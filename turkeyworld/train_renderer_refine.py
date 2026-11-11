"""Fine-tune only the existing compact Base and one-step RF on native Source RGB.

This is an independent run/optimizer. Frozen RSSM prior features are computed
from past observations before the next RGB target is inspected. The original
world and renderer bundles are never overwritten; no engine is queried here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from .distill_renderer import _digest, _save
from .gradient_safety import clip_finite_grad_norm_
from .models import load_sr_checkpoint, load_world_checkpoint
from .power_guard import KeepAwake
from .renderer_refinement import (causal_conditions, configure_refinement, evaluate_refinement,
    export_refined_bundle, frozen_fingerprint, refinement_loss)
from .renderer_student import _hash_tensors, load_renderer_student
from .staged_curriculum import StagedSequenceData
from .training_pipeline import atomic_json, now, replace_file


class SourcePools:
    def __init__(self, directories, split, *, seed=173):
        self.split, self.rng = split, np.random.default_rng(seed)
        self.pools, self.catalog, identities = [], [], []
        for directory in directories:
            directory = Path(directory).resolve()
            if directory.name != split:
                raise ValueError('Renderer supervision needs explicit train/val split directories')
            data = StagedSequenceData(directory, 'cpu', mode='stream', cache_bytes=64*1024**2)
            pool_index = len(self.pools)
            for episode_index, episode in enumerate(data.episodes):
                path = Path(episode['path'])
                with np.load(path, allow_pickle=False) as source:
                    if (str(source['split']) != split or source['frames'].dtype != np.uint8
                            or source['frames'].shape != (len(source['states']), 32, 32, 3)
                            or bool(source['enemies_enabled']) or np.any(source['states'][:, 12:14])):
                        raise ValueError('Expected genuine native32 no-enemy Source episodes')
                    length, seed_value = len(source['actions']), int(source['seed'])
                self.catalog.append({'pool': pool_index, 'episode': episode_index, 'actions': length,
                                     'path': str(path), 'seed': seed_value})
                identities.append({'path': str(path), 'sha256': _digest(path), 'seed': seed_value, 'actions': length})
            self.pools.append(data)
        if not self.catalog:
            raise ValueError('No Source episodes for renderer refinement')
        self.identity = {'split': split, 'files': identities}

    def fixed_panel(self, count, *, context=1, seed=173):
        eligible = [index for index, item in enumerate(self.catalog) if item['actions'] >= context]
        if count > len(eligible):
            raise ValueError('Validation needs one distinct Source episode per requested panel sample')
        rng = np.random.default_rng(seed)
        return [(int(index), int(rng.integers(self.catalog[int(index)]['actions']-context+1)))
                for index in rng.choice(eligible, count, replace=False)]

    def panel_fingerprint(self, panel):
        import hashlib
        return hashlib.sha256(json.dumps({'data': self.identity, 'panel': panel}, sort_keys=True).encode()).hexdigest()

    def batch(self, size, *, indices=None, context=1):
        if indices is None:
            indices = [(int(index), int(self.rng.integers(self.catalog[int(index)]['actions']-context+1)))
                       for index in self.rng.integers(len(self.catalog), size=size)]
        rows = []
        for index, start in indices:
            item = self.catalog[int(index)]
            rows.append(self.pools[item['pool']].batch(1, context=context, horizon=1,
                indices=np.asarray([[item['episode'], int(start)]], np.int64)))
        keys = ('states', 'frames', 'target_states', 'target_frames', 'actions', 'map')
        return {key: torch.cat([row[key] for row in rows]) for key in keys}


def _preview(arrays, path, step):
    count, size = min(4, len(arrays['rgb'])), 160
    rows = [('base', 'COMPACT BASE'), ('source', 'SOURCE native32'), ('rgb', 'RF1 Source-finetuned')]
    if 'sr' in arrays:
        rows.append(('sr', 'FROZEN SR128 (no HR target)'))
    image = Image.new('RGB', (count*size, len(rows)*(size+24)), (12, 17, 25))
    draw = ImageDraw.Draw(image)
    for row, (key, label) in enumerate(rows):
        for column in range(min(count, len(arrays[key]))):
            value = arrays[key][column].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
            image.paste(Image.fromarray(value).resize((size, size), Image.Resampling.NEAREST),
                        (column*size, row*(size+24)+24))
            draw.text((column*size+3, row*(size+24)+4), f'{label} | {step}', fill='white')
    temporary = Path(str(path)+'.tmp.png')
    image.save(temporary)
    replace_file(temporary, Path(path))


class RendererRefinement:
    def __init__(self, args):
        self.args, self.root = args, Path(args.run_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        self.state = {'run_id': self.root.name, 'stage': 'renderer_refinement', 'status': 'initializing',
            'model': 'Student compact Base + RF1; VQGAN / RSSM / map frozen', 'current': 0,
            'total': args.steps, 'started_at': now(), 'prediction_every': args.preview_every,
            'scope': 'Source-supervised navigation renderer only', 'world_training': False}

    def publish(self, **values):
        self.state.update(values, updated_at=now(), elapsed_seconds=time.perf_counter()-self.started)
        atomic_json(self.root/'status.json', self.state)

    def run(self):
        args, device = self.args, torch.device(self.args.device)
        if device.type == 'cuda':
            torch.cuda.init()
        torch.set_num_threads(args.threads)
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # Reference retains the ORIGINAL codec/Flow for strict export provenance.
        source, source_metadata = load_world_checkpoint(args.world, 'cpu')
        source_step = source_metadata.get('training_steps')
        source.eval().requires_grad_(False)
        working, metadata = load_world_checkpoint(args.world, 'cpu')
        del source_metadata, metadata
        initial = load_renderer_student(working, args.renderer_checkpoint,
                                       allow_unqualified_codec=args.allow_unqualified_codec)
        initial_renderer_steps = initial['training_steps']
        initial_base_steps = working.base_student_bundle['training_steps']
        working.float().to(device)
        policy = configure_refinement(working)
        if policy['base_trainable'] > 1_000_000 or policy['flow_trainable'] > 7_000_000 or policy['codec_parameters'] > 1_000_000:
            raise ValueError('Compact renderer parameter budgets changed')
        optimizer = torch.optim.AdamW([{'params': working.base_student.parameters(), 'lr': args.base_lr},
            {'params': [p for p in working.flow_renderer.parameters() if p.requires_grad], 'lr': args.lr}], weight_decay=.01)
        sr, sr_metadata = (None, None) if args.sr is None else load_sr_checkpoint(args.sr, device)
        if sr is not None:
            sr.float().eval().requires_grad_(False)
        sr_identity = _hash_tensors(sr.state_dict()) if sr is not None else None
        base = Path(args.data_root).resolve() if args.data_root else None
        train_dirs = ([base/'short/rgb/train'] if base is not None else [])+args.train_dir
        val_dirs = ([base/'short/rgb/val'] if base is not None else [])+args.val_dir
        train, validation = SourcePools(train_dirs, 'train', seed=args.seed), SourcePools(val_dirs, 'val', seed=args.seed+1)
        train_hashes, val_hashes = ({row['sha256'] for row in data.identity['files']} for data in (train, validation))
        train_seeds, val_seeds = ({row['seed'] for row in data.identity['files']} for data in (train, validation))
        if train_hashes & val_hashes or train_seeds & val_seeds:
            raise ValueError('Source train/validation files or seed identities overlap')
        panel = validation.fixed_panel(args.validation_frames, seed=args.seed)
        identity = {'world_sha256': _digest(args.world), 'renderer_sha256': _digest(args.renderer_checkpoint),
                    'train': train.identity, 'val': validation.identity, 'panel': panel,
                    'sr_sha256': _digest(args.sr) if args.sr else None}
        configuration = {key: [str(x) for x in value] if isinstance(value, list) else str(value) if isinstance(value, Path) else value
                         for key, value in vars(args).items()}
        frozen = frozen_fingerprint(working)
        codec_identity = _hash_tensors(working.flow_renderer.codec.state_dict())
        optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group['params']}
        codec_excluded = not bool(optimizer_ids & {id(parameter) for parameter in working.flow_renderer.codec.parameters()})
        if not codec_excluded:
            raise AssertionError('Codec parameters entered the renderer optimizer')
        atomic_json(self.root/'source_identity.json', identity)
        self.publish(model_parameters=policy, renderer_training={'active': True, 'base_active': True,
            'codec_frozen': True, 'rssm_frozen': True, 'map_frozen': True, 'gan_active': False,
            'reason': 'student_renderer_source_refinement'}, source_teacher_step=source_step,
            validation_panel_fingerprint=validation.panel_fingerprint(panel), solver_steps=1)
        checkpoint = self.root/'checkpoints/renderer_refine_last.pt'
        best_path, baseline_path = self.root/'checkpoints/renderer_refine_best.pt', self.root/'old_rf1_baseline.pt'
        step, best, streak = 0, None, 0
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location=device, weights_only=False)
            if ({k: v for k, v in saved['configuration'].items() if k != 'steps'} !=
                    {k: v for k, v in configuration.items() if k != 'steps'} or args.steps < saved['configuration']['steps']
                    or saved['identity'] != identity or saved['frozen_fingerprint'] != frozen):
                raise ValueError('Renderer refinement resume identities/objectives changed or budget shrank')
            working.base_student.load_state_dict(saved['base'], strict=True)
            working.flow_renderer.load_state_dict(saved['flow'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            step, best, streak = saved['step'], saved['best'], saved['streak']
            train.rng.bit_generator.state = saved['sampler_rng']
            torch.random.set_rng_state(saved['rng_torch'].cpu()); np.random.set_state(saved['rng_numpy']); random.setstate(saved['rng_python'])
            if device.type == 'cuda':
                torch.cuda.set_rng_state_all([value.cpu() for value in saved['rng_cuda']])
        def save():
            _save(checkpoint, {'base': working.base_student.state_dict(), 'flow': working.flow_renderer.state_dict(),
                'optimizer': optimizer.state_dict(), 'step': step, 'best': best, 'streak': streak,
                'configuration': configuration, 'identity': identity, 'frozen_fingerprint': frozen,
                'sampler_rng': train.rng.bit_generator.state, 'rng_torch': torch.random.get_rng_state(),
                'rng_numpy': np.random.get_state(), 'rng_python': random.getstate(),
                'rng_cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []})
        def verify_frozen(at_step):
            if (frozen_fingerprint(working) != frozen
                    or _hash_tensors(working.flow_renderer.codec.state_dict()) != codec_identity
                    or working.flow_renderer.codec.training or working.flow_renderer.codec.quantize.training
                    or sr is not None and _hash_tensors(sr.state_dict()) != sr_identity):
                raise AssertionError('Frozen RSSM/map/geometry/codec/SR parameters or buffers changed')
            self.publish(frozen_verification={'checked_after_optimizer_step': at_step,
                'all_frozen_parameters_buffers_bitexact': True, 'codec_all_state_dict_tensors_bitexact': True,
                'codec_optimizer_excluded': codec_excluded, 'codec_eval': True,
                'codec_sha256': codec_identity, 'frozen_sha256': frozen,
                'sr_loaded': sr is not None, 'sr_bitexact': True if sr is not None else None})
        def validate(count=None):
            chosen = panel if count is None else panel[:count]
            report, arrays = evaluate_refinement(working, validation, chosen, step=step,
                batch_size=args.validation_batch, baseline=baseline if count is None else None, sr=sr,
                target_ssim=args.target_ssim, target_mae=args.target_mae)
            _preview(arrays, self.root/'preview.png', step)
            atomic_json(self.root/'preview.json', {'kind': 'static_scene_query', 'step': step, 'every': args.preview_every,
                'source_held_out': True, 'renderer_only': True, 'world_rollout': False, 'coordinate_source': 'source_gt',
                'feature_source': 'frozen_prior_before_target_rgb', 'rows': ['base', 'source', 'rf1'],
                'source_quality_gate_evaluated': count is None})
            return report, arrays
        if baseline_path.exists():
            old = torch.load(baseline_path, map_location='cpu', weights_only=False)
            if old['identity'] != identity:
                raise ValueError('Original RF1 baseline identity changed')
            baseline = old['arrays']
        else:
            if step > 0:
                raise ValueError('Resume requires the immutable original RF1 validation baseline')
            report, baseline = evaluate_refinement(working, validation, panel, step=0,
                batch_size=args.validation_batch, sr=sr, target_ssim=args.target_ssim, target_mae=args.target_mae)
            _save(baseline_path, {'identity': identity, 'arrays': baseline, 'report': report})
            atomic_json(self.root/'validation_initial.json', report)
            _preview(baseline, self.root/'preview.png', 0)
        self.publish(validation_initial=report if 'report' in locals() else old['report'],
                     validation=report if 'report' in locals() else old['report'])
        first_update = step+1
        stop_at = step if streak >= args.required_passes else args.steps
        for step in range(step+1, stop_at+1):
            if (self.root/'stop_training').exists():
                step -= 1; save(); self.publish(status='stopped', current=step); return
            configure_refinement(working)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = refinement_loss(working, train.batch(args.batch), ssim_weight=args.ssim_weight,
                edge_weight=args.edge_weight, latent_weight=args.latent_weight, detach_base=not args.flow_to_base)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite Source refinement loss')
            loss.backward()
            metrics['gradient_norm'] = float(clip_finite_grad_norm_(working.named_parameters(), context=f'Renderer refine{step}'))
            optimizer.step()
            if step == first_update:
                verify_frozen(step)
            report = None
            if step % args.validate_every == 0 or step == args.steps:
                report, _ = validate()
                if step != first_update:
                    verify_frozen(step)
                report['frozen_components_bitexact'] = True
                score = report['rf1']['mae_mean']+.5*(1-report['rf1']['ssim_mean'])+report['base']['mae_mean']
                if best is None or score < best['score']:
                    best = {'score': score, 'step': step}
                    _save(best_path, {'base': working.base_student.state_dict(), 'flow': working.flow_renderer.state_dict(), **best})
                streak = streak+1 if report['passed'] else 0
                atomic_json(self.root/'validation.json', report)
            elif step % args.preview_every == 0:
                validate(4)
            with (self.root/'metrics.jsonl').open('a', encoding='utf-8') as log:
                log.write(json.dumps({'stage': 'renderer_refinement', 'step': step, 'timestamp': now(), **metrics,
                    **({'source_validation': report} if report is not None else {})})+'\n')
            self.publish(status='running', current=step, progress=step/args.steps, loss=metrics['train_loss'],
                metrics=metrics, validation=report or self.state.get('validation'), selected_best=best,
                eta_seconds=(time.perf_counter()-self.started)/step*(args.steps-step))
            if step % args.save_every == 0 or report is not None:
                save()
            if streak >= args.required_passes:
                break
        save()
        selected = torch.load(best_path, map_location=device, weights_only=False)
        working.base_student.load_state_dict(selected['base'], strict=True)
        working.flow_renderer.load_state_dict(selected['flow'], strict=True)
        final, arrays = evaluate_refinement(working, validation, panel, step=selected['step'], baseline=baseline,
            batch_size=args.validation_batch, sr=sr, target_ssim=args.target_ssim, target_mae=args.target_mae)
        if frozen_fingerprint(working) != frozen:
            raise AssertionError('Frozen assets changed before export')
        final['frozen_components_bitexact'] = True
        metadata = export_refined_bundle(source, working, self.root/'checkpoints', updates=selected['step'], report=final,
            initial_renderer_steps=initial_renderer_steps, initial_base_steps=initial_base_steps,
            provenance={'kind': 'Source_GT_native32_causal_prior_renderer_refinement', 'source_identity': identity,
                        'codec_frozen': True, 'dynamics_frozen': True})
        _preview(arrays, self.root/'preview.png', selected['step'])
        atomic_json(self.root/'validation_selected.json', final)
        self.publish(status='completed', current=step, selected_step=selected['step'], validation=final,
            renderer_export=metadata, source_quality_passed=final['passed'], gameplay_qualified=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('world', 'renderer-checkpoint', 'run-dir'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--train-dir', type=Path, action='append', default=[])
    parser.add_argument('--val-dir', type=Path, action='append', default=[])
    parser.add_argument('--sr', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--allow-unqualified-codec', action='store_true')
    parser.add_argument('--flow-to-base', action='store_true', help='Opt in to final-RF gradients into Base; default is detached')
    for name, default in (('steps', 2000), ('batch', 16), ('validation-frames', 128), ('validation-batch', 8),
                          ('validate-every', 100), ('preview-every', 10), ('save-every', 10), ('required-passes', 2), ('seed', 173), ('threads', 4)):
        parser.add_argument('--'+name, type=int, default=default)
    for name, default in (('lr', 1e-4), ('base-lr', 1e-4), ('ssim-weight', .2), ('edge-weight', .1),
                          ('latent-weight', 0.), ('target-ssim', .90), ('target-mae', .03)):
        parser.add_argument('--'+name, type=float, default=default)
    args = parser.parse_args(argv)
    if (args.data_root is None and (not args.train_dir or not args.val_dir)
            or min(args.steps, args.batch, args.validation_frames, args.validation_batch, args.validate_every,
                   args.preview_every, args.save_every, args.required_passes, args.threads) < 1
            or min(args.lr, args.base_lr) <= 0 or min(args.ssim_weight, args.edge_weight, args.latent_weight, args.target_mae) < 0):
        parser.error('Provide Source train/val data and valid positive budgets/rates')
    experiment = RendererRefinement(args)
    try:
        with KeepAwake():
            experiment.run()
    except BaseException as error:
        experiment.publish(status='failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
