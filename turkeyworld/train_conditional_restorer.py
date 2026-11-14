"""Learn a hinted native32 RGB restorer while the entire RF3 parent is frozen."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import os
import random
import time

import numpy as np
import torch

from .conditional_decoder_restorer import ConditionalDecoderRestorer, conditional_decoder_restorer_metadata
from .conditional_restorer_objectives import (
    PROTOCOL, TRAINABLE_PREFIXES, configure_conditional_restorer,
    conditional_restorer_protocol, conditional_restorer_loss)
from .conditional_restorer_validation import evaluate_conditional_restorer, save_conditional_restorer_preview
from .conditional_restorer_bundle import export_conditional_restorer
from .distill_renderer import _digest
from .gradient_safety import clip_finite_grad_norm_
from .models import load_world_checkpoint
from .power_guard import KeepAwake
from .renderer_combat_data import CombatRendererFrames
from .renderer_small_enemy_data import SmallEnemyBalancedFrames
from .renderer_moe_five_objectives import FiveProfiledFrames
from .renderer_moe_five_phase import original_renderer_configuration
from .renderer_native_phase import _same, restore_renderer_native_rng
from .renderer_student import load_renderer_student, _hash_tensors
from .train_renderer_rf5 import durable_save
from .training_pipeline import atomic_json, now


def frozen_hash(model):
    return _hash_tensors({name: value for name, value in model.state_dict().items()
                          if not name.startswith(TRAINABLE_PREFIXES)})


class ConditionalRestorerTraining:
    def __init__(self, args):
        self.args, self.root = args, args.run_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        self.state = dict(run_id=self.root.name, status='initializing', stage='conditional_restorer',
            loss_protocol=PROTOCOL, current=0, global_training_step=0, total=args.steps,
            prediction_every=10, validation_every=100, image_size=32,
            model='Conditional Transformer RGB restorer; frozen RF3 + decoder', gameplay_qualified=False)

    def publish(self, **values):
        self.state.update(values, updated_at=now(), elapsed_seconds=time.perf_counter()-self.started)
        atomic_json(self.root / 'status.json', self.state)

    def prepare(self):
        a = self.args
        if a.device == 'cuda':
            torch.cuda.init()
        torch.set_num_threads(a.threads)
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.publish(stage_label='Loading the frozen RF3 / decoder parent')
        parent = torch.load(a.parent_decoder, map_location='cpu', weights_only=False)
        boundary = torch.load(a.parent_training, map_location='cpu', weights_only=False)
        generator = torch.load(a.generator_full_checkpoint, map_location='cpu', weights_only=False)
        recipe = torch.load(a.source_recipe_parent, map_location='cpu', weights_only=False)
        inherited = original_renderer_configuration(recipe['configuration'])
        self.source, meta = load_world_checkpoint(a.world, 'cpu')
        if (meta['training_steps'] != 5906 or parent['training_steps'] != 30717
                or parent['decoder_refinement_updates'] != boundary['step']):
            raise ValueError('Conditional restorer changed the frozen source/decoder clocks')
        self.source.eval().requires_grad_(False)
        self.model = deepcopy(self.source)
        load_renderer_student(self.model, a.parent_decoder, allow_unqualified_codec=True)
        self.model.float().to(a.device).eval().requires_grad_(False)
        self.reference = deepcopy(self.model.flow_renderer.codec).eval().requires_grad_(False)
        restorer = ConditionalDecoderRestorer(seed=a.seed).to(a.device)
        self.model.flow_renderer.conditional_restorer = restorer
        policy = configure_conditional_restorer(self.model)
        self.names = [name for name, p in self.model.named_parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                                           lr=a.lr, weight_decay=.01)
        self.frozen, self.reference_hash = frozen_hash(self.model), _hash_tensors(self.reference.state_dict())
        self.publish(stage_label='Restoring unchanged Source sampling and held-out panels')
        raw = CombatRendererFrames(inherited['combat_data_root'], inherited['appearance_data_root'],
                                  'train', seed=a.seed)
        validation = CombatRendererFrames(inherited['combat_data_root'], inherited['appearance_data_root'],
            'val', seed=a.seed+1, exclude_scene_keys=raw.scene_keys)
        self.train = SmallEnemyBalancedFrames(raw)
        identity = generator['phase']['identity']['parent_identity']
        for key, value in (('train', raw.identity), ('val', validation.identity),
                           ('distance_sampling', self.train.identity['distance_sampling'])):
            if not _same(identity[key], value):
                raise ValueError('Conditional restorer changed the pinned Source corpus: '+key)
        self.panel, self.geometry_panel = list(identity['panel']), list(identity['geometry_wall_panel'])
        if (validation.panel_fingerprint(self.panel) != identity['panel_fingerprint']
                or validation.panel_fingerprint(self.geometry_panel) != identity['geometry_wall_fingerprint']):
            raise ValueError('Held-out restorer panels changed')
        self.data, self.validation_data = FiveProfiledFrames(self.train), FiveProfiledFrames(validation)
        self.preview_ids = []
        for category in ('large', 'partial', 'distant', 'empty_navigation'):
            index = next((i for i in self.panel if validation.rows[i]['category'] == category), None)
            if index is not None and index not in self.preview_ids:
                self.preview_ids.append(index)
        self.preview_ids += [i for i in self.panel if i not in self.preview_ids][:4-len(self.preview_ids)]
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
        self.phase = dict(protocol=conditional_restorer_protocol(), configuration=config,
            inputs={key: _digest(getattr(a, key)) for key in (
                'world', 'parent_decoder', 'parent_training', 'generator_full_checkpoint', 'source_recipe_parent')},
            restorer_architecture=conditional_decoder_restorer_metadata(restorer, include_state_hash=False),
            generator_training_steps=30717, generator_local_step=1108, base_training_steps=29209,
            decoder_parent_updates=boundary['step'], feature_world_step=5906,
            source_train_identity=raw.identity, source_val_identity=validation.identity,
            panel_fingerprint=identity['panel_fingerprint'], geometry_panel_fingerprint=identity['geometry_wall_fingerprint'],
            preview_panel_fingerprint=validation.panel_fingerprint(self.preview_ids),
            frozen_fingerprint=self.frozen, reference_codec_sha256=self.reference_hash,
            restorer_parameter_names=self.names, fresh_restorer_Adam=True, initial_restorer_Adam_states=0,
            frozen_decoder_Adam_archived=True, automatic_active_promotion=False)
        phase_path = self.root / 'phase_plan.json'
        if phase_path.exists():
            if json.loads(phase_path.read_text(encoding='utf-8')) != self.phase:
                raise ValueError('Immutable restorer phase configuration or input pins changed')
        else:
            atomic_json(phase_path, self.phase)
        self.plan_sha = _digest(phase_path)
        self.checkpoint = self.root / 'checkpoints/restorer_last.pt'
        self.step, saved = 0, boundary
        if self.checkpoint.exists() and not a.probe:
            saved = torch.load(self.checkpoint, map_location='cpu', weights_only=False)
            if (saved['phase'] != self.phase or saved['plan_sha256'] != self.plan_sha
                    or type(saved.get('step')) is not int or not 0 <= saved['step'] <= a.steps
                    or saved['optimizer_parameter_names'] != self.names):
                raise ValueError('Restorer continuation changed its parent, phase or optimizer')
            restorer.load_state_dict(saved['restorer'], strict=True)
            self.optimizer.load_state_dict(saved['optimizer'])
            self.step = saved['step']
        self.train.restore_sampler(saved['sampler'])
        if not _same(self.train.sampler_state(), saved['sampler']):
            raise AssertionError('Source sampler restoration was not exact')
        restore_renderer_native_rng(saved, device=a.device)
        self.training = dict(protocol=PROTOCOL, plan_sha256=self.plan_sha,
            feature_world_sha256=self.phase['inputs']['world'], feature_world_step=5906,
            parent_decoder_sha256=self.phase['inputs']['parent_decoder'],
            frozen_RF_training_steps=30717, frozen_RF_local_step=1108, base_training_steps=29209,
            frozen_decoder_updates=boundary['step'], solver_steps=3,
            restorer_parameters=policy['trainable_parameters'],
            restorer_parameter_tensors=policy['trainable_parameter_tensors'],
            rf_weight=.5, source_weight=.5, source_queries_per_update=a.batch,
            decoder_examples_per_update=2*a.batch, local_step=self.step, local_budget=a.steps,
            encoder_frozen=True, codebook_frozen=True, RF_frozen=True, Base_frozen=True,
            decoder_frozen=True, post_quant_conv_trainable=False, restorer_only=True,
            inputs=['RF3_quantized_codes', 'decoded_RGB', 'current_state20', 'raw_map16x16'],
            hints_are_predicted_in_game=True, Source_RGB_and_masks_are_labels_only=True,
            preview_panel_fingerprint=self.phase['preview_panel_fingerprint'])
        self.publish(conditional_restorer_training=self.training, current=self.step,
            global_training_step=self.step, renderer_training_steps=30717, base_training_steps=29209,
            validation_panel_fingerprint=self.phase['panel_fingerprint'],
            renderer_training=dict(active=True, rf_active=False, base_active=False, decoder_active=False,
                restorer_active=True, encoder_frozen=True, codebook_frozen=True, gan_active=False))

    def save(self):
        durable_save(self.checkpoint, dict(restorer=self.model.flow_renderer.conditional_restorer.state_dict(),
            optimizer=self.optimizer.state_dict(), optimizer_parameter_names=self.names,
            step=self.step, phase=self.phase, plan_sha256=self.plan_sha,
            sampler=self.train.sampler_state(), rng_torch=torch.get_rng_state(), rng_numpy=np.random.get_state(),
            rng_python=random.getstate(), rng_cuda=torch.cuda.get_rng_state_all() if self.args.device=='cuda' else []))

    def update(self):
        self.optimizer.zero_grad(set_to_none=True)
        batch = self.data.batch(self.args.batch)
        started = time.perf_counter()
        value, metrics = conditional_restorer_loss(self.model, batch)
        value.backward()
        outside = [name for name, p in self.model.named_parameters()
                   if p.grad is not None and not name.startswith(TRAINABLE_PREFIXES)]
        if outside:
            raise AssertionError('A frozen parent parameter received a gradient')
        norm = float(clip_finite_grad_norm_(self.model.named_parameters(), context='Conditional restorer'))
        self.optimizer.step(); self.step += 1
        if self.step <= 5 or self.step % 100 == 0:
            if frozen_hash(self.model) != self.frozen or _hash_tensors(self.reference.state_dict()) != self.reference_hash:
                raise AssertionError('Restorer changed a frozen parent or reference codec')
        return dict(metrics, step=self.step, loss_protocol=PROTOCOL, plan_sha256=self.plan_sha,
            feature_world_sha256=self.phase['inputs']['world'], gradient_norm=norm,
            optimizer_update_seconds=time.perf_counter()-started, timestamp=now(), restorer_updates=self.step)

    def preview(self):
        _, arrays = evaluate_conditional_restorer(self.model, self.reference, self.validation_data,
            self.preview_ids, step=self.step, batch_size=4)
        meta = save_conditional_restorer_preview(arrays, self.root / 'preview.png', self.step)
        meta.update(kind='conditional_restorer_GT_query', every=10, loss_protocol=PROTOCOL,
            plan_sha256=self.plan_sha, feature_world_sha256=self.phase['inputs']['world'],
            validation_panel_fingerprint=self.phase['panel_fingerprint'], solver_steps=3,
            renderer_only=True, world_rollout=False, future_source_rgb_in_prior=False,
            future_source_pose_in_prior=False, restorer_inputs=self.training['inputs'])
        atomic_json(self.root / 'preview.json', meta)
        self.publish(prediction_preview=meta, preview_url='/preview.png')

    def export(self, report=None):
        return export_conditional_restorer(self.model, self.args.parent_decoder,
            self.root / 'checkpoints/renderer_for_play.pt', restorer_updates=self.step,
            source_model=self.source, metrics=dict(validation=report or {}),
            provenance=dict(plan_sha256=self.plan_sha))

    def validate(self, baseline):
        report, _ = evaluate_conditional_restorer(self.model, self.reference, self.validation_data,
            self.panel, step=self.step, batch_size=8, geometry_panel=self.geometry_panel, baseline=baseline)
        report.update(plan_sha256=self.plan_sha, feature_world_sha256=self.phase['inputs']['world'])
        atomic_json(self.root / 'validation' / f'step_{self.step:06d}.json', report)
        atomic_json(self.root / 'validation.json', report)
        row = dict(step=self.step, loss_protocol=PROTOCOL, protocol=PROTOCOL,
            plan_sha256=self.plan_sha, feature_world_sha256=self.phase['inputs']['world'], solver_steps=3,
            panel_fingerprint=report['panel_fingerprint'], val_loss=report['val_loss'],
            validation_objective=report['validation_objective'], source_held_out=True,
            renderer_only=True, world_rollout=False)
        with (self.root / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        self.export(report)
        self.publish(validation=report, conditional_restorer_validation=report)
        self.preview()
        return report

    def run(self):
        self.prepare()
        if self.args.probe:
            before = deepcopy(self.model.flow_renderer.conditional_restorer.state_dict())
            rows = [self.update() for _ in range(3)]
            self.save(); self.export()
            after = self.model.flow_renderer.conditional_restorer.state_dict()
            changed = [k for k in before if not torch.equal(before[k], after[k])]
            proof = dict(passed=bool(changed), discarded_optimizer_updates=3, production_training_updates=0,
                rows=rows, frozen_assets_exact=True, changed_restorer_tensors=changed,
                trainable_parameters=self.training['restorer_parameters'],
                trainable_tensors=self.training['restorer_parameter_tensors'],
                seconds_per_update=sum(r['optimizer_update_seconds'] for r in rows)/3,
                peak_cuda_GiB=torch.cuda.max_memory_allocated()/2**30 if self.args.device=='cuda' else 0.)
            atomic_json(self.root / 'gpu_warmstart.json', proof)
            self.publish(status='completed', message='Three discarded hinted-restorer GPU updates passed')
            print(json.dumps(dict(probe_passed=proof['passed'], report=str(self.root/'gpu_warmstart.json'),
                seconds=proof['seconds_per_update'], peak_GiB=proof['peak_cuda_GiB'])), flush=True)
            return
        baseline_path = self.root / 'validation_initial.json'
        baseline = json.loads(baseline_path.read_text(encoding='utf-8')) if baseline_path.exists() else None
        if self.step > 0 and baseline is None:
            raise ValueError('Restorer resume requires its recorded initial baseline')
        if baseline is None:
            self.publish(stage_label='Measuring unchanged parent / zero-initialized hinted restorer')
            baseline = self.validate(None)
            atomic_json(baseline_path, baseline)
        self.save()
        self.publish(status='running', stage_label='Conditional restorer only: RF3 / Source codes 50:50')
        while self.step < self.args.steps:
            if (self.root / 'stop_training').exists():
                self.save(); self.publish(status='stopped'); return
            row = self.update()
            with (self.root / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            self.training['local_step'] = self.step
            self.save()
            self.publish(current=self.step, global_training_step=self.step, loss=row['train_loss'],
                latest_metrics=row, conditional_restorer_training=self.training)
            if self.step % 100 == 0:
                self.validate(baseline)
            elif self.step % 10 == 0:
                self.preview()
        self.save(); self.publish(status='completed', message='Restorer budget completed; quality is evaluated separately')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'world', 'parent-decoder', 'parent-training',
                 'generator-full-checkpoint', 'source-recipe-parent'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=173)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--probe', action='store_true')
    a = p.parse_args(argv)
    if a.steps < 1 or a.batch != 8 or a.lr <= 0:
        p.error('Restorer requires positive steps/LR and eight paired Source queries')
    for name, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, name, value.resolve())
    from .vqgan_pipeline import process_alive
    a.run_dir.mkdir(parents=True, exist_ok=True)
    pid = a.run_dir/'training.pid'
    if pid.exists() and process_alive(int(pid.read_text())):
        p.error('Conditional restorer worker is already active')
    pid.write_text(str(os.getpid()), encoding='ascii')
    experiment = ConditionalRestorerTraining(a)
    try:
        with KeepAwake():
            experiment.run()
    except BaseException as error:
        experiment.publish(status='failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
