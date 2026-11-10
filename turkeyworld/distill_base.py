"""Distill a compact Base head from GT poses and genuine one-step own contexts.

Source RGB initializes the frozen world encoder once for an own-context sample.
The compact image head receives numerical state, actual predicted128D features
and shared learned camera geometry; neither source RGB nor future labels enter it.
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
from torch.nn import functional as F

from .base_student import build_base_student, export_base_student
from .distill_renderer import _batch, _digest, _panel_identity, _save, select_pose_panel
from .gradient_safety import clip_finite_grad_norm_
from .models import load_world_checkpoint
from .object_losses import balanced_edge_l1, decoded_ssim_loss
from .power_guard import KeepAwake
from .training_pipeline import atomic_json, now, replace_file
from .visual_quality import image_quality_metrics, strict_render_precision


GEOMETRY_KEYS = ('ray_tokens', 'object_identity_tokens', 'object_appearance_tokens', 'render_gate')


def validate_resume_configuration(previous, current, *, completed_step):
    """Permit an explicit larger budget; all objective/data settings stay fixed."""
    old_budget, new_budget = previous.get('steps'), current.get('steps')
    if (not isinstance(old_budget, int) or isinstance(old_budget, bool)
            or not isinstance(new_budget, int) or isinstance(new_budget, bool)
            or new_budget < old_budget or new_budget < completed_step
            or {key: value for key, value in previous.items() if key != 'steps'} !=
               {key: value for key, value in current.items() if key != 'steps'}):
        raise ValueError('Base resume permits only a monotonic training-budget increase')


def prepare_base_targets(teacher, directory, *, count=512, own_count=None, batch_size=16,
                         seed=173, teacher_fingerprint, cache_path=None, progress=None):
    """Cache separate GT/zero-feature and own-predicted-state/feature conditions."""
    if hasattr(teacher, 'base_student'):
        raise ValueError('Base targets must use the original teacher Base, before attaching a student')
    directory = Path(directory).resolve()
    own_count = count if own_count is None else own_count
    panels = {'gt': select_pose_panel(directory, count, seed, image_size=teacher.config.image_size),
              'own': select_pose_panel(directory, own_count, seed+10000, image_size=teacher.config.image_size)}
    identity = {'format_version': 1, 'split': directory.name, 'teacher_fingerprint': teacher_fingerprint,
                'gt': _panel_identity(panels['gt'], directory.name),
                'own': _panel_identity(panels['own'], directory.name),
                'own_initialization': 'one_current_source_frame_and_state_then_one_neural_transition',
                'own_action': 'recorded_next_action_or_zero_if_terminal', 'future_source_feedback': False}
    if cache_path is not None and Path(cache_path).exists():
        saved = torch.load(cache_path, map_location='cpu', weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('Base target teacher/source/domain identity changed')
        return saved
    device = next(teacher.parameters()).device
    modes = [(module, module.training) for module in teacher.modules()]
    values = {name: [] for name in ('state', 'features', 'target_rgb', 'domain', *GEOMETRY_KEYS)}
    completed, total = 0, count+own_count
    try:
        teacher.eval()
        devices = [device.index or 0] if device.type == 'cuda' else []
        with torch.inference_mode(), strict_render_precision(), torch.random.fork_rng(devices=devices), \
                torch.autocast(device.type, enabled=False):
            for domain_id, (domain, panel) in enumerate(panels.items()):
                for first in range(0, len(panel), batch_size):
                    selected = panel[first:first+batch_size]
                    states, grids, images, actions = [], [], [], []
                    for path, frame in selected:
                        with np.load(path, allow_pickle=False) as source:
                            states.append(source['states'][frame])
                            grids.append(source['map'])
                            images.append(source['frames'][frame])
                            actions.append(source['actions'][frame] if frame < len(source['actions']) else np.zeros(4, np.float32))
                    state = torch.tensor(np.stack(states), device=device, dtype=torch.float32)
                    grid = torch.tensor(np.stack(grids), device=device, dtype=torch.long)
                    map_features = teacher.encode_map(grid)
                    features = state.new_zeros(len(state), 128)
                    if domain == 'own':
                        image = torch.tensor(np.stack(images), device=device).permute(0, 3, 1, 2).float()/255
                        memory = teacher.initialize_from_image(image, state, map_features)
                        controls = torch.tensor(np.stack(actions), device=device, dtype=torch.float32)
                        prediction = teacher.transition(state, controls, map_features, memory)
                        state, features = prediction['state'], prediction['features']
                    target = teacher.render_base(state, features, map_features)
                    geometry = teacher.vjepa_backend.geometry(state, map_features)
                    batch = {'state': state, 'features': features, 'target_rgb': target,
                             'domain': torch.full((len(state),), domain_id, device=device, dtype=torch.long),
                             **{name: geometry[name] for name in GEOMETRY_KEYS}}
                    if not all(bool(torch.isfinite(value).all()) for value in batch.values()):
                        raise FloatingPointError('Nonfinite Base target buffer')
                    for name, value in batch.items():
                        values[name].append(value.detach().cpu())
                    completed += len(selected)
                    if progress:
                        progress(completed, total, directory.name, domain)
    finally:
        for module, mode in modes:
            module.training = mode
    result = {'identity': identity, **{name: torch.cat(rows) for name, rows in values.items()}}
    if cache_path is not None:
        _save(cache_path, result)
    return result


def base_prediction(head, batch):
    return head(batch['state'], batch['features'], {name: batch[name] for name in GEOMETRY_KEYS})


def base_distillation_loss(head, batch):
    predicted = base_prediction(head, batch)
    target = batch['target_rgb'].detach()
    pixel = F.l1_loss(predicted, target)
    structural = decoded_ssim_loss(predicted, target)
    edges = balanced_edge_l1(predicted, target)
    loss = pixel+.2*structural+.1*edges
    return loss, {'base_pixel_l1': float(pixel.detach()), 'base_ssim_loss': float(structural.detach()),
                  'base_edge_l1': float(edges.detach()), 'train_loss': float(loss.detach())}


def evaluate_base_student(head, buffer, *, batch_size=16, step=0, ssim=.98, mae=.01):
    if buffer['identity']['split'] != 'val':
        raise ValueError('Base quality requires an explicit held-out val buffer')
    device = next(head.parameters()).device
    modes = [(module, module.training) for module in head.modules()]
    rows, previews = [], []
    devices = [device.index or 0] if device.type == 'cuda' else []
    try:
        head.eval()
        with torch.inference_mode(), strict_render_precision(), torch.random.fork_rng(devices=devices), \
                torch.autocast(device.type, enabled=False):
            for first in range(0, len(buffer['state']), batch_size):
                batch = _batch(buffer, slice(first, first+batch_size), device)
                predicted = base_prediction(head, batch)
                if (not bool(torch.isfinite(predicted).all()) or bool((predicted < 0).any())
                        or bool((predicted > 1).any())):
                    raise FloatingPointError('Invalid compact Base RGB')
                quality = image_quality_metrics(predicted.float(), batch['target_rgb'].float())
                for index in range(len(predicted)):
                    domain = int(batch['domain'][index])
                    rows.append({'domain': ('gt', 'own')[domain],
                                 **{name: float(value[index]) for name, value in quality.items()}})
                    if sum(item['domain'] == domain for item in previews) < 2:
                        previews.append({'domain': domain, 'images': torch.stack([
                            batch['target_rgb'][index], predicted[index],
                            (batch['target_rgb'][index]-predicted[index]).abs()]).cpu()})
    finally:
        for module, mode in modes:
            module.training = mode
    aggregate = lambda items: {name+'_mean': float(np.mean([row[name] for row in items]))
                              for name in ('ssim', 'mae', 'psnr')}
    domains = {name: aggregate([row for row in rows if row['domain'] == name]) for name in ('gt', 'own')}
    report = {'step': step, 'frames': len(rows), 'summary': aggregate(rows), 'domains': domains,
        'passed': all(value['ssim_mean'] >= ssim and value['mae_mean'] <= mae for value in domains.values()),
        'thresholds': {'ssim_mean_each_domain': ssim, 'mae_mean_each_domain': mae},
        'source_held_out': True, 'future_source_feedback': False,
        'scope': 'Compact Base versus frozen original teacher Base at GT and one-step own contexts',
        'per_frame': rows, 'buffer_identity': buffer['identity']}
    return report, previews


def _preview(previews, path, step):
    size = 160
    previews = sorted(previews, key=lambda item: item['domain'])
    canvas = Image.new('RGB', (len(previews)*size, 3*(size+24)), (10, 15, 24))
    draw = ImageDraw.Draw(canvas)
    for column, item in enumerate(previews):
        for row, (image, name) in enumerate(zip(item['images'], ('TEACHER BASE', 'COMPACT BASE', 'ABS DIFFERENCE'))):
            array = image.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
            canvas.paste(Image.fromarray(array).resize((size, size), Image.Resampling.NEAREST),
                         (column*size, row*(size+24)+24))
            draw.text((column*size+3, row*(size+24)+4), f'{name} {("GT", "OWN")[item["domain"]]} | {step}', fill='white')
    temporary = Path(str(path)+'.tmp.png')
    canvas.save(temporary)
    replace_file(temporary, Path(path))


class BaseDistillation:
    def __init__(self, args):
        self.args, self.root = args, Path(args.run_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        self.state = {'run_id': self.root.name, 'status': 'running', 'stage': 'base_distillation',
                      'current': 0, 'total': args.steps, 'world_training': False,
                      'source_teacher': str(Path(args.teacher).resolve()), 'started_at': now()}

    def publish(self, **values):
        self.state.update(values, updated_at=now(), elapsed_seconds=time.perf_counter()-self.started)
        atomic_json(self.root/'status.json', self.state)

    def run(self):
        args, device = self.args, torch.device(self.args.device)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        teacher, teacher_metadata = load_world_checkpoint(args.teacher, 'cpu')
        teacher.to(device).requires_grad_(False).eval()
        head = build_base_student(teacher, seed=args.seed).to(device).train()
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=.01)
        fingerprint = _digest(args.teacher)
        data = Path(args.data_dir).resolve()
        if (data/'rgb').is_dir():
            data = data/'rgb'
        self.publish(model_parameters={'base_head': sum(value.numel() for value in head.parameters()),
            'shared_map': sum(value.numel() for value in teacher.vjepa_backend.renderer.map_encoder.parameters()),
            'shared_geometry': sum(value.numel() for value in teacher.vjepa_backend.renderer.geometry_reader.parameters())})
        def progress(done, total, split, domain):
            self.publish(stage='base_targets', stage_label='Preparing teacher Base GT/own targets',
                         data_preparation={'completed': done, 'total': total, 'split': split, 'domain': domain})
        train = prepare_base_targets(teacher, data/'train', count=args.train_count, batch_size=args.batch,
            seed=args.seed, teacher_fingerprint=fingerprint, cache_path=self.root/'teacher_base_train.pt', progress=progress)
        validation = prepare_base_targets(teacher, data/'val', count=args.val_count, batch_size=args.batch,
            seed=args.seed+1, teacher_fingerprint=fingerprint, cache_path=self.root/'teacher_base_val.pt', progress=progress)
        hashes = lambda buffer: {row['source_sha256'] for domain in ('gt', 'own') for row in buffer['identity'][domain]['frames']}
        if hashes(train) & hashes(validation):
            raise ValueError('Base train/validation source files overlap')
        configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        identities = {'teacher': fingerprint, 'train': train['identity'], 'val': validation['identity']}
        checkpoint, best_path = self.root/'checkpoints/base_student_last.pt', self.root/'checkpoints/base_student_best.pt'
        step, best, streak = 0, None, 0
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location=device, weights_only=False)
            validate_resume_configuration(saved['configuration'], configuration, completed_step=saved['step'])
            if saved['identities'] != identities:
                raise ValueError('Base resume teacher/data changed')
            head.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            step, best, streak = saved['step'], saved['best'], saved['streak']
            torch.random.set_rng_state(saved['rng_torch'].cpu())
            np.random.set_state(saved['rng_numpy'])
            random.setstate(saved['rng_python'])
            if device.type == 'cuda':
                torch.cuda.set_rng_state_all([value.cpu() for value in saved['rng_cuda']])
        def save():
            _save(checkpoint, {'model': head.state_dict(), 'optimizer': optimizer.state_dict(), 'config': asdict(head.config),
                'step': step, 'best': best, 'streak': streak, 'configuration': configuration, 'identities': identities,
                'rng_torch': torch.random.get_rng_state(), 'rng_numpy': np.random.get_state(),
                'rng_python': random.getstate(), 'rng_cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []})
        def validate(full=True):
            selected = validation
            if not full:
                indices = torch.cat([(validation['domain'] == domain).nonzero().flatten()[:2] for domain in (0, 1)])
                selected = {**validation, **{key: value[indices] for key, value in validation.items() if torch.is_tensor(value)}}
            report, previews = evaluate_base_student(head, selected, batch_size=args.batch, step=step,
                                                    ssim=args.target_ssim, mae=args.target_mae)
            _preview(previews, self.root/'preview.png', step)
            atomic_json(self.root/'preview.json', {'step': step, 'every': args.preview_every, 'quality_gate_evaluated': full,
                'scope': report['scope'], 'rows': ['teacher_base', 'compact_base', 'absolute_difference']})
            return report
        if step == 0:
            atomic_json(self.root/'validation_initial.json', validate())
        stop_at = step if streak >= args.required_passes else args.steps
        for step in range(step+1, stop_at+1):
            if (self.root/'stop_training').exists():
                step -= 1
                save()
                self.publish(status='stopped', current=step, stage_label='Base stopped at a complete update')
                return
            head.train()
            sample = _batch(train, np.random.randint(0, len(train['state']), size=args.batch), device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = base_distillation_loss(head, sample)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite Base distillation loss')
            loss.backward()
            metrics['gradient_norm'] = float(clip_finite_grad_norm_(head.named_parameters(), context=f'Base update{step}'))
            optimizer.step()
            report = None
            if step % args.validate_every == 0 or step == args.steps:
                report = validate()
                atomic_json(self.root/'validation.json', report)
                score = report['summary']['mae_mean']+.5*(1-report['summary']['ssim_mean'])
                if best is None or score < best['score']:
                    best = {'score': score, 'step': step, 'report': report}
                    _save(best_path, {'model': head.state_dict(), **best})
                streak = streak+1 if report['passed'] else 0
            elif step % args.preview_every == 0:
                validate(False)
            with (self.root/'metrics.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps({'timestamp': now(), 'step': step, 'stage': 'base_distillation',
                    **metrics, **({'validation': report['summary']} if report else {})})+'\n')
            self.publish(stage='base_distillation', stage_label='Training compact Base', current=step,
                global_training_step=step, metrics=metrics, loss=metrics['train_loss'], best=best,
                validation=report or self.state.get('validation'), progress=step/args.steps)
            if step % args.save_every == 0 or report is not None:
                save()
            if streak >= args.required_passes:
                break
        save()
        selected = torch.load(best_path, map_location=device, weights_only=False)
        head.load_state_dict(selected['model'], strict=True)
        final, previews = evaluate_base_student(head, validation, batch_size=args.batch, step=selected['step'],
                                                ssim=args.target_ssim, mae=args.target_mae)
        _preview(previews, self.root/'preview.png', selected['step'])
        exported = export_base_student(teacher, head, self.root/'checkpoints/base_student.pt',
            training_steps=selected['step'], metrics=final,
            provenance={'teacher_checkpoint_sha256': fingerprint, 'teacher_training_steps': teacher_metadata.get('training_steps'),
                        'train': train['identity'], 'val': validation['identity']})
        atomic_json(self.root/'validation_selected.json', final)
        self.publish(status='completed', stage='base_distilled', current=step, selected_step=selected['step'],
            quality_gate_passed=final['passed'], validation=final, export=exported, world_quality_qualified=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('teacher', 'data-dir', 'run-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda' if torch.cuda.is_available() else 'cpu')
    for name, default in (('steps', 2000), ('train-count', 512), ('val-count', 128), ('batch', 16),
                          ('validate-every', 100), ('preview-every', 10), ('save-every', 10), ('required-passes', 2), ('seed', 173)):
        parser.add_argument('--'+name, type=int, default=default)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--target-ssim', type=float, default=.98)
    parser.add_argument('--target-mae', type=float, default=.01)
    args = parser.parse_args(argv)
    if (min(args.steps, args.train_count, args.val_count, args.batch, args.validate_every,
            args.preview_every, args.save_every, args.required_passes) < 1
            or not np.isfinite(args.lr) or args.lr <= 0 or not 0 <= args.target_ssim <= 1
            or not np.isfinite(args.target_mae) or args.target_mae < 0):
        parser.error('Counts, cadence, learning rate and quality thresholds must be valid')
    experiment = BaseDistillation(args)
    try:
        with KeepAwake():
            experiment.run()
    except BaseException as error:
        experiment.publish(status='failed', stage='error', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
