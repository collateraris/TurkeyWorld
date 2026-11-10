"""Standalone same-noise latent reflow of a frozen world's image generator.

Only Flow is optimized. Source states are fixed-pose rendering conditions,
not an autonomous-world validation or future input to a dynamics model.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from .gradient_safety import clip_finite_grad_norm_
from .models import load_world_checkpoint
from .power_guard import KeepAwake
from .structured_state import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
from .training_pipeline import atomic_json, now, replace_file
from .visual_quality import image_quality_metrics, strict_render_precision


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(4*1024**2), b''):
            value.update(block)
    return value.hexdigest()


def select_pose_panel(directory, count, seed, *, image_size):
    """Select unique fixed-pose frames from exactly one explicit source split."""
    directory = Path(directory).resolve()
    if directory.name not in {'train', 'val'} or count < 1:
        raise ValueError('Renderer buffer requires an explicit train/val directory and positive count')
    files, lengths = [], []
    for path in sorted(directory.glob('*.npz')):
        with np.load(path, allow_pickle=False) as data:
            if str(data['split']) != directory.name or data['states'].shape[1:] != (STATE_DIM,):
                raise ValueError('Renderer source split/state schema mismatch')
            if data['frames'].dtype != np.uint8 or data['frames'].shape != (len(data['states']), image_size, image_size, 3):
                raise ValueError('Renderer source must have native-size uint8 RGB')
            files.append(path)
            lengths.append(len(data['states']))
    if not files or sum(lengths) < count:
        raise ValueError('Insufficient distinct source poses for renderer buffer')
    # Uniform without replacement over frames, using a private RNG.
    choices = np.random.default_rng(seed).choice(sum(lengths), count, replace=False)
    ends = np.cumsum(lengths)
    panel = []
    for choice in choices:
        index = int(np.searchsorted(ends, choice, side='right'))
        panel.append((files[index], int(choice-(ends[index-1] if index else 0))))
    return panel


def _panel_identity(panel, split):
    files = {str(path): _digest(path) for path, _ in panel}
    seeds = {}
    for path, _ in panel:
        if str(path) not in seeds:
            with np.load(path, allow_pickle=False) as data:
                seeds[str(path)] = int(data['seed'])
    specification = {'split': split, 'frames': [{'episode': path.name, 'frame': frame,
        'source_sha256': files[str(path)], 'seed': seeds[str(path)]} for path, frame in panel]}
    return {**specification, 'fingerprint': hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()}


def _decode(renderer, latent):
    return renderer.codec.decode(renderer.codec.quantize(renderer.denormalize_latents(latent))['quantized'])


def prepare_teacher_buffer(teacher, directory, *, count=512, batch_size=16, seed=173,
                           teacher_fingerprint, cache_path=None, progress=None):
    """Cache full-sampler continuous endpoints and the SAME noise per pose.

    Runtime fixed noise is intentional: both the reference renderer and the
    eventual deterministic student use that exact realization. Continuous
    endpoints are never quantized/re-encoded as the velocity training target.
    """
    directory = Path(directory).resolve()
    panel = select_pose_panel(directory, count, seed, image_size=teacher.config.image_size)
    identity = _panel_identity(panel, directory.name)
    identity.update(teacher_fingerprint=teacher_fingerprint, teacher_steps=teacher.flow_renderer.steps,
        noise_policy='shared_runtime_fixed_noise', scope='fixed_pose_navigation_renderer', format_version=1)
    if cache_path is not None and Path(cache_path).exists():
        saved = torch.load(cache_path, map_location='cpu', weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('Existing teacher target buffer identity changed')
        return saved
    device = next(teacher.parameters()).device
    modes = [(module, module.training) for module in teacher.modules()]
    tensors = {name: [] for name in ('state', 'map', 'base', 'base_latents', 'noise', 'endpoint', 'teacher_rgb', 'source_rgb')}
    try:
        teacher.eval()
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
        with torch.inference_mode(), strict_render_precision(), torch.random.fork_rng(devices=devices), \
                torch.autocast(device.type, enabled=False):
            for first in range(0, count, batch_size):
                selected = panel[first:first+batch_size]
                states, grids, source = [], [], []
                for path, frame in selected:
                    with np.load(path, allow_pickle=False) as saved:
                        state = saved['states'][frame]
                        npc = state[GLOBAL_DIM:-2].reshape(ENTITY_SLOTS, ENTITY_DIM)
                        if np.any(npc[:, 5] >= .5):
                            raise ValueError('Navigation renderer distillation requires inactive NPC source poses')
                        states.append(state)
                        grids.append(saved['map'])
                        source.append(saved['frames'][frame])
                state = torch.tensor(np.stack(states), device=device, dtype=torch.float32)
                grid = torch.tensor(np.stack(grids), device=device, dtype=torch.long)
                source_rgb = torch.tensor(np.stack(source), device=device).permute(0, 3, 1, 2).float()/255
                features = teacher.encode_map(grid)
                base = teacher.render_base(state, state.new_zeros(len(state), 128), features)
                flow = teacher.flow_renderer
                noise = flow._noise(flow.fixed_noise, len(state), device).clone()
                endpoint = flow.latent_sample(base, state, noise=noise, map_condition=grid)
                rgb = _decode(flow, endpoint)
                values = dict(state=state, map=grid, base=base,
                    base_latents=flow.normalize_latents(flow.codec.encode(base)['quantized']),
                    noise=noise, endpoint=endpoint, teacher_rgb=rgb, source_rgb=source_rgb)
                if not all(bool(torch.isfinite(value).all()) for value in values.values()):
                    raise FloatingPointError('Teacher buffer contains non-finite conditions/targets')
                for name, value in values.items():
                    tensors[name].append(value.detach().cpu())
                if progress:
                    progress(min(count, first+len(selected)), count, directory.name)
    finally:
        for module, mode in modes:
            module.training = mode
    result = {'identity': identity, **{name: torch.cat(values) for name, values in tensors.items()}}
    if cache_path is not None:
        _save(cache_path, result)
    return result


def reflow_loss(student, batch, *, time_values=None, self_condition=True, decoded_weight=.1):
    """Fit a straight path from cached noise to its exact teacher endpoint."""
    target, noise = batch['endpoint'].detach().float(), batch['noise'].detach().float()
    if target.shape != noise.shape or target.ndim != 4:
        raise ValueError('Same-noise reflow needs matching continuous latent endpoint/noise')
    time_values = (student.draw_training_time(len(target), target.device) if time_values is None
                   else time_values.to(target.device).float())
    if time_values.shape != (len(target),) or not bool(torch.isfinite(time_values).all()):
        raise ValueError('Reflow times must be finite and batch aligned')
    fraction = time_values.clamp(.001, .999)[:, None, None, None]
    noisy = (1-fraction)*noise+fraction*target
    conditions = (batch['base_latents'].detach(), batch['state'].detach())
    maps = batch['map'].detach()
    previous = None
    if self_condition and bool(torch.rand((), device=target.device) < .5):
        with torch.no_grad():
            previous = noisy+(1-fraction)*student.velocity(noisy, time_values, *conditions, map_condition=maps).float()
    prediction = student.velocity(noisy, time_values, *conditions, previous, map_condition=maps).float()
    endpoint = noisy+(1-fraction)*prediction
    velocity = F.mse_loss(prediction, target-noise)
    endpoint_loss = F.l1_loss(endpoint, target)
    decoded = F.l1_loss(_decode(student, endpoint), batch['teacher_rgb'].detach()) if decoded_weight else endpoint_loss*0
    total = velocity+.1*endpoint_loss+decoded_weight*decoded
    return total, {'reflow_velocity': float(velocity.detach()), 'reflow_endpoint': float(endpoint_loss.detach()),
        'reflow_decoded_l1': float(decoded.detach()), 'train_loss': float(total.detach())}


def one_step_reflow_loss(student, batch, *, decoded_weight=.1, ssim_weight=.05, edge_weight=.05):
    """Teach the actual single Euler evaluation at t=0, with no intermediate t.

    The continuous original CFM endpoint and its exact cached noise define
    the velocity target. The frozen compact decoder supplies image gradients;
    Base, state, map, targets and codec parameters remain outside optimization.
    """
    from .object_losses import decoded_ssim_loss, balanced_edge_l1
    if student.steps != 1 or student.kind != 'student':
        raise ValueError('One-step reflow requires a one-Euler-step student')
    target, noise = batch['endpoint'].detach().float(), batch['noise'].detach().float()
    if target.shape != noise.shape or target.ndim != 4:
        raise ValueError('One-step reflow requires matched continuous teacher endpoint/noise')
    time_values = torch.zeros(len(target), device=target.device, dtype=torch.float32)
    prediction = student.velocity(noise, time_values, batch['base_latents'].detach(),
        batch['state'].detach(), None, map_condition=batch['map'].detach()).float()
    endpoint = noise+prediction
    velocity = F.mse_loss(prediction, target-noise)
    endpoint_loss = F.l1_loss(endpoint, target)
    decoded_loss = ssim_loss = edge_loss = endpoint_loss*0
    if decoded_weight or ssim_weight or edge_weight:
        decoded = _decode(student, endpoint)
        teacher_rgb = batch['teacher_rgb'].detach()
        if decoded_weight:
            decoded_loss = F.l1_loss(decoded, teacher_rgb)
        if ssim_weight:
            ssim_loss = decoded_ssim_loss(decoded, teacher_rgb)
        if edge_weight:
            edge_loss = balanced_edge_l1(decoded, teacher_rgb)
    total = velocity+.1*endpoint_loss+decoded_weight*decoded_loss+ssim_weight*ssim_loss+edge_weight*edge_loss
    return total, {'reflow_velocity': float(velocity.detach()), 'reflow_endpoint': float(endpoint_loss.detach()),
        'reflow_decoded_l1': float(decoded_loss.detach()), 'reflow_decoded_ssim': float(ssim_loss.detach()),
        'reflow_decoded_edge': float(edge_loss.detach()), 'flow_time_mean': 0., 'solver_steps': 1,
        'velocity_evaluations': 1, 'train_loss': float(total.detach())}


def _batch(buffer, indices, device):
    return {name: value[indices].to(device) for name, value in buffer.items() if torch.is_tensor(value)}


def _restore_random_state(saved, device):
    """Restore device-mapped checkpoint RNG tensors through CPU ByteTensors."""
    torch.random.set_rng_state(saved['rng_torch'].cpu())
    np.random.set_state(saved['rng_numpy'])
    random.setstate(saved['rng_python'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state_all([value.cpu() for value in saved['rng_cuda']])


def prepare_student_conditions(teacher, student, renderer_buffer, *, batch_size=16,
                               base_checkpoint=None, cache_path=None, allow_unqualified_codec=False):
    """Change only student Base/encoder conditions; teacher labels stay exact.

    The original immutable teacher buffer remains valid provenance. A compact
    Base and compact encoder need their own cache identity and cannot reuse
    the original teacher's encoded Base features.
    """
    from .renderer_student import codec_tensor_fingerprint
    conditioning = {'format_version': 1, 'student_codec_fingerprint': codec_tensor_fingerprint(student.codec),
        'student_config': getattr(student, '_renderer_student_config', None),
        'allow_unqualified_codec': bool(allow_unqualified_codec),
        'base_checkpoint_sha256': _digest(base_checkpoint) if base_checkpoint is not None else None,
        'base_features_policy': 'GT_pose_zero_128_appearance', 'teacher_targets_modified': False}
    identity = {**renderer_buffer['identity'], 'student_conditioning': conditioning}
    identity['fingerprint'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if cache_path is not None and Path(cache_path).exists():
        result = torch.load(cache_path, map_location='cpu', weights_only=False)
        if result['identity'] != identity:
            raise ValueError('Student condition cache codec/Base/configuration changed')
        return result
    device = next(student.parameters()).device
    if base_checkpoint is not None and not hasattr(teacher, 'base_student'):
        raise ValueError('Compact Base must be validated and attached before encoding its conditions')
    rows, latents = [], []
    with torch.inference_mode(), strict_render_precision(), torch.autocast(device.type, enabled=False):
        for first in range(0, len(renderer_buffer['state']), batch_size):
            batch = _batch(renderer_buffer, slice(first, first+batch_size), device)
            if base_checkpoint is not None:
                features = teacher.encode_map(batch['map'])
                base = teacher.render_base(batch['state'], batch['state'].new_zeros(len(batch['state']), 128), features)
            else:
                base = batch['base']
            encoded = student.normalize_latents(student.codec.encode(base)['quantized'])
            if not bool(torch.isfinite(base).all() and torch.isfinite(encoded).all()):
                raise FloatingPointError('Nonfinite compact Base/encoder conditions')
            rows.append(base.cpu()); latents.append(encoded.cpu())
    result = {**renderer_buffer, 'identity': identity, 'base': torch.cat(rows), 'base_latents': torch.cat(latents)}
    if cache_path is not None:
        _save(cache_path, result)
    return result


def evaluate_renderer_student(student, buffer, *, batch_size=16, step=0, ssim=.95, mae=.015):
    """Full deterministic Euler student versus held-out full Heun teacher RGB."""
    if buffer['identity']['split'] != 'val':
        raise ValueError('Renderer quality must use a held-out val buffer')
    device = next(student.parameters()).device
    modes = [(module, module.training) for module in student.modules()]
    values, reference_values, source_values, previews = [], [], [], []
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        student.eval()
        with torch.inference_mode(), strict_render_precision(), torch.random.fork_rng(devices=devices), \
                torch.autocast(device.type, enabled=False):
            for first in range(0, len(buffer['state']), batch_size):
                sample = _batch(buffer, slice(first, first+batch_size), device)
                rgb = student.sample(sample['base'], sample['state'], noise=sample['noise'], map_condition=sample['map'])
                if not bool(torch.isfinite(rgb).all()) or bool((rgb < 0).any() or (rgb > 1).any()):
                    raise FloatingPointError('Student produced invalid RGB during validation')
                measured = image_quality_metrics(rgb.float(), sample['teacher_rgb'].float())
                reference = image_quality_metrics(sample['teacher_rgb'].float(), sample['source_rgb'].float())
                source_measured = image_quality_metrics(rgb.float(), sample['source_rgb'].float())
                values.extend({name: float(value[index]) for name, value in measured.items()} for index in range(len(rgb)))
                reference_values.extend({name: float(value[index]) for name, value in reference.items()} for index in range(len(rgb)))
                source_values.extend({name: float(value[index]) for name, value in source_measured.items()} for index in range(len(rgb)))
                for index in range(min(len(rgb), max(0, 4-len(previews)))):
                    previews.append(torch.stack([sample['source_rgb'][index], sample['base'][index],
                                                 sample['teacher_rgb'][index], rgb[index]]).cpu())
    finally:
        for module, mode in modes:
            module.training = mode
    summary = {name+'_mean': float(np.mean([row[name] for row in values])) for name in ('ssim', 'mae', 'psnr')}
    summary['ssim_p05'] = float(np.quantile([row['ssim'] for row in values], .05))
    summary['mae_p95'] = float(np.quantile([row['mae'] for row in values], .95))
    passed = summary['ssim_mean'] >= ssim and summary['mae_mean'] <= mae
    report = {'step': step, 'frames': len(values), 'summary': summary, 'passed': passed,
        'teacher_vs_source': {name+'_mean': float(np.mean([row[name] for row in reference_values])) for name in ('ssim', 'mae', 'psnr')},
        'student_vs_source': {name+'_mean': float(np.mean([row[name] for row in source_values])) for name in ('ssim', 'mae', 'psnr')},
        'thresholds': {'ssim_mean': ssim, 'mae_mean': mae}, 'per_frame': values,
        'panel_fingerprint': buffer['identity']['fingerprint'], 'source_held_out': True,
        'scope': 'Student versus frozen teacher at held-out GT navigation poses; not world-rollout quality',
        'student_steps': student.steps, 'teacher_steps': buffer['identity']['teacher_steps']}
    return report, previews


def _preview(previews, path, step, solver_steps=4):
    size = 192
    canvas = Image.new('RGB', (size*len(previews), 4*(size+22)), (10, 15, 24))
    draw = ImageDraw.Draw(canvas)
    for column, images in enumerate(previews):
        for row, (image, label) in enumerate(zip(images, ('SOURCE GT pose', 'FROZEN BASE', 'TEACHER full Heun', f'STUDENT Euler{solver_steps}'))):
            pixels = image.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
            canvas.paste(Image.fromarray(pixels).resize((size, size), Image.Resampling.NEAREST), (column*size, row*(size+22)+22))
            draw.text((column*size+3, row*(size+22)+4), f'{label} | {step}', fill='white')
    temporary = Path(str(path)+'.tmp.png')
    canvas.save(temporary)
    replace_file(temporary, Path(path))


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary)
    replace_file(temporary, path)


class RendererDistillation:
    def __init__(self, args):
        self.args, self.root = args, Path(args.run_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        self.state = {'run_id': self.root.name, 'status': 'running', 'stage': 'renderer_distillation',
            'stage_label': 'Preparing frozen teacher targets', 'current': 0, 'total': args.steps,
            'teacher': str(Path(args.teacher).resolve()), 'scope': 'navigation_renderer_only',
            'world_training': False, 'started_at': now(), 'image_size': 32,
            'model': f'Frozen world + distilled {getattr(args, "solver_steps", 4)}-step RF renderer',
            'loss_protocol': ('renderer_single_step_reflow_v1' if getattr(args, 'solver_steps', 4) == 1
                              else 'renderer_reflow_distillation_v1'),
            'solver_steps': getattr(args, 'solver_steps', 4), 'prediction_every': args.preview_every,
            'validation_every': args.validate_every, 'global_training_step': 0}

    def publish(self, **values):
        self.state.update(values, updated_at=now(), elapsed_seconds=time.perf_counter()-self.started)
        atomic_json(self.root/'status.json', self.state)

    def run(self):
        from .renderer_student import build_renderer_student, export_renderer_student
        args, device = self.args, torch.device(self.args.device)
        torch.set_num_threads(int(os.environ.get('TW_TORCH_THREADS', '4')))
        if device.type == 'cuda':
            torch.cuda.init()
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        teacher, metadata = load_world_checkpoint(args.teacher, 'cpu')
        teacher_training_steps = int(metadata.get('training_steps', metadata.get('global_training_step', 0)))
        if teacher_training_steps < 1:
            raise ValueError('Renderer distillation requires a trained teacher with positive training_steps')
        teacher = teacher.to(device).requires_grad_(False).eval()
        fingerprint = _digest(args.teacher)
        codec_checkpoint, base_checkpoint = getattr(args, 'codec_checkpoint', None), getattr(args, 'base_checkpoint', None)
        allow_unqualified_codec = getattr(args, 'allow_unqualified_codec', False)
        if allow_unqualified_codec and codec_checkpoint is None:
            raise ValueError('Experimental codec opt-in requires an explicit codec checkpoint')
        data = Path(args.data_dir).resolve()
        if (data/'rgb').is_dir():
            data = data/'rgb'
        def progress(done, total, split):
            self.publish(stage='teacher_targets', stage_label='Preparing frozen teacher targets',
                data_preparation={'split': split, 'completed': done, 'total': total})
        base_parameters = teacher.vjepa_backend.renderer.renderer_parameter_count()
        shared_map_parameters = sum(p.numel() for p in teacher.vjepa_backend.renderer.map_encoder.parameters())
        shared_geometry_parameters = sum(p.numel() for p in teacher.vjepa_backend.renderer.geometry_reader.parameters())
        teacher_parameters = sum(p.numel() for p in teacher.parameters())
        teacher_flow_parameters = sum(p.numel() for p in teacher.flow_renderer.parameters())
        self.publish(teacher_training_steps=teacher_training_steps,
            model_parameters={'teacher_total': teacher_parameters,
            'teacher_renderer_total': base_parameters+teacher_flow_parameters,
            'shared_map_parameters': shared_map_parameters,
            'shared_geometry_parameters': shared_geometry_parameters})
        train = prepare_teacher_buffer(teacher, data/'train', count=args.train_count, batch_size=args.batch,
            seed=args.seed, teacher_fingerprint=fingerprint, cache_path=self.root/'teacher_train_buffer.pt', progress=progress)
        validation = prepare_teacher_buffer(teacher, data/'val', count=args.val_count, batch_size=args.batch,
            seed=args.seed+1, teacher_fingerprint=fingerprint, cache_path=self.root/'teacher_val_buffer.pt', progress=progress)
        # Attach the compact Base only AFTER the original full-teacher labels
        # have been prepared/verified. Never generate reference labels from it.
        if base_checkpoint is not None:
            from .base_student import load_base_student
            self.publish(base_student=load_base_student(teacher, base_checkpoint))
        # The student bundle must bind the approved Base identity at build
        # time, after original-teacher target generation and before training.
        options = {'codec_checkpoint': codec_checkpoint}
        if allow_unqualified_codec:
            options['allow_unqualified_codec'] = True
        solver_steps = getattr(args, 'solver_steps', 4)
        student = build_renderer_student(teacher, steps=solver_steps, seed=args.seed, **options).to(device).train()
        initial_renderer = getattr(args, 'initialize_renderer', None)
        initialization = {'mode': 'fresh_velocity_distillation', 'optimizer_reset': True,
                          'initial_optimizer_step': 0, 'seed': args.seed, 'solver_steps': solver_steps}
        if initial_renderer is not None:
            from .renderer_student import initialize_renderer_student
            source = initialize_renderer_student(teacher, student, initial_renderer,
                allow_unqualified_codec=allow_unqualified_codec)
            initialization.update(mode='renderer_weight_warmstart', source_renderer=str(Path(initial_renderer).resolve()),
                source_renderer_sha256=_digest(initial_renderer), source_training_steps=source['training_steps'],
                source_solver_steps=source['solver_steps'], target_solver_steps=solver_steps)
            # This is a new training phase, not continuation of the old Adam
            # moments or RNG stream. A resumed new-phase checkpoint restores
            # its own moments/RNG below after this deterministic initialization.
            random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        self.publish(initialization=initialization)
        codec_provenance = getattr(student, '_renderer_codec_distillation', None)
        if codec_provenance is not None:
            self.publish(codec_distillation=codec_provenance,
                codec_quality_gate_passed=codec_provenance['quality_gate']['passed'],
                experimental_preview=bool(allow_unqualified_codec and not codec_provenance['quality_gate']['passed']))
        optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr, weight_decay=.01)
        student_parameters = sum(p.numel() for p in student.parameters())
        self.state['model_parameters'].update(student_renderer_total=base_parameters+student_parameters,
            student_flow_including_shared_codec=student_parameters,
            hybrid_world_total=teacher_parameters-teacher_flow_parameters+student_parameters,
            student_trainable=sum(p.numel() for p in student.parameters() if p.requires_grad))
        if codec_checkpoint is not None or base_checkpoint is not None:
            train = prepare_student_conditions(teacher, student, train, batch_size=args.batch,
                base_checkpoint=base_checkpoint, cache_path=self.root/'student_train_conditions.pt',
                allow_unqualified_codec=allow_unqualified_codec)
            validation = prepare_student_conditions(teacher, student, validation, batch_size=args.batch,
                base_checkpoint=base_checkpoint, cache_path=self.root/'student_val_conditions.pt',
                allow_unqualified_codec=allow_unqualified_codec)
            head_parameters = sum(p.numel() for p in teacher.base_student.parameters()) if base_checkpoint is not None else 0
            self.state['model_parameters'].update(student_base_head_parameters=head_parameters,
                student_renderer_total=(head_parameters or base_parameters)+student_parameters,
                student_velocity_parameters=sum(p.numel() for name, p in student.named_parameters() if not name.startswith('codec.')),
                student_codec_parameters=sum(p.numel() for p in student.codec.parameters()),
                hybrid_world_total=teacher_parameters-teacher_flow_parameters+student_parameters+head_parameters,
                student_retained_renderer_modules_total=base_parameters+student_parameters+head_parameters)
        if set(row['source_sha256'] for row in train['identity']['frames']) & set(row['source_sha256'] for row in validation['identity']['frames']):
            raise ValueError('Teacher target train/validation source files overlap')
        if set(row['seed'] for row in train['identity']['frames']) & set(row['seed'] for row in validation['identity']['frames']):
            raise ValueError('Teacher target train/validation source seeds overlap')
        configuration = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
        identities = {'teacher': fingerprint, 'train': train['identity'], 'val': validation['identity']}
        if initial_renderer is not None:
            identities['initial_renderer_sha256'] = initialization['source_renderer_sha256']
        checkpoint = self.root/'checkpoints/renderer_student_last.pt'
        step, best, streak = 0, None, 0
        initial_budget, budget_history = args.steps, []
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location=device, weights_only=False)
            previous_configuration = saved['configuration']
            previous_budget = previous_configuration['steps']
            if ({name: value for name, value in previous_configuration.items() if name != 'steps'} !=
                    {name: value for name, value in configuration.items() if name != 'steps'}
                    or args.steps < previous_budget or saved['identities'] != identities):
                raise ValueError('Renderer distillation resume configuration/teacher/data changed')
            student.load_state_dict(saved['model'], strict=True); optimizer.load_state_dict(saved['optimizer'])
            step, best, streak = saved['step'], saved['best'], saved['streak']
            initial_budget = saved.get('initial_budget', previous_budget)
            budget_history = saved.get('budget_history', [])
            if args.steps > previous_budget:
                budget_history.append({'from_steps': previous_budget, 'to_steps': args.steps,
                    'at_optimizer_step': step, 'requested_at': now()})
            _restore_random_state(saved, device)
        def save():
            _save(checkpoint, {'model': student.state_dict(), 'optimizer': optimizer.state_dict(),
                'step': step, 'best': best, 'streak': streak, 'configuration': configuration, 'identities': identities,
                'initial_budget': initial_budget, 'budget_history': budget_history,
                'initialization': initialization,
                'rng_torch': torch.random.get_rng_state(), 'rng_numpy': np.random.get_state(),
                'rng_python': random.getstate(), 'rng_cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []})
        def validate(full):
            selected = validation if full else {**validation, **{name: value[:4] for name, value in validation.items() if torch.is_tensor(value)}}
            report, previews = evaluate_renderer_student(student, selected, batch_size=args.batch, step=step,
                ssim=args.target_ssim, mae=args.target_mae)
            _preview(previews, self.root/'preview.png', step, student.steps)
            atomic_json(self.root/'preview.json', {'step': step, 'every': args.preview_every,
                'scope': report['scope'], 'rows': ['source', 'base', 'teacher', 'student'],
                'solver': 'euler', 'solver_steps': student.steps, 'loss_protocol': self.state['loss_protocol'],
                'quality_gate_evaluated': full})
            return report
        self.publish(initial_budget=initial_budget, budget_history=budget_history,
            resumed_from_optimizer_step=step, global_training_step=step, current=step)
        if step == 0:
            initial = validate(True)
            atomic_json(self.root/'validation_initial.json', initial)
        training_started = time.perf_counter()
        first_step = step
        for step in range(step+1, args.steps+1):
            student.train()
            indices = np.random.randint(0, len(train['state']), size=args.batch)
            sample = _batch(train, indices, device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = (one_step_reflow_loss(student, sample) if solver_steps == 1
                             else reflow_loss(student, sample))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite reflow loss')
            loss.backward()
            metrics['gradient_norm'] = float(clip_finite_grad_norm_(student.named_parameters(), context=f'Reflow update{step}'))
            optimizer.step()
            report = None
            if step % args.validate_every == 0 or step == args.steps:
                report = validate(True)
                atomic_json(self.root/'validation.json', report)
                score = report['summary']['mae_mean']+.5*(1-report['summary']['ssim_mean'])
                if best is None or (not report['passed'], score) < (not best['report']['passed'], best['score']):
                    best = {'score': score, 'step': step, 'report': report}
                    _save(self.root/'checkpoints/renderer_student_best.pt', {'model': student.state_dict(), **best})
                streak = streak+1 if report['passed'] else 0
            elif step % args.preview_every == 0:
                validate(False)
            with (self.root/'metrics.jsonl').open('a', encoding='utf-8') as log:
                log.write(json.dumps({'timestamp': now(), 'step': step, 'global_training_step': step,
                    'stage': 'renderer_distillation', 'loss_protocol': self.state['loss_protocol'],
                    'solver_steps': solver_steps, **metrics, **({'validation': report['summary']} if report else {})})+'\n')
            self.publish(stage='renderer_distillation', stage_label='Training fast renderer', current=step,
                global_training_step=step, loss=metrics['train_loss'], metrics=metrics,
                validation=report if report else self.state.get('validation'), best=best,
                progress=step/args.steps,
                eta_seconds=(time.perf_counter()-training_started)/(step-first_step)*(args.steps-step))
            if step % args.save_every == 0 or report is not None:
                save()
            if streak >= args.required_passes:
                break
        save()
        selected = torch.load(self.root/'checkpoints/renderer_student_best.pt', map_location=device, weights_only=False)
        student.load_state_dict(selected['model'], strict=True)
        final, previews = evaluate_renderer_student(student, validation, batch_size=args.batch,
            step=selected['step'], ssim=args.target_ssim, mae=args.target_mae)
        _preview(previews, self.root/'preview.png', selected['step'], student.steps)
        exported = export_renderer_student(teacher, student, self.root/'checkpoints/renderer_student.pt',
            training_steps=selected['step'], metrics=final)
        atomic_json(self.root/'validation_selected.json', final)
        self.publish(status='completed', stage='renderer_distilled', stage_label='Fast renderer training finished',
            current=step, selected_step=selected['step'], quality_gate_passed=final['passed'],
            renderer_imitation_quality_gate_passed=final['passed'],
            validation=final, export=exported, world_quality_qualified=False, eta_seconds=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--codec-checkpoint', type=Path)
    parser.add_argument('--base-checkpoint', type=Path)
    parser.add_argument('--solver-steps', type=int, choices=(1, 4), default=4)
    parser.add_argument('--initialize-renderer', type=Path,
                        help='Validated trained renderer weights; starts a new phase with fresh Adam/RNG')
    parser.add_argument('--allow-unqualified-codec', action='store_true',
                        help='Explicit experimental preview with the codec quality gate still reported as failed')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    for name, default in [('steps', 1000), ('train-count', 512), ('val-count', 128), ('batch', 16),
                          ('validate-every', 100), ('preview-every', 10), ('save-every', 10), ('required-passes', 2), ('seed', 173)]:
        parser.add_argument('--'+name, type=int, default=default)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--target-ssim', type=float, default=.95)
    parser.add_argument('--target-mae', type=float, default=.015)
    args = parser.parse_args(argv)
    if min(args.steps, args.train_count, args.val_count, args.batch, args.validate_every,
           args.preview_every, args.save_every, args.required_passes) < 1 or args.lr <= 0:
        parser.error('Counts, cadence and learning rate must be positive')
    experiment = RendererDistillation(args)
    try:
        with KeepAwake():
            experiment.run()
    except BaseException as error:
        experiment.publish(status='failed', stage='error', stage_label='Renderer distillation failed',
            error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
