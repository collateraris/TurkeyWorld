"""Train a fresh learned x4 upscaler on genuinely paired source LR/HR RGB."""
from contextlib import nullcontext
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from .models import SingleFrameUpscaler
from .power_guard import KeepAwake
from .sr_data import SRPairData
from .training_pipeline import atomic_json, now, replace_file
from .visual_quality import image_quality_metrics, strict_render_precision


def _pixels(images):
    return images.detach().float().clamp(0, 1).mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def differentiable_ssim_loss(prediction, target):
    """Canonical Gaussian11/sigma1.5 RGB SSIM formula, FP32 for training.

    Validation uses visual_quality.image_quality_metrics' FP64 implementation.
    The window, valid convolution, constants and averaging are identical.
    """
    if prediction.shape != target.shape or prediction.ndim != 4 or min(prediction.shape[-2:]) < 11:
        raise ValueError('SSIM loss requires matching RGB images of at least11 pixels')
    with torch.autocast(prediction.device.type, enabled=False):
        first, second = prediction.float(), target.detach().float()
        axis = torch.arange(11, device=first.device, dtype=first.dtype)-5
        gaussian = torch.exp(-axis.square()/(2*1.5**2))
        gaussian /= gaussian.sum()
        window = (gaussian[:, None]*gaussian[None]).expand(3, 1, 11, 11).contiguous()
        mu1, mu2 = F.conv2d(first, window, groups=3), F.conv2d(second, window, groups=3)
        var1 = (F.conv2d(first.square(), window, groups=3)-mu1.square()).clamp_min(0)
        var2 = (F.conv2d(second.square(), window, groups=3)-mu2.square()).clamp_min(0)
        covariance = F.conv2d(first*second, window, groups=3)-mu1*mu2
        ssim = ((2*mu1*mu2+.01**2)*(2*covariance+.03**2)
                /((mu1.square()+mu2.square()+.01**2)*(var1+var2+.03**2)))
        return 1-ssim.flatten(1).mean(-1).clamp(-1, 1).mean()


def load_sr_codec(path, device, image_size):
    """Verify a qualified or explicitly user-accepted immutable codec once."""
    from .vq_world import _load_codec
    saved, codec, fingerprint = _load_codec(path, require_qualified=True,
        accept_user_requested=True, expected_image_size=image_size)
    return codec.to(device).eval().requires_grad_(False), {
        'checkpoint': str(Path(path).resolve()), 'codec_fingerprint': fingerprint,
        'training_steps': int(saved.get('training_steps', 0)),
        'quality_gate_passed': bool(saved.get('quality_gate', {}).get('passed', False)),
        'accepted_by_user': bool(saved.get('quality_acceptance', {}).get('accepted', False)),
        'frozen': True, 'source': 'source_LR_codec_reconstruction', 'world_generated_rgb': False}


def evaluate_sr(model, data, panel, *, batch_size=8, step=0, codec=None, input_kind='clean_source_lr'):
    """Strict FP32 held-out single-frame SR, never a world-rollout qualification."""
    if data.split != 'val' or batch_size < 1 or not panel or len(set(map(tuple, panel))) != len(panel):
        raise ValueError('SR evaluation requires unique explicit held-out val frames')
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError('SR validation requires FP32 model weights')
    if input_kind not in ('clean_source_lr', 'codec_reconstruction') or (input_kind == 'codec_reconstruction' and codec is None):
        raise ValueError('Decoded SR validation requires an explicitly supplied frozen codec')
    modes = [(module, module.training) for network in (model, codec) if network is not None for module in network.modules()]
    rng_state = copy.deepcopy(data.rng.bit_generator.state)
    global_numpy, global_python = np.random.get_state(), random.getstate()
    device = data.device
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    rows, previews, content = [], [], hashlib.sha256()
    try:
        model.eval()
        if codec is not None:
            codec.eval()
        with strict_render_precision(), torch.random.fork_rng(devices=cuda_devices), \
                torch.inference_mode(), torch.autocast(device.type, enabled=False):
            for start in range(0, len(panel), batch_size):
                chosen = panel[start:start+batch_size]
                batch = data.batch(len(chosen), indices=chosen)
                network_input = (codec(batch['lr'])['reconstruction'].float()
                                 if input_kind == 'codec_reconstruction' else batch['lr'])
                predicted = model(network_input).float()
                if predicted.shape != batch['hr'].shape or not bool(torch.isfinite(predicted).all()):
                    raise FloatingPointError('SR validation returned invalid or nonfinite RGB')
                if bool((predicted < 0).any() or (predicted > 1).any()):
                    raise FloatingPointError('SR validation RGB lies outside[0,1]')
                baseline = F.interpolate(network_input, size=predicted.shape[-2:], mode='bilinear', align_corners=False)
                values = image_quality_metrics(predicted, batch['hr'])
                reference = image_quality_metrics(baseline, batch['hr'])
                for offset, (episode, frame) in enumerate(chosen):
                    for key in ('lr', 'hr'):
                        content.update(batch[key][offset].cpu().contiguous().numpy().tobytes())
                    rows.append({'episode': data.episodes[int(episode)]['path'].name, 'frame': int(frame),
                        **{key: float(value[offset]) for key, value in values.items()},
                        'bilinear_psnr': float(reference['psnr'][offset]),
                        'bilinear_mae': float(reference['mae'][offset])})
                    if len(previews) < 4:
                        previews.append((network_input[offset].cpu(), predicted[offset].cpu(), batch['hr'][offset].cpu()))
    finally:
        for module, training in modes:
            module.training = training
        data.rng.bit_generator.state = rng_state
        np.random.set_state(global_numpy)
        random.setstate(global_python)
    keys = ('psnr', 'ssim', 'mae', 'bilinear_psnr', 'bilinear_mae')
    report = {'kind': 'single_frame_super_resolution', 'step': int(step), 'frames': len(rows),
        'source_held_out': True, 'source_input_allowed_for_sr': True, 'world_rollout': False,
        'input_kind': input_kind, 'primary_quality_metric': input_kind == 'clean_source_lr',
        'render_precision': 'fp32', 'image_size': data.image_size, 'output_size': data.image_size*data.scale,
        'summary': {key+'_mean': float(np.mean([row[key] for row in rows])) for key in keys},
        'per_frame': rows, 'panel_fingerprint': data.panel_fingerprint(panel),
        'source_content_sha256': content.hexdigest(),
        'scope': 'Held-out source LR to paired source HR; no autonomous-world quality claim'}
    preview = {key: _pixels(torch.stack([row[index] for row in previews]))
               for index, key in enumerate(('source_lr', 'sr', 'source_hr'))}
    preview['difference'] = np.abs(preview['sr'].astype(np.int16)-preview['source_hr'].astype(np.int16)).astype(np.uint8)
    preview['metadata'] = {key: report[key] for key in ('kind', 'step', 'source_held_out', 'world_rollout',
        'source_input_allowed_for_sr', 'image_size', 'output_size', 'panel_fingerprint', 'input_kind')}
    return report, preview


def save_sr_preview(preview, path, *, every=100):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new('RGB', (512, 576), (14, 22, 31))
    draw = ImageDraw.Draw(canvas)
    input_label = 'CODEC RECONSTRUCTED LR input' if preview['metadata'].get('input_kind') == 'codec_reconstruction' else 'SOURCE LR input'
    labels = (input_label+' | enlarged for display', 'LEARNED SR x4', 'PAIRED SOURCE HR target', 'ABSOLUTE DIFFERENCE x4')
    for row, (key, label) in enumerate(zip(('source_lr', 'sr', 'source_hr', 'difference'), labels)):
        top = row*144
        draw.text((4, top+3), label, fill=(235, 239, 245))
        for column, pixels in enumerate(preview[key][:4]):
            if key == 'difference':
                pixels = np.clip(pixels.astype(np.int16)*4, 0, 255).astype(np.uint8)
            canvas.paste(Image.fromarray(pixels).resize((128, 128), Image.Resampling.NEAREST), (column*128, top+16))
    temporary = path.with_name(path.stem+'.tmp.png')
    canvas.save(temporary, format='PNG')
    replace_file(temporary, path)
    metadata = {**preview['metadata'], 'path': str(path), 'generated_at': now(), 'every': every,
        'native_lr_size': [preview['source_lr'].shape[2], preview['source_lr'].shape[1]],
        'native_hr_size': [preview['source_hr'].shape[2], preview['source_hr'].shape[1]],
        'preview_image_size': [512, 576], 'caption': 'Held-out Source LR / learned SR x4 / paired Source HR / difference x4'}
    atomic_json(path.with_suffix('.json'), metadata)
    return metadata


class SRExperiment:
    def __init__(self, args):
        defaults = {'architecture': 'legacy', 'res_scale': .1, 'target_ssim': None,
            'quality_patience': 10, 'quality_min_delta': .0005, 'lr_min': 1e-6,
            'codec_checkpoint': None, 'codec_fraction': .25, 'ssim_weight': 0.}
        for name, value in defaults.items():
            if not hasattr(args, name):
                setattr(args, name, value)
        self.args, self.root, self.device = args, args.run_dir.resolve(), torch.device(args.device)
        self.root.mkdir(parents=True, exist_ok=True)
        self.step, self.examples_trained, self.elapsed_before = 0, 0, 0.
        self.started = time.time()
        self.rng = np.random.default_rng(args.seed)
        self.validation_step = None
        self.codec, self.codec_info, self.scheduler = None, None, None
        self.quality = {'best_ssim': None, 'best_step': None, 'anchor_ssim': None,
            'bad_checks': 0, 'target_streak': 0, 'last_checked_step': None, 'decision': 'continue'}
        self.state = {'run_id': self.root.name, 'status': 'running', 'stage': 'sr_initializing',
            'stage_label': 'SR x4: fresh paired-source training', 'started_at': now(), 'current': 0,
            'total': args.steps, 'progress': 0., 'prediction_every': args.preview_every,
            'prediction_preview': None, 'sr_training': {'from_scratch': True, 'pretrained_weights_loaded': False,
                'input_size': args.image_size, 'output_size': args.image_size*4, 'scale': 4,
                'world_rollout_qualified': False, 'requested_updates': args.steps,
                'target_ssim': args.target_ssim, 'target_scope': 'clean_source_lr_fixed_held_out_mean_ssim'}}
        self.update()

    def elapsed(self):
        return self.elapsed_before+time.time()-self.started

    def update(self, **fields):
        self.state.update(current=self.step, updated_at=now(), elapsed_seconds=round(self.elapsed(), 1),
                          progress=min(1., self.step/self.args.steps), **fields)
        atomic_json(self.root/'status.json', self.state)

    def protocol(self):
        result = {'image_size': self.args.image_size, 'scale': 4, 'data_directory': str(self.args.data_dir.resolve()),
                'target_field': 'hr_frames', 'paired_source_rgb': True, 'pixel_weight': 1., 'edge_weight': self.args.edge_weight}
        if self.args.ssim_weight:
            result['ssim_weight'] = self.args.ssim_weight
        if self.codec_info is not None:
            result['codec_augmentation'] = {'codec_fingerprint': self.codec_info['codec_fingerprint'],
                'fraction': self.args.codec_fraction, 'source': 'source_LR_codec_reconstruction', 'world_generated_rgb': False}
        return result

    def quality_config(self):
        if self.args.target_ssim is None:
            return None
        return {'target_ssim': self.args.target_ssim, 'required_passes': 2,
            'patience': self.args.quality_patience, 'min_delta': self.args.quality_min_delta,
            'lr_min': self.args.lr_min, 'scheduler_factor': .5,
            'scheduler_patience': max(1, self.args.quality_patience//3)}

    def training_inputs(self, clean):
        if self.codec is None or self.args.codec_fraction == 0:
            return clean, 0.
        selected = self.rng.random(len(clean)) < self.args.codec_fraction
        if not selected.any():
            return clean, 0.
        indices = torch.as_tensor(np.flatnonzero(selected), device=clean.device)
        with torch.no_grad(), torch.autocast(clean.device.type, enabled=False):
            decoded = self.codec(clean[indices])['reconstruction'].float()
        augmented = clean.clone()
        augmented[indices] = decoded
        return augmented, float(selected.mean())

    def save(self):
        path = self.root/'checkpoints'/'sr_last.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'format_version': 1, 'stage': 'sr', 'config': self.model.config,
            'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
            'input_size': self.args.image_size, 'output_size': self.args.image_size*4, 'scale': 4,
            'training_steps': self.step, 'examples_trained': self.examples_trained,
            'training_protocol': self.protocol(), 'rng_numpy': self.rng.bit_generator.state,
            'rng_torch': torch.random.get_rng_state(),
            'rng_cuda': torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else [],
            'elapsed_seconds': self.elapsed(), 'validation': self.state.get('sr_validation'),
            'validation_codec': self.state.get('sr_validation_codec'), 'codec_augmentation': self.codec_info,
            'quality_config': self.quality_config(), 'quality_state': self.quality,
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
            'pretrained_sr_weights_loaded': False, 'quality_status': 'Held-out SR metrics; autonomous world remains unverified',
            'created_at': now()}
        temporary = path.with_suffix('.pt.tmp')
        torch.save(payload, temporary)
        replace_file(temporary, path)
        self.state['checkpoint'] = {'name': str(path), 'step': self.step, 'created_at': payload['created_at']}
        return path

    def export_ready(self):
        completed_budget = self.step >= self.args.steps
        reason = self.quality['decision'] if self.args.target_ssim is not None else 'budget'
        if reason == 'continue' and completed_budget:
            reason = 'budget'
        if not completed_budget and reason not in ('target', 'plateau'):
            raise ValueError('Only completed SR training with held-out validation can be exported')
        if self.validation_step != self.step:
            raise ValueError('SR export requires current held-out validation')
        selected_step = self.step
        if self.args.target_ssim is not None:
            selected = torch.load(self.root/'checkpoints'/'sr_best.pt', map_location='cpu', weights_only=False)
            if selected['config'] != self.model.config or selected['training_protocol'] != self.protocol():
                raise ValueError('Best SR checkpoint does not match the current model or source protocol')
            # The full latest optimizer/RNG checkpoint has already been saved.
            # Only final inference/model state switches to the selected best.
            self.model.load_state_dict(selected['model'])
            selected_step = int(selected['training_steps'])
            self.state['sr_last_validation'] = self.state['sr_validation']
            atomic_json(self.root/'sr_validation_last.json', self.state['sr_validation'])
            if self.state.get('sr_validation_codec') is not None:
                atomic_json(self.root/'sr_validation_codec_last.json', self.state['sr_validation_codec'])
            self.state['sr_validation'] = selected['validation']
            self.state['sr_validation_codec'] = selected.get('validation_codec')
            atomic_json(self.root/'sr_validation.json', self.state['sr_validation'])
            if self.state['sr_validation_codec'] is not None:
                atomic_json(self.root/'sr_validation_codec.json', self.state['sr_validation_codec'])
            _, preview = evaluate_sr(self.model, self.validation, self.panel[:4],
                batch_size=self.args.validation_batch, step=selected_step)
            self.state['prediction_preview'] = save_sr_preview(preview, self.root/'preview.png', every=self.args.preview_every)
            if self.codec is not None:
                _, decoded_preview = evaluate_sr(self.model, self.validation, self.panel[:4],
                    batch_size=self.args.validation_batch, step=selected_step, codec=self.codec, input_kind='codec_reconstruction')
                self.state['prediction_preview_codec'] = save_sr_preview(decoded_preview,
                    self.root/'preview_codec.png', every=self.args.preview_every)
        target_met = bool(self.args.target_ssim is not None and reason == 'target'
                          and self.state['sr_validation']['summary']['ssim_mean'] >= self.args.target_ssim)
        path = self.root/'checkpoints'/'sr_for_game.pt'
        payload = {'format_version': 1, 'stage': 'sr', 'config': self.model.config,
            'model': self.model.state_dict(), 'training_steps': selected_step, 'last_training_step': self.step,
            'input_size': self.args.image_size, 'output_size': self.args.image_size*4, 'scale': 4,
            'training_completed': True, 'requested_training_steps': self.args.steps,
            'examples_trained': self.examples_trained, 'training_protocol': self.protocol(),
            'validation': self.state['sr_validation'], 'pretrained_sr_weights_loaded': False,
            'validation_codec': self.state.get('sr_validation_codec'), 'codec_augmentation': self.codec_info,
            'target_ssim': self.args.target_ssim, 'target_met': target_met, 'completed_budget': completed_budget,
            'completion_reason': reason, 'quality_config': self.quality_config(), 'quality_state': self.quality,
            'quality_status': f'SR training finished by {reason}; selected held-out metrics recorded, autonomous world unverified',
            'created_at': now()}
        temporary = path.with_suffix('.pt.tmp')
        torch.save(payload, temporary)
        replace_file(temporary, path)
        atomic_json(self.root/'sr_ready.json', {'checkpoint': str(path), 'training_steps': selected_step,
            'last_training_step': self.step, 'target_ssim': self.args.target_ssim, 'target_met': target_met,
            'completed_budget': completed_budget, 'completion_reason': reason,
            'validation': self.state['sr_validation']['summary'],
            'validation_codec': (self.state['sr_validation_codec']['summary'] if self.state.get('sr_validation_codec') else None),
            'requested_training_steps': self.args.steps, 'input_size': self.args.image_size,
            'output_size': self.args.image_size*4, 'scale': 4, 'training_completed': True,
            'pretrained_sr_weights_loaded': False, 'created_at': payload['created_at']})
        self.state['sr_training'].update(target_met=target_met, completion_reason=reason,
            completed_budget=completed_budget, selected_step=selected_step)
        return path

    def restore(self):
        path = self.root/'checkpoints'/'sr_last.pt'
        if not path.exists():
            return
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['stage'] != 'sr' or saved['config'] != self.model.config or saved['training_protocol'] != self.protocol():
            raise ValueError('SR model/native source resolution/objective or dataset changed on resume')
        if saved.get('quality_config') != self.quality_config():
            raise ValueError('SR quality target or plateau/scheduler configuration changed on resume')
        self.model.load_state_dict(saved['model'])
        self.optimizer.load_state_dict(saved['optimizer'])
        self.step, self.examples_trained = saved['training_steps'], saved['examples_trained']
        self.quality = saved.get('quality_state', self.quality)
        if self.quality['last_checked_step'] is not None and self.quality['last_checked_step'] > self.step:
            raise ValueError('SR quality history refers to a future checkpoint')
        if self.scheduler is not None:
            if saved.get('scheduler') is None:
                raise ValueError('Quality SR checkpoint is missing its learning-rate scheduler')
            self.scheduler.load_state_dict(saved['scheduler'])
            self.state['sr_quality'] = {**self.quality, **self.quality_config(),
                'target_met': self.quality['decision'] == 'target', 'learning_rate': self.optimizer.param_groups[0]['lr']}
        self.rng.bit_generator.state = saved['rng_numpy']
        torch.random.set_rng_state(saved['rng_torch'].cpu())
        if self.device.type == 'cuda' and saved.get('rng_cuda'):
            torch.cuda.set_rng_state_all([state.cpu() for state in saved['rng_cuda']])
        self.elapsed_before = float(saved.get('elapsed_seconds', 0.))
        self.started = time.time()
        if saved.get('validation'):
            self.state['sr_validation'] = saved['validation']
            self.validation_step = saved['validation']['step']
        if saved.get('validation_codec'):
            self.state['sr_validation_codec'] = saved['validation_codec']
        self.state['sr_training']['resumed_from_step'] = self.step

    def metric(self, metrics):
        value = {'timestamp': now(), 'stage': 'sr_train', 'step': self.step, **metrics}
        with (self.root/'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def validate(self, full=True):
        panel = self.panel if full else self.panel[:4]
        report, preview = evaluate_sr(self.model, self.validation, panel, batch_size=self.args.validation_batch, step=self.step)
        self.state['prediction_preview'] = save_sr_preview(preview, self.root/'preview.png', every=self.args.preview_every)
        decoded = None
        if self.codec is not None:
            decoded, decoded_preview = evaluate_sr(self.model, self.validation, panel,
                batch_size=self.args.validation_batch, step=self.step, codec=self.codec, input_kind='codec_reconstruction')
            decoded['codec'] = self.codec_info
            self.state['prediction_preview_codec'] = save_sr_preview(decoded_preview,
                self.root/'preview_codec.png', every=self.args.preview_every)
        if full:
            self.validation_step = self.step
            self.state['sr_validation'] = report
            atomic_json(self.root/'sr_validation.json', report)
            if decoded is not None:
                self.state['sr_validation_codec'] = decoded
                atomic_json(self.root/'sr_validation_codec.json', decoded)
            if self.args.target_ssim is not None:
                self.update_quality(report, decoded)
            self.metric({'validation': report['summary'], 'validation_frames': report['frames']})
        self.update()
        return report

    def update_quality(self, report, decoded=None):
        if self.quality['last_checked_step'] == self.step:
            return
        score = float(report['summary']['ssim_mean'])
        if not np.isfinite(score) or report.get('input_kind', 'clean_source_lr') != 'clean_source_lr':
            raise ValueError('The SR target requires finite mean SSIM on CLEAN source LR')
        if decoded is not None and decoded['panel_fingerprint'] != report['panel_fingerprint']:
            raise ValueError('Clean and decoded SR checks must use the same fixed held-out panel')
        previous_panel = self.quality.get('panel_fingerprint')
        if previous_panel is not None and previous_panel != report['panel_fingerprint']:
            raise ValueError('SR quality history cannot change its fixed held-out panel')
        self.quality['panel_fingerprint'] = report['panel_fingerprint']
        best_changed = self.quality['best_ssim'] is None or score > self.quality['best_ssim']
        if best_changed:
            self.quality.update(best_ssim=score, best_step=self.step)
        anchor = self.quality['anchor_ssim']
        if anchor is None or score >= anchor+self.args.quality_min_delta:
            self.quality.update(anchor_ssim=score, bad_checks=0)
        else:
            self.quality['bad_checks'] += 1
        self.quality['target_streak'] = self.quality['target_streak']+1 if score >= self.args.target_ssim else 0
        self.quality['last_checked_step'] = self.step
        self.quality['last_ssim'] = score
        if self.quality['target_streak'] >= 2:
            self.quality['decision'] = 'target'
        elif self.quality['bad_checks'] >= self.args.quality_patience:
            self.quality['decision'] = 'plateau'
        self.scheduler.step(score)
        if best_changed:
            path = self.root/'checkpoints'/'sr_best.pt'
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.pt.tmp')
            torch.save({'stage': 'sr', 'config': self.model.config, 'model': self.model.state_dict(),
                'training_steps': self.step, 'validation': report, 'validation_codec': decoded,
                'training_protocol': self.protocol()}, temporary)
            replace_file(temporary, path)
            self.state['sr_best_checkpoint'] = {'name': str(path), 'step': self.step, 'clean_ssim': score}
        self.state['sr_quality'] = {**self.quality, **self.quality_config(),
            'target_met': self.quality['decision'] == 'target', 'learning_rate': self.optimizer.param_groups[0]['lr']}

    def run(self):
        random.seed(self.args.seed); np.random.seed(self.args.seed); torch.manual_seed(self.args.seed)
        torch.set_num_threads(self.args.threads)
        from .models import build_upscaler
        model_config = {'channels': self.args.channels, 'blocks': self.args.blocks, 'scale': 4}
        if self.args.architecture != 'legacy':
            model_config.update(architecture=self.args.architecture, res_scale=self.args.res_scale)
        if self.args.codec_checkpoint is not None:
            self.codec, self.codec_info = load_sr_codec(self.args.codec_checkpoint, self.device, self.args.image_size)
            self.state['sr_training']['codec_augmentation'] = {**self.codec_info, 'fraction': self.args.codec_fraction}
        self.model = build_upscaler(model_config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=0.)
        if self.args.target_ssim is not None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max',
                factor=.5, patience=max(1, self.args.quality_patience//3),
                threshold=self.args.quality_min_delta, threshold_mode='abs', min_lr=self.args.lr_min)
        self.restore()
        self.data = SRPairData(self.args.data_dir/'rgb'/'train', self.device, split='train', image_size=self.args.image_size,
                               cache_bytes=self.args.cache_mb*1024**2, seed=self.args.seed)
        self.validation = SRPairData(self.args.data_dir/'rgb'/'val', self.device, split='val', image_size=self.args.image_size,
                                     cache_bytes=self.args.cache_mb*1024**2, seed=self.args.seed)
        panel_path = self.root/'validation_panel.json'
        if panel_path.exists():
            saved = json.loads(panel_path.read_text(encoding='utf-8'))
            self.panel = [tuple(pair) for pair in saved['pairs']]
            if (len(self.panel) != self.args.validation_frames
                    or saved['fingerprint'] != self.validation.panel_fingerprint(self.panel)):
                raise ValueError('Fixed held-out SR validation panel changed')
        else:
            self.panel = self.validation.fixed_panel(self.args.validation_frames, self.args.seed)
            atomic_json(panel_path, {'pairs': self.panel, 'fingerprint': self.validation.panel_fingerprint(self.panel)})
        atomic_json(self.root/'experiment_config.json', {key: str(value) if isinstance(value, Path) else value
                                                       for key, value in vars(self.args).items()})
        self.state['model_parameters'] = {'sr_total': sum(parameter.numel() for parameter in self.model.parameters())}
        self.update(stage='sr_train', stage_label='SR x4: learned reconstruction from paired source RGB', data=self.data.stats)
        initial_step, started = self.step, time.perf_counter()
        while self.step < self.args.steps and self.quality['decision'] == 'continue':
            if (self.root/'stop_training').exists():
                self.save()
                self.update(status='stopped', stage_label='SR checkpoint saved; stopped at optimizer boundary')
                return None
            batch = self.data.batch(self.args.batch, rng=self.rng)
            inputs, codec_fraction = self.training_inputs(batch['lr'])
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            amp = torch.autocast('cuda', dtype=torch.bfloat16) if self.device.type == 'cuda' else nullcontext()
            with amp:
                prediction = self.model(inputs).float()
                pixel = F.l1_loss(prediction, batch['hr'])
                edge = .5*(F.l1_loss(prediction[..., 1:]-prediction[..., :-1], batch['hr'][..., 1:]-batch['hr'][..., :-1])
                           +F.l1_loss(prediction[..., 1:, :]-prediction[..., :-1, :], batch['hr'][..., 1:, :]-batch['hr'][..., :-1, :]))
                ssim = differentiable_ssim_loss(prediction, batch['hr']) if self.args.ssim_weight else pixel.new_zeros(())
                loss = pixel+self.args.edge_weight*edge+self.args.ssim_weight*ssim
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite SR training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
            self.optimizer.step()
            self.step += 1
            self.examples_trained += len(batch['lr'])
            if self.step == initial_step+1 or self.step % self.args.log_every == 0:
                seconds = (time.perf_counter()-started)/(self.step-initial_step)
                self.metric({'train_loss': float(loss.detach()), 'pixel_l1': float(pixel.detach()),
                    'edge_l1': float(edge.detach()), 'ssim_loss': float(ssim.detach()),
                    'codec_fraction': codec_fraction, 'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'seconds_per_step': seconds})
                self.state['sr_training']['examples_trained'] = self.examples_trained
                self.update(loss=float(loss.detach()), eta_seconds=max(0., (self.args.steps-self.step)*seconds), data=self.data.stats)
            if self.step % self.args.validate_every == 0:
                self.validate()
            elif self.step == initial_step+1 or self.step % self.args.preview_every == 0:
                self.validate(full=False)
            if self.step == initial_step+1 or self.step % self.args.save_every == 0:
                self.save()
        if self.validation_step != self.step:
            self.validate()
        self.save()
        checkpoint = self.export_ready()
        self.update(status='completed', stage='sr_ready', stage_label='SR training finished; selected held-out metrics recorded',
                    eta_seconds=0, playable_sr_checkpoint=str(checkpoint))
        return checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--image-size', type=int, choices=(32, 64, 128), default=32)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--architecture', choices=('legacy', 'edsr'), default='legacy')
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--blocks', type=int, default=4)
    parser.add_argument('--res-scale', type=float, default=.1)
    parser.add_argument('--edge-weight', type=float, default=.1)
    parser.add_argument('--ssim-weight', type=float, default=0.)
    parser.add_argument('--target-ssim', type=float, default=None, help='Clean held-out mean SSIM target; requires two distinct full checks')
    parser.add_argument('--quality-patience', type=int, default=10, help='Stagnant full validations before selecting the best weights')
    parser.add_argument('--quality-min-delta', type=float, default=.0005)
    parser.add_argument('--lr-min', type=float, default=1e-6)
    parser.add_argument('--codec-checkpoint', type=Path, help='Verified frozen source-LR codec; never an untrained-world RGB source')
    parser.add_argument('--codec-fraction', type=float, default=.25, help='Expected fraction of training examples reconstructed by the frozen codec')
    parser.add_argument('--validation-frames', type=int, default=100)
    parser.add_argument('--validation-batch', type=int, default=8)
    parser.add_argument('--validate-every', type=int, default=100)
    parser.add_argument('--preview-every', type=int, default=100)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--cache-mb', type=int, default=256)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=373)
    args = parser.parse_args(argv)
    if min(args.batch, args.steps, args.channels, args.blocks, args.validation_frames, args.validation_batch,
           args.validate_every, args.preview_every, args.save_every, args.log_every, args.cache_mb, args.threads) < 1:
        parser.error('SR counts and intervals must be positive')
    if not np.isfinite(args.lr) or args.lr <= 0 or not np.isfinite(args.edge_weight) or args.edge_weight < 0:
        parser.error('SR learning rate must be positive and edge weight nonnegative')
    if (args.quality_patience < 1 or not np.isfinite(args.quality_min_delta) or args.quality_min_delta < 0
            or not np.isfinite(args.lr_min) or not 0 < args.lr_min <= args.lr
            or not np.isfinite(args.ssim_weight) or args.ssim_weight < 0
            or not np.isfinite(args.codec_fraction) or not 0 <= args.codec_fraction <= 1
            or not np.isfinite(args.res_scale) or args.res_scale <= 0
            or args.target_ssim is not None and (not np.isfinite(args.target_ssim) or not -1 <= args.target_ssim <= 1)):
        parser.error('Invalid SR quality target, schedule, codec fraction or SSIM/residual weight')
    experiment = SRExperiment(args)
    try:
        with KeepAwake() as awake:
            experiment.state['power_guard'] = {'active': awake.active, 'screen_lock_allowed': True,
                                              'display_required': False, 'error': awake.error}
            experiment.run()
    except Exception as error:
        experiment.update(status='failed', error=str(error), stage_label='SR failed; previous checkpoint retained')
        raise


if __name__ == '__main__':
    main()
