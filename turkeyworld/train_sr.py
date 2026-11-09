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


def evaluate_sr(model, data, panel, *, batch_size=8, step=0):
    """Strict FP32 held-out single-frame SR, never a world-rollout qualification."""
    if data.split != 'val' or batch_size < 1 or not panel or len(set(map(tuple, panel))) != len(panel):
        raise ValueError('SR evaluation requires unique explicit held-out val frames')
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError('SR validation requires FP32 model weights')
    modes = [(module, module.training) for module in model.modules()]
    rng_state = copy.deepcopy(data.rng.bit_generator.state)
    global_numpy, global_python = np.random.get_state(), random.getstate()
    device = data.device
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    rows, previews, content = [], [], hashlib.sha256()
    try:
        model.eval()
        with strict_render_precision(), torch.random.fork_rng(devices=cuda_devices), \
                torch.inference_mode(), torch.autocast(device.type, enabled=False):
            for start in range(0, len(panel), batch_size):
                chosen = panel[start:start+batch_size]
                batch = data.batch(len(chosen), indices=chosen)
                predicted = model(batch['lr']).float()
                if predicted.shape != batch['hr'].shape or not bool(torch.isfinite(predicted).all()):
                    raise FloatingPointError('SR validation returned invalid or nonfinite RGB')
                if bool((predicted < 0).any() or (predicted > 1).any()):
                    raise FloatingPointError('SR validation RGB lies outside[0,1]')
                baseline = F.interpolate(batch['lr'], size=predicted.shape[-2:], mode='bilinear', align_corners=False)
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
                        previews.append((batch['lr'][offset].cpu(), predicted[offset].cpu(), batch['hr'][offset].cpu()))
    finally:
        for module, training in modes:
            module.training = training
        data.rng.bit_generator.state = rng_state
        np.random.set_state(global_numpy)
        random.setstate(global_python)
    keys = ('psnr', 'ssim', 'mae', 'bilinear_psnr', 'bilinear_mae')
    report = {'kind': 'single_frame_super_resolution', 'step': int(step), 'frames': len(rows),
        'source_held_out': True, 'source_input_allowed_for_sr': True, 'world_rollout': False,
        'render_precision': 'fp32', 'image_size': data.image_size, 'output_size': data.image_size*data.scale,
        'summary': {key+'_mean': float(np.mean([row[key] for row in rows])) for key in keys},
        'per_frame': rows, 'panel_fingerprint': data.panel_fingerprint(panel),
        'source_content_sha256': content.hexdigest(),
        'scope': 'Held-out source LR to paired source HR; no autonomous-world quality claim'}
    preview = {key: _pixels(torch.stack([row[index] for row in previews]))
               for index, key in enumerate(('source_lr', 'sr', 'source_hr'))}
    preview['difference'] = np.abs(preview['sr'].astype(np.int16)-preview['source_hr'].astype(np.int16)).astype(np.uint8)
    preview['metadata'] = {key: report[key] for key in ('kind', 'step', 'source_held_out', 'world_rollout',
        'source_input_allowed_for_sr', 'image_size', 'output_size', 'panel_fingerprint')}
    return report, preview


def save_sr_preview(preview, path, *, every=100):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new('RGB', (512, 576), (14, 22, 31))
    draw = ImageDraw.Draw(canvas)
    labels = ('SOURCE LR input | enlarged for display', 'LEARNED SR x4', 'PAIRED SOURCE HR target', 'ABSOLUTE DIFFERENCE x4')
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
        self.args, self.root, self.device = args, args.run_dir.resolve(), torch.device(args.device)
        self.root.mkdir(parents=True, exist_ok=True)
        self.step, self.examples_trained, self.elapsed_before = 0, 0, 0.
        self.started = time.time()
        self.rng = np.random.default_rng(args.seed)
        self.validation_step = None
        self.state = {'run_id': self.root.name, 'status': 'running', 'stage': 'sr_initializing',
            'stage_label': 'SR x4: fresh paired-source training', 'started_at': now(), 'current': 0,
            'total': args.steps, 'progress': 0., 'prediction_every': args.preview_every,
            'prediction_preview': None, 'sr_training': {'from_scratch': True, 'pretrained_weights_loaded': False,
                'input_size': args.image_size, 'output_size': args.image_size*4, 'scale': 4,
                'world_rollout_qualified': False, 'requested_updates': args.steps}}
        self.update()

    def elapsed(self):
        return self.elapsed_before+time.time()-self.started

    def update(self, **fields):
        self.state.update(current=self.step, updated_at=now(), elapsed_seconds=round(self.elapsed(), 1),
                          progress=min(1., self.step/self.args.steps), **fields)
        atomic_json(self.root/'status.json', self.state)

    def protocol(self):
        return {'image_size': self.args.image_size, 'scale': 4, 'data_directory': str(self.args.data_dir.resolve()),
                'target_field': 'hr_frames', 'paired_source_rgb': True, 'pixel_weight': 1., 'edge_weight': self.args.edge_weight}

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
            'pretrained_sr_weights_loaded': False, 'quality_status': 'Held-out SR metrics; autonomous world remains unverified',
            'created_at': now()}
        temporary = path.with_suffix('.pt.tmp')
        torch.save(payload, temporary)
        replace_file(temporary, path)
        self.state['checkpoint'] = {'name': str(path), 'step': self.step, 'created_at': payload['created_at']}
        return path

    def export_ready(self):
        if self.step < self.args.steps or self.validation_step != self.step:
            raise ValueError('Only a completed SR budget with current held-out validation can be exported')
        path = self.root/'checkpoints'/'sr_for_game.pt'
        payload = {'format_version': 1, 'stage': 'sr', 'config': self.model.config,
            'model': self.model.state_dict(), 'training_steps': self.step,
            'input_size': self.args.image_size, 'output_size': self.args.image_size*4, 'scale': 4,
            'training_completed': True, 'requested_training_steps': self.args.steps,
            'examples_trained': self.examples_trained, 'training_protocol': self.protocol(),
            'validation': self.state['sr_validation'], 'pretrained_sr_weights_loaded': False,
            'quality_status': 'Completed SR budget; held-out metrics recorded, autonomous world unverified',
            'created_at': now()}
        temporary = path.with_suffix('.pt.tmp')
        torch.save(payload, temporary)
        replace_file(temporary, path)
        atomic_json(self.root/'sr_ready.json', {'checkpoint': str(path), 'training_steps': self.step,
            'requested_training_steps': self.args.steps, 'input_size': self.args.image_size,
            'output_size': self.args.image_size*4, 'scale': 4, 'training_completed': True,
            'pretrained_sr_weights_loaded': False, 'created_at': payload['created_at']})
        return path

    def restore(self):
        path = self.root/'checkpoints'/'sr_last.pt'
        if not path.exists():
            return
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['stage'] != 'sr' or saved['config'] != self.model.config or saved['training_protocol'] != self.protocol():
            raise ValueError('SR model/native source resolution/objective or dataset changed on resume')
        self.model.load_state_dict(saved['model'])
        self.optimizer.load_state_dict(saved['optimizer'])
        self.step, self.examples_trained = saved['training_steps'], saved['examples_trained']
        self.rng.bit_generator.state = saved['rng_numpy']
        torch.random.set_rng_state(saved['rng_torch'].cpu())
        if self.device.type == 'cuda' and saved.get('rng_cuda'):
            torch.cuda.set_rng_state_all([state.cpu() for state in saved['rng_cuda']])
        self.elapsed_before = float(saved.get('elapsed_seconds', 0.))
        self.started = time.time()
        if saved.get('validation'):
            self.state['sr_validation'] = saved['validation']
            self.validation_step = saved['validation']['step']
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
        if full:
            self.validation_step = self.step
            self.state['sr_validation'] = report
            atomic_json(self.root/'sr_validation.json', report)
            self.metric({'validation': report['summary'], 'validation_frames': report['frames']})
        self.update()
        return report

    def run(self):
        random.seed(self.args.seed); np.random.seed(self.args.seed); torch.manual_seed(self.args.seed)
        torch.set_num_threads(self.args.threads)
        self.model = SingleFrameUpscaler(channels=self.args.channels, blocks=self.args.blocks, scale=4).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=0.)
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
        while self.step < self.args.steps:
            if (self.root/'stop_training').exists():
                self.save()
                self.update(status='stopped', stage_label='SR checkpoint saved; stopped at optimizer boundary')
                return None
            batch = self.data.batch(self.args.batch, rng=self.rng)
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            amp = torch.autocast('cuda', dtype=torch.bfloat16) if self.device.type == 'cuda' else nullcontext()
            with amp:
                prediction = self.model(batch['lr']).float()
                pixel = F.l1_loss(prediction, batch['hr'])
                edge = .5*(F.l1_loss(prediction[..., 1:]-prediction[..., :-1], batch['hr'][..., 1:]-batch['hr'][..., :-1])
                           +F.l1_loss(prediction[..., 1:, :]-prediction[..., :-1, :], batch['hr'][..., 1:, :]-batch['hr'][..., :-1, :]))
                loss = pixel+self.args.edge_weight*edge
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
                             'edge_l1': float(edge.detach()), 'seconds_per_step': seconds})
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
        self.update(status='completed', stage='sr_ready', stage_label='SR training budget completed; held-out metrics recorded',
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
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--blocks', type=int, default=4)
    parser.add_argument('--edge-weight', type=float, default=.1)
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
