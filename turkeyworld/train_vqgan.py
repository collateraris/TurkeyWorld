"""Train a shared game VQGAN codec from scratch, then resume latent world models.

Encoder, decoder, codebook and PatchGAN are fresh random models. Optional
standard pretrained LPIPS/VGG is a frozen loss network, excluded from codec
inference/checkpoints. Growing source data and held-out validation stay apart.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .power_guard import KeepAwake
from .training_pipeline import atomic_json, now, replace_file
from .vqgan import VQGAN, VQGANConfig
from .vqgan_data import VQGANData
from .vqgan_losses import (FrozenLPIPSVGG, NLayerDiscriminator, VQGANObjective,
    adaptive_discriminator_weight, generator_hinge_loss, hinge_discriminator_loss)
from .vqgan_validation import evaluate_vqgan, save_vqgan_preview


def codec_fingerprint(model):
    value = hashlib.sha256(json.dumps(model.get_config(), sort_keys=True).encode())
    for name, parameter in sorted(model.state_dict().items()):
        value.update(name.encode())
        value.update(parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return value.hexdigest()


class CodecExperiment:
    def __init__(self, args):
        self.args, self.root = args, args.run_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(args.device)
        self.image_size = getattr(args, 'image_size', 128)
        teacher_size = getattr(args, 'teacher_size', '500m_vqgan')
        fresh_world = bool(getattr(args, 'world_from_scratch', False))
        source_run = getattr(args, 'world_source_run', None)
        self.world_plan = {'teacher_size': teacher_size, 'from_scratch': fresh_world,
            'target_seconds': getattr(args, 'target_seconds', 300.),
            'image_size': self.image_size,
            'world_source_run': None if fresh_world or source_run is None else str(source_run.resolve())}
        self.rng = np.random.default_rng(args.seed)
        self.started = time.time()
        self.step, self.streak, self.best_score = 0, 0, None
        self.examples_trained = 0
        self.last_validation = None
        self.last_validation_step = None
        self.state = {'run_id': self.root.name, 'status': 'running', 'stage': 'vqgan_initializing',
            'stage_label': 'VQGAN: подготовка обучения с нуля', 'model': 'Shared VQGAN codec → Teacher CFM / Student RF',
            'current': 0, 'total': args.max_steps or None, 'progress': None,
            'started_at': now(), 'prediction_every': args.validate_every, 'prediction_preview': None,
            'evaluation': {'status': 'pending', 'target_seconds': self.world_plan['target_seconds'], 'passed': False},
            'model_parameters': {'target_teacher_total': 175000000 if teacher_size == '175m_vqgan' else 500000000,
                'teacher_trainable': 0},
            'codec_training': {'from_scratch': True, 'pretrained_codec_weights_loaded': False,
                'pretrained_discriminator_weights_loaded': False, 'disc_start': args.disc_start,
                'required_passes': args.quality_passes, 'consecutive_passes': 0, 'frozen': False},
            'pipeline': {**self.world_plan,
                'teacher_state': 'fresh_pending_codec_quality' if fresh_world else 'checkpointed_waiting_for_frozen_codec',
                'shared_codec_for_teacher_and_student': True}}
        self.update()

    def update(self, **fields):
        self.state.update(current=self.step, updated_at=now(),
            elapsed_seconds=round(time.time()-self.started, 1), **fields)
        atomic_json(self.root/'status.json', self.state)

    def metric(self, metrics):
        value = {'timestamp': now(), 'stage': 'vqgan_codec', 'step': self.step, **metrics}
        with (self.root/'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def amp(self):
        return torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False) if self.device.type == 'cuda' else nullcontext()

    def save(self, metrics=None):
        checkpoint = {'format_version': 1, 'stage': 'vqgan_codec', 'codec_config': self.model.get_config(),
            'model': self.model.state_dict(), 'discriminator': self.discriminator.state_dict(),
            'optimizer': self.optimizer.state_dict(), 'discriminator_optimizer': self.d_optimizer.state_dict(),
            'training_steps': self.step, 'quality_streak': self.streak, 'best_score': self.best_score,
            'examples_trained': self.examples_trained,
            'last_validation_fingerprint': self.last_validation,
            'last_validation_step': self.last_validation_step,
            'rng_numpy': self.rng.bit_generator.state, 'rng_torch': torch.random.get_rng_state(),
            'rng_cuda': torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else [],
            'training_objective': self.objective_metadata(), 'metrics': metrics or {},
            'world_plan': self.world_plan,
            'initialization': 'from_scratch_codec_and_discriminator; own-training-feature codebook initialization',
            'created_at': now()}
        path = self.root/'checkpoints'/'vqgan_last.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.pt.tmp')
        torch.save(checkpoint, temporary)
        replace_file(temporary, path)
        self.state['checkpoint'] = {'name': str(path), 'step': self.step, 'created_at': checkpoint['created_at']}

    def restore(self):
        path = self.root/'checkpoints'/'vqgan_last.pt'
        if not path.exists():
            return
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if asdict(VQGANConfig(**saved['codec_config'])) != self.model.get_config():
            raise ValueError('Codec checkpoint configuration differs from this run')
        if saved['training_objective'] != self.objective_metadata():
            # Early checkpoints kept auxiliary coefficients in the immutable
            # run configuration rather than in the objective record.
            original_path = self.root/'experiment_config.json'
            original = json.loads(original_path.read_text(encoding='utf-8')) if original_path.exists() else {}
            auxiliary = ('temporal_weight', 'vq_weight', 'adversarial_weight', 'disc_ramp')
            if saved['training_objective'] != self.objective.metadata or any(
                    original.get(key) != getattr(self.args, key) for key in auxiliary):
                raise ValueError('Codec reconstruction/perceptual objective changed on resume')
        self.model.load_state_dict(saved['model'])
        self.discriminator.load_state_dict(saved['discriminator'])
        self.optimizer.load_state_dict(saved['optimizer'])
        self.d_optimizer.load_state_dict(saved['discriminator_optimizer'])
        self.step, self.streak = saved['training_steps'], saved.get('quality_streak', 0)
        self.examples_trained = int(saved.get('examples_trained', self.step*self.args.batch*2))
        self.best_score, self.last_validation = saved.get('best_score'), saved.get('last_validation_fingerprint')
        self.last_validation_step = saved.get('last_validation_step',
            self.step//self.args.validate_every*self.args.validate_every if self.last_validation else None)
        self.state['codec_training']['consecutive_passes'] = self.streak
        self.rng.bit_generator.state = saved['rng_numpy']
        torch.random.set_rng_state(saved['rng_torch'].cpu())
        if self.device.type == 'cuda' and saved.get('rng_cuda'):
            torch.cuda.set_rng_state_all([value.cpu() for value in saved['rng_cuda']])

    def objective_metadata(self):
        return {**self.objective.metadata, 'temporal_weight': self.args.temporal_weight,
            'vq_weight': self.args.vq_weight, 'adversarial_weight': self.args.adversarial_weight,
            'disc_ramp': self.args.disc_ramp}

    def validate(self):
        self.update(validation_progress={'status': 'running', 'step': self.step, 'frames': len(self.panel)*2})
        thresholds = {'psnr': self.args.quality_psnr, 'ssim': self.args.quality_ssim,
            'mae': self.args.quality_mae, 'required_fraction': self.args.quality_fraction}
        report, preview = evaluate_vqgan(self.model, self.validation, self.panel,
            batch_size=self.args.validation_batch, device=self.device, thresholds=thresholds,
            seed=self.args.seed, step=self.step)
        if self.last_validation != report['fingerprint']:
            self.streak = 0
            self.last_validation_step = None
        self.last_validation = report['fingerprint']
        distinct = self.last_validation_step is None or self.step-self.last_validation_step >= self.args.validate_every
        if not report['passed']:
            self.streak = 0
        elif distinct:
            self.streak += 1
        if distinct:
            self.last_validation_step = self.step
        report.update(consecutive_passes=self.streak, required_passes=self.args.quality_passes,
                      frozen=self.streak >= self.args.quality_passes)
        atomic_json(self.root/'codec_validation.json', report)
        self.state['codec_validation'] = report
        self.state['codec_training'].update(consecutive_passes=self.streak, frozen=report['frozen'])
        self.state['prediction_preview'] = save_vqgan_preview(preview, self.root/'preview.png', step=self.step)
        self.state['prediction_preview']['every'] = self.args.validate_every
        score = report['summary']['mae_mean']+.5*(1-report['summary']['ssim_mean'])
        if self.best_score is None or score < self.best_score:
            self.best_score = score
            path = self.root/'checkpoints'/'vqgan_best.pt'
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.pt.tmp')
            torch.save({'stage': 'vqgan_codec', 'codec_config': self.model.get_config(),
                'model': self.model.state_dict(), 'training_steps': self.step, 'validation': report}, temporary)
            replace_file(temporary, path)
        self.metric({'validation': report['summary'], 'codebook': {key: value for key, value in report['codebook'].items()
                    if key != 'histogram'}, 'quality_streak': self.streak})
        self.update(validation_progress={'status': 'complete', 'step': self.step, 'frames': len(self.panel)*2})
        return report

    def export_ready(self, report):
        required = max(3, self.args.quality_passes)
        if (report.get('passed') is not True or report.get('frozen') is not True
                or report.get('step') != self.step or self.streak < required
                or report.get('consecutive_passes') != self.streak
                or report.get('required_passes') != self.args.quality_passes
                or report.get('fingerprint') != self.last_validation):
            raise ValueError('Only the current weights after three distinct passed checks can become the shared codec')
        path = self.root/'checkpoints'/'codec_for_renderer.pt'
        temporary = path.with_suffix('.pt.tmp')
        fingerprint = codec_fingerprint(self.model)
        # Normalize the frozen code embeddings for continuous FM/RF. Only
        # an independent TRAIN calibration sample is measured. Validation
        # remains reserved for codec quality checks.
        total = squares = None
        count = 0
        calibration_rng = np.random.default_rng(self.args.seed+991)
        calibration_indices = []
        self.model.eval()
        with torch.inference_mode(), torch.autocast(self.device.type, enabled=False):
            for first in range(0, len(self.panel), self.args.validation_batch):
                values = self.data.batch(min(self.args.validation_batch, len(self.panel)-first), rng=calibration_rng)
                calibration_indices.extend(values['indices'].cpu().tolist())
                latent = self.model.encode(values['frames'].flatten(0, 1))['quantized'].double()
                one, two = latent.sum((0, 2, 3)), latent.square().sum((0, 2, 3))
                total = one if total is None else total+one
                squares = two if squares is None else squares+two
                count += latent.shape[0]*latent.shape[2]*latent.shape[3]
        mean = total/count
        std = (squares/count-mean.square()).clamp_min(1e-6).sqrt()
        normalization = {'mean': mean.cpu().tolist(), 'std': std.cpu().tolist(),
                         'source': 'independent_training_calibration', 'samples': count,
                         'seed': self.args.seed+991, 'pair_indices': calibration_indices}
        torch.save({'format_version': 1, 'stage': 'frozen_vqgan_codec',
            'codec_config': self.model.get_config(), 'model': self.model.state_dict(),
            'training_steps': self.step, 'codec_fingerprint': fingerprint,
            'quality_gate': report, 'training_objective': self.objective_metadata(),
            'world_plan': self.world_plan,
            'latent_normalization': normalization,
            'generator_parameters': sum(p.numel() for p in self.model.parameters()),
            'pretrained_codec_weights_loaded': False, 'created_at': now()}, temporary)
        replace_file(temporary, path)
        ready = {'checkpoint': str(path), 'codec_fingerprint': fingerprint, 'training_steps': self.step,
            'codec_config': self.model.get_config(), 'world_plan': self.world_plan,
            'quality_gate_passed': True, 'consecutive_passes': self.streak,
            'required_passes': self.args.quality_passes, 'shared_teacher_and_student': True, 'created_at': now()}
        atomic_json(self.root/'codec_ready.json', ready)
        self.state['codec_training']['frozen'] = True
        self.state['status'] = 'completed'
        self.update(stage='vqgan_ready', stage_label='VQGAN прошёл проверку и заморожен для обоих рендереров',
                    codec_ready=ready, eta_seconds=0)
        return path

    def run(self):
        random.seed(self.args.seed);np.random.seed(self.args.seed);torch.manual_seed(self.args.seed)
        torch.set_num_threads(4)
        config = VQGANConfig(image_size=self.image_size, n_embed=self.args.n_embed,
                             embedding_dim=self.args.embedding_dim)
        self.model = VQGAN(config).to(self.device)
        self.discriminator = NLayerDiscriminator(ndf=self.args.discriminator_width).to(self.device)
        self.update(stage='vqgan_initializing', stage_label='VQGAN: подготовка perceptual loss и потока Source RGB')
        perceptual = FrozenLPIPSVGG().to(self.device) if self.args.perceptual == 'lpips' else None
        self.objective = VQGANObjective(perceptual, pixel_weight=1., perceptual_weight=self.args.perceptual_weight,
            edge_weight=self.args.edge_weight, disc_start=self.args.disc_start).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(.5, .9))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr, betas=(.5, .9))
        self.restore()
        self.state['codec_training'].update(parameters=sum(p.numel() for p in self.model.parameters()),
            config=config.__dict__, discriminator_parameters=sum(p.numel() for p in self.discriminator.parameters()),
            perceptual=self.objective.metadata['perceptual'], actual_initialization='random_codec_and_patchgan')
        self.state['model_parameters']['codec_total'] = self.state['codec_training']['parameters']
        atomic_json(self.root/'experiment_config.json', {key: str(value) if isinstance(value, Path) else value
                                                        for key, value in vars(self.args).items()})
        ready_path = self.root/'codec_ready.json'
        if ready_path.exists():
            from .vq_world import validate_frozen_codec
            ready = json.loads(ready_path.read_text(encoding='utf-8'))
            verified = validate_frozen_codec(ready['checkpoint'])
            if verified['codec_config'] != self.model.get_config():
                raise ValueError('Ready codec configuration differs from requested configuration')
            gate = verified['quality_gate']
            thresholds = {'psnr': self.args.quality_psnr, 'ssim': self.args.quality_ssim,
                'mae': self.args.quality_mae, 'required_fraction': self.args.quality_fraction}
            if gate.get('thresholds') != thresholds or gate['required_passes'] != self.args.quality_passes:
                raise ValueError('Ready codec quality thresholds differ from this run')
            self.state.update(status='completed', codec_ready=ready, codec_validation=gate)
            self.state['codec_training'].update(frozen=True, consecutive_passes=gate['consecutive_passes'])
            self.update(stage='vqgan_ready', stage_label='Проверенный VQGAN готов для обоих рендереров')
            return Path(ready['checkpoint'])
        self.validation = VQGANData(self.args.data_dir/'rgb'/'val', self.device, split='val',
            image_size=self.image_size, require_native_size=True,
            cache_bytes=128*1024**2, seed=self.args.seed)
        panel_path = self.root/'validation_panel.json'
        self.panel = json.loads(panel_path.read_text(encoding='utf-8')) if panel_path.exists() else self.validation.fixed_panel(
            self.args.validation_pairs, self.args.seed)
        if not panel_path.exists():atomic_json(panel_path, self.panel)
        self.data = VQGANData(self.args.data_dir/'rgb'/'train', self.device, split='train',
            image_size=self.image_size, require_native_size=True,
            cache_bytes=self.args.cache_mb*1024**2, balance=True, event_fraction=.35, seed=self.args.seed,
            initial_file_limit=self.args.initial_files, growth_files=self.args.growth_files)
        self.state['data'] = self.data.stats
        self.update(stage='vqgan_codec', stage_label='VQGAN: реконструкция и обучение кодовой книги с нуля')
        started, initial_step, metrics = time.perf_counter(), self.step, {}
        while self.args.max_steps == 0 or self.step < self.args.max_steps:
            if (self.root/'stop_training').exists():
                self.save(metrics)
                self.state['status'] = 'stopped'
                self.update(stage_label='VQGAN сохранён и остановлен')
                raise SystemExit(0)
            self.step += 1
            batch = self.data.batch(self.args.batch, rng=self.rng)
            images = batch['frames'].flatten(0, 1)
            self.model.train();self.discriminator.eval().requires_grad_(False)
            self.optimizer.zero_grad(set_to_none=True);self.d_optimizer.zero_grad(set_to_none=True)
            factor = self.objective.discriminator_factor(self.step)
            factor *= min(1., max(0., (self.step-self.args.disc_start+1)/max(1, self.args.disc_ramp)))
            with self.amp():
                result = self.model(images)
                reconstruction = result['reconstruction']
                rec_loss, metrics = self.objective.reconstruction_loss(images, reconstruction)
                pair_reconstruction = reconstruction.reshape_as(batch['frames'])
                temporal = F.l1_loss(pair_reconstruction[:, 1].float()-pair_reconstruction[:, 0].float(),
                                    batch['frames'][:, 1]-batch['frames'][:, 0])
                rec_loss = rec_loss+self.args.temporal_weight*temporal
                g_adv, adaptive = rec_loss.new_zeros(()), rec_loss.new_zeros(())
                if factor:
                    with torch.autocast(self.device.type, enabled=False):
                        g_adv = generator_hinge_loss(self.discriminator(reconstruction.float()))
                    adaptive = adaptive_discriminator_weight(rec_loss, g_adv, self.model.get_last_layer())
                loss = rec_loss+self.args.vq_weight*result['vq_loss']+self.args.adversarial_weight*factor*adaptive*g_adv
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite codec loss at{self.step}')
            loss.backward();torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
            self.optimizer.step()
            self.examples_trained += len(images)
            d_loss = loss.new_zeros(())
            if factor:
                self.discriminator.train().requires_grad_(True)
                with torch.autocast(self.device.type, enabled=False):
                    d_loss = hinge_discriminator_loss(self.discriminator(images.detach().float()),
                                                       self.discriminator(reconstruction.detach().float()))
                if not torch.isfinite(d_loss):raise FloatingPointError('Nonfinite VQGAN discriminator')
                d_loss.backward();torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1., error_if_nonfinite=True)
                self.d_optimizer.step()
            metrics.update(train_loss=float(loss.detach()), vq_loss=float(result['vq_loss'].detach()),
                temporal_loss=float(temporal.detach()), generator_adversarial=float(g_adv.detach()),
                adaptive_weight=float(adaptive.detach()), discriminator_loss=float(d_loss.detach()),
                adversarial_factor=factor, **{key: float(value) for key, value in result['metrics'].items()})
            if self.step == initial_step+1 or self.step % self.args.log_every == 0:
                if self.device.type == 'cuda':torch.cuda.synchronize()
                seconds = (time.perf_counter()-started)/(self.step-initial_step)
                self.metric({**metrics, 'seconds_per_step': seconds})
                self.state['data'] = self.data.stats
                self.state['codec_training'].update(current=self.step, phase='adversarial' if factor else 'reconstruction_warmup')
                self.state['codec_training']['examples_trained'] = self.examples_trained
                self.update(loss=metrics['train_loss'], throughput={'value': 1/seconds, 'unit': 'шагов/с'},
                    eta_seconds=None, message='Encoder, codebook, decoder и PatchGAN учатся на Source; LPIPS — замороженная сеть потерь')
            if self.step % self.args.validate_every == 0:
                report = self.validate()
                self.save(metrics)
                if report['frozen']:
                    return self.export_ready(report)
            elif self.step == initial_step+1 or self.step % self.args.save_every == 0:
                self.save(metrics)
        self.save(metrics)
        self.state['status'] = 'budget_finished_unqualified'
        self.update(stage_label='Лимит шагов завершён; кодек ещё не прошёл проверку')
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('training_runs/vqgan_20261109'))
    parser.add_argument('--data-dir', type=Path, default=Path('training_runs/vjepa250_20261109/data'))
    parser.add_argument('--world-source-run', type=Path, default=Path('training_runs/vjepa500_20261109'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--image-size', type=int, choices=(32, 64, 128), default=128,
        help='Native source RGB resolution; input clips must match exactly')
    parser.add_argument('--teacher-size', choices=('175m_vqgan', '500m_vqgan'), default='500m_vqgan',
        help='Pending world profile for truthful codec-stage metadata; no teacher is trained here')
    parser.add_argument('--target-seconds', type=float, default=300., help='Pending world rollout qualification duration')
    parser.add_argument('--world-from-scratch', action='store_true',
        help='Record a fresh pending world; launch it with the separate fresh-world controller after codec qualification')
    parser.add_argument('--batch', type=int, default=8, help='Adjacent frame pairs; actual RGB batch is twice this value')
    parser.add_argument('--cache-mb', type=int, default=512)
    parser.add_argument('--initial-files', type=int, default=8192, help='Bootstrap checked training subset; 0 indexes all initially')
    parser.add_argument('--growth-files', type=int, default=2048, help='New clips indexed per discovery interval')
    parser.add_argument('--n-embed', type=int, default=1488)
    parser.add_argument('--embedding-dim', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--perceptual', choices=['lpips', 'none'], default='lpips')
    parser.add_argument('--perceptual-weight', type=float, default=1.)
    parser.add_argument('--edge-weight', type=float, default=.1)
    parser.add_argument('--temporal-weight', type=float, default=.1)
    parser.add_argument('--vq-weight', type=float, default=1.)
    parser.add_argument('--adversarial-weight', type=float, default=.1)
    parser.add_argument('--disc-start', type=int, default=1000)
    parser.add_argument('--disc-ramp', type=int, default=1000)
    parser.add_argument('--discriminator-width', type=int, default=64)
    parser.add_argument('--validation-pairs', type=int, default=128)
    parser.add_argument('--validation-batch', type=int, default=4)
    parser.add_argument('--validate-every', type=int, default=500)
    parser.add_argument('--save-every', type=int, default=250)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--quality-psnr', type=float, default=40.)
    parser.add_argument('--quality-ssim', type=float, default=.99)
    parser.add_argument('--quality-mae', type=float, default=1/255)
    parser.add_argument('--quality-fraction', type=float, default=.95)
    parser.add_argument('--quality-passes', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=0, help='0 means continue until the codec quality gate passes')
    parser.add_argument('--seed', type=int, default=273)
    parser.add_argument('--continue-world', action='store_true')
    parser.add_argument('--world-run-dir', type=Path)
    args = parser.parse_args(argv)
    if min(args.batch, args.validation_pairs, args.validation_batch, args.validate_every,
           args.save_every, args.log_every, args.cache_mb, args.quality_passes) < 1 or min(args.max_steps, args.initial_files, args.growth_files) < 0:
        parser.error('Counts must be positive; max-steps may be0')
    if args.quality_passes < 3:
        parser.error('Shared codec export requires at least three consecutive quality checks')
    if not np.isfinite(args.target_seconds) or args.target_seconds <= 0:
        parser.error('--target-seconds must be finite and positive')
    if args.continue_world and (args.world_from_scratch or args.teacher_size != '500m_vqgan' or args.image_size != 128):
        parser.error('--continue-world is the legacy 500M/128 migration; use the fresh-world controller for a new world')
    experiment = CodecExperiment(args)
    try:
        with KeepAwake() as awake:
            experiment.state['power_guard'] = {'active': awake.active, 'screen_lock_allowed': True,
                                              'display_required': False, 'error': awake.error}
            codec = experiment.run()
            if codec is not None and args.continue_world:
                from .vqgan_pipeline import continue_world_after_codec
                continue_world_after_codec(codec, args)
    except Exception as error:
        experiment.state['status'] = 'failed'
        experiment.update(stage_label='Ошибка VQGAN; предыдущий checkpoint сохранён', error=str(error))
        raise


if __name__ == '__main__':
    main()
