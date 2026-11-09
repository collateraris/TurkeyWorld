"""Compact V-JEPA2-AC game training, then Rectified Flow student distillation.

This is a 125M game adaptation trained from local data, not Meta pretrained
weights. Future RGB/state is supervision only: both short and long recurrent
streams consume their own predictions. The EMA encoder and GAN are training
tools, excluded from the playable generator parameter count.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import gc
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from .models import NeuralWorldModel, WorldModelConfig, load_world_checkpoint
from .training_pipeline import Experiment, atomic_json, replace_file, now, model_fingerprint
from .jepa_training_utils import EMATargetEncoder, SequenceData, latent_prediction_loss, variance_covariance_loss
from .plateau import PlateauTracker
from .train import state_loss, image_loss, supervised_event_losses


def short_rollout_loss(model, ema, batch, *, context=2, render=True, teacher=None,
                       distill_endpoint=False):
    """No ground-truth feedback after context; returns a differentiable loss.

    Targets are encoded by a stop-gradient EMA. In contrast, runtime memory
    carries the online context followed only by predicted visual tokens.
    """
    features = model.encode_map(batch['map'])
    online = model.core.encode_video(batch['context_frames'])
    memory = model.vjepa_backend.initialize_from_latents(online, batch['states'][:, :context], features)
    current = batch['states'][:, context-1]
    with torch.no_grad():
        target_latents = ema(batch['frames'])[:, context:]
    regularizer, regularizer_metrics = variance_covariance_loss(online)
    parts = {'regularizer': .1*regularizer}
    latent_predictions, state_values, event_values, motion_values = [], [], [], []
    for index in range(batch['actions'].shape[1]):
        result = model.transition(current, batch['actions'][:, index], features, memory)
        latent_predictions.append(result['latent_prediction'])
        state_values.append(state_loss(result['state'], batch['target_states'][:, index]))
        events = supervised_event_losses(result, batch['hits'][:, index], batch['attacks'][:, index],
                                         batch['events'][:, index])
        event_values.append(sum(events.values()))
        # A 1/30s displacement is tiny in normalized90D state. Scale motion
        # before loss so standing still cannot cheaply minimize the objective.
        reference_old = batch['previous_states'][:, index]
        reference_next = batch['target_states'][:, index]
        dt = 1/model.config.simulation_hz
        player_motion = F.smooth_l1_loss((result['state'][:, :2]-current[:, :2])*(16/(2.4*dt)),
                                        (reference_next[:, :2]-reference_old[:, :2])*(16/(2.4*dt)))
        predicted_npcs = result['state'][:, 8:88].reshape(-1, 8, 10)
        previous_npcs = current[:, 8:88].reshape(-1, 8, 10)
        reference_npcs = reference_next[:, 8:88].reshape(-1, 8, 10)
        old_reference_npcs = reference_old[:, 8:88].reshape(-1, 8, 10)
        continuing = (old_reference_npcs[:, :, 5]>.5)&(reference_npcs[:, :, 5]>.5)
        npc_motion = F.smooth_l1_loss((predicted_npcs[:, :, :2]-previous_npcs[:, :, :2])*(16/(1.2*dt)),
            (reference_npcs[:, :, :2]-old_reference_npcs[:, :, :2])*(16/(1.2*dt)), reduction='none')
        npc_motion = (npc_motion*continuing[:, :, None]).sum()/(2*continuing.sum()).clamp_min(1)
        motion_values.append(player_motion+npc_motion)
        current, memory = result['state'], result['memory']
    parts['state'] = 10*torch.stack(state_values).mean()
    parts['combat'] = torch.stack(event_values).mean()
    parts['motion'] = 5*torch.stack(motion_values).mean()
    parts['jepa'], latent_metrics = latent_prediction_loss(torch.stack(latent_predictions, 1), target_latents)
    base = model.render_base(current, result['features'], features)
    target_rgb = batch['target_frames'][:, -1]
    parts['base_image'] = image_loss(base, target_rgb)
    flow_metrics = {}
    if render:
        parts['flow'], flow_metrics = model.flow_renderer.loss(target_rgb, base, current)
    if teacher is not None:
        with torch.no_grad():
            teacher_map = teacher.encode_map(batch['map'])
            teacher_memory = teacher.initialize_from_image(batch['context_frames'], batch['states'][:, :context], teacher_map)
            teacher_state = batch['states'][:, context-1]
            for index in range(batch['actions'].shape[1]):
                teach = teacher.transition(teacher_state, batch['actions'][:, index], teacher_map, teacher_memory)
                teacher_state, teacher_memory = teach['state'], teach['memory']
        parts['teacher_state'] = .5*state_loss(current, teacher_state)
        parts['teacher_events'] = .2*(F.kl_div(F.log_softmax(result['target_logits'], -1),
            F.softmax(teach['target_logits'], -1), reduction='batchmean') +
            F.binary_cross_entropy_with_logits(result['attack_logits'], teach['attack_logits'].sigmoid()))
        if distill_endpoint:
            # True reflow: a teacher endpoint and the SAME initial noise field
            # define the student's straight conditional probability path.
            count = 1
            with torch.no_grad():
                noise = torch.randn_like(target_rgb[:count])
                teacher_base = teacher.render_base(teacher_state[:count], teach['features'][:count], teacher_map[:count])
                endpoint = teacher.flow_renderer.sample(teacher_base, teacher_state[:count], noise=noise)
            parts['reflow'], _ = model.flow_renderer.loss(target_rgb[:count], base[:count], current[:count],
                                                         noise=noise, target_endpoint=endpoint)
    metrics = {key: float(value.detach()) for key, value in parts.items()}
    metrics.update({key: float(value) for key, value in {**regularizer_metrics, **latent_metrics, **flow_metrics}.items()})
    return sum(parts.values()), metrics, {'base': base, 'state': current, 'target': target_rgb, 'features': result['features']}


class VJEPAExperiment(Experiment):
    def __init__(self, args):
        super().__init__(args)
        self.global_step = 0
        self.ema = None
        self.extra = {}
        self.validation = None
        self.panel = None
        start_path = self.root/'vjepa_run_started.json'
        if start_path.exists():
            started_at = json.loads(start_path.read_text(encoding='utf-8'))['started_at']
        else:
            started_at = now()
            metrics_path = self.root/'metrics.jsonl'
            if metrics_path.exists():
                with metrics_path.open(encoding='utf-8') as stream:
                    for line in stream:
                        try:
                            row = json.loads(line)
                        except ValueError:
                            continue
                        if row.get('stage') == 'vjepa_pretrain':
                            started_at = row['timestamp']
                            break
            atomic_json(start_path, {'started_at': started_at})
        self.started = datetime.fromisoformat(started_at).timestamp()
        self.state['started_at'] = started_at
        self.state.update(model='Compact V-JEPA 2-AC 125M / Flow Matching → Rectified Flow',
            teacher_world_architecture='vjepa2_ac_compact', teacher_size='125m',
            prediction_preview=None, estimated_total_range_seconds=None, eta_seconds=None,
            prediction_every=500, target_rollout_steps=9000,
            first_cycle_estimate={'range_seconds': [14400, 21600],
                'scope': 'First teacher plateau handoff plus16000 student updates and long evaluations; excludes repair rounds',
                'teacher_seconds_per_update': 2.76, 'student_seconds_per_update': .405,
                'target_quality_completion_time_known': False},
            training_protocol={'own_predictions': True, 'future_ground_truth_feedback': False,
                'persistent_map': True, 'persistent_object_slots': 8, 'tbptt_chunk': args.chunk,
                'curriculum': [32, 128, 512, 2048, 9000], 'teacher_before_student': True},
            plateau_policy={'enabled': True, 'patience': args.plateau_patience,
                'min_relative_improvement': args.plateau_delta, 'every': 500})
        self.state['teacher_plateau'] = {'enabled': True, 'patience': args.plateau_patience,
            'min_relative_improvement': args.plateau_delta, 'check_every': 500, 'bad_checks': 0}
        self.state['model_parameters'] = {'target_teacher_total': 125000000}

    def _publish_model_sizes(self, model):
        super()._publish_model_sizes(model)
        self.state['model_parameters']['target_teacher_total'] = 125000000
        if self.ema is not None:
            self.state['model_parameters']['ema_target_total'] = self.ema.parameter_count

    def save(self, model, optimizer, phase, step, metrics, extra=None):
        additional = {**self.extra, 'global_training_step': self.global_step}
        if self.ema is not None:
            additional['ema_target_encoder'] = self.ema.state_dict()
        additional.update(extra or {})
        return super().save(model, optimizer, phase, step, metrics, additional)

    def restore(self, model, optimizer, phase):
        step = super().restore(model, optimizer, phase)
        path = self.root/'checkpoints'/f'{phase}.pt'
        self.extra = {}
        if path.exists():
            saved = torch.load(path, map_location='cpu', weights_only=False)
            if phase.startswith('student') and saved.get('teacher_fingerprint') != self.state.get('teacher_fingerprint'):
                return 0
            self.global_step = max(self.global_step, int(saved.get('global_training_step', 0)))
            preview_path = self.root/'preview.json'
            if preview_path.exists():
                previous_preview = json.loads(preview_path.read_text(encoding='utf-8'))
                if (previous_preview.get('kind') == 'free_rollout'
                        and int(previous_preview.get('step', -1)) <= self.global_step):
                    self.state['prediction_preview'] = previous_preview
            if self.ema is not None and 'ema_target_encoder' in saved:
                self.ema.load_state_dict(saved['ema_target_encoder'])
            self.extra = {name: saved[name] for name in ('teacher_plateau', 'teacher_handoff_reason', 'teacher_selection', 'quality_streak', 'phase_complete',
                          'long_curriculum_state') if name in saved}
        return step

    def check_stop(self, model, optimizer, phase, step, metrics):
        if (self.root/'stop_training').exists():
            self.save(model, optimizer, phase, step, metrics)
            self.state['status'] = 'stopped'
            self.update(phase, step, None, 'Обучение сохранено и остановлено')
            raise SystemExit(0)

    def validation_report(self, model, *, preview=True):
        from .vjepa_validation import evaluate_autoregressive, save_autoregressive_preview
        self.state.update(global_training_step=self.global_step,
            validation_progress={'status': 'running', 'step': self.global_step, 'frames': 256})
        atomic_json(self.root/'status.json', self.state)
        report, contact = evaluate_autoregressive(model, self.validation, self.panel,
            context=self.args.context, horizon=self.args.horizon, batch_size=1,
            thresholds=self.quality_thresholds(), seed=self.args.seed, step=self.global_step)
        report['global_training_step'] = self.global_step
        atomic_json(self.root/f'{model.config.kind}_vjepa_quality_report.json', report)
        if preview:
            contact['metadata'].update(model_kind=model.config.kind, phase=self.state['stage'])
            self.state['prediction_preview'] = save_autoregressive_preview(contact, self.root/'preview.png',
                step=self.global_step, every=500)
        self.state['visual_quality'] = report
        self.state['validation_progress'] = {'status': 'complete', 'step': self.global_step, 'frames': 256,
                                              'seconds': report['seconds']}
        if model.config.kind == 'teacher':
            previous = self.state.get('teacher_quality', {})
            summary = report['summary']
            self.state['teacher_quality'] = {**previous, **summary, 'status': 'training',
                'matching_fraction': summary['pass_fraction'], 'frame_count': summary['frames'],
                'last_checked_step': self.global_step, 'required_passes': self.args.quality_passes,
                'target_psnr': self.args.quality_psnr, 'target_ssim': self.args.quality_ssim,
                'target_mae': self.args.quality_mae, 'required_fraction': self.args.quality_fraction,
                'scope': 'held_out_autoregressive_future_rgb', 'horizon': self.args.horizon}
        self.metric({'stage': self.state['stage'], 'step': self.global_step, 'validation': report['summary']})
        return report

    def preview_if_due(self, model):
        if self.global_step and self.global_step % 500 == 0:
            self.validation_report(model)

    def record_teacher_best(self, model, optimizer, phase, step, metrics, report):
        best = self.save(model, optimizer, phase+'_best', step, metrics)
        atomic_json(self.root/f'{phase}_best_report.json', report)
        self.state['teacher_plateau']['best_checkpoint'] = str(best)

    def observe_teacher_validation(self, model, optimizer, phase, step, metrics, report):
        """Optional selection hook; patience remains on the phase validation clock."""

    def fit(self, model, data, phase, steps=None, *, teacher=None, quality=False, pretrain=False):
        if pretrain:
            # The newly initialized encoder/predictor first learns the game;
            # preserve the warm-started renderer during this stage.
            for parameter in model.flow_renderer.parameters():
                parameter.requires_grad_(False)
            for parameter in model.vjepa_backend.renderer.parameters():
                parameter.requires_grad_(False)
        else:
            model.requires_grad_(True)
        self.ema = EMATargetEncoder(model.core.encoder, self.args.ema).to(self.device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=self.args.lr,
                                      weight_decay=.01)
        completed = self.restore(model, optimizer, phase)
        if self.extra.get('phase_complete'):
            if quality and not (self.root/'teacher_handoff_vjepa125.json').exists():
                reason = self.extra['teacher_handoff_reason']
                if reason == 'plateau':
                    best, _ = load_world_checkpoint(self.root/'checkpoints'/f'{phase}_best.pt', 'cpu')
                    model.load_state_dict(best.state_dict())
                    del best
                self.freeze_teacher(model, reason, self.extra['teacher_plateau'], completed)
            model.requires_grad_(True)
            return
        if not pretrain:
            self._setup_adversarial(phase)
        self._publish_model_sizes(model)
        panel_hash = self.extra.get('teacher_plateau', {}).get('fingerprint', '')
        tracker = PlateauTracker.from_state(self.extra.get('teacher_plateau'), patience=self.args.plateau_patience,
            min_relative_improvement=self.args.plateau_delta, phase=phase, fingerprint=panel_hash,
            current_step=completed)
        streak = int(self.extra.get('quality_streak', 0))
        label = ('Учитель: видеокодировщик и прогноз действий' if pretrain else
                 'Учитель: качество прогноза / Flow Matching / GAN' if quality else
                 'Студент: V-JEPA / Rectified Flow / дистилляция')
        self.update(phase, completed, steps, label, global_training_step=self.global_step,
                    message='Обучение на собственных состояниях; будущие RGB используются только как цели')
        model.train()
        started, initial_step, metrics = time.perf_counter(), completed, {}
        step = completed
        while steps is None or step < steps:
            self.check_stop(model, optimizer, phase, step, metrics)
            step += 1
            batch = data.batch(self.args.batch, context=self.args.context, horizon=self.args.horizon, rng=self.rng)
            optimizer.zero_grad(set_to_none=True)
            with self.amp(cache_enabled=False):
                loss_function = getattr(self, 'loss_function', short_rollout_loss)
                loss, metrics, endpoint = loss_function(model, self.ema, batch, context=self.args.context,
                    render=not pretrain, teacher=teacher, distill_endpoint=teacher is not None and step % 4 == 0)
                generated = None
                if not pretrain:
                    generated = model.flow_renderer.sample(endpoint['base'][:1], endpoint['state'][:1])
                    loss = loss + .5*F.l1_loss(generated.float(), endpoint['target'][:1])
            if generated is not None:
                gan_loss, gan_metrics = self._adversarial_loss(generated, endpoint['target'][:1],
                    endpoint['base'][:1], endpoint['state'][:1])
                loss = loss+gan_loss
                metrics.update(gan_metrics)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite {phase} loss at {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            self.ema.update(model.core.encoder)
            self.global_step += 1
            metrics['train_loss'] = metrics['optimization_loss'] = float(loss.detach())
            due = self.global_step % 500 == 0
            report = self.validation_report(model) if due else None
            if quality and report is not None:
                self.observe_teacher_validation(model, optimizer, phase, step, metrics, report)
            model.train()
            # The image publication clock is global. Plateau patience is tied
            # to500 actual updates of THIS teacher phase, including on resume.
            if quality and (step % 500 == 0 or step == 1):
                if report is None:
                    report = self.validation_report(model, preview=False)
                    model.train()
                summary = report['summary']
                score = summary['mae_mean']+.5*(1-summary['ssim_mean'])
                if tracker.fingerprint != report['panel']['fingerprint']:
                    tracker = PlateauTracker(patience=self.args.plateau_patience,
                        min_relative_improvement=self.args.plateau_delta, phase=phase,
                        fingerprint=report['panel']['fingerprint'])
                    streak = 0
                result = tracker.update_score(score, step)
                streak = streak+1 if report['passed'] else 0
                self.extra.update(teacher_plateau=tracker.to_state(), quality_streak=streak)
                self.state['teacher_plateau'] = {**result, 'enabled': True,
                    'status': 'plateau' if result['plateau'] else 'monitoring', 'score': score, 'check_every': 500}
                self.state['teacher_quality']['consecutive_passes'] = streak
                self.state['teacher_quality_gate'] = {'passed': bool(report['passed']), 'streak': streak,
                    'required_passes': self.args.quality_passes, 'thresholds': self.quality_thresholds(),
                    'summary': summary, 'scope': 'held_out_autoregressive_future_rgb'}
                if result['best_changed']:
                    self.record_teacher_best(model, optimizer, phase, step, metrics, report)
                if streak >= self.args.quality_passes or result['plateau']:
                    reason = 'quality_passed' if streak >= self.args.quality_passes else 'plateau'
                    self.state['teacher_quality']['status'] = 'passed' if reason == 'quality_passed' else 'plateau'
                    self.extra.update(phase_complete=True, teacher_handoff_reason=reason)
                    self.save(model, optimizer, phase, step, metrics, {'teacher_handoff_reason': reason})
                    if reason == 'plateau':
                        best, _ = load_world_checkpoint(self.root/'checkpoints'/f'{phase}_best.pt', 'cpu')
                        model.load_state_dict(best.state_dict())
                        del best
                    self.state['teacher_handoff_reason'] = reason
                    self.freeze_teacher(model, reason, tracker.to_state(), step)
                    break
            if step == initial_step+1 or step % self.args.log_every == 0 or due:
                if model.config.kind == 'student' and 'student_plan' in self.state:
                    self.state['student_plan'].update(status='training', last_phase=phase, last_phase_step=step)
                    atomic_json(self.root/'student_plan.json', self.state['student_plan'])
                if hasattr(data, 'files') and hasattr(data, 'count'):
                    self.state.setdefault('data', {}).update(rgb_training_files=data.files,
                        rgb_training_transitions=data.count)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                seconds = (time.perf_counter()-started)/max(1, step-initial_step)
                self.metric({'stage': phase, 'step': step, 'global_training_step': self.global_step,
                             **metrics, 'seconds_per_step': seconds})
                self.update(phase, step, steps, label, loss=metrics['train_loss'],
                    global_training_step=self.global_step,
                    eta_seconds=None if steps is None else round((steps-step)*seconds),
                    eta_scope='Текущий этап; время до требуемого качества заранее неизвестно',
                    throughput={'value': round(1/max(seconds, 1e-6), 3), 'unit': 'шагов/с'})
            if step == initial_step+1 or step % self.args.save_every == 0 or due:
                self.save(model, optimizer, phase, step, metrics)
        if steps is not None:
            self.extra['phase_complete'] = True
            self.save(model, optimizer, phase, step, metrics)
        del optimizer
        model.requires_grad_(True)

    def freeze_teacher(self, model, reason, plateau, step):
        fingerprint = model_fingerprint(model)
        selected_step = plateau.get('best_step', step) if reason == 'plateau' else step
        path = self.root/'checkpoints'/'teacher_for_student_vjepa125.pt'
        temporary = path.with_suffix('.pt.tmp')
        torch.save({'stage': 'world', 'config': asdict(model.config), 'model': model.state_dict(),
            'training_steps': selected_step, 'handoff_at_phase_step': step, 'global_training_step': self.global_step,
            'pipeline_phase': 'teacher_handoff_vjepa125', 'teacher_fingerprint': fingerprint,
            'quality_status': reason, 'teacher_plateau': plateau, 'created_at': now()}, temporary)
        replace_file(temporary, path)
        atomic_json(self.root/'teacher_handoff_vjepa125.json', {'checkpoint': str(path),
            'teacher_fingerprint': fingerprint, 'reason': reason, 'plateau': plateau,
            'global_training_step': self.global_step, 'created_at': now(),
            'quality_goal_passed': reason == 'quality_passed'})
        self.state.update(teacher_fingerprint=fingerprint, teacher_handoff_reason=reason)
        self.state['teacher_completion'] = {'reason': 'quality' if reason == 'quality_passed' else reason,
            'checkpoint': str(path), 'best_step': plateau.get('best_step'), 'best_score': plateau.get('best_score'),
            'quality_goal_passed': reason == 'quality_passed', 'teacher_fingerprint': fingerprint}

    def long_curriculum(self, model, kind):
        from .vjepa_long_rollout import train_curriculum
        phase = kind+'_vjepa_temporal'
        if self.state.get('adversarial'):
            self.state['adversarial']['active'] = False
        self.ema = EMATargetEncoder(model.core.encoder, self.args.ema).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr*.25, weight_decay=.01)
        completed = self.restore(model, optimizer, phase)
        if self.extra.get('phase_complete'):
            return
        def callback(progress):
            nonlocal completed
            completed += 1
            self.global_step += 1
            self.ema.update(model.core.encoder)
            self.extra['long_curriculum_state'] = progress['resume_state']
            metrics = progress['metrics']
            self.preview_if_due(model)
            model.train()
            self.update(phase, progress['completed_transitions'], progress['total_transitions'],
                ('Учитель' if kind == 'teacher' else 'Студент')+': автономные последовательности до 9 000 переходов',
                global_training_step=self.global_step, rollout_horizon=progress['horizon'],
                loss=metrics.get('loss'), eta_seconds=progress.get('eta_seconds'),
                message='TBPTT обрезает граф, но сохраняет собственные состояния и память без исправлений')
            if completed % self.args.log_every == 0:
                self.metric({'stage': phase, 'step': completed, 'global_training_step': self.global_step, **metrics})
            if completed == 1 or completed % self.args.save_every == 0 or self.global_step % 500 == 0:
                self.save(model, optimizer, phase, completed, metrics)
            self.check_stop(model, optimizer, phase, completed, metrics)
        report = train_curriculum(model, self.root/'data'/'long'/'train', optimizer,
            device=self.device, chunk_size=self.args.chunk, resume=self.extra.get('long_curriculum_state'),
            on_chunk=callback, seed=self.args.seed, return_directory=self.root/'data'/'memory',
            amp_dtype=torch.bfloat16 if self.device.type == 'cuda' else None,
            aligned_supervision=getattr(self.args, 'aligned_supervision', False))
        public_report = {key: value for key, value in report.items() if key != 'resume_state'}
        atomic_json(self.root/f'{kind}_vjepa_long_training.json', public_report)
        self.extra.update(phase_complete=True, long_curriculum_state=report['resume_state'])
        self.save(model, optimizer, phase, completed, report['metrics'])
        del optimizer

    def evaluate_9000(self, checkpoint, kind, round_index=0):
        """Independent held-out tapes, no resets/corrections during execution."""
        self.update(kind+'_evaluation', 0, 18000, ('Учитель' if kind == 'teacher' else 'Студент')+
                    ': проверка двух автономных прогонов по 5 минут', eta_seconds=None)
        reports = []
        report_prefix = kind if round_index == 0 else f'{kind}_round_{round_index:03d}'
        for index, scenario in enumerate(('explore', 'mixed')):
            output = self.root/f'{report_prefix}_vjepa_evaluation_{scenario}.json'
            command = [sys.executable, '-X', 'utf8', '-u', '-m', 'turkeyworld.evaluate_rollout',
                '--world', str(checkpoint), '--seconds', '300', '--episodes', '1', '--scenario', scenario,
                '--device', str(self.device), '--seed', str(41000+round_index*1000+index*100), '--threads', '4',
                '--render-every', '300', '--min-render-psnr', str(self.args.quality_psnr),
                '--continue-on-failure', '--output', str(output)]
            log_path = self.root/f'{report_prefix}.evaluation.{scenario}.log'
            with log_path.open('w', encoding='utf-8') as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                while process.poll() is None:
                    if (self.root/'stop_training').exists():
                        process.terminate()
                        process.wait()
                        self.state['status'] = 'stopped'
                        self.update(kind+'_evaluation', index*9000, 18000, 'Проверка остановлена; checkpoint сохранён')
                        raise SystemExit(0)
                    evaluation_step = 0
                    try:
                        for line in log_path.read_text(encoding='utf-8').splitlines()[-8:]:
                            try:
                                row = json.loads(line)
                                evaluation_step = max(evaluation_step, int(row.get('evaluation_step', 0)))
                            except (ValueError, TypeError):
                                continue
                    except OSError:
                        pass
                    self.update(kind+'_evaluation', index*9000+evaluation_step, 18000,
                        f'{kind}: автономный прогон {scenario}',
                        evaluation_progress={'model': kind, 'scenario': scenario, 'step': evaluation_step,
                            'requested_steps': 9000, 'simulated_seconds': evaluation_step/30,
                            'engine_corrections': 0, 'model_resets': 0})
                    time.sleep(10)
            if process.returncode:
                raise RuntimeError(f'Rollout evaluation failed; see {kind}.evaluation.{scenario}.log')
            reports.append(json.loads(output.read_text(encoding='utf-8')))
        result = {'status': 'complete', 'target_seconds': 300,
            'passed': all(r['300s_active_pass_count'] == r['episode_count'] for r in reports),
            'episodes': sum(r['episode_count'] for r in reports), 'reports': reports,
            'engine_corrections': 0, 'model_resets_during_rollout': 0,
            'checkpoint': str(checkpoint), 'round': round_index,
            'seed_base': 41000+round_index*1000}
        atomic_json(self.root/f'{kind}_vjepa_evaluation.json', result)
        self.state[kind+'_evaluation'] = result
        if kind == 'student':
            self.state['evaluation'] = result
        return result

    def qualify_student(self, student, teacher, data, initial_checkpoint):
        """Keep training a failed student; a finished schedule is not a pass.

        The teacher is frozen throughout. Every new repair round trains only
        the training corpus and uses fresh held-out evaluation trajectories.
        The durable pointer is written BEFORE starting that round, so restart
        continues its optimizer/TBPTT checkpoint rather than the baseline.
        """
        progress_path = self.root/'student_continuation_vjepa.json'
        if progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding='utf-8'))
            if progress.get('teacher_fingerprint') != self.state['teacher_fingerprint']:
                raise ValueError('Student continuation belongs to a different frozen teacher')
        else:
            progress = {'round': 0, 'status': 'evaluating', 'last_checkpoint': str(initial_checkpoint),
                        'teacher_fingerprint': self.state['teacher_fingerprint']}
            atomic_json(progress_path, progress)
        while True:
            round_index = int(progress['round'])
            self.state['student_continuation'] = progress
            if progress['status'] == 'qualified':
                qualified, metadata = load_world_checkpoint(progress['last_checkpoint'], 'cpu')
                if model_fingerprint(qualified) != progress['student_fingerprint']:
                    raise ValueError('Qualified student checkpoint fingerprint mismatch')
                student.load_state_dict(qualified.state_dict())
                self.global_step = max(self.global_step, int(metadata.get('global_training_step', 0)))
                del qualified
                student.eval()
                self.state['playable_checkpoint'] = progress['last_checkpoint']
                self.state['checkpoint'] = {'name': progress['last_checkpoint'], 'qualified': True,
                                           'global_training_step': self.global_step}
                result = json.loads((self.root/'student_vjepa_evaluation.json').read_text(encoding='utf-8'))
                self.state['evaluation'] = self.state['student_evaluation'] = result
                return result
            checkpoint = Path(progress['last_checkpoint'])
            if round_index:
                restored, metadata = load_world_checkpoint(checkpoint, 'cpu')
                student.load_state_dict(restored.state_dict())
                self.global_step = max(self.global_step, int(metadata.get('global_training_step', 0)))
                if 'rng_numpy' in metadata:
                    self.rng.bit_generator.state = metadata['rng_numpy']
                if 'rng_torch' in metadata:
                    torch.random.set_rng_state(metadata['rng_torch'].cpu())
                if self.device.type == 'cuda' and metadata.get('rng_cuda'):
                    torch.cuda.set_rng_state_all([value.cpu() for value in metadata['rng_cuda']])
                del restored
                self.fit(student, data, f'student_vjepa_repair_{round_index:03d}',
                         self.args.repair_steps, teacher=teacher)
                self._adversarial_trainer = None
                self._adversarial_phase = None
                long_kind = f'student_repair_{round_index:03d}'
                self.long_curriculum(student, long_kind)
                checkpoint = self.root/'checkpoints'/f'{long_kind}_vjepa_temporal.pt'
            self.ema = None
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            result = self.evaluate_9000(checkpoint, 'student', round_index=round_index)
            if result['passed']:
                progress.update(status='qualified', last_checkpoint=str(checkpoint), qualified_at=now(),
                                student_fingerprint=model_fingerprint(student))
                atomic_json(progress_path, progress)
                self.state['student_continuation'] = progress
                self.state['playable_checkpoint'] = str(checkpoint)
                self.state['checkpoint'] = {'name': str(checkpoint), 'qualified': True,
                                           'global_training_step': self.global_step}
                return result
            progress.update(round=round_index+1, status='repairing', last_checkpoint=str(checkpoint),
                last_failed_evaluation_round=round_index, updated_at=now(),
                reason='Independent five-minute evaluation did not pass; continue own-prediction training')
            atomic_json(progress_path, progress)
            self.metric({'stage': 'student_continuation', 'round': round_index+1,
                         'message': progress['reason'], 'evaluation_passed': False})

    def run(self):
        from .vjepa2_ac import migrate_vjepa_checkpoint
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.set_num_threads(4)
        atomic_json(self.root/'experiment_config_vjepa125.json', {k: str(v) if isinstance(v, Path) else v
                                                               for k, v in vars(self.args).items()})
        initial = self.root/'checkpoints'/'teacher_vjepa125_initial.pt'
        if not initial.exists():
            self.update('initializing', 0, 1, 'Перенос обученного Flow Matching-рендера; создание V-JEPA 125M')
            report = migrate_vjepa_checkpoint(self.args.source, initial)
            atomic_json(self.root/'vjepa125_migration.json', report)
        self.validation = SequenceData(self.root/'data'/'rgb'/'val', self.device, seed=self.args.seed)
        self.panel = self.validation.fixed_panel(count=64, context=self.args.context, horizon=self.args.horizon,
                                                 seed=self.args.seed)
        data = SequenceData(self.root/'data'/'rgb'/'train', self.device, seed=self.args.seed)
        handoff_path = self.root/'teacher_handoff_vjepa125.json'
        if handoff_path.exists():
            handoff = json.loads(handoff_path.read_text(encoding='utf-8'))
            teacher, _ = load_world_checkpoint(handoff['checkpoint'], self.device)
            if model_fingerprint(teacher) != handoff['teacher_fingerprint']:
                raise ValueError('Teacher handoff fingerprint mismatch')
            self.global_step = handoff['global_training_step']
            self.state.update(teacher_fingerprint=handoff['teacher_fingerprint'], teacher_handoff_reason=handoff['reason'])
            self.state['teacher_plateau'] = {**handoff['plateau'], 'enabled': True, 'check_every': 500}
            self.state['teacher_completion'] = {'reason': 'quality' if handoff['reason'] == 'quality_passed' else handoff['reason'],
                'checkpoint': handoff['checkpoint'], 'best_step': handoff['plateau'].get('best_step'),
                'best_score': handoff['plateau'].get('best_score'),
                'quality_goal_passed': handoff['quality_goal_passed']}
            quality_path = self.root/('teacher_quality_125m_best_report.json' if handoff['reason'] == 'plateau'
                                      else 'teacher_vjepa_quality_report.json')
            if quality_path.exists():
                quality = json.loads(quality_path.read_text(encoding='utf-8'))
                summary = quality['summary']
                self.state['teacher_quality'] = {**summary, 'status': 'plateau' if handoff['reason'] == 'plateau' else 'passed',
                    'matching_fraction': summary['pass_fraction'], 'frame_count': summary['frames'],
                    'last_checked_step': quality['global_training_step'], 'required_passes': self.args.quality_passes,
                    'consecutive_passes': self.args.quality_passes if handoff['quality_goal_passed'] else 0,
                    'scope': 'held_out_autoregressive_future_rgb', 'horizon': self.args.horizon}
        else:
            teacher, _ = load_world_checkpoint(initial, self.device)
            self.fit(teacher, data, 'vjepa_pretrain', self.args.pretrain_steps, pretrain=True)
            self.long_curriculum(teacher, 'teacher')
            self.fit(teacher, data, 'teacher_quality_125m', quality=True)
        teacher.eval().requires_grad_(False)
        self.ema = None
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        teacher_path = self.root/'checkpoints'/'teacher_for_student_vjepa125.pt'
        if not (self.root/'teacher_vjepa_evaluation.json').exists():
            self.evaluate_9000(teacher_path, 'teacher')
        else:
            self.state['teacher_evaluation'] = json.loads((self.root/'teacher_vjepa_evaluation.json').read_text(encoding='utf-8'))
        self.state['model_parameters']['teacher_total'] = sum(p.numel() for p in teacher.parameters())
        student = NeuralWorldModel(WorldModelConfig(kind='student', image_size=128, endless=True,
            architecture='vjepa2_ac', renderer_mode='rectified_flow', flow_steps=4)).to(self.device)
        self.fit(student, data, 'student_vjepa', self.args.student_steps, teacher=teacher)
        self._adversarial_trainer = None
        self._adversarial_phase = None
        self.long_curriculum(student, 'student')
        final = self.root/'checkpoints'/'student_vjepa_temporal.pt'
        self.ema = None
        result = self.qualify_student(student, teacher, data, final)
        self.state['status'] = 'completed'
        self.update('complete', 1, 1, 'Студент прошёл две независимые пятиминутные проверки', eta_seconds=0,
            message='Подробные результаты автономных проверок сохранены для учителя и студента')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--pretrain-steps', type=int, default=1500)
    parser.add_argument('--student-steps', type=int, default=16000)
    parser.add_argument('--repair-steps', type=int, default=2000)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ema', type=float, default=.996)
    parser.add_argument('--context', type=int, default=2)
    parser.add_argument('--horizon', type=int, default=4)
    parser.add_argument('--chunk', type=int, default=32)
    parser.add_argument('--seed', type=int, default=73)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--save-every', type=int, default=250)
    parser.add_argument('--plateau-patience', type=int, default=6)
    parser.add_argument('--plateau-delta', type=float, default=.005)
    parser.add_argument('--quality-passes', type=int, default=3)
    parser.add_argument('--quality-psnr', type=float, default=35.)
    parser.add_argument('--quality-ssim', type=float, default=.98)
    parser.add_argument('--quality-mae', type=float, default=2/255)
    parser.add_argument('--quality-fraction', type=float, default=.95)
    parser.set_defaults(teacher_steps=1500, temporal_updates=400, reflow_pairs=2048,
        validate_every=500, teacher_until_quality=True, teacher_size='125m', adversarial=True,
        adversarial_weight=.02, adversarial_feature_weight=.05, adversarial_warmup=64,
        adversarial_ramp=256, adversarial_r1_gamma=1., adversarial_r1_interval=16, adversarial_lr=1e-4)
    args = parser.parse_args(argv)
    if args.source is None:
        args.source = args.run_dir/'checkpoints'/'teacher_quality_75m.pt'
    if min(args.pretrain_steps, args.student_steps, args.repair_steps, args.batch, args.context, args.horizon, args.chunk,
           args.log_every, args.save_every, args.quality_passes) < 1:
        parser.error('Training counts must be positive')
    experiment = VJEPAExperiment(args)
    try:
        from .power_guard import KeepAwake
        with KeepAwake() as awake:
            experiment.state['power_guard'] = {'active': awake.active,
                'screen_lock_allowed': True, 'display_required': False, 'error': awake.error}
            experiment.run()
    except Exception as error:
        experiment.state['status'] = 'failed'
        experiment.update(experiment.state.get('stage', 'initializing'), experiment.state.get('current', 0),
                          experiment.state.get('total'), 'Ошибка обучения; checkpoint сохранён', error=str(error))
        raise


if __name__ == '__main__':
    main()
