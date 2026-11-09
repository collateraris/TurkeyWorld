"""250M teacher with expanded data and pose-aligned local supervision.

World learning is checked before long rollouts and adversarial rendering.
The independent main stream never receives future observations/corrections;
GT-pose auxiliary queries alone receive local combat/motion/image labels.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .aligned_training import aligned_short_rollout_loss
from .jepa_training_utils import SequenceData
from .models import NeuralWorldModel, WorldModelConfig, load_world_checkpoint
from .plateau import PlateauTracker
from .training_pipeline import Experiment, atomic_json, model_fingerprint, now, replace_file
from .vjepa_training import VJEPAExperiment
from .student_profile import STUDENT_22M_PARAMETERS, student_config_22m


class ExpandedExperiment(VJEPAExperiment):
    PROFILE = '250'
    PARAMETER_TARGET = 250000000
    STUDENT_PARAMETER_TARGET = STUDENT_22M_PARAMETERS
    STUDENT_PROFILE_NAME = 'vjepa_rf_22m'
    loss_function = staticmethod(aligned_short_rollout_loss)

    def __init__(self, args):
        super().__init__(args)
        self.state.update(model=f'Compact V-JEPA 2-AC {self.PROFILE}M / aligned dynamics / Flow Matching → Rectified Flow',
            teacher_size=f'{self.PROFILE}m', first_cycle_estimate=None,
            message=f'Подготовка расширенного корпуса и учителя{self.PROFILE}M; эталонные метки привязаны к точной позе')
        self.state['model_parameters'] = {'target_teacher_total': self.PARAMETER_TARGET}
        self.state['model_parameters']['target_student_total'] = self.STUDENT_PARAMETER_TARGET
        self.state['student_plan'] = {'status': 'pending', 'parameters': self.STUDENT_PARAMETER_TARGET,
            'profile': self.STUDENT_PROFILE_NAME, 'config': asdict(self.make_student_config()),
            'starts_after_teacher': True}
        plan_path = self.root/'student_plan.json'
        if plan_path.exists():
            previous_plan = json.loads(plan_path.read_text(encoding='utf-8'))
            if previous_plan.get('config') == self.state['student_plan']['config']:
                self.state['student_plan'].update(previous_plan)
                self.state['student_plan'].pop('training_steps', None)
        atomic_json(self.root/'student_plan.json', self.state['student_plan'])
        self.state['training_protocol'].update(aligned_auxiliary_supervision=True,
            future_ground_truth_feedback=False, dynamics_gate_before_gan=True, expanded_data=True)
        self.update('initializing', 0, 1, f'Подготовка V-JEPA{self.PROFILE}M и расширенного корпуса')

    def _publish_model_sizes(self, model):
        Experiment._publish_model_sizes(self, model)
        self.state['model_parameters']['target_teacher_total'] = self.PARAMETER_TARGET
        self.state['model_parameters']['target_student_total'] = self.STUDENT_PARAMETER_TARGET
        if self.ema is not None:
            self.state['model_parameters']['ema_target_total'] = self.ema.parameter_count

    def make_student_config(self):
        return student_config_22m()

    def make_student_model(self):
        return NeuralWorldModel(self.make_student_config()).to(self.device)

    def create_initial_teacher(self, initial):
        from . import vjepa2_ac
        migrate = getattr(vjepa2_ac, f'migrate_vjepa{self.PROFILE}_checkpoint')
        return migrate(self.args.source, initial)

    def record_teacher_best(self, model, optimizer, phase, step, metrics, report):
        score = report['summary']['mae_mean']+.5*(1-report['summary']['ssim_mean'])
        previous = self.extra.get('teacher_selection', {})
        if previous and score >= previous['score']:
            return
        self.extra['teacher_selection'] = {'score': score, 'phase_step': step,
            'global_step': self.global_step, 'phase': phase, 'source': 'all_actual_visual_validations'}
        path = self.save(model, optimizer, phase+'_best', step, metrics)
        atomic_json(self.root/f'{phase}_best_report.json', report)
        self.state['teacher_selection'] = {**self.extra['teacher_selection'], 'checkpoint': str(path)}

    def observe_teacher_validation(self, model, optimizer, phase, step, metrics, report):
        # All measured improvements preserve their weights; the plateau clock
        # still advances only on the distinct local500-update checks.
        self.record_teacher_best(model, optimizer, phase, step, metrics, report)

    def freeze_teacher(self, model, reason, plateau, step):
        fingerprint = model_fingerprint(model)
        selection = self.extra.get('teacher_selection', {})
        if reason == 'quality_passed':
            summary = self.state.get('visual_quality', {}).get('summary', {})
            selection = {'phase_step': step, 'global_step': self.global_step,
                'source': 'current_weights_after_required_consecutive_quality_passes'}
            if 'mae_mean' in summary and 'ssim_mean' in summary:
                selection['score'] = summary['mae_mean']+.5*(1-summary['ssim_mean'])
        selected_step = selection.get('phase_step', plateau.get('best_step', step))
        path = self.root/'checkpoints'/f'teacher_for_student_vjepa{self.PROFILE}.pt'
        temporary = path.with_suffix('.pt.tmp')
        torch.save({'stage': 'world', 'config': asdict(model.config), 'model': model.state_dict(),
            'training_steps': selected_step, 'handoff_at_phase_step': step,
            'global_training_step': self.global_step, 'pipeline_phase': f'teacher_handoff_vjepa{self.PROFILE}',
            'teacher_fingerprint': fingerprint, 'teacher_plateau': plateau,
            'teacher_selection': selection, 'quality_status': reason, 'created_at': now()}, temporary)
        replace_file(temporary, path)
        handoff = {'checkpoint': str(path), 'teacher_fingerprint': fingerprint, 'reason': reason,
            'plateau': plateau, 'selection': selection, 'global_training_step': self.global_step,
            'quality_goal_passed': reason == 'quality_passed', 'created_at': now()}
        atomic_json(self.root/f'teacher_handoff_vjepa{self.PROFILE}.json', handoff)
        self.state.update(teacher_fingerprint=fingerprint, teacher_handoff_reason=reason,
            teacher_completion={'reason': 'quality' if reason == 'quality_passed' else reason,
                'checkpoint': str(path), 'best_step': selection.get('global_step', selected_step),
                'best_score': selection.get('score', plateau.get('best_score')),
                'quality_goal_passed': reason == 'quality_passed'})

    def wait_for_data(self):
        status_path = self.root/'data'/'status.json'
        while True:
            if (self.root/'stop_training').exists():
                self.state['status'] = 'stopped'
                self.update('data', 0, None, 'Подготовка остановлена; данные сохраняются')
                raise SystemExit(0)
            if status_path.exists():
                report = json.loads(status_path.read_text(encoding='utf-8'))
                self.state['data_preparation'] = report
                state = report.get('status')
                ready_minimum = report.get('groups', {}).get('rgb/train', {}).get('files', 0) >= getattr(self.args, 'minimum_train_clips', 8192)
                if state in ('complete', 'completed') or ready_minimum:
                    own_path = self.root/'data'/'rgb'/'train'/'model_visited_manifest.json'
                    own = json.loads(own_path.read_text(encoding='utf-8')) if own_path.exists() else {}
                    if own.get('complete') and own.get('completed_clips', 0) >= self.args.visited_clips:
                        self.state['model_visited_data'] = {key: value for key, value in own.items() if key != 'records'}
                        break
                    self.update('data', own.get('completed_clips', 0), self.args.visited_clips,
                        'Разметка собственных состояний предыдущей модели', model_visited_data=own,
                        eta_seconds=None)
                    time.sleep(5)
                    continue
                if state == 'failed':
                    raise RuntimeError('Expanded data collection failed; see data/status.json')
                completed = report.get('completed', report.get('completed_jobs', 0))
                total = report.get('total', report.get('total_jobs'))
                self.update('data', completed, total, 'Сбор расширенного корпуса движения, боя и памяти',
                    eta_seconds=report.get('eta_seconds'))
            else:
                self.update('data', 0, None, 'Ожидание процесса сбора расширенного корпуса')
            time.sleep(5)

    def world_until_ready(self, model, data):
        from .dynamics_gate import evaluate_dynamics_gate
        pointer = self.root/f'teacher_world_progress{self.PROFILE}.json'
        progress = json.loads(pointer.read_text(encoding='utf-8')) if pointer.exists() else {'round': 0}
        if progress.get('finished'):
            restored, metadata = load_world_checkpoint(progress['checkpoint'], 'cpu')
            model.load_state_dict(restored.state_dict())
            self.global_step = max(self.global_step, metadata.get('global_training_step', 0))
            self.state['dynamics_gate'] = progress['gate']
            return progress['gate']['passed']
        tracker_state = progress.get('tracker')
        while True:
            index = progress['round']
            phase = f'teacher{self.PROFILE}_world' if index == 0 else f'teacher{self.PROFILE}_world_{index:03d}'
            if index and progress.get('checkpoint'):
                restored, metadata = load_world_checkpoint(progress['checkpoint'], 'cpu')
                model.load_state_dict(restored.state_dict())
                self.global_step = max(self.global_step, metadata.get('global_training_step', 0))
                del restored
            steps = self.args.pretrain_steps if index == 0 else self.args.world_repair_steps
            self.fit(model, data, phase, steps, pretrain=True)
            gate = evaluate_dynamics_gate(model, self.validation, getattr(self, 'gate_panel', self.panel), context=self.args.context,
                                          horizon=self.args.horizon, device=self.device)
            gate['global_training_step'] = self.global_step
            atomic_json(self.root/f'dynamics_gate_{index:03d}.json', gate)
            self.state['dynamics_gate'] = gate
            tracker = PlateauTracker.from_state(tracker_state, patience=self.args.world_patience,
                min_relative_improvement=.01, phase=f'teacher{self.PROFILE}_world_gate', fingerprint=gate['fingerprint'])
            result = tracker.update_score(gate['score'], self.global_step)
            checkpoint = self.root/'checkpoints'/f'{phase}.pt'
            progress.update(checkpoint=str(checkpoint), tracker=tracker.to_state(), gate=gate)
            if result['best_changed']:
                progress['best_checkpoint'] = str(checkpoint)
            self.metric({'stage': f'teacher{self.PROFILE}_dynamics_gate', 'step': self.global_step,
                         'passed': gate['passed'], 'score': gate['score'], **gate['metrics']})
            if gate['passed'] or result['plateau']:
                progress['finished'] = True
                if result['plateau'] and not gate['passed']:
                    best, metadata = load_world_checkpoint(progress['best_checkpoint'], 'cpu')
                    model.load_state_dict(best.state_dict())
                    self.extra['teacher_selection'] = {'phase_step': metadata['training_steps'],
                        'global_step': metadata['global_training_step'], 'score': tracker.best_score,
                        'source': 'held_out_dynamics_gate'}
                    del best
                    self.freeze_teacher(model, 'plateau', tracker.to_state(), self.global_step)
                    progress['checkpoint'] = str(self.root/'checkpoints'/f'teacher_for_student_vjepa{self.PROFILE}.pt')
                    self.state['teacher_world_plateau'] = tracker.to_state()
                atomic_json(pointer, progress)
                return bool(gate['passed'])
            progress.update(round=index+1)
            atomic_json(pointer, progress)
            tracker_state = tracker.to_state()

    def run(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.set_num_threads(4)
        atomic_json(self.root/f'experiment_config{self.PROFILE}.json',
            {key: str(value) if isinstance(value, Path) else value for key, value in vars(self.args).items()})
        initial = self.root/'checkpoints'/f'teacher_vjepa{self.PROFILE}_initial.pt'
        if not initial.exists():
            self.update('initializing', 0, 1, f'Расширение учителя до{self.PROFILE}M с сохранением обученных весов')
            report = self.create_initial_teacher(initial)
            atomic_json(self.root/f'vjepa{self.PROFILE}_migration.json', report)
        self.wait_for_data()
        # Training clips remain on disk with a bounded LRU; validation is small
        # enough to retain fixed source windows without changing the split.
        data = SequenceData(self.root/'data'/'rgb'/'train', self.device, seed=self.args.seed,
                            mode='stream', cache_bytes=self.args.cache_mb*1024**2)
        self.validation = SequenceData(self.root/'data'/'rgb'/'val', self.device, seed=self.args.seed,
                                      mode='stream', cache_bytes=min(self.args.cache_mb, 256)*1024**2)
        panel_path = self.root/f'validation_panel{self.PROFILE}.json'
        if panel_path.exists():
            panels = json.loads(panel_path.read_text(encoding='utf-8'))
            if panels['context'] != self.args.context or panels['horizon'] != self.args.horizon:
                raise ValueError('Frozen validation panel uses a different context/horizon')
            self.panel, self.gate_panel = panels['visual'], panels['dynamics']
        else:
            self.panel = self.validation.fixed_panel(count=64, context=self.args.context, horizon=self.args.horizon,
                                                     seed=self.args.seed)
            self.gate_panel = self.validation.fixed_panel(count=128, context=self.args.context, horizon=self.args.horizon,
                                                          seed=self.args.seed)
            atomic_json(panel_path, {'visual': self.panel, 'dynamics': self.gate_panel,
                'context': self.args.context, 'horizon': self.args.horizon, 'seed': self.args.seed})
        self.state['data'] = {'rgb_training_files': data.files, 'rgb_training_transitions': data.count,
            'rgb_validation_files': self.validation.files, 'rgb_validation_transitions': self.validation.count,
            'loader': 'bounded_episode_lru', 'cache_mb': self.args.cache_mb}
        handoff_path = self.root/f'teacher_handoff_vjepa{self.PROFILE}.json'
        if handoff_path.exists():
            handoff = json.loads(handoff_path.read_text(encoding='utf-8'))
            teacher, metadata = load_world_checkpoint(handoff['checkpoint'], self.device)
            if model_fingerprint(teacher) != handoff['teacher_fingerprint']:
                raise ValueError(f'{self.PROFILE}M teacher handoff fingerprint mismatch')
            self.global_step = handoff['global_training_step']
            self.state.update(teacher_fingerprint=handoff['teacher_fingerprint'], teacher_handoff_reason=handoff['reason'])
        else:
            teacher, _ = load_world_checkpoint(initial, self.device)
            if self.world_until_ready(teacher, data):
                self.long_curriculum(teacher, f'teacher{self.PROFILE}')
                original_lr = self.args.lr
                self.args.lr *= .2
                try:
                    self.fit(teacher, data, f'teacher_quality_{self.PROFILE}m', quality=True)
                finally:
                    self.args.lr = original_lr
        teacher.eval().requires_grad_(False)
        self.ema = None
        self._adversarial_trainer = None
        self._adversarial_phase = None
        self._publish_model_sizes(teacher)
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        teacher_path = self.root/'checkpoints'/f'teacher_for_student_vjepa{self.PROFILE}.pt'
        evaluation = self.root/'teacher_vjepa_evaluation.json'
        if not evaluation.exists():
            self.evaluate_9000(teacher_path, 'teacher')
        else:
            self.state['teacher_evaluation'] = json.loads(evaluation.read_text(encoding='utf-8'))
        student = self.make_student_model()
        self.state['student_plan']['status'] = 'training'
        self.state['student_plan']['actual_parameters'] = sum(parameter.numel() for parameter in student.parameters())
        if self.state['student_plan']['actual_parameters'] != self.STUDENT_PARAMETER_TARGET:
            raise ValueError('Instantiated student does not match the selected22.48M profile')
        atomic_json(self.root/'student_plan.json', self.state['student_plan'])
        self.fit(student, data, f'student_vjepa{self.PROFILE}_distill', self.args.student_steps, teacher=teacher)
        self._adversarial_trainer = None
        self._adversarial_phase = None
        self.long_curriculum(student, 'student')
        final = self.root/'checkpoints'/'student_vjepa_temporal.pt'
        self.qualify_student(student, teacher, data, final)
        self.state['student_plan'].update(status='qualified',
            checkpoint=self.state.get('playable_checkpoint', str(final)))
        atomic_json(self.root/'student_plan.json', self.state['student_plan'])
        self.state['status'] = 'completed'
        self.update('complete', 1, 1, 'Студент прошёл независимые пятиминутные проверки', eta_seconds=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('training_runs/vjepa250_20261109'))
    parser.add_argument('--source', type=Path, default=Path('training_runs/flow_20261108/checkpoints/teacher_for_student_vjepa125.pt'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--pretrain-steps', type=int, default=6000)
    parser.add_argument('--world-repair-steps', type=int, default=3000)
    parser.add_argument('--world-patience', type=int, default=6)
    parser.add_argument('--student-steps', type=int, default=16000)
    parser.add_argument('--repair-steps', type=int, default=2000)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument('--preview-every', type=int, default=10)
    parser.add_argument('--cache-mb', type=int, default=512)
    parser.add_argument('--visited-clips', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--ema', type=float, default=.996)
    parser.add_argument('--context', type=int, default=2)
    parser.add_argument('--horizon', type=int, default=4)
    parser.add_argument('--chunk', type=int, default=16)
    parser.add_argument('--seed', type=int, default=173)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--save-every', type=int, default=250)
    parser.add_argument('--plateau-patience', type=int, default=6)
    parser.add_argument('--plateau-delta', type=float, default=.005)
    parser.add_argument('--quality-passes', type=int, default=3)
    parser.add_argument('--quality-psnr', type=float, default=35.)
    parser.add_argument('--quality-ssim', type=float, default=.98)
    parser.add_argument('--quality-mae', type=float, default=2/255)
    parser.add_argument('--quality-fraction', type=float, default=.95)
    parser.set_defaults(teacher_steps=6000, temporal_updates=400, reflow_pairs=2048,
        validate_every=500, teacher_until_quality=True, teacher_size='250m', aligned_supervision=True,
        adversarial=True, adversarial_weight=.02, adversarial_feature_weight=.05,
        adversarial_warmup=512, adversarial_ramp=512, adversarial_r1_gamma=1.,
        adversarial_r1_interval=16, adversarial_lr=1e-4)
    args = parser.parse_args(argv)
    if min(args.pretrain_steps, args.world_repair_steps, args.world_patience, args.student_steps,
           args.repair_steps, args.batch, args.cache_mb, args.chunk, args.horizon, args.context,
           args.gradient_accumulation_steps, args.preview_every) < 1:
        parser.error('Counts must be positive')
    experiment = ExpandedExperiment(args)
    from .power_guard import KeepAwake
    try:
        with KeepAwake() as awake:
            experiment.state['power_guard'] = {'active': awake.active, 'display_required': False,
                'screen_lock_allowed': True, 'error': awake.error}
            experiment.run()
    except Exception as error:
        experiment.state['status'] = 'failed'
        experiment.update(experiment.state.get('stage', 'initializing'), experiment.state.get('current', 0),
                          experiment.state.get('total'), 'Ошибка обучения250M', error=str(error))
        raise


if __name__ == '__main__':
    main()
