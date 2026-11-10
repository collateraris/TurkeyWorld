"""Checkpointed renderer -> dynamics -> joint phases within a scene curriculum.

Phase handoffs are scheduling decisions. Plateau/budget never qualify gameplay.
The data plan and the full AdamW optimizer remain unchanged between phases.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from .navigation_experiment import experiment_fingerprint
from .navigation_sampling import NAVIGATION_CASES, NAVIGATION_FOCUS_WEIGHTS


COMPONENT_PHASES = (
    ('renderer', 'Base + Flow + GAN: настоящие состояния'),
    ('dynamics', 'Динамика: движение, скольжение и возвраты'),
    ('joint', 'Совместное дообучение и автономные прогоны'),
)


class ComponentCurriculum:
    def __init__(self, path, *, step, base_plan_fingerprint, previous=None):
        self.path = Path(path).resolve()
        plan = json.loads(self.path.read_text(encoding='utf-8'))
        if (plan.get('version') != 1 or plan.get('base_plan_fingerprint') != base_plan_fingerprint
                or plan.get('validation_every') != 100 or plan.get('preview_every') != 10
                or plan.get('microbatch_size') != 16 or plan.get('scene_at_start') != 'navigation'
                or [p.get('id') for p in plan.get('phases', [])] != [p[0] for p in COMPONENT_PHASES]
                or plan.get('navigation_weights') != NAVIGATION_FOCUS_WEIGHTS
                or plan.get('required_passes') != 2 or plan.get('plateau_patience') != 5
                or plan.get('plateau_relative_delta') != .01):
            raise ValueError('Invalid component phase plan or changed scene data identity')
        for phase in plan['phases'][:2]:
            low, high = phase.get('minimum_updates'), phase.get('budget_updates')
            if (not isinstance(low, int) or isinstance(low, bool) or low < 1
                    or not isinstance(high, int) or isinstance(high, bool) or high < low):
                raise ValueError('Component phases require positive minimum/budget bounds')
        self.plan = plan
        fingerprint = experiment_fingerprint(plan)
        if previous:
            if previous.get('fingerprint') != fingerprint or previous.get('last_optimizer_step') != step:
                raise ValueError('Component phase plan or saved optimizer boundary changed')
            self.state = copy.deepcopy(previous)
        else:
            if step != plan.get('start_optimizer_step'):
                raise ValueError('New component phases must start at the saved optimizer boundary')
            phases = []
            for index, ((name, label), spec) in enumerate(zip(COMPONENT_PHASES, plan['phases'])):
                enabled = name != 'dynamics'
                phases.append({**spec, 'label': label, 'status': 'active' if index == 0 else 'pending',
                    'updates': 0, 'losses': {}, 'gate': {'status': 'pending'},
                    'coordinate_source': 'source_gt' if name == 'renderer' else 'own_predictions',
                    'model_component_status': {'base': 'trainable' if enabled else 'frozen',
                        'flow': 'trainable' if enabled else 'frozen',
                        'gan': 'trainable' if enabled else 'frozen', 'codec': 'frozen',
                        'map': 'trainable' if name == 'joint' else 'frozen',
                        'dynamics': 'frozen' if name == 'renderer' else 'trainable'}})
            phases[0]['phase_start_step'] = step
            self.state = {'version': 1, 'fingerprint': fingerprint, 'plan': str(self.path),
                'phase_index': 0, 'phase_start_step': step, 'last_optimizer_step': step,
                'start_optimizer_step': step, 'validation_every': 100, 'preview_every': 10,
                'status': 'training', 'history': [], 'phases': phases,
                'scene_at_start': 'navigation', 'scene_progression': 'joint_phase_only',
                'plateau_patience': 5, 'plateau_relative_delta': .01,
                'gameplay_qualified': False, 'frozen_shared_map_until_joint': True}
        self.sync()

    @property
    def phase(self):
        return self.state['phases'][self.state['phase_index']]

    @property
    def name(self):
        return self.phase['id']

    def sync(self):
        phase = self.phase
        self.state.update(phase=phase['id'], phase_label=phase['label'],
            phase_updates=phase['updates'], phase_start_step=phase['phase_start_step'],
            losses=phase['losses'], loss_step=phase.get('loss_step'), gate=phase['gate'],
            model_component_status=phase['model_component_status'],
            coordinate_source=phase['coordinate_source'])
        return self.state

    def record_update(self, step, metrics):
        if step != self.state['last_optimizer_step']+1:
            raise ValueError('Component counters require exactly one completed optimizer update')
        self.state['last_optimizer_step'] = step
        self.phase['updates'] += 1
        self.phase['loss_step'] = step
        self.phase['losses'] = {key: value for key, value in metrics.items()
            if not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)}
        self.sync()

    def validation_due(self, step):
        gate_step = self.phase['gate'].get('step')
        return (gate_step is None or step//100 > gate_step//100
            or self.name != 'joint' and self.phase['updates'] >= self.phase['budget_updates'])

    def record_validation(self, report, step, *, gameplay_qualified=False):
        """Return True on handoff; every check records what was actually tested."""
        if step != self.state['last_optimizer_step']:
            raise ValueError('Component validation must match the completed optimizer boundary')
        phase = self.phase
        if phase['gate'].get('step') == step:
            raise ValueError('A component validation step cannot be counted twice')
        metrics = report.get('metrics', {})
        if self.name == 'renderer':
            valid = metrics.get('rgb_finite') is True and metrics.get('base_rgb_finite', True) is True
            passed = valid and metrics.get('rgb_ssim', -1) >= .90 and metrics.get('rgb_mae', 1) <= .03
            score = metrics.get('rgb_mae', 1)+(1-metrics.get('rgb_ssim', 0)) if valid else None
            limits = {'rgb_ssim_min': .90, 'rgb_mae_max': .03}
        elif self.name == 'dynamics':
            valid = metrics.get('all_states_finite') is True and metrics.get('all_memory_finite') is True
            passed = (valid and metrics.get('player_xy_mean', 1) <= .10
                and metrics.get('player_xy_p95', 1) <= .25 and metrics.get('wall_windows', 1) == 0)
            score = (metrics.get('player_xy_mean', 1)+.25*metrics.get('player_xy_p95', 1)
                +.25*metrics.get('wall_windows', 1)/max(1, metrics.get('trajectory_windows', 32))) if valid else None
            limits = {'player_xy_mean_max': .10, 'player_xy_p95_max': .25, 'wall_windows_max': 0}
        else:
            phase['gate'] = {'status': 'passed' if gameplay_qualified else 'pending', 'step': step,
                'scene_quality_passed': report.get('passed') is True, 'metrics': metrics,
                'thresholds': report.get('thresholds', {}), 'report': report.get('report_path'),
                'scope': report.get('scope'), 'gameplay_qualified': gameplay_qualified}
            if gameplay_qualified:
                phase.update(status='completed', completion_reason='quality', completed_step=step)
                self.state.update(status='completed', gameplay_qualified=True)
            self.sync()
            return False
        eligible = phase['updates'] >= phase['minimum_updates']
        streak = phase.get('pass_streak', 0)+1 if passed and eligible else 0
        best = phase.get('best_score')
        if score is not None and math.isfinite(score):
            improved = best is None or score < best*(1-self.plan['plateau_relative_delta'])
            if improved:
                phase.update(best_score=score, checks_without_improvement=0)
            else:
                phase['checks_without_improvement'] = phase.get('checks_without_improvement', 0)+1
        else:
            phase['checks_without_improvement'] = 0
        phase['pass_streak'] = streak
        phase['gate'] = {'status': 'passed' if passed else 'failed', 'step': step, 'metrics': metrics,
            'thresholds': limits, 'report': report.get('report_path'), 'quality_passed': bool(passed),
            'minimum_updates_met': eligible, 'pass_streak': streak, 'required_passes': 2,
            'checks_without_improvement': phase.get('checks_without_improvement', 0),
            'plateau_patience': 5, 'score': score, 'gameplay_qualified': False,
            'scope': 'GT-pose renderer queries' if self.name == 'renderer' else 'Own-state navigation trajectories'}
        reason = ('quality' if eligible and streak >= 2 else
            'plateau' if valid and eligible and phase.get('checks_without_improvement', 0) >= 5 else
            'budget' if phase['updates'] >= phase['budget_updates'] else None)
        if reason:
            phase.update(status='completed', completion_reason=reason, completed_step=step)
            self.state['history'].append(copy.deepcopy(phase))
            self.state['phase_index'] += 1
            self.phase.update(status='active', phase_start_step=step)
            if self.name == 'joint' and 'stop_optimizer_step' in self.plan:
                self.phase['budget_updates'] = max(0, self.plan['stop_optimizer_step']-step)
        self.sync()
        return reason is not None

    def dynamics_sampling(self):
        if self.name != 'dynamics':
            raise ValueError('Focused motion sampling is reserved for the dynamics phase')
        return self.phase.setdefault('sampling', {'weights': dict(NAVIGATION_FOCUS_WEIGHTS),
            'sample_counts': {case: 0 for case in NAVIGATION_CASES}, 'draws': 0,
            'source_fingerprint': self.plan['navigation_source_fingerprint']})
