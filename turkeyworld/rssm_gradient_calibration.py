"""Read-only component gradient measurements before independent reweighting."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math

import torch

from .rssm_component_balance import COMPONENT_GROUPS, SETTINGS


def _norm(values):
    return math.sqrt(sum(float(value.double().square().sum()) for value in values))


def _route(name):
    if name.startswith('vjepa_backend.enemy_rssm.core.'): return 'enemy_core'
    if name.startswith('vjepa_backend.enemy_rssm.'): return 'enemy_heads'
    if name.startswith('vjepa_backend.core.'): return 'player_core'
    if name.startswith(('vjepa_backend.spawn_', 'vjepa_backend.spatial_spawn_',
            'vjepa_backend.hit_target_head.', 'vjepa_backend.attack_head.',
            'vjepa_backend.spatial_combat_head.')): return 'shared_combat_spawn_heads'
    return 'other_trainable'


def reprioritize_calibration(measurement, *, settings=None):
    """CPU arithmetic only; retain every measured vector norm and source field."""
    config = deepcopy(SETTINGS if settings is None else settings)
    result = deepcopy(measurement)
    base = result['base_gradient_norm']
    if not math.isfinite(base) or base <= 0:
        raise ValueError('Reprioritization requires a real finite positive measured base norm')
    budgets = []
    for component, actors in COMPONENT_GROUPS.items():
        for actor in actors:
            row = result['components'][component][actor]
            raw, norm = row['raw'], row['raw_gradient_norm']
            if any(not math.isfinite(value) or value < 0 for value in (raw, norm)):
                raise ValueError('Invalid immutable raw component measurements')
            scale = max(raw, config['minimum_scale'])
            normalized = norm/scale
            target = config['target_gradient_fraction'][component]*base
            unclamped = target/normalized if normalized > 0 else config['minimum_priority']
            priority = max(config['minimum_priority'], min(config['maximum_priority'], unclamped))
            expected = norm*priority/scale
            passed = norm == 0 or expected <= target*config['maximum_calibrated_target_multiplier']+1e-8
            row.update(normalized_gradient_norm=normalized, chosen_priority=priority,
                effective_coefficient=priority/scale, expected_gradient_norm=expected,
                target_gradient_norm=target, unclamped_priority=unclamped,
                priority_lower_clamp_active=unclamped < config['minimum_priority'],
                priority_budget_passed=passed)
            budgets.append(passed)
    result['settings'] = config
    result['passed_priority_budget_check'] = all(budgets)
    return result


class ComponentGradientMeter:
    """Accumulate raw gradient vectors across all graphs without touching .grad.

    ``capture`` is called before the normal backward by the legacy actor
    accumulator. Actual coefficients remain unchanged during this probe.
    CPU vectors are retained only by this discarded diagnostic object.
    """
    def __init__(self, named_parameters, actor_coefficients):
        self.parameters = {name: parameter for name, parameter in named_parameters if parameter.requires_grad}
        if not self.parameters:
            raise ValueError('Gradient calibration needs actual trainable model parameters')
        self.coefficients = dict(actor_coefficients)
        if any(not math.isfinite(value) or value <= 0 for value in self.coefficients.values()):
            raise ValueError('Positive finite unchanged actor coefficients are required')
        self.raw = defaultdict(float)
        self.vectors = defaultdict(dict)
        self.calls = defaultdict(int)

    def capture(self, component, actor, loss):
        if component not in COMPONENT_GROUPS or actor not in COMPONENT_GROUPS[component]:
            raise ValueError('Unexpected measured actor/component assignment')
        if not torch.is_tensor(loss) or loss.numel() != 1 or not bool(torch.isfinite(loss)):
            raise FloatingPointError('Component calibration requires a finite scalar loss')
        key = (component, actor)
        self.raw[key] += float(loss.detach())
        self.calls[key] += 1
        if not loss.requires_grad: return
        gradients = torch.autograd.grad(loss, tuple(self.parameters.values()), retain_graph=True, allow_unused=True)
        for (name, _), gradient in zip(self.parameters.items(), gradients):
            if gradient is None: continue
            value = gradient.detach().to(device='cpu', dtype=torch.float64)
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError('Nonfinite raw component gradient: '+component+'/'+actor+'/'+name)
            if name in self.vectors[key]: self.vectors[key][name].add_(value)
            else: self.vectors[key][name] = value.clone()

    def finish(self):
        actual = {name: p.grad.detach().to(device='cpu', dtype=torch.float64).clone()
            for name, p in self.parameters.items() if p.grad is not None}
        if not actual or any(not bool(torch.isfinite(value).all()) for value in actual.values()):
            raise FloatingPointError('Actual baseline backward has no finite gradient')
        base = {name: value.clone() for name, value in actual.items()}
        for (_, actor), vectors in self.vectors.items():
            for name, value in vectors.items():
                if name not in base: base[name] = torch.zeros_like(value)
                base[name].sub_(value, alpha=self.coefficients[actor])
        base_norm = _norm(base.values())
        if not math.isfinite(base_norm) or base_norm <= 0:
            raise FloatingPointError('Cannot calibrate components against an absent/nonfinite base gradient')
        output = {'base_gradient_norm': base_norm, 'total_gradient_norm': _norm(actual.values()),
            'base_route_gradient_norms': {route: _norm(v for n, v in base.items() if _route(n) == route)
                for route in ('player_core', 'enemy_core', 'enemy_heads', 'shared_combat_spawn_heads', 'other_trainable')},
            'components': {}}
        for component, actors in COMPONENT_GROUPS.items():
            output['components'][component] = {}
            for actor in actors:
                key = (component, actor)
                vectors, raw = self.vectors[key], self.raw[key]
                norm = _norm(vectors.values())
                scale = max(raw, SETTINGS['minimum_scale'])
                normalized = norm/scale
                desired = SETTINGS['target_gradient_fraction'][component]*base_norm
                priority = (desired/normalized if normalized > 0 else SETTINGS['minimum_priority'])
                priority = max(SETTINGS['minimum_priority'], min(SETTINGS['maximum_priority'], priority))
                coefficient = priority/scale
                output['components'][component][actor] = {'raw': raw,
                    'raw_gradient_norm': norm, 'normalized_gradient_norm': normalized,
                    'chosen_priority': priority, 'effective_coefficient': coefficient,
                    'expected_gradient_norm': norm*coefficient, 'target_gradient_norm': desired,
                    'baseline_actor_coefficient': self.coefficients[actor],
                    'baseline_weighted_gradient_norm': norm*self.coefficients[actor],
                    'capture_calls': self.calls[key], 'active': norm > 0,
                    'route_gradient_norms': {route: _norm(v for n, v in vectors.items() if _route(n) == route)
                        for route in ('player_core', 'enemy_core', 'enemy_heads', 'shared_combat_spawn_heads', 'other_trainable')}}
        return reprioritize_calibration(output)
