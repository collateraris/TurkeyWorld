"""Independent spawn/recovery scales on top of the preserved actor balancer."""
from __future__ import annotations

from copy import deepcopy
import math

from .rssm_loss_balance import (GROUPS, GroupLossAccumulator, GroupLossBalancer,
                               LOSS_PROTOCOL as ACTOR_PROTOCOL)


LOSS_PROTOCOL = 'rssm_dual_combat_components_v6'
NORMALIZATION = 'lagged_actor_and_component_ema'
COMPONENT_GROUPS = {'spawn': ('enemy',), 'recovery': GROUPS}
SETTINGS = {'version': 1, 'ema_decay': .95, 'minimum_scale': .1,
    'maximum_priority': 1., 'minimum_priority': .001,
    'maximum_calibrated_target_multiplier': 1.1,
    'target_gradient_fraction': {'spawn': .20, 'recovery': .15},
    'normalization': NORMALIZATION}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


class ComponentLossAccumulator(GroupLossAccumulator):
    def __init__(self, scales, priorities, component_scales, component_priorities):
        super().__init__(scales, priorities)
        self.component_scales = deepcopy(component_scales)
        self.component_priorities = deepcopy(component_priorities)
        self.base_raw = dict.fromkeys(GROUPS, 0.)

    def combine(self, parts, *, floors=None, component=None):
        if component is None:
            before = dict(self.raw)
            result = super().combine(parts, floors=floors)
            for name in GROUPS:self.base_raw[name] += self.raw[name]-before[name]
            return result
        if component not in COMPONENT_GROUPS or set(parts)-set(COMPONENT_GROUPS[component]) or floors:
            raise ValueError('Unexpected actor/component assignment')
        # Reuse scalar validation and bookkeeping with independent coefficients.
        original_scales, original_priorities = self.scales, self.priorities
        try:
            self.scales = {**original_scales, **self.component_scales[component]}
            self.priorities = {**original_priorities, **self.component_priorities[component]}
            return super().combine(parts, component=component)
        finally:
            self.scales, self.priorities = original_scales, original_priorities

    def report(self):
        result = super().report()
        for name in GROUPS:
            row = result[name]
            row['base_raw'] = self.base_raw[name]
            row['base_normalized'] = self.base_raw[name]/row['scale']
            row['components'] = {}
            variable = row['base_normalized']*row['weight']
            for component, actors in COMPONENT_GROUPS.items():
                if name not in actors:continue
                raw = self.component_raw.get(component, {}).get(name, 0.)
                scale = self.component_scales[component][name]
                priority = self.component_priorities[component][name]
                detail = {'raw': raw, 'scale': scale, 'priority': priority,
                    'normalized': raw/scale, 'weighted': raw*priority/scale,
                    'effective_weight': priority/scale, 'active': raw > 1e-8}
                row['components'][component] = detail
                variable += detail['weighted']
            row['normalized'] = variable/row['weight']
            row['weighted'] = variable+row['kl_floor']
        result.update(total=sum(result[name]['weighted'] for name in GROUPS),
            normalization=NORMALIZATION, loss_protocol=LOSS_PROTOCOL)
        return result


def initialize_component_balancer(actor_state, calibration, *, step):
    """Seed only new component banks from a measured, non-optimizer probe."""
    state = deepcopy(actor_state)
    GroupLossBalancer(state)
    if state['last_optimizer_step'] != step or calibration.get('source_step') != step:
        raise ValueError('Gradient calibration must use the exact complete source boundary')
    if calibration.get('passed') is not True or state.get('components') is not None:
        raise ValueError('Need a passed fresh gradient calibration, without overwriting components')
    priorities, initial = {}, {}
    for component, actors in COMPONENT_GROUPS.items():
        priorities[component], initial[component] = {}, {}
        for name in actors:
            row = calibration['components'][component][name]
            raw, priority = row['raw'], row['chosen_priority']
            if (not _finite(raw) or not _finite(priority)
                    or not SETTINGS['minimum_priority'] <= priority <= SETTINGS['maximum_priority']):
                raise ValueError('Invalid measured component initialization')
            if row.get('active') is True:
                expected, target = row.get('expected_gradient_norm'), row.get('target_gradient_norm')
                if (not _finite(expected) or not _finite(target)
                        or expected > target*SETTINGS['maximum_calibrated_target_multiplier']+1e-8):
                    raise ValueError('Clamped component priority exceeds the measured gradient budget')
            priorities[component][name] = priority
            initial[component][name] = raw if raw > 1e-8 else None
    state['components'] = {'settings': deepcopy(SETTINGS), 'priorities': priorities,
        'committed_updates': 0, 'last_optimizer_step': step, 'start_optimizer_step': step,
        'calibration_source_step': step, 'initialization': 'offline_gradient_measurement_no_optimizer',
        'banks': {environment: {'updates': 0,
            'ema': deepcopy(initial) if environment == 'combat' else {
                component: {name: None for name in actors} for component, actors in COMPONENT_GROUPS.items()}}
            for environment in ('combat', 'navigation')}}
    return state


class ComponentLossBalancer(GroupLossBalancer):
    def __init__(self, state):
        super().__init__(state)
        comp = self.state.get('components')
        if (not isinstance(comp, dict) or comp.get('settings') != SETTINGS
                or type(comp.get('committed_updates')) is not int
                or comp['committed_updates'] < 0
                or comp.get('last_optimizer_step') != self.state['last_optimizer_step']
                or comp['committed_updates'] != comp['last_optimizer_step']-comp.get('start_optimizer_step', -1)
                or set(comp.get('banks', {})) != {'combat', 'navigation'}):
            raise ValueError('Component normalization checkpoint identity/counters changed')
        if sum(bank.get('updates', -1) for bank in comp['banks'].values()) != comp['committed_updates']:
            raise ValueError('Component environment clocks disagree')
        for component, actors in COMPONENT_GROUPS.items():
            priorities = comp.get('priorities', {}).get(component, {})
            if set(priorities) != set(actors) or any(not _finite(value)
                    or not SETTINGS['minimum_priority'] <= value <= SETTINGS['maximum_priority']
                    for value in priorities.values()):
                raise ValueError('Component priorities differ from the calibration')
            for bank in comp['banks'].values():
                values = bank.get('ema', {}).get(component, {})
                if (type(bank.get('updates')) is not int or bank['updates'] < 0
                        or set(values) != set(actors)
                        or any(value is not None and not _finite(value) for value in values.values())):
                    raise ValueError('Invalid component EMA bank')

    def new_microbatch(self, environment):
        parent = super().new_microbatch(environment)
        comp = self.state['components'];bank = comp['banks'][environment]
        scales = {component: {name: max(SETTINGS['minimum_scale'], value) if value is not None else 1.
            for name, value in values.items()} for component, values in bank['ema'].items()}
        return ComponentLossAccumulator(parent.scales, parent.priorities, scales, comp['priorities'])

    def summarize(self, reports, *, environment, step):
        if not reports:raise ValueError('No completed component microbatches')
        expected = self.new_microbatch(environment).report();rows = {}
        for name in GROUPS:
            row = {}
            for key in ('raw', 'base_raw', 'base_normalized', 'normalized', 'weighted', 'kl_floor'):
                values = [report[name][key] for report in reports]
                if not all(_finite(value) for value in values):
                    raise FloatingPointError('Nonfinite component accounting')
                row[key] = sum(values)/len(values)
            for key in ('scale', 'weight', 'effective_weight'):
                if any(report[name][key] != expected[name][key] for report in reports):
                    raise ValueError('Actor scale changed inside a component optimizer update')
                row[key] = expected[name][key]
            row['active'] = row['raw'] > 1e-8;row['components'] = {}
            for component, reference in expected[name]['components'].items():
                detail = {}
                for key in ('scale', 'priority', 'effective_weight'):
                    if any(report[name]['components'][component][key] != reference[key] for report in reports):
                        raise ValueError('Component coefficient changed inside accumulated update')
                    detail[key] = reference[key]
                for key in ('raw', 'normalized', 'weighted'):
                    values = [report[name]['components'][component][key] for report in reports]
                    if not all(_finite(value) for value in values):raise FloatingPointError('Nonfinite component loss')
                    detail[key] = sum(values)/len(values)
                detail['active'] = detail['raw'] > 1e-8
                row['components'][component] = detail
            rows[name] = row
        return {**rows, 'total': sum(row['weighted'] for row in rows.values()),
            'kl_floor': sum(row['kl_floor'] for row in rows.values()), 'step': step,
            'loss_protocol': LOSS_PROTOCOL, 'normalization': NORMALIZATION,
            'calibrating': False, 'ema_updates_before': self.state['committed_updates'],
            'component_updates_before': self.state['components']['committed_updates'],
            'environment': environment, 'microbatches': len(reports),
            'scale_update': 'after_optimizer_commit_only'}

    def commit(self, report, *, step):
        comp = self.state['components']
        if (report.get('loss_protocol') != LOSS_PROTOCOL or report.get('step') != step
                or report.get('component_updates_before') != comp['committed_updates']
                or step != comp['last_optimizer_step']+1):
            raise ValueError('Components commit only once after the corresponding Adam update')
        # Validate all new values before mutating either parent or component EMA.
        for name in GROUPS:
            if not _finite(report[name]['base_raw']):raise FloatingPointError('Nonfinite base actor loss')
            for detail in report[name]['components'].values():
                if not _finite(detail['raw']):raise FloatingPointError('Nonfinite component loss')
        parent = deepcopy(report);parent['loss_protocol'] = ACTOR_PROTOCOL
        for name in GROUPS:parent[name]['raw'] = report[name]['base_raw']
        super().commit(parent, step=step)
        bank = comp['banks'][report['environment']]
        for component, actors in COMPONENT_GROUPS.items():
            for name in actors:
                value = report[name]['components'][component]['raw']
                if value <= 1e-8:continue
                old = bank['ema'][component][name]
                bank['ema'][component][name] = value if old is None else SETTINGS['ema_decay']*old+(1-SETTINGS['ema_decay'])*value
        bank['updates'] += 1;comp['committed_updates'] += 1;comp['last_optimizer_step'] = step
