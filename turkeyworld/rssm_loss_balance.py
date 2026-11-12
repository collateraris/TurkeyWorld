"""Lagged, checkpointed task normalization for the two causal RSSMs.

Scales are frozen for an entire optimizer update, including all TBPTT graphs
and accumulated microbatches. Only successfully committed updates change EMA.
The constant free-KL floor is reported, but never enters the normalization.
"""
from __future__ import annotations

from copy import deepcopy
import math

import torch

from .structured_state import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS


GROUPS = ('player', 'enemy', 'interaction')
LOSS_PROTOCOL = 'rssm_dual_group_balance_v5'
SETTINGS = {'version': 1, 'ema_decay': .95, 'minimum_scale': .1,
            'priorities': {name: 1. for name in GROUPS},
            'normalization': 'lagged_ema_without_constant_kl_floor'}
COMBAT_GROUPS = {'combat_events': 'interaction', 'combat_target': 'interaction',
    'combat_attack': 'interaction', 'combat_damage_npc': 'interaction',
    'combat_damage_player': 'interaction', 'combat_death_npc': 'interaction',
    'combat_spawn': 'enemy', 'combat_npc_motion': 'enemy',
    'combat_kinematic_player_axes': 'player', 'combat_kinematic_npc_speed': 'enemy',
    'combat_kinematic_npc_direction': 'enemy'}


def split_state_loss(prediction, target):
    """Disjoint partition of the existing field-weighted 20D state objective.

    Keep its original B x state-dimension denominator. Independent task scales
    below provide normalization without silently changing individual labels.
    HP, life flags, combat timers, kills and clock belong to interaction.
    """
    if prediction.shape != target.shape or prediction.shape[-1] != STATE_DIM:
        raise ValueError('Grouped state loss requires matching explicit states')
    weights = torch.ones(STATE_DIM, device=prediction.device, dtype=prediction.dtype)
    weights[:4] = 4
    weights[4:6] = 2
    entity = torch.ones(ENTITY_DIM, device=prediction.device, dtype=prediction.dtype)
    entity[:2] = 3
    entity[4:6] = 4
    weights[GLOBAL_DIM:-2] = entity.repeat(ENTITY_SLOTS)
    indices = {'player': list(range(4)), 'enemy': []}
    for slot in range(ENTITY_SLOTS):
        first = GLOBAL_DIM + slot * ENTITY_DIM
        indices['enemy'].extend([first, first+1, first+2, first+3, first+9])
    occupied = set(indices['player'] + indices['enemy'])
    indices['interaction'] = [index for index in range(STATE_DIM) if index not in occupied]
    error = (prediction-target).square()*weights
    return {name: error[..., ids].sum()/error.numel() for name, ids in indices.items()}


def combat_loss_groups(terms):
    unknown = set(terms)-set(COMBAT_GROUPS)
    if unknown:
        raise ValueError('Unassigned combat objective: '+str(sorted(unknown)))
    result = {}
    for name, value in terms.items():
        group = COMBAT_GROUPS[name]
        result[group] = result.get(group, 0)+value
    return result


class GroupLossAccumulator:
    """One microbatch's differentiable task sums with immutable scale values."""
    def __init__(self, scales=None, priorities=None, *, gradient_meter=None):
        self.scales = {name: float((scales or {}).get(name, 1.)) for name in GROUPS}
        self.priorities = {name: float((priorities or {}).get(name, 1.)) for name in GROUPS}
        if not all(math.isfinite(x) and x > 0 for x in (*self.scales.values(), *self.priorities.values())):
            raise ValueError('Task scales and priorities must be finite and positive')
        self.raw = dict.fromkeys(GROUPS, 0.)
        self.floors = dict.fromkeys(GROUPS, 0.)
        self.calls = 0
        self.gradient_meter = gradient_meter
        self.component_raw = {}

    def combine(self, parts, *, floors=None, component=None):
        floors = floors or {}
        if component is not None and (component not in ('spawn', 'recovery') or floors):
            raise ValueError('Known components have no constant KL floors')
        if not parts or set(parts)-set(GROUPS) or set(floors)-set(parts):
            raise ValueError('Every objective must belong to exactly one known task')
        total = None
        pending = []
        for name, value in parts.items():
            floor = float(floors.get(name, 0.))
            if (not torch.is_tensor(value) or value.numel() != 1 or not bool(torch.isfinite(value))
                    or not math.isfinite(floor) or floor < 0):
                raise FloatingPointError('Nonfinite grouped objective or KL floor')
            variable = value-floor
            raw = float(variable.detach())
            if raw < -2e-5:
                raise ValueError('Grouped objective is below its declared constant KL floor')
            # Clipping only detached round-off in bookkeeping cannot change gradients.
            pending.append((name, max(raw, 0.), floor))
            if component is not None and self.gradient_meter is not None:
                self.gradient_meter.capture(component, name, variable)
            weighted = variable*(self.priorities[name]/self.scales[name])+floor
            total = weighted if total is None else total+weighted
        if not bool(torch.isfinite(total)):
            raise FloatingPointError('Nonfinite weighted task objective')
        for name, raw, floor in pending:
            self.raw[name] += raw
            self.floors[name] += floor
            if component is not None:
                bucket = self.component_raw.setdefault(component, {})
                bucket[name] = bucket.get(name, 0.)+raw
        self.calls += 1
        return total

    def report(self):
        rows = {}
        for name in GROUPS:
            raw, scale, weight, floor = self.raw[name], self.scales[name], self.priorities[name], self.floors[name]
            rows[name] = {'raw': raw, 'scale': scale, 'weight': weight,
                'effective_weight': weight/scale, 'normalized': raw/scale,
                'weighted': raw*weight/scale+floor, 'kl_floor': floor, 'active': raw > 1e-8}
        return {**rows, 'total': sum(row['weighted'] for row in rows.values()),
            'kl_floor': sum(self.floors.values()), 'normalization': SETTINGS['normalization'],
            'loss_protocol': LOSS_PROTOCOL}


class GroupLossBalancer:
    """EMA task magnitudes, updated once after Adam from all microbatch means."""
    def __init__(self, state=None):
        self.state = deepcopy(state) if state is not None else {
            'settings': deepcopy(SETTINGS), 'committed_updates': 0, 'last_optimizer_step': None,
            'banks': {environment: {'updates': 0, 'ema': {name: None for name in GROUPS}}
                      for environment in ('combat', 'navigation')}}
        if self.state.get('settings') != SETTINGS:
            raise ValueError('Loss balancing settings changed on resume')
        if (type(self.state.get('committed_updates')) is not int or self.state['committed_updates'] < 0
                or set(self.state.get('banks', {})) != {'combat', 'navigation'}):
            raise ValueError('Invalid task-balancer checkpoint counters')
        for bank in self.state['banks'].values():
            if type(bank.get('updates')) is not int or bank['updates'] < 0 or set(bank.get('ema', {})) != set(GROUPS):
                raise ValueError('Invalid task-balancer environment bank')
            if any(value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0)
                   for value in bank['ema'].values()):
                raise ValueError('Invalid task-balancer EMA')
        last = self.state.get('last_optimizer_step')
        if (sum(bank['updates'] for bank in self.state['banks'].values()) != self.state['committed_updates']
                or last is not None and (type(last) is not int or last < 0)
                or self.state['committed_updates'] > 0 and last is None):
            raise ValueError('Task-balancer total/environment/optimizer counters disagree')

    def new_microbatch(self, environment):
        bank = self.state['banks'][environment]
        scales = {name: max(SETTINGS['minimum_scale'], value) if value is not None else 1.
                  for name, value in bank['ema'].items()}
        return GroupLossAccumulator(scales, SETTINGS['priorities'])

    def summarize(self, reports, *, environment, step):
        if not reports:
            raise ValueError('Cannot balance an incomplete optimizer update')
        expected = self.new_microbatch(environment).report()
        groups = {}
        for name in GROUPS:
            for report in reports:
                row = report[name]
                if any(row[key] != expected[name][key] for key in ('scale', 'weight')):
                    raise ValueError('Loss weights changed inside accumulated optimizer update')
                if not all(math.isfinite(row[key]) and row[key] >= 0
                           for key in ('raw', 'normalized', 'weighted', 'kl_floor')):
                    raise FloatingPointError('Nonfinite accumulated task losses')
            groups[name] = {key: sum(report[name][key] for report in reports)/len(reports)
                for key in ('raw', 'normalized', 'weighted', 'kl_floor')}
            groups[name].update({key: expected[name][key] for key in ('scale', 'weight', 'effective_weight')})
            groups[name]['active'] = groups[name]['raw'] > 1e-8
        return {**groups, 'total': sum(row['weighted'] for row in groups.values()),
            'kl_floor': sum(row['kl_floor'] for row in groups.values()), 'step': step,
            'loss_protocol': LOSS_PROTOCOL, 'normalization': SETTINGS['normalization'],
            'calibrating': self.state['banks'][environment]['updates'] == 0,
            'ema_updates_before': self.state['committed_updates'], 'environment': environment,
            'microbatches': len(reports), 'scale_update': 'after_optimizer_commit_only'}

    def commit(self, report, *, step):
        if (report.get('step') != step or report.get('loss_protocol') != LOSS_PROTOCOL
                or report.get('ema_updates_before') != self.state['committed_updates']
                or self.state['last_optimizer_step'] is not None and step != self.state['last_optimizer_step']+1):
            raise ValueError('Task EMA commits only once at the corresponding optimizer boundary')
        values = {name: report[name]['raw'] for name in GROUPS}
        if not all(math.isfinite(value) and value >= 0 for value in values.values()):
            raise FloatingPointError('Cannot commit nonfinite task EMA')
        bank = self.state['banks'][report['environment']]
        for name, value in values.items():
            if value <= 1e-8:
                continue  # Missing/fully solved task never drives its scale to zero.
            old = bank['ema'][name]
            bank['ema'][name] = value if old is None else SETTINGS['ema_decay']*old+(1-SETTINGS['ema_decay'])*value
        bank['updates'] += 1
        self.state['committed_updates'] += 1
        self.state['last_optimizer_step'] = step

    def state_dict(self):
        return deepcopy(self.state)


def bind_loss_balance_revision(state, plan, *, path, sha256, source_checkpoint_sha256,
                               base_plan_sha256, batch_revision_sha256, loaded_checkpoint_sha256=None):
    """Opt in at a preserved complete Adam boundary, then validate each resume."""
    expected = {'version': 1, 'loss_protocol': LOSS_PROTOCOL, 'settings': SETTINGS,
        'source_future_feedback': False, 'optimizer_reset': False,
        'base_plan_sha256': base_plan_sha256, 'batch_revision_sha256': batch_revision_sha256,
        'source_checkpoint_sha256': source_checkpoint_sha256}
    if (not isinstance(sha256, str) or not sha256 or not isinstance(source_checkpoint_sha256, str)
            or not source_checkpoint_sha256
            or any(type(plan.get(key)) is not type(value) or plan.get(key) != value for key, value in expected.items())):
        raise ValueError('Task balance revision settings or parent pins changed')
    step, start = state['global_training_step'], plan.get('start_optimizer_step')
    if type(start) is not int or start < 0 or start > step:
        raise ValueError('Task balancing requires a complete source optimizer boundary')
    training = state['rssm_training']
    previous = training.get('loss_balance_revision')
    if previous is None:
        if (start != step or training.get('loss_balance_state') is not None
                or loaded_checkpoint_sha256 != source_checkpoint_sha256):
            raise ValueError('Initial task balancing must preserve its exact source boundary')
        balancer = GroupLossBalancer()
        balancer.state['last_optimizer_step'] = step
        previous = {'plan': str(path), 'sha256': sha256, 'start_optimizer_step': step,
            'last_optimizer_step': step, 'updates': 0, 'loss_protocol': LOSS_PROTOCOL,
            'initial_loaded_checkpoint_sha256': loaded_checkpoint_sha256}
        training['loss_balance_revision'] = previous
        training['loss_balance_state'] = balancer.state_dict()
        state.setdefault('objective_history', []).append({'loss_protocol': state['loss_protocol'],
            'last_step': step, 'reason': 'user_requested_separate_player_enemy_interaction_normalization',
            'constant_KL_floor_excluded_from_normalization': True})
    else:
        if (previous.get('sha256') != sha256 or previous.get('start_optimizer_step') != start
                or previous.get('last_optimizer_step') != step or previous.get('updates') != step-start
                or previous.get('loss_protocol') != LOSS_PROTOCOL
                or previous.get('initial_loaded_checkpoint_sha256') != source_checkpoint_sha256):
            raise ValueError('Task-balancing revision identity/counters changed')
        if training.get('loss_balance_state') is None:
            raise ValueError('Task balancing cannot resume without its saved EMA')
        balancer = GroupLossBalancer(training['loss_balance_state'])
        if balancer.state['last_optimizer_step'] != step or balancer.state['committed_updates'] != step-start:
            raise ValueError('Task EMA does not match the complete Adam boundary')
    state['loss_protocol'] = LOSS_PROTOCOL
    state['training_protocol']['loss_balance'] = {'settings': deepcopy(SETTINGS),
        'groups': list(GROUPS), 'constant_kl_floor': 'unscaled; excluded from task EMA',
        'scale_update': 'once_after_committed_optimizer_update',
        'scales_fixed_across_all_TBPTT_chunks_branches_and_microbatches': True}
    return previous, balancer


def balanced_batch_boundary_valid(state):
    training = state['rssm_training']
    revision = training.get('loss_balance_revision', {})
    ema = training.get('loss_balance_state', {})
    try:
        GroupLossBalancer(ema)
    except (ValueError, TypeError, KeyError):
        return False
    step = state['global_training_step']
    protocol_valid = state.get('loss_protocol') == LOSS_PROTOCOL
    if state.get('loss_protocol') == 'rssm_dual_combat_components_v6':
        from .rssm_combat_v6 import v6_boundary_valid
        protocol_valid = v6_boundary_valid(state)
    return (protocol_valid and bool(revision.get('sha256'))
        and revision.get('loss_protocol') == LOSS_PROTOCOL
        and revision.get('last_optimizer_step') == step
        and revision.get('updates') == step-revision.get('start_optimizer_step', step+1)
        and ema.get('settings') == SETTINGS and ema.get('last_optimizer_step') == step
        and ema.get('committed_updates') == revision.get('updates'))
