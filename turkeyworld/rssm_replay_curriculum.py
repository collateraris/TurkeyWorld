"""Pinned long replay with separate horizon EMA values and one global commit.

The original coordinate clock keeps advancing. Every fifth new update uses
450 own transitions; the other updates retain its scheduled horizon. Actor
and component EMA values are isolated by horizon, while their historical
optimizer/commit counters continue through the existing revision chain.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .rssm_component_balance import ComponentLossBalancer
from .rssm_loss_balance import GROUPS


REPLAY_PROTOCOL = 'rssm_coordinate_long_replay_v1'
EMA_KEY = 'horizon_ema_banks'
SETTINGS = {'version': 1, 'long_every': 5, 'long_horizon': 450,
    'ema_partition': 'curriculum_short_vs450', 'long_seed_step': 5014,
    'microbatch_size': 64, 'gradient_accumulation_steps': 4,
    'effective_batch_size': 256, 'validation_horizon': 450,
    'source_future_feedback': False, 'optimizer_reset': False}


def _ema_values(state):
    return {'actor_ema': deepcopy(state['banks']['combat']['ema']),
            'component_ema': deepcopy(state['components']['banks']['combat']['ema'])}


def _set_ema_values(state, bank):
    state['banks']['combat']['ema'] = deepcopy(bank['actor_ema'])
    state['components']['banks']['combat']['ema'] = deepcopy(bank['component_ema'])


class ReplayComponentLossBalancer(ComponentLossBalancer):
    """Switch values before four micros; keep one ordinary global EMA commit."""
    def __init__(self, state):
        super().__init__(state)
        saved = self.state.get(EMA_KEY)
        if (not isinstance(saved, dict) or saved.get('settings') != SETTINGS
                or saved.get('active') not in ('short', 'long450')
                or set(saved.get('banks', {})) != {'short', 'long450'}
                or type(saved.get('start_optimizer_step')) is not int):
            raise ValueError('Missing or changed horizon-separated EMA state')
        count = self.state['last_optimizer_step']-saved['start_optimizer_step']
        if count < 0 or sum(bank.get('updates', -1) for bank in saved['banks'].values()) != count:
            raise ValueError('Horizon EMA clocks differ from committed global updates')
        for bank in saved['banks'].values():
            if (type(bank.get('updates')) is not int or bank['updates'] < 0
                    or set(bank.get('actor_ema', {})) != set(GROUPS)
                    or set(bank.get('component_ema', {})) != {'spawn', 'recovery'}):
                raise ValueError('Invalid separated horizon EMA bank')
            values = list(bank['actor_ema'].values())
            for component, names in self.state['components']['priorities'].items():
                if set(bank['component_ema'][component]) != set(names):
                    raise ValueError('Horizon component actor assignment changed')
                values.extend(bank['component_ema'][component].values())
            if any(value is not None and (type(value) not in (int, float)
                   or not math.isfinite(value) or value < 0) for value in values):
                raise ValueError('Nonfinite horizon EMA values')
        if _ema_values(self.state) != {key: saved['banks'][saved['active']][key]
                                      for key in ('actor_ema', 'component_ema')}:
            raise ValueError('Canonical EMA does not match the selected saved horizon bank')

    def select_update(self, *, horizon):
        from .rssm_loss_gated_curriculum import HORIZONS
        if type(horizon) is not int or horizon not in HORIZONS:
            raise ValueError('Unknown replay training horizon')
        saved = self.state[EMA_KEY]
        name = 'long450' if horizon == 450 else 'short'
        _set_ema_values(self.state, saved['banks'][name])
        saved['active'] = name
        return name

    def summarize(self, reports, *, environment, step):
        if environment != 'combat':
            raise ValueError('Explicit long replay applies only to the combat curriculum')
        result = super().summarize(reports, environment=environment, step=step)
        saved = self.state[EMA_KEY]
        result.update(horizon_ema_bank=saved['active'],
            horizon_ema_bank_updates_before=saved['banks'][saved['active']]['updates'])
        return result

    def commit(self, report, *, step):
        saved = self.state[EMA_KEY]
        bank = saved['banks'][saved['active']]
        if (report.get('environment') != 'combat' or report.get('horizon_ema_bank') != saved['active']
                or report.get('horizon_ema_bank_updates_before') != bank['updates']):
            raise ValueError('Horizon EMA commits require the selected complete four-micro report')
        super().commit(report, step=step)
        bank.update(_ema_values(self.state))
        bank['updates'] += 1
        bank['last_optimizer_step'] = step


def replay_curriculum_plan(state, *, source_checkpoint, source_checkpoint_sha256,
                           long_seed_checkpoint, long_seed_sha256,
                           base_plan_sha256, late_prefix_protocol):
    from .rssm_late_prefix_recovery import normalize_late_prefix_protocol
    late = normalize_late_prefix_protocol(late_prefix_protocol)
    if late is None:
        raise ValueError('Long replay requires its explicit late own-prefix coverage protocol')
    q = state['rssm_training']
    coordinate = q['coordinate_curriculum']
    return {'version': 1, 'replay_protocol': REPLAY_PROTOCOL, 'settings': deepcopy(SETTINGS),
        'source_checkpoint': str(source_checkpoint), 'source_checkpoint_sha256': source_checkpoint_sha256,
        'start_optimizer_step': state['global_training_step'],
        'base_plan_sha256': base_plan_sha256,
        'coordinate_plan_sha256': coordinate['sha256'],
        'coordinate_start_optimizer_step': coordinate['start_optimizer_step'],
        'long_seed_checkpoint': str(long_seed_checkpoint), 'long_seed_checkpoint_sha256': long_seed_sha256,
        'late_prefix_protocol': late, 'short_scout_microbatch_index': 0}


def replay_training_horizon(state, *, coordinate_updates=None):
    from .rssm_coordinate_curriculum import coordinate_horizon
    q = state['rssm_training']; coordinate = q['coordinate_curriculum']
    if q.get('loss_gated_curriculum') is not None:
        from .rssm_loss_gated_curriculum import loss_gated_training_horizon
        return loss_gated_training_horizon(state)
    completed = coordinate['updates'] if coordinate_updates is None else coordinate_updates
    base = coordinate_horizon(completed)
    replay = q.get('replay_curriculum')
    if replay is None:
        return {'horizon': base, 'curriculum_horizon': base, 'forced_long_replay': False,
                'ema_bank': 'long450' if base == 450 else 'short'}
    relative = completed-replay['coordinate_updates_at_start']
    if type(relative) is not int or relative < 0:
        raise ValueError('Replay cannot rewind the preserved coordinate curriculum clock')
    forced = (relative+1) % SETTINGS['long_every'] == 0
    actual = 450 if forced else base
    return {'horizon': actual, 'curriculum_horizon': base, 'forced_long_replay': forced,
            'ema_bank': 'long450' if actual == 450 else 'short'}


def replay_coordinate_expectation(state, completed_coordinate_updates):
    """Actual horizon histogram, including the immutable pre-replay history."""
    from .rssm_coordinate_curriculum import coordinate_horizon
    if state['rssm_training'].get('loss_gated_curriculum') is not None:
        from .rssm_loss_gated_curriculum import loss_gated_coordinate_expectation
        return loss_gated_coordinate_expectation(state, completed_coordinate_updates)
    replay = state['rssm_training']['replay_curriculum']
    before = replay['coordinate_updates_at_start']
    count = completed_coordinate_updates-before
    if type(count) is not int or count < 0:
        raise ValueError('Replay coordinate history cannot precede its saved source')
    histogram = deepcopy(replay['coordinate_horizon_counts_at_start'])
    for index in range(count):
        horizon = 450 if (index+1) % SETTINGS['long_every'] == 0 else coordinate_horizon(before+index)
        histogram[str(horizon)] += 1
    transitions = sum(int(key)*value for key, value in histogram.items())
    next_horizon = replay_training_horizon(state, coordinate_updates=completed_coordinate_updates)['horizon']
    return histogram, transitions, next_horizon


def _check_source_boundary(state):
    from .rssm_coordinate_curriculum import _boundary
    q, step = _boundary(state)
    coordinate = q.get('coordinate_curriculum', {})
    if (state.get('training_schedule') != 'combat_only_v1'
            or coordinate.get('last_optimizer_step') != step
            or coordinate.get('updates') != step-coordinate.get('start_optimizer_step', step+1)):
        raise ValueError('Replay requires the preserved complete coordinate optimizer boundary')
    ComponentLossBalancer(q['loss_balance_state'])
    return q, step


def bind_replay_curriculum(state, plan, *, path, sha256, source_checkpoint_sha256,
                           loaded_checkpoint_sha256, long_seed_checkpoint_sha256,
                           long_seed_loss_balance_state, base_plan_sha256,
                           cli_microbatch=64, cli_accumulation=4):
    q, step = _check_source_boundary(state)
    if (cli_microbatch, cli_accumulation) != (64, 4):
        raise ValueError('Long replay preserves64x4 optimizer semantics')
    expected = replay_curriculum_plan(state, source_checkpoint=plan.get('source_checkpoint', ''),
        source_checkpoint_sha256=source_checkpoint_sha256, long_seed_checkpoint=plan.get('long_seed_checkpoint', ''),
        long_seed_sha256=long_seed_checkpoint_sha256, base_plan_sha256=base_plan_sha256,
        late_prefix_protocol=plan.get('late_prefix_protocol'))
    expected['start_optimizer_step'] = plan.get('start_optimizer_step')
    raw = Path(path).read_bytes()
    if (not sha256 or not source_checkpoint_sha256 or not long_seed_checkpoint_sha256 or not base_plan_sha256
            or not plan.get('source_checkpoint') or not plan.get('long_seed_checkpoint')
            or type(plan.get('start_optimizer_step')) is not int
            or not SETTINGS['long_seed_step'] <= plan['start_optimizer_step'] <= step
            or json.dumps(plan, sort_keys=True) != json.dumps(expected, sort_keys=True)
            or hashlib.sha256(raw).hexdigest() != sha256 or json.loads(raw.decode('utf-8')) != plan):
        raise ValueError('Replay settings, source, late-prefix or plan identity changed')
    seed = ComponentLossBalancer(long_seed_loss_balance_state).state_dict()
    if (seed['last_optimizer_step'] != SETTINGS['long_seed_step']
            or seed['components']['priorities'] != q['loss_balance_state']['components']['priorities']
            or seed['settings'] != q['loss_balance_state']['settings']
            or seed['components']['settings'] != q['loss_balance_state']['components']['settings']):
        raise ValueError('Long EMA seed must retain the exact5014 profile and measured priorities')
    revision = q.get('replay_curriculum')
    if revision is None:
        if (plan['start_optimizer_step'] != step or loaded_checkpoint_sha256 != source_checkpoint_sha256
                or EMA_KEY in q['loss_balance_state']):
            raise ValueError('Initial replay binding requires the exact archived complete Adam boundary')
        short = _ema_values(q['loss_balance_state']); long = _ema_values(seed)
        extended = deepcopy(q['loss_balance_state'])
        extended[EMA_KEY] = {'settings': deepcopy(SETTINGS), 'start_optimizer_step': step, 'active': 'short',
            'long_seed_checkpoint_sha256': long_seed_checkpoint_sha256,
            'banks': {'short': {**short, 'updates': 0, 'last_optimizer_step': None},
                      'long450': {**long, 'updates': 0, 'last_optimizer_step': None}}}
        revision = {'version': 1, 'replay_protocol': REPLAY_PROTOCOL, 'sha256': sha256, 'plan': str(path),
            'start_optimizer_step': step, 'last_optimizer_step': step, 'updates': 0,
            'initial_loaded_checkpoint_sha256': loaded_checkpoint_sha256,
            'long_seed_checkpoint_sha256': long_seed_checkpoint_sha256,
            'coordinate_plan_sha256': plan['coordinate_plan_sha256'],
            'coordinate_updates_at_start': q['coordinate_curriculum']['updates'],
            'coordinate_horizon_counts_at_start': deepcopy(q['coordinate_curriculum']['horizon_update_counts']),
            'forced_long_replay_updates': 0, 'long450_updates': 0, 'short_updates': 0,
            'settings': deepcopy(SETTINGS), 'late_prefix_protocol': deepcopy(plan['late_prefix_protocol'])}
        # Validate before mutating the caller's state.
        balancer = ReplayComponentLossBalancer(extended)
        q['loss_balance_state'] = extended
        q['replay_curriculum'] = revision
        state.setdefault('objective_history', []).append({'loss_protocol': state['loss_protocol'], 'last_step': step,
            'reason': 'user_requested_late_state_replay_with_separate_horizon_ema', 'optimizer_reset': False,
            'old_coordinate_clock_preserved': True, 'long_every': 5, 'validation_horizon': 450})
    else:
        count = step-plan['start_optimizer_step']
        expected_forced = (q['loss_gated_curriculum']['replay_forced_long_updates_at_start']
            if q.get('loss_gated_curriculum') is not None else count//5)
        if (revision.get('sha256') != sha256 or revision.get('start_optimizer_step') != plan['start_optimizer_step']
                or revision.get('last_optimizer_step') != step or revision.get('updates') != count or count < 0
                or revision.get('initial_loaded_checkpoint_sha256') != source_checkpoint_sha256
                or revision.get('long_seed_checkpoint_sha256') != long_seed_checkpoint_sha256
                or revision.get('coordinate_plan_sha256') != plan['coordinate_plan_sha256']
                or revision.get('settings') != SETTINGS or revision.get('late_prefix_protocol') != plan['late_prefix_protocol']
                or revision.get('forced_long_replay_updates') != expected_forced
                or revision.get('long450_updates', -1)+revision.get('short_updates', -1) != count
                or q['coordinate_curriculum']['updates']-revision.get('coordinate_updates_at_start', -1) != count):
            raise ValueError('Saved replay identity, coordinate clock or update counts changed')
        balancer = ReplayComponentLossBalancer(q['loss_balance_state'])
        banks = balancer.state[EMA_KEY]
        if (banks['start_optimizer_step'] != revision['start_optimizer_step']
                or banks['long_seed_checkpoint_sha256'] != long_seed_checkpoint_sha256
                or banks['banks']['short']['updates'] != revision['short_updates']
                or banks['banks']['long450']['updates'] != revision['long450_updates']):
            raise ValueError('Saved horizon EMA history differs from the actual replay commits')
    state['training_protocol']['long_horizon_replay'] = {'protocol': REPLAY_PROTOCOL, 'settings': deepcopy(SETTINGS),
        'plan_sha256': sha256, 'late_prefix_protocol': deepcopy(plan['late_prefix_protocol']),
        'ema_values_separate_by_horizon': True, 'global_ema_commit_once_per_adam': True,
        'original_coordinate_clock_preserved': True}
    return revision, balancer


def commit_replay_update(state, *, step, horizon, forced_long_replay):
    q, current = _check_source_boundary(state)
    revision = q['replay_curriculum']; count = revision['updates']
    coordinate = q['coordinate_curriculum']
    expected = replay_training_horizon(state, coordinate_updates=coordinate['updates']-1)
    if (step != current or step != revision['last_optimizer_step']+1
            or coordinate['updates']-revision['coordinate_updates_at_start'] != count+1
            or horizon != expected['horizon'] or forced_long_replay is not expected['forced_long_replay']):
        raise ValueError('Replay commits once after the matching complete coordinate/Adam update')
    balancer = ReplayComponentLossBalancer(q['loss_balance_state'])
    banks = balancer.state[EMA_KEY]
    short = revision['short_updates']+int(horizon != 450)
    long = revision['long450_updates']+int(horizon == 450)
    if (banks['banks']['short']['updates'] != short or banks['banks']['long450']['updates'] != long
            or banks['active'] != expected['ema_bank']):
        raise ValueError('Replay horizon EMA must commit exactly once before its ledger')
    revision.update(updates=count+1, last_optimizer_step=step, short_updates=short, long450_updates=long,
        forced_long_replay_updates=revision['forced_long_replay_updates']+int(forced_long_replay),
        last_horizon=horizon, last_ema_bank=expected['ema_bank'])
    return revision
