"""Pinned, optimizer-clock curriculum for short-to-450 coordinate learning.

This metadata layer never changes model tensors, Adam, RNG, Source clips, or
the held-out 450-step validation protocol. The 500th new optimizer update uses
450 transitions; subsequent updates continue at 450.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path


CURRICULUM_PROTOCOL = 'rssm_coordinate_horizon_v1'
COORDINATE_GRADIENT_PROTOCOL = {
    'version': 1, 'mode': 'detached_conditioning_integrated_xy',
    'exclude_predicted_respawns': True,
}
HORIZON_SCHEDULE = ((0, 16), (100, 32), (200, 64), (300, 128), (400, 256), (499, 450))
_PARENTS = ('combat_v6_revision', 'batch256_revision', 'combat_only_revision',
            'loss_balance_revision', 'batch_revision', 'combat_recovery_revision', 'feedback_revision')
_COUNTERS = ('combat_completed', 'source_completed', 'auxiliary_draws',
             'combat_auxiliary_draws', 'navigation_auxiliary_draws', 'navigation_replay_updates')


def coordinate_horizon(completed_updates):
    """Return the horizon of the NEXT update, from its committed local clock."""
    if type(completed_updates) is not int or completed_updates < 0:
        raise ValueError('Coordinate curriculum requires a nonnegative integer update clock')
    return next(horizon for start, horizon in reversed(HORIZON_SCHEDULE) if completed_updates >= start)


def _histogram(completed):
    result = {str(horizon): 0 for _, horizon in HORIZON_SCHEDULE}
    for index, (start, horizon) in enumerate(HORIZON_SCHEDULE):
        end = HORIZON_SCHEDULE[index+1][0] if index+1 < len(HORIZON_SCHEDULE) else completed
        result[str(horizon)] = max(0, min(completed, end)-start)
    return result


def coordinate_curriculum_plan(state, *, source_checkpoint, source_checkpoint_sha256,
                               kinematic_protocol, overshooting_protocol, base_plan_sha256=None):
    """Build a JSON-safe plan; the operator writes it once and pins its SHA."""
    from .rssm_kinematic_losses import normalize_kinematic_protocol
    from .rssm_memory_overshooting import normalize_memory_overshooting_protocol
    kinematic_protocol = normalize_kinematic_protocol(kinematic_protocol)
    overshooting_protocol = normalize_memory_overshooting_protocol(overshooting_protocol)
    if kinematic_protocol is None or overshooting_protocol is None:
        raise ValueError('Coordinate curriculum requires both kinematic and memory protocols from update one')
    q = state['rssm_training']
    return {
        'version': 1, 'curriculum_protocol': CURRICULUM_PROTOCOL,
        'source_checkpoint': str(source_checkpoint), 'source_checkpoint_sha256': source_checkpoint_sha256,
        'start_optimizer_step': state['global_training_step'], 'curriculum_updates': 500,
        'base_plan_sha256': base_plan_sha256 or state.get('combat_training_plan_sha256', state.get('plan_sha256')),
        'parent_v6_sha256': q['combat_v6_revision']['sha256'],
        'parent_batch256_sha256': q['batch256_revision']['sha256'],
        'horizon_schedule': [{'completed_updates': start, 'horizon': horizon} for start, horizon in HORIZON_SCHEDULE],
        'microbatch_size': 64, 'gradient_accumulation_steps': 4, 'effective_batch_size': 256,
        'tbptt_chunk': 16, 'auxiliary_horizon': 16, 'validation_horizon': 450,
        'validation_curriculum_updates': [0, 100, 200, 300, 400, 500],
        'main_window_start': 0, 'context_schedule_preserved': True,
        'combat_branch_protocol': deepcopy(state['training_protocol']['combat_branch_protocol']),
        'coordinate_gradient_protocol': deepcopy(COORDINATE_GRADIENT_PROTOCOL),
        'kinematic_protocol': deepcopy(kinematic_protocol),
        'overshooting_protocol': deepcopy(overshooting_protocol),
        'numeric_state_feedback': 'detached', 'optimizer_reset': False, 'source_future_feedback': False,
    }


def _boundary(state):
    q = state['rssm_training']; step = state['global_training_step']
    if type(step) is not int or step < 0:
        raise ValueError('Coordinate curriculum requires an exact saved optimizer clock')
    for key in _PARENTS:
        if q.get(key, {}).get('last_optimizer_step') != step:
            raise ValueError('Coordinate curriculum requires a complete parent boundary: '+key)
    protocol = state['training_protocol']
    if ((protocol.get('microbatch_size'), protocol.get('gradient_accumulation_steps'),
         protocol.get('effective_batch_size')) != (64, 4, 256)
            or q['feedback_revision'].get('numeric_state_feedback') != 'detached'):
        raise ValueError('Coordinate curriculum preserves the validated64x4 detached-feedback continuation')
    return q, step


def _validate_plan(state, plan, *, base_plan_sha256, source_checkpoint_sha256):
    q, _ = _boundary(state)
    if not isinstance(plan, dict):
        raise ValueError('Coordinate curriculum plan must be a mapping')
    expected = coordinate_curriculum_plan(state, source_checkpoint=plan.get('source_checkpoint', ''),
        source_checkpoint_sha256=source_checkpoint_sha256, kinematic_protocol=plan.get('kinematic_protocol'),
        overshooting_protocol=plan.get('overshooting_protocol'))
    expected['base_plan_sha256'] = base_plan_sha256
    expected['start_optimizer_step'] = plan.get('start_optimizer_step')
    if (not base_plan_sha256 or not source_checkpoint_sha256 or not plan.get('source_checkpoint')
            or type(plan.get('start_optimizer_step')) is not int or plan['start_optimizer_step'] < 0
            or not isinstance(plan.get('kinematic_protocol'), dict) or not plan['kinematic_protocol']
            or not isinstance(plan.get('overshooting_protocol'), dict) or not plan['overshooting_protocol']
            or json.dumps(plan, sort_keys=True) != json.dumps(expected, sort_keys=True)):
        raise ValueError('Coordinate curriculum settings, source or immutable parent pins changed')
    branches = plan['combat_branch_protocol']
    if ((branches.get('version'), branches.get('horizon'), branches.get('max_branches'),
         branches.get('max_direct_source_queries')) != (2, 16, 16, 256)
            or plan['parent_v6_sha256'] != q['combat_v6_revision']['sha256']
            or plan['parent_batch256_sha256'] != q['batch256_revision']['sha256']):
        raise ValueError('Coordinate curriculum must preserve the reviewed sixteen own-memory branches')


def _verify_counts(state, revision, count, *, sampler_draws=None):
    q = state['rssm_training']
    expected = {'combat_completed': count, 'source_completed': count,
        'auxiliary_draws': count*256, 'combat_auxiliary_draws': count*256,
        'navigation_auxiliary_draws': 0, 'navigation_replay_updates': 0}
    draws = q['combat_recovery_revision']['sampler_state']['draws'] if sampler_draws is None else sampler_draws
    if (any(q[key]-revision['counters_at_start'][key] != delta for key, delta in expected.items())
            or type(draws) is not int or draws-revision['sampler_draws_at_start'] != count*256):
        raise ValueError('Coordinate curriculum source/sampler counters differ from complete64x4 commits')


def bind_coordinate_curriculum(state, plan, *, path, sha256, source_checkpoint_sha256,
                               loaded_checkpoint_sha256, base_plan_sha256,
                               cli_microbatch=64, cli_accumulation=4):
    """Bind once at the pinned source, or verify the complete saved revision."""
    _validate_plan(state, plan, base_plan_sha256=base_plan_sha256,
                   source_checkpoint_sha256=source_checkpoint_sha256)
    if (cli_microbatch, cli_accumulation) != (64, 4):
        raise ValueError('Coordinate curriculum CLI batch must remain64x4')
    payload = Path(path).read_bytes()
    if (hashlib.sha256(payload).hexdigest() != sha256
            or json.loads(payload.decode('utf-8')) != plan):
        raise ValueError('Coordinate curriculum plan file identity changed')
    q, step = _boundary(state); start = plan['start_optimizer_step']
    previous = q.get('coordinate_curriculum')
    if previous is None:
        if step != start or loaded_checkpoint_sha256 != source_checkpoint_sha256:
            raise ValueError('First coordinate curriculum bind requires its exact archived model/Adam boundary')
        previous = {'version': 1, 'sha256': sha256, 'plan': str(path),
            'start_optimizer_step': start, 'last_optimizer_step': start, 'updates': 0,
            'initial_loaded_checkpoint_sha256': loaded_checkpoint_sha256,
            'parent_v6_sha256': plan['parent_v6_sha256'], 'parent_batch256_sha256': plan['parent_batch256_sha256'],
            'coordinate_gradient_protocol': deepcopy(plan['coordinate_gradient_protocol']),
            'kinematic_protocol': deepcopy(plan['kinematic_protocol']),
            'overshooting_protocol': deepcopy(plan['overshooting_protocol']),
            'horizon_update_counts': _histogram(0), 'main_transitions_per_example': 0,
            'main_transitions': 0, 'main_examples': 0, 'next_horizon': coordinate_horizon(0),
            'completed_validation_updates': [],
            'counters_at_start': {key: q[key] for key in _COUNTERS},
            'sampler_draws_at_start': q['combat_recovery_revision']['sampler_state']['draws']}
        q['coordinate_curriculum'] = previous
        state.setdefault('objective_history', []).append({'loss_protocol': state['loss_protocol'],
            'last_step': step, 'reason': 'user_requested_coordinate_horizon_curriculum_500_updates',
            'optimizer_reset': False, 'validation_horizon_unchanged': 450})
    else:
        count = step-start
        histogram = _histogram(max(count, 0))
        transitions = sum(int(key)*value for key, value in histogram.items())
        next_horizon = coordinate_horizon(count) if count >= 0 else None
        if q.get('loss_gated_curriculum') is not None:
            from .rssm_loss_gated_curriculum import loss_gated_coordinate_expectation
            histogram, transitions, next_horizon = loss_gated_coordinate_expectation(state, count)
        elif q.get('replay_curriculum') is not None:
            from .rssm_replay_curriculum import replay_coordinate_expectation
            histogram, transitions, next_horizon = replay_coordinate_expectation(state, count)
        if (count < 0 or previous.get('sha256') != sha256
                or previous.get('start_optimizer_step') != start or previous.get('last_optimizer_step') != step
                or previous.get('updates') != count
                or previous.get('initial_loaded_checkpoint_sha256') != source_checkpoint_sha256
                or previous.get('parent_v6_sha256') != plan['parent_v6_sha256']
                or previous.get('parent_batch256_sha256') != plan['parent_batch256_sha256']
                or previous.get('coordinate_gradient_protocol') != plan['coordinate_gradient_protocol']
                or previous.get('kinematic_protocol') != plan['kinematic_protocol']
                or previous.get('overshooting_protocol') != plan['overshooting_protocol']
                or previous.get('horizon_update_counts') != histogram
                or previous.get('main_transitions_per_example') != transitions
                or previous.get('main_transitions') != transitions*256
                or previous.get('main_examples') != count*256
                or previous.get('next_horizon') != next_horizon):
            raise ValueError('Saved coordinate curriculum identity, horizon or optimizer clock changed')
        _verify_counts(state, previous, count)
    state['training_protocol'].update(coordinate_horizon_curriculum={
        'protocol': CURRICULUM_PROTOCOL, 'plan_sha256': sha256, 'curriculum_updates': 500,
        'horizon_schedule': deepcopy(plan['horizon_schedule']), 'validation_horizon': 450,
        'auxiliary_horizon': 16, 'main_window_start': 0},
        coordinate_gradient_protocol=deepcopy(plan['coordinate_gradient_protocol']),
        kinematic_protocol=deepcopy(plan['kinematic_protocol']),
        overshooting_protocol=deepcopy(plan['overshooting_protocol']))
    return previous


def commit_coordinate_update(state, *, step, horizon, sampler_draws):
    """Commit once, after the model/Adam update and all old revision ledgers."""
    q, current = _boundary(state); revision = q['coordinate_curriculum']; count = revision['updates']
    expected_horizon = coordinate_horizon(count)
    if q.get('loss_gated_curriculum') is not None:
        from .rssm_loss_gated_curriculum import loss_gated_training_horizon
        expected_horizon = loss_gated_training_horizon(state)['horizon']
    elif q.get('replay_curriculum') is not None:
        from .rssm_replay_curriculum import replay_training_horizon
        expected_horizon = replay_training_horizon(state)['horizon']
    if (type(horizon) is not int or horizon != expected_horizon
            or step != current or step != revision['last_optimizer_step']+1
            or step-revision['start_optimizer_step'] != count+1):
        raise ValueError('Coordinate horizon commits only once after a complete matching optimizer update')
    _verify_counts(state, revision, count+1, sampler_draws=sampler_draws)
    revision['updates'] = count+1; revision['last_optimizer_step'] = step
    revision['horizon_update_counts'][str(horizon)] = revision['horizon_update_counts'].get(str(horizon), 0)+1
    revision['main_transitions_per_example'] += horizon
    revision['main_transitions'] += horizon*256; revision['main_examples'] += 256
    revision['last_horizon'] = horizon; revision['next_horizon'] = coordinate_horizon(count+1)
    if q.get('loss_gated_curriculum') is not None:
        revision['next_horizon'] = loss_gated_training_horizon(state)['horizon']
    elif q.get('replay_curriculum') is not None:
        revision['next_horizon'] = replay_training_horizon(state)['horizon']
    return revision
