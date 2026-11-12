"""Pinned continuation for aligned late-event learning and component scales."""
from __future__ import annotations

from copy import deepcopy

from .rssm_component_balance import (LOSS_PROTOCOL, SETTINGS, ComponentLossBalancer,
                                     initialize_component_balancer)


def bind_combat_v6(state, plan, calibration, *, path, sha256, calibration_sha256,
                   source_checkpoint_sha256, loaded_checkpoint_sha256, base_plan_sha256):
    from .rssm_combat_branches import normalize_combat_branch_protocol
    training = state['rssm_training'];step = state['global_training_step']
    for name in ('loss_balance_revision', 'batch_revision', 'combat_only_revision',
                 'combat_recovery_revision', 'feedback_revision'):
        revision = training.get(name, {})
        if revision.get('last_optimizer_step') != step:
            raise ValueError('Combat v6 parent is not at the complete optimizer boundary: '+name)
    expected = {'version': 6, 'loss_protocol': LOSS_PROTOCOL, 'component_settings': SETTINGS,
        'base_plan_sha256': base_plan_sha256,
        'parent_loss_balance_sha256': training['loss_balance_revision']['sha256'],
        'parent_batch_sha256': training['batch_revision']['sha256'],
        'parent_combat_only_sha256': training['combat_only_revision']['sha256'],
        'source_checkpoint_sha256': source_checkpoint_sha256,
        'gradient_calibration_sha256': calibration_sha256,
        'attack_positive_balanced_sampling': True, 'optimizer_reset': False,
        'source_future_feedback': False, 'main_horizon': 450, 'tbptt_chunk': 16,
        'validation_every': 250, 'preview_every': 10, 'control_validation_updates': [100, 250]}
    if (not sha256 or not calibration_sha256 or not source_checkpoint_sha256
            or any(type(plan.get(key)) is not type(value) or plan.get(key) != value for key, value in expected.items())):
        raise ValueError('Combat v6 settings or immutable parents changed')
    branches = normalize_combat_branch_protocol(plan.get('combat_branch_protocol'))
    if (branches is None or branches['version'] != 2 or branches['max_branches'] != 8
            or branches['horizon'] != 16 or branches['max_direct_source_queries'] != 128):
        raise ValueError('Combat v6 requires the reviewed aligned eight-branch protocol')
    start = plan.get('start_optimizer_step')
    if (type(start) is not int or start < 0 or start > step
            or calibration.get('passed') is not True or calibration.get('source_step') != start
            or calibration.get('source_checkpoint_sha256') != source_checkpoint_sha256
            or calibration.get('optimizer_updated') is not False
            or calibration.get('model_weights_and_buffers_unchanged', calibration.get('model_weights_unchanged')) is not True
            or calibration.get('settings', SETTINGS) != SETTINGS):
        raise ValueError('Combat v6 must use an unchanged-source gradient calibration')
    previous = training.get('combat_v6_revision')
    if previous is None:
        if (start != step or loaded_checkpoint_sha256 != source_checkpoint_sha256
                or training['loss_balance_state'].get('components') is not None):
            raise ValueError('First combat v6 update must preserve its exact archived Adam boundary')
        normalized = initialize_component_balancer(training['loss_balance_state'], calibration, step=step)
        previous = {'version': 6, 'sha256': sha256, 'plan': str(path),
            'start_optimizer_step': step, 'last_optimizer_step': step, 'updates': 0,
            'initial_loaded_checkpoint_sha256': loaded_checkpoint_sha256,
            'gradient_calibration_sha256': calibration_sha256, 'baseline_complete': False,
            'completed_control_validations': [], 'legacy_sampler_state': deepcopy(
                training['combat_recovery_revision']['sampler_state'])}
        training['combat_v6_revision'] = previous
        training['loss_balance_state'] = normalized
        state.setdefault('objective_history', []).append({'loss_protocol': state['loss_protocol'],
            'last_step': step, 'reason': 'user_requested_late_event_coverage_component_balance_and_positive_melee',
            'normalization_changed': True, 'optimizer_reset': False})
        state.update(loss=None, latest_metrics=None, loss_groups=None)
    else:
        if (previous.get('sha256') != sha256 or previous.get('start_optimizer_step') != start
                or previous.get('last_optimizer_step') != step or previous.get('updates') != step-start
                or previous.get('initial_loaded_checkpoint_sha256') != source_checkpoint_sha256
                or previous.get('gradient_calibration_sha256') != calibration_sha256):
            raise ValueError('Saved combat v6 identity/counters changed')
    balancer = ComponentLossBalancer(training['loss_balance_state'])
    components = balancer.state['components']
    if components['committed_updates'] != step-start:
        raise ValueError('Component EMA clock differs from the actual v6 optimizer boundary')
    for component, actors in calibration['components'].items():
        for name, value in actors.items():
            if components['priorities'][component][name] != value['chosen_priority']:
                raise ValueError('Component priority differs from the pinned measured calibration')
    state['loss_protocol'] = LOSS_PROTOCOL
    state['training_protocol'].update(combat_branch_protocol=branches,
        own_memory_combat_branches=True, combat_branch_active_environments=['combat'],
        auxiliary_combat_quotas={'batch': 32, 'spawn': 8, 'attack_negative': 8,
            'attack_positive': 8, 'shooting_replay': 8},
        attack_positive_balanced_sampling=True,
        component_normalization={'settings': SETTINGS, 'gradient_calibration_sha256': calibration_sha256,
            'base_actor_ema_preserved': True, 'spawn_and_recovery_bypass_base_actor_scale': True},
        direct_combat_head_supervised_steps=list(range(1,17)),
        direct_target_provenance='one-step Source from exact detached own state; no rollout correction')
    return previous, balancer, branches


def v6_boundary_valid(state):
    training = state['rssm_training'];revision = training.get('combat_v6_revision', {})
    step = state['global_training_step']
    try:
        balanced = ComponentLossBalancer(training['loss_balance_state'])
    except (KeyError, ValueError, TypeError):
        return False
    return (state.get('loss_protocol') == LOSS_PROTOCOL and bool(revision.get('sha256'))
        and revision.get('last_optimizer_step') == step
        and revision.get('updates') == step-revision.get('start_optimizer_step', step+1)
        and balanced.state['components']['committed_updates'] == revision.get('updates'))
