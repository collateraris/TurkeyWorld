"""An explicit, reversible curriculum override without resetting training clocks."""
from __future__ import annotations

from copy import deepcopy


COMBAT_ONLY_SCHEDULE = 'combat_only_v1'


def bind_combat_only_revision(state, plan, *, path, sha256, source_checkpoint_sha256,
                              loaded_checkpoint_sha256, base_plan_sha256, loss_balance_plan_sha256):
    expected = {'version': 1, 'training_schedule': COMBAT_ONLY_SCHEDULE,
        'update_cycle': ['combat'], 'navigation_replay_enabled': False,
        'navigation_validation_enabled': True, 'optimizer_reset': False,
        'source_future_feedback': False, 'base_plan_sha256': base_plan_sha256,
        'loss_balance_plan_sha256': loss_balance_plan_sha256,
        'source_checkpoint_sha256': source_checkpoint_sha256}
    if (not isinstance(sha256, str) or not sha256 or not source_checkpoint_sha256
            or any(type(plan.get(key)) is not type(value) or plan.get(key) != value
                   for key, value in expected.items())):
        raise ValueError('Combat-only curriculum or immutable parent pins changed')
    training = state['rssm_training'];step = state['global_training_step']
    batch = state['training_protocol'].get('effective_batch_size', 128)
    start = plan.get('start_optimizer_step')
    if type(start) is not int or not 0 <= start <= step:
        raise ValueError('Combat-only curriculum requires a complete optimizer boundary')
    balance = training.get('loss_balance_revision', {})
    if balance.get('sha256') != loss_balance_plan_sha256 or balance.get('last_optimizer_step') != step:
        raise ValueError('Combat-only curriculum must retain the current task-balancing boundary')
    previous = training.get('combat_only_revision')
    if previous is None:
        if start != step or loaded_checkpoint_sha256 != source_checkpoint_sha256:
            raise ValueError('Initial combat-only training must load its exact archived checkpoint')
        previous = {'version': 1, 'training_schedule': COMBAT_ONLY_SCHEDULE,
            'plan': str(path), 'sha256': sha256, 'start_optimizer_step': step,
            'last_optimizer_step': step, 'updates': 0,
            'initial_loaded_checkpoint_sha256': loaded_checkpoint_sha256,
            'combat_long_updates_at_start': training['combat_long_updates'],
            'navigation_replay_updates_at_start': training['navigation_replay_updates'],
            'combat_auxiliary_draws_at_start': training['combat_auxiliary_draws'],
            'navigation_auxiliary_draws_at_start': training['navigation_auxiliary_draws']}
        training['combat_only_revision'] = previous
        state.setdefault('curriculum_history', []).append({'last_step': step,
            'previous_update_cycle': deepcopy(state['training_protocol'].get('update_cycle', ['combat']*3+['navigation'])),
            'reason': 'user_requested_temporary_combat_only_training',
            'new_training_schedule': COMBAT_ONLY_SCHEDULE})
    else:
        if (previous.get('sha256') != sha256 or previous.get('start_optimizer_step') != start
                or previous.get('training_schedule') != COMBAT_ONLY_SCHEDULE
                or previous.get('initial_loaded_checkpoint_sha256') != source_checkpoint_sha256
                or previous.get('last_optimizer_step') != step or previous.get('updates') != step-start
                or training['combat_long_updates']-previous.get('combat_long_updates_at_start', -1) != step-start
                or training['navigation_replay_updates'] != previous.get('navigation_replay_updates_at_start')
                or training['combat_auxiliary_draws']-previous.get('combat_auxiliary_draws_at_start', -1) != (step-start)*batch
                or training['navigation_auxiliary_draws'] != previous.get('navigation_auxiliary_draws_at_start')):
            raise ValueError('Saved combat-only curriculum identity or counters changed')
    state['training_schedule'] = COMBAT_ONLY_SCHEDULE
    state['training_protocol'].update(update_cycle=['combat'], active_training_environments=['combat'],
        navigation_replay_enabled=False, navigation_validation_enabled=True,
        latent_and_context_schedules='unchanged_global_combat_completed_clock')
    return previous


def next_training_environment(state, base_cycle, completed):
    if type(completed) is not int or completed != state['rssm_training']['combat_completed']:
        raise ValueError('Training schedule must use the preserved completed-update clock')
    revision = state['rssm_training'].get('combat_only_revision')
    if revision is None:
        protocol = state.get('training_protocol', {})
        if (state.get('training_schedule') == COMBAT_ONLY_SCHEDULE
                or protocol.get('navigation_replay_enabled') is False
                or protocol.get('update_cycle') == ['combat']):
            raise ValueError('Saved combat-only schedule cannot silently lose its revision ledger')
        return base_cycle[completed % len(base_cycle)]
    step = state['global_training_step']
    if (revision.get('training_schedule') != COMBAT_ONLY_SCHEDULE
            or revision.get('last_optimizer_step') != step
            or revision.get('updates') != step-revision.get('start_optimizer_step', step+1)):
        raise ValueError('Combat-only schedule is not at a complete optimizer boundary')
    return 'combat'


def commit_combat_only_update(state, *, step, environment):
    revision = state['rssm_training']['combat_only_revision']
    training = state['rssm_training']
    expected_updates = revision['updates']+1
    batch = state['training_protocol'].get('effective_batch_size', 128)
    if (environment != 'combat' or step != state['global_training_step']
            or step != revision['last_optimizer_step']+1
            or step-revision['start_optimizer_step'] != expected_updates
            or training['combat_long_updates']-revision['combat_long_updates_at_start'] != expected_updates
            or training['combat_auxiliary_draws']-revision['combat_auxiliary_draws_at_start'] != expected_updates*batch
            or training['navigation_replay_updates'] != revision['navigation_replay_updates_at_start']
            or training['navigation_auxiliary_draws'] != revision['navigation_auxiliary_draws_at_start']):
        raise ValueError('Combat-only clocks commit only after a successful combat optimizer update')
    revision['updates'] = expected_updates
    revision['last_optimizer_step'] = step
