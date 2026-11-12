"""An explicit64x4 continuation, retaining the old128-example history."""
from __future__ import annotations

from copy import deepcopy

from .rssm_component_balance import LOSS_PROTOCOL


BATCH_PROTOCOL = 'rssm_effective256_v7'


def scaled_branch_protocol(base):
    from .rssm_combat_branches import normalize_combat_branch_protocol
    original = normalize_combat_branch_protocol(base)
    if original is None or original['version'] != 2:
        raise ValueError('Batch256 retains the aligned late-event branch protocol')
    return normalize_combat_branch_protocol({**original, 'max_branches': 16,
        'max_source_probes': 128, 'max_direct_source_queries': 256})


def bind_batch256_revision(state, plan, probe, *, path, sha256, probe_sha256,
                           source_checkpoint_sha256, loaded_checkpoint_sha256,
                           cli_microbatch, cli_accumulation):
    training = state['rssm_training'];step = state['global_training_step']
    expected = {'version': 7, 'batch_protocol': BATCH_PROTOCOL, 'loss_protocol': LOSS_PROTOCOL,
        'microbatch_size': 64, 'gradient_accumulation_steps': 4, 'effective_batch_size': 256,
        'previous_effective_batch_size': 128, 'gradient_reduction': 'mean_over_microbatches',
        'source_checkpoint_sha256': source_checkpoint_sha256, 'batch64_probe_sha256': probe_sha256,
        'parent_v6_sha256': training['combat_v6_revision']['sha256'],
        'parent_batch_sha256': training['batch_revision']['sha256'],
        'optimizer_reset': False, 'source_future_feedback': False}
    if (not sha256 or not probe_sha256 or not source_checkpoint_sha256
            or type(cli_microbatch) is not int or cli_microbatch != 64
            or type(cli_accumulation) is not int or cli_accumulation != 4
            or any(type(plan.get(key)) is not type(value) or plan.get(key) != value
                   for key, value in expected.items())):
        raise ValueError('Batch256 requires its explicit64x4 plan and unchanged v6 parents')
    if state.get('loss_protocol') != LOSS_PROTOCOL or state.get('training_schedule') != 'combat_only_v1':
        raise ValueError('Batch256 continues the current combat-only v6 objectives')
    for key in ('combat_v6_revision', 'loss_balance_revision', 'combat_only_revision',
                'batch_revision', 'combat_recovery_revision', 'feedback_revision'):
        if training[key]['last_optimizer_step'] != step:
            raise ValueError('Batch256 parent is not at the complete Adam boundary: '+key)
    start = plan.get('start_optimizer_step')
    if (type(start) is not int or not 0 <= start <= step or probe.get('passed') is not True
            or probe.get('source_step') != start or probe.get('source_checkpoint_sha256') != source_checkpoint_sha256
            or probe.get('microbatch_size') != 64 or probe.get('optimizer_updated') is not False
            or probe.get('model_weights_and_buffers_unchanged') is not True
            or probe.get('all_gradients_finite') is not True):
        raise ValueError('Batch256 requires a successful unchanged-source64-example backward probe')
    branches = scaled_branch_protocol(state['training_protocol']['combat_branch_protocol'])
    if plan.get('combat_branch_protocol') != branches:
        raise ValueError('Batch256 must scale recovery coverage together with the batch')
    previous = training.get('batch256_revision')
    if previous is None:
        if start != step or loaded_checkpoint_sha256 != source_checkpoint_sha256:
            raise ValueError('First64x4 update must preserve the measured exact source checkpoint')
        previous = {'version': 7, 'batch_protocol': BATCH_PROTOCOL, 'plan': str(path), 'sha256': sha256,
            'start_optimizer_step': step, 'last_optimizer_step': step, 'updates': 0,
            'initial_loaded_checkpoint_sha256': loaded_checkpoint_sha256,
            'batch64_probe_sha256': probe_sha256, 'microbatch_size': 64,
            'gradient_accumulation_steps': 4, 'effective_batch_size': 256,
            'previous_effective_batch_size': 128,
            'auxiliary_draws_at_start': training['auxiliary_draws'],
            'combat_auxiliary_draws_at_start': training['combat_auxiliary_draws'],
            'navigation_auxiliary_draws_at_start': training['navigation_auxiliary_draws'],
            'navigation_replay_updates_at_start': training['navigation_replay_updates'],
            'sampler_draws_at_start': training['combat_recovery_revision']['sampler_state']['draws']}
        training['batch256_revision'] = previous
        state.setdefault('batch_history', []).append({'last_step': step, 'previous_microbatch_size': 32,
            'previous_gradient_accumulation_steps': 4, 'previous_effective_batch_size': 128,
            'reason': 'explicit_user_requested64x4', 'model_Adam_RNG_and_EMA_preserved': True})
        state.update(loss=None, latest_metrics=None, loss_groups=None, long_training_progress=None)
    else:
        count = step-start
        if (previous.get('sha256') != sha256 or previous.get('start_optimizer_step') != start
                or previous.get('last_optimizer_step') != step or previous.get('updates') != count
                or previous.get('initial_loaded_checkpoint_sha256') != source_checkpoint_sha256
                or previous.get('batch64_probe_sha256') != probe_sha256
                or training['auxiliary_draws']-previous['auxiliary_draws_at_start'] != count*256
                or training['combat_auxiliary_draws']-previous['combat_auxiliary_draws_at_start'] != count*256
                or training['navigation_auxiliary_draws'] != previous['navigation_auxiliary_draws_at_start']
                or training['navigation_replay_updates'] != previous['navigation_replay_updates_at_start']
                or training['combat_recovery_revision']['sampler_state']['draws']-previous['sampler_draws_at_start'] != count*256):
            raise ValueError('Saved batch256 identity, counters or sampler history changed')
    state['batch_protocol'] = BATCH_PROTOCOL
    state['training_protocol'].update(microbatch_size=64, gradient_accumulation_steps=4,
        effective_batch_size=256, auxiliary_fragments_per_update=256,
        auxiliary_combat_quotas={'batch':64,'spawn':16,'attack_negative':16,'attack_positive':16,'shooting_replay':16},
        combat_branch_protocol=branches, gradient_reduction='mean_over_microbatches')
    return previous, branches


def batch256_boundary_valid(state):
    training = state['rssm_training'];revision = training.get('batch256_revision', {})
    step = state['global_training_step'];protocol = state.get('training_protocol', {})
    if (revision.get('version') != 7 or not revision.get('sha256')
            or state.get('batch_protocol') != BATCH_PROTOCOL or state.get('loss_protocol') != LOSS_PROTOCOL
            or revision.get('last_optimizer_step') != step
            or revision.get('updates') != step-revision.get('start_optimizer_step', step+1)
            or revision.get('previous_effective_batch_size') != 128
            or (revision.get('microbatch_size'),revision.get('gradient_accumulation_steps'),revision.get('effective_batch_size')) != (64,4,256)
            or (protocol.get('microbatch_size'),protocol.get('gradient_accumulation_steps'),protocol.get('effective_batch_size')) != (64,4,256)):
        return False
    from .rssm_combat_v6 import v6_boundary_valid
    return v6_boundary_valid(state)


def commit_batch256_update(state, *, step, sampler_draws):
    q = state['rssm_training'];revision = q['batch256_revision'];count = revision['updates']+1
    if (step != state['global_training_step'] or step != revision['last_optimizer_step']+1
            or step-revision['start_optimizer_step'] != count
            or q['auxiliary_draws']-revision['auxiliary_draws_at_start'] != count*256
            or q['combat_auxiliary_draws']-revision['combat_auxiliary_draws_at_start'] != count*256
            or q['navigation_auxiliary_draws'] != revision['navigation_auxiliary_draws_at_start']
            or q['navigation_replay_updates'] != revision['navigation_replay_updates_at_start']
            or sampler_draws-revision['sampler_draws_at_start'] != count*256):
        raise ValueError('Batch256 commits after exactly four complete64-example microbatches')
    revision['updates'] = count;revision['last_optimizer_step'] = step
