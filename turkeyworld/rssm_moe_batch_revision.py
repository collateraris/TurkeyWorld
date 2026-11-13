"""Explicit batch continuations with untouched historical segment accounting."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re


PROTOCOL = 'rssm_moe_micro16_accum4_v1'
MICRO_LAYOUTS = (
    {'combat': 9, 'enemy_motion': 4, 'navigation': 3},
    {'combat': 10, 'enemy_motion': 4, 'navigation': 2},
    {'combat': 9, 'enemy_motion': 4, 'navigation': 3},
    {'combat': 10, 'enemy_motion': 4, 'navigation': 2},
)
SETTINGS = {
    'version': 1, 'microbatch_size': 16, 'gradient_accumulation_steps': 4,
    'effective_batch_size': 64, 'micro_layouts': list(MICRO_LAYOUTS),
    'combat_main_actors_per_update': 38, 'enemy_motion_main_actors_per_update': 16,
    'navigation_main_actors_per_update': 10, 'combat_rng_draws_per_microbatch': 16,
    'discarded_combat_rng_draws_per_update': 26,
    'combat_auxiliary_actors_per_microbatch': 16, 'combat_late_actors_per_microbatch': 2,
    'auxiliary_batch_size': 16, 'late_own_prefix_actors': 2, 'main_episode_rng_draws': 16,
    'auxiliary_sampling_policy': {
        'kind': 'source_attack_balanced_micro16_v1', 'version': 1,
        'spawn': 4, 'positive_attack': {'near_wall': 2, 'open': 2},
        'negative_attack': {'occluded': 3, 'other': 1},
        'shooting_replay': {'near': 1, 'mid': 1, 'far': 1,
                            'alternating_miss_or_legacy': 1},
        'four_micro_shooting_counts': {'near': 4, 'mid': 4, 'far': 4, 'miss': 2, 'legacy': 2},
        'spawn_anchors': 'least_used_existing_counters',
        'miss_wall_other_rotation': 'actual_miss_draws',
        'preserve_old_v2_clock_and_counters': True,
    },
    'navigation_fraction': 10 / 64, 'enemy_motion_fraction': 16 / 64,
    'enemy_normalization_denominator_per_update': 54,
    'enemy_micro_weight': 'actor_count_times_accumulation_divided_by_54',
    'player_router_micro_weight': 'actor_count_divided_by_16',
    'historical_effective_batch_size': 256,
    'preserve_gate_streak_history': True, 'preserve_all_rng': True,
    'optimizer_reset': False, 'model_reset': False,
}
MICRO32_PROTOCOL = 'rssm_moe_micro32_accum1_v2'
MICRO32_SETTINGS = {
    'version': 2, 'microbatch_size': 32, 'gradient_accumulation_steps': 1,
    'effective_batch_size': 32,
    'micro_layouts': [{'combat': 19, 'enemy_motion': 8, 'navigation': 5}],
    'combat_main_actors_per_update': 19, 'enemy_motion_main_actors_per_update': 8,
    'navigation_main_actors_per_update': 5, 'combat_rng_draws_per_microbatch': 32,
    'discarded_combat_rng_draws_per_update': 13,
    'combat_auxiliary_actors_per_microbatch': 32, 'combat_late_actors_per_microbatch': 4,
    'auxiliary_batch_size': 32, 'late_own_prefix_actors': 4, 'main_episode_rng_draws': 32,
    'auxiliary_sampling_policy': {
        'kind': 'source_attack_balanced_micro32_continuation_v1', 'version': 1,
        'spawn': 8, 'positive_attack': {'near_wall': 4, 'open': 4},
        'negative_attack': {'occluded': 6, 'other': 2},
        'shooting_replay': {'near': 2, 'mid': 2, 'far': 2, 'miss': 1, 'legacy': 1},
        'spawn_anchors': 'least_used_existing_counters',
        'miss_wall_other_rotation': 'actual_miss_draws',
        'preserve_old_v2_clock_and_counters': True,
    },
    'navigation_fraction': 5 / 32, 'enemy_motion_fraction': 8 / 32,
    'enemy_normalization_denominator_per_update': 27,
    'enemy_micro_weight': 'actor_count_times_accumulation_divided_by_27',
    'player_router_micro_weight': 'actor_count_divided_by_32',
    'historical_effective_batch_sizes': [256, 64],
    'preserve_gate_streak_history': True, 'preserve_all_rng': True,
    'optimizer_reset': False, 'model_reset': False,
}
LEGACY_SETTINGS = {
    'microbatch_size': 64, 'gradient_accumulation_steps': 4,
    'effective_batch_size': 256,
    'micro_layouts': [{'combat': 38, 'enemy_motion': 16, 'navigation': 10}] * 4,
    'combat_main_actors_per_update': 152, 'enemy_motion_main_actors_per_update': 64,
    'navigation_main_actors_per_update': 40, 'combat_rng_draws_per_microbatch': 64,
    'discarded_combat_rng_draws_per_update': 104,
    'combat_auxiliary_actors_per_microbatch': 64, 'combat_late_actors_per_microbatch': 8,
    'auxiliary_batch_size': 64, 'late_own_prefix_actors': 8, 'main_episode_rng_draws': 64,
    'enemy_normalization_denominator_per_update': 216,
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def batch_revision_plan(state, *, source_checkpoint, source_checkpoint_sha256=None):
    from .rssm_moe_protocol import _validate_gate
    step = state.get('global_training_step')
    gate = state['rssm_training']['moe_training']
    _validate_gate(gate, step)
    if gate.get('batch_revision') is not None:
        raise ValueError('Micro16 revision can only replace the historical micro64 phase once')
    source = Path(source_checkpoint).resolve()
    actual_sha = _file_sha(source)
    if source_checkpoint_sha256 is not None and source_checkpoint_sha256 != actual_sha:
        raise ValueError('Micro16 source checkpoint SHA changed')
    navigation = state['rssm_training'].get('navigation_replay_revision')
    return {
        'version': 1, 'kind': PROTOCOL, 'settings': deepcopy(SETTINGS),
        'source_checkpoint': str(source), 'source_checkpoint_sha256': actual_sha,
        'source_optimizer_step': step, 'source_moe_updates': gate['updates'],
        'source_gate_sha256': _digest(gate), 'moe_training_plan_sha256': gate['sha256'],
        'source_navigation_replay_revision_sha256': _digest(navigation),
        'weights_reset': False, 'optimizer_reset': False, 'rng_reset': False,
        'gate_reset': False, 'historical_counts_rewritten': False,
    }


def micro32_batch_revision_plan(state, *, source_checkpoint, source_checkpoint_sha256=None):
    """Pin one direct micro16 -> micro32 continuation at a complete boundary."""
    from .rssm_moe_protocol import _validate_gate
    step = state.get('global_training_step')
    gate = state['rssm_training']['moe_training']
    _validate_gate(gate, step)
    previous = gate.get('batch_revision')
    if not isinstance(previous, dict) or previous.get('kind') != PROTOCOL:
        raise ValueError('Micro32 revision requires its exact direct micro16 parent')
    source = Path(source_checkpoint).resolve()
    actual_sha = _file_sha(source)
    if source_checkpoint_sha256 is not None and source_checkpoint_sha256 != actual_sha:
        raise ValueError('Micro32 source checkpoint SHA changed')
    return {
        'version': 2, 'kind': MICRO32_PROTOCOL, 'settings': deepcopy(MICRO32_SETTINGS),
        'source_checkpoint': str(source), 'source_checkpoint_sha256': actual_sha,
        'source_optimizer_step': step, 'source_moe_updates': gate['updates'],
        'source_gate_sha256': _digest(gate), 'moe_training_plan_sha256': gate['sha256'],
        'source_navigation_replay_revision_sha256':
            _digest(state['rssm_training'].get('navigation_replay_revision')),
        'previous_batch_revision_sha256': previous['sha256'],
        'weights_reset': False, 'optimizer_reset': False, 'rng_reset': False,
        'gate_reset': False, 'historical_counts_rewritten': False,
    }


def batch_settings_for_plan(plan):
    """Validate canonical dimensions before a trainer/checkpoint is constructed."""
    if (plan.get('kind'), type(plan.get('version')), plan.get('version')) == (PROTOCOL, int, 1):
        settings = SETTINGS
    elif (plan.get('kind'), type(plan.get('version')), plan.get('version')) == (MICRO32_PROTOCOL, int, 2):
        settings = MICRO32_SETTINGS
    else:
        raise ValueError('Unknown MoE microbatch revision protocol/version')
    if _digest(plan.get('settings')) != _digest(settings):
        raise ValueError('MoE microbatch revision settings changed')
    return deepcopy(settings)


def batch_settings_for_gate(gate):
    record = gate.get('batch_revision')
    if record is None:
        if gate.get('batch_protocol') is not None:
            raise ValueError('Micro16 batch metadata has no pinned revision')
        return deepcopy(LEGACY_SETTINGS)
    settings = batch_settings_for_plan(record)
    if gate.get('batch_protocol') != record['kind']:
        raise ValueError('MoE microbatch revision settings changed')
    return settings


def validate_moe_batch_accounting(gate, step):
    """Return expected examples/transitions, including the immutable old segment."""
    settings = batch_settings_for_gate(gate)
    record = gate.get('batch_revision')
    if record is None:
        return gate['updates'] * 256, sum(int(h) * n for h, n in
                                         gate['horizon_update_counts'].items()) * 256
    from .rssm_moe_protocol import _validate_gate, MOE_HORIZONS
    source = record.get('source_gate')
    source_step = record.get('source_optimizer_step')
    source_updates = record.get('source_moe_updates')
    updates = record.get('updates')
    counts = record.get('horizon_update_counts')
    parent = None if not isinstance(source, dict) else source.get('batch_revision')
    valid_parent = (parent is None if record['kind'] == PROTOCOL else
                    isinstance(parent, dict) and parent.get('kind') == PROTOCOL
                    and record.get('previous_batch_revision_sha256') == parent.get('sha256'))
    if (not isinstance(source, dict) or not valid_parent
            or not _sha(record.get('sha256')) or not _sha(record.get('source_checkpoint_sha256'))
            or type(source_step) is not int or type(source_updates) is not int
            or type(updates) is not int or updates < 0
            or source_step + updates != step or record.get('last_optimizer_step') != step
            or source_updates + updates != gate['updates']
            or source.get('last_optimizer_step') != source_step or source.get('updates') != source_updates
            or record.get('source_gate_sha256') != _digest(source)
            or record.get('source_navigation_replay_revision_sha256') !=
                _digest(record.get('source_navigation_replay_revision'))
            or source.get('sha256') != gate.get('sha256')
            or source.get('start_optimizer_step') != gate.get('start_optimizer_step')
            or not isinstance(counts, dict) or set(counts) != {str(h) for h in MOE_HORIZONS}
            or any(type(value) is not int or value < 0 for value in counts.values())
            or sum(counts.values()) != updates
            or any(gate['horizon_update_counts'][h] != source['horizon_update_counts'][h] + n
                   for h, n in counts.items())
            or gate['advance_history'][:len(source['advance_history'])] != source['advance_history']):
        raise ValueError('MoE batch source or piecewise update/horizon ledger changed')
    _validate_gate(source, source_step)
    examples = updates * settings['effective_batch_size']
    transitions = sum(int(h) * n for h, n in counts.items()) * settings['effective_batch_size']
    if record.get('main_examples') != examples or record.get('main_transitions') != transitions:
        raise ValueError('MoE micro16 actual example/transition ledger changed')
    return source['main_examples'] + examples, source['main_transitions'] + transitions


def bind_moe_batch_revision(state, plan, *, path, sha256, loaded_checkpoint_sha256):
    from .rssm_moe_protocol import _validate_gate, MOE_HORIZONS
    raw = Path(path).read_bytes()
    expected_keys = {'version', 'kind', 'settings', 'source_checkpoint',
        'source_checkpoint_sha256', 'source_optimizer_step', 'source_moe_updates',
        'source_gate_sha256', 'moe_training_plan_sha256',
        'source_navigation_replay_revision_sha256', 'weights_reset', 'optimizer_reset',
        'rng_reset', 'gate_reset', 'historical_counts_rewritten'}
    settings = batch_settings_for_plan(plan)
    protocol = plan['kind']
    if protocol == MICRO32_PROTOCOL:
        expected_keys.add('previous_batch_revision_sha256')
    if (set(plan) != expected_keys or not _sha(sha256)
            or type(plan.get('source_optimizer_step')) is not int
            or type(plan.get('source_moe_updates')) is not int
            or hashlib.sha256(raw).hexdigest() != sha256 or json.loads(raw.decode('utf-8')) != plan
            or any(plan[key] is not False for key in ('weights_reset', 'optimizer_reset',
                'rng_reset', 'gate_reset', 'historical_counts_rewritten'))
            or not _sha(plan.get('source_checkpoint_sha256'))
            or _file_sha(plan['source_checkpoint']) != plan['source_checkpoint_sha256']):
        raise ValueError('MoE immutable micro16 plan or full source changed')
    q = state['rssm_training']
    gate, step = q['moe_training'], state['global_training_step']
    _validate_gate(gate, step)
    record = gate.get('batch_revision')
    activating = record is None or protocol == MICRO32_PROTOCOL and record.get('kind') == PROTOCOL
    if activating:
        valid_parent = (record is None if protocol == PROTOCOL else
                        isinstance(record, dict) and record.get('kind') == PROTOCOL
                        and plan.get('previous_batch_revision_sha256') == record.get('sha256'))
        if (not valid_parent or step != plan['source_optimizer_step'] or gate['updates'] != plan['source_moe_updates']
                or loaded_checkpoint_sha256 != plan['source_checkpoint_sha256']
                or gate['sha256'] != plan['moe_training_plan_sha256']
                or _digest(gate) != plan['source_gate_sha256']
                or _digest(q.get('navigation_replay_revision')) !=
                    plan['source_navigation_replay_revision_sha256']):
            raise ValueError('MoE micro16 activation needs the exact full source gate and private samplers')
        source = deepcopy(gate)
        record = {
            'version': plan['version'], 'kind': protocol, 'plan': str(Path(path).resolve()), 'sha256': sha256,
            'settings': deepcopy(settings), 'source_checkpoint_sha256': plan['source_checkpoint_sha256'],
            'source_optimizer_step': step, 'source_moe_updates': gate['updates'],
            'source_gate_sha256': plan['source_gate_sha256'], 'source_gate': source,
            'source_navigation_replay_revision_sha256': plan['source_navigation_replay_revision_sha256'],
            'source_navigation_replay_revision': deepcopy(q.get('navigation_replay_revision')),
            'last_optimizer_step': step, 'updates': 0, 'main_examples': 0, 'main_transitions': 0,
            'horizon_update_counts': {str(h): 0 for h in MOE_HORIZONS},
        }
        if protocol == MICRO32_PROTOCOL:
            record['previous_batch_revision_sha256'] = plan['previous_batch_revision_sha256']
        gate.update(batch_revision=record, batch_protocol=protocol)
    elif (record.get('kind') != protocol or record.get('sha256') != sha256
            or record.get('source_checkpoint_sha256') != plan['source_checkpoint_sha256']
            or record.get('source_optimizer_step') != plan['source_optimizer_step']
            or record.get('source_moe_updates') != plan['source_moe_updates']
            or record.get('source_gate_sha256') != plan['source_gate_sha256']
            or record.get('source_navigation_replay_revision_sha256') !=
                plan['source_navigation_replay_revision_sha256']
            or gate['sha256'] != plan['moe_training_plan_sha256']
            or protocol == MICRO32_PROTOCOL and record.get('previous_batch_revision_sha256') !=
                plan['previous_batch_revision_sha256']):
        raise ValueError('Saved MoE micro16 revision source or plan identity changed')
    _validate_gate(gate, step)
    state.setdefault('training_protocol', {})['moe_batch_revision'] = {
        'protocol': protocol, 'plan_sha256': sha256, **deepcopy(settings)}
    return record


def moe_batch_settings(state):
    gate = state['rssm_training']['moe_training']
    validate_moe_batch_accounting(gate, state['global_training_step'])
    return batch_settings_for_gate(gate)


def microbatch_layout(state, index):
    settings = moe_batch_settings(state)
    accumulation = settings['gradient_accumulation_steps']
    if type(index) is not int or not 0 <= index < accumulation:
        raise ValueError('MoE microbatch index must fit the pinned accumulation count')
    layout = deepcopy(settings['micro_layouts'][index])
    total, denominator = settings['microbatch_size'], settings['enemy_normalization_denominator_per_update']
    c, e, n = (layout[key] for key in ('combat', 'enemy_motion', 'navigation'))
    layout.update(microbatch_size=total, effective_batch_size=settings['effective_batch_size'],
        combat_rng_draws=settings['combat_rng_draws_per_microbatch'],
        discarded_combat_rng_draws=settings['combat_rng_draws_per_microbatch'] - c,
        combat_auxiliary=settings['combat_auxiliary_actors_per_microbatch'],
        combat_late=settings['combat_late_actors_per_microbatch'],
        main_player_weight=c / total, main_router_weight=c / total,
        navigation_player_router_weight=n / total, enemy_motion_router_weight=e / total,
        main_enemy_weight=c * accumulation / denominator,
        enemy_motion_enemy_weight=e * accumulation / denominator)
    layout.update(combat_player_weight=layout['main_player_weight'],
                  combat_router_weight=layout['main_router_weight'],
                  combat_enemy_weight=layout['main_enemy_weight'])
    return layout


def commit_moe_batch_revision(state, *, step, horizon):
    """Called inside commit_moe_update after its gate clock, never separately."""
    gate = state['rssm_training']['moe_training']
    record = gate.get('batch_revision')
    if record is None:
        return None
    settings = batch_settings_for_gate(gate)
    if (step != state['global_training_step'] or step != record['last_optimizer_step'] + 1
            or gate['last_optimizer_step'] != step
            or gate['updates'] != record['source_moe_updates'] + record['updates'] + 1
            or str(horizon) not in record['horizon_update_counts']):
        raise ValueError('MoE micro16 ledger requires one complete ordered optimizer commit')
    count = settings['effective_batch_size']
    record.update(last_optimizer_step=step, updates=record['updates'] + 1,
                  main_examples=record['main_examples'] + count,
                  main_transitions=record['main_transitions'] + horizon * count)
    record['horizon_update_counts'][str(horizon)] += 1
    return record
