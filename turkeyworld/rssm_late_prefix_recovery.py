"""Bounded continuation of actual own prefixes for late sixteen-step recovery.

The scout consumes only recorded controls after the main loss has run. It
continues the main rollout's detached numerical state and packed h/z without
an observation, reset, Source state, or correction. Source labels remain the
separate, existing loss-only counterfactual queries after capture.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext

import torch


LATE_PREFIX_RECOVERY_PROTOCOL = {
    'version': 1, 'mode': 'continue_actual_main_prefix', 'scout_batch_size': 8,
    'scout_horizon': 450, 'branch_horizon': 16, 'minimum_prefix_steps': 16,
    'reserved_late_branches': 8, 'no_grad_scout': True,
    'new_observed_rgb_frames': 0, 'source_future_feedback': False,
}


def normalize_late_prefix_protocol(value=None):
    if value is None:
        return None
    if (not isinstance(value, dict) or value != LATE_PREFIX_RECOVERY_PROTOCOL
            or any(type(value.get(key)) is not type(item)
                   for key, item in LATE_PREFIX_RECOVERY_PROTOCOL.items())):
        raise ValueError('Unsupported actual-own late-prefix recovery protocol')
    return dict(LATE_PREFIX_RECOVERY_PROTOCOL)


@contextmanager
def _prefix_map_caches(model, count):
    """Scope batch-shaped map caches to the same first actors as the scout.

    The encoded main features stay unchanged. These nonpersistent caches are
    read by spawn/renderer heads independently of the features argument, so a
    B64 main prefix also needs B8 views here. Restore the original objects even
    when a transition fails, before any main-batch auxiliary or branch query.
    """
    saved = [(module, name, getattr(module, name)) for module in model.modules()
             for name in ('_spawn_walkable', '_map_condition')
             if torch.is_tensor(getattr(module, name, None))]
    try:
        for module, name, value in saved:
            if value.numel() and value.ndim and value.shape[0] > count:
                setattr(module, name, value[:count])
        yield
    finally:
        for module, name, value in saved:
            setattr(module, name, value)


def continue_own_prefix_candidates(model, state, memory, features, grid, actions, *,
                                   start, late_prefix_protocol, amp=None):
    """Continue first <=8 actual main actors, capture private no-grad windows."""
    config = normalize_late_prefix_protocol(late_prefix_protocol)
    if config is None:
        raise ValueError('Own late-prefix continuation requires its explicit protocol')
    count = len(state)
    if (type(start) is not int or not config['minimum_prefix_steps'] <= start < config['scout_horizon']
            or start % config['branch_horizon'] or not 1 <= count <= config['scout_batch_size']
            or state.ndim != 2 or state.shape[1] != 20 or memory.shape[0] != count
            or features.shape[0] != count or grid.shape != (count, 16, 16)
            or actions.shape != (count, config['scout_horizon'], 4)
            or not all(bool(torch.isfinite(value).all()) for value in (state, memory, features, actions))):
        raise ValueError('Late scout requires exact own state/h-z and matching450 controls for <=8 actors')
    from .rssm_combat_branches import capture_combat_candidate
    current, carried = state.detach(), memory.detach()
    actions, features, grid = actions.detach(), features.detach(), grid.detach()
    expected_memory = tuple(carried.shape)
    candidates = []
    autocast = nullcontext if amp is None else amp
    device = state.device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    # Scout sampling must not shift the existing auxiliary/branch RNG stream.
    with _prefix_map_caches(model, count), torch.random.fork_rng(devices=devices), torch.no_grad(), autocast():
        for first in range(start, config['scout_horizon'], config['branch_horizon']):
            old, old_memory = current, carried
            predictions = []
            for index in range(first, min(first+config['branch_horizon'], config['scout_horizon'])):
                result = model.transition(current, actions[:, index], features, carried)
                current, carried = result['state'], result['memory']
                if (current.shape != state.shape or tuple(carried.shape) != expected_memory
                        or not bool(torch.isfinite(current).all() and torch.isfinite(carried).all())):
                    raise FloatingPointError('Nonfinite own late-prefix scout; no reset or repair')
                predictions.append(current)
            candidates.append(capture_combat_candidate(first, old, old_memory,
                actions[:, first:first+config['branch_horizon']], grid, torch.stack(predictions, 1)))
    return candidates, {'active': True, 'mode': config['mode'], 'actors': count,
        'start_step': start, 'end_step': config['scout_horizon'],
        'transition_calls': config['scout_horizon']-start,
        'actor_transitions': count*(config['scout_horizon']-start),
        'captured_windows': len(candidates), 'new_observed_rgb_frames': 0,
        'source_future_states_consumed': 0, 'source_future_feedback': False,
        'memory_resets': 0, 'state_resets': 0, 'optimizer_steps': 0,
        'grad_enabled': False, 'sampling_rng_restored': True}


def is_late_prefix(row, protocol):
    return row['step'] >= protocol['minimum_prefix_steps']


def order_late_prefix_probes(rows, *, late_prefix_protocol, budget):
    """Reserve half of the existing Source probe budget for actual late rows."""
    config = normalize_late_prefix_protocol(late_prefix_protocol)
    late = [row for row in rows if is_late_prefix(row, config)]
    late.sort(key=lambda row: (not row['pre_wall'], not row.get('post_spawn_recovery', False),
                               row['step'], row['actor']))
    # Bound the front reservation so step0 still has real contrast coverage.
    reserved = late[:max(1, budget//2)]
    identities = {id(row) for row in reserved}
    return reserved+[row for row in rows if id(row) not in identities]


def select_late_prefix_rows(eligible, existing, *, late_prefix_protocol, max_branches):
    """Reserve late slots after canonical legality checks; never fill by repair."""
    config = normalize_late_prefix_protocol(late_prefix_protocol)
    late = [row for row in eligible if is_late_prefix(row, config)]
    buckets = [
        [row for row in late if row['pre_wall']],
        [row for row in late if row.get('post_spawn_recovery', False)],
        [row for row in late if row['cohort'] in ('npc_motion', 'combat', 'spawn')],
        late,
    ]
    for bucket in buckets:
        bucket.sort(key=lambda row: (row['first_source_error'] is None, row['step'], row['actor']))
    requested = min(config['reserved_late_branches'], max_branches)
    reserved = []
    while len(reserved) < requested:
        added = False
        for bucket in buckets:
            available = [row for row in bucket if all(row is not old for old in reserved)]
            if not available:
                continue
            used_actors = {row['actor'] for row in reserved}
            reserved.append(next((row for row in available if row['actor'] not in used_actors), available[0]))
            added = True
            if len(reserved) == requested:
                break
        if not added:
            break
    chosen = []
    initial = next((row for row in existing if row['step'] == 0), None)
    if initial is not None and len(reserved) < max_branches:
        chosen.append(initial)
    chosen.extend(reserved)
    for row in existing+eligible:
        if len(chosen) == max_branches:
            break
        if all(row is not old for old in chosen):
            chosen.append(row)
    selected = [row for row in chosen if is_late_prefix(row, config)]
    return chosen, {'late_prefix_selection_active': True,
        'late_prefix_requested_count': int(requested), 'late_prefix_selected_count': int(len(selected)),
        'late_prefix_unfilled_count': int(max(0, requested-len(selected))),
        'late_prefix_legal_probe_count': int(len(late)),
        # Source geometry/life flags can be NumPy bool scalars. Keep the
        # production JSON boundary native without relying on probe sanitizers.
        'late_prefix_pre_contact_count': int(sum(bool(row['pre_wall']) for row in selected)),
        'late_prefix_post_spawn_count': int(sum(bool(row.get('post_spawn_recovery', False)) for row in selected)),
        'late_prefix_motion_combat_count': int(sum(row['cohort'] in ('npc_motion', 'combat', 'spawn') for row in selected)),
        'late_prefix_selected_steps': [int(row['step']) for row in selected]}
