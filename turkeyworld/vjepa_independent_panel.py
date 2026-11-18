"""Episode-disjoint H180 panel and locally initialized one-step player probes."""
import hashlib
import json
from pathlib import Path

import torch

from .structured_state import player_command_delta

# Fingerprint registered from both held-out splits before either A/B arm ran.
INDEPENDENT_FINGERPRINT = '8b7feb02125196bc2ee91454f08745bb8e412baeb1cfcafd0168977f7d15b5ac'
PROBE_OFFSETS = tuple(range(0, 180, 16))
MIN_AXIS_COMMAND = 1e-5  # normalized map units per tick
LOW, HIGH = .25, .75


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def independent_windows(data, published_windows, stream, count=8):
    """Choose one window per unused VAL episode, without reading model outputs."""
    if data.split != 'val':
        raise ValueError('Independent panel requires the explicit val split')
    excluded = {w['episode'] for w in published_windows}
    available = []
    for episode in data.episodes:
        name = Path(episode['path']).name
        last = len(episode['frames']) - 181
        if name in excluded or last < 3:
            continue
        key = fingerprint(['vjepa_heldout_episode_panel_v1', stream, name])
        # Reserve four distinct observed states immediately preceding transition.
        start = 3 + int(key[16:32], 16) % (last - 2)
        available.append((key, name, start))
    if len({r[1] for r in available}) != len(available) or len(available) < count:
        raise ValueError(f'Need {count} distinct non-reference val episodes for {stream}')
    return [dict(episode=name, start=start) for _, name, start in sorted(available)[:count]]


def axis_errors(command, expected_delta, predicted_delta):
    """Classify active XY components: stop <=25% of command, pass >=75%."""
    if not all(torch.isfinite(x).all() for x in (command, expected_delta, predicted_delta)):
        raise ValueError('Non-finite displacement in independent panel probe')
    active = command.abs() > MIN_AXIS_COMMAND
    denominator = torch.where(active, command, 1.)
    expected = expected_delta / denominator
    predicted = predicted_delta / denominator
    free = active & (expected >= HIGH)
    blocked = active & (expected <= LOW)
    return dict(free_axes=int(free.sum()), blocked_axes=int(blocked.sum()),
                partial_axes=int((active & ~free & ~blocked).sum()),
                false_stops=int((free & (predicted <= LOW)).sum()),
                false_passes=int((blocked & (predicted >= HIGH)).sum()))


def common_start_probes(adapter, past, grid, actions, target):
    """Each one-step transition starts from the same four Source states in both arms.

    SR images come from the frozen feedback renderer; the main 180-step rollout
    is never restarted or given future Source states.
    """
    if adapter.feedback_renderer is None:
        raise ValueError('Independent common-start probes require the generated SR domain')
    history = torch.cat((past, target), dim=1)
    totals = {}
    for tick in PROBE_OFFSETS:
        context = history[:, tick:tick+4]
        source = context[:, -1]
        action = actions[:, tick]
        current = adapter.initialize(context, grid, source.new_zeros(len(source), 4, 3, 32, 32))
        result = adapter.transition_prior(current, action)
        values = axis_errors(player_command_delta(source, action),
                             target[:, tick, :2] - source[:, :2],
                             result.state[:, :2] - source[:, :2])
        for k, v in values.items():
            totals[k] = totals.get(k, 0) + v
    return totals


def summarize_probes(groups):
    totals = {k: sum(g[k] for g in groups) for k in groups[0]}
    return {**totals, 'probe_offsets_zero_based': list(PROBE_OFFSETS),
            'false_stop_rate': totals['false_stops'] / totals['free_axes'] if totals['free_axes'] else None,
            'false_pass_rate': totals['false_passes'] / totals['blocked_axes'] if totals['blocked_axes'] else None,
            'min_axis_command_map_units': MIN_AXIS_COMMAND, 'stop_max_fraction': LOW, 'pass_min_fraction': HIGH}
