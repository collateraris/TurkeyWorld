"""Opt-in Source-only displacement labels with whole-tape normalization.

These losses query an already computed prior. They do not run a model, read an
image, alter a rollout state, or use predicted life flags to hide motion error.
Speeds and axis residuals use cells (map tiles) per second.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from .motion_recovery_training import pass_fraction_targets
from .structured_state import ENTITY_DIM, ENTITY_SLOTS, GLOBAL_DIM, STATE_DIM


KINEMATIC_PROTOCOL_V1 = {
    'version': 1, 'player_axes_weight': 1., 'npc_speed_weight': 1.,
    'npc_direction_weight': .25, 'direction_min_speed': .05,
    'cosine_epsilon_speed': .05, 'source_free_fraction': .95,
    'source_blocked_fraction': .05,
}


def normalize_kinematic_protocol(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value)-set(KINEMATIC_PROTOCOL_V1):
        raise ValueError('Kinematic protocol contains unknown settings')
    result = {**KINEMATIC_PROTOCOL_V1, **value}
    if type(result['version']) is not int or result['version'] != 1:
        raise ValueError('Kinematic protocol requires version1')
    for name, item in result.items():
        if name == 'version':
            continue
        if (not isinstance(item, (int, float)) or isinstance(item, bool)
                or not math.isfinite(item) or item < 0):
            raise ValueError('Kinematic settings must be finite and nonnegative: '+name)
    if (result['direction_min_speed'] <= 0 or result['cosine_epsilon_speed'] <= 0
            or not 0 <= result['source_blocked_fraction'] < result['source_free_fraction'] <= 1):
        raise ValueError('Invalid kinematic direction or Source-axis thresholds')
    return result


def prepare_kinematic_supervision(previous, target, actions, *, kinematic_protocol,
                                   simulation_hz=30.):
    """Prepare detached masks and denominators over the entire B x H tape.

    Free, blocked and partial Source outcomes each receive equal present mass
    within each active axis; X and Y then receive equal present mass. Thus a
    sliding sample supervises both its blocked axis and its continuing axis.
    No per-timestep mean is taken, including for rare continuing NPC labels.
    """
    config = normalize_kinematic_protocol(kinematic_protocol)
    if config is None:
        raise ValueError('Kinematic supervision requires its opt-in protocol')
    if (not isinstance(simulation_hz, (int, float)) or isinstance(simulation_hz, bool)
            or not math.isfinite(simulation_hz) or simulation_hz <= 0):
        raise ValueError('Kinematic simulation rate must be finite and positive')
    if (previous.ndim != 3 or previous.shape[-1] != STATE_DIM or target.shape != previous.shape
            or previous.shape[0] < 1 or not 1 <= previous.shape[1] <= 16
            or actions.shape != (*previous.shape[:2], 4)
            or not all(bool(torch.isfinite(value).all()) for value in (previous, target, actions))):
        raise ValueError('Kinematic labels require finite Source[B,1..16,state] pairs and controls')
    previous, target, actions = (value.detach().float() for value in (previous, target, actions))
    batch, horizon = previous.shape[:2]
    fraction, active, _ = pass_fraction_targets(previous.flatten(0, 1), target.flatten(0, 1),
                                                actions.flatten(0, 1), dt=1/simulation_hz)
    fraction, active = (value.reshape(batch, horizon, 2) for value in (fraction, active))
    groups = {'free': active & (fraction >= config['source_free_fraction']),
              'blocked': active & (fraction <= config['source_blocked_fraction'])}
    groups['partial'] = active & ~groups['free'] & ~groups['blocked']
    counts = {name: mask.sum((0, 1)) for name, mask in groups.items()}
    present_classes = torch.stack([count > 0 for count in counts.values()]).sum(0)
    present_axes = (present_classes > 0).sum().clamp_min(1)
    axis_weight = torch.zeros_like(fraction)
    for name, mask in groups.items():
        axis_weight += (mask.float()/counts[name].clamp_min(1)
                        /present_classes.clamp_min(1)/present_axes)
    old = previous[..., GLOBAL_DIM:-2].reshape(batch, horizon, ENTITY_SLOTS, ENTITY_DIM)
    new = target[..., GLOBAL_DIM:-2].reshape_as(old)
    continuing = ((old[..., 5] >= .5) & (old[..., 4] > 0)
                  & (new[..., 5] >= .5) & (new[..., 4] > 0)
                  & (previous[..., 4:5] > 0))
    source_npc_velocity = (new[..., :2]-old[..., :2])*(16*simulation_hz)
    source_npc_speed = source_npc_velocity.norm(dim=-1)
    direction = continuing & (source_npc_speed >= config['direction_min_speed'])
    return {'config': config, 'simulation_hz': float(simulation_hz),
        'horizon': horizon, 'previous': previous, 'target': target,
        'source_player_velocity': (target[..., :2]-previous[..., :2])*(16*simulation_hz),
        'player_active': active, 'player_axis_weight': axis_weight.detach(),
        'player_axis_groups': groups, 'player_axis_group_counts': counts,
        'player_axis_counts': active.sum((0, 1)),
        'player_sliding': groups['free'].any(-1) & groups['blocked'].any(-1),
        'npc_continuing': continuing, 'npc_continuing_count': continuing.sum(),
        'source_npc_velocity': source_npc_velocity, 'source_npc_speed': source_npc_speed,
        'npc_direction': direction, 'npc_direction_count': direction.sum()}


def kinematic_supervised_terms(prediction, *, supervision, index):
    """Return additive losses and detached, additive diagnostics for one tick.

    Each term already uses complete-tape denominators. Callers sum ticks and
    TBPTT chunks directly; they must not divide these terms by H again.
    """
    if type(index) is not int or not 0 <= index < supervision['horizon']:
        raise ValueError('Kinematic timestep is outside its prepared Source tape')
    previous = supervision['previous'][:, index]
    if prediction.shape != previous.shape or not bool(torch.isfinite(prediction).all()):
        raise ValueError('Kinematic prediction must be a finite matching explicit state')
    prediction = prediction.float()
    config, scale = supervision['config'], 16*supervision['simulation_hz']
    player_velocity = (prediction[:, :2]-previous[:, :2])*scale
    player_error = player_velocity-supervision['source_player_velocity'][:, index]
    player_loss = (F.smooth_l1_loss(player_velocity, supervision['source_player_velocity'][:, index],
                                  reduction='none')*supervision['player_axis_weight'][:, index]).sum()
    old = previous[:, GLOBAL_DIM:-2].reshape(-1, ENTITY_SLOTS, ENTITY_DIM)
    new = prediction[:, GLOBAL_DIM:-2].reshape_as(old)
    npc_velocity = (new[..., :2]-old[..., :2])*scale
    npc_speed = npc_velocity.norm(dim=-1)
    source_speed = supervision['source_npc_speed'][:, index]
    continuing = supervision['npc_continuing'][:, index]
    speed_denominator = supervision['npc_continuing_count'].clamp_min(1)
    speed_loss = (F.smooth_l1_loss(npc_speed, source_speed, reduction='none')
                  *continuing).sum()/speed_denominator
    direction_mask = supervision['npc_direction'][:, index]
    direction_denominator = supervision['npc_direction_count'].clamp_min(1)
    cosine = F.cosine_similarity(npc_velocity, supervision['source_npc_velocity'][:, index],
                                 dim=-1, eps=config['cosine_epsilon_speed'])
    direction_loss = ((1-cosine)*direction_mask).sum()/direction_denominator
    terms = {'combat_kinematic_player_axes': config['player_axes_weight']*player_loss,
             'combat_kinematic_npc_speed': config['npc_speed_weight']*speed_loss,
             'combat_kinematic_npc_direction': config['npc_direction_weight']*direction_loss}
    active = supervision['player_active'][:, index]
    metrics = {'combat_kinematic_player_axes_count': active.sum(),
        'combat_kinematic_player_sliding_count': supervision['player_sliding'][:, index].sum(),
        'combat_kinematic_npc_speed_count': continuing.sum(),
        'combat_kinematic_npc_direction_count': direction_mask.sum(),
        'combat_kinematic_npc_stationary_count': (continuing & ~direction_mask).sum(),
        'combat_kinematic_npc_speed_mae': ((npc_speed-source_speed).abs()*continuing).sum()/speed_denominator,
        'combat_kinematic_npc_direction_cosine_error': direction_loss}
    for axis, name in enumerate(('x', 'y')):
        metrics['combat_kinematic_player_'+name+'_velocity_mae'] = (
            (player_error[:, axis].abs()*active[:, axis]).sum()
            /supervision['player_axis_counts'][axis].clamp_min(1))
        for outcome, mask in supervision['player_axis_groups'].items():
            selected = mask[:, index, axis]
            prefix = 'combat_kinematic_player_'+name+'_'+outcome
            metrics[prefix+'_count'] = selected.sum()
            metrics[prefix+'_velocity_mae'] = ((player_error[:, axis].abs()*selected).sum()
                /supervision['player_axis_group_counts'][outcome][axis].clamp_min(1))
    return terms, {name: value.detach() for name, value in metrics.items()}
