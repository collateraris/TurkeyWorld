"""Short action/contact supervision beside an unchanged autonomous long stream.

The first stream uses only its observed initialization and its own predictions.
The second stream is an explicitly separate Source-pose labeled query stream.
Neither can overwrite the caller's450-step state, memory or numerical history.
Only observed initial RGB is encoded online, even in the auxiliary stream.
"""
from __future__ import annotations

from contextlib import nullcontext
import math

import torch
from torch.nn import functional as F

from .gradient_safety import finite_gradient_norm
from .structured_state import STATE_DIM, player_command_delta
from .spatial_entities import entity_motion_basis
from .trajectory_objectives import trajectory_state_losses


PASS_WEIGHT = 1.
RAW_DELTA_WEIGHT = 2.
SCALED_MOTION_WEIGHT = 5.
AXIS_COMMAND_EPS = 1e-5
FREE_FRACTION = .9
BLOCKED_FRACTION = .1


def pass_fraction_targets(previous, target, action, *, dt=1/30):
    """Loss-only Source pass fractions; inactive command axes carry no label."""
    if (previous.ndim != 2 or previous.shape[-1] != STATE_DIM or target.shape != previous.shape
            or action.shape != (len(previous), 4) or not math.isfinite(dt) or dt <= 0):
        raise ValueError('Pass targets require matching Source state pairs and controls')
    previous, target, action = previous.detach().float(), target.detach().float(), action.detach().float()
    command = player_command_delta(previous, action, dt=dt).detach()
    active = (command.abs() >= AXIS_COMMAND_EPS) & (previous[:, 4:5] > 0)
    safe_command = torch.where(active, command, torch.ones_like(command))
    fractions = ((target[:, :2]-previous[:, :2])/safe_command).clamp(0, 1)
    fractions = torch.where(active, fractions, torch.zeros_like(fractions))
    return fractions, active, command


def balanced_pass_weights(fractions, active):
    """Equal present free/blocked/partial mass per axis over all BxH labels."""
    if fractions.shape != active.shape or fractions.shape[-1] != 2:
        raise ValueError('Pass weights require matching[...,2] labels/masks')
    labels, valid = fractions.detach(), active.detach().bool()
    groups = ((labels >= FREE_FRACTION) & valid, (labels <= BLOCKED_FRACTION) & valid,
              (labels > BLOCKED_FRACTION) & (labels < FREE_FRACTION) & valid)
    dimensions = tuple(range(labels.ndim-1))
    counts = [mask.sum(dimensions) for mask in groups]
    present = torch.stack([count > 0 for count in counts]).sum(0)
    axes = (present > 0).sum().clamp_min(1)
    weights = torch.zeros_like(labels)
    for mask, count in zip(groups, counts):
        weights = weights+mask.to(labels.dtype)/count.clamp_min(1)/present.clamp_min(1)/axes
    return weights.detach()


def _source_batch(batch, context):
    states, frames, actions, targets, grid = (batch[name].detach() for name in
        ('states', 'frames', 'actions', 'target_states', 'map'))
    horizon = actions.shape[1] if actions.ndim == 3 else 0
    if (not isinstance(context, int) or isinstance(context, bool) or context < 1 or not 1 <= horizon <= 16
            or states.ndim != 3 or states.shape[-1] != STATE_DIM or actions.shape[-1] != 4
            or states.shape[0] != actions.shape[0] or states.shape[1] < context+horizon
            or targets.shape != (len(states), horizon, STATE_DIM) or frames.ndim != 5
            or frames.shape[:2] != states.shape[:2] or frames.shape[2] != 3
            or grid.shape != (len(states), 16, 16)):
        raise ValueError('Short recovery requires genuine context and1..16 matching Source transitions')
    if not all(bool(torch.isfinite(value).all()) for value in (states, frames, actions, targets)):
        raise FloatingPointError('Nonfinite short motion Source labels')
    previous = batch.get('previous_states', states[:, context-1:context-1+horizon]).detach().float()
    if previous.shape != targets.shape or not torch.equal(previous[:, 0], states[:, context-1].float()):
        raise ValueError('Source auxiliary previous states do not match their initialization')
    return states.float(), frames, actions.float(), targets.float(), grid, previous


def _initial_memory(model, frames, states, map_features, context, observed):
    latents = model.core.encode_video(frames[:, context-observed:context])
    return model.initialize_from_latents(latents, states[:, context-observed:context], map_features)


def _local(delta, basis):
    return torch.stack([(delta*basis['forward'][:, 0]).sum(-1),
                        (delta*basis['right'][:, 0]).sum(-1)], -1)


def short_motion_backward(model, batch, *, context=2, horizon=None, observed_context=None,
                          amp=None, auxiliary_chunk_size=4, loss_weight=1., check_gradients=True):
    """Accumulate short-AR and globally balanced GT-query motion gradients.

    No optimizer/EMA step, renderer call, map gradient, mode change or Source
    state repair occurs. The caller performs its one optimizer update after
    the independent full450-step backward and this auxiliary backward.
    """
    observed = context if observed_context is None else observed_context
    if (not isinstance(observed, int) or isinstance(observed, bool) or not 1 <= observed <= context
            or not isinstance(auxiliary_chunk_size, int) or not 1 <= auxiliary_chunk_size <= 16
            or amp is not None and not callable(amp) or not math.isfinite(loss_weight) or loss_weight < 0):
        raise ValueError('Invalid short motion context/chunk/weight/AMP configuration')
    states, frames, actions, targets, grid, previous = _source_batch(batch, context)
    length, batch_size = actions.shape[1], len(actions)
    if horizon is not None and horizon != length:
        raise ValueError('Explicit short horizon must match the recorded Source batch')
    autocast = nullcontext if amp is None else amp
    with torch.no_grad(), autocast():
        features = model.encode_map(grid).detach()
    # Complete and backpropagate the independent own short trajectory before
    # querying any future Source poses. No Source future RGB is ever encoded.
    with autocast():
        memory = _initial_memory(model, frames, states, features, context, observed)
        current = states[:, context-1]
        own = []
        for index in range(length):
            result = model.transition(current, actions[:, index], features, memory)
            current, memory = result['state'], result['memory']
            if not bool(torch.isfinite(current).all() and torch.isfinite(memory).all()):
                raise FloatingPointError('Nonfinite short own state/memory')
            own.append(current)
        own_parts, own_metrics = trajectory_state_losses(torch.stack(own, 1), states[:, context-1], targets, grid,
            offsets=sorted({offset for offset in (1, 4, 8, 16, length) if offset <= length}))
        own_loss = loss_weight*sum(own_parts.values())
    if not bool(torch.isfinite(own_loss)):
        raise FloatingPointError('Nonfinite short own trajectory objective')
    own_loss.backward()
    maximum_norm = float(finite_gradient_norm(model.named_parameters(), context='Short own-motion backward')) if check_gradients else 0.
    del own, result, current, memory, own_loss
    # Global Source label weights are computed before chunking. A single free
    # event therefore keeps its class mass even among many blocked queries.
    fractions, active, commands = pass_fraction_targets(previous.flatten(0, 1), targets.flatten(0, 1),
                                                        actions.flatten(0, 1), dt=1/30)
    fractions, active, commands = (value.reshape(batch_size, length, 2) for value in (fractions, active, commands))
    weights = balanced_pass_weights(fractions, active)
    moving = (previous[:, :, 4] > 0) & actions[:, :, :2].ne(0).any(-1)
    motion_denominator = (2*moving.sum()).clamp_min(1)
    totals = {'pass_bce': 0., 'raw_delta': 0., 'scaled_motion': 0.}
    free = active & (fractions >= FREE_FRACTION); blocked = active & (fractions <= BLOCKED_FRACTION)
    false_stops = missed_blocks = 0
    error_sum = free_error = blocked_leak = 0.
    backward_calls = 1
    with autocast():
        memory = _initial_memory(model, frames, states, features, context, observed)
    for first in range(0, length, auxiliary_chunk_size):
        last = min(length, first+auxiliary_chunk_size)
        if first:
            memory = memory.detach()
        with autocast():
            parts = dict.fromkeys(totals, None)
            for index in range(first, last):
                query = model.transition(previous[:, index], actions[:, index], features, memory)
                memory = query['memory']
                if not bool(torch.isfinite(query['state']).all() and torch.isfinite(memory).all()):
                    raise FloatingPointError('Nonfinite Source-query state/memory')
                if any(name not in query for name in ('player_pass_logits', 'raw_player_delta', 'player_command_delta')):
                    raise ValueError('Short motion recovery requires axis_fraction player heads')
                logits, raw = query['player_pass_logits'].float(), query['raw_player_delta'].float()
                if logits.shape != (batch_size, 2) or raw.shape != logits.shape:
                    raise ValueError('Player pass/raw motion heads must return[B,2]')
                cross_entropy = F.binary_cross_entropy_with_logits(logits, fractions[:, index], reduction='none')
                pass_loss = (cross_entropy*weights[:, index]).sum()
                expected = targets[:, index, :2]-previous[:, index, :2]
                accepted = query['state'][:, :2]-previous[:, index, :2]
                basis = entity_motion_basis(previous[:, index], actions[:, index], dt=1/30)
                raw_error = F.smooth_l1_loss(_local(raw*200, basis), _local(expected*200, basis), reduction='none')
                tick_error = F.smooth_l1_loss(_local(accepted*200, basis), _local(expected*200, basis), reduction='none')
                values = {'pass_bce': pass_loss,
                    'raw_delta': (raw_error.sum(-1)*moving[:, index]).sum()/motion_denominator,
                    'scaled_motion': (tick_error.sum(-1)*moving[:, index]).sum()/motion_denominator}
                parts = {name: value if parts[name] is None else parts[name]+value for name, value in values.items()}
                with torch.no_grad():
                    probabilities = logits.sigmoid()
                    false_stops += int(((probabilities < .5) & free[:, index]).sum())
                    missed_blocks += int(((probabilities >= .5) & blocked[:, index]).sum())
                    free_error += float(((1-probabilities).abs()*free[:, index]).sum())
                    blocked_leak += float((probabilities*blocked[:, index]).sum())
                    error_sum += float(((accepted-expected).norm(dim=-1)*16/(2.4/30)*moving[:, index]).sum())
            value = loss_weight*(PASS_WEIGHT*parts['pass_bce']+RAW_DELTA_WEIGHT*parts['raw_delta']
                                 +SCALED_MOTION_WEIGHT*parts['scaled_motion'])
        if not bool(torch.isfinite(value)):
            raise FloatingPointError('Nonfinite Source-query motion objective')
        value.backward()
        backward_calls += 1
        if check_gradients:
            maximum_norm = max(maximum_norm, float(finite_gradient_norm(model.named_parameters(),
                context=f'Source-query motion backward{last}/{length}')))
        for name, part in parts.items():
            totals[name] += float(part.detach())
    own_values = {name: float(value.detach()) for name, value in own_parts.items()}
    total = loss_weight*(sum(own_values.values())+PASS_WEIGHT*totals['pass_bce']
        +RAW_DELTA_WEIGHT*totals['raw_delta']+SCALED_MOTION_WEIGHT*totals['scaled_motion'])
    return {'motion_aux_loss': total, 'motion_aux_train_loss': total,
        **{'motion_aux_'+name: value for name, value in totals.items()},
        **{'motion_aux_'+name: value for name, value in own_values.items()},
        'motion_aux_pass_weight': PASS_WEIGHT, 'motion_aux_raw_weight': RAW_DELTA_WEIGHT,
        'motion_aux_scaled_weight': SCALED_MOTION_WEIGHT, 'motion_aux_draws': batch_size,
        'motion_aux_horizon': length, 'motion_aux_observed_frames': observed,
        'motion_aux_ar_transitions': batch_size*length, 'motion_aux_source_query_transitions': batch_size*length,
        'motion_aux_backward_calls': backward_calls, 'motion_aux_max_gradient_norm': maximum_norm,
        'motion_aux_free_axes': int(free.sum()), 'motion_aux_blocked_axes': int(blocked.sum()),
        'motion_aux_partial_axes': int((active & ~free & ~blocked).sum()),
        'motion_aux_false_stop_rate': false_stops/max(int(free.sum()), 1),
        'motion_aux_missed_block_rate': missed_blocks/max(int(blocked.sum()), 1),
        'motion_aux_free_speed_error': free_error/max(int(free.sum()), 1),
        'motion_aux_blocked_speed_leak': blocked_leak/max(int(blocked.sum()), 1),
        'motion_aux_relative_velocity_error': error_sum/max(int(moving.sum()), 1),
        'motion_aux_gt_feedback_to_long': False, 'motion_aux_future_rgb_online_encoded': False,
        'motion_aux_online_context_encoder_calls': 2, 'motion_aux_renderer_calls': 0,
        'motion_aux_optimizer_steps': 0, 'motion_aux_ar_offsets': own_metrics['ar_xy_offsets']}
