"""One own-state/own-memory dynamics backward pass with bounded TBPTT graphs.

The caller clips and performs ONE optimizer/EMA update after this function.
No renderer, GAN, optimizer step, state repair or future-image input is used.
Future RGB is encoded only by the frozen EMA target, one small chunk at a time.
"""
from __future__ import annotations

from contextlib import nullcontext
import math

import torch
from torch.nn import functional as F

from .jepa_training_utils import latent_prediction_loss, variance_covariance_loss
from .structured_state import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
from .train import state_loss
from .trajectory_objectives import trajectory_state_losses, XY_WEIGHT
from .gradient_safety import finite_gradient_norm


LONG_COORDINATE_OFFSETS = (1, 4, 8, 16, 32, 64, 128, 256, 450)


def long_dynamics_backward(model, ema, batch, *, context=2, chunk_size=16, amp=None,
                           expected_horizon=450, progress_callback=None, check_gradients=True,
                           observed_context=None):
    """Accumulate gradients for a single full recorded-control own rollout.

    State/JEPA/wall chunk means receive chunk_length/full_horizon weights;
    all future XY offsets receive one denominator over the entire rollout.
    Thus a final2-step chunk is neither discarded nor over-weighted. The
    context regularizer is added once, with its ordinary coefficient .1.
    Only graph history is detached at chunk boundaries; numerical values carry.
    """
    if (any(not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in (context, chunk_size, expected_horizon))
            or amp is not None and not callable(amp)
            or progress_callback is not None and not callable(progress_callback)
            or not isinstance(check_gradients,bool)):
        raise ValueError('Long dynamics requires positive context/chunk_size and an AMP context factory')
    autocast = nullcontext if amp is None else amp
    observed = context if observed_context is None else observed_context
    if not isinstance(observed, int) or isinstance(observed, bool) or not 1 <= observed <= context:
        raise ValueError('Observed context must select1..context genuine past frames')
    actions, states, frames = (batch[name].detach() for name in ('actions', 'states', 'frames'))
    targets, grid = batch['target_states'].detach(), batch['map'].detach()
    if (actions.ndim != 3 or actions.shape[-1] != 4 or actions.shape[1] < 1
            or states.ndim != 3 or states.shape[0] != actions.shape[0] or states.shape[-1] != STATE_DIM
            or targets.shape != (*actions.shape[:2], STATE_DIM)
            or frames.ndim != 5 or frames.shape[0] != len(actions) or frames.shape[2] != 3
            or min(states.shape[1], frames.shape[1]) < context+actions.shape[1]
            or grid.shape != (len(actions), 16, 16)):
        raise ValueError('Long dynamics requires complete matching context/future states, frames, controls and maps')
    horizon = actions.shape[1]
    if horizon != expected_horizon:
        raise ValueError(f'Long dynamics requires exactly{expected_horizon} future transitions; received{horizon}')
    initial = states[:, context-1].float()
    if not all(bool(torch.isfinite(value).all()) for value in (actions, states, targets, frames)):
        raise FloatingPointError('Nonfinite recorded long dynamics inputs')
    with torch.no_grad(), autocast():
        map_features = model.encode_map(grid).detach()
    with autocast():
        context_latents = model.core.encode_video(frames[:, context-observed:context])
        memory = model.initialize_from_latents(context_latents, states[:, context-observed:context], map_features)
        regularizer, regularizer_metrics = variance_covariance_loss(context_latents)
        regularizer = .1*regularizer
    current = initial
    expected_memory_shape = tuple(memory.shape)
    offsets = list(range(1, horizon+1))
    selected = torch.tensor([offset-1 for offset in offsets], device=targets.device)
    source_npc = targets.float()[:, :, GLOBAL_DIM:-2].reshape(len(targets), horizon, ENTITY_SLOTS, ENTITY_DIM)
    first_npc = initial[:, GLOBAL_DIM:-2].reshape(len(targets), ENTITY_SLOTS, ENTITY_DIM)
    source_alive = (source_npc[..., 5] >= .5) & (source_npc[..., 4] > 0)
    initial_alive = (first_npc[..., 5] >= .5) & (first_npc[..., 4] > 0)
    previous_alive = torch.cat([initial_alive[:, None], source_alive[:, :-1]], 1)
    respawn_seen = ((~previous_alive & source_alive).long().cumsum(1) > 0)
    stable_npc = initial_alive[:, None] & source_alive & ~respawn_seen
    player_mask = (initial[:, None, 4] > 0) & (targets[:, :, 4] > 0)
    player_denominator = (2*player_mask[:, selected].sum()).clamp_min(1)
    npc_denominator = (2*stable_npc[:, selected].sum()).clamp_min(1)
    totals = dict.fromkeys(('state', 'jepa', 'ar_xy_player', 'ar_xy_npc',
                          'ar_wall_player', 'ar_wall_npc', 'regularizer'), 0.)
    overlaps_player = overlaps_npc = transitions = chunks = 0
    max_gradient_norm=0.
    for first in range(0, horizon, chunk_size):
        last = min(horizon, first+chunk_size)
        if first:
            current, memory = current.detach(), memory.detach()
        own_previous = current.detach()
        previous_source = initial if first == 0 else targets[:, first-1].float()
        with autocast():
            predictions, latents = [], []
            for offset in range(first, last):
                result = model.transition(current, actions[:, offset], map_features, memory)
                current, memory = result['state'], result['memory']
                predictions.append(current)
                latents.append(result['latent_prediction'])
            predicted, predicted_latents = torch.stack(predictions, 1), torch.stack(latents, 1)
        if (predicted.shape != (len(actions), last-first, STATE_DIM)
                or tuple(memory.shape) != expected_memory_shape
                or not bool(torch.isfinite(predicted).all()) or not bool(torch.isfinite(memory).all())):
            raise FloatingPointError(f'Invalid own state/memory at long future offset{last}')
        with torch.no_grad(), autocast():
            target_latents = ema(frames[:, context+first:context+last])
        with autocast():
            state_term = 10*state_loss(predicted.flatten(0, 1), targets[:, first:last].flatten(0, 1))
            jepa_term, _ = latent_prediction_loss(predicted_latents, target_latents)
            # Circle endpoints and source-X-then-Y paths cover every transition.
            # Only Source labels determine eligibility; the preceding point is
            # the model's carried prediction, never a new Source injection.
            wall_parts, wall_metrics = trajectory_state_losses(predicted, previous_source,
                targets[:, first:last], grid, offsets=range(1, last-first+1), xy_weight=0.,
                previous_prediction=own_previous)
            fraction = (last-first)/horizon
            contributions = {'state': fraction*state_term, 'jepa': fraction*jepa_term,
                'ar_wall_player': fraction*wall_parts['ar_wall_player'],
                'ar_wall_npc': fraction*wall_parts['ar_wall_npc']}
            local = [offset-1-first for offset in offsets if first < offset <= last]
            if local:
                index = torch.tensor(local, device=predicted.device)
                absolute = index+first
                player_error = F.smooth_l1_loss(predicted[:, index, :2].float()*16,
                    targets[:, absolute, :2].float()*16, reduction='none').sum(-1)
                npc_predicted = predicted[:, index, GLOBAL_DIM:-2].float().reshape(len(actions), len(local), ENTITY_SLOTS, ENTITY_DIM)
                npc_error = F.smooth_l1_loss(npc_predicted[..., :2]*16,
                    source_npc[:, absolute, :, :2]*16, reduction='none').sum(-1)
                contributions['ar_xy_player'] = XY_WEIGHT*(player_error*player_mask[:, absolute]).sum()/player_denominator
                contributions['ar_xy_npc'] = XY_WEIGHT*(npc_error*stable_npc[:, absolute]).sum()/npc_denominator
            if first == 0:
                contributions['regularizer'] = regularizer
            value = sum(contributions.values())
        if not bool(torch.isfinite(value)):
            raise FloatingPointError(f'Nonfinite long dynamics objective at future offset{last}')
        value.backward()
        if check_gradients:
            norm=finite_gradient_norm(model.named_parameters(),
                context=f'Long rollout backward at future offset{last}/{horizon}')
            max_gradient_norm=max(max_gradient_norm,float(norm))
        for name, part in contributions.items():
            totals[name] += float(part.detach())
        overlaps_player += wall_metrics['ar_wall_player_overlaps']
        overlaps_npc += wall_metrics['ar_wall_npc_overlaps']
        transitions += last-first
        chunks += 1
        if progress_callback is not None:
            progress_callback(last, horizon)
    train_loss = sum(totals.values())
    if not math.isfinite(train_loss):
        raise FloatingPointError('Nonfinite aggregated long dynamics loss')
    return {**totals, 'train_loss': train_loss,
        **{name: float(value) for name, value in regularizer_metrics.items()},
        'long_rollout': True, 'long_horizon': horizon, 'tbptt_chunk': chunk_size,
        'long_tbptt_chunks': chunks, 'long_forward_transitions': transitions,
        'long_coordinate_offsets': offsets, 'long_optimizer_updates': 1,
        'long_report_offsets': [value for value in LONG_COORDINATE_OFFSETS if value <= horizon],
        'optimizer_step_performed': False, 'long_memory_reset_count': 0,
        'long_online_context_encoder_calls': 1, 'long_ema_target_encoder_calls': chunks,
        'initial_observed_frames': observed,
        'initial_observed_frame_indices': list(range(context-observed, context)),
        'long_gt_feedback': False, 'long_future_rgb_online_encoded': False,
        'long_render_calls': 0, 'rgb_supervised_steps': 0, 'image_objectives_active': False,
        'dynamics_updated': True, 'ar_wall_player_overlaps': overlaps_player,
        'ar_wall_npc_overlaps': overlaps_npc,
        'long_gradient_checks':chunks if check_gradients else 0,
        'long_max_gradient_norm':max_gradient_norm,
        'loss_reduction': 'time_weighted_chunks_and_global_selected_xy_pairs'}
