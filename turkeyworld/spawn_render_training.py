"""Auxiliary spawn-position image supervision, separate from autonomous memory.

The public decoder keeps its hard map/time/separation policy. Only a training
surrogate of that same legal categorical choice is differentiable here. Source
labels select a render query and supply its pose/targets; they never enter the
spawn head or replace the main autoregressive state.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from .object_losses import object_region_weights, weighted_image_l1
from .spawn import decode_spawn_positions
from .structured_state import DEATH_DURATION, MAP_EXTENT, RESPAWN_DELAY


def _walkable_batch(walkable, batch, device):
    if not torch.is_tensor(walkable):
        raise ValueError('Spawn image supervision requires a matching walkable tensor')
    if walkable.ndim == 2:
        walkable = walkable[None]
    if walkable.shape == (1, 16, 16):
        walkable = walkable.expand(batch, -1, -1)
    if walkable.shape != (batch, 16, 16):
        raise ValueError('Spawn walkable map must have shape[B,16,16]')
    return walkable.detach().to(device=device, dtype=torch.bool)


def spawn_positions_with_grad(logits, previous, walkable, dt):
    """Return hard-decoder-exact positions/availability with legal ST gradients.

    A soft expectation supplies only the derivative. Sequential masks use
    previously accepted HARD positions, not soft positions or Source targets.
    Unready/terminal/full-wall cases preserve the legacy positions and have
    finite zero logit gradients, even if a row has no legal categorical cell.
    """
    if (not torch.is_tensor(previous) or previous.ndim != 2 or previous.shape[-1] != 90
            or not previous.is_floating_point()):
        raise ValueError('Spawn previous state must be floating point[B,90]')
    batch = len(previous)
    if (not torch.is_tensor(logits) or logits.shape != (batch, 8, 256)
            or not logits.is_floating_point() or logits.device != previous.device):
        raise ValueError('Spawn logits must be colocated floating point[B,8,256]')
    if (not math.isfinite(dt) or dt <= 0 or not bool(torch.isfinite(previous.detach()).all())
            or not bool(torch.isfinite(logits.detach()).all())):
        raise ValueError('Spawn inputs and positive timestep must be finite')
    previous = previous.detach()
    walkable = _walkable_batch(walkable, batch, previous.device)
    hard_positions, available = decode_spawn_positions(logits, previous, walkable, dt, deterministic=False)
    old = previous[:, 8:88].reshape(batch, 8, 10)
    alive = (old[:, :, 5] >= .5) & (old[:, :, 4] > 0)
    ready = (~alive & (old[:, :, 8]*DEATH_DURATION+dt >= RESPAWN_DELAY-1e-6)
             & (previous[:, 4:5] > 0))
    axis = (torch.arange(16, device=previous.device, dtype=previous.dtype)+.5)/MAP_EXTENT
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    centers = torch.stack([xx, yy], -1).reshape(256, 2)
    base_allowed = walkable.flatten(1) & (((centers[None]-previous[:, None, :2])*MAP_EXTENT)
                                         .square().sum(-1) >= 4.**2)
    accepted_positions = old[:, :, :2].clone()
    results = []
    for slot in range(8):
        distances = ((centers[None, :, None]-accepted_positions[:, None])*MAP_EXTENT).square().sum(-1)
        allowed = base_allowed & ~((distances < 1.) & alive[:, None]).any(-1)
        usable = ready[:, slot] & allowed.any(-1)
        # Empty softmax rows receive a dummy legal cell whose entire derivative
        # is then masked out. No all-minus-infinity softmax/NaN is constructed.
        safe = allowed.clone()
        safe[:, 0] |= ~allowed.any(-1)
        scores = logits[:, slot].double() if logits.dtype == torch.float64 else logits[:, slot].float()
        probability = scores.masked_fill(~safe, -torch.inf).softmax(-1)
        soft = (probability[:, :, None]*centers.to(probability.dtype)[None]).sum(1)
        soft = soft.to(previous.dtype)*usable[:, None].to(previous.dtype)
        results.append(hard_positions[:, slot]+(soft-soft.detach()))
        # Match exact ID-order acceptance, including each simultaneous spawn.
        accepted_positions[:, slot] = hard_positions[:, slot]
        alive[:, slot] |= usable
    return torch.stack(results, 1), available


def _payload_tensors(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get('contexts'), dict):
        raise ValueError('Spawn image payload requires captured contexts')
    context = payload['contexts']
    names = ('features', 'target_states', 'target_frames', 'object_masks', 'map_features')
    if any(name not in payload for name in names) or any(name not in context for name in ('state', 'pooled', 'spatial')):
        raise ValueError('Spawn image payload is missing aligned tensors')
    values = {name: payload[name] for name in names}
    values.update({'context_'+name: context[name] for name in ('state', 'pooled', 'spatial')})
    if any(not torch.is_tensor(value) or not value.is_floating_point() for value in values.values()):
        raise ValueError('Spawn image payload values must be floating point tensors')
    previous = values['context_state']
    if previous.ndim != 3 or previous.shape[-1] != 90 or min(previous.shape[:2]) < 1:
        raise ValueError('Spawn context states require shape[B,H,90]')
    batch, horizon = previous.shape[:2]
    pooled, frames, maps = (values[name] for name in ('context_pooled', 'target_frames', 'map_features'))
    if (pooled.ndim != 3 or pooled.shape[:2] != (batch, horizon) or pooled.shape[-1] < 1
            or values['context_spatial'].shape != (batch, horizon, 9, pooled.shape[-1])
            or values['features'].shape != (batch, horizon, 128)
            or values['target_states'].shape != previous.shape
            or frames.ndim != 5 or frames.shape[:3] != (batch, horizon, 3)
            or frames.shape[-1] != frames.shape[-2] or frames.shape[-1] < 1
            or maps.ndim != 4 or maps.shape[0] != batch or maps.shape[1] < 1
            or maps.shape[-2:] != (16, 16)):
        raise ValueError('Spawn image tensors have inconsistent batch/horizon/geometry shapes')
    mask = values['object_masks']
    if mask.shape == (batch, horizon, 1, *frames.shape[-2:]):
        mask = mask[:, :, 0]
    if mask.shape != (batch, horizon, *frames.shape[-2:]):
        raise ValueError('Spawn image masks must match native target images')
    values['object_masks'] = mask
    if (any(value.device != previous.device or not bool(torch.isfinite(value.detach()).all()) for value in values.values())
            or bool(((mask.detach() < 0) | (mask.detach() > 1)).any())):
        raise ValueError('Spawn image tensors must be finite and colocated with valid masks')
    return {name: value.detach() for name, value in values.items()}


def _spawn_zero(backend):
    parameters = [parameter for name, parameter in backend.named_parameters()
                  if name.startswith(('spawn_head.', 'spatial_spawn_'))]
    if not parameters:
        raise ValueError('Spawn image supervision requires trainable spawn heads')
    return sum((parameter.sum()*0 for parameter in parameters), parameters[0].new_zeros(()))


def spawn_image_objective(model, payload):
    """Return .25 weighted RGB L1+.1 spawn XY loss and JSON diagnostics.

    Select one Source respawn offset per image sample: most simultaneous new
    slots, with latest offset as the tie break. Only spawn heads and Base/Flow
    are recomputed/trained. All context/map/geometry inputs are detached, and
    this helper performs no backward, optimizer operation or runtime reset.
    """
    values = _payload_tensors(payload)
    previous, target = values['context_state'], values['target_states']
    batch, horizon = previous.shape[:2]
    old, future = (value[:, :, 8:88].reshape(batch, horizon, 8, 10) for value in (previous, target))
    old_alive = (old[..., 5] >= .5) & (old[..., 4] > 0)
    new_alive = (future[..., 5] >= .5) & (future[..., 4] > 0)
    revived = ~old_alive & new_alive
    counts = revived.sum(-1)
    score = counts*(horizon+1)+torch.arange(horizon, device=previous.device)[None]
    offsets = score.masked_fill(counts == 0, -1).argmax(-1)
    selected_samples = (counts > 0).any(-1)
    offsets = torch.where(selected_samples, offsets, torch.full_like(offsets, -1))
    metrics = {'spawn_image_selected_offsets': offsets.cpu().tolist(),
        'spawn_image_selected_samples': int(selected_samples.sum()),
        'spawn_image_skipped_samples': int((~selected_samples).sum()),
        'spawn_image_source_labels_selection_only': True, 'spawn_image_future_feedback': False,
        'spawn_image_loss': 0., 'spawn_image_rgb_l1': 0., 'spawn_image_coordinate_loss': 0.,
        'spawn_image_recorded_slots': 0, 'spawn_image_available_slots': 0}
    backend = model.vjepa_backend
    if not bool(selected_samples.any()):
        return _spawn_zero(backend), metrics
    if not bool(getattr(model.config, 'endless', False)):
        raise ValueError('Spawn image objective requires endless mode')
    hz = float(getattr(model.config, 'simulation_hz', 30))
    if not math.isfinite(hz) or hz <= 0:
        raise ValueError('Spawn image simulation frequency must be finite and positive')
    dt = 1/hz
    rows = torch.arange(batch, device=previous.device)[selected_samples]
    steps = offsets[selected_samples]
    state, pose = previous[rows, steps], target[rows, steps]
    maps = values['map_features'][rows]
    if 'walkable' in payload:
        walkable = _walkable_batch(payload['walkable'], batch, previous.device)
    else:
        renderer = getattr(backend, 'renderer', None)
        walkable = _walkable_batch(getattr(renderer, '_spawn_walkable', None), batch, previous.device)
    logits = backend.spawn_logits(state, values['context_pooled'][rows, steps],
                                  values['context_spatial'][rows, steps], maps)
    positions, available = spawn_positions_with_grad(logits, state, walkable[rows], dt)
    changed = revived[rows, steps]
    old_selected = state[:, 8:88].reshape(-1, 8, 10)
    target_selected = pose[:, 8:88].reshape_as(old_selected)
    xy = torch.where(changed[:, :, None], positions, target_selected[:, :, :2])
    # No Source resurrection when the own decoder cannot legally place a slot.
    corpse = old_selected.clone()
    corpse[:, :, 4:6] = 0
    corpse[:, :, 8] = (corpse[:, :, 8]+dt/DEATH_DURATION).clamp_max(RESPAWN_DELAY/DEATH_DURATION)
    fields = torch.where((changed & ~available)[:, :, None], corpse[:, :, 2:], target_selected[:, :, 2:])
    entities = torch.cat([xy, fields], -1)
    image_state = torch.cat([pose[:, :8], entities.flatten(1), pose[:, 88:]], -1)
    features = values['features'][rows, steps]
    base = model.render_base(image_state, features, maps)
    flow_options = {}
    if getattr(model.flow_renderer, 'uses_wall_conditioning', False):
        if 'map_grid' not in payload:
            raise ValueError('Wall-conditioned spawn images require the explicit source initialization map')
        flow_options['map_condition'] = payload['map_grid'].detach()[rows]
    prediction = model.flow_renderer.sample(base, image_state, checkpoint_velocity=True, **flow_options)
    mask = values['object_masks'][rows, steps]
    rgb = weighted_image_l1(prediction, values['target_frames'][rows, steps], object_region_weights(mask, boost=4.))
    coordinate_mask = changed & available
    error = F.smooth_l1_loss(positions*MAP_EXTENT, target_selected[:, :, :2]*MAP_EXTENT, reduction='none')
    coordinates = (error*coordinate_mask[:, :, None]).sum()/(2*coordinate_mask.sum()).clamp_min(1)
    total = .25*rgb+.1*coordinates
    metrics.update(spawn_image_loss=float(total.detach()), spawn_image_rgb_l1=float(rgb.detach()),
        spawn_image_coordinate_loss=float(coordinates.detach()), spawn_image_recorded_slots=int(changed.sum()),
        spawn_image_available_slots=int(coordinate_mask.sum()))
    return total, metrics
