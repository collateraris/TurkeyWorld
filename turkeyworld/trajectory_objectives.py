"""Loss-only spatial supervision of a fully predicted trajectory.

No simulator, projection, state repair or future-label model input is used.
The initialization map is static observed data. Circle clearance is a
piecewise differentiable training constraint; inference remains neural.
"""
from __future__ import annotations

from .engine import GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS, STATE_DIM

import torch
from torch.nn import functional as F

XY_WEIGHT = 1.
WALL_WEIGHT = .25
TRAJECTORY_OFFSETS = (1, 4, 8, 16)


def circle_wall_penalty(points, tilemap, radii):
    """Squared circle/tile/boundary overlap at continuous map-unit centers.

    points[B,...,2], raw tilemap[B,16,16], radius scalar or broadcastable to
    points[...,0]. Floor is0. Zero means every circle clears every wall and
    the arena boundary. Inside a wall, rectangle signed distances retain a
    useful gradient; an exact center uses a deterministic valid subgradient.
    """
    if points.ndim < 3 or points.shape[-1] != 2 or not points.is_floating_point():
        raise ValueError('Circle centers must be floating point[B,...,2]')
    if tilemap.shape != (points.shape[0], 16, 16):
        raise ValueError('Circle loss requires the matching raw map[B,16,16]')
    value = points.float()
    flat = value.reshape(len(value), -1, 2)
    axis = torch.arange(16, device=value.device, dtype=value.dtype)+.5
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    centers = torch.stack([x, y], -1).reshape(256, 2)
    delta = flat[:, :, None]-centers[None, None]
    # torch.abs has zero derivative at0, while the right-sided distance is a
    # valid subgradient and avoids stranding an exact wall-center prediction.
    q = torch.where(delta >= 0, delta, -delta)-.5
    signed = q.clamp_min(0).norm(dim=-1)+q.amax(-1).clamp_max(0)
    walls = tilemap.detach().to(value.device).ne(0).flatten(1)
    wall_clearance = signed.masked_fill(~walls[:, None], torch.inf).amin(-1)
    boundary = torch.cat([flat, 16-flat], -1).amin(-1)
    center_clearance = torch.minimum(wall_clearance, boundary).reshape(value.shape[:-1])
    radius = torch.as_tensor(radii, device=value.device, dtype=value.dtype)
    if not bool(torch.isfinite(radius).all()) or bool((radius < 0).any()):
        raise ValueError('Circle radius must be finite and nonnegative')
    return (radius-center_clearance).clamp_min(0).square()


def _masked_mean(values, mask, coordinate_dims=1):
    return (values*mask.to(values.dtype)).sum()/(mask.sum()*coordinate_dims).clamp_min(1)


def trajectory_state_losses(predictions, initial, targets, tilemap, *, offsets=TRAJECTORY_OFFSETS,
                            xy_weight=XY_WEIGHT, wall_weight=WALL_WEIGHT, segment_samples=5,
                            previous_prediction=None):
    """Return(additive loss parts, detached metrics) in tiles at selected offsets.

    Primary NPC XY masks depend on Source life only; false-missing predicted
    NPCs cannot disappear from the objective. A Source respawn starts a new
    life and excludes that ID from paired XY thereafter. Collision endpoints
    use currently Source-alive actors, and segment checks exclude Source
    respawn teleports. Neither loss ever modifies predictions or memory.
    """
    if (predictions.ndim != 3 or predictions.shape[-1] != STATE_DIM or targets.shape != predictions.shape
            or initial.shape != (predictions.shape[0], STATE_DIM) or predictions.shape[1] < 1
            or not isinstance(segment_samples, int) or segment_samples < 2):
        raise ValueError('Expected predictions/targets[B,H,STATE_DIM], initial[B,STATE_DIM] and segment_samples>=2')
    if xy_weight < 0 or wall_weight < 0:
        raise ValueError('Trajectory weights must be nonnegative')
    initial, targets = initial.detach().float(), targets.detach().float()
    predicted = predictions.float()
    batch, horizon = predicted.shape[:2]
    selected = sorted({int(offset)-1 for offset in offsets if 1 <= int(offset) <= horizon})
    if not selected:
        raise ValueError('At least one trajectory offset must fit the predicted horizon')
    indices = torch.as_tensor(selected, device=predicted.device, dtype=torch.long)
    old_source = torch.cat([initial[:, None], targets[:, :-1]], 1)
    source_entities = targets[:, :, GLOBAL_DIM:-2].reshape(batch, horizon, ENTITY_SLOTS, ENTITY_DIM)
    previous_entities = old_source[:, :, GLOBAL_DIM:-2].reshape_as(source_entities)
    initial_entities = initial[:, GLOBAL_DIM:-2].reshape(batch, ENTITY_SLOTS, ENTITY_DIM)
    source_alive = (source_entities[..., 5] >= .5) & (source_entities[..., 4] > 0)
    previous_alive = (previous_entities[..., 5] >= .5) & (previous_entities[..., 4] > 0)
    initial_alive = (initial_entities[..., 5] >= .5) & (initial_entities[..., 4] > 0)
    respawn = ~previous_alive & source_alive
    stable = initial_alive[:, None] & source_alive & (respawn.long().cumsum(1) == 0)
    predicted_entities = predicted[:, :, GLOBAL_DIM:-2].reshape_as(source_entities)
    player_mask = (targets[:, indices, 4] > 0) & (initial[:, None, 4] > 0)
    npc_mask = stable[:, indices]
    player_error = F.smooth_l1_loss(predicted[:, indices, :2]*16, targets[:, indices, :2]*16, reduction='none')
    npc_error = F.smooth_l1_loss(predicted_entities[:, indices, :, :2]*16,
                                source_entities[:, indices, :, :2]*16, reduction='none')
    parts = {'ar_xy_player': xy_weight*_masked_mean(player_error.sum(-1), player_mask, 2),
             'ar_xy_npc': xy_weight*_masked_mean(npc_error.sum(-1), npc_mask, 2)}
    current_points = torch.cat([predicted[:, :, None, :2], predicted_entities[..., :2]], 2)*16
    # TBPTT chunks carry their own numerical predecessor, even when it has
    # drifted from the Source state used only for labels and lifetime masks.
    own_previous = initial if previous_prediction is None else previous_prediction.detach().float()
    if own_previous.shape != initial.shape:
        raise ValueError('Own preceding prediction must match initial state shape')
    old_prediction = torch.cat([own_previous[:, None], predicted[:, :-1]], 1)
    old_npc = old_prediction[:, :, GLOBAL_DIM:-2].reshape_as(source_entities)
    old_points = torch.cat([old_prediction[:, :, None, :2], old_npc[..., :2]], 2)*16
    radii = predicted.new_tensor([.20]+[.25]*ENTITY_SLOTS)
    alive_mask = torch.cat([(targets[:, :, 4] > 0)[:, :, None], source_alive], -1)
    previous_mask = torch.cat([(old_source[:, :, 4] > 0)[:, :, None], previous_alive], -1)
    endpoint_error = circle_wall_penalty(current_points, tilemap, radii)
    # Source movement applies X then Y; a straight chord can cut a wall corner
    # even when both physical movement legs are legal. Include the elbow and
    # interior samples of both legs, not an invented diagonal traversal.
    elbow = torch.stack([current_points[..., 0], old_points[..., 1]], -1)
    fractions = torch.linspace(0, 1, segment_samples, device=predicted.device, dtype=predicted.dtype)[1:-1]
    first_leg = old_points[:, :, None]+fractions[None, None, :, None, None]*(elbow-old_points)[:, :, None]
    second_leg = elbow[:, :, None]+fractions[None, None, :, None, None]*(current_points-elbow)[:, :, None]
    segments = torch.cat([first_leg, elbow[:, :, None], second_leg], 2)
    segment_error = circle_wall_penalty(segments, tilemap, radii).sum(2)
    segment_mask = alive_mask & previous_mask
    combined = endpoint_error*alive_mask+segment_error*segment_mask
    counts = alive_mask.to(predicted.dtype)+segments.shape[2]*segment_mask
    parts['ar_wall_player'] = wall_weight*combined[:, :, 0].sum()/counts[:, :, 0].sum().clamp_min(1)
    parts['ar_wall_npc'] = wall_weight*combined[:, :, 1:].sum()/counts[:, :, 1:].sum().clamp_min(1)
    with torch.no_grad():
        distances = (predicted_entities[:, indices, :, :2]-source_entities[:, indices, :, :2]).norm(dim=-1)*16
        metrics = {'ar_xy_offsets': [index+1 for index in selected],
            'ar_xy_player_pairs': int(player_mask.sum()), 'ar_xy_npc_pairs': int(npc_mask.sum()),
            'ar_xy_npc_mean_tiles': float(_masked_mean(distances, npc_mask)),
            'ar_source_respawn_exclusions': int(respawn.sum()),
            'ar_wall_player_overlaps': int(((endpoint_error[:, :, 0] > 1e-14) & alive_mask[:, :, 0]).sum()),
            'ar_wall_npc_overlaps': int(((endpoint_error[:, :, 1:] > 1e-14) & alive_mask[:, :, 1:]).sum()),
            'ar_wall_segment_samples': segment_samples, 'ar_wall_path': 'x_then_y',
            'ar_wall_samples_per_transition': int(segments.shape[2]+1), 'ar_xy_weight': float(xy_weight),
            'ar_wall_weight': float(wall_weight), 'ar_geometry_future_feedback': False}
    return parts, metrics
