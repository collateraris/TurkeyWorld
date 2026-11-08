"""Differentiable camera coordinates and camera-relative map reads.

This module performs coordinate transforms and samples *learned map features*.
It does not run the reference renderer, trace collision rays, or decide wall
occlusion. Frustum membership means only that an object's center projects into
the horizontal field of view; wall visibility must be learned separately.
"""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F

FOV_DEGREES = 68.0
ENTITY_SLOTS = 8
ENTITY_DIM = 10
STATE_DIM = 90
OBJECT_CAMERA_DIM = 18


def _check_state(state):
    if state.ndim != 2 or state.shape[-1] != STATE_DIM:
        raise ValueError(f"state must be [B,{STATE_DIM}]")
    if not state.is_floating_point():
        raise TypeError("state must be a floating point tensor")


def _map_dimensions(map_size):
    if isinstance(map_size, int):
        height = width = map_size
    else:
        height, width = map_size
    if height <= 0 or width <= 0:
        raise ValueError("map dimensions must be positive")
    return float(height), float(width)


def camera_basis(state):
    """Return world-space forward/right vectors, each [B,2].

    State stores sin(angle), cos(angle) at indices 2 and 3. Positive rotation
    follows the reference game's y-down coordinates: yaw=0 faces +x; yaw=pi/2
    faces +y. A zero-length heading falls back to +x without producing NaNs.
    """
    _check_state(state)
    heading = torch.stack([state[:, 3], state[:, 2]], dim=-1)
    length = heading.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(heading)
    fallback[:, 0] = 1.0
    forward = torch.where(length > 1e-6, heading / length.clamp_min(1e-6), fallback)
    right = torch.stack([-forward[:, 1], forward[:, 0]], dim=-1)
    return forward, right


def sample_camera_frustum(map_features, state, rays=32, samples=64,
                          max_depth=24.0, fov_degrees=FOV_DEGREES):
    """Read map features along a perspective camera frustum.

    Args:
        map_features: [B,C,H,W], one spatial feature cell per world-map tile.
        state: [B,90], with normalized player x/W, y/H and sin/cos heading.
        rays: number of horizontal screen-column samples.
        samples: uniformly spaced forward-depth bins on each camera ray.

    Returns a dictionary containing:
        sampled: [B,R,K,C], differentiably sampled learned map features.
        forward_depths: [K], depth-bin centers in world tile units.
        screen_x: [R], pixel-center camera-plane x coordinates in [-1,1].
        in_bounds: [B,R,K], whether each position lies inside the map extent.
        world_xy / sampling_grid: [B,R,K,2], physical / normalized positions.
        directions: [B,R,2], rays parameterized by FORWARD depth, not unit range.

    Ray p(d) = camera_position + d*(forward + tan(FOV/2)*screen_x*right).
    Thus d matches the reference raycaster's perpendicular wall depth. With
    align_corners=False, tile-center world x=i+.5 maps to exact feature i.
    Out-of-map samples are zero. No wall-intersection oracle is called.
    """
    _check_state(state)
    if map_features.ndim != 4 or map_features.shape[0] != state.shape[0]:
        raise ValueError("map_features must be [B,C,H,W] with the same batch as state")
    if map_features.device != state.device or map_features.dtype != state.dtype:
        raise ValueError("map_features and state must share dtype and device")
    if not isinstance(rays, int) or not isinstance(samples, int) or min(rays, samples) < 1:
        raise ValueError("rays and samples must be positive integers")
    if not math.isfinite(max_depth) or max_depth <= 0:
        raise ValueError("max_depth must be finite and positive")
    if not math.isfinite(fov_degrees) or not 0 < fov_degrees < 179:
        raise ValueError("fov_degrees must be finite and in (0,179)")

    height, width = map_features.shape[-2:]
    forward, right = camera_basis(state)
    screen_x = (torch.arange(rays, device=state.device, dtype=state.dtype) + .5) * (2.0 / rays) - 1.0
    depths = (torch.arange(samples, device=state.device, dtype=state.dtype) + .5) * (max_depth / samples)
    tangent = math.tan(math.radians(fov_degrees) * .5)
    directions = forward[:, None, :] + tangent * screen_x[None, :, None] * right[:, None, :]
    scale = state.new_tensor([width, height])
    position = state[:, :2] * scale
    world_xy = position[:, None, None, :] + directions[:, :, None, :] * depths[None, None, :, None]
    grid = 2.0 * world_xy / scale - 1.0
    inside = ((world_xy[..., 0] >= 0) & (world_xy[..., 0] < width) &
              (world_xy[..., 1] >= 0) & (world_xy[..., 1] < height))
    sampled = F.grid_sample(map_features, grid, mode="bilinear", padding_mode="zeros",
                            align_corners=False).permute(0, 2, 3, 1)
    sampled = sampled * inside[..., None].to(dtype=sampled.dtype)
    return {"sampled": sampled, "forward_depths": depths, "screen_x": screen_x,
            "in_bounds": inside, "world_xy": world_xy, "sampling_grid": grid,
            "directions": directions}


def object_camera_features(state, map_size=16, max_depth=24.0,
                           fov_degrees=FOV_DEGREES):
    """Represent all eight persistent object slots relative to the camera.

    features[B,8,18] has the following fixed layout:
      forward/max_depth, right/max_depth, distance/max_depth,
      bearing_sin, bearing_cos, clipped_screen_x/4,
      clipped_projected_radius/4, center_frustum,
      camera_velocity_forward/2, camera_velocity_right/2,
      hp, alive, attack_cooldown, hurt_timer, death_timer, animation_phase,
      original_normalized_x, original_normalized_y.

    HP/alive/timers use the existing normalized state representation. A dead
    entity is NOT removed or masked: its corpse can remain visible. frustum
    describes centerline projection only, with no wall or inter-object test.
    """
    _check_state(state)
    height, width = _map_dimensions(map_size)
    if not math.isfinite(max_depth) or max_depth <= 0:
        raise ValueError("max_depth must be finite and positive")
    if not math.isfinite(fov_degrees) or not 0 < fov_degrees < 179:
        raise ValueError("fov_degrees must be finite and in (0,179)")
    entities = state[:, 8:88].reshape(-1, ENTITY_SLOTS, ENTITY_DIM)
    forward_basis, right_basis = camera_basis(state)
    scale = state.new_tensor([width, height])
    delta = (entities[..., :2] - state[:, None, :2]) * scale
    forward = (delta * forward_basis[:, None, :]).sum(-1)
    right = (delta * right_basis[:, None, :]).sum(-1)
    distance = delta.norm(dim=-1)
    safe_distance = distance.clamp_min(1e-6)
    tangent = math.tan(math.radians(fov_degrees) * .5)
    safe_forward = torch.where(forward.abs() > 1e-6, forward, torch.full_like(forward, 1e-6))
    screen_x = right / (safe_forward * tangent)
    projected_radius = .25 / (forward.clamp_min(.12) * tangent)
    frustum = (forward > .12) & (screen_x.abs() <= 1)
    velocity = entities[..., 2:4]  # Existing values already represent velocity/2.
    camera_v_forward = (velocity * forward_basis[:, None, :]).sum(-1)
    camera_v_right = (velocity * right_basis[:, None, :]).sum(-1)
    relative = torch.stack([
        forward / max_depth, right / max_depth, distance / max_depth,
        right / safe_distance, forward / safe_distance,
        screen_x.clamp(-4, 4) / 4, projected_radius.clamp(0, 4) / 4,
        frustum.to(state.dtype), camera_v_forward, camera_v_right,
    ], dim=-1)
    features = torch.cat([relative, entities[..., 4:10], entities[..., :2]], dim=-1)
    return {"features": features, "entities": entities, "delta_world": delta,
            "forward": forward, "right": right, "distance": distance,
            "screen_x": screen_x, "frustum": frustum,
            "projected_radius": projected_radius,
            "camera_velocity_forward": camera_v_forward,
            "camera_velocity_right": camera_v_right}
