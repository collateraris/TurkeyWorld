"""Training-only depth and object-visibility targets for the reference game.

This module is deliberately separate from the neural runtime.  Its DDA queries
are a supervision oracle; the deployed camera/Transformer model must predict
occlusion instead of calling this code every frame.

Visibility means that an NPC's *centre point* lies in the camera frustum and is
not hidden by a wall.  It does not account for sprite alpha, partial visibility,
the weapon overlay, or occlusion by another NPC.  Dead slots are included because
their corpses remain visible in the game.
"""
from __future__ import annotations

from .engine import GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS, STATE_DIM

import math
import numpy as np


def _wall_depths(tilemap, x, y, direction_x, direction_y, far):
    """First wall depth along each unnormalised camera ray.

    The ray is p + t * direction.  For direction = forward + u * right,
    t is camera-forward depth, matching ``Game.render`` rather than Euclidean
    distance along an oblique ray.  Exiting an unclosed map counts as a wall.
    """
    height, width = tilemap.shape
    direction_x = np.asarray(direction_x, dtype=np.float64)
    direction_y = np.asarray(direction_y, dtype=np.float64)
    shape = direction_x.shape
    dx, dy = direction_x.reshape(-1), direction_y.reshape(-1)
    count = len(dx)
    mx = np.full(count, math.floor(x), dtype=np.int64)
    my = np.full(count, math.floor(y), dtype=np.int64)
    step_x, step_y = np.where(dx >= 0, 1, -1), np.where(dy >= 0, 1, -1)
    inv_x = np.full(count, np.inf)
    inv_y = np.full(count, np.inf)
    np.divide(1.0, np.abs(dx), out=inv_x, where=np.abs(dx) > 1e-12)
    np.divide(1.0, np.abs(dy), out=inv_y, where=np.abs(dy) > 1e-12)
    tx = np.where(dx >= 0, mx + 1 - x, x - mx) * inv_x
    ty = np.where(dy >= 0, my + 1 - y, y - my) * inv_y
    result = np.full(count, far, dtype=np.float64)
    active = np.ones(count, dtype=bool)

    # A straight ray visits no more than width + height cells before exiting.
    for _ in range(width + height + 4):
        if not active.any():
            break
        on_x = (tx < ty) & active
        on_y = (~on_x) & active
        distance = np.where(on_x, tx, ty)
        mx[on_x] += step_x[on_x]
        my[on_y] += step_y[on_y]
        tx[on_x] += inv_x[on_x]
        ty[on_y] += inv_y[on_y]
        inside = (mx >= 0) & (mx < width) & (my >= 0) & (my < height)
        wall = np.ones(count, dtype=bool)
        wall[inside] = tilemap[my[inside], mx[inside]] != 0
        hit = active & wall
        result[hit] = np.minimum(distance[hit], far)
        active &= ~wall & (distance < far)
    return result.reshape(shape)


def build_geometry_targets(tilemap, state_vector, rays=32, max_depth=24.0,
                           fov_degrees=68.0):
    """Return oracle ``depth[R]``, ``visibility[8]`` and ``entity_forward[8]``.

    Coordinates in the 90D state are normalised by map width/height.  Heading is
    stored as sin/cos at indices 2/3; a zero heading falls back to looking right.
    Depth is clipped to [0.02, max_depth].  Invalid camera geometry (outside the
    map or inside a wall) produces minimum depths and zero visibility, allowing
    diagnostics on model-visited states without unsafe array indexing.
    """
    if isinstance(rays, bool) or int(rays) != rays or rays <= 0:
        raise ValueError("rays must be a positive integer")
    rays = int(rays)
    max_depth = float(max_depth)
    fov_degrees = float(fov_degrees)
    if not math.isfinite(max_depth) or max_depth < .02:
        raise ValueError("max_depth must be finite and at least 0.02")
    if not math.isfinite(fov_degrees) or not 0 < fov_degrees < 180:
        raise ValueError("fov_degrees must be finite and lie in (0,180)")
    tilemap = np.asarray(tilemap)
    if tilemap.ndim != 2 or min(tilemap.shape) < 1:
        raise ValueError("tilemap must be a non-empty 2D array")
    if not np.issubdtype(tilemap.dtype, np.number) and tilemap.dtype != bool:
        raise ValueError("tilemap must contain numeric tile IDs")
    if not np.isfinite(tilemap).all():
        raise ValueError("tilemap must contain finite tile IDs")
    state = np.asarray(state_vector, dtype=np.float64)
    if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
        raise ValueError("state_vector must contain exactly STATE_DIM finite values")

    height, width = tilemap.shape
    x, y = state[:2] * (width, height)
    sine, cosine = state[2:4]
    heading_norm = math.hypot(sine, cosine)
    if heading_norm > 1e-12:
        sine, cosine = sine / heading_norm, cosine / heading_norm
    else:
        sine, cosine = 0.0, 1.0
    entities = state[GLOBAL_DIM:-2].reshape(ENTITY_SLOTS, ENTITY_DIM)
    delta = entities[:, :2] * (width, height) - (x, y)
    forward = delta[:, 0] * cosine + delta[:, 1] * sine
    right = -delta[:, 0] * sine + delta[:, 1] * cosine
    result = {
        "depth": np.full(rays, .02, dtype=np.float32),
        "visibility": np.zeros(ENTITY_SLOTS, dtype=np.float32),
        "entity_forward": forward.astype(np.float32),
    }
    if not (0 <= x < width and 0 <= y < height):
        return result
    if tilemap[math.floor(y), math.floor(x)] != 0:
        return result

    tangent = math.tan(math.radians(fov_degrees) / 2)
    screen_x = (np.arange(rays, dtype=np.float64) + .5) / rays * 2 - 1
    ray_x = cosine - sine * tangent * screen_x
    ray_y = sine + cosine * tangent * screen_x
    depths = _wall_depths(tilemap, x, y, ray_x, ray_y, max_depth)
    result["depth"] = np.clip(depths, .02, max_depth).astype(np.float32)

    safe_forward = np.where(np.abs(forward) > 1e-12, forward, 1e-12)
    centre_screen_x = right / (safe_forward * tangent)
    object_x, object_y = entities[:, 0] * width, entities[:, 1] * height
    inside = ((object_x >= 0) & (object_x < width) &
              (object_y >= 0) & (object_y < height))
    free = np.zeros(ENTITY_SLOTS, dtype=bool)
    free[inside] = tilemap[object_y[inside].astype(int), object_x[inside].astype(int)] == 0
    candidate = (forward > .12) & (np.abs(centre_screen_x) <= 1) & free

    # Each object gets its exact centre ray; low-resolution depth interpolation
    # would produce incorrect labels close to a wall edge.
    query_far = max(max_depth, float(np.maximum(forward, 0).max()) + .01)
    centre_depth = _wall_depths(tilemap, x, y,
                                delta[:, 0] / safe_forward,
                                delta[:, 1] / safe_forward, query_far)
    result["visibility"] = (candidate & (centre_depth > forward + 1e-6)).astype(np.float32)
    return result


def geometry_targets(game, **kwargs):
    """Collection-only convenience wrapper; never invokes ``game.render``."""
    return build_geometry_targets(game.map, game.state_vector(), **kwargs)
