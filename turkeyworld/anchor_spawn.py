"""Learned anchor/fallback spawning with the existing legal-cell decoder.

No source time ranking or engine query is used. Geometry masks and ID-order
acceptance mirror spawn.decode_spawn_positions. All logits are finite so a
source label excluded by an earlier own simultaneous choice cannot create NaN.
"""
from __future__ import annotations

from .structured_state import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS

import math

import torch

from .spawn import SPAWN_ANCHORS
from .structured_state import DEATH_DURATION, MAP_EXTENT, RESPAWN_DELAY


def anchor_tile_indices(device=None):
    return torch.tensor([y*16+x for x, y in SPAWN_ANCHORS], device=device, dtype=torch.long)


def anchor_target_labels(positions):
    """Map normalized Source XY[...,2] to eight anchor classes or fallback8."""
    if not torch.is_tensor(positions) or positions.shape[-1] != 2:
        raise ValueError('Anchor targets require normalized XY[...,2]')
    tiles = (positions.detach()*MAP_EXTENT).long().clamp(0, 15)
    ids = tiles[..., 0]+16*tiles[..., 1]
    matches = ids[..., None] == anchor_tile_indices(positions.device)
    return torch.where(matches.any(-1), matches.long().argmax(-1), torch.full_like(ids, 8))


def hierarchical_spawn_logits(tile_logits, anchor_residual, previous, walkable, dt):
    """Return finite effective256 logits plus raw9 and raw fallback logits.

    The nine-way head adds to the old anchor scores and old legal fallback
    log-mass. A zero residual therefore preserves old legal per-cell rankings
    and sequential decoded positions. Fallback probability is distributed only
    over legal non-anchor cells. Raw9 scores retain blocked anchor classes for
    supervision; runtime scores apply all safety constraints.
    """
    if (not torch.is_tensor(previous) or previous.ndim != 2 or previous.shape[-1] != STATE_DIM
            or not previous.is_floating_point()):
        raise ValueError('Hierarchical spawn state must be floating point[B,20]')
    batch = len(previous)
    if (tile_logits.shape != (batch, ENTITY_SLOTS, 256) or anchor_residual.shape != (batch, ENTITY_SLOTS, 9)
            or tile_logits.device != previous.device or anchor_residual.device != previous.device
            or not tile_logits.is_floating_point() or not anchor_residual.is_floating_point()):
        raise ValueError('Hierarchical spawn requires colocated logits[B,1,256] and residual[B,1,9]')
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('Hierarchical spawn requires a positive finite timestep')
    if not torch.is_tensor(walkable):
        raise ValueError('Hierarchical spawn requires encode_map or an explicit walkable map')
    if walkable.ndim == 2:
        walkable = walkable[None]
    if walkable.shape == (1, 16, 16):
        walkable = walkable.expand(batch, -1, -1)
    if walkable.shape != (batch, 16, 16):
        raise ValueError('Hierarchical spawn walkable must match[B,16,16]')
    # Masks use only own known numerical state. Discrete accepted positions do
    # not acquire gradients or receive source pose/labels.
    state = previous.detach()
    dtype = torch.float64 if tile_logits.dtype == torch.float64 else torch.float32
    raw, residual = tile_logits.to(dtype), anchor_residual.to(dtype)
    indices = anchor_tile_indices(previous.device)
    is_anchor = torch.zeros(256, device=previous.device, dtype=torch.bool)
    is_anchor[indices] = True
    axis = (torch.arange(16, device=previous.device, dtype=state.dtype)+.5)/MAP_EXTENT
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    centers = torch.stack([xx, yy], -1).reshape(256, 2)
    old = state[:, GLOBAL_DIM:-2].reshape(batch, ENTITY_SLOTS, ENTITY_DIM)
    alive = (old[:, :, 5] >= .5) & (old[:, :, 4] > 0)
    ready = (~alive & (old[:, :, 8]*DEATH_DURATION+dt >= RESPAWN_DELAY-1e-6)
             & (state[:, 4:5] > 0))
    positions = old[:, :, :2].clone()
    base_allowed = walkable.to(device=state.device, dtype=torch.bool).flatten(1)
    base_allowed = base_allowed & (((centers[None]-state[:, None, :2])*MAP_EXTENT)
                                   .square().sum(-1) >= 4.**2)
    effective, categories, legal, category_legal = [], [], [], []
    negative = -1.e4
    for slot in range(ENTITY_SLOTS):
        distances = ((centers[None, :, None]-positions[:, None])*MAP_EXTENT).square().sum(-1)
        allowed = base_allowed & ~((distances < 1.) & alive[:, None]).any(-1)
        fallback = allowed & ~is_anchor[None]
        has_fallback = fallback.any(-1)
        # A dummy prevents logsumexp/softmax of an empty row. Its contribution
        # is subsequently removed from both the category and tile paths.
        safe_fallback = fallback.clone()
        safe_fallback[:, 0] |= ~has_fallback
        fallback_scores = raw[:, slot].masked_fill(~safe_fallback, -torch.inf)
        fallback_mass = fallback_scores.logsumexp(-1)
        fallback_mass = torch.where(has_fallback, fallback_mass, fallback_mass.new_full((), negative))
        coarse = torch.cat([raw[:, slot, indices], fallback_mass[:, None]], -1)+residual[:, slot]
        allowed_categories = torch.cat([allowed[:, indices], has_fallback[:, None]], -1)
        usable = ready[:, slot] & allowed.any(-1)
        safe_categories = allowed_categories.clone()
        safe_categories[:, 0] |= ~allowed_categories.any(-1)
        normalizer = coarse.masked_fill(~safe_categories, -torch.inf).logsumexp(-1, keepdim=True)
        # Algebraically cancel fallback log-mass against its conditional
        # normalization. Besides saving work, this preserves old exact ties
        # when the new residual is zero instead of introducing roundoff bias.
        scores = raw[:, slot]+residual[:, slot, 8:9]-normalizer
        scores = scores.masked_fill(~fallback, negative)
        # Direct assignment replaces fallback values only at the eight anchors.
        scores = scores.scatter(1, indices[None].expand(batch, -1),
                                (coarse[:, :8]-normalizer).masked_fill(~allowed[:, indices], negative))
        scores = torch.where(allowed.any(-1)[:, None], scores, raw[:, slot]*0)
        effective.append(scores)
        categories.append(coarse)
        legal.append(allowed)
        category_legal.append(allowed_categories)
        chosen = scores.detach().masked_fill(~allowed, -torch.finfo(scores.dtype).max).argmax(-1)
        accepted = torch.where(usable[:, None], centers[chosen], positions[:, slot])
        # Clone before writing: prior slot masks may be saved by autograd.
        positions = positions.clone()
        alive = alive.clone()
        positions[:, slot] = accepted
        alive[:, slot] |= usable
    return {'spawn_logits': torch.stack(effective, 1),
            'spawn_anchor_logits': torch.stack(categories, 1),
            'spawn_fallback_logits': raw,
            'spawn_legal_mask': torch.stack(legal, 1),
            'spawn_anchor_legal': torch.stack(category_legal, 1),
            'spawn_anchor_residual': residual}
