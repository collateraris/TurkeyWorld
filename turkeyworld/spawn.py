"""Decode learned spawn distributions using the disclosed endless spawn rules.

The initialization map supplies allowed floor-tile centers. This module does
not import the reference engine or query its collision, navigation or hit code.
The network chooses among legal distant, separated centers; it does not select
arbitrary floating-point positions and then ask the engine to repair them.
"""
from __future__ import annotations

import torch

from .structured_state import DEATH_DURATION, ENTITY_SLOTS, MAP_EXTENT, RESPAWN_DELAY


@torch.no_grad()
def decode_spawn_positions(logits, previous, walkable, dt):
    """Return normalized centers [B,8,2] and availability [B,8].

    Selecting a legal tile is a declared discrete game rule. Relative preference
    between legal tiles comes from the learned categorical logits. Different
    dead slots are decoded in ID order so newly selected centers stay apart.
    """
    batch = previous.shape[0]
    if logits.shape != (batch, ENTITY_SLOTS, 256):
        raise ValueError("spawn logits must have shape [B,8,256]")
    if walkable.ndim == 2:
        walkable = walkable[None]
    if walkable.shape[0] == 1 and batch != 1:
        walkable = walkable.expand(batch, -1, -1)
    if walkable.shape != (batch, 16, 16):
        raise ValueError("encode the matching initialization map before spawning")
    old = previous[:, 8:-2].reshape(batch, ENTITY_SLOTS, 10)
    alive = (old[:, :, 5] >= .5) & (old[:, :, 4] > 0)
    ready = (~alive & (old[:, :, 8] * DEATH_DURATION + dt >= RESPAWN_DELAY - 1e-6)
             & (previous[:, 4:5] > 0))
    positions = old[:, :, :2].clone()
    available = torch.zeros_like(alive)
    if not bool(ready.any()):
        return positions, available
    axis = (torch.arange(16, device=previous.device, dtype=previous.dtype) + .5) / MAP_EXTENT
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    centers = torch.stack([xx, yy], dim=-1).reshape(256, 2)
    floor = walkable.to(device=previous.device, dtype=torch.bool).flatten(1)
    player_distance = ((centers[None] - previous[:, None, :2]) * MAP_EXTENT).square().sum(-1)
    base_allowed = floor & (player_distance >= 4.0 ** 2)
    for slot in range(ENTITY_SLOTS):
        distances = ((centers[None, :, None] - positions[:, None]) * MAP_EXTENT).square().sum(-1)
        separated = ~((distances < 1.0) & alive[:, None]).any(-1)
        allowed = base_allowed & separated
        usable = ready[:, slot] & allowed.any(-1)
        chosen = logits[:, slot].masked_fill(~allowed, -torch.finfo(logits.dtype).max).argmax(-1)
        positions[:, slot] = torch.where(usable[:, None], centers[chosen], positions[:, slot])
        alive[:, slot] |= usable
        available[:, slot] = usable
    return positions, available
