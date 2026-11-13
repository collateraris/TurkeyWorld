"""Shared deterministic source/neural spawn rules and legacy learned decoding.

The initialization map supplies allowed floor-tile centers. This module does
not import the reference engine or query its collision, navigation or hit code.
The deterministic policy cycles through eight fixed anchors every three game
seconds. Safety filters retain player/NPC separation and wall exclusion; a
static row-major floor order is the fallback. No hidden RNG or time hash is
used. The learned decoder remains the default for neural model architectures.
No engine collision/navigation/hit query is used.
"""
from __future__ import annotations

BEHAVIOR_VERSION = 'deterministic_enemy_v4_single_spawn_cycle'
SPAWN_TICK_HZ = 30
SPAWN_PHASE_SECONDS = 3
SPAWN_PHASE_TICKS = SPAWN_PHASE_SECONDS * SPAWN_TICK_HZ
# All eight cells are floor on the reference16x16 arena. Clockwise ordering
# gives neighboring time phases neighboring regions, without a hash avalanche.
SPAWN_ANCHORS = ((2, 2), (8, 2), (13, 2), (13, 8),
                 (13, 13), (6, 13), (2, 13), (2, 8))
SPAWN_CYCLE_SECONDS = SPAWN_PHASE_SECONDS * len(SPAWN_ANCHORS)


def spawn_tile_rank(tick, slot, tile):
    """Coarse cyclic anchor priority; accepts integers or torch int64 tensors.

    Anchor floor(game_seconds/3) modulo8 comes first, followed by the
    remaining anchors in ring order. Other cells follow in static row-major
    order. The schedule repeats every24seconds; walls/occupancy may skip an
    anchor but cannot introduce randomness.
    """
    count = len(SPAWN_ANCHORS)
    # The public slot argument is retained for source-call compatibility; the
    # game has one persistent ID and no identity/time hash in its schedule.
    preferred = (tick // SPAWN_PHASE_TICKS) % count
    rank = count + tile
    for index, (x, y) in enumerate(SPAWN_ANCHORS):
        priority = (index-preferred) % count
        rank = rank + (tile == y*16+x) * (priority-rank)
    return rank


def deterministic_spawn_tile(walkable, player_xy, positions, alive, slot, tick):
    """Choose a floor tile using map-unit coordinates; return (x,y) or None.

    This small dependency-free implementation is used by the source engine.
    The neural decoder applies the same rank and distance rules in tensors.
    """
    candidates = []
    for iy, row in enumerate(walkable):
        for ix, floor in enumerate(row):
            if not floor:
                continue
            x, y = ix + .5, iy + .5
            if (x-player_xy[0])**2 + (y-player_xy[1])**2 < 16.:
                continue
            if any(living and (x-xy[0])**2 + (y-xy[1])**2 < 1.
                   for xy, living in zip(positions, alive)):
                continue
            tile = iy * 16 + ix
            candidates.append((spawn_tile_rank(tick, slot, tile), tile, x, y))
    if not candidates:
        return None
    _, _, x, y = min(candidates)
    return x, y


def decode_spawn_positions(logits, previous, walkable, dt, *, deterministic=False):
    """Return normalized centers [B,1,2] and availability [B,1].

    With deterministic=True, learned logits are auxiliary labels only. The
    shared game-time rule controls placement. Slots are processed in ID order
    so simultaneous spawns remain separated. Defaults retain old checkpoints.
    """
    import torch
    from .structured_state import DEATH_DURATION, ENTITY_SLOTS, MAP_EXTENT, RESPAWN_DELAY
    from .simulation_clock import normalized_timer_ready

    # Selection is a discrete, disclosed rule in both modes, never a gradient
    # path. Detach once so source-only users need no PyTorch import at all.
    previous, logits = previous.detach(), logits.detach()
    batch = previous.shape[0]
    if logits.shape != (batch, ENTITY_SLOTS, 256):
        raise ValueError(f"spawn logits must have shape [B,{ENTITY_SLOTS},256]")
    if walkable.ndim == 2:
        walkable = walkable[None]
    if walkable.shape[0] == 1 and batch != 1:
        walkable = walkable.expand(batch, -1, -1)
    if walkable.shape != (batch, 16, 16):
        raise ValueError("encode the matching initialization map before spawning")
    old = previous[:, 8:-2].reshape(batch, ENTITY_SLOTS, 10)
    alive = (old[:, :, 5] >= .5) & (old[:, :, 4] > 0)
    ready = (~alive & normalized_timer_ready(old[:, :, 8], duration=DEATH_DURATION,
             dt=dt, increasing=True, deadline=RESPAWN_DELAY)
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
    ticks = torch.round((previous[:, -1].double() * 300 + dt) * SPAWN_TICK_HZ).long()
    tiles = torch.arange(256, device=previous.device, dtype=torch.long)
    for slot in range(ENTITY_SLOTS):
        distances = ((centers[None, :, None] - positions[:, None]) * MAP_EXTENT).square().sum(-1)
        separated = ~((distances < 1.0) & alive[:, None]).any(-1)
        allowed = base_allowed & separated
        usable = ready[:, slot] & allowed.any(-1)
        if deterministic:
            ranks = spawn_tile_rank(ticks[:, None], slot, tiles[None]) * 256 + tiles[None]
            chosen = ranks.masked_fill(~allowed, torch.iinfo(torch.long).max).argmin(-1)
        else:
            chosen = logits[:, slot].masked_fill(~allowed, -torch.finfo(logits.dtype).max).argmax(-1)
        positions[:, slot] = torch.where(usable[:, None], centers[chosen], positions[:, slot])
        alive[:, slot] |= usable
        available[:, slot] = usable
    return positions, available
