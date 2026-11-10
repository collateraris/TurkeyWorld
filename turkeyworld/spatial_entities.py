"""Two persistent spatial entity tokens from state20 and learned map features.

There are no engine queries, future images or changes to the20-value state.
Slot0 is the player; slot1 retains NPC ID0, including corpses and
off-camera entities. NPC direction means observed velocity direction, not an
independent sprite heading. Stationary direction descriptors are zero/invalid.
Local sampling/motion coordinate frames fall back toward the player for a
stationary NPC; this does not invent an observed entity/sprite heading.
"""
from __future__ import annotations

from .structured_state import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS

import math

import torch
from torch import nn
from torch.nn import functional as F


ENTITY_SLOT_NAMES = ('player',) + tuple(f'npc_{index}' for index in range(ENTITY_SLOTS))
ENTITY_SLOT_COUNT = ENTITY_SLOTS+1
MAP_EXTENT = 16.
SPATIAL_FEATURE_LAYOUT = (
    ('absolute_xy', 2),                 # Normalized map coordinates x/16,y/16.
    ('direction_sin_cos', 2),           # Player heading / NPC velocity direction.
    ('direction_valid', 1),
    ('velocity_xy', 2),                 # NPC vx/2,vy/2; player velocity is unknown.
    ('velocity_valid', 1),              # 0 for player; 1 for each recorded NPC.
    ('relative_world_xy', 2),           # (EntityXY-playerXY), normalized map units.
    ('relative_forward_right', 2),      # Same displacement in player's camera axes.
    ('relative_distance', 1),           # Euclidean distance in normalized map units.
    ('relative_bearing_sin_cos', 2),     # right/distance,forward/distance.
    ('relative_bearing_valid', 1),
    ('speed', 1),                       # NPC tiles/second; unknown player speed=0.
    ('is_player', 1),
    ('alive', 1),
    ('hp', 1),                          # HP/100, as recorded in state20.
    ('animation_phase', 1),             # NPC phase; unavailable player phase=0.
    ('animation_valid', 1),
    ('cooldown', 1),                    # Recorded player fire / NPC attack timer.
    ('hurt_or_flash', 1),               # Recorded player flash / NPC hurt timer.
    ('death_timer', 1),                 # Recorded NPC death timer; player=0.
    ('relative_direction_sin_cos', 2),   # Entity direction relative to player yaw.
    ('player_heading_sin_cos', 2),       # Broadcast valid normalized player heading.
    ('player_heading_valid', 1),
)
FEATURE_SLICES = {}
SPATIAL_FEATURE_DIM = 0
for _name, _length in SPATIAL_FEATURE_LAYOUT:
    FEATURE_SLICES[_name] = slice(SPATIAL_FEATURE_DIM, SPATIAL_FEATURE_DIM+_length)
    SPATIAL_FEATURE_DIM += _length


def _check_state(state):
    if not torch.is_tensor(state) or state.ndim != 2 or state.shape[1] != STATE_DIM or state.shape[0] < 1:
        raise ValueError('Spatial entities require nonempty state[B,20]')
    if not state.is_floating_point():
        raise TypeError('Spatial state must be floating point')


def _geometry(state):
    _check_state(state)
    dtype = torch.float64 if state.dtype == torch.float64 else torch.float32
    state = state.to(dtype=dtype)
    batch = state.shape[0]
    npc = state[:, GLOBAL_DIM:-2].reshape(batch, ENTITY_SLOTS, ENTITY_DIM)
    xy = torch.cat([state[:, None, :2], npc[:, :, :2]], 1)
    player_sin_cos = state[:, 2:4]
    heading_norm = torch.linalg.vector_norm(player_sin_cos, dim=-1, keepdim=True)
    heading_valid = heading_norm > 1e-6
    player_direction = player_sin_cos/heading_norm.clamp_min(1e-6)
    fallback = torch.cat([torch.zeros_like(heading_norm), torch.ones_like(heading_norm)], -1)
    safe_heading = torch.where(heading_valid, player_direction, fallback)
    npc_velocity = npc[:, :, 2:4]
    speed_normalized = torch.linalg.vector_norm(npc_velocity, dim=-1, keepdim=True)
    npc_valid = speed_normalized > 1e-6
    npc_unit_xy = npc_velocity/speed_normalized.clamp_min(1e-6)
    npc_sin_cos = torch.stack([npc_unit_xy[:, :, 1], npc_unit_xy[:, :, 0]], -1)
    npc_direction = torch.where(npc_valid, npc_sin_cos, torch.zeros_like(npc_sin_cos))
    direction = torch.cat([torch.where(heading_valid, player_direction, torch.zeros_like(player_direction))[:, None],
                           npc_direction], 1)
    direction_valid = torch.cat([heading_valid[:, None], npc_valid], 1)
    return state, npc, xy, direction, direction_valid, safe_heading, heading_valid, speed_normalized


def entity_spatial_features(state20):
    """Return[B,2,30] descriptors with the exact exported FEATURE_SLICES layout.

    Relative coordinates retain off-screen/back-facing entities and are never
    multiplied by visibility or life flags. Unknown player velocity and absent
    stationary-NPC direction are explicitly marked, rather than invented.
    """
    state, npc, xy, direction, valid, heading, heading_valid, speed = _geometry(state20)
    zeros = torch.zeros_like(xy[:, :, :1])
    is_player = torch.cat([torch.ones_like(zeros[:, :1]), zeros[:, 1:]], 1)
    velocity = torch.cat([torch.zeros_like(xy[:, :1]), npc[:, :, 2:4]], 1)
    velocity_valid = 1-is_player
    relative = xy-xy[:, :1]
    forward = relative[:, :, 0]*heading[:, None, 1]+relative[:, :, 1]*heading[:, None, 0]
    right = -relative[:, :, 0]*heading[:, None, 0]+relative[:, :, 1]*heading[:, None, 1]
    distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
    bearing_valid = distance > 1e-6
    bearing = torch.stack([right, forward], -1)/distance.clamp_min(1e-6)
    relative_sin = direction[:, :, 0]*heading[:, None, 1]-direction[:, :, 1]*heading[:, None, 0]
    relative_cos = direction[:, :, 0]*heading[:, None, 0]+direction[:, :, 1]*heading[:, None, 1]
    relative_direction = torch.stack([relative_sin, relative_cos], -1)
    relative_direction = relative_direction*(valid & heading_valid[:, None]).to(state.dtype)
    player_heading = torch.where(heading_valid, heading, torch.zeros_like(heading))[:, None].expand(-1, ENTITY_SLOT_COUNT, -1)
    values = [xy, direction, valid.to(state.dtype), velocity, velocity_valid, relative,
        torch.stack([forward, right], -1), distance, bearing, bearing_valid.to(state.dtype),
        torch.cat([zeros[:, :1], speed*2], 1), is_player,
        torch.cat([(state[:, 4:5] > 0).to(state.dtype)[:, None], npc[:, :, 5:6]], 1),
        torch.cat([state[:, None, 4:5], npc[:, :, 4:5]], 1),
        torch.cat([zeros[:, :1], npc[:, :, 9:10]], 1), velocity_valid,
        torch.cat([state[:, None, 6:7], npc[:, :, 6:7]], 1),
        torch.cat([state[:, None, 7:8], npc[:, :, 7:8]], 1),
        torch.cat([zeros[:, :1], npc[:, :, 8:9]], 1), relative_direction,
        player_heading, heading_valid[:, None].expand(-1, ENTITY_SLOT_COUNT, -1).to(state.dtype)]
    return torch.cat(values, -1)


def entity_motion_basis(state, action=None, dt=1/30):
    """Return forward/right worldXY unit axes[B,2,2] for local motion outputs.

    Player axes use post-turn heading when controls are supplied. NPC axes use
    current velocity when nonzero, otherwise the known vector toward the player
    (V3 chase target). Coincident positions fall back to player forward. These
    are motion coordinate frames, not claims of stationary NPC sprite heading;
    the observed direction/validity descriptors remain unchanged.
    """
    state, npc, xy, direction, valid, heading, _, _ = _geometry(state)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('Motion basis timestep must be finite and positive')
    if action is not None:
        if not torch.is_tensor(action) or action.shape != (state.shape[0], 4) or action.device != state.device:
            raise ValueError('Motion basis action must have shape[B,4] on the state device')
        turn = action[:, 2].to(state.dtype).clamp(-1, 1)*(2.2*dt)*(state[:, 4] > 0).to(state.dtype)
        sine, cosine = turn.sin(), turn.cos()
        heading = torch.stack([heading[:, 0]*cosine+heading[:, 1]*sine,
                               heading[:, 1]*cosine-heading[:, 0]*sine], -1)
    player_forward = torch.stack([heading[:, 1], heading[:, 0]], -1)
    npc_forward = torch.stack([direction[:, 1:, 1], direction[:, 1:, 0]], -1)
    toward_player = xy[:, :1]-xy[:, 1:]
    distance = torch.linalg.vector_norm(toward_player, dim=-1, keepdim=True)
    fallback = torch.where(distance > 1e-6, toward_player/distance.clamp_min(1e-6), player_forward[:, None])
    npc_forward = torch.where(valid[:, 1:], npc_forward, fallback)
    forward = torch.cat([player_forward[:, None], npc_forward], 1)
    right = torch.stack([-forward[:, :, 1], forward[:, :, 0]], -1)
    return {'forward': forward, 'right': right}


def world_to_local(vectors, basis):
    """Project worldXY vectors[B,2,2] to forward/right components."""
    if vectors.shape != basis['forward'].shape or vectors.shape != basis['right'].shape:
        raise ValueError('Motion vectors must match the entity basis shape[B,2,2]')
    return torch.stack([(vectors*basis['forward']).sum(-1), (vectors*basis['right']).sum(-1)], -1)


def local_to_world(vectors, basis):
    """Rotate forward/right vectors[B,2,2] into worldXY without engine rules."""
    if vectors.shape != basis['forward'].shape or vectors.shape != basis['right'].shape:
        raise ValueError('Motion vectors must match the entity basis shape[B,2,2]')
    return vectors[:, :, :1]*basis['forward']+vectors[:, :, 1:2]*basis['right']


class SpatialEntityEncoder(nn.Module):
    """Encode nine descriptors/IDs and rotated3x3 local learned-map patches.

    Patch rows go from one tile behind to one tile ahead; columns go from one
    tile left to one tile right. Coordinates use align_corners=False, matching
    the world map's tile-center convention. This samples learned features, not
    an engine occupancy/collision oracle. All nine output tokens always exist.
    """
    def __init__(self, width, map_channels=192, *, patch_radius=1.):
        super().__init__()
        if (not isinstance(width, int) or isinstance(width, bool) or width < 1
                or not isinstance(map_channels, int) or isinstance(map_channels, bool) or map_channels < 1
                or not math.isfinite(patch_radius) or patch_radius <= 0):
            raise ValueError('Spatial width/map channels and patch radius must be positive')
        self.width, self.map_channels = width, map_channels
        self.patch_radius = float(patch_radius)
        self.descriptor_projection = nn.Sequential(nn.Linear(SPATIAL_FEATURE_DIM, width), nn.GELU())
        self.map_projection = nn.Sequential(nn.Linear(9*map_channels, 128), nn.GELU(), nn.Linear(128, width))
        self.slot_embedding = nn.Embedding(ENTITY_SLOT_COUNT, width)
        nn.init.normal_(self.slot_embedding.weight, std=.02)
        self.output_norm = nn.LayerNorm(width)
        forward, right = torch.meshgrid(torch.arange(-1, 2), torch.arange(-1, 2), indexing='ij')
        self.register_buffer('patch_offsets', torch.stack([forward, right], -1).flatten(0, 1).float(), persistent=False)

    def sampling_grid(self, state):
        """Return normalized[B,2,9,2] grid in current local motion coordinates."""
        state, npc, xy, direction, valid, heading, _, _ = _geometry(state)
        basis = entity_motion_basis(state)
        forward_xy, right_xy = basis['forward'], basis['right']
        offsets = self.patch_offsets.to(dtype=xy.dtype)*self.patch_radius/MAP_EXTENT
        positions = xy[:, :, None]+forward_xy[:, :, None]*offsets[None, None, :, :1]
        positions = positions+right_xy[:, :, None]*offsets[None, None, :, 1:2]
        return positions*2-1

    def sample_local_map(self, state, map_features):
        """Return[B,2,9,C] local features, preserving continuous coordinate grads."""
        _check_state(state)
        if (not torch.is_tensor(map_features) or map_features.ndim != 4
                or map_features.shape != (state.shape[0], self.map_channels, 16, 16)
                or not map_features.is_floating_point()):
            raise ValueError('Learned spatial map must have shape[B,map_channels,16,16]')
        if map_features.device != state.device:
            raise ValueError('Spatial state and learned map must share a device')
        dtype = torch.float64 if map_features.dtype == torch.float64 else torch.float32
        grid = self.sampling_grid(state).to(dtype=dtype).reshape(state.shape[0], ENTITY_SLOT_COUNT*9, 1, 2)
        # Avoid rounding the small map-coordinate movements to BF16 before the
        # interpolating read. Gradients still reach the supplied learned map.
        with torch.autocast(map_features.device.type, enabled=False):
            sampled = F.grid_sample(map_features.to(dtype=dtype), grid, mode='bilinear',
                                    padding_mode='border', align_corners=False)
        return sampled[:, :, :, 0].transpose(1, 2).reshape(state.shape[0], ENTITY_SLOT_COUNT, 9, self.map_channels)

    def forward(self, state, map_features):
        descriptors = entity_spatial_features(state).to(self.descriptor_projection[0].weight.dtype)
        local = self.sample_local_map(state, map_features).flatten(-2).to(self.map_projection[0].weight.dtype)
        tokens = self.descriptor_projection(descriptors)+self.map_projection(local)
        tokens = tokens+self.slot_embedding.weight[None].to(tokens.dtype)
        return self.output_norm(tokens)
