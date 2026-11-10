"""Compact supervised neural world models and a causal single-frame upscaler.

These are model definitions, not pretrained weights. In event_v1, the networks
predict XY motion, hit targets and attacks; explicit typed bookkeeping updates
health, life flags, timers, ammunition, heading and counters. Collision and hit
selection are not made exact by those rules. residual_v0 preserves the original
free-residual baseline for old checkpoints. Entity slots keep permanent IDs.
"""
from __future__ import annotations

from .structured_state import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS, STATE_SCHEMA_VERSION

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

GLOBAL_DIM = 8
ENTITY_DIM = 10
ENTITY_SLOTS = 1
STATE_DIM = GLOBAL_DIM + ENTITY_DIM * ENTITY_SLOTS + 2
ACTION_DIM = 4
EVENT_DIM = 4
FEATURE_DIM = 128
STATE_SCHEMA = {
    "version": STATE_SCHEMA_VERSION,
    "state_dim": STATE_DIM,
    "global": ["x/W", "y/H", "sin_angle", "cos_angle", "hp/100", "ammo/99",
               "fire_cooldown/.22", "shot_flash/.1"],
    "entities": ["x/W", "y/H", "vx/2", "vy/2", "hp/100", "alive",
                 "attack_cooldown/1.2", "hurt_timer/.15", "death_timer/.6", "phase"],
    "slots": ENTITY_SLOTS,
    "tail": ["kills/1", "time/300"],
    "actions": ["forward", "strafe", "turn", "fire"],
    "events": ["shot", "hit", "kill", "player_hit"],
}


@dataclass
class WorldModelConfig:
    kind: str = "student"
    entity_slots: int = ENTITY_SLOTS
    state_schema_version: str = STATE_SCHEMA_VERSION
    image_size: int = 256
    tile_types: int = 16
    architecture: str = "camera_transformer"
    transition_mode: str = "event_v1"
    simulation_hz: float = 30.0
    endless: bool = False
    navigation_only: bool = False
    renderer_mode: str = "direct"
    flow_kind: str = ""
    flow_steps: int = 0
    flow_width: int = 0
    flow_depth: int = 0
    flow_heads: int = 0
    flow_mlp_ratio: int = 4
    flow_mlp_hidden: int = 0
    renderer_codec: str = 'pixel'
    renderer_codec_config: dict | None = None
    flow_adapter_width: int = 0
    flow_adapter_depth: int = 0
    flow_adapter_heads: int = 0
    flow_adapter_mlp_ratio: int = 4
    vjepa_width: int = 0
    vjepa_encoder_depth: int = 0
    vjepa_predictor_depth: int = 0
    vjepa_heads: int = 0
    vjepa_context_frames: int = 4
    vjepa_patch_size: int = 16
    vjepa_mlp_ratio: int = 4
    vjepa_readout_mlp_ratio: int = 4
    vjepa_gradient_checkpointing: bool = False
    spatial_entities: bool = False
    spatial_combat: bool = False
    game_training_v5: bool = False
    flow_patch_size: int = 4
    flow_block_conditioning: bool = False

    def __post_init__(self):
        if not isinstance(self.navigation_only, bool):
            raise ValueError('navigation_only must be a boolean environment profile')
        if (not isinstance(self.entity_slots, int) or isinstance(self.entity_slots, bool)
                or self.entity_slots != ENTITY_SLOTS or self.state_schema_version != STATE_SCHEMA_VERSION):
            raise ValueError('Only the single-enemy20D state schema is supported; migrate legacy checkpoints explicitly')
        if not isinstance(self.spatial_entities, bool):
            raise ValueError('spatial_entities must be a boolean')
        if not isinstance(self.spatial_combat, bool):
            raise ValueError('spatial_combat must be a boolean')
        if self.spatial_combat and not self.spatial_entities:
            raise ValueError('spatial_combat requires spatial_entities=True')
        if not isinstance(self.game_training_v5, bool) or not isinstance(self.flow_block_conditioning, bool):
            raise ValueError('Training v5 and block conditioning flags must be booleans')
        if (not isinstance(self.flow_patch_size, int) or isinstance(self.flow_patch_size, bool)
                or self.flow_patch_size not in (1, 2, 4)):
            raise ValueError('Flow patch size must be 1, 2 or 4')
        if self.flow_block_conditioning and not self.spatial_combat:
            raise ValueError('Block conditioning requires explicit spatial combat conditions')
        if self.game_training_v5 and (not self.spatial_combat or self.architecture != 'vjepa2_ac'
                or self.renderer_codec != 'vqgan' or not self.endless):
            raise ValueError('Training v5 requires a spatial V-JEPA world with VQGAN and endless slots')
        if (self.flow_patch_size != 4 or self.flow_block_conditioning) and self.renderer_codec != 'vqgan':
            raise ValueError('Configurable latent patches and block conditions require VQGAN')
        if self.kind not in {"teacher", "student"}:
            raise ValueError("kind must be teacher or student")
        if self.flow_kind not in {'', 'teacher', 'student'}:
            raise ValueError('flow_kind must be empty, teacher or student')
        if self.image_size not in {32, 64, 128, 256}:
            raise ValueError("image_size must be 32, 64, 128 or 256")
        if self.architecture not in {"camera_transformer", "legacy_cnn", "vjepa2_ac"}:
            raise ValueError("architecture must be camera_transformer, legacy_cnn or vjepa2_ac")
        if self.transition_mode not in {"event_v1", "residual_v0"}:
            raise ValueError("transition_mode must be event_v1 or residual_v0")
        if self.endless and self.transition_mode != "event_v1":
            raise ValueError("endless mode requires event_v1 typed bookkeeping")
        if self.renderer_mode not in {"direct", "flow_matching", "rectified_flow"}:
            raise ValueError("renderer_mode must be direct, flow_matching or rectified_flow")
        if self.renderer_codec not in {'pixel', 'vqgan'}:
            raise ValueError('renderer_codec must be pixel or vqgan')
        if self.renderer_codec == 'vqgan' and (self.renderer_mode == 'direct' or not self.renderer_codec_config):
            raise ValueError('VQGAN rendering requires a flow renderer and explicit codec configuration')
        if (self.renderer_codec == 'vqgan' and self.renderer_codec_config
                and self.renderer_codec_config.get('image_size', self.image_size) != self.image_size):
            raise ValueError('World and VQGAN image sizes must match')
        if self.flow_mlp_hidden < 0:
            raise ValueError('Flow hidden width must be nonnegative')
        if self.flow_steps < 0:
            raise ValueError("flow_steps must be nonnegative; zero selects the kind default")
        if min(self.flow_width, self.flow_depth, self.flow_heads, self.flow_adapter_width,
               self.flow_adapter_depth, self.flow_adapter_heads) < 0:
            raise ValueError("flow dimensions must be nonnegative; zero selects defaults")
        if min(self.flow_mlp_ratio, self.flow_adapter_mlp_ratio) < 1:
            raise ValueError("flow MLP ratios must be positive")
        if bool(self.flow_adapter_width) != bool(self.flow_adapter_depth):
            raise ValueError("flow adapter width/depth must be enabled together")
        if min(self.vjepa_width, self.vjepa_encoder_depth, self.vjepa_predictor_depth, self.vjepa_heads) < 0:
            raise ValueError("V-JEPA dimensions must be nonnegative; zero selects compact kind defaults")
        if min(self.vjepa_context_frames, self.vjepa_patch_size, self.vjepa_mlp_ratio,
               self.vjepa_readout_mlp_ratio) < 1:
            raise ValueError("V-JEPA context, patch and MLP sizes must be positive")
        if self.architecture == 'vjepa2_ac' and (self.transition_mode != 'event_v1' or self.image_size % self.vjepa_patch_size):
            raise ValueError("V-JEPA requires event_v1 and patches dividing the image size")
        if not math.isfinite(self.simulation_hz) or self.simulation_hz <= 0:
            raise ValueError("simulation_hz must be finite and positive")

    @property
    def hidden(self):
        return 512 if self.kind == "teacher" else 256

    @property
    def map_channels(self):
        if self.architecture in {"camera_transformer", "vjepa2_ac"}:
            return 192 if self.kind == "teacher" else 128
        return 64 if self.kind == "teacher" else 32

    @property
    def render_channels(self):
        return 96 if self.kind == "teacher" else 48


class SeparableResidual(nn.Module):
    """Depthwise spatial mixing and pointwise channel mixing."""
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.SiLU(), nn.Conv2d(channels, channels, 1), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.SiLU(), nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x):
        return x + 0.2 * self.net(x)


class MapEncoder(nn.Module):
    def __init__(self, tile_types: int, channels: int):
        super().__init__()
        self.tile_types = tile_types
        self.embedding = nn.Embedding(tile_types, 8)
        self.net = nn.Sequential(
            nn.Conv2d(8, channels, 3, padding=1), nn.SiLU(),
            SeparableResidual(channels), SeparableResidual(channels),
        )

    def forward(self, grid):
        if grid.ndim == 2:
            grid = grid.unsqueeze(0)
        if tuple(grid.shape[-2:]) != (16, 16):
            raise ValueError("map must have shape [B,16,16]")
        if bool(((grid < 0) | (grid >= self.tile_types)).any()):
            raise ValueError("map tile ID outside configured embedding vocabulary")
        return self.net(self.embedding(grid.long()).permute(0, 3, 1, 2))


def project_state(raw, previous=None):
    """Bounds plus no-healing/absorbing-death invariants, without hit tests.

    This game has neither respawn nor healing inside an episode. An entity's
    HP/life flag can only decrease. A previously dead entity keeps its position
    and zero velocity/HP; its death-animation timers can still evolve. These
    explicit logical constraints are not presented as learned combat behavior.
    """
    bounded = raw.clamp(0.0, 1.0)
    direction = raw[:, 2:4]
    norm = direction.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(direction)
    fallback[:, 1] = 1.0
    direction = torch.where(norm > 1e-6, direction / norm.clamp_min(1e-6), fallback)
    globals_ = torch.cat([bounded[:, :2], direction, bounded[:, 4:8]], dim=-1)
    entities = raw[:, GLOBAL_DIM:-2].reshape(-1, ENTITY_SLOTS, ENTITY_DIM)
    limited = bounded[:, GLOBAL_DIM:-2].reshape_as(entities)
    entities = torch.cat([limited[:, :, :2], entities[:, :, 2:4].clamp(-1, 1),
                          limited[:, :, 4:]], dim=-1)
    if previous is not None:
        old = previous[:, GLOBAL_DIM:-2].reshape(-1, ENTITY_SLOTS, ENTITY_DIM)
        dead = old[:, :, 5:6] <= 0
        position = torch.where(dead, old[:, :, :2], entities[:, :, :2])
        velocity = torch.where(dead, torch.zeros_like(entities[:, :, 2:4]), entities[:, :, 2:4])
        hp = torch.minimum(old[:, :, 4:5].clamp(0, 1), entities[:, :, 4:5])
        hp = torch.where(dead, torch.zeros_like(hp), hp)
        alive = torch.minimum(old[:, :, 5:6].clamp(0, 1), entities[:, :, 5:6])
        entities = torch.cat([position, velocity, hp, alive, entities[:, :, 6:]], dim=-1)
    return torch.cat([globals_, entities.flatten(1), bounded[:, -2:]], dim=-1)


def render_state_view(state):
    """Only source-scene-visible values, without modifying persistent state.

    HUD is excluded from training RGB: ammo, fire cooldown, kills and elapsed
    time are therefore nonvisual. Positive player HP changes no scene pixels;
    only the alive/dead tint is visible. NPC animation/HP/life fields and the
    player's weapon flash stay available to the renderer.
    """
    player_alive = (state[:, 4:5] > 0).to(dtype=state.dtype)
    return torch.cat([state[:, :4], player_alive, torch.zeros_like(state[:, 5:7]),
                      state[:, 7:-2], torch.zeros_like(state[:, -2:])], dim=-1)


class NeuralWorldModel(nn.Module):
    """A wider teacher and a compact student sharing explicit-state interfaces.

    Static tile map is supplied at initialization and can be cached. Recurrent
    memory persists across transitions; rendered pixels are not re-encoded as
    the sole source of truth about pose, health or a dead turkey.
    """
    def __init__(self, config: WorldModelConfig | None = None):
        super().__init__()
        self.config = config or WorldModelConfig()
        if self.config.renderer_mode != "direct":
            from .flow_renderer import ConditionalFlowRenderer
            renderer = ConditionalFlowRenderer
            codec = {}
            if self.config.renderer_codec == 'vqgan':
                from .latent_flow_renderer import LatentFlowRenderer
                renderer = LatentFlowRenderer
                codec['codec_config'] = self.config.renderer_codec_config
                if self.config.spatial_combat:
                    codec['spatial_conditioning'] = True
                if self.config.flow_patch_size != 4:
                    codec['patch_size'] = self.config.flow_patch_size
                if self.config.flow_block_conditioning:
                    codec['block_conditioning'] = True
                if self.config.game_training_v5:
                    codec['improved_loss'] = True
            self.flow_renderer = renderer(kind=self.config.flow_kind or self.config.kind,
                image_size=self.config.image_size, mode=self.config.renderer_mode,
                steps=self.config.flow_steps, width=self.config.flow_width,
                depth=self.config.flow_depth, heads=self.config.flow_heads,
                mlp_ratio=self.config.flow_mlp_ratio,
                adapter_width=self.config.flow_adapter_width,
                adapter_depth=self.config.flow_adapter_depth,
                adapter_heads=self.config.flow_adapter_heads,
                adapter_mlp_ratio=self.config.flow_adapter_mlp_ratio,
                mlp_hidden=self.config.flow_mlp_hidden, **codec)
        if self.config.architecture == 'vjepa2_ac':
            from .vjepa2_ac import VJEPA2ACWorldBackend
            self.vjepa_backend = VJEPA2ACWorldBackend(self.config)
            return
        if self.config.architecture == "camera_transformer":
            from .transformer_model import CameraTransformerWorldModel
            self.camera_backend = CameraTransformerWorldModel(self.config)
            return
        c, h = self.config.map_channels, self.config.hidden
        self.map_encoder = MapEncoder(self.config.tile_types, c)
        self.map_summary = nn.Sequential(nn.Flatten(), nn.Linear(c * 4 * 4, 256), nn.SiLU())
        self.entity_encoder = nn.Sequential(nn.Linear(ENTITY_DIM, 64), nn.SiLU(), nn.Linear(64, 64))
        self.slot_embedding = nn.Parameter(torch.randn(ENTITY_SLOTS, 64) * 0.02)
        self.entity_summary = nn.Sequential(nn.Linear(ENTITY_SLOTS * 64, 256), nn.SiLU())
        self.global_encoder = nn.Sequential(nn.Linear(GLOBAL_DIM + 2, 128), nn.SiLU())
        self.action_encoder = nn.Sequential(nn.Linear(ACTION_DIM, 64), nn.SiLU())
        self.recurrent = nn.GRUCell(256 + 256 + 128 + 64, h)
        self.delta_head = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, STATE_DIM))
        if self.config.endless:
            self.spawn_head = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, ENTITY_SLOTS * 256))
            self.register_buffer("_spawn_walkable", torch.empty(0, dtype=torch.bool), persistent=False)
        if self.config.transition_mode == "residual_v0":
            self.event_head = nn.Sequential(nn.Linear(h, 128), nn.SiLU(), nn.Linear(128, EVENT_DIM))
        else:
            self.hit_target_head = nn.Sequential(nn.Linear(h, 128), nn.SiLU(), nn.Linear(128, ENTITY_SLOTS+1))
            self.attack_head = nn.Sequential(nn.Linear(h, 128), nn.SiLU(), nn.Linear(128, ENTITY_SLOTS))
            initialize_event_heads(self.hit_target_head, self.attack_head)
        self.feature_head = nn.Sequential(nn.Linear(h, FEATURE_DIM), nn.SiLU())
        # Zero residual initialization starts from a copy of the input state.
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

        rc = self.config.render_channels
        self.render_condition = nn.Sequential(nn.Linear(STATE_DIM + FEATURE_DIM, rc), nn.SiLU())
        self.render_input = nn.Conv2d(c + rc + 2, rc, 3, padding=1)
        channels = [rc, rc, rc // 2, rc // 2, max(16, rc // 4)]
        self.render_blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels[i], channels[i + 1], 3, padding=1),
                          nn.SiLU(), SeparableResidual(channels[i + 1]))
            for i in range(4)
        ])
        self.rgb128 = nn.Conv2d(channels[3], 3, 3, padding=1)
        self.rgb256 = nn.Conv2d(channels[4], 3, 3, padding=1)
        if self.config.image_size == 64:
            self.rgb64 = nn.Conv2d(channels[2], 3, 3, padding=1)
        if self.config.image_size == 32:
            self.rgb32 = nn.Conv2d(channels[1], 3, 3, padding=1)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 16), torch.linspace(-1, 1, 16), indexing="ij")
        self.register_buffer("coordinates", torch.stack([xx, yy])[None], persistent=False)

    def requires_grad_(self, requires_grad=True):
        super().requires_grad_(requires_grad)
        codec = getattr(getattr(self, 'flow_renderer', None), 'codec', None)
        if codec is not None:
            codec.eval().requires_grad_(False)
        return self

    def encode_map(self, grid):
        flow = getattr(self, 'flow_renderer', None)
        if bool(getattr(flow, 'uses_wall_conditioning', False)):
            flow.set_map_condition(grid)
        if self.config.architecture == 'vjepa2_ac':
            return self.vjepa_backend.encode_map(grid)
        if self.config.architecture == "camera_transformer":
            return self.camera_backend.encode_map(grid)
        features = self.map_encoder(grid)
        if self.config.endless:
            self._spawn_walkable = (grid == 0).to(device=features.device)
        return features

    def initial_memory(self, batch: int, device=None, dtype=None):
        """Neural memory follows weights in event_v1; world state stays FP32."""
        if self.config.architecture == 'vjepa2_ac':
            return self.vjepa_backend.initial_memory(batch, device, dtype)
        p = next(self.parameters())
        if self.config.transition_mode == "event_v1":
            dtype = p.dtype  # Persistent state is FP32; neural memory follows weights.
        return torch.zeros(batch, self.config.hidden, device=device or p.device, dtype=dtype or p.dtype)

    def transition(self, state, action, map_features, memory=None):
        if self.config.architecture == 'vjepa2_ac':
            return self.vjepa_backend.transition(state, action, map_features, memory, project_state)
        if self.config.architecture == "camera_transformer":
            return self.camera_backend.transition(state, action, map_features, memory, project_state)
        if state.ndim != 2 or state.shape[-1] != STATE_DIM:
            raise ValueError(f"state must be [B,{STATE_DIM}]")
        if action.shape != (state.shape[0], ACTION_DIM):
            raise ValueError("action must be [B,4]")
        # Never quantize persistent event_v1 bookkeeping with FP16/BF16 NN
        # inputs: a half-precision clock stops advancing on long trajectories.
        authoritative_state = state.float() if self.config.transition_mode == "event_v1" else state
        network_state = authoritative_state.to(dtype=map_features.dtype) if self.config.transition_mode == "event_v1" else state
        network_action = action.to(dtype=map_features.dtype) if self.config.transition_mode == "event_v1" else action
        entities = network_state[:, GLOBAL_DIM:-2].reshape(-1, ENTITY_SLOTS, ENTITY_DIM)
        tokens = self.entity_encoder(entities) + self.slot_embedding[None]
        globals_ = torch.cat([network_state[:, :GLOBAL_DIM], network_state[:, -2:]], dim=-1)
        inputs = torch.cat([
            self.map_summary(F.adaptive_avg_pool2d(map_features, (4, 4))),
            self.entity_summary(tokens.flatten(1)), self.global_encoder(globals_),
            self.action_encoder(network_action),
        ], dim=-1)
        if memory is None:
            memory = self.initial_memory(state.shape[0], state.device, network_state.dtype)
        elif self.config.transition_mode == "event_v1":
            memory = memory.to(dtype=self.recurrent.weight_hh.dtype)
        memory = self.recurrent(inputs, memory)
        delta = self.delta_head(memory)
        if self.config.transition_mode == "event_v1":
            candidate = project_state(authoritative_state + 0.1 * delta.float(), previous=authoritative_state)
        else:
            candidate = project_state(state + 0.1 * delta, previous=state)
        if self.config.transition_mode == "residual_v0":
            return {"state": candidate, "memory": memory,
                    "features": self.feature_head(memory), "event_logits": self.event_head(memory)}
        # Only candidate XY motion is consumed by the typed transition. HP,
        # alive, ammo, timers, counters and yaw cannot drift as free residuals.
        from .structured_state import structured_transition
        targets, attacks = self.hit_target_head(memory).float(), self.attack_head(memory).float()
        spawn = {}
        if self.config.endless:
            from .spawn import decode_spawn_positions
            logits = self.spawn_head(memory).float().reshape(-1, ENTITY_SLOTS, 256)
            positions, valid = decode_spawn_positions(logits, authoritative_state, self._spawn_walkable,
                                                      1. / self.config.simulation_hz)
            spawn = {"spawn_logits": logits, "spawn_positions": positions, "spawn_valid": valid}
        result = structured_transition(authoritative_state, candidate, action.float(), targets, attacks,
                                       dt=1. / self.config.simulation_hz, training=self.training,
                                       endless=self.config.endless,
                                       respawn_enabled=not self.config.navigation_only,
                                       spawn_positions=spawn.get("spawn_positions"), spawn_valid=spawn.get("spawn_valid"))
        return {**result, "memory": memory, "features": self.feature_head(memory),
                "target_logits": targets, "attack_logits": attacks, **spawn}

    def render_base(self, state, features, map_features):
        if self.config.architecture == 'vjepa2_ac' and hasattr(self, 'base_student'):
            geometry = self.vjepa_backend.geometry(state, map_features)
            return self.base_student(state, features, geometry)
        if self.config.architecture == 'vjepa2_ac':
            return self.vjepa_backend.render_with_aux(state, features, map_features)['rgb']
        if self.config.architecture == 'vjepa2_ac':
            result = self.vjepa_backend.render_with_aux(state, features, map_features)
        elif self.config.architecture == "camera_transformer":
            return self.camera_backend.render_with_aux(state, features, map_features)["rgb"]
        if self.config.transition_mode == "event_v1":
            # All animation/flash/tint inputs are explicit state fields. GRU
            # features remain auxiliary KD outputs, not hidden visual drift.
            state = state.to(dtype=map_features.dtype)
            features = torch.zeros_like(features, dtype=map_features.dtype)
            state = render_state_view(state)
        condition = self.render_condition(torch.cat([state, features], dim=-1))
        condition = condition[:, :, None, None].expand(-1, -1, 16, 16)
        coords = self.coordinates.to(dtype=condition.dtype).expand(state.shape[0], -1, -1, -1)
        image = F.silu(self.render_input(torch.cat([map_features, condition, coords], dim=1)))
        n_blocks = {32: 1, 64: 2, 128: 3, 256: 4}[self.config.image_size]
        for block in self.render_blocks[:n_blocks]:
            image = block(F.interpolate(image, scale_factor=2, mode="nearest"))
        head = self.rgb32 if n_blocks == 1 else (self.rgb64 if n_blocks == 2
            else (self.rgb128 if n_blocks == 3 else self.rgb256))
        return torch.sigmoid(head(image))

    def render(self, state, features, map_features):
        """Render through the configured direct or conditional-flow decoder."""
        base = self.render_base(state, features, map_features)
        if self.config.renderer_mode == "direct":
            return base
        condition = state if self.config.spatial_combat else render_state_view(state)
        return self.flow_renderer.sample(base, condition)

    def render_with_aux(self, state, features, map_features):
        """Render RGB and, for the token model, learned depth/visibility features."""
        if self.config.architecture == "camera_transformer":
            result = self.camera_backend.render_with_aux(state, features, map_features)
        else:
            result = {"rgb": self.render_base(state, features, map_features)}
        if self.config.renderer_mode != "direct":
            condition = state if self.config.spatial_combat else render_state_view(state)
            result = {**result, "base_rgb": result["rgb"],
                      "rgb": self.flow_renderer.sample(result["rgb"], condition)}
        return result

    def geometry(self, state, map_features):
        """Camera-conditioned learned auxiliaries; legacy renderer has none."""
        if self.config.architecture == 'vjepa2_ac':
            return self.vjepa_backend.geometry(state, map_features)
        if self.config.architecture == "camera_transformer":
            return self.camera_backend.geometry(state, map_features)
        return {}

    @property
    def core(self):
        if self.config.architecture != 'vjepa2_ac':
            raise AttributeError('This architecture has no V-JEPA core')
        return self.vjepa_backend.core

    def initialize_from_image(self, rgb, state, map_features=None):
        if self.config.architecture == 'vjepa2_ac':
            return self.vjepa_backend.initialize_from_image(rgb, state, map_features)
        return self.initial_memory(state.shape[0], state.device)

    def latent_features(self, memory):
        if self.config.architecture != 'vjepa2_ac':
            raise ValueError('Latent visual features require the V-JEPA architecture')
        return self.vjepa_backend.latent_features(memory)

    def initialize_from_latents(self, latents, state, map_features=None):
        if self.config.architecture != 'vjepa2_ac':
            raise ValueError('Visual latent initialization requires the V-JEPA architecture')
        return self.vjepa_backend.initialize_from_latents(latents, state, map_features)

    def step(self, state, action, grid=None, memory=None, map_features=None, render=True):
        if map_features is None:
            if grid is None:
                raise ValueError("supply grid or cached map_features")
            map_features = self.encode_map(grid)
        result = self.transition(state, action, map_features, memory)
        if render:
            result["rgb"] = self.render(result["state"], result["features"], map_features)
        return result

    forward = step


class SingleFrameUpscaler(nn.Module):
    """LR-only x4 SR. No hidden ground-truth HR input or temporal guarantees."""
    def __init__(self, channels: int = 32, blocks: int = 4, scale: int = 4):
        super().__init__()
        if scale != 4:
            raise ValueError("this prototype implements exactly x4 upscaling")
        self.config = {"channels": channels, "blocks": blocks, "scale": scale}
        self.input = nn.Conv2d(3, channels, 3, padding=1)
        self.body = nn.Sequential(*[SeparableResidual(channels) for _ in range(blocks)])
        self.output = nn.Sequential(nn.Conv2d(channels, 3 * scale * scale, 3, padding=1),
                                    nn.PixelShuffle(scale))
        nn.init.zeros_(self.output[0].weight)
        nn.init.zeros_(self.output[0].bias)

    def forward(self, low_rgb):
        residual = self.output(self.body(F.silu(self.input(low_rgb))))
        base = F.interpolate(low_rgb, scale_factor=4, mode="bilinear", align_corners=False)
        return (base + residual).clamp(0, 1)


class SRResidualBlock(nn.Module):
    """Dense EDSR-style LR residual branch: Conv/ReLU/Conv, without BN."""
    def __init__(self, channels, res_scale=.1):
        super().__init__()
        self.res_scale = float(res_scale)
        self.body = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                                  nn.ReLU(inplace=False),
                                  nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, value):
        return value+self.res_scale*self.body(value)


class EDSRUpscaler(nn.Module):
    """Single-frame dense residual x4 SR, with two learned x2 shuffle stages.

    The trunk stays on the native LR grid. A zero-initialized final RGB head
    starts exactly from bilinear enlargement. Its first update learns that
    head; subsequent updates reach both shuffle stages and the residual trunk.
    Training uses an identity straight-through derivative for output clipping:
    forward RGB stays in[0,1], while saturation cannot trap a wrong proposal.
    This is a compact project adaptation, not a pretrained EDSR checkpoint.
    """
    def __init__(self, channels=64, blocks=16, scale=4, res_scale=.1):
        super().__init__()
        if (not isinstance(channels, int) or isinstance(channels, bool) or channels < 1
                or not isinstance(blocks, int) or isinstance(blocks, bool) or blocks < 1
                or scale != 4 or not math.isfinite(res_scale) or res_scale <= 0):
            raise ValueError('EDSR requires positive channels/blocks/res_scale and exactly x4 scale')
        self.config = {'architecture': 'edsr', 'channels': channels, 'blocks': blocks,
                       'scale': scale, 'res_scale': float(res_scale)}
        self.input = nn.Conv2d(3, channels, 3, padding=1)
        self.body = nn.Sequential(*[SRResidualBlock(channels, res_scale) for _ in range(blocks)])
        self.body_output = nn.Conv2d(channels, channels, 3, padding=1)
        self.upsample = nn.Sequential(nn.Conv2d(channels, channels*4, 3, padding=1),
            nn.PixelShuffle(2), nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels*4, 3, padding=1),
            nn.PixelShuffle(2), nn.ReLU(inplace=False))
        self.output = nn.Conv2d(channels, 3, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, low_rgb):
        head = self.input(low_rgb)
        trunk = head+self.body_output(self.body(head))
        residual = self.output(self.upsample(trunk))
        base = F.interpolate(low_rgb, scale_factor=4, mode='bilinear', align_corners=False)
        proposed = base+residual
        bounded = proposed.clamp(0, 1)
        return bounded.detach()+(proposed-proposed.detach()) if self.training else bounded


def build_upscaler(config=None, **overrides):
    """Build an explicit SR architecture; omitted architecture preserves legacy.

    Old checkpoints contain only channels/blocks/scale. Their original class,
    tensor names, defaults and serialized config remain unchanged.
    """
    values = dict(config or {})
    values.update(overrides)
    architecture = values.pop('architecture', 'separable')
    if architecture == 'edsr':
        return EDSRUpscaler(**values)
    if architecture in ('separable', 'legacy'):
        return SingleFrameUpscaler(**values)
    raise ValueError(f'Unknown SR architecture: {architecture}')


def parameter_counts(model: NeuralWorldModel, upscaler: SingleFrameUpscaler | None = None):
    if model.config.architecture in {"camera_transformer", 'vjepa2_ac'}:
        backend = model.vjepa_backend if model.config.architecture == 'vjepa2_ac' else model.camera_backend
        renderer = backend.renderer if model.config.architecture == 'vjepa2_ac' else backend
        parts = {"map_encoder": sum(p.numel() for p in renderer.map_encoder.parameters()),
                 "renderer": backend.renderer_parameter_count(),
                 "shared_camera_geometry": sum(p.numel() for p in renderer.geometry_reader.parameters())}
        if model.config.architecture == 'vjepa2_ac':
            parts['latent_encoder'] = sum(p.numel() for p in backend.core.encoder.parameters())
            parts['action_conditioned_predictor'] = sum(p.numel() for p in backend.core.predictor.parameters())
    else:
        parts = {"map_encoder": sum(p.numel() for p in model.map_encoder.parameters())}
        renderer_names = ("render_condition.", "render_input.", "render_blocks.", "rgb32.", "rgb64.", "rgb128.", "rgb256.")
        parts["renderer"] = sum(p.numel() for n, p in model.named_parameters() if n.startswith(renderer_names))
    if model.config.renderer_mode != "direct":
        parts["renderer"] += sum(p.numel() for p in model.flow_renderer.parameters())
    if hasattr(model, 'base_student'):
        original = backend.renderer_parameter_count()
        compact = sum(p.numel() for p in model.base_student.parameters())
        parts['base_student'] = compact
        parts['inactive_original_base_renderer'] = original
        parts['renderer'] += compact
        parts['active_renderer'] = parts['renderer']-original
    parts["world_total"] = sum(p.numel() for p in model.parameters())
    parts["dynamics_including_map"] = parts["world_total"] - parts["renderer"]
    parts["upscaler"] = 0 if upscaler is None else sum(p.numel() for p in upscaler.parameters())
    parts["pipeline_total"] = parts["world_total"] + parts["upscaler"]
    return parts


def initialize_event_heads(target_head, attack_head):
    """Conservative initialization; still requires real combat supervision."""
    nn.init.zeros_(target_head[-1].weight)
    nn.init.zeros_(target_head[-1].bias)
    with torch.no_grad():
        target_head[-1].bias[ENTITY_SLOTS] = 2.0  # Explicit none target, not a random kill.
    nn.init.zeros_(attack_head[-1].weight)
    nn.init.constant_(attack_head[-1].bias, -4.0)


def _read_checkpoint(path, device):
    # Only load checkpoints you created or otherwise trust.
    return torch.load(Path(path), map_location=device, weights_only=False)


def load_world_checkpoint(path, device="cpu"):
    checkpoint = _read_checkpoint(path, device)
    if checkpoint.get("stage") != "world":
        raise ValueError("expected a world-model checkpoint")
    # v0.1 checkpoint dictionaries predate the architecture field. Their keys
    # and exact CNN topology remain loadable without pretending they are tokens.
    config = dict(checkpoint["config"])
    if checkpoint.get('state_schema', {}).get('slots', ENTITY_SLOTS) != ENTITY_SLOTS:
        raise ValueError('Legacy multi-enemy checkpoint requires explicit single_enemy_migration')
    if any(name.endswith(('state_encoder.weight', 'state_input.0.weight'))
           and value.ndim == 2 and value.shape[1] != STATE_DIM
           for name, value in checkpoint['model'].items()):
        raise ValueError('Legacy state dimensions require explicit single_enemy_migration')
    config.setdefault("architecture", "legacy_cnn")
    config.setdefault("transition_mode", "residual_v0")
    config.setdefault("simulation_hz", 30.0)
    model = NeuralWorldModel(WorldModelConfig(**config)).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint


def load_sr_checkpoint(path, device="cpu"):
    checkpoint = _read_checkpoint(path, device)
    if checkpoint.get("stage") != "sr":
        raise ValueError("expected an SR checkpoint")
    model = build_upscaler(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint
