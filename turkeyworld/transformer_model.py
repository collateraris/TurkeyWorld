"""Camera-aware token world model and patch Transformer renderer.

Only coordinate transforms and learned map sampling are used at runtime. Exact
ray/wall intersection labels live in geometry_targets.py and are training-only.
This module contains no Conv2d, engine import, or ground-truth visibility call.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .camera import sample_camera_frustum, object_camera_features

STATE_DIM, ENTITY_DIM, ENTITY_SLOTS = 90, 10, 8
FEATURE_DIM, OBJECT_CAMERA_DIM = 128, 18
RAY_COUNT, RAY_SAMPLES, MAX_DEPTH = 32, 64, 24.0


class MapTokenEncoder(nn.Module):
    """256 world-coordinate tile tokens, encoded once for a static level."""
    def __init__(self, vocabulary, width, heads, layers):
        super().__init__()
        self.vocabulary, self.width = vocabulary, width
        self.tile_embedding = nn.Embedding(vocabulary, width)
        self.position = nn.Sequential(nn.Linear(2, width), nn.GELU(), nn.Linear(width, width))
        layer = nn.TransformerEncoderLayer(width, heads, dim_feedforward=3 * width,
                                            dropout=0., activation="gelu", batch_first=True,
                                            norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        y, x = torch.meshgrid((torch.arange(16) + .5) / 8 - 1,
                              (torch.arange(16) + .5) / 8 - 1, indexing="ij")
        self.register_buffer("coordinates", torch.stack([x, y], -1).reshape(1, 256, 2), persistent=False)

    def forward(self, grid):
        if grid.ndim == 2:
            grid = grid[None]
        if tuple(grid.shape[-2:]) != (16, 16):
            raise ValueError("map must be [B,16,16]")
        if bool(((grid < 0) | (grid >= self.vocabulary)).any()):
            raise ValueError("map tile ID outside embedding vocabulary")
        positions = self.position(self.coordinates.to(dtype=self.tile_embedding.weight.dtype))
        tokens = self.tile_embedding(grid.long().flatten(1)) + positions
        tokens = self.norm(self.encoder(tokens))
        return tokens.transpose(1, 2).reshape(-1, self.width, 16, 16)


class TokenReadBlock(nn.Module):
    """Bounded-size self attention followed by cross attention into scene data."""
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.self_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.query_norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 3 * width), nn.GELU(), nn.Linear(3 * width, width))

    def forward(self, tokens, context, padding_mask=None, key_weights=None):
        normalized = self.self_norm(tokens)
        tokens = tokens + self.self_attention(normalized, normalized, normalized, need_weights=False)[0]
        query, keys = self.query_norm(tokens), self.context_norm(context)
        attention_bias = None
        if key_weights is not None:
            # Apply confidence after LayerNorm: scaling a whole token before
            # LayerNorm alone would mostly cancel out the intended soft gate.
            batch, queries, keys_count = tokens.shape[0], tokens.shape[1], context.shape[1]
            attention_bias = key_weights.clamp_min(1e-5).log()[:, None, None, :]
            attention_bias = attention_bias.expand(batch, self.heads, queries, keys_count)
            attention_bias = attention_bias.reshape(batch * self.heads, queries, keys_count)
        tokens = tokens + self.cross_attention(query, keys, keys,
                                                key_padding_mask=padding_mask,
                                                attn_mask=attention_bias,
                                                need_weights=False)[0]
        return tokens + self.ff(self.ff_norm(tokens))


class CameraGeometry(nn.Module):
    """Learned opacity/depth and center-line object visibility features.

    Each of 32 camera columns samples 64 map locations at forward depths.
    Learned alpha and transmittance integrate samples into a camera ray token.
    This is a differentiable learned approximation, not exact wall raycasting.
    Object visibility is center/FOV visibility against walls; it is not an
    alpha-accurate sprite mask and does not model inter-object occlusion.
    """
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.opacity_network = nn.Sequential(nn.Linear(width + 1, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.normal_(self.opacity_network[-1].weight, std=.01)
        nn.init.constant_(self.opacity_network[-1].bias, math.log(.05 / .95))
        self.background_token = nn.Parameter(torch.zeros(width))
        self.ray_position = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))
        self.object_encoder = nn.Sequential(nn.Linear(ENTITY_DIM + OBJECT_CAMERA_DIM, width),
                                             nn.GELU(), nn.Linear(width, width))
        self.object_position = nn.Sequential(nn.Linear(OBJECT_CAMERA_DIM, width), nn.GELU(),
                                              nn.Linear(width, width))
        self.slot_embedding = nn.Parameter(torch.randn(ENTITY_SLOTS, width) * .02)
        self.visibility_head = nn.Sequential(nn.Linear(OBJECT_CAMERA_DIM + 2, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, state, map_features):
        samples = sample_camera_frustum(map_features, state, rays=RAY_COUNT,
                                        samples=RAY_SAMPLES, max_depth=MAX_DEPTH)
        sampled = samples["sampled"]
        depths = samples["forward_depths"].to(dtype=sampled.dtype)
        depth_channel = (depths / MAX_DEPTH)[None, None, :, None].expand(*sampled.shape[:3], 1)
        alpha = self.opacity_network(torch.cat([sampled, depth_channel], -1)).squeeze(-1).sigmoid()
        # Leaving a finite map is opaque. This is only a bounds prior; walls
        # inside the map are represented by learned alpha, never oracle hits.
        alpha = torch.where(samples["in_bounds"], alpha, torch.ones_like(alpha))
        alpha32 = alpha.float().clamp(0., 1. - 1e-6)
        survival = 1 - alpha32
        inclusive = survival.cumprod(-1)
        transmittance = torch.cat([torch.ones_like(inclusive[..., :1]), inclusive[..., :-1]], -1)
        weights = transmittance * alpha32
        background = inclusive[..., -1]
        depth = (weights * depths.float()[None, None]).sum(-1) + background * MAX_DEPTH
        ray_tokens = (sampled * weights.to(sampled.dtype)[..., None]).sum(-2)
        ray_tokens = ray_tokens + background.to(sampled.dtype)[..., None] * self.background_token
        screen_x = samples["screen_x"].to(dtype=sampled.dtype)[None].expand(state.shape[0], -1)
        ray_descriptors = torch.stack([screen_x, depth.to(sampled.dtype) / MAX_DEPTH,
                                       background.to(sampled.dtype)], -1)
        ray_tokens = ray_tokens + self.ray_position(ray_descriptors)

        objects = object_camera_features(state, max_depth=MAX_DEPTH)
        object_features = objects["features"].to(dtype=sampled.dtype)
        raw_entities = state[:, 8:-2].reshape(-1, ENTITY_SLOTS, ENTITY_DIM)
        object_appearance = self.object_encoder(torch.cat([raw_entities, object_features], -1))
        object_identity = self.object_position(object_features) + self.slot_embedding[None]
        object_tokens = object_appearance + object_identity
        # Sample the predicted depth at each object's projected center column.
        x = objects["screen_x"].clamp(-1, 1)
        grid = torch.stack([x, torch.zeros_like(x)], -1)[:, :, None]
        column_depth = F.grid_sample(depth[:, None, None].to(sampled.dtype), grid.to(sampled.dtype),
                                     mode="bilinear", padding_mode="border", align_corners=False)[:, 0, :, 0]
        forward = objects["forward"].to(sampled.dtype)
        visibility_features = torch.cat([object_features,
                                          (column_depth / MAX_DEPTH)[..., None],
                                          ((column_depth - forward) / MAX_DEPTH)[..., None]], -1)
        raw_visibility_logits = self.visibility_head(visibility_features).squeeze(-1)
        frustum = objects["frustum"].bool()
        logits = torch.where(frustum, raw_visibility_logits, torch.full_like(raw_visibility_logits, -20.))
        visibility = logits.sigmoid() * frustum.to(logits.dtype)
        return {"depth": depth.to(sampled.dtype), "visibility_logits": logits,
                "visibility": visibility, "frustum": frustum,
                "ray_tokens": ray_tokens, "object_tokens": object_tokens,
                "object_identity_tokens": object_identity,
                "object_appearance_tokens": object_appearance,
                "object_camera": object_features, "object_forward": forward,
                "object_screen_x": objects["screen_x"],
                "opacity": alpha, "ray_weights": weights.to(sampled.dtype)}


class CameraTransformerWorldModel(nn.Module):
    """Token dynamics plus fixed-size recurrent memory and a patch renderer."""
    def __init__(self, config, *, render_only=False):
        super().__init__()
        self.config = config
        self.width = width = 192 if config.kind == "teacher" else 128
        heads = 6 if config.kind == "teacher" else 4
        layers = 4 if config.kind == "teacher" else 2
        hidden = config.hidden
        self.map_encoder = MapTokenEncoder(config.tile_types, width, heads, layers)
        self.geometry_reader = CameraGeometry(width)
        if not render_only:
            self.player_token = nn.Sequential(nn.Linear(10, width), nn.GELU(), nn.Linear(width, width))
            self.action_token = nn.Linear(4, width)
            self.memory_token = nn.Linear(hidden, width)
            self.scene_type_embeddings = nn.Parameter(torch.randn(11, width) * .02)
            self.scene_blocks = nn.ModuleList([TokenReadBlock(width, heads) for _ in range(layers)])
            self.recurrent = nn.GRUCell(width, hidden)
            self.delta_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, STATE_DIM))
            if config.endless:
                self.spawn_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                                nn.Linear(hidden, ENTITY_SLOTS * 256))
            if config.transition_mode == "residual_v0":
                self.event_head = nn.Sequential(nn.Linear(hidden, 128), nn.SiLU(), nn.Linear(128, 4))
            else:
                from .models import initialize_event_heads
                self.hit_target_head = nn.Sequential(nn.Linear(hidden, 128), nn.SiLU(), nn.Linear(128, 9))
                self.attack_head = nn.Sequential(nn.Linear(hidden, 128), nn.SiLU(), nn.Linear(128, 8))
                initialize_event_heads(self.hit_target_head, self.attack_head)
            self.feature_head = nn.Sequential(nn.Linear(hidden, FEATURE_DIM), nn.SiLU())
            nn.init.zeros_(self.delta_head[-1].weight)
            nn.init.zeros_(self.delta_head[-1].bias)
        if config.endless:
            self.register_buffer("_spawn_walkable", torch.empty(0, dtype=torch.bool), persistent=False)

        self.render_condition = nn.Sequential(nn.Linear(STATE_DIM + FEATURE_DIM, width), nn.GELU(),
                                                nn.Linear(width, width))
        self.screen_position = nn.Sequential(nn.Linear(18, width), nn.GELU(), nn.Linear(width, width))
        self.render_blocks = nn.ModuleList([TokenReadBlock(width, heads) for _ in range(layers)])
        self.render_norm = nn.LayerNorm(width)
        if not render_only or config.image_size == 128:
            self.rgb128 = nn.Linear(width, 3 * 8 * 8)
        if not render_only or config.image_size == 256:
            self.rgb256 = nn.Linear(width, 3 * 16 * 16)
        y, x = torch.meshgrid((torch.arange(16) + .5) / 8 - 1,
                              (torch.arange(16) + .5) / 8 - 1, indexing="ij")
        xy = torch.stack([x, y], -1).reshape(1, 256, 2)
        frequencies = math.pi * (2 ** torch.arange(4))
        angles = xy[..., None] * frequencies
        position_features = torch.cat([xy, angles.sin().flatten(-2), angles.cos().flatten(-2)], -1)
        self.register_buffer("screen_coordinates", position_features, persistent=False)

    def encode_map(self, grid):
        features = self.map_encoder(grid)
        if self.config.endless:
            self._spawn_walkable = (grid == 0).to(device=features.device)
        return features

    def initial_memory(self, batch, device=None, dtype=None):
        parameter = next(self.parameters())
        if self.config.transition_mode == "event_v1":
            dtype = parameter.dtype
        return torch.zeros(batch, self.config.hidden, device=device or parameter.device,
                           dtype=dtype or parameter.dtype)

    def geometry(self, state, map_features):
        if self.config.transition_mode == "event_v1":
            state = state.to(dtype=map_features.dtype)
        return self.geometry_reader(state, map_features)

    def transition(self, state, action, map_features, memory, projection):
        if state.ndim != 2 or state.shape[-1] != STATE_DIM:
            raise ValueError("state must be [B,90]")
        if action.shape != (state.shape[0], 4):
            raise ValueError("action must be [B,4]")
        authoritative_state = state.float() if self.config.transition_mode == "event_v1" else state
        network_state = authoritative_state.to(dtype=map_features.dtype) if self.config.transition_mode == "event_v1" else state
        network_action = action.to(dtype=map_features.dtype) if self.config.transition_mode == "event_v1" else action
        if memory is None:
            memory = self.initial_memory(state.shape[0], state.device, network_state.dtype)
        elif self.config.transition_mode == "event_v1":
            memory = memory.to(dtype=self.recurrent.weight_hh.dtype)
        geometry = self.geometry(network_state, map_features)
        global_state = torch.cat([network_state[:, :8], network_state[:, -2:]], -1)
        tokens = torch.cat([self.player_token(global_state)[:, None],
                             self.action_token(network_action)[:, None],
                             self.memory_token(memory)[:, None], geometry["object_tokens"]], 1)
        tokens = tokens + self.scene_type_embeddings[None]
        # 11 scene tokens cross-read 256 map tokens plus 32 camera ray tokens.
        context = torch.cat([map_features.flatten(2).transpose(1, 2), geometry["ray_tokens"]], 1)
        for block in self.scene_blocks:
            tokens = block(tokens, context)
        memory = self.recurrent(tokens[:, 0], memory)
        delta = self.delta_head(memory)
        if self.config.transition_mode == "event_v1":
            candidate = projection(authoritative_state + .1 * delta.float(), previous=authoritative_state)
        else:
            candidate = projection(state + .1 * delta, previous=state)
        if self.config.transition_mode == "residual_v0":
            return {"state": candidate, "memory": memory,
                    "features": self.feature_head(memory), "event_logits": self.event_head(memory)}
        # Bookkeeping is explicit and disclosed: the network selects targets,
        # attacks and XY motion; typed rules update HP/alive/ammo/yaw/timers.
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
                                       spawn_positions=spawn.get("spawn_positions"), spawn_valid=spawn.get("spawn_valid"))
        return {**result, "memory": memory, "features": self.feature_head(memory),
                "target_logits": targets, "attack_logits": attacks, **spawn}

    def render_with_aux(self, state, features, map_features):
        if self.config.transition_mode == "event_v1":
            # Render is a function of explicit world/animation state and the
            # cached map. Different recurrent histories cannot repaint walls
            # when those inputs are identical. 128D features remain KD-only.
            state = state.to(dtype=map_features.dtype)
            features = (features.to(dtype=map_features.dtype) if self.config.architecture == "vjepa2_ac"
                        else torch.zeros_like(features, dtype=map_features.dtype))
        geometry = self.geometry(state, map_features)
        if self.config.transition_mode == "event_v1":
            from .models import render_state_view
            condition_state = render_state_view(state)
        else:
            condition_state = state
        condition = self.render_condition(torch.cat([condition_state, features], -1))
        queries = self.screen_position(self.screen_coordinates.to(dtype=state.dtype))
        queries = queries.expand(state.shape[0], -1, -1) + condition[:, None]
        # Center visibility is only a feature, not a per-pixel silhouette mask:
        # a sprite can straddle a wall/FOV edge while its center is hidden.
        # Keep every ID/geometry token and retain a small appearance path even
        # for center-hidden objects; corpses are never removed by alive.
        gate = .05 + .95 * geometry["visibility"]
        objects = geometry["object_identity_tokens"] + gate[..., None] * geometry["object_appearance_tokens"]
        context = torch.cat([geometry["ray_tokens"], objects, condition[:, None]], 1)
        key_weights = torch.cat([torch.ones(state.shape[0], RAY_COUNT, device=state.device, dtype=state.dtype),
                                  gate, torch.ones(state.shape[0], 1, device=state.device, dtype=state.dtype)], 1)
        for block in self.render_blocks:
            queries = block(queries, context, key_weights=key_weights)
        patch = 8 if self.config.image_size == 128 else 16
        head = self.rgb128 if patch == 8 else self.rgb256
        pixels = head(self.render_norm(queries)).reshape(-1, 16, 16, 3, patch, patch)
        rgb = pixels.permute(0, 3, 1, 4, 2, 5).reshape(-1, 3, 16 * patch, 16 * patch).sigmoid()
        return {"rgb": rgb, **geometry}

    def renderer_parameter_count(self):
        names = ("render_condition.", "screen_position.", "render_blocks.", "render_norm.",
                  "rgb128.", "rgb256.")
        return sum(p.numel() for name, p in self.named_parameters() if name.startswith(names))
