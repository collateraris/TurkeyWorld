"""Compact game adaptation of V-JEPA2-AC principles; no pretrained claims.

The official MIT implementation at facebookresearch/vjepa2 provides the design
reference: video patch latents, interleaved action/state tokens, frame-block
causal action conditioning and separate temporal/height/width rotary axes.
This implementation uses game action4/state90, 128px images and a causal video
encoder, so encoding a full training clip cannot leak future images backward.
An EMA target encoder belongs to the trainer, never this inference model.

Unlike the earlier model, runtime memory contains autoregressively predicted
visual tokens plus their action/state conditioning. There is no GRU, original
engine call, future-image input or engine correction in a neural transition.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def block_causal_mask(frames, tokens_per_frame, device=None):
    """SDPA mask: full spatial attention in-frame, causal between frames."""
    frame = torch.arange(frames*tokens_per_frame, device=device)//tokens_per_frame
    return frame[:, None] >= frame[None, :]


def _positions(frames, grid, condition_tokens, device):
    time, yy, xx = torch.meshgrid(torch.arange(frames, device=device),
        torch.arange(grid, device=device), torch.arange(grid, device=device), indexing='ij')
    visual = torch.stack([time, yy, xx], -1).reshape(frames, grid*grid, 3)
    if condition_tokens:
        condition = torch.zeros(frames, condition_tokens, 3, device=device, dtype=visual.dtype)
        condition[:, :, 0] = torch.arange(frames, device=device)[:, None]
        visual = torch.cat([condition, visual], 1)
    return visual.flatten(0, 1)


class RotaryAttention3D(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        if width % heads or width//heads < 6:
            raise ValueError('RoPE width must divide heads and provide at least6 dimensions/head')
        self.heads, self.head_width = heads, width//heads
        self.axis_width = 2*((self.head_width//3)//2)
        self.qkv = nn.Linear(width, 3*width)
        self.proj = nn.Linear(width, width)
        frequency = 1/(10000**(torch.arange(self.axis_width//2).float()/(self.axis_width/2)))
        self.register_buffer('frequency', frequency, persistent=False)

    def rotate(self, value, positions):
        pieces = []
        for axis in range(3):
            start = axis*self.axis_width
            component = value[..., start:start+self.axis_width].float()
            phase = positions[:, axis].float()[:, None]*self.frequency[None]
            cosine = phase.cos().repeat_interleave(2, -1)[None, None]
            sine = phase.sin().repeat_interleave(2, -1)[None, None]
            pairs = component.reshape(*component.shape[:-1], -1, 2)
            perpendicular = torch.stack([-pairs[..., 1], pairs[..., 0]], -1).flatten(-2)
            pieces.append((component*cosine+perpendicular*sine).to(value.dtype))
        if self.axis_width*3 < self.head_width:
            pieces.append(value[..., self.axis_width*3:])
        return torch.cat(pieces, -1)

    def forward(self, tokens, positions, allowed):
        batch, length, width = tokens.shape
        qkv = self.qkv(tokens).reshape(batch, length, 3, self.heads, self.head_width).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query, key = self.rotate(query, positions), self.rotate(key, positions)
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=allowed[None, None], dropout_p=0.)
        return self.proj(attended.transpose(1, 2).reshape(batch, length, width))


class VideoBlock(nn.Module):
    def __init__(self, width, heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=1e-6)
        self.attention = RotaryAttention3D(width, heads)
        self.norm2 = nn.LayerNorm(width, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(width, width*mlp_ratio), nn.GELU(approximate='tanh'),
                                 nn.Linear(width*mlp_ratio, width))

    def forward(self, tokens, positions, allowed):
        tokens = tokens+self.attention(self.norm1(tokens), positions, allowed)
        return tokens+self.mlp(self.norm2(tokens))


def _initialize_blocks(module):
    for child in module.modules():
        if isinstance(child, (nn.Linear, nn.Conv3d)):
            nn.init.trunc_normal_(child.weight, std=.02)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.LayerNorm):
            nn.init.ones_(child.weight)
            nn.init.zeros_(child.bias)
    with torch.no_grad():
        for index, block in enumerate(module.blocks, start=1):
            block.attention.proj.weight.div_(math.sqrt(2*index))
            block.mlp[-1].weight.div_(math.sqrt(2*index))


def _run_blocks(module, tokens, positions, allowed):
    """Checkpoint only trainable execution; inference and EMA remain ordinary ViT."""
    for block in module.blocks:
        if module.gradient_checkpointing and module.training and torch.is_grad_enabled():
            tokens = checkpoint(block, tokens, positions, allowed, use_reentrant=False)
        else:
            tokens = block(tokens, positions, allowed)
    return tokens


class CausalVideoEncoder(nn.Module):
    """Online/EMA-compatible visual encoder with temporal tubelet size1."""
    def __init__(self, image_size=128, patch_size=16, width=512, depth=10, heads=8, mlp_ratio=4,
                 gradient_checkpointing=False):
        super().__init__()
        self.image_size, self.patch_size, self.width = image_size, patch_size, width
        self.gradient_checkpointing = gradient_checkpointing
        self.grid = image_size//patch_size
        self.tokens_per_frame = self.grid*self.grid
        self.patch_embed = nn.Conv3d(3, width, (1, patch_size, patch_size), stride=(1, patch_size, patch_size))
        self.blocks = nn.ModuleList(VideoBlock(width, heads, mlp_ratio) for _ in range(depth))
        self.norm = nn.LayerNorm(width, eps=1e-6)
        _initialize_blocks(self)

    def forward(self, video):
        if video.ndim != 5 or video.shape[2] != 3 or tuple(video.shape[-2:]) != (self.image_size, self.image_size):
            raise ValueError('video must have shape [B,T,3,image_size,image_size]')
        batch, frames = video.shape[:2]
        if frames < 1:
            raise ValueError('video must contain at least one frame')
        image = video.to(dtype=self.patch_embed.weight.dtype).permute(0, 2, 1, 3, 4)
        tokens = self.patch_embed(image).permute(0, 2, 3, 4, 1).reshape(batch, frames*self.tokens_per_frame, self.width)
        positions = _positions(frames, self.grid, 0, video.device)
        allowed = block_causal_mask(frames, self.tokens_per_frame, video.device)
        tokens = _run_blocks(self, tokens, positions, allowed)
        return self.norm(tokens).reshape(batch, frames, self.tokens_per_frame, self.width)


class ActionConditionedPredictor(nn.Module):
    def __init__(self, width=512, depth=6, heads=8, grid=8, mlp_ratio=4,
                 gradient_checkpointing=False):
        super().__init__()
        self.width, self.grid = width, grid
        self.gradient_checkpointing = gradient_checkpointing
        self.visual_tokens = grid*grid
        self.block_tokens = self.visual_tokens+2
        self.predictor_embed = nn.Linear(width, width)
        self.action_encoder = nn.Linear(4, width)
        self.state_encoder = nn.Linear(90, width)
        self.type_embedding = nn.Parameter(torch.randn(3, width)*.02)
        self.blocks = nn.ModuleList(VideoBlock(width, heads, mlp_ratio) for _ in range(depth))
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.projection = nn.Linear(width, width)
        _initialize_blocks(self)

    def pack(self, latents, actions, states):
        batch, frames, count, width = latents.shape
        if count != self.visual_tokens or width != self.width:
            raise ValueError('Visual latent token shape does not match predictor')
        if actions.shape != (batch, frames, 4) or states.shape != (batch, frames, 90):
            raise ValueError('Actions/states must align with every visual frame')
        dtype = self.action_encoder.weight.dtype
        action = self.action_encoder(actions.to(dtype))[:, :, None].to(latents.dtype)
        state = self.state_encoder(states.to(dtype))[:, :, None].to(latents.dtype)
        return torch.cat([action, state, latents], 2).flatten(1, 2)

    def forward_packed(self, packed):
        batch, length, width = packed.shape
        if width != self.width or length % self.block_tokens:
            raise ValueError('Packed memory must contain complete action/state/visual blocks')
        frames = length//self.block_tokens
        blocks = packed.reshape(batch, frames, self.block_tokens, width)
        visual = self.predictor_embed(blocks[:, :, 2:].to(self.predictor_embed.weight.dtype))
        conditions = blocks[:, :, :2].to(visual.dtype)+self.type_embedding[:2][None, None]
        visual = visual+self.type_embedding[2]
        tokens = torch.cat([conditions, visual], 2).flatten(1, 2)
        positions = _positions(frames, self.grid, 2, packed.device)
        allowed = block_causal_mask(frames, self.block_tokens, packed.device)
        tokens = _run_blocks(self, tokens, positions, allowed)
        visual = tokens.reshape(batch, frames, self.block_tokens, width)[:, :, 2:]
        return self.projection(self.norm(visual))

    def forward(self, latents, actions, states):
        return self.forward_packed(self.pack(latents, actions, states))


class CompactVJEPA2AC(nn.Module):
    def __init__(self, image_size=128, patch_size=16, width=512, encoder_depth=10,
                 predictor_depth=6, heads=8, context_frames=4, mlp_ratio=4,
                 gradient_checkpointing=False):
        super().__init__()
        self.width, self.context_frames = width, context_frames
        self.encoder = CausalVideoEncoder(image_size, patch_size, width, encoder_depth, heads, mlp_ratio,
                                         gradient_checkpointing)
        self.predictor = ActionConditionedPredictor(width, predictor_depth, heads, self.encoder.grid, mlp_ratio,
                                                   gradient_checkpointing)
        self.visual_tokens = self.encoder.tokens_per_frame
        self.block_tokens = self.visual_tokens+2
        self.memory_tokens = context_frames*self.block_tokens

    def encode_video(self, video):
        return self.encoder(video)

    def predict(self, latents, actions, states):
        return self.predictor(latents, actions, states)

    def pack_context(self, latents, actions, states):
        return self.predictor.pack(latents, actions, states)

    def predict_packed(self, packed):
        return self.predictor.forward_packed(packed)

    def initial_memory_from_latents(self, latents, states, actions=None):
        batch, frames = latents.shape[:2]
        if states.ndim == 2:
            states = states[:, None].expand(-1, frames, -1)
        if actions is None:
            actions = states.new_zeros(batch, frames, 4)
        if frames < self.context_frames:
            padding = self.context_frames-frames
            latents = torch.cat([latents[:, :1].expand(-1, padding, -1, -1), latents], 1)
            states = torch.cat([states[:, :1].expand(-1, padding, -1), states], 1)
            actions = torch.cat([actions.new_zeros(batch, padding, 4), actions], 1)
        return self.pack_context(latents[:, -self.context_frames:], actions[:, -self.context_frames:],
                                 states[:, -self.context_frames:])


class VJEPA2ACWorldBackend(nn.Module):
    """Visual-latent predictor plus typed game readouts; no legacy GRU modules."""
    def __init__(self, config):
        super().__init__()
        from .transformer_model import CameraTransformerWorldModel
        from .models import initialize_event_heads
        self.config = config
        width = config.vjepa_width or (512 if config.kind == 'teacher' else 256)
        encoder_depth = config.vjepa_encoder_depth or (10 if config.kind == 'teacher' else 6)
        predictor_depth = config.vjepa_predictor_depth or (6 if config.kind == 'teacher' else 4)
        heads = config.vjepa_heads or (8 if config.kind == 'teacher' else 4)
        self.renderer = CameraTransformerWorldModel(config, render_only=True)
        self.core = CompactVJEPA2AC(config.image_size, config.vjepa_patch_size, width,
            encoder_depth, predictor_depth, heads, config.vjepa_context_frames, config.vjepa_mlp_ratio,
            config.vjepa_gradient_checkpointing)
        map_channels = 192 if config.kind == 'teacher' else 128
        rank = min(128, width)
        self.map_projection = nn.Sequential(nn.Linear(map_channels*6, rank), nn.GELU(),
                                             nn.Linear(rank, width), nn.LayerNorm(width))
        # A used latent-to-game readout neck, rather than dormant parameters:
        # every motion/event/spawn/appearance head consumes this residual MLP.
        neck_width = width*config.vjepa_readout_mlp_ratio
        self.readout_neck = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, neck_width),
                                           nn.GELU(), nn.Linear(neck_width, width))
        # Only playerXY and eight NPC XY pairs are learned motion outputs;
        # HP, resources, heading and clocks remain the disclosed typed rules.
        self.delta_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 18))
        nn.init.normal_(self.delta_head[-1].weight, std=.001)
        nn.init.zeros_(self.delta_head[-1].bias)
        self.hit_target_head = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 9))
        self.attack_head = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 8))
        initialize_event_heads(self.hit_target_head, self.attack_head)
        self.feature_head = nn.Linear(width, 128)
        nn.init.zeros_(self.feature_head.weight)
        nn.init.zeros_(self.feature_head.bias)
        if config.endless:
            self.spawn_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 8*256))

    def encode_map(self, grid):
        return self.renderer.encode_map(grid)

    def initial_memory(self, batch, device=None, dtype=None):
        first = next(self.core.parameters())
        return torch.zeros(batch, self.core.memory_tokens, self.core.width,
                           device=device or first.device, dtype=dtype or first.dtype)

    def initialize_from_image(self, rgb, state, map_features=None):
        if rgb.ndim == 4:
            rgb = rgb[:, None]
        if rgb.dtype == torch.uint8:
            rgb = rgb.float()/255
        latents = self.core.encode_video(rgb)
        return self.initialize_from_latents(latents, state, map_features)

    def initialize_from_latents(self, latents, state, map_features=None):
        """Reuse an already encoded context, including persistent map tokens."""
        memory = self.core.initial_memory_from_latents(latents, state)
        if map_features is not None:
            if state.ndim == 2:
                context = self._map_context(map_features, state)[:, None].expand(-1, self.core.context_frames, -1)
            else:
                sequence = state[:, -self.core.context_frames:]
                if sequence.shape[1] < self.core.context_frames:
                    sequence = torch.cat([sequence[:, :1].expand(-1, self.core.context_frames-sequence.shape[1], -1), sequence], 1)
                context = torch.stack([self._map_context(map_features, sequence[:, index])
                                       for index in range(self.core.context_frames)], 1)
            blocks = memory.reshape(memory.shape[0], self.core.context_frames, self.core.block_tokens, self.core.width)
            memory = torch.cat([blocks[:, :, :1], blocks[:, :, 1:2]+context[:, :, None], blocks[:, :, 2:]], 2).flatten(1, 2)
        return memory

    def _map_context(self, map_features, state):
        """Read the persistent learned map globally and at all known objects.

        These are differentiable feature reads, not exact collision/raycast
        rules. The unchanged90D numerical state retains all8 entity slots even
        outside the camera; visibility never deletes their persistent state.
        """
        entity_xy = state[:, 8:88].reshape(-1, 8, 10)[:, :, :2]
        coordinates = torch.cat([state[:, None, :2], entity_xy], 1)*2-1
        grid = coordinates[:, :, None].to(map_features.dtype)
        local = F.grid_sample(map_features, grid, align_corners=False, padding_mode='border')[:, :, :, 0]
        descriptor = torch.cat([F.adaptive_avg_pool2d(map_features, (2, 2)).flatten(1),
                                 local[:, :, 0], local[:, :, 1:].mean(-1)], -1)
        return self.map_projection(descriptor.to(self.map_projection[0].weight.dtype))

    def latent_features(self, memory):
        blocks = memory.reshape(memory.shape[0], self.core.context_frames, self.core.block_tokens, self.core.width)
        pooled = blocks[:, -1, 2:].mean(1).to(self.feature_head.weight.dtype)
        pooled = pooled+self.readout_neck(pooled)
        return self.feature_head(pooled)

    def transition(self, state, action, map_features, memory, projection):
        state = state.float()
        batch = state.shape[0]
        if memory is None or not bool(memory.detach().any()):
            # Compatibility initialization only. This image is a neural render
            # from supplied numerical state/map, never an original-engine frame.
            rgb = self.renderer.render_with_aux(state, state.new_zeros(batch, 128), map_features)['rgb']
            memory = self.initialize_from_image(rgb, state, map_features)
        blocks = memory.reshape(batch, self.core.context_frames, self.core.block_tokens, self.core.width)
        current = self.core.pack_context(blocks[:, -1:, 2:], action[:, None], state[:, None])
        context = self._map_context(map_features, state)
        current = torch.cat([current[:, :1], current[:, 1:2]+context[:, None], current[:, 2:]], 1)
        packed = torch.cat([blocks[:, :-1].flatten(1, 2), current], 1)
        prediction = self.core.predict_packed(packed)[:, -1]
        pooled = prediction.mean(1).to(self.readout_neck[1].weight.dtype)
        pooled = pooled+self.readout_neck(pooled)
        delta = self.delta_head(pooled.to(self.delta_head[0].weight.dtype))
        motion = delta.float()
        old_entities = state[:, 8:88].reshape(batch, 8, 10)
        candidate_entities = torch.cat([old_entities[:, :, :2]+.1*motion[:, 2:].reshape(batch, 8, 2),
                                        old_entities[:, :, 2:]], -1)
        candidate = torch.cat([state[:, :2]+.1*motion[:, :2], state[:, 2:8],
                               candidate_entities.flatten(1), state[:, -2:]], -1)
        candidate = projection(candidate, previous=state)
        targets = self.hit_target_head(pooled.to(self.hit_target_head[0].weight.dtype)).float()
        attacks = self.attack_head(pooled.to(self.attack_head[0].weight.dtype)).float()
        spawn = {}
        if self.config.endless:
            from .spawn import decode_spawn_positions
            logits = self.spawn_head(pooled.to(self.spawn_head[0].weight.dtype)).float().reshape(batch, 8, 256)
            positions, valid = decode_spawn_positions(logits, state, self.renderer._spawn_walkable,
                                                      1/self.config.simulation_hz)
            spawn = {'spawn_logits': logits, 'spawn_positions': positions, 'spawn_valid': valid}
        from .structured_state import structured_transition
        result = structured_transition(state, candidate, action.float(), targets, attacks,
            dt=1/self.config.simulation_hz, training=self.training, endless=self.config.endless,
            spawn_positions=spawn.get('spawn_positions'), spawn_valid=spawn.get('spawn_valid'))
        next_block = self.core.pack_context(prediction[:, None], action.new_zeros(batch, 1, 4), result['state'][:, None])
        next_context = self._map_context(map_features, result['state'])
        next_block = torch.cat([next_block[:, :1], next_block[:, 1:2]+next_context[:, None], next_block[:, 2:]], 1)
        next_memory = torch.cat([packed.reshape_as(blocks)[:, 1:].flatten(1, 2), next_block], 1)
        return {**result, **spawn, 'memory': next_memory, 'latent_prediction': prediction,
                'features': self.feature_head(pooled.to(self.feature_head.weight.dtype)),
                'target_logits': targets, 'attack_logits': attacks}

    def geometry(self, state, map_features):
        return self.renderer.geometry(state, map_features)

    def render_with_aux(self, state, features, map_features):
        return self.renderer.render_with_aux(state, features, map_features)

    def renderer_parameter_count(self):
        return self.renderer.renderer_parameter_count()


def load_vjepa125_from75_checkpoint(source, device='cpu', *, require_profile=True, **config_overrides):
    """Warm-start renderer/map/geometry only; V-JEPA dynamics are new weights.

    No GRU, scene-token dynamics or old event/spawn heads are represented as a
    pretrained V-JEPA predictor. The external trainer must learn the new online
    encoder/predictor and maintain its separate frozen EMA target encoder.
    """
    from .models import NeuralWorldModel, WorldModelConfig, load_world_checkpoint
    original, metadata = load_world_checkpoint(source, 'cpu')
    if original.config.kind != 'teacher' or original.config.architecture != 'camera_transformer':
        raise ValueError('Compact125M migration requires the trained camera teacher renderer')
    if original.config.image_size != 128:
        raise ValueError('The compact125M game profile uses128px source images')
    config = asdict(original.config)
    config.update(architecture='vjepa2_ac')
    config.update(config_overrides)
    model = NeuralWorldModel(WorldModelConfig(**config))
    target = model.state_dict()
    allowed = ('map_encoder.', 'geometry_reader.', 'render_condition.', 'screen_position.',
               'render_blocks.', 'render_norm.', 'rgb128.', 'rgb256.')
    transfer, source_names = {}, {}
    for name, tensor in original.state_dict().items():
        destination = None
        if name.startswith('flow_renderer.'):
            destination = name
        elif name.startswith('camera_backend.'):
            local = name[len('camera_backend.'):]
            if local.startswith(allowed):
                destination = 'vjepa_backend.renderer.'+local
        if destination is not None and destination in target:
            if tensor.shape != target[destination].shape:
                raise ValueError(f'Renderer warm-start shape mismatch: {name}')
            transfer[destination] = tensor
            source_names[destination] = name
    copied = model.load_state_dict(transfer, strict=False)
    if copied.unexpected_keys or any(name.startswith(('flow_renderer.', 'vjepa_backend.renderer.'))
                                     for name in copied.missing_keys):
        raise ValueError('The complete trained renderer/map/geometry did not transfer')
    for name, value in transfer.items():
        if not torch.equal(value, model.state_dict()[name]):
            raise RuntimeError(f'A transferred renderer tensor changed: {name}')
    total = sum(parameter.numel() for parameter in model.parameters())
    if require_profile and abs(total-125_000_000) > 500_000:
        raise ValueError(f'Source/configuration does not produce the selected125M generator: {total}')
    transferred_parameters = sum(parameter.numel() for name, parameter in model.named_parameters() if name in transfer)
    retained_source = set(source_names.values())
    discarded_parameters = sum(parameter.numel() for name, parameter in original.named_parameters() if name not in retained_source)
    report = {'architecture': 'compact_vjepa2_ac_game_adaptation',
              'source_checkpoint': str(Path(source).resolve()),
              'source_training_steps': int(metadata.get('training_steps', 0)),
              'generator_parameters': total, 'transferred_parameters': transferred_parameters,
              'new_latent_dynamics_parameters': total-transferred_parameters,
              'discarded_legacy_parameters': discarded_parameters,
              'transferred_tensor_count': len(transfer), 'all_transferred_tensors_exact': True,
              'renderer_function_preserved_at_zero_latent_features': True,
              'latent_encoder_reinitialized': True, 'action_predictor_reinitialized': True,
              'official_pretrained_weights_loaded': False, 'ema_target_included': False,
              'discriminator_included': False,
              'latent_width': model.core.width, 'visual_tokens_per_frame': model.core.visual_tokens,
              'conditioning_tokens_per_frame': 2, 'runtime_memory_tokens': model.core.memory_tokens,
              'context_frames': model.core.context_frames,
              'created_at': datetime.now(timezone.utc).isoformat()}
    excluded = {'model', 'optimizer', 'adversarial_training', 'teacher_quality_gate', 'teacher_plateau'}
    payload = {name: value for name, value in metadata.items() if name not in excluded}
    model = model.to(device).eval()
    payload.update(stage='world', config=asdict(model.config), model=model.state_dict(),
        pipeline_phase='vjepa2_ac_initial', vjepa_migration=report,
        optimizer_reset_required=True, latent_training_steps=0,
        quality_status='Renderer warm-started; compact V-JEPA encoder/predictor newly initialized and unverified.')
    return model, payload


def migrate_vjepa_checkpoint(source, output, *, kind='teacher', require_profile=True, **config_overrides):
    """Save a separate inference-generator artifact; source checkpoint remains intact."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if kind != 'teacher':
        raise ValueError('Student V-JEPA is trained separately; this migration preserves the75M teacher renderer')
    if source == output or output.exists():
        raise FileExistsError('V-JEPA migration needs a new output path')
    _, payload = load_vjepa125_from75_checkpoint(source, 'cpu', require_profile=require_profile, **config_overrides)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name+f'.{os.getpid()}.tmp')
    torch.save(payload, temporary)
    for attempt in range(40):
        try:
            os.replace(temporary, output)
            return payload['vjepa_migration']
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.05)


# The renderer is retained at66M; additional capacity belongs to the online
# visual encoder and the action-conditioned predictor, not parameter padding.
# Width512 keeps every learned125M weight transferable and runtime memory264x512.
VJEPA250_PROFILE = {
    'vjepa_width': 512,
    'vjepa_encoder_depth': 32,
    'vjepa_predictor_depth': 23,
    'vjepa_heads': 8,
    'vjepa_readout_mlp_ratio': 7,
    'vjepa_gradient_checkpointing': True,
}

# Continue the same width and token interface while allocating the added
# capacity to executed visual/action transformer blocks and the game readout.
VJEPA500_PROFILE = {
    'vjepa_width': 512,
    'vjepa_encoder_depth': 76,
    'vjepa_predictor_depth': 58,
    'vjepa_heads': 8,
    'vjepa_readout_mlp_ratio': 9,
    'vjepa_gradient_checkpointing': True,
}


def _identity_initialize(block):
    """Net2Deeper: new residual blocks initially compute the exact identity.

    All branches execute and their output projections receive gradients on the
    first backward pass; inner branch weights learn after those projections move.
    """
    with torch.no_grad():
        for layer in (block.attention.proj, block.mlp[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)


def _load_vjepa_expansion(source, device, *, require_profile, profile, source_size,
                          target_size, source_tolerance, target_tolerance, **config_overrides):
    """Expand a trained V-JEPA teacher while preserving its initial function.

    Every original tensor is copied exactly, with the three widened readout
    tensors retained as exact leading slices. Added encoder/predictor blocks are
    residual identities; new readout channels initially have zero output weight.
    This is continued training of the existing compact game adaptation, without
    official pretrained weights, an inference-time EMA or a discriminator.
    """
    from .models import NeuralWorldModel, WorldModelConfig, load_world_checkpoint
    original, metadata = load_world_checkpoint(source, 'cpu')
    if original.config.architecture != 'vjepa2_ac' or original.config.kind != 'teacher':
        raise ValueError(f'{target_size}M expansion requires a compact V-JEPA teacher checkpoint')
    source_total = sum(parameter.numel() for parameter in original.parameters())
    if require_profile and abs(source_total-source_size*1_000_000) > source_tolerance:
        raise ValueError(f'{target_size}M expansion expected the {source_size}M source teacher, got {source_total}')
    config = asdict(original.config)
    config.update(profile)
    config.update(config_overrides)
    model = NeuralWorldModel(WorldModelConfig(**config))
    if model.core.width != original.core.width:
        raise ValueError('Function-preserving expansion must retain the trained latent width')
    added = {}
    for name in ('encoder', 'predictor'):
        old_part, new_part = getattr(original.core, name), getattr(model.core, name)
        if len(new_part.blocks) < len(old_part.blocks):
            raise ValueError('Function-preserving expansion cannot remove trained blocks')
        added[name] = len(new_part.blocks)-len(old_part.blocks)
        for block in new_part.blocks[len(old_part.blocks):]:
            _identity_initialize(block)
    # Attention/RoPE and temporal packing must retain their original semantics.
    for name in ('vjepa_patch_size', 'vjepa_context_frames', 'vjepa_mlp_ratio'):
        if getattr(model.config, name) != getattr(original.config, name):
            raise ValueError(f'Function-preserving expansion cannot change {name}')
    if model.core.encoder.blocks[0].attention.heads != original.core.encoder.blocks[0].attention.heads:
        raise ValueError('Function-preserving expansion cannot change the attention heads')
    old_state, new_state = original.state_dict(), model.state_dict()
    widened = {
        'vjepa_backend.readout_neck.1.weight',
        'vjepa_backend.readout_neck.1.bias',
        'vjepa_backend.readout_neck.3.weight',
    }
    exact, grown = [], []
    with torch.no_grad():
        for name, value in old_state.items():
            if name not in new_state:
                raise ValueError(f'Expansion would discard a trained tensor: {name}')
            target = new_state[name]
            if target.shape == value.shape:
                target.copy_(value)
                exact.append(name)
            elif name in widened and target.ndim == value.ndim and all(
                    new >= old for new, old in zip(target.shape, value.shape)):
                # New hidden neurons are active but initially make no output.
                if name.endswith('readout_neck.3.weight'):
                    target.zero_()
                slices = tuple(slice(0, size) for size in value.shape)
                target[slices].copy_(value)
                grown.append(name)
            else:
                raise ValueError(f'Expansion changed an unsupported trained shape: {name}')
    for name, value in old_state.items():
        slices = tuple(slice(0, size) for size in value.shape)
        if not torch.equal(value, new_state[name][slices]):
            raise RuntimeError(f'Expansion failed to retain trained weights exactly: {name}')
    total = sum(parameter.numel() for parameter in model.parameters())
    if require_profile and abs(total-target_size*1_000_000) > target_tolerance:
        raise ValueError(f'Configuration does not produce the selected {target_size}M generator: {total}')
    if total <= source_total:
        raise ValueError('Expansion must add trainable generator capacity')
    report = {
        'architecture': f'compact_vjepa2_ac_game_adaptation_{target_size}m',
        'source_checkpoint': str(Path(source).resolve()),
        'source_training_steps': int(metadata.get('training_steps', 0)),
        'source_global_training_step': metadata.get('global_training_step'),
        'source_generator_parameters': source_total,
        'generator_parameters': total,
        'transferred_parameters': source_total,
        'new_parameters': total-source_total,
        'exact_shape_tensor_count': len(exact),
        'widened_tensor_count': len(grown),
        'widened_tensors': grown,
        'all_original_weight_values_exact': True,
        'all_exact_shape_tensors_exact': True,
        'function_preserving_initialization': True,
        'encoder_blocks_added': added['encoder'],
        'predictor_blocks_added': added['predictor'],
        'encoder_depth': len(model.core.encoder.blocks),
        'predictor_depth': len(model.core.predictor.blocks),
        'latent_width': model.core.width,
        'readout_mlp_ratio': model.config.vjepa_readout_mlp_ratio,
        'gradient_checkpointing': model.config.vjepa_gradient_checkpointing,
        'optimizer_reset_required': True,
        'official_pretrained_weights_loaded': False,
        'ema_target_included': False,
        'discriminator_included': False,
        'created_at': datetime.now(timezone.utc).isoformat(),
    }
    # The old quality/evaluation gates describe another architecture artifact.
    # Preserve its provenance and training count without accepting those gates.
    excluded = {'model', 'optimizer', 'adversarial_training', 'teacher_quality_gate',
                'teacher_plateau', 'plateau', 'scaler', 'ema_target_encoder'}
    payload = {name: value for name, value in metadata.items() if name not in excluded}
    model = model.to(device).eval()
    payload.update(stage='world', config=asdict(model.config), model=model.state_dict(),
        pipeline_phase=f'vjepa2_ac_{target_size}m_initial',
        optimizer_reset_required=True, ema_reset_required=True, expansion_training_steps=0,
        quality_status=f'{source_size}M teacher function retained; new {target_size}M capacity requires training and fresh validation.')
    payload[f'vjepa{target_size}_expansion'] = report
    return model, payload


def load_vjepa250_checkpoint(source, device='cpu', *, require_profile=True, **config_overrides):
    """Warm-start a 250M generator from the trained 125M teacher exactly."""
    return _load_vjepa_expansion(source, device, require_profile=require_profile,
        profile=VJEPA250_PROFILE, source_size=125, target_size=250,
        source_tolerance=500_000, target_tolerance=1_000_000, **config_overrides)


def load_vjepa500_checkpoint(source, device='cpu', *, require_profile=True, **config_overrides):
    """Warm-start a 500M generator from all trained 250M weight values.

    Added encoder/predictor blocks initially compute identity; widened readout
    neurons have zero output columns. Optimizer and external EMA are rebuilt
    for the new architecture. No official pretrained weights are introduced.
    """
    return _load_vjepa_expansion(source, device, require_profile=require_profile,
        profile=VJEPA500_PROFILE, source_size=250, target_size=500,
        source_tolerance=1_000_000, target_tolerance=2_500_000, **config_overrides)


def migrate_vjepa250_checkpoint(source, output, *, require_profile=True, **config_overrides):
    """Persist a separate250M teacher checkpoint; never overwrite the source."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.exists():
        raise FileExistsError('250M expansion needs a new output path')
    _, payload = load_vjepa250_checkpoint(source, 'cpu', require_profile=require_profile, **config_overrides)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name+f'.{os.getpid()}.tmp')
    torch.save(payload, temporary)
    for attempt in range(40):
        try:
            os.replace(temporary, output)
            return payload['vjepa250_expansion']
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.05)


def migrate_vjepa500_checkpoint(source, output, *, require_profile=True, **config_overrides):
    """Save a separate 500M teacher artifact without overwriting its 250M source."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.exists():
        raise FileExistsError('500M expansion needs a new output path')
    _, payload = load_vjepa500_checkpoint(source, 'cpu', require_profile=require_profile, **config_overrides)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name+f'.{os.getpid()}.tmp')
    torch.save(payload, temporary)
    for attempt in range(40):
        try:
            os.replace(temporary, output)
            return payload['vjepa500_expansion']
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.05)
