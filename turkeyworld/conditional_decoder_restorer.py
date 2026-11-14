"""Learned native32 correction from current RF codes, RGB, state and raw map.

There is no Source renderer, ray cast, depth/visibility oracle, target RGB or
target mask in this module.  The map is a known material-ID input.  Camera-
relative tile coordinates are ordinary features for learned cross-attention.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F


KIND = 'conditional_decoder_restorer_v1'
SCHEMA = 'quantized_RF_RGB_current_state_raw_material_map_v1'
VERSION = 1


@dataclass(frozen=True)
class ConditionalDecoderRestorerConfig:
    native_size: int = 32
    latent_size: int = 8
    code_channels: int = 16
    map_size: int = 16
    materials: int = 16
    state_dim: int = 20
    width: int = 64
    heads: int = 4
    hidden: int = 128
    depth: int = 2
    rgb_amplitude: float = 1.
    init_seed: int = 173

    def __post_init__(self):
        contract = dict(native_size=32, latent_size=8, code_channels=16,
                        map_size=16, materials=16, state_dim=20)
        if any(type(getattr(self, key)) is not int or getattr(self, key) != value
               for key, value in contract.items()):
            raise ValueError('Conditional decoder requires native32, RF16x8x8, state20 and map16/material16')
        if (any(type(value) is not int or value < 1 for value in
                (self.width, self.heads, self.hidden, self.depth))
                or self.width % self.heads or self.depth != 2
                or type(self.init_seed) is not int or self.init_seed < 0
                or type(self.rgb_amplitude) not in (int, float)
                or not math.isfinite(self.rgb_amplitude) or self.rgb_amplitude != 1.):
            raise ValueError('Conditional decoder needs two valid attention blocks and full-range RGB amplitude1')


class _RestorerBlock(nn.Module):
    """One screen self-attention and one map/state cross-attention, dropout0."""
    def __init__(self, width, heads, hidden):
        super().__init__()
        self.self_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.cross_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))

    def forward(self, queries, memory):
        normalized = self.self_norm(queries)
        queries = queries + self.self_attention(normalized, normalized, normalized, need_weights=False)[0]
        key_value = self.memory_norm(memory)
        queries = queries + self.cross_attention(self.cross_norm(queries), key_value, key_value,
                                                need_weights=False)[0]
        return queries + self.feedforward(self.feedforward_norm(queries))


class ConditionalDecoderRestorer(nn.Module):
    """A zero-output residual appended after the existing native32 decoder.

    Codes and RGB can retain their autograd graph.  The phase owner freezes
    the RF encoder/quantizer and chooses whether its existing decoder trains.
    State and map are current numeric inputs, never future Source targets.
    """
    def __init__(self, config=None, *, seed=None):
        super().__init__()
        if config is None:
            config = ConditionalDecoderRestorerConfig()
        elif isinstance(config, dict):
            config = ConditionalDecoderRestorerConfig(**config)
        if not isinstance(config, ConditionalDecoderRestorerConfig):
            raise TypeError('Expected an explicit ConditionalDecoderRestorerConfig or config dictionary')
        if seed is not None:
            config = replace(config, init_seed=seed)
        self.config = config
        width = config.width
        # Constructor uses private CPU initialization; the caller's training
        # CPU/CUDA random streams do not advance and no GPU is initialized.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.init_seed)
            self.input_projection = nn.Conv2d(config.code_channels + 3 * 4**2, width, 1)
            self.screen_position = nn.Parameter(torch.empty(1, 8 * 8, width))
            self.material_embedding = nn.Embedding(config.materials, width)
            self.map_position = nn.Parameter(torch.empty(1, 16 * 16, width))
            self.map_coordinates = nn.Sequential(nn.Linear(7, width), nn.SiLU(), nn.Linear(width, width))
            self.map_global = nn.Linear(width, width)
            self.map_global_position = nn.Parameter(torch.empty(1, 1, width))
            self.actor_state = nn.Sequential(nn.Linear(20, 2 * width), nn.SiLU(),
                                             nn.Linear(2 * width, 2 * width))
            self.actor_position = nn.Parameter(torch.empty(1, 2, width))
            self.blocks = nn.ModuleList([_RestorerBlock(width, config.heads, config.hidden)
                                         for _ in range(config.depth)])
            self.output_norm = nn.LayerNorm(width)
            self.upsample = nn.Sequential(nn.Conv2d(width, 128, 3, padding=1), nn.PixelShuffle(2), nn.SiLU(),
                nn.Conv2d(32, 64, 3, padding=1), nn.PixelShuffle(2), nn.SiLU())
            self.output = nn.Conv2d(16, 3, 3, padding=1)
            for value in (self.screen_position, self.map_position, self.map_global_position, self.actor_position):
                nn.init.normal_(value, std=.02)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)
        axis = torch.arange(config.map_size, dtype=torch.float32) + .5
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        self.register_buffer('_tile_centers', torch.stack((xx, yy), -1).reshape(1, 256, 2), persistent=False)

    def get_config(self):
        return asdict(self.config)

    def _validate(self, codes, rgb, state, raw_map):
        values = (codes, rgb, state, raw_map)
        if not all(torch.is_tensor(value) for value in values):
            raise TypeError('Conditional decoder inputs must be tensors')
        size = len(rgb) if rgb.ndim else 0
        if (size < 1 or tuple(rgb.shape) != (size, 3, 32, 32)
                or tuple(codes.shape) != (size, 16, 8, 8)
                or tuple(state.shape) != (size, 20) or tuple(raw_map.shape) != (size, 16, 16)
                or not all(value.is_floating_point() for value in (codes, rgb, state))
                or raw_map.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)):
            raise ValueError('Conditional decoder expects RF codes[B16,8,8], RGB[B3,32,32], state[B20], integer map[B16,16]')
        device = self.input_projection.weight.device
        if any(value.device != device for value in values):
            raise ValueError('Conditional decoder inputs and weights must share one device')
        if (not all(bool(torch.isfinite(value).all()) for value in values)
                or bool(((rgb < 0) | (rgb > 1)).any())
                or bool(((raw_map < 0) | (raw_map >= self.config.materials)).any())):
            raise ValueError('Conditional decoder needs finite inputs, RGB[0,1] and material IDs0..15')

    def _memory(self, state, raw_map, dtype):
        count = len(state)
        # All coordinates are features only.  No DDA, sampled ray depth,
        # analytic occlusion/visibility or collision lookup is performed.
        centers = self._tile_centers.float().expand(count, -1, -1)
        pose = state.float()
        relative = (centers - 16 * pose[:, None, :2]) / 16
        sine, cosine = pose[:, 2:3], pose[:, 3:4]
        forward = relative[..., 0] * cosine + relative[..., 1] * sine
        lateral = -relative[..., 0] * sine + relative[..., 1] * cosine
        features = torch.cat((centers / 8 - 1, relative, forward[..., None], lateral[..., None],
                              relative.norm(dim=-1, keepdim=True)), -1)
        tiles = (self.material_embedding(raw_map.long().flatten(1)) + self.map_position
                 + self.map_coordinates(features.to(dtype)))
        global_map = self.map_global(tiles.mean(1, keepdim=True)) + self.map_global_position
        actors = self.actor_state(state.to(dtype)).reshape(count, 2, self.config.width) + self.actor_position
        return torch.cat((tiles, global_map, actors), 1)

    def forward_with_aux(self, codes, rgb, state, raw_map):
        self._validate(codes, rgb, state, raw_map)
        dtype = self.input_projection.weight.dtype
        queries = self.input_projection(torch.cat((codes.to(dtype), F.pixel_unshuffle(rgb.to(dtype), 4)), 1))
        queries = queries.flatten(2).transpose(1, 2) + self.screen_position
        memory = self._memory(state, raw_map, dtype)
        for block in self.blocks:
            queries = block(queries, memory)
        screen = self.output_norm(queries).transpose(1, 2).reshape(len(rgb), self.config.width, 8, 8)
        logits = self.output(self.upsample(screen))
        delta = (logits.tanh() * self.config.rgb_amplitude).to(rgb.dtype)
        image = (rgb + delta).clamp(0, 1)
        if not all(bool(torch.isfinite(value).all()) for value in (logits, delta, image)):
            raise FloatingPointError('Nonfinite conditional decoder output')
        return dict(image=image, delta=delta, residual_logits=logits, screen_tokens=queries)

    def forward(self, codes, rgb, state, raw_map):
        return self.forward_with_aux(codes, rgb, state, raw_map)['image']


def _state_sha256(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def conditional_decoder_restorer_metadata(module, *, include_state_hash=True):
    from .conditional_restorer_7m import LargeConditionalDecoderRestorer, large_conditional_restorer_metadata
    if isinstance(module, LargeConditionalDecoderRestorer):
        return large_conditional_restorer_metadata(module, include_state_hash=include_state_hash)
    from .conditional_restorer_expansion import ExpandedConditionalDecoderRestorer, expanded_conditional_restorer_metadata
    if isinstance(module, ExpandedConditionalDecoderRestorer):
        return expanded_conditional_restorer_metadata(module, include_state_hash=include_state_hash)
    if not isinstance(module, ConditionalDecoderRestorer):
        raise TypeError('Expected the actual ConditionalDecoderRestorer module')
    result = dict(kind=KIND, version=VERSION, schema=SCHEMA, config=module.get_config(),
        parameters=sum(value.numel() for value in module.parameters()),
        parameter_tensors=sum(1 for _ in module.parameters()),
        screen_tokens=64, map_material_tokens=256, pooled_map_tokens=1, actor_state_tokens=2,
        cross_attention_memory_tokens=259, attention_blocks=module.config.depth,
        dropout=0., initialization='zero_RGB_output_layer', rgb_amplitude=module.config.rgb_amplitude,
        input_scope='current_quantized_RF_codes_decoded_RGB_current_state_raw_material_map',
        Source_RGB_inputs=False, Source_masks_inputs=False, future_Source_inputs=False,
        Source_raycast_or_occlusion_oracle=False, learned_RGB_residual=True,
        output='clamp(decoded_RGB_plus_tanh_RGB_residual,0,1)', masks_or_gates_required=False)
    if include_state_hash:
        result['state_sha256'] = _state_sha256(module.state_dict())
    return result


def validate_conditional_decoder_restorer_state(module, state, *, metadata=None):
    """Validate the complete candidate state before touching any module tensor."""
    from .conditional_restorer_7m import LargeConditionalDecoderRestorer, validate_large_conditional_restorer_state
    if isinstance(module, LargeConditionalDecoderRestorer):
        return validate_large_conditional_restorer_state(module, state, metadata=metadata)
    from .conditional_restorer_expansion import ExpandedConditionalDecoderRestorer, validate_expanded_conditional_restorer_state
    if isinstance(module, ExpandedConditionalDecoderRestorer):
        return validate_expanded_conditional_restorer_state(module, state, metadata=metadata)
    if not isinstance(module, ConditionalDecoderRestorer) or not isinstance(state, dict):
        raise ValueError('Conditional decoder state requires its explicit module and complete tensor dictionary')
    expected = module.state_dict()
    if set(state) != set(expected) or any(not torch.is_tensor(value)
            or value.shape != expected[name].shape or value.dtype != expected[name].dtype
            or not bool(torch.isfinite(value).all()) for name, value in state.items()):
        raise ValueError('Conditional decoder state has missing/extra, wrong-shape/dtype or nonfinite tensors')
    state_sha = _state_sha256(state)
    if metadata is not None:
        architecture = conditional_decoder_restorer_metadata(module, include_state_hash=False)
        if (not isinstance(metadata, dict) or set(metadata) != set(architecture) | {'state_sha256'}
                or any(metadata.get(key) != value for key, value in architecture.items())
                or metadata.get('state_sha256') != state_sha):
            raise ValueError('Conditional decoder version/config/schema/inputs or state binding differs')
    return dict(passed=True, state_sha256=state_sha, parameters=sum(p.numel() for p in module.parameters()),
                parameter_tensors=len(list(module.parameters())))


def build_conditional_decoder_restorer(config=None):
    """Explicit architecture factory; old plain configs keep their exact class."""
    from .conditional_restorer_expansion import (
        KIND as expanded_kind, ExpandedConditionalDecoderRestorer, ExpandedConditionalDecoderRestorerConfig)
    from .conditional_restorer_7m import (
        KIND as large_kind, LargeConditionalDecoderRestorer, LargeConditionalDecoderRestorerConfig)
    if isinstance(config, LargeConditionalDecoderRestorerConfig):
        return LargeConditionalDecoderRestorer(config=config)
    if isinstance(config, ExpandedConditionalDecoderRestorerConfig):
        return ExpandedConditionalDecoderRestorer(config=config)
    if isinstance(config, dict) and 'architecture' in config:
        if config['architecture'] == large_kind:
            return LargeConditionalDecoderRestorer(config=config)
        if config['architecture'] != expanded_kind:
            raise ValueError('Unknown conditional decoder restorer architecture')
        return ExpandedConditionalDecoderRestorer(config=config)
    return ConditionalDecoderRestorer(config)


def load_conditional_decoder_restorer(config, state, *, metadata):
    """Construct/load one strictly pinned private module, without advancing RNG."""
    module = build_conditional_decoder_restorer(config)
    validate_conditional_decoder_restorer_state(module, state, metadata=metadata)
    module.load_state_dict(state, strict=True)
    return module
