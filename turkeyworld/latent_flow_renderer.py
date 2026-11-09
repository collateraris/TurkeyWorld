"""Conditional continuous-latent flow through a frozen, locally trained VQGAN.

The interface accepts/returns RGB so existing image losses and teacher reflow
continue to work. Flow velocity is learned over continuous VQ embedding grids;
only the terminal grid is quantized and decoded. Codec weights are included in
the generator, frozen and kept in eval mode. Loading a codec is explicit; its
qualification decision belongs to the experiment controller, not this module.
"""
from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .flow_renderer import FlowBlock
from .vqgan import VQGAN, VQGANConfig


def _velocity(network, noisy, time, base_latents, state, self_condition=None, map_condition=None):
    from .models import render_state_view
    dtype = network.image_input.weight.dtype
    noisy, base_latents = noisy.to(dtype), base_latents.to(dtype)
    previous = torch.zeros_like(noisy) if self_condition is None else self_condition.to(dtype)
    tokens = network.image_input(torch.cat([noisy, base_latents, previous], 1))
    tokens = tokens.flatten(2).transpose(1, 2)+network.position
    angles = time.float()[:, None]*network.frequencies[None]*1000
    time_embedding = torch.cat([angles.sin(), angles.cos()], -1).to(dtype)
    condition = network.time_input(time_embedding)+network.state_input(render_state_view(state).to(dtype))
    for block in network.blocks:
        tokens = block(tokens, condition)
    if hasattr(network, 'spatial_condition'):
        tokens = tokens+network.spatial_condition(tokens, state)
    if hasattr(network, 'wall_condition'):
        tokens = tokens+network.wall_condition(tokens, state, map_condition)
    shift, scale = network.final_modulation(condition).chunk(2, -1)
    tokens = network.final_norm(tokens)*(1+scale[:, None])+shift[:, None]
    patches = network.output(tokens).reshape(-1, network.grid, network.grid,
                                             network.latent_channels, network.patch, network.patch)
    return patches.permute(0, 3, 1, 4, 2, 5).reshape(-1, network.latent_channels,
                                                  network.latent_size, network.latent_size)


class LatentVelocityNetwork(nn.Module):
    """Conditional DiT with dynamic latent channels/grid and a zero output head."""
    def __init__(self, latent_channels, latent_size, width, depth, heads, mlp_ratio=4, mlp_hidden=0):
        super().__init__()
        if min(latent_channels, latent_size, width, depth, heads, mlp_ratio) < 1 or width % heads:
            raise ValueError('Latent flow dimensions must be positive and width divisible by heads')
        if latent_size % 4:
            raise ValueError('The latent grid must divide four-pixel transformer patches')
        self.latent_channels, self.latent_size = latent_channels, latent_size
        self.patch, self.grid, self.width = 4, latent_size//4, width
        self.image_input = nn.Conv2d(latent_channels*3, width, self.patch, stride=self.patch)
        self.position = nn.Parameter(torch.randn(1, self.grid*self.grid, width)*.02)
        self.state_input = nn.Sequential(nn.Linear(90, width), nn.SiLU(), nn.Linear(width, width))
        self.time_input = nn.Sequential(nn.Linear(64, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList(FlowBlock(width, heads, mlp_ratio, mlp_hidden) for _ in range(depth))
        self.final_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, width*2))
        self.output = nn.Linear(width, latent_channels*self.patch*self.patch)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer('frequencies', torch.exp(-math.log(10000)*torch.arange(32)/32), persistent=False)

    def forward(self, noisy, time, base_latents, state, self_condition=None, map_condition=None):
        return _velocity(self, noisy, time, base_latents, state, self_condition, map_condition)

    def enable_spatial_conditioning(self, image_size):
        if not hasattr(self, 'spatial_condition'):
            # Adding an opt-in bridge must not shift the initialization of old modules.
            with torch.random.fork_rng(devices=[]):
                self.spatial_condition = SpatialFlowCondition(self.width, self.grid, image_size)

    def enable_wall_conditioning(self):
        if not hasattr(self, 'wall_condition'):
            with torch.random.fork_rng(devices=[]):
                self.wall_condition = WallFlowCondition(self.width)


class SpatialFlowCondition(nn.Module):
    """Nine identity-preserving participant conditions read by latent patches.

    Uses current continuous coordinates, observed direction/validity, relative
    pose, HP/life/timers and health-derived hit-progress descriptors. Projection
    only supplies an image-space prior; no engine visibility or future RGB enters.
    """
    def __init__(self, width, grid, image_size, rank=64):
        super().__init__()
        from .spatial_entities import SPATIAL_FEATURE_DIM
        self.grid, self.image_size, self.rank = grid, image_size, rank
        self.features = nn.Sequential(nn.Linear(SPATIAL_FEATURE_DIM+6, rank), nn.SiLU(), nn.Linear(rank, width))
        self.identity = nn.Embedding(9, width)
        self.entity_norm, self.query_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.query, self.key, self.value = (nn.Linear(width, rank, bias=False) for _ in range(3))
        self.output = nn.Linear(rank, width)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        y, x = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing='ij')
        boxes = torch.stack([x, y, x+1, y+1], -1).reshape(-1, 4).float()/grid
        self.register_buffer('query_boxes', boxes, persistent=False)

    def entity_tokens(self, state):
        from .spatial_entities import entity_spatial_features
        from .combat_objectives import combat_count_features
        features = torch.cat([entity_spatial_features(state.float()), combat_count_features(state.float())], -1)
        encoded = self.features(features.to(self.features[0].weight.dtype))
        identities = self.identity(torch.arange(9, device=state.device))[None]
        return self.entity_norm(encoded+identities.to(encoded.dtype))

    def projection_weights(self, state):
        from .camera import project_sprite_bounds
        boxes = project_sprite_bounds(state.float(), image_size=self.image_size)['boxes']
        patch = self.query_boxes.to(state.device)[None, :, None]
        lower = torch.maximum(patch[..., :2], boxes[:, None, :, :2])
        upper = torch.minimum(patch[..., 2:], boxes[:, None, :, 2:])
        overlap = (upper-lower).clamp_min(0).prod(-1)*(self.grid*self.grid)
        player = torch.ones(state.shape[0], self.grid*self.grid, 1, device=state.device, dtype=overlap.dtype)
        return torch.cat([player, overlap], -1)

    def forward(self, queries, state):
        entities = self.entity_tokens(state)
        query = self.query(self.query_norm(queries)).float()
        key, value = self.key(entities).float(), self.value(entities).float()
        weights = self.projection_weights(state)
        bias = torch.where(weights > 0, weights.clamp_min(1e-6).log(), torch.full_like(weights, -10000.))
        attention = (torch.matmul(query, key.transpose(-1, -2))/math.sqrt(self.rank)+bias).softmax(-1)
        context = torch.matmul(attention, value)
        return self.output(context.to(self.output.weight.dtype)).to(queries.dtype)


def _raw_map(grid, *, device=None):
    """Validate raw material IDs without guessing a batch subset or map origin."""
    if (not torch.is_tensor(grid) or grid.ndim not in (2, 3) or grid.shape[-2:] != (16, 16)
            or (grid.ndim == 3 and grid.shape[0] < 1) or grid.is_floating_point() or grid.is_complex()):
        raise ValueError('Wall map_condition must be integer[16,16] or[B,16,16]')
    value = grid.detach().to(device=device, dtype=torch.long)
    if bool(((value < 0) | (value > 15)).any()):
        raise ValueError('Wall map material IDs must be in0..15 (0=floor)')
    return value


def _map_batch(grid, batch, *, device=None):
    value = _raw_map(grid, device=device)
    if value.ndim == 2:
        return value[None].expand(batch, -1, -1)
    if value.shape[0] != batch:
        raise ValueError('Wall map_condition batch must match the current call exactly')
    return value


class WallFlowCondition(nn.Module):
    """Image patches attend to known wall coordinates/materials plus a null key.

    Geometry includes normalized tile center/bounds and continuous camera-local
    forward/right/distance/bearing. Every wall is retained, including behind
    the player. This is learned conditioning, not raycasting or an occlusion
    oracle. Floor keys are masked; the always-valid null key keeps empty maps
    finite. Zero output initialization preserves the previous flow function.
    """
    def __init__(self, width, rank=64):
        super().__init__()
        self.rank = rank
        self.features = nn.Sequential(nn.Linear(11, rank), nn.SiLU(), nn.Linear(rank, width))
        self.material = nn.Embedding(15, width)
        self.null = nn.Parameter(torch.zeros(1, 1, width))
        self.wall_norm, self.query_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.query, self.key, self.value = (nn.Linear(width, rank, bias=False) for _ in range(3))
        self.output = nn.Linear(rank, width)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        y, x = torch.meshgrid(torch.arange(16), torch.arange(16), indexing='ij')
        bounds = torch.stack([x, y, x+1, y+1], -1).reshape(256, 4).float()/16
        centers = (bounds[:, :2]+bounds[:, 2:])*.5
        self.register_buffer('geometry', torch.cat([centers, bounds], -1), persistent=False)

    def wall_tokens(self, state, map_condition):
        grid = _map_batch(map_condition, len(state), device=state.device)
        materials = grid.flatten(1)
        values = state.float()
        geometry = self.geometry.to(device=state.device).float()
        delta = geometry[None, :, :2]-values[:, None, :2]
        heading = values[:, 2:4]
        norm = heading.norm(dim=-1, keepdim=True)
        fallback = torch.cat([torch.zeros_like(norm), torch.ones_like(norm)], -1)
        heading = torch.where(norm > 1e-6, heading/norm.clamp_min(1e-6), fallback)
        sine, cosine = heading.unbind(-1)
        forward = delta[:, :, 0]*cosine[:, None]+delta[:, :, 1]*sine[:, None]
        right = -delta[:, :, 0]*sine[:, None]+delta[:, :, 1]*cosine[:, None]
        distance = delta.norm(dim=-1)
        relative = torch.stack([forward, right, distance,
                                right/distance.clamp_min(1e-6), forward/distance.clamp_min(1e-6)], -1)
        descriptor = torch.cat([geometry[None].expand(len(state), -1, -1), relative], -1)
        encoded = self.features(descriptor.to(self.features[0].weight.dtype))
        encoded = encoded+self.material((materials-1).clamp_min(0)).to(encoded.dtype)
        tokens = torch.cat([encoded, self.null.expand(len(state), -1, -1).to(encoded.dtype)], 1)
        valid = torch.cat([materials > 0, torch.ones(len(state), 1, dtype=torch.bool, device=state.device)], 1)
        return self.wall_norm(tokens), valid

    def forward(self, queries, state, map_condition):
        if map_condition is None:
            raise ValueError('Wall-conditioned Flow requires an initialization map or explicit map_condition')
        walls, valid = self.wall_tokens(state, map_condition)
        query = self.query(self.query_norm(queries)).float()
        key, value = self.key(walls).float(), self.value(walls).float()
        scores = (torch.matmul(query, key.transpose(-1, -2))/math.sqrt(self.rank)).masked_fill(~valid[:, None], -torch.inf)
        context = torch.matmul(scores.softmax(-1), value)
        return self.output(context.to(self.output.weight.dtype)).to(queries.dtype)


class LatentFlowRenderer(LatentVelocityNetwork):
    def __init__(self, kind='teacher', image_size=128, mode='flow_matching', steps=0,
                 width=0, depth=0, heads=0, mlp_ratio=4,
                 adapter_width=0, adapter_depth=0, adapter_heads=0, adapter_mlp_ratio=4,
                 codec_config=None, mlp_hidden=0, spatial_conditioning=False):
        if kind not in {'teacher', 'student'} or mode not in {'flow_matching', 'rectified_flow'}:
            raise ValueError('Unsupported latent flow kind/mode')
        if steps < 0 or min(width, depth, heads, adapter_width, adapter_depth, adapter_heads) < 0:
            raise ValueError('Steps and optional flow dimensions must be nonnegative')
        values = asdict(codec_config) if isinstance(codec_config, VQGANConfig) else dict(codec_config or {})
        if values.get('image_size', image_size) != image_size:
            raise ValueError('Codec and renderer image sizes must match')
        values['image_size'] = image_size
        config = VQGANConfig(**values)
        if config.embedding_dim > 3*config.downsample_factor**2:
            raise ValueError('RGB noise does not contain enough independent polyphase channels for this codec')
        defaults = (256, 6, 8) if kind == 'teacher' else (128, 3, 4)
        super().__init__(config.embedding_dim, config.latent_size, width or defaults[0],
                         depth or defaults[1], heads or defaults[2], mlp_ratio, mlp_hidden)
        self.kind, self.mode, self.image_size = kind, mode, image_size
        self.steps = steps or (12 if kind == 'teacher' else 4)
        self.codec = VQGAN(config).eval().requires_grad_(False)
        self.register_buffer('codec_loaded', torch.tensor(False, dtype=torch.bool))
        self.register_buffer('latent_mean', torch.zeros(1, config.embedding_dim, 1, 1))
        self.register_buffer('latent_std', torch.ones(1, config.embedding_dim, 1, 1))
        # RGB-shaped fixed noise preserves existing teacher/student reflow APIs.
        generator = torch.Generator().manual_seed(91831)
        self.register_buffer('fixed_noise', torch.randn(1, 3, image_size, image_size, generator=generator))
        if bool(adapter_width) != bool(adapter_depth):
            raise ValueError('Adapter width and depth must be enabled together')
        if adapter_width:
            self.residual_adapter = LatentVelocityNetwork(config.embedding_dim, config.latent_size,
                adapter_width, adapter_depth, adapter_heads or defaults[2], adapter_mlp_ratio)
        if spatial_conditioning:
            self.enable_spatial_conditioning(image_size)
            self.enable_wall_conditioning()
            self.register_buffer('_map_condition', torch.empty(0, dtype=torch.long), persistent=False)
            if hasattr(self, 'residual_adapter'):
                self.residual_adapter.enable_spatial_conditioning(image_size)
                self.residual_adapter.enable_wall_conditioning()

    @property
    def uses_wall_conditioning(self):
        return hasattr(self, 'wall_condition')

    def set_map_condition(self, grid):
        """Cache a detached level map; explicit per-call subsets take precedence."""
        if self.uses_wall_conditioning:
            self._map_condition = _raw_map(grid, device=self.image_input.weight.device).clone()
        return self

    def _resolve_map_condition(self, map_condition, batch, device):
        if not self.uses_wall_conditioning:
            return None
        if map_condition is None:
            map_condition = getattr(self, '_map_condition', None)
            if map_condition is None or map_condition.numel() == 0:
                raise ValueError('Wall-conditioned Flow requires encode_map or explicit map_condition')
        # A per-call snapshot is an explicit checkpoint argument, never a
        # mutable cache lookup during backward recomputation.
        return _map_batch(map_condition, batch, device=device).clone()

    def train(self, mode=True):
        super().train(mode)
        self.codec.eval().requires_grad_(False)
        return self

    def requires_grad_(self, requires_grad=True):
        super().requires_grad_(requires_grad)
        self.codec.requires_grad_(False)
        return self

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Earlier latent-renderer checkpoints used raw embedding coordinates.
        # Their absent normalization pair therefore means exact identity.
        if prefix+'latent_mean' not in state_dict and prefix+'latent_std' not in state_dict:
            state_dict[prefix+'latent_mean'] = torch.zeros_like(self.latent_mean)
            state_dict[prefix+'latent_std'] = torch.ones_like(self.latent_std)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def _check_codec(self):
        if not bool(self.codec_loaded):
            raise RuntimeError('Load the trained VQGAN codec before latent flow loss or sampling')
        # A parent module may have toggled every parameter directly.
        self.codec.eval().requires_grad_(False)

    def load_codec(self, checkpoint):
        """Load locally trained codec weights; the controller verifies qualification."""
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False) if isinstance(
            checkpoint, (str, Path)) else checkpoint
        if not isinstance(saved, dict):
            raise ValueError('Codec checkpoint must be a state or checkpoint dictionary')
        config = saved.get('codec_config', saved.get('config'))
        if config is not None:
            if asdict(VQGANConfig(**config)) != self.codec.get_config():
                raise ValueError('Trained codec configuration does not match latent flow configuration')
        normalization = saved.get('latent_normalization')
        if normalization is None:
            mean, std = torch.zeros(self.latent_channels), torch.ones(self.latent_channels)
        else:
            if not isinstance(normalization, dict) or 'mean' not in normalization or 'std' not in normalization:
                raise ValueError('Latent normalization must contain per-channel mean and std')
            mean = torch.as_tensor(normalization['mean'], dtype=torch.float32)
            std = torch.as_tensor(normalization['std'], dtype=torch.float32)
            if (mean.shape != (self.latent_channels,) or std.shape != (self.latent_channels,)
                    or not bool(torch.isfinite(mean).all() and torch.isfinite(std).all())
                    or not bool((std > 0).all())):
                raise ValueError('Latent normalization needs finite channel vectors with positive std')
        state = saved.get('codec_state_dict', saved.get('state_dict', saved.get('model', saved)))
        self.codec.load_state_dict(state, strict=True)
        self.latent_mean.copy_(mean.reshape_as(self.latent_mean))
        self.latent_std.copy_(std.reshape_as(self.latent_std))
        self.codec.eval().requires_grad_(False)
        self.codec_loaded.fill_(True)
        return self

    def normalize_latents(self, value):
        return (value.float()-self.latent_mean.float())/self.latent_std.float()

    def denormalize_latents(self, value):
        return value.float()*self.latent_std.float()+self.latent_mean.float()

    def _noise(self, noise, batch, device):
        value = self.fixed_noise if noise is None else noise
        if value.ndim != 4 or value.shape[0] not in (1, batch):
            raise ValueError('Flow noise must be a batch-aligned RGB or latent tensor')
        if tuple(value.shape[1:]) == (3, self.image_size, self.image_size):
            # Pixel-unshuffle is a permutation: the selected channels retain
            # independent N(0,1) coordinates without encoding noise as an image.
            value = F.pixel_unshuffle(value, self.codec.config.downsample_factor)[:, :self.latent_channels]
        elif tuple(value.shape[1:]) != (self.latent_channels, self.latent_size, self.latent_size):
            raise ValueError('Noise shape does not match RGB or the codec latent grid')
        return value.to(device=device, dtype=torch.float32).expand(batch, -1, -1, -1)

    def velocity(self, noisy, time, base_latents, state, self_condition=None, map_condition=None):
        maps = self._resolve_map_condition(map_condition, len(state), state.device)
        result = _velocity(self, noisy, time, base_latents, state, self_condition, maps)
        if hasattr(self, 'residual_adapter'):
            if self.uses_wall_conditioning:
                result = result+self.residual_adapter(noisy, time, base_latents, state, self_condition, maps)
            else:
                result = result+self.residual_adapter(noisy, time, base_latents, state, self_condition)
        return result

    def forward(self, noisy, time, base_latents, state, self_condition=None, map_condition=None):
        if self.uses_wall_conditioning:
            return self.velocity(noisy, time, base_latents, state, self_condition, map_condition)
        return self.velocity(noisy, time, base_latents, state, self_condition)

    def loss(self, target_rgb, base_rgb, state, noise=None, time=None, target_endpoint=None, *, pixel_weights=None,
             detach_base=False, map_condition=None):
        """Continuous-latent velocity regression, including same-noise RGB reflow."""
        self._check_codec()
        maps = self._resolve_map_condition(map_condition, len(state), state.device)
        with torch.no_grad():
            target = self.normalize_latents(self.codec.encode(
                target_rgb if target_endpoint is None else target_endpoint)['quantized'])
        # Base may learn solely from its direct RGB objective. Detaching here
        # preserves the exact conditional flow forward values while preventing
        # denoising loss from overwhelming reconstruction of the condition.
        condition_rgb = base_rgb.detach() if detach_base else base_rgb
        base = self.normalize_latents(self.codec.encode(condition_rgb)['quantized'])
        noise = self._noise(torch.randn_like(target) if noise is None else noise, target.shape[0], target.device)
        if time is None:
            values = torch.randn(target.shape[0], device=target.device)
            time = values.sigmoid() if self.mode == 'flow_matching' else torch.rand_like(values)
        time = time.to(device=target.device, dtype=torch.float32)
        if time.shape != (target.shape[0],):
            raise ValueError('Flow time must have one value per image')
        time = time.clamp(.001, .999)
        fraction = time[:, None, None, None]
        noisy = (1-fraction)*noise+fraction*target
        previous = None
        if self.kind == 'teacher' and self.training and bool(torch.rand((), device=target.device) < .5):
            with torch.no_grad():
                first = (self.velocity(noisy, time, base, state, None, maps) if self.uses_wall_conditioning
                         else self.velocity(noisy, time, base, state)).float()
                previous = noisy+(1-fraction)*first
        prediction = (self.velocity(noisy, time, base, state, previous, maps) if self.uses_wall_conditioning
                      else self.velocity(noisy, time, base, state, previous)).float()
        endpoint = noisy+(1-fraction)*prediction
        if pixel_weights is not None:
            from .object_losses import weighted_mse, weighted_image_l1
            velocity_loss = weighted_mse(prediction, target-noise, pixel_weights)
            endpoint_loss = weighted_image_l1(endpoint, target, pixel_weights)
        else:
            velocity_loss = F.mse_loss(prediction, target-noise)
            endpoint_loss = F.l1_loss(endpoint, target)
        return velocity_loss+.1*endpoint_loss, {
            'flow_velocity': velocity_loss.detach(), 'flow_endpoint': endpoint_loss.detach(),
            'flow_latent_std': target.detach().std(unbiased=False),
        }

    def latent_sample(self, base_rgb, state, noise=None, steps=None, *, checkpoint_velocity=False, map_condition=None):
        """Return continuous normalized latents, before unscaling/quantization.

        Optional activation recomputation checkpoints each ODE velocity call
        during gradient-enabled training. Evaluation/no-grad keep the ordinary
        deterministic sampler, including the teacher's full Heun endpoint.
        """
        self._check_codec()
        maps = self._resolve_map_condition(map_condition, len(state), state.device)
        steps = self.steps if steps is None else steps
        if not isinstance(steps, int) or steps < 1:
            raise ValueError('Flow integration requires a positive integer number of steps')
        base = self.normalize_latents(self.codec.encode(base_rgb)['quantized'])
        current = self._noise(noise, base_rgb.shape[0], base_rgb.device)
        previous, dt = None, 1/steps
        recompute = bool(checkpoint_velocity and self.training and torch.is_grad_enabled())

        def evaluate_velocity(*inputs):
            if self.uses_wall_conditioning:
                inputs = (*inputs, maps)
            output = (checkpoint(self.velocity, *inputs, use_reentrant=False, preserve_rng_state=True)
                      if recompute else self.velocity(*inputs))
            return output.float()

        for index in range(steps):
            time = torch.full((current.shape[0],), index*dt, device=current.device)
            velocity = evaluate_velocity(current, time, base, state, previous)
            proposal = current+dt*velocity
            previous = current+(1-index*dt)*velocity
            if self.kind == 'teacher':
                end_velocity = evaluate_velocity(proposal, time+dt, base, state, previous)
                current = current+.5*dt*(velocity+end_velocity)
            else:
                current = proposal
        return current

    def sample(self, base_rgb, state, noise=None, steps=None, *, checkpoint_velocity=False, map_condition=None):
        continuous = self.latent_sample(base_rgb, state, noise=noise, steps=steps,
                                        checkpoint_velocity=checkpoint_velocity, map_condition=map_condition)
        terminal = self.codec.quantize(self.denormalize_latents(continuous))['quantized']
        return self.codec.decode(terminal)


ConditionalLatentFlowRenderer = LatentFlowRenderer
