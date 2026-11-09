"""Conditional flow matching and teacher-coupled rectified-flow rendering.

Teacher: AdaLN-zero conditional image Transformer, logit-normal time sampling,
self-conditioning and Heun integration. Student: a smaller Transformer fitted
to straight noise/teacher-endpoint couplings (reflow), sampled with Euler.
The conditional probability path is x_t=(1-t)*noise+t*image. This implements
velocity regression, not an RGB decoder renamed as a flow model.

References: https://arxiv.org/abs/2210.02747 and https://arxiv.org/abs/2209.03003.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class FlowBlock(nn.Module):
    def __init__(self, width, heads, mlp_ratio=4, mlp_hidden=0):
        super().__init__()
        if mlp_hidden < 0:
            raise ValueError('Flow MLP hidden width must be nonnegative')
        hidden = mlp_hidden or width*mlp_ratio
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(approximate="tanh"),
                                 nn.Linear(hidden, width))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, width * 6))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, tokens, condition):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(condition).chunk(6, -1)
        first = self.norm1(tokens) * (1 + scale1[:, None]) + shift1[:, None]
        attention = self.attention(first, first, first, need_weights=False)[0]
        tokens = tokens + gate1[:, None] * attention
        second = self.norm2(tokens) * (1 + scale2[:, None]) + shift2[:, None]
        return tokens + gate2[:, None] * self.mlp(second)


def _conditional_velocity(network, noisy, time, base_rgb, state, self_condition=None):
    """Shared velocity-network computation; module names remain checkpoint-stable."""
    from .models import render_state_view
    dtype = network.image_input.weight.dtype
    noisy, base_rgb = noisy.to(dtype), base_rgb.to(dtype)
    self_condition = torch.zeros_like(noisy) if self_condition is None else self_condition.to(dtype)
    tokens = network.image_input(torch.cat([noisy, base_rgb*2-1, self_condition], 1))
    tokens = tokens.flatten(2).transpose(1, 2)+network.position
    angles = time.float()[:, None]*network.frequencies[None]*1000
    embedding = torch.cat([angles.sin(), angles.cos()], -1).to(dtype)
    condition = network.time_input(embedding)+network.state_input(render_state_view(state).to(dtype))
    for block in network.blocks:
        tokens = block(tokens, condition)
    shift, scale = network.final_modulation(condition).chunk(2, -1)
    tokens = network.final_norm(tokens)*(1+scale[:, None])+shift[:, None]
    patches = network.output(tokens).reshape(-1, 16, 16, 3, network.patch, network.patch)
    return patches.permute(0, 3, 1, 4, 2, 5).reshape(-1, 3, network.image_size, network.image_size)


class ResidualVelocityAdapter(nn.Module):
    """Additional conditional capacity whose initial velocity is exactly zero.

    Its zero output projection preserves the trained backbone at expansion.
    All attention/MLP parameters participate in subsequent residual learning;
    there are no padding parameters or outputs discarded by the sampler.
    """
    def __init__(self, image_size, width=512, depth=12, heads=8, mlp_ratio=4):
        super().__init__()
        if min(width, depth, heads, mlp_ratio) < 1 or width % heads:
            raise ValueError("adapter dimensions must be positive and width divisible by heads")
        self.image_size, self.patch, self.width = image_size, image_size//16, width
        self.image_input = nn.Conv2d(9, width, self.patch, stride=self.patch)
        self.position = nn.Parameter(torch.randn(1, 256, width)*.02)
        self.state_input = nn.Sequential(nn.Linear(90, width), nn.SiLU(), nn.Linear(width, width))
        self.time_input = nn.Sequential(nn.Linear(64, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList(FlowBlock(width, heads, mlp_ratio) for _ in range(depth))
        self.final_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, width*2))
        self.output = nn.Linear(width, 3*self.patch*self.patch)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer("frequencies", torch.exp(-math.log(10000)*torch.arange(32)/32),
                             persistent=False)

    def forward(self, noisy, time, base_rgb, state, self_condition=None):
        return _conditional_velocity(self, noisy, time, base_rgb, state, self_condition)


class ConditionalFlowRenderer(nn.Module):
    def __init__(self, kind="teacher", image_size=128, mode="flow_matching", steps=0,
                 width=0, depth=0, heads=0, mlp_ratio=4,
                 adapter_width=0, adapter_depth=0, adapter_heads=0, adapter_mlp_ratio=4,
                 mlp_hidden=0):
        super().__init__()
        if kind not in {"teacher", "student"} or mode not in {"flow_matching", "rectified_flow"}:
            raise ValueError("unsupported flow kind/mode")
        if image_size not in {128, 256}:
            raise ValueError("flow image size must be 128 or 256")
        self.kind, self.mode, self.image_size = kind, mode, image_size
        self.steps = steps or (12 if kind == "teacher" else 4)
        self.patch = image_size // 16
        default_width, default_depth, default_heads = (256, 6, 8) if kind == "teacher" else (128, 3, 4)
        width, depth, heads = width or default_width, depth or default_depth, heads or default_heads
        if min(width, depth, heads, mlp_ratio) < 1 or width % heads:
            raise ValueError("flow dimensions must be positive and width divisible by heads")
        if bool(adapter_width) != bool(adapter_depth):
            raise ValueError("adapter_width and adapter_depth must be enabled together")
        self.width = width
        self.image_input = nn.Conv2d(9, width, self.patch, stride=self.patch)
        self.position = nn.Parameter(torch.randn(1, 256, width) * .02)
        self.state_input = nn.Sequential(nn.Linear(90, width), nn.SiLU(), nn.Linear(width, width))
        self.time_input = nn.Sequential(nn.Linear(64, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList(FlowBlock(width, heads, mlp_ratio, mlp_hidden) for _ in range(depth))
        self.final_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, width * 2))
        self.output = nn.Linear(width, 3 * self.patch * self.patch)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        # A stable inference noise field avoids unrelated frame-to-frame noise.
        generator = torch.Generator().manual_seed(91831)
        self.register_buffer("fixed_noise", torch.randn(1, 3, image_size, image_size,
                                                       generator=generator), persistent=True)
        self.register_buffer("frequencies", torch.exp(-math.log(10000) * torch.arange(32) / 32),
                             persistent=False)
        if adapter_width:
            self.residual_adapter = ResidualVelocityAdapter(image_size, adapter_width, adapter_depth,
                adapter_heads or default_heads, adapter_mlp_ratio)

    def velocity(self, noisy, time, base_rgb, state, self_condition=None):
        velocity = _conditional_velocity(self, noisy, time, base_rgb, state, self_condition)
        if hasattr(self, 'residual_adapter'):
            velocity = velocity+self.residual_adapter(noisy, time, base_rgb, state, self_condition)
        return velocity

    def loss(self, target_rgb, base_rgb, state, noise=None, time=None, target_endpoint=None):
        """Regress an actual conditional-path velocity; reflow accepts teacher endpoints."""
        target = target_rgb.float() * 2 - 1 if target_endpoint is None else target_endpoint.float() * 2 - 1
        noise = torch.randn_like(target) if noise is None else noise.float()
        if time is None:
            samples = torch.randn(target.shape[0], device=target.device)
            time = samples.sigmoid() if self.mode == "flow_matching" else torch.rand_like(samples)
        time = time.float().clamp(.001, .999)
        mixture = time[:, None, None, None]
        noisy = (1 - mixture) * noise + mixture * target
        self_condition = None
        # Teacher self-conditioning is learned with the same conditional state.
        if self.kind == "teacher" and self.training and torch.rand((), device=target.device) < .5:
            with torch.no_grad():
                preliminary = self.velocity(noisy, time, base_rgb, state)
                self_condition = (noisy + (1 - mixture) * preliminary.float()).clamp(-1, 1)
        prediction = self.velocity(noisy, time, base_rgb, state, self_condition).float()
        expected = target - noise
        velocity_loss = F.mse_loss(prediction, expected)
        endpoint = noisy + (1 - mixture) * prediction
        reconstruction = F.l1_loss(endpoint, target)
        total = velocity_loss + .1 * reconstruction
        return total, {"flow_velocity": velocity_loss.detach(),
                       "flow_endpoint": reconstruction.detach()}

    def sample(self, base_rgb, state, noise=None, steps=None):
        steps = steps or self.steps
        if steps < 1:
            raise ValueError("flow integration requires at least one step")
        x = (self.fixed_noise.expand(base_rgb.shape[0], -1, -1, -1)
             if noise is None else noise).to(device=base_rgb.device, dtype=torch.float32)
        self_condition = None
        dt = 1. / steps
        for index in range(steps):
            time = torch.full((x.shape[0],), index * dt, device=x.device)
            velocity = self.velocity(x, time, base_rgb, state, self_condition).float()
            proposal = x + dt * velocity
            self_condition = (x + (1 - index * dt) * velocity).clamp(-1, 1)
            if self.kind == "teacher":
                end_time = torch.full_like(time, (index + 1) * dt)
                end_velocity = self.velocity(proposal, end_time, base_rgb, state, self_condition).float()
                x = x + .5 * dt * (velocity + end_velocity)
            else:
                x = proposal
        return ((x + 1) * .5).clamp(0, 1)
