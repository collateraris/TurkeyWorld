"""Synthesize combat detail from neural RGB and the scene's own predicted state.

The per-pixel decoder stores network weights only. The camera projection supplies
a local coordinate system for detail on a moving character; no combat art or
reference visibility buffer is consulted by this module.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .camera import project_sprite_bounds
from .enemy_animation import enemy_animation_index


class CombatAppearanceSR(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "frequencies", torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0, 32.0]), persistent=False
        )
        self.decoder = nn.Sequential(
            nn.Linear(39, 96),
            nn.SiLU(),
            nn.Linear(96, 96),
            nn.SiLU(),
            nn.Linear(96, 64),
            nn.SiLU(),
            nn.Linear(64, 4),
        )
        self.weapon_phase = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 3))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def phase_logits(self, state):
        return self.weapon_phase(state[:, 7:8].to(self.weapon_phase[0].weight))

    def decode(self, rgb_low, rgb_prior, state, x, y):
        """RGB values [B,N,3], integer output pixel coordinates [B,N]."""
        state = state.float()
        phase = enemy_animation_index(state)
        projection = project_sprite_bounds(state, image_size=128)
        left, top, right, _ = (projection["unclipped_boxes"][:, 0] * 128).unbind(-1)
        size = (right - left).clamp_min(1.0)
        forward = projection["forward"][:, 0]
        depth = forward.clamp_min(0.12)
        u = (x.float() - left[:, None] + 0.5) / size[:, None]
        v = (y.float() - top[:, None] + 0.5) / size[:, None]
        coords = torch.stack((u, v), dim=-1)
        active = (phase >= 3) & (forward > 0.12)
        # Include a narrow rim to repair blur along silhouette edges.
        support = active[:, None] & ((coords >= -0.06) & (coords <= 1.06)).all(-1)
        uv = coords.clamp(-0.06, 1.06)
        trig = uv.unsqueeze(-1) * (2 * math.pi) * self.frequencies.float()
        phase_features = F.one_hot((phase - 3).clamp(0, 3), 4).float()
        camera = torch.cat(
            (
                phase_features,
                (depth / 16).clamp(0, 1)[:, None],
                (size / 128).clamp(max=5)[:, None],
                state[:, 15:16],
            ),
            dim=-1,
        )
        features = torch.cat(
            (
                uv,
                trig.sin().flatten(-2),
                trig.cos().flatten(-2),
                rgb_low.float(),
                rgb_prior.float(),
                camera[:, None].expand(-1, x.shape[1], -1),
            ),
            dim=-1,
        )
        raw = self.decoder(features.to(self.decoder[0].weight))
        residual = raw[..., :3].float() * support[..., None]
        return (rgb_prior.float() + residual).clamp(0, 1).to(rgb_prior.dtype), raw[..., 3], support

    def forward(self, low, prior, state):
        if state is None:
            return prior
        active = enemy_animation_index(state) >= 3
        if not bool(active.any()):
            return prior
        indexes = torch.nonzero(active, as_tuple=True)[0]
        base = F.interpolate(low[indexes].float(), size=(128, 128), mode="bilinear", align_corners=False)
        pixels = torch.arange(128 * 128, device=state.device)
        x = (pixels % 128).expand(len(indexes), -1)
        y = (pixels // 128).expand_as(x)
        rgb, _, _ = self.decode(
            base.flatten(2).transpose(1, 2),
            prior[indexes].flatten(2).transpose(1, 2),
            state[indexes],
            x,
            y,
        )
        return prior.index_copy(0, indexes, rgb.transpose(1, 2).reshape(-1, 3, 128, 128))
