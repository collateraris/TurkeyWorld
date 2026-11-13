"""RGB-only learned object restoration appended to the existing dense EDSR.

No sprite, state, source mask, detector crop or reference image is an inference
input. Source masks supervise the two auxiliary recognition heads only during
training. A zero RGB readout preserves the complete parent SR function.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .models import EDSRUpscaler, SRResidualBlock


class SemanticEDSRUpscaler(EDSRUpscaler):
    def __init__(self, channels=64, blocks=16, scale=4, res_scale=.1,
                 semantic_channels=32, semantic_blocks=4):
        if (type(semantic_channels) is not int or semantic_channels < 1
                or type(semantic_blocks) is not int or semantic_blocks < 1):
            raise ValueError('Semantic SR widths/depth must be positive integers')
        super().__init__(channels, blocks, scale, res_scale)
        self.config.update(architecture='semantic_edsr', semantic_channels=semantic_channels,
                           semantic_blocks=semantic_blocks)
        width = semantic_channels
        self.semantic_input = nn.Conv2d(3, width, 3, padding=1)
        self.semantic_body = nn.Sequential(*[SRResidualBlock(width, res_scale)
                                            for _ in range(semantic_blocks)])
        self.semantic_mask = nn.Conv2d(width, 1, 1)
        self.semantic_presence = nn.Linear(width, 1)
        self.semantic_upsample = nn.Sequential(
            nn.Conv2d(width, width*4, 3, padding=1), nn.PixelShuffle(2), nn.ReLU(),
            nn.Conv2d(width, width*4, 3, padding=1), nn.PixelShuffle(2), nn.ReLU())
        self.semantic_output = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.semantic_output.weight)
        nn.init.zeros_(self.semantic_output.bias)

    def forward_with_aux(self, low_rgb):
        if (low_rgb.ndim != 4 or low_rgb.shape[1] != 3 or low_rgb.shape[-2:] != (32, 32)
                or not low_rgb.is_floating_point()):
            raise ValueError('Semantic SR accepts RGB[B,3,32,32] only')
        base = super().forward(low_rgb)
        features = self.semantic_body(F.relu(self.semantic_input(low_rgb)))
        mask_logits = self.semantic_mask(features)
        presence_logits = self.semantic_presence(features.mean((-2, -1)))
        gate = F.interpolate(mask_logits.sigmoid(), scale_factor=4,
                             mode='bilinear', align_corners=False)
        gate = gate * presence_logits.sigmoid()[:, :, None, None]
        residual = .5 * self.semantic_output(self.semantic_upsample(features)).tanh()
        proposed = base + gate * residual
        bounded = proposed.clamp(0, 1)
        image = bounded.detach()+(proposed-proposed.detach()) if self.training else bounded
        return {'image': image, 'mask_logits_lr': mask_logits,
                'presence_logits': presence_logits, 'object_residual': residual,
                'object_gate_hr': gate}

    def forward(self, low_rgb):
        return self.forward_with_aux(low_rgb)['image']
