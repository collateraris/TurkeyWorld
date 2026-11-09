"""Conditional spectral-normalized PatchGAN auxiliaries for final rendered RGB.

Real and generated images must use the SAME detached base-render condition and
numerical state. Source RGB is a discriminator example/loss target; it is never
a generator condition. Train the discriminator on detached final Heun samples,
then freeze/evaluate it for the generator update. No noisy flow-path images are
substituted for the final generated image.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


class _PatchDiscriminator(nn.Module):
    def __init__(self, input_channels, base_channels):
        super().__init__()
        widths = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        blocks = []
        channels = input_channels
        for index, width in enumerate(widths):
            blocks.append(nn.Sequential(
                spectral_norm(nn.Conv2d(channels, width, 4, stride=2 if index < 3 else 1, padding=1)),
                nn.LeakyReLU(.2, inplace=False)))
            channels = width
        self.blocks = nn.ModuleList(blocks)
        self.logits = spectral_norm(nn.Conv2d(channels, 1, 4, padding=1))
        self.state_projection = spectral_norm(nn.Linear(90, channels, bias=False))

    def forward(self, paired, state):
        features = []
        value = paired
        for block in self.blocks:
            value = block(value)
            features.append(value)
        logits = self.logits(value)
        if state is not None:
            embedding = self.state_projection(state.to(dtype=value.dtype))
            # Match the final PatchGAN head's spatial grid with a local feature
            # average; state projection adds a conditional score to every patch.
            projected = (value * embedding[:, :, None, None]).sum(1, keepdim=True) / math.sqrt(value.shape[1])
            projected = F.avg_pool2d(projected, 4, stride=1, padding=1)
            logits = logits + projected
        return logits, features


class MultiScaleDiscriminator(nn.Module):
    """Two default PatchGANs at input resolution and half resolution.

    Native32 inputs use one scale explicitly, retaining their actual32 pixels
    rather than enlarging them for the critic. Existing64/128 profiles use two.

    Inputs are RGB [B,C,H,W] in the same [0,1] convention as the renderer.
    The base condition may be coarse and is resized after detaching. Optional
    state [B,90] is restricted to render_state_view, excluding invisible HUD
    counters. Only candidate RGB receives input gradients, including for R1.
    """

    def __init__(self, image_channels=3, condition_channels=3, base_channels=32, num_scales=2):
        super().__init__()
        if min(image_channels, condition_channels, base_channels, num_scales) < 1:
            raise ValueError("Discriminator channel counts and scale count must be positive")
        self.image_channels, self.condition_channels = image_channels, condition_channels
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList(
            _PatchDiscriminator(image_channels + condition_channels, base_channels)
            for _ in range(num_scales))

    def forward(self, rgb, condition, state=None):
        if rgb.ndim != 4 or condition.ndim != 4:
            raise ValueError("RGB and base condition must have shape [B,C,H,W]")
        if (rgb.shape[0] != condition.shape[0] or rgb.shape[1] != self.image_channels
                or condition.shape[1] != self.condition_channels):
            raise ValueError("RGB/base batches or channels do not match the discriminator")
        if min(rgb.shape[-2:]) < 32 * 2 ** (self.num_scales - 1):
            raise ValueError("Image is too small for the requested PatchGAN scales")
        condition = condition.detach().to(device=rgb.device, dtype=rgb.dtype)
        if condition.shape[-2:] != rgb.shape[-2:]:
            condition = F.interpolate(condition, rgb.shape[-2:], mode="bilinear", align_corners=False)
        if state is not None:
            if state.shape != (rgb.shape[0], 90):
                raise ValueError("Conditional state must have shape [B,90]")
            from .models import render_state_view
            state = render_state_view(state.detach().to(device=rgb.device, dtype=rgb.dtype))
        paired = torch.cat([rgb * 2 - 1, condition * 2 - 1], dim=1)
        logits, features = [], []
        for index, discriminator in enumerate(self.discriminators):
            score, activations = discriminator(paired, state)
            logits.append(score)
            features.append(activations)
            if index + 1 < self.num_scales:
                paired = F.avg_pool2d(paired, 3, stride=2, padding=1, count_include_pad=False)
        return {"logits": logits, "features": features}


def _logits(output):
    values = output["logits"] if isinstance(output, dict) else output
    if torch.is_tensor(values):
        values = [values]
    if not isinstance(values, (tuple, list)) or not values or not all(torch.is_tensor(value) for value in values):
        raise ValueError("Expected nonempty discriminator logits")
    return values


def discriminator_hinge_loss(real, fake):
    """Mean max(0,1-D(real)) + max(0,1+D(fake)), equally weighting scales."""
    real_logits, fake_logits = _logits(real), _logits(fake)
    if len(real_logits) != len(fake_logits):
        raise ValueError("Real/fake discriminator scale counts differ")
    return torch.stack([F.relu(1 - first.float()).mean() + F.relu(1 + second.float()).mean()
                        for first, second in zip(real_logits, fake_logits)]).mean()


def generator_hinge_loss(fake):
    """Generator maximizes the discriminator's score for final generated RGB."""
    return -torch.stack([value.float().mean() for value in _logits(fake)]).mean()


def feature_matching_loss(real, fake):
    """Mean L1 intermediate-feature error; source features are detached."""
    real_features, fake_features = real["features"], fake["features"]
    if len(real_features) != len(fake_features):
        raise ValueError("Real/fake feature scale counts differ")
    losses = []
    for first_scale, second_scale in zip(real_features, fake_features):
        if len(first_scale) != len(second_scale):
            raise ValueError("Real/fake feature layer counts differ")
        for first, second in zip(first_scale, second_scale):
            if first.shape != second.shape:
                raise ValueError("Real/fake feature shapes differ")
            losses.append(F.l1_loss(second.float(), first.detach().float()))
    if not losses:
        raise ValueError("Feature matching requires nonempty intermediate features")
    return torch.stack(losses).mean()


def r1_penalty(real_logits, real_images):
    """R1 on per-image mean patch scores, with float32 gradient accumulation.

    For lazy R1, compute the real discriminator forward in FP32 with autocast
    disabled and real_images.float().requires_grad_(True). The caller applies
    gamma/2 and the lazy-update interval; neither factor is hidden here.
    """
    if not real_images.requires_grad:
        raise ValueError("R1 requires real_images.requires_grad=True")
    values = _logits(real_logits)
    batch = real_images.shape[0]
    score = torch.stack([value.float().reshape(batch, -1).mean(-1) for value in values]).mean(0).sum()
    gradients = torch.autograd.grad(score, real_images, create_graph=True, retain_graph=True,
                                    only_inputs=True)[0]
    return gradients.float().square().flatten(1).sum(-1).mean()
