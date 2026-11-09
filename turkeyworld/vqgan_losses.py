"""From-scratch VQGAN discriminator and reconstruction/adversarial objectives.

The adaptive last-decoder-layer gradient ratio, delayed GAN factor and hinge
normalization follow CompVis/taming-transformers:
https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py
The RGB-only PatchGAN uses the same receptive-field schedule, with GroupNorm
instead of BatchNorm/ActNorm for small game batches. Pixel-art edge loss is
an explicit project adaptation. This is separate from the conditional game GAN.

All public images use [0,1]. Encoder, decoder, codebook and discriminator can
start from random weights. Optional pretrained LPIPS/VGG is a frozen TRAINING
LOSS only; it is never an inference component or a pretrained image codec.
Importing this module does not import LPIPS or download any weights.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _validate_images(real, reconstruction):
    if (real.ndim != 4 or real.shape != reconstruction.shape or real.shape[1] != 3
            or min(real.shape) < 1):
        raise ValueError("Real and reconstructed RGB must share nonempty [B,3,H,W] shape")
    if not real.is_floating_point() or not reconstruction.is_floating_point():
        raise ValueError("VQGAN images must be floating-point RGB in [0,1]")


def _group_norm(channels):
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class NLayerDiscriminator(nn.Module):
    """Randomly initialized RGB PatchGAN; n_layers=3 gives a 70-pixel field.

    The three stride-two stages and two stride-one stages follow the standard
    VQGAN PatchGAN. GroupNorm is an intentional small-batch adaptation and has
    no running image statistics. There is no condition, sigmoid or spectral
    normalization, and no pretrained discriminator weights are loaded.
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super().__init__()
        if input_nc != 3:
            raise ValueError("The VQGAN discriminator accepts RGB only (input_nc=3)")
        if not isinstance(ndf, int) or not isinstance(n_layers, int) or min(ndf, n_layers) < 1:
            raise ValueError("PatchGAN width and layer count must be positive integers")
        self.input_nc, self.ndf, self.n_layers = input_nc, ndf, n_layers
        self.minimum_image_size = 3 * 2 ** n_layers
        blocks = [nn.Conv2d(3, ndf, 4, stride=2, padding=1), nn.LeakyReLU(.2, inplace=False)]
        channels = ndf
        for layer in range(1, n_layers):
            width = ndf * min(2 ** layer, 8)
            blocks.extend((nn.Conv2d(channels, width, 4, stride=2, padding=1, bias=False),
                           _group_norm(width), nn.LeakyReLU(.2, inplace=False)))
            channels = width
        width = ndf * min(2 ** n_layers, 8)
        blocks.extend((nn.Conv2d(channels, width, 4, padding=1, bias=False),
                       _group_norm(width), nn.LeakyReLU(.2, inplace=False),
                       nn.Conv2d(width, 1, 4, padding=1)))
        self.main = nn.Sequential(*blocks)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0., std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    def metadata(self):
        return {"architecture": "rgb_patchgan", "input_channels": 3,
                "base_channels": self.ndf, "n_layers": self.n_layers,
                "normalization": "group_norm_small_batch_adaptation",
                "initialization": "random_normal_std_0.02", "pretrained": False}

    def forward(self, rgb):
        if (rgb.ndim != 4 or rgb.shape[1] != 3 or rgb.shape[0] < 1
                or min(rgb.shape[-2:]) < self.minimum_image_size):
            raise ValueError(f"PatchGAN requires RGB [B,3,H,W] with H,W>={self.minimum_image_size}")
        if not rgb.is_floating_point():
            raise ValueError("PatchGAN RGB must be floating point")
        return self.main(rgb)


def hinge_discriminator_loss(real_logits, fake_logits):
    """VQGAN convention: half the real-margin plus fake-margin losses."""
    return .5 * (F.relu(1. - real_logits.float()).mean()
                 + F.relu(1. + fake_logits.float()).mean())


def generator_hinge_loss(fake_logits):
    return -fake_logits.float().mean()


def adaptive_discriminator_weight(rec_loss, g_loss, last_layer, *, epsilon=1e-4, max_weight=1e4):
    """Balance GAN against reconstruction by actual decoder-output gradients.

    Both losses must still be connected to the same trainable last layer.
    Autograd graphs are retained for the caller's final generator backward;
    this helper does not populate .grad or construct second-order gradients.
    The returned FP32 ratio is detached and bounded, as in the VQGAN objective.
    """
    if rec_loss.numel() != 1 or g_loss.numel() != 1:
        raise ValueError("Adaptive weighting requires scalar losses")
    if epsilon <= 0 or max_weight <= 0 or not math.isfinite(epsilon + max_weight):
        raise ValueError("Adaptive weight bounds must be positive and finite")
    rec_gradient = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
    gan_gradient = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    ratio = rec_gradient.float().norm() / (gan_gradient.float().norm() + epsilon)
    return ratio.clamp(0., max_weight).detach()


def discriminator_factor(global_step, disc_start=1000, weight=1.):
    """Zero before the configured warmup, then weight; threshold is inclusive.

    The runner must skip the discriminator optimizer update while this is 0,
    rather than stepping Adam on zero gradients. Typical starts are 500/1000.
    """
    if global_step < 0 or disc_start < 0 or not math.isfinite(float(weight)) or weight < 0:
        raise ValueError("GAN step, warmup and weight must be nonnegative")
    return float(weight) if global_step >= disc_start else 0.


class FrozenLPIPSVGG(nn.Module):
    """Optional conventional pretrained LPIPS/VGG loss for public RGB[0,1].

    Construction explicitly requests standard pretrained perceptual weights;
    the optional lpips package may download/cache VGG weights at that point.
    The caller owns dependency installation and download/cache policy.
    Gradients pass through reconstructed pixels, never perceptual parameters.
    """

    def __init__(self):
        super().__init__()
        try:
            import lpips
        except ImportError as error:
            raise ImportError("FrozenLPIPSVGG requires the optional 'lpips' package") from error
        self.network = lpips.LPIPS(net="vgg", pretrained=True, pnet_rand=False, eval_mode=True)
        self.network.eval().requires_grad_(False)
        self.eval()

    @property
    def metadata(self):
        return {"name": "LPIPS", "backbone": "VGG16", "pretrained": True,
                "trainable": False, "training_loss_only": True,
                "input_range": "0_1", "network_input_range": "minus1_1"}

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, real, reconstruction):
        _validate_images(real, reconstruction)
        return self.network(real.detach().float() * 2. - 1., reconstruction.float() * 2. - 1.)


class VQGANObjective(nn.Module):
    """L1 + optional frozen LPIPS + finite-difference pixel-art edge loss.

    The codec's separate vq_loss and adaptive adversarial term are added by the
    runner. Supplying perceptual=None disables perceptual supervision honestly;
    it never silently substitutes random features for pretrained LPIPS.
    """

    def __init__(self, perceptual=None, pixel_weight=1., perceptual_weight=1., edge_weight=.1,
                 disc_start=1000):
        super().__init__()
        weights = (pixel_weight, perceptual_weight, edge_weight)
        if any(not math.isfinite(float(value)) or value < 0 for value in weights) or disc_start < 0:
            raise ValueError("Reconstruction weights and GAN warmup must be nonnegative and finite")
        if perceptual is not None and not isinstance(perceptual, nn.Module):
            raise TypeError("Perceptual loss must be a torch.nn.Module or None")
        self.perceptual = perceptual
        self.pixel_weight, self.perceptual_weight, self.edge_weight = map(float, weights)
        self.disc_start = disc_start
        if perceptual is not None:
            perceptual.eval().requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.perceptual is not None:
            self.perceptual.eval()
        return self

    @property
    def metadata(self):
        perceptual = None
        if self.perceptual is not None:
            perceptual = getattr(self.perceptual, "metadata", {
                "name": type(self.perceptual).__name__, "provided_by_caller": True,
                "trainable": False, "training_loss_only": True})
        return {"pixel_weight": self.pixel_weight, "perceptual_weight": self.perceptual_weight,
                "edge_weight": self.edge_weight, "edge_loss_is_project_adaptation": True,
                "disc_start": self.disc_start, "perceptual": perceptual}

    def discriminator_factor(self, global_step, weight=1.):
        return discriminator_factor(global_step, self.disc_start, weight)

    def reconstruction_loss(self, real, reconstruction):
        _validate_images(real, reconstruction)
        target, predicted = real.detach().float(), reconstruction.float()
        pixel = F.l1_loss(predicted, target)
        perceptual = predicted.sum() * 0.
        if self.perceptual is not None and self.perceptual_weight:
            perceptual = self.perceptual(target, predicted).float().mean()
        directions = []
        if predicted.shape[-1] > 1:
            directions.append(F.l1_loss(predicted[..., 1:] - predicted[..., :-1],
                                        target[..., 1:] - target[..., :-1]))
        if predicted.shape[-2] > 1:
            directions.append(F.l1_loss(predicted[..., 1:, :] - predicted[..., :-1, :],
                                        target[..., 1:, :] - target[..., :-1, :]))
        edge = torch.stack(directions).mean() if directions else predicted.sum() * 0.
        total = self.pixel_weight * pixel + self.perceptual_weight * perceptual + self.edge_weight * edge
        return total, {"pixel_l1": float(pixel.detach()), "perceptual_loss": float(perceptual.detach()),
                       "edge_l1": float(edge.detach()), "reconstruction_loss": float(total.detach()),
                       "perceptual_enabled": self.perceptual is not None and self.perceptual_weight > 0}
