"""Compact VQGAN first-stage codec trained from scratch on game RGB.

Independent adaptation of the CompVis MIT-licensed taming-transformers design:
https://github.com/CompVis/taming-transformers
The reference uses residual CNNs, GroupNorm/Swish, bottleneck attention, learned
vector quantization and a convolutional decoder. This compact implementation
uses two residual blocks per resolution in both directions and SDPA attention;
it is not compatible with official pretrained checkpoints. No weights download
or external engine query occurs here. GAN/perceptual objectives belong to the
trainer. Public RGB inputs and reconstructions use [0,1].

The VQ objective applies beta to encoder commitment, fixing the legacy beta
placement described by the official VectorQuantizer2(legacy=False).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class VQGANConfig:
    image_size: int = 128
    base_channels: int = 64
    channel_multipliers: tuple[int, ...] = (1, 2, 2)
    num_res_blocks: int = 2
    latent_channels: int = 128
    embedding_dim: int = 16
    n_embed: int = 1024
    commitment_beta: float = .25
    attention: bool = True
    dropout: float = 0.
    data_init: bool = True
    kmeans_iterations: int = 5
    max_init_samples: int = 16384
    distance_chunk_size: int = 4096
    usage_decay: float = .99
    dead_code_threshold: float = 1e-5

    def __post_init__(self):
        self.channel_multipliers = tuple(self.channel_multipliers)
        if (not self.channel_multipliers or min(self.channel_multipliers) < 1
                or min(self.image_size, self.base_channels, self.num_res_blocks,
                       self.latent_channels, self.embedding_dim, self.n_embed,
                       self.max_init_samples, self.distance_chunk_size) < 1):
            raise ValueError('Codec dimensions and channel multipliers must be positive')
        if self.image_size % self.downsample_factor or self.latent_size < 2:
            raise ValueError('Image size must divide the downsampling factor and leave a spatial bottleneck')
        if (not math.isfinite(self.commitment_beta) or self.commitment_beta < 0
                or not 0 <= self.dropout < 1 or self.kmeans_iterations < 0
                or not 0 <= self.usage_decay < 1 or not 0 <= self.dead_code_threshold < 1):
            raise ValueError('Invalid VQ commitment, dropout, initialization or usage settings')

    @property
    def downsample_factor(self):
        return 2**(len(self.channel_multipliers)-1)

    @property
    def latent_size(self):
        return self.image_size//self.downsample_factor


def _normalization(channels):
    return nn.GroupNorm(math.gcd(32, channels), channels, eps=1e-6)


class ResidualBlock(nn.Module):
    def __init__(self, incoming, outgoing, dropout=0.):
        super().__init__()
        self.norm1, self.norm2 = _normalization(incoming), _normalization(outgoing)
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, padding=1)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Conv2d(incoming, outgoing, 1) if incoming != outgoing else nn.Identity()

    def forward(self, value):
        hidden = self.conv1(F.silu(self.norm1(value)))
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return self.shortcut(value)+hidden


class SpatialAttention(nn.Module):
    """Single-head bottleneck attention with memory-efficient PyTorch SDPA."""
    def __init__(self, channels):
        super().__init__()
        self.norm = _normalization(channels)
        self.qkv = nn.Conv2d(channels, channels*3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, value):
        batch, channels, height, width = value.shape
        query, key, content = self.qkv(self.norm(value)).chunk(3, 1)
        tokens = [tensor.flatten(2).transpose(1, 2)[:, None] for tensor in (query, key, content)]
        attended = F.scaled_dot_product_attention(*tokens, dropout_p=0.)
        attended = attended[:, 0].transpose(1, 2).reshape(batch, channels, height, width)
        return value+self.proj(attended)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2)

    def forward(self, value):
        return self.conv(F.pad(value, (0, 1, 0, 1)))


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, value):
        return self.conv(F.interpolate(value, scale_factor=2, mode='nearest'))


def _middle(channels, config):
    return nn.Sequential(ResidualBlock(channels, channels, config.dropout),
        SpatialAttention(channels) if config.attention else nn.Identity(),
        ResidualBlock(channels, channels, config.dropout))


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = config.base_channels
        self.conv_in = nn.Conv2d(3, channels, 3, padding=1)
        levels = []
        for index, multiplier in enumerate(config.channel_multipliers):
            outgoing = config.base_channels*multiplier
            for _ in range(config.num_res_blocks):
                levels.append(ResidualBlock(channels, outgoing, config.dropout))
                channels = outgoing
            if index < len(config.channel_multipliers)-1:
                levels.append(Downsample(channels))
        self.levels = nn.Sequential(*levels)
        self.middle = _middle(channels, config)
        self.norm_out = _normalization(channels)
        self.conv_out = nn.Conv2d(channels, config.latent_channels, 3, padding=1)

    def forward(self, value):
        value = self.middle(self.levels(self.conv_in(value)))
        return self.conv_out(F.silu(self.norm_out(value)))


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = config.base_channels*config.channel_multipliers[-1]
        self.conv_in = nn.Conv2d(config.latent_channels, channels, 3, padding=1)
        self.middle = _middle(channels, config)
        levels = []
        for index in reversed(range(len(config.channel_multipliers))):
            outgoing = config.base_channels*config.channel_multipliers[index]
            for _ in range(config.num_res_blocks):
                levels.append(ResidualBlock(channels, outgoing, config.dropout))
                channels = outgoing
            if index:
                levels.append(Upsample(channels))
        self.levels = nn.Sequential(*levels)
        self.norm_out = _normalization(channels)
        self.conv_out = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, value):
        value = self.levels(self.middle(self.conv_in(value)))
        return (self.conv_out(F.silu(self.norm_out(value))).tanh()+1)*.5


class VectorQuantizer(nn.Module):
    """Learned codebook with correct commitment beta and explicit usage statistics.

    The moving usage histogram is diagnostic only; this is an ordinary
    gradient-trained embedding table, not an EMA-updated quantizer. No automatic
    dead-code replacement or validation-dependent initialization occurs.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.n_embed, config.embedding_dim)
        nn.init.normal_(self.embedding.weight, std=1/math.sqrt(config.embedding_dim))
        self.register_buffer('initialized', torch.tensor(not config.data_init, dtype=torch.bool))
        self.register_buffer('usage_ema', torch.zeros(config.n_embed))
        self.register_buffer('usage_updates', torch.zeros((), dtype=torch.long))

    def _nearest(self, flat, embeddings=None):
        embeddings = self.embedding.weight if embeddings is None else embeddings
        # argmin is discrete: keep its large distance matrices outside autograd
        # and compute in FP32 even under the trainer's mixed-precision context.
        with torch.no_grad(), torch.autocast(flat.device.type, enabled=False):
            table = embeddings.float()
            table_norm = table.square().sum(1)
            indices = []
            for values in flat.detach().float().split(self.config.distance_chunk_size):
                distances = values.square().sum(1, keepdim=True)+table_norm[None]-2*values@table.t()
                indices.append(distances.argmin(1))
        return torch.cat(indices)

    @torch.no_grad()
    def initialize_from_features(self, features):
        """Initialize once from actual training features, never external weights."""
        if bool(self.initialized):
            return False
        flat = features.detach().permute(0, 2, 3, 1).reshape(-1, self.config.embedding_dim).float()
        if not flat.numel() or not bool(torch.isfinite(flat).all()):
            raise ValueError('Data initialization requires finite nonempty encoder features')
        if flat.shape[0] > self.config.max_init_samples:
            flat = flat[torch.randperm(flat.shape[0], device=flat.device)[:self.config.max_init_samples]]
        order = torch.randperm(flat.shape[0], device=flat.device)
        if flat.shape[0] >= self.config.n_embed:
            centers = flat[order[:self.config.n_embed]].clone()
        else:
            repeated = order.repeat(math.ceil(self.config.n_embed/len(order)))[:self.config.n_embed]
            centers = flat[repeated].clone()
            scale = flat.std(0, unbiased=False).clamp_min(1e-3)
            centers[len(order):] += torch.randn_like(centers[len(order):])*scale*.01
        for _ in range(self.config.kmeans_iterations):
            labels = self._nearest(flat, centers)
            counts = torch.bincount(labels, minlength=self.config.n_embed)
            sums = torch.zeros_like(centers).index_add_(0, labels, flat)
            occupied = counts > 0
            centers[occupied] = sums[occupied]/counts[occupied, None]
        self.embedding.weight.copy_(centers.to(self.embedding.weight.dtype))
        self.initialized.fill_(True)
        return True

    def forward(self, features):
        if features.ndim != 4 or features.shape[1] != self.config.embedding_dim:
            raise ValueError('Quantizer expects [B,embedding_dim,H,W] features')
        if self.training and self.config.data_init and not bool(self.initialized):
            self.initialize_from_features(features)
        batch, channels, height, width = features.shape
        flat = features.permute(0, 2, 3, 1).reshape(-1, channels)
        indices = self._nearest(flat)
        selected = self.embedding(indices).view(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()
        # Correct VQ-VAE objective: beta multiplies encoder commitment, not the
        # codebook's embedding-learning term (the old reference's legacy bug).
        with torch.autocast(features.device.type, enabled=False):
            codebook_loss = F.mse_loss(selected.float(), features.detach().float())
            commitment_loss = F.mse_loss(features.float(), selected.detach().float())
            vq_loss = codebook_loss+self.config.commitment_beta*commitment_loss
        quantized = features+(selected.to(features.dtype)-features).detach()
        counts = torch.bincount(indices, minlength=self.config.n_embed).float()
        probabilities = counts/counts.sum().clamp_min(1)
        if self.training:
            with torch.no_grad():
                self.usage_ema.mul_(self.config.usage_decay).add_(probabilities*(1-self.config.usage_decay))
                self.usage_updates.add_(1)
        # Bias correction makes the first batch comparable with later batches.
        updates = self.usage_updates.to(self.usage_ema.dtype)
        corrected_usage = self.usage_ema/(1-self.config.usage_decay**updates).clamp_min(1e-12)
        metrics = {
            'codebook_loss': codebook_loss.detach(), 'commitment_loss': commitment_loss.detach(),
            'perplexity': (-(probabilities*probabilities.clamp_min(1e-12).log()).sum()).exp().detach(),
            'active_codes': (counts > 0).sum().detach(),
            'codebook_usage': (counts > 0).float().mean().detach(),
            'dead_codes': (corrected_usage <= self.config.dead_code_threshold).sum().detach(),
            'codebook_initialized': self.initialized.detach().clone(),
            'usage_updates': self.usage_updates.detach().clone(),
            'prequant_std': features.detach().float().std(unbiased=False),
        }
        return {'quantized': quantized, 'indices': indices.view(batch, height, width),
                'vq_loss': vq_loss, 'metrics': metrics}


class VQGAN(nn.Module):
    """From-scratch VQGAN codec; adversarial training is supplied externally."""
    def __init__(self, config=None):
        super().__init__()
        self.config = VQGANConfig() if config is None else config
        if isinstance(self.config, dict):
            self.config = VQGANConfig(**self.config)
        self.encoder, self.decoder = Encoder(self.config), Decoder(self.config)
        self.quant_conv = nn.Conv2d(self.config.latent_channels, self.config.embedding_dim, 1)
        self.quantize = VectorQuantizer(self.config)
        self.post_quant_conv = nn.Conv2d(self.config.embedding_dim, self.config.latent_channels, 1)

    def encode(self, value):
        if value.ndim != 4 or value.shape[1:] != (3, self.config.image_size, self.config.image_size):
            raise ValueError('Codec RGB input must have shape [B,3,image_size,image_size]')
        value = value.to(self.encoder.conv_in.weight.dtype)
        prequant = self.quant_conv(self.encoder(value*2-1))
        return {'prequant': prequant, **self.quantize(prequant)}

    def decode(self, latents):
        if (latents.ndim != 4 or latents.shape[1] != self.config.embedding_dim
                or tuple(latents.shape[-2:]) != (self.config.latent_size, self.config.latent_size)):
            raise ValueError('Decoder expects [B,embedding_dim,latent_size,latent_size]')
        return self.decoder(self.post_quant_conv(latents.to(self.post_quant_conv.weight.dtype)))

    def decode_indices(self, indices):
        if indices.ndim != 3 or tuple(indices.shape[-2:]) != (self.config.latent_size, self.config.latent_size):
            raise ValueError('Code indices must have shape [B,latent_size,latent_size]')
        embeddings = self.quantize.embedding(indices.long()).permute(0, 3, 1, 2).contiguous()
        return self.decode(embeddings)

    def decode_codes(self, indices):
        """Decode integer code IDs; continuous embedding tensors use decode()."""
        return self.decode_indices(indices)

    def get_config(self):
        return asdict(self.config)

    def forward(self, value):
        encoded = self.encode(value)
        return {**encoded, 'reconstruction': self.decode(encoded['quantized'])}

    def get_last_layer(self):
        return self.decoder.conv_out.weight
