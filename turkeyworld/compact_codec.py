"""Sub-million-parameter native32 codec with the teacher's exact latent codebook.

This changes encoder/decoder capacity, not VQGAN's implementation or latent
coordinates. Encoder and decoder are new weights; only the complete trained
quantizer state is copied. The quantizer stays frozen during codec distillation.
"""
from __future__ import annotations

from dataclasses import asdict

import torch

from .vqgan import VQGAN, VQGANConfig


COMPACT_CODEC_PARAMETERS = 997_539
COMPACT_CODEC_MAX_PARAMETERS = 1_000_000


def compact_codec_config(source_config=None):
    """Preserve quantization settings and the native32/8x8x16 interface."""
    if source_config is None:
        source_config = VQGANConfig(image_size=32, n_embed=1488)
    values = (asdict(source_config) if isinstance(source_config, VQGANConfig)
              else dict(source_config))
    source = VQGANConfig(**values)
    if (source.image_size != 32 or source.downsample_factor != 4
            or source.embedding_dim != 16 or source.n_embed != 1488):
        raise ValueError('Compact codec requires native32, downsample4 and the1488x16 teacher codebook')
    values = asdict(source)
    values.update(base_channels=24, channel_multipliers=(1, 2, 3),
                  num_res_blocks=1, latent_channels=128, attention=True)
    return VQGANConfig(**values)


class CompactCodec(VQGAN):
    """Existing VQGAN architecture with an immutable quantizer during training."""

    def train(self, mode=True):
        super().train(mode)
        self.quantize.eval().requires_grad_(False)
        return self

    def requires_grad_(self, requires_grad=True):
        super().requires_grad_(requires_grad)
        self.quantize.requires_grad_(False)
        return self


def build_compact_codec(source, *, seed=173):
    """Create fresh E/D and copy the teacher codebook without changing RNG/source.

    The result starts on CPU in the source train/eval mode. Its quantizer weights
    and diagnostic buffers have private storage and remain unchanged by train().
    """
    if not isinstance(source, VQGAN):
        raise TypeError('source must be a loaded VQGAN codec')
    config = compact_codec_config(source.config)
    if not bool(source.quantize.initialized):
        raise ValueError('Teacher codebook must be initialized before distillation')
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError('Codec initialization seed must be an integer')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        student = CompactCodec(config)
    student.quantize.load_state_dict(source.quantize.state_dict(), strict=True)
    student.quantize.requires_grad_(False)
    student.train(source.training)
    count = sum(value.numel() for value in student.parameters())
    if count != COMPACT_CODEC_PARAMETERS or count > COMPACT_CODEC_MAX_PARAMETERS:
        raise RuntimeError('Compact codec architecture exceeds its exact parameter contract')
    return student


def compact_codec_parameter_counts(codec):
    """Count real E/Q/D parameters, including frozen embeddings exactly once."""
    count = lambda module: sum(value.numel() for value in module.parameters())
    return {'total': count(codec), 'encoder': count(codec.encoder)+count(codec.quant_conv),
            'decoder': count(codec.decoder)+count(codec.post_quant_conv),
            'codebook': count(codec.quantize),
            'trainable': sum(value.numel() for value in codec.parameters() if value.requires_grad)}
