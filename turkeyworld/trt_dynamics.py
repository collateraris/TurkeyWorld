"""Fixed-shape V-JEPA latent prediction for TensorRT gameplay inference."""
import torch
from torch import nn
from torch.nn import functional as F

from .vjepa2_ac import _positions, block_causal_mask


class GameplayPredictor(nn.Module):
    """Export the trained predictor at the game's fixed batch and context length."""

    def __init__(self, core):
        super().__init__()
        self.predictor = core.predictor
        self.frames = core.context_frames
        attention = self.predictor.blocks[0].attention
        self.heads = attention.heads
        self.head_width = attention.head_width

        # RoPE positions and the frame-causal mask are identical at every tick.
        # Encode all three axes in one lookup, including unrotated tail channels.
        positions = _positions(self.frames, self.predictor.grid, 2, attention.frequency.device)
        phase = (positions.float()[:, :, None] * attention.frequency.float()).flatten(1)
        phase = phase.repeat_interleave(2, dim=-1)
        nrot = phase.shape[-1]
        phase = F.pad(phase, (0, self.head_width - nrot))
        indices = torch.arange(self.head_width, device=phase.device)
        partners = torch.where(indices < nrot, indices.bitwise_xor(1), indices)
        signs = torch.where(indices.remainder(2) == 0, -1, 1)
        self.register_buffer('partners', partners)
        self.register_buffer('cosine', phase.cos())
        self.register_buffer('signed_sine', phase.sin() * signs)
        allowed = block_causal_mask(self.frames, self.predictor.block_tokens, phase.device)
        mask = torch.zeros_like(allowed, dtype=self.predictor.predictor_embed.weight.dtype)
        self.register_buffer('mask', mask.masked_fill(~allowed, float('-inf'))[None, None])

    def forward(self, packed):
        p = self.predictor
        blocks = packed.reshape(1, self.frames, p.block_tokens, p.width)
        visual = p.predictor_embed(blocks[:, :, 2:]) + p.type_embedding[2]
        cond = blocks[:, :, :2] + p.type_embedding[:2]
        tokens = torch.cat((cond, visual), dim=2).flatten(1, 2)
        length = self.frames * p.block_tokens

        for block in p.blocks:
            qkv = block.attention.qkv(block.norm1(tokens)).reshape(
                1, length, 3, self.heads, self.head_width).permute(2, 0, 3, 1, 4)
            qk = qkv[:2].float()
            rotated = (qk * self.cosine + qk.index_select(-1, self.partners) * self.signed_sine).to(qkv.dtype)
            attention = F.scaled_dot_product_attention(rotated[0], rotated[1], qkv[2],
                                                       attn_mask=self.mask, dropout_p=0.)
            tokens = tokens + block.attention.proj(attention.transpose(1, 2).reshape(1, length, p.width))
            tokens = tokens + block.mlp(block.norm2(tokens))

        visual = tokens.reshape(1, self.frames, p.block_tokens, p.width)[:, :, 2:]
        return p.projection(p.norm(visual))


def enable_dynamics_trt(runtime, factory):
    """Install fixed-shape prediction for this game instance after building the engine."""
    if runtime.world_architecture != 'vjepa2_ac':
        return []
    core = runtime.model.core
    sample = runtime.initial_memory.to(core.predictor.predictor_embed.weight.dtype).contiguous()
    predictor = GameplayPredictor(core).eval()
    name = 'vjepa_predictor'
    engine = factory(name, predictor, (sample,), ('latent_prediction',))
    def predict_packed(packed):
        return engine(packed)

    core.predict_packed = predict_packed
    return [name]
