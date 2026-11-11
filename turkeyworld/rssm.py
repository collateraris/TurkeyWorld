"""PyTorch categorical RSSM inspired by the official DreamerV3 world core.

Uses grouped recurrent gates, RMS normalization, categorical straight-through
latents, uniform mixing and separately stopped dynamics/representation KL.
This module is not the complete Dreamer actor/critic algorithm or an official
checkpoint loader. Reference: https://github.com/danijar/dreamerv3/blob/main/dreamerv3/rssm.py
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class RSSMConfig:
    deter_dim: int = 1008
    hidden_dim: int = 1024
    stoch: int = 32
    classes: int = 32
    embed_dim: int = 512
    action_dim: int = 4
    blocks: int = 8
    unimix: float = .01

    def __post_init__(self):
        if (any(not isinstance(value, int) or isinstance(value, bool) or value < 1
                for value in (self.deter_dim, self.hidden_dim, self.stoch, self.classes,
                              self.embed_dim, self.action_dim, self.blocks))
                or self.classes < 2 or self.deter_dim % self.blocks
                or not isinstance(self.unimix, (int, float)) or isinstance(self.unimix, bool)
                or not math.isfinite(self.unimix) or not 0 <= self.unimix < 1):
            raise ValueError('Invalid RSSM dimensions, grouped state size or uniform mixture')


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, value):
        normalized = value.float()*torch.rsqrt(value.float().square().mean(-1, keepdim=True)+1e-6)
        return (normalized*self.weight.float()).to(value.dtype)


class BlockLinear(nn.Module):
    def __init__(self, incoming, outgoing, groups):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(groups, outgoing, incoming))
        self.bias = nn.Parameter(torch.empty(groups, outgoing))
        nn.init.uniform_(self.weight, -1/math.sqrt(incoming), 1/math.sqrt(incoming))
        nn.init.uniform_(self.bias, -1/math.sqrt(incoming), 1/math.sqrt(incoming))

    def forward(self, value):
        return torch.einsum('bgi,goi->bgo', value.to(self.weight.dtype), self.weight)+self.bias


def _mlp(incoming, hidden, outgoing=None):
    layers = [nn.Linear(incoming, hidden), RMSNorm(hidden), nn.GELU()]
    if outgoing is not None:
        layers.append(nn.Linear(hidden, outgoing))
    return nn.Sequential(*layers)


def _call(module, value):
    return module(value.to(next(module.parameters()).dtype)).float()


class DreamerRSSM(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or RSSMConfig()
        if isinstance(self.config, dict):
            self.config = RSSMConfig(**self.config)
        c = self.config
        self.deter_dim, self.stoch, self.classes = c.deter_dim, c.stoch, c.classes
        self.memory_size = c.deter_dim+c.stoch*c.classes
        self.deter_input = _mlp(c.deter_dim, c.hidden_dim)
        self.stoch_input = _mlp(c.stoch*c.classes, c.hidden_dim)
        self.action_input = _mlp(c.action_dim, c.hidden_dim)
        group = c.deter_dim//c.blocks
        self.recurrent_hidden = BlockLinear(group+3*c.hidden_dim, group, c.blocks)
        self.recurrent_norm = RMSNorm(c.deter_dim)
        self.recurrent_gates = BlockLinear(group, 3*group, c.blocks)
        self.prior_head = nn.Sequential(*_mlp(c.deter_dim, c.hidden_dim),
            *_mlp(c.hidden_dim, c.hidden_dim), nn.Linear(c.hidden_dim, c.stoch*c.classes))
        self.posterior_head = _mlp(c.deter_dim+c.embed_dim, c.hidden_dim, c.stoch*c.classes)

    def initial(self, batch_size, device=None, dtype=None):
        device = device or next(self.parameters()).device
        # Packed recurrent state remains FP32 even with FP16/BF16 network weights.
        return {'deter': torch.zeros(batch_size, self.deter_dim, device=device, dtype=torch.float32),
            'stoch': torch.zeros(batch_size, self.stoch, self.classes, device=device, dtype=torch.float32),
            'logits': torch.zeros(batch_size, self.stoch, self.classes, device=device, dtype=torch.float32)}

    def probabilities(self, logits):
        probs = logits.float().softmax(-1)
        return probs*(1-self.config.unimix)+self.config.unimix/self.classes

    def categorical(self, logits, *, sample=None):
        sample = self.training if sample is None else sample
        if not isinstance(sample, bool):
            raise ValueError('RSSM sampling mode must be an explicit boolean')
        probs = self.probabilities(logits)
        indices = (torch.multinomial(probs.reshape(-1, self.classes), 1).reshape(probs.shape[:-1])
                   if sample else probs.argmax(-1))
        hard = F.one_hot(indices, self.classes).float()
        return hard+(probs-probs.detach())

    def features(self, state):
        return torch.cat([state['deter'].float(), state['stoch'].float().flatten(1)], -1)

    def pack(self, state):
        return self.features(state)

    def unpack(self, memory):
        if memory.ndim != 2 or memory.shape[-1] != self.memory_size:
            raise ValueError('RSSM packed memory has the wrong recurrent dimension')
        memory = memory.float()
        stoch = memory[:, self.deter_dim:].reshape(-1, self.stoch, self.classes)
        return {'deter': memory[:, :self.deter_dim], 'stoch': stoch, 'logits': torch.zeros_like(stoch)}

    def _recurrent(self, state, action):
        c = self.config
        deter, stoch = state['deter'].float(), state['stoch'].float().flatten(1)
        if (deter.ndim != 2 or deter.shape[-1] != c.deter_dim
                or stoch.shape != (len(deter), c.stoch*c.classes)
                or action.shape != (len(deter), c.action_dim)):
            raise ValueError('RSSM state/action shape differs from its configuration')
        action = action.float()/action.float().abs().clamp_min(1).detach()
        context = torch.cat([_call(self.deter_input, deter), _call(self.stoch_input, stoch),
                             _call(self.action_input, action)], -1)
        grouped = torch.cat([deter.reshape(len(deter), c.blocks, -1),
            context[:, None].expand(-1, c.blocks, -1)], -1)
        hidden = self.recurrent_hidden(grouped).float().flatten(1)
        hidden = F.gelu(self.recurrent_norm(hidden)).reshape(len(deter), c.blocks, -1)
        reset, candidate, update = self.recurrent_gates(hidden).float().chunk(3, -1)
        reset, candidate, update = [value.flatten(1) for value in (reset, candidate, update)]
        candidate = torch.tanh(reset.sigmoid()*candidate)
        update = (update-1).sigmoid()
        return (update*candidate+(1-update)*deter).float()

    def imagine_step(self, previous, action, *, sample=None):
        deter = self._recurrent(previous, action)
        logits = _call(self.prior_head, deter).reshape(-1, self.stoch, self.classes)
        return {'deter': deter, 'stoch': self.categorical(logits, sample=sample), 'logits': logits}

    def posterior(self, prior, embed, *, sample=None):
        if embed.shape != (len(prior['deter']), self.config.embed_dim):
            raise ValueError('RSSM posterior embedding has the wrong shape')
        logits = _call(self.posterior_head, torch.cat([prior['deter'].float(), embed.float()], -1))
        logits = logits.reshape(-1, self.stoch, self.classes)
        return {'deter': prior['deter'], 'stoch': self.categorical(logits, sample=sample), 'logits': logits}

    def observe_step(self, previous, action, embed, *, sample=None):
        prior = self.imagine_step(previous, action, sample=sample)
        return self.posterior(prior, embed, sample=sample), prior


def balanced_kl(posterior_logits, prior_logits, *, free_nats=1., dynamics_scale=1.,
                representation_scale=.1, unimix=.01):
    """KL(q||p), with separately stopped q/p and a per-sample free-nats floor."""
    if (posterior_logits.shape != prior_logits.shape or posterior_logits.ndim < 3
            or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                   or not math.isfinite(value) or value < 0
                   for value in (free_nats, dynamics_scale, representation_scale))
            or not isinstance(unimix, (int, float)) or isinstance(unimix, bool)
            or not math.isfinite(unimix) or not 0 <= unimix < 1):
        raise ValueError('Invalid RSSM KL shapes, scales or uniform mixture')
    classes = posterior_logits.shape[-1]
    q = posterior_logits.float().softmax(-1)*(1-unimix)+unimix/classes
    p = prior_logits.float().softmax(-1)*(1-unimix)+unimix/classes
    tiny = torch.finfo(torch.float32).tiny
    kl = lambda left, right: (left*(left.clamp_min(tiny).log()-right.clamp_min(tiny).log())).sum((-1, -2))
    dynamics = kl(q.detach(), p).clamp_min(free_nats).mean()
    representation = kl(q, p.detach()).clamp_min(free_nats).mean()
    parts = {'dynamics_kl': dynamics, 'representation_kl': representation,
        'dynamics_kl_weighted': dynamics_scale*dynamics,
        'representation_kl_weighted': representation_scale*representation,
        'raw_kl': kl(q, p).detach().mean(),
        'prior_entropy': -(p*p.clamp_min(tiny).log()).sum((-1, -2)).detach().mean(),
        'posterior_entropy': -(q*q.clamp_min(tiny).log()).sum((-1, -2)).detach().mean()}
    return parts['dynamics_kl_weighted']+parts['representation_kl_weighted'], parts
