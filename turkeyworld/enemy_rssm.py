"""Independent one-NPC categorical memory with detached shared conditioning.

Only this module's own parameters and h/z carry enemy gradients. Numerical
state, player proposals, shared map/spatial features and shared RGB embeddings
are values, never differentiable paths into the established player model.
There are no engine, visibility, hit, HP, death or spawn rules here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .rssm import DreamerRSSM, RSSMConfig, RMSNorm
from .structured_state import STATE_DIM


@dataclass
class EnemyRSSMConfig(RSSMConfig):
    deter_dim: int = 384
    hidden_dim: int = 384
    stoch: int = 8
    classes: int = 32
    embed_dim: int = 128
    action_dim: int = 128
    blocks: int = 4
    unimix: float = .01

    @property
    def memory_size(self):
        return self.deter_dim+self.stoch*self.classes


def _call(module, value):
    return module(value.to(next(module.parameters()).dtype)).float()


class EnemyRSSM(nn.Module):
    """Own640D FP32 memory by default; observations enter only the posterior.

    ``motion`` is a worldXY proposal in the existing raw motion units: the
    backend applies ``.1*motion`` to normalized NPC coordinates. Attack logits
    are learned intent; shared structured_transition accepts or rejects them.
    """
    def __init__(self, config=None, width=512, observation_dim=512):
        super().__init__()
        if config is None:
            config = EnemyRSSMConfig()
        elif isinstance(config, dict):
            config = EnemyRSSMConfig(**config)
        elif isinstance(config, RSSMConfig):
            config = EnemyRSSMConfig(**asdict(config))
        else:
            raise TypeError('Enemy RSSM config must be an RSSMConfig or explicit dictionary')
        if any(type(value) is not int or value < 1 for value in (width, observation_dim)):
            raise ValueError('Positive enemy readout and shared observation widths are required')
        self.config, self.width, self.observation_dim = config, width, observation_dim
        self.core = DreamerRSSM(config)
        self.memory_size = self.core.memory_size
        # Full state20 + action4 + proposed playerXY2 + mapW + two actor tokens.
        self.conditioner = nn.Sequential(nn.Linear(STATE_DIM+6+3*width, config.action_dim),
            nn.LayerNorm(config.action_dim), nn.SiLU(),
            nn.Linear(config.action_dim, config.action_dim), nn.Tanh())
        self.observation_projection = nn.Sequential(nn.Linear(observation_dim, config.embed_dim),
            nn.LayerNorm(config.embed_dim), nn.SiLU())
        self.readout = nn.Sequential(nn.Linear(self.memory_size, config.hidden_dim),
            RMSNorm(config.hidden_dim), nn.GELU(), nn.Linear(config.hidden_dim, width))
        rank = min(128, width)
        self.motion_head = nn.Sequential(nn.Linear(width, rank), nn.GELU(), nn.Linear(rank, 2))
        self.attack_head = nn.Sequential(nn.Linear(width, rank), nn.GELU(), nn.Linear(rank, 1))
        nn.init.normal_(self.motion_head[-1].weight, std=.001)
        nn.init.zeros_(self.motion_head[-1].bias)
        nn.init.zeros_(self.attack_head[-1].weight)
        nn.init.constant_(self.attack_head[-1].bias, -4.)

    def initial(self, batch_size, device=None, dtype=None):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('Enemy memory requires a positive batch size')
        return self.core.pack(self.core.initial(batch_size, device=device, dtype=dtype)).float()

    def condition(self, state, action, proposed_player_xy, shared_map_context, spatial_tokens):
        if state.ndim != 2 or state.shape[1] != STATE_DIM:
            raise ValueError('Enemy conditioning requires numerical state[B,20]')
        batch = len(state)
        if (action.shape != (batch, 4) or proposed_player_xy.shape != (batch, 2)
                or shared_map_context.shape != (batch, self.width)
                or spatial_tokens.shape != (batch, 2, self.width)):
            raise ValueError('Enemy action/player/map/two-token conditioning shapes differ')
        state, action, proposed_player_xy, shared_map_context, spatial_tokens = (
            value.detach().float() for value in (state, action, proposed_player_xy, shared_map_context, spatial_tokens))
        controls = torch.cat([action[:, :3].clamp(-1, 1), action[:, 3:4].clamp(0, 1)], -1)
        values = torch.cat([state, controls, proposed_player_xy, shared_map_context, spatial_tokens.flatten(1)], -1)
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError('Nonfinite detached enemy conditioning; no repair')
        return _call(self.conditioner, values)

    def _observation(self, shared_rgb_embed, batch):
        if shared_rgb_embed.shape != (batch, self.observation_dim):
            raise ValueError('Enemy posterior requires the matching shared RGB embedding')
        value = shared_rgb_embed.detach().float()
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError('Nonfinite enemy posterior observation; no repair')
        return _call(self.observation_projection, value)

    def _unpack(self, memory):
        if (not torch.is_tensor(memory) or memory.ndim != 2 or memory.shape[1] != self.memory_size
                or not bool(torch.isfinite(memory).all())):
            raise ValueError('Enemy memory must be a finite packed[B,h+z] tensor')
        return self.core.unpack(memory)

    def _readout(self, latent):
        memory = self.core.pack(latent).float()
        pooled = _call(self.readout, memory)
        return {'memory': memory, 'pooled': pooled,
            'motion': _call(self.motion_head, pooled), 'attack_logits': _call(self.attack_head, pooled),
            'rssm_state': latent, 'logits': latent['logits'],
            'rssm_h': latent['deter'], 'rssm_z': latent['stoch']}

    def prior(self, memory, state, action, proposed_player_xy, shared_map_context, spatial_tokens, *, sample=None):
        """Imagine one tick using own h/z and current values; there is no RGB arg."""
        previous = self._unpack(memory)
        controls = self.condition(state, action, proposed_player_xy, shared_map_context, spatial_tokens)
        latent = self.core.imagine_step(previous, controls, sample=sample)
        result = self._readout(latent)
        result.update(prior_logits=latent['logits'], rssm_prior_logits=latent['logits'])
        return result

    def posterior(self, prior, shared_rgb_embed, *, sample=None):
        """Observe once from the exact pre-RGB prior, without a second recurrence."""
        if not isinstance(prior, dict) or 'rssm_state' not in prior:
            raise ValueError('Enemy posterior requires the actual previously computed prior')
        latent = prior['rssm_state']
        observation = self._observation(shared_rgb_embed, len(latent['deter']))
        posterior = self.core.posterior(latent, observation, sample=sample)
        result = self._readout(posterior)
        result.update(prior_logits=prior['rssm_prior_logits'], rssm_prior_logits=prior['rssm_prior_logits'],
            posterior_logits=posterior['logits'], rssm_posterior_logits=posterior['logits'])
        return result

    def observe_initial(self, shared_rgb_embed, state, shared_map_context, spatial_tokens, *, sample=None):
        """Explicit initialization only; death/offscreen never resets this memory."""
        prior = self.prior(self.initial(len(state), device=state.device), state,
            state.new_zeros(len(state), 4), state[:, :2], shared_map_context, spatial_tokens, sample=sample)
        return self.posterior(prior, shared_rgb_embed, sample=sample)
