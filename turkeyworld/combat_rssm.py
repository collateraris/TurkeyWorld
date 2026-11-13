"""Persistent interaction RSSM with zero-residual typed combat readouts.

Player/enemy prior contexts are differentiable. Numerical state, controls,
player proposal, map and spatial descriptors are detached values. Observed RGB
embeddings enter only explicit initialization/posterior calls. Every new latent
sampling call preserves the existing Player/Enemy global Torch RNG streams.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .rssm import DreamerRSSM, RSSMConfig, RMSNorm
from .structured_state import STATE_DIM


COMBAT_RSSM_PARAMETERS = 20_007_020
COMBAT_RESIDUAL_SLICES = {'target': slice(0, 2), 'attack': slice(2, 3),
                         'spawn_tile': slice(3, 259), 'spawn_anchor': slice(259, 268)}


@dataclass
class CombatRSSMConfig(RSSMConfig):
    deter_dim: int = 1344
    hidden_dim: int = 1392
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


@contextmanager
def _preserve_sampling_rng(device):
    """No mutable private stream or counter; existing actor streams stay exact."""
    device = torch.device(device)
    devices = ([device.index if device.index is not None else torch.cuda.current_device()]
               if device.type == 'cuda' else [])
    with torch.random.fork_rng(devices=devices):
        yield


class CombatRSSM(nn.Module):
    """Own1600D FP32 h/z and20,007,020 parameters at the production widths.

    There is no separate coordinate, HP, life or event truth. Four residual
    outputs adjust target/attack/spawn logits before the shared typed rules.
    The zero fusion adapter adjusts the existing combat role query without
    adding an attention token or changing the old MoE parameter namespace.
    """
    def __init__(self, config=None, width=512, observation_dim=512, fusion_width=256):
        super().__init__()
        if config is None:
            config = CombatRSSMConfig()
        elif isinstance(config, dict):
            config = CombatRSSMConfig(**config)
        elif isinstance(config, RSSMConfig):
            config = CombatRSSMConfig(**asdict(config))
        else:
            raise TypeError('Combat RSSM requires an RSSMConfig or explicit dictionary')
        if any(type(value) is not int or value < 1 for value in (width, observation_dim, fusion_width)):
            raise ValueError('Positive combat readout, observation and fusion widths are required')
        self.config, self.width = config, width
        self.observation_dim, self.fusion_width = observation_dim, fusion_width
        self.core = DreamerRSSM(config)
        self.memory_size = self.core.memory_size
        # state20 + known action4 + proposed playerXY2 + mapW + two spatialW
        # + current player priorW + current enemy priorW.
        self.conditioner = nn.Sequential(nn.Linear(STATE_DIM+6+5*width, config.action_dim),
            nn.LayerNorm(config.action_dim), nn.SiLU(),
            nn.Linear(config.action_dim, config.action_dim), nn.Tanh())
        self.observation_projection = nn.Sequential(nn.Linear(observation_dim, config.embed_dim),
            nn.LayerNorm(config.embed_dim), nn.SiLU())
        self.readout = nn.Sequential(nn.Linear(self.memory_size, config.hidden_dim),
            RMSNorm(config.hidden_dim), nn.GELU(), nn.Linear(config.hidden_dim, width))
        rank = min(128, width)
        self.task_head = nn.Sequential(nn.Linear(width, rank), nn.GELU(), nn.Linear(rank, 268))
        self.fusion_adapter = nn.Linear(width, fusion_width)
        for head in (self.task_head[-1], self.fusion_adapter):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def initial(self, batch_size, device=None, dtype=None):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('Combat memory requires a positive integer batch size')
        return self.core.pack(self.core.initial(batch_size, device=device, dtype=dtype)).float()

    def _validate(self, value, shape, label):
        parameter = next(self.parameters())
        if (not torch.is_tensor(value) or value.shape != shape or not value.is_floating_point()
                or value.device != parameter.device):
            raise ValueError('Combat '+label+' must be a matching colocated floating tensor'+str(tuple(shape)))
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError('Nonfinite combat '+label+'; no reset or repair')

    def condition(self, state, action, proposed_player_xy, shared_map_context, spatial_tokens, *,
                  player_context, enemy_context):
        if not torch.is_tensor(state) or state.ndim != 2 or len(state) < 1:
            raise ValueError('Combat conditioning requires nonempty numerical state[B,20]')
        count = len(state)
        values = (state, action, proposed_player_xy, shared_map_context, spatial_tokens,
                  player_context, enemy_context)
        shapes = ((count, STATE_DIM), (count, 4), (count, 2), (count, self.width),
                  (count, 2, self.width), (count, self.width), (count, self.width))
        labels = ('state', 'action', 'player proposal', 'map', 'spatial tokens',
                  'player prior context', 'enemy prior context')
        for value, shape, label in zip(values, shapes, labels):
            self._validate(value, shape, label)
        # Only the two current learned prior contexts carry shared gradients.
        state, action, proposed_player_xy, shared_map_context, spatial_tokens = (
            value.detach().float() for value in values[:5])
        controls = torch.cat([action[:, :3].clamp(-1, 1), action[:, 3:4].clamp(0, 1)], -1)
        inputs = torch.cat([state, controls, proposed_player_xy, shared_map_context,
                            spatial_tokens.flatten(1), player_context.float(), enemy_context.float()], -1)
        return _call(self.conditioner, inputs)

    def _unpack(self, memory):
        if not torch.is_tensor(memory) or memory.ndim != 2 or len(memory) < 1:
            raise ValueError('Combat memory must be a persistent finite packed[B,h+z] tensor')
        self._validate(memory, (len(memory), self.memory_size), 'memory')
        return self.core.unpack(memory.float())

    def _pooled(self, latent):
        return _call(self.readout, self.core.pack(latent).float())

    def _readout(self, latent):
        memory = self.core.pack(latent).float()
        pooled = self._pooled(latent)
        raw = _call(self.task_head, pooled)
        return {'memory': memory, 'pooled': pooled,
            'target_residual': raw[:, COMBAT_RESIDUAL_SLICES['target']],
            'attack_residual': raw[:, COMBAT_RESIDUAL_SLICES['attack']],
            'spawn_tile_residual': raw[:, COMBAT_RESIDUAL_SLICES['spawn_tile']][:, None],
            'spawn_anchor_residual': raw[:, COMBAT_RESIDUAL_SLICES['spawn_anchor']][:, None],
            'rssm_state': latent, 'logits': latent['logits'],
            'rssm_h': latent['deter'], 'rssm_z': latent['stoch']}

    def fusion_context(self, previous_memory):
        """Previous own h/z only; no recurrence, observation, RNG or reset."""
        return _call(self.fusion_adapter, self._pooled(self._unpack(previous_memory)))

    def prior(self, memory, state, action, proposed_player_xy, shared_map_context, spatial_tokens, *,
              player_context, enemy_context, sample=None):
        """Advance every tick, after both actor priors, without an RGB argument."""
        previous = self._unpack(memory)
        if len(memory) != len(state):
            raise ValueError('Combat state and persistent memory batch dimensions differ')
        controls = self.condition(state, action, proposed_player_xy, shared_map_context, spatial_tokens,
                                  player_context=player_context, enemy_context=enemy_context)
        with _preserve_sampling_rng(memory.device):
            latent = self.core.imagine_step(previous, controls, sample=sample)
        result = self._readout(latent)
        result.update(prior_logits=latent['logits'], rssm_prior_logits=latent['logits'])
        return result

    def posterior(self, prior, shared_rgb_embed, *, sample=None):
        """Observe the exact pre-RGB prior; never imagine a second tick."""
        if (not isinstance(prior, dict) or 'rssm_state' not in prior or 'rssm_prior_logits' not in prior
                or not isinstance(prior['rssm_state'], dict) or 'deter' not in prior['rssm_state']):
            raise ValueError('Combat posterior requires the actual previously computed prior wrapper')
        latent = prior['rssm_state']
        self._validate(shared_rgb_embed, (len(latent['deter']), self.observation_dim), 'observed RGB embedding')
        # Source observations are values. Current P/E contexts in condition()
        # provide the explicit permitted gradient paths into both old cores.
        embed = _call(self.observation_projection, shared_rgb_embed.detach().float())
        with _preserve_sampling_rng(shared_rgb_embed.device):
            posterior = self.core.posterior(latent, embed, sample=sample)
        result = self._readout(posterior)
        result.update(prior_logits=prior['rssm_prior_logits'], rssm_prior_logits=prior['rssm_prior_logits'],
                      posterior_logits=posterior['logits'], rssm_posterior_logits=posterior['logits'])
        return result

    def observe_initial(self, shared_rgb_embed, state, shared_map_context, spatial_tokens, *,
                        player_context, enemy_context, sample=None):
        """Explicit real observed initialization; death/offscreen never calls it."""
        if not torch.is_tensor(state) or state.ndim != 2 or len(state) < 1:
            raise ValueError('Combat initialization needs actual observed state[B,20]')
        prior = self.prior(self.initial(len(state), device=state.device), state,
            state.new_zeros(len(state), 4), state[:, :2], shared_map_context, spatial_tokens,
            player_context=player_context, enemy_context=enemy_context, sample=sample)
        return self.posterior(prior, shared_rgb_embed, sample=sample)
