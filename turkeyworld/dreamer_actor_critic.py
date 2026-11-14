"""Compact Dreamer-style discrete actor/critic trained on frozen imagination.

Current-reference return/lambda/entropy/slow-value settings are explicit below;
the512-unit PyTorch heads are a compact game-specific architecture. Imagination
is collected elsewhere with a frozen world and learned reward/continue heads.
Every feature, reward, action and continuation is detached at this boundary.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import math

import torch
from torch import nn
from torch.nn import functional as F

from .dreamer_actor_critic_math import (VALUE_ENCODINGS, make_value_bins,
    twohot_encode, twohot_mean, twohot_loss, twohot_target_clip_stats, lambda_returns, imagination_weights)

PROTOCOL = 'dreamer_frozen_rssm_actor_critic_v1'
ACTION_PROTOCOL = 'factorized_move_strafe_turn_3_fire_2_v1'
ACTION_SIZES = (3, 3, 3, 2)
REFERENCE_URLS = (
    'https://github.com/danijar/dreamerv3/blob/main/dreamerv3/configs.yaml',
    'https://github.com/danijar/dreamerv3/blob/main/dreamerv3/agent.py',
    'https://github.com/danijar/dreamerv3/blob/main/embodied/jax/utils.py',
    'https://github.com/danijar/dreamerv3/blob/main/embodied/jax/heads.py')


@dataclass(frozen=True)
class ActorCriticConfig:
    feature_dim: int
    hidden_dim: int = 512
    layers: int = 3
    gamma: float = 332/333
    lambda_: float = .95
    entropy_scale: float = 3e-4
    slow_value_rate: float = .02
    slow_target: bool = False
    slow_regularization: float = 1.
    return_norm: bool = True
    return_norm_rate: float = .01
    return_norm_low: float = .05
    return_norm_high: float = .95
    return_norm_floor: float = 1.
    value_encoding: str = 'symexp_twohot'
    num_bins: int = 255
    value_low: float = -20.
    value_high: float = 20.
    action_unimix: float = 0.
    actor_output_scale: float = .01
    norm_epsilon: float = 1e-4

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in (self.feature_dim, self.hidden_dim, self.layers, self.num_bins)):
            raise ValueError('Actor/critic dimensions must be positive integer sizes')
        numeric = (self.gamma, self.lambda_, self.entropy_scale, self.slow_value_rate, self.slow_regularization,
            self.return_norm_rate, self.return_norm_low, self.return_norm_high, self.return_norm_floor,
            self.value_low, self.value_high, self.action_unimix, self.actor_output_scale, self.norm_epsilon)
        if (any(type(value) not in (int, float) or not math.isfinite(value) for value in numeric)
                or not 0 <= self.gamma <= 1 or not 0 <= self.lambda_ <= 1 or self.entropy_scale < 0
                or not 0 < self.slow_value_rate <= 1 or self.slow_regularization < 0
                or type(self.slow_target) is not bool or type(self.return_norm) is not bool
                or not 0 < self.return_norm_rate <= 1
                or not 0 <= self.return_norm_low < self.return_norm_high <= 1 or self.return_norm_floor <= 0
                or self.value_encoding not in VALUE_ENCODINGS or self.num_bins < 3
                or not -40 <= self.value_low < self.value_high <= 40 or not 0 <= self.action_unimix < 1
                or self.actor_output_scale <= 0 or self.norm_epsilon <= 0):
            raise ValueError('Invalid explicit actor/critic return, entropy, support or normalizer setting')

    def to_dict(self):
        return dict(protocol=PROTOCOL, **asdict(self))

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.get('protocol') != PROTOCOL:
            raise ValueError('Unknown actor/critic configuration protocol')
        names = {field.name for field in fields(cls)}
        if set(value) != names | {'protocol'}:
            raise ValueError('Actor/critic serialized configuration fields changed')
        return cls(**{name: value[name] for name in names})


class _RMSNorm(nn.Module):
    def __init__(self, size, epsilon):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.epsilon = epsilon

    def forward(self, value):
        return value*torch.rsqrt(value.square().mean(-1, keepdim=True)+self.epsilon)*self.weight


def _mlp(config):
    modules = []
    previous = config.feature_dim
    for _ in range(config.layers):
        layer = nn.Linear(previous, config.hidden_dim)
        # Compact architecture choice; the return/optimization formulas below
        # follow the reference, without claiming its full1024-unit network.
        nn.init.trunc_normal_(layer.weight, std=1/math.sqrt(previous),
            a=-2/math.sqrt(previous), b=2/math.sqrt(previous))
        nn.init.zeros_(layer.bias)
        modules.extend((layer, _RMSNorm(config.hidden_dim, config.norm_epsilon), nn.SiLU()))
        previous = config.hidden_dim
    return nn.Sequential(*modules)


def _features(value, module, size):
    if (not torch.is_tensor(value) or not value.is_floating_point() or value.ndim < 1
            or value.shape[-1] != size or not value.numel() or not bool(torch.isfinite(value).all())):
        raise ValueError('Actor/critic features must be finite tensors with the pinned trailing dimension')
    return value.detach().to(next(module.parameters()))


def action_indices(actions):
    if (not torch.is_tensor(actions) or actions.ndim < 1 or actions.shape[-1] != 4
            or not actions.is_floating_point() or not bool(torch.isfinite(actions).all())
            or not bool(((actions[..., :3] == -1) | (actions[..., :3] == 0) | (actions[..., :3] == 1)).all())
            or not bool(((actions[..., 3] == 0) | (actions[..., 3] == 1)).all())):
        raise ValueError('Policy actions require move/strafe/turn in{-1,0,1} and fire in{0,1}')
    value = actions.detach()
    return torch.cat(((value[..., :3]+1).long(), value[..., 3:].long()), -1)


class FactorizedActor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.trunk = _mlp(config)
        self.output = nn.Linear(config.hidden_dim, sum(ACTION_SIZES))
        standard_deviation = config.actor_output_scale/math.sqrt(config.hidden_dim)
        nn.init.trunc_normal_(self.output.weight, std=standard_deviation,
            a=-2*standard_deviation, b=2*standard_deviation)
        nn.init.zeros_(self.output.bias)

    def forward(self, features):
        return self.output(self.trunk(_features(features, self, self.config.feature_dim)))

    def _log_probabilities(self, logits):
        result = []
        for values in logits.split(ACTION_SIZES, -1):
            log_probability = F.log_softmax(values.float(), -1)
            if self.config.action_unimix:
                probability = (1-self.config.action_unimix)*log_probability.exp()+self.config.action_unimix/values.shape[-1]
                log_probability = probability.log()
            result.append(log_probability)
        return result

    def evaluate_actions(self, features, actions):
        logits = self(features)
        indices = action_indices(actions).to(logits.device)
        if indices.shape[:-1] != logits.shape[:-1]:
            raise ValueError('Policy sampled-action shape differs from actor features')
        probabilities = self._log_probabilities(logits)
        log_parts = [values.gather(-1, indices[..., index:index+1]).squeeze(-1)
            for index, values in enumerate(probabilities)]
        entropies = [-(values.exp()*values).sum(-1) for values in probabilities]
        return dict(log_prob=torch.stack(log_parts, -1).sum(-1),
            entropy=torch.stack(entropies, -1).sum(-1), logits=logits,
            factor_entropies=torch.stack(entropies, -1), indices=indices)

    @torch.no_grad()
    def act(self, features, deterministic=False, generator=None):
        if type(deterministic) is not bool:
            raise ValueError('Deterministic policy selection must be boolean')
        logits = self(features)
        parts = self._log_probabilities(logits)
        indices = []
        for values in parts:
            if deterministic:
                selected = values.argmax(-1)
            else:
                selected = torch.multinomial(values.exp().reshape(-1, values.shape[-1]), 1,
                    generator=generator).reshape(values.shape[:-1])
            indices.append(selected)
        result = torch.stack(indices, -1).float()
        result[..., :3] -= 1
        return result


class SymlogTwoHotCritic(nn.Module):
    """Name covers both explicit symlog and current symexp-spaced encodings."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.trunk = _mlp(config)
        self.output = nn.Linear(config.hidden_dim, config.num_bins)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer('bins', make_value_bins(config.num_bins, config.value_low, config.value_high,
            encoding=config.value_encoding))

    def forward(self, features):
        return self.output(self.trunk(_features(features, self, self.config.feature_dim)))

    def value(self, features):
        return twohot_mean(self(features), self.bins, self.config.value_encoding)

    def loss(self, features, targets):
        return twohot_loss(self(features), targets, self.bins, self.config.value_encoding)


class ReturnPercentileNormalizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.register_buffer('low', torch.tensor(0.))
        self.register_buffer('high', torch.tensor(0.))
        self.register_buffer('updates', torch.tensor(0, dtype=torch.int64))

    @torch.no_grad()
    def update(self, returns):
        if not self.config.return_norm:
            return
        if not torch.is_tensor(returns) or not returns.numel() or not bool(torch.isfinite(returns).all()):
            raise ValueError('Return normalization requires a finite nonempty target tape')
        values = returns.detach().float().to(self.low.device).flatten()
        quantiles = torch.quantile(values, torch.tensor([self.config.return_norm_low, self.config.return_norm_high],
            device=values.device))
        self.low.lerp_(quantiles[0], self.config.return_norm_rate)
        self.high.lerp_(quantiles[1], self.config.return_norm_rate)
        self.updates.add_(1)

    def scale(self):
        if not self.config.return_norm:
            return self.low.detach()*0+1
        return (self.high-self.low).clamp_min(self.config.return_norm_floor).detach()


class DreamerActorCritic(nn.Module):
    def __init__(self, config):
        super().__init__()
        if not isinstance(config, ActorCriticConfig):
            raise ValueError('DreamerActorCritic requires its explicit serialized configuration')
        self.config = config
        self.actor = FactorizedActor(config)
        self.critic = SymlogTwoHotCritic(config)
        self.slow_critic = deepcopy(self.critic).requires_grad_(False).eval()
        self.return_normalizer = ReturnPercentileNormalizer(config)
        self.register_buffer('slow_updates', torch.tensor(0, dtype=torch.int64))

    def train(self, mode=True):
        super().train(mode)
        self.slow_critic.eval()
        return self

    def losses(self, imagined, update_normalizer=True):
        if type(update_normalizer) is not bool:
            raise ValueError('Normalizer update selection must be boolean')
        feature = imagined.get('features')
        actions, rewards, continues = [imagined.get(name) for name in ('actions', 'rewards', 'continues')]
        if (not torch.is_tensor(feature) or feature.ndim != 3 or feature.shape[-1] != self.config.feature_dim
                or not torch.is_tensor(rewards) or rewards.ndim != 2 or not rewards.numel()
                or feature.shape[:2] != (rewards.shape[0], rewards.shape[1]+1)
                or not torch.is_tensor(actions) or actions.shape != (*rewards.shape, 4)
                or not torch.is_tensor(continues) or continues.shape != rewards.shape):
            raise ValueError('Imagined features must be[B,H+1,D] with transition actions/rewards/continues[B,H]')
        parameter = next(self.actor.parameters())
        feature = feature.detach().to(parameter)
        actions = actions.detach().to(parameter)
        rewards = rewards.detach().to(parameter)
        continues = continues.detach().to(parameter)
        initial_alive = imagined.get('initial_alive')
        if initial_alive is not None:
            initial_alive = initial_alive.detach().to(parameter)
        value_logits = self.critic(feature)
        with torch.no_grad():
            online_values = twohot_mean(value_logits.detach(), self.critic.bins, self.config.value_encoding)
            slow_values = self.slow_critic.value(feature)
            bootstrap = slow_values if self.config.slow_target else online_values
            returns = lambda_returns(rewards, continues, bootstrap, gamma=self.config.gamma, lambda_=self.config.lambda_)
            weights = imagination_weights(continues, gamma=self.config.gamma, initial_alive=initial_alive)
            if update_normalizer:
                self.return_normalizer.update(returns)
            scale = self.return_normalizer.scale()
            advantages = (returns-bootstrap[:, :-1])/scale
        policy = self.actor.evaluate_actions(feature[:, :-1], actions)
        actor_loss = -(weights*(policy['log_prob']*advantages+self.config.entropy_scale*policy['entropy'])).mean()
        value_error = twohot_loss(value_logits[:, :-1], returns, self.critic.bins, self.config.value_encoding)
        slow_error = twohot_loss(value_logits[:, :-1], slow_values[:, :-1], self.critic.bins, self.config.value_encoding)
        critic_loss = (weights*(value_error+self.config.slow_regularization*slow_error)).mean()
        if not bool(torch.isfinite(actor_loss)) or not bool(torch.isfinite(critic_loss)):
            raise FloatingPointError('Nonfinite actor/critic imagination objective')
        with torch.no_grad():
            metrics = dict(protocol=PROTOCOL, actor_loss=float(actor_loss), critic_loss=float(critic_loss),
                total_loss=float(actor_loss+critic_loss), value_target_loss=float((weights*value_error).mean()),
                slow_value_regularizer_loss=float((weights*slow_error).mean()), entropy=float(policy['entropy'].mean()),
                entropy_move=float(policy['factor_entropies'][..., 0].mean()),
                entropy_strafe=float(policy['factor_entropies'][..., 1].mean()),
                entropy_turn=float(policy['factor_entropies'][..., 2].mean()),
                entropy_fire=float(policy['factor_entropies'][..., 3].mean()),
                imagined_reward_mean=float(rewards.mean()), imagined_continue_mean=float(continues.mean()),
                raw_return_mean=float(returns.mean()), raw_return_min=float(returns.min()), raw_return_max=float(returns.max()),
                value_mean=float(online_values.mean()), slow_value_mean=float(slow_values.mean()),
                advantage_mean=float(advantages.mean()), advantage_std=float(advantages.std(unbiased=False)),
                imagination_weight_mean=float(weights.mean()), alive_weighted_transitions=float(weights.sum()),
                return_norm_low=float(self.return_normalizer.low), return_norm_high=float(self.return_normalizer.high),
                return_norm_scale=float(scale), return_norm_updates=int(self.return_normalizer.updates),
                actor_gradient='REINFORCE_factorized_discrete', critic_encoding=self.config.value_encoding,
                bootstrap_critic='slow' if self.config.slow_target else 'online_stop_gradient',
                world_inputs_detached=True, reward_continue_targets_detached=True, critic_targets_detached=True,
                actor_uses_critic_gradient=False, actor_updates_world=False, imagined_horizon=rewards.shape[1],
                imagined_actors=rewards.shape[0], slow_updates=int(self.slow_updates))
            metrics.update({'critic_target_'+name: value for name, value in twohot_target_clip_stats(
                returns, self.critic.bins, self.config.value_encoding).items()})
            metrics.update({'critic_slow_target_'+name: value for name, value in twohot_target_clip_stats(
                slow_values[:, :-1], self.critic.bins, self.config.value_encoding).items()})
        return dict(actor=actor_loss, critic=critic_loss), metrics

    @torch.no_grad()
    def update_slow_critic(self):
        rate = self.config.slow_value_rate
        source = dict(self.critic.named_parameters())
        for name, parameter in self.slow_critic.named_parameters():
            parameter.lerp_(source[name], rate)
        for name, buffer in self.slow_critic.named_buffers():
            buffer.copy_(dict(self.critic.named_buffers())[name])
        self.slow_updates.add_(1)
