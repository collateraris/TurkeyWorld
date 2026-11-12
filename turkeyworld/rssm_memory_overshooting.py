"""Stopped posterior targets for an independent autonomous RSSM memory branch.

Source future RGB belongs exclusively to the no-grad target stream. The
trainable branch initializes from observed context once and carries its own
numerical state and packed memories, with no posterior feedback or reset.
"""
from __future__ import annotations

from copy import deepcopy

import torch
from torch.nn import functional as F

from .gradient_safety import finite_gradient_norm
from .rssm_coordinate_gradients import (
    normalize_coordinate_gradient_protocol, integrate_coordinate_gradients)


MEMORY_OVERSHOOTING_PROTOCOL = {
    'version': 1, 'offsets': [1, 4, 8, 16], 'weight': .1,
    'deterministic_weight': .1, 'unimix': .01,
    'target_stop_gradient': True, 'future_rgb_feedback': False,
}


def normalize_memory_overshooting_protocol(settings=None):
    if settings is None:
        return None
    if (not isinstance(settings, dict) or settings != MEMORY_OVERSHOOTING_PROTOCOL
            or type(settings.get('version')) is not int
            or type(settings.get('offsets')) is not list
            or any(type(value) is not int for value in settings['offsets'])
            or any(type(settings.get(key)) not in (int, float)
                   for key in ('weight', 'deterministic_weight', 'unimix'))
            or settings.get('target_stop_gradient') is not True
            or settings.get('future_rgb_feedback') is not False):
        raise ValueError('Unsupported memory-overshooting protocol')
    return deepcopy(MEMORY_OVERSHOOTING_PROTOCOL)


def _categorical_kl(target, predicted, unimix):
    """KL per categorical variable, averaged over batch and latent variables."""
    if target.shape != predicted.shape or target.ndim != 3 or target.shape[-1] < 2:
        raise ValueError('Overshooting categorical targets require matching[B,stoch,classes]')
    classes = predicted.shape[-1]
    q = target.detach().float().softmax(-1)*(1-unimix)+unimix/classes
    p = predicted.float().softmax(-1)*(1-unimix)+unimix/classes
    tiny = torch.finfo(torch.float32).tiny
    return (q*(q.clamp_min(tiny).log()-p.clamp_min(tiny).log())).sum(-1).mean()


def _deterministic_error(target, predicted):
    if target.shape != predicted.shape or target.ndim != 2:
        raise ValueError('Overshooting deterministic targets require matching[B,deter]')
    return F.mse_loss(F.layer_norm(predicted.float(), (predicted.shape[-1],), eps=1e-6),
        F.layer_norm(target.detach().float(), (target.shape[-1],), eps=1e-6))


def memory_overshooting_backward(model, batch, *, context, observed_context, protocol,
                                coordinate_gradient_protocol=None, loss_groups=None,
                                check_gradients=True):
    """Backpropagate once; leave Adam, EMA, model weights and modes untouched.

``loss`` is protocol-weighted before actor normalization. ``weighted_loss``
is the actual backpropagated scalar, including the caller's fixed actor
coefficients. No constant KL/free-nats floor is added by this objective.
"""
    settings = normalize_memory_overshooting_protocol(protocol)
    if settings is None:
        return {'loss': 0., 'weighted_loss': 0., 'unweighted_loss': 0.,
            'player_kl': 0., 'enemy_kl': 0., 'player_h': 0., 'enemy_h': 0.,
            'offsets': [], 'no_future_feedback': True, 'backward_calls': 0,
            'prior_future_rgb_inputs': 0, 'optimizer_steps': 0}
    coordinate = normalize_coordinate_gradient_protocol(coordinate_gradient_protocol)
    # Reuse the exact main-objective Source alignment and context contract.
    from .rssm_objectives import _batch_data, _observed_count, _initialize, _valid_result
    states, frames, actions, _, grid = _batch_data(batch, context)
    observed = _observed_count(context, observed_context)
    horizon = min(actions.shape[1], max(settings['offsets']))
    offsets = [offset for offset in settings['offsets'] if offset <= horizon]
    dual = bool(getattr(getattr(model, 'config', None), 'enemy_rssm_enabled', False))
    target_keys = {'player': ('rssm_posterior_logits', 'rssm_h')}
    prior_keys = {'player': ('rssm_prior_logits', 'rssm_h')}
    if dual:
        target_keys['enemy'] = ('enemy_rssm_posterior_logits', 'enemy_rssm_h')
        prior_keys['enemy'] = ('enemy_rssm_prior_logits', 'enemy_rssm_h')
    core = model.core
    modes = [(module, module.training) for module in core.modules()]
    targets = {}
    try:
        # Preserve training=True on the backend for its existing typed/ST
        # derivatives. Only categorical sampling becomes deterministic.
        core.eval()
        with torch.no_grad():
            fmap = model.encode_map(grid).detach()
            memory = _initialize(model, frames, states, fmap, context, observed).detach()
            expected_shape = tuple(memory.shape)
            for index in range(horizon):
                posterior = model.vjepa_backend.observe_step(states[:, context-1+index],
                    actions[:, index], fmap, memory, frames[:, context+index], sample=False)
                _valid_result(posterior, expected_shape, f'overshooting target {index+1}')
                memory = posterior['memory'].detach()
                if index+1 in offsets:
                    for actor, keys in target_keys.items():
                        if any(key not in posterior for key in keys):
                            raise ValueError('Missing '+actor+' overshooting posterior target')
                    targets[index+1] = {key: posterior[key].detach().clone()
                        for keys in target_keys.values() for key in keys}
        # A separate observed-context initialization supplies the online
        # encoder/core gradients; no target memory is copied into this branch.
        memory = _initialize(model, frames, states, fmap, context, observed)
        if tuple(memory.shape) != expected_shape:
            raise ValueError('Overshooting target/prior packed memory shapes differ')
        current = states[:, context-1]
        parts = {}
        for index in range(horizon):
            prior = model.transition(current.detach(), actions[:, index], fmap, memory)
            _valid_result(prior, expected_shape, f'overshooting own prior {index+1}')
            integrated = integrate_coordinate_gradients(current, prior, coordinate)
            current, memory = integrated['state'], prior['memory']
            if index+1 not in targets:
                continue
            target = targets[index+1]
            for actor in prior_keys:
                logits, hidden = prior_keys[actor]
                target_logits, target_hidden = target_keys[actor]
                if logits not in prior or hidden not in prior:
                    raise ValueError('Missing '+actor+' overshooting own prior output')
                kl = _categorical_kl(target[target_logits], prior[logits], settings['unimix'])
                deterministic = _deterministic_error(target[target_hidden], prior[hidden])
                for suffix, value in (('kl', kl), ('h', deterministic)):
                    name = actor+'_'+suffix
                    parts[name] = parts.get(name, 0)+value/len(offsets)
        groups = {actor: settings['weight']*(parts[actor+'_kl']
            +settings['deterministic_weight']*parts[actor+'_h']) for actor in prior_keys}
        raw = sum(groups.values())
        if not bool(torch.isfinite(raw)):
            raise FloatingPointError('Nonfinite RSSM memory-overshooting objective')
        actual = loss_groups.combine(groups) if loss_groups is not None else raw
        if not bool(torch.isfinite(actual)):
            raise FloatingPointError('Nonfinite balanced RSSM memory-overshooting objective')
        actual.backward()
        if check_gradients:
            finite_gradient_norm(model.named_parameters(), context='RSSM memory overshooting backward')
        numbers = {name: float(value.detach()) for name, value in parts.items()}
        return {'loss': float(raw.detach()), 'weighted_loss': float(actual.detach()),
            'unweighted_loss': float(raw.detach())/settings['weight'],
            'player_kl': numbers['player_kl'], 'player_h': numbers['player_h'],
            'enemy_kl': numbers.get('enemy_kl', 0.), 'enemy_h': numbers.get('enemy_h', 0.),
            'offsets': offsets, 'horizon': horizon, 'target_stop_gradient': True,
            'no_future_feedback': True, 'prior_future_rgb_inputs': 0,
            'target_rgb_frames': len(states)*horizon, 'initial_observed_frames': observed,
            'prior_numerical_resets': 0, 'memory_resets': 0, 'backward_calls': 1,
            'optimizer_steps': 0, 'kl_floor': 0., 'categorical_reduction': 'mean_per_latent_variable',
            'sampling': 'deterministic', 'weight': settings['weight'],
            'deterministic_weight': settings['deterministic_weight']}
    finally:
        for module, training in modes:
            module.training = training
