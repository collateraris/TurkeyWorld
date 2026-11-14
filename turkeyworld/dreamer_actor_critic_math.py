"""Distributional values and transition-indexed lambda returns for imagination.

The current DreamerV3 reference uses raw two-hot interpolation on a symmetric
symexp-spaced support. The older symlog-interpolation variant is an explicit,
separately serialized option. No world model or engine is imported here.
"""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F

VALUE_ENCODINGS = ('symexp_twohot', 'symlog_twohot')


def symlog(value):
    return value.sign()*torch.log1p(value.abs())


def symexp(value):
    return value.sign()*torch.expm1(value.abs())


def make_value_bins(num_bins=255, low=-20., high=20., *, encoding='symexp_twohot',
                    device=None, dtype=torch.float32):
    if (type(num_bins) is not int or num_bins < 3 or encoding not in VALUE_ENCODINGS
            or type(low) not in (int, float) or type(high) not in (int, float)
            or not math.isfinite(low) or not math.isfinite(high) or not -40 <= low < high <= 40):
        raise ValueError('Invalid finite distributional value support or encoding')
    if low == -high and num_bins % 2:
        half = torch.linspace(low, 0, (num_bins-1)//2+1, device=device, dtype=dtype)
        half = symexp(half) if encoding == 'symexp_twohot' else half
        return torch.cat((half, -half[:-1].flip(0)))
    bins = torch.linspace(low, high, num_bins, device=device, dtype=dtype)
    return symexp(bins) if encoding == 'symexp_twohot' else bins


def _support(bins, encoding):
    if (encoding not in VALUE_ENCODINGS or not torch.is_tensor(bins) or bins.ndim != 1
            or len(bins) < 3 or not bins.is_floating_point() or not bool(torch.isfinite(bins).all())
            or not bool((bins[1:] > bins[:-1]).all())):
        raise ValueError('Value bins must be a finite strictly ascending vector')


def twohot_encode(values, bins, encoding='symexp_twohot'):
    """Detached labels with mass on the two surrounding bins; extremes clamp."""
    _support(bins, encoding)
    if not torch.is_tensor(values) or not values.is_floating_point() or not bool(torch.isfinite(values).all()):
        raise ValueError('Two-hot targets must be finite floating values')
    target = values.detach().to(device=bins.device, dtype=bins.dtype)
    if encoding == 'symlog_twohot':
        target = symlog(target)
    target = target.clamp(bins[0], bins[-1]).contiguous()
    upper = torch.searchsorted(bins, target, right=True).clamp(1, len(bins)-1)
    lower = upper-1
    upper_mass = (target-bins[lower])/(bins[upper]-bins[lower])
    result = torch.zeros((*target.shape, len(bins)), device=bins.device, dtype=bins.dtype)
    result.scatter_add_(-1, lower[..., None], (1-upper_mass)[..., None])
    result.scatter_add_(-1, upper[..., None], upper_mass[..., None])
    return result


def twohot_mean(logits, bins, encoding='symexp_twohot'):
    """Raw value prediction; paired summation is exactly zero for uniform logits."""
    _support(bins, encoding)
    if (not torch.is_tensor(logits) or not logits.is_floating_point() or logits.shape[-1] != len(bins)
            or logits.device != bins.device or not bool(torch.isfinite(logits).all())):
        raise ValueError('Distributional value logits do not match their finite support')
    probability = logits.float().softmax(-1)
    weighted = probability*bins.to(probability)
    middle = len(bins)//2
    if len(bins) % 2:
        mean = weighted[..., middle] + (weighted[..., :middle].flip(-1)+weighted[..., middle+1:]).sum(-1)
    else:
        mean = (weighted[..., :middle].flip(-1)+weighted[..., middle:]).sum(-1)
    return symexp(mean) if encoding == 'symlog_twohot' else mean


def twohot_loss(logits, targets, bins, encoding='symexp_twohot'):
    if logits.shape[:-1] != targets.shape:
        raise ValueError('Distributional target shape must equal logits without the bin axis')
    labels = twohot_encode(targets, bins, encoding)
    if logits.device != bins.device or not bool(torch.isfinite(logits).all()):
        raise ValueError('Distributional logits must be finite and colocated with bins')
    return -(labels*F.log_softmax(logits.float(), -1)).sum(-1)


@torch.no_grad()
def twohot_target_clip_stats(values, bins, encoding='symexp_twohot'):
    """Report raw label mass that the support would clamp, before encoding."""
    _support(bins, encoding)
    if (not torch.is_tensor(values) or not values.is_floating_point() or not values.numel()
            or not bool(torch.isfinite(values).all())):
        raise ValueError('Clipping diagnostics require finite nonempty raw targets')
    target = values.detach().to(bins)
    low, high = (symexp(bins[0]), symexp(bins[-1])) if encoding == 'symlog_twohot' else (bins[0], bins[-1])
    clipped = (target < low) | (target > high)
    return dict(clipped_count=int(clipped.sum()), target_count=target.numel(),
        clipped_fraction=float(clipped.float().mean()), support_low=float(low), support_high=float(high),
        raw_target_min=float(target.min()), raw_target_max=float(target.max()))


def lambda_returns(rewards, continues, values, *, gamma=332/333, lambda_=.95):
    """Return[B,H], with r_t/c_t for s_t->s_(t+1), values[B,H+1].

    A terminal transition retains its own reward and cancels all later rewards
    and bootstrap. A truncated nonterminal tail bootstraps values[:,-1]. This
    helper detaches every target input to prevent actor/world bootstrap paths.
    """
    if (not torch.is_tensor(rewards) or rewards.ndim != 2 or not rewards.is_floating_point()
            or not torch.is_tensor(continues) or continues.shape != rewards.shape
            or not torch.is_tensor(values) or values.shape != (rewards.shape[0], rewards.shape[1]+1)
            or not rewards.numel() or not 0 <= gamma <= 1 or not 0 <= lambda_ <= 1
            or any(value.device != rewards.device or not value.is_floating_point()
                or not bool(torch.isfinite(value).all()) for value in (rewards, continues, values))
            or bool(((continues < 0) | (continues > 1)).any())):
        raise ValueError('Lambda return needs finite aligned rewards/continuations/value bootstrap')
    reward, continuation, value = (item.detach().float() for item in (rewards, continues, values))
    future = value[:, -1]
    result = []
    for step in reversed(range(reward.shape[1])):
        future = reward[:, step] + gamma*continuation[:, step]*((1-lambda_)*value[:, step+1]+lambda_*future)
        result.append(future)
    return torch.stack(list(reversed(result)), 1)


def imagination_weights(continues, *, gamma=332/333, initial_alive=None):
    """Weight the action before its outcome; terminal-reward actions stay valid."""
    if (not torch.is_tensor(continues) or continues.ndim != 2 or not continues.is_floating_point()
            or not continues.numel() or not bool(torch.isfinite(continues).all())
            or bool(((continues < 0) | (continues > 1)).any()) or not 0 <= gamma <= 1):
        raise ValueError('Imagined continuation probabilities must be finite[B,H] in[0,1]')
    alive = torch.ones(continues.shape[0], device=continues.device)
    if initial_alive is not None:
        if (not torch.is_tensor(initial_alive) or initial_alive.shape != alive.shape
                or initial_alive.device != continues.device or not bool(torch.isfinite(initial_alive).all())
                or bool(((initial_alive < 0) | (initial_alive > 1)).any())):
            raise ValueError('Initial alive weights must align with imagined actors')
        alive = initial_alive.detach().float()
    prefix = torch.cat((torch.ones_like(continues[:, :1]), gamma*continues.detach()[:, :-1]), 1)
    return prefix.cumprod(1)*alive[:, None]
