"""Inspect and clip dense gradients without overflowing an FP32 norm.

These functions never repair or silently discard invalid gradients. The
caller must invoke them before its optimizer/EMA/counter updates.
"""
from __future__ import annotations

import math

import torch


def _gradient_entries(named_parameters):
    entries = []
    for name, parameter in named_parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        if gradient.layout != torch.strided:
            raise ValueError(f"Dense gradient required for parameter {name!r}; received {gradient.layout}")
        if not gradient.is_floating_point():
            raise ValueError(f"Real floating gradient required for parameter {name!r}")
        entries.append((str(name), gradient))
    return entries


def _finite_norm(entries, context):
    if not entries:
        return torch.zeros((), dtype=torch.float64)
    device = entries[0][1].device
    # Promoting the reduction, rather than its FP32 result, is essential:
    # squaring an otherwise finite 1e25 FP32 gradient overflows in FP32.
    norms = [torch.linalg.vector_norm(gradient.detach(), ord=2, dtype=torch.float64).to(device)
             for _, gradient in entries]
    norm = torch.linalg.vector_norm(torch.stack(norms), ord=2)
    if not bool(torch.isfinite(norm)):
        # The ordinary finite path needs only the single scalar check above.
        # Parameter-level scans are reserved for an actual failure.
        invalid = [name for name, gradient in entries if not bool(torch.isfinite(gradient).all())]
        prefix = f"{context}: " if context else ""
        if invalid:
            raise FloatingPointError(prefix + "Non-finite gradients in parameters: " + ", ".join(invalid))
        raise FloatingPointError(prefix + "Non-finite FP64 gradient norm despite finite gradient elements")
    return norm


@torch.no_grad()
def finite_gradient_norm(named_parameters, *, context=""):
    """Return scalar FP64 L2 norm, or raise with invalid parameter names.

    The named iterable is consumed once. Neither parameters nor gradients
    are modified, including when validation fails. Missing gradients count0.
    """
    return _finite_norm(_gradient_entries(named_parameters), context)


@torch.no_grad()
def clip_finite_grad_norm_(named_parameters, max_norm=1., *, context=""):
    """Clip dense gradients in place and return their *original* FP64 norm.

    The coefficient matches PyTorch's L2 clipping rule, including its 1e-6
    epsilon. Validation completes before any gradient is modified.
    """
    limit = float(max_norm)
    if not math.isfinite(limit) or limit < 0:
        raise ValueError("max_norm must be finite and nonnegative")
    entries = _gradient_entries(named_parameters)
    norm = _finite_norm(entries, context)
    coefficient = (limit / (norm + 1e-6)).clamp(max=1.)
    for _, gradient in entries:
        gradient.mul_(coefficient.to(device=gradient.device, dtype=gradient.dtype))
    return norm
