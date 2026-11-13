"""Shared 30 Hz timer arithmetic at the Source/float32-state boundary.

The state-vector adapter keeps its literal normalized values.  Only a timer
*advance* can recover a nearby simulation phase: elapsed timers start at zero;
countdowns start at their disclosed duration.  The acceptance interval is two
float32 representation ULP bounds, not an enlarged gameplay deadline. Genuine
off-grid timers and non-integral 30 Hz steps keep their ordinary arithmetic.

The scalar entry point has no NumPy/PyTorch dependency. Tensor arithmetic uses
float64 intermediates, then encodes once in the incoming state dtype.
"""
from __future__ import annotations

import math

SIMULATION_HZ = 30
NORMALIZED_TIMER_VERSION = 'float32_phase_30hz_v1'
PHASE_REPRESENTATION_ULPS = 2
_FLOAT32_EPS = 2.0 ** -23


def _settings(duration, dt, maximum):
    for name, value in (('duration', duration), ('dt', dt)):
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value <= 0):
            raise ValueError(name + ' must be finite and positive')
    if maximum is not None and (not isinstance(maximum, (int, float))
            or isinstance(maximum, bool) or not math.isfinite(maximum) or maximum < 0):
        raise ValueError('maximum must be finite and nonnegative')
    ticks = round(dt * SIMULATION_HZ)
    return ticks if ticks > 0 and abs(dt * SIMULATION_HZ - ticks) <= 1e-10 else None


def advance_timer_seconds(value, *, duration, dt, increasing=False, maximum=None):
    """Advance a physical-seconds scalar, recovering only a nearby 30 Hz phase.

    ``maximum`` is in seconds. ``duration`` is the state's normalization scale,
    also the origin of a countdown (e.g. fire .22s, death normalization .6s).
    Calling this function never mutates a state/vector or changes a deadline.
    """
    ticks = _settings(duration, dt, maximum)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('timer must be a finite scalar')
    value = max(0., float(value))
    if maximum is not None:
        value = min(value, maximum)
    result = value + dt if increasing else value - dt
    if ticks is not None:
        phase = (value if increasing else duration - value) * SIMULATION_HZ
        nearest = round(phase)
        bound = (PHASE_REPRESENTATION_ULPS * _FLOAT32_EPS
                 * max(1., abs(value / duration)) * duration * SIMULATION_HZ)
        if abs(phase - nearest) <= bound:
            result = ((nearest + ticks) / SIMULATION_HZ if increasing
                      else duration - (nearest + ticks) / SIMULATION_HZ)
    result = max(0., result)
    return min(result, maximum) if maximum is not None else result


def advance_normalized_timer(value, *, duration, dt, increasing=False, maximum=None):
    """Tensor equivalent of :func:`advance_timer_seconds`, returning normalized values.

    The forward phase correction has the identity derivative of the original
    arithmetic away from clipping; rounding is not a learned timer decision.
    """
    import torch
    ticks = _settings(duration, dt, maximum)
    if not torch.is_tensor(value) or not value.is_floating_point():
        raise ValueError('normalized timer must be a floating tensor')
    if not bool(torch.isfinite(value).all()):
        raise ValueError('normalized timer must be finite')
    clipped = value.clamp_min(0)
    if maximum is not None:
        clipped = clipped.clamp_max(maximum / duration)
    # Preserve native tensor arithmetic exactly outside a recovered phase.
    ordinary_result = clipped + dt / duration if increasing else clipped - dt / duration
    ordinary_result = ordinary_result.clamp_min(0)
    if maximum is not None:
        ordinary_result = ordinary_result.clamp_max(maximum / duration)
    if ticks is None:
        return ordinary_result
    seconds = clipped.double() * duration
    phase = (seconds if increasing else duration - seconds) * SIMULATION_HZ
    nearest = phase.round()
    bound = (PHASE_REPRESENTATION_ULPS * _FLOAT32_EPS
             * (seconds / duration).abs().clamp_min(1) * duration * SIMULATION_HZ)
    canonical = ((nearest + ticks) / SIMULATION_HZ if increasing
                 else duration - (nearest + ticks) / SIMULATION_HZ)
    result = (canonical / duration).clamp_min(0)
    if maximum is not None:
        result = result.clamp_max(maximum / duration)
    canonical_result = result.to(dtype=value.dtype).detach() + (ordinary_result - ordinary_result.detach())
    return torch.where((phase - nearest).abs() <= bound, canonical_result, ordinary_result)


def normalized_timer_ready(value, *, duration, dt, increasing=False, deadline=None):
    """Post-advance readiness of a recovered simulation phase.

    Elapsed timers require a deadline; countdowns are ready at zero. Recovered
    phases use physical Source tolerance 1e-8s. Off-grid/generic-step inputs
    preserve the old typed readiness predicates as well as their arithmetic.
    """
    if increasing and (deadline is None or deadline <= 0):
        raise ValueError('An elapsed timer needs a positive deadline')
    import torch
    ticks = _settings(duration, dt, deadline if increasing else duration)
    if not torch.is_tensor(value) or not value.is_floating_point():
        raise ValueError('normalized timer must be a floating tensor')
    legacy = (value * duration + dt >= deadline - 1e-6 if increasing
              else (value - dt / duration).clamp_min(0) <= 1e-6)
    if ticks is None:
        return legacy
    seconds = value.clamp_min(0).double() * duration
    if increasing:
        seconds = seconds.clamp_max(deadline)
    else:
        seconds = seconds.clamp_max(duration)
    phase = (seconds if increasing else duration - seconds) * SIMULATION_HZ
    nearest = phase.round()
    bound = (PHASE_REPRESENTATION_ULPS * _FLOAT32_EPS
             * (seconds / duration).abs().clamp_min(1) * duration * SIMULATION_HZ)
    canonical_seconds = ((nearest + ticks) / SIMULATION_HZ if increasing
                         else duration - (nearest + ticks) / SIMULATION_HZ)
    canonical_ready = (canonical_seconds >= deadline - 1e-8 if increasing
                       else canonical_seconds <= 1e-8)
    return torch.where((phase - nearest).abs() <= bound, canonical_ready, legacy)
