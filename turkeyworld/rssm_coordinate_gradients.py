"""Train-only XY integration credit with detached neural conditioning."""
from __future__ import annotations

import torch

from .structured_state import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS


COORDINATE_GRADIENT_PROTOCOL = {
    'version': 1,
    'mode': 'detached_conditioning_integrated_xy',
    'exclude_predicted_respawns': True,
}


def normalize_coordinate_gradient_protocol(settings=None):
    if settings is None:
        return None
    if (not isinstance(settings, dict) or settings != COORDINATE_GRADIENT_PROTOCOL
            or type(settings.get('version')) is not int
            or settings.get('exclude_predicted_respawns') is not True):
        raise ValueError('Unsupported coordinate-gradient protocol')
    return dict(COORDINATE_GRADIENT_PROTOCOL)


def integrate_coordinate_gradients(previous_state, result, protocol=None):
    """Carry XY gradients only; retain original result for local Source heads.

The neural transition must have received previous_state.detach(). A zero-value
identity term restores additive XY credit within TBPTT without the unstable
typed-state Jacobian. Predicted respawn teleports begin a fresh NPC chain.
"""
    if normalize_coordinate_gradient_protocol(protocol) is None:
        return result
    predicted = result['state']
    if (previous_state.ndim != 2 or previous_state.shape[-1] != STATE_DIM
            or predicted.shape != previous_state.shape
            or predicted.device != previous_state.device or predicted.dtype != previous_state.dtype):
        raise ValueError('Coordinate credit requires matching floating states[B,STATE_DIM]')
    respawn = result.get('respawn_flags')
    if (not torch.is_tensor(respawn) or respawn.shape != (len(predicted), ENTITY_SLOTS)
            or respawn.dtype != torch.bool or respawn.device != predicted.device):
        raise ValueError('Coordinate credit requires predicted bool respawn_flags[B,slots]')
    mask = torch.zeros_like(predicted)
    mask[:, :2] = 1
    for slot in range(ENTITY_SLOTS):
        first = GLOBAL_DIM+slot*ENTITY_DIM
        mask[:, first:first+2] = (~respawn[:, slot:slot+1]).to(mask.dtype)
    identity = (previous_state-previous_state.detach())*mask
    return {**result, 'state': predicted+identity}
