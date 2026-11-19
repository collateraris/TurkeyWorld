"""Learn spawn category preferences from the model's own public simulation clock."""
import torch
from torch import nn

from .spawn import SPAWN_ANCHORS, SPAWN_PHASE_TICKS, SPAWN_TICK_HZ


class SpawnPhaseHead(nn.Module):
    def __init__(self):
        super().__init__()
        # No preferred tile, ordering, spawn schedule or labels are initialized.
        n = len(SPAWN_ANCHORS)
        self.scores = nn.Parameter(torch.zeros(n, n + 1))

    def forward(self, state, *, dt):
        tick = torch.round((state[:, -1].double() * 300 + dt) * SPAWN_TICK_HZ).long()
        phase = (tick // SPAWN_PHASE_TICKS).remainder(len(self.scores))
        return self.scores[phase, None].float()


def verified_migration(model, saved):
    """Qualify the single newly attached frozen tensor without changing older receipts."""
    from .renderer_student import _hash_tensors
    proof = saved.get('vjepa_spawn_phase_migration', {})
    param = 'vjepa_backend.spawn_phase_head.scores'
    if (not model.config.vjepa_spawn_phase_head or
            proof.get('protocol') != 'vjepa_spawn_phase_migration_v1' or
            proof.get('transferred_parameter') != param or param not in saved['model']):
        return False
    ledger = saved.get('vjepa_sr_training', {})
    return (proof.get('new_frozen_sha256') == ledger.get('frozen_parameters_sha256') ==
            ledger.get('combat_migration', {}).get('frozen_sha256')
            and isinstance(proof.get('parent_step'), int) and proof['parent_step'] <= saved['training_steps']
            and proof.get('fitted_sha256') == _hash_tensors({param: model.vjepa_backend.spawn_phase_head.scores}))


def compatible_policy_refresh(previous, model, saved):
    from dataclasses import replace
    return (not getattr(previous, 'vjepa_spawn_phase_head', False)
            and replace(previous, vjepa_spawn_phase_head=True) == model.config
            and verified_migration(model, saved))
