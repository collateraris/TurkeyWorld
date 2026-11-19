"""Transfer only a fitted respawn head to a saved three-stream dynamics boundary."""
import argparse
from pathlib import Path

import torch

from .distill_renderer import _save
from .models import load_world_checkpoint
from .renderer_student import _hash_tensors
from .small_enemy_data import digest
from .training_pipeline import atomic_json
from .vjepa_combat_training import THREE_MODE, dynamics_parameter_trainable


KEY = 'vjepa_backend.spawn_phase_head.scores'


def attach(parent_path, candidate_path, output):
    if output.exists():
        raise FileExistsError(output)
    model, snapshot = load_world_checkpoint(parent_path, 'cpu')
    donor = torch.load(candidate_path, weights_only=False, mmap=True, map_location='cpu')
    profile = snapshot['vjepa_sr_training']
    if (snapshot['config'].get('vjepa_spawn_phase_head') or KEY in snapshot['model']
            or profile['training_profile'] != THREE_MODE or model.config.architecture != 'vjepa2_ac'
            or not model.config.endless or not model.config.game_training_v5 or not model.config.spatial_combat):
        raise ValueError('Expected an original saved three-mode V-JEPA checkpoint')
    if (not donor['config'].get('vjepa_spawn_phase_head') or
            {k:v for k,v in donor['config'].items() if k != 'vjepa_spawn_phase_head'}
            != {k:v for k,v in snapshot['config'].items() if k != 'vjepa_spawn_phase_head'} or
            donor['vjepa_sr_feedback'] != snapshot['vjepa_sr_feedback'] or
            donor['vjepa_sr_training']['source_step'] != profile['source_step'] or
            donor['training_steps'] > snapshot['training_steps']):
        raise ValueError('Incompatible model architecture, training lineage or visual feedback')
    scores = donor['model'][KEY]
    if (scores.shape != (8, 9) or scores.dtype != torch.float32 or not torch.isfinite(scores).all() or
            donor['vjepa_spawn_phase_training']['train_events'] <= 0):
        raise ValueError('Invalid fitted spawn weights')
    frozen = {name: p for name, p in model.named_parameters() if not dynamics_parameter_trainable(name, True)}
    before = _hash_tensors(frozen)
    if (profile['frozen_parameters_sha256'] != before or
            profile['combat_migration']['frozen_sha256'] != before):
        raise ValueError('Source boundary failed its recorded frozen-parameter fingerprint')
    after = _hash_tensors({**frozen, KEY:scores})
    snapshot['model'][KEY] = scores.clone()
    snapshot['config']['vjepa_spawn_phase_head'] = True
    snapshot['vjepa_sr_training'] = {
        **profile, 'frozen_parameters_sha256': after,
        'combat_migration': {**profile['combat_migration'], 'frozen_sha256': after},
    }
    snapshot['vjepa_spawn_phase_training'] = donor['vjepa_spawn_phase_training']
    audit = dict(protocol='vjepa_spawn_phase_migration_v1', parent_step=snapshot['training_steps'], world_updates=profile['updates'],
                 donor_step=donor['training_steps'], donor_path=str(candidate_path.resolve()),
                 transferred_parameter=KEY, fitted_sha256=_hash_tensors({KEY:scores}),
                 old_frozen_sha256=before, new_frozen_sha256=after,
                 previous_adam_slots=len(snapshot['optimizer']['state']),
                 parent=str(parent_path.resolve()), parent_sha256=digest(parent_path))
    snapshot['vjepa_spawn_phase_migration'] = audit
    _save(output, snapshot)
    atomic_json(output.with_suffix('.migration.json'), audit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    attach(args.parent, args.candidate, args.output)
