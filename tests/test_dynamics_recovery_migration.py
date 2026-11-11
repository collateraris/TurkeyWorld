"""Real AdamW continuity for an appended player pass head; CPU/tempfiles only."""
import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.dynamics_recovery_migration import enable_axis_fraction_checkpoint, migrate_dynamics_checkpoint
from turkeyworld.models import NeuralWorldModel, STATE_SCHEMA
from test_player_axis_fraction import axis_config


class DynamicsRecoveryMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def source(self):
        model = NeuralWorldModel(axis_config()).eval()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
        tracked = model.vjepa_backend.delta_head[-1].weight
        tracked.sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        saved = {'stage': 'world', 'pipeline_phase': 'teacher175vq_world',
            'model': model.state_dict(), 'config': asdict(model.config), 'state_schema': copy.deepcopy(STATE_SCHEMA),
            'optimizer': optimizer.state_dict(), 'optimizer_parameter_names': [list(dict(model.named_parameters()))],
            'training_steps': 1633, 'global_training_step': 1633,
            'training_counters': {'optimizer_updates': 1633, 'pending_microbatches': 0, 'fragments_seen': 45678},
            'ema_target_encoder': {'test': torch.tensor([2.])}, 'discriminator': {'weight': torch.tensor([3.])},
            'rng_torch': torch.random.get_rng_state(), 'rng_numpy': np.random.get_state(),
            'component_training': {'phase': 'dynamics', 'history': [{'phase': 'renderer', 'step': 1444}]}}
        return model, saved

    def test_old_tensors_optimizer_states_payloads_and_rng_are_preserved_exactly(self):
        old_model, saved = self.source()
        rng = torch.random.get_rng_state().clone()
        migrated, report = enable_axis_fraction_checkpoint(saved)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        self.assertEqual(report['source_step'], 1633)
        self.assertEqual(report['added_parameter_count'], 9346)
        self.assertEqual(len(report['added_parameters']), 4)
        for name, value in saved['model'].items():
            self.assertIs(migrated['model'][name], value)
        for key in ('ema_target_encoder', 'discriminator', 'rng_numpy', 'rng_torch', 'component_training', 'training_counters'):
            self.assertIs(migrated[key], saved[key])
        old_group, new_group = saved['optimizer']['param_groups'][0], migrated['optimizer']['param_groups'][0]
        self.assertEqual(new_group['params'][:len(old_group['params'])], old_group['params'])
        self.assertEqual(new_group['param_names'][:len(old_group['params'])], saved['optimizer_parameter_names'][0])
        self.assertEqual(new_group['lr'], old_group['lr'])
        self.assertEqual(new_group['betas'], old_group['betas'])
        self.assertEqual(len(migrated['optimizer']['state']), len(saved['optimizer']['state']))
        for identifier, state in saved['optimizer']['state'].items():
            self.assertIs(migrated['optimizer']['state'][identifier], state)
        model = NeuralWorldModel(axis_config('axis_fraction'))
        model.load_state_dict(migrated['model'], strict=True)
        optimizer = torch.optim.AdamW(model.parameters())
        optimizer.load_state_dict(migrated['optimizer'])
        old_names = list(dict(old_model.named_parameters()))
        self.assertEqual(list(dict(model.named_parameters()))[:len(old_names)], old_names)
        for name, parameter in model.named_parameters():
            if name.startswith('vjepa_backend.player_pass_head.'):
                self.assertNotIn(parameter, optimizer.state)
        head = model.vjepa_backend.player_pass_head
        self.assertEqual(head[-1].weight.abs().sum().item(), 0)
        self.assertEqual(head[-1].bias.abs().sum().item(), 0)

    def test_partial_or_unrelated_missing_keys_bad_optimizer_and_pending_group_rejected(self):
        _, source = self.source()
        for case in ('missing', 'partial', 'optimizer_names', 'pending'):
            with self.subTest(case=case):
                saved = {**source, 'model': dict(source['model'])}
                if case == 'missing':
                    del saved['model'][next(iter(saved['model']))]
                elif case == 'partial':
                    saved['model']['vjepa_backend.player_pass_head.0.weight'] = torch.zeros(1)
                elif case == 'optimizer_names':
                    saved['optimizer_parameter_names'] = [['wrong']]
                else:
                    saved['training_counters'] = {'pending_microbatches': 1}
                with self.assertRaises((ValueError, RuntimeError)):
                    enable_axis_fraction_checkpoint(saved)

    def test_idempotent_migration_never_duplicates_head_ids_or_resets_moments(self):
        _, saved = self.source()
        migrated, _ = enable_axis_fraction_checkpoint(saved)
        again, report = enable_axis_fraction_checkpoint(migrated)
        self.assertEqual(report['status'], 'already_enabled')
        self.assertEqual(report['added_parameter_count'], 0)
        self.assertEqual(again['optimizer']['param_groups'], migrated['optimizer']['param_groups'])
        self.assertEqual(again['optimizer_parameter_names'], migrated['optimizer_parameter_names'])

    def test_file_wrapper_keeps_source_immutable_and_refuses_live_or_same_destination(self):
        _, saved = self.source()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root/'checkpoints/archive.pt', root/'checkpoints/recovered.pt'
            source.parent.mkdir()
            torch.save(saved, source)
            before = source.read_bytes()
            with self.assertRaises(ValueError):
                migrate_dynamics_checkpoint(source, source)
            (root/'training.pid').write_text('123', encoding='ascii')
            with patch('turkeyworld.dynamics_recovery_migration.process_alive', return_value=True), self.assertRaises(RuntimeError):
                migrate_dynamics_checkpoint(source, output)
            (root/'training.pid').unlink()
            report = migrate_dynamics_checkpoint(source, output)
            self.assertEqual(source.read_bytes(), before)
            self.assertTrue(output.with_suffix('.pt.migration.json').is_file())
            self.assertEqual(report['player_motion_mode'], 'axis_fraction')
            loaded = torch.load(output, map_location='cpu', weights_only=False)
            self.assertEqual(loaded['global_training_step'], 1633)
            self.assertEqual(loaded['component_training'], saved['component_training'])


if __name__ == '__main__':
    unittest.main()
