"""Real moments and stopped-run recovery survive adding spatial branches."""
from dataclasses import asdict
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


class SpatialMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def source(self):
        from turkeyworld.models import NeuralWorldModel, WorldModelConfig
        from turkeyworld.optimizer_migration import optimizer_parameter_names
        from turkeyworld.spatial_migration import config_fingerprint
        config = WorldModelConfig(kind='teacher', architecture='vjepa2_ac', image_size=32,
            vjepa_patch_size=4, endless=True, renderer_mode='direct', vjepa_width=32,
            vjepa_encoder_depth=1, vjepa_predictor_depth=1, vjepa_heads=4)
        model = NeuralWorldModel(config)
        old_config = asdict(config)
        old_config.pop('spatial_entities')
        count = sum(parameter.numel() for parameter in model.parameters())
        marker = {'mode': 'from_scratch_world_and_renderer', 'teacher_size': '175m_vqgan',
            'image_size': 32, 'teacher_parameters': count, 'teacher_config_fingerprint': config_fingerprint(old_config),
            'codec_fingerprint': 'frozen-codec', 'seed': 173, 'behavior_version': 'deterministic_enemy_v3_simple_spawn'}
        initial = {'stage': 'world', 'config': old_config, 'model': copy.deepcopy(model.state_dict()),
            'training_steps': 0, 'global_training_step': 0, 'pipeline_phase': 'teacher175vq_initial',
            'initialization': copy.deepcopy(marker)}
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(.9, .95))
        parameters = dict(model.named_parameters())
        selected = ['vjepa_backend.renderer.map_encoder.tile_embedding.weight',
            'vjepa_backend.core.encoder.patch_embed.weight', 'vjepa_backend.delta_head.2.weight']
        sum(parameters[name].square().mean() for name in selected).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        saved = {'stage': 'world', 'config': old_config, 'model': model.state_dict(),
            'training_steps': 66, 'global_training_step': 66, 'pipeline_phase': 'teacher175vq_world',
            'optimizer': optimizer.state_dict(), 'optimizer_parameter_names': optimizer_parameter_names(model, optimizer),
            'ema_target_encoder': {'steps': 66, 'encoder': copy.deepcopy(model.core.encoder.state_dict())},
            'adversarial_training': {'updates': 66, 'discriminator': {'weight': torch.randn(4, 3)},
                                     'optimizer': {'state': {0: {'step': torch.tensor(66.), 'exp_avg': torch.ones(4)}}}},
            'training_counters': {'optimizer_updates': 66, 'fragments_seen': 66528, 'pending_microsteps': 0},
            'rng_numpy': np.random.default_rng(73).bit_generator.state,
            'rng_torch': torch.random.get_rng_state(), 'rng_cuda': [torch.arange(4)],
            'initialization': copy.deepcopy(marker), 'last_metrics': {'train_loss': .7}}
        return model, saved, initial, marker, selected

    def fixture(self, root):
        _, saved, initial, marker, selected = self.source()
        checkpoints = root/'checkpoints'
        checkpoints.mkdir(parents=True)
        main, origin = checkpoints/'teacher175vq_world.pt', checkpoints/'teacher_vjepa175vq_initial.pt'
        torch.save(saved, main)
        torch.save(initial, origin)
        (root/'initialization.json').write_text(json.dumps(marker), encoding='utf-8')
        budget = b'{"limit_updates":6000,"start_optimizer_step":0,"stop_optimizer_step":6000}\n'
        (root/'teacher_budget.json').write_bytes(budget)
        sr = root/'updated_sr.pt'
        torch.save({'stage': 'sr', 'config': {'architecture': 'edsr', 'channels': 4, 'blocks': 1, 'scale': 4, 'res_scale': .1},
                    'training_steps': 20000, 'input_size': 32, 'output_size': 128, 'target_met': False}, sr)
        return saved, initial, main, origin, sr, budget

    def test_all_old_tensors_payload_and_global_rng_preserved_while_new_moments_start_empty(self):
        from turkeyworld.models import NeuralWorldModel, WorldModelConfig
        from turkeyworld.optimizer_migration import optimizer_state_for_trainables
        from turkeyworld.spatial_migration import enable_spatial_checkpoint
        _, source, _, _, selected = self.source()
        before = torch.random.get_rng_state()
        migrated, report = enable_spatial_checkpoint(source)
        torch.testing.assert_close(before, torch.random.get_rng_state(), rtol=0, atol=0)
        self.assertTrue(migrated['config']['spatial_entities'])
        self.assertGreater(report['added_parameter_count'], 0)
        self.assertEqual(report['start_optimizer_step'], 66)
        for name, value in source['model'].items():
            self.assertIs(migrated['model'][name], value)
            torch.testing.assert_close(migrated['model'][name], value, rtol=0, atol=0)
        for name in set(source)-{'model', 'config', 'initialization'}:
            self.assertIs(migrated[name], source[name])
        model = NeuralWorldModel(WorldModelConfig(**migrated['config']))
        model.load_state_dict(migrated['model'], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.9)
        mapped, _ = optimizer_state_for_trainables(migrated, model, optimizer)
        optimizer.load_state_dict(mapped)
        self.assertEqual(optimizer.param_groups[0]['lr'], 5e-5)
        parameters = dict(model.named_parameters())
        names = source['optimizer_parameter_names'][0]
        identifiers = source['optimizer']['param_groups'][0]['params']
        for name in selected:
            old = source['optimizer']['state'][identifiers[names.index(name)]]
            for key, value in old.items():
                torch.testing.assert_close(optimizer.state[parameters[name]][key], value, rtol=0, atol=0)
        self.assertTrue(all(parameters[name] not in optimizer.state for name in report['added_parameters']))
        self.assertEqual(int(model.vjepa_backend.spatial_visual_attention.output.weight.count_nonzero()), 0)
        self.assertEqual(int(model.vjepa_backend.spatial_motion_head[-1].weight.count_nonzero()), 0)

    def test_partial_or_unnamed_ambiguous_sources_are_rejected(self):
        from turkeyworld.spatial_migration import enable_spatial_checkpoint
        _, source, _, _, _ = self.source()
        wrong = {**source, 'optimizer_parameter_names': None}
        with self.assertRaisesRegex(ValueError, 'explicitly named'):
            enable_spatial_checkpoint(wrong)
        wrong = {**source, 'stage': 'sr'}
        with self.assertRaisesRegex(ValueError, 'world-model'):
            enable_spatial_checkpoint(wrong)
        wrong = {**source, 'model': dict(source['model'])}
        wrong['model'].pop('vjepa_backend.delta_head.2.weight')
        with self.assertRaisesRegex(ValueError, 'non-spatial'):
            enable_spatial_checkpoint(wrong)

    def test_run_migrates_resume_and_initial_preserving_backups_budget_and_start66(self):
        from turkeyworld.spatial_migration import config_fingerprint, migrate_spatial_run
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            saved, initial, main, origin, sr, budget = self.fixture(root)
            hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (main, origin)}
            report = migrate_spatial_run(root, sr_checkpoint=sr)
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['start_optimizer_step'], 66)
            self.assertEqual(report['target_seconds'], 30.)
            self.assertEqual(report['sr_checkpoint'], str(sr.resolve()))
            self.assertEqual((root/'teacher_budget.json').read_bytes(), budget)
            for path, previous in ((main, saved), (origin, initial)):
                upgraded = torch.load(path, weights_only=False)
                self.assertTrue(upgraded['config']['spatial_entities'])
                self.assertEqual(upgraded['training_steps'], previous['training_steps'])
                for name, value in previous['model'].items():
                    torch.testing.assert_close(upgraded['model'][name], value, rtol=0, atol=0)
                backup = Path(report['backups'][str(path)])
                self.assertEqual(hashlib.sha256(backup.read_bytes()).hexdigest(), hashes[path])
            marker = json.loads((root/'initialization.json').read_text())
            self.assertTrue(marker['spatial_entities'])
            self.assertEqual(marker['spatial_entity_count'], 2)
            self.assertEqual(marker['teacher_config_fingerprint'], config_fingerprint(torch.load(main, weights_only=False)['config']))
            mtimes = {path: path.stat().st_mtime_ns for path in (main, origin, root/'initialization.json')}
            self.assertEqual(migrate_spatial_run(root, sr_checkpoint=sr), report)
            self.assertEqual(mtimes, {path: path.stat().st_mtime_ns for path in mtimes})

    def test_interrupted_two_checkpoint_commit_recovers_without_overwriting_old_backups(self):
        from turkeyworld import spatial_migration as migration
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, _, main, origin, sr, budget = self.fixture(root)
            main_hash = hashlib.sha256(main.read_bytes()).hexdigest()
            replace = migration.replace_file
            def fail_main(source, destination):
                if destination == main:
                    raise OSError('simulated interrupted commit')
                return replace(source, destination)
            with patch.object(migration, 'replace_file', side_effect=fail_main):
                with self.assertRaisesRegex(OSError, 'interrupted'):
                    migration.migrate_spatial_run(root, sr_checkpoint=sr)
            self.assertEqual(json.loads((root/'spatial_entities_migration.json').read_text())['status'], 'pending')
            self.assertTrue(torch.load(origin, weights_only=False)['config']['spatial_entities'])
            self.assertFalse(torch.load(main, weights_only=False)['config'].get('spatial_entities', False))
            report = migration.migrate_spatial_run(root, sr_checkpoint=sr)
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['world_report']['status'], 'expanded')
            self.assertEqual(hashlib.sha256(Path(report['backups'][str(main)]).read_bytes()).hexdigest(), main_hash)
            self.assertEqual((root/'teacher_budget.json').read_bytes(), budget)
            self.assertFalse(list((root/'checkpoints').glob('*.tmp')))

    def test_live_worker_or_wrong_initial_stage_rejected_before_mutation(self):
        from turkeyworld import spatial_migration as migration
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, initial, main, origin, sr, _ = self.fixture(root)
            before = hashlib.sha256(main.read_bytes()).hexdigest()
            (root/'training.pid').write_text('12345')
            with patch.object(migration, 'process_alive', return_value=True), \
                    patch.object(migration.torch, 'load', side_effect=AssertionError('No live checkpoint load')):
                with self.assertRaisesRegex(RuntimeError, 'Stop the world'):
                    migration.migrate_spatial_run(root, sr_checkpoint=sr)
            (root/'training.pid').unlink()
            initial['training_steps'] = 1
            torch.save(initial, origin)
            with self.assertRaisesRegex(ValueError, 'source phase or step'):
                migration.migrate_spatial_run(root, sr_checkpoint=sr)
            self.assertEqual(hashlib.sha256(main.read_bytes()).hexdigest(), before)
            self.assertFalse((root/'spatial_entities_migration.json').exists())


if __name__ == '__main__':
    unittest.main()
