"""Old spatial weights/moments survive new combat, spawn and Flow branches."""
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


class CombatMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def source(self, explicit_false=False):
        from turkeyworld.models import NeuralWorldModel
        from turkeyworld.optimizer_migration import optimizer_parameter_names
        from turkeyworld.combat_migration import config_fingerprint
        from turkeyworld.vqgan import VQGANConfig
        from turkeyworld.vq_world import teacher175_vq_config
        codec = VQGANConfig(image_size=32, base_channels=8, latent_channels=16,
            embedding_dim=4, n_embed=32, data_init=False, kmeans_iterations=0)
        config = teacher175_vq_config(codec, spatial_entities=True,
            vjepa_width=32, vjepa_heads=4, vjepa_encoder_depth=1, vjepa_predictor_depth=1,
            vjepa_readout_mlp_ratio=1, vjepa_gradient_checkpointing=False,
            flow_width=32, flow_depth=1, flow_heads=4,
            flow_adapter_width=32, flow_adapter_depth=1, flow_adapter_heads=4)
        model = NeuralWorldModel(config)
        old_config = asdict(config)
        if not explicit_false:
            old_config.pop('spatial_combat')
        fingerprint = config_fingerprint(old_config) if not explicit_false else hashlib.sha256(
            json.dumps(old_config, sort_keys=True).encode()).hexdigest()
        marker = {'mode': 'from_scratch_world_and_renderer', 'teacher_size': '175m_vqgan',
            'image_size': 32, 'teacher_parameters': sum(p.numel() for p in model.parameters()),
            'teacher_config_fingerprint': fingerprint, 'codec_fingerprint': 'frozen-codec',
            'seed': 173, 'behavior_version': 'deterministic_enemy_v3_simple_spawn',
            'spatial_entities': True, 'spatial_entity_count': 9}
        initial = {'stage': 'world', 'config': old_config, 'model': copy.deepcopy(model.state_dict()),
            'training_steps': 0, 'global_training_step': 0, 'pipeline_phase': 'teacher175vq_initial',
            'initialization': copy.deepcopy(marker)}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=5e-5, betas=(.9, .95))
        parameters = dict(model.named_parameters())
        selected = ['vjepa_backend.renderer.map_encoder.tile_embedding.weight',
            'vjepa_backend.core.encoder.patch_embed.weight', 'vjepa_backend.delta_head.2.weight',
            'vjepa_backend.spatial_motion_head.2.weight', 'flow_renderer.output.weight',
            'flow_renderer.residual_adapter.output.weight']
        sum((parameters[name]-.1).square().mean() for name in selected).backward()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        saved = {'stage': 'world', 'config': old_config, 'model': model.state_dict(),
            'training_steps': 103, 'global_training_step': 103, 'pipeline_phase': 'teacher175vq_world',
            'optimizer': optimizer.state_dict(), 'optimizer_parameter_names': optimizer_parameter_names(model, optimizer),
            'ema_target_encoder': {'steps': 103, 'encoder': copy.deepcopy(model.core.encoder.state_dict())},
            'adversarial_training': {'updates': 103, 'discriminator': {'weight': torch.randn(4, 3)},
                'optimizer': {'state': {0: {'step': torch.tensor(103.), 'exp_avg': torch.ones(4)}}}},
            'flow_optimizer': {'state': {0: {'step': torch.tensor(103.), 'exp_avg': torch.ones(3)}}},
            'training_counters': {'optimizer_updates': 103, 'fragments_seen': 45360, 'pending_microsteps': 0},
            'motion_curriculum': {'start_optimizer_step': 66, 'phase': 'straight_1',
                'protocol': {'context': 4, 'max_horizon': 29, 'version': 'local_motion_curriculum_v2_future_cases',
                             'fingerprint': 'unchanged-source-corpus'}},
            'optimizer_budget': {'limit_updates': 6000, 'start_optimizer_step': 0, 'stop_optimizer_step': 6000},
            'rng_numpy': np.random.default_rng(73).bit_generator.state,
            'rng_torch': torch.random.get_rng_state(), 'rng_cuda': [torch.arange(4)],
            'initialization': copy.deepcopy(marker), 'last_metrics': {'train_loss': .7}}
        return model, saved, initial, marker, selected

    def fixture(self, root, explicit_false=False):
        _, saved, initial, marker, selected = self.source(explicit_false)
        checkpoints = root/'checkpoints'; checkpoints.mkdir(parents=True)
        main, origin = checkpoints/'teacher175vq_world.pt', checkpoints/'teacher_vjepa175vq_initial.pt'
        torch.save(saved, main); torch.save(initial, origin)
        (root/'initialization.json').write_text(json.dumps(marker), encoding='utf-8')
        budget = b'{"limit_updates":6000,"start_optimizer_step":0,"stop_optimizer_step":6000}\n'
        (root/'teacher_budget.json').write_bytes(budget)
        spatial = b'{"status":"complete","spatial_entities":true,"spatial_entity_count":9,"start_optimizer_step":66}\n'
        (root/'spatial_entities_migration.json').write_bytes(spatial)
        (root/'experiment_config175vq.json').write_text(json.dumps({'training_subset_manifest': 'fixed128k.json'}), encoding='utf-8')
        sr = root/'updated_sr.pt'
        torch.save({'stage': 'sr', 'config': {'architecture': 'edsr', 'channels': 4, 'blocks': 1, 'scale': 4},
            'training_steps': 20000, 'input_size': 32, 'output_size': 128, 'target_met': False}, sr)
        return saved, initial, main, origin, sr, budget, spatial

    def test_old_tensors_objects_rng_and_real_adam_moments_preserved_new_moments_empty(self):
        from turkeyworld.models import NeuralWorldModel, WorldModelConfig
        from turkeyworld.optimizer_migration import optimizer_state_for_trainables
        from turkeyworld.combat_migration import COMBAT_PREFIXES, enable_combat_checkpoint
        _, source, _, _, selected = self.source()
        before = torch.random.get_rng_state()
        migrated, report = enable_combat_checkpoint(source)
        torch.testing.assert_close(before, torch.random.get_rng_state(), rtol=0, atol=0)
        self.assertTrue(migrated['config']['spatial_combat'])
        self.assertEqual(report['start_optimizer_step'], 103)
        self.assertFalse(report['source_spatial_combat'])
        self.assertGreater(report['added_parameter_count'], 0)
        self.assertTrue(all(name.startswith(COMBAT_PREFIXES) for name in report['added_parameters']))
        self.assertTrue(any(name.startswith('flow_renderer.spatial_condition.') for name in report['added_parameters']))
        self.assertTrue(any(name.startswith('flow_renderer.residual_adapter.spatial_condition.') for name in report['added_parameters']))
        self.assertTrue(any(name.startswith('flow_renderer.wall_condition.') for name in report['added_parameters']))
        self.assertTrue(any(name.startswith('flow_renderer.residual_adapter.wall_condition.') for name in report['added_parameters']))
        self.assertTrue(any(name.startswith('vjepa_backend.spatial_spawn_output.') for name in report['added_parameters']))
        for name, value in source['model'].items():
            self.assertIs(migrated['model'][name], value)
            torch.testing.assert_close(migrated['model'][name], value, rtol=0, atol=0)
        for name in set(source)-{'model', 'config', 'initialization'}:
            self.assertIs(migrated[name], source[name])
        self.assertEqual(migrated['motion_curriculum']['start_optimizer_step'], 66)
        self.assertFalse(migrated['combat_conditioning_migration']['source_spatial_combat'])
        self.assertTrue(migrated['combat_conditioning_migration']['allowed_source_sampling_upgrade'])
        self.assertEqual(migrated['combat_conditioning_migration']['start_optimizer_step'], 103)
        model = NeuralWorldModel(WorldModelConfig(**migrated['config']))
        model.load_state_dict(migrated['model'], strict=True)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.9)
        mapped, mapping_report = optimizer_state_for_trainables(migrated, model, optimizer)
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
        self.assertTrue(set(report['added_parameters']).issubset(mapping_report['added_parameters']))
        for name in ('vjepa_backend.spatial_combat_head.2.weight', 'vjepa_backend.spatial_spawn_output.weight',
                     'vjepa_backend.renderer.spatial_combat_counts.weight', 'flow_renderer.spatial_condition.output.weight',
                     'flow_renderer.residual_adapter.spatial_condition.output.weight'):
            self.assertEqual(int(parameters[name].count_nonzero()), 0, name)

    def test_pure_idempotence_and_false_flag_fingerprint_compatibility(self):
        from turkeyworld.combat_migration import config_fingerprint, enable_combat_checkpoint
        _, source, _, _, _ = self.source()
        base = dict(source['config'])
        self.assertEqual(config_fingerprint(base), config_fingerprint({**base, 'spatial_combat': False}))
        minimal = {'kind': 'teacher'}
        self.assertEqual(config_fingerprint(minimal), config_fingerprint({**minimal, 'spatial_entities': False, 'spatial_combat': False}))
        migrated, first = enable_combat_checkpoint(source)
        again, report = enable_combat_checkpoint(migrated, seed=999)
        self.assertEqual(report['status'], 'already_enabled')
        self.assertEqual(report['added_parameter_count'], 0)
        self.assertIs(again['combat_conditioning_migration'], migrated['combat_conditioning_migration'])
        self.assertFalse(again['combat_conditioning_migration']['source_spatial_combat'])
        for name, value in migrated['model'].items():
            self.assertIs(again['model'][name], value)

    def test_partial_nonspatial_wrong_stage_and_unnamed_optimizer_rejected(self):
        from turkeyworld.combat_migration import enable_combat_checkpoint
        _, source, _, _, _ = self.source()
        for issue in ('stage', 'spatial', 'missing_old', 'unnamed', 'partial', 'wrong_native'):
            with self.subTest(issue=issue):
                wrong = {**source, 'config': dict(source['config']), 'model': dict(source['model'])}
                if issue == 'stage': wrong['stage'] = 'sr'
                elif issue == 'spatial': wrong['config']['spatial_entities'] = False
                elif issue == 'missing_old': wrong['model'].pop('vjepa_backend.delta_head.2.weight')
                elif issue == 'unnamed': wrong['optimizer_parameter_names'] = None
                elif issue == 'wrong_native': wrong['config']['image_size'] = 64
                else: wrong['model']['vjepa_backend.spatial_combat_head.malformed'] = torch.zeros(1)
                with self.assertRaises((ValueError, RuntimeError)): enable_combat_checkpoint(wrong)

    def test_transaction_backups_budget_old_spatial_marker_and_curriculum_survive_idempotent_rerun(self):
        from turkeyworld.combat_migration import config_fingerprint, migrate_combat_run
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            saved, initial, main, origin, sr, budget, spatial = self.fixture(root)
            hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (main, origin, root/'initialization.json')}
            report = migrate_combat_run(root, sr)
            self.assertEqual(report['status'], 'complete')
            self.assertFalse(report['source_spatial_combat'])
            self.assertEqual(report['start_optimizer_step'], 103)
            self.assertEqual(report['target_seconds'], 30.)
            self.assertTrue(report['allowed_source_sampling_upgrade'])
            self.assertEqual(report['sr_quality'], {'training_steps': 20000, 'target_met': False})
            self.assertEqual((root/'teacher_budget.json').read_bytes(), budget)
            self.assertEqual((root/'spatial_entities_migration.json').read_bytes(), spatial)
            for path, previous in ((main, saved), (origin, initial)):
                upgraded = torch.load(path, weights_only=False)
                self.assertTrue(upgraded['config']['spatial_entities'])
                self.assertTrue(upgraded['config']['spatial_combat'])
                self.assertEqual(upgraded['training_steps'], previous['training_steps'])
                self.assertEqual(upgraded['global_training_step'], previous['global_training_step'])
                self.assertFalse(upgraded['combat_conditioning_migration']['source_spatial_combat'])
                self.assertTrue(upgraded['combat_conditioning_migration']['allowed_source_sampling_upgrade'])
                for name, value in previous['model'].items():
                    torch.testing.assert_close(upgraded['model'][name], value, rtol=0, atol=0)
                backup = Path(report['backups'][str(path)])
                self.assertIn('_before_combat', backup.name)
                self.assertEqual(hashlib.sha256(backup.read_bytes()).hexdigest(), hashes[path])
            self.assertEqual(torch.load(main, weights_only=False)['motion_curriculum'], saved['motion_curriculum'])
            marker = json.loads((root/'initialization.json').read_text())
            self.assertEqual(hashlib.sha256(Path(report['backups'][str(root/'initialization.json')]).read_bytes()).hexdigest(),
                             hashes[root/'initialization.json'])
            self.assertTrue(marker['spatial_combat'])
            self.assertEqual(marker['teacher_parameters'], report['generator_parameters'])
            self.assertEqual(marker['teacher_config_fingerprint'], config_fingerprint(torch.load(main, weights_only=False)['config']))
            paths = (main, origin, root/'initialization.json', root/'combat_conditioning_migration.json')
            mtimes = {path: path.stat().st_mtime_ns for path in paths}
            self.assertEqual(migrate_combat_run(root, sr), report)
            self.assertEqual(mtimes, {path: path.stat().st_mtime_ns for path in paths})

    def test_interrupted_each_commit_boundary_recovers_without_resetting_source_or_backups(self):
        from turkeyworld import combat_migration as migration
        for failure in ('initial', 'world', 'initialization', 'complete'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                saved, _, main, origin, sr, budget, spatial = self.fixture(root, explicit_false=True)
                main_hash = hashlib.sha256(main.read_bytes()).hexdigest()
                replace, atomic = migration.replace_file, migration.atomic_json
                def interrupt_replace(source, destination):
                    if (failure == 'world' and destination == main) or (failure == 'initial' and destination == origin):
                        raise OSError('simulated interrupted commit')
                    return replace(source, destination)
                def interrupt_json(path, value):
                    if (failure == 'initialization' and path.name == 'initialization.json'
                            or failure == 'complete' and path.name == 'combat_conditioning_migration.json'
                            and value.get('status') == 'complete'):
                        raise OSError('simulated interrupted commit')
                    return atomic(path, value)
                with patch.object(migration, 'replace_file', side_effect=interrupt_replace),\
                        patch.object(migration, 'atomic_json', side_effect=interrupt_json):
                    with self.assertRaisesRegex(OSError, 'interrupted'):
                        migration.migrate_combat_run(root, sr)
                self.assertEqual(json.loads((root/'combat_conditioning_migration.json').read_text())['status'], 'pending')
                report = migration.migrate_combat_run(root, sr)
                self.assertEqual(report['status'], 'complete')
                self.assertFalse(report['source_spatial_combat'])
                self.assertEqual(report['world_report']['status'], 'expanded')
                self.assertEqual(report['initial_report']['status'], 'expanded')
                self.assertEqual(report['start_optimizer_step'], 103)
                self.assertEqual(hashlib.sha256(Path(report['backups'][str(main)]).read_bytes()).hexdigest(), main_hash)
                upgraded = torch.load(main, weights_only=False)
                self.assertEqual(upgraded['motion_curriculum']['start_optimizer_step'], 66)
                for name, value in saved['model'].items():
                    torch.testing.assert_close(upgraded['model'][name], value, rtol=0, atol=0)
                self.assertEqual((root/'teacher_budget.json').read_bytes(), budget)
                self.assertEqual((root/'spatial_entities_migration.json').read_bytes(), spatial)
                self.assertFalse(list((root/'checkpoints').glob('*.tmp')))

    def test_live_process_or_invalid_sr_and_initial_rejected_before_mutation(self):
        from turkeyworld import combat_migration as migration
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, initial, main, origin, sr, _, _ = self.fixture(root)
            original = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (main, origin, root/'initialization.json')}
            (root/'training.pid').write_text('12345')
            with patch.object(migration, 'process_alive', return_value=True),\
                    patch.object(migration.torch, 'load', side_effect=AssertionError('No live checkpoint load')):
                with self.assertRaisesRegex(RuntimeError, 'Stop the world'): migration.migrate_combat_run(root, sr)
            (root/'training.pid').unlink()
            initial['training_steps'] = 1; torch.save(initial, origin)
            with self.assertRaisesRegex(ValueError, 'source phase or step'): migration.migrate_combat_run(root, sr)
            initial['training_steps'] = 0; torch.save(initial, origin)
            torch.save({'stage': 'sr', 'training_steps': 0, 'input_size': 32, 'output_size': 128, 'config': {'scale': 4}}, sr)
            with self.assertRaisesRegex(ValueError, 'trained x4'): migration.migrate_combat_run(root, sr)
            self.assertEqual(hashlib.sha256(main.read_bytes()).hexdigest(), original[main])
            self.assertEqual(hashlib.sha256((root/'initialization.json').read_bytes()).hexdigest(), original[root/'initialization.json'])
            self.assertFalse((root/'combat_conditioning_migration.json').exists())
            self.assertFalse(list((root/'checkpoints').glob('*_before_combat.pt')))

    def test_pending_checkpoint_advance_and_changed_transaction_request_are_rejected(self):
        from turkeyworld import combat_migration as migration
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            saved, _, main, origin, sr, _, _ = self.fixture(root)
            replace = migration.replace_file
            def fail_world(source, destination):
                if destination == main:
                    raise OSError('interrupted before world')
                return replace(source, destination)
            with patch.object(migration, 'replace_file', side_effect=fail_world):
                with self.assertRaises(OSError): migration.migrate_combat_run(root, sr)
            journal = root/'combat_conditioning_migration.json'
            before = journal.read_bytes()
            with self.assertRaisesRegex(ValueError, 'different run/duration/seed'):
                migration.migrate_combat_run(root, sr, seed=174)
            self.assertEqual(journal.read_bytes(), before)
            advanced = dict(saved); advanced['global_training_step'] = 104
            torch.save(advanced, main)
            original_backup = Path(json.loads(before)['backups'][str(main)])
            backup_hash = hashlib.sha256(original_backup.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'advanced while combat migration was pending'):
                migration.migrate_combat_run(root, sr)
            self.assertEqual(journal.read_bytes(), before)
            self.assertEqual(hashlib.sha256(original_backup.read_bytes()).hexdigest(), backup_hash)

    def test_real_profile_parameter_count_includes_new_wall_branches_and_frozen_codec(self):
        from turkeyworld.models import NeuralWorldModel
        from turkeyworld.vqgan import VQGANConfig
        from turkeyworld.vq_world import teacher175_vq_config
        with torch.device('meta'):
            model = NeuralWorldModel(teacher175_vq_config(VQGANConfig(image_size=32, n_embed=1488),
                spatial_entities=True, spatial_combat=True))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 176_492_456)
        self.assertEqual(sum(parameter.numel() for name, parameter in model.named_parameters()
            if name.startswith(('flow_renderer.wall_condition.',
                                'flow_renderer.residual_adapter.wall_condition.'))), 264_192)


if __name__ == '__main__':
    unittest.main()
