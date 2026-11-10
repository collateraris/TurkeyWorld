"""Isolated CPU migration/transaction checks; no real training files or GPU."""
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
from torch.nn import functional as F


PATCH_KEYS = {prefix+name for prefix in ('flow_renderer.', 'flow_renderer.residual_adapter.')
              for name in ('image_input.weight', 'position', 'output.weight', 'output.bias')}


class TrainingV5MigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def source(self, omit_false=False):
        from turkeyworld.combat_migration import config_fingerprint
        from turkeyworld.models import NeuralWorldModel
        from turkeyworld.optimizer_migration import optimizer_parameter_names
        from turkeyworld.vq_world import teacher175_vq_config
        from turkeyworld.vqgan import VQGANConfig
        codec = VQGANConfig(image_size=32, base_channels=8, latent_channels=16,
            embedding_dim=4, n_embed=16, data_init=False, kmeans_iterations=0)
        config = teacher175_vq_config(codec, spatial_entities=True, spatial_combat=True,
            vjepa_width=16, vjepa_heads=2, vjepa_encoder_depth=1, vjepa_predictor_depth=1,
            vjepa_readout_mlp_ratio=1, vjepa_mlp_ratio=2, vjepa_gradient_checkpointing=False,
            flow_width=16, flow_depth=1, flow_heads=4, flow_mlp_hidden=32,
            flow_adapter_width=16, flow_adapter_depth=1, flow_adapter_heads=4,
            flow_adapter_mlp_ratio=2)
        model = NeuralWorldModel(config)
        old_config = asdict(config)
        if omit_false:
            for key in ('game_training_v5', 'flow_block_conditioning', 'flow_patch_size'):
                old_config.pop(key)
        marker = {'teacher_config_fingerprint': config_fingerprint(old_config),
            'teacher_parameters': sum(p.numel() for p in model.parameters()),
            'teacher_size': '175m_vqgan', 'mode': 'from_scratch_world_and_renderer',
            'codec_fingerprint': 'frozen-codec-identity', 'spatial_entities': True,
            'spatial_combat': True, 'image_size': 32, 'seed': 173}
        initial = {'stage': 'world', 'config': copy.deepcopy(old_config),
            'model': copy.deepcopy(model.state_dict()), 'training_steps': 0,
            'global_training_step': 0, 'pipeline_phase': 'teacher175vq_initial',
            'initialization': copy.deepcopy(marker)}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=5e-5, betas=(.9, .95), weight_decay=.01)
        parameters = dict(model.named_parameters())
        unchanged = ['vjepa_backend.core.encoder.patch_embed.weight',
                     'vjepa_backend.delta_head.2.weight', 'flow_renderer.spatial_condition.output.weight']
        # A real AdamW step creates all eight patch moments and non-patch ones.
        for name in sorted(PATCH_KEYS)+unchanged:
            parameters[name].grad = torch.full_like(parameters[name], .01)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        saved = {'stage': 'world', 'config': old_config, 'model': model.state_dict(),
            'training_steps': 724, 'global_training_step': 724, 'pipeline_phase': 'teacher175vq_world',
            'optimizer': optimizer.state_dict(), 'optimizer_parameter_names': optimizer_parameter_names(model, optimizer),
            'ema_target_encoder': {'steps': 724, 'encoder': copy.deepcopy(model.core.encoder.state_dict())},
            'adversarial_training': {'updates': 700, 'counterfactual_updates': 680,
                'discriminator': {'weight': torch.arange(12).reshape(4, 3).float()},
                'optimizer': {'state': {0: {'step': torch.tensor(700.), 'exp_avg': torch.ones(4)}}}},
            'training_counters': {'optimizer_updates': 724, 'fragments_seen': 123456,
                'gradient_accumulation_steps': 5, 'pending_microbatches': 0},
            'motion_curriculum': {'start_optimizer_step': 66, 'phase': 'long_29'},
            'optimizer_budget': {'limit_updates': 6000, 'stop_optimizer_step': 6000},
            'rng_numpy': np.random.default_rng(17).bit_generator.state,
            'rng_torch': torch.random.get_rng_state(), 'rng_cuda': [torch.arange(8, dtype=torch.uint8)],
            'initialization': copy.deepcopy(marker), 'last_metrics': {'train_loss': .7}}
        return model, saved, initial, marker, unchanged

    def fixture(self, root):
        _, saved, initial, marker, _ = self.source(omit_false=True)
        directory = root/'checkpoints'; directory.mkdir(parents=True)
        main, origin = directory/'teacher175vq_world.pt', directory/'teacher_vjepa175vq_initial.pt'
        torch.save(saved, main); torch.save(initial, origin)
        (root/'initialization.json').write_text(json.dumps(marker), encoding='utf-8')
        (root/'teacher_budget.json').write_bytes(b'{"stop_optimizer_step":6000,"start_optimizer_step":0}\n')
        (root/'training.pid').write_text('12345')
        parent = root/'controller'; parent.mkdir()
        (parent/'training.pid').write_text('54321')
        (root/'pipeline_parent.json').write_text(json.dumps({'run_dir': str(parent)}))
        return saved, initial, main, origin

    def test_all_unchanged_tensors_payloads_rng_and_nonpatch_adam_states_are_exact(self):
        from turkeyworld.training_v5_migration import upgrade_training_checkpoint, NEW_PREFIXES
        from turkeyworld.models import NeuralWorldModel, WorldModelConfig
        from turkeyworld.optimizer_migration import optimizer_state_for_trainables
        _, saved, _, _, unchanged = self.source()
        rng = torch.random.get_rng_state().clone()
        upgraded, report = upgrade_training_checkpoint(saved)
        torch.testing.assert_close(rng, torch.random.get_rng_state(), atol=0, rtol=0)
        self.assertEqual(set(report['resized_tensors']), PATCH_KEYS)
        self.assertEqual(set(report['reset_patch_optimizer_states']), PATCH_KEYS)
        self.assertTrue(report['new_tensors'])
        self.assertTrue(all(name.startswith(NEW_PREFIXES) for name in report['new_tensors']))
        for name, value in saved['model'].items():
            if name not in PATCH_KEYS:
                self.assertIs(upgraded['model'][name], value)
                torch.testing.assert_close(upgraded['model'][name], value, atol=0, rtol=0)
        for name in set(saved)-{'model', 'config', 'optimizer', 'initialization'}:
            self.assertIs(upgraded[name], saved[name], name)
        self.assertEqual(upgraded['training_counters']['fragments_seen'], 123456)
        self.assertEqual(upgraded['global_training_step'], 724)
        self.assertEqual(upgraded['motion_curriculum']['start_optimizer_step'], 66)
        model = NeuralWorldModel(WorldModelConfig(**upgraded['config']))
        model.load_state_dict(upgraded['model'], strict=True)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.5)
        mapped, _ = optimizer_state_for_trainables(upgraded, model, optimizer)
        optimizer.load_state_dict(mapped)
        self.assertEqual(optimizer.param_groups[0]['lr'], 5e-5)
        self.assertEqual(optimizer.param_groups[0]['betas'], (.9, .95))
        parameters = dict(model.named_parameters())
        ids = dict(zip(saved['optimizer_parameter_names'][0], saved['optimizer']['param_groups'][0]['params']))
        for name in unchanged:
            for key, value in saved['optimizer']['state'][ids[name]].items():
                torch.testing.assert_close(optimizer.state[parameters[name]][key], value, atol=0, rtol=0)
        self.assertEqual(set(saved['optimizer']['state'])-set(upgraded['optimizer']['state']), {ids[name] for name in PATCH_KEYS})
        for name in PATCH_KEYS | set(report['new_tensors']):
            if name in parameters:
                self.assertNotIn(parameters[name], optimizer.state, name)
        self.assertEqual(model.flow_renderer.patch, 2)
        self.assertEqual(tuple(model.flow_renderer.position.shape), (1, 16, 16))
        self.assertTrue(upgraded['config']['game_training_v5'])
        self.assertTrue(upgraded['config']['flow_block_conditioning'])
        self.assertEqual(int(model.vjepa_backend.spatial_spawn_anchor_head.weight.count_nonzero()), 0)
        state = torch.zeros(1, 90); state[:, 3:6] = 1
        with torch.inference_mode():
            model.eval(); model.encode_map(torch.zeros(1, 16, 16, dtype=torch.long))
            velocity = model.flow_renderer.velocity(torch.randn(1, 4, 8, 8), torch.tensor([.5]),
                                                    torch.zeros(1, 4, 8, 8), state)
        self.assertEqual(tuple(velocity.shape), (1, 4, 8, 8))
        self.assertTrue(bool(torch.isfinite(velocity).all()))

    def test_patch_input_output_and_position_mapping_have_known_constant_responses(self):
        from turkeyworld.training_v5_migration import resize_patch_tensor
        config = {'flow_patch_size': 2, 'renderer_codec_config': {'embedding_dim': 2}}
        image = torch.arange(3*6*4*4, dtype=torch.float32).reshape(3, 6, 4, 4)/100
        mapped = resize_patch_tensor('flow_renderer.image_input.weight', image, torch.empty(3, 6, 2, 2), config)
        torch.testing.assert_close(mapped, image.reshape(3, 6, 2, 2, 2, 2).sum((3, 5)), atol=1e-6, rtol=1e-6)
        old = F.conv2d(torch.ones(1, 6, 4, 4), image)
        new = F.conv2d(torch.ones(1, 6, 2, 2), mapped)
        torch.testing.assert_close(old, new, atol=2e-5, rtol=1e-6)
        for suffix in ('weight', 'bias'):
            source = torch.full((2*16, 5), .7) if suffix == 'weight' else torch.full((2*16,), .7)
            target = torch.empty(2*4, 5) if suffix == 'weight' else torch.empty(2*4)
            result = resize_patch_tensor('flow_renderer.residual_adapter.output.'+suffix, source, target, config)
            torch.testing.assert_close(result, torch.full_like(target, .7), atol=0, rtol=0)
        position = torch.full((1, 4, 5), .3)
        result = resize_patch_tensor('flow_renderer.position', position, torch.empty(1, 16, 5), config)
        torch.testing.assert_close(result, torch.full_like(result, .3), atol=3e-8, rtol=0)

    def test_unrecognized_resize_partial_groups_and_missing_legacy_keys_are_rejected(self):
        from turkeyworld.training_v5_migration import upgrade_training_checkpoint, resize_patch_tensor
        _, saved, _, _, _ = self.source()
        config = {'flow_patch_size': 2, 'renderer_codec_config': {'embedding_dim': 4}}
        with self.assertRaises(ValueError):
            resize_patch_tensor('flow_renderer.spatial_condition.output.weight',
                torch.ones(64, 16), torch.empty(16, 16), config)
        for issue in ('unexpected_shape', 'missing_old', 'pending', 'unnamed', 'unknown_tensor'):
            with self.subTest(issue=issue):
                wrong = {**saved, 'model': dict(saved['model'])}
                if issue == 'unexpected_shape':
                    wrong['model']['flow_renderer.spatial_condition.output.weight'] = torch.ones(64, 16)
                elif issue == 'missing_old': wrong['model'].pop('vjepa_backend.delta_head.2.weight')
                elif issue == 'pending': wrong['training_counters'] = {'pending_microbatches': 1}
                elif issue == 'unnamed': wrong['optimizer_parameter_names'] = None
                else: wrong['model']['flow_renderer.unknown.weight'] = torch.zeros(1)
                with self.assertRaises((ValueError, RuntimeError)):
                    upgrade_training_checkpoint(wrong)

    def test_legacy_false_configuration_identity_and_pure_upgrade_idempotence(self):
        from turkeyworld.combat_migration import config_fingerprint
        from turkeyworld.training_v5_migration import upgrade_training_checkpoint
        _, saved, _, _, _ = self.source(omit_false=True)
        normalized = {**saved['config'], 'game_training_v5': False,
                      'flow_block_conditioning': False, 'flow_patch_size': 4}
        self.assertEqual(config_fingerprint(saved['config']), config_fingerprint(normalized))
        first, _ = upgrade_training_checkpoint(saved)
        again, report = upgrade_training_checkpoint(first)
        self.assertEqual(report['status'], 'already_enabled')
        self.assertFalse(report['resized_tensors'])
        self.assertFalse(report['new_tensors'])
        for name, value in first['model'].items():
            self.assertIs(again['model'][name], value)
        self.assertEqual(first['optimizer'], again['optimizer'])

    def test_transaction_preserves_backups_budget_and_completed_rerun_is_read_only(self):
        from turkeyworld import training_v5_migration as migration
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            saved, initial, main, origin = self.fixture(root)
            paths = (main, origin, root/'initialization.json')
            hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
            budget = (root/'teacher_budget.json').read_bytes()
            with patch.object(migration, 'process_alive', return_value=False) as alive:
                report = migration.migrate_training_run(root)
            self.assertEqual(alive.call_count, 2)
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['source_step'], 724)
            self.assertEqual((root/'teacher_budget.json').read_bytes(), budget)
            for path, source in ((main, saved), (origin, initial)):
                upgraded = torch.load(path, map_location='cpu', weights_only=False)
                self.assertTrue(upgraded['config']['game_training_v5'])
                self.assertEqual(upgraded['global_training_step'], source['global_training_step'])
                for name, value in source['model'].items():
                    if name not in PATCH_KEYS:
                        torch.testing.assert_close(upgraded['model'][name], value, atol=0, rtol=0)
            for source, backup in report['backups'].items():
                self.assertEqual(hashlib.sha256(Path(backup).read_bytes()).hexdigest(), hashes[source])
            watched = paths+(root/'training_v5_migration.json',)
            mtimes = {path: path.stat().st_mtime_ns for path in watched}
            with patch.object(migration, 'process_alive', return_value=False), \
                    patch.object(migration, 'upgrade_training_checkpoint', side_effect=AssertionError('double migration')):
                self.assertEqual(migration.migrate_training_run(root), report)
            self.assertEqual(mtimes, {path: path.stat().st_mtime_ns for path in watched})
            self.assertFalse(list((root/'checkpoints').glob('*.tmp')))

    def test_pending_commit_recovery_uses_immutable_sources_and_does_not_double_pool(self):
        from turkeyworld import training_v5_migration as migration
        for boundary in ('initial', 'initialization', 'complete'):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                saved, _, main, origin = self.fixture(root)
                main_hash = hashlib.sha256(main.read_bytes()).hexdigest()
                replace, atomic = migration.replace_file, migration.atomic_json

                def fail_replace(source, destination):
                    if boundary == 'initial' and destination == origin:
                        raise OSError('simulated interrupted commit')
                    return replace(source, destination)

                def fail_json(path, value):
                    if (boundary == 'initialization' and path.name == 'initialization.json'
                            or boundary == 'complete' and path.name == 'training_v5_migration.json'
                            and value.get('status') == 'complete'):
                        raise OSError('simulated interrupted commit')
                    return atomic(path, value)

                with patch.object(migration, 'process_alive', return_value=False), \
                        patch.object(migration, 'replace_file', side_effect=fail_replace), \
                        patch.object(migration, 'atomic_json', side_effect=fail_json):
                    with self.assertRaisesRegex(OSError, 'interrupted'):
                        migration.migrate_training_run(root)
                self.assertEqual(json.loads((root/'training_v5_migration.json').read_text())['status'], 'pending')
                with patch.object(migration, 'process_alive', return_value=False):
                    result = migration.migrate_training_run(root)
                self.assertEqual(result['status'], 'complete')
                self.assertEqual(hashlib.sha256(Path(result['backups'][str(main)]).read_bytes()).hexdigest(), main_hash)
                upgraded = torch.load(main, map_location='cpu', weights_only=False)
                expected = F.avg_pool2d(saved['model']['flow_renderer.image_input.weight'], 2)*4
                torch.testing.assert_close(upgraded['model']['flow_renderer.image_input.weight'], expected, atol=0, rtol=0)

    def test_live_worker_and_controller_rejected_before_any_backup(self):
        from turkeyworld import training_v5_migration as migration
        for live in (12345, 54321):
            with self.subTest(live=live), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                self.fixture(root)
                with patch.object(migration, 'process_alive', side_effect=lambda pid: pid == live):
                    with self.assertRaisesRegex(RuntimeError, 'Stop training'):
                        migration.migrate_training_run(root)
                self.assertFalse(list(root.rglob('*_before_v5*')))
                self.assertFalse((root/'training_v5_migration.json').exists())

    def test_second_staging_failure_cleans_first_temporary_and_preserves_sources(self):
        from turkeyworld import training_v5_migration as migration
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, _, main, origin = self.fixture(root)
            hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (main, origin)}
            stage, calls = migration._stage, 0

            def fail_second(payload, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError('simulated second-stage write failure')
                return stage(payload, destination)

            with patch.object(migration, 'process_alive', return_value=False), \
                    patch.object(migration, '_stage', side_effect=fail_second):
                with self.assertRaisesRegex(OSError, 'second-stage'):
                    migration.migrate_training_run(root)
            self.assertEqual(hashes, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (main, origin)})
            self.assertFalse(list((root/'checkpoints').glob('*.tmp')))
            self.assertEqual(json.loads((root/'training_v5_migration.json').read_text())['status'], 'pending')


if __name__ == '__main__':
    unittest.main()
