"""Copied render assets require explicit, hash-bound JEPA-to-RSSM lineage."""
import copy
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_renderer_student import TinyTeacher, tiny_config
from turkeyworld.base_student import build_base_student, export_base_student, load_base_student
from turkeyworld.models import NeuralWorldModel, STATE_DIM, STATE_SCHEMA, load_world_checkpoint
from turkeyworld.renderer_student import (build_renderer_student, export_renderer_student,
    load_renderer_student, renderer_condition_identity)
from turkeyworld.transformer_model import CameraTransformerWorldModel


class RSSMAssetCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.teacher = TinyTeacher().eval()
        self.source_config = asdict(self.teacher.config)
        self.identity = renderer_condition_identity(self.teacher)
        self.renderer_path = self.directory/'renderer.pt'
        student = build_renderer_student(self.teacher, width=32, depth=1, heads=4, mlp_hidden=64)
        export_renderer_student(self.teacher, student, self.renderer_path, training_steps=7)
        self.base_path = self.directory/'base.pt'
        export_base_student(self.teacher, build_base_student(self.teacher), self.base_path,
                            training_steps=3)

    def migrated_view(self, *, approved=True):
        target = copy.deepcopy(self.teacher)
        target.config = replace(target.config, architecture='dreamer_rssm',
                                player_motion_mode='axis_fraction', rssm_deter_dim=32)
        if approved:
            target.rssm_migration = {
                'format_version': 1, 'source_architecture': 'vjepa2_ac',
                'architecture': 'dreamer_rssm', 'frozen_render_assets_bitexact': True,
                'source_teacher_config': self.source_config,
                'source_renderer_conditioning_identity': self.identity,
            }
        return target

    def test_scratch_rssm_rejects_old_base_and_flow_without_mutation(self):
        target = self.migrated_view(approved=False)
        flow, config = target.flow_renderer, target.config
        for loader, path in ((load_base_student, self.base_path),
                             (load_renderer_student, self.renderer_path)):
            with self.subTest(loader=loader.__name__), self.assertRaisesRegex(ValueError, 'lineage'):
                loader(target, path)
            self.assertIs(target.flow_renderer, flow)
            self.assertIs(target.config, config)
            self.assertFalse(hasattr(target, 'base_student'))

    def test_approved_renderer_reuse_preserves_live_rssm_config_and_nonflow_tensors(self):
        target = self.migrated_view()
        original = {name: value.clone() for name, value in target.state_dict().items()
                    if not name.startswith('flow_renderer.')}
        load_renderer_student(target, self.renderer_path)
        self.assertEqual(target.config.architecture, 'dreamer_rssm')
        self.assertEqual(target.config.player_motion_mode, 'axis_fraction')
        self.assertEqual(target.config.rssm_deter_dim, 32)
        self.assertEqual((target.flow_renderer.kind, target.flow_renderer.steps), ('student', 4))
        for name, value in original.items():
            torch.testing.assert_close(target.state_dict()[name], value, rtol=0, atol=0)
        # A Base bundle independently validates the same migration and hashes.
        base_target = self.migrated_view()
        metadata = load_base_student(base_target, self.base_path)
        self.assertEqual(metadata['training_steps'], 3)
        self.assertEqual(base_target.config.architecture, 'dreamer_rssm')

    def test_bad_lineage_or_changed_assets_cannot_be_approved_by_config_normalization(self):
        for change in ('false_copy', 'wrong_config', 'wrong_hash', 'changed_tensor'):
            with self.subTest(change=change):
                target = self.migrated_view()
                target.rssm_migration = copy.deepcopy(target.rssm_migration)
                if change == 'false_copy':
                    target.rssm_migration['frozen_render_assets_bitexact'] = False
                elif change == 'wrong_config':
                    target.rssm_migration['source_teacher_config']['vjepa_width'] = 64
                elif change == 'wrong_hash':
                    target.rssm_migration['source_renderer_conditioning_identity']['base_map_sha256'] = '0'*64
                else:
                    with torch.no_grad():
                        target.vjepa_backend.renderer.weight[0, 0] += 1
                flow = target.flow_renderer
                with self.assertRaises(ValueError):
                    load_renderer_student(target, self.renderer_path)
                self.assertIs(target.flow_renderer, flow)
                self.assertFalse(hasattr(target, 'base_student'))

    def test_rssm_camera_renderer_preserves_actual_appearance_features(self):
        config = replace(tiny_config(), architecture='dreamer_rssm')
        renderer = CameraTransformerWorldModel(config, render_only=True).eval()
        state = torch.zeros(1, STATE_DIM)
        state[:, :5] = torch.tensor([.25, .25, 0., 1., 1.])
        features = torch.linspace(-1, 1, 128)[None]
        maps = torch.randn(1, config.map_channels, 16, 16)
        with torch.inference_mode(), patch.object(renderer.render_condition, 'forward',
                wraps=renderer.render_condition.forward) as condition:
            result = renderer.render_with_aux(state, features, maps)
        torch.testing.assert_close(condition.call_args.args[0][:, -128:], features, rtol=0, atol=0)
        self.assertTrue(bool(torch.isfinite(result['rgb']).all()))

    def test_world_loader_retains_checkpoint_migration_for_asset_validation(self):
        config = tiny_config(architecture='dreamer_rssm', player_motion_mode='axis_fraction',
            rssm_deter_dim=32, rssm_hidden_dim=24, rssm_stoch=4, rssm_classes=4,
            rssm_embed_dim=12, rssm_action_dim=16, rssm_blocks=4)
        model = NeuralWorldModel(config)
        path = self.directory/'world.pt'
        report = {'format_version': 1, 'source_architecture': 'vjepa2_ac',
                  'frozen_render_assets_bitexact': True}
        torch.save({'stage': 'world', 'config': asdict(config), 'model': model.state_dict(),
                    'state_schema': STATE_SCHEMA, 'rssm_migration': report}, path)
        loaded, checkpoint = load_world_checkpoint(path)
        self.assertEqual(loaded.rssm_migration, report)
        self.assertIs(loaded.rssm_migration, checkpoint['rssm_migration'])
        self.assertEqual(loaded.core.memory_size, 48)


if __name__ == '__main__':
    unittest.main()
