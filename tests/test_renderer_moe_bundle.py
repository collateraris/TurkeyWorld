"""Strict bundled Base/RF MoE export, conditioning and transactional loading."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_renderer_enemy_objectives import batch
from test_renderer_geometry_opacity import working_model
from turkeyworld.base_student import load_base_student
from turkeyworld.renderer_geometry_opacity import enable_geometry_opacity
from turkeyworld.renderer_moe import enable_renderer_moe, renderer_moe_metadata
from turkeyworld.renderer_refinement import export_refined_bundle
from turkeyworld.renderer_student import _hash_tensors, load_renderer_student


def unloaded_base(model):
    for name in ('base_student', 'base_student_metadata', 'base_student_bundle'):
        if hasattr(model, name):
            delattr(model, name)
    return model


class RendererMoEBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make_bundle(self, directory):
        source = working_model()
        target = unloaded_base(deepcopy(source))
        working = deepcopy(source)
        enable_geometry_opacity(working)
        enable_renderer_moe(working)
        with torch.no_grad():
            working.base_student.output_moe.output.bias.add_(.025)
            working.flow_renderer.output_moe.output.bias.add_(.015)
            working.flow_renderer.geometry_opacity[-1].bias.add_(.2)
        export_refined_bundle(source, working, directory, updates=3,
            report={'passed': False}, initial_renderer_steps=10,
            initial_base_steps=10, provenance={})
        return working, target, Path(directory)/'renderer_student.pt'

    def test_strict_export_roundtrip_preserves_both_private_expert_heads_and_private_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            working, target, path = self.make_bundle(directory)
            saved = torch.load(path, map_location='cpu', weights_only=False)
            header = saved['renderer_output_moe']
            self.assertEqual(header, renderer_moe_metadata(working.flow_renderer, working.base_student))
            self.assertEqual(header['base'], saved['base_student_bundle']['output_moe'])
            self.assertTrue(saved['config']['flow_output_moe_enabled'])
            self.assertTrue(saved['base_student_bundle']['config']['output_moe_enabled'])
            self.assertEqual(header['base']['expert_width'], 8)
            self.assertEqual(header['flow']['expert_width'], 32)
            self.assertEqual(header['base']['roles'], ['scene_walls', 'enemy', 'combat'])
            self.assertEqual(saved['geometry_opacity_override'],
                             saved['base_student_bundle']['geometry_opacity_binding'])
            self.assertFalse(saved['gameplay_qualified'])
            query = batch((False, False))
            rng = torch.get_rng_state().clone()
            with patch('turkeyworld.engine.Game.render', side_effect=AssertionError('No Source at inference')):
                metadata = load_renderer_student(target, path)
                self.assertEqual(_hash_tensors(working.base_student.state_dict()),
                                 _hash_tensors(target.base_student.state_dict()))
                self.assertEqual(_hash_tensors(working.flow_renderer.state_dict()),
                                 _hash_tensors(target.flow_renderer.state_dict()))
                # Strict loader freezes an inference Base. Match that execution
                # policy: CPU attention selects a different fused fast path for
                # trainable versus frozen parameter sets, even in inference_mode.
                working.requires_grad_(False)
                target.requires_grad_(False)
                features = torch.zeros(2, 128)
                with torch.inference_mode():
                    fmap = working.encode_map(query['map'])
                    actual_map = target.encode_map(query['map'])
                    self.assertTrue(torch.equal(working.render_base(query['state'], features, fmap),
                                                target.render_base(query['state'], features, actual_map)))
                    self.assertTrue(torch.equal(working.render(query['state'], features, fmap),
                                                target.render(query['state'], features, actual_map)))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(metadata['renderer_output_moe'], header)

    def test_incomplete_or_tampered_head_bindings_fail_before_mutating_live_world(self):
        with tempfile.TemporaryDirectory() as directory:
            _, original, path = self.make_bundle(directory)
            saved = torch.load(path, map_location='cpu', weights_only=False)
            modes = ('missing_header', 'wrong_flow_hash', 'wrong_Base_header', 'wrong_Base_flag',
                     'missing_flow_expert', 'missing_Base_expert', 'changed_expert', 'wrong_flow_flag')
            for mode in modes:
                broken = deepcopy(saved)
                if mode == 'missing_header':
                    broken.pop('renderer_output_moe')
                elif mode == 'wrong_flow_hash':
                    broken['renderer_output_moe']['flow']['state_sha256'] = 'f'*64
                elif mode == 'wrong_Base_header':
                    broken['base_student_bundle']['output_moe']['expert_depth'] = 2
                elif mode == 'wrong_Base_flag':
                    broken['base_student_bundle']['config']['output_moe_enabled'] = False
                elif mode == 'missing_flow_expert':
                    broken['model'].pop('output_moe.experts.1.linear1.weight')
                elif mode == 'missing_Base_expert':
                    broken['base_student_bundle']['model'].pop('output_moe.experts.2.linear1.weight')
                elif mode == 'changed_expert':
                    broken['model']['output_moe.experts.0.linear1.weight'].add_(.02)
                else:
                    broken['config']['flow_output_moe_enabled'] = False
                output = Path(directory)/(mode+'.pt')
                torch.save(broken, output)
                target = deepcopy(original)
                old_hash = _hash_tensors(target.state_dict())
                old_config = asdict(target.config)
                old_flow = target.flow_renderer
                rng = torch.get_rng_state().clone()
                with self.subTest(mode=mode), self.assertRaises((ValueError, RuntimeError)):
                    load_renderer_student(target, output)
                self.assertIs(target.flow_renderer, old_flow)
                self.assertFalse(hasattr(target, 'base_student'))
                self.assertEqual(_hash_tensors(target.state_dict()), old_hash)
                self.assertEqual(asdict(target.config), old_config)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_standalone_MoE_Base_requires_its_matching_loaded_RF_bundle(self):
        source = working_model()
        target = unloaded_base(deepcopy(source))
        working = deepcopy(source)
        enable_renderer_moe(working)
        with tempfile.TemporaryDirectory() as directory:
            export_refined_bundle(source, working, directory, updates=1,
                report={'passed': False}, initial_renderer_steps=10,
                initial_base_steps=10, provenance={})
            directory = Path(directory)
            old_hash = _hash_tensors(target.state_dict())
            old_flow = target.flow_renderer
            with self.assertRaisesRegex(ValueError, 'matching.*MoE|MoE.*matching'):
                load_base_student(target, directory/'base_student.pt')
            self.assertIs(target.flow_renderer, old_flow)
            self.assertEqual(old_hash, _hash_tensors(target.state_dict()))
            self.assertFalse(hasattr(target, 'base_student'))
            load_renderer_student(target, directory/'renderer_student.pt')
            expected = _hash_tensors(target.base_student.state_dict())
            load_base_student(target, directory/'base_student.pt')
            self.assertEqual(expected, _hash_tensors(target.base_student.state_dict()))


if __name__ == '__main__':
    unittest.main()
