"""Independent residual export/runtime and frozen-codec audit."""
from argparse import Namespace
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.base_student import (build_base_student, enable_base_pixel_residual,
    export_base_student, load_base_student, prepare_base_student)
from turkeyworld.models import STATE_SCHEMA, apply_pixel_residual
from turkeyworld.neural_play import NeuralRuntime
from turkeyworld.renderer_enemy_objectives import enemy_conditions
from turkeyworld.renderer_refinement import configure_refinement, refinement_forward
from turkeyworld.renderer_student import build_renderer_student, export_renderer_student
from test_renderer_enemy_objectives import batch
from test_renderer_refinement import source_model


class PixelResidualAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_zero_upgrade_preserves_old_weights_base_flow_input_and_decoded_output_exactly(self):
        original = source_model()
        flow = build_renderer_student(original, steps=1, width=16, depth=1, heads=2, mlp_hidden=32)
        original.flow_renderer = flow
        original.base_student = build_base_student(original).eval()
        configure_refinement(original)
        source = batch((False, True))
        conditions = enemy_conditions(original, source)
        old_base = original.base_student
        before = {key: value.clone() for key, value in old_base.state_dict().items()}
        rng = torch.get_rng_state().clone()
        upgraded = enable_base_pixel_residual(old_base)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(sum(value.numel() for value in old_base.parameters()), 996844)
        self.assertEqual(sum(value.numel() for value in upgraded.parameters()), 998584)
        for key, value in before.items():
            self.assertTrue(torch.equal(old_base.state_dict()[key], value))
            self.assertTrue(torch.equal(upgraded.state_dict()[key], value))
        with torch.no_grad():
            left = refinement_forward(original, conditions)
            original.base_student = upgraded
            right = refinement_forward(original, conditions, return_residual=True)
        self.assertTrue(torch.equal(left[0], right[0]))
        self.assertTrue(torch.equal(left[1], right[1]))
        self.assertTrue(torch.equal(left[2], right[2]))
        self.assertEqual(float(right[3].abs().max()), 0)

    def test_nonzero_residual_roundtrip_runs_real_neural_runtime_with_no_future_source_or_codec_updates(self):
        original = source_model()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            world, base_path, bundle = root/'world.pt', root/'base.pt', root/'renderer.pt'
            torch.save({'stage': 'world', 'state_schema': STATE_SCHEMA,
                'config': asdict(original.config), 'model': original.state_dict(), 'training_steps': 339}, world)
            head = enable_base_pixel_residual(build_base_student(original))
            with torch.no_grad():
                head.pixel_residual.bias.fill_(.1)
            export_base_student(original, head, base_path, training_steps=1, metrics={'passed': False})
            load_base_student(original, base_path)
            student = build_renderer_student(original, steps=1, width=16, depth=1, heads=2, mlp_hidden=32)
            export_renderer_student(original, student, bundle, training_steps=1)
            options = Namespace(world=world, renderer_checkpoint=bundle, base_checkpoint=None,
                allow_unqualified_codec=False, device='cpu', threads=2, sr=None, fp16=False,
                seed=0, display_size=32, smooth=True, headless=True, fps=60, warmup=0)
            runtime = NeuralRuntime(options)
            self.assertTrue(runtime.model.base_student.config.pixel_residual)
            self.assertTrue(torch.equal(runtime.model.base_student.pixel_residual.bias, head.pixel_residual.bias))
            codec_before = {key: value.clone() for key, value in runtime.model.flow_renderer.codec.state_dict().items()}
            with patch('turkeyworld.engine.Game.step', side_effect=AssertionError('No Source future')), \
                    patch('turkeyworld.engine.Game.render', side_effect=AssertionError('No Source future')), \
                    patch('turkeyworld.engine.Game.set_state_vector', side_effect=AssertionError('No corrections')), \
                    patch.object(runtime.model.flow_renderer, 'velocity', wraps=runtime.model.flow_renderer.velocity) as velocity:
                frame = runtime.step([.5, .1, .2, 0])
            self.assertEqual(velocity.call_count, 1)
            self.assertEqual(frame.shape, (32, 32, 3))
            self.assertEqual(frame.dtype, np.uint8)
            self.assertEqual(runtime.engine_renders_for_initialization, 1)
            self.assertTrue(all(torch.equal(value, runtime.model.flow_renderer.codec.state_dict()[key])
                                for key, value in codec_before.items()))
            self.assertTrue(all(value.grad is None for value in runtime.model.parameters()))
            # The only correction is added AFTER decoding, not to Base or Flow inputs.
            with torch.no_grad():
                state = runtime.state
                features = state.new_zeros(1, 128)
                model = runtime.model
                base, residual = model.render_base_with_residual(state, features, runtime.map_features)
                raw = model.flow_renderer.sample(base, state)
                rendered = model.render(state, features, runtime.map_features)
                self.assertTrue(torch.equal(rendered, apply_pixel_residual(raw, residual)))
                self.assertGreater(float(residual.abs().mean()), 0)

    def test_original_base_payload_without_new_flag_keeps_default_off_and_new_provenance_is_strict(self):
        original = source_model()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'old.pt'
            export_base_student(original, build_base_student(original), path, training_steps=1)
            old = torch.load(path, weights_only=False)
            old['config'].pop('pixel_residual')
            head, _, _ = prepare_base_student(original, old)
            self.assertFalse(head.config.pixel_residual)
            self.assertFalse(hasattr(head, 'pixel_residual'))
            upgraded = enable_base_pixel_residual(head)
            export_base_student(original, upgraded, root/'new.pt', training_steps=1)
            payload = torch.load(root/'new.pt', weights_only=False)
            corrupted = deepcopy(payload)
            corrupted['pixel_residual']['position'] = 'before_codec'
            with self.assertRaisesRegex(ValueError, 'provenance/configuration'):
                prepare_base_student(original, corrupted)


if __name__ == '__main__':
    unittest.main()
