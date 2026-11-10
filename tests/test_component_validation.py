"""Direct renderer validation must never invoke visual/world dynamics."""
import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from turkeyworld.component_validation import evaluate_renderer_component, renderer_component_preview
from turkeyworld.engine import Game, STATE_DIM


class QueryData:
    def __init__(self, split='val'):
        self.directory, self.split = Path('/offline/navigation')/split, split
        self.episodes = [{'path': str(self.directory/f'episode_{index}.npz'),
                          'frames': np.zeros((8, 16, 16, 3), np.uint8)} for index in range(8)]
        self.rng = np.random.default_rng(91)
        self.calls = []

    def batch(self, count, *, context, horizon, indices):
        assert horizon == 1
        self.rng.random()
        self.calls.extend(indices)
        target = torch.zeros(count, 1, STATE_DIM)
        target[:, 0, 0] = torch.tensor([.1+.01*index+.02*(start+context) for index, start in indices])
        frames = target[:, :, :1, None, None].expand(-1, -1, 3, 16, 16).clone()
        return {'target_states': target, 'target_frames': frames,
                'target_object_masks': torch.zeros(count, 1, 1, 16, 16),
                'target_background_frames': frames.clone(), 'map': torch.zeros(count, 16, 16, dtype=torch.long)}


class QueryFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('_map_condition', torch.ones(1, 16, 16, dtype=torch.long), persistent=False)
        self.fail = False

    def sample(self, base, state, *, map_condition):
        assert not torch.is_grad_enabled()
        assert base.dtype == state.dtype == torch.float32
        torch.rand(()); np.random.rand(); random.random()
        if self.fail:
            raise RuntimeError('query failure')
        return base


class QueryWorld(nn.Module):
    def __init__(self):
        super().__init__()
        self.error = nn.Parameter(torch.zeros(()))
        self.flow_renderer = QueryFlow()
        self.core = nn.Linear(1, 1)
        self.register_buffer('_spawn_walkable', torch.ones(1, 16, 16, dtype=torch.bool), persistent=False)
        self.received_states = []

    def encode_map(self, grid):
        assert not torch.is_grad_enabled()
        self._spawn_walkable = grid == 0
        self.flow_renderer._map_condition = grid.clone()
        return torch.zeros(len(grid), 1, 16, 16)

    def render_base(self, state, features, maps):
        assert not torch.is_grad_enabled()
        assert not self.training
        assert features.shape == (len(state), 128) and not bool(features.any())
        self.received_states.append(state.clone())
        return (state[:, :1]+self.error)[:, :, None, None].expand(-1, 3, 16, 16)

    def transition(self, *args, **kwargs):
        raise AssertionError('No world transition is permitted')

    def initialize_from_image(self, *args, **kwargs):
        raise AssertionError('No visual encoder is permitted')


class ComponentValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_gt_alignment_passes_without_core_engine_or_transition_calls(self):
        data, model = QueryData(), QueryWorld()
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(Game, 'step', side_effect=AssertionError('No oracle')), \
                patch.object(Game, 'render', side_effect=AssertionError('No oracle')), \
                patch.object(model.core, 'forward', side_effect=AssertionError('No visual core')):
            report = evaluate_renderer_component(model, data, count=6, batch_size=2, output_dir=folder, step=777)
            self.assertTrue(report['passed'], report['failures'])
            self.assertAlmostEqual(report['metrics']['rgb_ssim'], 1.)
            self.assertEqual(report['metrics']['rgb_mae'], 0.)
            self.assertEqual(report['metrics']['rgb_frames'], 6)
            self.assertEqual(report['metrics']['base_rgb_mae'], 0.)
            self.assertFalse(report['world_rollout'])
            self.assertEqual(report['transitions_evaluated'], 0)
            self.assertEqual(report['coordinate_source'], 'source_gt')
            self.assertEqual(json.loads(Path(report['report_path']).read_text())['step'], 777)
        actual = torch.cat(model.received_states)[:, 0]
        expected = torch.tensor([.1+.01*row['episode_index']+.02*row['source_frame'] for row in report['per_frame']])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_wrong_renderer_fails_image_gate_without_any_dynamics_judgment(self):
        model = QueryWorld()
        model.error.data.fill_(.25)
        report = evaluate_renderer_component(model, QueryData(), count=4)
        self.assertFalse(report['passed'])
        self.assertAlmostEqual(report['metrics']['rgb_mae'], .25, places=6)
        self.assertNotIn('player_xy_mean', report['metrics'])
        self.assertNotIn('attack_recall', report['metrics'])
        self.assertEqual(report['component_phase'], 'renderer')

    def test_nonfinite_sample_fails_and_still_writes_strict_json(self):
        model = QueryWorld()
        with tempfile.TemporaryDirectory() as folder, patch.object(model.flow_renderer, 'sample',
                side_effect=lambda base, state, **options: base*float('nan')):
            report = evaluate_renderer_component(model, QueryData(), count=4, output_dir=folder)
            self.assertFalse(report['passed'])
            self.assertFalse(report['metrics']['rgb_finite'])
            self.assertTrue(report['metrics']['base_rgb_finite'])
            self.assertEqual(report['metrics']['rgb_ssim'], -1.)
            self.assertEqual(report['metrics']['rgb_mae'], 1.)
            self.assertFalse(json.loads(Path(report['report_path']).read_text())['passed'])

    def test_rng_modes_map_caches_and_weights_restore_on_success_or_failure(self):
        for failure in (False, True):
            with self.subTest(failure=failure):
                data, model = QueryData(), QueryWorld().train()
                model.core.eval(); model.flow_renderer.fail = failure
                before = {name: value.clone() for name, value in model.state_dict().items()}
                caches = (model._spawn_walkable.clone(), model.flow_renderer._map_condition.clone())
                modes = [module.training for module in model.modules()]
                torch_rng, numpy_rng, python_rng = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
                data_rng = copy.deepcopy(data.rng.bit_generator.state)
                if failure:
                    with self.assertRaisesRegex(RuntimeError, 'query failure'):
                        evaluate_renderer_component(model, data, count=4)
                else:
                    evaluate_renderer_component(model, data, count=4)
                torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)
                np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
                self.assertEqual(np.random.get_state()[2:], numpy_rng[2:])
                self.assertEqual(random.getstate(), python_rng)
                self.assertEqual(data.rng.bit_generator.state, data_rng)
                self.assertEqual([module.training for module in model.modules()], modes)
                torch.testing.assert_close(model._spawn_walkable, caches[0])
                torch.testing.assert_close(model.flow_renderer._map_condition, caches[1])
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_preview_has_four_gt_pose_columns_and_truthful_renderer_labels(self):
        model, data = QueryWorld(), QueryData()
        preview = renderer_component_preview(model, data, 800)
        self.assertEqual(preview['source'].shape, (4, 16, 16, 3))
        self.assertEqual(preview['generated_float'].shape, (4, 3, 16, 16))
        np.testing.assert_array_equal(preview['source'], preview['generated'])
        np.testing.assert_array_equal(preview['context'], preview['generated'])
        metadata = preview['metadata']
        self.assertEqual(metadata['kind'], 'static_scene_query')
        self.assertEqual(metadata['coordinate_source'], 'source_gt')
        self.assertTrue(metadata['renderer_only'])
        self.assertTrue(metadata['source_held_out'])
        self.assertFalse(metadata['world_rollout'])
        self.assertFalse(metadata['source_rgb_is_model_input'])
        self.assertIn('GT navigation', metadata['row_labels']['context'])
        self.assertEqual(metadata['horizon'], 0)

    def test_training_split_and_non_fp32_models_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'held-out'):
            evaluate_renderer_component(QueryWorld(), QueryData('train'))
        with self.assertRaisesRegex(ValueError, 'FP32'):
            evaluate_renderer_component(QueryWorld().half(), QueryData())


if __name__ == '__main__':
    unittest.main()
