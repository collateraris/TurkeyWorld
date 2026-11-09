"""Frequent image previews stay separate from full quality and optimizer clocks."""
from copy import deepcopy
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Optional PyTorch dependency is not installed')
class PreviewCadenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def experiment(self, folder, preview_every=10):
        from turkeyworld.vjepa_training import VJEPAExperiment
        experiment = object.__new__(VJEPAExperiment)
        experiment.root = Path(folder)
        experiment.args = SimpleNamespace(context=2, horizon=4, seed=73, preview_every=preview_every,
            quality_psnr=35., quality_ssim=.98, quality_mae=2/255, quality_fraction=.95,
            quality_passes=3)
        experiment.global_step, experiment.fragments_seen = 10, 80
        experiment.panel = [(index, 0) for index in range(64)]
        experiment.validation = object()
        experiment.state = {'stage': 'teacher500vq_world',
            'visual_quality': {'saved_full_check': 500}, 'teacher_quality': {'last_checked_step': 500},
            'teacher_plateau': {'bad_checks': 2, 'checks': 3}, 'teacher_quality_gate': {'streak': 0},
            'training_batch': {'microstep': 8, 'optimizer_step': 10},
            'validation_progress': {'status': 'complete', 'step': 500}}
        experiment.extra = {'teacher_plateau': {'checks': 3}, 'quality_streak': 0}
        experiment.metric = Mock(side_effect=AssertionError('Preview must not publish quality metrics'))
        return experiment

    def test_ten_update_preview_does_not_trigger_quality_until_500_and_legacy_default_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment = self.experiment(folder)
            experiment.prediction_preview_report = Mock(return_value={'preview': True})
            full_report = {'full_quality': True}
            experiment.validation_report = Mock(return_value=full_report)
            for step in (0, 9, 10, 499):
                experiment.global_step = step
                self.assertIsNone(experiment.preview_if_due(object()))
            experiment.prediction_preview_report.assert_called_once()
            experiment.validation_report.assert_not_called()
            experiment.global_step = 500
            self.assertIs(experiment.preview_if_due(object()), full_report)
            experiment.validation_report.assert_called_once()
            self.assertEqual(experiment.prediction_preview_report.call_count, 1)
            del experiment.args.preview_every
            experiment.global_step = 10
            self.assertIsNone(experiment.preview_if_due(object()))
            self.assertEqual(experiment.preview_interval(), 500)
            self.assertEqual(experiment.prediction_preview_report.call_count, 1)

    def test_small_preview_delegates_four_heldout_windows_without_mutating_quality_or_counters(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment = self.experiment(folder)
            protected = {key: deepcopy(experiment.state[key]) for key in experiment.state if key != 'stage'}
            extra = deepcopy(experiment.extra)
            model = torch.nn.Linear(1, 1).train()
            model.config = SimpleNamespace(kind='teacher')
            report = {'seconds': .1, 'summary': {'psnr_mean': 120., 'frames': 16}, 'passed': True}
            contact = {'metadata': {'kind': 'free_rollout', 'source_held_out': True,
                                    'future_source_feedback': False, 'context': 2, 'horizon': 4}}
            before_rng = torch.random.get_rng_state().clone()

            def save(preview, path, *, step, every):
                self.assertEqual(path, Path(folder)/'preview.png')
                return {**preview['metadata'], 'step': step, 'every': every}

            with patch('turkeyworld.vjepa_validation.evaluate_autoregressive', return_value=(report, contact)) as evaluate, \
                 patch('turkeyworld.vjepa_validation.save_autoregressive_preview', side_effect=save):
                self.assertIs(experiment.prediction_preview_report(model), report)
            args, options = evaluate.call_args
            self.assertIs(args[0], model)
            self.assertIs(args[1], experiment.validation)
            self.assertEqual(args[2], experiment.panel[:4])
            self.assertEqual(options['context'], 2)
            self.assertEqual(options['horizon'], 4)
            self.assertEqual(options['preview_every'], 10)
            self.assertEqual(experiment.state['prediction_progress']['frames'], 16)
            self.assertFalse(experiment.state['prediction_preview']['quality_gate_evaluated'])
            self.assertEqual(experiment.state['prediction_preview']['scope'], 'training_preview')
            self.assertEqual(experiment.state['prediction_preview']['every'], 10)
            for key, value in protected.items():
                self.assertEqual(experiment.state[key], value, key)
            self.assertEqual(experiment.extra, extra)
            self.assertEqual((experiment.global_step, experiment.fragments_seen), (10, 80))
            self.assertTrue(model.training)
            torch.testing.assert_close(torch.random.get_rng_state(), before_rng, rtol=0, atol=0)
            self.assertFalse((Path(folder)/'teacher_vjepa_quality_report.json').exists())
            experiment.metric.assert_not_called()

    def test_real_shared_evaluator_restores_rng_and_modes_and_never_feeds_future_gt(self):
        from turkeyworld.jepa_training_utils import SequenceData

        class World(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.child = torch.nn.Dropout(.5)
                self.config = SimpleNamespace(kind='teacher', architecture='vjepa2_ac', image_size=16)
                self.inputs = []

            def encode_map(self, grid):
                return grid[:, None].float()

            def initialize_from_image(self, images, states, features):
                self.asserted_eval = not self.training and not self.child.training
                torch.rand(1)
                return states[:, -1:, :1].clone()

            def transition(self, state, action, features, memory):
                self.inputs.append(state.detach().clone())
                following = state.clone()
                following[:, 0] += action[:, 0]
                return {'state': following, 'memory': memory+1, 'features': state.new_zeros(len(state), 128)}

            def render(self, state, features, map_features):
                return (state[:, 0]/255)[:, None, None, None].expand(-1, 3, 16, 16)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data_dir = root/'val'
            data_dir.mkdir()
            values = np.arange(9, dtype=np.float32)
            frames = np.broadcast_to(values.astype(np.uint8)[:, None, None, None], (9, 16, 16, 3)).copy()
            states = np.zeros((9, 90), np.float32)
            states[:, 0], states[:, 3:5] = values, 1
            np.savez(data_dir/'example.npz', frames=frames, states=states,
                actions=np.ones((8, 4), np.float32), events=np.zeros((8, 4), np.float32),
                hit_ids=np.full(8, -1, np.int64), attack_flags=np.zeros((8, 8), np.float32),
                map=np.zeros((16, 16), np.int8), split=np.asarray('val'), dt=np.asarray(1/30))
            experiment = self.experiment(root)
            experiment.panel, experiment.validation = [(0, 0)], SequenceData(data_dir, split='val')
            model = World().train()
            model.child.eval()
            generated = []

            def save(preview, path, *, step, every):
                generated.append(preview['generated'].copy())
                return {**preview['metadata'], 'step': step, 'every': every}

            torch_state, python_state = torch.random.get_rng_state().clone(), random.getstate()
            with patch('turkeyworld.vjepa_validation.save_autoregressive_preview', side_effect=save):
                experiment.prediction_preview_report(model)
                initial_inputs = [value.clone() for value in model.inputs]
                experiment.validation.episodes[0]['frames'][2:] = 200
                experiment.validation.episodes[0]['states'][2:, 0] = 1000
                model.inputs.clear()
                experiment.prediction_preview_report(model)
            self.assertTrue(model.asserted_eval)
            self.assertTrue(model.training)
            self.assertFalse(model.child.training)
            torch.testing.assert_close(torch.random.get_rng_state(), torch_state, rtol=0, atol=0)
            self.assertEqual(random.getstate(), python_state)
            np.testing.assert_array_equal(generated[0], generated[1])
            for before, after in zip(initial_inputs, model.inputs):
                torch.testing.assert_close(before, after, rtol=0, atol=0)
            self.assertEqual(experiment.state['visual_quality'], {'saved_full_check': 500})
            self.assertEqual(experiment.state['teacher_plateau']['checks'], 3)


if __name__ == '__main__':
    unittest.main()
