from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from turkeyworld.engine import Game
from turkeyworld.motion_recovery_validation import (_short, evaluate_motion_recovery,
                                                   evaluate_prior_navigation_tape)
from turkeyworld.navigation_recovery_data import collect
from turkeyworld.staged_curriculum import StagedSequenceData


class PublicMemoryModel(nn.Module):
    """Backend-agnostic deterministic own-memory stub, independent of labels."""
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(simulation_hz=30., image_size=32)
        self.initial_images, self.initial_states, self.inputs = [], [], []
        self.register_buffer('_map_condition', torch.tensor([17.]), persistent=False)

    def encode_map(self, grid):
        self._map_condition = grid.clone()
        return grid[:, None].float()

    def initialize_from_image(self, images, states, fmap):
        self.initial_images.append(images.detach().clone())
        self.initial_states.append(states.detach().clone())
        return states.new_zeros((len(states), 3))

    def transition(self, state, action, fmap, memory):
        self.inputs.append((state.detach().clone(), memory.detach().clone()))
        controls = action[:, :2]/torch.linalg.vector_norm(action[:, :2], dim=-1, keepdim=True).clamp_min(1)
        angle = torch.atan2(state[:, 2], state[:, 3])+action[:, 2]*2.2/30
        command = torch.stack([angle.cos()*controls[:, 0]-angle.sin()*controls[:, 1],
                               angle.sin()*controls[:, 0]+angle.cos()*controls[:, 1]], -1)*2.4/30
        result = state.clone()
        result[:, :2] += command/16
        result[:, 2], result[:, 3] = angle.sin(), angle.cos()
        result[:, 16] = (result[:, 16]+1/30/.6).clamp_max(5)
        result[:, -1] += 1/30/300
        return {'state': result, 'memory': memory+1, 'player_pass_logits': state.new_full((len(state), 2), 10.)}


class CorruptFuture:
    def __init__(self, data):
        self.data = data
        self.episodes, self.split, self.directory, self.rng = data.episodes, data.split, data.directory, data.rng

    def batch(self, *args, **kwargs):
        batch = self.data.batch(*args, **kwargs)
        batch['target_states'] = batch['target_states'].clone()
        batch['target_states'][..., :2] += .25
        batch['target_frames'] = torch.ones_like(batch['target_frames'])*.998
        return batch


class MotionRecoveryValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_future_labels_change_metrics_but_never_model_inputs_or_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collect(root, short_train=0, short_val=16, long_train=0, long_val=0)
            data = StagedSequenceData(root/'short/rgb/val', 'cpu', mode='stream')
            model = PublicMemoryModel().train()
            rng, cache = torch.random.get_rng_state().clone(), model._map_condition.clone()
            with (patch.object(Game, 'step', side_effect=AssertionError('No Source future')),
                  patch.object(Game, 'render', side_effect=AssertionError('No Source renderer'))):
                clean, expected = _short(model, data, context=1, count=16, batch_size=4, seed=173)
                image_inputs = [value.clone() for value in model.initial_images]
                state_inputs = [value.clone() for value in model.initial_states]
                model.initial_images.clear(); model.initial_states.clear()
                changed, actual = _short(model, CorruptFuture(data), context=1, count=16, batch_size=4, seed=173)
            self.assertTrue(np.array_equal(expected['predicted'], actual['predicted']))
            self.assertTrue(all(torch.equal(left, right) for left, right in zip(image_inputs, model.initial_images)))
            self.assertTrue(all(torch.equal(left, right) for left, right in zip(state_inputs, model.initial_states)))
            self.assertGreater(changed['metrics']['velocity_error_tiles_s']['mean'], clean['metrics']['velocity_error_tiles_s']['mean'])
            self.assertTrue(model.training)
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            self.assertTrue(torch.equal(cache, model._map_condition))
            self.assertFalse(clean['source_future_feedback'])
            self.assertTrue(clean['axis_fraction_head_present'])

    def test_prior_tape450_uses_only_initial_render_and_own_memory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            game = Game(seed=0, endless=True, enemies_enabled=False)
            states = [game.state_vector().copy()]
            actions = np.zeros((450, 4), np.float32)
            for action in actions:
                game.step(action)
                states.append(game.state_vector().copy())
            tape = root/'tape.npz'
            np.savez(tape, reference=np.stack(states), actions=actions, map=game.map)
            model = PublicMemoryModel().train()
            with (patch.object(Game, 'step', side_effect=AssertionError('No Source future steps')),
                  patch.object(Game, 'render', wraps=None) as render):
                # Preserve the actual initial observation while counting calls.
                render.side_effect = lambda *a, **k: np.zeros((32, 32, 3), np.uint8)
                report = evaluate_prior_navigation_tape(model, tape, root/'report')
            self.assertEqual(render.call_count, 1)
            self.assertEqual(len(model.inputs), 450)
            self.assertTrue(all(float(memory[0, 0]) == index for index, (_, memory) in enumerate(model.inputs)))
            self.assertEqual(report['horizon'], 450)
            self.assertLess(report['final_player_xy_error_tiles']['mean'], 1e-5)
            self.assertEqual(report['model_wall_steps'], 0)
            self.assertTrue(model.training)

    def test_combined_report_requires_expanded_long450_and_labels_default_not_heldout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collect(root/'data', short_train=16, short_val=16, long_train=0, long_val=1)
            model = PublicMemoryModel().train()
            with (patch.object(Game, 'step', side_effect=AssertionError('No reference rollout in validation')),
                  patch.object(Game, 'render', side_effect=AssertionError('Recorded RGB only'))):
                result = evaluate_motion_recovery(model, root/'data/short/rgb/val', root/'data/long/rgb/val',
                    root/'report', count=16, long_count=1, batch_size=4,
                    default_regression_data=root/'data/short/rgb/train')
            self.assertEqual(result['long450']['horizon'], 450)
            self.assertEqual(result['long450']['windows'], 1)
            self.assertEqual(result['short']['context'], 1)
            self.assertFalse(result['default_spawn']['source_held_out'])
            self.assertTrue(result['default_spawn']['default_spawn_regression'])
            self.assertFalse(result['passed'])
            self.assertTrue(model.training)
            self.assertTrue((root/'report/motion_recovery_validation.json').is_file())
            self.assertTrue((root/'report/long450/heldout_trajectory_states.npz').is_file())


if __name__ == '__main__':
    unittest.main()
