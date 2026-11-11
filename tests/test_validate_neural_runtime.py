from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from turkeyworld.validate_neural_runtime import (evaluate_runtime_case, initialize_runtime,
                                                load_heldout_cases, _measure)


class TinyFlow(nn.Module):
    kind, steps = 'student', 1

    def __init__(self):
        super().__init__()
        self.codec = nn.Linear(1, 1).requires_grad_(False)

    def velocity(self, value):
        return value*0


class TinyWorld(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.flow_renderer = TinyFlow()
        self.config = SimpleNamespace(architecture='dreamer_rssm', transition_mode='event_v1',
            simulation_hz=30., image_size=32, endless=True, navigation_only=True,
            kind='teacher', renderer_mode='direct')
        self.images, self.own_memories = [], []

    def encode_map(self, grid):
        return grid[:, None].float()

    def initialize_from_image(self, image, state, fmap):
        self.images.append(image.clone())
        return state.new_zeros(len(state), 8)

    def step(self, state, action, memory, map_features):
        self.own_memories.append(memory.clone())
        predicted = state.clone()
        predicted[:, 0] += action[:, 0]*.001
        predicted[:, -1] += 1/9000
        value = self.flow_renderer.velocity(state[:, :1])
        rgb = torch.ones(len(state), 3, 32, 32, device=state.device)*(.3+value[:, :, None, None])
        return {'state': predicted, 'memory': memory+1, 'rgb': rgb,
                'events': state.new_zeros(len(state), 4)}


def case(steps=5):
    state = np.zeros(20, np.float32)
    state[:2] = 3.5/16
    state[3], state[4] = 1, 1
    action = np.zeros((steps, 4), np.float32)
    action[:, 0] = 1
    states = np.repeat(state[None], steps+1, axis=0)
    for index in range(steps):
        states[index+1] = states[index]
        states[index+1, 0] += .001
        states[index+1, -1] += 1/9000
    grid = np.zeros((16, 16), np.int8)
    grid[[0, -1], :] = 1
    grid[:, [0, -1]] = 1
    frames = np.full((steps+1, 32, 32, 3), 76, np.uint8)
    return {'name': 'test', 'states': states, 'actions': action, 'frames': frames,
            'initial_rgb': frames[0], 'map': grid, 'dt': 1/30,
            'source_held_out': True, 'selection': 'test held-out tape'}


def args():
    return Namespace(world=Path('immutable.pt'), device='cpu', threads=1, sr=None, fp16=False,
        seed=0, display_size=32, smooth=True, headless=True, fps=60, warmup=0,
        renderer_checkpoint=None, base_checkpoint=None)


class ValidateNeuralRuntimeTests(unittest.TestCase):
    def initialize(self, model, source, shared=None):
        with patch('turkeyworld.neural_play.load_world_checkpoint', return_value=(model, {'training_steps': 3000})):
            return initialize_runtime(args(), source, shared)

    def test_actual_runtime450_keeps_own_memory_one_initial_observation_and_one_euler(self):
        model = TinyWorld()
        source = case(450)
        runtime = self.initialize(model, source)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        with TemporaryDirectory() as directory, \
                patch('turkeyworld.neural_play.parameter_counts', return_value={}), \
                patch('turkeyworld.engine.Game.step', side_effect=AssertionError('No engine rollout')) as step, \
                patch('turkeyworld.engine.Game.render', side_effect=AssertionError('No engine rendering')) as render:
            report = evaluate_runtime_case(runtime, source, Path(directory)/'case')
            with np.load(Path(directory)/'case/trajectory.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['predicted_states'], source['states'])
                self.assertEqual(saved['neural_native_rgb'].shape, (6, 32, 32, 3))
            step.assert_not_called()
            render.assert_not_called()
        self.assertEqual(report['simulated_steps'], 450)
        self.assertEqual(report['actual_renderer_velocity_evaluations'], 450)
        self.assertEqual(report['initial_observed_rgb_frames'], 1)
        self.assertEqual(report['numerical_resets_during_rollout'], 0)
        self.assertEqual(report['metrics']['final_player_xy_error_tiles'], 0)
        self.assertEqual(report['metrics']['model_wall_steps'], 0)
        self.assertTrue(report['codec_all_state_dict_tensors_bitexact'])
        self.assertEqual(len(model.images), 1)
        for i, memory in enumerate(model.own_memories):
            self.assertTrue(torch.equal(memory, torch.full((1, 8), float(i))))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)

    def test_future_rgb_and_numeric_targets_never_change_runtime_predictions_or_memory(self):
        first = case()
        second = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in first.items()}
        second['states'][1:, 0] += .05
        second['frames'][1:] = 250
        model = TinyWorld()
        runtime = self.initialize(model, first)
        with TemporaryDirectory() as directory, patch('turkeyworld.neural_play.parameter_counts', return_value={}):
            report1 = evaluate_runtime_case(runtime, first, Path(directory)/'first')
            next_runtime = self.initialize(model, second, shared=runtime)
            report2 = evaluate_runtime_case(next_runtime, second, Path(directory)/'second')
            with np.load(Path(directory)/'first/trajectory.npz') as a, np.load(Path(directory)/'second/trajectory.npz') as b:
                np.testing.assert_array_equal(a['predicted_states'], b['predicted_states'])
                np.testing.assert_array_equal(a['neural_native_rgb'], b['neural_native_rgb'])
            self.assertNotEqual(report1['metrics']['final_player_xy_error_tiles'],
                                report2['metrics']['final_player_xy_error_tiles'])
        self.assertTrue(torch.equal(model.images[0], model.images[1]))
        self.assertTrue(torch.equal(runtime.memory, next_runtime.memory))

    def test_geometry_detects_swept_wall_without_endpoint_overlap(self):
        source = case(1)
        source['map'][3, 4] = 1
        predicted = source['states'].copy()
        predicted[1, 0] = 5.5/16
        result = _measure(source, predicted)
        self.assertEqual(result['model_wall_steps'], 0)
        self.assertEqual(result['model_wall_path_steps'], 1)
        self.assertEqual(result['first_model_wall_step'], 1)

    def test_heldout_selection_requires_val_and_genuine_control_reversal(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for index, label in enumerate(('slide', 'free')):
                source = case(4)
                source['actions'][2:, 0] = -1
                np.savez_compressed(root/f'episode_{index}.npz', states=source['states'],
                    actions=source['actions'], frames=source['frames'], map=source['map'], dt=1/30,
                    split='val', environment_profile='navigation_no_enemies_v1', movement_case=label)
            chosen = load_heldout_cases(root, steps=4)
            self.assertEqual([row['name'] for row in chosen], ['heldout_slide', 'heldout_return'])
            self.assertNotEqual(chosen[0]['source_path'], chosen[1]['source_path'])
            np.savez(root/'bad.npz', split='train')
            with self.assertRaisesRegex(ValueError, 'val provenance'):
                load_heldout_cases(root, steps=4)


if __name__ == '__main__':
    unittest.main()
