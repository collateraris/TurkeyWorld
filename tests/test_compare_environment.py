"""Explicit paired scene changes preserve each game's camera and own RSSM h/z."""
from dataclasses import dataclass
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from turkeyworld.compare_runtime import PairedRuntime
from turkeyworld.engine import Game
from test_compare_runtime import NeuralStub, PoseRenderer


@dataclass(frozen=True)
class Config:
    navigation_only: bool = True
    endless: bool = True
    image_size: int = 32
    simulation_hz: float = 30.


class CompareEnvironmentTests(unittest.TestCase):
    def paired(self):
        neural = NeuralStub(seed=137, navigation_only=True)
        source = Game(seed=137, endless=True, enemies_enabled=False)
        neural.initial_state = torch.tensor(source.state_vector())[None]
        neural.initial_map = source.map.copy()
        neural.initial_rgb = source.render(size=32, overlays=False)
        neural.model = torch.nn.Module()
        neural.model.config = Config()
        neural.model.weight_identity = object()
        neural.model.register_parameter('frozen_renderer_weight', torch.nn.Parameter(torch.tensor([.25, .75]), requires_grad=False))
        neural.model.vjepa_backend = torch.nn.Module()
        neural.model.vjepa_backend.config = Config()
        neural.model.vjepa_backend.register_parameter('dynamics_weight', torch.nn.Parameter(torch.tensor([1., 2.])))
        neural.model.vjepa_backend.register_buffer('frozen_codec_usage', torch.tensor([3., 4.]))
        neural.world_metadata['config'] = {'navigation_only': False}
        renderer = PoseRenderer()
        pair = PairedRuntime(neural, source_size=32, render_pose=renderer,
            shared_camera_entities='neural')
        return pair, neural, renderer

    def test_real_source_toggle_only_initializes_npc_preserves_player_clocks_own_memory_and_weights(self):
        pair, neural, _ = self.paired()
        pair.step([1, 0, .3, 0]); pair.step([0, -1, -.1, 1])
        source_player = pair.source.get_state()['player']
        own = neural.state.clone(); memory = neural.memory.clone()
        memory_ptr = neural.memory.data_ptr(); identity = neural.model.weight_identity
        weights = {name: value.clone() for name, value in neural.model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        cached = (neural.initial_state.clone(), neural.initial_memory.clone())
        times = pair.source.time, float(neural.state[0, -1])
        tape = list(pair.actions)
        pair.set_environment('combat')
        self.assertTrue(pair.source.enemies_enabled)
        self.assertEqual(pair.source.get_state()['player'], source_player)
        self.assertTrue(torch.equal(neural.state[:, :8], own[:, :8]))
        self.assertTrue(torch.equal(neural.state[:, -2:], own[:, -2:]))
        self.assertTrue(torch.equal(neural.memory, memory))
        self.assertEqual(neural.memory.data_ptr(), memory_ptr)
        self.assertIs(neural.model.weight_identity, identity)
        self.assertTrue(all(torch.equal(neural.model.state_dict()[name], value) for name, value in weights.items()))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertTrue(torch.equal(neural.initial_state, cached[0]))
        self.assertTrue(torch.equal(neural.initial_memory, cached[1]))
        self.assertFalse(neural.model.config.navigation_only)
        self.assertFalse(neural.model.vjepa_backend.config.navigation_only)
        self.assertEqual((pair.steps, neural.steps, pair.source.time, float(neural.state[0, -1])), (2, 2, *times))
        self.assertEqual(pair.actions, tape)
        self.assertEqual(float(neural.state[0, 13]), 1.)
        self.assertTrue(pair.source.enemies[0]['alive'])
        changes = len(pair.environment_changes)
        pair.set_environment('combat')
        self.assertEqual(len(pair.environment_changes), changes)
        pair.toggle_environment()
        self.assertFalse(pair.source.enemies_enabled)
        self.assertFalse(pair.source.enemies[0]['alive'])
        self.assertTrue(torch.equal(neural.state[:, 12:14], torch.zeros(1, 2)))
        self.assertTrue(torch.equal(neural.memory, memory))
        self.assertTrue(neural.model.config.navigation_only)
        self.assertTrue(neural.model.vjepa_backend.config.navigation_only)
        status = pair.status()
        self.assertTrue(status['runtime_initial_navigation_only'])
        self.assertFalse(status['checkpoint_navigation_only'])
        self.assertEqual(status['environment'], 'navigation')
        self.assertEqual(status['source_steps'], 2)
        self.assertEqual(status['engine_corrections'], 0)
        self.assertEqual(neural.reset_calls, 1)

    def test_neural_entities_shared_camera_uses_source_xy_heading_only_and_never_aliases(self):
        pair, neural, renderer = self.paired()
        pair.set_environment('combat'); pair.step([1, 0, .2, 0])
        neural.state[0, 4:8] = torch.tensor([.45, .8, .2, .6])
        neural.state[0, 8:10] = torch.tensor([.6, .7])
        neural.state[0, 12] = .25
        own, memory = neural.state.clone(), neural.memory.clone()
        pair.set_mode('shared_camera')
        query = renderer.calls[-1]
        torch.testing.assert_close(query[:, :4], torch.tensor(pair.source.state_vector()[:4])[None], rtol=0, atol=0)
        self.assertTrue(torch.equal(query[:, 4:], own[:, 4:]))
        self.assertFalse(torch.equal(query[:, 8:10], torch.tensor(pair.source.state_vector()[8:10])[None]))
        self.assertTrue(torch.equal(neural.state, own))
        self.assertTrue(torch.equal(neural.memory, memory))
        renderer.poison_argument = True
        source_before = pair.source.state_vector().copy()
        pair.set_mode('shared_camera')
        self.assertTrue(torch.equal(neural.state, own))
        np.testing.assert_array_equal(pair.source.state_vector(), source_before)
        self.assertTrue(pair.status()['reference_player_camera_only'])
        self.assertFalse(pair.status()['rssm_future_source_feedback'])

    def test_reset_keeps_selected_profile_but_uses_original_cached_initial_observation_and_hz(self):
        pair, neural, _ = self.paired()
        pair.step([1, 0, .2, 0]); pair.toggle_environment()
        pair.reset()
        self.assertEqual(pair.environment, 'combat')
        self.assertTrue(pair.source.enemies_enabled)
        self.assertFalse(neural.model.vjepa_backend.config.navigation_only)
        self.assertTrue(torch.equal(neural.memory, neural.initial_memory))
        self.assertTrue(torch.equal(neural.state[:, :8], neural.initial_state[:, :8]))
        self.assertEqual(float(neural.state[0, 13]), 1.)
        self.assertEqual((pair.steps, neural.steps, pair.simulated_seconds), (0, 0, 0.))
        self.assertEqual(pair.neural.simulation_hz, 30.)
        pair.toggle_environment(); pair.reset()
        self.assertFalse(pair.source.enemies_enabled)
        self.assertTrue(neural.model.config.navigation_only)
        self.assertTrue(torch.equal(neural.state, neural.initial_state))
        before = pair.source.state_vector().copy()
        with self.assertRaises(ValueError):
            pair.set_environment('two_enemies')
        np.testing.assert_array_equal(pair.source.state_vector(), before)
        self.assertEqual(pair.environment, 'navigation')


if __name__ == '__main__':
    unittest.main()
