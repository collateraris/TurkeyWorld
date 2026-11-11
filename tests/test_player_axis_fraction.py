"""Real source kinematics and learned axis gates, without runtime engine use."""
from dataclasses import asdict
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.engine import Game
from turkeyworld.models import NeuralWorldModel, WorldModelConfig
from turkeyworld.structured_state import STATE_DIM, player_command_delta, structured_transition


def axis_config(mode='delta', **overrides):
    values = dict(kind='teacher', architecture='vjepa2_ac', image_size=32,
        endless=True, navigation_only=True, spatial_entities=True, player_motion_mode=mode,
        vjepa_width=32, vjepa_encoder_depth=1, vjepa_predictor_depth=1,
        vjepa_heads=4, vjepa_patch_size=8, vjepa_context_frames=2)
    values.update(overrides)
    return WorldModelConfig(**values)


class PlayerAxisFractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def game(wall=False):
        game = Game(seed=173, endless=True, enemies_enabled=False)
        game.map[:] = 0
        game.map[[0, -1], :] = 1
        game.map[:, [0, -1]] = 1
        if wall:
            game.map[:, 4] = 1
        game.player.update(x=3.78 if wall else 2.5, y=2.5, angle=0.)
        return game

    def test_command_matches_typed_turn_diagonal_limits_and_dead_player(self):
        game = self.game()
        state = torch.tensor(np.stack([game.state_vector()]*5))
        state[3, 4] = 0
        actions = torch.tensor([[1., 1., .3, 0.], [-1., 0., -.5, 0.],
            [0., 0., 1., 0.], [1., 0., 0., 0.], [2., -2., 2., 0.]])
        command = player_command_delta(state, actions)
        candidate = state.clone()
        candidate[:, :2] += command
        result = structured_transition(state, candidate, actions, torch.zeros(5, 2),
            torch.full((5, 1), -10.), endless=True, respawn_enabled=False)
        torch.testing.assert_close(result['state'][:, :2]-state[:, :2], command, rtol=0, atol=2e-8)
        self.assertEqual(command[2:].shape, (3, 2))
        self.assertEqual(command[2].abs().sum().item(), 0)
        self.assertEqual(command[3].abs().sum().item(), 0)
        self.assertTrue(bool((command.norm(dim=-1) <= .005+1e-7).all()))

    def test_zero_logits_are_an_explicit_untrained_half_pass_prior(self):
        model = NeuralWorldModel(axis_config('axis_fraction')).eval()
        game = self.game()
        state = torch.tensor(game.state_vector())[None]
        grid = torch.tensor(game.map)[None]
        action = torch.tensor([[1., 0., 0., 0.]])
        maps = model.encode_map(grid)
        memory = torch.full((1, model.core.memory_tokens, model.core.width), .01)
        with torch.inference_mode(), patch.object(Game, 'step', side_effect=AssertionError('No engine step')), \
             patch.object(Game, 'render', side_effect=AssertionError('No engine render')):
            result = model.transition(state, action, maps, memory)
        torch.testing.assert_close(result['player_pass_logits'], torch.zeros(1, 2), rtol=0, atol=0)
        torch.testing.assert_close(result['player_pass_fraction'], torch.full((1, 2), .5), rtol=0, atol=0)
        torch.testing.assert_close(result['raw_player_delta'], .5*result['player_command_delta'], rtol=0, atol=0)
        torch.testing.assert_close(result['state'][:, :2]-state[:, :2], result['raw_player_delta'], rtol=0, atol=2e-8)

    def test_source_free_slide_and_stop_can_be_represented_without_a_collision_oracle(self):
        model = NeuralWorldModel(axis_config('axis_fraction')).eval()
        for case, wall, controls, logits in [
            ('free', False, [1., 0., 0., 0.], [20., 20.]),
            ('slide', True, [1., 1., 0., 0.], [-20., 20.]),
            ('stop', True, [1., 0., 0., 0.], [-20., 20.])]:
            with self.subTest(case=case):
                game = self.game(wall)
                state = torch.tensor(game.state_vector())[None]
                game.step(controls)
                target = torch.tensor(game.state_vector())[None]
                with torch.no_grad():
                    model.vjepa_backend.player_pass_head[-1].bias.copy_(torch.tensor(logits))
                maps = model.encode_map(torch.tensor(game.map)[None])
                memory = torch.full((1, model.core.memory_tokens, model.core.width), .01)
                with torch.inference_mode(), patch.object(Game, 'step', side_effect=AssertionError('No source query')):
                    result = model.transition(state, torch.tensor([controls]), maps, memory)
                torch.testing.assert_close(result['state'][:, :2], target[:, :2], rtol=0, atol=2e-7)
                if case == 'slide':
                    self.assertEqual((target[0, 0]-state[0, 0]).item(), 0)
                    self.assertGreater((target[0, 1]-state[0, 1]).item(), 0)

    def test_raw_delta_gradient_teaches_active_axis_and_masks_inactive_axis(self):
        model = NeuralWorldModel(axis_config('axis_fraction')).train()
        game = self.game()
        state = torch.tensor(game.state_vector())[None]
        maps = model.encode_map(torch.tensor(game.map)[None])
        memory = torch.full((1, model.core.memory_tokens, model.core.width), .01)
        result = model.transition(state, torch.tensor([[1., 0., 0., 0.]]), maps, memory)
        result['player_pass_logits'].retain_grad()
        ((result['raw_player_delta']-result['player_command_delta']).square().sum()/.005**2).backward()
        self.assertLess(result['player_pass_logits'].grad[0, 0].item(), 0)
        self.assertEqual(result['player_pass_logits'].grad[0, 1].item(), 0)
        self.assertGreater(model.vjepa_backend.player_pass_head[-1].weight.grad.abs().sum().item(), 0)
        self.assertEqual(model.vjepa_backend.player_pass_head[-1].bias.grad[1].item(), 0)

    def test_legacy_default_keeps_rng_weights_outputs_and_new_branch_appends_parameters(self):
        torch.manual_seed(173)
        original = NeuralWorldModel(axis_config()).eval()
        rng = torch.random.get_rng_state().clone()
        torch.manual_seed(173)
        explicit = NeuralWorldModel(axis_config('delta')).eval()
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0, atol=0)
        torch.manual_seed(173)
        expanded = NeuralWorldModel(axis_config('axis_fraction')).eval()
        self.assertEqual(list(dict(expanded.named_parameters()))[:len(list(original.parameters()))],
                         list(dict(original.named_parameters())))
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, expanded.state_dict()[name], rtol=0, atol=0)
        for key in set(expanded.state_dict())-set(original.state_dict()):
            self.assertTrue(key.startswith('vjepa_backend.player_pass_head.'))

    def test_head_count_and_mode_validation(self):
        with torch.device('meta'):
            model = NeuralWorldModel(axis_config('axis_fraction', vjepa_width=512, vjepa_heads=8))
        self.assertEqual(sum(p.numel() for p in model.vjepa_backend.player_pass_head.parameters()), 132226)
        for overrides in ({'spatial_entities': False}, {'endless': False}, {'player_motion_mode': 'other'}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                axis_config('axis_fraction', **overrides)


if __name__ == '__main__':
    unittest.main()
