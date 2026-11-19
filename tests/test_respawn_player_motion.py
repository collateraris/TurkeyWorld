"""The corrected combat profile keeps the player's learned wall contacts between lives."""
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from test_single_enemy_core import model as single_enemy_model
from turkeyworld.engine import Game
from turkeyworld.models import NeuralWorldModel
from turkeyworld.structured_state import player_command_delta


class RespawnPlayerMotionTest(unittest.TestCase):
    def test_between_lives_player_uses_collision_head_only_in_corrected_profile(self):
        torch.set_num_threads(2)
        game = Game(seed=1, endless=True)
        state = torch.tensor(game.state_vector())[None]
        state[:, 12:14] = 0.0
        state[:, 16] = 1.0
        action = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
        grid = torch.tensor(game.map.copy())[None].long()
        combat_config = single_enemy_model().config
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                config = replace(combat_config, player_motion_mode='learned_axis_quiet_pair',
                                 navigation_only=False, vjepa_spawn_phase_head=enabled)
                model = NeuralWorldModel(config).eval()
                memory = torch.full((1, model.core.memory_tokens, model.core.width), 0.01)
                with torch.inference_mode(), patch.object(
                    model.vjepa_backend.quiet_player_head, 'forward',
                    wraps=model.vjepa_backend.quiet_player_head.forward,
                ) as player, patch.object(
                    model.vjepa_backend.quiet_pursuit_head, 'forward',
                    side_effect=AssertionError('corpse should not use live pursuit dynamics'),
                ):
                    result = model.transition(state, action, model.encode_map(grid), memory)
                self.assertEqual(player.call_count, int(enabled))
                self.assertFalse(result['quiet_pair_active'].item())
                self.assertEqual(result['state'][0, 13].item(), 0.)
                self.assertGreater(result['state'][0, 16].item(), state[0, 16].item())
                if enabled:
                    expected = state[:, :2] + player_command_delta(state, action)
                    torch.testing.assert_close(result['state'][:, :2], expected, atol=1e-7, rtol=0)
