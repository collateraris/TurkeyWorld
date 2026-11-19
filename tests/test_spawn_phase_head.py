import unittest
from unittest.mock import patch

import torch

from turkeyworld.anchor_spawn import anchor_tile_indices, hierarchical_spawn_logits
from turkeyworld.spawn import decode_spawn_positions
from turkeyworld.spawn_phase_head import SpawnPhaseHead


class SpawnPhaseHeadTest(unittest.TestCase):
    def test_predicted_clock_uses_next_tick_and_independently_fitted_weights(self):
        head = SpawnPhaseHead()
        with torch.no_grad():
            head.scores.copy_(torch.arange(72).reshape(8, 9))
        state = torch.zeros(2, 20)
        state[:, -1] = torch.tensor([88., 89.]) / 9000
        logits = head(state, dt=1/30)
        torch.testing.assert_close(logits[:, 0], head.scores[:2])

    def test_learned_anchor_respects_wall_player_and_native_fallback(self):
        state = torch.zeros(3, 20)
        state[:, 3:5] = 1
        state[:, 16] = 5
        state[:, :2] = torch.tensor([8.5, 8.5]) / 16
        anchors = anchor_tile_indices()
        raw = torch.zeros(3, 1, 256)
        raw[:, :, 16+3] = 200
        grid = torch.ones(3, 16, 16, dtype=torch.bool)
        ranks = torch.zeros(3, 1, 9)
        # An arbitrary trained preference independent of Source's ordering.
        ranks[:, :, 7] = 20
        ranks[:, :, 5] = 10
        grid[1, 8, 2] = False
        grid[2].view(-1)[anchors] = False
        with patch('turkeyworld.spawn.spawn_tile_rank', side_effect=AssertionError('No oracle')):
            result = hierarchical_spawn_logits(raw, torch.zeros_like(ranks), state, grid, 1/30,
                                                category_logits=ranks)
            pos, valid = decode_spawn_positions(result['spawn_logits'], state, grid, 1/30)
        self.assertTrue(valid.all())
        self.assertEqual((pos * 16).tolist(), [[[2.5, 8.5]], [[6.5, 13.5]], [[3.5, 1.5]]])

    def test_frozen_resume_mask_keeps_calibrated_phase_head_and_old_spawn_heads_out_of_adam(self):
        from turkeyworld.vjepa_combat_training import dynamics_parameter_trainable
        for name in ('vjepa_backend.spawn_phase_head.scores', 'vjepa_backend.spawn_head.0.weight',
                     'vjepa_backend.core.encoder.blocks.15.norm1.weight'):
            self.assertFalse(dynamics_parameter_trainable(name, True))
        for name in ('vjepa_backend.hit_target_head.0.weight',
                     'vjepa_backend.core.encoder.blocks.16.norm1.weight'):
            self.assertTrue(dynamics_parameter_trainable(name, True))
        self.assertFalse(dynamics_parameter_trainable('vjepa_backend.hit_target_head.0.weight', False))
