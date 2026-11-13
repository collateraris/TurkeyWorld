import copy
from dataclasses import dataclass
import json
import unittest
from unittest.mock import patch

import torch
from torch import nn

from turkeyworld.rssm_fusion_moe import RSSMFusionMoE, RSSMFusionMoEConfig
from turkeyworld.rssm_moe_navigation_objective import navigation_replay_backward


@dataclass
class Config:
    simulation_hz: float = 30.
    navigation_only: bool = False


class TinyNavigationModel(nn.Module):
    """Three recurrent memories and the actual shared production Transformer MoE."""
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.vjepa_backend = nn.Module()
        self.vjepa_backend.config = Config()
        self.p_core, self.e_core, self.c_core = nn.Linear(12, 12), nn.Linear(10, 10), nn.Linear(8, 8)
        self.p_head, self.c_context = nn.Linear(12, 2), nn.Linear(8, 16)
        self.fusion = RSSMFusionMoE(RSSMFusionMoEConfig(width=16, depth=1, heads=4, expert_hidden=32),
            player_memory_dim=12, enemy_memory_dim=10, map_width=16)
        self.frozen_renderer = nn.Linear(2, 2).requires_grad_(False)
        self.seen_frames, self.calls, self.returned_memories = [], [], []

    def encode_map(self, grid):
        return torch.ones(len(grid), 16, device=grid.device)

    def initialize_from_image(self, frames, states, fmap):
        self.seen_frames.append(frames.detach().clone())
        return torch.ones(len(states), 30, device=states.device)*.2

    def transition(self, state, action, fmap, memory):
        assert self.config.navigation_only and self.vjepa_backend.config.navigation_only
        self.calls.append((state.detach().clone(), memory.detach().clone()))
        p, e, c = memory[:, :12], memory[:, 12:22], memory[:, 22:]
        fusion = self.fusion(p, e, state, action, fmap, fmap[:, None].expand(-1, 2, -1),
            combat_context=self.c_context(c))
        p, e, c = self.p_core(p).tanh(), self.e_core(e).tanh(), self.c_core(c).tanh()
        passed = self.p_head(p)+fusion['player_pass_residual']
        new_state = torch.cat((state[:, :2]+.004*passed.sigmoid(), state[:, 2:]), -1)
        packed = torch.cat((p, e, c), -1)
        self.returned_memories.append(packed.detach().clone())
        return {'state': new_state, 'memory': packed,
            'respawn_flags': torch.zeros(len(state), 1, device=state.device, dtype=torch.bool),
            'rssm_fusion_moe': fusion}


def batch(horizon, *, count=2, context=2):
    states = torch.zeros(count, horizon+context, 20)
    states[..., :2] = .25
    states[..., 3:6] = 1
    states[..., 8:10] = .5
    states[..., 11] = 1
    for tick in range(horizon):
        states[:, context+tick, :2] += .003*(tick+1)
    frames = torch.zeros(count, horizon+context, 3, 8, 8)
    frames[:, :context] = .25
    frames[:, context:] = 99.  # Must never be observed by an encoder.
    return {'states':states, 'frames':frames, 'target_states':states[:, context:].clone(),
        'actions':torch.ones(count, horizon, 4), 'map':torch.zeros(count, 16, 16)}


class NavigationObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_own_memory_all_horizons_and_only_past_images(self):
        for horizon in (4, 8, 16, 32):
            with self.subTest(horizon=horizon):
                torch.manual_seed(7)
                model = TinyNavigationModel()
                initial_config, backend_config = model.config, model.vjepa_backend.config
                model.p_core.eval()
                modes = [m.training for m in model.modules()]
                labels = batch(horizon)
                for value in labels.values():
                    value.requires_grad_(True)
                progress = []
                with patch('turkeyworld.rssm_combat_branches._source_step_at_own_state',
                           side_effect=AssertionError('No Source query allowed')):
                    result = navigation_replay_backward(model, labels, context=2, observed_context=1,
                        expected_horizon=horizon, progress_callback=lambda a,b:progress.append((a,b)))
                self.assertEqual(result['nav_forward_transitions'], 2*horizon)
                self.assertEqual(result['nav_memory_resets_after_initialization'], 0)
                self.assertEqual(result['nav_npc_spawns'], 0)
                self.assertEqual(len(model.seen_frames), 1)
                self.assertEqual(tuple(model.seen_frames[0].shape[:2]), (2, 1))
                self.assertEqual(float(model.seen_frames[0].max()), .25)
                self.assertEqual(len(model.calls), horizon)
                for tick in range(1, horizon):
                    self.assertTrue(torch.equal(model.calls[tick][1], model.returned_memories[tick-1]))
                    self.assertTrue(torch.all(model.calls[tick][0][:,:2] > model.calls[tick-1][0][:,:2]))
                self.assertEqual(progress[-1], (horizon, horizon))
                self.assertIs(model.config, initial_config)
                self.assertIs(model.vjepa_backend.config, backend_config)
                self.assertEqual([m.training for m in model.modules()], modes)
                self.assertGreater(float(model.p_core.weight.grad.abs().sum()), 0.)
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.fusion.parameters() if p.grad is not None), 0.)
                self.assertTrue(all(value.grad is None for value in labels.values()))
                self.assertTrue(all(p.grad is None for p in model.frozen_renderer.parameters()))
                json.dumps(result, allow_nan=False)

    def test_weight_scales_losses_and_all_gradients(self):
        torch.manual_seed(8)
        model = TinyNavigationModel()
        half = copy.deepcopy(model)
        first = navigation_replay_backward(model, batch(8), context=2, weight=1.)
        second = navigation_replay_backward(half, batch(8), context=2, weight=.15625)
        self.assertAlmostEqual(second['nav_loss'], first['nav_loss']*.15625, places=9)
        self.assertAlmostEqual(second['nav_raw_player_loss'], first['nav_raw_player_loss'], places=9)
        for (name,a),(_,b) in zip(model.named_parameters(),half.named_parameters()):
            if a.grad is None:
                self.assertIsNone(b.grad)
            else:
                self.assertTrue(torch.allclose(b.grad,a.grad*.15625,atol=1e-9,rtol=1e-5),name)

    def test_reject_enemy_source_and_phantom_spawn(self):
        for field in (12, 13):
            labels = batch(4)
            labels['target_states'][0, -1, field] = 1.
            with self.assertRaisesRegex(ValueError, 'No-enemy replay rejects'):
                navigation_replay_backward(TinyNavigationModel(), labels, context=2)
        model = TinyNavigationModel()
        original = model.transition
        def phantom(*args):
            result = original(*args)
            result['respawn_flags'][0,0] = True
            return result
        model.transition = phantom
        with self.assertRaisesRegex(ValueError, 'respawn'):
            navigation_replay_backward(model, batch(4), context=2)
        self.assertFalse(model.config.navigation_only)
        self.assertFalse(model.vjepa_backend.config.navigation_only)

    def test_profile_and_child_modes_restored_on_error(self):
        model = TinyNavigationModel()
        config, backend = model.config, model.vjepa_backend.config
        modes = [m.training for m in model.modules()]
        def fail(*args):
            model.p_core.eval()
            raise RuntimeError('controlled test failure')
        model.transition = fail
        with self.assertRaisesRegex(RuntimeError, 'controlled'):
            navigation_replay_backward(model, batch(4), context=2)
        self.assertIs(model.config, config)
        self.assertIs(model.vjepa_backend.config, backend)
        self.assertEqual([m.training for m in model.modules()], modes)

    def test_horizon_450_uses_all_ticks_with_bounded_tbptt(self):
        model = TinyNavigationModel()
        result = navigation_replay_backward(model, batch(450, count=1, context=1),
            context=1, expected_horizon=450, check_gradients=False)
        self.assertEqual(len(model.calls),450)
        self.assertEqual(result['nav_player_axis_label_count'],900)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


if __name__ == '__main__':
    unittest.main()
