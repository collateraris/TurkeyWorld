import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from turkeyworld.rssm_coordinate_gradients import COORDINATE_GRADIENT_PROTOCOL
from turkeyworld.rssm_drift_only_objective import drift_only_backward, drift_only_branches_backward
from turkeyworld.rssm_late_prefix_recovery import LATE_PREFIX_RECOVERY_PROTOCOL
from turkeyworld.rssm_loss_gated_curriculum import HORIZONS


XY = [0, 1, 8, 9]


def batch(horizon=16, count=1):
    states = torch.zeros(count, horizon+1, 20)
    states[..., XY] = .25
    states[..., [3, 4, 5, 12, 13]] = 1.
    frames = torch.full((count, horizon+1, 3, 2, 2), 9.)
    frames[:, 0] = 0.
    return {'states': states, 'target_states': states[:, 1:].clone(),
            'frames': frames, 'actions': torch.zeros(count, horizon, 4),
            'map': torch.zeros(count, 16, 16)}


class OnlyXY(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion = nn.Parameter(torch.full((500, 4), .0001))
        self.combat = nn.Parameter(torch.tensor(-1.))
        self.decoder = nn.Parameter(torch.tensor(.2))
        self.calls, self.outputs = [], []
        self.initializations = 0
        self.config = SimpleNamespace(navigation_only=False, simulation_hz=30.)
        self.vjepa_backend = SimpleNamespace(observe_step=self.forbidden, decode=self.forbidden)

    def forbidden(self, *args, **kwargs):
        raise AssertionError('Disabled posterior/decoder must never run')

    def encode_map(self, grid):
        return self.motion.new_zeros(len(grid), 1)

    def initialize_from_image(self, frames, states, features):
        self.initializations += 1
        assert frames.shape[1] == 1
        assert not bool(frames.any()), 'Future RGB entered initialization'
        return self.motion.new_zeros(len(states), 1)

    def transition(self, previous, action, features, memory):
        index = len(self.calls)
        self.calls.append((previous.detach().clone(), memory.detach().clone(), previous.requires_grad))
        delta = torch.zeros_like(previous)
        delta[:, XY] = self.motion[index]
        prediction = previous+delta
        # Predicted missing/dead actors must not remove Source XY support.
        prediction[:, [4, 12, 13]] = self.combat
        self.outputs.append(prediction)
        return {'state': prediction, 'memory': memory+self.motion[index].sum()*0,
            'event_logits': self.combat.expand(len(previous), 4),
            'rssm_reconstruction': self.decoder.expand(len(previous), 3, 2, 2),
            'respawn_flags': torch.zeros(len(previous), 1, dtype=torch.bool)}


class DriftOnlyObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_disabled_objectives_never_run_and_source_labels_receive_no_gradients(self):
        model, data = OnlyXY(), batch()
        model.combat.requires_grad_(False); model.decoder.requires_grad_(False)
        data = {name: value.requires_grad_() if value.is_floating_point() else value for name, value in data.items()}
        paths = ['turkeyworld.rssm_objectives._posterior_backward',
            'turkeyworld.rssm_objectives.state_loss', 'turkeyworld.rssm_objectives.trajectory_state_losses',
            'turkeyworld.rssm_combat_branches._direct_head_terms',
            'turkeyworld.combat_objectives.combat_transition_losses',
            'turkeyworld.contact_objectives.differentiable_contact_terms',
            'turkeyworld.rssm_memory_overshooting.memory_overshooting_backward']
        from contextlib import ExitStack
        with ExitStack() as stack:
            for path in paths:
                stack.enter_context(patch(path, side_effect=AssertionError('Disabled loss ran: '+path)))
            metrics = drift_only_backward(model, data, context=1, expected_horizon=16)
        self.assertEqual(model.initializations, 1)
        self.assertEqual(len(model.calls), 16)
        self.assertFalse(any(item[2] for item in model.calls))
        self.assertIsNone(model.combat.grad)
        self.assertIsNone(model.decoder.grad)
        self.assertTrue(bool((model.motion.grad[:16].abs().sum(-1) > 0).all()))
        self.assertTrue(all(value.grad is None for value in data.values() if value.is_floating_point()))
        self.assertEqual(metrics['rssm_source_posterior_calls'], 0)
        self.assertEqual(metrics['rssm_decoder_calls'], 0)
        self.assertEqual(metrics['rssm_kl_minimum_floor'], 0.)
        self.assertEqual(metrics['loss_groups']['interaction']['weighted'], 0.)
        self.assertEqual(metrics['train_loss'], metrics['loss_groups']['total'])
        json.dumps(metrics, allow_nan=False)

    def test_source_continuous_life_masks_exclude_exact_respawn_but_keep_later_lives_and_model_missing(self):
        data = batch(count=2)
        data['states'][0, 2, [12, 13]] = 0.
        data['states'][1, 5:, 4] = 0.
        data['target_states'] = data['states'][:, 1:].clone()
        model = OnlyXY()
        metrics = drift_only_backward(model, data, context=1)
        self.assertEqual(metrics['rssm_player_xy_supervised_transitions'], 20)
        self.assertEqual(metrics['rssm_npc_xy_supervised_transitions'], 30)
        self.assertGreater(metrics['rssm_xy_player'], 0.)
        self.assertGreater(metrics['rssm_xy_npc'], 0.)
        pred = torch.stack([value.detach() for value in model.outputs], 1)
        previous, target = data['states'][:, :-1], data['target_states']
        pm = (previous[..., 4] > 0) & (target[..., 4] > 0)
        nm = ((previous[..., 12] > 0) & (previous[..., 13] >= .5)
              & (target[..., 12] > 0) & (target[..., 13] >= .5))
        for name, ids, mask in [('player', [0, 1], pm), ('npc', [8, 9], nm)]:
            error = torch.nn.functional.smooth_l1_loss(pred[..., ids]*16, target[..., ids]*16,
                                                      reduction='none').sum(-1)
            expected = float((error*mask).sum()/(2*mask.sum()))
            self.assertAlmostEqual(metrics['rssm_xy_'+name], expected, places=6)

    def test_all_curriculum_horizons_keep_global_denominators_integrated_xy_and_exact_tbptt_tail(self):
        for horizon in HORIZONS:
            with self.subTest(horizon=horizon):
                model = OnlyXY()
                metrics = drift_only_backward(model, batch(horizon), context=1, expected_horizon=horizon)
                self.assertEqual(metrics['rssm_prior_tbptt_chunks'], (horizon+15)//16)
                self.assertEqual(metrics['rssm_npc_xy_supervised_transitions'], horizon)
                indices = {0, 7, 15, 16, 31, horizon-17, horizon-16, horizon-2, horizon-1}
                for index in sorted(value for value in indices if 0 <= value < horizon):
                    stop = min(horizon, (index//16+1)*16)
                    expected = 256*.0001/(2*horizon)*sum(range(index+1, stop+1))
                    torch.testing.assert_close(model.motion.grad[index], torch.full((4,), expected), rtol=5e-4, atol=2e-6)
                self.assertTrue(bool(torch.isfinite(model.motion.grad).all()))

    def test_horizons_outside_the_approved_path_reject_before_any_forward(self):
        for horizon in (48, 80, 449):
            with self.subTest(horizon=horizon):
                model = OnlyXY()
                with self.assertRaisesRegex(ValueError, 'horizon'):
                    drift_only_backward(model, batch(horizon), context=1, expected_horizon=horizon)
                self.assertEqual(model.initializations, 0)
                self.assertEqual(model.calls, [])

    def test_frozen_disabled_head_weights_and_saved_adam_momenta_do_not_change(self):
        model = OnlyXY()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
        sum(value.sum() for value in model.parameters()).backward()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        model.combat.requires_grad_(False); model.decoder.requires_grad_(False)
        before = {name: (value.detach().clone(), copy.deepcopy(optimizer.state[value]))
                  for name, value in [('combat', model.combat), ('decoder', model.decoder)]}
        drift_only_backward(model, batch(), context=1)
        optimizer.step()
        for name, value in [('combat', model.combat), ('decoder', model.decoder)]:
            self.assertIsNone(value.grad)
            torch.testing.assert_close(value, before[name][0], rtol=0, atol=0)
            for key, item in optimizer.state[value].items():
                torch.testing.assert_close(item, before[name][1][key], rtol=0, atol=0)

    def test_real_own_memory_branches_compute_only_xy_without_rgb_reset_or_combat_helpers(self):
        from test_rssm_combat_branches import fixture
        from turkeyworld.rssm_combat_protocol import environment_profile
        from turkeyworld.engine import Game
        model, candidate = fixture()
        with environment_profile(model, navigation_only=False), \
             patch.object(model, 'initialize_from_image', side_effect=AssertionError('No reset/RGB')), \
             patch.object(model.vjepa_backend, 'observe_step', side_effect=AssertionError('No posterior')), \
             patch.object(Game, 'render', side_effect=AssertionError('No Source RGB')), \
             patch('turkeyworld.rssm_combat_branches._direct_head_terms', side_effect=AssertionError('No head losses')), \
             patch('turkeyworld.rssm_combat_objectives.combat_supervised_terms', side_effect=AssertionError('No combat loss')):
            metrics = drift_only_branches_backward(model, [candidate], combat_branch_protocol={'version': 2},
                coordinate_gradient_protocol=COORDINATE_GRADIENT_PROTOCOL)
        self.assertGreater(metrics['branches'], 0)
        self.assertEqual(metrics['source_xy_offsets'], [1, 4, 8, 16])
        self.assertEqual(metrics['combat_loss_calls'], 0)
        self.assertEqual(metrics['direct_source_queries'], 0)
        self.assertEqual(metrics['memory_resets'], 0)
        self.assertAlmostEqual(metrics['loss'], metrics['player_xy_loss']+metrics['enemy_xy_loss'], places=6)
        self.assertGreater(sum(float(value.grad.abs().sum()) for value in model.core.parameters()
                               if value.grad is not None), 0.)
        json.dumps(metrics, allow_nan=False)

    def test_short_scout_continues_main_then_trains_only_late_source_xy(self):
        for horizon in (16, 32, 96, 448):
            with self.subTest(horizon=horizon):
                model, data = OnlyXY(), batch(horizon, count=2)
                with torch.no_grad():
                    model.combat.fill_(1.)
                model.combat.requires_grad_(False); model.decoder.requires_grad_(False)
                extension = {'actions': torch.zeros(2, 450, 4), 'map': data['map'].clone()}
                with patch('turkeyworld.rssm_combat_branches._direct_head_terms', side_effect=AssertionError('No direct heads')), \
                     patch('turkeyworld.rssm_combat_objectives.combat_supervised_terms', side_effect=AssertionError('No combat loss')):
                    metrics = drift_only_backward(model, data, context=1,
                        combat_branch_protocol={'version': 2}, late_prefix_protocol=LATE_PREFIX_RECOVERY_PROTOCOL,
                        late_prefix_batch=extension)
                self.assertEqual(model.initializations, 1)
                self.assertEqual(metrics['rssm_prior_tbptt_chunks'], horizon//16)
                self.assertEqual(metrics['rssm_late_prefix_scout_start_step'], horizon)
                self.assertEqual(metrics['rssm_late_prefix_scout_transition_calls'], 450-horizon)
                self.assertEqual(metrics['rssm_combat_branch_late_prefix_selected_count'], 8)
                self.assertGreater(metrics['rssm_combat_branch_loss'], 0.)
                self.assertEqual(metrics['rssm_combat_branch_combat_loss_calls'], 0)
                torch.testing.assert_close(model.calls[horizon][0], model.outputs[horizon-1].detach(), rtol=0, atol=0)
                self.assertIsNone(model.combat.grad)
                self.assertIsNone(model.decoder.grad)
                json.dumps(metrics, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
