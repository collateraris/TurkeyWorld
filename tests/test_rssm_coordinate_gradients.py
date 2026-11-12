"""XY integration credits earlier motion without restoring typed-state feedback."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from turkeyworld.rssm_coordinate_gradients import (
    normalize_coordinate_gradient_protocol, integrate_coordinate_gradients)
from turkeyworld.rssm_objectives import rssm_dynamics_backward


PROTOCOL = {'version': 1, 'mode': 'detached_conditioning_integrated_xy',
            'exclude_predicted_respawns': True}
XY = [0, 1, 8, 9]


def state(count=1, dtype=torch.float64):
    value = torch.zeros(count, 20, dtype=dtype)
    value[:, [0, 1, 8, 9]] = .25
    value[:, [3, 4, 5, 12, 13]] = 1.
    return value


def xy_values(value):
    result = value.new_zeros(len(value), 20)
    result[:, XY] = value
    return result


class IndependentMotion(nn.Module):
    """Per-tick independent deltas make the missing kinematic derivative visible."""
    def __init__(self, horizon):
        super().__init__()
        self.deltas = nn.Parameter(torch.full((horizon, 4), .001))
        self.calls = []
        self.outputs = []

    def encode_map(self, grid):
        return self.deltas.new_zeros(len(grid), 1)

    def initialize_from_image(self, frames, states, features):
        return self.deltas.new_zeros(len(states), 4)

    def transition(self, previous, action, features, memory):
        index = len(self.calls)
        self.calls.append((previous.detach().clone(), memory.detach().clone(), previous.requires_grad))
        # The model input is deliberately detached by the production caller.
        current = previous + xy_values(self.deltas[index:index+1].expand(len(previous), -1))
        current.retain_grad()
        self.outputs.append(current)
        return {'state': current, 'memory': memory + self.deltas[index].sum()*0,
                'respawn_flags': torch.zeros(len(previous), 1, dtype=torch.bool)}


def rollout_batch(horizon):
    states = state(dtype=torch.float32)[:, None].repeat(1, horizon+1, 1)
    return {'states': states, 'target_states': states[:, 1:].clone(),
            'frames': torch.zeros(1, horizon+1, 3, 2, 2),
            'actions': torch.zeros(1, horizon, 4), 'map': torch.zeros(1, 16, 16)}


def zero_geometry(prediction, *args, **kwargs):
    zero = prediction.sum()*0
    return {'ar_wall_player': zero, 'ar_wall_npc': zero}, {}


class CoordinateGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_protocol_and_legacy_identity_fail_closed(self):
        self.assertIsNone(normalize_coordinate_gradient_protocol(None))
        self.assertEqual(normalize_coordinate_gradient_protocol(PROTOCOL), PROTOCOL)
        previous = state().requires_grad_()
        result = {'state': previous.square(), 'memory': torch.ones(1, 3)}
        self.assertIs(integrate_coordinate_gradients(previous, result, None), result)
        for bad in ({**PROTOCOL, 'version': True}, {**PROTOCOL, 'version': 2},
                    {**PROTOCOL, 'exclude_predicted_respawns': False},
                    {**PROTOCOL, 'mode': 'attached'}, {**PROTOCOL, 'unknown': 1}):
            with self.assertRaises(ValueError): normalize_coordinate_gradient_protocol(bad)
        with self.assertRaises(ValueError): integrate_coordinate_gradients(previous, result, PROTOCOL)

    def test_only_xy_gets_identity_and_predicted_respawn_cuts_npc_only(self):
        previous = state(2).requires_grad_()
        local = state(2).add(.013).requires_grad_()
        memory = torch.randn(2, 7, requires_grad=True)
        flags = torch.tensor([[False], [True]])
        result = {'state': local, 'memory': memory, 'respawn_flags': flags,
                  'features': torch.randn(2, 128), 'event_logits': torch.randn(2, 4)}
        original = {key: value.detach().clone() for key, value in result.items()}
        output = integrate_coordinate_gradients(previous, result, PROTOCOL)
        self.assertIsNot(output, result)
        torch.testing.assert_close(output['state'], local, rtol=0, atol=0)
        for key in result:
            torch.testing.assert_close(result[key], original[key], rtol=0, atol=0)
            if key != 'state': self.assertIs(output[key], result[key])
        output['state'].sum().backward()
        expected = torch.zeros_like(previous)
        expected[:, :2] = 1
        expected[0, 8:10] = 1
        torch.testing.assert_close(previous.grad, expected, rtol=0, atol=0)
        torch.testing.assert_close(local.grad, torch.ones_like(local), rtol=0, atol=0)
        self.assertIsNone(memory.grad)

    def test_sixteen_deltas_receive_analytic_endpoint_credit_with_bitexact_forward(self):
        def run(protocol):
            deltas = torch.full((16, 4), .001, dtype=torch.float64, requires_grad=True)
            current = state()
            trajectory = []
            for index in range(16):
                result = {'state': current.detach()+xy_values(deltas[index:index+1]),
                          'respawn_flags': torch.zeros(1, 1, dtype=torch.bool),
                          'memory': torch.tensor([[index]], dtype=torch.float64)}
                current = integrate_coordinate_gradients(current, result, protocol)['state']
                trajectory.append(current.detach().clone())
            current[:, XY].sum().backward()
            return torch.stack(trajectory), deltas.grad
        legacy, old_grad = run(None)
        integrated, new_grad = run(PROTOCOL)
        torch.testing.assert_close(integrated, legacy, rtol=0, atol=0)
        expected_old = torch.zeros_like(old_grad); expected_old[-1] = 1.
        torch.testing.assert_close(old_grad, expected_old, rtol=0, atol=0)
        torch.testing.assert_close(new_grad, torch.ones_like(new_grad), rtol=0, atol=0)

    def test_spawn_breaks_earlier_npc_credit_but_keeps_local_spawn_gradient(self):
        early = torch.full((1, 4), .001, dtype=torch.float64, requires_grad=True)
        spawn = torch.tensor([[.4, .5]], dtype=torch.float64, requires_grad=True)
        late = torch.full((1, 4), .002, dtype=torch.float64, requires_grad=True)
        current = state()
        current = integrate_coordinate_gradients(current,
            {'state': current.detach()+xy_values(early),
             'respawn_flags': torch.zeros(1, 1, dtype=torch.bool)}, PROTOCOL)['state']
        local = torch.cat([current.detach()[:, :8], spawn, current.detach()[:, 10:]], -1)
        current = integrate_coordinate_gradients(current,
            {'state': local, 'respawn_flags': torch.ones(1, 1, dtype=torch.bool)}, PROTOCOL)['state']
        current = integrate_coordinate_gradients(current,
            {'state': current.detach()+xy_values(late),
             'respawn_flags': torch.zeros(1, 1, dtype=torch.bool)}, PROTOCOL)['state']
        current[:, XY].sum().backward()
        torch.testing.assert_close(early.grad, torch.tensor([[1., 1., 0., 0.]], dtype=early.dtype))
        torch.testing.assert_close(spawn.grad, torch.ones_like(spawn), rtol=0, atol=0)
        torch.testing.assert_close(late.grad, torch.ones_like(late), rtol=0, atol=0)

    def test_real_main_loop_all_step_gradient_and_tbptt_boundary(self):
        def run(protocol):
            model = IndependentMotion(32)
            batch = rollout_batch(32)
            with patch('turkeyworld.rssm_objectives._posterior_backward', return_value={'loss': 0.}), \
                 patch('turkeyworld.rssm_objectives.state_loss', side_effect=lambda p, t: p.sum()*0), \
                 patch('turkeyworld.rssm_objectives.trajectory_state_losses', side_effect=zero_geometry):
                metrics = rssm_dynamics_backward(model, batch, context=1, expected_horizon=32,
                    chunk_size=16, numeric_state_feedback='detached',
                    coordinate_gradient_protocol=protocol)
            return model, metrics
        old, old_metrics = run(None)
        new, new_metrics = run(PROTOCOL)
        self.assertEqual(old_metrics['train_loss'], new_metrics['train_loss'])
        self.assertEqual(new_metrics['rssm_prior_tbptt_chunks'], 2)
        self.assertEqual(new_metrics['rssm_prior_future_rgb_inputs'], 0)
        for left, right in zip(old.calls, new.calls):
            self.assertFalse(right[2])
            torch.testing.assert_close(left[0], right[0], rtol=0, atol=0)
            torch.testing.assert_close(left[1], right[1], rtol=0, atol=0)
        # Whole-tape 2*H denominator is unchanged. The first delta receives
        # positions1..16, but not17..32 across the explicit TBPTT boundary.
        multiplier = .001*16*16/(2*32)
        for index in range(32):
            stop = 16 if index < 16 else 32
            expected = multiplier*sum(range(index+1, stop+1))
            torch.testing.assert_close(new.deltas.grad[index], torch.full((4,), expected), rtol=2e-5, atol=1e-6)
            torch.testing.assert_close(old.deltas.grad[index], torch.full((4,), multiplier*(index+1)), rtol=2e-5, atol=1e-6)
        self.assertTrue(torch.isfinite(new.deltas.grad).all())

    def test_real_branch_endpoint_credit_preserves_own_memory_and_no_source_images(self):
        from test_rssm_combat_branches import fixture
        from turkeyworld.rssm_combat_branches import combat_branches_backward
        from turkeyworld.rssm_combat_protocol import environment_profile
        from turkeyworld.engine import Game
        original, candidate = fixture()
        protocol = {'max_branches': 1, 'offsets': [16], 'state_weight': 0.,
                    'wall_weight': 0., 'anchor_combat_weight': 0.}
        traces, gradients = [], []
        for coordinate in (None, PROTOCOL):
            model = deepcopy(original)
            transition = model.transition
            calls, outputs = [], []
            def record(previous, action, fmap, memory):
                calls.append((previous.detach().clone(), memory.detach().clone(), previous.requires_grad))
                result = transition(previous, action, fmap, memory)
                result['state'].retain_grad(); outputs.append(result['state'])
                return result
            model.transition = record
            with environment_profile(model, navigation_only=False), \
                 patch.object(model, 'initialize_from_image', side_effect=AssertionError('No reset')), \
                 patch.object(Game, 'render', side_effect=AssertionError('No Source RGB')):
                metrics = combat_branches_backward(model, [deepcopy(candidate)], combat_branch_protocol=protocol,
                    coordinate_gradient_protocol=coordinate)
            self.assertEqual(metrics['new_observed_rgb_frames'], 0)
            self.assertEqual(metrics['memory_resets'], 0)
            self.assertEqual(len(calls), 16)
            traces.append(calls)
            gradients.append(outputs[0].grad[:, XY].abs().sum().item())
        self.assertEqual(gradients[0], 0.)
        self.assertGreater(gradients[1], 0.)
        for left, right in zip(*traces):
            self.assertFalse(right[2])
            torch.testing.assert_close(left[0], right[0], rtol=0, atol=0)
            torch.testing.assert_close(left[1], right[1], rtol=0, atol=0)

    def test_direct_local_branch_supervision_does_not_acquire_spurious_earlier_delta_credit(self):
        from test_rssm_combat_branches import fixture
        from turkeyworld.rssm_combat_branches import combat_branches_backward
        from turkeyworld.rssm_combat_protocol import environment_profile
        original, candidate = fixture()
        protocol = {'version': 2, 'max_branches': 1, 'xy_weight': 0.,
                    'state_weight': 0., 'wall_weight': 0., 'anchor_combat_weight': 1.}
        traces = []
        for coordinate in (None, PROTOCOL):
            model = deepcopy(original)
            with environment_profile(model, navigation_only=False):
                metrics = combat_branches_backward(model, [deepcopy(candidate)],
                    combat_branch_protocol=protocol, coordinate_gradient_protocol=coordinate)
            self.assertGreater(metrics['direct_supervised_queries'], 0)
            traces.append((metrics['loss'], {name: parameter.grad.detach().clone()
                for name, parameter in model.named_parameters() if parameter.grad is not None}))
        self.assertEqual(traces[0][0], traces[1][0])
        self.assertEqual(set(traces[0][1]), set(traces[1][1]))
        for name, gradient in traces[0][1].items():
            torch.testing.assert_close(gradient, traces[1][1][name], rtol=0, atol=0, msg=name)


if __name__ == '__main__': unittest.main()
