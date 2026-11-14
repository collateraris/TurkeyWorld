"""Actual RF3/VQ/D gradients and the whole-frame Source objective."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import torch

from test_renderer_rf5_objectives import model, source
from turkeyworld.object_losses import decoded_ssim_loss
from turkeyworld.renderer_moe_objectives import moe_conditions
from turkeyworld.renderer_rf_codec_objectives import (
    PROTOCOL, renderer_rf_codec_protocol, rf_codec_forward, rf_codec_refinement_loss)
from turkeyworld.renderer_student import _hash_tensors


def network():
    return model(3)


def frozen_state(net):
    return {name: value.clone() for name, value in net.state_dict().items()
            if not name.startswith('flow_renderer.')
            or name.startswith(('flow_renderer.codec.', 'flow_renderer.geometry_opacity.'))}


class RendererRFCodecObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_three_real_velocity_calls_terminal_Q_and_D_carry_gradients_without_Source_forward(self):
        net, data = network(), source()
        conditions = moe_conditions(net, data)
        before = _hash_tensors(net.state_dict())
        rng = torch.get_rng_state().clone()
        with patch('turkeyworld.engine.Game.render', side_effect=AssertionError('No Source rendering')), \
             patch('turkeyworld.engine.Game.step', side_effect=AssertionError('No Source transition')), \
             patch.object(net.flow_renderer, 'velocity', wraps=net.flow_renderer.velocity) as calls, \
             patch.object(net.flow_renderer.codec.quantize, 'forward', wraps=net.flow_renderer.codec.quantize.forward) as q, \
             patch.object(net.flow_renderer.codec, 'decode', wraps=net.flow_renderer.codec.decode) as decode:
            base, rgb, endpoint, routers = rf_codec_forward(net, conditions, checkpoint_velocity=False)
            self.assertEqual(calls.call_count, 3)
            # Frozen Base encoding has its own Q; terminal Q is the last call.
            self.assertGreaterEqual(q.call_count, 2)
            self.assertEqual(decode.call_count, 1)
        endpoint.retain_grad()
        rgb.square().mean().backward()
        self.assertFalse(base.requires_grad)
        self.assertEqual(len(routers), 3)
        self.assertGreater(float(endpoint.grad.abs().sum()), 0)
        self.assertGreater(float(net.flow_renderer.image_input.weight.grad.abs().sum()), 0)
        for expert in net.flow_renderer.output_moe.experts:
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in expert.parameters() if p.grad is not None), 0)
        for name, parameter in net.named_parameters():
            if (not name.startswith('flow_renderer.')
                    or name.startswith(('flow_renderer.codec.', 'flow_renderer.geometry_opacity.'))):
                self.assertIsNone(parameter.grad, name)
        self.assertEqual(before, _hash_tensors(net.state_dict()))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_main_uses_unmasked_whole_SSIM_and_exact_new_formula_without_Source_image_branch(self):
        net, data = network(), source()
        with patch('turkeyworld.renderer_rf_codec_objectives.decoded_ssim_loss', wraps=decoded_ssim_loss) as ssim, \
             patch.object(net.flow_renderer.codec, 'encode', wraps=net.flow_renderer.codec.encode) as encode, \
             patch.object(net.flow_renderer.codec, 'decode', wraps=net.flow_renderer.codec.decode) as decode:
            loss, metrics = rf_codec_refinement_loss(net, data, checkpoint_velocity=False)
        self.assertEqual(ssim.call_count, 1)
        self.assertIsNone(ssim.call_args.kwargs['mask'])
        self.assertEqual(encode.call_count, 2)  # Base condition and detached Source-Q label.
        self.assertEqual(decode.call_count, 1)  # RF-Q only, never Source-Q RGB reconstruction.
        self.assertEqual(set(metrics['weighted_terms']),
                         {'global_ssim', 'region_l1', 'region_edge', 'native_crop', 'geometry_wall', 'source_latent'})
        self.assertAlmostEqual(metrics['weighted_terms']['region_l1'], .5 * metrics['region_l1'], places=7)
        self.assertAlmostEqual(metrics['weighted_terms']['global_ssim'], metrics['global_ssim_loss'], places=7)
        self.assertAlmostEqual(metrics['weighted_terms']['source_latent'], .1 * metrics['source_latent_mse'], places=7)
        self.assertAlmostEqual(float(loss.detach()), sum(metrics['weighted_terms'].values())
            + .01 * metrics['renderer_moe_role_loss'] + .001 * metrics['renderer_moe_balance_loss'], places=6)
        self.assertFalse(metrics['base_regression_loss_active'])
        self.assertFalse(metrics['Source_Q_image_branch'])
        self.assertFalse(metrics['restorer_active'])
        self.assertFalse(metrics['global_ssim_region_masked'])
        json.dumps(metrics, allow_nan=False)

    def test_checkpoint_and_ordinary_gradients_match_and_router_labels_remain_three_after_recompute(self):
        first = network(); second = deepcopy(first); data = source()
        loss_a, metrics_a = rf_codec_refinement_loss(first, data, checkpoint_velocity=False)
        loss_a.backward()
        with patch.object(second.flow_renderer, 'velocity', wraps=second.flow_renderer.velocity) as calls:
            loss_b, metrics_b = rf_codec_refinement_loss(second, data, checkpoint_velocity=True)
            self.assertEqual(calls.call_count, 3)
            self.assertEqual(len(second.flow_renderer.output_moe._forward_hooks), 0)
            loss_b.backward()
            self.assertGreater(calls.call_count, 3)
        self.assertEqual(metrics_a['rf_router_steps'], metrics_b['rf_router_steps'])
        self.assertEqual(metrics_b['router_velocity_calls'], 3)
        self.assertEqual([row['euler_time'] for row in metrics_b['rf_router_steps']], [0, 1/3, 2/3])
        self.assertAlmostEqual(metrics_b['renderer_moe_role_loss'],
            sum(row['role_ce'] for row in metrics_b['rf_router_steps']) / 3, places=6)
        self.assertTrue(torch.allclose(loss_a, loss_b, atol=1e-7, rtol=1e-6))
        for name, parameter in first.named_parameters():
            other = dict(second.named_parameters())[name]
            if parameter.grad is None:
                self.assertIsNone(other.grad, name)
            else:
                self.assertTrue(torch.allclose(parameter.grad, other.grad, atol=2e-6, rtol=2e-4), name)

    def test_two_actual_updates_leave_every_parent_tensor_and_Source_label_unchanged(self):
        net, data = network(), source()
        for key in ('state', 'target', 'background', 'object_mask'):
            data[key] = data[key].detach().requires_grad_()
        before = frozen_state(net)
        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-4)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            rng = torch.get_rng_state().clone()
            loss, metrics = rf_codec_refinement_loss(net, data, checkpoint_velocity=True)
            loss.backward(); optimizer.step()
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for name, value in before.items():
                self.assertTrue(torch.equal(value, net.state_dict()[name]), name)
            for key in ('state', 'target', 'background', 'object_mask'):
                self.assertIsNone(data[key].grad, key)
        self.assertEqual(metrics['protocol'], PROTOCOL)

    def test_named_auxiliaries_are_finite_without_main_or_Source_latent_branch(self):
        for kind in ('far', 'geometry'):
            with self.subTest(kind=kind):
                net = network()
                with patch.object(net.flow_renderer.codec, 'encode', wraps=net.flow_renderer.codec.encode) as encode:
                    loss, metrics = rf_codec_refinement_loss(net, source(), kind=kind, checkpoint_velocity=False)
                loss.backward()
                self.assertEqual(encode.call_count, 1)
                self.assertEqual(set(metrics['weighted_terms']), {kind + '_wall'})
                self.assertFalse(metrics['global_ssim_active'])
                self.assertFalse(metrics['source_latent_active'])
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(metrics['router_velocity_calls'], 3)
                json.dumps(metrics, allow_nan=False)

    def test_target_poison_does_not_change_forward_and_invalid_policy_fails_before_RF(self):
        net = network(); conditions = moe_conditions(net, source())
        with torch.no_grad():
            reference = rf_codec_forward(net, conditions, checkpoint_velocity=False)
            changed = dict(conditions)
            for key in ('target', 'background', 'object_mask'):
                changed[key] = torch.full_like(changed[key], float('nan'))
            actual = rf_codec_forward(net, changed, checkpoint_velocity=False)
        for before, after in zip(reference[:3], actual[:3]):
            self.assertTrue(torch.equal(before, after))
        for mutate in (
            lambda n: setattr(n.flow_renderer, 'steps', 5),
            lambda n: setattr(n.flow_renderer, 'conditional_restorer', torch.nn.Identity()),
            lambda n: n.base_student.train(),
            lambda n: next(n.flow_renderer.codec.decoder.parameters()).requires_grad_(True),
            lambda n: n.flow_renderer.codec.quantize.train(),
            lambda n: next(n.flow_renderer.geometry_opacity.parameters()).requires_grad_(True),
        ):
            candidate = network(); mutate(candidate)
            with patch.object(candidate.flow_renderer, 'latent_sample', side_effect=AssertionError('Validation must precede RF')):
                with self.assertRaises(ValueError):
                    rf_codec_refinement_loss(candidate, source())
        with self.assertRaises(ValueError):
            rf_codec_refinement_loss(net, source(), kind='unknown')
        settings = renderer_rf_codec_protocol()
        settings['weights']['global_ssim'] = 999
        self.assertEqual(renderer_rf_codec_protocol()['weights']['global_ssim'], 1.)


if __name__ == '__main__':
    unittest.main()
