"""Base-only Source losses and frozen-encoder INPUT gradients on real modules."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import torch

from test_renderer_rf5_objectives import model as tiny_renderer, source
from turkeyworld.base_expanded_experts import upgrade_base_experts
from turkeyworld.base_geometry_phase import configure_base_geometry_training, base_geometry_frozen_fingerprint
from turkeyworld.base_geometry_objectives import base_geometry_forward, base_geometry_refinement_loss
from turkeyworld.renderer_moe_objectives import moe_conditions
from turkeyworld.renderer_moe import RendererTokenMoE, FIVE_SCHEMA
from turkeyworld.renderer_rf5_experts import upgrade_rf5_experts


def network():
    result = tiny_renderer(3)
    result.flow_renderer.output_moe = RendererTokenMoE(256, 64, 32, 4, role_schema=FIVE_SCHEMA, experts=5)
    upgrade_rf5_experts(result.flow_renderer.output_moe)
    upgrade_base_experts(result.base_student)
    configure_base_geometry_training(result)
    return result


class BaseGeometryObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_main_never_calls_RF_and_frozen_E_Q_input_gradient_reaches_Base(self):
        model, data = network(), source()
        for key in ('state', 'target', 'background', 'object_mask'):
            data[key] = data[key].detach().requires_grad_()
        fingerprint = base_geometry_frozen_fingerprint(model)
        encoded = []
        original = model.flow_renderer.codec.encode
        def encode(rgb):
            result = original(rgb)
            if result['quantized'].requires_grad: result['quantized'].retain_grad()
            encoded.append((rgb, result['quantized']))
            return result
        with (patch.object(model.flow_renderer, 'velocity', side_effect=AssertionError('No RF')),
             patch.object(model.flow_renderer, 'latent_sample', side_effect=AssertionError('No RF')),
             patch.object(model.flow_renderer.codec, 'encode', side_effect=encode)):
            loss, metrics = base_geometry_refinement_loss(model, data)
            loss.backward()
        self.assertEqual(len(encoded), 2)
        self.assertTrue(encoded[0][1].requires_grad); self.assertFalse(encoded[1][1].requires_grad)
        self.assertGreater(float(encoded[0][1].grad.abs().sum()), 0)
        self.assertAlmostEqual(float(loss.detach()), sum(metrics['weighted_terms'].values())+
            .01*metrics['renderer_moe_role_loss']+.001*metrics['renderer_moe_balance_loss'], places=6)
        self.assertEqual(set(metrics['renderer_moe_heads']), {'base'})
        self.assertFalse(metrics['RF_forward_active']); self.assertTrue(metrics['frozen_encoder_input_gradient_allowed'])
        for key in ('state','target','background','object_mask'): self.assertIsNone(data[key].grad, key)
        for name,p in model.named_parameters():
            if not name.startswith('base_student.'): self.assertIsNone(p.grad, name)
        self.assertEqual(fingerprint, base_geometry_frozen_fingerprint(model))
        json.dumps(metrics, allow_nan=False)

    def test_actual_full_image_SSIM_formula_and_far_geometry_no_latent_or_RF_call(self):
        from turkeyworld.object_losses import decoded_ssim_loss
        for kind in ('main','far','geometry'):
            model = network()
            with (patch.object(model.flow_renderer.codec, 'encode', wraps=model.flow_renderer.codec.encode) as enc,
                 patch('turkeyworld.base_geometry_objectives.decoded_ssim_loss', wraps=decoded_ssim_loss) as ssim,
                 patch.object(model.flow_renderer, 'velocity', side_effect=AssertionError('No RF'))):
                loss, metrics = base_geometry_refinement_loss(model, source(), kind=kind)
                loss.backward()
            self.assertEqual(enc.call_count, 2 if kind=='main' else 0)
            self.assertEqual(ssim.call_count, 1 if kind=='main' else 0)
            if kind=='main':
                self.assertIsNone(ssim.call_args.kwargs['mask'])
                self.assertEqual(metrics['weighted_terms']['global_ssim'], metrics['global_ssim_loss'])
                self.assertAlmostEqual(metrics['weighted_terms']['region_l1'], .5*metrics['region_l1'])
                self.assertAlmostEqual(metrics['weighted_terms']['source_latent'], .1*metrics['source_latent_mse'])
            else: self.assertEqual(set(metrics['weighted_terms']), {'far_wall' if kind=='far' else 'geometry_wall'})

    def test_forward_ignores_poisoned_Source_labels_and_frozen_policy_rejects_leaks(self):
        model = network(); conditions = moe_conditions(model, source())
        with torch.no_grad(): expected, _ = base_geometry_forward(model, conditions)
        for key in ('target','background','object_mask'):
            conditions[key] = torch.full_like(conditions[key], float('nan'))
        with torch.no_grad(): actual, _ = base_geometry_forward(model, conditions)
        self.assertTrue(torch.equal(expected, actual))
        for mutate in (lambda m:m.flow_renderer.train(), lambda m:m.flow_renderer.codec.quantize.train(),
                       lambda m:next(m.flow_renderer.codec.encoder.parameters()).requires_grad_(True)):
            other = deepcopy(model); mutate(other)
            with self.assertRaisesRegex(ValueError, 'frozen eval'): base_geometry_refinement_loss(other, source())
        with self.assertRaises(ValueError): base_geometry_refinement_loss(model, source(), kind='bad')


if __name__ == '__main__': unittest.main()
