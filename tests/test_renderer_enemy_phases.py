import unittest
from unittest.mock import patch

import torch

from turkeyworld.object_losses import (balanced_edge_l1, balanced_image_l1, decoded_ssim_loss,
                                      foreground_contrastive_loss)
from turkeyworld.renderer_enemy_objectives import enemy_conditions, enemy_refinement_loss
from turkeyworld.renderer_refinement import configure_refinement, frozen_fingerprint, refinement_forward
from test_renderer_enemy_objectives import tiny_model, batch


class RendererEnemyPhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_default_joint_objective_is_bitexact_to_original_sum_and_explicit_false(self):
        model = tiny_model()
        source = batch((False, True))
        conditions = enemy_conditions(model, source)
        base, rgb, _ = refinement_forward(model, conditions, detach_base=True)
        expected = sum(balanced_image_l1(prediction, conditions['target'], conditions['object_mask'])
            +.2*decoded_ssim_loss(prediction, conditions['target'], conditions['object_mask'])
            +.1*balanced_edge_l1(prediction, conditions['target'], conditions['object_mask'])
            +.1*foreground_contrastive_loss(prediction, conditions['target'], conditions['background'], conditions['object_mask'])
            for prediction in (base, rgb))
        default, metrics = enemy_refinement_loss(model, source)
        explicit, _ = enemy_refinement_loss(model, source, base_only=False)
        self.assertTrue(torch.equal(expected, default))
        self.assertTrue(torch.equal(default, explicit))
        self.assertFalse(metrics['base_only'])
        self.assertTrue(metrics['flow_forward_active'])
        self.assertEqual(metrics['solver_steps'], 1)
        self.assertFalse(metrics['pixel_residual_active'])

    def test_base_only_never_calls_flow_or_codec_and_only_base_receives_gradients(self):
        model = tiny_model()
        source = batch((False,))
        frozen = frozen_fingerprint(model)
        with patch.object(model.flow_renderer, 'latent_sample', side_effect=AssertionError('No Flow in Base curriculum')), \
                patch.object(model.flow_renderer.codec, 'encode', side_effect=AssertionError('No codec encoder')), \
                patch.object(model.flow_renderer.codec, 'decode', side_effect=AssertionError('No codec decoder')), \
                patch.object(model.flow_renderer.codec.quantize, 'forward', side_effect=AssertionError('No quantizer')):
            loss, metrics = enemy_refinement_loss(model, source, base_only=True)
            loss.backward()
        self.assertGreater(float(model.base_student.rgb.weight.grad.abs().sum()), 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.flow_renderer.parameters()))
        self.assertEqual(frozen, frozen_fingerprint(model))
        self.assertTrue(metrics['base_only'])
        self.assertFalse(metrics['flow_forward_active'])
        self.assertEqual(metrics['solver_steps'], 0)
        self.assertNotIn('rf1_source_l1', metrics)
        self.assertNotIn('pixel_residual_abs_mean', metrics)
        with self.assertRaisesRegex(ValueError, 'explicit boolean'):
            enemy_refinement_loss(model, source, base_only=1)

    def test_enabled_postcodec_residual_gets_joint_rgb_gradient_but_is_idle_in_base_only(self):
        from turkeyworld.base_student import enable_base_pixel_residual
        model = tiny_model()
        previous = model.base_student
        model.base_student = enable_base_pixel_residual(previous)
        configure_refinement(model)
        source = batch((False, True))
        codec = {key: value.clone() for key, value in model.flow_renderer.codec.state_dict().items()}
        loss, metrics = enemy_refinement_loss(model, source)
        loss.backward()
        self.assertGreater(float(model.base_student.pixel_residual.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.base_student.rgb.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.flow_renderer.output.weight.grad.abs().sum()), 0)
        self.assertTrue(metrics['pixel_residual_active'])
        self.assertEqual(metrics['pixel_residual_abs_mean'], 0)
        self.assertTrue(all(value.grad is None for value in model.flow_renderer.codec.parameters()))
        self.assertTrue(all(torch.equal(value, model.flow_renderer.codec.state_dict()[key]) for key, value in codec.items()))
        model.zero_grad(set_to_none=True)
        with patch.object(model.flow_renderer, 'latent_sample', side_effect=AssertionError('No Flow in Base-only phase')):
            base_loss, base_metrics = enemy_refinement_loss(model, source, base_only=True)
            base_loss.backward()
        self.assertTrue(all(value.grad is None for value in model.base_student.pixel_residual.parameters()))
        self.assertTrue(all(value.grad is None for value in model.flow_renderer.parameters()))
        self.assertFalse(base_metrics['pixel_residual_active'])
        self.assertEqual(base_metrics['solver_steps'], 0)


if __name__ == '__main__':
    unittest.main()
