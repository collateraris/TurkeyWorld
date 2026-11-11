"""Independent checks for the opt-in Source Base/RF preservation objective."""
import copy
import unittest
from unittest.mock import patch

import torch

from test_renderer_enemy_objectives import tiny_model, batch
from turkeyworld.object_losses import balanced_image_l1
from turkeyworld.renderer_enemy_objectives import balanced_base_regression_loss, enemy_refinement_loss
from turkeyworld.renderer_refinement import frozen_fingerprint


class RendererBaseGuardAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_regions_and_examples_cannot_compensate_and_labels_have_no_gradient(self):
        target = torch.zeros(2, 1, 2, 2, requires_grad=True)
        mask = torch.zeros(2, 1, 2, 2)
        mask[:, :, 0, 0] = 1
        mask.requires_grad_(True)
        reference = torch.full((2, 1, 2, 2), .8)
        reference[0, :, 0, 0] = .2
        reference.requires_grad_(True)
        predicted = torch.zeros_like(reference)
        predicted[0] = .2
        predicted[0, :, 0, 0] = .8
        predicted.requires_grad_(True)
        # Image0 improves overall MAE, but makes its foreground worse.
        self.assertLess(float(predicted[0].detach().mean()), float(reference[0].detach().mean()))
        loss = balanced_base_regression_loss(predicted, target, reference, mask)
        self.assertAlmostEqual(float(loss.detach()), .15, places=6)
        loss.backward()
        self.assertGreater(float(predicted.grad[0, :, 0, 0].sum()), 0)
        self.assertEqual(float(predicted.grad[0, :, 1:, :].abs().sum()), 0)
        self.assertEqual(float(predicted.grad[1].abs().sum()), 0)
        self.assertIsNone(target.grad)
        self.assertIsNone(reference.grad)
        self.assertIsNone(mask.grad)

    def test_guard_is_area_normalized_and_empty_or_full_foreground_is_finite(self):
        for side, area in ((2, 1), (8, 16)):
            mask = torch.zeros(1, 1, side, side)
            mask.flatten()[:area] = 1
            reference = .8-.6*mask
            predicted = .2+.6*mask
            loss = balanced_base_regression_loss(predicted, torch.zeros_like(predicted), reference, mask)
            self.assertAlmostEqual(float(loss), .3, places=6)
        for occupancy in (0., 1.):
            prediction = torch.full((1, 3, 4, 4), .5, requires_grad=True)
            target, base = torch.zeros_like(prediction), torch.full_like(prediction, .2)
            mask = torch.full((1, 1, 4, 4), occupancy)
            loss = balanced_base_regression_loss(prediction, target, base, mask)
            self.assertAlmostEqual(float(loss.detach()), .3, places=6)
            loss.backward()
            self.assertTrue(bool(torch.isfinite(prediction.grad).all()))
            self.assertGreater(float(prediction.grad.abs().sum()), 0)

    def test_actual_rf_only_path_reaches_base_only_with_optin_and_codec_is_bitexact(self):
        original = tiny_model()
        # A trained nonzero velocity output is needed to depend on its condition.
        with torch.no_grad():
            weight = original.flow_renderer.output.weight
            weight.copy_(torch.arange(weight.numel()).reshape_as(weight).sin()*.01)
        state_dict = copy.deepcopy(original.state_dict())
        source = batch((False, False))
        for name, value in source.items():
            if value.is_floating_point():
                source[name] = value.requires_grad_(True)
        outputs = []
        for coupled in (False, True):
            model = tiny_model()
            model.load_state_dict(state_dict)
            before_codec = {name: value.clone() for name, value in model.flow_renderer.codec.state_dict().items()}
            before_frozen = frozen_fingerprint(model)
            base_before = model.base_student.rgb.weight.detach().clone()
            calls = 0

            def rf_only_l1(prediction, target, mask):
                nonlocal calls
                calls += 1
                # Isolate final-RGB and guard/latent gradients from direct Base supervision.
                return prediction.sum()*0 if calls == 1 else balanced_image_l1(prediction, target, mask)

            with patch('turkeyworld.renderer_enemy_objectives.balanced_image_l1', side_effect=rf_only_l1), \
                    patch.object(model, 'transition', side_effect=AssertionError('Static Source labels only')), \
                    patch('turkeyworld.engine.Game.render', side_effect=AssertionError('No engine in training')), \
                    patch.object(model.flow_renderer, 'velocity', wraps=model.flow_renderer.velocity) as velocity:
                loss, metrics = enemy_refinement_loss(model, source, ssim_weight=0., edge_weight=0.,
                    contrast_weight=0., latent_weight=.03, detach_base=not coupled, base_regression_weight=1.)
                self.assertEqual(velocity.call_count, 1)
                loss.backward()
            base_gradient = model.base_student.rgb.weight.grad
            base_norm = 0. if base_gradient is None else float(base_gradient.abs().sum())
            if coupled:
                self.assertGreater(base_norm, 0.)
            else:
                self.assertEqual(base_norm, 0.)
            self.assertGreater(float(model.flow_renderer.output.weight.grad.abs().sum()), 0.)
            optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
            optimizer.step()
            if coupled:
                self.assertFalse(torch.equal(base_before, model.base_student.rgb.weight))
            else:
                self.assertTrue(torch.equal(base_before, model.base_student.rgb.weight))
            self.assertEqual(before_frozen, frozen_fingerprint(model))
            self.assertTrue(all(torch.equal(value, model.flow_renderer.codec.state_dict()[name])
                                for name, value in before_codec.items()))
            self.assertTrue(all(p.grad is None for p in model.flow_renderer.codec.parameters()))
            self.assertFalse(model.flow_renderer.codec.training)
            self.assertFalse(model.flow_renderer.codec.quantize.training)
            self.assertFalse(metrics['future_source_rgb_in_prior'])
            self.assertEqual(metrics['flow_base_gradient_active'], coupled)
            self.assertTrue(metrics['base_regression_reference_detached'])
            outputs.append(metrics['common_rgb_objective'])
        self.assertEqual(outputs[0], outputs[1])  # detach controls gradients, not forward RGB.
        self.assertTrue(all(value.grad is None for value in source.values() if value.is_floating_point()))


if __name__ == '__main__':
    unittest.main()
