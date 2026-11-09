"""Visible-object losses preserve legacy reductions and train only predictions."""
import unittest

import torch
from torch.nn import functional as F

from turkeyworld.object_losses import (npc_region_weights, object_region_weights,
    weighted_image_l1, weighted_mse, weighted_image_loss, object_image_diagnostics,
    region_error_metrics)


class ObjectLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def state(self):
        state = torch.zeros(1, 90)
        state[:, :2] = .5
        state[:, 3:6] = 1.
        slots = state[:, 8:88].reshape(1, 8, 10)
        slots[:, :, :2] = torch.tensor([.25, .5])  # All behind the player.
        slots[:, :, 4:6] = torch.tensor([.6, 1.])
        return state

    def test_exact_ones_recover_legacy_rgb_and_latent_losses_and_gradients(self):
        generator = torch.Generator().manual_seed(340)
        for size, channels in ((64, 3), (128, 3), (32, 16)):
            for helper, reference in ((weighted_image_l1, F.l1_loss), (weighted_mse, F.mse_loss)):
                with self.subTest(size=size, channels=channels, helper=helper.__name__):
                    prediction = torch.randn(2, channels, size, size, generator=generator, requires_grad=True)
                    target = torch.randn(prediction.shape, generator=generator)
                    actual = helper(prediction, target, torch.ones(2, 1, size, size))
                    expected = reference(prediction, target)
                    self.assertTrue(torch.equal(actual, expected))
                    gradient, = torch.autograd.grad(actual, prediction, retain_graph=True)
                    baseline, = torch.autograd.grad(expected, prediction)
                    self.assertTrue(torch.equal(gradient, baseline))

    def test_empty_full_and_unit_boost_masks_are_exact_uniform_weights(self):
        for mask in (torch.zeros(2, 1, 64, 64), torch.ones(2, 1, 64, 64)):
            self.assertTrue(torch.equal(object_region_weights(mask), torch.ones_like(mask)))
        mixed = torch.zeros(1, 64, 64)
        mixed[:, 10:30, 20:40] = 1
        self.assertTrue(torch.equal(object_region_weights(mixed, boost=1.), torch.ones(1, 1, 64, 64)))

    def test_foreground_is_four_times_background_and_only_prediction_receives_gradient(self):
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 2:4, 2:4] = 1
        mask.requires_grad_()
        prediction = torch.ones(1, 3, 8, 8, requires_grad=True)
        target = torch.zeros_like(prediction, requires_grad=True)
        weights = object_region_weights(mask, boost=4.)
        self.assertFalse(weights.requires_grad)
        weighted_image_l1(prediction, target, weights).backward()
        self.assertAlmostEqual(float(prediction.grad[0, 0, 2, 2]/prediction.grad[0, 0, 0, 0]), 4., places=6)
        self.assertIsNone(mask.grad)
        self.assertIsNone(target.grad)
        torch.testing.assert_close(weights.mean((-2, -1)), torch.ones(1, 1), rtol=1e-6, atol=1e-7)

    def test_rgb_weights_are_area_resized_for_latent_mse_and_channel_expansion_is_equivalent(self):
        mask = torch.zeros(1, 1, 128, 128)
        mask[:, :, 17:64, 29:99] = 1
        weights = object_region_weights(mask)
        area = F.interpolate(weights, (32, 32), mode='area')
        torch.testing.assert_close(area.mean(), torch.tensor(1.), rtol=1e-6, atol=1e-7)
        prediction = torch.linspace(0, 1, 16*32*32).reshape(1, 16, 32, 32).requires_grad_()
        target = torch.zeros_like(prediction)
        actual = weighted_mse(prediction, target, weights)
        expanded = area.expand_as(prediction)
        expected = (prediction.square()*expanded).sum()/expanded.sum()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, weighted_mse(prediction, target, expanded), rtol=0, atol=0)
        actual.backward()
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))

    def test_npc_bounds_exclude_behind_camera_include_corpses_and_detach_state(self):
        state = self.state()
        self.assertTrue(torch.equal(npc_region_weights(state, 64, 64), torch.ones(1, 1, 64, 64)))
        state[:, 8:10] = torch.tensor([.7, .5])
        state[:, 12:14] = 0.  # A corpse remains a rendered sprite.
        state.requires_grad_()
        weights = npc_region_weights(state, 128, 128)
        self.assertFalse(weights.requires_grad)
        self.assertGreater(float(weights[0, 0, 70, 64]), float(weights[0, 0, 0, 0]))
        self.assertGreater(float(weights.min()), 0)
        self.assertLessEqual(float(weights.max()), 4.)
        torch.testing.assert_close(weights.mean(), torch.tensor(1.), rtol=1e-6, atol=1e-7)

    def test_sprite_whose_center_is_outside_fov_keeps_clipped_edge_region(self):
        from turkeyworld.camera import object_camera_features
        state = self.state()
        state[:, 8:10] = torch.tensor([.5+1/16, .5+.8/16])
        self.assertFalse(bool(object_camera_features(state)['frustum'][0, 0]))
        weights = npc_region_weights(state, 64, 64)
        self.assertGreater(float(weights[0, 0, :, -1].max()), float(weights[0, 0, :, 0].max()))
        self.assertTrue(bool(torch.isfinite(weights).all()))

    def test_zero_loss_mask_is_finite_and_diagnostics_distinguish_object_error(self):
        prediction = torch.zeros(1, 3, 8, 8, requires_grad=True)
        target = torch.ones_like(prediction)
        zero = weighted_image_l1(prediction, target, torch.zeros(1, 1, 8, 8))
        self.assertEqual(float(zero.detach()), 0.)
        zero.backward()
        self.assertTrue(torch.equal(prediction.grad, torch.zeros_like(prediction)))
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 2:4, 2:4] = 1
        estimate = target.detach().clone()
        estimate[:, :, 2:4, 2:4] = 0
        diagnostics = object_image_diagnostics(estimate, target, mask)
        self.assertEqual(float(diagnostics['foreground_l1']), 1.)
        self.assertEqual(float(diagnostics['background_l1']), 0.)
        self.assertEqual(float(diagnostics['foreground_fraction']), 4/64)

    def test_invalid_masks_boosts_weights_and_shapes_raise(self):
        for mask in (torch.tensor([float('nan')]).reshape(1, 1, 1, 1),
                     torch.full((1, 1, 1, 1), -1.), torch.full((1, 1, 1, 1), 1.1)):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                object_region_weights(mask)
        for boost in (0., -1., float('inf')):
            with self.subTest(boost=boost), self.assertRaises(ValueError):
                object_region_weights(torch.zeros(1, 1, 2, 2), boost)
        with self.assertRaises(ValueError):
            weighted_mse(torch.zeros(1, 3, 8, 8), torch.zeros(1, 4, 8, 8))
        with self.assertRaises(ValueError):
            weighted_mse(torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 8, 8), torch.full((1, 1, 8, 8), -1.))

    def test_uniform_image_objective_preserves_legacy_l1_and_both_edge_losses_exactly(self):
        from turkeyworld.train import image_loss
        generator = torch.Generator().manual_seed(350)
        for size in (64, 128):
            prediction = torch.randn(1, 3, size, size, generator=generator, requires_grad=True)
            target = torch.randn(prediction.shape, generator=generator)
            baseline = image_loss(prediction, target)
            for weights in (None, torch.ones(1, 1, size, size)):
                with self.subTest(size=size, weights=weights is not None):
                    actual = weighted_image_loss(prediction, target, weights)
                    self.assertTrue(torch.equal(actual, baseline))
                    gradient, = torch.autograd.grad(actual, prediction, retain_graph=True)
                    expected, = torch.autograd.grad(baseline, prediction, retain_graph=True)
                    self.assertTrue(torch.equal(gradient, expected))

    def test_image_edges_use_average_adjacent_weights(self):
        prediction = torch.tensor([[[[0., 1., 0.], [0., 1., 0.]]]], requires_grad=True)
        target = torch.zeros_like(prediction, requires_grad=True)
        weights = torch.tensor([[[[1., 4., 1.], [1., 4., 1.]]]])
        # Base = 8/12; both horizontal differences have magnitude1, while
        # vertical differences are zero. Edge-weight mass normalizes separately.
        actual = weighted_image_loss(prediction, target, weights)
        self.assertAlmostEqual(float(actual.detach()), 8/12+.1, places=6)
        actual.backward()
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))
        self.assertIsNone(target.grad)

    def test_singleton_edges_and_empty_region_metrics_stay_finite(self):
        for height, width in ((1, 1), (1, 8), (8, 1)):
            prediction = torch.ones(1, 3, height, width, requires_grad=True)
            target = torch.zeros_like(prediction)
            loss = weighted_image_loss(prediction, target, torch.ones(1, 1, height, width))
            self.assertEqual(float(loss.detach()), 1.)
            loss.backward()
            self.assertTrue(bool(torch.isfinite(prediction.grad).all()))
        prediction, target = torch.ones(1, 3, 8, 8), torch.zeros(1, 3, 8, 8)
        empty = region_error_metrics(prediction, target, torch.zeros(1, 1, 8, 8))
        self.assertEqual(float(empty['npc_rgb_mae']), 0.)
        self.assertEqual(float(empty['background_rgb_mae']), 1.)
        self.assertFalse(bool(empty['npc_present']))
        self.assertTrue(bool(empty['background_present']))
        full = region_error_metrics(prediction, target, torch.ones(1, 1, 8, 8))
        self.assertEqual(float(full['npc_rgb_mae']), 1.)
        self.assertEqual(float(full['background_rgb_mae']), 0.)
        self.assertFalse(bool(full['background_present']))


if __name__ == '__main__':
    unittest.main()
