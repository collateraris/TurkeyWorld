"""CPU checks of actual VQGAN gradient balancing and separate GAN updates."""
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class VQGANLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        from turkeyworld.vqgan_losses import (NLayerDiscriminator, VQGANObjective, FrozenLPIPSVGG,
            adaptive_discriminator_weight, discriminator_factor, generator_hinge_loss,
            hinge_discriminator_loss)
        cls.Discriminator = NLayerDiscriminator
        cls.Objective = VQGANObjective
        cls.LPIPS = FrozenLPIPSVGG
        cls.adaptive = staticmethod(adaptive_discriminator_weight)
        cls.factor = staticmethod(discriminator_factor)
        cls.g_hinge = staticmethod(generator_hinge_loss)
        cls.d_hinge = staticmethod(hinge_discriminator_loss)

    def setUp(self):
        torch.manual_seed(98)

    def test_rgb_patchgan_random_weights_shape_gradients_and_exact_roundtrip(self):
        discriminator = self.Discriminator(ndf=4, n_layers=3)
        rgb = torch.rand(2, 3, 64, 64, requires_grad=True)
        scores = discriminator(rgb)
        self.assertEqual(tuple(scores.shape), (2, 1, 6, 6))
        scores.square().mean().backward()
        self.assertGreater(float(rgb.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in discriminator.parameters()))
        self.assertTrue(any(isinstance(layer, torch.nn.GroupNorm) for layer in discriminator.modules()))
        self.assertFalse(any(isinstance(layer, (torch.nn.BatchNorm2d, torch.nn.Sigmoid))
                             for layer in discriminator.modules()))
        self.assertFalse(any(hasattr(layer, "weight_orig") for layer in discriminator.modules()))
        self.assertFalse(discriminator.metadata["pretrained"])
        another = self.Discriminator(ndf=4, n_layers=3)
        self.assertFalse(torch.equal(discriminator.main[0].weight, another.main[0].weight))
        stream = io.BytesIO()
        torch.save(discriminator.state_dict(), stream)
        stream.seek(0)
        another.load_state_dict(torch.load(stream, weights_only=True))
        with torch.no_grad():
            torch.testing.assert_close(discriminator(rgb), another(rgb), rtol=0., atol=0.)

    def test_hinge_signs_use_vqgan_half_scale_and_fp32_accumulation(self):
        self.assertEqual(float(self.d_hinge(torch.tensor([2.]), torch.tensor([-2.]))), 0.)
        self.assertEqual(float(self.d_hinge(torch.tensor([-2.]), torch.tensor([2.]))), 3.)
        self.assertEqual(float(self.g_hinge(torch.tensor([2.]))), -2.)
        fake = torch.tensor([.5], dtype=torch.bfloat16, requires_grad=True)
        loss = self.g_hinge(fake)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertEqual(float(fake.grad), -1.)

    def test_reconstruction_metric_values_edges_and_frozen_perceptual_gradients(self):
        class Perceptual(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(2.))
                self.dropout = torch.nn.Dropout(.9)

            def forward(self, real, reconstruction):
                return self.dropout((real - reconstruction).abs()).mean() * self.scale

        perceptual = Perceptual()
        objective = self.Objective(perceptual=perceptual).train()
        self.assertFalse(perceptual.training)
        self.assertFalse(perceptual.scale.requires_grad)
        real = torch.zeros(1, 3, 2, 2, requires_grad=True)
        reconstruction = torch.tensor([[[[0., 1.], [0., 1.]]]]).repeat(1, 3, 1, 1).requires_grad_()
        loss, metrics = objective.reconstruction_loss(real, reconstruction)
        self.assertAlmostEqual(metrics["pixel_l1"], .5)
        self.assertAlmostEqual(metrics["perceptual_loss"], 1.)
        self.assertAlmostEqual(metrics["edge_l1"], .5)
        self.assertAlmostEqual(float(loss.detach()), 1.55, places=6)
        self.assertAlmostEqual(metrics["reconstruction_loss"], 1.55, places=6)
        self.assertTrue(metrics["perceptual_enabled"])
        loss.backward()
        self.assertIsNone(real.grad)
        self.assertIsNone(perceptual.scale.grad)
        self.assertGreater(float(reconstruction.grad.abs().sum()), 0.)
        self.assertTrue(objective.metadata["perceptual"]["training_loss_only"])

    def test_no_perceptual_loss_is_explicit_and_one_pixel_edges_are_finite(self):
        objective = self.Objective(perceptual=None)
        real, reconstruction = torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1, requires_grad=True)
        loss, metrics = objective.reconstruction_loss(real, reconstruction)
        self.assertEqual(float(loss.detach()), 1.)
        self.assertEqual(metrics["edge_l1"], 0.)
        self.assertFalse(metrics["perceptual_enabled"])
        self.assertIsNone(objective.metadata["perceptual"])
        loss.backward()
        self.assertTrue(torch.isfinite(reconstruction.grad).all())

    def test_optional_lpips_vgg_normalization_frozen_eval_and_loss_only_metadata(self):
        calls = []

        class StubLPIPS(torch.nn.Module):
            def __init__(self, **options):
                super().__init__()
                calls.append(options)
                self.weight = torch.nn.Parameter(torch.ones(()))

            def forward(self, real, reconstruction):
                calls.append((real.detach().clone(), reconstruction.detach().clone()))
                return (real - reconstruction).square().mean((1, 2, 3), keepdim=True) * self.weight

        with patch.dict("sys.modules", {"lpips": SimpleNamespace(LPIPS=StubLPIPS)}):
            perceptual = self.LPIPS().train()
        self.assertEqual(calls[0], {"net": "vgg", "pretrained": True,
                                    "pnet_rand": False, "eval_mode": True})
        self.assertFalse(perceptual.training)
        self.assertFalse(perceptual.network.training)
        real = torch.zeros(1, 3, 4, 4, requires_grad=True)
        reconstruction = torch.ones_like(real, requires_grad=True)
        value = perceptual(real, reconstruction)
        torch.testing.assert_close(calls[1][0], -torch.ones_like(real))
        torch.testing.assert_close(calls[1][1], torch.ones_like(real))
        value.mean().backward()
        self.assertIsNone(real.grad)
        self.assertIsNone(perceptual.network.weight.grad)
        self.assertGreater(float(reconstruction.grad.abs().sum()), 0.)
        self.assertTrue(perceptual.metadata["pretrained"])
        self.assertTrue(perceptual.metadata["training_loss_only"])

    def test_adaptive_weight_matches_actual_last_layer_gradient_ratio_and_keeps_graph(self):
        layer = torch.nn.Parameter(torch.tensor([1., 2.]))
        rec, gan = layer.square().sum(), (layer * 3.).sum()
        weight = self.adaptive(rec, gan, layer)
        expected = torch.tensor([2., 4.]).norm() / (torch.tensor([3., 3.]).norm() + 1e-4)
        torch.testing.assert_close(weight, expected)
        self.assertFalse(weight.requires_grad)
        self.assertEqual(weight.dtype, torch.float32)
        self.assertIsNone(layer.grad)
        (rec + weight * gan).backward()
        torch.testing.assert_close(layer.grad, torch.tensor([2., 4.]) + weight * 3.)

    def test_adaptive_weight_bounds_zero_gan_gradient_without_hiding_disconnected_losses(self):
        layer = torch.nn.Parameter(torch.ones(2))
        ratio = self.adaptive((layer * 1e6).sum(), (layer * 0.).sum(), layer)
        self.assertEqual(float(ratio), 1e4)
        other = torch.nn.Parameter(torch.ones(2))
        with self.assertRaises(RuntimeError):
            self.adaptive(layer.sum(), other.sum(), layer)

    def test_discriminator_update_and_generator_backward_are_isolated_with_real_decoder_gradients(self):
        generator = torch.nn.Conv2d(3, 3, 1)
        discriminator = self.Discriminator(ndf=4)
        optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
        real = torch.rand(1, 3, 32, 32, requires_grad=True)
        reconstruction = generator(torch.rand_like(real)).sigmoid()
        before = discriminator.main[0].weight.detach().clone()
        d_loss = self.d_hinge(discriminator(real.detach()), discriminator(reconstruction.detach()))
        d_loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.assertFalse(torch.equal(before, discriminator.main[0].weight))
        self.assertTrue(all(parameter.grad is None for parameter in generator.parameters()))
        self.assertIsNone(real.grad)
        discriminator.eval().requires_grad_(False)
        rec_loss, _ = self.Objective().reconstruction_loss(real, reconstruction)
        gan_loss = self.g_hinge(discriminator(reconstruction))
        adaptive = self.adaptive(rec_loss, gan_loss, generator.weight)
        self.assertGreater(float(adaptive), 0.)
        (rec_loss + adaptive * gan_loss).backward()
        self.assertGreater(float(generator.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in discriminator.parameters()))
        self.assertIsNone(real.grad)

    def test_gan_warmup_is_delayed_and_invalid_shapes_or_weights_fail(self):
        self.assertEqual(self.factor(999), 0.)
        self.assertEqual(self.factor(1000), 1.)
        objective = self.Objective(disc_start=500)
        self.assertEqual(objective.discriminator_factor(499, .2), 0.)
        self.assertEqual(objective.discriminator_factor(500, .2), .2)
        with self.assertRaises(ValueError):
            self.Discriminator(input_nc=6)
        with self.assertRaises(ValueError):
            self.Discriminator(ndf=4)(torch.zeros(1, 3, 16, 16))
        with self.assertRaises(ValueError):
            self.Objective(edge_weight=-1)
        with self.assertRaises(ValueError):
            self.Objective().reconstruction_loss(torch.zeros(1, 3, 4, 4), torch.zeros(1, 3, 5, 4))


if __name__ == "__main__":
    unittest.main()
