"""Conditional PatchGAN, hinge objectives, feature matching and lazy-R1 checks."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
import io
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class AdversarialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        from turkeyworld.adversarial import (MultiScaleDiscriminator, discriminator_hinge_loss,
                                            generator_hinge_loss, feature_matching_loss, r1_penalty)
        cls.Discriminator = MultiScaleDiscriminator
        cls.d_hinge = staticmethod(discriminator_hinge_loss)
        cls.g_hinge = staticmethod(generator_hinge_loss)
        cls.match = staticmethod(feature_matching_loss)
        cls.r1 = staticmethod(r1_penalty)

    def setUp(self):
        torch.manual_seed(392)

    def test_discriminator_shapes_backprop_and_state_roundtrip(self):
        discriminator = self.Discriminator(base_channels=8)
        real = torch.rand(2, 3, 64, 64)
        fake = torch.rand_like(real)
        condition = torch.rand(2, 3, 32, 32, requires_grad=True)
        state = torch.zeros(2, STATE_DIM, requires_grad=True)
        real_output, fake_output = discriminator(real, condition, state), discriminator(fake, condition, state)
        self.assertEqual(len(real_output["logits"]), 2)
        self.assertEqual([len(scale) for scale in real_output["features"]], [4, 4])
        self.assertEqual(tuple(real_output["logits"][0].shape), (2, 1, 6, 6))
        self.assertEqual(tuple(real_output["logits"][1].shape), (2, 1, 2, 2))
        loss = self.d_hinge(real_output, fake_output)
        loss.backward()
        gradients = [parameter.grad for parameter in discriminator.parameters() if parameter.grad is not None]
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in gradients))
        self.assertTrue(any(float(value.abs().sum()) > 0 for value in gradients))
        self.assertIsNone(condition.grad)
        self.assertIsNone(state.grad)
        self.assertFalse(any(isinstance(module, (torch.nn.BatchNorm2d, torch.nn.Sigmoid))
                             for module in discriminator.modules()))
        self.assertTrue(all(hasattr(module, "weight_orig") for module in discriminator.modules()
                            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))))
        discriminator.eval()
        with torch.no_grad():
            expected = discriminator(real, condition, state)
        buffer = io.BytesIO()
        torch.save(discriminator.state_dict(), buffer)
        buffer.seek(0)
        restored = self.Discriminator(base_channels=8).eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        with torch.no_grad():
            actual = restored(real, condition, state)
        for first, second in zip(expected["logits"], actual["logits"]):
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_hinge_signs_and_generator_gradient(self):
        self.assertEqual(float(self.d_hinge(torch.tensor([2.]), torch.tensor([-2.]))), 0.)
        self.assertEqual(float(self.d_hinge(torch.tensor([-2.]), torch.tensor([2.]))), 6.)
        self.assertEqual(float(self.g_hinge(torch.tensor([2.]))), -2.)
        fake = torch.tensor([-2., 0., 2.], requires_grad=True)
        self.g_hinge(fake).backward()
        self.assertTrue(bool((fake.grad < 0).all()))

    def test_state_projection_uses_visual_state_and_masks_hud_counters(self):
        discriminator = self.Discriminator(base_channels=8)
        image, condition = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)
        state = torch.zeros(1, STATE_DIM)
        state[:, 3:5] = 1
        # Initialize the spectral power-iteration buffers before fixed eval.
        discriminator(image, condition, state)
        discriminator.eval()
        hud = state.clone()
        hud[:, 5:7], hud[:, -2:] = .8, 25.
        visual = state.clone()
        visual[:, 0] = .7
        with torch.no_grad():
            original = discriminator(image, condition, state)["logits"]
            hidden_change = discriminator(image, condition, hud)["logits"]
            visible_change = discriminator(image, condition, visual)["logits"]
        for first, second in zip(original, hidden_change):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertTrue(any(float((first - second).abs().sum()) > 0
                            for first, second in zip(original, visible_change)))

    def test_feature_matching_detaches_source_features(self):
        real = torch.ones(2, 4, 3, 3, requires_grad=True)
        fake = torch.zeros_like(real, requires_grad=True)
        loss = self.match({"features": [[real]]}, {"features": [[fake]]})
        self.assertEqual(float(loss.detach()), 1.)
        loss.backward()
        self.assertIsNone(real.grad)
        self.assertGreater(float(fake.grad.abs().sum()), 0.)

    def test_frozen_eval_discriminator_preserves_candidate_rgb_gradient(self):
        discriminator = self.Discriminator(base_channels=8).eval().requires_grad_(False)
        fake = torch.rand(1, 3, 64, 64, requires_grad=True)
        condition = torch.rand_like(fake, requires_grad=True)
        state = torch.zeros(1, STATE_DIM)
        before = {name: value.clone() for name, value in discriminator.named_buffers()}
        self.g_hinge(discriminator(fake, condition, state)).backward()
        self.assertGreater(float(fake.grad.abs().sum()), 0.)
        self.assertIsNone(condition.grad)
        self.assertTrue(all(parameter.grad is None for parameter in discriminator.parameters()))
        for name, value in discriminator.named_buffers():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_lazy_r1_is_finite_and_backpropagates_in_float32(self):
        discriminator = self.Discriminator(base_channels=8)
        real = torch.rand(1, 3, 64, 64, requires_grad=True)
        condition = torch.rand_like(real)
        state = torch.zeros(1, STATE_DIM)
        state[:, 3:5] = 1
        with torch.autocast("cpu", enabled=False):
            output = discriminator(real, condition, state)
            penalty = self.r1(output["logits"], real)
        self.assertEqual(penalty.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(penalty)))
        self.assertGreaterEqual(float(penalty.detach()), 0.)
        penalty.backward()
        gradients = [parameter.grad for parameter in discriminator.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
