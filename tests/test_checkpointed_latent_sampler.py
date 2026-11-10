"""ODE activation recomputation preserves RGB, gradients and frozen codec state."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
import copy
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class CheckpointedLatentSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def renderer(kind="teacher"):
        from turkeyworld.latent_flow_renderer import LatentFlowRenderer
        from turkeyworld.vqgan import VQGAN
        config = dict(image_size=32, base_channels=8, latent_channels=16, embedding_dim=4,
                      n_embed=32, attention=False, data_init=False, kmeans_iterations=0)
        model = LatentFlowRenderer(kind=kind, image_size=32, width=32, depth=1, heads=4,
            mode="flow_matching" if kind == "teacher" else "rectified_flow", codec_config=config)
        codec = VQGAN(config)
        model.load_codec({"config": codec.get_config(), "model": codec.state_dict()})
        with torch.no_grad():
            model.output.weight.normal_(std=.01)
            model.blocks[0].modulation[-1].bias.fill_(.1)
        return model.train()

    @staticmethod
    def inputs(requires_grad=False):
        base = torch.rand(1, 3, 32, 32, requires_grad=requires_grad)
        state = torch.zeros(1, STATE_DIM)
        state[:, :8] = torch.tensor([.2, .3, 0., 1., 1., 1., 0., 0.])
        state.requires_grad_(requires_grad)
        noise = torch.randn(1, 4, 8, 8, requires_grad=requires_grad)
        return base, state, noise

    def assert_gradients_match(self, plain, checked):
        gradients = dict(checked.named_parameters())
        for name, parameter in plain.named_parameters():
            other = gradients[name]
            if name.startswith("codec."):
                self.assertFalse(parameter.requires_grad)
                self.assertFalse(other.requires_grad)
                self.assertIsNone(parameter.grad)
                self.assertIsNone(other.grad)
            elif parameter.grad is None:
                self.assertIsNone(other.grad, name)
            else:
                self.assertIsNotNone(other.grad, name)
                torch.testing.assert_close(parameter.grad, other.grad, rtol=2e-5, atol=2e-7, msg=name)
        self.assertGreater(float(checked.output.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(checked.blocks[0].attention.in_proj_weight.grad.abs().sum()), 0.)

    def test_full_12_step_heun_rgb_and_all_gradients_match_without_codec_updates(self):
        from turkeyworld.latent_flow_renderer import checkpoint
        torch.manual_seed(620)
        plain = self.renderer()
        checked = copy.deepcopy(plain)
        codec_before = {name: value.clone() for name, value in checked.codec.state_dict().items()}
        base, state, noise = self.inputs(requires_grad=True)
        checked_inputs = [value.detach().clone().requires_grad_(True) for value in (base, state, noise)]
        expected = plain.sample(base, state, noise=noise)
        with patch("turkeyworld.latent_flow_renderer.checkpoint", wraps=checkpoint) as checkpoint_call:
            actual = checked.sample(*checked_inputs[:2], noise=checked_inputs[2], checkpoint_velocity=True)
            self.assertEqual(checkpoint_call.call_count, 24)
            first = checkpoint_call.call_args_list[0]
            self.assertIsNone(first.args[-1])
            self.assertFalse(first.kwargs["use_reentrant"])
            self.assertTrue(first.kwargs["preserve_rng_state"])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            expected.square().mean().backward()
            actual.square().mean().backward()
        self.assert_gradients_match(plain, checked)
        for value, other in zip((base, state, noise), checked_inputs):
            self.assertGreater(float(value.grad.abs().sum()), 0.)
            torch.testing.assert_close(value.grad, other.grad, rtol=2e-5, atol=2e-7)
        self.assertFalse(checked.codec.training)
        for name, value in checked.codec.state_dict().items():
            torch.testing.assert_close(value, codec_before[name], rtol=0, atol=0)

    def test_euler_with_detached_conditions_still_trains_flow_parameters(self):
        from turkeyworld.latent_flow_renderer import checkpoint
        torch.manual_seed(621)
        plain = self.renderer("student")
        checked = copy.deepcopy(plain)
        base, state, noise = self.inputs()
        expected = plain.sample(base, state, noise=noise, steps=3)
        with patch("turkeyworld.latent_flow_renderer.checkpoint", wraps=checkpoint) as checkpoint_call:
            actual = checked.sample(base, state, noise=noise, steps=3, checkpoint_velocity=True)
            self.assertEqual(checkpoint_call.call_count, 3)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            expected.mean().backward()
            actual.mean().backward()
        self.assert_gradients_match(plain, checked)
        self.assertIsNone(base.grad)
        self.assertIsNone(state.grad)

    def test_eval_no_grad_inference_and_default_flag_bypass_checkpoint(self):
        model = self.renderer()
        base, state, noise = self.inputs()
        with torch.no_grad():
            expected = model.sample(base, state, noise=noise, steps=2)
        with patch("turkeyworld.latent_flow_renderer.checkpoint", side_effect=AssertionError("Unexpected recomputation")):
            ordinary = model.sample(base, state, noise=noise, steps=2)
            model.eval()
            evaluated = model.sample(base, state, noise=noise, steps=2, checkpoint_velocity=True)
            model.train()
            with torch.no_grad():
                no_grad = model.sample(base, state, noise=noise, steps=2, checkpoint_velocity=True)
            with torch.inference_mode():
                inference = model.sample(base, state, noise=noise, steps=2, checkpoint_velocity=True)
        for actual in (ordinary, evaluated, no_grad, inference):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_checkpointed_heun_keeps_the_final_endpoint_time_and_constant_field_integral(self):
        model = self.renderer()
        base, state, _ = self.inputs()
        initial = torch.full((1, 4, 8, 8), -.3)
        times = []

        def velocity(noisy, time, condition, labels, self_condition=None):
            times.append(float(time[0]))
            return torch.full_like(noisy, .4)

        with patch.object(model, "velocity", side_effect=velocity):
            result = model.latent_sample(base, state, noise=initial, checkpoint_velocity=True)
        self.assertEqual(len(times), 24)
        self.assertAlmostEqual(times[0], 0.)
        self.assertAlmostEqual(times[-1], 1.)
        torch.testing.assert_close(result, torch.full_like(initial, .1), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(initial, torch.full_like(initial, -.3), rtol=0, atol=0)

    def test_stochastic_velocity_recomputation_preserves_rng_and_gradients(self):
        from torch.nn import functional as F
        plain = self.renderer()
        checked = copy.deepcopy(plain)
        base, state, noise = self.inputs()

        def stochastic(model):
            def velocity(noisy, time, condition, labels, self_condition=None):
                return F.dropout(noisy, p=.3, training=True)*model.output.weight.mean()+.1*condition
            return velocity

        torch.manual_seed(152)
        with patch.object(plain, "velocity", side_effect=stochastic(plain)):
            expected = plain.latent_sample(base, state, noise=noise, steps=2)
            expected.square().mean().backward()
        expected_rng = torch.random.get_rng_state()
        torch.manual_seed(152)
        with patch.object(checked, "velocity", side_effect=stochastic(checked)):
            actual = checked.latent_sample(base, state, noise=noise, steps=2, checkpoint_velocity=True)
            actual.square().mean().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(checked.output.weight.grad, plain.output.weight.grad, rtol=1e-6, atol=1e-8)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), expected_rng))


if __name__ == "__main__":
    unittest.main()
