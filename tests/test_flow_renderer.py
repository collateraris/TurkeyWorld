"""Flow objectives, actual Transformer gradients, ODE sampling and compatibility."""
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class ConditionalFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        from turkeyworld.flow_renderer import ConditionalFlowRenderer
        from turkeyworld.models import NeuralWorldModel, WorldModelConfig, load_world_checkpoint
        cls.Renderer = ConditionalFlowRenderer
        cls.Model, cls.Config = NeuralWorldModel, WorldModelConfig
        cls.load_checkpoint = staticmethod(load_world_checkpoint)

    def setUp(self):
        torch.manual_seed(131)
        self.state = torch.zeros(1, 90)
        self.state[:, :8] = torch.tensor([.2, .3, 0., 1., 1., 1., 0., 0.])
        self.base = torch.rand(1, 3, 128, 128)

    def test_matching_loss_backpropagates_through_flow_blocks_after_zero_initialization(self):
        for kind, mode in (("teacher", "flow_matching"), ("student", "rectified_flow")):
            with self.subTest(kind=kind):
                renderer = self.Renderer(kind=kind, image_size=128, mode=mode, steps=2).train()
                optimizer = torch.optim.AdamW(renderer.parameters(), lr=1e-3)
                target = torch.rand_like(self.base)
                noise = torch.randn_like(target)
                for _ in range(4):
                    optimizer.zero_grad(set_to_none=True)
                    loss, metrics = renderer.loss(target, self.base, self.state,
                                                  noise=noise, time=torch.tensor([.4]))
                    self.assertTrue(bool(torch.isfinite(loss)))
                    self.assertGreater(float(loss.detach()), 0.)
                    self.assertIn("flow_velocity", metrics)
                    self.assertIn("flow_endpoint", metrics)
                    loss.backward()
                    optimizer.step()
                # AdaLN-zero gates and a zero output intentionally delay these
                # gradients; testing only the first update misses the blocks.
                block = renderer.blocks[0]
                for parameter in (block.attention.in_proj_weight, block.mlp[0].weight,
                                  block.modulation[-1].weight, renderer.state_input[0].weight):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                    self.assertGreater(float(parameter.grad.abs().sum()), 0.)

    def test_fixed_noise_is_deterministic_and_supplied_noise_changes_samples(self):
        renderer = self.Renderer(kind="student", image_size=128, steps=2).eval()
        with torch.no_grad():
            # Activate the velocity network so this exercises learned paths too.
            renderer.output.weight.normal_(std=.01)
            first = renderer.sample(self.base, self.state)
            torch.manual_seed(999)
            second = renderer.sample(self.base, self.state)
            explicit_same = renderer.sample(self.base, self.state, noise=renderer.fixed_noise.clone())
            changed = renderer.sample(self.base, self.state, noise=-renderer.fixed_noise)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(first, explicit_same, rtol=0, atol=0)
        self.assertGreater(float((first - changed).abs().mean()), .1)
        self.assertTrue(bool(((first >= 0) & (first <= 1)).all()))

    def test_euler_and_heun_integrate_constant_velocity_from_supplied_noise(self):
        noise = torch.full_like(self.base, -.3)
        velocity = torch.full_like(noise, .4)
        expected = torch.full_like(noise, .55)
        for kind in ("teacher", "student"):
            with self.subTest(kind=kind):
                renderer = self.Renderer(kind=kind, image_size=128).eval()
                with patch.object(renderer, "velocity", return_value=velocity):
                    result = renderer.sample(self.base, self.state, noise=noise, steps=4)
                torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)
                # Sampling must not mutate caller-owned noise or its fixed buffer.
                torch.testing.assert_close(noise, torch.full_like(noise, -.3))

    def test_heun_integrates_a_linear_time_velocity_over_the_complete_interval(self):
        renderer = self.Renderer(kind="teacher", image_size=128).eval()
        noise = torch.zeros_like(self.base)

        def velocity(noisy, time, base_rgb, state, self_condition=None):
            return time[:, None, None, None].expand_as(noisy)

        with patch.object(renderer, "velocity", side_effect=velocity):
            result = renderer.sample(self.base, self.state, noise=noise, steps=4)
        # Integral_0^1 t dt = .5, then latent -> RGB maps .5 to .75.
        torch.testing.assert_close(result, torch.full_like(result, .75), atol=1e-6, rtol=1e-6)

    def test_reflow_uses_teacher_endpoint_and_the_same_supplied_noise(self):
        teacher = self.Renderer(kind="teacher", image_size=128, mode="flow_matching").eval()
        student = self.Renderer(kind="student", image_size=128, mode="rectified_flow").train()
        noise = torch.full_like(self.base, -.3)
        velocity = torch.full_like(noise, .4)
        with patch.object(teacher, "velocity", return_value=velocity):
            endpoint = teacher.sample(self.base, self.state, noise=noise, steps=3)
        supplied_time = torch.tensor([.25])
        unrelated_ground_truth = torch.full_like(endpoint, .05)
        captured = []

        def predict(noisy, time, base_rgb, state, self_condition=None):
            captured.append(noisy.clone())
            return velocity

        with patch.object(student, "velocity", side_effect=predict):
            loss, _ = student.loss(unrelated_ground_truth, self.base, self.state,
                                   noise=noise, time=supplied_time, target_endpoint=endpoint)
        self.assertLess(float(loss), 1e-8)
        torch.testing.assert_close(captured[0], noise * .75 + (endpoint * 2 - 1) * .25)
        with patch.object(student, "velocity", return_value=velocity):
            mismatched, _ = student.loss(unrelated_ground_truth, self.base, self.state,
                                         noise=noise + .5, time=supplied_time,
                                         target_endpoint=endpoint)
        self.assertGreater(float(mismatched), .1)

    def test_old_direct_checkpoint_loads_and_keeps_the_original_render_path(self):
        config = self.Config(kind="student", image_size=128, architecture="legacy_cnn",
                             transition_mode="residual_v0")
        model = self.Model(config).eval()
        self.assertFalse(hasattr(model, "flow_renderer"))
        old_config = asdict(config)
        for key in ("architecture", "transition_mode", "simulation_hz", "endless",
                    "renderer_mode", "flow_steps"):
            old_config.pop(key)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old_direct.pt"
            torch.save({"stage": "world", "config": old_config, "model": model.state_dict()}, path)
            restored, _ = self.load_checkpoint(path)
        self.assertEqual(restored.config.renderer_mode, "direct")
        grid = torch.zeros(1, 16, 16, dtype=torch.long)
        features = torch.zeros(1, 128)
        with torch.no_grad():
            encoded = restored.encode_map(grid)
            actual = restored.render(self.state, features, encoded)
            expected = restored.render_base(self.state, features, encoded)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(any(name.startswith("flow_renderer.") for name in restored.state_dict()))


if __name__ == "__main__":
    unittest.main()
