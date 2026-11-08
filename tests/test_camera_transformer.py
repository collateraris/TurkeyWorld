"""Camera geometry and runtime boundaries for the Transformer world model.

These tests verify coordinate conventions, supervision contracts and executable
models. They do not establish trained visibility accuracy or a target FPS.
"""
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def simple_state():
    state = np.zeros(90, dtype=np.float32)
    state[:8] = [2.5 / 16, 2.5 / 16, 0, 1, 1, 1, 0, 0]
    for slot in range(8):
        offset = 8 + 10 * slot
        state[offset:offset + 2] = [5.5 / 16, 2.5 / 16]
        state[offset + 4:offset + 6] = [.6, 1]
    return state


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class CameraGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        from turkeyworld import camera
        from turkeyworld.geometry_targets import build_geometry_targets
        cls.camera = camera
        cls.oracle = staticmethod(build_geometry_targets)

    def test_yaw_changes_forward_and_right_with_engine_convention(self):
        state = simple_state()
        state[8:10] = [4.5 / 16, 3.5 / 16]
        state = torch.tensor(state)[None]
        relative = self.camera.object_camera_features(state)
        torch.testing.assert_close(relative["forward"][0, 0], torch.tensor(2.))
        torch.testing.assert_close(relative["right"][0, 0], torch.tensor(1.))
        state[:, 2], state[:, 3] = 1., 0.
        relative = self.camera.object_camera_features(state)
        torch.testing.assert_close(relative["forward"][0, 0], torch.tensor(1.))
        torch.testing.assert_close(relative["right"][0, 0], torch.tensor(-2.))

    def test_sampling_matches_world_cell_centres_and_has_pose_gradient(self):
        # A linear field must interpolate to its world-coordinate value. This
        # detects half-cell shifts and align_corners mistakes.
        yy, xx = torch.meshgrid(torch.arange(16) + .5,
                                torch.arange(16) + .5, indexing="ij")
        features = (xx + 100 * yy)[None, None].requires_grad_()
        state = torch.tensor(simple_state())[None].requires_grad_()
        result = self.camera.sample_camera_frustum(
            features, state, rays=1, samples=4, max_depth=2.)
        expected = result["world_xy"][..., 0] + 100 * result["world_xy"][..., 1]
        torch.testing.assert_close(result["sampled"][..., 0], expected,
                                   atol=1e-4, rtol=1e-5)
        result["sampled"].square().mean().backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertTrue(torch.isfinite(state.grad).all())
        self.assertGreater(float(state.grad[:, :2].abs().sum()), 0)

    def test_outside_map_samples_are_explicitly_marked(self):
        features = torch.ones(1, 2, 16, 16)
        state = torch.tensor(simple_state())[None]
        state[:, 0] = 15.5 / 16
        result = self.camera.sample_camera_frustum(
            features, state, rays=1, samples=4, max_depth=8.)
        self.assertFalse(bool(result["in_bounds"].all()))
        far_outside = result["world_xy"][..., 0] >= 17
        self.assertTrue(bool(far_outside.any()))
        torch.testing.assert_close(result["sampled"][far_outside],
                                   torch.zeros_like(result["sampled"][far_outside]))

    def test_frustum_does_not_delete_dead_objects(self):
        state = simple_state()
        state[8 + 4:8 + 6] = 0  # A corpse still has a visible sprite.
        result = self.camera.object_camera_features(torch.tensor(state)[None])
        self.assertTrue(bool(result["frustum"][0, 0]))
        self.assertEqual(result["features"].shape[:2], (1, 8))

    def test_training_visibility_distinguishes_wall_occlusion_and_corpses(self):
        tilemap = np.zeros((16, 16), dtype=np.uint8)
        tilemap[[0, -1], :] = 1
        tilemap[:, [0, -1]] = 1
        state = simple_state()
        state[12:14] = 0  # corpse in front, same camera projection
        open_scene = self.oracle(tilemap, state, rays=1, max_depth=24.)
        self.assertEqual(float(open_scene["visibility"][0]), 1.)
        self.assertAlmostEqual(float(open_scene["depth"][0]), 12.5, places=5)
        tilemap[:, 4] = 1
        blocked_scene = self.oracle(tilemap, state, rays=1, max_depth=24.)
        self.assertEqual(float(blocked_scene["visibility"][0]), 0.)
        self.assertAlmostEqual(float(blocked_scene["depth"][0]), 1.5, places=5)


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class TransformerRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        from turkeyworld.models import (NeuralWorldModel, WorldModelConfig,
                                        load_world_checkpoint)
        cls.Model = NeuralWorldModel
        cls.Config = WorldModelConfig
        cls.load_checkpoint = staticmethod(load_world_checkpoint)

    def test_rgb_geometry_and_gradients_without_runtime_oracle(self):
        torch.manual_seed(5)
        model = self.Model(self.Config(architecture="camera_transformer", image_size=128))
        self.assertFalse(any(isinstance(module, torch.nn.Conv2d) for module in model.modules()))
        state = torch.tensor(np.stack([simple_state(), simple_state()]))
        action = torch.tensor([[.4, 0, .1, 1], [0, .2, -.1, 0.]])
        grid = torch.zeros(2, 16, 16, dtype=torch.long)
        grid[:, :, 4] = 1
        with patch("turkeyworld.geometry_targets.build_geometry_targets", side_effect=AssertionError("runtime oracle")), \
             patch("turkeyworld.engine.Game.step", side_effect=AssertionError("runtime engine step")), \
             patch("turkeyworld.engine.Game.render", side_effect=AssertionError("runtime engine render")):
            encoded = model.encode_map(grid)
            output = model.step(state, action, map_features=encoded)
            geometry = model.geometry(output["state"], encoded)
        self.assertEqual(tuple(output["rgb"].shape), (2, 3, 128, 128))
        self.assertEqual(tuple(output["state"].shape), (2, 90))
        self.assertEqual(tuple(geometry["depth"].shape), (2, 32))
        self.assertEqual(tuple(geometry["visibility"].shape), (2, 8))
        self.assertEqual(tuple(geometry["object_tokens"].shape[:2]), (2, 8))
        loss = (output["rgb"].square().mean()
                + .01 * geometry["depth"].square().mean()
                + .1 * geometry["visibility_logits"].square().mean())
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
        self.assertTrue(any(float(g.abs().sum()) > 0 for g in gradients))
        model_256 = self.Model(self.Config(architecture="camera_transformer", image_size=256)).eval()
        with torch.inference_mode():
            large = model_256.step(state[:1], action[:1], grid=grid[:1])
        self.assertEqual(tuple(large["rgb"].shape), (1, 3, 256, 256))

    def test_visibility_affects_renderer_after_token_normalization(self):
        # A positive scalar before LayerNorm is nearly normalized away. Hold
        # state/map/object appearance fixed and change only confidence: the
        # rendered image must still respond to learned center visibility.
        torch.manual_seed(11)
        model = self.Model(self.Config(architecture="camera_transformer", image_size=128)).eval()
        state = torch.tensor(simple_state())[None]
        grid = torch.zeros(1, 16, 16, dtype=torch.long)
        features = torch.zeros(1, 128)
        with torch.inference_mode():
            encoded = model.encode_map(grid)
            geometry = model.geometry(state, encoded)
            high_visibility = {**geometry, "visibility": torch.ones(1, 8)}
            low_visibility = {**geometry, "visibility": torch.full((1, 8), .1)}
            with patch.object(model.camera_backend, "geometry", return_value=high_visibility):
                high = model.render(state, features, encoded)
            with patch.object(model.camera_backend, "geometry", return_value=low_visibility):
                low = model.render(state, features, encoded)
        self.assertGreater(float((high-low).abs().mean()), 1e-4)

    def test_old_checkpoint_without_architecture_remains_legacy(self):
        model = self.Model(self.Config(architecture="legacy_cnn", image_size=128))
        old_config = asdict(model.config)
        old_config.pop("architecture")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old.pt"
            torch.save({"stage": "world", "config": old_config,
                        "model": model.state_dict()}, path)
            restored, _ = self.load_checkpoint(path)
        self.assertEqual(restored.config.architecture, "legacy_cnn")
        grid = torch.zeros(1, 16, 16, dtype=torch.long)
        state = torch.tensor(simple_state())[None]
        action = torch.zeros(1, 4)
        with torch.inference_mode():
            original = model.eval().step(state, action, grid=grid)["rgb"]
            actual = restored.step(state, action, grid=grid)["rgb"]
        torch.testing.assert_close(actual, original)


if __name__ == "__main__":
    unittest.main()
