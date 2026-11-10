"""V-JEPA receives one initial source image; rollout only uses cached/model memory."""
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class VJEPAInitializationTests(unittest.TestCase):
    @staticmethod
    def model(architecture="vjepa2_ac"):
        class StubWorld(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.config = SimpleNamespace(architecture=architecture, transition_mode="event_v1",
                    simulation_hz=30., image_size=128, endless=False, kind="teacher", renderer_mode="direct")
                self.initialized_images = []
                self.memory_inputs = []
                self.initial_memory_calls = 0

            def encode_map(self, grid):
                return grid[:, None].float()

            def initial_memory(self, batch, device=None, dtype=None):
                self.initial_memory_calls += 1
                if architecture == "vjepa2_ac":
                    raise AssertionError("V-JEPA must initialize from its source image")
                return torch.zeros(batch, 64, device=device, dtype=dtype)

            def initialize_from_image(self, image, state, map_features):
                if architecture != "vjepa2_ac":
                    raise AssertionError("Legacy models must not receive source RGB")
                self.initialized_images.append(image.detach().clone())
                return torch.full((image.shape[0], 264, 512), float(image.float().mean()), device=image.device)

            def transition(self, state, action, map_features, memory):
                self.memory_inputs.append(memory.detach().clone())
                output = state.clone()
                output[:, -1] += 1 / 9000
                return {"state": output, "memory": memory + 1,
                        "features": state.new_zeros(len(state), 128),
                        "events": state.new_zeros(len(state), 4)}

            def render(self, state, features, map_features):
                return state.new_full((len(state), 3, 128, 128), .3)

            def step(self, state, action, memory, map_features):
                result = self.transition(state, action, map_features, memory)
                result["rgb"] = self.render(result["state"], result["features"], map_features)
                return result

        return StubWorld()

    @staticmethod
    def arguments():
        return Namespace(world=Path("stub.pt"), device="cpu", threads=1, sr=None, fp16=False,
                         seed=1, display_size=128, smooth=False, headless=True, fps=60, warmup=2)

    @staticmethod
    def evaluation_arguments(render_every=0):
        return Namespace(seconds=2 / 30, scenario="idle", no_shadow_reference=False,
                         max_failures=50, render_every=render_every, min_render_psnr=None,
                         continue_on_failure=True, progress_every=0)

    def test_runtime_renders_initial_image_once_and_resets_cached_packed_memory(self):
        from turkeyworld.neural_play import NeuralRuntime
        model = self.model()
        rgb = np.full((128, 128, 3), 64, dtype=np.uint8)
        with patch("turkeyworld.neural_play.load_world_checkpoint", return_value=(model, {"training_steps": 7})), \
             patch("turkeyworld.neural_play.parameter_counts", return_value={"world_total": 1}), \
             patch("turkeyworld.neural_play.Game.render", autospec=True, return_value=rgb) as render, \
             patch("turkeyworld.neural_play.Game.step", side_effect=AssertionError("engine rollout forbidden")):
            runtime = NeuralRuntime(self.arguments())
            self.assertEqual(tuple(runtime.memory.shape), (1, 264, 512))
            self.assertEqual(len(model.initialized_images), 1)
            torch.testing.assert_close(model.initialized_images[0], torch.full((1, 3, 128, 128), 64 / 255))
            original = runtime.initial_memory.clone()
            runtime.memory.add_(5)
            torch.testing.assert_close(runtime.initial_memory, original, rtol=0, atol=0)
            runtime.reset_from_cached_initial_state()
            self.assertNotEqual(runtime.memory.data_ptr(), runtime.initial_memory.data_ptr())
            runtime.warmup()
            torch.testing.assert_close(runtime.memory, original, rtol=0, atol=0)
            runtime.step([0, 0, 0, 0])
            report = runtime.report(.01)
            self.assertEqual(render.call_count, 1)
            self.assertEqual(model.initial_memory_calls, 0)
            self.assertEqual(report["engine_renders_for_initialization"], 1)
            self.assertEqual(report["engine_renders_during_rollout"], 0)
            self.assertEqual(report["engine_steps_during_rollout"], 0)
            self.assertEqual(report["world_architecture"], "vjepa2_ac")

    def test_old_runtime_still_uses_no_initial_source_render(self):
        from turkeyworld.neural_play import NeuralRuntime
        model = self.model("camera_transformer")
        with patch("turkeyworld.neural_play.load_world_checkpoint", return_value=(model, {})), \
             patch("turkeyworld.neural_play.parameter_counts", return_value={}), \
             patch("turkeyworld.neural_play.Game.render", side_effect=AssertionError("legacy RGB query forbidden")):
            runtime = NeuralRuntime(self.arguments())
            runtime.warmup()
            self.assertEqual(tuple(runtime.memory.shape), (1, 64))
            self.assertEqual(model.initialized_images, [])
            self.assertEqual(runtime.report(.01)["engine_renders_for_initialization"], 0)

    def test_action_tape_renders_before_any_reference_transition(self):
        from turkeyworld.engine import Game
        from turkeyworld.evaluate_rollout import build_action_tape
        original_step = Game.step
        calls = []
        rgb = np.full((128, 128, 3), 73, dtype=np.uint8)

        def render(game, *values, **options):
            calls.append(("render", game.time))
            return rgb

        def step(game, *values, **options):
            calls.append(("step", game.time))
            return original_step(game, *values, **options)

        with patch.object(Game, "render", autospec=True, side_effect=render) as render_mock, \
             patch.object(Game, "step", autospec=True, side_effect=step):
            tape = build_action_tape(1, "idle", 3, 1 / 30, initialize_rgb=True)
        self.assertEqual(calls[0], ("render", 0.))
        self.assertEqual(render_mock.call_count, 1)
        np.testing.assert_array_equal(tape["initial_rgb"], rgb)
        with patch.object(Game, "render", side_effect=AssertionError("unexpected old tape RGB")):
            old_tape = build_action_tape(1, "idle", 2, 1 / 30)
        self.assertNotIn("initial_rgb", old_tape)

    def test_evaluator_uses_initial_image_once_and_never_future_diagnostic_images(self):
        from turkeyworld.evaluate_rollout import evaluate_episode
        model = self.model()
        initial = np.full((128, 128, 3), 64, dtype=np.uint8)
        diagnostic = np.full_like(initial, 224)
        with patch("turkeyworld.evaluate_rollout.Game.render", autospec=True,
                   side_effect=[initial, diagnostic, diagnostic]) as render:
            report = evaluate_episode(model, {}, self.evaluation_arguments(render_every=1), seed=1)
        self.assertEqual(render.call_count, 3)
        self.assertEqual(len(model.initialized_images), 1)
        torch.testing.assert_close(model.initialized_images[0], torch.full((1, 3, 128, 128), 64 / 255))
        self.assertEqual(len(model.memory_inputs), 2)
        torch.testing.assert_close(model.memory_inputs[1], model.memory_inputs[0] + 1)
        self.assertEqual(report["world_architecture"], "vjepa2_ac")
        self.assertEqual(report["engine_renders_for_initialization"], 1)
        self.assertEqual(report["rendered_sample_count"], 2)

    def test_old_evaluator_performs_no_initial_rgb_render(self):
        from turkeyworld.evaluate_rollout import evaluate_episode
        model = self.model("camera_transformer")
        with patch("turkeyworld.evaluate_rollout.Game.render", side_effect=AssertionError("legacy RGB query forbidden")):
            report = evaluate_episode(model, {}, self.evaluation_arguments(), seed=1)
        self.assertEqual(report["engine_renders_for_initialization"], 0)
        self.assertEqual(model.initial_memory_calls, 1)
        self.assertEqual(model.initialized_images, [])


if __name__ == "__main__":
    unittest.main()
