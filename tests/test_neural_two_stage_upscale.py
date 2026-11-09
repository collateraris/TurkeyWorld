"""Learned x4 SR must run before the separate bilinear x4 display stage."""
from argparse import Namespace
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from turkeyworld.models import SingleFrameUpscaler
from turkeyworld.neural_play import NeuralRuntime, _arguments


class StubWorld(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(image_size=size, architecture="vjepa2_ac", simulation_hz=30.,
            transition_mode="event_v1", endless=True, kind="teacher", renderer_mode="direct")
        self.rgb = torch.linspace(.1, .8, size*size).reshape(1, 1, size, size).expand(1, 3, size, size)

    def encode_map(self, grid):
        return grid.float()

    def initialize_from_image(self, image, state, map_features):
        return torch.zeros(1, 2, 4)

    def step(self, state, action, memory, map_features):
        return {"state": state, "memory": memory+1, "rgb": self.rgb,
                "events": torch.zeros(1, 4)}


class TwoStageUpscaleTests(unittest.TestCase):
    @staticmethod
    def args(**changes):
        values = dict(world=Path("world.pt"), sr=Path("sr.pt"), device="cpu", threads=1,
            fp16=False, seed=1, display_size=None, two_stage_upscale=True, max_catchup_steps=1,
            smooth=False, headless=True, fps=60, warmup=0)
        return Namespace(**(values | changes))

    def runtime(self, size, args=None, steps=3, sr=None, metadata=None):
        model = StubWorld(size)
        upscaler = sr if sr is not None else SingleFrameUpscaler(channels=4, blocks=1).eval()
        with torch.no_grad():
            if isinstance(upscaler, SingleFrameUpscaler):
                upscaler.output[0].bias.fill_(.03)
        with patch("turkeyworld.neural_play.load_world_checkpoint", return_value=(model, {"training_steps": 7})), \
             patch("turkeyworld.neural_play.load_sr_checkpoint", return_value=(upscaler,
                   {"training_steps": steps, **(metadata or {})})), \
             patch("turkeyworld.neural_play.Game.render", return_value=np.zeros((size, size, 3), np.uint8)), \
             patch("turkeyworld.neural_play.parameter_counts", return_value={}):
            return NeuralRuntime(args or self.args())

    def test_native32_and64_execute_learned_sr_then_bilinear_and_report_actual_stages(self):
        for native in (32, 64):
            with self.subTest(native=native):
                runtime = self.runtime(native)
                original = F.interpolate
                with torch.inference_mode():
                    expected = original(runtime.sr(runtime.model.rgb), size=(native*16, native*16),
                                        mode="bilinear", align_corners=False)
                with patch("turkeyworld.neural_play.F.interpolate", wraps=original) as interpolate, \
                     patch("turkeyworld.neural_play.Game.step", side_effect=AssertionError("engine step forbidden")), \
                     patch("turkeyworld.neural_play.Game.render", side_effect=AssertionError("future source forbidden")), \
                     patch("turkeyworld.neural_play.parameter_counts", return_value={}):
                    frame = runtime.step([0, 0, 0, 0])
                    report = runtime.report(.5)
                expected_pixels = expected[0].mul(255).round().byte().permute(1, 2, 0).numpy()
                np.testing.assert_array_equal(frame, expected_pixels)
                self.assertEqual(frame.shape, (native*16, native*16, 3))
                final_call = interpolate.call_args
                self.assertEqual(final_call.args[0].shape[-2:], (native*4, native*4))
                self.assertEqual(final_call.kwargs, {"size": (native*16, native*16),
                    "mode": "bilinear", "align_corners": False})
                self.assertFalse(any(call.args[0].shape[-1] == native and
                    call.kwargs.get("size") == (native*16, native*16) for call in interpolate.call_args_list))
                self.assertEqual(report["upscale"], "neural_x4_then_bilinear_x4")
                self.assertEqual(report["base_resolution"], [native, native])
                self.assertEqual(report["output_resolution"], [native*16, native*16])
                self.assertEqual(report["upscale_stages"], [
                    {"method": "neural_x4", "input_resolution": [native]*2,
                     "output_resolution": [native*4]*2, "scale": 4},
                    {"method": "bilinear", "input_resolution": [native*4]*2,
                     "output_resolution": [native*16]*2, "scale": 4, "align_corners": False}])
                self.assertEqual(report["sr_training_steps"], 3)
                self.assertEqual(report["engine_renders_during_rollout"], 0)
                self.assertEqual(report["max_catchup_steps_per_input_poll"], 1)
                self.assertFalse(report["gameplay_correctness_evaluated"])

    def test_wrong_sr_scale_does_not_fall_back_to_classical_resize(self):
        runtime = self.runtime(32, sr=torch.nn.Upsample(scale_factor=2))
        with self.assertRaisesRegex(ValueError, "exactly x4"):
            runtime._resize(runtime.model.rgb)

    def test_two_stage_requires_trained_sr_and_matching_native_display(self):
        for steps in (0, None, float("nan"), True):
            with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, "positive training_steps"):
                self.runtime(32, steps=steps)
        with self.assertRaisesRegex(ValueError, "times16"):
            self.runtime(32, self.args(display_size=1024))
        with self.assertRaisesRegex(ValueError, "native32/64"):
            self.runtime(128)
        with self.assertRaisesRegex(ValueError, "trained SR checkpoint"):
            self.runtime(32, self.args(sr=None))
        with self.assertRaisesRegex(ValueError, "resolution differs"):
            self.runtime(32, metadata={"input_size": 64, "output_size": 256})

    def test_cli_requires_sr_and_keeps_legacy_defaults(self):
        with TemporaryDirectory() as directory:
            world, sr = Path(directory)/"world.pt", Path(directory)/"sr.pt"
            world.touch()
            sr.touch()
            legacy = _arguments(["--world", str(world)])
            self.assertEqual((legacy.display_size, legacy.max_catchup_steps), (1024, 4))
            new = _arguments(["--world", str(world), "--sr", str(sr), "--two-stage-upscale"])
            self.assertIsNone(new.display_size)
            self.assertEqual(new.max_catchup_steps, 1)
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                _arguments(["--world", str(world), "--two-stage-upscale"])

    def test_legacy_resize_and_sr_acceptance_remain_unchanged(self):
        runtime = self.runtime(32, self.args(two_stage_upscale=False, display_size=128), steps=0)
        result = runtime._resize(runtime.model.rgb)
        torch.testing.assert_close(result, runtime.sr(runtime.model.rgb))
        runtime.sr = None
        runtime.args.smooth = False
        torch.testing.assert_close(runtime._resize(runtime.model.rgb),
                                   F.interpolate(runtime.model.rgb, size=(128, 128), mode="nearest"))


if __name__ == "__main__":
    unittest.main()
