"""Held-out future images/states never enter the autoregressive predictor."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class VJEPAValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def write_episode(path, episode_id=0, split="val", steps=8):
        values = np.arange(steps + 1, dtype=np.float32)
        frames = np.broadcast_to((values + episode_id * 20).astype(np.uint8)[:, None, None, None],
                                 (steps + 1, 16, 16, 3)).copy()
        states = np.zeros((steps + 1, STATE_DIM), dtype=np.float32)
        states[:, 0], states[:, 3:5] = values, 1
        np.savez_compressed(path, frames=frames, states=states,
                            actions=np.ones((steps, 4), dtype=np.float32),
                            events=np.zeros((steps, 4), dtype=np.float32),
                            hit_ids=np.full(steps, -1, dtype=np.int64),
                            attack_flags=np.zeros((steps, ENTITY_SLOTS), dtype=np.float32),
                            map=np.zeros((16, 16), dtype=np.int8), split=np.asarray(split), dt=np.asarray(1 / 30))

    @staticmethod
    def model():
        class Stub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.child = torch.nn.Dropout(.5)
                self.config = SimpleNamespace(architecture="vjepa2_ac", image_size=16)
                self.initializations, self.state_inputs = [], []

            def encode_map(self, grid):
                return grid[:, None].float()

            def initialize_from_image(self, images, states, map_features):
                self.initializations.append((images.detach().clone(), states.detach().clone()))
                # Consume randomness intentionally: validation must seed and
                # restore it even for a custom stochastic inference backend.
                torch.rand(1)
                return states[:, -1:, :1].clone()

            def transition(self, state, action, map_features, memory):
                self.state_inputs.append(state.detach().clone())
                next_state = state.clone()
                next_state[:, 0] += action[:, 0]
                return {"state": next_state, "memory": memory + 1,
                        "features": next_state.new_zeros(len(state), 128)}

            def render(self, state, features, map_features):
                return (state[:, 0] / 255)[:, None, None, None].expand(-1, 3, 16, 16)

        return Stub()

    def test_metrics_use_all_future_predictions_and_do_not_feed_future_states(self):
        from turkeyworld.jepa_training_utils import SequenceData
        from turkeyworld.vjepa_validation import evaluate_autoregressive
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            self.write_episode(path / "a.npz")
            data = SequenceData(path, split="val")
            model = self.model()
            panel = [(0, 0)]
            first, first_preview = evaluate_autoregressive(model, data, panel, step=500)
            self.assertTrue(first["passed"])
            self.assertEqual(first["summary"]["frames"], 4)
            self.assertEqual(first["summary"]["psnr_mean"], 120.)
            self.assertEqual([int(state[0, 0]) for state in model.state_inputs], [1, 2, 3, 4])
            self.assertEqual(tuple(model.initializations[0][0].shape), (1, 2, 3, 16, 16))
            self.assertEqual(tuple(model.initializations[0][1].shape), (1, 2, STATE_DIM))
            # Change every future measurement target, preserving initial context
            # and controls. Generated RGB/state inputs must remain identical.
            data.episodes[0]["frames"][2:] = 200
            data.episodes[0]["states"][2:, 0] = 1000
            model.state_inputs.clear()
            second, second_preview = evaluate_autoregressive(model, data, panel, step=1000)
            self.assertFalse(second["passed"])
            np.testing.assert_array_equal(first_preview["generated"], second_preview["generated"])
            self.assertEqual([int(state[0, 0]) for state in model.state_inputs], [1, 2, 3, 4])
            self.assertNotEqual(first["panel"]["fingerprint"], second["panel"]["fingerprint"])
            self.assertTrue(second["source_held_out"])
            self.assertFalse(second["future_source_feedback"])
            json.dumps(second, allow_nan=False)

    def test_modes_rng_and_matmul_settings_are_preserved_and_batches_are_supported(self):
        from turkeyworld.jepa_training_utils import SequenceData
        from turkeyworld.vjepa_validation import evaluate_autoregressive
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            self.write_episode(path / "a.npz")
            self.write_episode(path / "b.npz", 1)
            data, model = SequenceData(path, split="val"), self.model()
            model.train()
            model.child.eval()
            torch_state = torch.get_rng_state().clone()
            numpy_state = np.random.get_state()
            python_state = random.getstate()
            data_rng = json.dumps(data.rng.bit_generator.state, sort_keys=True)
            precision = torch.get_float32_matmul_precision()
            tf32 = torch.backends.cuda.matmul.allow_tf32
            report, _ = evaluate_autoregressive(model, data, [(0, 0), (1, 1)], batch_size=2)
            self.assertEqual(report["summary"]["frames"], 8)
            self.assertEqual([frame["window"] for frame in report["per_frame"]], [0] * 4 + [1] * 4)
            self.assertTrue(model.training)
            self.assertFalse(model.child.training)
            torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
            restored_numpy = np.random.get_state()
            self.assertEqual(numpy_state[0], restored_numpy[0])
            np.testing.assert_array_equal(numpy_state[1], restored_numpy[1])
            self.assertEqual(numpy_state[2:], restored_numpy[2:])
            self.assertEqual(random.getstate(), python_state)
            self.assertEqual(json.dumps(data.rng.bit_generator.state, sort_keys=True), data_rng)
            self.assertEqual(torch.get_float32_matmul_precision(), precision)
            self.assertEqual(torch.backends.cuda.matmul.allow_tf32, tf32)

    def test_preview_contact_sheet_and_metadata_are_actual_free_rollout(self):
        from turkeyworld.jepa_training_utils import SequenceData
        from turkeyworld.vjepa_validation import evaluate_autoregressive, save_autoregressive_preview
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            self.write_episode(path / "a.npz")
            data = SequenceData(path, split="val")
            _, preview = evaluate_autoregressive(self.model(), data, [(0, 0)], step=500)
            output = path / "preview.png"
            metadata = save_autoregressive_preview(preview, output)
            with Image.open(output) as image:
                self.assertEqual(image.size, (512, 430))
                # Frame0/source2/model2 have known pixel values and rows.
                self.assertEqual(image.getpixel((64, 80)), (0, 0, 0))
                self.assertEqual(image.getpixel((64, 220)), (2, 2, 2))
                self.assertEqual(image.getpixel((64, 360)), (2, 2, 2))
            self.assertEqual(metadata["step"], 500)
            self.assertEqual(metadata["every"], 500)
            self.assertEqual(metadata["context"], 2)
            self.assertEqual(metadata["horizon"], 4)
            self.assertEqual(metadata["kind"], "free_rollout")
            self.assertTrue(metadata["source_held_out"])
            self.assertEqual(json.loads(output.with_suffix(".json").read_text())["step"], 500)

    def test_training_split_and_duplicate_validation_windows_are_rejected(self):
        from turkeyworld.jepa_training_utils import SequenceData
        from turkeyworld.vjepa_validation import evaluate_autoregressive
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            self.write_episode(path / "a.npz", split="train")
            train = SequenceData(path, split="train")
            with self.assertRaisesRegex(ValueError, "held-out"):
                evaluate_autoregressive(self.model(), train)
            train.split = "val"
            with self.assertRaisesRegex(ValueError, "unique"):
                evaluate_autoregressive(self.model(), train, [(0, 0), (0, 0)])

    def test_validation_failure_still_restores_modes_rng_and_precision(self):
        from turkeyworld.jepa_training_utils import SequenceData
        from turkeyworld.vjepa_validation import evaluate_autoregressive
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            self.write_episode(path / "a.npz")
            data, model = SequenceData(path, split="val"), self.model()
            model.train()
            model.child.eval()
            rng = torch.get_rng_state().clone()
            precision = torch.get_float32_matmul_precision()
            with patch.object(model, "render", side_effect=RuntimeError("render failed")):
                with self.assertRaisesRegex(RuntimeError, "render failed"):
                    evaluate_autoregressive(model, data, [(0, 0)])
            self.assertTrue(model.training)
            self.assertFalse(model.child.training)
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            self.assertEqual(torch.get_float32_matmul_precision(), precision)

    def test_nonfinite_predicted_state_fails_without_feeding_it_into_more_transitions(self):
        from turkeyworld.jepa_training_utils import SequenceData
        from turkeyworld.vjepa_validation import evaluate_autoregressive
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            self.write_episode(path / "a.npz")
            data, model = SequenceData(path, split="val"), self.model()

            def invalid(state, action, map_features, memory):
                return {"state": state.new_full(state.shape, float("nan")), "memory": memory}

            with patch.object(model, "transition", side_effect=invalid) as transition, \
                 patch.object(model, "render", side_effect=AssertionError("Invalid pose must not be rendered")):
                report, preview = evaluate_autoregressive(model, data, [(0, 0)])
            self.assertEqual(transition.call_count, 1)
            self.assertFalse(report["passed"])
            self.assertEqual(report["summary"]["nonfinite_frames"], 4)
            self.assertFalse(preview["metadata"]["generated_finite"])
            json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
