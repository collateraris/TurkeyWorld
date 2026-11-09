"""Codec held-out reconstruction is measured separately from world rollout."""
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
class VQGANValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self, root, split="val"):
        from turkeyworld.vqgan_data import VQGANData
        directory = Path(root) / split
        directory.mkdir()
        frames = np.broadcast_to(np.arange(7, dtype=np.uint8)[:, None, None, None], (7, 16, 16, 3)).copy()
        events = np.zeros((6, 4), np.float32)
        events[0, 0] = 1
        states = np.zeros((7, 90), np.float32)
        np.savez_compressed(directory / "episode.npz", frames=frames, states=states,
            actions=np.zeros((6, 4), np.float32), events=events,
            hit_ids=np.full(6, -1, np.int64), attack_flags=np.zeros((6, 8), np.float32),
            map=np.zeros((16, 16), np.int8), split=np.asarray(split), dt=np.asarray(1/30),
            frame_source=np.asarray("engine"))
        return VQGANData(directory, split=split, image_size=16, cache_bytes=1_000_000)

    def model(self):
        class Codec(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.child = torch.nn.Dropout(.5)
                self.config = SimpleNamespace(n_embed=8, image_size=16)
                self.mode, self.offset = "identity", 0.
                self.calls = []

            def forward(self, source):
                self.calls.append((tuple(source.shape), source.dtype, self.training,
                                   torch.is_autocast_enabled("cpu"), torch.is_grad_enabled()))
                torch.rand(1)
                np.random.rand()
                random.random()
                if self.mode == "raise":
                    raise RuntimeError("intentional codec failure")
                reconstruction = source.clone() + self.anchor * 0 + self.offset
                if self.mode == "double":
                    reconstruction = source * 2
                elif self.mode == "nan":
                    reconstruction[0, 0, 0, 0] = float("nan")
                elif self.mode == "range":
                    reconstruction[0, 0, 0, 0] = 1.1
                code = (source.mean((1, 2, 3)) * 255).round().long() % 4
                if self.mode == "collapse":
                    code.zero_()
                indices = code[:, None, None].expand(-1, 2, 2).clone()
                if self.mode == "invalid_code":
                    indices[0, 0, 0] = 8
                prequant = source.mean((1, 2, 3))[:, None, None, None].expand(-1, 2, 2, 2)
                quantized = prequant + .5
                if self.mode == "invalid_quantization":
                    quantized[0, 0, 0, 0] = float("nan")
                return {"reconstruction": reconstruction, "indices": indices,
                    "prequant": prequant, "quantized": quantized,
                    "metrics": {"active_codes": 1024, "perplexity": 1024}, "vq_loss": source.new_zeros(())}

        return Codec()

    def test_perfect_codec_metrics_global_panel_histogram_flash_and_batch_independence(self):
        from turkeyworld.vqgan_validation import evaluate_vqgan
        with tempfile.TemporaryDirectory() as folder:
            data, model = self.fixture(folder), self.model()
            panel = [(0, 0), (0, 2)]
            first, _ = evaluate_vqgan(model, data, panel, batch_size=1, step=500)
            second, _ = evaluate_vqgan(model, data, panel, batch_size=2, step=1000)
            self.assertTrue(first["passed"])
            self.assertEqual(first["summary"]["frames"], 4)
            self.assertEqual(first["summary"]["psnr_mean"], 120.)
            self.assertEqual(first["summary"]["mae_mean"], 0.)
            self.assertEqual(first["summary"]["edge_l1"], 0.)
            self.assertAlmostEqual(first["summary"]["quantization_mse"], .25)
            self.assertEqual(first["summary"]["temporal_reconstruction_error"], 0.)
            self.assertEqual(first["codebook"]["histogram"], [4, 4, 4, 4, 0, 0, 0, 0])
            self.assertEqual(first["codebook"]["active_codes"], 4)
            self.assertEqual(first["codebook"]["perplexity"], 4.)
            self.assertEqual(first["codebook"]["live_fraction"], .5)
            self.assertEqual(first["codebook"]["dead_fraction"], .5)
            self.assertEqual(first["codebook"]["scope"], "fixed_validation_panel")
            self.assertEqual(first["counts"]["shot_pairs"], 1)
            self.assertEqual(first["summary"]["flash_frames"], 1)
            self.assertIsNotNone(first["flash_summary"])
            self.assertEqual(first["fingerprint"], second["fingerprint"])
            self.assertEqual(first["summary"], second["summary"])
            self.assertTrue(first["source_input_allowed_for_codec"])
            self.assertFalse(first["world_rollout"])
            json.dumps(first, allow_nan=False)

    def test_temporal_error_compares_pair_differences_and_collapsed_codes_cannot_qualify(self):
        from turkeyworld.vqgan_validation import evaluate_vqgan
        with tempfile.TemporaryDirectory() as folder:
            data, model = self.fixture(folder), self.model()
            model.mode = "double"
            report, _ = evaluate_vqgan(model, data, [(0, 0), (0, 2)], batch_size=2)
            self.assertAlmostEqual(report["summary"]["temporal_reconstruction_error"], 1/255, places=8)
            model.mode = "collapse"
            report, _ = evaluate_vqgan(model, data, [(0, 0), (0, 2)])
            self.assertTrue(report["visual_passed"])
            self.assertFalse(report["passed"])
            self.assertEqual(report["codebook"]["active_codes"], 1)
            self.assertIn("codebook_broad_usage", report["failure_reasons"])

    def test_nonfinite_or_out_of_range_frames_fail_strictly_and_json_remains_finite(self):
        from turkeyworld.vqgan_validation import evaluate_vqgan
        with tempfile.TemporaryDirectory() as folder:
            data, model = self.fixture(folder), self.model()
            # The visual fraction can tolerate one bad frame; codec range/finite cannot.
            for mode in ("nan", "range"):
                with self.subTest(mode=mode):
                    model.mode = mode
                    report, preview = evaluate_vqgan(model, data, [(0, 0), (0, 2)],
                        thresholds={"required_fraction": .5, "mae": 1., "ssim": -1., "psnr": -120.})
                    self.assertFalse(report["passed"])
                    self.assertIn("all_frames_finite" if mode == "nan" else "all_frames_in_range",
                                  report["failure_reasons"])
                    self.assertEqual(preview["reconstruction"].dtype, np.uint8)
                    json.dumps(report, allow_nan=False)

    def test_perfect_pixels_do_not_hide_invalid_indices_or_nonfinite_quantization(self):
        from turkeyworld.vqgan_validation import evaluate_vqgan
        with tempfile.TemporaryDirectory() as folder:
            data, model = self.fixture(folder), self.model()
            for mode, failed in (("invalid_code", "valid_code_indices"),
                                 ("invalid_quantization", "quantization_finite")):
                with self.subTest(mode=mode):
                    model.mode = mode
                    report, _ = evaluate_vqgan(model, data, [(0, 0), (0, 2)])
                    self.assertTrue(report["visual_passed"])
                    self.assertFalse(report["passed"])
                    self.assertIn(failed, report["failure_reasons"])
                    json.dumps(report, allow_nan=False)

    def test_rng_mixed_modes_and_precision_restore_even_when_codec_raises(self):
        from turkeyworld.vqgan_validation import evaluate_vqgan
        with tempfile.TemporaryDirectory() as folder:
            data, model = self.fixture(folder), self.model()
            model.train()
            model.child.eval()
            original_batch = data.batch

            def batch(*args, **kwargs):
                data.rng.random()
                data.data.rng.random()
                return original_batch(*args, **kwargs)

            saved_torch = torch.get_rng_state().clone()
            saved_numpy, saved_python = np.random.get_state(), random.getstate()
            rngs = [json.dumps(owner.rng.bit_generator.state, sort_keys=True) for owner in (data, data.data)]
            precision = torch.get_float32_matmul_precision()
            tf32, cudnn_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
            with patch.object(data, "batch", side_effect=batch), torch.autocast("cpu", dtype=torch.bfloat16):
                evaluate_vqgan(model, data, [(0, 0), (0, 2)])
            model.mode = "raise"
            with patch.object(data, "batch", side_effect=batch), self.assertRaisesRegex(RuntimeError, "intentional"):
                evaluate_vqgan(model, data, [(0, 0)])
            self.assertTrue(model.training)
            self.assertFalse(model.child.training)
            self.assertTrue(all(call[1:] == (torch.float32, False, False, False) for call in model.calls))
            torch.testing.assert_close(torch.get_rng_state(), saved_torch, rtol=0, atol=0)
            now = np.random.get_state()
            self.assertEqual(now[0], saved_numpy[0])
            np.testing.assert_array_equal(now[1], saved_numpy[1])
            self.assertEqual(now[2:], saved_numpy[2:])
            self.assertEqual(random.getstate(), saved_python)
            self.assertEqual(rngs, [json.dumps(owner.rng.bit_generator.state, sort_keys=True) for owner in (data, data.data)])
            self.assertEqual(torch.get_float32_matmul_precision(), precision)
            self.assertEqual(torch.backends.cuda.matmul.allow_tf32, tf32)
            self.assertEqual(torch.backends.cudnn.allow_tf32, cudnn_tf32)

    def test_real_codec_validation_does_not_initialize_codebook_or_update_training_usage(self):
        from turkeyworld.vqgan import VQGAN, VQGANConfig
        from turkeyworld.vqgan_validation import evaluate_vqgan
        with tempfile.TemporaryDirectory() as folder:
            data = self.fixture(folder)
            model = VQGAN(VQGANConfig(image_size=16, base_channels=8, channel_multipliers=(1, 2),
                num_res_blocks=1, latent_channels=8, embedding_dim=4, n_embed=8, attention=False))
            before = {name: value.detach().clone() for name, value in model.state_dict().items()}
            report, _ = evaluate_vqgan(model, data, [(0, 0), (0, 2)])
            self.assertFalse(report["passed"])
            self.assertEqual(report["codebook"]["total_tokens"], 4 * 8 * 8)
            self.assertTrue(report["checks"]["quantization_finite"])
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
            self.assertTrue(model.training)

    def test_preview_source_reconstruction_and_scaled_difference_rows_are_unambiguous(self):
        from turkeyworld.vqgan_validation import evaluate_vqgan, save_vqgan_preview
        with tempfile.TemporaryDirectory() as folder:
            data, model = self.fixture(folder), self.model()
            model.offset = 2/255
            _, preview = evaluate_vqgan(model, data, [(0, 0), (0, 2)], step=500)
            output = Path(folder) / "codec.png"
            metadata = save_vqgan_preview(preview, output, step=1000)
            with Image.open(output) as image:
                self.assertEqual(image.size, (512, 430))
                self.assertEqual(image.getpixel((64, 80)), (0, 0, 0))
                self.assertEqual(image.getpixel((64, 220)), (2, 2, 2))
                self.assertEqual(image.getpixel((64, 360)), (8, 8, 8))
            self.assertEqual(metadata["step"], 1000)
            self.assertEqual(metadata["kind"], "vqgan_codec_reconstruction")
            self.assertTrue(metadata["source_input_allowed_for_codec"])
            self.assertFalse(metadata["world_rollout"])
            self.assertEqual(metadata, json.loads(output.with_suffix(".json").read_text("utf-8")))

    def test_train_half_precision_duplicate_or_boundary_crossing_panels_are_rejected(self):
        from turkeyworld.vqgan_validation import evaluate_vqgan
        with tempfile.TemporaryDirectory() as folder:
            data, model = self.fixture(folder), self.model()
            for panel in ([(0, 0), (0, 0)], [(0, 6)], [(0, .5)]):
                with self.subTest(panel=panel), self.assertRaises(ValueError):
                    evaluate_vqgan(model, data, panel)
            with self.assertRaisesRegex(ValueError, "FP32"):
                evaluate_vqgan(model.half(), data, [(0, 0)])
            model.float()
            data.split = "train"
            with self.assertRaisesRegex(ValueError, "held-out"):
                evaluate_vqgan(model, data, [(0, 0)])


if __name__ == "__main__":
    unittest.main()
