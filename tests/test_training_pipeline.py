"""A tiny real CPU experiment exercises all training stages and resumable files."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
from argparse import Namespace
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class TrainingPipelineSmokeTests(unittest.TestCase):
    def test_temporal_bfloat16_chunks_refresh_weight_casts_between_optimizer_updates(self):
        from turkeyworld.long_train import run_segment
        from turkeyworld.models import NeuralWorldModel, WorldModelConfig
        from turkeyworld.prepare_training_data import prepare
        from turkeyworld.training_pipeline import Experiment

        torch.set_num_threads(2)
        torch.manual_seed(191)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            prepare(root / "data", rgb_train=0, rgb_val=0, long_train=1, long_val=0,
                    long_frames=48, size=32, seed=812)
            args = Namespace(run_dir=root, device="cpu", seed=191, teacher_steps=1,
                             student_steps=1, temporal_updates=1, reflow_pairs=2,
                             batch=1, lr=3e-4)
            experiment = Experiment(args)
            model = NeuralWorldModel(WorldModelConfig(kind="student", image_size=128, endless=True))
            updates, cache_settings, chunk_counts = [], [], []
            head = model.camera_backend.delta_head[-1].weight
            original_adam = torch.optim.AdamW

            def optimizer_with_update_tracking(parameters, *values, **options):
                optimizer = original_adam(parameters, *values, **options)
                original_step = optimizer.step

                def track_step(*step_values, **step_options):
                    previous = head.detach().clone()
                    result = original_step(*step_values, **step_options)
                    updates.append(float((head.detach() - previous).abs().sum()))
                    return result

                optimizer.step = track_step
                return optimizer

            def cpu_amp(cache_enabled=True):
                cache_settings.append(cache_enabled)
                return torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=cache_enabled)

            def two_real_chunks(*values, **options):
                # One curriculum update is enough to exercise the multi-chunk
                # case that a 16-step smoke test cannot cover.
                options.update(rollout=32, chunk=16, prefix_start=0)
                result = run_segment(*values, **options)
                chunk_counts.append(result["declaration"]["optimizer_steps"])
                return result

            with patch.object(experiment, "amp", side_effect=cpu_amp), \
                 patch("turkeyworld.long_train.run_segment", side_effect=two_real_chunks), \
                 patch("turkeyworld.training_pipeline.torch.optim.AdamW", side_effect=optimizer_with_update_tracking):
                experiment.temporal(model, "student_temporal")
            self.assertEqual(cache_settings, [False])
            self.assertEqual(chunk_counts, [2])
            self.assertEqual(len(updates), 2)
            self.assertTrue(all(np.isfinite(update) and update > 0 for update in updates))
            self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_teacher_reflow_student_temporal_updates_and_checkpoint_resume(self):
        from turkeyworld.models import NeuralWorldModel, WorldModelConfig, load_world_checkpoint
        from turkeyworld.prepare_training_data import prepare
        from turkeyworld.training_pipeline import Experiment, MemoryData

        torch.set_num_threads(2)
        torch.manual_seed(313)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            prepare(root / "data", rgb_train=2, rgb_val=1, clip_frames=4, size=128,
                    long_train=1, long_val=1, long_frames=32, seed=734)
            args = Namespace(run_dir=root, device="cpu", seed=313, teacher_steps=1,
                             student_steps=1, temporal_updates=1, reflow_pairs=2,
                             batch=1, lr=3e-4, log_every=1, save_every=1, validate_every=1)
            experiment = Experiment(args)
            data = MemoryData(root / "data/rgb/train", torch.device("cpu"))
            validation = MemoryData(root / "data/rgb/val", torch.device("cpu"))
            self.assertEqual(data.count, 8)
            self.assertEqual(validation.count, 4)
            teacher = NeuralWorldModel(WorldModelConfig(kind="teacher", image_size=128,
                                                       endless=True, renderer_mode="flow_matching", flow_steps=2))
            experiment.joint(teacher, "teacher", 1, data, validation)
            experiment.temporal(teacher, "teacher_temporal")
            pairs = experiment.reflow(teacher, data)
            self.assertEqual(pairs["states"].shape, (2, STATE_DIM))
            self.assertEqual(pairs["noise"].shape, (2, 3, 128, 128))
            self.assertEqual(pairs["endpoints"].shape, (2, 3, 128, 128))
            self.assertTrue(np.isfinite(pairs["noise"]).all())
            self.assertTrue(all(any(np.array_equal(state, candidate) for candidate in data.data["new"])
                                for state in pairs["states"]))
            self.assertFalse(any(np.array_equal(state, candidate) for state in pairs["states"]
                                 for candidate in validation.data["new"]))
            student = NeuralWorldModel(WorldModelConfig(kind="student", image_size=128,
                                                       endless=True, renderer_mode="rectified_flow", flow_steps=2))
            experiment.joint(student, "student", 1, data, validation, teacher, pairs)
            experiment.temporal(student, "student_temporal")
            for phase in ("teacher", "teacher_temporal", "student", "student_temporal"):
                path = root / "checkpoints" / f"{phase}.pt"
                self.assertTrue(path.exists(), phase)
                loaded, metadata = load_world_checkpoint(path)
                self.assertEqual(metadata["training_steps"], 1)
                self.assertEqual(metadata["pipeline_phase"], phase)
                self.assertEqual(loaded.config.endless, True)
            records = [json.loads(line) for line in (root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            for phase in ("teacher", "teacher_temporal", "student", "student_temporal"):
                training = [record for record in records if record["stage"] == phase and "train_loss" in record]
                self.assertEqual(len(training), 1)
                self.assertTrue(np.isfinite(training[0]["train_loss"]))
            student_training = next(record for record in records if record["stage"] == "student" and "train_loss" in record)
            self.assertIn("flow", student_training)
            self.assertIn("reflow", student_training)
            self.assertTrue((root / "preview.png").exists())
            # A finished phase restores its exact checkpoint without another
            # optimizer update, and cached reflow pairs stay byte-for-byte fixed.
            old_metrics = (root / "metrics.jsonl").read_bytes()
            old_pairs = (root / "reflow_pairs.npz").read_bytes()
            experiment.joint(student, "student", 1, data, validation, teacher, pairs)
            experiment.reflow(teacher, data)
            self.assertEqual(old_metrics, (root / "metrics.jsonl").read_bytes())
            self.assertEqual(old_pairs, (root / "reflow_pairs.npz").read_bytes())


if __name__ == "__main__":
    unittest.main()
