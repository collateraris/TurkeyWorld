"""Teacher advancement requires checkpoint-backed, held-out visual qualification."""
from argparse import Namespace
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def report(passed=True, fingerprint="fixed-panel"):
    return {"passed": passed, "panel": {"fingerprint": fingerprint},
            "summary": {"frames": 4, "pass_fraction": 1. if passed else .25,
                        "psnr_mean": 40. if passed else 20.,
                        "psnr_p05": 38. if passed else 18.,
                        "ssim_mean": .995 if passed else .8,
                        "mae_mean": .002 if passed else .05}}


class TinyData:
    def __init__(self):
        self.count, self.files = 4, 2
        self.data = {"new": np.zeros((4, 90), dtype=np.float32),
                     "images": np.zeros((4, 16, 16, 3), dtype=np.uint8)}
        self.data["new"][:, 3:5] = 1
        self.grid = torch.zeros(1, 16, 16, dtype=torch.long)

    def batch(self, count, rng, indices=None):
        chosen = np.arange(count) if indices is None else indices
        return {"new": torch.from_numpy(self.data["new"][chosen]),
                "images": torch.from_numpy(self.data["images"][chosen]).permute(0, 3, 1, 2).float(),
                "map": self.grid.expand(len(chosen), -1, -1)}


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class TeacherQualityGateTests(unittest.TestCase):
    def fixture(self, folder):
        from turkeyworld.models import WorldModelConfig
        from turkeyworld.training_pipeline import Experiment

        class TinyTeacher(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = WorldModelConfig(kind="teacher", image_size=128,
                                               endless=True, renderer_mode="flow_matching", flow_steps=12)
                self.flow_renderer = torch.nn.Linear(1, 1)
                self.dynamics = torch.nn.Linear(1, 1)

        args = Namespace(run_dir=Path(folder), device="cpu", seed=313, teacher_steps=1,
                         student_steps=1, temporal_updates=1, reflow_pairs=2, batch=1,
                         lr=.01, log_every=1, save_every=1, validate_every=1,
                         teacher_until_quality=True, quality_frames=4, quality_every=1,
                         quality_passes=3, quality_psnr=35., quality_ssim=.98,
                         quality_mae=2 / 255, quality_fraction=.95, quality_endpoint_batch=1)
        experiment, model = Experiment(args), TinyTeacher()
        return experiment, model, TinyData()

    @staticmethod
    def train_step(model, optimizer, data):
        if any(parameter.requires_grad for parameter in model.dynamics.parameters()):
            raise AssertionError("Visual refinement must freeze dynamics")
        optimizer.zero_grad(set_to_none=True)
        loss = sum(parameter.square().sum() for parameter in model.flow_renderer.parameters())
        loss.backward()
        optimizer.step()
        return {"train_loss": float(loss.detach())}

    def seed_checkpoint(self, experiment, model, *, checked_step=5, fingerprint="fixed-panel",
                        consecutive=2, phase="teacher_quality"):
        optimizer = torch.optim.AdamW(model.flow_renderer.parameters(), lr=experiment.args.lr / 3)
        gate = {"phase": phase, "fingerprint": fingerprint, "last_checked_step": checked_step,
                "consecutive_passes": consecutive, "required_passes": 3, "passed": True,
                "best_score": .0045, "stale_checks": 0,
                "thresholds": experiment.quality_thresholds()}
        return experiment.save(model, optimizer, phase, 5, {}, {"teacher_quality_gate": gate})

    def test_failure_keeps_teacher_training_until_three_consecutive_passes(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            experiment, model, data = self.fixture(folder)
            before = {name: value.clone() for name, value in model.dynamics.state_dict().items()}
            with patch("turkeyworld.visual_quality.evaluate_visual_quality",
                       side_effect=[report(False), report(), report(), report()]) as evaluate, \
                 patch.object(experiment, "_teacher_quality_step", side_effect=self.train_step) as train, \
                 patch.object(experiment, "preview"):
                result = experiment.fit_teacher_until_quality(model, data, data)
            self.assertEqual(evaluate.call_count, 4)
            self.assertEqual(train.call_count, 3)
            self.assertEqual(result["consecutive_passes"], 3)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["last_checked_step"], 3)
            for name, value in model.dynamics.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
            self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
            checkpoint = torch.load(Path(folder, "checkpoints/teacher_quality.pt"), weights_only=False)
            self.assertEqual(checkpoint["teacher_quality_gate"]["consecutive_passes"], 3)

    def test_failed_check_resets_a_partly_successful_streak(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            experiment, model, data = self.fixture(folder)
            sequence = [report(), report(), report(False), report(), report(), report()]
            with patch("turkeyworld.visual_quality.evaluate_visual_quality", side_effect=sequence) as evaluate, \
                 patch.object(experiment, "_teacher_quality_step", side_effect=self.train_step) as train, \
                 patch.object(experiment, "preview"):
                result = experiment.fit_teacher_until_quality(model, data, data)
            self.assertEqual(evaluate.call_count, 6)
            self.assertEqual(train.call_count, 5)
            self.assertEqual(result["consecutive_passes"], 3)

    def test_resume_restores_weights_and_does_not_count_a_repeated_same_step(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            experiment, model, data = self.fixture(folder)
            self.seed_checkpoint(experiment, model)
            saved_weight = model.flow_renderer.weight.detach().clone()
            with torch.no_grad():
                model.flow_renderer.weight.add_(10)
            observed = []

            def evaluate(*values, **options):
                observed.append(model.flow_renderer.weight.detach().clone())
                return report()

            with patch("turkeyworld.visual_quality.evaluate_visual_quality", side_effect=evaluate) as evaluate_mock, \
                 patch.object(experiment, "_teacher_quality_step", side_effect=self.train_step) as train, \
                 patch.object(experiment, "preview"):
                result = experiment.fit_teacher_until_quality(model, data, data)
            torch.testing.assert_close(observed[0], saved_weight, rtol=0, atol=0)
            self.assertEqual(evaluate_mock.call_count, 2)
            self.assertEqual(train.call_count, 1)
            self.assertEqual(result["last_checked_step"], 6)

    def test_changed_panel_or_future_checkpoint_metadata_invalidates_cached_streak(self):
        for name, checked_step, saved_fingerprint in (("panel", 5, "old-panel"),
                                                     ("future", 6, "fixed-panel")):
            with self.subTest(reason=name), tempfile.TemporaryDirectory() as folder, \
                 contextlib.redirect_stdout(io.StringIO()):
                experiment, model, data = self.fixture(folder)
                self.seed_checkpoint(experiment, model, checked_step=checked_step,
                                     fingerprint=saved_fingerprint)
                with patch("turkeyworld.visual_quality.evaluate_visual_quality",
                           side_effect=[report(), report(), report()]) as evaluate, \
                     patch.object(experiment, "_teacher_quality_step", side_effect=self.train_step) as train, \
                     patch.object(experiment, "preview"):
                    result = experiment.fit_teacher_until_quality(model, data, data)
                self.assertEqual(evaluate.call_count, 3)
                self.assertEqual(train.call_count, 2)
                self.assertEqual(result["last_checked_step"], 7)

    def test_post_temporal_phase_rechecks_without_overwriting_trained_dynamics(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            experiment, model, data = self.fixture(folder)
            self.seed_checkpoint(experiment, model, consecutive=3)
            # Represent dynamics learned after the first visual qualification.
            with torch.no_grad():
                model.dynamics.weight.add_(7)
            trained_dynamics = model.dynamics.weight.detach().clone()
            observed = []

            def evaluate(*values, **options):
                observed.append(model.dynamics.weight.detach().clone())
                return report(False) if len(observed) == 1 else report()

            with patch("turkeyworld.visual_quality.evaluate_visual_quality", side_effect=evaluate) as evaluate_mock, \
                 patch.object(experiment, "_teacher_quality_step", side_effect=self.train_step) as train, \
                 patch.object(experiment, "preview"):
                result = experiment.fit_teacher_until_quality(model, data, data,
                    phase="teacher_final_quality", reset_streak=True)
            self.assertEqual(evaluate_mock.call_count, 4)
            self.assertEqual(train.call_count, 3)
            self.assertTrue(Path(folder, "checkpoints/teacher_final_quality.pt").exists())
            for value in observed:
                torch.testing.assert_close(value, trained_dynamics, rtol=0, atol=0)
            self.assertEqual(result["consecutive_passes"], 3)


if __name__ == "__main__":
    unittest.main()
