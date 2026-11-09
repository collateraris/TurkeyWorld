"""CPU coverage of durable 250M selection, world gates and dataset prerequisites."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from turkeyworld.models import WorldModelConfig
    from turkeyworld.train_250m import ExpandedExperiment

    class TinyTeacher(torch.nn.Module):
        def __init__(self, value=0.):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(float(value)))
            self.encoder = torch.nn.Linear(2, 3)
            self.config = WorldModelConfig(kind="teacher")

    class Harness(ExpandedExperiment):
        def __init__(self, root):
            self.root = Path(root)
            (self.root / "checkpoints").mkdir(exist_ok=True)
            self.device = torch.device("cpu")
            self.args = SimpleNamespace(pretrain_steps=6000, world_repair_steps=3000,
                world_patience=2, context=2, horizon=4, visited_clips=256, teacher_size="250m")
            self.global_step = 123
            self.extra, self.state = {}, {}
            self.validation, self.panel = object(), [(0, 0)]
            self.ema = None
            self._adversarial_trainer = None
            self.calls, self.saved = [], []

        def save(self, model, optimizer, phase, step, metrics, extra=None):
            path = self.root / "checkpoints" / (phase + ".pt")
            torch.save({"model": model.state_dict(), "config": asdict(model.config),
                "training_steps": step, "global_training_step": self.global_step,
                **copy.deepcopy(self.extra), **(extra or {})}, path)
            self.saved.append((phase, step, float(model.value.detach())))
            return path

        def fit(self, model, data, phase, steps, **options):
            self.calls.append((phase, steps, options))
            with torch.no_grad():
                model.value.fill_(len(self.calls))
            self.global_step += steps
            self.save(model, None, phase, steps, {})

        def metric(self, values):
            pass

        def update(self, stage, current, total, label, **values):
            self.state.update(stage=stage, current=current, total=total, **values)

    def load_tiny(path, device="cpu"):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        model = TinyTeacher()
        model.load_state_dict(saved["model"])
        return model, saved


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class ExpandedTrainingTests(unittest.TestCase):
    @staticmethod
    def report(score):
        return {"summary": {"mae_mean": score, "ssim_mean": 1.}}

    @staticmethod
    def gate(score, passed=False):
        return {"passed": passed, "score": score, "fingerprint": "fixed-heldout-world",
                "metrics": {"pose_error_tiles_mean": score}}

    def test_global_preview_best_survives_worse_local_plateau_candidate_and_handoff(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment, model = Harness(folder), TinyTeacher(2.)
            phase = "teacher_quality_250m"
            # A global500 preview falls between phase-local patience checks.
            experiment.global_step = 500
            experiment.observe_teacher_validation(model, None, phase, 377, {}, self.report(.05))
            with torch.no_grad():
                model.value.fill_(9.)
            experiment.global_step = 623
            # This can be a numeric best on the LOCAL clock, yet is worse globally.
            experiment.record_teacher_best(model, None, phase, 500, {}, self.report(.15))
            best, metadata = load_tiny(Path(folder, "checkpoints", phase + "_best.pt"))
            self.assertEqual(float(best.value.detach()), 2.)
            self.assertEqual(metadata["teacher_selection"]["global_step"], 500)
            self.assertEqual(len(experiment.saved), 1)
            model.load_state_dict(best.state_dict())  # Inherited fit restores this on plateau.
            experiment.freeze_teacher(model, "plateau", {"best_step": 500}, 3500)
            selected = torch.load(Path(folder, "checkpoints/teacher_for_student_vjepa250.pt"),
                                  weights_only=False)
            self.assertEqual(float(selected["model"]["value"]), 2.)
            self.assertEqual(selected["training_steps"], 377)
            handoff = json.loads(Path(folder, "teacher_handoff_vjepa250.json").read_text("utf-8"))
            self.assertFalse(handoff["quality_goal_passed"])
            self.assertEqual(handoff["selection"]["global_step"], 500)

    def test_failed_world_gates_repair_pretraining_until_pass_before_any_gan_phase(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment, model = Harness(folder), TinyTeacher()
            gates = [self.gate(3.), self.gate(2.), self.gate(1., True)]
            with patch("turkeyworld.dynamics_gate.evaluate_dynamics_gate", side_effect=gates), \
                 patch("turkeyworld.train_250m.load_world_checkpoint", side_effect=load_tiny):
                self.assertTrue(experiment.world_until_ready(model, object()))
            self.assertEqual([(phase, steps) for phase, steps, _ in experiment.calls],
                [("teacher250_world", 6000), ("teacher250_world_001", 3000),
                 ("teacher250_world_002", 3000)])
            self.assertTrue(all(options == {"pretrain": True} for _, _, options in experiment.calls))
            self.assertEqual(experiment.global_step, 12123)
            progress = json.loads(Path(folder, "teacher_world_progress250.json").read_text("utf-8"))
            self.assertTrue(progress["finished"])
            self.assertTrue(progress["gate"]["passed"])
            self.assertFalse(Path(folder, "teacher_handoff_vjepa250.json").exists())

    def test_world_plateau_restores_actual_best_and_durable_resume_skips_training(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment, model = Harness(folder), TinyTeacher()
            with patch("turkeyworld.dynamics_gate.evaluate_dynamics_gate",
                       side_effect=[self.gate(1.), self.gate(2.), self.gate(3.)]), \
                 patch("turkeyworld.train_250m.load_world_checkpoint", side_effect=load_tiny):
                self.assertFalse(experiment.world_until_ready(model, object()))
            self.assertEqual(float(model.value.detach()), 1.)
            progress = json.loads(Path(folder, "teacher_world_progress250.json").read_text("utf-8"))
            self.assertTrue(progress["finished"])
            self.assertIn("teacher_for_student_vjepa250.pt", progress["checkpoint"])
            self.assertEqual(experiment.extra["teacher_selection"]["global_step"], 6123)
            restored, current = Harness(folder), TinyTeacher(-10.)
            with patch("turkeyworld.dynamics_gate.evaluate_dynamics_gate",
                       side_effect=AssertionError("finished gate must not rerun")), \
                 patch("turkeyworld.train_250m.load_world_checkpoint", side_effect=load_tiny):
                self.assertFalse(restored.world_until_ready(current, object()))
            self.assertEqual(float(current.value.detach()), 1.)
            self.assertEqual(restored.global_step, 12123)
            self.assertEqual(restored.calls, [])
            self.assertEqual(restored.state["dynamics_gate"], progress["gate"])

    def test_finished_passed_world_resume_loads_checkpoint_and_global_step(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment = Harness(folder)
            experiment.global_step = 9000
            checkpoint = experiment.save(TinyTeacher(7.), None, "selected_world", 6000, {})
            gate = self.gate(.01, True)
            Path(folder, "teacher_world_progress250.json").write_text(json.dumps(
                {"finished": True, "checkpoint": str(checkpoint), "gate": gate}), encoding="utf-8")
            experiment.global_step = 500
            model = TinyTeacher(-1.)
            with patch("turkeyworld.train_250m.load_world_checkpoint", side_effect=load_tiny):
                self.assertTrue(experiment.world_until_ready(model, object()))
            self.assertEqual(float(model.value.detach()), 7.)
            self.assertEqual(experiment.global_step, 9000)
            self.assertEqual(experiment.calls, [])

    def test_interrupted_world_round_starts_from_prior_checkpoint_and_global_step(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment = Harness(folder)
            experiment.global_step = 9123
            checkpoint = experiment.save(TinyTeacher(6.), None, "teacher250_world_001", 3000, {})
            Path(folder, "teacher_world_progress250.json").write_text(json.dumps(
                {"round": 2, "checkpoint": str(checkpoint)}), encoding="utf-8")
            experiment.global_step = 123
            model = TinyTeacher(-1.)
            before = []
            original_fit = experiment.fit

            def fit(*values, **options):
                before.append((float(model.value.detach()), experiment.global_step))
                original_fit(*values, **options)

            with patch.object(experiment, "fit", side_effect=fit), \
                 patch("turkeyworld.dynamics_gate.evaluate_dynamics_gate", return_value=self.gate(.01, True)), \
                 patch("turkeyworld.train_250m.load_world_checkpoint", side_effect=load_tiny):
                self.assertTrue(experiment.world_until_ready(model, object()))
            self.assertEqual(before, [(6., 9123)])
            self.assertEqual(experiment.calls[0][0], "teacher250_world_002")
            self.assertEqual(experiment.global_step, 12123)

    def test_quality_pass_handoff_describes_current_passed_weights_not_old_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment, model = Harness(folder), TinyTeacher(7.)
            experiment.global_step = 19000
            experiment.extra["teacher_selection"] = {"phase_step": 500, "global_step": 17000,
                                                       "score": .001}
            experiment.state["visual_quality"] = self.report(.002)
            experiment.freeze_teacher(model, "quality_passed", {"best_step": 500}, 2500)
            selected = torch.load(Path(folder, "checkpoints/teacher_for_student_vjepa250.pt"),
                                  weights_only=False)
            self.assertEqual(float(selected["model"]["value"]), 7.)
            self.assertEqual(selected["training_steps"], 2500)
            self.assertEqual(selected["teacher_selection"]["global_step"], 19000)
            self.assertEqual(selected["teacher_selection"]["score"], .002)
            handoff = json.loads(Path(folder, "teacher_handoff_vjepa250.json").read_text("utf-8"))
            self.assertTrue(handoff["quality_goal_passed"])

    def test_generator_count_and_frozen_checkpoint_exclude_ema_and_discriminator(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment, model = Harness(folder), TinyTeacher(4.)
            experiment.ema = SimpleNamespace(parameter_count=98765)
            experiment._adversarial_trainer = SimpleNamespace(parameter_count=87654)
            model.eval().requires_grad_(False)
            experiment._publish_model_sizes(model)
            sizes = experiment.state["model_parameters"]
            self.assertEqual(sizes["teacher_total"], sum(p.numel() for p in model.parameters()))
            self.assertEqual(sizes["teacher_trainable"], 0)
            self.assertEqual(sizes["ema_target_total"], 98765)
            self.assertEqual(sizes["discriminator_total"], 87654)
            self.assertEqual(sizes["target_teacher_total"], 250000000)
            experiment.freeze_teacher(model, "plateau", {"best_step": 6}, 10)
            saved = torch.load(Path(folder, "checkpoints/teacher_for_student_vjepa250.pt"),
                               weights_only=False)
            self.assertEqual(set(saved["model"]), set(model.state_dict()))
            self.assertNotIn("ema_target_encoder", saved)
            self.assertNotIn("adversarial_training", saved)

    def test_complete_expansion_still_waits_for_256_complete_model_visited_clips(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment = Harness(folder)
            directory = Path(folder, "data/rgb/train")
            directory.mkdir(parents=True)
            Path(folder, "data/status.json").write_text('{"status":"complete"}', encoding="utf-8")
            manifest = directory / "model_visited_manifest.json"
            manifest.write_text('{"complete":true,"completed_clips":255}', encoding="utf-8")
            sleeps = []

            def advance(_):
                sleeps.append(experiment.state["current"])
                # Even the requested count is insufficient until the producer closes its manifest.
                manifest.write_text(json.dumps({"complete": len(sleeps) > 1,
                                                "completed_clips": 256}), encoding="utf-8")

            with patch("turkeyworld.train_250m.time.sleep", side_effect=advance):
                experiment.wait_for_data()
            self.assertEqual(sleeps, [255, 256])
            self.assertEqual(experiment.state["model_visited_data"]["completed_clips"], 256)


if __name__ == "__main__":
    unittest.main()
