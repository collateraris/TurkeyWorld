"""Small real autograd tests for JEPA prediction and teacher/student coupling."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class VJEPATrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def model(width=4):
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(3, width)
                self.calls = []

            def forward(self, video):
                self.calls.append(video.detach().clone())
                return self.projection(video.mean((-1, -2)))[:, :, None]

        class Core(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = Encoder()
                self.predictor = torch.nn.Linear(width, width)
                self.context_states = []

            def encode_video(self, video):
                return self.encoder(video)

            def initial_memory_from_latents(self, latents, states):
                self.context_states.append(states.detach().clone())
                return latents[:, -1].clone()

        class Flow(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.tensor(.01))
                self.loss_calls, self.sample_calls = [], []

            def loss(self, target, base, state, noise=None, target_endpoint=None):
                self.loss_calls.append({"noise": None if noise is None else noise.detach().clone(),
                    "endpoint": None if target_endpoint is None else target_endpoint.detach().clone()})
                desired = target if target_endpoint is None else target_endpoint
                value = torch.nn.functional.l1_loss(base + self.bias, desired)
                return value, {"flow_velocity": value.detach()}

            def sample(self, base, state, noise=None):
                self.sample_calls.append(None if noise is None else noise.detach().clone())
                return (base + self.bias + (0 if noise is None else noise * .01)).clamp(0, 1)

        class World(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(simulation_hz=30.)
                self.core, self.flow_renderer = Core(), Flow()
                self.motion = torch.nn.Linear(width, STATE_DIM)
                self.feature = torch.nn.Linear(width, 128)
                self.target = torch.nn.Linear(width, ENTITY_SLOTS+1)
                self.attack = torch.nn.Linear(width, ENTITY_SLOTS)
                self.state_inputs, self.memory_inputs, self.state_outputs, self.memory_outputs = [], [], [], []

            @property
            def vjepa_backend(self):
                return self

            def encode_map(self, grid):
                return grid[:, None].float()

            def initialize_from_image(self, frames, states, map_features):
                return self.core.initial_memory_from_latents(self.core.encode_video(frames), states)

            def initialize_from_latents(self, latents, states, map_features):
                return self.core.initial_memory_from_latents(latents, states)

            def transition(self, state, action, map_features, memory):
                self.state_inputs.append(state.detach().clone())
                self.memory_inputs.append(memory.detach().clone())
                prediction = self.core.predictor(memory) + .01 * action[:, None, :1]
                pooled = prediction.mean(1)
                next_state = state + .01 * self.motion(pooled)
                self.state_outputs.append(next_state.detach().clone())
                self.memory_outputs.append(prediction.detach().clone())
                return {"state": next_state, "memory": prediction, "latent_prediction": prediction,
                        "features": self.feature(pooled), "target_logits": self.target(pooled),
                        "attack_logits": self.attack(pooled)}

            def render_base(self, state, features, map_features):
                color = (features[:, :3] + state[:, :1]).sigmoid()
                return color[:, :, None, None].expand(-1, -1, 16, 16)

        return World()

    @staticmethod
    def batch():
        frames = torch.rand(2, 6, 3, 16, 16)
        states = torch.zeros(2, 6, STATE_DIM)
        states[:, :2, 0] = .1
        states[:, 2:, 0] = torch.tensor([10., 20., 30., 40.])
        states[:, :, 3:5] = 1
        return {"frames": frames, "context_frames": frames[:, :2], "target_frames": frames[:, 2:],
                "states": states, "target_states": states[:, 2:], "previous_states": states[:, 1:-1],
                "actions": torch.ones(2, 4, 4), "events": torch.zeros(2, 4, 4),
                "hits": torch.full((2, 4), -1, dtype=torch.long),
                "attacks": torch.zeros(2, 4, ENTITY_SLOTS), "map": torch.zeros(2, 16, 16, dtype=torch.long)}

    def setUp(self):
        torch.manual_seed(72)

    def test_short_stream_is_autoregressive_and_ema_target_has_no_gradients(self):
        from turkeyworld.jepa_training_utils import EMATargetEncoder
        from turkeyworld.vjepa_training import short_rollout_loss
        model, batch = self.model(), self.batch()
        ema = EMATargetEncoder(model.core.encoder)
        loss, metrics, endpoint = short_rollout_loss(model, ema, batch)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(len(model.state_inputs), 4)
        torch.testing.assert_close(model.state_inputs[0], batch["states"][:, 1])
        for index in range(1, 4):
            torch.testing.assert_close(model.state_inputs[index], model.state_outputs[index - 1])
            torch.testing.assert_close(model.memory_inputs[index], model.memory_outputs[index - 1])
        self.assertEqual(model.core.encoder.calls[0].shape[1], 2)
        self.assertEqual(ema.encoder.calls[0].shape[1], 6)
        loss.backward()
        self.assertGreater(float(model.core.encoder.projection.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.core.predictor.weight.grad.abs().sum()), 0)
        self.assertTrue(all(parameter.grad is None for parameter in ema.parameters()))
        self.assertIn("jepa", metrics)
        self.assertEqual(endpoint["state"].shape, (2, STATE_DIM))

    def test_changing_future_supervision_changes_loss_but_not_generated_endpoint(self):
        from turkeyworld.jepa_training_utils import EMATargetEncoder
        from turkeyworld.vjepa_training import short_rollout_loss
        model, batch = self.model(), self.batch()
        ema = EMATargetEncoder(model.core.encoder)
        first_loss, _, first = short_rollout_loss(model, ema, batch)
        changed = {name: value.clone() for name, value in batch.items()}
        changed["states"][:, 2:] += 100
        changed["target_states"] += 100
        changed["frames"][:, 2:] = .9
        changed["target_frames"][:] = .9
        changed["events"][:] = 1
        changed["hits"][:] = 0
        changed["attacks"][:] = 1
        second_loss, _, second = short_rollout_loss(model, ema, changed)
        torch.testing.assert_close(first["state"], second["state"], rtol=0, atol=0)
        torch.testing.assert_close(first["base"], second["base"], rtol=0, atol=0)
        self.assertGreater(float((first_loss - second_loss).abs().detach()), 1.)

    def test_student_teacher_reflow_uses_same_noise_and_teacher_stays_frozen(self):
        from turkeyworld.jepa_training_utils import EMATargetEncoder
        from turkeyworld.vjepa_training import short_rollout_loss
        student = self.model(width=4)
        teacher = self.model(width=6).eval().requires_grad_(False)
        ema = EMATargetEncoder(student.core.encoder)
        loss, metrics, _ = short_rollout_loss(student, ema, self.batch(), teacher=teacher,
                                             distill_endpoint=True)
        self.assertIn("teacher_state", metrics)
        self.assertIn("teacher_events", metrics)
        self.assertIn("reflow", metrics)
        self.assertEqual(len(teacher.flow_renderer.sample_calls), 1)
        self.assertEqual(len(student.flow_renderer.loss_calls), 2)
        torch.testing.assert_close(teacher.flow_renderer.sample_calls[0],
                                   student.flow_renderer.loss_calls[1]["noise"], rtol=0, atol=0)
        self.assertIsNotNone(student.flow_renderer.loss_calls[1]["endpoint"])
        self.assertEqual(tuple(teacher.core.context_states[0].shape), (2, 2, STATE_DIM))
        loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))
        self.assertGreater(float(student.core.predictor.weight.grad.abs().sum()), 0)

    def test_visual_encoder_pretrain_can_skip_flow_but_still_receives_prediction_gradients(self):
        from turkeyworld.jepa_training_utils import EMATargetEncoder
        from turkeyworld.vjepa_training import short_rollout_loss
        model = self.model()
        ema = EMATargetEncoder(model.core.encoder)
        loss, metrics, _ = short_rollout_loss(model, ema, self.batch(), render=False)
        self.assertNotIn("flow", metrics)
        self.assertEqual(model.flow_renderer.loss_calls, [])
        loss.backward()
        self.assertTrue(bool(torch.isfinite(model.core.encoder.projection.weight.grad).all()))
        self.assertGreater(float(model.core.encoder.projection.weight.grad.abs().sum()), 0.)


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class StudentContinuationTests(unittest.TestCase):
    @staticmethod
    def model(value):
        from turkeyworld.models import WorldModelConfig
        model = torch.nn.Linear(1, 1)
        model.config = WorldModelConfig(kind="student", image_size=128, architecture="vjepa2_ac")
        with torch.no_grad():
            model.weight.fill_(value)
            model.bias.zero_()
        return model

    def fixture(self, directory, outcomes):
        from turkeyworld.training_pipeline import model_fingerprint
        from turkeyworld.vjepa_training import VJEPAExperiment
        teacher = self.model(7).eval().requires_grad_(False)
        fingerprint = model_fingerprint(teacher)
        owner = self

        class FakeExperiment(VJEPAExperiment):
            def __init__(self):
                self.root = Path(directory)
                (self.root / "checkpoints").mkdir()
                self.device = torch.device("cpu")
                self.args = SimpleNamespace(repair_steps=2000)
                self.state = {"teacher_fingerprint": fingerprint}
                self.rng = np.random.default_rng(17)
                self.global_step = 0
                self.ema = None
                self._adversarial_trainer = self._adversarial_phase = None
                self.events = []
                self.outcomes = iter(outcomes)

            def fit(self, student, data, phase, steps, teacher=None):
                self.events.append(("fit", phase, steps, float(student.weight.detach())))
                if any(parameter.requires_grad for parameter in teacher.parameters()):
                    raise AssertionError("Teacher must stay frozen during student repairs")
                with torch.no_grad():
                    student.weight.add_(.1)
                self.global_step += steps

            def long_curriculum(self, student, kind):
                self.events.append(("long", kind))
                with torch.no_grad():
                    student.weight.add_(.2)
                self.global_step += 10
                owner.save_checkpoint(self, self.root / "checkpoints" / f"{kind}_vjepa_temporal.pt", student)

            def evaluate_9000(self, checkpoint, kind, round_index=0):
                self.events.append(("evaluate", round_index, str(checkpoint)))
                result = {"passed": next(self.outcomes), "episodes": 2,
                          "round": round_index, "checkpoint": str(checkpoint)}
                (self.root / "student_vjepa_evaluation.json").write_text(json.dumps(result), encoding="utf-8")
                self.state["evaluation"] = self.state["student_evaluation"] = result
                return result

            def metric(self, value):
                self.events.append(("metric", value["stage"]))

        experiment = FakeExperiment()

        def load(path, device="cpu"):
            experiment.events.append(("load", str(path)))
            saved = torch.load(path, map_location="cpu", weights_only=False)
            restored = owner.model(0)
            restored.load_state_dict(saved["model"])
            return restored, saved

        return experiment, teacher, load

    @staticmethod
    def save_checkpoint(experiment, path, model):
        torch.save({"model": model.state_dict(), "global_training_step": experiment.global_step,
                    "teacher_fingerprint": experiment.state["teacher_fingerprint"],
                    "rng_numpy": experiment.rng.bit_generator.state,
                    "rng_torch": torch.get_rng_state()}, path)

    def test_failed_baseline_repairs_only_student_then_uses_a_new_evaluation_round(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment, teacher, load = self.fixture(folder, [False, True])
            student = self.model(1)
            initial = experiment.root / "checkpoints" / "baseline.pt"
            self.save_checkpoint(experiment, initial, student)
            teacher_before = {key: value.clone() for key, value in teacher.state_dict().items()}
            with patch("turkeyworld.vjepa_training.load_world_checkpoint", side_effect=load):
                result = experiment.qualify_student(student, teacher, object(), initial)
            self.assertTrue(result["passed"])
            timeline = [event[:2] for event in experiment.events if event[0] in {"fit", "long", "evaluate"}]
            self.assertEqual(timeline, [("evaluate", 0), ("fit", "student_vjepa_repair_001"),
                                        ("long", "student_repair_001"), ("evaluate", 1)])
            self.assertEqual(next(event for event in experiment.events if event[0] == "fit")[2], 2000)
            progress = json.loads((experiment.root / "student_continuation_vjepa.json").read_text())
            self.assertEqual(progress["status"], "qualified")
            self.assertEqual(progress["round"], 1)
            self.assertIn("student_repair_001_vjepa_temporal.pt", progress["last_checkpoint"])
            self.assertEqual(experiment.state["playable_checkpoint"], progress["last_checkpoint"])
            for name, value in teacher.state_dict().items():
                torch.testing.assert_close(value, teacher_before[name], rtol=0, atol=0)
            self.assertFalse(any(parameter.requires_grad for parameter in teacher.parameters()))

    def test_qualified_pointer_restores_selected_weights_without_training_or_evaluation(self):
        from turkeyworld.training_pipeline import model_fingerprint
        with tempfile.TemporaryDirectory() as folder:
            experiment, teacher, load = self.fixture(folder, [])
            qualified = self.model(9)
            checkpoint = experiment.root / "checkpoints" / "qualified.pt"
            experiment.global_step = 9001
            self.save_checkpoint(experiment, checkpoint, qualified)
            experiment.global_step = 0
            progress = {"status": "qualified", "round": 3, "last_checkpoint": str(checkpoint),
                        "teacher_fingerprint": experiment.state["teacher_fingerprint"],
                        "student_fingerprint": model_fingerprint(qualified)}
            (experiment.root / "student_continuation_vjepa.json").write_text(json.dumps(progress))
            (experiment.root / "student_vjepa_evaluation.json").write_text(json.dumps({"passed": True, "round": 3}))
            student = self.model(1)
            with patch("turkeyworld.vjepa_training.load_world_checkpoint", side_effect=load):
                result = experiment.qualify_student(student, teacher, object(), checkpoint)
            self.assertTrue(result["passed"])
            self.assertEqual([event[0] for event in experiment.events], ["load"])
            torch.testing.assert_close(student.weight, qualified.weight, rtol=0, atol=0)
            self.assertEqual(experiment.global_step, 9001)
            self.assertFalse(student.training)
            self.assertEqual(experiment.state["playable_checkpoint"], str(checkpoint))

    def test_resumed_second_repair_loads_prior_final_then_resumes_unique_phases(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment, teacher, load = self.fixture(folder, [True])
            previous = self.model(4)
            checkpoint = experiment.root / "checkpoints" / "student_repair_001_vjepa_temporal.pt"
            experiment.global_step = 12000
            self.save_checkpoint(experiment, checkpoint, previous)
            experiment.global_step = 0
            progress = {"status": "repairing", "round": 2, "last_checkpoint": str(checkpoint),
                        "teacher_fingerprint": experiment.state["teacher_fingerprint"]}
            (experiment.root / "student_continuation_vjepa.json").write_text(json.dumps(progress))
            student = self.model(0)
            with patch("turkeyworld.vjepa_training.load_world_checkpoint", side_effect=load):
                result = experiment.qualify_student(student, teacher, object(), checkpoint)
            self.assertTrue(result["passed"])
            events = [event for event in experiment.events if event[0] in {"load", "fit", "long", "evaluate"}]
            self.assertEqual(events[0], ("load", str(checkpoint)))
            self.assertEqual(events[1], ("fit", "student_vjepa_repair_002", 2000, 4.))
            self.assertEqual(events[2], ("long", "student_repair_002"))
            self.assertEqual(events[3][0:2], ("evaluate", 2))
            self.assertGreaterEqual(experiment.global_step, 12000)


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class VJEPAFitPlateauTests(unittest.TestCase):
    def fixture(self, directory):
        from turkeyworld.vjepa_training import VJEPAExperiment

        class Flow(torch.nn.Module):
            def sample(self, base, state):
                return base

        class ScalarModel(torch.nn.Module):
            def __init__(self, value=1.):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(value))
                self.core = torch.nn.Module()
                self.core.encoder = torch.nn.Linear(1, 1)
                self.flow_renderer = Flow()
                self.config = SimpleNamespace(kind="teacher")

        class FakeExperiment(VJEPAExperiment):
            def __init__(self):
                self.root = Path(directory)
                (self.root / "checkpoints").mkdir()
                self.device = torch.device("cpu")
                self.args = SimpleNamespace(ema=.996, lr=.001, plateau_patience=6, plateau_delta=.005,
                    quality_passes=3, batch=1, context=2, horizon=4, save_every=100000, log_every=100000)
                self.global_step = 123
                self.rng = np.random.default_rng(1)
                self.state = {"teacher_quality": {}}
                self.extra = {}
                self.validations, self.freezes = [], []

            def restore(self, model, optimizer, phase):
                path = self.root / "checkpoints" / f"{phase}.pt"
                if not path.exists():
                    self.extra = {}
                    return 0
                saved = torch.load(path, weights_only=False)
                model.load_state_dict(saved["model"])
                self.extra = saved["extra"]
                self.global_step = max(self.global_step, saved["global_training_step"])
                return saved["training_steps"]

            def save(self, model, optimizer, phase, step, metrics, extra=None):
                path = self.root / "checkpoints" / f"{phase}.pt"
                torch.save({"model": model.state_dict(), "training_steps": step,
                            "global_training_step": self.global_step,
                            "extra": dict(self.extra, **(extra or {}))}, path)
                return path

            def check_stop(self, *values):
                pass

            def _setup_adversarial(self, phase):
                pass

            def _publish_model_sizes(self, model):
                pass

            def _adversarial_loss(self, generated, *values):
                return generated.new_zeros(()), {}

            def validation_report(self, model, *, preview=True):
                self.validations.append((self.global_step, preview))
                # The first actual check is best; all later checks are worse.
                mae = .05 if len(self.validations) == 1 else .15
                report = {"passed": False, "summary": {"mae_mean": mae, "ssim_mean": .9,
                    "pass_fraction": 0., "frames": 256}, "panel": {"fingerprint": "unchanged-heldout"}}
                self.state["teacher_quality"].update(report["summary"])
                return report

            def quality_thresholds(self):
                return {"psnr": 35., "ssim": .98, "mae": 2 / 255, "required_fraction": .95}

            def freeze_teacher(self, model, reason, plateau, step):
                self.freezes.append((float(model.value.detach()), reason, plateau, step))
                (self.root / "teacher_handoff_vjepa125.json").write_text(json.dumps({"reason": reason}))

            def metric(self, value):
                pass

            def update(self, stage, current, total, label, **values):
                self.state.update(stage=stage, **values)

        experiment, model = FakeExperiment(), ScalarModel()

        def loss(model, ema, batch, **options):
            base = model.value.expand(1, 3, 1, 1)
            endpoint = {"base": base, "state": base.new_zeros(1, STATE_DIM), "target": torch.zeros_like(base)}
            return -model.value, {}, endpoint

        def load(path, device="cpu"):
            saved = torch.load(path, weights_only=False)
            restored = ScalarModel()
            restored.load_state_dict(saved["model"])
            return restored, saved

        data = SimpleNamespace(batch=lambda *values, **options: {})
        return experiment, model, data, loss, load

    def test_six_stale_distinct_checks_stop_at_3000_and_restore_numeric_best_before_handoff(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment, model, data, loss, load = self.fixture(folder)
            with patch("turkeyworld.vjepa_training.short_rollout_loss", side_effect=loss) as train, \
                 patch("turkeyworld.vjepa_training.load_world_checkpoint", side_effect=load):
                experiment.fit(model, data, "teacher_quality_125m", quality=True)
            self.assertEqual(train.call_count, 3000)
            self.assertEqual(experiment.global_step, 3123)
            self.assertEqual([step for step, preview in experiment.validations if preview],
                             [500, 1000, 1500, 2000, 2500, 3000])
            self.assertEqual([step for step, preview in experiment.validations if not preview],
                             [124, 623, 1123, 1623, 2123, 2623, 3123])
            best = torch.load(Path(folder, "checkpoints/teacher_quality_125m_best.pt"), weights_only=False)
            last = torch.load(Path(folder, "checkpoints/teacher_quality_125m.pt"), weights_only=False)
            self.assertEqual(best["training_steps"], 1)
            self.assertGreater(float(last["model"]["value"]), float(best["model"]["value"]))
            selected, reason, tracker, handoff_step = experiment.freezes[0]
            self.assertEqual(selected, float(best["model"]["value"]))
            self.assertEqual(reason, "plateau")
            self.assertEqual(tracker["bad_checks"], 6)
            self.assertEqual(tracker["checks"], 7)
            self.assertEqual(handoff_step, 3000)
            self.assertFalse(experiment.state["teacher_quality_gate"]["passed"])

    def test_completed_quality_resume_reconstructs_missing_handoff_from_best_without_training(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment, model, data, loss, load = self.fixture(folder)
            experiment.extra = {"phase_complete": True, "teacher_handoff_reason": "plateau",
                                "teacher_plateau": {"best_step": 500, "bad_checks": 6}}
            optimizer = torch.optim.AdamW(model.parameters())
            with torch.no_grad():
                model.value.fill_(2.)
            experiment.save(model, optimizer, "teacher_quality_125m_best", 500, {})
            with torch.no_grad():
                model.value.fill_(9.)
            experiment.save(model, optimizer, "teacher_quality_125m", 3500, {})
            with patch("turkeyworld.vjepa_training.short_rollout_loss", side_effect=AssertionError("must not retrain")), \
                 patch("turkeyworld.vjepa_training.load_world_checkpoint", side_effect=load):
                experiment.fit(model, data, "teacher_quality_125m", quality=True)
            self.assertEqual(experiment.validations, [])
            self.assertEqual(experiment.freezes[0][:2], (2., "plateau"))
            self.assertEqual(experiment.freezes[0][3], 3500)
            self.assertTrue(Path(folder, "teacher_handoff_vjepa125.json").exists())
            self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
