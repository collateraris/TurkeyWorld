"""Independent tiny-world optimizer, counter and checkpoint accumulation tests."""
import copy
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from turkeyworld.models import WorldModelConfig
    from turkeyworld.vjepa_training import VJEPAExperiment

    class TinyWorld(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.core = torch.nn.Module()
            self.core.encoder = torch.nn.Module()
            self.core.encoder.weight = torch.nn.Parameter(torch.tensor([.4, -.3]))
            self.flow_renderer = torch.nn.Linear(1, 1, bias=False)
            self.vjepa_backend = torch.nn.Module()
            self.vjepa_backend.renderer = torch.nn.Linear(1, 1, bias=False)
            self.config = WorldModelConfig(kind="teacher", image_size=128)

    class TrackingEMA:
        def __init__(self, encoder, momentum):
            self.steps, self.calls = 0, 0
            self.shadow = encoder.weight.detach().clone()

        def to(self, device):
            return self

        def update(self, encoder):
            self.steps += 1
            self.calls += 1
            self.shadow = encoder.weight.detach().clone()

        def state_dict(self):
            return {"steps": self.steps, "shadow": self.shadow.clone()}

        def load_state_dict(self, state):
            self.steps = state["steps"]
            self.shadow = state["shadow"].clone()

    class DistinctData:
        def __init__(self, *, random_indices=False, on_sample=None):
            self.calls, self.history = 0, []
            self.random_indices, self.on_sample = random_indices, on_sample

        @staticmethod
        def example(index):
            return {"x": torch.tensor([1., float(index % 5 - 2)]),
                    "y": torch.tensor((index - 11.5) / 2), "index": index}

        def batch(self, batch_size, *, context, horizon, rng):
            if batch_size != 1 or context != 1 or horizon != 1:
                raise AssertionError("Expected the requested batch1/context1/horizon1 protocol")
            index = int(rng.integers(48)) if self.random_indices else self.calls
            self.calls += 1
            self.history.append(index)
            if self.on_sample is not None:
                self.on_sample(self.calls)
            return self.example(index)

    def scalar_loss(model, ema, batch, *, render, **options):
        if render:
            raise AssertionError("Pretraining must not render or enable GAN")
        error = (model.core.encoder.weight * batch["x"]).sum() - batch["y"]
        loss = .5 * error.square()
        return loss, {"component": float(loss.detach()), "target": float(batch["y"]),
                      "all_good": batch["index"] != 7, "unused_text": "not a numeric metric"}, {}

    class Harness(VJEPAExperiment):
        loss_function = staticmethod(scalar_loss)

        def __init__(self, root, *, accumulation=24, completed=1345, honor_stop=False):
            self.root = Path(root)
            self.root.mkdir(parents=True, exist_ok=True)
            self.device = torch.device("cpu")
            self.started = time.time()
            self.args = SimpleNamespace(ema=.996, lr=.05, plateau_patience=6, plateau_delta=.005,
                quality_passes=3, batch=1, context=1, horizon=1, save_every=100000,
                log_every=1, control_run_dir=None)
            if accumulation is not None:
                self.args.gradient_accumulation_steps = accumulation
            self.global_step, self.initial_completed = completed, completed
            self.fragments_seen = completed
            self.rng = np.random.default_rng(17)
            self.extra, self.state = {}, {"teacher_quality": {}}
            self.ema = None
            self._adversarial_trainer = self._adversarial_phase = None
            self.metrics, self.saved, self.validations, self.optimizers = [], [], [], []
            self.data = None
            self.honor_stop = honor_stop

        def restore(self, model, optimizer, phase):
            self.optimizers.append(optimizer)
            if (self.root / "checkpoints" / (phase + ".pt")).exists():
                completed = super().restore(model, optimizer, phase)
            else:
                completed = self.initial_completed
            self.restored_optimizer = copy.deepcopy(optimizer.state_dict())
            return completed

        def save(self, model, optimizer, phase, step, metrics, extra=None):
            path = super().save(model, optimizer, phase, step, metrics, extra)
            self.saved.append({"step": step, "calls": self.data.calls if self.data is not None else 0,
                               "batch": dict(self.state.get("training_batch", {})), "metrics": dict(metrics)})
            return path

        def check_stop(self, *args):
            if self.honor_stop:
                return super().check_stop(*args)

        def _publish_model_sizes(self, model):
            pass

        def metric(self, value):
            self.metrics.append(dict(value))

        def update(self, stage, current, total, label, **values):
            self.state.update(stage=stage, current=current, total=total, **values)

        def validation_report(self, model, *, preview=True):
            self.validations.append((self.global_step, self.data.calls, self.state["training_batch"]["status"]))
            return {"passed": False}


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class GradientAccumulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fit(self, experiment, model, data, steps, **patches):
        experiment.data = data
        with patch("turkeyworld.vjepa_training.EMATargetEncoder", TrackingEMA):
            experiment.fit(model, data, "teacher500vq_world", steps, pretrain=True)

    def reference_sgd(self, weight, indices, lr=.05):
        parameter = torch.nn.Parameter(weight.detach().clone())
        optimizer = torch.optim.SGD([parameter], lr=lr, weight_decay=.01)
        losses = []
        for index in indices:
            batch = DistinctData.example(index)
            losses.append(.5 * ((parameter * batch["x"]).sum() - batch["y"]).square())
        mean = torch.stack(losses).mean()
        mean.backward()
        gradient_norm = float(parameter.grad.norm())
        torch.nn.utils.clip_grad_norm_([parameter], 1., error_if_nonfinite=True)
        optimizer.step()
        return parameter.detach(), float(mean.detach()), gradient_norm

    def test_24_distinct_fragments_match_one_clipped_mean_gradient_update_and_ema(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment, model, data = Harness(folder), TinyWorld(), DistinctData()
            expected, mean_loss, norm = self.reference_sgd(model.core.encoder.weight, range(24))
            self.assertGreater(norm, 1.)  # The once-per-update clipping is exercised.
            optimizers = []
            real_sgd, real_clip = torch.optim.SGD, torch.nn.utils.clip_grad_norm_

            def optimizer(parameters, **options):
                value = real_sgd(parameters, **options)
                optimizers.append(value)
                return value

            with patch("turkeyworld.vjepa_training.torch.optim.AdamW", side_effect=optimizer), \
                 patch("turkeyworld.vjepa_training.torch.nn.utils.clip_grad_norm_", wraps=real_clip) as clip:
                self.fit(experiment, model, data, 1346)
            torch.testing.assert_close(model.core.encoder.weight, expected, rtol=1e-6, atol=1e-7)
            self.assertEqual(clip.call_count, 1)
            self.assertEqual(len(optimizers), 1)
            self.assertEqual(data.history, list(range(24)))
            self.assertEqual(experiment.global_step, 1346)
            self.assertEqual(experiment.ema.calls, 1)
            torch.testing.assert_close(experiment.ema.shadow, model.core.encoder.weight, rtol=0, atol=0)
            self.assertEqual(experiment.fragments_seen, 1345 + 24)
            metrics = experiment.metrics[-1]
            self.assertAlmostEqual(metrics["train_loss"], mean_loss, places=5)
            self.assertAlmostEqual(metrics["component"], mean_loss, places=5)
            self.assertEqual(metrics["target"], 0.)
            self.assertFalse(metrics["all_good"])
            self.assertNotIn("unused_text", metrics)
            self.assertEqual(metrics["effective_batch_size"], 24)
            self.assertEqual(metrics["microbatch_size"], 1)
            self.assertEqual(metrics["gradient_accumulation_steps"], 24)
            self.assertEqual(metrics["loss_aggregation"], "mean_of_microbatches")
            self.assertTrue(all(item["calls"] == 24 and item["batch"]["status"] == "updated"
                                for item in experiment.saved))

    def test_validation_clock_runs_after_full_group_at_global1500(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment = Harness(folder, completed=1499)
            self.fit(experiment, TinyWorld(), DistinctData(), 1500)
            self.assertEqual(experiment.validations, [(1500, 24, "updated")])
            self.assertEqual(experiment.ema.calls, 1)

    def test_missing_accumulation_argument_preserves_legacy_single_sample_update(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment, model, data = Harness(folder, accumulation=None), TinyWorld(), DistinctData()
            expected, mean_loss, _ = self.reference_sgd(model.core.encoder.weight, [0])
            with patch("turkeyworld.vjepa_training.torch.optim.AdamW", side_effect=torch.optim.SGD):
                self.fit(experiment, model, data, 1346)
            torch.testing.assert_close(model.core.encoder.weight, expected, rtol=0, atol=0)
            self.assertEqual(data.calls, 1)
            self.assertEqual(experiment.ema.calls, 1)
            self.assertAlmostEqual(experiment.metrics[-1]["train_loss"], mean_loss)
            self.assertEqual(experiment.state["training_batch"]["effective_batch_size"], 1)

    def test_stop_created_at_micro5_is_deferred_to_safe24_sample_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment = Harness(folder, honor_stop=True)
            data = DistinctData(on_sample=lambda count: (experiment.root / "stop_training").touch() if count == 5 else None)
            with self.assertRaises(SystemExit) as stopped:
                self.fit(experiment, TinyWorld(), data, 1348)
            self.assertEqual(stopped.exception.code, 0)
            self.assertEqual(data.calls, 24)
            self.assertEqual(experiment.global_step, 1346)
            self.assertEqual(experiment.ema.calls, 1)
            self.assertEqual(experiment.state["status"], "stopped")
            checkpoint = torch.load(experiment.root / "checkpoints/teacher500vq_world.pt", weights_only=False)
            self.assertEqual(checkpoint["training_steps"], 1346)
            self.assertEqual(checkpoint["training_counters"]["pending_microbatches"], 0)
            self.assertEqual(checkpoint["training_counters"]["fragments_seen"], 1369)
            self.assertTrue(all(item["calls"] == 24 for item in experiment.saved))
            self.assertNotIn("phase_complete", checkpoint)

    def test_partial_group_save_guard_rejects_before_creating_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment, model = Harness(folder), TinyWorld()
            experiment.publish_accumulation(5, "accumulating")
            optimizer = torch.optim.SGD(model.parameters(), lr=.01)
            with self.assertRaisesRegex(ValueError, "completed optimizer update"):
                experiment.save(model, optimizer, "teacher500vq_world", 1345, {})
            self.assertFalse((experiment.root / "checkpoints/teacher500vq_world.pt").exists())

    def test_real_adamw_resume_keeps_moments_rng_ema_and_new_fragment_counters(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            full, uninterrupted = Harness(root / "full"), TinyWorld()
            full_data = DistinctData(random_indices=True)
            self.fit(full, uninterrupted, full_data, 1347)
            stopped, intermediate = Harness(root / "resume", honor_stop=True), TinyWorld()
            first_data = DistinctData(random_indices=True, on_sample=lambda count:
                (stopped.root / "stop_training").touch() if count == 5 else None)
            with self.assertRaises(SystemExit):
                self.fit(stopped, intermediate, first_data, 1347)
            (stopped.root / "stop_training").unlink()
            resumed, restored = Harness(root / "resume", completed=0), TinyWorld()
            next_data = DistinctData(random_indices=True)
            self.fit(resumed, restored, next_data, 1347)
            self.assertEqual(first_data.history + next_data.history, full_data.history)
            torch.testing.assert_close(restored.core.encoder.weight, uninterrupted.core.encoder.weight, rtol=0, atol=0)
            self.assertEqual(float(resumed.restored_optimizer["state"][0]["step"]), 1.)
            final = resumed.optimizers[-1].state_dict()["state"][0]
            reference = full.optimizers[-1].state_dict()["state"][0]
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(final[key], reference[key], rtol=0, atol=0)
            self.assertEqual(resumed.global_step, 1347)
            self.assertEqual(resumed.fragments_seen, 1393)
            self.assertEqual(resumed.ema.steps, 2)
            self.assertEqual(resumed.ema.calls, 1)
            self.assertEqual(next_data.calls, 24)

    def test_public_constructor_declares_optimizer_step_and_effective_batch_protocol(self):
        from turkeyworld.train_500m import build_parser
        with tempfile.TemporaryDirectory() as folder:
            args = build_parser().parse_args(["--run-dir", folder, "--device", "cpu", "--batch", "1",
                                              "--gradient-accumulation-steps", "24"])
            experiment = VJEPAExperiment(args)
            protocol = experiment.state["training_protocol"]
            self.assertEqual(protocol["gradient_accumulation_steps"], 24)
            self.assertEqual(protocol["microbatch_size"], 1)
            self.assertEqual(protocol["effective_batch_size"], 24)
            self.assertEqual(protocol["step_unit"], "optimizer_updates")
            self.assertEqual(protocol["loss_aggregation"], "mean_of_microbatches")


if __name__ == "__main__":
    unittest.main()
