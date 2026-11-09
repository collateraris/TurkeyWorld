"""Generator/discriminator isolation, lazy regularization and exact continuation."""
import io
import copy
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class AdversarialTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        from turkeyworld.adversarial_training import AdversarialTrainer
        cls.Trainer = AdversarialTrainer

    def setUp(self):
        torch.manual_seed(122)
        self.real = torch.rand(1, 3, 64, 64)
        self.condition = torch.rand_like(self.real)
        self.state = torch.zeros(1, 90)
        self.state[:, 3:5] = 1

    def trainer(self, **options):
        config = dict(device="cpu", base_channels=4, warmup_steps=0, ramp_steps=1,
                      r1_interval=16)
        config.update(options)
        return self.Trainer(**config)

    def test_discriminator_update_and_generator_backward_are_isolated(self):
        trainer = self.trainer()
        generator = torch.nn.Conv2d(3, 3, 1)
        predicted = generator(torch.rand_like(self.real)).sigmoid()
        predicted.retain_grad()
        real, condition = self.real.requires_grad_(), self.condition.requires_grad_()
        state = self.state.requires_grad_()
        previous = {name: value.detach().clone() for name, value in trainer.discriminator.named_parameters()}
        loss, metrics = trainer.generator_loss(predicted, real, condition, state)
        self.assertTrue(any(not torch.equal(previous[name], value)
                            for name, value in trainer.discriminator.named_parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in generator.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in trainer.discriminator.parameters()))
        self.assertFalse(trainer.discriminator.training)
        self.assertFalse(any(parameter.requires_grad for parameter in trainer.discriminator.parameters()))
        loss.backward()
        self.assertGreater(float(predicted.grad.abs().sum()), 0)
        self.assertGreater(float(generator.weight.grad.abs().sum()), 0)
        self.assertTrue(all(parameter.grad is None for parameter in trainer.discriminator.parameters()))
        self.assertIsNone(real.grad)
        self.assertIsNone(condition.grad)
        self.assertIsNone(state.grad)
        self.assertTrue(metrics["enabled"])
        self.assertEqual(metrics["step"], 1)
        self.assertGreater(trainer.parameter_count, 0)
        expected = .02 * metrics["generator_loss"] + .05 * metrics["feature_matching"]
        self.assertAlmostEqual(float(loss.detach()), expected, places=6)

    def test_warmup_and_capped_ramp_use_completed_discriminator_updates(self):
        trainer = self.trainer(warmup_steps=2, ramp_steps=2)
        for expected_step, expected_ramp in enumerate((0., 0., .5, 1., 1.), start=1):
            predicted = torch.rand_like(self.real, requires_grad=True)
            loss, metrics = trainer.generator_loss(predicted, self.real, self.condition, self.state)
            self.assertEqual(metrics["step"], expected_step)
            self.assertEqual(metrics["ramp"], expected_ramp)
            self.assertAlmostEqual(metrics["weight"], .02 * expected_ramp)
            self.assertEqual(metrics["enabled"], expected_ramp > 0)
            loss.backward()
            self.assertEqual(float(predicted.grad.abs().sum()) == 0, expected_ramp == 0)

    def test_resume_reproduces_the_next_discriminator_and_generator_update(self):
        first = self.trainer(r1_interval=2)
        first.generator_loss(torch.rand_like(self.real, requires_grad=True),
                             self.real, self.condition, self.state)
        buffer = io.BytesIO()
        torch.save(first.state_dict(), buffer)
        buffer.seek(0)
        second = self.trainer(r1_interval=2)
        second.load_state_dict(torch.load(buffer, weights_only=True))
        self.assertEqual(second.updates, 1)
        fake = torch.rand_like(self.real)
        first_input, second_input = fake.clone().requires_grad_(), fake.clone().requires_grad_()
        first_loss, first_metrics = first.generator_loss(first_input, self.real, self.condition, self.state)
        second_loss, second_metrics = second.generator_loss(second_input, self.real, self.condition, self.state)
        torch.testing.assert_close(first_loss, second_loss, rtol=0, atol=0)
        self.assertEqual(first_metrics, second_metrics)
        first_loss.backward()
        second_loss.backward()
        torch.testing.assert_close(first_input.grad, second_input.grad, rtol=0, atol=0)
        for name, value in first.discriminator.state_dict().items():
            torch.testing.assert_close(value, second.discriminator.state_dict()[name], rtol=0, atol=0)

    def test_lazy_r1_runs_at_the_exact_interval(self):
        from turkeyworld.adversarial_training import r1_penalty
        trainer = self.trainer(r1_interval=2, r1_gamma=1.)
        observed = []
        with patch("turkeyworld.adversarial_training.r1_penalty", wraps=r1_penalty) as regularize:
            for _ in range(4):
                loss, metrics = trainer.generator_loss(torch.rand_like(self.real, requires_grad=True),
                                                       self.real, self.condition, self.state)
                observed.append(metrics["r1"])
                loss.backward()
        self.assertEqual(regularize.call_count, 2)
        self.assertEqual(observed[0], 0.)
        self.assertEqual(observed[2], 0.)
        self.assertGreater(observed[1], 0.)
        self.assertGreater(observed[3], 0.)

    def test_mismatched_resume_configuration_is_rejected(self):
        trainer = self.trainer()
        changed = self.trainer(weight=.03)
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            changed.load_state_dict(trainer.state_dict())

    def test_generator_cast_preserves_bfloat16_input_gradient(self):
        trainer = self.trainer()
        predicted = torch.rand_like(self.real, dtype=torch.bfloat16, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss, _ = trainer.generator_loss(predicted, self.real, self.condition, self.state)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertEqual(predicted.grad.dtype, torch.bfloat16)
        self.assertTrue(bool(torch.isfinite(predicted.grad).all()))
        self.assertGreater(float(predicted.grad.abs().sum()), 0.)

    def test_two_microbatches_average_discriminator_gradients_and_step_only_once(self):
        from turkeyworld.adversarial import discriminator_hinge_loss
        trainer = self.trainer(r1_gamma=0.)
        # SGD makes a missing/extra divide observable in the actual parameter
        # delta; Adam's first-update normalization could conceal that error.
        trainer.optimizer = torch.optim.SGD(trainer.discriminator.parameters(), lr=.01)
        reference = copy.deepcopy(trainer.discriminator)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
        reference_optimizer.zero_grad(set_to_none=True)
        initial = {name: value.detach().clone() for name, value in trainer.discriminator.named_parameters()}
        with patch.object(trainer.optimizer, "step", wraps=trainer.optimizer.step) as update:
            for microstep in range(2):
                fake = torch.rand_like(self.real)
                real, base, state = self.real.clone().requires_grad_(), self.condition.clone().requires_grad_(), self.state.clone().requires_grad_()
                reference.train().requires_grad_(True)
                expected_loss = discriminator_hinge_loss(reference(real.detach(), base.detach(), state.detach()),
                                                          reference(fake, base.detach(), state.detach()))
                (expected_loss / 2).backward()
                reference.eval().requires_grad_(False)
                predicted = fake.clone().requires_grad_()
                loss, metrics = trainer.generator_loss(predicted, real, base, state,
                    accumulation_steps=2, microstep=microstep)
                self.assertEqual(metrics["step"], 1)
                self.assertEqual(metrics["accumulation_microstep"], microstep)
                self.assertEqual(metrics["accumulation_steps"], 2)
                self.assertEqual(metrics["optimizer_updated"], microstep == 1)
                self.assertEqual(trainer.updates, microstep)
                if microstep == 0:
                    self.assertEqual(update.call_count, 0)
                    for (name, value), (_, expected) in zip(trainer.discriminator.named_parameters(), reference.named_parameters()):
                        torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
                        torch.testing.assert_close(value.grad, expected.grad, rtol=0, atol=0)
                    before_generator_backward = [parameter.grad.detach().clone() for parameter in trainer.discriminator.parameters()]
                    (loss / 2).backward()
                    for parameter, expected in zip(trainer.discriminator.parameters(), before_generator_backward):
                        torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)
                else:
                    reference_optimizer.step()
                    for actual, expected in zip(trainer.discriminator.parameters(), reference.parameters()):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(update.call_count, 1)
                    (loss / 2).backward()
                    self.assertTrue(all(parameter.grad is None for parameter in trainer.discriminator.parameters()))
                self.assertGreater(float(predicted.grad.abs().sum()), 0.)
                self.assertIsNone(real.grad)
                self.assertIsNone(base.grad)
                self.assertIsNone(state.grad)

    def test_three_microbatches_keep_warmup_ramp_and_lazy_r1_on_optimizer_clock(self):
        from turkeyworld.adversarial_training import r1_penalty
        trainer = self.trainer(warmup_steps=1, ramp_steps=2, r1_interval=2, r1_gamma=1.)
        steps, completed, ramps, updated, regularization = [], [], [], [], []
        with patch("turkeyworld.adversarial_training.r1_penalty", wraps=r1_penalty) as r1, \
             patch.object(trainer.optimizer, "step", wraps=trainer.optimizer.step) as optimize:
            for group in range(3):
                for microstep in range(3):
                    predicted = torch.rand_like(self.real, requires_grad=True)
                    loss, metrics = trainer.generator_loss(predicted, self.real, self.condition, self.state,
                        accumulation_steps=3, microstep=microstep)
                    (loss / 3).backward()
                    steps.append(metrics["step"])
                    completed.append(metrics["completed_updates"])
                    ramps.append(metrics["ramp"])
                    updated.append(metrics["optimizer_updated"])
                    regularization.append(metrics["r1"])
            self.assertEqual(optimize.call_count, 3)
            self.assertEqual(r1.call_count, 3)  # All three micros of eligible update2.
        self.assertEqual(steps, [1] * 3 + [2] * 3 + [3] * 3)
        self.assertEqual(completed, [0, 0, 1, 1, 1, 2, 2, 2, 3])
        self.assertEqual(ramps, [0.] * 3 + [.5] * 3 + [1.] * 3)
        self.assertEqual(updated, [False, False, True] * 3)
        self.assertTrue(all(value == 0. for value in regularization[:3] + regularization[6:]))
        self.assertTrue(all(value > 0. for value in regularization[3:6]))

    def test_pending_groups_cannot_checkpoint_or_change_order_and_load_clears_pending(self):
        trainer = self.trainer(r1_gamma=0.)
        initial = copy.deepcopy(trainer.state_dict())
        for steps, microstep in ((0, 0), (2, 2), (2, 1), (True, 0)):
            with self.subTest(steps=steps, microstep=microstep), self.assertRaises(ValueError):
                trainer.generator_loss(torch.rand_like(self.real), self.real, self.condition, self.state,
                                       accumulation_steps=steps, microstep=microstep)
        loss, _ = trainer.generator_loss(torch.rand_like(self.real, requires_grad=True), self.real,
                                        self.condition, self.state, accumulation_steps=3, microstep=0)
        loss.backward()
        with self.assertRaisesRegex(RuntimeError, "completed discriminator accumulation"):
            trainer.state_dict()
        for steps, microstep in ((3, 0), (2, 1), (3, 2)):
            with self.subTest(steps=steps, microstep=microstep), self.assertRaises(ValueError):
                trainer.generator_loss(torch.rand_like(self.real), self.real, self.condition, self.state,
                                       accumulation_steps=steps, microstep=microstep)
        trainer.load_state_dict(initial)
        self.assertEqual(trainer.updates, 0)
        self.assertTrue(all(parameter.grad is None for parameter in trainer.discriminator.parameters()))
        self.assertEqual(trainer.state_dict()["updates"], 0)
        # A different group size is allowed after restoring a completed boundary.
        for microstep in range(2):
            loss, _ = trainer.generator_loss(torch.rand_like(self.real, requires_grad=True), self.real,
                self.condition, self.state, accumulation_steps=2, microstep=microstep)
            loss.backward()
        self.assertEqual(trainer.state_dict()["updates"], 1)

    def test_resume_after_complete_group_reproduces_next_accumulated_update(self):
        first = self.trainer(r1_interval=2)
        for microstep in range(3):
            loss, _ = first.generator_loss(torch.rand_like(self.real, requires_grad=True), self.real,
                self.condition, self.state, accumulation_steps=3, microstep=microstep)
            (loss / 3).backward()
        checkpoint = copy.deepcopy(first.state_dict())
        second = self.trainer(r1_interval=2)
        second.load_state_dict(checkpoint)
        for microstep in range(3):
            fake = torch.rand_like(self.real)
            first_input, second_input = fake.clone().requires_grad_(), fake.clone().requires_grad_()
            first_loss, first_metrics = first.generator_loss(first_input, self.real, self.condition,
                self.state, accumulation_steps=3, microstep=microstep)
            second_loss, second_metrics = second.generator_loss(second_input, self.real, self.condition,
                self.state, accumulation_steps=3, microstep=microstep)
            torch.testing.assert_close(first_loss, second_loss, rtol=0, atol=0)
            self.assertEqual(first_metrics, second_metrics)
            (first_loss / 3).backward()
            (second_loss / 3).backward()
            torch.testing.assert_close(first_input.grad, second_input.grad, rtol=0, atol=0)
        self.assertEqual(first.updates, 2)
        for name, value in first.discriminator.state_dict().items():
            torch.testing.assert_close(value, second.discriminator.state_dict()[name], rtol=0, atol=0)

    def test_nonfinite_single_update_remains_retryable_and_failed_group_cannot_checkpoint(self):
        bad_real = self.real.clone()
        bad_real[0, 0, 0, 0] = float("nan")
        trainer = self.trainer(r1_gamma=0.)
        with self.assertRaises(FloatingPointError):
            trainer.generator_loss(torch.rand_like(self.real), bad_real, self.condition, self.state)
        self.assertEqual(trainer.updates, 0)
        loss, _ = trainer.generator_loss(torch.rand_like(self.real, requires_grad=True),
                                        self.real, self.condition, self.state)
        loss.backward()
        self.assertEqual(trainer.state_dict()["updates"], 1)
        loss, _ = trainer.generator_loss(torch.rand_like(self.real, requires_grad=True), self.real,
            self.condition, self.state, accumulation_steps=2, microstep=0)
        loss.backward()
        with self.assertRaises(FloatingPointError):
            trainer.generator_loss(torch.rand_like(self.real), bad_real, self.condition, self.state,
                                   accumulation_steps=2, microstep=1)
        self.assertEqual(trainer.updates, 1)
        with self.assertRaisesRegex(RuntimeError, "completed discriminator accumulation"):
            trainer.state_dict()


if __name__ == "__main__":
    unittest.main()
