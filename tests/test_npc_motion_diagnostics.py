"""NPC proposal telemetry is masked, detached and separate from training loss."""
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class NPCMotionDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def states(batch=1):
        previous = torch.zeros(batch, 90)
        previous[:, 3:5] = 1
        return previous, previous.clone()

    def test_continuing_alive_gt_mask_controls_all_fractions_and_outputs_are_detached(self):
        from turkeyworld.aligned_training import npc_motion_diagnostics
        previous, target = self.states(2)
        old = previous[:, 8:88].reshape(2, 8, 10)
        expected = target[:, 8:88].reshape(2, 8, 10)
        old[0, :4, 5] = expected[0, :4, 5] = 1
        old[1, :2, 5] = expected[1, :2, 5] = 1
        expected[0, 2:4, 0] = .001
        expected[1, 1, 0] = .001
        # Revived/dead/half-alive slots are excluded even with huge proposals.
        expected[0, 4, 5] = 1
        old[0, 5, 5] = 1
        old[1, 3, 5], expected[1, 3, 5] = .5, 1
        proposal = torch.zeros(2, 8, 2)
        proposal[0, (0, 2), 0] = .01
        proposal[1, 1, 0] = .01
        proposal[:, 4:, :] = 100
        proposal[1, 3, :] = 100
        previous.requires_grad_()
        target.requires_grad_()
        proposal.requires_grad_()
        originals = [value.detach().clone() for value in (previous, target, proposal)]
        with torch.enable_grad():
            metrics = npc_motion_diagnostics({"proposed_npc_displacement": proposal}, previous, target)
        self.assertAlmostEqual(float(metrics["npc_saturated_fraction"]), .5)
        self.assertAlmostEqual(float(metrics["npc_stopped_target_fraction"]), .5)
        self.assertAlmostEqual(float(metrics["npc_stopped_saturated_fraction"]), 1/6)
        for value in metrics.values():
            self.assertFalse(value.requires_grad)
            self.assertIsNone(value.grad_fn)
            self.assertTrue(torch.isfinite(value))
        for original, value in zip(originals, (previous, target, proposal)):
            torch.testing.assert_close(value, original, rtol=0, atol=0)
            self.assertIsNone(value.grad)

    def test_no_continuing_npcs_returns_finite_zero_even_when_all_revive(self):
        from turkeyworld.aligned_training import npc_motion_diagnostics
        previous, target = self.states(2)
        target[:, 8:88].reshape(2, 8, 10)[:, :, 5] = 1
        metrics = npc_motion_diagnostics({"proposed_npc_displacement": torch.full((2, 8, 2), 100.)},
                                          previous, target)
        self.assertEqual(len(metrics), 3)
        self.assertTrue(all(float(value) == 0. and torch.isfinite(value) for value in metrics.values()))

    def test_missing_or_none_proposal_omits_optional_metrics(self):
        from turkeyworld.aligned_training import npc_motion_diagnostics
        previous, target = self.states()
        self.assertEqual(npc_motion_diagnostics({}, previous, target), {})
        self.assertEqual(npc_motion_diagnostics({"proposed_npc_displacement": None}, previous, target), {})

    def test_euclidean_saturation_limit_uses_hz_and_tolerance(self):
        from turkeyworld.aligned_training import npc_motion_diagnostics
        previous, target = self.states()
        previous[0, 8 + 5] = target[0, 8 + 5] = 1
        proposal = torch.zeros(1, 8, 2)
        limit = 1.2 / 30 / 16
        proposal[0, 0, 0] = limit
        self.assertEqual(float(npc_motion_diagnostics({"proposed_npc_displacement": proposal}, previous, target)
                               ["npc_saturated_fraction"]), 0.)
        proposal[0, 0, 0] = limit * (1 + 2e-5)
        self.assertEqual(float(npc_motion_diagnostics({"proposed_npc_displacement": proposal}, previous, target)
                               ["npc_saturated_fraction"]), 1.)
        proposal[0, 0] = limit * .8  # Both axes individually below the cap, norm above.
        self.assertEqual(float(npc_motion_diagnostics({"proposed_npc_displacement": proposal}, previous, target)
                               ["npc_saturated_fraction"]), 1.)
        proposal[0, 0] = torch.tensor([limit * .75, 0.])
        self.assertEqual(float(npc_motion_diagnostics({"proposed_npc_displacement": proposal}, previous, target,
                                                     simulation_hz=30.)["npc_saturated_fraction"]), 0.)
        self.assertEqual(float(npc_motion_diagnostics({"proposed_npc_displacement": proposal}, previous, target,
                                                     simulation_hz=60.)["npc_saturated_fraction"]), 1.)

    def test_proposal_telemetry_changes_without_changing_any_auxiliary_loss_or_gradients(self):
        from turkeyworld.aligned_training import aligned_auxiliary_step_loss, npc_motion_diagnostics
        previous, target = self.states()
        old, expected = previous[:, 8:88].reshape(1, 8, 10), target[:, 8:88].reshape(1, 8, 10)
        old[:, :3, 5] = expected[:, :3, 5] = 1
        expected[:, 3, 5] = 1  # Exercise real spawn supervision as well.
        predicted = (target + .001).requires_grad_()
        common = {"state": predicted, "target_logits": torch.zeros(1, 9, requires_grad=True),
            "attack_logits": torch.zeros(1, 8, requires_grad=True),
            "event_logits": torch.zeros(1, 4, requires_grad=True),
            "spawn_logits": torch.zeros(1, 8, 256, requires_grad=True)}
        low, high = torch.zeros(1, 8, 2, requires_grad=True), torch.ones(1, 8, 2, requires_grad=True)
        first, second = dict(common, proposed_npc_displacement=low), dict(common, proposed_npc_displacement=high)
        hits, attacks, events = torch.tensor([0]), torch.ones(1, 8), torch.ones(1, 4)
        before = aligned_auxiliary_step_loss(first, previous, target, hits, attacks, events)
        after = aligned_auxiliary_step_loss(second, previous, target, hits, attacks, events)
        self.assertEqual(set(before), {"events", "target", "attack", "state", "motion", "spawn"})
        self.assertEqual(set(before), set(after))
        for name in before:
            torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
        low_metrics = npc_motion_diagnostics(first, previous, target)
        high_metrics = npc_motion_diagnostics(second, previous, target)
        self.assertEqual(float(low_metrics["npc_saturated_fraction"]), 0.)
        self.assertEqual(float(high_metrics["npc_saturated_fraction"]), 1.)
        sum(after.values()).backward()
        self.assertIsNone(low.grad)
        self.assertIsNone(high.grad)
        self.assertGreater(float(predicted.grad.abs().sum()), 0.)
        self.assertGreater(float(common["target_logits"].grad.abs().sum()), 0.)
        self.assertGreater(float(common["spawn_logits"].grad.abs().sum()), 0.)

    def test_short_metrics_are_optional_and_never_added_to_total_objective(self):
        # Reuse the established tiny causal-world fixture, not a real model.
        from test_aligned_training import AlignedTrainingTests
        from turkeyworld.aligned_training import aligned_short_rollout_loss
        from turkeyworld.jepa_training_utils import EMATargetEncoder
        torch.manual_seed(92)
        model, batch = AlignedTrainingTests.model(), AlignedTrainingTests.batch()
        ema = EMATargetEncoder(model.core.encoder)
        original_transition = model.transition
        baseline, omitted, _ = aligned_short_rollout_loss(model, ema, batch, render=False)
        self.assertFalse(any(name.startswith("npc_") for name in omitted))
        proposal_size = [0.]

        def transition(*args, **kwargs):
            result = original_transition(*args, **kwargs)
            result["proposed_npc_displacement"] = result["state"].new_full((1, 8, 2), proposal_size[0])
            return result

        model.transition = transition
        low, low_metrics, _ = aligned_short_rollout_loss(model, ema, batch, render=False)
        proposal_size[0] = 1.
        high, high_metrics, _ = aligned_short_rollout_loss(model, ema, batch, render=False)
        self.assertEqual(low_metrics["npc_saturated_fraction"], 0.)
        self.assertEqual(high_metrics["npc_saturated_fraction"], 1.)
        torch.testing.assert_close(baseline, low, rtol=0, atol=0)
        torch.testing.assert_close(baseline, high, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
