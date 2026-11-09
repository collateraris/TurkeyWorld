"""Short dynamics cannot qualify from silent events or empty source coverage."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class DynamicsGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def data(folder, stationary=False, empty_events=False):
        from turkeyworld.jepa_training_utils import SequenceData
        states = np.zeros((6, 90), np.float32)
        states[:, 0] = (1.82 + (0 if stationary else np.arange(6) * .08)) / 16
        states[:, 1], states[:, 3:6] = 2.5 / 16, 1
        enemies = states[:, 8:88].reshape(6, 8, 10)
        enemies[:, :, :2], enemies[:, :, 4:6] = 5.5 / 16, [.6, 1]
        actions = np.zeros((5, 4), np.float32)
        actions[:, 0] = 0 if stationary else 1
        actions[:, 2], actions[:, 3] = [0, 0, 1, 0, 1], [0, 1, 0, 1, 0]
        events, attacks = np.zeros((5, 4), np.float32), np.zeros((5, 8), np.float32)
        if not empty_events:
            events[:, 0] = events[:, 1] = actions[:, 3]
            events[:, 3], attacks[:, 0] = actions[:, 2], actions[:, 2]
        grid = np.zeros((16, 16), np.int8)
        grid[[0, -1], :] = grid[:, [0, -1]] = 1
        np.savez_compressed(Path(folder) / "episode.npz", states=states, actions=actions,
            events=events, attack_flags=attacks, hit_ids=np.where(events[:, 1], 0, -1),
            frames=np.full((6, 16, 16, 3), 70, np.uint8), map=grid, dt=np.asarray(1 / 30), split=np.asarray("val"))
        return SequenceData(folder, split="val")

    @staticmethod
    def model(silent=False, collision=False):
        class World(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.child = torch.nn.Dropout(.5)
                self.config = SimpleNamespace(architecture="vjepa2_ac", image_size=16, simulation_hz=30.)
                self.inputs, self.initializations = [], []

            def encode_map(self, grid):
                return grid[:, None].float()

            def initialize_from_image(self, frames, states, map_features):
                self.initializations.append((frames.clone(), states.clone()))
                torch.rand(1)
                return states.new_ones(len(states), 1, 2)

            def transition(self, state, action, map_features, memory):
                self.inputs.append(state.clone())
                predicted = state.clone()
                predicted[:, 0] += action[:, 0] * (.08 / 16)
                if collision:
                    predicted[:, 0] = .01
                events = state.new_zeros(len(state), 4)
                attacks = torch.zeros(len(state), 8, device=state.device, dtype=torch.bool)
                if not silent:
                    events[:, 1] = action[:, 3]
                    events[:, 3] = action[:, 2]
                    attacks[:, 0] = action[:, 2] > .5
                return {"state": predicted, "memory": memory + 1, "events": events,
                        "accepted_attacks": attacks}

            def render(self, *values):
                raise AssertionError("Dynamics gate must not render or receive future RGB")

        return World()

    def test_correct_moving_combat_predictions_pass_with_real_positive_coverage(self):
        from turkeyworld.dynamics_gate import evaluate_dynamics_gate
        with tempfile.TemporaryDirectory() as folder:
            report = evaluate_dynamics_gate(self.model(), self.data(folder), [(0, 0)])
            self.assertTrue(report["passed"], report["failure_reasons"])
            self.assertEqual(report["metrics"]["hit_f1"], 1.)
            self.assertEqual(report["metrics"]["hit_recall"], 1.)
            self.assertEqual(report["metrics"]["attack_f1"], 1.)
            self.assertEqual(report["metrics"]["collision_fraction"], 0.)
            self.assertEqual(report["counts"]["source_hit_positives"], 2)
            self.assertEqual(report["counts"]["source_attack_positives"], 2)
            self.assertEqual(report["counts"]["source_moving_transitions"], 4)
            self.assertEqual(report["horizon"], 4)

    def test_silent_predictions_fail_even_when_all_positions_match_source(self):
        from turkeyworld.dynamics_gate import evaluate_dynamics_gate
        with tempfile.TemporaryDirectory() as folder:
            report = evaluate_dynamics_gate(self.model(silent=True), self.data(folder), [(0, 0)])
            self.assertFalse(report["passed"])
            self.assertIn("hit_recall", report["failure_reasons"])
            self.assertIn("attack_f1", report["failure_reasons"])
            self.assertLess(report["metrics"]["pose_error_tiles_mean"], 1e-6)

    def test_empty_stationary_source_cannot_manufacture_a_pass(self):
        from turkeyworld.dynamics_gate import evaluate_dynamics_gate
        with tempfile.TemporaryDirectory() as folder:
            report = evaluate_dynamics_gate(self.model(silent=True), self.data(folder, True, True), [(0, 0)])
            self.assertFalse(report["passed"])
            for name in ("source_motion", "source_hits", "source_attacks", "source_coverage"):
                self.assertFalse(report["checks"][name])
            self.assertTrue(np.isfinite(report["score"]))

    def test_future_gt_is_measurement_only_and_modes_rng_are_restored(self):
        from turkeyworld.dynamics_gate import evaluate_dynamics_gate
        with tempfile.TemporaryDirectory() as folder:
            data, model = self.data(folder), self.model()
            model.train()
            model.child.eval()
            rng = torch.get_rng_state().clone()
            first = evaluate_dynamics_gate(model, data, [(0, 0)])
            before = [state.clone() for state in model.inputs]
            data.episodes[0]["states"][2:, 0] += .3
            data.episodes[0]["frames"][2:] = 250
            model.inputs.clear()
            changed = evaluate_dynamics_gate(model, data, [(0, 0)])
            for original, current in zip(before, model.inputs):
                torch.testing.assert_close(original, current, rtol=0, atol=0)
            self.assertNotEqual(first["fingerprint"], changed["fingerprint"])
            self.assertFalse(changed["passed"])
            self.assertTrue(model.training)
            self.assertFalse(model.child.training)
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            self.assertEqual(tuple(model.initializations[0][0].shape), (1, 2, 3, 16, 16))

    def test_geometry_violation_is_measured_without_repairing_prediction(self):
        from turkeyworld.dynamics_gate import evaluate_dynamics_gate
        with tempfile.TemporaryDirectory() as folder:
            model = self.model(collision=True)
            report = evaluate_dynamics_gate(model, self.data(folder), [(0, 0)])
            self.assertFalse(report["passed"])
            self.assertEqual(report["metrics"]["collision_fraction"], 1.)
            self.assertAlmostEqual(float(model.inputs[1][0, 0]), .01, places=6)

    def test_nonfinite_event_outputs_cannot_pass_and_keep_a_finite_score(self):
        from turkeyworld.dynamics_gate import evaluate_dynamics_gate
        with tempfile.TemporaryDirectory() as folder:
            model, data = self.model(), self.data(folder)
            transition = model.transition

            def corrupt(*values):
                result = transition(*values)
                # Corrupt a non-hit event column: hit/attack F1 alone would
                # otherwise stay perfect despite invalid model outputs.
                result["events"][:, 0] = float("nan")
                return result

            with patch.object(model, "transition", side_effect=corrupt):
                report = evaluate_dynamics_gate(model, data, [(0, 0)])
            self.assertFalse(report["passed"])
            self.assertFalse(report["checks"]["complete_predictions"])
            self.assertTrue(np.isfinite(report["score"]))


if __name__ == "__main__":
    unittest.main()
