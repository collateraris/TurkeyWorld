"""Requested-duration qualification retains the independent legacy 300s gate."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from turkeyworld.runtime_checks import RolloutMonitor

try:
    import torch
except ImportError:
    torch = None


def active_monitor(requested_steps=3600, actual_steps=None):
    """Recorded-activity summary fixture; does not substitute for physics tests."""
    monitor = RolloutMonitor(np.zeros((16, 16), dtype=np.uint8), "mixed",
                             requested_steps, endless=True)
    monitor.steps = requested_steps if actual_steps is None else actual_steps
    monitor.path_tiles = monitor.reference_path_tiles = 60.
    monitor.movement_commands = monitor.responsive_movement_steps = 1500
    monitor.reference_movement_steps = 1500
    monitor.visited_cells = {(index, 2) for index in range(10)}
    monitor.npc_path_tiles[:] = monitor.reference_npc_path_tiles[:] = 20.
    monitor.event_counts.update(shot=60, hit=10, kill=2)
    monitor.respawn_count = monitor.reference_respawn_count = 2
    return monitor


class RequestedDurationMonitorTests(unittest.TestCase):
    def test_full_120_seconds_with_activity_passes_without_claiming_300_seconds(self):
        report = active_monitor().report()
        self.assertEqual(report["requested_steps"], 3600)
        self.assertEqual(report["requested_seconds"], 120.)
        self.assertEqual(report["checked_simulated_seconds"], 120.)
        self.assertTrue(report["state_integrity_active_requested_pass"])
        self.assertFalse(report["state_integrity_active_300s_pass"])
        self.assertTrue(report["combat_coverage"]["pass"])
        self.assertTrue(report["npc_responsiveness"]["pass"])

    def test_119_point_9_seconds_does_not_complete_or_reset_requested_session(self):
        monitor = active_monitor(actual_steps=3597)
        first = monitor.report()
        second = monitor.report()
        self.assertEqual(first, second)
        self.assertEqual(monitor.steps, 3597)
        self.assertEqual(first["checked_simulated_seconds"], 119.9)
        self.assertEqual(first["requested_steps"], 3600)
        self.assertFalse(first["completed_requested_steps"])
        self.assertFalse(first["state_integrity_active_requested_pass"])

    def test_duration_cannot_bypass_existing_activity_integrity_or_survival_gates(self):
        changes = {
            "idle": lambda m: setattr(m, "scenario", "idle"),
            "frozen_player": lambda m: setattr(m, "path_tiles", 0.),
            "frozen_npcs": lambda m: m.npc_path_tiles.fill(0.),
            "no_hits": lambda m: m.event_counts.update(hit=-m.event_counts["hit"]),
            "no_kills": lambda m: m.event_counts.update(kill=-m.event_counts["kill"]),
            "no_model_respawn": lambda m: setattr(m, "respawn_count", 0),
            "no_source_respawn": lambda m: setattr(m, "reference_respawn_count", 0),
            "dead_player": lambda m: setattr(m, "model_player_alive", False),
            "dead_source_player": lambda m: setattr(m, "reference_player_alive", False),
            "early_terminal": lambda m: setattr(m, "terminal_step", 3597),
            "early_source_terminal": lambda m: setattr(m, "reference_terminal_step", 3597),
            "invalid_transition": lambda m: setattr(m, "first_failure", {"code": "wall_tunneling"}),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                monitor = active_monitor()
                change(monitor)
                self.assertFalse(monitor.report()["state_integrity_active_requested_pass"])
        self.assertFalse(active_monitor().report(reference_completed=False)
                         ["state_integrity_active_requested_pass"])

    def test_full_300_seconds_keeps_legacy_result(self):
        monitor = active_monitor(9000)
        report = monitor.report()
        self.assertEqual(report["requested_seconds"], 300.)
        self.assertTrue(report["state_integrity_active_requested_pass"])
        self.assertTrue(report["state_integrity_active_300s_pass"])
        monitor.terminal_step = 8999
        failed = monitor.report()
        self.assertFalse(failed["state_integrity_active_requested_pass"])
        self.assertFalse(failed["state_integrity_active_300s_pass"])


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class RequestedDurationEvaluatorTests(unittest.TestCase):
    @staticmethod
    def model():
        class TinyWorld(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.config = SimpleNamespace(simulation_hz=30, image_size=16,
                    architecture="legacy_cnn", transition_mode="event_v1", endless=True)

            def encode_map(self, grid):
                return grid.float()

            def initial_memory(self, batch, device, dtype):
                return torch.zeros(batch, 1, device=device, dtype=dtype)

            def transition(self, state, action, map_features, memory):
                result = state.clone()
                result[:, -1] += 1 / 9000
                return {"state": result, "memory": memory + 1,
                        "features": state.new_zeros(len(state), 1),
                        "events": state.new_zeros(len(state), 4)}

            def render(self, state, features, map_features):
                return state.new_zeros(len(state), 3, 16, 16)

        return TinyWorld()

    def evaluate(self, *, integrity=True, psnr=35., render_every=1, oracle_rejection=False):
        from turkeyworld.evaluate_rollout import evaluate_episode
        monitor = active_monitor(2)
        monitor.observe = Mock(return_value=[])
        if not integrity:
            monitor.first_failure = {"code": "wall_tunneling"}
        initial = np.zeros(STATE_DIM, dtype=np.float32)
        initial[3:6] = 1.
        tape = {"map": np.zeros((16, 16), dtype=np.uint8), "initial_state": initial,
                "actions": np.zeros((2, 4), dtype=np.float32), "reference_states": None,
                "reference_events": None, "reference_completed": True,
                "reference_terminal_step": None, "controller": "controlled_test_fixture"}
        oracle = Mock()
        oracle.render.return_value = np.zeros((16, 16, 3), dtype=np.uint8)
        if oracle_rejection:
            oracle.set_state_vector.side_effect = ValueError("invalid predicted state")
        args = Namespace(seconds=2 / 30, scenario="mixed", no_shadow_reference=False,
                         max_failures=50, render_every=render_every, min_render_psnr=psnr,
                         continue_on_failure=True, progress_every=0)
        with patch("turkeyworld.evaluate_rollout.build_action_tape", return_value=tape), \
             patch("turkeyworld.evaluate_rollout.RolloutMonitor", return_value=monitor), \
             patch("turkeyworld.evaluate_rollout.Game", return_value=oracle):
            report = evaluate_episode(self.model(), {}, args, seed=1)
        self.assertEqual(report["engine_corrections"], 0)
        self.assertEqual(report["model_resets_during_rollout"], 0)
        return report

    def test_requested_pass_requires_both_integrity_and_configured_sampled_visual_gate(self):
        good = self.evaluate()
        self.assertTrue(good["state_integrity_active_requested_pass"])
        self.assertTrue(good["requested_active_pass"])
        self.assertFalse(good["300s_active_pass"])
        self.assertFalse(self.evaluate(integrity=False)["requested_active_pass"])
        self.assertFalse(self.evaluate(psnr=121.)["requested_active_pass"])
        self.assertFalse(self.evaluate(psnr=None)["requested_active_pass"])
        self.assertFalse(self.evaluate(render_every=0)["requested_active_pass"])
        self.assertFalse(self.evaluate(oracle_rejection=True)["requested_active_pass"])

    def test_cli_reports_requested_target_and_separate_legacy_aggregate(self):
        from turkeyworld.evaluate_rollout import main
        for seconds, steps, legacy in [(120, 3600, False), (300, 9000, True)]:
            with self.subTest(seconds=seconds), TemporaryDirectory() as folder:
                output = Path(folder) / "report.json"
                episode = {"checked_simulated_seconds": seconds, "integrity_pass": True,
                           "qualifying_active_session": True, "first_failure": None,
                           "state_integrity_active_requested_pass": True, "requested_active_pass": True,
                           "state_integrity_active_300s_pass": legacy, "300s_active_pass": legacy}
                with patch("turkeyworld.models.load_world_checkpoint", return_value=(self.model(), {})), \
                     patch("turkeyworld.evaluate_rollout.evaluate_episode", return_value=episode) as evaluate, \
                     patch("torch.set_num_threads"), redirect_stdout(StringIO()):
                    report = main(["--world", "tiny.pt", "--seconds", str(seconds),
                                   "--episodes", "2", "--output", str(output)])
                self.assertEqual(evaluate.call_count, 2)
                self.assertEqual(report["target_seconds"], seconds)
                self.assertEqual(report["target_steps"], steps)
                self.assertEqual(report["requested_active_pass_count"], 2)
                self.assertEqual(report["state_integrity_active_requested_pass_count"], 2)
                self.assertEqual(report["300s_active_pass_count"], 2 if legacy else 0)
                self.assertEqual(report["state_integrity_active_300s_pass_count"], 2 if legacy else 0)
                self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)


if __name__ == "__main__":
    unittest.main()
