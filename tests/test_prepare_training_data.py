"""Prepared datasets preserve oracle replay, episode boundaries and split seeds."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from turkeyworld.engine import ENTITY_SLOTS, EVENT_NAMES, Game
from turkeyworld.prepare_training_data import _atomic_json, prepare


class PreparedDataTests(unittest.TestCase):
    def test_atomic_status_write_retries_a_windows_reader_lock(self):
        import os
        replace = os.replace
        calls = []

        def sharing_violation_once(source, target):
            calls.append((source, target))
            if len(calls) == 1:
                raise PermissionError("Windows reader has not released the status file")
            return replace(source, target)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "status.json"
            path.write_text('{"completed_files": 1}', encoding="utf-8")
            with patch("turkeyworld.prepare_training_data.os.replace", side_effect=sharing_violation_once), \
                 patch("turkeyworld.prepare_training_data.time.sleep"):
                _atomic_json(path, {"completed_files": 2})
            self.assertEqual(len(calls), 2)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["completed_files"], 2)

    def test_exact_replay_geometry_splits_and_resume(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            options = dict(rgb_train=5, rgb_val=2, clip_frames=8, size=32,
                           long_train=1, long_val=1, long_frames=12, seed=321)
            summary = prepare(folder, **options)
            self.assertEqual(summary["completed_files"], 9)
            paths = sorted(Path(folder).rglob("*.npz"))
            seeds = {"train": set(), "val": set()}
            before_times = {path: path.stat().st_mtime_ns for path in paths}
            for path in paths:
                with np.load(path, allow_pickle=False) as data:
                    split = str(data["split"])
                    seeds[split].add(int(data["seed"]))
                    game = Game()
                    game.set_state(json.loads(str(data["initial_state"])))
                    np.testing.assert_array_equal(game.state_vector(), data["states"][0])
                    self.assertTrue(game.endless)
                    if path.parts[-3] == "rgb":
                        self.assertEqual(data["frames"].dtype, np.uint8)
                        self.assertEqual(data["frames"].shape, (9, 32, 32, 3))
                        self.assertEqual(data["depth"].shape, (9, 32))
                        self.assertEqual(data["visibility"].shape, (9, ENTITY_SLOTS))
                        np.testing.assert_array_equal(game.render(size=32, overlays=False), data["frames"][0])
                    else:
                        self.assertNotIn("frames", data.files)
                    for index, action in enumerate(data["actions"]):
                        alive_before = np.array([enemy["alive"] for enemy in game.enemies])
                        event = game.step(action, float(data["dt"]))
                        np.testing.assert_array_equal(game.state_vector(), data["states"][index + 1])
                        np.testing.assert_array_equal([float(event[name]) for name in EVENT_NAMES],
                                                      data["events"][index])
                        self.assertEqual(event["hit_id"], int(data["hit_ids"][index]))
                        attacks = np.zeros(ENTITY_SLOTS, dtype=np.float32)
                        attacks[event["attacker_ids"]] = 1
                        np.testing.assert_array_equal(attacks, data["attack_flags"][index])
                        respawns = ~alive_before & np.array([enemy["alive"] for enemy in game.enemies])
                        np.testing.assert_array_equal(respawns, data["respawn_flags"][index])
            self.assertFalse(seeds["train"] & seeds["val"])
            resumed = prepare(folder, **options)
            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(before_times, {path: path.stat().st_mtime_ns for path in paths})
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                prepare(folder, **dict(options, size=64))

    def test_parallel_jobs_write_independent_episodes(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            summary = prepare(folder, rgb_train=2, rgb_val=1, clip_frames=2, size=32,
                              long_train=0, long_val=0, long_frames=2, workers=2)
            self.assertEqual(summary["completed_files"], 3)
            progress = Path(folder, "progress.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(progress), 3)
            self.assertEqual(len({json.loads(line)["job_id"] for line in progress}), 3)


if __name__ == "__main__":
    unittest.main()
