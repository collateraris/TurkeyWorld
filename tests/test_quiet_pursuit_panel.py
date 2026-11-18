import tempfile
import unittest
from pathlib import Path

import numpy as np

from turkeyworld.validate_quiet_pursuit import KINDS, N, prepare


class QuietPursuitPanelTest(unittest.TestCase):
    def test_fresh_holdout_rejects_invalid_starts_and_seed_mismatch(self):
        with tempfile.TemporaryDirectory() as path:
            _, manifest = prepare(Path(path), seed=645912043)
            self.assertEqual(manifest["seed"], 645912043)
            self.assertEqual(len(manifest["windows"]), len(KINDS) * N)
            with self.assertRaisesRegex(ValueError, "Wrong panel"):
                prepare(Path(path))

    def test_locked_noncombat_source_events_and_coverage(self):
        with tempfile.TemporaryDirectory() as path:
            eps, manifest = prepare(Path(path))
            again, loaded = prepare(Path(path))
            self.assertEqual(loaded, manifest)
            self.assertEqual(len({r["trajectory_sha256"] for r in eps}), len(KINDS) * N)
            for a, b in zip(eps, again):
                np.testing.assert_array_equal(a["states"], b["states"])
                self.assertEqual(a["states"].shape, (184, 20))
                self.assertEqual(a["actions"].shape, (180, 4))
                self.assertFalse(a["actions"][:, 3].any())
                self.assertTrue((a["states"][:, [4, 12, 13]] == a["states"][0, [4, 12, 13]]).all())
                if a["kind"] == "occluded_turn":
                    self.assertGreaterEqual(a["occluded_turn_frames"], 6)
                if a["kind"] == "strafe_facing":
                    self.assertGreaterEqual(a["visible_frames"], 60)
                if a["kind"] == "distance_change":
                    self.assertGreaterEqual(a["gap_max"] - a["gap_min"], 2.0)
