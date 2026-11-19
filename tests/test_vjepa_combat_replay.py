import unittest
from pathlib import Path

import numpy as np

from turkeyworld.vjepa_combat_replay import HORIZON, collection_jobs, collection_plan, window_groups


class CombatReplayTest(unittest.TestCase):
    def test_last_target_transition_and_first_start_are_indexed(self):
        e = np.zeros((HORIZON + 2, 4), np.float32)
        r = np.zeros((len(e), 1), np.float32)
        e[0, 0] = 1
        e[HORIZON, :2] = 1
        e[HORIZON + 1, 3] = 1
        r[1, 0] = 1
        groups = window_groups(e, r)
        np.testing.assert_array_equal(groups["uniform"], [0, 1, 2])
        np.testing.assert_array_equal(groups["shot_miss"], [0])
        np.testing.assert_array_equal(groups["hit"], [1, 2])
        np.testing.assert_array_equal(groups["player_hit"], [2])
        np.testing.assert_array_equal(groups["respawn"], [0, 1])
        self.assertEqual(len(groups["kill"]), 0)

    def test_final_round_seed_does_not_overlap_next_round(self):
        jobs = [list(collection_jobs(Path("round"), collection_plan(n))) for n in (8, 9)]
        for group in jobs:
            self.assertEqual(len(group), 64)
            self.assertTrue(all(j["frames"] == 360 and j["split"] == "train" for j in group))
        self.assertEqual(max(j["seed"] for j in jobs[0]) + 1, min(j["seed"] for j in jobs[1]))


if __name__ == "__main__":
    unittest.main()
