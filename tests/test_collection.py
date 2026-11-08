"""Long-trajectory format and source-event alignment, without RGB overhead."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from turkeyworld.collect import collect
from turkeyworld.engine import Game


class StateOnlyCollectionTests(unittest.TestCase):
    def test_no_render_or_geometry_is_needed_for_state_only_data(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            with patch.object(Game, "render", side_effect=AssertionError("RGB must not be rendered")), \
                 patch("turkeyworld.collect.build_geometry_targets", side_effect=AssertionError("optional geometry")):
                paths = collect(folder, episodes=1, frames=24, seed=3200, states_only=True)
            with np.load(paths[0], allow_pickle=False) as episode:
                self.assertNotIn("frames", episode.files)
                self.assertNotIn("depth", episode.files)
                self.assertEqual(episode["hit_ids"].shape, (24,))
                self.assertEqual(episode["attack_flags"].shape, (24, 8))
                self.assertEqual(episode["states"].shape, (25, 90))

    def test_event_labels_match_exact_source_replay(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            paths = collect(folder, episodes=1, frames=32, seed=3210, states_only=True)
            with np.load(paths[0], allow_pickle=False) as episode:
                game = Game(seed=3210)
                game.set_state(json.loads(str(episode["initial_state"])))
                for t, action in enumerate(episode["actions"]):
                    result = game.step(action, float(episode["dt"]))
                    np.testing.assert_array_equal(game.state_vector(), episode["states"][t+1])
                    self.assertEqual(result["hit_id"], int(episode["hit_ids"][t]))
                    attackers = np.zeros(8, dtype=np.float32)
                    attackers[result["attacker_ids"]] = 1
                    np.testing.assert_array_equal(attackers, episode["attack_flags"][t])

    def test_explicit_geometry_labels_can_accompany_state_only_data(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            paths = collect(folder, episodes=1, frames=2, seed=3220,
                            states_only=True, geometry=True, policy_name="patrol")
            with np.load(paths[0], allow_pickle=False) as episode:
                self.assertNotIn("frames", episode.files)
                self.assertEqual(episode["depth"].shape, (3, 32))
                self.assertEqual(episode["visibility"].shape, (3, 8))


if __name__ == "__main__":
    unittest.main()
