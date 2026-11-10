"""Expanded clips retain exact oracle labels, split integrity and real quotas."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from turkeyworld.data_expansion import prepare_expansion
from turkeyworld.engine import EVENT_NAMES, Game
from turkeyworld.prepare_training_data import _generate_job, prepare


class DataExpansionTests(unittest.TestCase):
    def test_balanced_clips_replay_and_delayed_attacks_survive_context_alignment(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            status = prepare_expansion(folder, rgb_train=9, rgb_val=8, long_train=0, long_val=0,
                clip_frames=32, long_frames=4, size=32, workers=1)
            self.assertEqual(status["status"], "complete")
            seeds = {"train": set(), "val": set()}
            for path in Path(folder).glob("rgb/*/*.npz"):
                with np.load(path, allow_pickle=False) as episode:
                    split, policy = str(episode["split"]), str(episode["policy"])
                    seeds[split].add(int(episode["seed"]))
                    game = Game()
                    game.set_state(json.loads(str(episode["initial_state"])))
                    for index, action in enumerate(episode["actions"]):
                        event = game.step(action, dt=float(episode["dt"]))
                        np.testing.assert_array_equal(game.state_vector(), episode["states"][index+1])
                        np.testing.assert_array_equal([float(event[key]) for key in EVENT_NAMES], episode["events"][index])
                    if policy == "attack":
                        # One enemy can attack once in this32/30-second clip;
                        # the delayed event must still be after context.
                        self.assertEqual(int(episode["events"][1:, 3].sum()), 1)
                    if policy == "kill":
                        self.assertGreater(int(episode["events"][1:, 2].sum()), 0)
                    if policy == "perturbed":
                        self.assertEqual(split, "train")
                        self.assertFalse(bool(episode["model_visited"]))
                        self.assertEqual(str(episode["initialization_source"]), "perturbed_reference_state")
            self.assertFalse(seeds["train"] & seeds["val"])
            self.assertGreater(status["groups"]["rgb/val"]["attack_events_after_first_action"], 0)
            self.assertGreater(status["policies"]["rgb/train/collision"]["blocked_movement_steps"], 0)
            self.assertNotIn("rgb/val/perturbed", status["policies"])

    def test_hardlink_reuse_and_resume_never_rewrite_source_or_completed_clips(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            source, expanded = root/"source", root/"expanded"
            prepare(source, rgb_train=1, rgb_val=1, long_train=0, long_val=0,
                    clip_frames=8, long_frames=4, size=32)
            options = dict(source=source, rgb_train=3, rgb_val=2, long_train=0, long_val=0,
                           clip_frames=8, long_frames=4, size=32, workers=1)
            first = prepare_expansion(expanded, **options)
            linked = expanded/"rgb/train/episode_0000000000.npz"
            original = source/"rgb/train/episode_0000000000.npz"
            self.assertTrue(os.path.samefile(linked, original))
            times = {path: path.stat().st_mtime_ns for path in expanded.glob("rgb/*/*.npz")}
            resumed = prepare_expansion(expanded, **options)
            self.assertEqual(first["completed"], resumed["completed"])
            self.assertEqual(times, {path: path.stat().st_mtime_ns for path in times})
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                prepare_expansion(expanded, **dict(options, size=64))

    def test_short_recording_is_preserved_and_an_extra_full_long_episode_is_added(self):
        calls = []

        def source(job):
            calls.append(job)
            return _generate_job(dict(job, frames=2)) if len(calls) == 1 else _generate_job(job)

        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            with patch("turkeyworld.data_expansion._safe_generate", side_effect=source):
                result = prepare_expansion(folder, rgb_train=0, rgb_val=0, long_train=1, long_val=0,
                    clip_frames=4, long_frames=4, size=32, workers=1)
            group = result["groups"]["long/train"]
            self.assertEqual(group["files"], 2)
            self.assertEqual(group["eligible_requested_length"], 1)
            self.assertEqual(group["transitions"], 6)
            self.assertTrue(result["full_length_long_targets_satisfied"])


if __name__ == "__main__":
    unittest.main()
