"""Fresh engine behavior provenance cannot silently reuse old random recordings."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from turkeyworld import engine
from turkeyworld.data_expansion import prepare_expansion
from turkeyworld.prepare_training_data import (_generate_job, _perturb_reference_initial_state,
                                               current_behavior_version, prepare)


class DataBehaviorVersionTests(unittest.TestCase):
    @staticmethod
    def options(**changes):
        values = dict(rgb_train=1, rgb_val=1, long_train=0, long_val=0,
                      clip_frames=4, long_frames=3600, size=32, workers=1)
        values.update(changes)
        return values

    @staticmethod
    def run_quiet(function, output, **options):
        with contextlib.redirect_stdout(io.StringIO()):
            return function(output, **options)

    @staticmethod
    def change_episode_version(path, value):
        with np.load(path, allow_pickle=False) as saved:
            payload = {name: saved[name] for name in saved.files if name != "behavior_version"}
        if value is not None:
            payload["behavior_version"] = np.asarray(value)
        with path.open("wb") as stream:
            np.savez_compressed(stream, **payload)

    def test_prepare_stamps_current_engine_in_manifest_npz_progress_and_exact_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = self.run_quiet(prepare, root, **self.options(long_train=1, long_frames=5))
            version = current_behavior_version()
            self.assertEqual(version, engine.BEHAVIOR_VERSION)
            self.assertEqual(summary["behavior_version"], version)
            manifest = json.loads((root / "manifest.json").read_text("utf-8"))
            self.assertEqual(manifest["behavior_version"], version)
            self.assertEqual(manifest["config"]["behavior_version"], version)
            for line in (root / "progress.jsonl").read_text("utf-8").splitlines():
                self.assertEqual(json.loads(line)["behavior_version"], version)
            for path in root.glob("*/*/*.npz"):
                with np.load(path, allow_pickle=False) as episode:
                    self.assertEqual(str(episode["behavior_version"]), version)
                    if "frames" in episode.files:
                        self.assertEqual(str(episode["frame_source"]), "reference_engine")
                    game = engine.Game()
                    game.set_state(json.loads(str(episode["initial_state"])))
                    for index, action in enumerate(episode["actions"]):
                        game.step(action, dt=float(episode["dt"]))
                        np.testing.assert_array_equal(game.state_vector(), episode["states"][index + 1])

    def test_expansion_plan_append_and_frozen_validation_keep_current_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            options = self.options(rgb_train=9, rgb_val=2)
            result = self.run_quiet(prepare_expansion, root, **options)
            self.assertEqual(result["reused_files"], 0)
            val = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in (root / "rgb/val").glob("*.npz")}
            self.run_quiet(prepare_expansion, root, **dict(options, rgb_train=12))
            manifest = json.loads((root / "manifest.json").read_text("utf-8"))
            self.assertIsNone(manifest["config"]["source"])
            self.assertEqual(manifest["config"]["long_frames"], 3600)
            version = current_behavior_version()
            plan = json.loads((root / "expansion_plan.json").read_text("utf-8"))
            self.assertTrue(all(job["behavior_version"] == version for job in plan))
            self.assertEqual(val, {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in val})
            for job in plan:
                with np.load(job["path"], allow_pickle=False) as episode:
                    self.assertEqual(str(episode["behavior_version"]), version)

    def test_old_or_unversioned_sources_are_rejected_without_linking_or_rewriting(self):
        for version in ("legacy_random_v1", None):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "old"
                self.run_quiet(prepare, source, **self.options(rgb_val=0))
                path = next((source / "rgb/train").glob("*.npz"))
                self.change_episode_version(path, version)
                before = (path.stat().st_mtime_ns, path.read_bytes())
                with self.assertRaisesRegex(ValueError, "behavior version"):
                    self.run_quiet(prepare_expansion, root / "new", source=source, **self.options(rgb_val=0))
                self.assertFalse(list((root / "new").glob("*/*/*.npz")))
                self.assertEqual(before, (path.stat().st_mtime_ns, path.read_bytes()))

    def test_changed_engine_version_rejects_both_resume_manifests_before_episode_mutation(self):
        for function in (prepare, prepare_expansion):
            with self.subTest(producer=function.__name__), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.run_quiet(function, root, **self.options())
                before = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in root.rglob("*") if path.is_file()}
                with patch.object(engine, "BEHAVIOR_VERSION", "simulated_future_behavior"), \
                     self.assertRaisesRegex(ValueError, "configuration changed"):
                    self.run_quiet(function, root, **self.options())
                self.assertEqual(before, {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in before})

    def test_version_argument_is_assertion_not_permission_to_mislabel_and_worker_checks_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for function in (prepare, prepare_expansion):
                destination = root / function.__name__
                with self.subTest(producer=function.__name__), self.assertRaisesRegex(ValueError, "running engine"):
                    self.run_quiet(function, destination, **self.options(behavior_version="wrong"))
                self.assertFalse(destination.exists())
            with patch("turkeyworld.prepare_training_data.Game") as game, \
                 self.assertRaisesRegex(ValueError, "running engine"):
                _generate_job({"path": str(root / "bad.npz"), "behavior_version": "wrong"})
            game.assert_not_called()

    def test_perturbed_reference_rejects_old_behavior_before_applying_source_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_quiet(prepare, root, **self.options(rgb_val=0))
            source = next((root / "rgb/train").glob("*.npz"))
            self.change_episode_version(source, "legacy_random_v1")
            game = engine.Game(endless=True)
            with patch.object(game, "set_state") as apply, self.assertRaisesRegex(ValueError, "behavior version"):
                _perturb_reference_initial_state(game, np.random.default_rng(73), source)
            apply.assert_not_called()

    def test_actual_120_second_long_recording_uses_requested_length_not_9000_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_quiet(prepare_expansion, Path(directory),
                **self.options(rgb_train=0, rgb_val=0, long_train=1, long_frames=3600))
            group = result["groups"]["long/train"]
            self.assertEqual(group["eligible_requested_length"], 1)
            self.assertEqual(group["eligible_9000"], 0)
            self.assertEqual(group["transitions"], 3600)
            self.assertTrue(result["full_length_long_targets_satisfied"])
            path = next(Path(directory).glob("long/train/*.npz"))
            with np.load(path, allow_pickle=False) as saved:
                self.assertEqual(len(saved["actions"]), 3600)
                self.assertAlmostEqual(len(saved["actions"]) * float(saved["dt"]), 120.)
                self.assertEqual(str(saved["behavior_version"]), current_behavior_version())


if __name__ == "__main__":
    unittest.main()
