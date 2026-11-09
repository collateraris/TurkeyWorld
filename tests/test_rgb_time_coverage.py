"""Seeded RGB initial clocks cover spawn phases without claiming survival."""
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from turkeyworld.data_expansion import prepare_expansion
from turkeyworld.engine import Game
from turkeyworld.prepare_training_data import _generate_job, prepare


class RGBTimeCoverageTests(unittest.TestCase):
    @staticmethod
    def job(root, seed=173, kind="rgb", policy="movement", **changes):
        relative = f"{kind}/train/episode_{seed:010d}.npz"
        job = {"path": str(root / relative), "relative_path": relative, "job_id": f"{kind}/train/{seed}",
               "seed": seed, "kind": kind, "split": "train", "policy": policy,
               "frames": 4, "size": 32, "hr_size": 0, "hz": 30, "endless": True,
               "long_geometry": False, "rgb_time_span": 120.}
        job.update(changes)
        return job

    @staticmethod
    def quiet(function, *values, **options):
        with contextlib.redirect_stdout(io.StringIO()):
            return function(*values, **options)

    @staticmethod
    def options(**changes):
        options = dict(rgb_train=1, rgb_val=1, long_train=1, long_val=0,
                       clip_frames=4, long_frames=4, size=32, workers=1, rgb_time_span=120.)
        options.update(changes)
        return options

    def test_seeded_offsets_observe_all_eight_phases_and_only_real_clip_duration_is_counted(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            phases = set()
            for seed in range(64):
                job = self.job(root, seed=seed)
                result = _generate_job(job)
                with np.load(job["path"], allow_pickle=False) as saved:
                    offset = float(saved["initial_time_offset"])
                    initial = float(saved["initial_game_time"])
                    phases.add(int(initial // 3) % 8)
                    self.assertGreaterEqual(offset, 0.)
                    self.assertLess(offset, 120.)
                    self.assertAlmostEqual(offset*30, round(offset*30))
                    self.assertEqual(initial, offset)
                    self.assertEqual(json.loads(str(saved["initial_state"]))["time"], initial)
                    self.assertAlmostEqual(float(saved["states"][0, -1])*300, initial, delta=1e-5)
                    np.testing.assert_allclose(np.diff(saved["states"][:, -1]), 1/9000, rtol=0, atol=3e-8)
                    self.assertEqual(float(saved["rgb_time_span"]), 120.)
                    self.assertEqual(result["initial_game_time"], initial)
                    self.assertAlmostEqual(result["simulated_seconds"], 4/30)
            self.assertEqual(phases, set(range(8)))

    def test_deterministic_seed_exact_replay_rgb_and_warmup_start_metadata(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs = [self.job(root / name, policy="return_memory", return_offset=6) for name in ("a", "b")]
            for job in jobs:
                _generate_job(job)
            with np.load(jobs[0]["path"], allow_pickle=False) as first, \
                 np.load(jobs[1]["path"], allow_pickle=False) as second:
                for key in ("states", "actions", "events", "frames", "initial_time_offset", "initial_game_time"):
                    np.testing.assert_array_equal(first[key], second[key])
                self.assertEqual(int(first["warmup_steps"]), 6)
                self.assertAlmostEqual(float(first["initial_game_time"]), float(first["initial_time_offset"])+6/30)
                self.assertAlmostEqual(float(first["states"][0, -1])*300,
                                       float(first["initial_game_time"]), delta=1e-5)
                game = Game()
                game.set_state(json.loads(str(first["initial_state"])))
                for index in range(5):
                    np.testing.assert_array_equal(game.state_vector(), first["states"][index])
                    np.testing.assert_array_equal(game.render(size=32, overlays=False), first["frames"][index])
                    if index < 4:
                        game.step(first["actions"][index], dt=float(first["dt"]))

    def test_long_initialization_stays_at_zero_and_default_zero_preserves_rgb_sequence(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            long_job = self.job(root, kind="long", policy="survival")
            with patch.object(Game, "render", side_effect=AssertionError("States-only long recording must not render")):
                _generate_job(long_job)
            with np.load(long_job["path"], allow_pickle=False) as saved:
                self.assertEqual(float(saved["initial_game_time"]), 0.)
                self.assertEqual(float(saved["initial_time_offset"]), 0.)
                self.assertEqual(float(saved["rgb_time_span"]), 0.)
            zero = self.job(root / "zero", rgb_time_span=0.)
            omitted = self.job(root / "omitted")
            omitted.pop("rgb_time_span")
            for job in (zero, omitted):
                _generate_job(job)
            with np.load(zero["path"], allow_pickle=False) as first, np.load(omitted["path"], allow_pickle=False) as second:
                for key in ("states", "actions", "events", "frames"):
                    np.testing.assert_array_equal(first[key], second[key])
                self.assertEqual(float(first["initial_game_time"]), 0.)

    def test_both_producers_propagate_configs_jobs_progress_and_reject_changed_span(self):
        for producer in (prepare, prepare_expansion):
            with self.subTest(producer=producer.__name__), TemporaryDirectory() as temporary:
                root = Path(temporary)
                options = self.options()
                self.quiet(producer, root, **options)
                manifest = json.loads((root / "manifest.json").read_text("utf-8"))
                self.assertEqual(manifest["config"]["rgb_time_span"], 120.)
                for line in (root / "progress.jsonl").read_text("utf-8").splitlines():
                    record = json.loads(line)
                    self.assertEqual(record["rgb_time_span"], 120. if record["kind"] == "rgb" else 0.)
                if producer is prepare_expansion:
                    plan = json.loads((root / "expansion_plan.json").read_text("utf-8"))
                    self.assertTrue(all(job["rgb_time_span"] == (120. if job["kind"] == "rgb" else 0.) for job in plan))
                before = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in root.rglob("*") if path.is_file()}
                self.quiet(producer, root, **options)
                for path in root.rglob("*.npz"):
                    self.assertEqual((path.stat().st_mtime_ns, path.read_bytes()), before[path])
                after_resume = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in before}
                with self.assertRaisesRegex(ValueError, "configuration changed"):
                    self.quiet(producer, root, **dict(options, rgb_time_span=0.))
                self.assertEqual(after_resume, {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in before})

    def test_reuse_and_existing_npz_refuse_different_time_span_without_relabeling(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            options = self.options(rgb_val=0, long_train=0)
            self.quiet(prepare_expansion, source, **options)
            episode = next((source / "rgb/train").glob("*.npz"))
            original = (episode.stat().st_mtime_ns, episode.read_bytes())
            with self.assertRaisesRegex(ValueError, "time span"):
                self.quiet(prepare_expansion, root / "wrong", source=source, **dict(options, rgb_time_span=0.))
            self.assertFalse(list((root / "wrong").glob("rgb/*/*.npz")))
            self.quiet(prepare_expansion, root / "same", source=source, **options)
            with patch("turkeyworld.prepare_training_data.Game", side_effect=AssertionError("Resume must not construct a game")), \
                 self.assertRaisesRegex(ValueError, "time span"):
                _generate_job(self.job(root, rgb_time_span=0., path=str(episode), seed=10000000))
            self.assertEqual((episode.stat().st_mtime_ns, episode.read_bytes()), original)

    def test_invalid_time_spans_are_rejected(self):
        for producer in (prepare, prepare_expansion):
            for span in (-1., float("inf"), float("nan")):
                with self.subTest(producer=producer.__name__, span=span), TemporaryDirectory() as temporary:
                    with self.assertRaisesRegex(ValueError, "time span"):
                        self.quiet(producer, Path(temporary) / "bad", **self.options(rgb_time_span=span))


if __name__ == "__main__":
    unittest.main()
