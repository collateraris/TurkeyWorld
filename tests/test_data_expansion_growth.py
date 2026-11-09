"""Count increases append new oracle clips without changing the existing corpus."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from turkeyworld.data_expansion import prepare_expansion
import turkeyworld.data_expansion as expansion


class ExpansionGrowthTests(unittest.TestCase):
    @staticmethod
    def options(**changes):
        values = dict(rgb_train=2, rgb_val=1, long_train=0, long_val=0,
                      clip_frames=8, long_frames=4, size=32, seed=10_000_000, workers=1)
        values.update(changes)
        return values

    @staticmethod
    def prepare(output, options):
        with contextlib.redirect_stdout(io.StringIO()):
            return prepare_expansion(output, **options)

    @staticmethod
    def read(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def identity(path):
        return path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_train_growth_preserves_existing_jobs_validation_and_unplanned_oracle_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            options = self.options()
            initial = self.prepare(root, options)
            self.assertEqual(initial["completed"], 3)
            original_plan = self.read(root / "expansion_plan.json")
            original_jobs = {job["job_id"]: job for job in original_plan}
            # Independent DAgger/oracle additions are outside expansion_plan.json.
            oracle = root / "rgb/train/user_oracle_visited_000001.npz"
            np.savez_compressed(oracle, provenance=np.asarray("user_oracle"), values=np.arange(7))
            panel = root / "heldout_panel.json"
            panel.write_text(json.dumps({"seed": 73, "windows": [[0, 1], [0, 4]]}), encoding="utf-8")
            preserved = [Path(job["path"]) for job in original_plan] + [oracle, panel]
            for index, path in enumerate(preserved):
                fixed_ns = 1_700_000_000_000_000_000 + index * 1_000_000
                os.utime(path, ns=(fixed_ns, fixed_ns))
            identities = {path: self.identity(path) for path in preserved}
            old_val = list((root / "rgb/val").glob("*.npz"))

            expanded_options = dict(options, rgb_train=5)
            grown = self.prepare(root, expanded_options)
            self.assertEqual(grown["status"], "complete")
            self.assertEqual(grown["completed"], 6)
            self.assertEqual(grown["groups"]["rgb/train"]["files"], 5)
            self.assertEqual(grown["groups"]["rgb/val"]["files"], 1)
            self.assertEqual(identities, {path: self.identity(path) for path in preserved})
            self.assertEqual(old_val, list((root / "rgb/val").glob("*.npz")))
            new_plan = self.read(root / "expansion_plan.json")
            by_id = {job["job_id"]: job for job in new_plan}
            self.assertEqual(len(new_plan), 6)
            self.assertEqual(len(by_id), len(new_plan))
            self.assertEqual(original_jobs, {key: by_id[key] for key in original_jobs})
            self.assertEqual(len({job["path"] for job in new_plan}), 6)
            self.assertEqual(len({job["seed"] for job in new_plan}), 6)
            self.assertNotIn(str(oracle), {job["path"] for job in new_plan})
            manifest = self.read(root / "manifest.json")
            self.assertEqual(manifest["config"]["targets"]["rgb_train"], 5)
            self.assertEqual(manifest["config"]["targets"]["rgb_val"], 1)
            self.assertEqual(manifest["targets"], manifest["config"]["targets"])

            all_identities = {Path(job["path"]): self.identity(Path(job["path"])) for job in new_plan}
            repeated = self.prepare(root, expanded_options)
            self.assertEqual(repeated["completed"], 6)
            self.assertEqual(new_plan, self.read(root / "expansion_plan.json"))
            self.assertEqual(all_identities, {path: self.identity(path) for path in all_identities})

    def test_count_shrink_and_noncount_config_changes_reject_without_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            options = self.options()
            self.prepare(root, options)
            paths = [root / "manifest.json", root / "expansion_plan.json", root / "status.json"]
            paths.extend(root.glob("rgb/*/*.npz"))
            unchanged = {path: self.identity(path) for path in paths}
            changes = [{"rgb_train": 1}, {"rgb_val": 0}, {"seed": 10_000_001},
                       {"size": 64}, {"hz": 15}, {"clip_frames": 9},
                       {"long_frames": 5}, {"source": root / "different_source"}]
            for change in changes:
                with self.subTest(change=change), self.assertRaises(ValueError):
                    self.prepare(root, dict(options, **change))
                self.assertEqual(unchanged, {path: self.identity(path) for path in paths})

    def test_interrupted_growth_manifest_write_resumes_without_duplicate_or_lost_jobs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            options = self.options()
            self.prepare(root, options)
            original = self.read(root / "expansion_plan.json")
            preserved = {Path(job["path"]): self.identity(Path(job["path"])) for job in original}
            atomic = expansion._atomic_json
            interrupted = []

            def crash_once(path, value):
                if (Path(path).name == "manifest.json" and not interrupted
                        and value.get("config", {}).get("targets", {}).get("rgb_train") == 5):
                    interrupted.append(True)
                    raise OSError("simulated interrupted growth manifest write")
                return atomic(path, value)

            with patch("turkeyworld.data_expansion._atomic_json", side_effect=crash_once), \
                 self.assertRaisesRegex(OSError, "simulated interrupted growth"):
                self.prepare(root, dict(options, rgb_train=5))
            self.assertEqual(interrupted, [True])
            resumed = self.prepare(root, dict(options, rgb_train=5))
            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(resumed["completed"], 6)
            plan = self.read(root / "expansion_plan.json")
            self.assertEqual(len(plan), 6)
            self.assertEqual(len({job["job_id"] for job in plan}), 6)
            self.assertEqual(len({job["path"] for job in plan}), 6)
            by_id = {job["job_id"]: job for job in plan}
            self.assertEqual(original, [by_id[job["job_id"]] for job in original])
            self.assertEqual(preserved, {path: self.identity(path) for path in preserved})

    def test_long_and_validation_targets_cannot_shrink_either(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            options = dict(self.options(), rgb_train=1, rgb_val=1, long_train=1, long_val=1,
                           clip_frames=4, long_frames=4)
            first = self.prepare(root, options)
            self.assertEqual(first["status"], "complete")
            plan = self.read(root / "expansion_plan.json")
            for name in ("rgb_train", "rgb_val", "long_train", "long_val"):
                with self.subTest(target=name), self.assertRaises(ValueError):
                    self.prepare(root, dict(options, **{name: 0}))
                self.assertEqual(plan, self.read(root / "expansion_plan.json"))


if __name__ == "__main__":
    unittest.main()
