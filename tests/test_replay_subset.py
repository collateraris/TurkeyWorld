"""Offline old/new replay membership, quota arithmetic and immutable contents."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from turkeyworld.data_expansion import TRAIN_POLICIES
from turkeyworld.prepare_training_data import current_behavior_version
from turkeyworld.replay_subset import build_replay_subset, _joint_policy_quotas
from turkeyworld.training_subset import create_training_subset, load_training_subset


class ReplaySubsetTests(unittest.TestCase):
    @staticmethod
    def write(path, seed, policy, *, split="train", steps=32, behavior=None, pixel=0):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, behavior_version=np.asarray(behavior or current_behavior_version()),
                 split=np.asarray(split), policy=np.asarray(policy), seed=np.asarray(seed),
                 actions=np.full((steps, 4), .2, dtype=np.float32),
                 frames=np.full((steps + 1, 1, 1, 3), pixel, dtype=np.uint8))

    @classmethod
    def fixture(cls, root):
        old, new = root / "old/rgb/train", root / "new/rgb/train"
        for index in range(27):
            cls.write(old / f"episode_{1000+index:010d}.npz", 1000+index, TRAIN_POLICIES[index % 9])
        current = root / "current.json"
        create_training_subset(old, current, transitions=18*32)
        for index in range(15):
            cls.write(new / f"episode_{40000+index:010d}.npz", 40000+index, TRAIN_POLICIES[index % 9])
        for index in range(2):
            cls.write(old.parent / "val" / f"episode_{90000+index:010d}.npz",
                      90000+index, TRAIN_POLICIES[index], split="val")
        args = (current, old, new, root / "prepared/rgb/train", root / "training_subset_64k_B.json")
        return args, {"old_transitions": 5*32, "new_transitions": 15*32, "seed": 173}

    @staticmethod
    def change(path, **changes):
        with np.load(path, allow_pickle=False) as saved:
            values = {name: saved[name] for name in saved.files}
        values.update({name: np.asarray(value) for name, value in changes.items()})
        np.savez(path, **values)

    def test_exact_64k_joint_quota_fits_actual_1500_new_capacity(self):
        old_capacity = {p: 111 + int(i == 0) for i, p in enumerate(TRAIN_POLICIES)}
        new_capacity = {p: 166 + int(i < 6) for i, p in enumerate(TRAIN_POLICIES)}
        old, new = _joint_policy_quotas(500, 1500, old_capacity, new_capacity, 173)
        self.assertEqual(sum(old.values()), 500)
        self.assertEqual(sum(new.values()), 1500)
        self.assertEqual(sorted(old.values()), [55]*4 + [56]*5)
        self.assertEqual(new, new_capacity)
        self.assertEqual(sorted(old[p]+new[p] for p in TRAIN_POLICIES), [222]*7+[223]*2)
        self.assertEqual((old, new), _joint_policy_quotas(500, 1500, old_capacity, new_capacity, 173))
        self.assertEqual((500+1500)*32, 64000)
        with self.assertRaisesRegex(ValueError, "capacity"):
            _joint_policy_quotas(500, 1500, old_capacity, {p: 166 for p in TRAIN_POLICIES}, 173)

    def test_end_to_end_standard_loader_membership_and_no_rgb_decoding(self):
        with TemporaryDirectory() as temporary:
            args, kwargs = self.fixture(Path(temporary))
            current = load_training_subset(args[0], args[1])
            with np.load(next(args[1].glob("*.npz")), allow_pickle=False) as saved:
                archive = type(saved)
            getitem = archive.__getitem__

            def metadata_only(saved, key):
                self.assertNotIn(key, {"frames", "states"})
                return getitem(saved, key)

            with patch.object(archive, "__getitem__", metadata_only):
                manifest = build_replay_subset(*args, **kwargs)
                self.assertEqual(load_training_subset(args[4], args[3]), manifest)
            self.assertEqual(manifest["count_files"], 20)
            self.assertEqual(manifest["total_transitions"], 640)
            self.assertEqual(sorted(manifest["policy_counts"].values()), [2]*7+[3]*2)
            provenance = manifest["replay_provenance"]
            self.assertEqual(provenance["old_fraction"], .25)
            self.assertTrue(provenance["prepared_only"])
            self.assertEqual(len(provenance["old_members"]), 5)
            self.assertEqual(len(provenance["new_members"]), 15)
            self.assertTrue({m["filename"] for m in provenance["old_members"]}
                            <= set(current["filenames"]))
            self.assertEqual(provenance["current_subset_fingerprint"], current["fingerprint"])
            self.assertTrue(all(len(m["sha256"]) == 64
                                for m in provenance["old_members"]+provenance["new_members"]))

    def test_existing_output_byte_identical_when_sources_grow_and_request_changes_rejected(self):
        with TemporaryDirectory() as temporary:
            args, kwargs = self.fixture(Path(temporary))
            first = build_replay_subset(*args, **kwargs)
            before = args[4].read_bytes(), args[4].stat().st_mtime_ns
            self.write(args[2] / "episode_0000099999.npz", 99999, "movement")
            self.assertEqual(build_replay_subset(*args, **kwargs), first)
            self.assertEqual((args[4].read_bytes(), args[4].stat().st_mtime_ns), before)
            with self.assertRaisesRegex(ValueError, "different request"):
                build_replay_subset(*args, **{**kwargs, "seed": 174})

    def test_provenance_and_same_size_rgb_tampering_are_rejected(self):
        for kind in ("provenance", "rgb"):
            with self.subTest(kind=kind), TemporaryDirectory() as temporary:
                args, kwargs = self.fixture(Path(temporary))
                manifest = build_replay_subset(*args, **kwargs)
                if kind == "provenance":
                    manifest["replay_provenance"]["old_fraction"] = .5
                    args[4].write_text(json.dumps(manifest), encoding="utf-8")
                    message = "provenance fingerprint"
                else:
                    member = manifest["replay_provenance"]["new_members"][0]
                    source = args[2] / member["filename"]
                    before = source.stat().st_size
                    self.change(source, frames=np.ones((33, 1, 1, 3), dtype=np.uint8))
                    self.assertEqual(source.stat().st_size, before)
                    # Ordinary metadata/action identity cannot detect RGB-only edits.
                    load_training_subset(args[4], args[3])
                    message = "content changed"
                with self.assertRaisesRegex(ValueError, message):
                    build_replay_subset(*args, **kwargs)

    def test_partial_wrong_split_behavior_and_duplicate_new_seeds_fail_before_destination(self):
        for change, message in [({"actions": np.zeros((31, 4), np.float32)}, "complete 32"),
                                ({"split": "val"}, "non-training"),
                                ({"behavior_version": "old_v2"}, "behavior version"),
                                ({"seed": 40001}, "Duplicate new")]:
            with self.subTest(change=list(change)), TemporaryDirectory() as temporary:
                args, kwargs = self.fixture(Path(temporary))
                self.change(sorted(args[2].glob("*.npz"))[0], **change)
                with self.assertRaisesRegex(ValueError, message):
                    build_replay_subset(*args, **kwargs)
                self.assertFalse(args[3].exists())
                self.assertFalse(args[4].exists())

    def test_new_overlap_with_entire_current_manifest_or_validation_rejected(self):
        for origin in ("old", "val"):
            with self.subTest(origin=origin), TemporaryDirectory() as temporary:
                args, kwargs = self.fixture(Path(temporary))
                seed = load_training_subset(args[0], args[1])["records"][-1]["seed"] if origin == "old" else 90000
                self.change(sorted(args[2].glob("*.npz"))[0], seed=seed)
                with self.assertRaisesRegex(ValueError, "overlap current subset or validation"):
                    build_replay_subset(*args, **kwargs)
                self.assertFalse(args[3].exists())

    def test_copy_fallback_and_partial_preparation_recovery(self):
        with TemporaryDirectory() as temporary:
            args, kwargs = self.fixture(Path(temporary))
            import os
            original_link = os.link

            def copy_episodes_only(source, destination, *other, **options):
                if Path(source).suffix == ".npz":
                    raise OSError("Cross-volume test")
                return original_link(source, destination, *other, **options)

            with patch("turkeyworld.replay_subset.os.link", side_effect=copy_episodes_only):
                first = build_replay_subset(*args, **kwargs)
            self.assertEqual(load_training_subset(args[4], args[3]), first)
            # Crash before the manifest commit: identical prepared members are reusable.
            args[4].unlink()
            self.assertEqual(build_replay_subset(*args, **kwargs)["fingerprint"], first["fingerprint"])

    def test_unrelated_destination_or_missing_validation_is_never_repurposed(self):
        for reason in ("unrelated", "validation"):
            with self.subTest(reason=reason), TemporaryDirectory() as temporary:
                args, kwargs = self.fixture(Path(temporary))
                if reason == "unrelated":
                    self.write(args[3] / "unrelated.npz", 777777, "movement")
                    expected = "unrelated"
                else:
                    kwargs["validation_directory"] = Path(temporary) / "absent/val"
                    expected = "explicit val"
                with self.assertRaisesRegex(ValueError, expected):
                    build_replay_subset(*args, **kwargs)
                self.assertFalse(args[4].exists())


if __name__ == "__main__":
    unittest.main()
