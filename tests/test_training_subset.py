"""Fixed transition quotas, provenance isolation and immutable subset resume."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from turkeyworld.data_expansion import TRAIN_POLICIES
from turkeyworld.prepare_training_data import current_behavior_version
from turkeyworld.training_subset import (create_training_subset, load_training_subset,
                                         _policy_quotas, _select_names)


class TrainingSubsetTests(unittest.TestCase):
    @staticmethod
    def corpus(root, count=27):
        directory = root / "rgb/train"
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            path = directory / f"episode_{1000+index:010d}.npz"
            if not path.exists():
                np.savez_compressed(path, behavior_version=np.asarray(current_behavior_version()),
                    split=np.asarray("train"), policy=np.asarray(TRAIN_POLICIES[index % 9]),
                    seed=np.asarray(1000+index), actions=np.full((32, 4), index/100, dtype=np.float32),
                    frames=np.zeros((33, 1, 1, 3), dtype=np.uint8))
        manifest = {"expansion_version": 1, "behavior_version": current_behavior_version(),
                    "train_policies": list(TRAIN_POLICIES), "config": {"source": None,
                        "seed": 1000, "targets": {"rgb_train": count}, "clip_frames": 32,
                        "behavior_version": current_behavior_version()}}
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return directory

    @staticmethod
    def change(path, **changes):
        with np.load(path, allow_pickle=False) as saved:
            values = {name: saved[name] for name in saved.files}
        values.update({name: np.asarray(value) for name, value in changes.items()})
        np.savez_compressed(path, **values)

    def test_actual_128k_name_selection_is_exact_balanced_seeded_and_order_independent(self):
        groups = {policy: [f"{policy}_{index:05d}.npz" for index in range(500)]
                  for policy in TRAIN_POLICIES}
        selected = _select_names(groups, 4000, 173)
        self.assertEqual(len(selected), 4000)
        self.assertEqual(len({name for name, _ in selected}), 4000)
        self.assertEqual(len(selected)*32, 128000)
        quotas = _policy_quotas(4000, 173)
        self.assertEqual(sorted(quotas.values()), [444]*5+[445]*4)
        self.assertEqual({policy: sum(p == policy for _, p in selected) for policy in TRAIN_POLICIES}, quotas)
        reversed_groups = {policy: list(reversed(names)) for policy, names in groups.items()}
        self.assertEqual(selected, _select_names(reversed_groups, 4000, 173))
        self.assertNotEqual(selected, _select_names(groups, 4000, 174))
        self.assertNotEqual(quotas, _policy_quotas(4000, 174))

    def test_exact_32k_corpus_uses_available_extra_policy_instead_of_requesting_missing_clips(self):
        capacities = {policy: 111+int(policy == "movement") for policy in TRAIN_POLICIES}
        groups = {policy: [f"{policy}_{index}.npz" for index in range(capacity)]
                  for policy, capacity in capacities.items()}
        self.assertEqual(_policy_quotas(1000, 173, capacities), capacities)
        selected = _select_names(groups, 1000, 173)
        self.assertEqual(len(selected)*32, 32000)
        self.assertEqual({policy: sum(p == policy for _, p in selected) for policy in TRAIN_POLICIES}, capacities)
        with self.assertRaisesRegex(ValueError, "capacity"):
            _policy_quotas(1000, 173, {policy: 111 for policy in TRAIN_POLICIES})
        with self.assertRaisesRegex(ValueError, "capacity"):
            _policy_quotas(1000, 173, {**capacities, "attack": 110})
        # Also exercise the complete create/load path on the smallest uneven pool.
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = self.corpus(root / "data", count=10)
            output = root / "subset.json"
            manifest = create_training_subset(directory, output, transitions=320)
            self.assertEqual(manifest["count_files"], 10)
            self.assertEqual(manifest["policy_counts"], {policy: 1+int(policy == "movement") for policy in TRAIN_POLICIES})
            self.assertEqual(load_training_subset(output, directory), manifest)

    def test_real_npz_exact_quota_and_fingerprint_without_decoding_rgb(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = self.corpus(root / "data")
            output = root / "subset.json"
            with np.load(next(directory.glob("*.npz")), allow_pickle=False) as saved:
                archive_type = type(saved)
            original = archive_type.__getitem__

            def forbid_rgb(archive, name):
                self.assertNotIn(name, {"frames", "states"}, "Subset selection decoded RGB/state arrays")
                return original(archive, name)

            with patch.object(archive_type, "__getitem__", forbid_rgb):
                manifest = create_training_subset(directory, output, transitions=320)
                loaded = load_training_subset(output, directory, current_behavior_version())
            self.assertEqual(manifest, loaded)
            self.assertEqual(manifest["count_files"], 10)
            self.assertEqual(manifest["total_transitions"], 320)
            self.assertEqual(manifest["source_directory"], str(directory.resolve()))
            self.assertEqual(manifest["policy_counts"], _policy_quotas(10, 173))
            self.assertEqual(manifest["candidate_policy_source"], "fresh_expansion_seed_rule")
            self.assertEqual(len(manifest["fingerprint"]), 64)

    def test_resume_remains_byte_identical_when_source_grows_and_val_is_untouched(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = self.corpus(root / "data")
            val = directory.parent / "val"
            val.mkdir()
            sentinel = val / "frozen_panel.npz"
            sentinel.write_bytes(b"frozen validation sentinel")
            output = root / "subset.json"
            first = create_training_subset(directory, output, transitions=320)
            before = (output.read_bytes(), output.stat().st_mtime_ns)
            selected = {name: (directory/name).stat().st_mtime_ns for name in first["filenames"]}
            val_before = (sentinel.read_bytes(), sentinel.stat().st_mtime_ns)
            self.corpus(root / "data", count=45)
            self.assertEqual(create_training_subset(directory, output, transitions=320), first)
            self.assertEqual((output.read_bytes(), output.stat().st_mtime_ns), before)
            self.assertEqual({name: (directory/name).stat().st_mtime_ns for name in selected}, selected)
            self.assertEqual((sentinel.read_bytes(), sentinel.stat().st_mtime_ns), val_before)
            for changed in ({"transitions": 352}, {"seed": 174}):
                with self.assertRaisesRegex(ValueError, "immutable subset"):
                    create_training_subset(directory, output, **{"transitions": 320, **changed})
            self.assertEqual((output.read_bytes(), output.stat().st_mtime_ns), before)

    def test_wrong_split_behavior_source_and_changed_actions_are_rejected(self):
        for field, value, message in [("split", "val", "non-training split"),
                                      ("behavior_version", "legacy_random_v1", "behavior version"),
                                      ("actions", np.ones((32, 4), dtype=np.float32), "episode changed")]:
            with self.subTest(field=field), TemporaryDirectory() as temporary:
                root = Path(temporary)
                directory = self.corpus(root / "data")
                output = root / "subset.json"
                manifest = create_training_subset(directory, output, transitions=288)
                self.change(directory / manifest["filenames"][0], **{field: value})
                with self.assertRaisesRegex(ValueError, message):
                    load_training_subset(output, directory)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = self.corpus(root / "data")
            output = root / "subset.json"
            create_training_subset(directory, output, transitions=288)
            val = directory.parent / "val"
            val.mkdir()
            with self.assertRaisesRegex(ValueError, "train directory"):
                create_training_subset(val, root / "bad.json", transitions=288)
            other = self.corpus(root / "other")
            with self.assertRaisesRegex(ValueError, "source directory"):
                load_training_subset(output, other)
            with self.assertRaisesRegex(ValueError, "behavior mismatch"):
                load_training_subset(output, directory, "another_behavior")

    def test_new_selection_validates_actual_split_and_policy_not_only_filename_hint(self):
        for field, value in [("split", "val"), ("behavior_version", "old"), ("policy", "attack"), ("seed", 999)]:
            with self.subTest(field=field), TemporaryDirectory() as temporary:
                root = Path(temporary)
                directory = self.corpus(root / "data", count=9)
                self.change(directory / "episode_0000001000.npz", **{field: value})
                with self.assertRaises(ValueError):
                    create_training_subset(directory, root / "subset.json", transitions=288)
                self.assertFalse((root / "subset.json").exists())

    def test_partial_clips_are_skipped_and_insufficient_or_nonmultiple_quota_never_commits(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = self.corpus(root / "data")
            self.change(directory / "episode_0000001000.npz", actions=np.zeros((12, 4), dtype=np.float32))
            manifest = create_training_subset(directory, root / "subset.json", transitions=576)
            self.assertEqual(manifest["total_transitions"], 576)
            self.assertNotIn("episode_0000001000.npz", manifest["filenames"])
            for transitions in (289, 32*28):
                output = root / f"bad_{transitions}.json"
                with self.assertRaises(ValueError):
                    create_training_subset(directory, output, transitions=transitions)
                self.assertFalse(output.exists())

    def test_fingerprint_or_path_tampering_cannot_replace_fixed_subset(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = self.corpus(root / "data")
            output = root / "subset.json"
            manifest = create_training_subset(directory, output, transitions=288)
            manifest["filenames"][0] = "../val/frozen.npz"
            output.write_text(json.dumps(manifest), encoding="utf-8")
            before = output.read_bytes()
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                create_training_subset(directory, output, transitions=288)
            self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
