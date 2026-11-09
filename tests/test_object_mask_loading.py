"""Object sidecars are strict future labels, independent of runtime engines."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class ObjectMaskLoadingTests(unittest.TestCase):
    @staticmethod
    def corpus(root, count=3):
        from turkeyworld.object_mask_cache import MASK_KIND, source_identity
        data, masks = root / "data", root / "object_masks"
        records, expected = [], {}
        for split, files in (("train", count), ("val", 1)):
            (data / "rgb" / split).mkdir(parents=True)
            (masks / split).mkdir(parents=True)
            for index in range(files):
                path = data / "rgb" / split / f"episode_{index:04d}.npz"
                states = np.zeros((8, 90), dtype=np.float32)
                states[:, 0] = np.arange(8)
                np.savez_compressed(path, frames=np.full((8, 4, 6, 3), index, np.uint8), states=states,
                    actions=np.zeros((7, 4), np.float32), events=np.zeros((7, 4), np.float32),
                    hit_ids=np.full(7, -1, np.int64), attack_flags=np.zeros((7, 8), np.float32),
                    map=np.zeros((16, 16), np.int8), dt=np.asarray(1/30),
                    split=np.asarray(split), behavior_version=np.asarray("test_behavior_v3"))
                label = np.zeros((8, 4, 6), dtype=np.bool_)
                for frame in range(8):
                    label[frame, (frame+index) % 4, (2*frame+index) % 6] = True
                metadata = {**source_identity(path), "format_version": 1, "mask_kind": MASK_KIND,
                            "split": split, "behavior_version": "test_behavior_v3", "verified_source_frames": 8}
                np.savez_compressed(masks / split / path.name, mask_bits=np.packbits(label, axis=-1),
                                    shape=np.asarray(label.shape), metadata=np.asarray(json.dumps(metadata)))
                records.append(metadata)
                expected[(split, index)] = label
        (data / "manifest.json").write_text(json.dumps({"behavior_version": "test_behavior_v3"}), encoding="utf-8")
        (masks / "manifest.json").write_text(json.dumps({"format_version": 1, "mask_kind": MASK_KIND,
            "status": "complete", "source_data_root": str(data.resolve()), "behavior_version": "test_behavior_v3",
            "total_episodes": len(records), "completed_episodes": len(records), "records": records}), encoding="utf-8")
        return data, masks, expected

    @staticmethod
    def change_sidecar(path, *, metadata_changes=None, label=None):
        with np.load(path, allow_pickle=False) as saved:
            values = {name: saved[name] for name in saved.files}
        if metadata_changes:
            metadata = json.loads(str(values["metadata"]))
            metadata.update(metadata_changes)
            values["metadata"] = np.asarray(json.dumps(metadata))
        if label is not None:
            values["shape"] = np.asarray(label.shape)
            values["mask_bits"] = np.packbits(label, axis=-1)
        np.savez_compressed(path, **values)

    def test_memory_and_stream_masks_use_exact_future_offsets_and_leave_inputs_unchanged(self):
        from turkeyworld.jepa_training_utils import SequenceData
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, masks, expected = self.corpus(root)
            for mode in ("memory", "stream"):
                with self.subTest(mode=mode):
                    plain = SequenceData(data / "rgb/train", mode=mode)
                    annotated = SequenceData(data / "rgb/train", mode=mode, object_masks_directory=masks / "train")
                    panel = [(0, 1), (1, 2)]
                    original = plain.batch(2, context=2, horizon=3, indices=panel)
                    with patch("turkeyworld.engine.Game.render", side_effect=AssertionError("Engine render during training")), \
                         patch("turkeyworld.engine.Game.step", side_effect=AssertionError("Engine step during training")), \
                         patch("turkeyworld.object_mask_cache._generate_masks", side_effect=AssertionError("Online annotation forbidden")):
                        batch = annotated.batch(2, context=2, horizon=3, indices=panel)
                    self.assertEqual(set(batch)-set(original), {"target_object_masks"})
                    for key, value in original.items():
                        torch.testing.assert_close(batch[key], value, rtol=0, atol=0)
                    target = batch["target_object_masks"]
                    self.assertEqual(target.shape, (2, 3, 1, 4, 6))
                    self.assertEqual(target.dtype, torch.float32)
                    np.testing.assert_array_equal(target[0, :, 0].numpy(), expected[("train", 0)][3:6])
                    np.testing.assert_array_equal(target[1, :, 0].numpy(), expected[("train", 1)][4:7])

    def test_nearest_resize_and_explicit_validation_split(self):
        from turkeyworld.jepa_training_utils import SequenceData
        from torch.nn import functional as F
        with TemporaryDirectory() as temporary:
            data, masks, expected = self.corpus(Path(temporary))
            for mode in ("memory", "stream"):
                dataset = SequenceData(data / "rgb/val", mode=mode, image_size=8, object_masks_directory=masks / "val")
                batch = dataset.batch(1, context=2, horizon=3, indices=[(0, 1)])
                target = torch.from_numpy(expected[("val", 0)][3:6]).float()[:, None]
                resized = F.interpolate(target, (8, 8), mode="nearest")
                torch.testing.assert_close(batch["target_object_masks"][0], resized, rtol=0, atol=0)
                self.assertEqual(batch["target_object_masks"].shape, (1, 3, 1, 8, 8))
                self.assertTrue(set(batch["target_object_masks"].unique().tolist()).issubset({0., 1.}))

    def test_manifest_must_be_complete_and_match_source_behavior_and_split(self):
        from turkeyworld.jepa_training_utils import SequenceData
        changes = [{"status": "running"}, {"behavior_version": "old"}, {"source_data_root": "wrong"},
                   {"completed_episodes": 0}, {"mask_kind": "bounding_box_approximation"}]
        for change in changes:
            with self.subTest(change=change), TemporaryDirectory() as temporary:
                data, masks, _ = self.corpus(Path(temporary))
                path = masks / "manifest.json"
                manifest = json.loads(path.read_text("utf-8"));manifest.update(change)
                path.write_text(json.dumps(manifest), encoding="utf-8")
                for mode in ("memory", "stream"):
                    with self.assertRaisesRegex(ValueError, "mask"):
                        SequenceData(data / "rgb/train", mode=mode, object_masks_directory=masks / "train")
        with TemporaryDirectory() as temporary:
            data, masks, _ = self.corpus(Path(temporary))
            with self.assertRaisesRegex(ValueError, "split"):
                SequenceData(data / "rgb/train", object_masks_directory=masks / "val")
            (masks / "manifest.json").unlink()
            with self.assertRaisesRegex(ValueError, "manifest"):
                SequenceData(data / "rgb/train", object_masks_directory=masks / "train")

    def test_sidecar_split_source_identity_behavior_shape_and_missing_labels_fail_without_fallback(self):
        from turkeyworld.jepa_training_utils import SequenceData
        for defect in ("split", "identity", "behavior", "shape", "missing"):
            for mode in ("memory", "stream"):
                with self.subTest(defect=defect, mode=mode), TemporaryDirectory() as temporary:
                    data, masks, _ = self.corpus(Path(temporary))
                    dataset = SequenceData(data / "rgb/train", mode=mode, object_masks_directory=masks / "train")
                    sidecar = masks / "train/episode_0000.npz"
                    if defect == "split":
                        self.change_sidecar(sidecar, metadata_changes={"split": "val"})
                    elif defect == "identity":
                        self.change_sidecar(sidecar, metadata_changes={"source_size": -1})
                    elif defect == "behavior":
                        self.change_sidecar(sidecar, metadata_changes={"behavior_version": "old"})
                    elif defect == "shape":
                        self.change_sidecar(sidecar, label=np.zeros((7, 4, 6), dtype=np.bool_))
                    else:
                        sidecar.unlink()
                    with patch("turkeyworld.engine.Game.render", side_effect=AssertionError("No annotation fallback")), \
                         patch("turkeyworld.engine.Game.step", side_effect=AssertionError("No engine fallback")), \
                         self.assertRaisesRegex(ValueError, "mask"):
                        dataset.batch(1, context=2, horizon=3, indices=[(0, 0)])

    def test_label_lru_is_bounded_separate_from_rgb_and_clearable(self):
        from turkeyworld.jepa_training_utils import SequenceData
        from turkeyworld.object_mask_cache import read_object_masks
        with TemporaryDirectory() as temporary:
            data, masks, _ = self.corpus(Path(temporary), count=21)
            for mode in ("memory", "stream"):
                dataset = SequenceData(data / "rgb/train", mode=mode, object_masks_directory=masks / "train")
                with patch("turkeyworld.object_mask_cache.read_object_masks", wraps=read_object_masks) as read:
                    for index in range(21):
                        dataset.batch(1, context=2, horizon=3, indices=[(index, 0)])
                    self.assertEqual(read.call_count, 21)
                    self.assertEqual(len(dataset._object_mask_cache), 16)
                    self.assertEqual(dataset.object_mask_cached_bytes, 16*8*4*6)
                    dataset.batch(1, context=2, horizon=3, indices=[(20, 0)])
                    self.assertEqual(read.call_count, 21)
                    dataset.batch(1, context=2, horizon=3, indices=[(0, 0)])
                    self.assertEqual(read.call_count, 22)
                for label, identity in dataset._object_mask_cache.values():
                    self.assertEqual(label.shape, (8, 4, 6))
                    self.assertEqual(label.dtype, np.bool_)
                    self.assertEqual(len(identity), 2)
                dataset.clear_cache()
                self.assertFalse(dataset._object_mask_cache)
                self.assertEqual(dataset.object_mask_cached_bytes, 0)

    def test_cached_labels_detect_source_file_change(self):
        from turkeyworld.jepa_training_utils import SequenceData
        with TemporaryDirectory() as temporary:
            data, masks, _ = self.corpus(Path(temporary))
            dataset = SequenceData(data / "rgb/train", object_masks_directory=masks / "train")
            dataset.batch(1, context=2, horizon=3, indices=[(0, 0)])
            source = data / "rgb/train/episode_0000.npz"
            stat = source.stat()
            os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns+1_000_000))
            with self.assertRaisesRegex(ValueError, "source episode changed"):
                dataset.batch(1, context=2, horizon=3, indices=[(0, 0)])


if __name__ == "__main__":
    unittest.main()
