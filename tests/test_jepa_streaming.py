"""Large RGB corpora preserve alignment while bounding decoded LRU storage."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def write_episode(path, identity=0, split="train"):
    count = 10
    frames = np.full((count+1, 16, 16, 3), identity, np.uint8)
    states = np.zeros((count+1, STATE_DIM), np.float32)
    states[:, 0] = np.arange(count+1)
    states[:, 1] = identity
    np.savez_compressed(path, frames=frames, states=states,
        actions=np.broadcast_to(np.arange(count)[:, None], (count, 4)).astype(np.float32),
        events=np.zeros((count, 4), np.float32), hit_ids=np.full(count, -1, np.int64),
        attack_flags=np.zeros((count, ENTITY_SLOTS), np.float32), map=np.zeros((16, 16), np.int8),
        split=np.asarray(split), dt=np.asarray(1/30))


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class StreamingSequenceTests(unittest.TestCase):
    def test_subset_validation_panel_does_not_alias_periodic_source_policies(self):
        from turkeyworld.jepa_training_utils import SequenceData
        data = object.__new__(SequenceData)
        data.files = 512
        data.episodes = [{"frames": range(33)} for _ in range(data.files)]
        panel = data.fixed_panel(128, seed=73)
        self.assertEqual(len(set(panel)), 128)
        self.assertEqual({index % 8 for index, _ in panel}, set(range(8)))
        self.assertEqual(panel, data.fixed_panel(128, seed=73))

    def test_constructor_panels_and_window_count_never_decode_rgb(self):
        from turkeyworld.jepa_training_utils import SequenceData
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_episode(root/"a.npz")
            write_episode(root/"b.npz", 2)
            with np.load(root/"a.npz") as sample:
                archive_type = type(sample)
            original = archive_type.__getitem__
            reads = []

            def tracked(archive, key):
                reads.append(key)
                return original(archive, key)

            with patch.object(archive_type, "__getitem__", tracked):
                streamed = SequenceData(root, mode="stream", cache_bytes=20000)
                self.assertEqual(streamed.window_count(), 12)
                self.assertEqual(len(streamed.fixed_panel(4)), 4)
                self.assertEqual(len(streamed.episodes[0]["frames"]), 11)
                self.assertEqual(streamed.raw_bytes, 0)
                self.assertNotIn("frames", reads)
                _ = streamed.episodes[0]["frames"][0]
                self.assertIn("frames", reads)

    def test_stream_and_memory_batches_and_validation_panels_are_identical(self):
        from turkeyworld.jepa_training_utils import SequenceData
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for index in range(4):
                write_episode(root/f"{index}.npz", index, "val")
            memory = SequenceData(root, split="val", image_size=32)
            stream = SequenceData(root, split="val", mode="stream", cache_bytes=20000, image_size=32)
            self.assertEqual(memory.fixed_panel(8), stream.fixed_panel(8))
            fixed = memory.fixed_panel(8)
            first = memory.batch(8, indices=fixed)
            second = stream.batch(8, indices=fixed)
            for key in first:
                torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
            self.assertEqual(stream.count, memory.count)
            self.assertLessEqual(stream.raw_bytes, 20000)
            self.assertGreater(stream.total_raw_bytes, 20000)

    def test_lru_eviction_refresh_and_mixed_split_validation(self):
        from turkeyworld.jepa_training_utils import SequenceData
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_episode(root/"a.npz", 1)
            data = SequenceData(root, mode="stream", cache_bytes=20000, refresh_interval=0)
            _ = data.episodes[0]["frames"][0]
            self.assertEqual(len(data._stream_cache), 1)
            write_episode(root/"b.npz", 2)
            self.assertEqual(data.refresh(), 1)
            self.assertTrue(data.episodes[0]["path"].endswith("a.npz"))
            _ = data.episodes[1]["frames"][0]
            self.assertEqual(len(data._stream_cache), 1)
            self.assertTrue(next(iter(data._stream_cache)).endswith("b.npz"))
            self.assertLessEqual(data.raw_bytes, 20000)
            data.clear_cache()
            self.assertEqual(data.raw_bytes, 0)
            write_episode(root/"c.npz", 3)
            data.batch(1)
            self.assertEqual(data.files, 3)
            write_episode(root/"bad.npz", 4, "val")
            with self.assertRaisesRegex(ValueError, "mix dataset splits"):
                data.refresh()
            with self.assertRaisesRegex(ValueError, "cache_bytes"):
                SequenceData(root, split="train", mode="stream", cache_bytes=1)


if __name__ == "__main__":
    unittest.main()
