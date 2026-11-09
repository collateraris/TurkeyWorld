"""VQGAN targets are adjacent source frames, with immutable validation pairs."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def write_episode(path, identity=0, split="train", policy="movement", attack_index=None, source="engine"):
    steps = 6
    values = np.arange(steps+1)+identity*30
    frames = np.broadcast_to(values[:, None, None, None], (steps+1, 16, 16, 3)).astype(np.uint8).copy()
    states = np.zeros((steps+1, 90), np.float32)
    states[:, 0] = np.arange(steps+1)
    states[:, 1] = identity
    events = np.zeros((steps, 4), np.float32)
    if attack_index is not None:
        events[attack_index, 3] = 1
    np.savez_compressed(path, frames=frames, predicted_frames=np.full_like(frames, 255),
        states=states, actions=np.zeros((steps, 4), np.float32), events=events,
        hit_ids=np.full(steps, -1, np.int64), attack_flags=np.zeros((steps, 8), np.float32),
        map=np.full((16, 16), identity, np.int8), split=np.asarray(split), dt=np.asarray(1/30),
        policy=np.asarray(policy), frame_source=np.asarray(source))


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class VQGANPairDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_pairs_are_adjacent_source_rgb_and_never_cross_episodes(self):
        from turkeyworld.vqgan_data import VQGANData
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"train"
            root.mkdir()
            write_episode(root/"a.npz", 1)
            write_episode(root/"b.npz", 2)
            data = VQGANData(root, image_size=16, cache_bytes=12000)
            first = data.batch(16, rng=np.random.default_rng(273))
            again = data.batch(16, rng=np.random.default_rng(273))
            self.assertEqual(tuple(first["frames"].shape), (16, 2, 3, 16, 16))
            self.assertEqual(tuple(first["states"].shape), (16, 2, 90))
            self.assertEqual(tuple(first["events"].shape), (16, 1, 4))
            self.assertEqual(tuple(first["map"].shape), (16, 16, 16))
            self.assertEqual(tuple(first["indices"].shape), (16, 2))
            for key in first:
                torch.testing.assert_close(first[key], again[key], rtol=0, atol=0)
            values = first["frames"][:, :, 0, 0, 0]*255
            torch.testing.assert_close(values[:, 1]-values[:, 0], torch.ones(16))
            torch.testing.assert_close(values, first["states"][:, :, 0]+first["states"][:, :, 1]*30)
            self.assertTrue(bool((first["states"][:, 0, 1] == first["states"][:, 1, 1]).all()))
            self.assertLess(float(first["frames"].max()), 1.)  # Ignores255-valued predictions.
            self.assertLessEqual(data.stats["cache_bytes"], 12000)
            self.assertEqual(data.stats["rgb_frames"], 14)
            self.assertEqual(data.stats["adjacent_pairs"], 12)
            self.assertEqual(data.stats["augmentations"], [])

    def test_validation_panels_are_unique_replayable_and_val_only(self):
        from turkeyworld.vqgan_data import VQGANData
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_episode(root/"a.npz", 1, "val")
            write_episode(root/"b.npz", 2, "val")
            data = VQGANData(root, split="val", image_size=16, balance=True)
            self.assertFalse(data.balance)
            panel = data.fixed_panel(8, seed=273)
            self.assertEqual(len(panel), len(set(panel)))
            self.assertEqual(panel, data.fixed_panel(8, seed=273))
            first, second = data.batch(8, indices=panel), data.batch(8, indices=panel)
            torch.testing.assert_close(first["frames"], second["frames"], rtol=0, atol=0)
            self.assertEqual(first["indices"].tolist(), [list(pair) for pair in panel])
            with self.assertRaisesRegex(ValueError, "unique adjacent"):
                data.fixed_panel(128)
            with self.assertRaisesRegex(ValueError, "crosses an episode"):
                data.batch(1, indices=[(0, 6)])
            with self.assertRaisesRegex(ValueError, "integer"):
                data.batch(1, indices=[(.1, 1.)])
            with self.assertRaisesRegex(ValueError, "mix dataset splits"):
                VQGANData(root, split="train")

    def test_event_balance_uses_real_intervening_labels_and_refresh_appends(self):
        from turkeyworld.vqgan_data import VQGANData
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"train"
            root.mkdir()
            write_episode(root/"a.npz", 1, attack_index=3, policy="attack")
            data = VQGANData(root, image_size=16, balance=True, event_fraction=1., refresh_interval=0)
            batch = data.batch(12)
            self.assertTrue(bool((batch["indices"][:, 1] == 3).all()))
            self.assertTrue(bool((batch["events"][:, 0, 3] == 1).all()))
            self.assertEqual(data.stats["event_positive_pairs"], 1)
            original = data.episodes[0]["path"]
            write_episode(root/"b.npz", 2, attack_index=2, policy="combat")
            self.assertEqual(data.refresh(force=True), 1)
            self.assertEqual(data.episodes[0]["path"], original)
            self.assertEqual(data.stats["source_files"], 2)
            self.assertEqual(data.stats["adjacent_pairs"], 12)
            self.assertEqual(data.stats["event_positive_pairs"], 2)
            with self.assertRaisesRegex(ValueError, "val split"):
                data.fixed_panel(4)

    def test_declared_model_images_cannot_be_codec_targets(self):
        from turkeyworld.vqgan_data import VQGANData
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"val"
            root.mkdir()
            write_episode(root/"a.npz", source="neural_prediction", split="val")
            data = VQGANData(root, image_size=16)
            with self.assertRaisesRegex(ValueError, "clean source RGB"):
                data.batch(1, indices=[(0, 0)])

    def test_bounded_bootstrap_grows_without_reindexing_or_changing_old_pairs(self):
        from turkeyworld.vqgan_data import VQGANData
        from turkeyworld.streaming_sequences import _metadata
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"train"
            root.mkdir()
            for index in range(7):
                write_episode(root/f"{index:03d}.npz", index, attack_index=index % 6)
            with mock.patch("turkeyworld.streaming_sequences._metadata", wraps=_metadata) as headers:
                data = VQGANData(root, image_size=16, balance=True, event_fraction=1.,
                    initial_file_limit=2, growth_files=2, refresh_interval=30)
                self.assertEqual(headers.call_count, 2)
                self.assertEqual(data.files, 2)
                self.assertEqual(data.stats["available_source_files"], 7)
                self.assertEqual(data.stats["unindexed_source_files"], 5)
                self.assertFalse(data.stats["source_index_complete"])
                self.assertEqual(data.stats["indexing_scope"], "indexed_subset")
                self.assertEqual(data.stats["rgb_frames"], 14)
                # Even the underlying loader's automatic discovery must retain
                # the same persistent cap; it cannot scan all remaining NPZ.
                data.data._last_refresh = 0.
                data.data.batch(1, context=1, horizon=1)
                self.assertEqual(headers.call_count, 2)
                self.assertEqual(data.data.files, 2)
                with mock.patch.object(Path, "glob", side_effect=AssertionError("stats must use cached availability")):
                    self.assertEqual(data.stats["available_source_files"], 7)
                old_paths = [episode["path"] for episode in data.episodes]
                original = data.batch(1, indices=[(1, 2)])
                self.assertEqual(data.refresh(), 0)
                self.assertEqual(data.stats["current_file_limit"], 2)
                with mock.patch.object(data, "_read_source", wraps=data._read_source) as catalog:
                    self.assertEqual(data.refresh(force=True), 2)
                    self.assertEqual([call.args[0] for call in catalog.call_args_list], [2, 3])
                self.assertEqual(headers.call_count, 4)
                self.assertEqual(data.files, 4)
                self.assertEqual([episode["path"] for episode in data.episodes[:2]], old_paths)
                replay = data.batch(1, indices=[(1, 2)])
                torch.testing.assert_close(original["frames"], replay["frames"], rtol=0, atol=0)
                self.assertEqual(data.refresh(force=True), 2)
                self.assertEqual(data.refresh(force=True), 1)
                self.assertEqual(data.files, 7)
                self.assertEqual(headers.call_count, 7)
                self.assertEqual(data.stats["event_positive_pairs"], 7)
                self.assertEqual(data.stats["unindexed_source_files"], 0)
                self.assertTrue(data.stats["source_index_complete"])
                self.assertEqual(data.stats["adjacent_pairs"], 42)
                write_episode(root/"007.npz", 7, attack_index=2)
                self.assertEqual(data.refresh(force=True), 1)
                self.assertEqual(data.stats["available_source_files"], 8)
                self.assertEqual(data.stats["source_files"], 8)
                sampled = data.batch(12)
                self.assertTrue(bool((sampled["events"][:, 0, 3] == 1).all()))

    def test_bootstrap_defaults_leave_validation_full_and_reject_invalid_limits(self):
        from turkeyworld.vqgan_data import VQGANData
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"val"
            root.mkdir()
            for index in range(3):
                write_episode(root/f"{index}.npz", index, split="val")
            full = VQGANData(root, image_size=16)
            self.assertEqual(full.files, 3)
            self.assertTrue(full.stats["source_index_complete"])
            self.assertEqual(full.stats["current_file_limit"], 0)
            self.assertEqual(len(full.fixed_panel(12)), 12)
            for kwargs in ({"initial_file_limit": -1}, {"growth_files": -1},
                           {"initial_file_limit": 1.5}, {"growth_files": .5}):
                with self.assertRaisesRegex(ValueError, "nonnegative integers"):
                    VQGANData(root, image_size=16, **kwargs)

    def test_growth_cannot_admit_a_different_split(self):
        from turkeyworld.vqgan_data import VQGANData
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"train"
            root.mkdir()
            write_episode(root/"a.npz", split="train")
            write_episode(root/"b.npz", split="val")
            data = VQGANData(root, image_size=16, initial_file_limit=1, growth_files=1)
            with self.assertRaisesRegex(ValueError, "mix dataset splits"):
                data.refresh(force=True)
            self.assertEqual(data.files, 1)


if __name__ == "__main__":
    unittest.main()
