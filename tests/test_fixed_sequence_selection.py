"""A frozen episode manifest cannot silently grow with the streamed corpus."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def write_episode(path, identity=0, count=8, split='train'):
    states = np.zeros((count+1, 90), np.float32)
    states[:, 0] = np.arange(count+1)
    states[:, 1] = identity
    np.savez_compressed(path, frames=np.full((count+1, 16, 16, 3), identity, np.uint8),
        states=states, actions=np.zeros((count, 4), np.float32),
        events=np.zeros((count, 4), np.float32), hit_ids=np.full(count, -1, np.int64),
        attack_flags=np.zeros((count, 8), np.float32), map=np.zeros((16, 16), np.int8),
        split=np.asarray(split), dt=np.asarray(1/30))


@unittest.skipIf(torch is None, 'Optional PyTorch dependency is not installed')
class FixedSequenceSelectionTests(unittest.TestCase):
    def test_memory_and_stream_keep_manifest_order_and_actual_transition_counts(self):
        from turkeyworld.jepa_training_utils import SequenceData
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'train'
            root.mkdir()
            write_episode(root/'a.npz', identity=1, count=8)
            write_episode(root/'b.npz', identity=2, count=10)
            write_episode(root/'ignored.npz', identity=3, count=15)
            for mode in ('memory', 'stream'):
                with self.subTest(mode=mode):
                    data = SequenceData(root, mode=mode, episode_paths=['b.npz', root/'a.npz'])
                    self.assertEqual(data.files, 2)
                    self.assertEqual(data.count, 18)
                    self.assertEqual([Path(episode['path']).name for episode in data.episodes], ['b.npz', 'a.npz'])
                    self.assertEqual([(r['start'], r['stop']) for r in data.episode_ranges], [(0, 10), (10, 18)])
                    batch = data.batch(2, context=2, horizon=4, indices=[(0, 0), (1, 0)])
                    self.assertEqual(batch['states'][:, 0, 1].tolist(), [2., 1.])

    def test_appended_unselected_files_are_ignored_by_forced_and_automatic_refresh(self):
        from turkeyworld.jepa_training_utils import SequenceData
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'train'
            root.mkdir()
            write_episode(root/'a.npz', identity=1)
            data = SequenceData(root, mode='stream', refresh_interval=0, episode_paths=['a.npz'])
            before = list(data.episode_ranges)
            write_episode(root/'new.npz', identity=2, count=20)
            write_episode(root/'unselected-val.npz', split='val')
            (root/'unselected-broken.npz').write_bytes(b'not an archive')
            self.assertEqual(data.refresh(), 0)
            data.batch(1, context=2, horizon=4)
            self.assertEqual((data.files, data.count), (1, 8))
            self.assertEqual(data.episode_ranges, before)

    def test_invalid_fixed_paths_are_rejected_in_both_modes_before_loading(self):
        from turkeyworld.jepa_training_utils import SequenceData
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, val = root/'train', root/'val'
            train.mkdir(); val.mkdir()
            write_episode(train/'a.npz')
            write_episode(val/'v.npz', split='val')
            (train/'text.txt').write_text('not NPZ', encoding='utf-8')
            invalid = [([], 'nonempty'), (['a.npz', './a.npz'], 'duplicates'),
                (['a.npz', train/'a.npz'], 'duplicates'), ([val/'v.npz'], 'split directory'),
                (['../val/v.npz'], 'split directory'), (['missing.npz'], 'does not exist'),
                (['text.txt'], '.npz'), ('a.npz', 'sequence')]
            for mode in ('memory', 'stream'):
                for paths, message in invalid:
                    with self.subTest(mode=mode, paths=paths):
                        with self.assertRaisesRegex(ValueError, message):
                            SequenceData(train, mode=mode, episode_paths=paths)

    def test_in_directory_val_labels_are_still_rejected_for_explicit_train_split(self):
        from turkeyworld.jepa_training_utils import SequenceData
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'train'
            root.mkdir()
            write_episode(root/'wrong-label.npz', split='val')
            for mode in ('memory', 'stream'):
                with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'mix dataset splits'):
                    SequenceData(root, mode=mode, episode_paths=['wrong-label.npz'])

    def test_limit_keeps_provided_order_and_validation_cannot_be_hidden_after_limit(self):
        from turkeyworld.jepa_training_utils import SequenceData
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_episode(root/'a.npz', identity=1)
            write_episode(root/'b.npz', identity=2)
            for mode in ('memory', 'stream'):
                with self.subTest(mode=mode):
                    data = SequenceData(root, mode=mode, limit=1, episode_paths=['b.npz', 'a.npz'])
                    self.assertEqual(data.files, 1)
                    self.assertEqual(Path(data.episodes[0]['path']).name, 'b.npz')
                    with self.assertRaisesRegex(ValueError, 'duplicates'):
                        SequenceData(root, mode=mode, limit=1, episode_paths=['b.npz', 'b.npz'])

    def test_direct_stream_initializer_accepts_fixed_relative_selection(self):
        from turkeyworld.streaming_sequences import initialize_stream, refresh_stream
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_episode(root/'a.npz')
            write_episode(root/'b.npz', count=10)
            owner = SimpleNamespace()
            initialize_stream(owner, root, episode_paths=['b.npz'])
            self.assertEqual((owner.files, owner.count), (1, 10))
            write_episode(root/'c.npz')
            self.assertEqual(refresh_stream(owner, force=True), 0)
            self.assertEqual(owner._fixed_episode_paths, ((root/'b.npz').resolve(),))


if __name__ == '__main__':
    unittest.main()
