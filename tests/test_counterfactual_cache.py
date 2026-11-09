"""Offline background upgrades and label-only, bounded training loading."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from turkeyworld.engine import Game
from turkeyworld.object_mask_cache import (_generate_masks, main, prepare_object_masks,
                                           read_object_backgrounds, read_object_masks)
from turkeyworld.training_subset import create_training_subset
from test_object_mask_cache import make_episode, file_digest
import test_object_mask_loading as loading_fixtures

try:
    import torch
except ImportError:
    torch = None


def change_sidecar(path, update):
    with np.load(path, allow_pickle=False) as saved:
        payload = {key: saved[key] for key in saved.files}
    update(payload)
    np.savez_compressed(path, **payload)


def synthetic_backgrounds(root, count=3):
    data, masks, _ = loading_fixtures.ObjectMaskLoadingTests.corpus(root, count=count)
    expected = {}
    for split in ('train', 'val'):
        for index, path in enumerate(sorted((masks/split).glob('*.npz'))):
            frames = (np.arange(8*4*6*3).reshape(8, 4, 6, 3)+index).astype(np.uint8)
            def update(payload):
                metadata = json.loads(str(payload['metadata']))
                metadata['background_frames_available'] = True
                payload['metadata'] = np.asarray(json.dumps(metadata))
                payload['background_rgb'] = frames
            change_sidecar(path, update)
            expected[(split, index)] = frames
    manifest = json.loads((masks/'manifest.json').read_text('utf-8'))
    manifest['background_frames_available'] = True
    for record in manifest['records']:
        record['background_frames_available'] = True
    (masks/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return data, masks, expected


class CounterfactualCacheTests(unittest.TestCase):
    def test_mask_only_upgrade_preserves_masks_sources_and_natural_non_npc_pixels(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary)/'train/episode.npz'
            job = make_episode(source)
            _generate_masks(job)
            before_source = (file_digest(source), source.stat().st_size, source.stat().st_mtime_ns)
            old_mask, old_metadata = read_object_masks(job['output'], source, 'train')
            with np.load(job['output'], allow_pickle=False) as saved:
                old_bits = saved['mask_bits'].copy()
            upgraded = _generate_masks({**job, 'include_backgrounds': True})
            mask, metadata = read_object_masks(job['output'], source, 'train')
            backgrounds, background_metadata = read_object_backgrounds(job['output'], source, 'train')
            self.assertTrue(upgraded['background_frames_available'])
            self.assertEqual(metadata, background_metadata)
            self.assertEqual(metadata['source_frames_sha256'], old_metadata['source_frames_sha256'])
            np.testing.assert_array_equal(mask, old_mask)
            with np.load(job['output'], allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['mask_bits'], old_bits)
            with np.load(source, allow_pickle=False) as saved:
                frames = saved['frames']
                self.assertEqual(backgrounds.dtype, np.uint8)
                self.assertEqual(backgrounds.shape, frames.shape)
                np.testing.assert_array_equal(np.any(frames != backgrounds, axis=-1), mask)
                np.testing.assert_array_equal(backgrounds[~mask], frames[~mask])
                game = Game()
                game.set_state(json.loads(str(saved['initial_state'])))
                for index in range(len(frames)):
                    enemies = game.enemies
                    game.enemies = []
                    expected = game.render(size=frames.shape[1], overlays=False)
                    game.enemies = enemies
                    np.testing.assert_array_equal(backgrounds[index], expected)
                    if index < len(saved['actions']):
                        game.step(saved['actions'][index], dt=float(saved['dt']))
            self.assertEqual(before_source, (file_digest(source), source.stat().st_size, source.stat().st_mtime_ns))

    def test_completed_background_read_and_reuse_never_calls_engine_or_decodes_source_rgb(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary)/'train/episode.npz'
            job = {**make_episode(source), 'include_backgrounds': True}
            original = _generate_masks(job)
            sidecar = Path(job['output'])
            identity = (file_digest(sidecar), sidecar.stat().st_mtime_ns)
            with np.load(source, allow_pickle=False) as saved:
                archive_type = type(saved)
            read = archive_type.__getitem__
            def no_source_rgb(archive, key):
                self.assertNotEqual(key, 'frames')
                return read(archive, key)
            with patch.object(Game, '__init__', side_effect=AssertionError('No online engine')), \
                 patch.object(Game, 'render', side_effect=AssertionError('No online render')), \
                 patch.object(Game, 'step', side_effect=AssertionError('No online step')), \
                 patch.object(archive_type, '__getitem__', no_source_rgb):
                frames, metadata = read_object_backgrounds(sidecar, source, 'train')
                self.assertEqual(_generate_masks(job), original)
            self.assertEqual(metadata, original)
            self.assertEqual(frames.shape, (4, 64, 64, 3))
            self.assertEqual((file_digest(sidecar), sidecar.stat().st_mtime_ns), identity)

    def test_corrupted_old_mask_cannot_be_silently_replaced_by_background_upgrade(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary)/'train/episode.npz'
            job = make_episode(source)
            _generate_masks(job)
            change_sidecar(job['output'], lambda payload: payload['mask_bits'].__setitem__((0, 0, 0), payload['mask_bits'][0, 0, 0]^np.uint8(1)))
            before = file_digest(job['output'])
            with self.assertRaisesRegex(ValueError, 'masks differ'):
                _generate_masks({**job, 'include_backgrounds': True})
            self.assertEqual(file_digest(job['output']), before)

    def test_prepare_upgrades_manifest_and_all_records_without_changing_subset(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root/'data'
            train = data/'rgb/train/episode.npz'
            val = data/'rgb/val/episode.npz'
            make_episode(train);make_episode(val, split='val', seed=1073)
            subset = root/'subset.json'
            create_training_subset(train.parent, subset, transitions=3)
            before = {path: (file_digest(path), path.stat().st_mtime_ns) for path in (train, val, subset)}
            output = root/'object_masks'
            with patch('turkeyworld.object_mask_cache.ProcessPoolExecutor', ThreadPoolExecutor):
                legacy = prepare_object_masks(data, output, subset, workers=1)
                upgraded = prepare_object_masks(data, output, subset, workers=1, include_backgrounds=True)
            self.assertFalse(legacy['background_frames_available'])
            self.assertTrue(upgraded['background_frames_available'])
            self.assertTrue(all(record['background_frames_available'] for record in upgraded['records']))
            self.assertEqual(before, {path: (file_digest(path), path.stat().st_mtime_ns) for path in before})

    def test_background_reader_rejects_mask_only_wrong_split_changed_source_and_bad_rgb(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary)/'train/episode.npz'
            job = make_episode(source)
            _generate_masks(job)
            with self.assertRaisesRegex(ValueError, 'no counterfactual'):
                read_object_backgrounds(job['output'], source, 'train')
            _generate_masks({**job, 'include_backgrounds': True})
            with self.assertRaisesRegex(ValueError, 'split mismatch'):
                read_object_backgrounds(job['output'], source, 'val')
            change_sidecar(job['output'], lambda payload: payload.__setitem__('background_rgb', payload['background_rgb'][:-1]))
            with self.assertRaisesRegex(ValueError, 'shape/dtype'):
                read_object_backgrounds(job['output'], source, 'train')

    def test_cli_forwards_explicit_background_upgrade(self):
        fake = {'status': 'complete', 'completed_episodes': 2, 'verified_source_frames': 8}
        with patch('turkeyworld.object_mask_cache.prepare_object_masks', return_value=fake) as prepare:
            main(['--data-root', 'data', '--output', 'cache', '--training-subset-manifest', 'subset', '--include-backgrounds'])
        self.assertTrue(prepare.call_args.kwargs['include_backgrounds'])


@unittest.skipIf(torch is None, 'Optional PyTorch dependency is not installed')
class CounterfactualBackgroundLoadingTests(unittest.TestCase):
    def test_both_modes_attach_only_aligned_float_future_labels_and_resize_like_rgb(self):
        from turkeyworld.jepa_training_utils import SequenceData
        from torch.nn import functional as F
        with TemporaryDirectory() as temporary:
            data, masks, expected = synthetic_backgrounds(Path(temporary))
            for mode in ('memory', 'stream'):
                for image_size in (None, 8):
                    with self.subTest(mode=mode, image_size=image_size):
                        options = dict(mode=mode, image_size=image_size, object_masks_directory=masks/'train')
                        plain = SequenceData(data/'rgb/train', **options).batch(1, context=2, horizon=3, indices=[(0, 1)])
                        dataset = SequenceData(data/'rgb/train', counterfactual_backgrounds=True, **options)
                        with patch.object(Game, 'render', side_effect=AssertionError('No runtime render')), \
                             patch.object(Game, 'step', side_effect=AssertionError('No runtime engine step')):
                            batch = dataset.batch(1, context=2, horizon=3, indices=[(0, 1)])
                        self.assertEqual(set(batch)-set(plain), {'target_background_frames'})
                        for key in plain:
                            torch.testing.assert_close(batch[key], plain[key], rtol=0, atol=0)
                        target = torch.from_numpy(expected[('train', 0)][3:6]).permute(0, 3, 1, 2).float()/255
                        if image_size:
                            target = F.interpolate(target, (8, 8), mode='bilinear', align_corners=False, antialias=True)
                        torch.testing.assert_close(batch['target_background_frames'][0], target, rtol=0, atol=0)
                        self.assertEqual(batch['target_background_frames'].dtype, torch.float32)
                        self.assertTrue(bool(((batch['target_background_frames'] >= 0)&(batch['target_background_frames'] <= 1)).all()))

    def test_unavailable_backgrounds_or_missing_sidecars_fail_without_fallback(self):
        from turkeyworld.jepa_training_utils import SequenceData
        with TemporaryDirectory() as temporary:
            data, masks, _ = loading_fixtures.ObjectMaskLoadingTests.corpus(Path(temporary))
            with self.assertRaisesRegex(ValueError, 'require'):
                SequenceData(data/'rgb/train', counterfactual_backgrounds=True)
            with self.assertRaisesRegex(ValueError, 'unavailable'):
                SequenceData(data/'rgb/train', object_masks_directory=masks/'train', counterfactual_backgrounds=True)
        with TemporaryDirectory() as temporary:
            data, masks, _ = synthetic_backgrounds(Path(temporary))
            dataset = SequenceData(data/'rgb/train', object_masks_directory=masks/'train', counterfactual_backgrounds=True)
            change_sidecar(masks/'train/episode_0000.npz', lambda payload: payload.pop('background_rgb'))
            with self.assertRaisesRegex(ValueError, 'no counterfactual'):
                dataset.batch(1, context=2, horizon=3, indices=[(0, 0)])

    def test_background_lru_is_separate_bounded_and_refreshes_changed_sidecar(self):
        from turkeyworld.jepa_training_utils import SequenceData
        with TemporaryDirectory() as temporary:
            data, masks, _ = synthetic_backgrounds(Path(temporary), count=18)
            dataset = SequenceData(data/'rgb/train', mode='stream', object_masks_directory=masks/'train', counterfactual_backgrounds=True)
            for index in range(18):
                dataset.batch(1, context=2, horizon=3, indices=[(index, 0)])
            self.assertEqual(len(dataset._object_background_cache), 16)
            self.assertEqual(dataset.object_background_cached_bytes, 16*8*4*6*3)
            before = dataset.batch(1, context=2, horizon=3, indices=[(17, 0)])['target_background_frames'].clone()
            change_sidecar(masks/'train/episode_0017.npz', lambda payload: payload.__setitem__('background_rgb', 255-payload['background_rgb']))
            after = dataset.batch(1, context=2, horizon=3, indices=[(17, 0)])['target_background_frames']
            torch.testing.assert_close(after, 1-before, rtol=0, atol=1e-7)
            dataset.clear_cache()
            self.assertFalse(dataset._object_background_cache)
            self.assertEqual(dataset.object_background_cached_bytes, 0)


if __name__ == '__main__':
    unittest.main()
