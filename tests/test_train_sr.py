"""Genuine paired SR streaming and real optimizer/RNG resume without GPU."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
import contextlib
import copy
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


def write_pair(directory, name='episode.npz', *, split='train', size=32, identity=1, **changes):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(identity)
    lr = rng.integers(0, 256, (5, size, size, 3), dtype=np.uint8)
    hr = rng.integers(0, 256, (5, size*4, size*4, 3), dtype=np.uint8)
    values = dict(frames=lr, hr_frames=hr, states=np.zeros((5, STATE_DIM), np.float32),
        actions=np.zeros((4, 4), np.float32), split=np.asarray(split),
        frame_source=np.asarray('reference_engine'), hr_frame_source=np.asarray('reference_engine'),
        paired_source_rgb=np.asarray(True), behavior_version=np.asarray('deterministic_enemy_v3_simple_spawn'))
    values.update(changes)
    np.savez_compressed(directory/name, **values)
    return lr, hr


def arguments(root, data, **changes):
    values = dict(run_dir=root, data_dir=data, device='cpu', image_size=32, batch=2, steps=3,
        lr=1e-4, channels=4, blocks=1, edge_weight=.1, validation_frames=4,
        validation_batch=2, validate_every=3, preview_every=2, save_every=1,
        log_every=1, cache_mb=2, threads=2, seed=373)
    values.update(changes)
    return SimpleNamespace(**values)


class PairedSRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_indexing_reads_headers_only_and_batches_retain_exact_pairs_under_eviction(self):
        from turkeyworld.sr_data import SRPairData
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)/'train'
            expected = [write_pair(directory, f'{index}.npz', identity=index+1) for index in range(4)]
            with np.load(directory/'0.npz', allow_pickle=False) as saved:
                npz_type = type(saved)
            get = npz_type.__getitem__

            def small_only(saved, name):
                if name in ('frames', 'hr_frames', 'states', 'actions'):
                    raise AssertionError('Indexing must read array headers, not decoded RGB/state arrays')
                return get(saved, name)

            episode_bytes = sum(array.nbytes for array in expected[0])
            with patch.object(npz_type, '__getitem__', small_only), \
                    patch('turkeyworld.engine.Game.render', side_effect=AssertionError('No renderer in SR loader')):
                data = SRPairData(directory, cache_bytes=episode_bytes*2)
            self.assertEqual(data.raw_bytes, 0)
            indices = [(index, index) for index in range(4)]
            batch = data.batch(4, indices=indices)
            self.assertEqual(tuple(batch['lr'].shape), (4, 3, 32, 32))
            self.assertEqual(tuple(batch['hr'].shape), (4, 3, 128, 128))
            for index in range(4):
                for key, position in (('lr', 0), ('hr', 1)):
                    target = torch.from_numpy(expected[index][position][index]).permute(2, 0, 1).float()/255
                    torch.testing.assert_close(batch[key][index], target, rtol=0, atol=0)
            self.assertLessEqual(data.raw_bytes, episode_bytes*2)
            self.assertEqual(len(data._cache), 2)
            self.assertEqual(data.stats['transitions'], 16)

    def test_unproven_hr_pseudo_source_misalignment_and_wrong_split_fail(self):
        from turkeyworld.sr_data import SRPairData
        cases = [dict(paired_source_rgb=np.asarray(False)), dict(hr_frame_source=np.asarray('upsampled_lr')),
                 dict(frame_source=np.asarray('model_prediction')), dict(states=np.zeros((4, STATE_DIM), np.float32)),
                 dict(hr_frames=np.zeros((5, 64, 64, 3), np.uint8)), dict(split=np.asarray('val'))]
        for index, changes in enumerate(cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as folder:
                directory = Path(folder)/'train'
                write_pair(directory, **changes)
                with self.assertRaises(ValueError):
                    SRPairData(directory)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)/'train'
            write_pair(directory)
            path = directory/'episode.npz'
            with np.load(path, allow_pickle=False) as saved:
                values = {key: saved[key] for key in saved.files if key != 'hr_frames'}
            np.savez_compressed(path, **values)
            with self.assertRaisesRegex(ValueError, 'genuine paired'):
                SRPairData(directory)

    def test_fixed_val_panel_is_unique_reproducible_and_source_mutation_is_rejected(self):
        from turkeyworld.sr_data import SRPairData
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)/'val'
            for index in range(3):
                write_pair(directory, f'{index}.npz', split='val', identity=index+1)
            data = SRPairData(directory)
            panel = data.fixed_panel(12, seed=73)
            self.assertEqual(panel, data.fixed_panel(12, seed=73))
            self.assertEqual(len(set(panel)), 12)
            self.assertEqual(len({episode for episode, _ in panel[:3]}), 3)
            fingerprint = data.panel_fingerprint(panel)
            self.assertEqual(fingerprint, SRPairData(directory).panel_fingerprint(panel))
            data.batch(1, indices=[(0, 0)])
            write_pair(directory, '0.npz', split='val', identity=81)
            with self.assertRaisesRegex(ValueError, 'immutable paired SR source'):
                data.batch(1, indices=[(0, 0)])
            self.assertNotEqual(fingerprint, SRPairData(directory).panel_fingerprint(panel))
            with self.assertRaises(ValueError):
                data.fixed_panel(16)

    def test_append_preserves_existing_indices_and_sampling_seed(self):
        from turkeyworld.sr_data import SRPairData
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)/'train'
            write_pair(directory, 'b.npz')
            data = SRPairData(directory)
            original = data.episodes[0]['path']
            write_pair(directory, 'a.npz', identity=2)
            self.assertEqual(data.refresh(force=True), 1)
            self.assertEqual(data.episodes[0]['path'], original)
            first = data.batch(4, rng=np.random.default_rng(71))
            second = data.batch(4, rng=np.random.default_rng(71))
            for key in first:
                torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'held-out'):
                data.fixed_panel(2)

    def test_actual_optimizer_resume_is_bitexact_and_standard_sr_loader_accepts_checkpoint(self):
        from turkeyworld.models import load_sr_checkpoint
        from turkeyworld.train_sr import SRExperiment
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_pair(root/'data'/'rgb'/'train')
            write_pair(root/'data'/'rgb'/'val', split='val', identity=2)
            with contextlib.redirect_stdout(io.StringIO()):
                baseline = SRExperiment(arguments(root/'baseline', root/'data'))
                baseline_path = baseline.run()
                first = SRExperiment(arguments(root/'resumed', root/'data', steps=1))
                first.run()
                resumed = SRExperiment(arguments(root/'resumed', root/'data'))
                resumed_path = resumed.run()
            self.assertEqual((resumed.step, resumed.examples_trained), (3, 6))
            self.assertEqual(resumed.state['sr_training']['resumed_from_step'], 1)
            for name, value in baseline.model.state_dict().items():
                torch.testing.assert_close(value, resumed.model.state_dict()[name], rtol=0, atol=0)
            saved = torch.load(root/'resumed'/'checkpoints'/'sr_last.pt', weights_only=False)
            original = torch.load(root/'baseline'/'checkpoints'/'sr_last.pt', weights_only=False)
            self.assertEqual(saved['rng_numpy'], original['rng_numpy'])
            torch.testing.assert_close(saved['rng_torch'], original['rng_torch'], rtol=0, atol=0)
            for index, moments in original['optimizer']['state'].items():
                for name, value in moments.items():
                    torch.testing.assert_close(value, saved['optimizer']['state'][index][name], rtol=0, atol=0)
            model, metadata = load_sr_checkpoint(resumed_path, 'cpu')
            self.assertEqual(metadata['stage'], 'sr')
            self.assertEqual(model.config, {'channels': 4, 'blocks': 1, 'scale': 4})
            self.assertEqual(tuple(model(torch.rand(1, 3, 32, 32)).shape), (1, 3, 128, 128))
            self.assertEqual(saved['training_protocol']['image_size'], 32)
            exported = torch.load(resumed_path, weights_only=False)
            self.assertEqual(resumed_path.name, 'sr_for_game.pt')
            self.assertNotIn('optimizer', exported)
            self.assertTrue(exported['training_completed'])
            self.assertEqual((exported['input_size'], exported['output_size'], exported['scale']), (32, 128, 4))
            self.assertEqual(json.loads((root/'resumed'/'sr_ready.json').read_text())['checkpoint'], str(resumed_path))
            self.assertFalse(saved['pretrained_sr_weights_loaded'])
            self.assertEqual(saved['validation']['frames'], 4)
            self.assertEqual(resumed.state['status'], 'completed')
            self.assertFalse(resumed.state['sr_training']['world_rollout_qualified'])
            preview = json.loads((root/'resumed'/'preview.json').read_text(encoding='utf-8'))
            self.assertEqual(preview['native_lr_size'], [32, 32])
            self.assertEqual(preview['native_hr_size'], [128, 128])
            self.assertEqual(preview['preview_image_size'], [512, 576])

    def test_stop_marker_saves_completed_optimizer_boundary_and_changed_protocol_rejects(self):
        from turkeyworld.train_sr import SRExperiment
        class StopAfterOne(SRExperiment):
            def metric(self, metrics):
                super().metric(metrics)
                if self.step == 1:
                    (self.root/'stop_training').touch()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_pair(root/'data'/'rgb'/'train')
            write_pair(root/'data'/'rgb'/'val', split='val')
            experiment = StopAfterOne(arguments(root/'run', root/'data'))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertIsNone(experiment.run())
            self.assertEqual((experiment.step, experiment.state['status']), (1, 'stopped'))
            saved = torch.load(root/'run'/'checkpoints'/'sr_last.pt', weights_only=False)
            self.assertEqual(saved['training_steps'], 1)
            (root/'run'/'stop_training').unlink()
            changed = SRExperiment(arguments(root/'run', root/'data', image_size=64))
            with self.assertRaisesRegex(ValueError, 'native source resolution'):
                changed.run()

    def test_validation_restores_modes_rng_and_fp32_even_inside_autocast(self):
        from turkeyworld.models import SingleFrameUpscaler
        from turkeyworld.sr_data import SRPairData
        from turkeyworld.train_sr import evaluate_sr
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)/'val'
            write_pair(directory, split='val')
            data = SRPairData(directory)
            model = SingleFrameUpscaler(channels=4, blocks=1).train()
            model.body[0].eval()
            modes = [module.training for module in model.modules()]
            rng = copy.deepcopy(data.rng.bit_generator.state)
            torch_state, numpy_state, python_state = torch.random.get_rng_state(), np.random.get_state(), random.getstate()
            forward = model.forward
            calls = []
            def recording(image):
                calls.append((image.shape, image.dtype, torch.is_autocast_enabled('cpu'), torch.is_grad_enabled()))
                torch.rand(1); np.random.rand(); random.random()
                return forward(image)
            with patch.object(model, 'forward', side_effect=recording), torch.autocast('cpu', dtype=torch.bfloat16):
                report, _ = evaluate_sr(model, data, [(0, 0), (0, 2)], batch_size=1)
            self.assertEqual(modes, [module.training for module in model.modules()])
            self.assertEqual(rng, data.rng.bit_generator.state)
            torch.testing.assert_close(torch_state, torch.random.get_rng_state(), rtol=0, atol=0)
            np.testing.assert_equal(numpy_state, np.random.get_state())
            self.assertEqual(python_state, random.getstate())
            self.assertTrue(all(tuple(shape) == (1, 3, 32, 32) and dtype == torch.float32 and not amp and not grad
                                for shape, dtype, amp, grad in calls))
            self.assertTrue(report['source_held_out'])
            self.assertFalse(report['world_rollout'])
            json.dumps(report, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
