"""Latest stopped codec acceptance preserves failed metrics and original weights."""
from dataclasses import asdict
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Optional PyTorch dependency is not installed')
class FinalizeVQGANTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self, folder):
        from turkeyworld.vqgan import VQGAN, VQGANConfig
        from turkeyworld.vq_world import codec_fingerprint
        root = Path(folder)/'codec'
        checkpoints = root/'checkpoints'
        checkpoints.mkdir(parents=True)
        config = VQGANConfig(image_size=32, base_channels=8, latent_channels=16,
                            embedding_dim=4, n_embed=32, data_init=True, kmeans_iterations=1)
        model = VQGAN(config).train()
        model(torch.rand(1, 3, 32, 32))
        model.eval()
        saved = {'stage': 'vqgan_codec', 'codec_config': asdict(config), 'model': model.state_dict(),
            'training_steps': 11586, 'quality_streak': 0, 'last_validation_fingerprint': 'fixed-heldout-panel',
            'training_objective': {'vq_weight': 1., 'temporal_weight': .1}, 'optimizer': {'preserved': True},
            'world_plan': {'teacher_size': '175m_vqgan', 'from_scratch': True,
                           'image_size': 32, 'target_seconds': 120., 'world_source_run': None}}
        last = checkpoints/'vqgan_last.pt'
        torch.save(saved, last)
        torch.save({'model': {'fake_best': torch.tensor(99.)}, 'training_steps': 5}, checkpoints/'vqgan_best.pt')
        (root/'experiment_config.json').write_text(json.dumps({'seed': 77, 'quality_passes': 3,
            'quality_psnr': 40., 'quality_ssim': .99, 'quality_mae': 1/255,
            'quality_fraction': .95, 'validate_every': 100}), encoding='utf-8')
        panel = [[0, 1], [1, 2]]
        (root/'validation_panel.json').write_text(json.dumps(panel), encoding='utf-8')
        (root/'status.json').write_text(json.dumps({'status': 'stopped', 'codec_training': {'frozen': False}}), encoding='utf-8')
        return root, model, saved, panel, codec_fingerprint(model), hashlib.sha256(last.read_bytes()).hexdigest()

    def report(self):
        from turkeyworld.finalize_vqgan import INTEGRITY_CHECKS
        return {'passed': False, 'fingerprint': 'fixed-heldout-panel', 'step': 11586,
            'summary': {'psnr_mean': 31.7, 'ssim_mean': .941, 'mae_mean': .02},
            'checks': {key: True for key in INTEGRITY_CHECKS}}

    def patches(self, records, report, fingerprint):
        from turkeyworld.vq_world import codec_fingerprint

        class Data:
            def __init__(self, directory, device, *, split, **options):
                self.split = split
                records.append(('open', split, options))

            def fixed_panel(self, **kwargs):
                raise AssertionError('Existing validation panel must be reused')

            def batch(self, count, rng):
                if self.split != 'train':
                    raise AssertionError('Normalization may not sample validation')
                records.append(('calibration', count, int(rng.integers(1_000_000))))
                frames = torch.empty(count, 2, 3, 32, 32)
                frames[:, 0].fill_(.25)
                frames[:, 1].fill_(.75)
                return {'frames': frames, 'indices': torch.stack([torch.arange(count), torch.zeros(count)], 1).long()}

        def evaluate(model, data, panel, **kwargs):
            self.assertEqual(data.split, 'val')
            self.assertEqual(panel, [[0, 1], [1, 2]])
            self.assertEqual(kwargs['step'], 11586)
            self.assertEqual(codec_fingerprint(model), fingerprint)
            records.append(('validation', kwargs))
            frames = np.zeros((4, 32, 32, 3), np.uint8)
            return copy.deepcopy(report), {'source': frames, 'reconstruction': frames,
                'difference': frames, 'metadata': {'step': 11586, 'every': 500,
                    'kind': 'vqgan_codec_reconstruction', 'source_held_out': True}}

        return Data, evaluate

    def test_explicit_acceptance_and_stopped_worker_are_required_before_loading_weights(self):
        from turkeyworld.finalize_vqgan import finalize_current_vqgan
        with self.assertRaisesRegex(ValueError, 'accept-current-quality'):
            finalize_current_vqgan('missing', 'missing', device='cpu')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'training.pid').write_text('12345', encoding='ascii')
            with patch('turkeyworld.finalize_vqgan.process_alive', return_value=True), \
                 patch('turkeyworld.finalize_vqgan.torch.load', side_effect=AssertionError('No checkpoint load')):
                with self.assertRaisesRegex(RuntimeError, 'still running'):
                    finalize_current_vqgan(root, root/'data', device='cpu', accept_current_quality=True)

    def test_latest_weights_export_with_failed_quality_honest_acceptance_and_train_only_normalization(self):
        from turkeyworld.finalize_vqgan import finalize_current_vqgan
        from turkeyworld.vq_world import codec_fingerprint
        with tempfile.TemporaryDirectory() as folder:
            root, model, source, _, fingerprint, original_sha = self.fixture(folder)
            records = []
            data, evaluate = self.patches(records, self.report(), fingerprint)
            with patch('turkeyworld.finalize_vqgan.VQGANData', data), \
                 patch('turkeyworld.finalize_vqgan.evaluate_vqgan', side_effect=evaluate), \
                 patch('turkeyworld.finalize_vqgan.save_vqgan_preview', return_value={'step': 11586}) as preview:
                ready = finalize_current_vqgan(root, root/'data', device='cpu', accept_current_quality=True,
                    validation_batch=2, calibration_pairs=4, train_initial_files=1024)
            exported = torch.load(ready['checkpoint'], weights_only=False)
            self.assertEqual(exported['training_steps'], 11586)
            self.assertEqual(exported['stage'], 'frozen_vqgan_codec')
            self.assertFalse(exported['quality_gate']['passed'])
            self.assertTrue(exported['quality_gate']['frozen'])
            self.assertEqual(exported['quality_gate']['consecutive_passes'], 0)
            self.assertFalse(ready['quality_gate_passed'])
            self.assertTrue(ready['accepted_by_user'])
            self.assertEqual(exported['quality_acceptance']['mode'], 'user_requested')
            self.assertFalse(exported['quality_acceptance']['quality_gate_passed'])
            self.assertEqual(exported['codec_fingerprint'], fingerprint)
            self.assertEqual(exported['training_objective'], source['training_objective'])
            self.assertEqual(exported['world_plan'], source['world_plan'])
            self.assertEqual(ready['world_plan'], source['world_plan'])
            self.assertEqual(ready['codec_config'], source['codec_config'])
            for name, value in exported['model'].items():
                torch.testing.assert_close(value, source['model'][name], rtol=0, atol=0)
            self.assertEqual(hashlib.sha256((root/'checkpoints'/'vqgan_last.pt').read_bytes()).hexdigest(), original_sha)
            preview.assert_called_once()
            self.assertEqual([entry[1] for entry in records if entry[0] == 'open'], ['val', 'train'])
            train_options = next(entry[2] for entry in records if entry[:2] == ('open', 'train'))
            self.assertEqual(train_options['initial_file_limit'], 1024)
            self.assertEqual(train_options['growth_files'], 0)
            self.assertTrue(all(entry[2]['require_native_size'] for entry in records if entry[0] == 'open'))
            normalization = exported['latent_normalization']
            self.assertEqual(normalization['source'], 'independent_training_calibration')
            self.assertEqual(normalization['seed'], 77+991)
            images = torch.cat([torch.full((1, 3, 32, 32), .25), torch.full((1, 3, 32, 32), .75)])
            with torch.no_grad():
                latent = model.encode(images)['quantized'].double()
            expected_mean = latent.mean((0, 2, 3))
            expected_std = (latent.square().mean((0, 2, 3))-expected_mean.square()).clamp_min(1e-6).sqrt()
            np.testing.assert_allclose(normalization['mean'], expected_mean.numpy(), rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(normalization['std'], expected_std.numpy(), rtol=1e-10, atol=1e-10)
            self.assertEqual(codec_fingerprint(model), fingerprint)
            status = json.loads((root/'status.json').read_text(encoding='utf-8'))
            self.assertEqual(status['stage'], 'vqgan_ready_user_accepted')
            self.assertFalse(status['codec_training']['quality_gate_passed'])
            self.assertTrue(status['codec_training']['accepted_by_user'])
            self.assertEqual(status['prediction_every'], 100)
            self.assertEqual(status['prediction_preview']['every'], 100)
            self.assertNotIn('teacher', exported['quality_acceptance']['reason'].lower())
            self.assertNotIn('2026', exported['quality_acceptance']['reason'])

    def test_preview_file_and_status_both_preserve_actual100_update_cadence(self):
        from turkeyworld.finalize_vqgan import finalize_current_vqgan
        with tempfile.TemporaryDirectory() as folder:
            root, _, _, _, fingerprint, _ = self.fixture(folder)
            records = []
            data, evaluate = self.patches(records, self.report(), fingerprint)
            with patch('turkeyworld.finalize_vqgan.VQGANData', data), \
                    patch('turkeyworld.finalize_vqgan.evaluate_vqgan', side_effect=evaluate):
                finalize_current_vqgan(root, root/'data', device='cpu', accept_current_quality=True,
                                       validation_batch=2, calibration_pairs=2)
            preview = json.loads((root/'preview.json').read_text(encoding='utf-8'))
            status = json.loads((root/'status.json').read_text(encoding='utf-8'))
            self.assertEqual(preview['every'], 100)
            self.assertEqual(preview['step'], 11586)
            self.assertEqual(status['prediction_preview'], preview)
            self.assertTrue((root/'preview.png').exists())
            self.assertFalse(status['codec_validation']['passed'])

    def test_finalization_refuses_old128_source_for_native32_codec(self):
        from turkeyworld.finalize_vqgan import finalize_current_vqgan
        with tempfile.TemporaryDirectory() as folder:
            root, _, _, _, _, _ = self.fixture(folder)
            directory = root/'data'/'rgb'/'val'
            directory.mkdir(parents=True)
            np.savez_compressed(directory/'episode.npz', frames=np.zeros((3, 128, 128, 3), np.uint8),
                states=np.zeros((3, 90), np.float32), actions=np.zeros((2, 4), np.float32),
                events=np.zeros((2, 4), np.float32), hit_ids=np.full(2, -1, np.int64),
                attack_flags=np.zeros((2, 8), np.float32), map=np.zeros((16, 16), np.int8),
                split=np.asarray('val'), dt=np.asarray(1/30))
            with patch('turkeyworld.finalize_vqgan.evaluate_vqgan', side_effect=AssertionError('Wrong native size must fail first')):
                with self.assertRaisesRegex(ValueError, 'native source RGB 32x32'):
                    finalize_current_vqgan(root, root/'data', device='cpu', accept_current_quality=True)
            self.assertFalse((root/'codec_ready.json').exists())

    def test_integrity_failure_keeps_actual_final_report_and_never_publishes_ready(self):
        from turkeyworld.finalize_vqgan import finalize_current_vqgan
        with tempfile.TemporaryDirectory() as folder:
            root, _, _, _, fingerprint, _ = self.fixture(folder)
            report = self.report()
            report['checks']['quantization_finite'] = False
            records = []
            data, evaluate = self.patches(records, report, fingerprint)
            with patch('turkeyworld.finalize_vqgan.VQGANData', data), \
                 patch('turkeyworld.finalize_vqgan.evaluate_vqgan', side_effect=evaluate), \
                 patch('turkeyworld.finalize_vqgan.save_vqgan_preview', return_value={'step': 11586}):
                with self.assertRaisesRegex(ValueError, 'integrity'):
                    finalize_current_vqgan(root, root/'data', device='cpu', accept_current_quality=True)
            self.assertFalse((root/'codec_ready.json').exists())
            final = json.loads((root/'codec_validation_final.json').read_text(encoding='utf-8'))
            self.assertFalse(final['checks']['quantization_finite'])
            self.assertFalse(any(entry[:2] == ('open', 'train') for entry in records))


if __name__ == '__main__':
    unittest.main()
