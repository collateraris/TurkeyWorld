"""Native64 codec training, resolution lineage and unchanged strict gates."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


def write_clip(directory, *, size=64, split='train', name='episode.npz'):
    directory.mkdir(parents=True, exist_ok=True)
    frames = np.random.default_rng(71).integers(0, 256, (5, size, size, 3), dtype=np.uint8)
    events = np.zeros((4, 4), np.float32)
    events[1, 0] = 1
    path = directory/name
    np.savez_compressed(path, frames=frames, states=np.zeros((5, STATE_DIM), np.float32),
        actions=np.zeros((4, 4), np.float32), events=events,
        hit_ids=np.full(4, -1, np.int64), attack_flags=np.zeros((4, ENTITY_SLOTS), np.float32),
        map=np.zeros((16, 16), np.int8), split=np.asarray(split), dt=np.asarray(1/30),
        policy=np.asarray('combat'), frame_source=np.asarray('engine'))
    return path, frames


def arguments(root, **changes):
    values = dict(run_dir=root/'codec', data_dir=root/'data', world_source_run=root/'old_world',
        device='cpu', seed=73, max_steps=1, image_size=64, teacher_size='175m_vqgan',
        target_seconds=120., world_from_scratch=True, validate_every=500, disc_start=0,
        quality_passes=3, temporal_weight=.1, vq_weight=1., adversarial_weight=.1,
        disc_ramp=1, n_embed=1488, embedding_dim=16, discriminator_width=8,
        perceptual='none', perceptual_weight=1., edge_weight=.1, lr=2e-4,
        cache_mb=8, validation_batch=2, validation_pairs=2, batch=1,
        save_every=250, log_every=10, quality_psnr=40., quality_ssim=.99,
        quality_mae=1/255, quality_fraction=.95, initial_files=0, growth_files=0)
    values.update(changes)
    return SimpleNamespace(**values)


class Native64CodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_64_profile_keeps_capacity_and_has_16_by_16_codes(self):
        from turkeyworld.vqgan import VQGAN, VQGANConfig
        for size, latent in ((32, 8), (64, 16), (128, 32)):
            config = VQGANConfig(image_size=size, n_embed=1488)
            with torch.device('meta'):
                model = VQGAN(config)
            self.assertEqual(sum(p.numel() for p in model.parameters()), 4_763_731)
            self.assertEqual((config.latent_size, config.embedding_dim), (latent, 16))

    def test_actual32_codec_and_patchgan_optimizer_step(self):
        from turkeyworld.train_vqgan import CodecExperiment
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_clip(root/'data'/'rgb'/'train', size=32)
            write_clip(root/'data'/'rgb'/'val', size=32, split='val')
            experiment = CodecExperiment(arguments(root, image_size=32))
            with contextlib.redirect_stdout(io.StringIO()):
                experiment.run()
            self.assertEqual(experiment.step, 1)
            self.assertEqual(experiment.model.config.latent_size, 8)
            self.assertEqual(tuple(experiment.data.batch(1)['frames'].shape), (1, 2, 3, 32, 32))
            self.assertTrue(experiment.optimizer.state)
            self.assertTrue(experiment.d_optimizer.state)
            self.assertEqual(experiment.state['codec_training']['config']['image_size'], 32)
            self.assertFalse((root/'codec'/'codec_ready.json').exists())

    def test_native64_pairs_are_exact_and_128_sources_are_rejected_without_rgb_decode(self):
        from turkeyworld.vqgan_data import VQGANData
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, frames = write_clip(root/'train')
            data = VQGANData(root/'train', image_size=64, require_native_size=True)
            batch = data.batch(1, indices=[(0, 1)])
            torch.testing.assert_close(batch['frames'][0],
                torch.from_numpy(frames[1:3]).permute(0, 3, 1, 2).float()/255, rtol=0, atol=0)
            self.assertEqual(tuple(batch['frames'].shape), (1, 2, 3, 64, 64))
            self.assertEqual(data.stats['image_size'], 64)
            write_clip(root/'wrong'/'train', size=128)
            with patch('turkeyworld.streaming_sequences._load', side_effect=AssertionError('No RGB decoding')):
                with self.assertRaisesRegex(ValueError, 'native source RGB 64x64'):
                    VQGANData(root/'wrong'/'train', image_size=64, require_native_size=True)
            # Existing generic data clients may still explicitly resize.
            resized = VQGANData(root/'wrong'/'train', image_size=64)
            self.assertEqual(tuple(resized.batch(1)['frames'].shape), (1, 2, 3, 64, 64))

    def test_growth_cannot_admit_another_native_resolution(self):
        from turkeyworld.vqgan_data import VQGANData
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)/'train'
            write_clip(directory)
            data = VQGANData(directory, image_size=64, require_native_size=True)
            write_clip(directory, size=128, name='wrong.npz')
            with self.assertRaisesRegex(ValueError, 'native source RGB'):
                data.refresh(force=True)

    def test_actual64_codec_and_patchgan_update_from_scratch_and_save_resolution(self):
        from turkeyworld.train_vqgan import CodecExperiment
        from turkeyworld.vqgan import VQGAN, VQGANConfig
        from turkeyworld.vqgan_validation import evaluate_vqgan, save_vqgan_preview
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_clip(root/'data'/'rgb'/'train')
            write_clip(root/'data'/'rgb'/'val', split='val')
            experiment = CodecExperiment(arguments(root))
            initial = {}

            def create(config):
                model = VQGAN(config)
                initial['encoder'] = model.encoder.conv_in.weight.detach().clone()
                initial['decoder'] = model.get_last_layer().detach().clone()
                initial['codebook'] = model.quantize.embedding.weight.detach().clone()
                return model

            with patch('turkeyworld.train_vqgan.VQGAN', side_effect=create), contextlib.redirect_stdout(io.StringIO()):
                self.assertIsNone(experiment.run())
            self.assertEqual(experiment.step, 1)
            self.assertEqual(experiment.examples_trained, 2)
            self.assertFalse(torch.equal(initial['encoder'], experiment.model.encoder.conv_in.weight))
            self.assertFalse(torch.equal(initial['decoder'], experiment.model.get_last_layer()))
            self.assertFalse(torch.equal(initial['codebook'], experiment.model.quantize.embedding.weight))
            self.assertTrue(experiment.d_optimizer.state)
            self.assertTrue(experiment.optimizer.state)
            checkpoint = torch.load(root/'codec'/'checkpoints'/'vqgan_last.pt', weights_only=False)
            self.assertEqual(checkpoint['codec_config']['image_size'], 64)
            self.assertEqual(checkpoint['codec_config']['n_embed'], 1488)
            self.assertEqual(checkpoint['world_plan']['teacher_size'], '175m_vqgan')
            self.assertEqual(checkpoint['world_plan']['target_seconds'], 120.)
            self.assertIsNone(checkpoint['world_plan']['world_source_run'])
            self.assertNotIn('teacher_total', experiment.state['model_parameters'])
            self.assertEqual(experiment.state['model_parameters']['target_teacher_total'], 175_000_000)
            self.assertFalse((root/'codec'/'codec_ready.json').exists())
            report, preview = evaluate_vqgan(experiment.model, experiment.validation, [(0, 0), (0, 2)], batch_size=1)
            self.assertFalse(report['passed'])
            self.assertEqual(report['image_size'], 64)
            self.assertEqual(report['thresholds'], {'psnr': 40., 'ssim': .99, 'mae': 1/255, 'required_fraction': .95})
            self.assertEqual(report['codebook']['total_tokens'], 4*16*16)
            self.assertEqual(tuple(preview['source'].shape), (4, 64, 64, 3))
            self.assertEqual(preview['metadata']['source_image_size'], [64, 64])
            metadata = save_vqgan_preview(preview, root/'preview.png')
            self.assertEqual(metadata['image_size'], [512, 430])
            self.assertEqual(metadata['source_image_size'], [64, 64])
            # Equal parameter shapes do not authorize loading a128 codec checkpoint into64.
            experiment.model.config = VQGANConfig(image_size=128, n_embed=1488)
            with self.assertRaisesRegex(ValueError, 'configuration differs'):
                experiment.restore()

    def test_validation_rejects_wrong_resolution_before_codec_forward(self):
        from turkeyworld.vqgan import VQGAN, VQGANConfig
        from turkeyworld.vqgan_data import VQGANData
        from turkeyworld.vqgan_validation import evaluate_vqgan
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)/'val'
            write_clip(directory, split='val', size=128)
            data = VQGANData(directory, image_size=128)
            model = VQGAN(VQGANConfig(image_size=64, base_channels=8, n_embed=8))
            with patch.object(model, 'forward', side_effect=AssertionError('Do not infer wrong resolution')):
                with self.assertRaisesRegex(ValueError, 'source resolution'):
                    evaluate_vqgan(model, data, [(0, 0)])

    def test_cli_exposes_64_fresh175_plan_without_enabling_legacy_migration(self):
        from turkeyworld.train_vqgan import main
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = ['--run-dir', str(root), '--image-size', '64', '--teacher-size', '175m_vqgan',
                    '--target-seconds', '120', '--world-from-scratch', '--device', 'cpu']
            with patch('turkeyworld.train_vqgan.CodecExperiment') as factory, \
                    patch('turkeyworld.train_vqgan.KeepAwake'):
                factory.return_value.state = {}
                factory.return_value.run.return_value = None
                main(args)
                parsed = factory.call_args.args[0]
                self.assertEqual((parsed.image_size, parsed.teacher_size, parsed.target_seconds), (64, '175m_vqgan', 120.))
                self.assertTrue(parsed.world_from_scratch)
            with patch('turkeyworld.train_vqgan.CodecExperiment') as factory, \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main(args+['--continue-world'])
                factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
