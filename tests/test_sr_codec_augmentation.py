"""Frozen accepted-codec SR inputs retain source alignment and reproducible RNG."""
import copy
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from turkeyworld.models import EDSRUpscaler
from turkeyworld.train_sr import SRExperiment, evaluate_sr, load_sr_codec
from turkeyworld.vq_world import codec_fingerprint
from turkeyworld.vqgan import VQGAN, VQGANConfig


class SRCodecAugmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def checkpoint(path):
        model = VQGAN(VQGANConfig(image_size=32, base_channels=8, latent_channels=16,
            embedding_dim=16, n_embed=1488, data_init=False)).eval()
        gate = {'passed': False, 'frozen': True, 'required_passes': 3, 'consecutive_passes': 0,
            'checks': {key: True for key in ('all_frames_finite', 'all_frames_in_range',
                'quantization_finite', 'valid_code_indices', 'codebook_broad_usage')}}
        saved = {'stage': 'frozen_vqgan_codec', 'codec_config': model.get_config(),
            'model': model.state_dict(), 'training_steps': 19000, 'quality_gate': gate,
            'codec_fingerprint': codec_fingerprint(model),
            'latent_normalization': {'mean': [0.]*16, 'std': [1.]*16},
            'quality_acceptance': {'mode': 'user_requested', 'accepted': True,
                'reason': 'User requested freezing current quality', 'accepted_at': '2026-11-09T20:00:00Z',
                'quality_gate_passed': False}}
        torch.save(saved, path)
        return saved

    def test_real_failed_gate_accepted_codec_is_verified_and_reports_honest_provenance(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'codec.pt'
            saved = self.checkpoint(path)
            codec, info = load_sr_codec(path, 'cpu', 32)
            self.assertEqual(info['codec_fingerprint'], saved['codec_fingerprint'])
            self.assertFalse(info['quality_gate_passed'])
            self.assertTrue(info['accepted_by_user'])
            self.assertTrue(info['frozen'])
            self.assertFalse(info['world_generated_rgb'])
            self.assertFalse(codec.training)
            self.assertTrue(all(not p.requires_grad for p in codec.parameters()))
            self.assertEqual(info['source'], 'source_LR_codec_reconstruction')

    def test_forged_acceptance_changed_weights_or_wrong_native_size_are_rejected(self):
        for problem in ('unaccepted', 'weight', 'integrity', 'resolution'):
            with self.subTest(problem=problem), TemporaryDirectory() as directory:
                path = Path(directory)/'codec.pt'
                saved = self.checkpoint(path)
                if problem == 'unaccepted':
                    saved.pop('quality_acceptance')
                elif problem == 'weight':
                    saved['model']['decoder.conv_out.bias'].add_(.1)
                elif problem == 'integrity':
                    saved['quality_gate']['checks']['quantization_finite'] = False
                torch.save(saved, path)
                with self.assertRaises(ValueError):
                    load_sr_codec(path, 'cpu', 64 if problem == 'resolution' else 32)

    def test_mixed_augmentation_matches_only_selected_source_rows_without_global_rng_or_codec_mutation(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'codec.pt'
            self.checkpoint(path)
            codec, info = load_sr_codec(path, 'cpu', 32)
            experiment = object.__new__(SRExperiment)
            experiment.codec, experiment.codec_info = codec, info
            experiment.args = SimpleNamespace(codec_fraction=.5)
            experiment.rng = np.random.default_rng(373)
            expected_rng = np.random.default_rng()
            expected_rng.bit_generator.state = copy.deepcopy(experiment.rng.bit_generator.state)
            selected = expected_rng.random(8) < .5
            self.assertTrue(selected.any() and (~selected).any())
            clean = torch.linspace(0, 1, 8*3*32*32).reshape(8, 3, 32, 32)
            original = clean.clone()
            with torch.inference_mode():
                expected = codec(clean[torch.as_tensor(selected)])['reconstruction']
            fingerprint = codec_fingerprint(codec)
            torch_state = torch.random.get_rng_state().clone()
            numpy_state, python_state = np.random.get_state(), random.getstate()
            augmented, fraction = experiment.training_inputs(clean)
            self.assertTrue(torch.equal(clean, original))
            self.assertTrue(torch.equal(augmented[torch.as_tensor(selected)], expected))
            self.assertTrue(torch.equal(augmented[torch.as_tensor(~selected)], original[torch.as_tensor(~selected)]))
            self.assertEqual(fraction, float(selected.mean()))
            self.assertEqual(experiment.rng.bit_generator.state, expected_rng.bit_generator.state)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))
            actual_numpy = np.random.get_state()
            self.assertEqual(actual_numpy[0], numpy_state[0])
            np.testing.assert_array_equal(actual_numpy[1], numpy_state[1])
            self.assertEqual(actual_numpy[2:], numpy_state[2:])
            self.assertEqual(random.getstate(), python_state)
            self.assertEqual(codec_fingerprint(codec), fingerprint)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in codec.parameters()))
            model = EDSRUpscaler(channels=8, blocks=1).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            F.l1_loss(model(augmented), torch.zeros(8, 3, 128, 128)).backward()
            self.assertGreater(float(model.output.weight.grad.abs().sum()), 0.)
            optimizer.step()
            self.assertEqual(codec_fingerprint(codec), fingerprint)
            self.assertTrue(all(p.grad is None for p in codec.parameters()))

    def test_zero_fraction_does_not_decode_or_advance_augmentation_rng(self):
        experiment = object.__new__(SRExperiment)
        experiment.codec = lambda value: self.fail('Zero fraction may not invoke codec')
        experiment.args = SimpleNamespace(codec_fraction=0.)
        experiment.rng = np.random.default_rng(9)
        before = copy.deepcopy(experiment.rng.bit_generator.state)
        source = torch.rand(2, 3, 32, 32)
        result, fraction = experiment.training_inputs(source)
        self.assertIs(result, source)
        self.assertEqual(fraction, 0.)
        self.assertEqual(before, experiment.rng.bit_generator.state)

    def test_dual_validation_uses_same_source_hr_and_restores_rng_modes_and_codec(self):
        class Data:
            split, image_size, scale, device = 'val', 32, 4, torch.device('cpu')
            episodes = [{'path': Path('episode_51000000.npz')}, {'path': Path('episode_51000001.npz')}]

            def __init__(self):
                self.rng = np.random.default_rng(8)

            def panel_fingerprint(self, panel):
                return 'identical-panel'

            def batch(self, count, indices):
                self.rng.random(); np.random.random(); random.random(); torch.rand(())
                lr = torch.stack([torch.full((3, 32, 32), .2+.1*int(episode)) for episode, frame in indices])
                return {'lr': lr, 'hr': F.interpolate(lr, scale_factor=4, mode='nearest')}

        with TemporaryDirectory() as directory:
            path = Path(directory)/'codec.pt'
            self.checkpoint(path)
            codec, _ = load_sr_codec(path, 'cpu', 32)
            model, data = EDSRUpscaler(channels=8, blocks=1).train(), Data()
            fingerprint = codec_fingerprint(codec)
            torch_state = torch.random.get_rng_state().clone()
            numpy_state, python_state = np.random.get_state(), random.getstate()
            data_rng = copy.deepcopy(data.rng.bit_generator.state)
            panel = [(0, 3), (1, 4)]
            clean, _ = evaluate_sr(model, data, panel, batch_size=2, step=50)
            decoded, preview = evaluate_sr(model, data, panel, batch_size=2, step=50,
                                            codec=codec, input_kind='codec_reconstruction')
            self.assertEqual(clean['source_content_sha256'], decoded['source_content_sha256'])
            self.assertEqual(clean['panel_fingerprint'], decoded['panel_fingerprint'])
            self.assertTrue(clean['primary_quality_metric'])
            self.assertFalse(decoded['primary_quality_metric'])
            self.assertEqual(preview['metadata']['input_kind'], 'codec_reconstruction')
            self.assertFalse(decoded['world_rollout'])
            self.assertTrue(model.training)
            self.assertFalse(codec.training)
            self.assertEqual(codec_fingerprint(codec), fingerprint)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))
            self.assertEqual(data.rng.bit_generator.state, data_rng)
            self.assertEqual(random.getstate(), python_state)
            actual_numpy = np.random.get_state()
            np.testing.assert_array_equal(actual_numpy[1], numpy_state[1])
            self.assertEqual(actual_numpy[2:], numpy_state[2:])


if __name__ == '__main__':
    unittest.main()
