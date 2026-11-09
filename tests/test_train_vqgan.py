"""Codec training resume contracts and qualification exports, without GPU/LPIPS."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Optional PyTorch dependency is not installed')
class TrainVQGANTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def args(self, folder, **changes):
        values = dict(run_dir=Path(folder), world_source_run=Path(folder)/'source_world',
            data_dir=Path(folder)/'data', device='cpu', seed=73, max_steps=0,
            validate_every=500, disc_start=1000, quality_passes=3,
            temporal_weight=.1, vq_weight=1., adversarial_weight=.1, disc_ramp=1000,
            n_embed=1488, embedding_dim=16, discriminator_width=8, perceptual='none',
            perceptual_weight=1., edge_weight=.1, lr=2e-4, cache_mb=8,
            validation_batch=2, validation_pairs=4, batch=2, save_every=250, log_every=10,
            quality_psnr=40., quality_ssim=.99, quality_mae=1/255, quality_fraction=.95,
            initial_files=8, growth_files=4)
        values.update(changes)
        return SimpleNamespace(**values)

    def tiny_model(self, config=None, weight=2.):
        from turkeyworld.vqgan import VQGANConfig

        class TinyCodec(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(float(weight)))
                self.config = config or VQGANConfig(n_embed=1488)

            def get_config(self):
                return asdict(self.config)

            def encode(self, images):
                one = torch.nn.functional.avg_pool2d(images[:, :1], 4)*self.weight
                return {'quantized': one.repeat(1, self.config.embedding_dim, 1, 1)}

        return TinyCodec()

    def experiment(self, folder, **changes):
        from turkeyworld.train_vqgan import CodecExperiment
        from turkeyworld.vqgan_losses import VQGANObjective
        experiment = CodecExperiment(self.args(folder, **changes))
        experiment.model = self.tiny_model()
        experiment.discriminator = torch.nn.Conv2d(3, 1, 1)
        experiment.objective = VQGANObjective(None, perceptual_weight=experiment.args.perceptual_weight,
            edge_weight=experiment.args.edge_weight, disc_start=experiment.args.disc_start)
        experiment.optimizer = torch.optim.Adam(experiment.model.parameters(), lr=experiment.args.lr)
        experiment.d_optimizer = torch.optim.Adam(experiment.discriminator.parameters(), lr=experiment.args.lr)
        return experiment

    def report(self, experiment, **changes):
        result = {'passed': True, 'frozen': True, 'consecutive_passes': 3,
            'required_passes': 3, 'step': experiment.step, 'fingerprint': 'fixed-panel',
            'summary': {'mae_mean': .001, 'ssim_mean': .999},
            'thresholds': {'psnr': 40., 'ssim': .99, 'mae': 1/255, 'required_fraction': .95}}
        result.update(changes)
        return result

    def test_objective_metadata_includes_every_auxiliary_training_coefficient(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment = self.experiment(folder)
            metadata = experiment.objective_metadata()
            for name in ('temporal_weight', 'vq_weight', 'adversarial_weight', 'disc_ramp'):
                self.assertEqual(metadata[name], getattr(experiment.args, name))
            for name, value in experiment.objective.metadata.items():
                self.assertEqual(metadata[name], value)
            self.assertNotIn('temporal_weight', experiment.objective.metadata)

    def test_restore_rejects_changed_auxiliary_coefficients_from_current_checkpoint_metadata(self):
        for name in ('temporal_weight', 'vq_weight', 'adversarial_weight', 'disc_ramp'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                saved = self.experiment(folder)
                saved.step, saved.streak = 42, 2
                saved.save()
                restored = self.experiment(folder, **{name: getattr(saved.args, name)+1})
                with self.assertRaisesRegex(ValueError, 'objective changed'):
                    restored.restore()

    def test_legacy_metadata_uses_original_config_and_rejects_each_changed_auxiliary_coefficient(self):
        for name in ('temporal_weight', 'vq_weight', 'adversarial_weight', 'disc_ramp'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                saved = self.experiment(folder)
                saved.save()
                path = Path(folder)/'checkpoints'/'vqgan_last.pt'
                payload = torch.load(path, weights_only=False)
                payload['training_objective'] = saved.objective.metadata
                torch.save(payload, path)
                original = {field: getattr(saved.args, field)
                            for field in ('temporal_weight', 'vq_weight', 'adversarial_weight', 'disc_ramp')}
                (Path(folder)/'experiment_config.json').write_text(json.dumps(original), encoding='utf-8')
                compatible = self.experiment(folder)
                compatible.restore()
                changed = self.experiment(folder, **{name: getattr(saved.args, name)+1})
                with self.assertRaisesRegex(ValueError, 'objective changed'):
                    changed.restore()

    def test_restore_preserves_streak_and_exposes_it_in_actual_ui_status(self):
        with tempfile.TemporaryDirectory() as folder:
            saved = self.experiment(folder)
            saved.step, saved.streak, saved.last_validation = 73, 2, 'frozen-panel'
            saved.model.weight.data.fill_(5.)
            saved.save()
            restored = self.experiment(folder)
            restored.restore()
            restored.update()
            self.assertEqual((restored.step, restored.streak), (73, 2))
            self.assertEqual(restored.last_validation, 'frozen-panel')
            self.assertEqual(float(restored.model.weight.detach()), 5.)
            status = json.loads((Path(folder)/'status.json').read_text(encoding='utf-8'))
            self.assertEqual(status['codec_training']['consecutive_passes'], 2)
            self.assertEqual(status['current'], 73)

    def test_export_uses_passed_current_weights_and_only_independent_training_calibration(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment = self.experiment(folder)
            experiment.step, experiment.streak = 17, 3
            experiment.last_validation = 'fixed-panel'
            experiment.panel = [(0, 0), (0, 2), (1, 0), (1, 2)]
            calls = []

            class TrainingData:
                def batch(self, count, rng):
                    calls.append((count, rng, int(rng.integers(1_000_000))))
                    frames = torch.empty(count, 2, 3, 8, 8)
                    frames[:, 0].fill_(.25)
                    frames[:, 1].fill_(.75)
                    return {'frames': frames, 'indices': torch.stack([torch.arange(count), torch.zeros(count)], 1).long()}

            experiment.data = TrainingData()
            experiment.validation = SimpleNamespace(batch=lambda *args, **kwargs: self.fail('No validation calibration'))
            (Path(folder)/'checkpoints').mkdir()
            torch.save({'model': {'weight': torch.tensor(99.)}}, Path(folder)/'checkpoints'/'vqgan_best.pt')
            rng_before = copy.deepcopy(experiment.rng.bit_generator.state)
            report = self.report(experiment)
            path = experiment.export_ready(report)
            exported = torch.load(path, weights_only=False)
            self.assertEqual(float(exported['model']['weight']), 2.)
            self.assertEqual(exported['quality_gate'], report)
            self.assertEqual(exported['training_steps'], 17)
            normalization = exported['latent_normalization']
            self.assertEqual(normalization['source'], 'independent_training_calibration')
            self.assertEqual(normalization['seed'], experiment.args.seed+991)
            np.testing.assert_allclose(normalization['mean'], np.ones(16), rtol=0, atol=0)
            np.testing.assert_allclose(normalization['std'], np.full(16, .5), rtol=0, atol=0)
            self.assertEqual(normalization['samples'], 32)
            self.assertEqual(len(normalization['pair_indices']), 4)
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(rng is not experiment.rng for _, rng, _ in calls))
            self.assertEqual(calls[0][2], int(np.random.default_rng(experiment.args.seed+991).integers(1_000_000)))
            self.assertEqual(experiment.rng.bit_generator.state, rng_before)
            ready = json.loads((Path(folder)/'codec_ready.json').read_text(encoding='utf-8'))
            self.assertTrue(ready['quality_gate_passed'])
            self.assertEqual(ready['consecutive_passes'], 3)

    def test_export_rejects_failed_or_stale_quality_instead_of_publishing_ready(self):
        for changes in ({'passed': False}, {'frozen': False}, {'consecutive_passes': 2}, {'step': 16}):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as folder:
                experiment = self.experiment(folder)
                experiment.step, experiment.streak = 17, 3
                experiment.last_validation = 'fixed-panel'
                experiment.panel = [(0, 0)]
                experiment.data = SimpleNamespace(batch=lambda *args, **kwargs: self.fail('Guard must precede calibration'))
                with self.assertRaises(ValueError):
                    experiment.export_ready(self.report(experiment, **changes))
                self.assertFalse((Path(folder)/'codec_ready.json').exists())

    def test_ready_artifact_rejects_changed_config_or_thresholds_before_data_indexing(self):
        from turkeyworld.train_vqgan import CodecExperiment
        from turkeyworld.vqgan import VQGANConfig
        for change in ('config', 'thresholds'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                args = self.args(folder)
                experiment = CodecExperiment(args)
                (Path(folder)/'codec_ready.json').write_text(json.dumps({'checkpoint': 'qualified.pt'}), encoding='utf-8')
                expected_config = asdict(VQGANConfig(n_embed=args.n_embed, embedding_dim=args.embedding_dim))
                if change == 'config':expected_config['embedding_dim'] += 1
                gate = self.report(experiment)
                if change == 'thresholds':gate['thresholds']['psnr'] = 35.
                verified = {'codec_config': expected_config, 'quality_gate': gate}
                with patch('turkeyworld.train_vqgan.VQGAN', side_effect=lambda config: self.tiny_model(config)), \
                     patch('turkeyworld.train_vqgan.NLayerDiscriminator', return_value=torch.nn.Conv2d(3, 1, 1)), \
                     patch('turkeyworld.train_vqgan.FrozenLPIPSVGG', side_effect=AssertionError('No LPIPS download')), \
                     patch('turkeyworld.train_vqgan.VQGANData', side_effect=AssertionError('No dataset indexing')), \
                     patch('turkeyworld.vq_world.validate_frozen_codec', return_value=verified):
                    with self.assertRaisesRegex(ValueError, 'configuration|thresholds'):
                        experiment.run()


if __name__ == '__main__':
    unittest.main()
