"""Qualified-codec migration retains dynamics and valid optimizer progress."""
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Optional PyTorch dependency is not installed')
class VQWorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def codec(self, path):
        from turkeyworld.vqgan import VQGAN, VQGANConfig
        from turkeyworld.vq_world import codec_fingerprint
        model = VQGAN(VQGANConfig(image_size=128, base_channels=8, latent_channels=16,
                                embedding_dim=4, n_embed=32, data_init=False))
        model.quantize.usage_updates.fill_(3)
        payload = {'stage': 'frozen_vqgan_codec', 'codec_config': model.get_config(),
            'model': model.state_dict(), 'codec_fingerprint': codec_fingerprint(model),
            'quality_gate': {'passed': True, 'frozen': True, 'required_passes': 3, 'consecutive_passes': 3},
            'latent_normalization': {'mean': [.1]*4, 'std': [.5]*4}, 'training_steps': 500}
        torch.save(payload, path)
        return model, payload

    def world(self, path, **metadata):
        from turkeyworld.models import NeuralWorldModel, WorldModelConfig
        from turkeyworld.jepa_training_utils import EMATargetEncoder
        from turkeyworld.vq_world import _pretrain_parameters
        model = NeuralWorldModel(WorldModelConfig(kind='teacher', architecture='vjepa2_ac', image_size=128,
            endless=True, renderer_mode='flow_matching', flow_width=32, flow_depth=1, flow_heads=4,
            flow_adapter_width=32, flow_adapter_depth=1, flow_adapter_heads=4,
            vjepa_width=32, vjepa_encoder_depth=1, vjepa_predictor_depth=1,
            vjepa_heads=4, vjepa_context_frames=2)).eval()
        with torch.no_grad():
            model.flow_renderer.fixed_noise.normal_()
            model.flow_renderer.output.weight.normal_(std=.02)
        parameters = [value for _, value in _pretrain_parameters(model)]
        optimizer = torch.optim.AdamW(parameters, lr=1e-4)
        first = parameters[0]
        optimizer.state[first] = {'step': torch.tensor(17.), 'exp_avg': torch.randn_like(first),
                                  'exp_avg_sq': torch.ones_like(first)}
        ema = EMATargetEncoder(model.core.encoder)
        ema.update(model.core.encoder)
        payload = {'stage': 'world', 'config': asdict(model.config), 'model': model.state_dict(),
            'pipeline_phase': 'teacher500_world', 'training_steps': 17, 'global_training_step': 211,
            'optimizer': optimizer.state_dict(), 'ema_target_encoder': ema.state_dict(),
            'rng_numpy': np.random.default_rng(717).bit_generator.state,
            'rng_torch': torch.random.get_rng_state(), 'rng_cuda': [],
            'teacher_selection': {'old': True}, 'teacher_quality_gate': {'passed': False},
            'adversarial_training': {'old': True}, 'phase_complete': True, **metadata}
        torch.save(payload, path)
        return model, payload

    def test_student_factory_budget_includes_all_codec_weights_and_old_defaults_are_untouched(self):
        from turkeyworld.vq_world import student_vq_config, STUDENT_VQ_PARAMETERS, TEACHER_VQ_PARAMETERS
        from turkeyworld.vqgan import VQGANConfig
        from turkeyworld.models import NeuralWorldModel, WorldModelConfig
        config = student_vq_config(asdict(VQGANConfig(n_embed=1488)))
        self.assertEqual((config.vjepa_encoder_depth, config.vjepa_predictor_depth,
                          config.vjepa_readout_mlp_ratio, config.flow_mlp_hidden), (4, 5, 5, 736))
        self.assertEqual((config.renderer_codec, config.flow_steps, config.vjepa_width), ('vqgan', 4, 320))
        with torch.device('meta'):
            model = NeuralWorldModel(config)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), STUDENT_VQ_PARAMETERS)
        self.assertEqual(STUDENT_VQ_PARAMETERS, 22_475_096)
        self.assertEqual(TEACHER_VQ_PARAMETERS, 504_857_752)
        self.assertEqual(WorldModelConfig().renderer_codec, 'pixel')
        self.assertEqual(WorldModelConfig().vjepa_encoder_depth, 0)

    def test_migration_retains_core_and_compatible_renderer_weights_and_resumes_optimizer_exactly(self):
        from turkeyworld.vq_world import migrate_teacher_to_vq, _pretrain_parameters
        from turkeyworld.models import load_world_checkpoint
        from turkeyworld.jepa_training_utils import EMATargetEncoder
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source, codec = folder/'source.pt', folder/'codec.pt'
            initial, resume = folder/'initial.pt', folder/'resume.pt'
            original, payload = self.world(source)
            _, codec_payload = self.codec(codec)
            report = migrate_teacher_to_vq(source, codec, initial, resume, require_qualified=False)
            model, initial_payload = load_world_checkpoint(initial)
            _, resumed = load_world_checkpoint(resume)
            self.assertTrue(report['retained_core_all_values_exact'])
            self.assertTrue(report['optimizer_preserved'])
            self.assertTrue(report['ema_preserved'])
            self.assertTrue(report['rng_preserved'])
            self.assertFalse(report['renderer_function_preserved'])
            self.assertEqual(resumed['training_steps'], 17)
            self.assertEqual(resumed['global_training_step'], 211)
            self.assertEqual(resumed['pipeline_phase'], 'teacher500vq_world')
            self.assertFalse(resumed['phase_complete'])
            self.assertNotIn('optimizer', initial_payload)
            for key in ('teacher_selection', 'teacher_quality_gate', 'adversarial_training'):
                self.assertNotIn(key, resumed)
            differences = {key for key in asdict(model.config)
                           if asdict(model.config)[key] != asdict(original.config)[key]}
            self.assertEqual(differences, {'renderer_codec', 'renderer_codec_config'})
            for name, value in original.state_dict().items():
                if not name.startswith('flow_renderer.'):
                    torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
            torch.testing.assert_close(model.flow_renderer.fixed_noise, original.flow_renderer.fixed_noise, rtol=0, atol=0)
            for name in ('position', 'residual_adapter.position'):
                before = original.flow_renderer.state_dict()[name]
                expected = torch.nn.functional.interpolate(before.transpose(1, 2).reshape(1, 32, 16, 16),
                    size=(8, 8), mode='bilinear', align_corners=False).flatten(2).transpose(1, 2)
                torch.testing.assert_close(model.flow_renderer.state_dict()[name], expected, rtol=0, atol=0)
            self.assertIn('flow_renderer.output.weight', report['reinitialized_source_renderer_keys'])
            self.assertTrue(bool(model.flow_renderer.codec_loaded))
            self.assertTrue(all(not value.requires_grad for value in model.flow_renderer.codec.parameters()))
            torch.testing.assert_close(model.flow_renderer.latent_mean, torch.full((1, 4, 1, 1), .1), rtol=0, atol=0)
            self.assertEqual(initial_payload['codec_fingerprint'], codec_payload['codec_fingerprint'])
            optimizer = torch.optim.AdamW([value for _, value in _pretrain_parameters(model)], lr=1e-4)
            optimizer.load_state_dict(resumed['optimizer'])
            torch.testing.assert_close(optimizer.state_dict()['state'][0]['exp_avg'],
                                       payload['optimizer']['state'][0]['exp_avg'], rtol=0, atol=0)
            ema = EMATargetEncoder(model.core.encoder)
            ema.load_state_dict(resumed['ema_target_encoder'])
            self.assertEqual(ema.steps, 1)
            torch.testing.assert_close(resumed['rng_torch'], payload['rng_torch'], rtol=0, atol=0)
            self.assertEqual(resumed['rng_numpy'], payload['rng_numpy'])

    def test_incompatible_phase_or_optimizer_produces_initial_only_with_honest_reset_reason(self):
        from turkeyworld.vq_world import migrate_teacher_to_vq
        for problem in ('phase', 'optimizer'):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                source, codec = folder/'source.pt', folder/'codec.pt'
                _, payload = self.world(source)
                self.codec(codec)
                if problem == 'phase':
                    payload['pipeline_phase'] = 'teacher_quality_500m'
                else:
                    payload['optimizer']['param_groups'][0]['params'].pop()
                torch.save(payload, source)
                initial, resume = folder/'initial.pt', folder/'resume.pt'
                report = migrate_teacher_to_vq(source, codec, initial, resume, require_qualified=False)
                self.assertTrue(initial.exists())
                self.assertFalse(resume.exists())
                self.assertFalse(report['optimizer_preserved'])
                self.assertFalse(report['resume_compatible'])
                self.assertIsNone(report['resume_checkpoint'])
                self.assertTrue(report['resume_reason'])

    def test_bad_frozen_quality_fingerprint_or_missing_actual_codec_state_is_rejected(self):
        from turkeyworld.vq_world import validate_frozen_codec
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'codec.pt'
            _, payload = self.codec(path)
            verified = validate_frozen_codec(path, require_qualified=False)
            self.assertEqual(verified['codec_fingerprint'], payload['codec_fingerprint'])
            for problem in ('gate', 'streak', 'fingerprint', 'initialization', 'normalization'):
                _, bad = self.codec(path)
                if problem == 'gate':bad['quality_gate']['passed'] = False
                elif problem == 'streak':bad['quality_gate']['consecutive_passes'] = 2
                elif problem == 'fingerprint':bad['model']['decoder.conv_out.weight'].add_(1.)
                elif problem == 'initialization':bad['model']['quantize.initialized'].fill_(False)
                else:bad.pop('latent_normalization')
                torch.save(bad, path)
                with self.subTest(problem=problem), self.assertRaises(ValueError):
                    validate_frozen_codec(path, require_qualified=problem in ('gate', 'streak'))

    def test_migration_never_overwrites_sources_or_existing_outputs(self):
        from turkeyworld.vq_world import migrate_teacher_to_vq
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source, codec, target = folder/'source.pt', folder/'codec.pt', folder/'target.pt'
            self.world(source)
            self.codec(codec)
            target.write_bytes(b'existing')
            for output in (target, source, codec):
                with self.subTest(output=output), self.assertRaises(FileExistsError):
                    migrate_teacher_to_vq(source, codec, output, require_qualified=False)
            self.assertEqual(target.read_bytes(), b'existing')


if __name__ == '__main__':
    unittest.main()
