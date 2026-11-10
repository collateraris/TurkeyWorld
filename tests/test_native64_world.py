"""Native64 world/Base/latent-flow execution, budgets and codec lineage."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from turkeyworld.models import NeuralWorldModel, WorldModelConfig, load_world_checkpoint
from turkeyworld.vqgan import VQGAN, VQGANConfig
from turkeyworld.vq_world import (teacher175_vq_config, student_vq_config, codec_fingerprint,
    validate_frozen_codec, vq_profile_parameter_count, TEACHER175_VQ64_PARAMETERS,
    STUDENT_VQ64_PARAMETERS, TEACHER175_VQ_PARAMETERS, STUDENT_VQ_PARAMETERS)


class Native64WorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def tiny():
        codec_config = VQGANConfig(image_size=64, base_channels=8, latent_channels=16,
            embedding_dim=4, n_embed=32, data_init=False, kmeans_iterations=0)
        config = teacher175_vq_config(codec_config, vjepa_width=32, vjepa_heads=4,
            vjepa_encoder_depth=1, vjepa_predictor_depth=1, vjepa_readout_mlp_ratio=1,
            vjepa_gradient_checkpointing=False, flow_width=32, flow_depth=1, flow_heads=4,
            flow_steps=1, flow_adapter_width=0, flow_adapter_depth=0, flow_adapter_heads=0)
        model, codec = NeuralWorldModel(config), VQGAN(codec_config).eval()
        model.flow_renderer.load_codec({'codec_config': codec.get_config(), 'model': codec.state_dict()})
        return model

    @staticmethod
    def state():
        value = torch.zeros(1, STATE_DIM)
        value[:, :2], value[:, 3:6] = .5, 1.
        entities = value[:, GLOBAL_DIM:-2].reshape(1, ENTITY_SLOTS, ENTITY_DIM)
        entities[:, :, :2] = torch.tensor([.7, .6])
        entities[:, :, 4:6] = torch.tensor([.6, 1.])
        return value

    def test_native64_actual_meta_counts_and_all128_budgets_unchanged(self):
        for factory, count64, count128 in ((teacher175_vq_config, TEACHER175_VQ64_PARAMETERS,
                                          TEACHER175_VQ_PARAMETERS),
                                         (student_vq_config, STUDENT_VQ64_PARAMETERS, STUDENT_VQ_PARAMETERS)):
            with self.subTest(factory=factory.__name__):
                config = factory(VQGANConfig(image_size=64, n_embed=1488))
                self.assertEqual(vq_profile_parameter_count(config), count64)
                self.assertEqual(vq_profile_parameter_count(factory(VQGANConfig(n_embed=1488))), count128)
                with torch.device('meta'):
                    model = NeuralWorldModel(config)
                self.assertEqual(model.core.visual_tokens, 64)
                self.assertEqual(model.core.memory_tokens, 264)
                self.assertEqual(config.vjepa_patch_size, 8)
                self.assertEqual(model.flow_renderer.latent_size, 16)
                self.assertEqual(model.flow_renderer.grid, 4)
                self.assertEqual(model.flow_renderer.fixed_noise.shape, (1, 3, 64, 64))
                self.assertEqual(model.vjepa_backend.renderer.rgb64.out_features, 3*4*4)
                self.assertFalse(hasattr(model.vjepa_backend.renderer, 'rgb128'))
                self.assertFalse(hasattr(model.vjepa_backend.renderer, 'rgb256'))

    def test_native64_real_memory_base_flow_gradients_and_frozen_codec_without128_images(self):
        torch.manual_seed(64)
        model = self.tiny().train().requires_grad_(True)
        frozen = {name: value.clone() for name, value in model.flow_renderer.codec.state_dict().items()}
        image_shapes = []
        hook = model.flow_renderer.codec.encoder.conv_in.register_forward_pre_hook(
            lambda module, inputs: image_shapes.append(tuple(inputs[0].shape[-2:])))
        state = self.state()
        fmap = model.encode_map(torch.zeros(1, 16, 16, dtype=torch.long))
        memory = model.initialize_from_image(torch.rand(1, 2, 3, 64, 64), state, fmap)
        output = model.transition(state, torch.tensor([[1., 0., 0., 0.]]), fmap, memory)
        base = model.render_base(output['state'], output['features'], fmap)
        self.assertEqual(tuple(base.shape), (1, 3, 64, 64))
        self.assertEqual(tuple(output['latent_prediction'].shape), (1, 64, 32))
        flow_loss, _ = model.flow_renderer.loss(torch.rand_like(base), base, output['state'])
        rendered = model.flow_renderer.sample(base, output['state'])
        self.assertEqual(tuple(rendered.shape), (1, 3, 64, 64))
        self.assertTrue(bool(torch.isfinite(rendered).all()))
        loss = flow_loss+(base-.3).abs().mean()+output['latent_prediction'].square().mean()
        loss.backward()
        for parameter in (model.core.encoder.patch_embed.weight,
                          model.vjepa_backend.renderer.rgb64.weight,
                          model.flow_renderer.output.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        self.assertTrue(image_shapes and all(shape == (64, 64) for shape in image_shapes))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.flow_renderer.codec.parameters()))
        for name, value in model.flow_renderer.codec.state_dict().items():
            self.assertTrue(torch.equal(value, frozen[name]), name)
        hook.remove()

    def test_native64_checkpoint_roundtrip_and_direct_camera_legacy_shapes(self):
        model = self.tiny().eval()
        with TemporaryDirectory() as directory:
            path = Path(directory)/'world64.pt'
            torch.save({'stage': 'world', 'config': asdict(model.config), 'model': model.state_dict()}, path)
            restored, _ = load_world_checkpoint(path)
            self.assertEqual(restored.config.image_size, 64)
            self.assertTrue(bool(restored.flow_renderer.codec_loaded))
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[name]), name)
        for architecture in ('camera_transformer', 'legacy_cnn'):
            direct = NeuralWorldModel(WorldModelConfig(image_size=64, architecture=architecture)).eval()
            with torch.inference_mode():
                result = direct.step(self.state(), torch.zeros(1, 4),
                                     grid=torch.zeros(1, 16, 16, dtype=torch.long))
            self.assertEqual(tuple(result['rgb'].shape), (1, 3, 64, 64))

    def test_qualified_codec64_validation_checks_resolution_and_preserves_fingerprint(self):
        codec = VQGAN(VQGANConfig(image_size=64, base_channels=8, latent_channels=16,
                                embedding_dim=16, n_embed=1488, data_init=False)).eval()
        payload = {'stage': 'frozen_vqgan_codec', 'codec_config': codec.get_config(),
            'model': codec.state_dict(), 'codec_fingerprint': codec_fingerprint(codec),
            'quality_gate': {'passed': True, 'frozen': True, 'required_passes': 3, 'consecutive_passes': 3},
            'latent_normalization': {'mean': [0.]*16, 'std': [1.]*16}, 'training_steps': 500}
        with TemporaryDirectory() as directory:
            path = Path(directory)/'codec64.pt'
            torch.save(payload, path)
            info = validate_frozen_codec(path, expected_image_size=64)
            self.assertTrue(info['qualified'])
            self.assertEqual(info['codec_config']['image_size'], 64)
            self.assertEqual(info['codec_fingerprint'], payload['codec_fingerprint'])
            with self.assertRaisesRegex(ValueError, 'requested world resolution'):
                validate_frozen_codec(path, expected_image_size=128)
        with self.assertRaisesRegex(ValueError, 'image sizes must match'):
            teacher175_vq_config(VQGANConfig(image_size=64), image_size=128)

    def test_native64_runner_metadata_and_resolution_guard_before_parent_startup(self):
        from turkeyworld.train_vq_world import LatentWorldExperiment
        from turkeyworld.train_500m import MassiveExperiment, build_parser
        codec_info = {'codec_config': asdict(VQGANConfig(image_size=64, n_embed=1488)),
            'generator_parameters': 4_763_731, 'codec_fingerprint': 'new-codec64', 'training_steps': 500,
            'quality_gate': {'passed': True, 'frozen': True, 'required_passes': 3, 'consecutive_passes': 3}}
        with TemporaryDirectory() as directory:
            args = build_parser().parse_args(['--device', 'cpu', '--run-dir', directory])
            args.codec_checkpoint, args.image_size = Path(directory)/'codec.pt', 64
            args.teacher_size, args.from_scratch = '175m_vqgan', True
            with patch('turkeyworld.train_vq_world.validate_frozen_codec', return_value=codec_info) as validate:
                experiment = LatentWorldExperiment(args)
            validate.assert_called_once_with(args.codec_checkpoint, expected_image_size=64)
            self.assertEqual(experiment.state['base_resolution'], [64, 64])
            self.assertEqual(experiment.state['initialization']['image_size'], 64)
            self.assertEqual(experiment.state['model_parameters']['teacher_total'], TEACHER175_VQ64_PARAMETERS)
            self.assertEqual(experiment.state['student_plan']['parameters'], STUDENT_VQ64_PARAMETERS)
            self.assertEqual(experiment.state['student_plan']['config']['image_size'], 64)
            self.assertTrue(args.train_renderer_during_pretrain)
        args = SimpleNamespace(codec_checkpoint=Path('codec.pt'), image_size=128)
        with patch('turkeyworld.train_vq_world.validate_frozen_codec', return_value=codec_info), \
             patch.object(MassiveExperiment, '__init__', side_effect=AssertionError('No startup')):
            with self.assertRaisesRegex(ValueError, 'requested world resolution'):
                LatentWorldExperiment(args)


if __name__ == '__main__':
    unittest.main()
