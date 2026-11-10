"""Actual native32 aligned world/CFM/Heun/GAN gradients without RGB resizing."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

from turkeyworld.adversarial_training import AdversarialTrainer
from turkeyworld.aligned_training import aligned_short_rollout_loss
from turkeyworld.jepa_training_utils import EMATargetEncoder
from turkeyworld.models import NeuralWorldModel
from turkeyworld.object_losses import weighted_image_l1
from turkeyworld.vqgan import VQGAN, VQGANConfig
from turkeyworld.vq_world import (teacher175_vq_config, student_vq_config, vq_profile_parameter_count,
    TEACHER175_VQ32_PARAMETERS, STUDENT_VQ32_PARAMETERS, codec_fingerprint, validate_frozen_codec)


class Native32WorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_native32_meta_parameter_counts_and_spatial_shapes(self):
        for factory, expected in ((teacher175_vq_config, TEACHER175_VQ32_PARAMETERS),
                                  (student_vq_config, STUDENT_VQ32_PARAMETERS)):
            config = factory(VQGANConfig(image_size=32, n_embed=1488))
            self.assertEqual(vq_profile_parameter_count(config), expected)
            with torch.device('meta'):
                model = NeuralWorldModel(config)
            self.assertEqual(config.image_size, 32)
            self.assertEqual(config.vjepa_patch_size, 4)
            self.assertEqual(model.core.visual_tokens, 64)
            self.assertEqual(model.core.memory_tokens, 264)
            self.assertEqual(model.flow_renderer.latent_size, 8)
            self.assertEqual(model.flow_renderer.grid**2, 4)
            self.assertEqual(model.vjepa_backend.renderer.rgb32.out_features, 12)
            self.assertFalse(hasattr(model.vjepa_backend.renderer, 'rgb64'))
            self.assertFalse(hasattr(model.vjepa_backend.renderer, 'rgb128'))
            self.assertFalse(hasattr(model.vjepa_backend.renderer, 'rgb256'))

    def test_real_native32_aligned_cfm_full_heun_gan_r1_backward_codec_frozen(self):
        torch.manual_seed(32)
        codec_config = VQGANConfig(image_size=32, base_channels=8, latent_channels=16,
            embedding_dim=4, n_embed=32, data_init=False, kmeans_iterations=0)
        model = NeuralWorldModel(teacher175_vq_config(codec_config, vjepa_width=32, vjepa_heads=4,
            vjepa_encoder_depth=1, vjepa_predictor_depth=1, vjepa_readout_mlp_ratio=1,
            vjepa_gradient_checkpointing=False, flow_width=32, flow_depth=1, flow_heads=4,
            flow_adapter_width=0, flow_adapter_depth=0, flow_adapter_heads=0)).train()
        codec = VQGAN(codec_config).eval()
        model.flow_renderer.load_codec({'codec_config': codec.get_config(), 'model': codec.state_dict()})
        frozen = {name: value.clone() for name, value in model.flow_renderer.codec.state_dict().items()}
        frames = torch.rand(1, 4, 3, 32, 32)
        states = torch.zeros(1, 4, STATE_DIM)
        states[:, :, :2], states[:, :, 3:6] = .5, 1.
        entities = states[:, :, GLOBAL_DIM:-2].reshape(1, 4, ENTITY_SLOTS, ENTITY_DIM)
        entities[:, :, :, :2] = torch.tensor([.7, .6])
        entities[:, :, :, 4:6] = torch.tensor([.6, 1.])
        masks = torch.zeros(1, 2, 1, 32, 32)
        masks[:, :, :, 8:24, 9:22] = 1.
        batch = {'frames': frames, 'states': states, 'context_frames': frames[:, :2],
            'target_frames': frames[:, 2:], 'previous_states': states[:, 1:-1],
            'target_states': states[:, 2:], 'actions': torch.ones(1, 2, 4),
            'events': torch.zeros(1, 2, 4), 'hit_ids': torch.full((1, 2), -1, dtype=torch.long),
            'attack_flags': torch.zeros(1, 2, ENTITY_SLOTS), 'target_object_masks': masks,
            'map': torch.zeros(1, 16, 16, dtype=torch.long)}
        loss, metrics, endpoint = aligned_short_rollout_loss(model, EMATargetEncoder(model.core.encoder), batch,
            object_region_boost=4., detach_flow_base=True)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(endpoint['base'].shape[-2:], (32, 32))
        self.assertIn('flow', metrics)
        loss.backward()
        core_gradient = model.core.encoder.patch_embed.weight.grad.clone()
        generated = model.flow_renderer.sample(endpoint['base'].detach(), endpoint['state'].detach(),
                                                checkpoint_velocity=True)
        self.assertEqual(model.flow_renderer.steps, 12)
        self.assertEqual(generated.shape[-2:], (32, 32))
        trainer = AdversarialTrainer('cpu', base_channels=4, num_scales=1,
            weight=.005, feature_weight=.01, warmup_steps=0, ramp_steps=1,
            r1_gamma=.1, r1_interval=1)
        observed = []
        hook = trainer.discriminator.discriminators[0].register_forward_pre_hook(
            lambda module, inputs: observed.append(tuple(inputs[0].shape[-2:])))
        gan, gan_metrics = trainer.generator_loss(generated, endpoint['target'],
            endpoint['base'].detach(), endpoint['state'].detach())
        refinement = gan+.5*weighted_image_l1(generated, endpoint['target'], endpoint['pixel_weights'])
        refinement.backward()
        hook.remove()
        self.assertTrue(observed and all(shape == (32, 32) for shape in observed))
        self.assertTrue(bool(torch.isfinite(refinement)))
        self.assertEqual(trainer.updates, 1)
        self.assertGreater(gan_metrics['r1'], 0.)
        self.assertTrue(torch.equal(core_gradient, model.core.encoder.patch_embed.weight.grad))
        self.assertGreater(float(core_gradient.abs().sum()), 0.)
        for parameter in (model.vjepa_backend.renderer.rgb32.weight, model.flow_renderer.output.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        for name, value in model.flow_renderer.codec.state_dict().items():
            self.assertTrue(torch.equal(value, frozen[name]), name)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.flow_renderer.codec.parameters()))

    def test_setup_selects_native32_one_scale_and_preserves_legacy_configuration(self):
        from turkeyworld.training_pipeline import Experiment
        for size, scales in ((32, 1), (64, 2), (128, 2)):
            with self.subTest(size=size), TemporaryDirectory() as directory:
                experiment = object.__new__(Experiment)
                experiment.root, experiment.device = Path(directory), torch.device('cpu')
                experiment.args = SimpleNamespace(adversarial=True, image_size=size)
                experiment.state = {}
                experiment._adversarial_trainer = experiment._adversarial_phase = None
                experiment._setup_adversarial('teacher_world')
                trainer = experiment._adversarial_trainer
                self.assertEqual(trainer.discriminator.num_scales, scales)
                self.assertEqual(trainer.config.get('num_scales', 2), scales)
                if scales == 2:
                    self.assertNotIn('num_scales', trainer.config)
        old = AdversarialTrainer('cpu', base_channels=4)
        restored = AdversarialTrainer('cpu', base_channels=4)
        restored.load_state_dict(old.state_dict())
        self.assertEqual(restored.config, old.config)

    def test_qualified_codec32_matches_requested_native_resolution(self):
        codec = VQGAN(VQGANConfig(image_size=32, base_channels=8, latent_channels=16,
            embedding_dim=16, n_embed=1488, data_init=False)).eval()
        payload = {'stage': 'frozen_vqgan_codec', 'codec_config': codec.get_config(),
            'model': codec.state_dict(), 'codec_fingerprint': codec_fingerprint(codec),
            'quality_gate': {'passed': True, 'frozen': True, 'consecutive_passes': 3, 'required_passes': 3},
            'latent_normalization': {'mean': [0.]*16, 'std': [1.]*16}}
        with TemporaryDirectory() as directory:
            path = Path(directory)/'codec32.pt'
            torch.save(payload, path)
            self.assertEqual(validate_frozen_codec(path, expected_image_size=32)['codec_config']['image_size'], 32)
            with self.assertRaisesRegex(ValueError, 'requested world resolution'):
                validate_frozen_codec(path, expected_image_size=128)


if __name__ == '__main__':
    unittest.main()
