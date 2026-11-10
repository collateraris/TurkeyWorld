"""The fresh compact teacher budget includes world, Base, Flow and frozen VQGAN."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
from dataclasses import asdict
import unittest

import torch

from turkeyworld.models import NeuralWorldModel, WorldModelConfig
from turkeyworld.vq_world import (teacher175_vq_config, student_vq_config,
    TEACHER175_VQ_PARAMETERS, TEACHER_VQ_PARAMETERS, STUDENT_VQ_PARAMETERS,
    TEACHER175_VQ64_PARAMETERS, TEACHER175_VQ32_PARAMETERS,
    STUDENT_VQ64_PARAMETERS, STUDENT_VQ32_PARAMETERS)
from turkeyworld.vqgan import VQGAN, VQGANConfig


class Teacher175ProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_actual_meta_count_and_component_budget_are_173877948_with_codec(self):
        config = teacher175_vq_config(VQGANConfig(n_embed=1488))
        with torch.device('meta'):
            model = NeuralWorldModel(config)
        count = lambda module: sum(parameter.numel() for parameter in module.parameters())
        self.assertEqual(count(model), TEACHER175_VQ_PARAMETERS)
        self.assertEqual(TEACHER175_VQ_PARAMETERS, 173_877_948)
        components = {'encoder': count(model.core.encoder), 'predictor': count(model.core.predictor),
            'world_readout_heads': count(model.vjepa_backend)-count(model.core)-count(model.vjepa_backend.renderer),
            'base_map_geometry_rgb': count(model.vjepa_backend.renderer),
            'flow_without_codec': count(model.flow_renderer)-count(model.flow_renderer.codec),
            'codec': count(model.flow_renderer.codec)}
        self.assertEqual(components, {'encoder': 57_137_664, 'predictor': 38_369_792,
            'world_readout_heads': 3_696_263, 'base_map_geometry_rgb': 3_877_602,
            'flow_without_codec': 66_032_896, 'codec': 4_763_731})
        self.assertEqual(sum(components.values()), TEACHER175_VQ_PARAMETERS)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 169_114_217)
        self.assertTrue(all(not p.requires_grad for p in model.flow_renderer.codec.parameters()))
        self.assertEqual((model.core.width, len(model.core.encoder.blocks), len(model.core.predictor.blocks)),
                         (512, 18, 12))
        self.assertEqual((len(model.flow_renderer.blocks), len(model.flow_renderer.residual_adapter.blocks)), (6, 12))
        self.assertEqual(model.flow_renderer.steps, 12)
        self.assertTrue(config.vjepa_gradient_checkpointing)

    def test_current_single_enemy_500m_and_student_budgets_include_codec(self):
        from turkeyworld.vjepa2_ac import VJEPA500_PROFILE
        codec = asdict(VQGANConfig(n_embed=1488))
        teacher = WorldModelConfig(kind='teacher', architecture='vjepa2_ac', image_size=128,
            endless=True, renderer_mode='flow_matching', flow_steps=12, renderer_codec='vqgan',
            renderer_codec_config=codec, flow_adapter_width=512, flow_adapter_depth=12,
            flow_adapter_heads=8, **VJEPA500_PROFILE)
        with torch.device('meta'):
            old = NeuralWorldModel(teacher)
            student = NeuralWorldModel(student_vq_config(codec))
        self.assertEqual(sum(p.numel() for p in old.parameters()), TEACHER_VQ_PARAMETERS)
        self.assertEqual(TEACHER_VQ_PARAMETERS, 503_825_084)
        self.assertEqual(sum(p.numel() for p in student.parameters()), STUDENT_VQ_PARAMETERS)
        self.assertEqual(STUDENT_VQ_PARAMETERS, 21_847_868)

    def test_native_64_and_32_counts_keep_core_depth_and_spatial_grids(self):
        for size, teacher_count, student_count in (
                (64, TEACHER175_VQ64_PARAMETERS, STUDENT_VQ64_PARAMETERS),
                (32, TEACHER175_VQ32_PARAMETERS, STUDENT_VQ32_PARAMETERS)):
            with self.subTest(image_size=size), torch.device('meta'):
                codec = VQGANConfig(image_size=size, n_embed=1488)
                teacher = NeuralWorldModel(teacher175_vq_config(codec))
                student = NeuralWorldModel(student_vq_config(codec))
                self.assertEqual(sum(p.numel() for p in teacher.parameters()), teacher_count)
                self.assertEqual(sum(p.numel() for p in student.parameters()), student_count)
                for model in (teacher, student):
                    self.assertEqual(model.config.entity_slots, 1)
                    self.assertEqual(model.core.visual_tokens, 64)
                    self.assertEqual(model.core.memory_tokens, 264)
                    self.assertEqual(model.flow_renderer.codec.config.image_size, size)
        self.assertEqual((TEACHER175_VQ64_PARAMETERS, TEACHER175_VQ32_PARAMETERS),
                         (173_518_380, 173_428_488))
        self.assertEqual((STUDENT_VQ64_PARAMETERS, STUDENT_VQ32_PARAMETERS),
                         (21_635_756, 21_582_728))

    def test_configuration_roundtrip_input_immutability_and_profile_guards(self):
        source = {'n_embed': 1488}
        config = teacher175_vq_config(source)
        self.assertEqual(source, {'n_embed': 1488})
        self.assertEqual(WorldModelConfig(**asdict(config)), config)
        self.assertEqual(config, teacher175_vq_config(VQGANConfig(n_embed=1488)))
        for change in ({'kind': 'student'}, {'architecture': 'legacy_cnn'},
                       {'renderer_codec': 'pixel'}, {'renderer_mode': 'rectified_flow'}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, '175M profile'):
                teacher175_vq_config(source, **change)

    def test_tiny_profile_executes_visual_memory_state_and_cfm_with_frozen_codec(self):
        torch.manual_seed(175)
        codec_config = VQGANConfig(base_channels=8, latent_channels=16, embedding_dim=4,
            n_embed=32, data_init=False, kmeans_iterations=0)
        config = teacher175_vq_config(codec_config, vjepa_width=32,
            vjepa_encoder_depth=1, vjepa_predictor_depth=1, vjepa_heads=4,
            vjepa_readout_mlp_ratio=2, vjepa_gradient_checkpointing=False,
            flow_width=32, flow_depth=1, flow_heads=4, flow_steps=1,
            flow_adapter_width=0, flow_adapter_depth=0, flow_adapter_heads=0)
        model = NeuralWorldModel(config).train()
        codec = VQGAN(codec_config)
        model.flow_renderer.load_codec({'codec_config': codec.get_config(), 'model': codec.state_dict()})
        frozen = {name: value.clone() for name, value in model.flow_renderer.codec.state_dict().items()}
        state = torch.zeros(1, STATE_DIM)
        state[:, :2] = .5
        state[:, 3:6] = 1.
        slots = state[:, GLOBAL_DIM:-2].reshape(1, ENTITY_SLOTS, ENTITY_DIM)
        slots[:, :, :2] = torch.tensor([.7, .6])
        slots[:, :, 4:6] = torch.tensor([.6, 1.])
        fmap = model.encode_map(torch.zeros(1, 16, 16, dtype=torch.long))
        memory = model.initialize_from_image(torch.rand(1, 2, 3, 128, 128), state, fmap)
        result = model.transition(state, torch.tensor([[1., 0., 0., 0.]]), fmap, memory)
        self.assertEqual(result['state'].shape, (1, STATE_DIM))
        self.assertEqual(result['memory'].shape, (1, 264, 32))
        self.assertEqual(result['latent_prediction'].shape, (1, 64, 32))
        self.assertTrue(bool(torch.isfinite(result['state']).all()))
        base = model.render_base(result['state'], result['features'], fmap)
        target = torch.rand_like(base)
        flow_loss, _ = model.flow_renderer.loss(target, base, result['state'])
        loss = result['latent_prediction'][..., 0].square().mean()+flow_loss+(base-target).abs().mean()
        loss.backward()
        for parameter in (model.core.encoder.patch_embed.weight, model.flow_renderer.output.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        for name, value in model.flow_renderer.codec.state_dict().items():
            self.assertTrue(torch.equal(value, frozen[name]), name)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.flow_renderer.codec.parameters()))


if __name__ == '__main__':
    unittest.main()
