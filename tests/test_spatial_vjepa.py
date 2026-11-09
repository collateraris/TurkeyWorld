"""Nine spatial entity tokens augment V-JEPA without changing legacy memory."""
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F


SPATIAL_PREFIXES = ('vjepa_backend.spatial_encoder.', 'vjepa_backend.spatial_visual_attention.',
    'vjepa_backend.spatial_motion_context.', 'vjepa_backend.spatial_motion_head.',
    'vjepa_backend.renderer.spatial_encoder.', 'vjepa_backend.renderer.spatial_render.')


class SpatialVJEPATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, spatial=False):
        from turkeyworld.models import NeuralWorldModel, WorldModelConfig
        return NeuralWorldModel(WorldModelConfig(kind='student', architecture='vjepa2_ac',
            image_size=32, vjepa_patch_size=4, endless=True, renderer_mode='direct',
            vjepa_width=32, vjepa_encoder_depth=1, vjepa_predictor_depth=1,
            vjepa_heads=4, vjepa_context_frames=4, spatial_entities=spatial))

    def inputs(self, model, batch=1):
        state = torch.zeros(batch, 90)
        state[:, :2] = torch.tensor([4.5/16, 4.5/16])
        state[:, 3:6] = 1
        npc = state[:, 8:88].reshape(batch, 8, 10)
        npc[:, :, :2] = torch.tensor([[8.5, 4.5], [2.5, 4.5], [9.5, 7.5], [3.5, 12.5],
                                     [12.5, 2.5], [8.5, 12.5], [13.5, 9.5], [1.5, 13.5]])/16
        npc[:, :, 4], npc[:, :, 5] = .6, 1.
        npc[:, 0, 2] = -.2
        npc[:, 7, 4:6] = 0  # Hidden corpse keeps its permanent ninth token.
        grid = torch.zeros(batch, 16, 16, dtype=torch.long)
        grid[:, 0], grid[:, -1], grid[:, :, 0], grid[:, :, -1] = 1, 1, 1, 1
        grid[:, 6, 5:12] = 2
        map_features = model.encode_map(grid)
        rgb = torch.rand(batch, 2, 3, 32, 32)
        memory = model.initialize_from_image(rgb, state, map_features)
        return state, grid, map_features, rgb, memory

    def migrate(self):
        original, enabled = self.model(), self.model(True)
        missing, unexpected = enabled.load_state_dict(original.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(missing)
        self.assertTrue(all(name.startswith(SPATIAL_PREFIXES) for name in missing), missing)
        for name, tensor in original.state_dict().items():
            torch.testing.assert_close(tensor, enabled.state_dict()[name], rtol=0, atol=0)
        return original, enabled

    def test_default_false_preserves_state_dict_and_known_legacy_parameter_count(self):
        from turkeyworld.models import NeuralWorldModel, WorldModelConfig
        torch.manual_seed(17)
        first = self.model()
        config = asdict(first.config)
        config.pop('spatial_entities')  # Simulate an existing checkpoint config.
        torch.manual_seed(17)
        old_config_model = NeuralWorldModel(WorldModelConfig(**config))
        self.assertFalse(old_config_model.config.spatial_entities)
        self.assertEqual(first.state_dict().keys(), old_config_model.state_dict().keys())
        self.assertFalse(any('spatial_' in name for name in first.state_dict()))
        for name, tensor in first.state_dict().items():
            torch.testing.assert_close(tensor, old_config_model.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(first.core.memory_tokens, 264)
        self.assertEqual(first.core.block_tokens, 66)
        with torch.device('meta'):
            legacy = NeuralWorldModel(WorldModelConfig(kind='teacher', image_size=128,
                architecture='vjepa2_ac', endless=True, renderer_mode='flow_matching', flow_steps=12,
                flow_adapter_width=512, flow_adapter_depth=12, flow_adapter_heads=8))
        self.assertEqual(sum(parameter.numel() for parameter in legacy.parameters()), 125_439_429)
        with self.assertRaisesRegex(ValueError, 'boolean'):
            WorldModelConfig(spatial_entities=1)

    def test_legacy_checkpoint_loads_strictly_without_new_config_field(self):
        from turkeyworld.models import load_world_checkpoint
        original = self.model().eval()
        config = asdict(original.config)
        config.pop('spatial_entities')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'old.pt'
            torch.save({'stage': 'world', 'config': config, 'model': original.state_dict(), 'training_steps': 40}, path)
            restored, metadata = load_world_checkpoint(path)
        self.assertEqual(metadata['training_steps'], 40)
        self.assertFalse(restored.config.spatial_entities)
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)

    def test_zero_new_heads_preserve_imported_state_latents_memory_and_rgb_exactly(self):
        original, enabled = self.migrate()
        original.eval(); enabled.eval()
        action = torch.tensor([[.7, .2, .3, 0.]])
        with torch.inference_mode():
            state, grid, fmap, rgb, memory = self.inputs(original)
            new_map = enabled.encode_map(grid)
            new_memory = enabled.initialize_from_image(rgb, state, new_map)
            first = original.step(state, action, memory=memory, map_features=fmap)
            second = enabled.step(state, action, memory=new_memory, map_features=new_map)
        torch.testing.assert_close(fmap, new_map, rtol=0, atol=0)
        torch.testing.assert_close(memory, new_memory, rtol=0, atol=0)
        for name in ('state', 'memory', 'latent_prediction', 'features', 'rgb', 'target_logits', 'attack_logits'):
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
        self.assertEqual(second['memory'].shape, (1, 264, 32))
        self.assertEqual(second['spatial_tokens'].shape, (1, 9, 32))
        self.assertEqual(second['local_motion_residual'].shape, (1, 9, 2))
        self.assertEqual(int(second['local_motion_residual'].count_nonzero()), 0)

    def test_enabled_cross_attention_and_local_motion_heads_receive_training_gradients(self):
        model = self.model(True).train()
        state, _, fmap, _, memory = self.inputs(model, batch=2)
        action = torch.tensor([[.7, .2, .2, 0.], [.4, -.1, -.2, 0.]])
        output = model.transition(state, action, fmap, memory)
        target = torch.randn_like(output['local_motion_prediction'])*.03
        loss = output['latent_prediction'].square().mean()+F.mse_loss(output['local_motion_prediction'], target)
        loss.backward()
        backend = model.vjepa_backend
        for parameter in (backend.spatial_visual_attention.output.weight, backend.spatial_motion_head[-1].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        # Zero output bridges train their outputs first, then their input encoders.
        with torch.no_grad():
            backend.spatial_visual_attention.output.weight.normal_(std=.01)
            backend.spatial_motion_head[-1].weight.normal_(std=.01)
        model.zero_grad(set_to_none=True)
        _, grid, _, rgb, _ = self.inputs(model, batch=2)
        fmap = model.encode_map(grid)
        memory = model.initialize_from_image(rgb, state, fmap)
        output = model.transition(state, action, fmap, memory)
        (output['latent_prediction'].square().mean()+output['local_motion_prediction'].square().mean()).backward()
        self.assertGreater(sum(float(parameter.grad.abs().sum()) for parameter in backend.spatial_encoder.parameters()
                               if parameter.grad is not None), 0)
        self.assertGreater(float(backend.spatial_visual_attention.value.weight.grad.abs().sum()), 0)
        self.assertGreater(float(backend.spatial_motion_context.weight.grad.abs().sum()), 0)
        self.assertTrue(all(bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters() if parameter.grad is not None))

    def test_each_fixed_slot_keeps_its_coordinates_and_drives_a_separate_local_residual(self):
        model = self.model(True).eval()
        state, _, fmap, _, memory = self.inputs(model)
        changed = state.clone()
        changed[:, 78:80] = torch.tensor([2.5/16, 10.5/16])
        backend = model.vjepa_backend
        with torch.no_grad():
            backend.spatial_visual_attention.output.weight.normal_(std=.03)
            backend.spatial_motion_head[-1].weight.normal_(std=.03)
        first = backend.spatial_tokens(state, fmap)
        second = backend.spatial_tokens(changed, fmap)
        self.assertEqual(first.shape, (1, 9, 32))
        torch.testing.assert_close(first[:, :8], second[:, :8], rtol=0, atol=0)
        self.assertGreater(float((first[:, 8]-second[:, 8]).detach().abs().sum()), 0)
        fixed_visual = memory.reshape(1, 4, 66, 32)[:, -1, 2:]
        attention1 = backend.spatial_visual_attention(fixed_visual, first)
        attention2 = backend.spatial_visual_attention(fixed_visual, second)
        self.assertGreater(float((attention1-attention2).detach().abs().sum()), 0)
        context = torch.zeros(1, 9, backend.spatial_motion_context.out_features)
        residual1 = backend.spatial_motion_head(torch.cat([first, context], -1))
        residual2 = backend.spatial_motion_head(torch.cat([second, context], -1))
        torch.testing.assert_close(residual1[:, :8], residual2[:, :8], rtol=0, atol=0)
        self.assertGreater(float((residual1[:, 8]-residual2[:, 8]).detach().abs().sum()), 0)

    def test_local_motion_prediction_uses_postturn_axes_and_matches_world_proposal(self):
        from turkeyworld.spatial_entities import entity_motion_basis, world_to_local, local_to_world
        model = self.model(True).eval()
        state, _, fmap, _, memory = self.inputs(model)
        action = torch.tensor([[.5, .2, 1., 0.]])
        backend = model.vjepa_backend
        with torch.no_grad():
            backend.spatial_motion_head[-1].bias.copy_(torch.tensor([.02, -.01]))
            output = model.transition(state, action, fmap, memory)
            pooled = output['latent_prediction'].mean(1)
            pooled = pooled+backend.readout_neck(pooled)
            old = backend.delta_head(pooled).reshape(1, 9, 2)
        basis = entity_motion_basis(state, action, dt=1/30)
        torch.testing.assert_close(output['local_motion_prior'], world_to_local(old, basis))
        torch.testing.assert_close(output['motion_basis_forward'], basis['forward'])
        torch.testing.assert_close(output['motion_basis_right'], basis['right'])
        torch.testing.assert_close(output['spatial_motion_residual'], local_to_world(output['local_motion_residual'], basis))
        expected = old+local_to_world(output['local_motion_residual'], basis)
        actual = local_to_world(output['local_motion_prediction'], basis)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
        self.assertGreater(float(basis['forward'][0, 0, 1]), 0)

    def test_rollout_uses_only_cached_context_and_preserves_all_nine_tokens(self):
        from turkeyworld.engine import Game
        model = self.model(True).eval()
        state, _, fmap, _, memory = self.inputs(model)
        action = torch.tensor([[.5, 0., .1, 0.]])
        with patch.object(Game, 'step', side_effect=AssertionError('No source steps')), \
             patch.object(Game, 'render', side_effect=AssertionError('No source frames')), \
             patch.object(model.core.encoder, 'forward', side_effect=AssertionError('No future source encoding')):
            with torch.inference_mode():
                for _ in range(3):
                    result = model.step(state, action, memory=memory, map_features=fmap)
                    self.assertEqual(result['spatial_tokens'].shape, (1, 9, 32))
                    self.assertEqual(result['memory'].shape, (1, 264, 32))
                    self.assertTrue(bool(torch.isfinite(result['rgb']).all()))
                    state, memory = result['state'], result['memory']

    def test_complete_teacher_with_spatial_branches_stays_below176m(self):
        from turkeyworld.models import NeuralWorldModel
        from turkeyworld.vqgan import VQGANConfig
        from turkeyworld.vq_world import teacher175_vq_config
        codec = VQGANConfig(image_size=32, n_embed=1488)
        with torch.device('meta'):
            model = NeuralWorldModel(teacher175_vq_config(codec, spatial_entities=True))
        count = sum(parameter.numel() for parameter in model.parameters())
        self.assertGreater(count, 174_461_156)
        self.assertLess(count, 176_000_000)
        self.assertEqual(model.core.memory_tokens, 264)


if __name__ == '__main__':
    unittest.main()
