"""Exactly one persistent NPC; latent image grids and anchor count stay spatial."""
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import torch

from turkeyworld.models import NeuralWorldModel, WorldModelConfig, ENTITY_SLOTS, STATE_DIM, load_world_checkpoint
from turkeyworld.structured_state import structured_transition
from turkeyworld.single_enemy_migration import migrate_single_enemy_checkpoint, legacy_state_indices
from turkeyworld.vqgan import VQGAN, VQGANConfig
from turkeyworld.vq_world import teacher175_vq_config


def model():
    codec = VQGANConfig(image_size=32, base_channels=8, latent_channels=16, embedding_dim=4,
                        n_embed=32, attention=False, data_init=False, kmeans_iterations=0)
    config = teacher175_vq_config(codec, vjepa_width=32, vjepa_heads=4, vjepa_encoder_depth=1,
        vjepa_predictor_depth=1, vjepa_readout_mlp_ratio=1, vjepa_gradient_checkpointing=False,
        spatial_entities=True, spatial_combat=True, game_training_v5=True, flow_patch_size=2,
        flow_block_conditioning=True, flow_width=32, flow_depth=1, flow_heads=4,
        flow_steps=1, flow_adapter_width=32, flow_adapter_depth=1, flow_adapter_heads=4)
    value = NeuralWorldModel(config)
    source = VQGAN(codec)
    value.flow_renderer.load_codec({'codec_config': source.get_config(), 'model': source.state_dict()})
    return value


class SingleEnemyCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_real_source_initialization_transition_render_and_conditional_shapes(self):
        from turkeyworld.engine import Game
        from turkeyworld.camera import object_camera_features, project_sprite_bounds
        from turkeyworld.combat_objectives import combat_count_features
        from turkeyworld.spatial_entities import entity_spatial_features
        world = model().eval()
        source = Game(endless=True)
        state = torch.tensor(source.state_vector())[None]
        grid = torch.tensor(source.map)[None]
        image = torch.tensor(source.render(32, overlays=False)).permute(2, 0, 1)[None].float()/255
        features = world.encode_map(grid)
        memory = world.initialize_from_image(image, state, features)
        result = world.transition(state, torch.zeros(1, 4), features, memory)
        rgb = world.render(result['state'], result['features'], features)
        self.assertEqual((ENTITY_SLOTS, STATE_DIM, len(source.enemies)), (1, 20, 1))
        self.assertEqual(result['state'].shape, (1, 20))
        self.assertEqual(result['target_logits'].shape, (1, 2))
        self.assertEqual(result['attack_logits'].shape, (1, 1))
        self.assertEqual(result['spawn_logits'].shape, (1, 1, 256))
        self.assertEqual(result['spawn_anchor_logits'].shape, (1, 1, 9))
        self.assertEqual(entity_spatial_features(state).shape, (1, 2, 30))
        self.assertEqual(combat_count_features(state).shape, (1, 2, 6))
        self.assertEqual(object_camera_features(state)['features'].shape, (1, 1, 18))
        self.assertEqual(project_sprite_bounds(state)['boxes'].shape, (1, 1, 4))
        self.assertEqual(world.flow_renderer.spatial_condition.entity_tokens(state).shape, (1, 2, 32))
        self.assertEqual(world.core.visual_tokens, 64)
        self.assertEqual(world.flow_renderer.codec.config.latent_size, 8)
        self.assertEqual(world.flow_renderer.state_input[0].in_features, 20)
        self.assertEqual(rgb.shape, (1, 3, 32, 32))
        self.assertTrue(torch.isfinite(rgb).all())

    def test_typed_combat_respawn_and_head_gradients_only_have_one_slot(self):
        state = torch.zeros(1, 20); state[:, :2] = .2; state[:, 3:6] = 1
        state[:, 8:10] = .6; state[:, 12:14] = torch.tensor([.6, 1.])
        logits = torch.tensor([[2., -2.]], requires_grad=True)
        attacks = torch.tensor([[-2.]], requires_grad=True)
        result = structured_transition(state, state, torch.tensor([[0., 0., 0., 1.]]), logits, attacks,
                                       training=True, endless=True)
        torch.testing.assert_close(result['state'][:, 12], torch.tensor([.25]))
        self.assertEqual(result['accepted_target'].item(), 0)
        self.assertEqual(result['accepted_attacks'].shape, (1, 1))
        result['state'][:, 12].sum().backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0.)
        corpse = state.clone(); corpse[:, 12:14] = 0; corpse[:, 16] = 5
        spawned = structured_transition(corpse, corpse, torch.zeros(1, 4), logits.detach(), attacks.detach(),
            training=False, endless=True, spawn_positions=torch.tensor([[[.8, .8]]]),
            spawn_valid=torch.ones(1, 1, dtype=torch.bool))
        self.assertTrue(spawned['respawn_flags'].item())
        torch.testing.assert_close(spawned['state'][:, 12:14], torch.tensor([[.6, 1.]]))
        self.assertEqual(spawned['state'].shape[-1], 20)

    def test_wrong_schema_and_legacy_state_tensor_are_rejected_explicitly(self):
        for options in ({'entity_slots': 8}, {'entity_slots': True}, {'state_schema_version': 'legacy90'}):
            with self.assertRaisesRegex(ValueError, 'schema'):
                WorldModelConfig(**options)
        with self.assertRaisesRegex(ValueError, '20'):
            structured_transition(torch.zeros(1, 90), torch.zeros(1, 90), torch.zeros(1, 4),
                                  torch.zeros(1, 2), torch.zeros(1, 1))

    def test_gan_uses_twenty_state_values_and_rgb_channels_are_unchanged(self):
        from turkeyworld.adversarial import MultiScaleDiscriminator
        from turkeyworld.flow_renderer import ConditionalFlowRenderer
        critic = MultiScaleDiscriminator(base_channels=4, num_scales=1)
        rgb = torch.rand(2, 3, 32, 32, requires_grad=True)
        output = critic(rgb, torch.rand_like(rgb), torch.zeros(2, 20))
        sum(value.mean() for value in output['logits']).backward()
        self.assertTrue(torch.isfinite(rgb.grad).all())
        self.assertEqual(critic.discriminators[0].state_projection.in_features, 20)
        direct = ConditionalFlowRenderer(kind='student', image_size=128, width=32, depth=1, heads=4)
        self.assertEqual(direct.image_input.in_channels, 9)  # RGB noise/base/selfcondition, not9actors.
        self.assertEqual(direct.state_input[0].in_features, 20)

    @staticmethod
    def legacy_payload(world):
        tensors = {}
        for name, value in world.state_dict().items():
            shape = list(value.shape)
            if name.endswith(('state_encoder.weight', 'state_input.0.weight')):
                shape[1] = 90
            elif name.endswith('render_condition.0.weight'):
                shape[1] = 218
            elif name.endswith(('hit_target_head.2.weight', 'hit_target_head.2.bias')):
                shape[0] = 9
            elif name.endswith(('attack_head.2.weight', 'attack_head.2.bias')):
                shape[0] = 8
            elif name.endswith(('delta_head.2.weight', 'delta_head.2.bias')):
                shape[0] = 18
            elif name.endswith(('spawn_head.2.weight', 'spawn_head.2.bias')):
                shape[0] = 2048
            elif name.endswith(('slot_embedding', 'slot_embedding.weight', 'identity.weight')):
                shape[0] = 9 if value.shape[0] == 2 else 8
            tensors[name] = value.clone() if tuple(shape) == value.shape else torch.randn(shape, dtype=value.dtype)
        config = asdict(world.config); config.pop('entity_slots'); config.pop('state_schema_version')
        return {'stage': 'world', 'config': config, 'model': tensors, 'global_training_step': 818,
                'training_steps': 818, 'state_schema': {'slots': 8}, 'optimizer': {'old': True}}

    def test_explicit_migration_slices_interfaces_copies_codec_and_resets_training_honestly(self):
        source_world = model()
        old = self.legacy_payload(source_world)
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/'legacy.pt', Path(directory)/'single.pt'
            torch.save(old, source)
            with self.assertRaisesRegex(ValueError, 'migration'):
                load_world_checkpoint(source)
            report = migrate_single_enemy_checkpoint(source, output)
            current, payload = load_world_checkpoint(output)
            self.assertEqual(payload['training_steps'], 0)
            self.assertEqual(payload['source_optimizer_step'], 818)
            self.assertNotIn('optimizer', payload)
            self.assertTrue(report['optimizer_reset'])
            self.assertFalse(report['whole_function_preserved'])
            self.assertTrue(report['codec_tensors_bit_exact'])
            new = current.state_dict()
            key = 'vjepa_backend.core.predictor.state_encoder.weight'
            wanted = old['model'][key][:, legacy_state_indices()].clone(); wanted[:, -2] /= 8
            torch.testing.assert_close(new[key], wanted, rtol=0, atol=0)
            key = 'vjepa_backend.hit_target_head.2.weight'
            torch.testing.assert_close(new[key], old['model'][key][[0, 8]], rtol=0, atol=0)
            key = 'vjepa_backend.renderer.render_condition.0.weight'
            wanted = old['model'][key][:, legacy_state_indices()+list(range(90, 218))].clone()
            wanted[:, 18] /= 8
            torch.testing.assert_close(new[key], wanted, rtol=0, atol=0)
            for key in new:
                if key.startswith('flow_renderer.codec.'):
                    torch.testing.assert_close(new[key], old['model'][key], rtol=0, atol=0)
            with self.assertRaises(ValueError):
                migrate_single_enemy_checkpoint(source, output)
            with self.assertRaises(ValueError):
                migrate_single_enemy_checkpoint(output, Path(directory)/'twice.pt')


if __name__ == '__main__':
    unittest.main()
