"""Causal visual dynamics, persistent conditioning and honest renderer migration."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Optional PyTorch dependency is not installed')
class CompactVJEPA2ACTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def core(self):
        from turkeyworld.vjepa2_ac import CompactVJEPA2AC
        return CompactVJEPA2AC(width=32, encoder_depth=1, predictor_depth=2, heads=4, context_frames=2)

    def model(self, **options):
        from turkeyworld.models import NeuralWorldModel, WorldModelConfig
        config = dict(kind='student', architecture='vjepa2_ac', image_size=128,
            endless=True, renderer_mode='rectified_flow', flow_steps=1,
            flow_width=32, flow_depth=1, flow_heads=4, vjepa_width=32,
            vjepa_encoder_depth=1, vjepa_predictor_depth=2, vjepa_heads=4, vjepa_context_frames=2)
        config.update(options)
        return NeuralWorldModel(WorldModelConfig(**config))

    def initial(self, model):
        from turkeyworld.engine import Game
        game = Game(endless=True)
        state = torch.tensor(game.state_vector())[None]
        rgb = torch.tensor(game.render(128, overlays=False)).permute(2, 0, 1)[None].float()/255
        features = model.encode_map(torch.tensor(game.map, dtype=torch.long)[None])
        return game, state, rgb, features

    def test_video_encoder_is_frame_causal_without_future_image_leakage(self):
        core = self.core().eval()
        video = torch.rand(1, 3, 3, 128, 128)
        changed = video.clone()
        changed[:, 2] = torch.rand_like(changed[:, 2])*4
        with torch.inference_mode():
            first, second = core.encode_video(video), core.encode_video(changed)
        self.assertEqual(first.shape, (1, 3, 64, 32))
        torch.testing.assert_close(first[:, :2], second[:, :2], rtol=0, atol=0)
        self.assertGreater(float((first[:, 2]-second[:, 2]).abs().sum()), 0.)

    def test_interleaved_predictor_is_block_causal_and_actions_change_next_latents(self):
        from turkeyworld.vjepa2_ac import block_causal_mask
        core = self.core().eval()
        latents, actions, states = torch.randn(1, 3, 64, 32), torch.randn(1, 3, 4), torch.randn(1, 3, STATE_DIM)
        with torch.inference_mode():
            first = core.predict(latents, actions, states)
            changed_latents, changed_actions, changed_states = latents.clone(), actions.clone(), states.clone()
            changed_latents[:, 2] += torch.randn_like(changed_latents[:, 2])*10
            changed_actions[:, 2] += 10
            changed_states[:, 2] += 10
            second = core.predict(changed_latents, changed_actions, changed_states)
            altered_action = actions.clone()
            altered_action[:, 1, 0] += 1.
            action_result = core.predict(latents, altered_action, states)
        torch.testing.assert_close(first[:, :2], second[:, :2], rtol=0, atol=0)
        self.assertGreater(float((first[:, 2]-second[:, 2]).abs().sum()), 0.)
        torch.testing.assert_close(first[:, 0], action_result[:, 0], rtol=0, atol=0)
        self.assertGreater(float((first[:, 1]-action_result[:, 1]).abs().sum()), 0.)
        allowed = block_causal_mask(3, 66)
        self.assertTrue(allowed[:66, :66].all())
        self.assertFalse(allowed[:66, 66:].any())
        self.assertTrue(allowed[66:132, :132].all())

    def test_three_rotary_axes_preserve_norm_and_condition_tokens_only_rotate_in_time(self):
        from turkeyworld.vjepa2_ac import RotaryAttention3D
        attention = RotaryAttention3D(32, 4)
        value = torch.randn(1, 4, 5, 8)
        positions = torch.stack([torch.arange(5), torch.zeros(5), torch.zeros(5)], -1)
        rotated = attention.rotate(value, positions)
        torch.testing.assert_close(value.norm(dim=-1), rotated.norm(dim=-1), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(rotated[..., attention.axis_width:], value[..., attention.axis_width:], rtol=0, atol=0)
        self.assertGreater(float((value[..., :attention.axis_width]-rotated[..., :attention.axis_width]).abs().sum()), 0.)

    def test_runtime_carries_its_prediction_and_executed_action_without_engine_or_new_rgb(self):
        from turkeyworld.engine import Game
        model = self.model()
        _, state, rgb, features = self.initial(model)
        memory = model.initialize_from_image(rgb[:, None].expand(-1, 2, -1, -1, -1), state, features)
        action = torch.tensor([[.5, -.2, .1, 1.]])
        with patch.object(Game, 'step', side_effect=AssertionError('No original-engine transitions')), \
             patch.object(Game, 'render', side_effect=AssertionError('No original-engine renders')), \
             patch.object(model.core.encoder, 'forward', side_effect=AssertionError('No future RGB encoding')):
            result = model.step(state, action, memory=memory, map_features=features)
        self.assertEqual(result['memory'].shape, (1, 132, 32))
        self.assertEqual(result['latent_prediction'].shape, (1, 64, 32))
        blocks = result['memory'].reshape(1, 2, 66, 32)
        torch.testing.assert_close(blocks[:, -1, 2:], result['latent_prediction'])
        torch.testing.assert_close(blocks[:, -2, 0], model.core.predictor.action_encoder(action))
        self.assertFalse(any(isinstance(module, torch.nn.GRUCell) for module in model.modules()))
        self.assertFalse(any('scene_blocks' in name or 'recurrent' in name for name, _ in model.named_parameters()))

    def test_visual_memory_and_persistent_map_both_affect_predictions_and_receive_gradients(self):
        model = self.model()
        _, state, rgb, features = self.initial(model)
        memory = model.initialize_from_image(rgb, state, features)
        action = torch.tensor([[1., 0., 0., 0.]])
        result = model.transition(state, action, features, memory)
        changed_memory = memory+torch.randn_like(memory)*.2
        changed = model.transition(state, action, features, changed_memory)
        changed_map = model.transition(state, action, features.roll(2, -1), memory)
        self.assertGreater(float((result['latent_prediction']-changed['latent_prediction']).detach().abs().sum()), 0.)
        self.assertGreater(float((result['latent_prediction']-changed_map['latent_prediction']).detach().abs().sum()), 0.)
        loss = result['latent_prediction'].square().mean()
        loss.backward()
        gradient = model.vjepa_backend.map_projection[0].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.)
        self.assertGreater(sum(float(parameter.grad.abs().sum()) for parameter in model.core.encoder.parameters()
                               if parameter.grad is not None), 0.)
        self.assertGreater(sum(float(parameter.grad.abs().sum()) for parameter in model.core.predictor.parameters()
                               if parameter.grad is not None), 0.)

    def test_all_entity_slots_persist_outside_the_view_and_new_features_condition_rgb(self):
        model = self.model(renderer_mode='direct')
        _, state, rgb, features = self.initial(model)
        state[:, 8:10] = torch.tensor([1.5/16, 12.5/16])
        state[:, 12:14] = 0.  # Persistent dead NPC0, away from the initial view.
        memory = model.initialize_from_image(rgb, state, features)
        result = model.transition(state, torch.zeros(1, 4), features, memory)
        torch.testing.assert_close(result['state'][:, 8:10], state[:, 8:10], rtol=0, atol=0)
        self.assertEqual(result['state'].shape, (1, STATE_DIM))
        self.assertEqual(float(result['state'][0, 13].detach()), 0.)
        with torch.inference_mode():
            zero = model.render_base(state, torch.zeros(1, 128), features)
            one = model.render_base(state, torch.ones(1, 128), features)
        self.assertGreater(float((zero-one).abs().sum()), 0.)

    def test_numeric_state_fallback_uses_a_neural_initial_image(self):
        from turkeyworld.engine import Game
        model = self.model()
        _, state, _, features = self.initial(model)
        with patch.object(Game, 'render', side_effect=AssertionError('No engine fallback')):
            result = model.transition(state, torch.zeros(1, 4), features, model.initial_memory(1))
        self.assertTrue(torch.isfinite(result['memory']).all())
        self.assertGreater(float(result['memory'].detach().abs().sum()), 0.)

    def test_preencoded_initialization_reuses_encoder_output_with_identical_map_context(self):
        model = self.model().eval()
        _, state, rgb, features = self.initial(model)
        video = rgb[:, None].expand(-1, 2, -1, -1, -1)
        with torch.inference_mode():
            expected = model.initialize_from_image(video, state, features)
            encoded = model.core.encode_video(video)
            with patch.object(model.core.encoder, 'forward', side_effect=AssertionError('Encoder must not run twice')):
                actual = model.initialize_from_latents(encoded, state, features)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_full_teacher_profile_is_125m_without_ema_or_discriminator(self):
        from turkeyworld.models import NeuralWorldModel, WorldModelConfig, parameter_counts
        with torch.device('meta'):
            model = NeuralWorldModel(WorldModelConfig(kind='teacher', image_size=128, architecture='vjepa2_ac',
                endless=True, renderer_mode='flow_matching', flow_steps=12,
                flow_adapter_width=512, flow_adapter_depth=12, flow_adapter_heads=8))
        self.assertEqual(parameter_counts(model)['world_total'], 124_406_761)
        self.assertEqual(model.core.memory_tokens, 264)
        self.assertEqual(model.core.width, 512)
        self.assertFalse(any('ema' in name or 'discriminator' in name for name, _ in model.named_parameters()))

    def test_renderer_migration_is_exact_and_does_not_relabel_old_dynamics(self):
        from turkeyworld.models import NeuralWorldModel, WorldModelConfig, load_world_checkpoint
        from turkeyworld.vjepa2_ac import migrate_vjepa_checkpoint
        original = NeuralWorldModel(WorldModelConfig(kind='teacher', image_size=128, endless=True,
            renderer_mode='flow_matching', flow_steps=1, flow_width=32, flow_depth=1, flow_heads=4)).eval()
        with torch.no_grad():
            original.flow_renderer.output.weight.normal_(std=.01)
        _, state, _, source_features = self.initial(original)
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory)/'teacher75.pt', Path(directory)/'vjepa125.pt'
            torch.save({'stage': 'world', 'model': original.state_dict(), 'config': asdict(original.config),
                        'training_steps': 1110, 'optimizer': {}, 'teacher_quality_gate': {'passed': False}}, source)
            report = migrate_vjepa_checkpoint(source, target, require_profile=False,
                vjepa_width=32, vjepa_encoder_depth=1, vjepa_predictor_depth=1, vjepa_heads=4, vjepa_context_frames=2)
            model, payload = load_world_checkpoint(target)
            from turkeyworld.engine import Game
            fmap = model.encode_map(torch.tensor(Game().map, dtype=torch.long)[None])
            with torch.inference_mode():
                source_rgb = original.render(state, torch.zeros(1, 128), source_features)
                migrated_rgb = model.render(state, torch.zeros(1, 128), fmap)
            torch.testing.assert_close(source_rgb, migrated_rgb, rtol=0, atol=0)
            self.assertTrue(report['all_transferred_tensors_exact'])
            self.assertTrue(report['latent_encoder_reinitialized'])
            self.assertTrue(report['action_predictor_reinitialized'])
            self.assertFalse(report['official_pretrained_weights_loaded'])
            self.assertNotIn('optimizer', payload)
            self.assertNotIn('teacher_quality_gate', payload)
            self.assertEqual(payload['training_steps'], 1110)
            self.assertFalse(any('recurrent' in name for name in model.state_dict()))


if __name__ == '__main__':
    unittest.main()
