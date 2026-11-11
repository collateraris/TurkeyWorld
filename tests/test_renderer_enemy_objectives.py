from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.base_student import build_base_student
from turkeyworld.engine import Game
from turkeyworld.models import NeuralWorldModel, WorldModelConfig
from turkeyworld.renderer_enemy_objectives import enemy_conditions, enemy_refinement_loss
from turkeyworld.renderer_refinement import configure_refinement, frozen_fingerprint, refinement_forward
from turkeyworld.renderer_student import build_renderer_student


def tiny_model():
    config = WorldModelConfig(kind='teacher', architecture='dreamer_rssm', image_size=32,
        endless=True, navigation_only=True, spatial_entities=True, spatial_combat=True,
        player_motion_mode='axis_fraction', renderer_mode='flow_matching', renderer_codec='vqgan',
        renderer_codec_config=dict(image_size=32, base_channels=4, latent_channels=8,
            embedding_dim=4, n_embed=8, data_init=False), flow_steps=1,
        flow_width=16, flow_depth=1, flow_heads=2, vjepa_width=16,
        vjepa_readout_mlp_ratio=2, rssm_deter_dim=16, rssm_hidden_dim=16,
        rssm_stoch=2, rssm_classes=4, rssm_embed_dim=16, rssm_action_dim=20, rssm_blocks=2)
    model = NeuralWorldModel(config).eval()
    model.flow_renderer.codec_loaded.fill_(True)
    with torch.no_grad():
        model.vjepa_backend.feature_head.weight.fill_(.005)
        model.vjepa_backend.feature_head.bias.fill_(.02)
    model.flow_renderer = build_renderer_student(model, steps=1, width=16, depth=1, heads=2, mlp_hidden=32)
    model.base_student = build_base_student(model)
    configure_refinement(model)
    return model


def batch(dynamic=(False, True, False)):
    rows = []
    for index, flag in enumerate(dynamic):
        game = Game(seed=index, endless=True)
        previous = game.state_vector().copy()
        previous_rgb = game.render(size=32, overlays=False)
        action = np.array([.5, .2, .3, 0], np.float32)
        game.step(action, dt=1/30)
        target = game.render(size=32, overlays=False)
        enemies = game.enemies
        game.enemies = []
        background = game.render(size=32, overlays=False)
        game.enemies = enemies
        rows.append({'state': game.state_vector(), 'map': game.map.copy(), 'target': target,
            'object_mask': np.any(target != background, axis=-1), 'background': background,
            'previous_state': previous, 'previous_rgb': previous_rgb, 'action': action})
    values = {}
    for key in rows[0]:
        value = torch.from_numpy(np.stack([row[key] for row in rows]))
        if key in ('target', 'background', 'previous_rgb'):
            value = value.permute(0, 3, 1, 2).float()/255
        elif key == 'object_mask':
            value = value[:, None].float()
        else:
            value = value.long() if key == 'map' else value.float()
        values[key] = value
    values['dynamic'] = torch.tensor(dynamic, dtype=torch.bool)
    return values


class RendererEnemyObjectivesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_static_snapshots_never_encode_or_transition_and_all_features_are_zero(self):
        model = tiny_model()
        source = batch((False, False))
        # Invalid nonexistent previous observations are allowed for static rows;
        # they are never read, repaired, or passed into an encoder.
        for key in ('previous_state', 'previous_rgb', 'action'):
            del source[key]
        with patch.object(model, 'initialize_from_image', side_effect=AssertionError('No static image encoder')), \
                patch.object(model, 'transition', side_effect=AssertionError('No static dynamics')), \
                patch('turkeyworld.engine.Game.step', side_effect=AssertionError('Recorded data only')), \
                patch('turkeyworld.engine.Game.render', side_effect=AssertionError('Recorded data only')):
            conditions = enemy_conditions(model, source)
        self.assertTrue(torch.equal(conditions['features'], torch.zeros(2, 128)))
        self.assertIsNone(conditions['dynamic_prior_memory'])
        self.assertTrue(torch.equal(conditions['state'], source['state']))
        self.assertTrue(all(not value.requires_grad for value in conditions['geometry'].values()))

    def test_dynamic_subset_uses_previous_rgb_only_then_restores_full_batch_map_cache(self):
        model = tiny_model()
        source = batch()
        with patch.object(model, 'initialize_from_image', wraps=model.initialize_from_image) as initialize, \
                patch.object(model, 'transition', wraps=model.transition) as transition, \
                patch.object(model, 'encode_map', wraps=model.encode_map) as maps:
            first = enemy_conditions(model, source)
        self.assertEqual(initialize.call_count, 1)
        self.assertEqual(transition.call_count, 1)
        self.assertTrue(torch.equal(initialize.call_args.args[0], source['previous_rgb'][1:2]))
        self.assertTrue(torch.equal(initialize.call_args.args[1], source['previous_state'][1:2]))
        self.assertTrue(torch.equal(transition.call_args.args[1], source['action'][1:2]))
        self.assertEqual([len(call.args[0]) for call in maps.call_args_list], [1, 3])
        self.assertTrue(torch.equal(first['features'][[0, 2]], torch.zeros(2, 128)))
        self.assertGreater(float(first['features'][1].abs().sum()), 0)
        altered = {key: value.clone() for key, value in source.items()}
        altered['target'] = 1-altered['target']
        altered['background'] = 1-altered['background']
        altered['object_mask'] = 1-altered['object_mask']
        altered['state'][:, 0] += .001
        second = enemy_conditions(model, altered)
        self.assertTrue(torch.equal(first['features'], second['features']))
        self.assertTrue(torch.equal(first['dynamic_prior_memory'], second['dynamic_prior_memory']))
        self.assertFalse(torch.equal(first['state'], second['state']))

    def test_real_backward_updates_base_and_rf_only_with_codec_parameters_and_buffers_bitexact(self):
        model = tiny_model()
        source = batch()
        for key, value in source.items():
            if value.is_floating_point():
                source[key] = value.requires_grad_(True)
        frozen_before = frozen_fingerprint(model)
        codec_before = {key: value.clone() for key, value in model.flow_renderer.codec.state_dict().items()}
        base_before = model.base_student.rgb.weight.clone()
        flow_before = model.flow_renderer.output.weight.clone()
        optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad], lr=1e-4)
        optimizer_ids = {id(value) for group in optimizer.param_groups for value in group['params']}
        self.assertFalse(optimizer_ids & {id(value) for value in model.flow_renderer.codec.parameters()})
        with patch('turkeyworld.engine.Game.step', side_effect=AssertionError('No Source replay')), \
                patch('turkeyworld.engine.Game.render', side_effect=AssertionError('No Source rendering')), \
                patch.object(model.flow_renderer, 'velocity', wraps=model.flow_renderer.velocity) as velocity:
            loss, metrics = enemy_refinement_loss(model, source)
            loss.backward()
            self.assertEqual(velocity.call_count, 1)
        self.assertGreater(float(model.base_student.rgb.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.flow_renderer.output.weight.grad.abs().sum()), 0)
        self.assertTrue(all(value.grad is None for name, value in model.named_parameters() if not value.requires_grad))
        self.assertTrue(all(value.grad is None for value in source.values() if value.is_floating_point()))
        optimizer.step()
        self.assertFalse(torch.equal(base_before, model.base_student.rgb.weight))
        self.assertFalse(torch.equal(flow_before, model.flow_renderer.output.weight))
        self.assertEqual(frozen_before, frozen_fingerprint(model))
        self.assertTrue(all(torch.equal(value, model.flow_renderer.codec.state_dict()[key]) for key, value in codec_before.items()))
        self.assertFalse(model.flow_renderer.codec.training)
        self.assertFalse(model.flow_renderer.codec.quantize.training)
        self.assertEqual((metrics['static_examples'], metrics['dynamic_examples']), (2, 1))
        self.assertFalse(metrics['future_source_rgb_in_prior'])
        self.assertTrue(metrics['flow_base_detached'])

    def test_unfrozen_codec_and_invalid_weights_fail_before_training(self):
        model = tiny_model()
        source = batch((False,))
        model.flow_renderer.codec.train()
        with self.assertRaisesRegex(ValueError, 'codec/quantizer frozen'):
            enemy_refinement_loss(model, source)
        model.flow_renderer.codec.eval()
        for weight in (float('nan'), -1, True):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                enemy_refinement_loss(model, source, contrast_weight=weight)


if __name__ == '__main__':
    unittest.main()
