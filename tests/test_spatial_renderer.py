"""Persistent entity tokens have an independent, camera-grounded visual read."""
from contextlib import ExitStack
import math
import unittest
from unittest.mock import patch

import torch

from turkeyworld.camera import object_camera_features
from turkeyworld.models import WorldModelConfig
from turkeyworld.transformer_model import CameraTransformerWorldModel


def state90():
    state = torch.zeros(1, 90)
    state[:, :8] = torch.tensor([.5, .5, 0., 1., 1., .8, .1, 0.])
    entities = state[:, 8:88].reshape(1, 8, 10)
    entities[:, :, :2] = torch.tensor([7/16, 8/16])  # Other slots behind camera.
    entities[:, :, 4:6] = torch.tensor([.8, 1.])
    entities[:, 0, :4] = torch.tensor([9/16, 8/16, .2, 0.])
    return state


class SpatialRendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def model(enabled=True, kind='student'):
        config = WorldModelConfig(kind=kind, image_size=32, architecture='vjepa2_ac',
                                  spatial_entities=enabled)
        return CameraTransformerWorldModel(config, render_only=True).eval()

    @staticmethod
    def activate(model):
        with torch.no_grad():
            model.spatial_render.attention.out_proj.weight.copy_(torch.eye(model.width))

    def test_disabled_flag_preserves_legacy_keys_counts_and_initialization(self):
        for kind, parameters, renderer, keys in [('teacher', 3857646, 2202252, 169),
                                                  ('student', 942062, 529164, 105)]:
            with self.subTest(kind=kind):
                torch.manual_seed(73)
                legacy = self.model(False, kind)
                torch.manual_seed(73)
                spatial = self.model(True, kind)
                self.assertEqual(sum(p.numel() for p in legacy.parameters()), parameters)
                self.assertEqual(legacy.renderer_parameter_count(), renderer)
                self.assertEqual(len(legacy.state_dict()), keys)
                self.assertFalse(hasattr(legacy, 'spatial_encoder'))
                for name, value in legacy.state_dict().items():
                    self.assertFalse(name.startswith('spatial_'))
                    torch.testing.assert_close(spatial.state_dict()[name], value, rtol=0, atol=0)
                self.assertGreater(spatial.renderer_parameter_count(), renderer)

    def test_zero_residual_preserves_the_same_loaded_renderer_rgb(self):
        torch.manual_seed(11)
        legacy, spatial = self.model(False), self.model(True)
        spatial.load_state_dict(legacy.state_dict(), strict=False)
        state, features = state90(), torch.zeros(1, 128)
        maps = torch.randn(1, legacy.width, 16, 16)
        with torch.no_grad():
            before = legacy.render_with_aux(state, features, maps)
            after = spatial.render_with_aux(state, features, maps)
        torch.testing.assert_close(after['rgb'], before['rgb'], rtol=0, atol=0)
        self.assertEqual(after['spatial_tokens'].shape, (1, 9, spatial.width))
        self.assertEqual(after['spatial_render_weights'].shape, (1, 256, 9))
        self.assertNotIn('spatial_tokens', before)

    def test_spatial_positions_and_directions_change_actual_rgb_with_legacy_path_held_fixed(self):
        torch.manual_seed(17)
        model, original = self.model(), state90()
        maps, features = torch.randn(1, model.width, 16, 16), torch.zeros(1, 128)
        with torch.no_grad():
            fixed_geometry = model.geometry(original, maps)
            for parameter in model.render_condition.parameters():
                parameter.zero_()
        variants = []
        moved = original.clone(); moved[:, 9] += .25/16
        turned_npc = original.clone(); turned_npc[:, 10:12] = torch.tensor([0., .2])
        turned_player = original.clone(); turned_player[:, 2:4] = torch.tensor([math.sin(.2), math.cos(.2)])
        variants.extend([moved, turned_npc, turned_player])
        with patch.object(model, 'geometry', return_value=fixed_geometry), torch.no_grad():
            # Existing state/geometry conditioning is constant in this test:
            # without the new residual all three changes must remain invisible.
            baseline = model.render_with_aux(original, features, maps)['rgb']
            for state in variants:
                torch.testing.assert_close(model.render_with_aux(state, features, maps)['rgb'],
                                           baseline, rtol=0, atol=0)
            self.activate(model)
            baseline = model.render_with_aux(original, features, maps)['rgb']
            for index, state in enumerate(variants):
                changed = model.render_with_aux(state, features, maps)['rgb']
                self.assertGreater(float((changed-baseline).abs().max()), 1e-6, index)

    def test_visual_projection_retains_partial_edge_corpses_but_masks_behind_keys(self):
        model, state = self.model(), state90()
        state[:, 9] = 8.8/16  # NPC center beyond right FOV; sprite still overlaps.
        state[:, 12:14] = 0.  # Corpse still has a sprite.
        original = state.clone()
        with torch.no_grad():
            result = model.render_with_aux(state, torch.zeros(1, 128), torch.randn(1, model.width, 16, 16))
        self.assertFalse(bool(object_camera_features(state)['frustum'][0, 0]))
        weights = result['spatial_render_weights']
        self.assertTrue(bool((weights[:, :, 1] > 0).any()))
        self.assertTrue(bool((weights[:, :, 1] == 0).any()))
        self.assertTrue(bool((weights[:, :, 2:] == 0).all()))
        torch.testing.assert_close(weights[:, :, 0], torch.ones(1, 256), rtol=0, atol=0)
        self.assertEqual(result['spatial_tokens'].shape[1], 9)
        self.assertTrue(bool((result['spatial_tokens'].abs().sum(-1) > 0).all()))
        torch.testing.assert_close(state, original, rtol=0, atol=0)

    def test_masked_behind_token_cannot_contribute_to_the_spatial_visual_read(self):
        model, state = self.model(), state90()
        self.activate(model)
        maps = torch.randn(1, model.width, 16, 16)
        with torch.no_grad():
            geometry = model.geometry(state, maps)
            weights = model.spatial_render_weights(state, geometry)
            tokens = model.spatial_encoder(state, maps)
            queries = torch.randn(1, 256, model.width)
            before = model.spatial_render(queries, tokens, weights)
            changed = tokens.clone()
            changed[:, 2] = torch.randn_like(changed[:, 2])*20
            after = model.spatial_render(queries, changed, weights)
        self.assertEqual(tokens.shape[1], 9)
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_training_from_zero_opens_projection_then_reaches_spatial_encoder_and_rgb_head(self):
        torch.manual_seed(29)
        model = self.model().train()
        state = state90().requires_grad_()
        maps = torch.randn(1, model.width, 16, 16, requires_grad=True)
        features, target = torch.zeros(1, 128), torch.rand(1, 3, 32, 32)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        for iteration in range(2):
            optimizer.zero_grad(set_to_none=True)
            image = model.render_with_aux(state, features, maps)['rgb']
            (image-target).square().mean().backward()
            self.assertGreater(float(model.spatial_render.attention.out_proj.weight.grad.abs().sum()), 0.)
            self.assertGreater(float(model.rgb32.weight.grad.abs().sum()), 0.)
            if iteration == 1:
                self.assertGreater(float(model.spatial_encoder.descriptor_projection[0].weight.grad.abs().sum()), 0.)
                self.assertGreater(float(model.spatial_encoder.map_projection[0].weight.grad.abs().sum()), 0.)
                self.assertGreater(float(model.spatial_render.attention.in_proj_weight.grad.abs().sum()), 0.)
                self.assertTrue(bool(torch.isfinite(state.grad).all()))
                self.assertTrue(bool(torch.isfinite(maps.grad).all()))
            optimizer.step()

    def test_amp_keeps_original_fp32_spatial_coordinates_before_local_map_sampling(self):
        model, state = self.model(), state90()
        state[:, 0] += .0001
        observed = []
        handle = model.spatial_encoder.register_forward_pre_hook(
            lambda module, inputs: observed.append(inputs[0].detach().clone()))
        try:
            with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
                result = model.render_with_aux(state, torch.zeros(1, 128),
                                               torch.randn(1, model.width, 16, 16).bfloat16())
        finally:
            handle.remove()
        self.assertEqual(observed[0].dtype, torch.float32)
        torch.testing.assert_close(observed[0][:, :4], state[:, :4], rtol=0, atol=0)
        self.assertTrue(bool(torch.isfinite(result['rgb']).all()))

    def test_spatial_renderer_adds_no_engine_or_nonvisual_hud_conditioning(self):
        model, state = self.model(), state90()
        self.activate(model)
        maps, features = torch.randn(1, model.width, 16, 16), torch.zeros(1, 128)
        changed = state.clone()
        changed[:, 4:7] = torch.tensor([.2, .1, .9])
        changed[:, -2:] = torch.tensor([5., 2.])
        with ExitStack() as stack:
            for method in ('turkeyworld.engine.Game.__init__', 'turkeyworld.engine.Game.step',
                           'turkeyworld.engine.Game.render', 'turkeyworld.geometry_targets.build_geometry_targets'):
                stack.enter_context(patch(method, side_effect=AssertionError('Source oracle forbidden')))
            with torch.no_grad():
                before = model.render_with_aux(state, features, maps)
                after = model.render_with_aux(changed, features, maps)
        torch.testing.assert_close(before['spatial_tokens'], after['spatial_tokens'], rtol=0, atol=0)
        torch.testing.assert_close(before['rgb'], after['rgb'], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
