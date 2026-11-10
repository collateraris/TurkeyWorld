"""Player/single-NPC geometry, rotated map reads, identity persistence and gradients."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
import math
import unittest

import torch

from turkeyworld.spatial_entities import (SpatialEntityEncoder, entity_spatial_features,
    entity_motion_basis, world_to_local, local_to_world, FEATURE_SLICES,
    SPATIAL_FEATURE_DIM, ENTITY_SLOT_NAMES)


class SpatialEntityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def state():
        state = torch.zeros(1, STATE_DIM)
        state[:, :2], state[:, 3:6] = .5, 1.
        entities = state[:, GLOBAL_DIM:-2].reshape(1, ENTITY_SLOTS, ENTITY_DIM)
        entities[:, :, 0] = torch.linspace(.2, .8, ENTITY_SLOTS)
        entities[:, :, 1] = torch.linspace(.3, .7, ENTITY_SLOTS)
        entities[:, :, 2:4] = torch.tensor([.2, .1])
        entities[:, :, 4:6] = torch.tensor([.6, 1.])
        entities[:, :, 9] = torch.arange(ENTITY_SLOTS)/ENTITY_SLOTS
        return state

    def test_exact_layout_two_persistent_slots_and_numeric_attributes(self):
        state = self.state()
        features = entity_spatial_features(state)
        self.assertEqual(features.shape, (1, ENTITY_SLOTS+1, 30))
        self.assertEqual(SPATIAL_FEATURE_DIM, 30)
        self.assertEqual(ENTITY_SLOT_NAMES, ('player',)+tuple(f'npc_{i}' for i in range(ENTITY_SLOTS)))
        self.assertEqual(FEATURE_SLICES['player_heading_valid'].stop, 30)
        torch.testing.assert_close(features[:, 0, FEATURE_SLICES['absolute_xy']], state[:, :2])
        torch.testing.assert_close(features[:, 1:, FEATURE_SLICES['absolute_xy']], state[:, GLOBAL_DIM:-2].reshape(1, ENTITY_SLOTS, ENTITY_DIM)[:, :, :2])
        self.assertEqual(features[0, :, FEATURE_SLICES['is_player']].flatten().tolist(), [1.]+[0.]*ENTITY_SLOTS)
        self.assertEqual(features[0, :, FEATURE_SLICES['velocity_valid']].flatten().tolist(), [0.]+[1.]*ENTITY_SLOTS)
        torch.testing.assert_close(features[0, 1:, FEATURE_SLICES['speed']], torch.full((ENTITY_SLOTS, 1), math.sqrt(.2**2+.1**2)*2))

    def test_translation_and_global_rotation_preserve_player_relative_geometry(self):
        state = self.state()
        translated = state.clone()
        translated[:, :2] += torch.tensor([.05, -.04])
        translated[:, GLOBAL_DIM:-2].reshape(1, ENTITY_SLOTS, ENTITY_DIM)[:, :, :2] += torch.tensor([.05, -.04])
        rotated = state.clone()
        entities = rotated[:, GLOBAL_DIM:-2].reshape(1, ENTITY_SLOTS, ENTITY_DIM)
        old = state[:, GLOBAL_DIM:-2].reshape(1, ENTITY_SLOTS, ENTITY_DIM)
        delta = old[:, :, :2]-state[:, None, :2]
        entities[:, :, :2] = state[:, None, :2]+torch.stack([-delta[:, :, 1], delta[:, :, 0]], -1)
        entities[:, :, 2:4] = torch.stack([-old[:, :, 3], old[:, :, 2]], -1)
        rotated[:, 2:4] = torch.tensor([1., 0.])
        first = entity_spatial_features(state)
        for changed in (translated, rotated):
            result = entity_spatial_features(changed)
            for name in ('relative_forward_right', 'relative_distance', 'relative_bearing_sin_cos', 'relative_direction_sin_cos'):
                torch.testing.assert_close(first[..., FEATURE_SLICES[name]], result[..., FEATURE_SLICES[name]], atol=2e-6, rtol=1e-5)

    def test_stationary_direction_is_invalid_and_dead_behind_entities_are_retained(self):
        state = self.state()
        npc = state[:, GLOBAL_DIM:-2].reshape(1, ENTITY_SLOTS, ENTITY_DIM)
        npc[:, 0, 2:4], npc[:, 0, 4:6] = 0., 0.
        npc[:, 0, 8:10] = torch.tensor([2., .7])
        features = entity_spatial_features(state)
        self.assertEqual(features[0, 1, FEATURE_SLICES['direction_valid']].item(), 0.)
        self.assertEqual(features[0, 1, FEATURE_SLICES['direction_sin_cos']].abs().sum().item(), 0.)
        self.assertEqual(features[0, 1, FEATURE_SLICES['alive']].item(), 0.)
        self.assertLess(features[0, 1, FEATURE_SLICES['relative_forward_right']][0].item(), 0.)
        self.assertEqual(features[0, 1, FEATURE_SLICES['death_timer']].item(), 2.)
        torch.testing.assert_close(features[0, 1, FEATURE_SLICES['absolute_xy']], npc[0, 0, :2])
        encoder = SpatialEntityEncoder(32, 2)
        self.assertEqual(encoder(state, torch.zeros(1, 2, 16, 16)).shape, (1, ENTITY_SLOTS+1, 32))
        zero = torch.zeros(2, STATE_DIM)
        self.assertTrue(bool(torch.isfinite(entity_spatial_features(zero)).all()))
        self.assertTrue(bool(torch.isfinite(encoder(zero, torch.zeros(2, 2, 16, 16))).all()))

    def test_single_npc_content_changes_without_changing_player_or_stable_identity(self):
        state, moved = self.state(), self.state()
        moved[:, GLOBAL_DIM:GLOBAL_DIM+2] += .02
        first, second = entity_spatial_features(state), entity_spatial_features(moved)
        torch.testing.assert_close(first[:, 0], second[:, 0])
        self.assertFalse(torch.equal(first[:, 1], second[:, 1]))
        model = SpatialEntityEncoder(16, 2)
        with torch.no_grad():
            for projection in (model.descriptor_projection, model.map_projection):
                for parameter in projection.parameters():
                    parameter.zero_()
        fmap = torch.zeros(1, 2, 16, 16)
        torch.testing.assert_close(model(state, fmap), model(moved, fmap))
        self.assertEqual(model.slot_embedding.num_embeddings, ENTITY_SLOTS+1)
        self.assertFalse(torch.equal(model(state, fmap)[:, 0], model(state, fmap)[:, 1]))

    def test_map_tile_centers_rotated_patch_axes_and_stationary_target_fallback(self):
        state = self.state()
        state[:, GLOBAL_DIM:-2].reshape(1, ENTITY_SLOTS, ENTITY_DIM)[:, :, 2:4] = 0.
        yy, xx = torch.meshgrid((torch.arange(16)+.5)/16, (torch.arange(16)+.5)/16, indexing='ij')
        fmap = torch.stack([xx, yy])[None]
        model = SpatialEntityEncoder(16, 2)
        sampled = model.sample_local_map(state, fmap)
        xy = entity_spatial_features(state)[..., FEATURE_SLICES['absolute_xy']]
        torch.testing.assert_close(sampled[:, :, 4], xy)
        torch.testing.assert_close(sampled[0, 0, 7]-sampled[0, 0, 4], torch.tensor([1/16, 0.]))
        rotated = state.clone(); rotated[:, 2:4] = torch.tensor([1., 0.])
        patch = model.sample_local_map(rotated, fmap)
        torch.testing.assert_close(patch[0, 0, 7]-patch[0, 0, 4], torch.tensor([0., 1/16]))
        basis = entity_motion_basis(state)
        target = state[0, :2]-state[0, 8:10]
        torch.testing.assert_close(basis['forward'][0, 1], target/target.norm())

    def test_local_motion_basis_post_turn_direction_and_inverse_projection(self):
        state = self.state()
        action = torch.tensor([[1., 0., 1., 0.]])
        basis = entity_motion_basis(state, action, dt=1/30)
        torch.testing.assert_close(basis['forward'][0, 0], torch.tensor([math.cos(2.2/30), math.sin(2.2/30)]))
        torch.testing.assert_close(basis['forward'][0, 1], torch.tensor([.2, .1])/math.sqrt(.05))
        vectors = torch.randn(1, ENTITY_SLOTS+1, 2)
        torch.testing.assert_close(local_to_world(world_to_local(vectors, basis), basis), vectors)
        coincident = torch.zeros(1, STATE_DIM)
        axes = entity_motion_basis(coincident)
        torch.testing.assert_close(axes['forward'], torch.tensor([1., 0.]).expand(1, ENTITY_SLOTS+1, 2))
        self.assertTrue(bool(torch.isfinite(axes['right']).all()))

    def test_gradients_reach_geometry_learned_map_and_all_encoder_parameters(self):
        torch.manual_seed(19)
        state = self.state().requires_grad_()
        yy, xx = torch.meshgrid(torch.arange(16), torch.arange(16), indexing='ij')
        features = torch.stack([xx.float()/16, yy.float()/16, (xx >= 8).float(), (yy >= 7).float()])[None].requires_grad_()
        model = SpatialEntityEncoder(32, 4)
        tokens = model(state, features)
        weights = torch.linspace(-1., 1., tokens.numel()).reshape_as(tokens)
        (tokens*weights).sum().backward()
        self.assertGreater(float(state.grad[:, :2].abs().sum()), 0.)
        self.assertGreater(float(state.grad[:, 2:4].abs().sum()), 0.)
        self.assertGreater(float(state.grad[:, 8:12].abs().sum()), 0.)
        self.assertGreater(float(features.grad.abs().sum()), 0.)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0., name)
        moved = state.detach().clone(); moved[:, 0] += .04
        self.assertFalse(torch.equal(model.sample_local_map(state.detach(), features.detach()),
                                     model.sample_local_map(moved, features.detach())))

    def test_budget_and_shape_validation(self):
        model = SpatialEntityEncoder(512)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 305_280)
        self.assertLess(sum(p.numel() for p in model.parameters()), 500_000)
        for bad_state, fmap in ((torch.zeros(1, STATE_DIM-1), torch.zeros(1, 192, 16, 16)),
                                 (self.state(), torch.zeros(1, 191, 16, 16)),
                                 (self.state(), torch.zeros(1, 192, 15, 16))):
            with self.assertRaises(ValueError):
                model(bad_state, fmap)


if __name__ == '__main__':
    unittest.main()
