"""Learned anchor categories retain the legacy decoder's safety contract."""
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from turkeyworld.anchor_spawn import (anchor_target_labels, anchor_tile_indices,
                                     hierarchical_spawn_logits)
from turkeyworld.spawn import SPAWN_ANCHORS, decode_spawn_positions


class AnchorSpawnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def inputs(self, batch=1, simultaneous=False):
        state = torch.zeros(batch, 90)
        state[:, :2] = torch.tensor([8.5, 8.5])/16
        state[:, 3:6] = 1
        npc = state[:, 8:88].reshape(batch, 8, 10)
        npc[:, :, :2] = torch.tensor([8.5, 8.5])/16
        npc[:, :, 8] = 5 if simultaneous else 0
        npc[:, 0, 8] = 5
        grid = torch.ones(batch, 16, 16, dtype=torch.bool)
        grid[:, 0] = grid[:, -1] = False
        grid[:, :, 0] = grid[:, :, -1] = False
        return state, grid

    def test_labels_cover_eight_anchors_and_the_nonanchor_fallback(self):
        xy = torch.tensor([list(pair) for pair in SPAWN_ANCHORS]+[[4, 7]], dtype=torch.float32)
        self.assertEqual(anchor_target_labels((xy+.5)/16).tolist(), list(range(9)))
        self.assertEqual(anchor_tile_indices().tolist(), [y*16+x for x, y in SPAWN_ANCHORS])

    def test_zero_residual_preserves_random_and_tied_legacy_decodes_exactly(self):
        state, grid = self.inputs(batch=3, simultaneous=True)
        state[1, :2] = torch.tensor([2.5, 2.5])/16
        grid[2, 2, 2] = False
        for logits in (torch.randn(3, 8, 256), torch.zeros(3, 8, 256)):
            old = decode_spawn_positions(logits, state, grid, 1/30)
            heads = hierarchical_spawn_logits(logits, torch.zeros(3, 8, 9), state, grid, 1/30)
            new = decode_spawn_positions(heads['spawn_logits'], state, grid, 1/30)
            for first, second in zip(old, new):
                torch.testing.assert_close(first, second, atol=0, rtol=0)
            probability = heads['spawn_logits'].exp()
            torch.testing.assert_close(probability.sum(-1), torch.ones(3, 8), atol=2e-6, rtol=0)
            self.assertTrue(bool(torch.isfinite(heads['spawn_logits']).all()))

    def test_player_wall_and_npc_masks_override_learned_anchor_preferences(self):
        state, grid = self.inputs()
        state[:, :2] = torch.tensor([8.5, 2.5])/16
        npc = state[:, 8:88].reshape(1, 8, 10)
        npc[:, 1, :2] = torch.tensor([2.5, 2.5])/16
        npc[:, 1, 4:6] = torch.tensor([.6, 1.])
        grid[:, 2, 13] = False
        residual = torch.zeros(1, 8, 9)
        residual[:, :, :4] = torch.tensor([100., 90., 80., 70.])
        heads = hierarchical_spawn_logits(torch.zeros(1, 8, 256), residual, state, grid, 1/30)
        positions, valid = decode_spawn_positions(heads['spawn_logits'], state, grid, 1/30)
        self.assertEqual(heads['spawn_anchor_legal'][0, 0, :4].tolist(), [False, False, False, True])
        torch.testing.assert_close(positions[0, 0]*16, torch.tensor([13.5, 8.5]))
        self.assertTrue(bool(valid[0, 0]))

    def test_blocked_anchors_use_a_genuine_learned_nonanchor_fallback(self):
        state, _ = self.inputs()
        grid = torch.zeros(1, 16, 16, dtype=torch.bool)
        grid[:, 14, 14] = True
        residual = torch.zeros(1, 8, 9); residual[:, :, 0] = 1000
        heads = hierarchical_spawn_logits(torch.randn(1, 8, 256), residual, state, grid, 1/30)
        positions, valid = decode_spawn_positions(heads['spawn_logits'], state, grid, 1/30)
        self.assertFalse(bool(heads['spawn_anchor_legal'][0, 0, :8].any()))
        self.assertTrue(bool(heads['spawn_anchor_legal'][0, 0, 8]))
        self.assertTrue(bool(valid[0, 0]))
        torch.testing.assert_close(positions[0, 0]*16, torch.tensor([14.5, 14.5]))

    def test_simultaneous_slots_respect_id_order_and_separation(self):
        state, grid = self.inputs(simultaneous=True)
        residual = torch.zeros(1, 8, 9); residual[:, :, :8] = torch.arange(28., 20., -1)
        heads = hierarchical_spawn_logits(torch.zeros(1, 8, 256), residual, state, grid, 1/30)
        positions, valid = decode_spawn_positions(heads['spawn_logits'], state, grid, 1/30)
        expected = (torch.tensor(SPAWN_ANCHORS, dtype=torch.float32)+.5)/16
        torch.testing.assert_close(positions[0], expected, atol=0, rtol=0)
        self.assertTrue(bool(valid.all()))
        self.assertFalse(bool(heads['spawn_anchor_legal'][0, 1, 0]))

    def test_empty_map_and_terminal_player_are_finite_and_do_not_revive(self):
        for terminal in (False, True):
            state, grid = self.inputs(simultaneous=True)
            if terminal:
                state[:, 4] = 0
            else:
                grid.zero_()
            raw, residual = torch.randn(1, 8, 256), torch.randn(1, 8, 9)
            heads = hierarchical_spawn_logits(raw, residual, state, grid, 1/30)
            positions, valid = decode_spawn_positions(heads['spawn_logits'], state, grid, 1/30)
            self.assertTrue(bool(torch.isfinite(heads['spawn_logits']).all()))
            self.assertTrue(bool(torch.isfinite(heads['spawn_anchor_logits']).all()))
            self.assertFalse(bool(valid.any()))
            torch.testing.assert_close(positions, state[:, 8:88].reshape(1, 8, 10)[:, :, :2])

    def test_anchor_and_fallback_gradients_exist_without_state_or_label_gradients(self):
        state, grid = self.inputs(simultaneous=True)
        state.requires_grad_()
        raw = torch.randn(1, 8, 256, requires_grad=True)
        residual = torch.randn(1, 8, 9, requires_grad=True)
        heads = hierarchical_spawn_logits(raw, residual, state, grid, 1/30)
        anchors = anchor_tile_indices()
        allowed = heads['spawn_legal_mask'][0, 1].clone(); allowed[anchors] = False
        fallback_label = allowed.nonzero()[0, 0]
        anchor_label = heads['spawn_anchor_legal'][0, 0, :8].nonzero()[0, 0]
        loss = F.cross_entropy(heads['spawn_logits'][:, 0], anchors[anchor_label][None])
        loss = loss+F.cross_entropy(heads['spawn_logits'][:, 1], fallback_label[None])
        loss.backward()
        self.assertIsNone(state.grad)
        self.assertTrue(bool(torch.isfinite(raw.grad).all()))
        self.assertTrue(bool(torch.isfinite(residual.grad).all()))
        self.assertGreater(float(raw.grad[:, :, anchors].abs().sum()), 0)
        self.assertGreater(float(raw.grad[:, :, allowed].abs().sum()), 0)
        self.assertGreater(float(residual.grad[:, :, :8].abs().sum()), 0)
        self.assertGreater(float(residual.grad[:, :, 8].abs().sum()), 0)

    def test_unmasked_coarse_supervision_can_train_an_excluded_source_anchor(self):
        state, grid = self.inputs()
        grid[:, 2, 2] = False
        residual = torch.zeros(1, 8, 9, requires_grad=True)
        heads = hierarchical_spawn_logits(torch.zeros(1, 8, 256), residual, state, grid, 1/30)
        self.assertFalse(bool(heads['spawn_anchor_legal'][0, 0, 0]))
        loss = F.cross_entropy(heads['spawn_anchor_logits'][:, 0], torch.zeros(1, dtype=torch.long))
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertLess(float(residual.grad[0, 0, 0]), 0)


class AnchorBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def backend(self, enabled):
        from turkeyworld.models import WorldModelConfig
        from turkeyworld.vjepa2_ac import VJEPA2ACWorldBackend
        config = WorldModelConfig(kind='student', architecture='vjepa2_ac', image_size=32,
            vjepa_patch_size=4, endless=True, renderer_mode='flow_matching',
            renderer_codec='vqgan', renderer_codec_config={'image_size': 32}, vjepa_width=32,
            vjepa_encoder_depth=1, vjepa_predictor_depth=1, vjepa_heads=4,
            spatial_entities=True, spatial_combat=True, game_training_v5=enabled)
        return VJEPA2ACWorldBackend(config).eval()

    def test_v5_adds_only_zero_anchor_head_and_retains_old_weights_and_decodes(self):
        from turkeyworld.engine import Game
        old, new = self.backend(False), self.backend(True)
        missing, extra = new.load_state_dict(old.state_dict(), strict=False)
        self.assertFalse(extra)
        self.assertEqual(set(missing), {'spatial_spawn_anchor_head.weight', 'spatial_spawn_anchor_head.bias'})
        for key, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
        state, grid = AnchorSpawnTests().inputs(batch=2, simultaneous=True)
        grid = (~grid).long()
        rgb, action = torch.rand(2, 2, 3, 32, 32), torch.zeros(2, 4)
        with (torch.inference_mode(), patch.object(Game, 'step', side_effect=AssertionError('engine')),
                patch.object(Game, 'render', side_effect=AssertionError('engine'))):
            first_map, second_map = old.encode_map(grid), new.encode_map(grid)
            first_memory = old.initialize_from_image(rgb, state, first_map)
            second_memory = new.initialize_from_image(rgb, state, second_map)
            from turkeyworld.models import project_state
            first = old.transition(state, action, first_map, first_memory, project_state)
            second = new.transition(state, action, second_map, second_memory, project_state)
        for key in ('state', 'memory', 'spawn_positions', 'spawn_valid', 'target_logits', 'attack_logits'):
            torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
        self.assertEqual(second['spawn_anchor_logits'].shape, (2, 8, 9))
        self.assertEqual(second['spawn_fallback_logits'].shape, (2, 8, 256))
        self.assertEqual(sum(parameter.numel() for parameter in new.spatial_spawn_anchor_head.parameters()), 297)
        self.assertFalse(hasattr(old, 'spatial_spawn_anchor_head'))

    def test_v5_spawn_image_uses_selected_maps_and_balanced_actual_mask(self):
        import test_spawn_render_training as fixture
        from turkeyworld.object_losses import balanced_image_l1
        from turkeyworld.spawn_render_training import spawn_image_objective
        model = fixture.TinySpawnModel()
        model.config.game_training_v5 = True
        data, _ = fixture.payload(model)
        data['object_masks'] = torch.zeros_like(data['object_masks'])
        data['object_masks'][:, :, 0, 0] = 1
        targets = data['target_frames'].detach().clone()
        targets[:, :, :, 0, 0] = .1
        data['target_frames'] = targets.requires_grad_()
        data['walkable'][0, 0] = False
        data['walkable'][1, -1] = False
        maps_seen, old_method = [], model.vjepa_backend.spawn_logits

        def hierarchical(state, pooled, spatial, maps, *, walkable=None):
            maps_seen.append(walkable.detach().clone())
            raw = old_method(state, pooled, spatial, maps)
            return hierarchical_spawn_logits(raw, raw.new_zeros(len(state), 8, 9),
                                             state, walkable, 1/30)['spawn_logits']

        model.vjepa_backend.spawn_logits = hierarchical
        loss, metrics = spawn_image_objective(model, data)
        rows, steps = torch.tensor([0, 1]), torch.tensor([3, 2])
        torch.testing.assert_close(maps_seen[0], data['walkable'][rows])
        expected = balanced_image_l1(model.flow_renderer.last_prediction,
            targets[rows, steps], data['object_masks'][rows, steps])
        self.assertAlmostEqual(metrics['spawn_image_rgb_l1'], float(expected.detach()), places=6)
        loss.backward()
        self.assertIsNone(data['target_frames'].grad)
        self.assertIsNone(model.core.weight.grad)
        self.assertGreater(float(model.vjepa_backend.spawn_head.weight.grad.abs().sum()), 0)

    def test_actual_anchor_head_and_transferred_tile_head_both_train(self):
        backend = self.backend(True).train()
        state, walkable = AnchorSpawnTests().inputs(batch=2, simultaneous=True)
        maps = backend.encode_map((~walkable).long())
        spatial = backend.spatial_tokens(state, maps)
        heads = backend.spawn_outputs(state, torch.randn(2, 32), spatial, maps)
        labels = torch.zeros(2*8, dtype=torch.long)
        loss = F.cross_entropy(heads['spawn_anchor_logits'].reshape(-1, 9), labels)
        loss.backward()
        for weight in (backend.spatial_spawn_anchor_head.weight, backend.spawn_head[-1].weight):
            self.assertIsNotNone(weight.grad)
            self.assertTrue(bool(torch.isfinite(weight.grad).all()))
            self.assertGreater(float(weight.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
