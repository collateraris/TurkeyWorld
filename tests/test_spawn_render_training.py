"""Hard-safe spawning has a separate differentiable, label-only image query."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from turkeyworld.object_losses import object_region_weights, weighted_image_l1
from turkeyworld.spawn import decode_spawn_positions
from turkeyworld.spawn_render_training import spawn_image_objective, spawn_positions_with_grad


def initial(batch=1, dtype=torch.float32):
    state = torch.zeros(batch, STATE_DIM, dtype=dtype)
    state[:, :2] = 2.5/16
    state[:, 3:6] = 1
    npc = state[:, GLOBAL_DIM:-2].reshape(batch, ENTITY_SLOTS, ENTITY_DIM)
    for slot in range(ENTITY_SLOTS):
        npc[:, slot, :2] = torch.tensor([(slot+5.5)/16, 4.5/16], dtype=dtype)
        npc[:, slot, 8] = 5
    return state


class TinySpawnBackend(nn.Module):
    def __init__(self):
        super().__init__()
        self.spawn_head = nn.Linear(6, ENTITY_SLOTS*256)
        self.spatial_spawn_query = nn.Linear(6, 256)
        self.hit_target_head = nn.Linear(6, (ENTITY_SLOTS+1))
        self.attack_head = nn.Linear(6, ENTITY_SLOTS)
        self.spatial_encoder = nn.Linear(6, 6)
        self.renderer = nn.Module()
        self.renderer.register_buffer('_spawn_walkable', torch.ones(1, 16, 16, dtype=torch.bool), persistent=False)
        self.calls = []
        with torch.no_grad():
            self.spawn_head.weight.normal_(std=.01); self.spawn_head.bias.zero_()
            self.spawn_head.bias.reshape(ENTITY_SLOTS, 256)[:, 9*16+9] = 1.
            self.spatial_spawn_query.weight.fill_(.001); self.spatial_spawn_query.bias.zero_()

    def spawn_logits(self, state, pooled, spatial, map_features):
        self.calls.append(tuple(value.detach().clone() for value in (state, pooled, spatial, map_features)))
        return self.spawn_head(pooled).reshape(-1, ENTITY_SLOTS, 256)+self.spatial_spawn_query(spatial[:, 1:])


class TinySampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(.3))
        self.calls = []

    def sample(self, base, state, *, checkpoint_velocity=False):
        self.calls.append({'state': state.detach().clone(), 'checkpoint_velocity': checkpoint_velocity})
        self.last_prediction = .7*base+.3*torch.sigmoid(self.gain*state[:, 8:10].sum(-1)[:, None, None, None])
        return self.last_prediction


class TinySpawnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(simulation_hz=30, endless=True)
        self.vjepa_backend = TinySpawnBackend()
        self.core = nn.Linear(6, 6)
        self.feature_head = nn.Linear(6, 128)
        self.map_encoder = nn.Linear(6, 2*16*16)
        self.base_gain = nn.Parameter(torch.tensor(.4))
        self.flow_renderer = TinySampler()
        self.calls = []

    def render_base(self, state, features, map_features):
        self.calls.append({'state': state.detach().clone(), 'features_requires_grad': features.requires_grad,
                           'map_requires_grad': map_features.requires_grad})
        position = torch.stack([state[:, 8], state[:, 9], state[:, 8]+state[:, 9]], -1)
        return torch.sigmoid(self.base_gain*position[:, :, None, None]).expand(-1, -1, 4, 4)


def payload(model, batch=3, horizon=4):
    source = torch.arange(batch*horizon*6, dtype=torch.float32).reshape(batch, horizon, 6)/100
    source.requires_grad_()
    pooled = model.core(source)
    spatial = model.vjepa_backend.spatial_encoder(pooled)[:, :, None].expand(-1, -1, ENTITY_SLOTS+1, -1)
    previous = initial(batch*horizon).reshape(batch, horizon, STATE_DIM)
    previous.requires_grad_()
    target = previous.detach().clone()
    target[:, :, 0] += .02
    if batch >= 1:
        target[0, 1, 8:10] = torch.tensor([13.5/16, 13.5/16])
        target[0, 1, 12:14] = torch.tensor([.6, 1.])
        target[0, 3, 8:10] = torch.tensor([12.5/16, 13.5/16])
        target[0, 3, 12:14] = torch.tensor([.6, 1.])
    if batch >= 2:
        target[1, 2, 8:10] = torch.tensor([13.5/16, 13.5/16])
        target[1, 2, 12:14] = torch.tensor([.6, 1.])
    target.requires_grad_()
    masks = torch.zeros(batch, horizon, 4, 4); masks[:, :, 1:3, 1:3] = 1
    result = {'contexts': {'state': previous, 'pooled': pooled, 'spatial': spatial},
        'features': model.feature_head(pooled),
        'map_features': model.map_encoder(pooled[:, 0]).reshape(batch, 2, 16, 16),
        'target_states': target, 'target_frames': torch.full((batch, horizon, 3, 4, 4), .9, requires_grad=True),
        'object_masks': masks.requires_grad_(), 'walkable': torch.ones(batch, 16, 16, dtype=torch.bool)}
    return result, source


class DifferentiableSpawnDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_hard_forward_exact_for_random_masks_ready_and_live_slots(self):
        rng = torch.Generator().manual_seed(281)
        for dtype in (torch.float32, torch.float64):
            previous = initial(4, dtype)
            npc = previous[:, GLOBAL_DIM:-2].reshape(4, ENTITY_SLOTS, ENTITY_DIM)
            npc[1, 0, 4:6] = torch.tensor([.6, 1.], dtype=dtype)
            npc[1, :, 8] = 4.
            previous[2, 4] = 0.
            maps = torch.rand(4, 16, 16, generator=rng) > .2
            maps[3] = False
            logits = torch.randn(4, ENTITY_SLOTS, 256, generator=rng, dtype=dtype, requires_grad=True)
            expected = decode_spawn_positions(logits, previous, maps, 1/30, deterministic=False)
            actual = spawn_positions_with_grad(logits, previous, maps, 1/30)
            torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
            self.assertTrue(torch.equal(actual[1], expected[1]))
            self.assertTrue(torch.isfinite(actual[0]).all())
            actual[0].square().sum().backward()
            self.assertGreater(float(logits.grad[0].abs().sum()), 0.)
            torch.testing.assert_close(logits.grad[1:], torch.zeros_like(logits.grad[1:]), rtol=0, atol=0)

    def test_single_slot_safety_and_illegal_cell_zero_gradients(self):
        previous = initial(); previous[:, :2] = torch.tensor([7.5/16, 7.5/16])
        walkable = torch.ones(16, 16, dtype=torch.bool); walkable[0] = False
        logits = torch.zeros(1, ENTITY_SLOTS, 256, requires_grad=True)
        positions, available = spawn_positions_with_grad(logits, previous, walkable, 1/30)
        expected = decode_spawn_positions(logits, previous, walkable, 1/30)
        torch.testing.assert_close(positions, expected[0], rtol=0, atol=0)
        self.assertTrue(available.all())
        locations = positions.detach()[0]*16
        self.assertEqual(locations.shape, (1, 2))
        self.assertTrue(walkable[locations[:, 1].long(), locations[:, 0].long()].all())
        self.assertTrue((((locations-previous[0, :2]*16)**2).sum(-1) >= 16).all())
        positions[0, 0, 0].backward()
        self.assertGreater(float(logits.grad[0, 0].abs().sum()), 0.)
        self.assertTrue((logits.grad[0, 0, :16] == 0).all())
        # Player's own tile is forbidden in the soft surrogate too.
        self.assertEqual(float(logits.grad[0, 0, 7*16+7]), 0.)

    def test_empty_floor_unready_timer_and_no_player_life_are_finite_zero_gradients(self):
        for full_wall, death_clock, player_hp in ((True, 5., 1.), (False, 4., 1.), (False, 5., 0.)):
            previous = initial(); previous[:, 4] = player_hp
            previous[:, GLOBAL_DIM:-2].reshape(1, ENTITY_SLOTS, ENTITY_DIM)[:, :, 8] = death_clock
            logits = torch.randn(1, ENTITY_SLOTS, 256, requires_grad=True)
            result, available = spawn_positions_with_grad(logits, previous, torch.full((16, 16), not full_wall), 1/30)
            self.assertFalse(available.any())
            self.assertTrue(torch.isfinite(result).all())
            result.sum().backward()
            self.assertTrue((logits.grad == 0).all())
        with self.assertRaises(ValueError):
            spawn_positions_with_grad(logits, previous, torch.ones(15, 16), 1/30)


class SpawnImageObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_selects_most_spawned_then_latest_and_uses_same_source_pose(self):
        model = TinySpawnModel(); data, _ = payload(model)
        loss, metrics = spawn_image_objective(model, data)
        self.assertEqual(metrics['spawn_image_selected_offsets'], [3, 2, -1])
        self.assertEqual(metrics['spawn_image_recorded_slots'], 2)
        self.assertEqual(metrics['spawn_image_available_slots'], 2)
        self.assertEqual(metrics['spawn_image_selected_samples'], 2)
        self.assertTrue(torch.isfinite(loss))
        json.dumps(metrics, allow_nan=False)
        rows, steps = torch.tensor([0, 1]), torch.tensor([3, 2])
        pose = data['target_states'][rows, steps].detach()
        captured = model.vjepa_backend.calls[-1]
        logits = model.vjepa_backend.spawn_logits(*captured)
        expected, _ = decode_spawn_positions(logits, captured[0], data['walkable'][rows], 1/30)
        rendered = model.calls[-1]['state']
        torch.testing.assert_close(rendered[:, :8], pose[:, :8], rtol=0, atol=0)
        torch.testing.assert_close(rendered[:, -2:], pose[:, -2:], rtol=0, atol=0)
        torch.testing.assert_close(rendered[:, 8:10], expected[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(rendered[:, 10:18], pose[:, 10:18], rtol=0, atol=0)
        torch.testing.assert_close(model.flow_renderer.calls[-1]['state'], rendered, rtol=0, atol=0)
        self.assertTrue(model.flow_renderer.calls[-1]['checkpoint_velocity'])

    def test_rgb_gradient_reaches_spawn_heads_and_no_freed_context_or_labels(self):
        from turkeyworld.engine import Game
        model = TinySpawnModel(); data, source = payload(model)
        sum(data['contexts'][name].square().mean() for name in ('pooled', 'spatial')).backward(retain_graph=True)
        (data['features'].square().mean()+data['map_features'].square().mean()).backward()
        model.zero_grad(set_to_none=True); source.grad = None
        with patch.object(Game, 'step', side_effect=AssertionError('No engine')),\
                patch.object(Game, 'render', side_effect=AssertionError('No engine RGB')),\
                patch.object(model.core, 'forward', side_effect=AssertionError('No core replay')),\
                patch.object(model.vjepa_backend.spatial_encoder, 'forward', side_effect=AssertionError('No spatial replay')):
            _, metrics = spawn_image_objective(model, data)
            rows, steps = torch.tensor([0, 1]), torch.tensor([3, 2])
            rgb = weighted_image_l1(model.flow_renderer.last_prediction, data['target_frames'][rows, steps].detach(),
                object_region_weights(data['object_masks'][rows, steps].detach(), boost=4.))
            rgb.backward()
        for parameter in (model.vjepa_backend.spawn_head.bias, model.vjepa_backend.spatial_spawn_query.weight,
                          model.base_gain, model.flow_renderer.gain):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        for module in (model.core, model.feature_head, model.map_encoder, model.vjepa_backend.spatial_encoder,
                       model.vjepa_backend.hit_target_head, model.vjepa_backend.attack_head):
            self.assertTrue(all(parameter.grad is None for parameter in module.parameters()))
        for value in (source, data['contexts']['state'], data['target_states'], data['target_frames'], data['object_masks']):
            self.assertIsNone(value.grad)
        self.assertFalse(model.calls[-1]['features_requires_grad'])
        self.assertFalse(model.calls[-1]['map_requires_grad'])
        self.assertFalse(metrics['spawn_image_future_feedback'])

    def test_target_spawn_coordinates_and_rgb_do_not_enter_head_or_own_spawn_positions(self):
        model = TinySpawnModel(); data, _ = payload(model)
        spawn_image_objective(model, data)
        before_inputs = model.vjepa_backend.calls[-1]
        before_state = model.calls[-1]['state']
        changed = dict(data); target = data['target_states'].detach().clone()
        target[:, :, 8:10] = torch.tensor([14.5/16, 12.5/16])
        changed['target_states'] = target
        changed['target_frames'] = torch.zeros_like(data['target_frames'])
        spawn_image_objective(model, changed)
        for first, second in zip(before_inputs, model.vjepa_backend.calls[-1]):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(before_state[:, 8:10], model.calls[-1]['state'][:, 8:10], rtol=0, atol=0)

    def test_no_source_respawn_returns_connected_zero_and_never_renders(self):
        model = TinySpawnModel(); data, _ = payload(model)
        data['target_states'] = data['contexts']['state'].detach().clone()
        before = copy.deepcopy(data['contexts']['state'].detach())
        with patch.object(model, 'render_base', side_effect=AssertionError('No unnecessary Base')),\
                patch.object(model.flow_renderer, 'sample', side_effect=AssertionError('No unnecessary sampler')):
            loss, metrics = spawn_image_objective(model, data)
            loss.backward()
        self.assertEqual(float(loss.detach()), 0.)
        self.assertEqual(metrics['spawn_image_selected_offsets'], [-1, -1, -1])
        for parameter in model.vjepa_backend.spawn_head.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue((parameter.grad == 0).all())
        torch.testing.assert_close(before, data['contexts']['state'].detach(), rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in model.core.parameters()))

    def test_unavailable_spawn_retains_own_corpse_not_source_life_and_cached_maps_work(self):
        model = TinySpawnModel(); data, _ = payload(model)
        data['walkable'].zero_()
        _, metrics = spawn_image_objective(model, data)
        self.assertEqual(metrics['spawn_image_available_slots'], 0)
        rendered = model.calls[-1]['state']
        self.assertTrue((rendered[:, 12:14] == 0).all())
        self.assertEqual(metrics['spawn_image_coordinate_loss'], 0.)
        data.pop('walkable')
        model.vjepa_backend.renderer._spawn_walkable = torch.ones(1, 16, 16, dtype=torch.bool)
        _, metrics = spawn_image_objective(model, data)
        self.assertEqual(metrics['spawn_image_available_slots'], 2)


if __name__ == '__main__':
    unittest.main()
