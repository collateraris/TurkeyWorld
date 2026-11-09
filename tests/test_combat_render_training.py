"""Post-backward RGB supervision reaches combat heads, not saved visual graphs."""
import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from turkeyworld.combat_render_training import combat_image_objective
from turkeyworld.object_losses import foreground_contrastive_loss, object_region_weights, weighted_image_l1
from turkeyworld.structured_state import structured_transition


class TinyCombatBackend(nn.Module):
    def __init__(self):
        super().__init__()
        self.hit_target_head = nn.Linear(6, 9)
        self.attack_head = nn.Linear(6, 8)
        self.spatial_combat_head = nn.Linear(6, 2, bias=False)
        self.spatial_encoder = nn.Linear(6, 6)
        self.calls = []
        with torch.no_grad():
            self.hit_target_head.weight.zero_(); self.hit_target_head.bias.fill_(-1.)
            self.hit_target_head.bias[0] = 2.
            self.attack_head.weight.zero_(); self.attack_head.bias.fill_(-2.)
            self.attack_head.bias[1] = .5
            self.spatial_combat_head.weight.fill_(.01)

    def combat_logits(self, state, pooled, spatial=None):
        self.calls.append(tuple(value.detach().clone() for value in (state, pooled, spatial)))
        pair = self.spatial_combat_head(spatial[:, 1:])
        targets = self.hit_target_head(pooled)
        targets = torch.cat([targets[:, :8]+pair[..., 0], targets[:, 8:]], -1)
        return targets, self.attack_head(pooled)+pair[..., 1]


class TinyFullSampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(.3))
        self.calls = []
        self.last_prediction = None

    def sample(self, base, state, *, checkpoint_velocity=False):
        self.calls.append({'base': base.detach().clone(), 'state': state.detach().clone(),
                           'checkpoint_velocity': checkpoint_velocity})
        # A visible HP/tint channel supplies a genuine image derivative for
        # accepted melee damage; no new image classifier is supervised.
        combat = torch.stack([state[:, 12]+state[:, 13], state[:, 4], state[:, 15]], -1)
        self.last_prediction = .7*base+.3*torch.sigmoid(self.gain*combat[:, :, None, None])
        return self.last_prediction


class TinyCombatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(simulation_hz=30, endless=True, spatial_combat=True)
        self.vjepa_backend = TinyCombatBackend()
        self.flow_renderer = TinyFullSampler()
        self.core = nn.Linear(6, 6)
        self.feature_head = nn.Linear(6, 128)
        self.map_encoder = nn.Linear(6, 2*16*16)
        self.base_gain = nn.Parameter(torch.tensor(.4))
        self.render_calls = []

    def render_base(self, state, features, map_features):
        self.render_calls.append({'state': state.detach().clone(), 'features_requires_grad': features.requires_grad,
            'map_requires_grad': map_features.requires_grad})
        combat = torch.stack([state[:, 12]+state[:, 13]+.2*state[:, 15],
                              state[:, 4], state[:, 12]+.5*state[:, 15]], -1)
        return torch.sigmoid(self.base_gain*combat[:, :, None, None]).expand(-1, -1, 4, 4)


class CombatRenderTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def payload(model, batch=3, horizon=5, eventful=True):
        source = torch.arange(batch*horizon*6, dtype=torch.float32).reshape(batch, horizon, 6)/100
        source.requires_grad_()
        pooled = model.core(source)
        spatial = model.vjepa_backend.spatial_encoder(pooled)[:, :, None].expand(-1, -1, 9, -1)
        features = model.feature_head(pooled)
        map_features = model.map_encoder(pooled[:, 0]).reshape(batch, 2, 16, 16)
        state = torch.zeros(batch, horizon, 90)
        state[:, :, :2] = torch.tensor([2.5/16, 2.5/16])
        state[:, :, 3] = 1.
        state[:, :, 4:6] = torch.tensor([.8, 1.])
        state[:, :, 8:10] = torch.tensor([3.1/16, 2.5/16])
        state[:, :, 12:14] = torch.tensor([.25, 1.])
        state[:, :, 18:20] = torch.tensor([3.2/16, 2.5/16])
        state[:, :, 22:24] = torch.tensor([.6, 1.])
        state.requires_grad_()
        target = state.detach().clone()
        target[:, :, 0] += .02
        target[:, :, 8] += .03
        target[:, :, 88:] = .6
        target.requires_grad_()
        events = torch.zeros(batch, horizon, 4)
        if eventful and horizon >= 5:
            events[0, 1, 2] = 1.; events[0, 4, 1] = 1.
            if batch > 1: events[1, 2, 1] = 1.; events[1, 4, 3] = 1.
            if batch > 2: events[2, 0, 0] = 1.
        actions = torch.zeros(batch, horizon, 4); actions[:, :, 3] = 1.
        mask = torch.zeros(batch, horizon, 4, 4); mask[:, :, 1:3, 1:3] = 1.
        payload = {'contexts': {'state': state, 'pooled': pooled, 'spatial': spatial},
            'features': features, 'target_states': target, 'actions': actions.requires_grad_(),
            'events': events.requires_grad_(), 'target_frames': torch.full((batch, horizon, 3, 4, 4), .9, requires_grad=True),
            'object_masks': mask.requires_grad_(), 'backgrounds': torch.full((batch, horizon, 3, 4, 4), .1, requires_grad=True),
            'map_features': map_features}
        return payload, source

    def test_event_priority_and_latest_default_offsets_and_json_metrics(self):
        model = TinyCombatModel()
        payload, _ = self.payload(model)
        loss, metrics = combat_image_objective(model, payload)
        self.assertEqual(metrics['combat_image_selected_offsets'], [1, 2, 0])
        self.assertEqual(metrics['combat_image_selected_samples'], 3)
        self.assertEqual(metrics['combat_image_selected_event_counts'], {'shot': 1, 'hit': 1, 'kill': 1, 'player_hit': 0})
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.detach()), 0)
        json.dumps(metrics, allow_nan=False)
        self.assertTrue(metrics['combat_image_source_labels_selection_only'])
        self.assertFalse(metrics['combat_image_future_feedback'])
        payload, _ = self.payload(model, eventful=False)
        payload['events'].data[0, 1, 1] = payload['events'].data[0, 3, 1] = 1.
        _, metrics = combat_image_objective(model, payload)
        self.assertEqual(metrics['combat_image_selected_offsets'], [3, 4, 4])

    def test_physical_forward_fields_and_gt_geometry_are_separate(self):
        model = TinyCombatModel(); payload, _ = self.payload(model)
        _, metrics = combat_image_objective(model, payload)
        rows, offsets = torch.arange(3), torch.tensor(metrics['combat_image_selected_offsets'])
        previous = payload['contexts']['state'][rows, offsets].detach()
        target = payload['target_states'][rows, offsets].detach()
        inputs = model.vjepa_backend.calls[-1]
        target_logits, attack_logits = model.vjepa_backend.combat_logits(*inputs)
        expected = structured_transition(previous, target, payload['actions'][rows, offsets].detach(),
            target_logits, attack_logits, training=False, endless=True)['state']
        rendered = model.render_calls[-1]['state']
        torch.testing.assert_close(rendered[:, :4], target[:, :4], rtol=0, atol=0)
        torch.testing.assert_close(rendered[:, 5:7], target[:, 5:7], rtol=0, atol=0)
        torch.testing.assert_close(rendered[:, 88:], target[:, 88:], rtol=0, atol=0)
        torch.testing.assert_close(rendered[:, 4], expected[:, 4], rtol=0, atol=0)
        torch.testing.assert_close(rendered[:, 7], expected[:, 7], rtol=0, atol=0)
        actual_npc, target_npc, hard_npc = (value[:, 8:88].reshape(-1, 8, 10) for value in (rendered, target, expected))
        for columns in ((0, 1, 2, 3, 9),):
            torch.testing.assert_close(actual_npc[:, :, columns], target_npc[:, :, columns], rtol=0, atol=0)
        torch.testing.assert_close(actual_npc[:, :, 4:9], hard_npc[:, :, 4:9], rtol=0, atol=0)
        self.assertTrue((actual_npc[:, 0, 5] == 0).all())
        torch.testing.assert_close(model.flow_renderer.calls[-1]['state'], rendered, rtol=0, atol=0)
        self.assertTrue(model.flow_renderer.calls[-1]['checkpoint_velocity'])

    def test_post_primary_backward_recomputes_heads_without_upstream_or_source_gradients(self):
        from turkeyworld.engine import Game
        model = TinyCombatModel(); payload, source = self.payload(model)
        main = sum(payload['contexts'][name].square().mean() for name in ('pooled', 'spatial'))
        main = main+payload['features'].square().mean()+payload['map_features'].square().mean()
        main.backward()
        model.zero_grad(set_to_none=True); source.grad = None
        with patch.object(Game, 'step', side_effect=AssertionError('No engine step')),\
                patch.object(Game, 'render', side_effect=AssertionError('No engine RGB')),\
                patch.object(model.core, 'forward', side_effect=AssertionError('No visual recomputation')),\
                patch.object(model.vjepa_backend.spatial_encoder, 'forward', side_effect=AssertionError('No token recomputation')):
            loss, _ = combat_image_objective(model, payload)
            loss.backward()
        backend = model.vjepa_backend
        for parameter in (backend.hit_target_head.bias, backend.attack_head.bias,
                          backend.spatial_combat_head.weight, model.base_gain, model.flow_renderer.gain):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertTrue(all(parameter.grad is None for module in
            (model.core, model.feature_head, model.map_encoder, backend.spatial_encoder) for parameter in module.parameters()))
        for value in (source, payload['contexts']['state'], payload['target_states'], payload['actions'],
                      payload['events'], payload['target_frames'], payload['object_masks'], payload['backgrounds']):
            self.assertIsNone(value.grad)
        self.assertFalse(model.render_calls[-1]['features_requires_grad'])
        self.assertFalse(model.render_calls[-1]['map_requires_grad'])

    def test_target_damage_never_enters_head_inputs_or_accepted_image_fields(self):
        model = TinyCombatModel(); payload, _ = self.payload(model)
        combat_image_objective(model, payload)
        first_inputs = model.vjepa_backend.calls[-1]
        first_state = model.render_calls[-1]['state']
        changed = dict(payload); target = payload['target_states'].detach().clone()
        target[:, :, 4] = .01
        target[:, :, 12] = .01
        target[:, :, 22] = .02
        target[:, :, 15:17] = .8
        changed['target_states'] = target
        combat_image_objective(model, changed)
        for actual, expected in zip(model.vjepa_backend.calls[-1], first_inputs):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(model.render_calls[-1]['state'], first_state, rtol=0, atol=0)

    def test_recorded_respawn_offsets_filtered_and_all_invalid_is_connected_zero(self):
        model = TinyCombatModel(); payload, _ = self.payload(model, batch=3, eventful=False)
        # Slot7 is dead in the previous state. A recorded respawn must exclude
        # the entire query, rather than substitute future life into own combat.
        payload['target_states'].data[0, 4, 82:84] = torch.tensor([.6, 1.])
        payload['target_states'].data[1, :, 82:84] = torch.tensor([.6, 1.])
        payload['events'].data[0, 4, 2] = 1.
        loss, metrics = combat_image_objective(model, payload)
        self.assertEqual(metrics['combat_image_selected_offsets'], [3, -1, 4])
        self.assertEqual(metrics['combat_image_skipped_respawn_offsets'], 6)
        self.assertEqual(metrics['combat_image_selected_samples'], 2)
        self.assertEqual(model.render_calls[-1]['state'].shape[0], 2)
        loss.backward(); model.zero_grad(set_to_none=True)
        payload['target_states'].data[:, :, 82:84] = torch.tensor([.6, 1.])
        with patch.object(model, 'render_base', side_effect=AssertionError('No valid image')),\
                patch.object(model.flow_renderer, 'sample', side_effect=AssertionError('No valid Flow')):
            zero, metrics = combat_image_objective(model, payload)
            self.assertEqual(float(zero.detach()), 0.)
            self.assertTrue(zero.requires_grad)
            zero.backward()
        self.assertEqual(metrics['combat_image_selected_offsets'], [-1, -1, -1])
        self.assertEqual(metrics['combat_image_skipped_samples'], 3)
        for head in (model.vjepa_backend.hit_target_head, model.vjepa_backend.attack_head,
                     model.vjepa_backend.spatial_combat_head):
            for parameter in head.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertEqual(float(parameter.grad.abs().sum()), 0.)
        self.assertIsNone(model.base_gain.grad)
        self.assertTrue(all(parameter.grad is None for parameter in model.core.parameters()))

    def test_exact_loss_weights_present_pixel_metrics_and_empty_masks(self):
        model = TinyCombatModel(); payload, _ = self.payload(model)
        loss, metrics = combat_image_objective(model, payload)
        rows, offsets = torch.arange(3), torch.tensor(metrics['combat_image_selected_offsets'])
        prediction = model.flow_renderer.last_prediction
        mask, target, background = (payload[name][rows, offsets].detach() for name in
                                    ('object_masks', 'target_frames', 'backgrounds'))
        rgb = weighted_image_l1(prediction, target, object_region_weights(mask, 4.))
        contrast = foreground_contrastive_loss(prediction, target, background, mask, .25)
        torch.testing.assert_close(loss, .25*rgb+.25*contrast)
        self.assertAlmostEqual(metrics['combat_image_npc_mae'], float(weighted_image_l1(prediction, target, mask).detach()))
        self.assertEqual(metrics['combat_image_npc_pixel_fraction'], .25)
        payload['object_masks'] = torch.zeros_like(payload['object_masks'])[:, :, None]
        empty_loss, empty_metrics = combat_image_objective(model, payload)
        self.assertTrue(torch.isfinite(empty_loss))
        self.assertEqual(empty_metrics['combat_image_npc_mae'], 0.)
        self.assertEqual(empty_metrics['combat_image_foreground_contrast'], 0.)
        self.assertEqual(empty_metrics['combat_image_npc_pixel_fraction'], 0.)

    def test_malformed_nonfinite_and_misaligned_payloads_are_rejected_before_model_calls(self):
        model = TinyCombatModel(); payload, _ = self.payload(model)
        for problem in ('missing', 'shape', 'nan', 'mask_range', 'event_range', 'wrong_context'):
            with self.subTest(problem=problem):
                broken = dict(payload); broken['contexts'] = dict(payload['contexts'])
                if problem == 'missing': del broken['backgrounds']
                elif problem == 'shape': broken['features'] = torch.zeros(3, 5, 127)
                elif problem == 'wrong_context': broken['contexts']['spatial'] = torch.zeros(3, 5, 8, 6)
                else:
                    key = {'nan': 'target_frames', 'mask_range': 'object_masks', 'event_range': 'events'}[problem]
                    broken[key] = broken[key].detach().clone()
                    broken[key].flatten()[0] = float('nan') if problem == 'nan' else 2.
                with self.assertRaises(ValueError): combat_image_objective(model, broken)
        self.assertFalse(model.vjepa_backend.calls)


if __name__ == '__main__':
    unittest.main()
