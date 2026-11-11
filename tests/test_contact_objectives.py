"""Balanced physical contacts, legal Source margins and nonmutating probes."""
import copy
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.contact_objectives import (CONTACT_PROTOCOL_V2, normalize_contact_protocol,
    prepare_contact_targets, differentiable_contact_terms, swept_player_wall_loss,
    contact_gradient_probe, _swept)
from turkeyworld.structured_state import player_command_delta
from turkeyworld.trajectory_objectives import circle_wall_penalty
from turkeyworld.rssm_objectives import rssm_dynamics_backward
from turkeyworld.engine import Game
from test_rssm_objectives import SpyRSSM
from test_long_dynamics_training import batch as long_batch
from test_motion_recovery_training import source_batch


def states(count):
    value = torch.zeros(count, 20)
    value[:, :2] = torch.tensor([3.5, 5.5])/16
    value[:, 3:6] = 1.
    return value


def prior(previous, action, logits):
    raw = player_command_delta(previous, action)*logits.sigmoid()
    return {'state': torch.cat([previous[:, :2]+raw, previous[:, 2:]], -1), 'player_pass_logits': logits,
            'raw_player_delta': raw}


def mixed_source():
    rows = []
    for index in range(2):
        game = Game(seed=index, endless=True, enemies_enabled=False)
        if index:
            initial = game.get_state()
            initial['player'].update(x=1.205, y=2.5, angle=np.pi)
            game.set_state(initial)
        first, image = torch.tensor(game.state_vector()), torch.tensor(game.render(32, overlays=False))
        game.step([1., 0., 0., 0.], dt=1/30)
        rows.append((first, torch.tensor(game.state_vector()), image,
                     torch.tensor(game.render(32, overlays=False)), torch.tensor(game.map)))
    value = torch.stack([torch.stack(row[:2]) for row in rows])
    return {'states': value, 'target_states': value[:, 1:], 'actions': torch.tensor([[[1., 0., 0., 0.]]]*2),
        'frames': torch.stack([torch.stack(row[2:4]) for row in rows]).permute(0, 1, 4, 2, 3).float()/255,
        'map': torch.stack([row[4] for row in rows])}


class ContactObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_rare_blocked_physical_gradient_has_equal_mass_and_inactive_axes_zero(self):
        gradients, values = [], []
        for free_count in (1, 50):
            old = states(free_count+1); old[-1, 0] = 5.795/16
            action = torch.tensor([[1., 0., 0., 0.]]).expand(len(old), -1)
            target = old.clone(); target[:-1, :2] += player_command_delta(old, action)[:-1]
            logits = torch.zeros(len(old), 2, requires_grad=True)
            maps = torch.zeros(len(old), 16, 16, dtype=torch.long); maps[:, 5, 6] = 1
            supervision = prepare_contact_targets(old, target, action, contact_protocol=CONTACT_PROTOCOL_V2)
            self.assertAlmostEqual(float(supervision['sample_weight'][-1]), .5)
            self.assertEqual(float(supervision['axis_weight'][..., 1].sum()), 0.)
            terms = differentiable_contact_terms(prior(old, action, logits), old, target, action, maps,
                contact_protocol=CONTACT_PROTOCOL_V2, supervision=supervision)
            terms['physical_motion'].backward()
            gradients.append(float(logits.grad[-1, 0])); values.append(float(terms['physical_motion'].detach()))
            self.assertTrue(torch.equal(logits.grad[:, 1], torch.zeros(len(old))))
        self.assertAlmostEqual(gradients[0], gradients[1], places=6)
        self.assertAlmostEqual(values[0], values[1], places=6)

    def test_stop_free_and_slide_endpoint_margins_push_correct_directions(self):
        old = states(1); old[:, :2] = torch.tensor([5.5, 5.795])/16
        action = torch.tensor([[1., 1., 0., 0.]])
        command = player_command_delta(old, action)
        target = old.clone(); target[:, 0] += command[:, 0]
        logits = torch.tensor([[-2., 2.]], requires_grad=True)
        grid = torch.zeros(1, 16, 16, dtype=torch.long); grid[:, 6, 5:7] = 1
        labels = prepare_contact_targets(old, target, action, contact_protocol=CONTACT_PROTOCOL_V2)
        self.assertEqual(labels['groups']['slide'], 1)
        terms = differentiable_contact_terms(prior(old, action, logits), old, target, action, grid,
                                             contact_protocol=CONTACT_PROTOCOL_V2)
        terms['endpoint_margin'].backward()
        self.assertLess(float(logits.grad[0, 0]), 0.)
        self.assertGreater(float(logits.grad[0, 1]), 0.)
        self.assertGreater(float(terms['contact_wall'].detach()), 0.)
        idle = torch.zeros_like(action); idle_logits = torch.zeros(1, 2, requires_grad=True)
        empty = differentiable_contact_terms(prior(old, idle, idle_logits), old, old, idle, grid,
                                             contact_protocol=CONTACT_PROTOCOL_V2)
        sum(empty.values()).backward()
        self.assertEqual(float(sum(empty.values()).detach()), 0.)
        self.assertTrue(torch.equal(idle_logits.grad, torch.zeros(1, 2)))

    def test_legal_source_gap005_has_zero_loss_and_gradient_but_crossing_wall_is_penalized(self):
        grid = torch.zeros(1, 16, 16, dtype=torch.long); grid[:, 5, 6] = 1
        source = torch.tensor([[[5.795/16, 5.5/16]]])
        predicted = source.clone().requires_grad_()
        weight = torch.ones(1, 1)
        loss = swept_player_wall_loss(source, predicted, source, source, grid, weight,
                                      contact_protocol=CONTACT_PROTOCOL_V2)
        loss.backward()
        self.assertEqual(float(loss), 0.)
        self.assertTrue(torch.equal(predicted.grad, torch.zeros_like(predicted)))
        for x in (5.805, 6.5):  # tiny overlap, then an exact wall-center stress case
            predicted = torch.tensor([[[x/16, 5.5/16]]], requires_grad=True)
            loss = swept_player_wall_loss(source, predicted, source, source, grid, weight,
                                          contact_protocol=CONTACT_PROTOCOL_V2)
            loss.backward()
            self.assertGreater(float(loss.detach()), 0.)
            self.assertTrue(torch.isfinite(predicted.grad).all())
            self.assertGreater(float(predicted.grad.norm()), 0.)

    def test_swept_geometry_uses_source_x_then_y_legs_instead_of_corner_chord(self):
        grid = torch.zeros(1, 16, 16, dtype=torch.long); grid[:, 6, 6] = 1
        old = torch.tensor([[[6.2, 5.8]]])/16; new = torch.tensor([[[5.8, 6.2]]])/16
        points = _swept(old, new, 5)
        self.assertLess(float(circle_wall_penalty(points, grid, .2).max()), 1e-10)
        chord_midpoint = (old+new)*8  # normalized average converted to tiles
        self.assertGreater(float(circle_wall_penalty(chord_midpoint, grid, .2)), .01)

    def test_probe_preserves_parameters_existing_gradients_rng_modes_and_map_cache(self):
        world = SpyRSSM()
        world.register_buffer('_map_condition', torch.full((16, 16), 7, dtype=torch.long), persistent=False)
        original_encoder = world.encode_map
        def changed_cache(grid):
            world._map_condition = grid.clone()
            return original_encoder(grid)
        world.encode_map = changed_cache
        for parameter in world.parameters():
            if parameter.requires_grad: parameter.grad = torch.ones_like(parameter)*.123
        before = copy.deepcopy(world.state_dict())
        gradients = {name: p.grad.clone() if p.grad is not None else None for name, p in world.named_parameters()}
        modes = [module.training for module in world.modules()]
        cpu_rng, numpy_rng, python_rng = torch.random.get_rng_state(), np.random.get_state(), random.getstate()
        report = contact_gradient_probe(world, mixed_source(), context=1, observed_context=1,
                                        contact_protocol=CONTACT_PROTOCOL_V2)
        self.assertEqual(set(report['components']), {'pass_bce', 'physical_motion', 'endpoint_margin', 'contact_wall'})
        self.assertGreater(report['components']['contact_wall']['unweighted_norm'], 0.)
        self.assertFalse(report['parameter_gradients_mutated'])
        for name, value in world.state_dict().items(): torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        for name, parameter in world.named_parameters():
            if gradients[name] is None: self.assertIsNone(parameter.grad)
            else: torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)
        self.assertTrue(torch.equal(world._map_condition, torch.full((16, 16), 7, dtype=torch.long)))
        self.assertEqual(modes, [module.training for module in world.modules()])
        self.assertTrue(torch.equal(cpu_rng, torch.random.get_rng_state()))
        self.assertTrue(np.array_equal(numpy_rng[1], np.random.get_state()[1]))
        self.assertEqual(python_rng, random.getstate())

    def test_opt_in_rssm_has_one_physical_term_configured_kl_and_keeps_450_prior_unchanged(self):
        original = SpyRSSM(); upgraded = SpyRSSM(); upgraded.load_state_dict(original.state_dict())
        original.config = upgraded.config = SimpleNamespace(rssm_free_nats=.25,
            rssm_dynamics_scale=.4, rssm_representation_scale=.2, rssm_unimix=.02)
        data, aux = long_batch(), mixed_source()
        legacy = rssm_dynamics_backward(original, data, auxiliary_batch=aux, auxiliary_context=1)
        modern = rssm_dynamics_backward(upgraded, data, auxiliary_batch=aux, auxiliary_context=1,
                                        contact_protocol=CONTACT_PROTOCOL_V2)
        self.assertEqual(modern['rssm_contact_protocol_version'], 2)
        self.assertEqual(modern['rssm_aux_raw_delta'], 0.)
        self.assertEqual(modern['rssm_aux_scaled_motion'], 0.)
        self.assertGreater(modern['rssm_aux_physical_motion'], 0.)
        self.assertAlmostEqual(modern['rssm_kl_minimum_floor'], .15)
        self.assertAlmostEqual(modern['train_loss_without_kl_floor'], modern['train_loss']-.15)
        self.assertEqual(modern['rssm_prior_horizon'], 450)
        self.assertEqual(modern['rssm_prior_tbptt_chunks'], 29)
        for old, new in zip(original.calls[:450], upgraded.calls[:450]):
            torch.testing.assert_close(old[0], new[0], rtol=0, atol=0)
            torch.testing.assert_close(old[1], new[1], rtol=0, atol=0)
        self.assertIsNone(upgraded.renderer.weight.grad)
        self.assertIsNone(upgraded.map_encoder.weight.grad)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in upgraded.parameters()))
        self.assertGreater(modern['rssm_aux_physical_missed_block_rate'], 0.)
        with self.assertRaises(ValueError): normalize_contact_protocol({'version': 3})

    def test_autonomous_wall_weight_is_independent_of_source_auxiliary_wall_weight(self):
        first = SpyRSSM(); second = SpyRSSM(); second.load_state_dict(first.state_dict())
        data, aux = source_batch(context=1, horizon=4, batch_size=1), mixed_source()
        initial = normalize_contact_protocol({'version': 2, 'contact_wall_weight': 10.})
        changed = {**initial, 'autonomous_wall_weight': 16.}
        self.assertEqual(initial['autonomous_wall_weight'], 2.)
        # A connected, fixed penalty isolates the multiplier from differing
        # trajectories and from the separate real Source auxiliary geometry.
        def wall_value(old, predicted, *args, **kwargs):
            return predicted.sum()*0+.25
        with patch('turkeyworld.rssm_objectives.swept_player_wall_loss', side_effect=wall_value):
            before = rssm_dynamics_backward(first, data, auxiliary_batch=aux,
                context=1, auxiliary_context=1, expected_horizon=4, contact_protocol=initial)
            after = rssm_dynamics_backward(second, data, auxiliary_batch=aux,
                context=1, auxiliary_context=1, expected_horizon=4, contact_protocol=changed)
        self.assertAlmostEqual(before['rssm_wall_player'], .5)
        self.assertAlmostEqual(after['rssm_wall_player'], 4.)
        self.assertEqual(before['rssm_aux_contact_wall'], after['rssm_aux_contact_wall'])
        self.assertEqual(before['rssm_auxiliary_loss'], after['rssm_auxiliary_loss'])
        self.assertAlmostEqual(after['train_loss']-before['train_loss'], 3.5, places=5)

    def test_autonomous_xy_weight_scales_only_tracking_and_core_eval_keeps_gradients(self):
        original = SpyRSSM(); implicit = SpyRSSM(); amplified = SpyRSSM()
        implicit.load_state_dict(original.state_dict()); amplified.load_state_dict(original.state_dict())
        for world in (original, implicit, amplified):
            world.train(); world.core.eval()
        self.assertEqual(normalize_contact_protocol({'version': 2})['autonomous_xy_weight'], 1.)
        data, aux = source_batch(context=1, horizon=4, batch_size=1), mixed_source()
        # Keep a stable living NPC with nonzero recorded motion so both XY
        # multipliers are tested, independently of the navigation-only data.
        data['states'][:, :, 8:10] = torch.tensor([5., 5.])/16
        data['states'][:, :, 8] += torch.arange(5)*.003
        data['states'][:, :, 12:14] = 1.
        data['target_states'] = data['states'][:, 1:].clone()
        common = dict(auxiliary_batch=aux, context=1, auxiliary_context=1, expected_horizon=4)
        baseline = rssm_dynamics_backward(original, data, contact_protocol={'version': 2}, **common)
        explicit = rssm_dynamics_backward(implicit, data,
            contact_protocol={'version': 2, 'autonomous_xy_weight': 1.}, **common)
        scaled = rssm_dynamics_backward(amplified, data,
            contact_protocol={'version': 2, 'autonomous_xy_weight': 16.}, **common)
        self.assertEqual(baseline, explicit)
        for name in ('rssm_xy_player', 'rssm_xy_npc'):
            self.assertGreater(baseline[name], 0.)
            self.assertAlmostEqual(scaled[name], 16*baseline[name])
        for name in ('rssm_state', 'rssm_wall_player', 'rssm_wall_npc', 'rssm_auxiliary_loss'):
            self.assertEqual(scaled[name], baseline[name])
        added = 15*(baseline['rssm_xy_player']+baseline['rssm_xy_npc'])
        self.assertAlmostEqual(scaled['train_loss']-baseline['train_loss'], added)
        self.assertFalse(amplified.core.training)
        self.assertTrue(amplified.training)
        gradients = [p.grad for p in amplified.core.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertGreater(float(sum(g.abs().sum() for g in gradients)), 0.)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertIsNone(amplified.renderer.weight.grad)
        self.assertIsNone(amplified.map_encoder.weight.grad)


if __name__ == '__main__': unittest.main()
