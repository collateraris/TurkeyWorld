"""V5 fixes spatial/event loss scales while preserving the independent AR stream."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
import copy
import math
from types import MethodType
import unittest

import torch
from torch.nn import functional as F

from turkeyworld.aligned_training import (aggregate_spawn_supervision, spawn_supervision_statistics,
                                          aligned_short_rollout_loss)
from turkeyworld.combat_objectives import (aggregate_balanced_attack, balanced_attack_loss,
                                          combat_transition_losses)
from turkeyworld.trajectory_objectives import circle_wall_penalty, trajectory_state_losses


def state(batch=1):
    value = torch.zeros(batch, STATE_DIM)
    value[:, :2] = 4.5/16
    value[:, 3:6] = 1
    npc = value[:, GLOBAL_DIM:-2].reshape(batch, ENTITY_SLOTS, ENTITY_DIM)
    npc[:, :, :2] = 9.5/16
    npc[:, :, 4] = .6; npc[:, :, 5] = 1
    return value


class TrajectoryV5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_xy_uses_tile_units_selected_offsets_and_keeps_false_missing_npc(self):
        initial = state().requires_grad_()
        target = initial.detach()[:, None].repeat(1, 16, 1).requires_grad_()
        predicted = target.detach().clone()
        predicted[:, :, 8] += 1/16
        predicted[:, :, 12:14] = 0  # Wrong predicted death cannot hide positional error.
        predicted.requires_grad_()
        before = predicted.detach().clone()
        parts, metrics = trajectory_state_losses(predicted, initial, target, torch.zeros(1, 16, 16, dtype=torch.long))
        self.assertEqual(metrics['ar_xy_offsets'], [1, 4, 8, 16])
        self.assertAlmostEqual(float(parts['ar_xy_npc'].detach()), .5/(ENTITY_SLOTS*2))
        self.assertEqual(metrics['ar_xy_npc_pairs'], 4*ENTITY_SLOTS)
        parts['ar_xy_npc'].backward()
        for offset in (0, 3, 7, 15):
            self.assertGreater(float(predicted.grad[0, offset, 8]), 0.)
        self.assertEqual(float(predicted.grad[0, 1, 8]), 0.)
        self.assertEqual(float(predicted.grad[..., 13].abs().sum()), 0.)
        self.assertIsNone(target.grad); self.assertIsNone(initial.grad)
        torch.testing.assert_close(predicted.detach(), before, rtol=0, atol=0)

    def test_source_respawn_excludes_new_life_but_preserves_independent_continuing_life(self):
        initial = state(2); initial[0, 12:14] = 0
        targets = initial[:, None].repeat(1, 16, 1)
        targets[0, 3:, 12:14] = torch.tensor([.6, 1.])
        targets[0, 3:, 8:10] = 13.5/16
        predicted = targets.clone(); predicted[:, :, 8] += 1/16
        predicted.requires_grad_()
        parts, metrics = trajectory_state_losses(predicted, initial, targets, torch.zeros(2, 16, 16, dtype=torch.long))
        parts['ar_xy_npc'].backward()
        self.assertEqual(metrics['ar_source_respawn_exclusions'], 1)
        self.assertEqual(float(predicted.grad[0, :, 8:10].abs().sum()), 0.)
        self.assertGreater(float(predicted.grad[1, :, 8:10].abs().sum()), 0.)

    def test_circle_clearance_walls_radius_boundary_and_inside_center_gradients(self):
        grid = torch.zeros(1, 16, 16, dtype=torch.long); grid[0, 7, 7] = 2
        points = torch.tensor([[[6.5, 7.5], [6.9, 7.5], [7.2, 7.5], [7.5, 7.5], [.1, 4.]]], requires_grad=True)
        penalty = circle_wall_penalty(points, grid, .25)
        self.assertEqual(float(penalty[0, 0]), 0.)
        self.assertTrue((penalty[0, 1:] > 0).all())
        penalty.sum().backward()
        self.assertGreater(float(points.grad[0, 1, 0]), 0.)  # SGD moves left out of wall.
        self.assertGreater(float(points.grad[0, 2, 0]), 0.)
        self.assertGreater(float(points.grad[0, 3].abs().sum()), 0.)
        self.assertLess(float(points.grad[0, 4, 0]), 0.)  # SGD moves away from arena edge.
        self.assertTrue(torch.isfinite(points.grad).all())

    def test_segments_catch_tunneling_and_source_alive_prevents_life_flag_hiding(self):
        initial = state(); initial[:, :2] = torch.tensor([6.5/16, 7.5/16])
        targets = initial[:, None].clone(); targets[:, :, 0] = 8.5/16
        predicted = targets.clone(); predicted[:, :, 4] = 0  # Wrong death cannot disable wall loss.
        predicted.requires_grad_()
        grid = torch.zeros(1, 16, 16, dtype=torch.long); grid[:, 7, 7] = 1
        endpoints = circle_wall_penalty(predicted[:, :, :2]*16, grid, .2)
        self.assertEqual(float(endpoints.sum()), 0.)
        parts, _ = trajectory_state_losses(predicted, initial, targets, grid)
        self.assertGreater(float(parts['ar_wall_player']), 0.)
        parts['ar_wall_player'].backward()
        self.assertGreater(float(predicted.grad[:, :, :2].abs().sum()), 0.)
        self.assertEqual(float(predicted.grad[:, :, 4].abs().sum()), 0.)

    def test_real_source_legal_axis_separated_corner_slide_has_no_wall_loss(self):
        from turkeyworld.engine import Game
        game = Game()
        snapshot = game.get_state()
        snapshot['player'].update(x=4-.20001*math.cos(math.radians(50)),
                                   y=3-.20001*math.sin(math.radians(50)), angle=0.)
        game.set_state(snapshot)
        previous = torch.from_numpy(game.state_vector().copy())[None].float()
        game.step([-1., 1., 0., 0.])
        current = torch.from_numpy(game.state_vector().copy())[None, None].float()
        grid = torch.from_numpy(game.map.copy())[None]
        parts, metrics = trajectory_state_losses(current, previous, current, grid)
        self.assertLess(float(parts['ar_wall_player']), 1e-10)
        self.assertLess(float(parts['ar_wall_npc']), 1e-10)
        self.assertEqual(metrics['ar_wall_path'], 'x_then_y')


class AttackAndSpawnV5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_balanced_attack_positive_gradient_not_diluted_by_h_or_irrelevant_slots(self):
        logits = torch.zeros(2, ENTITY_SLOTS, requires_grad=True)
        labels = torch.zeros(2, ENTITY_SLOTS); labels[0, 0] = 1
        eligible = torch.ones(2, ENTITY_SLOTS, dtype=torch.bool)
        loss, event = balanced_attack_loss(logits, labels, eligible)
        blank_logits = torch.zeros(1, ENTITY_SLOTS, requires_grad=True)
        _, blank = balanced_attack_loss(blank_logits, torch.zeros(1, ENTITY_SLOTS), torch.zeros(1, ENTITY_SLOTS, dtype=torch.bool))
        aggregate, counts = aggregate_balanced_attack([event]+[blank]*15)
        torch.testing.assert_close(aggregate, loss, rtol=0, atol=0)
        self.assertEqual(float(counts['attack_positive_count']), 1.)
        self.assertEqual(float(counts['attack_negative_count']), 1.)
        aggregate.backward()
        self.assertAlmostEqual(float(logits.grad[0, 0]), -.4)
        self.assertAlmostEqual(float(logits.grad[1, 0]), .1)
        self.assertTrue(blank_logits.grad is None or not bool(blank_logits.grad.any()))

    def test_source_near_occluded_negative_and_positive_survive_wrong_predicted_pose_life(self):
        previous = state(2); previous[:, 8:10] = previous[:, :2]+.02
        target = previous.clone()
        wrong = previous.clone(); wrong[:, :2] = .9
        wrong[:, GLOBAL_DIM:-2].reshape(2, ENTITY_SLOTS, ENTITY_DIM)[:, :, 4:6] = 0
        target_logits = torch.zeros(2, (ENTITY_SLOTS+1), requires_grad=True)
        attack_logits = torch.zeros(2, ENTITY_SLOTS, requires_grad=True)
        flags = torch.zeros(2, ENTITY_SLOTS); flags[0, 0] = 1
        result = {'state': wrong, 'target_logits': target_logits, 'attack_logits': attack_logits, 'shot': torch.tensor([False, False])}
        values = combat_transition_losses(result, previous, target, torch.tensor([-1, -1]), flags,
                                           torch.zeros(2, 4), game_training_v5=True)
        self.assertEqual(float(values['attack_positive_count']), 1.)
        self.assertEqual(float(values['attack_negative_count']), 1.)
        values['attack'].backward()
        self.assertLess(float(attack_logits.grad[0, 0]), 0.)
        self.assertGreater(float(attack_logits.grad[1, 0]), 0.)
        # Positive labels remain included even if source guard is malformed.
        scalar, stats = balanced_attack_loss(torch.zeros(1, ENTITY_SLOTS), flags[:1], torch.zeros(1, ENTITY_SLOTS, dtype=torch.bool))
        self.assertEqual(float(stats['attack_positive_count']), 1.)
        self.assertAlmostEqual(float(scalar), math.log(2), places=6)

    def test_global_spawn_event_ce_does_not_shrink_with_empty_offsets(self):
        previous = state(); previous[:, 12:14] = 0
        target = previous.clone(); target[:, 12:14] = torch.tensor([.6, 1.])
        logits = torch.zeros(1, ENTITY_SLOTS, 256, requires_grad=True)
        result = {'state': target, 'spawn_logits': logits}
        event = spawn_supervision_statistics(result, previous, target)
        empty = spawn_supervision_statistics(result, target, target)
        loss, diagnostics = aggregate_spawn_supervision([event]+[empty]*15)
        self.assertAlmostEqual(float(loss), math.log(256), places=6)
        self.assertEqual(float(diagnostics['spawn_event_count']), 1.)
        loss.backward()
        self.assertGreater(float(logits.grad[0, 0].abs().sum()), 0.)
        self.assertTrue((logits.grad[0, 1:] == 0).all())
        zero, _ = aggregate_spawn_supervision([empty]*4)
        self.assertEqual(float(zero), 0.)
        self.assertTrue(torch.isfinite(zero))

    def test_hierarchical_spawn_anchor_and_conditional_fallback_are_not_double_counted(self):
        previous = state(); previous[:, 12:14] = 0
        target = previous.clone(); target[:, 12:14] = torch.tensor([.6, 1.])
        effective = torch.randn(1, ENTITY_SLOTS, 256, requires_grad=True)
        anchor = torch.zeros(1, ENTITY_SLOTS, 9, requires_grad=True)
        fallback = torch.zeros(1, ENTITY_SLOTS, 256, requires_grad=True)
        result = {'state': target, 'spawn_logits': effective, 'spawn_anchor_logits': anchor,
                  'spawn_fallback_logits': fallback, 'spawn_anchor_legal': torch.zeros(1, ENTITY_SLOTS, 9, dtype=torch.bool)}
        target[:, 8:10] = 2.5/16  # Known anchor0, even if own legal metadata says blocked.
        event = spawn_supervision_statistics(result, previous, target)
        loss, _ = aggregate_spawn_supervision([event])
        self.assertAlmostEqual(float(loss), math.log(9), places=6)
        gradient = torch.autograd.grad(loss, effective, allow_unused=True, retain_graph=True)[0]
        self.assertTrue(gradient is None or not bool(gradient.any()))
        self.assertEqual(float(event['spawn_fallback_sum']), 0.)
        target[:, 8:10] = torch.tensor([5.5/16, 5.5/16])  # A non-anchor floor target.
        event = spawn_supervision_statistics(result, previous, target)
        loss, diagnostics = aggregate_spawn_supervision([event])
        self.assertAlmostEqual(float(loss), math.log(9)+math.log(248), places=5)
        loss.backward()
        self.assertLess(float(anchor.grad[0, 0, 8]), 0.)
        self.assertLess(float(fallback.grad[0, 0, 5*16+5]), 0.)
        self.assertEqual(float(fallback.grad[0, 0, 2*16+2]), 0.)
        self.assertEqual(float(diagnostics['spawn_event_count']), 1.)


class AlignedV5IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_v5_future_labels_change_losses_only_and_diagnostics_are_not_summed(self):
        from test_aligned_training import AlignedTrainingTests
        from turkeyworld.jepa_training_utils import EMATargetEncoder
        fixture = AlignedTrainingTests()
        model, batch = fixture.model(), fixture.batch()
        model.config.game_training_v5 = True
        model.config.spatial_combat = True
        original_transition = model.transition
        def transition(self, previous, action, maps, memory):
            result = original_transition(previous, action, maps, memory)
            pooled = result['latent_prediction'].mean(1)
            result['shot'] = action[:, 3] > .5
            result['combat_context'] = {'state': previous, 'pooled': pooled,
                                       'spatial': pooled[:, None].expand(-1, 9, -1)}
            return result
        model.transition = MethodType(transition, model)
        original_flow = model.flow_renderer.loss
        masks_seen = []
        def flow(self, target, base, state, **options):
            masks_seen.append(options['object_mask'].detach().clone())
            self.assert_unused = options
            return original_flow(target, base, state)
        model.flow_renderer.loss = MethodType(flow, model.flow_renderer)
        batch['target_object_masks'] = torch.zeros(1, 4, 1, 16, 16)
        batch['target_object_masks'][:, :, :, 2:6, 2:6] = 1
        batch['target_background_frames'] = torch.zeros_like(batch['target_frames'])
        ema = EMATargetEncoder(model.core.encoder)
        loss, metrics, first = aligned_short_rollout_loss(model, ema, batch)
        self.assertTrue(torch.isfinite(loss))
        primary_inputs = [value.clone() for value in model.state_inputs[:4]]
        self.assertTrue(torch.equal(masks_seen[-1], batch['target_object_masks'][:, -1]))
        combat = sum(metrics['combat_'+name] for name in ('events', 'target', 'attack', 'damage_npc', 'damage_player', 'death_npc'))
        self.assertAlmostEqual(metrics['combat'], combat, places=5)
        parts = ('regularizer', 'state', 'jepa', 'ar_xy_player', 'ar_xy_npc', 'ar_wall_player', 'ar_wall_npc',
                 'combat', 'motion', 'spawn', 'aux_state', 'aux_jepa', 'base_image', 'foreground_base', 'flow')
        self.assertAlmostEqual(float(loss), sum(metrics[name] for name in parts), places=4)
        changed = {key: value.clone() for key, value in batch.items()}
        changed['states'][:, 2:, 8] += .25
        changed['previous_states'] = changed['states'][:, 1:-1]
        changed['target_states'] = changed['states'][:, 2:]
        changed['frames'][:, 2:] = .95; changed['target_frames'][:] = .95
        model.state_inputs.clear()
        _, _, second = aligned_short_rollout_loss(model, ema, changed)
        for old, new in zip(primary_inputs, model.state_inputs[:4]):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
        torch.testing.assert_close(first['ar_state'], second['ar_state'], rtol=0, atol=0)
        torch.testing.assert_close(first['ar_memory'], second['ar_memory'], rtol=0, atol=0)
        loss.backward()
        self.assertGreater(float(model.pose.weight.grad.abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
