"""Combat labels supervise own legal probabilities; HP never supplies a history ledger."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
import unittest

import torch
from torch.nn import functional as F

from turkeyworld.combat_objectives import (
    COMBAT_COUNT_FEATURE_DIM, COMBAT_FEATURE_DIM, COMBAT_COUNT_FEATURE_SLICES,
    combat_count_features, combat_image_state, combat_transition_losses)
from turkeyworld.structured_state import structured_transition


def state(batch=1):
    value = torch.zeros(batch, STATE_DIM)
    value[:, :2] = .4
    value[:, 3:6] = 1
    npc = value[:, GLOBAL_DIM:-2].reshape(batch, ENTITY_SLOTS, ENTITY_DIM)
    npc[:, :, :2] = .8
    npc[:, :, 4] = .6
    npc[:, :, 5] = 1
    return value


def transition(previous, *, chosen=ENTITY_SLOTS, fire=True, attacking=False, near=False):
    targets = torch.full((len(previous), ENTITY_SLOTS+1), -2.)
    targets[:, chosen] = 2.
    targets.requires_grad_()
    attacks = torch.full((len(previous), ENTITY_SLOTS), -2.)
    if attacking:
        attacks[:, 0] = 2.
    attacks.requires_grad_()
    action = previous.new_zeros((len(previous), 4))
    action[:, 3] = float(fire)
    result = structured_transition(previous, previous, action, targets, attacks,
                                   training=True, endless=True)
    return result, targets, attacks, action


def losses(result, previous, target=None, hits=None, attacks=None, events=None):
    if target is None:
        target = result['state'].detach()
    if hits is None:
        hits = result['accepted_target'].detach()
    if attacks is None:
        attacks = result['accepted_attacks'].float().detach()
    if events is None:
        events = result['events'].detach()
    return combat_transition_losses(result, previous, target, hits, attacks, events)


class CombatCountTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_nominal_source_ladders_death_and_respawn(self):
        previous = state(4)
        previous[:, 4] = torch.tensor([1., .92, .04, 0.])
        npc = previous[:, GLOBAL_DIM:-2].reshape(4, ENTITY_SLOTS, ENTITY_DIM)
        npc[:, 0, 4] = torch.tensor([.6, .25, 0., .6])
        npc[2, 0, 5] = 0
        snapshot = previous.clone()
        features = combat_count_features(previous)
        self.assertEqual(features.shape, (4, ENTITY_SLOTS+1, COMBAT_COUNT_FEATURE_DIM))
        self.assertEqual(COMBAT_FEATURE_DIM, 6)
        self.assertEqual(COMBAT_COUNT_FEATURE_SLICES['hp_ladder_valid'], slice(4, 5))
        torch.testing.assert_close(features[:, 0, 0], torch.tensor([0., 1., 12., 13.]))
        torch.testing.assert_close(features[:, 1, 0], torch.tensor([0., 1., 2., 0.]))
        torch.testing.assert_close(features[:, 0, 3], torch.tensor([13., 12., 1., 0.]))
        self.assertTrue(bool(features[:, :, 4].all()))
        self.assertEqual(float(features[2, 1, 1]), 1.)
        torch.testing.assert_close(snapshot, previous, rtol=0, atol=0)

    def test_augmented_hp_is_flagged_and_nominal_counts_have_hp_gradients(self):
        previous = state()
        previous[0, 12] = .35  # Source's forced-kill augmentation is off the60-35n ladder.
        previous.requires_grad_()
        features = combat_count_features(previous)
        self.assertEqual(float(features[0, 1, 4].detach()), 0.)
        self.assertEqual(float(features[0, 1, 0].detach()), 1.)
        self.assertTrue(bool(torch.isfinite(features).all()))
        features[..., [0, 2]].sum().backward()
        self.assertLess(float(previous.grad[0, 12]), 0.)
        self.assertEqual(float(previous.grad[0, 13]), 0.)
        with self.assertRaises(ValueError):
            combat_count_features(torch.zeros(2, STATE_DIM-1))


class CombatTransitionObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_old_target_none_weight_attack_pos_weight_and_events_are_exact(self):
        from turkeyworld.train import supervised_event_losses
        previous = state(2)
        previous[1, 8:10] = previous[1, :2]+.02
        result, _, _, _ = transition(previous, chosen=0, attacking=True)
        hits = torch.tensor([-1, 0])
        flags = torch.zeros(2, ENTITY_SLOTS); flags[1, 0] = 1
        events = torch.tensor([[1., 0., 0., 0.], [1., 1., 0., 1.]])
        before = supervised_event_losses(result, hits, flags, events)
        after = losses(result, previous, hits=hits, attacks=flags, events=events)
        self.assertEqual(set(after), {'events', 'target', 'attack', 'damage_npc', 'damage_player', 'death_npc'})
        for key in before:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)

    def test_lethal_hp_clamp_has_no_gradient_but_death_and_damage_are_learnable(self):
        previous = state()
        previous[0, 12] = .25
        result, targets, _, _ = transition(previous, chosen=0)
        self.assertEqual(float(result['state'][0, 12]), 0.)
        old_gradient = torch.autograd.grad(result['state'][0, 12], targets, retain_graph=True)[0]
        self.assertEqual(float(old_gradient.abs().sum()), 0.)
        # Recorded shot missed (e.g. wall occlusion), despite the wrong own lethal decision.
        target = previous.clone()
        objective = losses(result, previous, target=target, hits=torch.tensor([-1]),
                           attacks=torch.zeros(1, ENTITY_SLOTS), events=torch.tensor([[1., 0., 0., 0.]]))
        for key in ('damage_npc', 'death_npc'):
            gradient = torch.autograd.grad(objective[key], targets, retain_graph=True)[0]
            self.assertGreater(float(gradient[0, 0]), 0., key)
            self.assertTrue(bool(torch.isfinite(gradient).all()))

    def test_wrong_none_prediction_receives_positive_lethal_hit_supervision(self):
        previous = state(); previous[0, 12] = .25
        result, targets, _, _ = transition(previous, chosen=ENTITY_SLOTS)
        target = previous.clone(); target[0, 12:14] = 0
        objective = losses(result, previous, target=target, hits=torch.tensor([0]),
                           attacks=torch.zeros(1, ENTITY_SLOTS), events=torch.tensor([[1., 1., 1., 0.]]))
        (objective['damage_npc']+objective['death_npc']).backward()
        self.assertLess(float(targets.grad[0, 0]), 0.)
        self.assertGreater(float(targets.grad[0, ENTITY_SLOTS]), 0.)

    def test_player_overkill_expectation_survives_clamped_hp_and_occluded_negative(self):
        previous = state(2); previous[:, 4] = .04
        previous[0, 8:10] = previous[0, :2]+.02
        result, _, attacks, _ = transition(previous, fire=False, attacking=True)
        self.assertEqual(float(result['state'][0, 4]), 0.)
        old_gradient = torch.autograd.grad(result['state'][0, 4], attacks, retain_graph=True)[0]
        self.assertEqual(float(old_gradient.abs().sum()), 0.)
        # The nearby source NPC is occluded; target labels teach the attack head to abstain.
        objective = losses(result, previous, target=previous, hits=torch.full((2,), -1),
                           attacks=torch.zeros(2, ENTITY_SLOTS), events=torch.zeros(2, 4))
        gradient = torch.autograd.grad(objective['damage_player'], attacks)[0]
        self.assertGreater(float(gradient[0, 0]), 0.)
        # Use a second query with the same single ID out of range, rather than
        # an empty assertion over removed extra NPC slots.
        torch.testing.assert_close(gradient[1], torch.zeros(ENTITY_SLOTS), rtol=0, atol=0)
        probability = attacks.detach()[0, 0].sigmoid()
        expected = F.smooth_l1_loss(probability*.04/.08, torch.tensor(0.))/2
        torch.testing.assert_close(objective['damage_player'], expected)

    def test_prior_dead_respawn_no_fire_and_player_terminal_mask_new_objectives(self):
        previous = state()
        npc = previous[:, GLOBAL_DIM:-2].reshape(1, ENTITY_SLOTS, ENTITY_DIM)
        npc[:, :, 4:6] = 0
        npc[:, :, 8] = 5
        target_logits = torch.zeros(1, ENTITY_SLOTS+1, requires_grad=True)
        attack_logits = torch.zeros(1, ENTITY_SLOTS, requires_grad=True)
        result = structured_transition(previous, previous, torch.zeros(1, 4),
            target_logits, attack_logits, training=True, endless=True,
            spawn_positions=torch.full((1, ENTITY_SLOTS, 2), .8), spawn_valid=torch.ones(1, ENTITY_SLOTS, dtype=torch.bool))
        self.assertTrue(bool(result['respawn_flags'].all()))
        objective = losses(result, previous)
        for name in ('damage_npc', 'damage_player', 'death_npc'):
            self.assertEqual(float(objective[name]), 0., name)
        no_fire = state(); no_fire[0, 12] = .25
        result, _, _, _ = transition(no_fire, chosen=0, fire=False)
        self.assertEqual(float(losses(result, no_fire)['death_npc']), 0.)
        terminal = state(); terminal[:, 4] = 0
        terminal[:, 8:10] = terminal[:, :2]
        result, _, _, _ = transition(terminal, attacking=True)
        self.assertEqual(float(losses(result, terminal)['damage_player']), 0.)

    def test_shot_gate_uses_own_controls_never_recorded_shot_events(self):
        previous = state(); previous[0, 12] = .25
        result, _, _, action = transition(previous, chosen=0, fire=False)
        own = {key: value for key, value in result.items() if key not in ('shot', 'events')}
        target = previous.clone(); target[0, 12:14] = 0
        objective = combat_transition_losses(own, previous, target, torch.tensor([0]),
            torch.zeros(1, ENTITY_SLOTS), torch.ones(1, 4), action=action)
        self.assertEqual(float(objective['death_npc']), 0.)
        self.assertAlmostEqual(float(objective['damage_npc']), .5*(.25/.35)**2/ENTITY_SLOTS, places=7)
        with self.assertRaises(ValueError):
            combat_transition_losses(own, previous, target, torch.tensor([0]),
                                     torch.zeros(1, ENTITY_SLOTS), torch.ones(1, 4))


class CombatImageConditionTests(unittest.TestCase):
    def test_own_hard_combat_forward_gt_pose_and_only_combat_logit_gradients(self):
        previous = state(); previous[0, 12] = .25
        previous.requires_grad_()
        result, targets, attacks, action = transition(previous, chosen=0)
        pose = state(); pose[:, :2] = .55
        pose[:, 8:12] = torch.tensor([.2, .3, .1, -.1])
        pose.requires_grad_()
        condition = combat_image_state(result, previous, pose, action=action)
        hard = result['state'].detach()
        torch.testing.assert_close(condition[:, :4], pose[:, :4], rtol=0, atol=0)
        for index in (4, 7, 12, 13, 14, 15, 16):
            torch.testing.assert_close(condition[:, index], hard[:, index], rtol=0, atol=0)
        torch.testing.assert_close(condition[:, 8:12], pose[:, 8:12], rtol=0, atol=0)
        self.assertEqual(float(condition[0, 13]), 0.)
        # Distinct derivatives avoid accidental HP/life/hurt cancellation.
        (condition[0, 12]+2*condition[0, 13]+.3*condition[0, 15]).backward()
        self.assertGreater(float(targets.grad.abs().sum()), 0.)
        self.assertIsNone(previous.grad)
        self.assertIsNone(pose.grad)
        self.assertTrue(attacks.grad is None or not bool(attacks.grad.any()))

    def test_clamped_player_hp_and_hp_derived_counts_remain_image_differentiable(self):
        previous = state(); previous[0, 4] = .04
        previous[0, 8:10] = previous[0, :2]+.02
        result, targets, attacks, _ = transition(previous, fire=False, attacking=True)
        condition = combat_image_state(result, previous, previous)
        self.assertEqual(float(condition[0, 4]), 0.)
        features = combat_count_features(condition)
        features[0, 0, 2].backward()
        self.assertGreater(float(attacks.grad[0, 0]), 0.)
        self.assertTrue(targets.grad is None or not bool(targets.grad.any()))

    def test_respawned_slot_is_physical_and_has_no_surrogate_damage_gradient(self):
        previous = state(); previous[:, 12:14] = 0; previous[:, 16] = 5
        targets = torch.zeros(1, ENTITY_SLOTS+1, requires_grad=True)
        attacks = torch.zeros(1, ENTITY_SLOTS, requires_grad=True)
        result = structured_transition(previous, previous, torch.zeros(1, 4), targets, attacks,
            training=True, endless=True, spawn_positions=torch.full((1, ENTITY_SLOTS, 2), .8),
            spawn_valid=torch.ones(1, ENTITY_SLOTS, dtype=torch.bool))
        condition = combat_image_state(result, previous, previous)
        self.assertEqual(float(condition[0, 12]), float(result['state'][0, 12]))
        self.assertEqual(float(condition[0, 13]), 1.)
        gradient = torch.autograd.grad(condition[0, 12:17].sum(), targets, allow_unused=True)[0]
        self.assertTrue(gradient is None or not bool(gradient.any()))


if __name__ == '__main__':
    unittest.main()
