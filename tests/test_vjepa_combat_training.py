import unittest
import torch

from turkeyworld.vjepa_combat_training import combat_npc_motion, combat_objective, training_profile


class ThreeModeTrainingTests(unittest.TestCase):
    def test_profile_resume(self):
        self.assertEqual(training_profile(None, {}), 'motion_only')
        self.assertEqual(training_profile('three_mode', {}), 'three_mode')
        self.assertEqual(training_profile(None, {'training_profile': 'three_mode'}), 'three_mode')
        with self.assertRaises(ValueError):
            training_profile('motion_only', {'training_profile': 'three_mode'})

    def test_heads_receive_gradients_on_each_eligible_decision(self):
        state = torch.zeros(1, 4, 20)
        state[..., 4] = 1
        state[..., 12] = .6
        state[..., 13] = 1
        ev = torch.tensor([[[1., 1, 0, 0], [1, 0, 0, 1], [0, 0, 0, 0]]])
        targets = state[:, 1:].clone()
        targets[:, :, 12] = .25
        targets[:, 1:, 4] = .92
        batch = dict(states=state, target_states=targets, events=ev, hit_ids=torch.tensor([[0, -1, -1]]),
                     attack_flags=ev[:, :, 3:4])
        target = torch.zeros(1, 3, 2, requires_grad=True)
        attack = torch.zeros(1, 3, 1, requires_grad=True)
        probabilities = torch.ones((1, 3, 4), requires_grad=True)
        terms = combat_objective(target, attack, probabilities, batch, targets)
        terms['combat_total'].backward()
        self.assertLess(target.grad[0, 0, 0], 0)
        self.assertGreater(target.grad[0, 1, 0], 0)
        self.assertEqual(target.grad[0, 2].abs().sum(), 0)
        self.assertLess(attack.grad[0, 1, 0], 0)
        self.assertGreater(attack.grad[0, 0, 0], 0)
        self.assertEqual(terms['hp_mae'], 0)

    def test_respawn_teleport_is_excluded_from_motion(self):
        source = torch.zeros(1, 4, 20)
        source[..., 4] = 1
        source[:, 1:, 8:10] = .7
        source[:, 1:, 13] = 1
        pred = source.clone()
        pred[:, 0, 8:10] = .5
        init = torch.zeros(1, 20)
        grid = torch.zeros(1, 16, 16, dtype=torch.uint8)
        # Both teleport to the same point from different corpse positions.
        self.assertEqual(combat_npc_motion(pred, source, init, grid, .25), 0)
        pred[:, 2:, 8] += .02
        self.assertGreater(combat_npc_motion(pred, source, init, grid, .25), 0)
