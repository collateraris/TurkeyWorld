import unittest

import torch
from torch.nn import functional as F

from turkeyworld.aligned_training import motion_error_parts
from turkeyworld.spatial_entities import entity_motion_basis, local_to_world


class LocalMotionLossTests(unittest.TestCase):
    def state(self):
        value = torch.zeros(2, 90)
        value[:, :2] = .25
        value[:, 3] = 1
        value[:, 4] = 1
        entities = value[:, 8:88].reshape(2, 8, 10)
        entities[:, :, :2] = .5
        entities[:, :, 4:6] = 1
        return value

    def test_local_forward_targets_supervise_both_player_and_each_living_npc(self):
        previous = self.state()
        action = torch.tensor([[1., 0., .7, 0.], [1., 0., -.3, 0.]])
        basis = entity_motion_basis(previous, action=action)
        local = torch.zeros(2, 9, 2)
        local[:, :, 0] = .001
        movement = local_to_world(local, basis)
        target = previous.clone()
        target[:, :2] += movement[:, 0]
        target[:, 8:88].reshape(2, 8, 10)[:, :, :2] += movement[:, 1:]
        predicted = previous.clone().requires_grad_()
        player, npc = motion_error_parts(predicted, previous, target, local_motion=True, action=action)
        self.assertGreater(float(player), 0)
        self.assertGreater(float(npc), 0)
        (player+npc).backward()
        self.assertGreater(float(predicted.grad[:, :2].abs().sum()), 0)
        self.assertTrue(bool((predicted.grad[:, 8:88].reshape(2, 8, 10)[:, :, :2].abs().sum(-1) > 0).all()))
        first, second = motion_error_parts(target, previous, target, local_motion=True, action=action)
        self.assertEqual(float(first+second), 0)

    def test_respawn_teleport_is_excluded_without_hiding_player_motion_error(self):
        previous = self.state()
        previous[:, 8:88].reshape(2, 8, 10)[:, :, 5] = 0
        target = previous.clone()
        target[:, 8:88].reshape(2, 8, 10)[:, :, :2] = .9
        target[:, 8:88].reshape(2, 8, 10)[:, :, 5] = 1
        target[:, :2] += .001
        player, npc = motion_error_parts(previous, previous, target, local_motion=True)
        self.assertGreater(float(player), 0)
        self.assertEqual(float(npc), 0)

    def test_legacy_world_axis_objective_keeps_original_scale(self):
        previous = self.state()
        torch.manual_seed(17)
        predicted = previous+torch.randn_like(previous)*.001
        target = previous+torch.randn_like(previous)*.001
        target[:, 8:88].reshape(2, 8, 10)[:, :, 5] = 1
        player, npc = motion_error_parts(predicted, previous, target)
        expected_player = F.smooth_l1_loss((predicted[:, :2]-previous[:, :2])*200,
                                          (target[:, :2]-previous[:, :2])*200)
        old = previous[:, 8:88].reshape(2, 8, 10)
        a = predicted[:, 8:88].reshape(2, 8, 10)
        b = target[:, 8:88].reshape(2, 8, 10)
        expected_npc = F.smooth_l1_loss((a[:, :, :2]-old[:, :, :2])*400,
                                       (b[:, :, :2]-old[:, :, :2])*400)
        torch.testing.assert_close(player, expected_player, rtol=0, atol=0)
        torch.testing.assert_close(npc, expected_npc, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
