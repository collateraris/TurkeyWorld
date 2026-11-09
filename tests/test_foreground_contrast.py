import unittest

import torch

from turkeyworld.object_losses import foreground_contrastive_loss


class ForegroundContrastTests(unittest.TestCase):
    def test_empty_scene_is_penalized_and_gradient_points_toward_real_npc(self):
        target = torch.zeros(1, 3, 4, 4, requires_grad=True)
        target.data[:, :, 1:3, 1:3] = 1
        background = torch.zeros_like(target, requires_grad=True)
        mask = torch.zeros(1, 4, 4);mask[:, 1:3, 1:3] = 1
        predicted = torch.zeros_like(target, requires_grad=True)
        loss = foreground_contrastive_loss(predicted, target, background, mask)
        self.assertAlmostEqual(float(loss.detach()), 1.25)
        loss.backward()
        self.assertTrue(bool((predicted.grad[:, :, 1:3, 1:3] < 0).all()))
        self.assertEqual(float(predicted.grad[:, :, 0].abs().sum()), 0)
        self.assertIsNone(target.grad);self.assertIsNone(background.grad)
        self.assertEqual(float(foreground_contrastive_loss(target, target, background, mask)), 0)

    def test_absent_npcs_contribute_zero_without_nan_or_gradient(self):
        predicted = torch.rand(2, 3, 8, 8, requires_grad=True)
        target, background = torch.rand_like(predicted), torch.rand_like(predicted)
        loss = foreground_contrastive_loss(predicted, target, background, torch.zeros(2, 8, 8))
        self.assertEqual(float(loss.detach()), 0)
        loss.backward();self.assertEqual(float(predicted.grad.abs().sum()), 0)

    def test_background_changes_outside_npc_mask_do_not_change_loss(self):
        predicted = torch.full((1, 3, 5, 5), .2)
        target = torch.ones_like(predicted)
        background = torch.zeros_like(predicted)
        mask = torch.zeros(1, 5, 5);mask[:, 2, 2] = 1
        original = foreground_contrastive_loss(predicted, target, background, mask)
        background[:, :, 0] = .9;target[:, :, 0] = .1
        torch.testing.assert_close(original, foreground_contrastive_loss(predicted, target, background, mask), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
