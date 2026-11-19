import unittest

import torch

from turkeyworld.combat_appearance_sr import CombatAppearanceSR


class CombatAppearanceTests(unittest.TestCase):
    def test_correction_stays_inside_visible_combat_projection(self):
        torch.set_num_threads(2)
        model = CombatAppearanceSR().eval()
        low = torch.zeros(3, 3, 32, 32)
        scene = torch.full((3, 3, 128, 128), 0.1)
        state = torch.zeros(3, 20)
        state[:, 3] = 1
        state[:, 13] = 1
        state[:, 8] = 0.25  # Four tiles in front of the player.
        state[1:, 15] = 1  # Hurt state; the first enemy stays idle.
        state[2, 8] = -0.25

        with torch.inference_mode():
            model.decoder[-1].bias[:3].fill_(0.2)
            result = model(low, scene, state)
            self.assertTrue(torch.equal(model(low, scene, None), scene))
        self.assertTrue(torch.equal(result[0], scene[0]))
        self.assertTrue(torch.equal(result[2], scene[2]))
        self.assertTrue(torch.equal(result[1, :, 0, 0], scene[1, :, 0, 0]))
        self.assertTrue(torch.allclose(result[1, :, 64, 64], torch.full((3,), 0.3)))
