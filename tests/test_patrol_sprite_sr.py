import unittest
import torch
from turkeyworld.patrol_sprite_sr import patrol_alpha

class PatrolSpriteTests(unittest.TestCase):
    def test_only_idle_and_walk_can_be_composited(self):
        alpha=torch.ones(7,1,4,4)
        actual=patrol_alpha(alpha,torch.arange(7))
        self.assertTrue(torch.equal(actual[:3],alpha[:3]))
        self.assertEqual(int(actual[3:].count_nonzero()),0)

    def test_navigation_disables_even_a_confident_visible_enemy(self):
        alpha=torch.ones(3,1,4,4)
        self.assertEqual(int(patrol_alpha(alpha,torch.arange(3),enemies_enabled=False).count_nonzero()),0)

    def test_predicted_combat_or_death_vetoes_rgb_idle_classification(self):
        state=torch.zeros(5,20);state[:,13]=1
        state[1,14]=1;state[2,15]=.1;state[3,13]=0
        actual=patrol_alpha(torch.ones(5,1,4,4),torch.zeros(5,dtype=torch.long),predicted_state=state)
        self.assertEqual(int(actual[1:4].count_nonzero()),0)
        self.assertTrue(bool((actual[[0,4]]==1).all()))
