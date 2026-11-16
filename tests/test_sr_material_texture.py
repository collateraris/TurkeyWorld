import unittest
from unittest.mock import patch
import torch
from turkeyworld.engine import Game
from turkeyworld.sr_material_texture import material_labels,texture_loss

class MaterialTextureTests(unittest.TestCase):
    def test_material_strata_are_disjoint_and_exclude_npc(self):
        game=Game(seed=0)
        batch=dict(map=torch.tensor(game.map)[None],state=torch.tensor(game.state_vector())[None],mask_hr=torch.zeros(1,1,128,128))
        batch['mask_hr'][:,:,40:75,40:75]=1
        with patch.object(Game,'render',side_effect=AssertionError('labels must not render')):
            labels=material_labels(batch)
        self.assertTrue(all(v.shape==(1,1,128,128) for v in labels.values()))
        self.assertFalse(bool((labels['brick']*labels['earth']).any()))
        self.assertFalse(bool((labels['grass']*labels['earth']).any()))
        self.assertTrue(bool((labels['mortar']<=labels['brick']).all()))
        self.assertFalse(any(bool((v*batch['mask_hr']).any()) for v in labels.values()))

    def test_texture_loss_detects_blur_and_preserves_gradient(self):
        target=torch.zeros(1,3,128,128);target[:,:,:,::4]=1
        image=torch.nn.functional.avg_pool2d(target,3,1,1).detach().requires_grad_()
        labels={k:torch.ones(1,1,128,128) for k in ('brick','mortar','earth','grass','grass_edge')}
        zero,_=texture_loss(target,target,labels);self.assertEqual(float(zero),0)
        loss,metrics=texture_loss(image,target,labels);loss.backward()
        self.assertGreater(metrics['grass_edge_high_frequency'],0)
        self.assertGreater(float(image.grad.abs().sum()),0)
