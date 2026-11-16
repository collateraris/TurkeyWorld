import unittest
from unittest.mock import patch
import numpy as np
import torch

from turkeyworld.engine import Game
from turkeyworld.renderer_wall_supervision import source_wall_masks
from turkeyworld.sr_wall_patchgan import (SETTINGS,WallPatchDiscriminator,wall_patch_pairs,
    discriminator_loss,generator_loss,gan_weight,wall_reconstruction)


class WallPatchGANTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_native128_wall_labels_exclude_real_npc_mask_without_render_or_step(self):
        game=Game(seed=0);state=torch.tensor(game.state_vector())[None];grid=torch.tensor(game.map)[None]
        mask=torch.zeros(1,1,128,128);mask[:,:,50:75,50:75]=1
        with patch.object(Game,'render',side_effect=AssertionError('No Source image query')),patch.object(Game,'step',side_effect=AssertionError('No Source transition')):
            labels=source_wall_masks(grid,state,mask,native_size=128)
            old=source_wall_masks(grid,state,torch.zeros(1,1,32,32))
        self.assertEqual(labels['wall'].shape,(1,1,128,128));self.assertEqual(old['wall'].shape,(1,1,32,32))
        self.assertFalse(bool((labels['wall']*mask).any()))

    def test_pairs_are_aligned_strict_wall_crops_and_fake_retains_gradient(self):
        fake=torch.rand(2,3,128,128,requires_grad=True);real=torch.rand_like(fake)
        mask=torch.zeros(2,1,128,128);mask[0,:,:80,:80]=1
        pairs=wall_patch_pairs(torch.rand(2,3,32,32),fake,real,mask,np.random.default_rng(1))
        self.assertEqual(len(pairs['coordinates']),2)
        for j,(i,y,x) in enumerate(pairs['coordinates']):
            self.assertEqual(i,0);self.assertTrue(bool(mask[i,:,y:y+32,x:x+32].all()))
            self.assertTrue(torch.equal(pairs['real'][j],real[i,:,y:y+32,x:x+32]))
        pairs['fake'].sum().backward();self.assertTrue(bool(fake.grad.abs().sum()>0))
        self.assertIsNone(wall_patch_pairs(torch.rand(2,3,32,32),fake,real,torch.zeros_like(mask),np.random.default_rng(1)))

    def test_discriminator_update_and_generator_gradients_are_separate(self):
        discriminator=WallPatchDiscriminator();fake=torch.rand(2,3,32,32,requires_grad=True)
        pairs=dict(condition=torch.rand_like(fake),real=torch.rand_like(fake),fake=fake)
        discriminator_loss(discriminator,pairs).backward();self.assertIsNone(fake.grad)
        discriminator.zero_grad(set_to_none=True);before={k:v.clone() for k,v in discriminator.state_dict().items()}
        loss,_=generator_loss(discriminator,pairs);loss.backward()
        self.assertTrue(bool(torch.isfinite(fake.grad).all()));self.assertGreater(float(fake.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in discriminator.parameters()))
        self.assertTrue(all(torch.equal(v,discriminator.state_dict()[k]) for k,v in before.items()))

    def test_low_rates_schedule_and_wall_detail_loss(self):
        self.assertEqual(SETTINGS['generator_lr'],1e-5);self.assertEqual(SETTINGS['discriminator_lr'],5e-6)
        self.assertEqual(gan_weight(25),0);self.assertEqual(gan_weight(125),.002)
        image=torch.rand(1,3,128,128,requires_grad=True);mask=torch.ones(1,1,128,128)
        loss,m=wall_reconstruction(image,image.detach(),mask);self.assertEqual(float(loss.detach()),0)
        target=image.detach().roll(1,-1);loss,m=wall_reconstruction(image,target,mask);loss.backward()
        self.assertGreater(m['wall_high_frequency'],0);self.assertGreater(float(image.grad.abs().sum()),0)
