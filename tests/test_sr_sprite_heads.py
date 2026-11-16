import unittest
import torch
from torch import nn
from types import SimpleNamespace
from turkeyworld.sr_sprite_heads import configure,objective


class SpriteHeadTests(unittest.TestCase):
    def test_ground_projection_recovers_Source_height_and_bottom_without_Source_inputs(self):
        from turkeyworld.atlas_enemy_sr import ground_anchor_box
        from turkeyworld.atlas_enemy_objectives import source_full_sprite_geometry
        for distance in (1.,2.,3.,5.,8.,12.):
            state=torch.zeros(1,20);state[:,0]=1.5/16;state[:,1]=.5;state[:,3]=1
            state[:,8]=(1.5+distance)/16;state[:,9]=.5;state[:,12:14]=1
            target=source_full_sprite_geometry(state)['full_bbox']
            bad=target-torch.tensor([0.,.2,0.,.2])
            result=ground_anchor_box(bad)
            self.assertLessEqual(float((result[:,3]-target[:,3]).abs().max()*128),1.001)
            self.assertTrue(torch.allclose(result[:,2]-result[:,0],result[:,3]-result[:,1]))

    def test_ground_anchor_moves_a_high_sprite_down_toward_Source(self):
        from turkeyworld.atlas_enemy_objectives import source_full_sprite_geometry
        state=torch.zeros(1,20);state[:,0]=.2;state[:,1]=.5;state[:,3]=1;state[:,4]=1
        state[:,8]=.4;state[:,9]=.5;state[:,12:14]=1
        target=source_full_sprite_geometry(state)['full_bbox']
        box=(target-torch.tensor([0.,.15,0.,.15])).detach().requires_grad_()
        output=dict(visual_full_bbox=box,visual_visible_logits_hr=torch.zeros(1,1,128,128,requires_grad=True),
            presence_logits=torch.zeros(1,1,requires_grad=True),visual_confirmation_logits=torch.zeros(1,1,requires_grad=True),
            visual_animation_logits=torch.zeros(1,7,requires_grad=True))
        mask=torch.zeros(1,1,128,128);mask[:,:,50:80,50:80]=1
        loss,metrics=objective(output,dict(state=state,mask_hr=mask),ground=True);loss.backward()
        self.assertAlmostEqual(metrics['bottom_anchor'],.15,places=5)
        self.assertLess(float(box.grad[0,1]),0);self.assertLess(float(box.grad[0,3]),0)

    def test_only_named_sprite_heads_train_and_wall_detail_is_frozen(self):
        m=nn.Module();m.detail_residual=nn.Linear(1,1);m.sprite_renderer=nn.Module()
        m.sprite_renderer.enemy_decoder_animation=nn.Linear(1,7)
        m.sprite_renderer.enemy_decoder_visual=nn.Linear(1,1)
        m.sprite_renderer.input=nn.Linear(1,1)
        names=configure(m)
        self.assertTrue(all(n.startswith('sprite_renderer.enemy_decoder_') for n in names))
        self.assertFalse(m.detail_residual.weight.requires_grad);self.assertFalse(m.sprite_renderer.input.weight.requires_grad)

    def test_alive_walking_labels_train_wrong_death_logits(self):
        state=torch.zeros(2,20);state[:,3]=1;state[:,4]=1;state[:,8]=.8;state[:,9]=.5;state[:,12:14]=1;state[:,10]=.35
        phase=torch.zeros(2,7,requires_grad=True)
        with torch.no_grad():phase[:,6]=4
        box=torch.tensor([[.3,.3,.7,.7]]*2,requires_grad=True)
        output=dict(visual_full_bbox=box,visual_visible_logits_hr=torch.zeros(2,1,128,128,requires_grad=True),
            presence_logits=torch.zeros(2,1,requires_grad=True),visual_confirmation_logits=torch.zeros(2,1,requires_grad=True),
            visual_animation_logits=phase)
        mask=torch.zeros(2,1,128,128);mask[:,:,45:75,50:75]=1
        loss,_=objective(output,dict(state=state,mask_hr=mask));loss.backward()
        self.assertGreater(float(phase.grad[:,6].sum()),0)
        self.assertLess(float(phase.grad[:,1].sum()),0)
        self.assertTrue(bool(torch.isfinite(box.grad).all()))
