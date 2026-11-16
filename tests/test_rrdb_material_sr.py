import unittest
from unittest.mock import patch
import torch
from torch import nn
from turkeyworld.models import build_upscaler
from turkeyworld.rrdb_material_sr import RRDBMaterialSR
from turkeyworld.train_rrdb_material_sr import objective
from turkeyworld.engine import Game

class SpriteStub(nn.Module):
    def __init__(self):super().__init__();self.weight=nn.Parameter(torch.ones(()))
    def forward_with_aux(self,low):
        b=len(low);device=low.device
        return dict(visual_composite_alpha_hr=torch.ones(b,1,128,128,device=device),visual_animation_index=torch.ones(b,dtype=torch.long,device=device),visual_raw_warped_rgba=torch.ones(b,4,128,128,device=device)*.8)

class RRDBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def make(self):
        with patch('turkeyworld.models.build_upscaler',return_value=SpriteStub()):return RRDBMaterialSR(sprite_config={},channels=8,growth=4,blocks=1)
    def test_shapes_material_gradients_and_separate_sprite_veto(self):
        m=self.make().configure_training();m.enemies_enabled=False;low=torch.rand(1,3,32,32);o=m.forward_with_aux(low)
        self.assertEqual(o['image'].shape,(1,3,128,128));self.assertEqual(o['material_logits'].shape,(1,4,128,128));self.assertEqual(o['seam_logits'].shape,(1,2,128,128));self.assertFalse(bool(o['sprite_alpha'].any()))
        g=Game(seed=0);batch=dict(hr_source=torch.rand(1,3,128,128),mask_hr=torch.zeros(1,1,128,128),map=torch.tensor(g.map)[None],state=torch.tensor(g.state_vector())[None])
        loss,_=objective(o,batch);loss.backward()
        for head in (m.material_head,m.seam_head,m.color_head):self.assertGreater(float(head.weight.grad.abs().sum()),0)
        self.assertIsNone(m.sprite_renderer.weight.grad)
        m.enemies_enabled=True;m.eval();o=m.forward_with_aux(low);self.assertTrue(torch.allclose(o['image'],torch.full_like(o['image'],.8)))
        state=batch['state'].clone();state[:,13]=0;m.predicted_sprite_state=state
        self.assertFalse(bool(m.forward_with_aux(low)['sprite_alpha'].any()))
    def test_configuration_roundtrip_and_strict_shape(self):
        m=self.make();saved=m.state_dict()
        with patch('turkeyworld.models.build_upscaler',return_value=SpriteStub()):
            copy=build_upscaler(m.config)
        copy.load_state_dict(saved,strict=True)
        self.assertEqual(copy.config,m.config)
        with self.assertRaises(ValueError):m(torch.rand(1,3,64,64))

    def test_weapon_is_composited_after_fully_opaque_enemy(self):
        from turkeyworld.weapon_memory_sr import initialize_weapon_memory,source_weapon_rgba
        with patch('turkeyworld.models.build_upscaler',return_value=SpriteStub()):
            model=RRDBMaterialSR(sprite_config={},channels=8,growth=4,blocks=1,weapon_memory=True)
        initialize_weapon_memory(model.weapon_memory);model.eval()
        output=model.forward_with_aux(torch.rand(1,3,32,32))
        rgba=source_weapon_rgba()[0:1]
        expected=.8*(1-rgba[:,3:])+rgba[:,:3]*rgba[:,3:]
        support=rgba[:,3:]>0
        error=(output['image']-expected).abs()*support
        self.assertLess(float(error.max().detach()),3e-5)
        self.assertTrue(bool(output['sprite_alpha'].all()))
