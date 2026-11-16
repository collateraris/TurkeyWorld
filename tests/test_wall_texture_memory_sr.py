import unittest
from unittest.mock import patch
import torch
from turkeyworld.wall_texture_memory_sr import WallTextureMemory,initialize_wall_memory,geometry_labels,wall_memory_objective
from turkeyworld.engine import Game

class WallMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_brick_cannot_sample_green_earth_texture_and_runtime_has_no_source(self):
        m=WallTextureMemory(8);initialize_wall_memory(m);m.eval()
        material=torch.zeros(1,4,128,128);material[:,1]=10
        with patch.object(Game,'render',side_effect=AssertionError('No source runtime')):
            o=m(torch.rand(1,8,128,128),material)
        self.assertEqual(o['rgb'].shape,(1,3,128,128))
        self.assertFalse(bool((o['rgb'][:,1]>o['rgb'][:,0]).any()))
        # A predicted earth wall can only contain grass at its geometrical top.
        material[:,1]=0;material[:,2]=10;o=m(torch.rand(1,8,128,128),material)
        green=(o['rgb'][:,1:2]>o['rgb'][:,:1])&(o['alpha']>.5)
        self.assertFalse(bool((green&(o['v']>.3)).any()))
    def test_source_geometry_labels_and_false_green_penalty_have_gradients(self):
        g=Game(seed=0);batch=dict(map=torch.tensor(g.map)[None],state=torch.tensor(g.state_vector())[None],mask_hr=torch.zeros(1,1,128,128),hr_source=torch.zeros(1,3,128,128))
        depth,phase=geometry_labels(batch);self.assertEqual(depth.shape,(1,1,128));self.assertTrue(torch.allclose(phase.square().sum(1),torch.ones(1,128)))
        m=WallTextureMemory(8);initialize_wall_memory(m);m.train();o=m(torch.rand(1,8,128,128),torch.randn(1,4,128,128))
        image=torch.zeros(1,3,128,128);image[:,1]=.5;image.requires_grad_()
        loss,metrics=wall_memory_objective(dict(wall_memory=o,image=image),batch);loss.backward()
        self.assertGreater(metrics['false_green_excess'],0);self.assertGreater(float(image.grad[:,1].abs().sum()),0)
        self.assertGreater(float(m.geometry[-1].weight.grad.abs().sum()),0)

    def test_geometry_supervised_mode_does_not_fit_uv_to_rgb_photo_error(self):
        m=WallTextureMemory(8,geometry_supervised=True);initialize_wall_memory(m);m.train()
        material=torch.zeros(1,4,128,128);material[:,1]=10
        o=m(torch.rand(1,8,128,128),material)
        o['rgb'].sum().backward()
        self.assertIsNone(m.geometry[-1].weight.grad)
        self.assertIsNotNone(m.rgb_logits.grad)

    def test_coherent_uv_has_perspective_progress_and_red_walls_stay_red(self):
        m=WallTextureMemory(8,geometry_supervised=True,coherent_uv=True);initialize_wall_memory(m);m.eval()
        material=torch.zeros(1,4,128,128);material[:,1:3]=5
        rgb=torch.tensor([154.,65.,46.])[None,:,None,None].expand(1,3,32,32)/255
        o=m(torch.rand(1,8,128,128),material,rgb)
        steps=torch.remainder(o['u'].diff(dim=-1),1)
        self.assertGreater(float(steps.min()),0);self.assertLess(float(steps.max()),.1)
        self.assertFalse(bool((o['rgb'][:,1]>o['rgb'][:,0]).any()))
        self.assertTrue(bool(((o['alpha']==0)|(o['alpha']==1)).all()))

class WallMemoryV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_red_column_cannot_acquire_green_roof_from_green_pixels_at_top(self):
        from turkeyworld.wall_texture_memory_sr import WallTextureMemoryV2
        m=WallTextureMemoryV2(8,geometry_supervised=True,coherent_uv=True);initialize_wall_memory(m);m.eval()
        material=torch.zeros(1,4,128,128);material[:,1:3]=5
        low=torch.tensor([154.,65.,46.])[None,:,None,None].expand(1,3,32,32).clone()/255
        low[:,:,:12]=torch.tensor([.1,.8,.1])[None,:,None,None]
        output=m(torch.rand(1,8,128,128),material,low)
        self.assertTrue(bool((output['column_logits'].argmax(1)==0).all()))
        self.assertFalse(bool((output['rgb'][:,1]>output['rgb'][:,0]).any()))
        expected=(1-.24*.5)*(1-output['depth']*.024).clamp(.5,1)
        self.assertTrue(torch.allclose(output['shade'],expected))
    def test_column_and_side_labels_train_new_head(self):
        from turkeyworld.wall_texture_memory_sr import WallTextureMemoryV2,column_objective,column_labels
        g=Game(seed=0);batch=dict(map=torch.tensor(g.map)[None],state=torch.tensor(g.state_vector())[None],mask_hr=torch.zeros(1,1,128,128))
        types,sides=column_labels(batch);self.assertEqual(types.shape,(1,128));self.assertTrue(bool(((types==0)|(types==1)).all()))
        m=WallTextureMemoryV2(8,geometry_supervised=True,coherent_uv=True);initialize_wall_memory(m)
        o=m(torch.rand(1,8,128,128),torch.randn(1,4,128,128),torch.rand(1,3,32,32))
        loss,metrics=column_objective(o,batch);loss.backward()
        self.assertGreater(float(m.column_head[-1].weight.grad.abs().sum()),0)
        self.assertTrue(0<=metrics['wall_column_material_accuracy']<=1)
