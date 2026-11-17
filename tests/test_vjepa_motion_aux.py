import unittest
import torch
from turkeyworld.vjepa_motion_aux import player_auxiliary,window_kind,commands

class MotionAuxTests(unittest.TestCase):
    def fixture(self, moving=True):
        initial=torch.zeros(1,20);initial[:,3]=1;initial[:,4]=1
        actions=torch.zeros(1,32,4);actions[:,:,0]=float(moving)
        states=initial[:,None].repeat(1,33,1)
        for i in range(32):states[:,i+1,:2]=states[:,i,:2]+commands(states[:,i:i+1],actions[:,i:i+1])[:,0]
        return initial,actions,states
    def test_correct_motion_zero_and_stall_gradient(self):
        initial,actions,states=self.fixture();target=states[:,1:]
        v,c=player_auxiliary(target,target,initial,actions)
        self.assertAlmostEqual(float(v+c),0,places=6)
        pred=states[:,:1].repeat(1,32,1).clone().requires_grad_()
        v,c=player_auxiliary(pred,target,initial,actions);(v+c).backward()
        self.assertGreater(float(v.detach()),0);self.assertGreater(float(c.detach()),0)
        self.assertTrue(torch.isfinite(pred.grad).all());self.assertGreater(float(pred.grad.abs().sum()),0)
    def test_categories(self):
        initial,actions,states=self.fixture()
        self.assertEqual(window_kind(states[0,:-1],actions[0],states[0,1:]),'moving')
        self.assertEqual(window_kind(states[0,:-1],actions[0],states[0,:-1]),'blocked')
        self.assertEqual(window_kind(states[0,:-1],actions[0]*0,states[0,:-1]),'idle')
    def test_idle_mask_finite(self):
        initial,actions,states=self.fixture(False)
        v,c=player_auxiliary(states[:,1:],states[:,1:],initial,actions)
        self.assertEqual(float(v+c),0)
