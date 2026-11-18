import unittest
import numpy as np
import torch
from turkeyworld.vjepa_motion_aux import segmented_player_auxiliary,commands,window_kind
from turkeyworld.vjepa_balanced_windows import motion_window_index,draw_available
class MotionBlocksTests(unittest.TestCase):
 def test_late_errors_and_all_block_boundaries(self):
  initial=torch.zeros(1,20);initial[:,3:5]=1;actions=torch.zeros(1,180,4);actions[:,:,0]=1
  states=initial[:,None].repeat(1,181,1)
  for tick in range(180):states[:,tick+1,:2]=states[:,tick,:2]+commands(states[:,tick:tick+1],actions[:,tick:tick+1])[:,0]
  target=states[:,1:];v,c,blocks=segmented_player_auxiliary(target,target,initial,actions)
  self.assertEqual([(x[0],x[1]) for x in blocks],[(i+1,min(i+16,180)) for i in range(0,180,16)])
  self.assertAlmostEqual(float(v+c),0,places=6)
  prediction=target.clone();prediction[:,100,0]+=.01;prediction.requires_grad_()
  v,c,_=segmented_player_auxiliary(prediction,target,initial,actions);(v+c).backward()
  self.assertGreater(float(prediction.grad[:,100,:2].abs().sum()),0)
  # A persistent offset must not become a phantom velocity jump at every boundary.
  shifted=target.clone();shifted[:,:,0]+=.1
  _,_,b=segmented_player_auxiliary(shifted,target,initial,actions)
  self.assertAlmostEqual(float(b[1][2]),0,places=5)
 def test_available_pools_exact_no_fallback(self):
  episodes=[]
  for action,move in [([1,0,0,0],True),([1,0,0,0],False),([0,0,0,0],False)]:
   s=np.zeros((201,20),np.float32);s[:,3:5]=1;a=np.tile(action,(200,1)).astype(np.float32)
   if move:s[:,0]=np.arange(201)*.005
   episodes.append(dict(states=s,actions=a))
  pools=motion_window_index(episodes);self.assertEqual(set(pools),{'moving','blocked','idle'})
  for kind,rows in pools.items():
   for ep,start in rows:
    e=episodes[ep];self.assertEqual(window_kind(torch.tensor(e['states'][start:start+32]),torch.tensor(e['actions'][start:start+32]),torch.tensor(e['states'][start+1:start+33])),kind)
  only=motion_window_index(episodes[:1]);rng=np.random.default_rng(2)
  for i in range(12):self.assertEqual(draw_available(only,rng,i)[2],'moving')
