import unittest
import numpy as np
import torch
from turkeyworld.rssm_enemy_drift import SETTINGS, enemy_velocity_terms, hard_episode_classes


class EnemyDriftTests(unittest.TestCase):
    def trajectory(self,speed=.7):
        initial=torch.zeros(1,20);initial[:,8:10]=.5;initial[:,12:14]=1
        target=initial[:,None].repeat(1,4,1)
        target[:,:,8]+=(torch.arange(1,5)*speed/480)[None]
        previous=torch.cat((initial[:,None],target[:,:-1]),1)
        target[:,:,10]=(target[:,:,8]-previous[:,:,8])*240
        return initial,target

    def test_correct_velocity_has_zero_loss_and_wrong_speed_has_correct_gradient(self):
        initial,target=self.trajectory()
        loss,_=enemy_velocity_terms(target,initial,target)
        self.assertLess(float(loss),1e-7)
        speed=torch.tensor(1.2,requires_grad=True)
        prediction=target.clone();prediction[:,:,8]=initial[:,None,8]+torch.arange(1,5)[None]*speed/480
        loss,metrics=enemy_velocity_terms(prediction,initial,target);loss.backward()
        self.assertGreater(float(speed.grad),0)
        self.assertGreater(metrics['enemy_long_speed_loss'],0)

    def test_direction_error_distinguishes_equal_speed_opposite_motion(self):
        initial,target=self.trajectory();wrong=target.clone()
        wrong[:,:,8]=2*initial[:,None,8]-target[:,:,8]
        _,metrics=enemy_velocity_terms(wrong,initial,target)
        self.assertGreater(metrics['enemy_long_direction_loss'],1.9)
        self.assertLess(metrics['enemy_long_speed_loss'],1e-7)

    def test_stationary_and_respawn_masks_are_finite_and_target_has_no_gradient(self):
        initial,target=self.trajectory(0);target.requires_grad_();prediction=target.detach().clone().requires_grad_()
        loss,_=enemy_velocity_terms(prediction,initial,target);loss.backward()
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()));self.assertIsNone(target.grad)
        loss,m=enemy_velocity_terms(prediction,initial,target,respawns=torch.ones(1,4,1,dtype=torch.bool))
        self.assertEqual(float(loss.detach()),0);self.assertEqual(m['enemy_long_valid_transitions'],0)

    def test_hard_classes_detect_a_corner_and_near_wall_without_Source_queries(self):
        states=np.zeros((5,20),np.float32)
        states[:,8:10]=np.array([[2.3,2.5],[2.31,2.5],[2.32,2.5],[2.32,2.51],[2.32,2.52]])/16
        grid=np.zeros((16,16),np.uint8);grid[2,1]=1
        self.assertEqual(hard_episode_classes(states,grid,horizon=4),dict(near_wall=True,turn=True))
        self.assertFalse(hard_episode_classes(states,np.zeros_like(grid),horizon=4)['near_wall'])
