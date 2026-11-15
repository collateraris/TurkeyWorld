import unittest
from unittest.mock import patch
import numpy as np
import torch
from turkeyworld.rssm_first_contact import first_contact_terms,SETTINGS
from turkeyworld.rssm_player_local_motion import PLAYER_LOCAL_MOTION_PROTOCOL

class FirstContactTests(unittest.TestCase):
    def batch(self):
        grid=np.zeros((16,16),np.uint8);grid[[0,-1],:]=1;grid[:,[0,-1]]=1;grid[5,6]=1
        state=torch.zeros(1,20);state[:,0]=5.78/16;state[:,1]=5.5/16;state[:,3]=1;state[:,4]=1;state[:,5]=1;state[:,8:10]=.5
        prediction=state[:,None].repeat(1,3,1);prediction[0,:,0]+=torch.tensor([.08,.16,.24])/16;prediction.requires_grad_()
        actions=torch.zeros(1,3,4);actions[:,:,0]=1;logits=torch.zeros(1,3,2,requires_grad=True)
        return prediction,state,actions,torch.tensor(grid)[None],logits
    def test_first_crossing_is_supervised_from_legal_own_state_without_correcting_it(self):
        data=self.batch();before=[x.detach().clone() for x in data]
        with patch('turkeyworld.engine.Game.render',side_effect=AssertionError('No renderer')):
            loss,m=first_contact_terms(*data,protocol=PLAYER_LOCAL_MOTION_PROTOCOL)
        self.assertEqual(m['first_error_ticks'],[0]);self.assertEqual(m['selected_actors'],1)
        loss.backward();self.assertGreater(float(data[0].grad[0,0,0]),0)
        self.assertGreater(float(data[4].grad.abs().sum()),0)
        for x,original in zip(data,before):self.assertTrue(torch.equal(x.detach(),original))
        self.assertEqual(m['own_states_corrected'],0)
    def test_invalid_own_body_is_skipped_and_empty_loss_retains_graph(self):
        data=list(self.batch());data[1][:,0]=6.1/16
        loss,m=first_contact_terms(*data,protocol=PLAYER_LOCAL_MOTION_PROTOCOL);loss.backward()
        self.assertEqual(m['selected_actors'],0);self.assertEqual(float(loss.detach()),0)
    def test_configuration_cannot_change_silently(self):
        with self.assertRaises(ValueError):first_contact_terms(*self.batch(),protocol=PLAYER_LOCAL_MOTION_PROTOCOL,settings={**SETTINGS,'weight':3})
    def test_enemy_first_wrong_velocity_uses_own_state_and_keeps_graph(self):
        from turkeyworld.engine import Game
        from turkeyworld.rssm_first_contact import first_enemy_terms
        g=Game(seed=0,endless=True,enemies_enabled=True)
        grid=np.zeros((16,16),np.uint8);grid[[0,-1],:]=1;grid[:,[0,-1]]=1;g.map=grid
        snapshot=g.get_state();snapshot['player'].update(x=3.5,y=3.5,angle=0)
        snapshot['enemies'][0].update(x=8.5,y=7.5,hp=100,alive=True)
        g.set_state(snapshot);initial=torch.tensor(g.state_vector())[None]
        pred=initial[:,None].repeat(1,4,1);pred[:,:,8]+=torch.arange(1,5)[None]*.04/16;pred.requires_grad_()
        before=pred.detach().clone();actions=torch.zeros(1,4,4)
        loss,m=first_enemy_terms(pred,initial,actions,torch.tensor(grid)[None]);loss.backward()
        self.assertEqual(m['first_error_ticks'],[0]);self.assertEqual(m['selected_actors'],1)
        self.assertGreater(float(pred.grad[:,:,8:10].abs().sum()),0);self.assertTrue(torch.equal(pred.detach(),before))
