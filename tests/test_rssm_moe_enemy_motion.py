from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import torch
from torch import nn

from test_rssm_moe_navigation_objective import TinyNavigationModel,batch
from turkeyworld.rssm_moe_enemy_motion import enemy_motion_backward


class EnemyModel(TinyNavigationModel):
    def __init__(self):
        super().__init__();self.enemy_head=nn.Linear(10,2)
    def transition(self,state,action,fmap,memory):
        assert not self.config.navigation_only and not self.vjepa_backend.config.navigation_only
        self.calls.append((state.detach().clone(),memory.detach().clone()))
        p,e,c=memory[:,:12],memory[:,12:22],memory[:,22:]
        fusion=self.fusion(p,e,state,action,fmap,fmap[:,None].expand(-1,2,-1),combat_context=self.c_context(c))
        p,e,c=self.p_core(p).tanh(),self.e_core(e).tanh(),self.c_core(c).tanh()
        movement=self.enemy_head(e)+fusion['enemy_motion_residual']
        new=state.clone();new[:,8:10]=state[:,8:10]+.002*movement.tanh()
        packed=torch.cat((p,e,c),-1);self.returned_memories.append(packed.detach().clone())
        return dict(state=new,memory=packed,respawn_flags=torch.zeros(len(state),1,dtype=torch.bool),rssm_fusion_moe=fusion)


def quiet_batch(horizon):
    result=batch(horizon);result['actions'][...,3]=0
    for name in ('states','target_states'):
        result[name][...,12]=.6;result[name][...,13]=1
    for tick in range(horizon):
        result['target_states'][:,tick,8:10]+=.001*(tick+1)
        result['states'][:,2+tick,8:10]=result['target_states'][:,tick,8:10]
    return result


class EnemyMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_real_moe_backward_all_horizons_own_memory_and_only_enemy_objective(self):
        for horizon in (4,8,16,32):
            model=EnemyModel();data=quiet_batch(horizon)
            old_config=model.config;modes=[m.training for m in model.modules()]
            for value in data.values():value.requires_grad_()
            with patch('turkeyworld.rssm_combat_branches._source_step_at_own_state',side_effect=AssertionError('No combat oracle')):
                report=enemy_motion_backward(model,data,context=2,expected_horizon=horizon)
            self.assertEqual(len(model.calls),horizon)
            for tick in range(1,horizon):
                self.assertTrue(torch.equal(model.calls[tick][1],model.returned_memories[tick-1]))
            self.assertIs(model.config,old_config);self.assertEqual(modes,[m.training for m in model.modules()])
            self.assertEqual(float(model.seen_frames[0].max()),.25)
            self.assertEqual(report['enemy_motion_valid_npc_transitions'],2*horizon)
            self.assertEqual(report['enemy_motion_memory_resets_after_initialization'],0)
            self.assertEqual(report['enemy_motion_combat_loss'],0)
            self.assertEqual(report['enemy_motion_player_loss'],0)
            self.assertGreater(float(model.enemy_head.weight.grad.abs().sum()),0)
            self.assertIsNone(model.p_head.weight.grad)
            self.assertTrue(all(v.grad is None for v in data.values()))
            self.assertTrue(all(p.grad is None for p in model.frozen_renderer.parameters()))
            json.dumps(report,allow_nan=False)

    def test_all_weighted_losses_and_gradients_scale_linearly(self):
        torch.manual_seed(1);a=EnemyModel();b=deepcopy(a);data=quiet_batch(8)
        x=enemy_motion_backward(a,data,context=2,enemy_weight=1.,router_weight=1.)
        y=enemy_motion_backward(b,data,context=2,enemy_weight=.25,router_weight=.25)
        self.assertAlmostEqual(y['enemy_motion_loss'],.25*x['enemy_motion_loss'],places=7)
        for p,q in zip(a.parameters(),b.parameters()):
            if p.grad is not None:torch.testing.assert_close(q.grad,p.grad*.25,atol=1e-7,rtol=1e-5)

    def test_nonquiet_source_and_wrong_horizon_rejected(self):
        for kind in ('shoot','death','damage','horizon'):
            model=EnemyModel();data=quiet_batch(4)
            if kind=='shoot':data['actions'][0,0,3]=1
            elif kind=='death':data['target_states'][0,0,13]=0
            elif kind=='damage':data['target_states'][0,0,12]=.1
            with self.assertRaises(ValueError):
                enemy_motion_backward(model,data,context=2,expected_horizon=8 if kind=='horizon' else 4)

    def test_environment_profiles_restored_after_failure(self):
        model=EnemyModel();model.config.navigation_only=True;model.vjepa_backend.config.navigation_only=True
        a,b=model.config,model.vjepa_backend.config
        with patch.object(model,'transition',side_effect=RuntimeError('probe')):
            with self.assertRaises(RuntimeError):enemy_motion_backward(model,quiet_batch(4),context=2)
        self.assertIs(model.config,a);self.assertIs(model.vjepa_backend.config,b)


if __name__=='__main__':unittest.main()
