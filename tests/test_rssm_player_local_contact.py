"""Quiet own-prefix quotas and exact local labels after full own rollout."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.engine import Game
from turkeyworld.rssm_player_contact_replay import(
    PLAYER_LOCAL_MOTION_PROTOCOL,PlayerContactReplaySampler,capture_player_contact_candidate,
    select_player_contact_branches,contact_replay_backward,_scout)
from test_rssm_player_contact_replay import OwnWorld,Data


class BadEnemyWorld(OwnWorld):
    def transition(self,state,action,features,memory):
        result=super().transition(state,action,features,memory)
        if not self.config.navigation_only:
            full=result['state'].clone();full[:,8:10]=full.new_tensor([6.5,5.5])/16
            full[:,12:14]=1;result['state']=full
        return result


class DiesAfterMovementWorld(OwnWorld):
    def transition(self,state,action,features,memory):
        result=super().transition(state,action,features,memory)
        full=result['state'].clone();full[:,4]=0;result['state']=full
        return result


def quiet_prefix(world,batch,step=16):
    state=batch['states'][:2,0].clone();state[:,:2]=state.new_tensor([[3.5,3.5],[3.6,3.4]])/16
    state[:,8:10]=state.new_tensor([8.5,2.5])/16;state[:,12:14]=1
    action=batch['actions'][:2,:32].clone();grid=batch['map'][:2]
    with torch.no_grad():
        features=world.encode_map(grid);memory=world.initialize_from_image(batch['frames'][:2,:1],state[:,None],features)
        for tick in range(step):
            result=world.transition(state,action[:,tick%32],features,memory)
            state,memory=result['state'],result['memory']
        prefix_state,prefix_memory=state.clone(),memory.clone();future=[]
        for tick in range(16):
            result=world.transition(state,action[:,tick],features,memory)
            state,memory=result['state'],result['memory'];future.append(state.clone())
    return capture_player_contact_candidate(step,prefix_state,prefix_memory,action,grid,torch.stack(future,1),
        navigation_only=False,stream='enemy_motion')


class PlayerLocalContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.data=Data(self.temp.name);self.batch,_=PlayerContactReplaySampler(self.data).batch()

    def test_quota_selects_two_legal_nonzero_quiet_prefixes_even_with_NPC_in_wall(self):
        world=BadEnemyWorld();quiet=quiet_prefix(world,self.batch)
        candidates=[quiet,*_scout(world,self.batch,None)]
        selected,report=select_player_contact_branches(candidates,horizon=8,
            player_local_motion_protocol=PLAYER_LOCAL_MOTION_PROTOCOL)
        self.assertEqual(report['quiet_nonzero_reserved'],2)
        self.assertGreaterEqual(report['quiet_nonzero_selected'],2)
        self.assertEqual(report['quiet_nonzero_shortfall'],0)
        self.assertLessEqual(report['source_probes'],32)
        self.assertGreater(report['selection_player_labels_despite_invalid_NPC'],0)
        for row in selected[:2]:
            self.assertEqual(row['item']['stream'],'enemy_motion');self.assertEqual(row['step'],16)
            self.assertFalse(row['item']['navigation_only'])
            self.assertNotIn('labels',row) # No fabricated no-NPC H-step future tape.
            self.assertTrue(row['selection_player_label']['own_NPC_body_invalid'])

    def test_no_legal_nonzero_quiet_prefix_reports_honest_shortfall(self):
        world=BadEnemyWorld();quiet=quiet_prefix(world,self.batch)
        quiet['state'][:,:2]=quiet['state'].new_tensor([6.5,5.5])/16
        selected,report=select_player_contact_branches([quiet,*_scout(world,self.batch,None)],horizon=8,
            player_local_motion_protocol=PLAYER_LOCAL_MOTION_PROTOCOL)
        self.assertEqual(report['quiet_nonzero_selected'],0)
        self.assertEqual(report['quiet_nonzero_shortfall'],2)
        self.assertLessEqual(report['source_probes'],32)
        self.assertTrue(all(row['item']['stream']!='enemy_motion'for row in selected))

    def test_backward_keeps_full_quiet_NPC_hz_and_source_direct_queries_follow_every_forward(self):
        world=BadEnemyWorld();quiet=quiet_prefix(world,self.batch);old_calls=len(world.calls)
        old_initializations=len(world.initializations);source_times=[];real=Game.step
        before={n:p.detach().clone()for n,p in world.named_parameters()}
        def step(game,*args,**kwargs):
            source_times.append(len(world.calls));return real(game,*args,**kwargs)
        with patch.object(Game,'step',new=step),patch.object(Game,'render',side_effect=AssertionError('No rendering')):
            report=contact_replay_backward(world,self.batch,candidates=[quiet],horizon=8,
                player_local_motion_protocol=PLAYER_LOCAL_MOTION_PROTOCOL)
        self.assertEqual(report['quiet_nonzero_reserved'],2)
        self.assertEqual(len(world.initializations),old_initializations+1) # Scout only.
        self.assertGreaterEqual(min(source_times),old_calls+32)
        self.assertTrue(all(n==len(world.calls)for n in source_times[-report['direct_source_valid']:]))
        self.assertGreater(report['direct_labels_despite_invalid_NPC'],0)
        cursor=old_calls+32
        for actor in range(2):
            actual=world.calls[cursor]
            torch.testing.assert_close(actual[0],quiet['state'][actor:actor+1],rtol=0,atol=0)
            torch.testing.assert_close(actual[1],quiet['memory'][actor:actor+1],rtol=0,atol=0)
            self.assertFalse(actual[2]);self.assertFalse(world.hidden_graph[cursor])
            self.assertTrue(world.hidden_graph[cursor+1]);cursor+=report['selected_branch_horizons'][actor]
        self.assertGreater(float(world.axis_logits.grad.abs().sum()),0)
        self.assertGreater(float(world.core.weight.grad.abs().sum()),0)
        self.assertIsNone(world.map_encoder.weight.grad);self.assertIsNone(world.renderer.weight.grad)
        for n,p in world.named_parameters():torch.testing.assert_close(p,before[n],rtol=0,atol=0)
        self.assertFalse(report['fabricated_no_NPC_future_trajectory'])
        self.assertEqual(report['state_resets_after_prefix'],0);self.assertEqual(report['memory_resets_after_prefix'],0)
        self.assertIn('x_free_y_blocked',report['player_local_motion'])
        self.assertIn('x_blocked_y_free',report['player_local_motion'])
        self.assertEqual(report['player_contact_tbptt'],64)

    def test_explicit_none_is_bitexact_legacy_report_and_gradients(self):
        a,b=OwnWorld(),OwnWorld();b.load_state_dict(a.state_dict())
        first=contact_replay_backward(a,self.batch,horizon=4)
        second=contact_replay_backward(b,self.batch,horizon=4,player_local_motion_protocol=None)
        self.assertEqual(first,second)
        for p,q in zip(a.parameters(),b.parameters()):
            if p.grad is None:self.assertIsNone(q.grad)
            else:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)

    def test_predicted_combat_death_does_not_create_no_enemy_future_labels(self):
        world=DiesAfterMovementWorld()
        report=contact_replay_backward(world,self.batch,horizon=8,
            player_local_motion_protocol=PLAYER_LOCAL_MOTION_PROTOCOL)
        self.assertEqual(report['branches'],4)
        self.assertEqual(report['direct_source_valid'],4)
        self.assertEqual(report['direct_source_skip_counts'],{'dead_player_motion_source':28})
        self.assertEqual(report['player_local_cumulative_residual_tiles_mean'],{})
        self.assertFalse(report['fabricated_no_NPC_future_trajectory'])
        self.assertEqual(report['quiet_nonzero_shortfall'],2)


if __name__=='__main__':unittest.main()
