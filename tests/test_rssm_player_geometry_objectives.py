from copy import deepcopy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from turkeyworld.rssm_player_geometry_objectives import(
    PLAYER_GEOMETRY_PROTOCOL,prepare_player_geometry,player_geometry_terms,add_player_geometry_metrics)
from turkeyworld.rssm_moe_enemy_motion import enemy_motion_backward
from turkeyworld.rssm_moe_navigation_objective import navigation_replay_backward
from test_rssm_moe_navigation_objective import TinyNavigationModel,batch,Config
from test_rssm_moe_enemy_motion import EnemyModel,quiet_batch
from test_rssm_moe_training import TinyModel,data as combat_batch
from turkeyworld.rssm_moe_training import moe_dynamics_backward,COMBAT_COMPONENTS


def pose(xy,horizon=1):
    value=torch.zeros(1,horizon,20);value[...,4]=1;value[...,:2]=torch.tensor(xy)/16
    return value


class MovingEnemyModel(EnemyModel):
    def transition(self,state,action,fmap,memory):
        result=super().transition(state,action,fmap,memory)
        passed=self.p_head(result['memory'][:,:12])+result['rssm_fusion_moe']['player_pass_residual']
        result['state']=torch.cat((state[:,:2]+.004*passed.sigmoid(),result['state'][:,2:]),-1)
        return result


class MemoryCreditModel(nn.Module):
    def __init__(self):
        super().__init__();self.scale=nn.Parameter(torch.tensor(1.))
        self.config=Config()
        self.vjepa_backend=nn.Module();self.vjepa_backend.config=Config()
        self.returned=[];self.memory_input_flags=[]
    def encode_map(self,grid):return grid
    def initialize_from_image(self,frames,states,fmap):return torch.ones(len(states),2)
    def transition(self,state,action,fmap,memory):
        self.memory_input_flags.append(memory.requires_grad)
        new=memory*self.scale
        if new.requires_grad:new.retain_grad()
        self.returned.append(new)
        predicted=torch.cat((state[:,:2]+.001*new,state[:,2:]),-1)
        zero=self.scale*0
        return dict(state=predicted,memory=new,respawn_flags=torch.zeros(len(state),1,dtype=torch.bool),
            rssm_fusion_moe=dict(router_role_loss=zero,router_balance_loss=zero))


def exact_credit_batch(horizon):
    data=batch(horizon,count=1,context=1);model=MemoryCreditModel()
    state=data['states'][:,0].clone();memory=torch.ones(1,2);values=[]
    with torch.no_grad():
        for i in range(horizon):
            result=model.transition(state,data['actions'][:,i],data['map'],memory)
            state,memory=result['state'],result['memory'];values.append(state.clone())
    target=torch.stack(values,1);target[:,-1,:2]+=.01
    data['target_states']=target;data['states']=torch.cat((data['states'][:,:1],target),1)
    return data


class PlayerGeometryObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_legal_x_then_y_corner_has_zero_penalty_even_when_chord_crosses_wall(self):
        grid=torch.zeros(1,16,16);grid[:,5,5]=1
        previous=pose([4.5,6.3]);target=pose([6.3,4.5])
        labels=prepare_player_geometry(previous,target,grid,player_navigation_protocol=PLAYER_GEOMETRY_PROTOCOL)
        prediction=target.clone().requires_grad_()
        loss,metrics=player_geometry_terms(prediction,previous[:,0],labels)
        self.assertEqual(float(loss.detach()),0);self.assertEqual(metrics['player_wall_axis_order'],'x_then_y')
        from turkeyworld.trajectory_objectives import circle_wall_penalty
        chord=(previous[...,:2]+target[...,:2])*.5*16
        self.assertGreater(float(circle_wall_penalty(chord,grid,.2)),0)
        loss.backward();self.assertEqual(float(prediction.grad.abs().sum()),0)

    def test_radius_normalized_penetration_gradient_and_source_masks_cannot_be_hidden(self):
        grid=torch.zeros(1,16,16,requires_grad=True);grid.data[:,5,5]=1
        previous=pose([4.7,5.5]).requires_grad_();target=pose([4.8,5.5]).requires_grad_()
        labels=prepare_player_geometry(previous,target,grid,player_navigation_protocol={'geometry':PLAYER_GEOMETRY_PROTOCOL})
        prediction=pose([5.1,5.5]).requires_grad_();prediction.data[...,4]=0 # False model death must not mask collision.
        loss,m=player_geometry_terms(prediction,previous[:,0],labels)
        self.assertGreater(float(loss.detach()),0)
        self.assertAlmostEqual(m['player_wall_endpoint_loss'],m['player_wall_endpoint_penetration_squared']/.04*.125,places=6)
        loss.backward()
        self.assertGreater(float(prediction.grad[...,0]),0) # Gradient descent exits the nearest left wall edge.
        self.assertIsNone(previous.grad);self.assertIsNone(target.grad);self.assertIsNone(grid.grad)
        self.assertEqual(m['player_wall_overlap_transitions'],1)
        dead=target.detach().clone();dead[...,4]=0
        mask=prepare_player_geometry(previous,dead,grid,player_navigation_protocol=PLAYER_GEOMETRY_PROTOCOL)
        zero,_=player_geometry_terms(prediction.detach(),previous[:,0],mask)
        self.assertEqual(float(zero),0)

    def test_full_tape_denominators_are_additive_across_tbptt_without_count_multiplication(self):
        grid=torch.zeros(1,16,16);grid[:,5,5]=1
        previous=pose([4.7,5.5],32);target=pose([4.8,5.5],32)
        predicted=pose([5.1,5.5],32).requires_grad_()
        labels=prepare_player_geometry(previous,target,grid,player_navigation_protocol=PLAYER_GEOMETRY_PROTOCOL)
        full,_=player_geometry_terms(predicted,previous[:,0],labels)
        total=0.;metrics={}
        for first in(0,16):
            old=previous[:,0]if first==0 else predicted[:,first-1].detach()
            value,part=player_geometry_terms(predicted[:,first:first+16],old,labels,first=first)
            total=total+value;add_player_geometry_metrics(metrics,part)
        self.assertAlmostEqual(float(full.detach()),float(total.detach()),places=6)
        self.assertEqual(metrics['player_wall_source_alive_transitions'],32)
        self.assertEqual(metrics['player_wall_overlap_transitions'],32)
        self.assertAlmostEqual(metrics['player_wall_loss'],float(full.detach()),places=6)

    def test_quiet_player_labels_train_player_head_without_changing_npc_objective_or_rollout(self):
        torch.manual_seed(173);legacy=MovingEnemyModel();current=deepcopy(legacy);data=quiet_batch(32)
        for value in data.values():value.requires_grad_()
        old=enemy_motion_backward(legacy,data,context=2,player_weight=0.)
        snapshots=[]
        def capture(step,state,memory,actions,grid,prediction):
            snapshots.append((step,state.detach().clone(),memory.detach().clone(),prediction.detach().clone()))
        with patch('turkeyworld.engine.Game.step',side_effect=AssertionError('Geometry must not query Source')):
            new=enemy_motion_backward(current,data,context=2,player_weight=.25,
                player_navigation_protocol={'geometry':PLAYER_GEOMETRY_PROTOCOL},player_contact_callback=capture)
        self.assertGreater(new['enemy_motion_player_loss'],0)
        self.assertGreater(float(current.p_head.weight.grad.abs().sum()),0)
        self.assertTrue(legacy.p_head.weight.grad is None or float(legacy.p_head.weight.grad.abs().sum())==0)
        self.assertAlmostEqual(new['enemy_motion_enemy_loss'],old['enemy_motion_enemy_loss'],places=7)
        self.assertAlmostEqual(new['enemy_motion_loss'],old['enemy_motion_loss']+new['enemy_motion_player_loss'],places=7)
        self.assertTrue(all(torch.equal(a[0],b[0])for a,b in zip(legacy.calls,current.calls)))
        self.assertEqual([r[0]for r in snapshots],[0,16])
        self.assertTrue(torch.equal(snapshots[1][2],current.returned_memories[15]))
        self.assertTrue(all(value.grad is None for value in data.values()))
        self.assertTrue(all(p.grad is None for p in current.frozen_renderer.parameters()))

    def test_navigation_contact_capture_keeps_actual_prefix_memory_without_new_images(self):
        model=TinyNavigationModel();snapshots=[]
        def capture(step,state,memory,actions,grid,prediction):
            snapshots.append((step,state.detach().clone(),memory.detach().clone(),actions.detach().clone(),prediction.detach().clone()))
        result=navigation_replay_backward(model,batch(32),context=2,weight=5/32,
            player_navigation_protocol={'geometry':PLAYER_GEOMETRY_PROTOCOL},player_contact_callback=capture)
        self.assertEqual([row[0]for row in snapshots],[0,16]);self.assertEqual(len(model.seen_frames),1)
        self.assertTrue(torch.equal(snapshots[1][2],model.returned_memories[15]))
        self.assertTrue(torch.equal(snapshots[1][1],model.calls[16][0]))
        self.assertAlmostEqual(result['nav_player_loss'],result['nav_player_xy_loss']+result['nav_player_wall_loss'],places=7)
        self.assertEqual(result['nav_memory_resets_after_initialization'],0)

    def test_protocol_activation_is_explicit_and_invalid_shapes_rejected(self):
        self.assertIsNone(prepare_player_geometry(None,None,None,player_navigation_protocol=None))
        cfg={**PLAYER_GEOMETRY_PROTOCOL,'radius':.25}
        with self.assertRaises(ValueError):prepare_player_geometry(pose([2,2]),pose([2,2]),torch.zeros(1,16,16),player_navigation_protocol=cfg)
        with self.assertRaisesRegex(ValueError,'explicit'):
            enemy_motion_backward(MovingEnemyModel(),quiet_batch(4),context=2,player_weight=.25)

    def test_tbptt64_carries_final_loss_to_first_memory_and_detaches_at65(self):
        protocol={'geometry':PLAYER_GEOMETRY_PROTOCOL,'tbptt':64}
        old=MemoryCreditModel();new=MemoryCreditModel();data=exact_credit_batch(64);snapshots=[]
        legacy=navigation_replay_backward(old,data,context=1,weight=1.)
        report=navigation_replay_backward(new,data,context=1,weight=1.,
            player_navigation_protocol=protocol,player_contact_callback=lambda step,*args:snapshots.append(step))
        self.assertEqual(legacy['nav_tbptt'],16);self.assertEqual(report['nav_tbptt'],64)
        self.assertEqual(float(old.returned[0].grad.abs().sum()),0.)
        self.assertGreater(float(new.returned[0].grad.abs().sum()),0.)
        self.assertFalse(old.memory_input_flags[16]);self.assertTrue(new.memory_input_flags[16])
        self.assertEqual(snapshots,[0,16,32,48]) # Contact capture is independent from64-step graph detaches.
        tail=MemoryCreditModel()
        navigation_replay_backward(tail,exact_credit_batch(80),context=1,weight=1.,player_navigation_protocol=protocol)
        self.assertFalse(tail.memory_input_flags[64])
        self.assertEqual(float(tail.returned[0].grad.abs().sum()),0.)
        self.assertGreater(float(tail.returned[64].grad.abs().sum()),0.)

    def test_tbptt450_carries_terminal_loss_to_first_memory_and_detaches_at451(self):
        protocol={'geometry':PLAYER_GEOMETRY_PROTOCOL,'tbptt':450}
        model=MemoryCreditModel();snapshots=[]
        report=navigation_replay_backward(model,exact_credit_batch(450),context=1,weight=1.,
            player_navigation_protocol=protocol,player_contact_callback=lambda tick,*args:snapshots.append(tick))
        self.assertEqual(report['nav_tbptt'],450)
        self.assertGreater(float(model.returned[0].grad.abs().sum()),0.)
        self.assertTrue(model.memory_input_flags[449])
        self.assertEqual(snapshots,list(range(0,447,16)))
        # Production stops at450. Extend only the test tape to exercise the next boundary.
        tail=MemoryCreditModel()
        with patch('turkeyworld.rssm_moe_navigation_objective.MOE_HORIZONS',(466,)):
            navigation_replay_backward(tail,exact_credit_batch(466),context=1,weight=1.,
                player_navigation_protocol=protocol)
        self.assertFalse(tail.memory_input_flags[450])
        self.assertEqual(float(tail.returned[0].grad.abs().sum()),0.)
        self.assertGreater(float(tail.returned[450].grad.abs().sum()),0.)

    def test_main64_geometry_is_counted_once_and_contact_snapshots_still_every16(self):
        model=TinyModel();data=combat_batch(64);aux=combat_batch(16)
        data['map'][:,5,5]=1
        snapshots=[];memories=[];original=model.transition
        def transition(*args):
            result=original(*args);memories.append(result['memory'].detach().clone());return result
        model.transition=transition
        def capture(step,state,memory,*args):snapshots.append((step,memory.detach().clone()))
        def fake_branch(*args,**kwargs):
            return dict(player=0.,enemy=0.,combat=0.,horizon=16,actors=4,
                direct_source_supervision_available=True,direct_per_tick_counts=[],
                direct_spawn_anchor_counts={},**{key:0. for key in COMBAT_COMPONENTS})
        with patch('turkeyworld.rssm_moe_training._own_branch_backward',side_effect=fake_branch):
            result=moe_dynamics_backward(model,data,aux,expected_horizon=64,chunk_size=64,
                player_navigation_protocol={'geometry':PLAYER_GEOMETRY_PROTOCOL,'tbptt':64},
                player_contact_callback=capture)
        self.assertEqual(result['moe_main_tbptt'],64)
        self.assertEqual([item[0]for item in snapshots],[0,16,32,48])
        self.assertTrue(torch.equal(snapshots[2][1],memories[31]))
        self.assertGreater(result['moe_main_player_wall_loss'],0)
        self.assertAlmostEqual(result['moe_player_loss'],result['moe_main_player_xy_loss']+result['moe_main_player_wall_loss'],places=7)
        self.assertAlmostEqual(result['train_loss'],sum(result[key]for key in
            ('moe_player_loss','moe_enemy_loss','moe_combat_loss','moe_router_loss')),places=7)
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all())for p in model.parameters()))


if __name__=='__main__':unittest.main()
