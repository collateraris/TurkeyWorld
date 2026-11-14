from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from turkeyworld.rssm_coordinate_gradients import COORDINATE_GRADIENT_PROTOCOL,integrate_coordinate_gradients
from turkeyworld.rssm_moe_enemy_motion import enemy_motion_backward
from turkeyworld.rssm_moe_training import moe_dynamics_backward,COMBAT_COMPONENTS
from turkeyworld.rssm_npc_geometry_objectives import NPC_NAVIGATION_PROTOCOL
from turkeyworld.rssm_objectives import _initialize
from turkeyworld.rssm_player_geometry_objectives import PLAYER_GEOMETRY_PROTOCOL
from test_rssm_moe_enemy_motion import EnemyModel,quiet_batch
from test_rssm_moe_training import TinyModel,data as combat_batch


def auxiliary_zero(*args,**kwargs):
    return dict(player=0.,enemy=0.,combat=0.,horizon=16,actors=4,
        direct_source_supervision_available=True,direct_per_tick_counts=[],
        direct_spawn_anchor_counts={},**{key:0. for key in COMBAT_COMPONENTS})


def approach_wall(batch):
    for name in ('states','target_states'):
        value=batch[name];value[...,8]=7.5/16;value[...,9]=8.5/16
    batch['map'][:,8,8]=1
    return batch


def add_wall_crossing(model,*,respawn_tick=None):
    original=model.transition;outputs=[]
    def transition(*args):
        result=original(*args);offset=torch.zeros_like(result['state']);offset[:,8]=.025
        result['state']=result['state']+offset
        if respawn_tick is not None and len(outputs)==respawn_tick:
            result['respawn_flags']=torch.ones_like(result['respawn_flags'])
        outputs.append({key:value.detach().clone()for key,value in result.items()if torch.is_tensor(value)})
        return result
    model.transition=transition
    return outputs


class NPCLongObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_explicit_none_preserves_legacy_main_quiet_values_and_gradients(self):
        torch.manual_seed(174);a=EnemyModel();b=deepcopy(a);data=quiet_batch(8)
        old=enemy_motion_backward(a,data,context=2)
        same=enemy_motion_backward(b,data,context=2,npc_navigation_protocol=None)
        self.assertEqual(old,same)
        for p,q in zip(a.parameters(),b.parameters()):
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
        x=TinyModel();y=deepcopy(x)
        with patch('turkeyworld.rssm_moe_training._own_branch_backward',side_effect=auxiliary_zero):
            old=moe_dynamics_backward(x,combat_batch(4),combat_batch(4))
            same=moe_dynamics_backward(y,combat_batch(4),combat_batch(4),npc_navigation_protocol=None)
        self.assertEqual(old,same)
        for p,q in zip(x.parameters(),y.parameters()):
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)

    def test_new_quiet_gradient_is_global_xy_while_recorded_velocity_direction_are_diagnostics(self):
        torch.manual_seed(175);model=EnemyModel();manual=deepcopy(model);legacy=deepcopy(model)
        data=quiet_batch(8)
        result=enemy_motion_backward(model,data,context=2,enemy_weight=1.,router_weight=0.,
            npc_navigation_protocol=NPC_NAVIGATION_PROTOCOL)
        previous=data['states'][:,1].detach();grid=data['map']
        with torch.no_grad():features=manual.encode_map(grid).detach()
        memory=_initialize(manual,data['frames'],data['states'],features,2,2)
        predictions=[]
        for tick in range(8):
            step=manual.transition(previous.detach(),data['actions'][:,tick],features,memory)
            step=integrate_coordinate_gradients(previous,step,COORDINATE_GRADIENT_PROTOCOL)
            previous,memory=step['state'],step['memory'];predictions.append(previous[:,8:10])
        objective=F.smooth_l1_loss(torch.stack(predictions,1)*16,data['target_states'][...,8:10]*16,
                                  reduction='sum')/(2*len(previous)*8)
        objective.backward()
        self.assertAlmostEqual(result['enemy_motion_enemy_loss'],float(objective.detach()),places=7)
        self.assertEqual(result['enemy_motion_npc_wall_loss'],0.)
        self.assertEqual(result['enemy_motion_velocity_weight'],0.)
        self.assertEqual(result['enemy_motion_direction_weight'],0.)
        self.assertGreater(result['enemy_motion_raw_velocity_loss'],0.)
        self.assertGreater(result['enemy_motion_raw_direction_loss'],0.)
        self.assertTrue(result['enemy_motion_recorded_velocity_direction_diagnostic_only'])
        for p,q in zip(model.parameters(),manual.parameters()):
            pg=torch.zeros_like(p)if p.grad is None else p.grad
            qg=torch.zeros_like(q)if q.grad is None else q.grad
            torch.testing.assert_close(pg,qg,atol=1e-7,rtol=1e-5)
        old=enemy_motion_backward(legacy,data,context=2,enemy_weight=1.,router_weight=0.)
        self.assertGreater(old['enemy_motion_enemy_loss'],result['enemy_motion_enemy_loss'])
        self.assertFalse(torch.allclose(model.enemy_head.weight.grad,legacy.enemy_head.weight.grad))

    def test_quiet_wall_is_actor_weighted_once_and_preserves_actual_rollout_memory_and_spawn_flags(self):
        torch.manual_seed(176);model=EnemyModel();legacy=deepcopy(model)
        data=approach_wall(quiet_batch(8));outputs=add_wall_crossing(model,respawn_tick=2)
        add_wall_crossing(legacy,respawn_tick=2)
        for tensor in data.values():tensor.requires_grad_()
        with patch('turkeyworld.engine.Game.step',side_effect=AssertionError('No Source query in long geometry')):
            report=enemy_motion_backward(model,data,context=2,enemy_weight=2/3,router_weight=0.,
                npc_navigation_protocol=NPC_NAVIGATION_PROTOCOL)
            old=enemy_motion_backward(legacy,data,context=2,enemy_weight=2/3,router_weight=0.)
        self.assertGreater(report['enemy_motion_npc_wall_loss'],0.)
        self.assertAlmostEqual(report['enemy_motion_enemy_loss'],
            (2/3)*(report['enemy_motion_raw_xy_loss']+report['enemy_motion_raw_npc_wall_loss']),places=7)
        self.assertEqual(report['enemy_motion_npc_geometry']['npc_wall_predicted_spawn_path_exclusions'],len(data['states']))
        self.assertEqual(report['enemy_motion_npc_geometry']['npc_wall_source_alive_transitions'],16)
        self.assertEqual(len(outputs),8)
        self.assertTrue(all(torch.equal(a[0],b[0])and torch.equal(a[1],b[1])
                            for a,b in zip(model.calls,legacy.calls)))
        self.assertAlmostEqual(report['enemy_motion_raw_xy_loss'],old['enemy_motion_raw_xy_loss'],places=7)
        self.assertGreater(float(model.enemy_head.weight.grad.abs().sum()),0.)
        self.assertTrue(all(tensor.grad is None for tensor in data.values()))
        self.assertTrue(all(p.grad is None for p in model.frozen_renderer.parameters()))
        json.dumps(report,allow_nan=False)

    def test_main_npc_wall_is_counted_once_before_enemy_weight_with_own_prefix_captures(self):
        torch.manual_seed(177);model=TinyModel();scaled=deepcopy(model)
        data=approach_wall(combat_batch(32));aux=combat_batch(16)
        outputs=add_wall_crossing(model,respawn_tick=2);add_wall_crossing(scaled,respawn_tick=2)
        snapshots=[]
        player={'geometry':PLAYER_GEOMETRY_PROTOCOL,'tbptt':64}
        with patch('turkeyworld.rssm_moe_training._own_branch_backward',side_effect=auxiliary_zero),\
             patch('turkeyworld.engine.Game.step',side_effect=AssertionError('No Source query in long geometry')):
            report=moe_dynamics_backward(model,data,aux,chunk_size=64,player_navigation_protocol=player,
                npc_navigation_protocol=NPC_NAVIGATION_PROTOCOL,main_enemy_weight=1.,
                player_contact_callback=lambda tick,state,memory,*args:snapshots.append((tick,memory.detach().clone())))
            quarter=moe_dynamics_backward(scaled,data,aux,chunk_size=64,player_navigation_protocol=player,
                npc_navigation_protocol=NPC_NAVIGATION_PROTOCOL,main_enemy_weight=.25)
        self.assertGreater(report['moe_main_npc_wall_loss'],0.)
        self.assertAlmostEqual(report['moe_enemy_loss'],report['moe_main_enemy_xy_loss']+report['moe_main_npc_wall_loss'],places=6)
        self.assertAlmostEqual(quarter['moe_main_npc_wall_loss'],report['moe_main_npc_wall_loss']*.25,places=7)
        self.assertAlmostEqual(report['train_loss'],sum(report[key]for key in
            ('moe_player_loss','moe_enemy_loss','moe_combat_loss','moe_router_loss')),places=7)
        torch.testing.assert_close(scaled.e_head.weight.grad,model.e_head.weight.grad*.25,atol=1e-7,rtol=1e-5)
        self.assertEqual([row[0]for row in snapshots],[0,16])
        self.assertTrue(torch.equal(snapshots[1][1],outputs[15]['memory']))
        self.assertEqual(report['moe_npc_geometry']['npc_wall_source_alive_transitions'],64)
        self.assertEqual(report['moe_npc_geometry']['npc_wall_predicted_spawn_path_exclusions'],2)
        json.dumps(report,allow_nan=False)

    def test_malformed_explicit_npc_revision_is_rejected_without_silently_muting_legacy_terms(self):
        protocol=deepcopy(NPC_NAVIGATION_PROTOCOL);protocol['recorded_velocity_direction']=True
        model=EnemyModel()
        with self.assertRaises(ValueError):enemy_motion_backward(model,quiet_batch(4),context=2,npc_navigation_protocol=protocol)
        self.assertEqual(model.calls,[])


if __name__=='__main__':unittest.main()
