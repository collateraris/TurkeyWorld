"""Own-memory NPC physics targets, bounded probes and no neural Source feedback."""
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from turkeyworld.engine import Game
from turkeyworld.rssm_enemy_local_recovery import(ENEMY_LOCAL_RECOVERY_PROTOCOL,
    capture_enemy_local_candidate,select_enemy_local_branches,enemy_local_recovery_backward,_class_weights,
    _candidate_rows,_source_label)


@dataclass
class Config:
    navigation_only:bool=False
    simulation_hz:int=30


class EnemyWorld(nn.Module):
    def __init__(self):
        super().__init__();self.config=Config();self.vjepa_backend=SimpleNamespace(config=Config())
        self.head=nn.Parameter(torch.tensor([1.,.3]));self.core=nn.Linear(4,4,bias=False)
        with torch.no_grad():self.core.weight.copy_(torch.eye(4)*.01)
        self.map_encoder=nn.Linear(1,2);self.renderer=nn.Linear(1,1).requires_grad_(False)
        self.calls=[];self.graph=[];self.force_spawn=False
    def encode_map(self,grid):return self.map_encoder(grid.float().mean((1,2))[:,None])
    def initialize_from_image(self,*args):raise AssertionError('Recovery never observes RGB or resets h/z')
    def render(self,*args):raise AssertionError('No renderer in NPC recovery')
    def transition(self,state,action,features,memory):
        self.calls.append((state.detach().clone(),memory.detach().clone()));self.graph.append(memory.requires_grad)
        intent=self.head[None]+.05*memory[:,:2]
        delta=F.normalize(intent,dim=-1)*(.04/16)
        following=torch.cat([state[:,:8],state[:,8:10]+delta,state[:,10:]],-1)
        spawn=torch.full((len(state),1),self.force_spawn and len(self.calls)%8==2,dtype=torch.bool)
        return dict(state=following,memory=memory+.05*self.core(memory),respawn_flags=spawn,
            proposed_npc_displacement=(.1*intent)[:,None])


def fixture(world,*,steps=64):
    grid=np.zeros((16,16),np.uint8);grid[[0,-1],:]=1;grid[:,[0,-1]]=1
    states=[]
    for index in range(2):
        game=Game(seed=0,endless=True,enemies_enabled=True);game.map=grid.copy()
        initial=game.get_state();initial['player'].update(x=3.5,y=3.5,angle=0)
        initial['enemies'][0].update(x=8.5+index,y=7.5,hp=100,alive=True)
        game.set_state(initial);states.append(torch.tensor(game.state_vector()))
    state=torch.stack(states);maps=torch.tensor(grid)[None].expand(2,-1,-1).clone()
    actions=torch.zeros((2,steps+32,4));items=[]
    for stream,base in(('combat',.3),('enemy_motion',1.3)):
        current=state.clone();memory=torch.arange(8).reshape(2,4).float()/7+base;snapshots={};pred=[]
        with torch.no_grad():
            features=world.encode_map(maps)
            for tick in range(steps):
                if tick%16==0:snapshots[tick]=(current.clone(),memory.clone())
                result=world.transition(current,actions[:,tick],features,memory)
                current,memory=result['state'],result['memory'];pred.append(current.clone())
        for tick,(current,memory)in snapshots.items():
            items.append(capture_enemy_local_candidate(tick,current,memory,actions[:,tick:tick+32],maps,
                torch.stack(pred[tick:tick+16],1),stream=stream))
    world.calls.clear();world.graph.clear()
    return items


class EnemyLocalRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_default_off_does_not_call_model_or_Source_and_protocol_is_strict(self):
        world=EnemyWorld()
        with patch.object(Game,'step',side_effect=AssertionError('Default off')):
            result=enemy_local_recovery_backward(world,[],protocol=None)
        self.assertFalse(result['enemy_local_active']);self.assertEqual(result['enemy_local_loss'],0)
        self.assertFalse(world.calls)
        malformed=deepcopy(ENEMY_LOCAL_RECOVERY_PROTOCOL);malformed['version']=True
        with self.assertRaises(ValueError):enemy_local_recovery_backward(world,[],protocol=malformed)

    def test_selection_two_quiet_two_combat_with_initial_and_retained_memory_anchors(self):
        world=EnemyWorld();candidates=fixture(world)
        with patch.object(Game,'render',side_effect=AssertionError('No Source images')):
            selected,metrics=select_enemy_local_branches(candidates,horizon=16)
        self.assertEqual(len(selected),4);self.assertLessEqual(metrics['source_probes'],32)
        self.assertEqual(metrics['selected_streams'].count('combat'),2)
        self.assertEqual(metrics['selected_streams'].count('enemy_motion'),2)
        self.assertEqual(metrics['selected_initial_prefixes'],2);self.assertEqual(metrics['selected_nonzero_prefixes'],2)
        self.assertTrue(all(row['step']<row['first_fault']if row['first_fault']is not None else True for row in selected))
        self.assertTrue(all(value>.2 for value in metrics['selected_first_velocity_errors']))

    def test_full32_graph_keeps_actual_memory_and_local_labels_follow_all_neural_branches(self):
        world=EnemyWorld();candidates=fixture(world);before={n:p.detach().clone()for n,p in world.named_parameters()}
        original_step=Game.step;source_times=[]
        def Source_step(game,*args,**kwargs):source_times.append(len(world.calls));return original_step(game,*args,**kwargs)
        with patch.object(Game,'step',new=Source_step),patch.object(Game,'render',side_effect=AssertionError('No future RGB')):
            report=enemy_local_recovery_backward(world,candidates,horizon=32,protocol=ENEMY_LOCAL_RECOVERY_PROTOCOL)
        self.assertEqual(report['branches'],4);self.assertEqual(report['enemy_local_forward_transitions'],128)
        self.assertGreater(report['enemy_local_valid_source_labels'],0);self.assertEqual(report['branch_horizon_graph'],32)
        self.assertTrue(all(count in(0,128)for count in source_times))
        self.assertEqual(report['state_resets_after_prefix'],0);self.assertEqual(report['memory_resets_after_prefix'],0)
        self.assertTrue(report['full_own_memory_preserved']);self.assertEqual(report['new_observed_rgb_frames_after_prefix'],0)
        self.assertGreater(float(world.head.grad.norm()),0);self.assertGreater(float(world.core.weight.grad.norm()),0)
        self.assertIsNone(world.map_encoder.weight.grad);self.assertIsNone(world.renderer.weight.grad)
        for start in range(0,128,32):
            self.assertFalse(world.graph[start]);self.assertTrue(world.graph[start+16])
        selected,_=select_enemy_local_branches(candidates,horizon=32)
        for index,row in enumerate(selected):
            torch.testing.assert_close(world.calls[index*32][1],row['item']['memory'][row['actor']:row['actor']+1],rtol=0,atol=0)
        for name,parameter in world.named_parameters():torch.testing.assert_close(parameter,before[name],rtol=0,atol=0)
        self.assertAlmostEqual(report['enemy_local_loss'],.5*(report['velocity_loss']+report['speed_loss']
            +.25*report['direction_loss']+.1*report['trajectory_loss']),places=6)

    def test_illegal_anchors_are_skipped_without_repair_or_new_observation(self):
        world=EnemyWorld();items=fixture(world)
        for item in items:item['state'][:,8:10]=torch.tensor([0.,0.])
        with patch.object(world,'transition',side_effect=AssertionError('No legal prefix')):
            report=enemy_local_recovery_backward(world,items,horizon=8,protocol=ENEMY_LOCAL_RECOVERY_PROTOCOL)
        self.assertEqual(report['branches'],0);self.assertEqual(report['enemy_local_loss'],0)
        self.assertTrue(report['selection']['source_skip_counts'])

    def test_corner_path_is_x_then_y_instead_of_a_blocked_straight_chord(self):
        world=EnemyWorld();item=fixture(world)[0]
        item={key:value[:1].clone()if torch.is_tensor(value)else value for key,value in item.items()}
        item['map'][0,5,5]=1;item['state'][0,8:10]=torch.tensor([4.5,4.5])/16
        item['prediction'][0,:,8:10]=torch.tensor([6.5,6.5])/16
        rows,faults,_=_candidate_rows([item],16)
        self.assertEqual(len(rows),1);self.assertFalse(faults)
        from turkeyworld.runtime_checks import circle_is_clear
        self.assertFalse(all(circle_is_clear(item['map'][0].numpy(),p,.25)
            for p in np.linspace([4.5,4.5],[6.5,6.5],41)))

    def test_changed_future_Source_labels_change_loss_but_never_branch_inputs(self):
        first,second=EnemyWorld(),EnemyWorld();second.load_state_dict(first.state_dict());candidates=fixture(first)
        chosen,selection=select_enemy_local_branches(candidates,horizon=4)
        def changed(*args,**kwargs):
            label=_source_label(*args,**kwargs);label={**label,'target':label['target'].copy()}
            label['target'][8]+=.001
            return label
        with patch('turkeyworld.rssm_enemy_local_recovery.select_enemy_local_branches',return_value=(chosen,selection)):
            a=enemy_local_recovery_backward(first,candidates,horizon=4,protocol=ENEMY_LOCAL_RECOVERY_PROTOCOL)
            with patch('turkeyworld.rssm_enemy_local_recovery._source_label',side_effect=changed):
                b=enemy_local_recovery_backward(second,candidates,horizon=4,protocol=ENEMY_LOCAL_RECOVERY_PROTOCOL)
        self.assertNotEqual(a['enemy_local_loss'],b['enemy_local_loss'])
        for before,after in zip(first.calls,second.calls):
            torch.testing.assert_close(before[0],after[0],rtol=0,atol=0)
            torch.testing.assert_close(before[1],after[1],rtol=0,atol=0)

    def test_balancing_gives_first_rest_stream_and_stopped_moving_equal_mass(self):
        records=[dict(tick=0,stream='combat'),dict(tick=0,stream='combat')]
        classes=['moving','stopped']
        for stream in('combat','enemy_motion'):
            for tick in range(1,17):records.append(dict(tick=tick,stream=stream));classes.append('moving')
        weights=_class_weights(records,classes)
        self.assertAlmostEqual(float(weights.sum()),1)
        self.assertAlmostEqual(float(weights[:2].sum()),.5)
        self.assertAlmostEqual(float(weights[0]),.25);self.assertAlmostEqual(float(weights[1]),.25)
        self.assertAlmostEqual(float(weights[2:18].sum()),.25)
        self.assertAlmostEqual(float(weights[18:].sum()),.25)

    def test_retained_memory_selection_keeps_available_stationary_Source_class(self):
        world=EnemyWorld();items=fixture(world)
        def Source_with_stop(*args,**kwargs):
            label=_source_label(*args,**kwargs)
            if label['previous'][8]*16>10.2:
                label={**label,'target':label['target'].copy()}
                label['target'][8:10]=label['previous'][8:10]
            return label
        with patch('turkeyworld.rssm_enemy_local_recovery._source_label',side_effect=Source_with_stop):
            selected,report=select_enemy_local_branches(items,horizon=16)
        self.assertIn('stationary',report['selected_cohorts'])
        self.assertEqual(report['selected_initial_prefixes'],2)
        self.assertEqual(report['selected_nonzero_prefixes'],2)

    def test_predicted_spawn_only_excludes_local_velocity_not_own_source_XY(self):
        world=EnemyWorld();items=fixture(world);world.force_spawn=True
        report=enemy_local_recovery_backward(world,items,horizon=8,protocol=ENEMY_LOCAL_RECOVERY_PROTOCOL)
        self.assertGreater(report['predicted_spawn_velocity_exclusions'],0)
        self.assertLess(report['enemy_local_valid_source_labels'],report['enemy_local_forward_transitions'])
        self.assertGreater(report['trajectory_loss'],0);self.assertFalse(report['global_recorded_XY_objective_changed'])


if __name__=='__main__':unittest.main()
