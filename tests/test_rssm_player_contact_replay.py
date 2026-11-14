"""Bounded TRAIN replay, own memory and post-rollout Source supervision."""
import copy
from dataclasses import dataclass
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from turkeyworld.engine import Game
from turkeyworld.structured_state import player_command_delta
from turkeyworld.rssm_player_contact_replay import (PlayerContactReplaySampler,
    capture_player_contact_candidate, contact_replay_backward,
    select_player_contact_branches, _axis_path_clear, _flatten, _scout, _game_at)


@dataclass
class Config:
    navigation_only: bool = False
    simulation_hz: int = 30


class OwnWorld(nn.Module):
    def __init__(self):
        super().__init__()
        self.config=Config();self.vjepa_backend=SimpleNamespace(config=Config())
        self.axis_logits=nn.Parameter(torch.zeros(2))
        self.core=nn.Linear(4,4,bias=False)
        self.map_encoder=nn.Linear(1,2)
        self.renderer=nn.Linear(1,1).requires_grad_(False)
        with torch.no_grad(): self.core.weight.copy_(torch.eye(4)*.01)
        self.calls=[];self.initializations=[];self.hidden_graph=[]

    def encode_map(self,grid):
        return self.map_encoder(grid.float().mean((1,2))[:,None])

    def initialize_from_image(self,frames,states,features):
        self.initializations.append((frames.detach().clone(),states.detach().clone()))
        if frames.shape[1]!=1: raise AssertionError('Only one genuine frame is observed')
        return .3+torch.arange(4,device=states.device)[None].expand(len(states),-1)/5

    def render(self,*args): raise AssertionError('No renderer during player contact learning')

    def transition(self,state,action,features,memory):
        self.calls.append((state.detach().clone(),memory.detach().clone(),self.config.navigation_only))
        self.hidden_graph.append(memory.requires_grad)
        logits=self.axis_logits[None]+.01*memory[:,:2]
        delta=player_command_delta(state,action)*logits.sigmoid()
        return dict(state=torch.cat([state[:,:2]+delta,state[:,2:]],-1),
            memory=memory+.05*self.core(memory),player_pass_logits=logits,
            respawn_flags=torch.zeros((len(state),1),dtype=torch.bool,device=state.device))


def source_tapes():
    grid=np.zeros((16,16),np.uint8);grid[[0,-1],:]=1;grid[:,[0,-1]]=1
    grid[5,6]=1;grid[6,5]=1
    poses=((3.5,3.5),(6.5,4.795),(5.795,5.5),(5.795,5.795))
    states=[];actions=np.repeat(np.array([[1.,0.,0.,0.]],np.float32),32,axis=0)
    for x,y in poses:
        game=Game(seed=0,endless=True,enemies_enabled=False);game.map=grid.copy()
        state=game.get_state();state['player'].update(x=x,y=y,angle=np.pi/4)
        game.set_state(state);tape=[game.state_vector().copy()]
        for action in actions: game.step(action,dt=1/30);tape.append(game.state_vector().copy())
        states.append(np.asarray(tape,np.float32))
    return np.stack(states),np.repeat(actions[None],4,axis=0),np.repeat(grid[None],4,axis=0)


class Data:
    def __init__(self,directory,*,split='train'):
        self.split=split;self.rng=np.random.default_rng(77);self.episodes=[];self.batch_calls=[]
        states,self.actions,self.maps=source_tapes();self.states=torch.tensor(states)
        self.frames=torch.zeros((4,33,3,4,4));self.frames[:,0]=.1
        for index in range(4):
            path=Path(directory)/f'episode_{index}.npz'
            np.savez(path,states=states[index],actions=self.actions[index],split=split)
            self.episodes.append({'path':str(path)})

    def batch(self,actors,*,context,horizon,indices):
        self.batch_calls.append(indices.copy());ids=indices[:,0]
        assert actors==4 and context==1 and horizon==32 and not indices[:,1].any()
        return dict(states=self.states[ids].clone(),frames=self.frames[ids].clone(),
            actions=torch.tensor(self.actions[ids]),target_states=self.states[ids,1:].clone(),
            map=torch.tensor(self.maps[ids]))


class PlayerContactReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.data=Data(self.directory.name)

    def test_four_cohorts_private_rng_exact_resume_and_inventory_pin(self):
        original=copy.deepcopy(self.data.rng.bit_generator.state)
        sampler=PlayerContactReplaySampler(self.data)
        batch,first=sampler.batch();saved=sampler.state_dict();_,second=sampler.batch()
        restored=PlayerContactReplaySampler(self.data);restored.restore_state(saved)
        _,resumed=restored.batch()
        self.assertEqual(second,resumed)
        self.assertEqual(set(first['cohorts']),{'both_free','x_free_y_blocked','x_blocked_y_free','both_blocked'})
        self.assertEqual(batch['actions'].shape,(4,32,4));self.assertEqual(original,self.data.rng.bit_generator.state)
        self.assertEqual(restored.draws,8)
        broken=copy.deepcopy(saved);broken['draws']=3
        with self.assertRaises(ValueError): restored.restore_state(broken)
        pinned=Path(self.directory.name)/'inventory.json';PlayerContactReplaySampler(self.data,cache_path=pinned)
        self.data.episodes.reverse()
        with self.assertRaisesRegex(ValueError,'inventory changed'):
            PlayerContactReplaySampler(self.data,cache_path=pinned)

    def test_refuses_val_or_enemy_tapes(self):
        self.data.split='val'
        with self.assertRaisesRegex(ValueError,'TRAIN'):PlayerContactReplaySampler(self.data)
        self.data.split='train';path=Path(self.data.episodes[0]['path'])
        states=self.data.states[0].numpy().copy();states[:,12:14]=1
        np.savez(path,states=states,actions=self.data.actions[0],split='train')
        with self.assertRaisesRegex(ValueError,'no-enemy'):PlayerContactReplaySampler(self.data)

    def test_source_queries_follow_scout_and_branches_keep_actual_hidden_memory(self):
        world=OwnWorld();batch,_=PlayerContactReplaySampler(self.data).batch()
        before={name:value.detach().clone() for name,value in world.named_parameters()}
        source_times=[];real_step=Game.step
        def step(game,*args,**kwargs):
            source_times.append(len(world.calls));return real_step(game,*args,**kwargs)
        with patch.object(Game,'step',new=step),patch.object(Game,'render',side_effect=AssertionError('No future RGB')):
            report=contact_replay_backward(world,batch,horizon=32)
        self.assertEqual(len(world.initializations),1);self.assertGreaterEqual(min(source_times),32)
        self.assertEqual(report['player_contact_tbptt'],64)
        self.assertEqual(report['scout_forward_transitions'],128)
        self.assertLessEqual(report['branches'],4);self.assertLessEqual(report['source_probes'],32)
        self.assertEqual(report['state_resets_after_prefix'],0);self.assertEqual(report['memory_resets_after_prefix'],0)
        self.assertFalse(world.config.navigation_only);self.assertFalse(world.vjepa_backend.config.navigation_only)
        self.assertGreater(float(world.axis_logits.grad.norm()),0);self.assertGreater(float(world.core.weight.grad.norm()),0)
        self.assertIsNone(world.map_encoder.weight.grad);self.assertIsNone(world.renderer.weight.grad)
        for name,value in world.named_parameters():torch.testing.assert_close(value,before[name],rtol=0,atol=0)
        # Every chosen H32 contact branch starts at a detached real prefix;
        # its memory graph stays connected on transitions17..32 (no H16 cut).
        cursor=32
        for _ in range(report['branches']):
            self.assertFalse(world.hidden_graph[cursor]);self.assertTrue(world.hidden_graph[cursor+16])
            cursor+=32
        self.assertEqual(cursor,len(world.calls))
        self.assertEqual(report['direct_source_attempts'],report['forward_transitions'])
        self.assertGreater(report['direct_source_valid'],0)
        self.assertTrue(all(count==len(world.calls)for count in source_times[-report['direct_source_valid']:]))
        for axis in range(2):
            self.assertGreater(report['axis_group_counts']['free'][axis],0)
            self.assertGreater(report['axis_group_counts']['blocked'][axis],0)
        self.assertIn('32',report['xy_offsets_tiles_mean'])

    def test_future_source_targets_and_images_never_change_own_inputs(self):
        batch,_=PlayerContactReplaySampler(self.data).batch();altered=copy.deepcopy(batch)
        altered['states'][:,1:,:2]+=.01;altered['target_states'][...,:2]+=.02;altered['frames'][:,1:]=.99
        first,second=OwnWorld(),OwnWorld();second.load_state_dict(first.state_dict())
        _scout(first,batch,None);_scout(second,altered,None)
        for left,right in zip(first.calls,second.calls):
            torch.testing.assert_close(left[0],right[0],rtol=0,atol=0)
            torch.testing.assert_close(left[1],right[1],rtol=0,atol=0)
        torch.testing.assert_close(first.initializations[0][0],second.initializations[0][0],rtol=0,atol=0)

    def test_source_trajectory_rooted_at_own_prefix_and_horizon_aware_contact_selection(self):
        world=OwnWorld();batch,_=PlayerContactReplaySampler(self.data).batch()
        candidates=_scout(world,batch,None)
        selected,report=select_player_contact_branches(candidates,horizon=4)
        self.assertEqual(len(selected),4)
        for row in selected:
            np.testing.assert_allclose(row['labels'][0],row['item']['state'][row['actor']].numpy(),rtol=0,atol=2e-7)
            game=_game_at(row['state_numpy'],row['map_numpy'],navigation_only=True)
            for tick,action in enumerate(row['item']['actions'][row['actor'],:4].numpy()):
                game.step(action,dt=1/30)
                np.testing.assert_allclose(row['labels'][tick+1],game.state_vector(),rtol=0,atol=2e-7)
        self.assertLessEqual(report['source_probes'],32)

    def test_h4_prioritizes_prefix_four_ticks_before_recorded_event(self):
        world=OwnWorld();batch=self.data.batch(4,context=1,horizon=32,
            indices=np.array([[index,0]for index in range(4)]))
        candidates=[]
        # The free actor has no first-wall override. The event hint may guide
        # which own snapshot is queried, but cannot replace its Source label.
        for item in _scout(world,batch,None):
            actor=int((batch['states'][:,0,0]*16-3.5).abs().argmin())
            one=capture_player_contact_candidate(item['step'],item['state'][actor:actor+1],
                item['memory'][actor:actor+1],item['actions'][actor:actor+1],
                item['map'][actor:actor+1],item['prediction'][actor:actor+1],stream='contact_short')
            one['source_event_offsets']=[23];candidates.append(one)
        selected,_=select_player_contact_branches(candidates,horizon=4)
        self.assertEqual(selected[0]['step'],20)
        self.assertEqual(selected[0]['cohort'],'both_free')

    def test_invalid_own_state_is_skipped_without_projection_and_stream_ids_are_local(self):
        world=OwnWorld();batch,_=PlayerContactReplaySampler(self.data).batch();good=_scout(world,batch,None)[0]
        invalid={**good,'state':good['state'].clone(),'stream':'combat'}
        invalid['state'][:,:2]=torch.tensor([6.5,5.5])/16
        selected,stats=select_player_contact_branches([invalid],horizon=4)
        self.assertFalse(selected);self.assertEqual(stats['source_probes'],4)
        self.assertEqual(stats['source_skip_counts'],{'invalid_contact_own_state_geometry':4})
        good['step']=48
        rows,_=_flatten([invalid,good])
        self.assertTrue(any(row['item']['stream']=='contact_short' and row['step']==48 for row in rows))

    def test_axis_ordered_path_checks_legal_slide_around_corner(self):
        grid=np.zeros((16,16),np.uint8);grid[5,5]=1
        # X first passes above the wall, then Y descends to its right. A
        # straight diagonal would cross the wall, but Source does not use it.
        self.assertTrue(_axis_path_clear(grid,np.array([4.5,4.5]),np.array([6.5,6.5])))
        from turkeyworld.runtime_checks import circle_is_clear
        chord=np.linspace([4.5,4.5],[6.5,6.5],25)
        self.assertFalse(all(circle_is_clear(grid,point,.2)for point in chord))


if __name__=='__main__':unittest.main()
