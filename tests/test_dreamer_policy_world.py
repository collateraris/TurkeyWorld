from dataclasses import replace
from unittest.mock import patch
import unittest

import torch

from test_rssm_fusion_integration import raw_dual,SMALL_FUSION
from test_rssm_combat_memory_migration import SMALL_COMBAT
from turkeyworld.models import NeuralWorldModel
from turkeyworld.dreamer_policy_world import RSSMImaginationAdapter,WorldState
from turkeyworld.dreamer_reward_continue import DreamerRewardContinue


def fixture(count=2):
    config=replace(raw_dual().config,rssm_fusion_moe_enabled=True,combat_rssm_enabled=True,
        **SMALL_FUSION,**SMALL_COMBAT)
    world=NeuralWorldModel(config)
    state=torch.zeros(count,20);state[:,:2]=.2;state[:,3:6]=1
    state[:,8:10]=torch.tensor([.4,.3]);state[:,12:14]=torch.tensor([.6,1.])
    grid=torch.zeros(count,16,16,dtype=torch.long);grid[:,0]=1;grid[:,-1]=1;grid[:,:,0]=1;grid[:,:,-1]=1
    rgb=torch.rand(count,3,32,32)
    return world,state,grid,rgb


class DreamerPolicyWorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_actual_three_RSSM_observe_interface_and_complete_memory_feature_schema(self):
        world,state,grid,rgb=fixture();adapter=RSSMImaginationAdapter(world,sample_latents=False)
        initial=adapter.initialize(state,grid,rgb)
        self.assertEqual(initial.memory.shape[-1],world.vjepa_backend.memory_size)
        self.assertEqual(adapter.feature_dim,20+initial.memory.shape[-1])
        parts=world.vjepa_backend.split_all_memory(initial.memory)
        self.assertEqual(len(parts),3);self.assertTrue(all(p is not None and bool(p.abs().sum())for p in parts))
        next_state=state.clone();next_state[:,:2]+=.01
        observed=adapter.observe(initial,torch.zeros(2,4),next_state,rgb)
        self.assertTrue(torch.equal(observed.state,next_state))
        self.assertFalse(torch.equal(observed.memory,initial.memory))
        self.assertTrue(torch.equal(adapter.features(observed)[:,:20],next_state))
        self.assertTrue(torch.equal(adapter.features(observed)[:,20:],observed.memory))
        self.assertTrue(all(not p.requires_grad for p in world.parameters()))

    def test_prior_imagination_has_no_observation_Source_or_renderer_calls_and_keeps_h_z(self):
        world,state,grid,rgb=fixture();heads=DreamerRewardContinue(20+world.vjepa_backend.memory_size,hidden=16)
        adapter=RSSMImaginationAdapter(world,heads,sample_latents=False);initial=adapter.initialize(state,grid,rgb)
        memories=[];results=[];original=world.transition
        def transition(s,a,memory_map,memory):
            memories.append(memory.clone());out=original(s,a,memory_map,memory);results.append(out['memory'].clone());return out
        with patch.object(world,'transition',side_effect=transition),\
             patch.object(world.vjepa_backend,'observe_step',side_effect=AssertionError('No future Source RGB')) ,\
             patch.object(world,'initialize_from_image',side_effect=AssertionError('No reset during imagination')),\
             patch.object(world,'render',side_effect=AssertionError('No renderer in latent environment')),\
             patch('turkeyworld.engine.Game.step',side_effect=AssertionError('No Source game in imagination')):
            tape=adapter.imagine(initial,lambda features:torch.zeros(len(features),4),horizon=3)
        self.assertEqual(tape['features'].shape,(2,4,adapter.feature_dim));self.assertEqual(tape['actions'].shape,(2,3,4))
        self.assertTrue(torch.equal(memories[0],initial.memory))
        self.assertTrue(torch.equal(memories[1],results[0]));self.assertTrue(torch.equal(memories[2],results[1]))
        self.assertTrue(all(not value.requires_grad for key,value in tape.items()if torch.is_tensor(value)))
        self.assertTrue(all(p.grad is None for p in world.parameters()));self.assertTrue(all(p.grad is None for p in heads.parameters()))

    def test_source_observation_tape_has_initial_plus_each_next_feature_and_selects_per_actor(self):
        world,state,grid,rgb=fixture();adapter=RSSMImaginationAdapter(world,sample_latents=False)
        states=state[:,None].repeat(1,4,1);states[:,1:,:2]+=.01
        batch=dict(states=states,map=grid,context_frames=rgb[:,None],target_states=states[:,1:],
            target_frames=rgb[:,None].repeat(1,3,1,1,1),actions=torch.zeros(2,3,4))
        features,observed=adapter.observe_batch(batch,return_states=True)
        self.assertEqual(features.shape,(2,4,adapter.feature_dim));self.assertEqual(len(observed),4)
        selected=adapter.select_observed(observed,torch.tensor([0,3]))
        self.assertTrue(torch.equal(selected.state[0],observed[0].state[0]))
        self.assertTrue(torch.equal(selected.memory[1],observed[3].memory[1]))
        for index,current in enumerate(observed):self.assertTrue(torch.equal(features[:,index],adapter.features(current)))

    def test_sample_latents_does_not_turn_structured_readout_training_on_and_restores_modes(self):
        world,state,grid,rgb=fixture();adapter=RSSMImaginationAdapter(world,sample_latents=True)
        initial=adapter.initialize(state,grid,rgb);original=world.transition;flags=[]
        def transition(*args):
            flags.append((world.training,world.vjepa_backend.training,world.core.training));return original(*args)
        modes=[m.training for m in world.modules()]
        with patch.object(world,'transition',side_effect=transition):adapter.transition_prior(initial,torch.zeros(2,4))
        self.assertEqual(flags,[(False,False,True)]);self.assertEqual(modes,[m.training for m in world.modules()])

    def test_invalid_partial_memory_action_and_observation_alignment_fail_closed(self):
        world,state,grid,rgb=fixture();adapter=RSSMImaginationAdapter(world,sample_latents=False);initial=adapter.initialize(state,grid,rgb)
        with self.assertRaises(ValueError):adapter.features(replace(initial,memory=initial.memory[:,:-1]))
        with self.assertRaises(ValueError):adapter.transition_prior(initial,torch.full((2,4),3.))
        with self.assertRaises(ValueError):adapter.step(initial,torch.zeros(2,4))
        with self.assertRaises(ValueError):adapter.initialize(state,grid,rgb[:,None].repeat(1,2,1,1,1))
        with self.assertRaises(ValueError):adapter.select_observed([initial],torch.tensor([0,1]))

    def test_interleaved_batches_restore_each_static_map_for_prior_and_posterior(self):
        world,state,grid,rgb=fixture();grid[:,5,5]=1
        adapter=RSSMImaginationAdapter(world,sample_latents=False)
        main=adapter.initialize(state,grid,rgb)
        aux_grid=grid[:1].clone();aux_grid[:,5,5]=0;aux_grid[:,7,7]=1
        aux=adapter.initialize(state[:1],aux_grid,rgb[:1])
        self.assertEqual(world.vjepa_backend.renderer._spawn_walkable.shape[0],1)
        original=world.transition;seen=[]
        def transition(*args):
            seen.append(world.vjepa_backend.renderer._spawn_walkable.clone());return original(*args)
        with patch.object(world,'transition',side_effect=transition):adapter.transition_prior(main,torch.zeros(2,4))
        self.assertTrue(torch.equal(seen[-1],grid==0))
        adapter.transition_prior(aux,torch.zeros(1,4))
        original_observe=world.vjepa_backend.observe_step
        def observe(*args,**kwargs):
            seen.append(world.vjepa_backend.renderer._spawn_walkable.clone());return original_observe(*args,**kwargs)
        with patch.object(world.vjepa_backend,'observe_step',side_effect=observe):
            adapter.observe(main,torch.zeros(2,4),state,rgb)
        self.assertTrue(torch.equal(seen[-1],grid==0))
        selected=adapter.select_observed([main,main],torch.tensor([1,0]))
        self.assertTrue(torch.equal(selected.grid,grid))


if __name__=='__main__':unittest.main()
