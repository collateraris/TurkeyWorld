"""Own-memory branches: actual numerical carry, Source-only labels and bounded cost."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from turkeyworld.engine import Game
from turkeyworld.structured_state import player_command_delta
from turkeyworld.rssm_contact_branches import (CONTACT_BRANCH_DEFAULTS,AXIS_COHORTS,
    normalize_contact_branch_protocol,capture_contact_candidate,select_contact_branches,
    contact_branches_backward)
from turkeyworld.rssm_objectives import rssm_dynamics_backward
from test_rssm_objectives import SpyRSSM
from test_long_dynamics_training import batch as long_batch
from test_motion_recovery_training import source_batch
from test_rssm_world_backend import tiny_config
from turkeyworld.models import NeuralWorldModel
from turkeyworld.rssm_migration import configure_rssm_training


class BranchWorld(nn.Module):
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(navigation_only=True,simulation_hz=30)
        self.axis_logits=nn.Parameter(torch.zeros(2))
        self.core=nn.Linear(4,4,bias=False)
        with torch.no_grad(): self.core.weight.copy_(torch.eye(4)*.01)
        self.map_encoder=nn.Linear(1,2)
        self.renderer=nn.Linear(1,1).requires_grad_(False)
        self.calls=[]
    def encode_map(self,grid): return self.map_encoder(grid.float().mean((1,2))[:,None])
    def initialize_from_image(self,*args): raise AssertionError('Do not reset branch h/z or observe RGB')
    def render(self,*args): raise AssertionError('No renderer in dynamics branch')
    def transition(self,state,action,features,memory):
        self.calls.append((state.detach().clone(),memory.detach().clone()))
        logits=self.axis_logits[None]+.01*memory[:,:2]
        delta=player_command_delta(state,action)*logits.sigmoid()
        following=torch.cat([state[:,:2]+delta,state[:,2:]],-1)
        return {'state':following,'memory':memory+.05*self.core(memory),
                'player_pass_logits':logits,'raw_player_delta':delta}


def fixture(world,*,horizon=16):
    grid=np.zeros((16,16),np.uint8)
    grid[[0,-1],:]=1;grid[:,[0,-1]]=1
    grid[5,6]=1;grid[6,5]=1
    poses=[(3.5,3.5),(6.5,4.795),(5.795,5.5),(5.795,5.795)]
    states=[]
    for x,y in poses:
        game=Game(seed=0,endless=True,enemies_enabled=False);game.map=grid.copy()
        value=game.get_state();value['player'].update(x=x,y=y,angle=np.pi/4)
        game.set_state(value);states.append(torch.tensor(game.state_vector()))
    state=torch.stack(states);memory=torch.arange(16).reshape(4,4).float()/5+.3
    actions=torch.tensor([1.,0.,0.,0.])[None,None].expand(4,horizon,4).clone()
    maps=torch.tensor(grid)[None].expand(4,-1,-1).clone()
    with torch.no_grad():
        features=world.encode_map(maps);current=state;hidden=memory
        predictions=[]
        for i in range(horizon):
            result=world.transition(current,actions[:,i],features,hidden)
            current,hidden=result['state'],result['memory'];predictions.append(current)
    world.calls.clear()
    return capture_contact_candidate(0,state,memory,actions,maps,torch.stack(predictions,1))


class OwnContactBranchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_default_protocol_copy_validation_and_candidate_is_detached_immutable(self):
        full=normalize_contact_branch_protocol({'version':1})
        self.assertEqual(full,CONTACT_BRANCH_DEFAULTS)
        full['offsets'].append(32);self.assertEqual(CONTACT_BRANCH_DEFAULTS['offsets'],[1,4,8,16])
        for value in ({'version':2},{'max_branches':5},{'max_source_probes':65},
                      {'weight':float('nan')},{'canonical_atol':.01}):
            with self.assertRaises(ValueError): normalize_contact_branch_protocol(value)
        item=fixture(BranchWorld())
        original=item['memory'].clone()
        external=item['memory'].requires_grad_()
        captured=capture_contact_candidate(0,item['state'],external,item['actions'],item['map'],item['prediction'])
        with torch.no_grad(): external.add_(1)
        self.assertTrue(torch.equal(captured['memory'],original))
        self.assertFalse(captured['memory'].requires_grad)
        self.assertEqual(captured['memory_provenance'],'actualownprefix')

    def test_real_source_labels_balance_four_axis_outcomes_before_first_wall(self):
        world=BranchWorld();item=fixture(world)
        with patch.object(Game,'render',side_effect=AssertionError('No future Source RGB')):
            selected,stats=select_contact_branches([item],contact_branch_protocol={'version':1})
        self.assertEqual(len(selected),4)
        self.assertEqual({r['cohort'] for r in selected},set(AXIS_COHORTS))
        self.assertTrue(stats['axis_outcome_coverage_complete'])
        self.assertEqual(stats['source_probes'],4)
        for row in selected:
            if row['first_wall_step'] is not None:self.assertLess(row['step'],row['first_wall_step'])
            np.testing.assert_allclose(row['labels'][0],row['state_numpy'],atol=2e-7,rtol=0)
        self.assertFalse(np.array_equal(selected[0]['labels'][1],selected[0]['labels'][0]))

    def test_actual_hidden_memory_carry_finite_gradients_and_no_rgb_renderer_or_optimizer(self):
        world=BranchWorld();item=fixture(world)
        before={name:p.detach().clone() for name,p in world.named_parameters()}
        with patch.object(Game,'render',side_effect=AssertionError('No Source images')):
            result=contact_branches_backward(world,[item],contact_branch_protocol={'version':1})
        self.assertEqual(result['branches'],4);self.assertEqual(result['forward_transitions'],64)
        self.assertEqual(len(world.calls),16)
        self.assertEqual(result['new_observed_rgb_frames'],0)
        self.assertEqual(result['state_resets'],result['memory_resets'])
        self.assertTrue(result['prior_initial_memory_preserved'])
        selected,_=select_contact_branches([item],contact_branch_protocol={'version':1})
        expected=torch.stack([item['memory'][r['actor']] for r in selected])
        torch.testing.assert_close(world.calls[0][1],expected,rtol=0,atol=0)
        for a,b in zip(world.calls,world.calls[1:]):self.assertFalse(torch.equal(a[1],b[1]))
        self.assertGreater(float(world.axis_logits.grad.norm()),0.)
        self.assertGreater(float(world.core.weight.grad.norm()),0.)
        self.assertIsNone(world.map_encoder.weight.grad);self.assertIsNone(world.renderer.weight.grad)
        for name,p in world.named_parameters():
            torch.testing.assert_close(p,before[name],rtol=0,atol=0)
            if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all())
        for offset in (1,4,8,16):self.assertIn(f'xy_offset{offset}_tiles_mean',result)
        self.assertAlmostEqual(result['loss'],.25*result['unweighted_loss'])

    def test_future_labels_change_loss_but_never_branch_state_or_memory_inputs(self):
        first,second=BranchWorld(),BranchWorld();second.load_state_dict(first.state_dict())
        item=fixture(first);chosen,stats=select_contact_branches([item],contact_branch_protocol={'version':1})
        altered=[{**r,'labels':r['labels'].copy()} for r in chosen]
        for row in altered:row['labels'][1:,:2]+=.001
        with patch('turkeyworld.rssm_contact_branches.select_contact_branches',return_value=(chosen,stats)):
            a=contact_branches_backward(first,[item],contact_branch_protocol={'version':1})
        with patch('turkeyworld.rssm_contact_branches.select_contact_branches',return_value=(altered,stats)):
            b=contact_branches_backward(second,[item],contact_branch_protocol={'version':1})
        self.assertNotEqual(a['loss'],b['loss'])
        for x,y in zip(first.calls,second.calls):
            torch.testing.assert_close(x[0],y[0],rtol=0,atol=0)
            torch.testing.assert_close(x[1],y[1],rtol=0,atol=0)

    def test_invalid_anchor_skipped_without_repair_and_source_probes_are_bounded(self):
        world=BranchWorld();item=fixture(world)
        invalid={**item,'state':item['state'].clone()}
        invalid['state'][:,:2]=torch.tensor([6.5,5.5])/16
        world.axis_logits.grad=torch.ones_like(world.axis_logits)
        before=world.axis_logits.grad.clone()
        report=contact_branches_backward(world,[invalid],contact_branch_protocol={'version':1})
        self.assertEqual(report['branches'],0);self.assertEqual(report['loss'],0)
        torch.testing.assert_close(world.axis_logits.grad,before,rtol=0,atol=0)
        self.assertTrue(report['skip_counts'])
        # Many actual candidate rows still cannot exceed the64 Source budget.
        many={key:value.repeat((30,)+(1,)*(value.ndim-1)) if torch.is_tensor(value) else value
              for key,value in item.items()}
        _,stats=select_contact_branches([many],contact_branch_protocol={'version':1})
        self.assertEqual(stats['source_probes'],64)

    def test_wall_only_penalty_has_finite_nonzero_braking_gradient(self):
        world=BranchWorld();item=fixture(world)
        result=contact_branches_backward(world,[item],contact_branch_protocol={
            'version':1,'xy_weight':0.,'anchor_bce_weight':0.,'anchor_margin_weight':0.})
        self.assertGreater(result['wall_loss'],0.)
        self.assertGreater(float(world.axis_logits.grad.abs().sum()),0.)
        self.assertTrue(torch.isfinite(world.axis_logits.grad).all())

    def test_moderate_free_progress_does_not_hide_later_true_fb_bf_contact_anchors(self):
        world=BranchWorld()
        with torch.no_grad():world.axis_logits.fill_(float(torch.logit(torch.tensor(.7))))
        grid=np.zeros((16,16),np.uint8)
        grid[[0,-1],:]=1;grid[:,[0,-1]]=1
        grid[6,1:15]=1;grid[1:15,6]=1
        states=[]
        for x,y in ((3.5,5.1),(5.1,3.5)):
            game=Game(seed=0,endless=True,enemies_enabled=False);game.map=grid.copy()
            initial=game.get_state();initial['player'].update(x=x,y=y,angle=np.pi/4)
            game.set_state(initial);states.append(torch.tensor(game.state_vector()))
        state=torch.stack(states);memory=torch.full((2,4),.3)
        maps=torch.tensor(grid)[None].expand(2,-1,-1).clone()
        actions=state.new_tensor([1.,0.,0.,0.])[None,None].expand(2,16,4).clone()
        candidates=[]
        with torch.no_grad():
            features=world.encode_map(maps)
            for first in (0,16):
                anchor,hidden=state.clone(),memory.clone();predicted=[]
                for offset in range(16):
                    result=world.transition(state,actions[:,offset],features,memory)
                    state,memory=result['state'],result['memory'];predicted.append(state)
                candidates.append(capture_contact_candidate(first,anchor,hidden,actions,maps,torch.stack(predicted,1)))
        from turkeyworld.rssm_contact_branches import _source_labels,_flatten
        flat,_=_flatten(candidates)
        initial_rows=[row for row in flat if row['step']==0]
        for row in initial_rows:
            diagnostic=_source_labels(row,normalize_contact_branch_protocol({'version':1}),1/30)
            self.assertIsNone(diagnostic['first_false_stop'])
            self.assertGreater(diagnostic['free_velocity_residual_axes'],0)
            if diagnostic['first_physical_error'] is not None:
                self.assertGreater(diagnostic['first_physical_error'],1)
        selected,stats=select_contact_branches(candidates,contact_branch_protocol={'version':1})
        self.assertEqual([row['step'] for row in selected[:2]],[16,16])
        self.assertEqual({row['cohort'] for row in selected[:2]}, {'x_free_y_blocked','x_blocked_y_free'})
        self.assertEqual(stats['false_stop_probe_count'],0)
        self.assertGreater(stats['free_velocity_residual_axes'],0)
        for row in selected[:2]:
            self.assertTrue(row['first_wall_chunk'])
            self.assertLess(row['step'],row['first_wall_step'])

    def test_terminal_two_step_candidate_does_not_shorten_full16_branches(self):
        world=BranchWorld();full=fixture(world)
        terminal={**full,'step':448,'actions':full['actions'][:,:2].clone(),
                  'prediction':full['prediction'][:,:2].clone()}
        result=contact_branches_backward(world,[full,terminal],contact_branch_protocol={'version':1})
        self.assertEqual(result['horizon'],16)
        # Three actors already had an earlier wall event and are discarded
        # before deferral; the remaining terminal anchor still cannot truncate.
        self.assertEqual(result['terminal_candidates_deferred'],1)
        self.assertFalse(result['short_horizon_fallback'])
        self.assertIn('xy_offset16_tiles_mean',result)
        world=BranchWorld()
        fallback=contact_branches_backward(world,[terminal],contact_branch_protocol={'version':1})
        self.assertTrue(fallback['short_horizon_fallback'])
        self.assertEqual(fallback['horizon'],2)
        self.assertNotIn('xy_offset16_tiles_mean',fallback)

    def test_real_tiny_rssm_prefix_hidden_state_backward_with_all_render_assets_frozen(self):
        torch.manual_seed(54)
        world=NeuralWorldModel(tiny_config());configure_rssm_training(world);world.core.eval()
        with torch.no_grad():world.vjepa_backend.player_pass_head[-1].weight.normal_(std=.02)
        game=Game(seed=0,endless=True,enemies_enabled=False)
        state=torch.tensor(game.state_vector())[None];grid=torch.tensor(game.map)[None]
        rgb=torch.tensor(game.render(32,overlays=False)).permute(2,0,1)[None].float()/255
        actions=state.new_tensor([1.,0.,0.,0.])[None,None].expand(1,16,4).clone()
        with torch.no_grad():
            fmap=world.encode_map(grid);memory=world.initialize_from_image(rgb,state,fmap)
            initial_memory=memory.clone()
            for _ in range(8):
                result=world.transition(state,actions[:,0],fmap,memory)
                state,memory=result['state'],result['memory']
            anchor_state,anchor_memory=state.clone(),memory.clone()
            predicted=[]
            for offset in range(16):
                result=world.transition(state,actions[:,offset],fmap,memory)
                state,memory=result['state'],result['memory'];predicted.append(state)
        self.assertFalse(torch.equal(anchor_memory,initial_memory))
        item=capture_contact_candidate(8,anchor_state,anchor_memory,actions,grid,torch.stack(predicted,1))
        frozen={name:p.detach().clone() for name,p in world.named_parameters() if not p.requires_grad}
        with patch.object(world,'initialize_from_image',side_effect=AssertionError('No reset')), \
             patch.object(world.vjepa_backend.obs_encoder,'forward',side_effect=AssertionError('No RGB')), \
             patch.object(world,'render',side_effect=AssertionError('No renderer')), \
             patch.object(Game,'render',side_effect=AssertionError('No Source images')):
            metrics=contact_branches_backward(world,[item],contact_branch_protocol={'version':1})
        self.assertEqual(metrics['branches'],1);self.assertEqual(metrics['horizon'],16)
        self.assertGreater(float(world.vjepa_backend.player_pass_head[-1].weight.grad.abs().sum()),0.)
        core_grads=[p.grad for p in world.core.parameters() if p.grad is not None]
        self.assertGreater(float(sum(g.abs().sum() for g in core_grads)),0.)
        for name,p in world.named_parameters():
            if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all(),name)
            if name in frozen:
                self.assertIsNone(p.grad,name)
                torch.testing.assert_close(p,frozen[name],rtol=0,atol=0)

    def test_opt_in_full450_preserves_main_and_adds_fresh_branch_graph_before_posterior(self):
        legacy,modern=SpyRSSM(),SpyRSSM();modern.load_state_dict(legacy.state_dict())
        for model in (legacy,modern):model.config=SimpleNamespace(navigation_only=True,simulation_hz=30)
        data=long_batch();data['states'][:,:,8:10]=torch.tensor([5.5,2.5])/16
        data['target_states']=data['states'][:,2:].clone()
        aux=source_batch(context=1,horizon=4,batch_size=1)
        original={k:v.clone() for k,v in data.items()}
        a=rssm_dynamics_backward(legacy,data,auxiliary_batch=aux,auxiliary_context=1)
        b=rssm_dynamics_backward(modern,data,auxiliary_batch=aux,auxiliary_context=1,
            contact_branch_protocol={'version':1})
        for old,new in zip(legacy.calls[:450],modern.calls[:450]):
            torch.testing.assert_close(old[0],new[0],rtol=0,atol=0)
            torch.testing.assert_close(old[1],new[1],rtol=0,atol=0)
        self.assertEqual(len(modern.initializations),2) # main context and separate posterior only
        self.assertGreater(b['rssm_contact_branch_branches'],0)
        self.assertEqual(modern.observations[0]['prior_calls_before'],
            450+b['rssm_contact_branch_horizon']) # one batched transition call per offset
        self.assertEqual(b['rssm_contact_branch_forward_transitions'],
            b['rssm_contact_branch_branches']*b['rssm_contact_branch_horizon'])
        self.assertAlmostEqual(b['train_loss'],a['train_loss']+b['rssm_contact_branch_loss'],places=5)
        self.assertAlmostEqual(b['train_loss_without_kl_floor'],b['train_loss']-1.1)
        for key,value in data.items():torch.testing.assert_close(value,original[key],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in modern.parameters()))


if __name__=='__main__':unittest.main()

