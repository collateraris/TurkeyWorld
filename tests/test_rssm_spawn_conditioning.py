"""Own-clock/candidate conditioning is learned, causal and zero-append exact."""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from test_rssm_fusion_integration import raw_dual
import test_rssm_fusion_integration as fusion_fixture
from turkeyworld.anchor_spawn import hierarchical_spawn_logits
from turkeyworld.models import NeuralWorldModel, WorldModelConfig, load_world_checkpoint,STATE_SCHEMA
from turkeyworld.renderer_student import renderer_condition_identity, validate_spawn_conditioning_renderer_lineage
from turkeyworld.rssm_five_migration import report_sha
from turkeyworld.rssm_spawn_conditioning import (
    RSSMSpawnConditioning, enable_spawn_conditioning, spawn_conditioning_metadata, PREFIX)


def inputs(batch=9, width=16):
    state=torch.zeros(batch,20);state[:,:2]=torch.tensor([.5,.5]);state[:,3]=1
    state[:,4]=1;state[:,8:10]=torch.tensor([.7,.7]);state[:,16]=5
    grid=torch.ones(batch,16,16,dtype=torch.bool)
    grid[:,0]=False;grid[:,-1]=False;grid[:,:,0]=False;grid[:,:,-1]=False
    return state,torch.randn(batch,width),torch.randn(batch,2,width),grid


def migration(model, previous):
    meta=spawn_conditioning_metadata(model.vjepa_backend.spawn_conditioning)
    report=dict(format_version=1,kind='append_current_state_spawn_conditioning',
        old_model_tensors_bitexact=True,frozen_render_assets_bitexact=True,zero_residual_initialization=True,
        source_config=asdict(previous),target_config=asdict(model.config),
        allowed_config_changes=['rssm_spawn_conditioning_enabled'],architecture=meta,
        added_parameter_names=[PREFIX+name for name in meta['parameter_names']],
        added_parameter_count=meta['parameters'],memory_size=model.vjepa_backend.memory_size,
        source_renderer_conditioning_identity=renderer_condition_identity(model))
    report['revision_sha256']=report_sha(report)
    model.rssm_spawn_conditioning_migration=report
    return report


class RSSMSpawnConditioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_exact_capacity_and_all_eight_next_tick_boundaries(self):
        rng=torch.get_rng_state().clone();module=RSSMSpawnConditioning()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        meta=spawn_conditioning_metadata(module)
        self.assertEqual(meta['parameters'],118736);self.assertEqual(meta['parameter_tensors'],12)
        self.assertIsNone(module.score[-1].bias)
        for phase in range(8):
            state,pool,spatial,grid=inputs(3,512)
            # State is the previous tick; the proposed next tick is -1/0/+1
            # around the boundary, including the24s wrap back to phase0.
            ticks=torch.tensor([90*(phase+1)-1,90*(phase+1),90*(phase+1)+1])
            state[:,19]=(ticks.double()-1).float()/(30*300)
            result=module(state,pool,spatial,grid)
            self.assertEqual(result['next_tick'].tolist(),ticks.tolist())
            self.assertEqual(result['phase_index'].tolist(),[phase,(phase+1)%8,(phase+1)%8])
            self.assertTrue(torch.allclose(result['within_phase'],torch.tensor([89/90,0.,1/90])))
            self.assertEqual(float(result['anchor_residual'].detach().abs().sum()),0.)

    def test_candidate_legality_exactly_matches_existing_hierarchy_and_XY_convention(self):
        module=RSSMSpawnConditioning(width=16)
        state,pool,spatial,grid=inputs(9)
        state[:,0]=torch.linspace(.12,.88,9);state[:,1]=.2
        state[1::2,12]=.6;state[1::2,13]=1
        state[2:,8:10]=torch.tensor([13.5/16,2.5/16])
        grid[::2,2,2]=False;grid[1::2,13,13]=False
        result=module(state,pool,spatial,grid)
        previous=hierarchical_spawn_logits(torch.zeros(9,1,256),torch.zeros(9,1,9),state,grid,1/30)
        self.assertTrue(torch.equal(result['candidate_legal'],previous['spawn_anchor_legal'][:,0]))
        self.assertTrue(torch.equal(result['candidate_xy'][:,0],torch.tensor([2.5/16,2.5/16]).expand(9,-1)))
        self.assertTrue(torch.equal(result['candidate_xy'][:,2],torch.tensor([13.5/16,2.5/16]).expand(9,-1)))
        grid.zero_();empty=module(state,pool,spatial,grid)
        self.assertFalse(bool(empty['candidate_legal'].any()))
        self.assertTrue(torch.isfinite(empty['candidate_features']).all())

    def test_zero_head_first_then_deep_gradients_from_real_nine_class_CE(self):
        module=RSSMSpawnConditioning(width=16)
        optimizer=torch.optim.AdamW(module.parameters(),lr=1e-3)
        state,pool,spatial,grid=inputs(9)
        state[:,19]=torch.arange(9).float()*3/300
        pool.requires_grad_();spatial.requires_grad_();state.requires_grad_()
        target=torch.arange(9)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True);pool.grad=spatial.grad=None
            with patch('turkeyworld.engine.Game.step',side_effect=AssertionError('No Source step')), \
                 patch('turkeyworld.engine.Game.render',side_effect=AssertionError('No Source image')), \
                 patch('turkeyworld.spawn.spawn_tile_rank',side_effect=AssertionError('No Source placement rank')):
                result=module(state,pool,spatial,grid)
                loss=F.cross_entropy(result['anchor_residual'][:,0],target)
            loss.backward()
            self.assertGreater(float(module.score[-1].weight.grad.abs().sum()),0)
            for name,value in module.named_parameters():
                self.assertIsNotNone(value.grad,name);self.assertTrue(torch.isfinite(value.grad).all(),name)
                if step:self.assertGreater(float(value.grad.abs().sum()),0,name)
                elif name!='score.2.weight':self.assertEqual(float(value.grad.abs().sum()),0,name)
            if step:
                self.assertGreater(float(pool.grad.abs().sum()),0)
                self.assertGreater(float(spatial.grad.abs().sum()),0)
            self.assertIsNone(state.grad)
            optimizer.step()
        with torch.no_grad():
            baseline=module(state,pool,spatial,grid)['anchor_residual']
            shifted=state.detach().clone();shifted[:,19]+=3/300
            self.assertFalse(torch.equal(baseline,module(shifted,pool,spatial,grid)['anchor_residual']))
            moved=state.detach().clone();moved[:,:2]+=.1
            self.assertFalse(torch.equal(baseline,module(moved,pool,spatial,grid)['anchor_residual']))

    def test_actual_backend_warmstart_old_weights_memory_outputs_and_sampling_RNG_exact(self):
        old=raw_dual();new=deepcopy(old);previous=new.config
        names={name:id(value) for name,value in new.named_parameters()}
        values={name:value.clone() for name,value in new.state_dict().items()}
        rng=torch.get_rng_state().clone();enable_spawn_conditioning(new)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(old.vjepa_backend.memory_size,new.vjepa_backend.memory_size)
        for name,identity in names.items():
            self.assertEqual(id(dict(new.named_parameters())[name]),identity)
            self.assertTrue(torch.equal(values[name],new.state_dict()[name]))
        self.assertEqual(new.config,new.vjepa_backend.config)
        state,grid,rgb,action=fusion_fixture.RSSMFusionIntegrationTests.controls()
        state=state.clone();state[:,12:14]=0;state[:,16]=5
        for training in (False,True):
            old.train(training);new.train(training)
            maps=[model.encode_map(grid) for model in (old,new)]
            rng=torch.get_rng_state().clone()
            with torch.no_grad():
                a_mem=old.initialize_from_image(rgb,state,maps[0]);after=torch.get_rng_state().clone()
                torch.set_rng_state(rng);b_mem=new.initialize_from_image(rgb,state,maps[1])
                self.assertTrue(torch.equal(after,torch.get_rng_state()))
                a_state,b_state=state.clone(),state.clone()
                for _ in range(4):
                    rng=torch.get_rng_state().clone();a=old.transition(a_state,action,maps[0],a_mem)
                    after=torch.get_rng_state().clone();torch.set_rng_state(rng)
                    b=new.transition(b_state,action,maps[1],b_mem)
                    self.assertTrue(torch.equal(after,torch.get_rng_state()))
                    for key in ('state','memory','events','spawn_logits','spawn_anchor_logits','spawn_fallback_logits','respawn_flags'):
                        self.assertTrue(torch.equal(a[key],b[key]),key)
                    self.assertIn('spawn_conditioning',b)
                    a_state,a_mem=a['state'],a['memory'];b_state,b_mem=b['state'],b['memory']
        migration(new,previous)
        validate_spawn_conditioning_renderer_lineage(new,asdict(previous))

    def test_constructor_false_legacy_RNG_and_true_strict_reload_require_migration(self):
        old=raw_dual();rng=torch.get_rng_state().clone()
        original=NeuralWorldModel(old.config);after=torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        new=NeuralWorldModel(replace(old.config,rssm_spawn_conditioning_enabled=True))
        self.assertTrue(torch.equal(after,torch.get_rng_state()))
        for name,value in original.state_dict().items():self.assertTrue(torch.equal(value,new.state_dict()[name]),name)
        previous=old.config;enable_spawn_conditioning(old);proof=migration(old,previous)
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'world.pt'
            payload=dict(stage='world',state_schema=dict(STATE_SCHEMA),config=asdict(old.config),model=old.state_dict(),training_steps=1,
                         rssm_spawn_conditioning_migration=proof)
            torch.save(payload,path)
            loaded,_=load_world_checkpoint(path,'cpu')
            self.assertTrue(all(torch.equal(value,loaded.state_dict()[name]) for name,value in old.state_dict().items()))
            for mutate in ('missing_report','wrong_names','wrong_frozen_identity'):
                changed=deepcopy(payload)
                if mutate=='missing_report':del changed['rssm_spawn_conditioning_migration']
                else:
                    report=changed['rssm_spawn_conditioning_migration']
                    if mutate=='wrong_names':report['added_parameter_names']=[]
                    else:report['source_renderer_conditioning_identity']['appearance_sha256']='0'*64
                    report['revision_sha256']=report_sha(report)
                torch.save(changed,path)
                with self.subTest(mutate=mutate),self.assertRaises(ValueError):load_world_checkpoint(path,'cpu')

    def test_bad_inputs_and_wrong_architecture_flags_fail(self):
        module=RSSMSpawnConditioning(width=16);args=list(inputs(2))
        for position,value in ((0,torch.zeros(2,19)),(1,torch.zeros(2,17)),(3,torch.ones(2,16,16))):
            changed=list(args);changed[position]=value
            with self.assertRaises(ValueError):module(*changed)
        for dt in (0.,-1.,float('nan'),True):
            with self.assertRaises(ValueError):module(*args,dt=dt)
        with self.assertRaises(ValueError):replace(raw_dual().config,rssm_spawn_conditioning_enabled='true')
        with self.assertRaises(ValueError):replace(raw_dual().config,rssm_spawn_conditioning_enabled=True,game_training_v5=False)


if __name__=='__main__':unittest.main()
