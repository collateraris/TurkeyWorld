"""Spawn append keeps the current optimizer/curriculum and uses existing CE."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

import test_rssm_five_expanded as expanded_fixture
from test_rssm_combat_objectives import source_batch
import test_rssm_fusion_integration as fusion_fixture
from turkeyworld.aligned_training import spawn_supervision_statistics
from turkeyworld.models import NeuralWorldModel, WorldModelConfig, load_world_checkpoint
from turkeyworld.renderer_native_phase import _same
from turkeyworld.rssm_five_expanded_migration import enable_expanded_five_checkpoint
from turkeyworld.rssm_five_migration import report_sha
from turkeyworld.rssm_migration import configure_rssm_training
from turkeyworld.rssm_spawn_conditioning_migration import (enable_spawn_conditioning_checkpoint,
    prepare_spawn_conditioning_continuation, bind_spawn_conditioning_plan, PLAN_KIND, PREFIX)
from turkeyworld.train_rssm_moe import restore_optimizer


class SpawnConditioningMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def parent(self):
        helper=expanded_fixture.ExpandedFiveTests();self.addCleanup(helper.doCleanups)
        _,source=helper.fixture()
        source,_=enable_expanded_five_checkpoint(source,source_checkpoint_sha256='a'*64)
        network,optimizer=helper.restore(source)
        # Initialize the existing 70 expanded moments, as in the live parent.
        sum(p.square().mean()+.001*p.mean()for p in network.parameters()if p.requires_grad).backward()
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        source['model']={n:v.detach().clone()for n,v in network.state_dict().items()}
        source['optimizer']=deepcopy(optimizer.state_dict())
        source['run_state']['independent_sampler_receipts']={'nav':{'draws':65,'PCG':'nav'},'quiet':{'draws':104,'PCG':'enemy'}}
        return source

    @staticmethod
    def restore(saved):
        network=NeuralWorldModel(WorldModelConfig(**saved['config']))
        network.load_state_dict(saved['model'],strict=True)
        for key in ('rssm_fusion_migration','combat_rssm_migration','rssm_five_migration',
                    'rssm_five_expanded_migration','rssm_spawn_conditioning_migration'):
            setattr(network,key,saved.get(key))
        configure_rssm_training(network)
        optimizer,_,_=restore_optimizer(network,saved)
        return network,optimizer

    def test_append_preserves_all_old_names_six_Adam_groups_RNG_sampler_and_entire_gate(self):
        source=self.parent();before=deepcopy(source);rng=torch.get_rng_state().clone()
        migrated,report=enable_spawn_conditioning_checkpoint(source,source_checkpoint_sha256='b'*64)
        self.assertTrue(_same(source,before));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(_same(source['optimizer']['state'],migrated['optimizer']['state']))
        self.assertTrue(_same(source['optimizer']['param_groups'],migrated['optimizer']['param_groups'][:6]))
        self.assertEqual(len(migrated['optimizer']['param_groups']),7)
        self.assertEqual(report['added_tensor_count'],12)
        self.assertEqual(migrated['optimizer_parameter_names'],source['optimizer_parameter_names']+report['added_parameter_names'])
        self.assertTrue(all(n.startswith(PREFIX)for n in report['added_parameter_names']))
        self.assertTrue(all(_same(v,migrated['model'][n])for n,v in source['model'].items()))
        self.assertTrue(_same(source['run_state']['rssm_training'],migrated['run_state']['rssm_training']))
        for key in ('rng_state','torch_rng_state','cuda_rng_state','rssm_training','plan_sha256'):
            self.assertTrue(_same(source[key],migrated[key]),key)
        self.assertTrue(_same(source['run_state']['independent_sampler_receipts'],migrated['run_state']['independent_sampler_receipts']))
        self.assertFalse(report['objective_formula_changed']);self.assertTrue(report['H450_total_loss_definition_unchanged'])
        _,optimizer=self.restore(migrated)
        self.assertTrue(_same(optimizer.state_dict()['state'],source['optimizer']['state']))
        self.assertFalse(set(optimizer.state_dict()['param_groups'][-1]['params'])&set(optimizer.state_dict()['state']))

    def test_genuine_Source_respawn_existing_anchor_CE_reaches_new_adapter_without_new_losses(self):
        source=self.parent();migrated,_=enable_spawn_conditioning_checkpoint(source,source_checkpoint_sha256='b'*64)
        model,optimizer=self.restore(migrated)
        tape=source_batch();states=tape['states']
        revived=(states[:,1:,13]>=.5)&(states[:,1:,12]>0)&~((states[:,:-1,13]>=.5)&(states[:,:-1,12]>0))
        actor,tick=revived.nonzero()[0].tolist()
        previous=states[actor:actor+1,tick];target=states[actor:actor+1,tick+1]
        _,grid,_,_=fusion_fixture.RSSMFusionIntegrationTests.controls()
        with torch.no_grad(): maps=model.encode_map(grid[:1]).detach()
        width=model.config.vjepa_width
        pooled=torch.ones(1,width);spatial=torch.ones(1,2,width)
        source_state=deepcopy(migrated['model'])
        for update in range(2):
            optimizer.zero_grad(set_to_none=True)
            heads=model.vjepa_backend.spawn_outputs(previous,pooled,spatial,maps)
            statistics=spawn_supervision_statistics(dict(heads,state=previous),previous,target)
            loss=statistics['spawn_sum']/statistics['spawn_count'].clamp_min(1)
            loss.backward()
            self.assertEqual(float(statistics['spawn_count']),1)
            adapter=model.vjepa_backend.spawn_conditioning
            self.assertGreater(float(adapter.score[-1].weight.grad.abs().sum()),0)
            if update:
                self.assertGreater(float(adapter.query[0].weight.grad.abs().sum()),0)
                self.assertGreater(float(adapter.candidate_id.weight.grad.abs().sum()),0)
            optimizer.step()
        self.assertFalse(any(not _same(v,model.state_dict()[n])for n,v in source_state.items()if n.startswith(('flow_renderer.','vjepa_backend.renderer.','vjepa_backend.feature_head.'))))
        self.assertTrue(all(value.grad is None for value in (previous,target)))

    def test_plan_resume_strict_pins_and_invalid_parent_names_moments_fail(self):
        source=self.parent();migrated,report=enable_spawn_conditioning_checkpoint(source,source_checkpoint_sha256='b'*64)
        plan=dict(kind=PLAN_KIND,version=1,migration_sha256=report['revision_sha256'],
            source_checkpoint_sha256='b'*64,source_step=report['source_step'],architecture=report['architecture'])
        state=deepcopy(migrated['run_state']);oldq=deepcopy(state['rssm_training'])
        bind_spawn_conditioning_plan(state,plan,path=Path('plan.json'),sha256='c'*64,report=report)
        before=deepcopy(state)
        bind_spawn_conditioning_plan(state,plan,path=Path('plan.json'),sha256='c'*64,report=report)
        self.assertTrue(_same(state,before));self.assertTrue(_same(state['rssm_training'],oldq))
        for key,value in (('source_step',report['source_step']+1),('architecture',{}),('migration_sha256','d'*64)):
            bad=deepcopy(plan);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                bind_spawn_conditioning_plan(state,bad,path=Path('plan.json'),sha256='c'*64,report=report)
        with self.assertRaises(ValueError):
            bind_spawn_conditioning_plan(state,plan,path=Path('plan.json'),sha256='d'*64,report=report)
        for mutate in (lambda p:p['optimizer_parameter_names'].reverse(),
                       lambda p:p['optimizer']['state'][0]['step'].zero_(),
                       lambda p:p['optimizer']['state'][0]['exp_avg'].fill_(float('nan'))):
            bad=deepcopy(source);mutate(bad)
            with self.assertRaises(ValueError):enable_spawn_conditioning_checkpoint(bad,source_checkpoint_sha256='b'*64)

    def test_separate_atomic_prepare_and_strict_saved_report_load(self):
        source=self.parent()
        with TemporaryDirectory()as folder:
            path=Path(folder)/'source.pt';target=Path(folder)/'spawn.pt'
            torch.save(source,path);before=path.read_bytes()
            proof=prepare_spawn_conditioning_continuation(path,target)
            self.assertEqual(before,path.read_bytes())
            model,payload=load_world_checkpoint(target,'cpu')
            self.assertTrue(model.config.rssm_spawn_conditioning_enabled)
            self.assertEqual(proof['added_tensor_count'],12)
            payload.pop('rssm_spawn_conditioning_migration');torch.save(payload,Path(folder)/'bad.pt')
            with self.assertRaises(ValueError):load_world_checkpoint(Path(folder)/'bad.pt','cpu')


if __name__=='__main__':unittest.main()
