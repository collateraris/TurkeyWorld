from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

import test_rssm_fusion_integration as fusion_fixture
import test_rssm_five_migration as five_fixture
from turkeyworld.models import NeuralWorldModel,WorldModelConfig,load_world_checkpoint
from turkeyworld.renderer_native_phase import _same
from turkeyworld.renderer_student import validate_five_role_renderer_lineage
from turkeyworld.rssm_five_migration import enable_rssm_five_checkpoint
from turkeyworld.rssm_five_expanded import expand_five_experts,expanded_five_metadata
from turkeyworld.rssm_five_expanded_migration import (enable_expanded_five_checkpoint,
    prepare_expanded_five_continuation,bind_expanded_five_plan)
from turkeyworld.rssm_migration import configure_rssm_training
from turkeyworld.train_rssm_moe import restore_optimizer


class ExpandedFiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def fixture(self):
        helper=five_fixture.RSSMFiveMigrationTests();self.addCleanup(helper.doCleanups)
        with patch.dict(fusion_fixture.SMALL_FUSION,dict(rssm_fusion_moe_width=256,
                rssm_fusion_moe_heads=4,rssm_fusion_moe_expert_hidden=512)):
            _,_,saved=helper.fixture()
        saved,_=enable_rssm_five_checkpoint(saved)
        model,optimizer=helper.restore(saved)
        sum(p.square().mean()+.001*p.mean()for p in model.parameters()if p.requires_grad).backward()
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        saved['model']={n:v.detach().clone()for n,v in model.state_dict().items()}
        saved['optimizer']=deepcopy(optimizer.state_dict())
        # This tiny legacy fixture began with one dual-RSSM optimizer group;
        # production retained two. Split that canonical prefix without changing
        # any ID/name/state to exercise the actual five-group source contract.
        first=saved['optimizer']['param_groups'][0];half=len(first['params'])//2
        saved['optimizer']['param_groups']=[{**deepcopy(first),'params':first['params'][:half]},
            {**deepcopy(first),'params':first['params'][half:]},*saved['optimizer']['param_groups'][1:]]
        return model,saved

    @staticmethod
    def restore(saved):
        model=NeuralWorldModel(WorldModelConfig(**saved['config']))
        model.load_state_dict(saved['model'],strict=True)
        for k in('rssm_fusion_migration','combat_rssm_migration','rssm_five_migration','rssm_five_expanded_migration'):
            setattr(model,k,saved.get(k))
        configure_rssm_training(model)
        optimizer,_,_=restore_optimizer(model,saved)
        return model,optimizer

    def test_actual_count_zero_append_distinct_seeds_and_global_RNG_preserved(self):
        old,_=self.fixture();module=old.vjepa_backend.five_role_fusion.eval()
        inputs=(torch.randn(2,3,256),torch.zeros(2,20),torch.zeros(2,4))
        with torch.no_grad():expected=module(*inputs)
        old_params=dict(module.named_parameters());clock=torch.get_rng_state().clone()
        meta=expand_five_experts(module)
        self.assertTrue(torch.equal(clock,torch.get_rng_state()))
        self.assertEqual(meta['parameters_per_expert'],1005088)
        self.assertEqual(meta['added_parameters'],2389920)
        self.assertEqual(len(meta['added_parameter_names']),70)
        self.assertEqual(meta['total_fusion_parameters'],5107388)
        for source,destination in meta['legacy_parameter_name_map'].items():
            self.assertIs(old_params[source],dict(module.named_parameters())[destination])
        with torch.no_grad():actual=module(*inputs)
        self.assertTrue(_same(expected,actual));self.assertTrue(torch.equal(clock,torch.get_rng_state()))
        self.assertFalse(torch.equal(module.experts[0].private_transformer.linear1.weight,
                                     module.experts[1].private_transformer.linear1.weight))

    def test_full_old_Adam_config_RNG_and_gate_are_retained_under_name_bijection(self):
        _,saved=self.fixture();original=deepcopy(saved);clock=torch.get_rng_state().clone()
        migrated,report=enable_expanded_five_checkpoint(saved,source_checkpoint_sha256='a'*64)
        self.assertTrue(_same(saved,original));self.assertTrue(torch.equal(clock,torch.get_rng_state()))
        self.assertEqual(len(migrated['optimizer']['param_groups']),6)
        self.assertTrue(_same(saved['optimizer']['state'],migrated['optimizer']['state']))
        self.assertEqual(saved['optimizer']['param_groups'],migrated['optimizer']['param_groups'][:5])
        for key in('rng_state','torch_rng_state','cuda_rng_state','rssm_training','plan_sha256'):
            self.assertTrue(_same(saved[key],migrated[key]))
        self.assertTrue(_same(saved['run_state']['rssm_training'],migrated['run_state']['rssm_training']))
        for name,value in saved['model'].items():
            self.assertTrue(torch.equal(value,migrated['model'][report['old_model_tensor_name_map'][name]]))
        self.assertEqual(report['memory_size'],72)
        model,optimizer=self.restore(migrated)
        self.assertTrue(_same(optimizer.state_dict()['state'],saved['optimizer']['state']))
        self.assertFalse(set(optimizer.state_dict()['param_groups'][-1]['params'])&set(optimizer.state_dict()['state']))
        validate_five_role_renderer_lineage(model,saved['config'])
        bad=deepcopy(migrated);bad['optimizer_parameter_names'][0:2]=list(reversed(bad['optimizer_parameter_names'][0:2]))
        with self.assertRaises(ValueError):restore_optimizer(model,bad)

    def test_argmax_and_stochastic_own_state_memory_features_and_RNG_remain_exact(self):
        old,saved=self.fixture();migrated,_=enable_expanded_five_checkpoint(saved)
        new,_=self.restore(migrated)
        state,grid,rgb,action=fusion_fixture.RSSMFusionIntegrationTests.controls()
        for mode in(False,True):
            for model in(old,new):configure_rssm_training(model);model.train(mode)
            with torch.no_grad():
                maps=[model.encode_map(grid)for model in(old,new)]
                rng=torch.get_rng_state().clone();memories=[]
                for model,fmap in zip((old,new),maps):
                    torch.set_rng_state(rng);memories.append(model.initialize_from_image(rgb,state,fmap))
                states=[state.clone(),state.clone()]
                for _ in range(4):
                    rng=torch.get_rng_state().clone();results=[];clocks=[]
                    for model,current,memory,fmap in zip((old,new),states,memories,maps):
                        torch.set_rng_state(rng);results.append(model.transition(current,action,fmap,memory))
                        clocks.append(torch.get_rng_state().clone())
                    self.assertTrue(torch.equal(*clocks))
                    for key in('state','memory','features','events','target_logits','attack_logits','player_pass_logits'):
                        self.assertTrue(torch.equal(results[0][key],results[1][key]),(mode,key))
                    states=[v['state']for v in results];memories=[v['memory']for v in results]

    def test_new_five_private_blocks_receive_gradient_after_output_opens_and_strict_reload(self):
        _,saved=self.fixture();migrated,report=enable_expanded_five_checkpoint(saved)
        model,optimizer=self.restore(migrated);fusion=model.vjepa_backend.five_role_fusion
        for update in range(2):
            optimizer.zero_grad(set_to_none=True)
            output=fusion(torch.ones(2,3,256),torch.zeros(2,20),torch.zeros(2,4))
            loss=sum(output[k].square().mean()for k in('player_pass_residual','enemy_motion_residual',
                'interaction_target_residual','interaction_attack_residual','interaction_spawn_tile_residual',
                'interaction_spawn_anchor_residual'))+.01*output['router_role_loss']
            loss.backward()
            if update:
                for expert in fusion.experts:
                    self.assertGreater(sum(float(p.grad.abs().sum())for p in expert.private_transformer.parameters()if p.grad is not None),0.)
            optimizer.step()
        with TemporaryDirectory()as folder:
            source=Path(folder)/'source.pt';target=Path(folder)/'expanded.pt'
            torch.save(saved,source);proof=prepare_expanded_five_continuation(source,target)
            loaded,payload=load_world_checkpoint(target,'cpu')
            self.assertTrue(loaded.config.rssm_five_expanded_enabled)
            self.assertEqual(proof['added_parameter_count'],2389920)
            payload.pop('rssm_five_expanded_migration');torch.save(payload,Path(folder)/'bad.pt')
            with self.assertRaises(ValueError):load_world_checkpoint(Path(folder)/'bad.pt','cpu')


if __name__=='__main__':unittest.main()
