"""Append-only five-role dynamics fusion and strict frozen-image lineage."""
from copy import deepcopy
from dataclasses import asdict,replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import test_combat_rssm_backend as combat_fixture
import test_rssm_fusion_integration as fusion_fixture
from turkeyworld.base_student import build_base_student,export_base_student
from turkeyworld.models import NeuralWorldModel,WorldModelConfig,load_world_checkpoint,parameter_counts
from turkeyworld.renderer_student import (build_renderer_student,export_renderer_student,
    load_renderer_student,renderer_condition_identity,_hash_tensors,validate_five_role_renderer_lineage)
from turkeyworld.rssm_five_role_fusion import RESIDUAL_KEYS
from turkeyworld.rssm_migration import configure_rssm_training


def append_five(old):
    new=NeuralWorldModel(replace(old.config,rssm_five_role_enabled=True))
    missing,extra=new.load_state_dict(old.state_dict(),strict=False)
    assert missing and all(n.startswith('vjepa_backend.five_role_fusion.')for n in missing)and not extra
    for name in('rssm_fusion_migration','combat_rssm_migration'):
        setattr(new,name,deepcopy(getattr(old,name)))
    previous,migrated=asdict(old.config),asdict(new.config)
    module=new.vjepa_backend.five_role_fusion
    new.rssm_five_migration=dict(format_version=1,kind='append_five_role_transformer_fusion',
        old_model_tensors_bitexact=True,frozen_render_assets_bitexact=True,zero_residual_initialization=True,
        source_config=previous,target_config=migrated,
        allowed_config_changes=sorted(k for k in previous if previous[k]!=migrated[k]),
        added_parameter_names=['vjepa_backend.five_role_fusion.'+n for n,_ in module.named_parameters()],
        added_parameter_count=sum(p.numel()for p in module.parameters()),memory_size=new.vjepa_backend.memory_size,
        source_renderer_conditioning_identity=renderer_condition_identity(new))
    configure_rssm_training(new)
    return new


class RSSMFiveBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);combat_fixture.CombatRSSMBackendTests.setUpClass()

    def old(self):return combat_fixture.CombatRSSMBackendTests().pair()[1]

    def test_append_preserves_old_parameter_prefix_initialization_rng_and_16_tick_argmax_stochastic_outputs(self):
        old=self.old()
        with torch.no_grad():
            old.vjepa_backend.fusion_moe.player_head.weight.normal_(0,.01)
            old.vjepa_backend.fusion_moe.combat_head.weight.normal_(0,.01)
        rng=torch.get_rng_state().clone();baseline=NeuralWorldModel(old.config)
        after=torch.get_rng_state().clone();torch.set_rng_state(rng)
        constructed=NeuralWorldModel(replace(old.config,rssm_five_role_enabled=True))
        self.assertTrue(torch.equal(after,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v,constructed.state_dict()[n])for n,v in baseline.state_dict().items()))
        new=append_five(old);old_names=list(dict(old.named_parameters()))
        self.assertEqual(list(dict(new.named_parameters()))[:len(old_names)],old_names)
        self.assertEqual(old.vjepa_backend.memory_size,new.vjepa_backend.memory_size)
        state,grid,rgb,action=fusion_fixture.RSSMFusionIntegrationTests.controls()
        keys=('state','events','memory','rssm_h','rssm_z','enemy_rssm_h','enemy_rssm_z',
            'combat_rssm_h','combat_rssm_z','target_logits','attack_logits','spawn_logits','player_pass_logits')
        for training in(False,True):
            old.train(training);new.train(training)
            maps=[m.encode_map(grid)for m in(old,new)]
            rng=torch.get_rng_state().clone()
            with torch.no_grad():
                a_mem=old.initialize_from_image(rgb,state,maps[0]);after=torch.get_rng_state().clone()
                torch.set_rng_state(rng);b_mem=new.initialize_from_image(rgb,state,maps[1])
                self.assertTrue(torch.equal(after,torch.get_rng_state()))
                self.assertTrue(torch.equal(a_mem,b_mem))
                a_state,b_state=state.clone(),state.clone()
                for tick in range(16):
                    rng=torch.get_rng_state().clone();a=old.transition(a_state,action,maps[0],a_mem)
                    after=torch.get_rng_state().clone();torch.set_rng_state(rng)
                    b=new.transition(b_state,action,maps[1],b_mem)
                    self.assertTrue(torch.equal(after,torch.get_rng_state()))
                    for key in keys:self.assertTrue(torch.equal(a[key],b[key]),(training,tick,key))
                    self.assertTrue(torch.equal(a['rssm_fusion_moe']['role_tokens'],b['rssm_fusion_moe']['role_tokens']))
                    self.assertEqual(b['five_role_fusion']['role_tokens'].shape,(2,5,16))
                    for key in RESIDUAL_KEYS:self.assertTrue(torch.equal(a['rssm_fusion_moe'][key],b['rssm_fusion_moe'][key]))
                    a_state,b_state,a_mem,b_mem=a['state'],b['state'],a['memory'],b['memory']

    def test_five_context_is_previous_trunk_and_single_causal_prior_even_with_future_RGB(self):
        model=append_five(self.old()).eval();backend=model.vjepa_backend
        state,grid,rgb,action=fusion_fixture.RSSMFusionIntegrationTests.controls();maps=model.encode_map(grid)
        memory=model.initial_memory(2)
        calls=[];old_fusion=backend.fusion_moe.forward;five=backend.five_role_fusion.forward
        prior=backend.core.imagine_step
        def shared(*args,**kwargs):calls.append('shared');return old_fusion(*args,**kwargs)
        def extra(*args,**kwargs):calls.append('five');return five(*args,**kwargs)
        def imagine(*args,**kwargs):calls.append('prior');return prior(*args,**kwargs)
        with patch.object(backend.fusion_moe,'forward',side_effect=shared),\
                patch.object(backend.five_role_fusion,'forward',side_effect=extra)as spy,\
                patch.object(backend.core,'imagine_step',side_effect=imagine):
            a=backend.observe_step(state,action,maps,memory,rgb,sample=False)
        self.assertEqual(calls,['shared','five','prior']);self.assertEqual(spy.call_count,1)
        self.assertIs(spy.call_args.args[0],a['prior_result']['rssm_fusion_moe']['role_tokens'])
        self.assertEqual(a['rssm_fusion_moe']['role_tokens'].shape,(2,3,16))
        self.assertIs(a['five_role_fusion'],a['prior_result']['five_role_fusion'])
        b=backend.observe_step(state,action,maps,memory,1-rgb,sample=False)
        for key in('state','memory','target_logits','attack_logits'):
            self.assertTrue(torch.equal(a['prior_result'][key],b['prior_result'][key]))
        addition=a['five_role_fusion'];fusion=a['rssm_fusion_moe']
        self.assertTrue(torch.equal(a['five_role_router_role_loss'],addition['router_role_loss']))
        self.assertGreater(float(fusion['router_role_loss'].detach()),float(addition['router_role_loss'].detach()))

    def test_opened_five_head_reaches_shared_previous_memories_without_renderer_gradients(self):
        model=append_five(self.old());configure_rssm_training(model);model.eval();backend=model.vjepa_backend
        with torch.no_grad():
            backend.combat_rssm.fusion_adapter.weight.copy_(torch.eye(16)*.2)
            for head in backend.five_role_fusion.player_heads:
                head.weight[0].copy_(torch.linspace(.01,.12,16))
        state,grid,_,action=fusion_fixture.RSSMFusionIntegrationTests.controls();maps=model.encode_map(grid)
        memory=model.initial_memory(2).detach().requires_grad_()
        result=model.transition(state,action,maps,memory)
        (result['player_pass_logits'][:,0].sum()+.01*result['five_role_router_role_loss']).backward()
        p,e,c=backend.split_all_memory(memory.grad)
        self.assertTrue(all(float(v.abs().sum())>0 for v in(p,e,c)))
        for expert in backend.five_role_fusion.experts:
            grads=[p.grad for p in expert.parameters()if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all()for g in grads));self.assertGreater(sum(float(g.abs().sum())for g in grads),0.)
        self.assertTrue(all(p.grad is None for p in model.flow_renderer.parameters()))
        self.assertEqual(parameter_counts(model)['rssm_five_role_fusion'],sum(p.numel()for p in backend.five_role_fusion.parameters()))

    def test_old_RF_bundle_and_checkpoint_require_exact_composed_five_proof(self):
        old=self.old().eval();model=append_five(old).eval()
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);base=build_base_student(old)
            export_base_student(old,base,root/'base.pt',training_steps=3)
            old.base_student=base;old.base_student_bundle=torch.load(root/'base.pt',weights_only=False)
            renderer=build_renderer_student(old,steps=1,width=16,depth=1,heads=2,mlp_hidden=32)
            export_renderer_student(old,renderer,root/'rf.pt',training_steps=9)
            proof=deepcopy(model.rssm_five_migration)
            for mode in('missing','wrong_memory','wrong_names','changed_backbone','wrong_identity'):
                broken=deepcopy(proof)
                if mode=='missing':broken=None
                elif mode=='wrong_memory':broken['memory_size']+=1
                elif mode=='wrong_names':broken['added_parameter_names']=broken['added_parameter_names'][::-1]
                elif mode=='changed_backbone':broken['target_config']['rssm_deter_dim']+=2
                else:broken['source_renderer_conditioning_identity']['appearance_sha256']='0'*64
                model.rssm_five_migration=broken;before=_hash_tensors(model.state_dict())
                with self.subTest(mode=mode),self.assertRaisesRegex(ValueError,'[Ff]ive-role'):load_renderer_student(model,root/'rf.pt')
                self.assertEqual(before,_hash_tensors(model.state_dict()))
            model.rssm_five_migration=proof;load_renderer_student(model,root/'rf.pt')
            self.assertTrue(model.config.rssm_five_role_enabled)
            cp=dict(stage='world',config=asdict(model.config),model=model.state_dict(),
                rssm_fusion_migration=model.rssm_fusion_migration,combat_rssm_migration=model.combat_rssm_migration,
                rssm_five_migration=proof)
            cp['model']={n:v for n,v in cp['model'].items()if not n.startswith('base_student.')}
            torch.save(cp,root/'world.pt');clone,_=load_world_checkpoint(root/'world.pt','cpu')
            self.assertEqual(clone.rssm_five_migration,proof)
            self.assertEqual(_hash_tensors(clone.state_dict()),_hash_tensors(cp['model']))
            cp.pop('rssm_five_migration');torch.save(cp,root/'bad.pt')
            with self.assertRaisesRegex(ValueError,'five-role'):load_world_checkpoint(root/'bad.pt','cpu')
        with self.assertRaises(ValueError):WorldModelConfig(rssm_five_role_enabled=True)


if __name__=='__main__':unittest.main()
