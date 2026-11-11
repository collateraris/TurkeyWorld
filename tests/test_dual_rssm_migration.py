import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_rssm_combat_training import raw_model
from turkeyworld.base_student import build_base_student, export_base_student
from turkeyworld.dual_rssm_migration import (ENEMY_PREFIX, enable_dual_rssm_checkpoint,
    prepare_dual_rssm_continuation)
from turkeyworld.models import NeuralWorldModel, WorldModelConfig, STATE_SCHEMA
from turkeyworld.renderer_student import build_renderer_student, export_renderer_student, load_renderer_student, _hash_tensors
from turkeyworld.rssm_migration import configure_rssm_training


class DualRSSMMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def fixture(self):
        model=raw_model();model.config=replace(model.config,navigation_only=False)
        model.vjepa_backend.config=model.config;model.flow_renderer.codec_loaded.fill_(True)
        configure_rssm_training(model)
        names=[name for name,p in model.named_parameters()if p.requires_grad]
        optimizer=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=.0001,weight_decay=.01)
        sum(p.square().sum()*1e-6 for p in model.parameters()if p.requires_grad).backward();optimizer.step()
        state={'run_id':'old','global_training_step':4398,'current':4398,'loss_protocol':'unchanged_batch_v4',
            'training_protocol':{'microbatch_size':32,'gradient_accumulation_steps':4,'effective_batch_size':128},
            'rssm_training':{'source_completed':4198,'combat_completed':525,'quality_streak':0,
                'batch_revision':{'last_optimizer_step':4398,'updates':0,'start_optimizer_step':4398},
                'combat_recovery_revision':{'sampler_state':{'draws':123456,'anchor_counts':list(range(8))}}},
            'navigation_training_history':{'source_completed':3673},'objective_history':[{'protocol':'v3','step':4398}]}
        payload={'stage':'world','state_schema':dict(STATE_SCHEMA),'config':asdict(model.config),
            'model':{n:v.detach().clone()for n,v in model.state_dict().items()},
            'optimizer':copy.deepcopy(optimizer.state_dict()),'optimizer_parameter_names':names,
            'global_training_step':4398,'training_steps':4398,'run_state':state,
            'rssm_training':copy.deepcopy(state['rssm_training']), 'rssm_migration':{'format_version':1,'original':'retained'},
            'plan_sha256':'same_base_plan','rng_state':np.random.default_rng(9).bit_generator.state,
            'torch_rng_state':torch.get_rng_state().clone(),'cuda_rng_state':[]}
        return model,optimizer,payload

    def assert_tree(self,first,second):
        if torch.is_tensor(first):torch.testing.assert_close(first,second,rtol=0,atol=0)
        elif isinstance(first,dict):
            self.assertEqual(first.keys(),second.keys())
            for key in first:self.assert_tree(first[key],second[key])
        elif isinstance(first,(list,tuple)):
            self.assertEqual(len(first),len(second))
            for a,b in zip(first,second):self.assert_tree(a,b)
        else:self.assertEqual(first,second)

    def restore(self,payload):
        model=NeuralWorldModel(WorldModelConfig(**payload['config']));model.load_state_dict(payload['model'],strict=True)
        configure_rssm_training(model);parameters=dict(model.named_parameters());names=payload['optimizer_parameter_names']
        groups=[{**{k:v for k,v in group.items()if k!='params'},
            'params':[parameters[names[i]]for i in group['params']]}for group in payload['optimizer']['param_groups']]
        optimizer=torch.optim.AdamW(groups);optimizer.load_state_dict(payload['optimizer'])
        model.dual_rssm_migration=payload['dual_rssm_migration'];model.rssm_migration=payload['rssm_migration']
        return model,optimizer

    def test_all_legacy_weights_groups_adam_rng_and_clocks_survive_with_only_new_group_fresh(self):
        old,optimizer,saved=self.fixture();before=copy.deepcopy(saved);rng=torch.get_rng_state().clone()
        with patch.object(torch.cuda,'manual_seed_all',side_effect=AssertionError('Do not seed CUDA')):
            migrated,report=enable_dual_rssm_checkpoint(saved,173,run_id='rssm_dual')
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assert_tree(saved,before)
        self.assertFalse(saved['config']['enemy_rssm_enabled']);self.assertTrue(migrated['config']['enemy_rssm_enabled'])
        for name,value in saved['model'].items():torch.testing.assert_close(value,migrated['model'][name],rtol=0,atol=0)
        added=set(migrated['model'])-set(saved['model']);self.assertTrue(added);self.assertTrue(all(n.startswith(ENEMY_PREFIX)for n in added))
        self.assertEqual(migrated['optimizer_parameter_names'][:len(saved['optimizer_parameter_names'])],saved['optimizer_parameter_names'])
        self.assert_tree(migrated['optimizer']['param_groups'][:-1],saved['optimizer']['param_groups'])
        self.assert_tree(migrated['optimizer']['state'],saved['optimizer']['state'])
        self.assertEqual(set(migrated['optimizer']['state'])&set(migrated['optimizer']['param_groups'][-1]['params']),set())
        for key in ['rng_state','torch_rng_state','cuda_rng_state','plan_sha256','rssm_migration']:
            self.assert_tree(migrated[key],saved[key])
        for key,value in saved['rssm_training'].items():self.assert_tree(migrated['rssm_training'][key],value)
        for key in ['global_training_step','current','training_protocol','navigation_training_history']:
            self.assert_tree(migrated['run_state'][key],saved['run_state'][key])
        self.assert_tree(migrated['run_state']['objective_history'][:-1],saved['run_state']['objective_history'])
        self.assertEqual(migrated['rssm_training']['memory_size'],report['combined_memory_size'])
        self.assertEqual(migrated['rssm_training']['enemy_memory_size'],640)
        self.assertEqual(migrated['run_state']['run_id'],'rssm_dual');self.assertFalse(report['inference_functions_bitexact_claimed'])
        restored,new_optimizer=self.restore(migrated)
        # Old moments perform exactly the same next update. New parameters get
        # their own first state without shifting or resetting any old ID.
        for model,opt in [(old,optimizer),(restored,new_optimizer)]:
            opt.zero_grad(set_to_none=True)
            sum(p.square().sum()*1e-6 for p in model.parameters()if p.requires_grad).backward();opt.step()
        old_parameters=dict(old.named_parameters());new_parameters=dict(restored.named_parameters())
        for name in saved['optimizer_parameter_names']:
            torch.testing.assert_close(old_parameters[name],new_parameters[name],rtol=0,atol=0)
            self.assert_tree(optimizer.state[old_parameters[name]],new_optimizer.state[new_parameters[name]])
        self.assertTrue(all(float(new_optimizer.state[new_parameters[n]]['step'])==1 for n in report['added_parameter_names']))

    def test_seed_only_changes_new_enemy_and_production_profile_adds_real2438019_parameters(self):
        _,_,saved=self.fixture();first,a=enable_dual_rssm_checkpoint(saved,173);again,_=enable_dual_rssm_checkpoint(saved,173);other,_=enable_dual_rssm_checkpoint(saved,174)
        self.assertTrue(all(torch.equal(v,again['model'][n])for n,v in first['model'].items()))
        self.assertTrue(any(not torch.equal(first['model'][n],other['model'][n])for n in a['added_parameter_names']))
        self.assertTrue(all(torch.equal(v,other['model'][n])for n,v in saved['model'].items()))
        config=WorldModelConfig(kind='teacher',architecture='dreamer_rssm',image_size=32,spatial_entities=True,spatial_combat=True,
            vjepa_width=512,rssm_embed_dim=512,rssm_deter_dim=1008,rssm_hidden_dim=1024,enemy_rssm_enabled=True)
        with torch.device('meta'):model=NeuralWorldModel(config)
        self.assertEqual(sum(p.numel()for p in model.vjepa_backend.enemy_rssm.parameters()),2438019)
        self.assertEqual(model.vjepa_backend.enemy_rssm.memory_size,640);self.assertEqual(model.vjepa_backend.memory_size,2672)

    def test_partial_or_wrong_source_and_optimizer_mappings_are_rejected_without_mutating_input(self):
        _,_,saved=self.fixture()
        for change in ['missing_weight','extra_enemy','bad_names','bad_moment','bad_schema','partial_step']:
            bad=copy.deepcopy(saved)
            if change=='missing_weight':bad['model'].pop(next(iter(bad['model'])))
            elif change=='extra_enemy':bad['model'][ENEMY_PREFIX+'fake']=torch.ones(1)
            elif change=='bad_names':bad['optimizer_parameter_names'][0]='not_a_parameter'
            elif change=='bad_moment':
                key=next(iter(bad['optimizer']['state']));bad['optimizer']['state'][key]['exp_avg']=torch.ones(1)
            elif change=='bad_schema':bad['state_schema']={}
            else:bad['training_steps']-=1
            before=copy.deepcopy(bad)
            with self.assertRaises(ValueError):enable_dual_rssm_checkpoint(bad)
            self.assert_tree(before,bad)

    def test_real_legacy_base_rf_bundle_requires_verified_migration_and_preserves_dual_flag_and_nonflow_weights(self):
        model,_,saved=self.fixture();migrated,report=enable_dual_rssm_checkpoint(saved)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);base=build_base_student(model);export_base_student(model,base,root/'base.pt',training_steps=3)
            model.base_student=base;model.base_student_bundle=torch.load(root/'base.pt',weights_only=False)
            student=build_renderer_student(model,steps=1,width=16,depth=1,heads=2,mlp_hidden=32)
            export_renderer_student(model,student,root/'renderer.pt',training_steps=7,metrics={'source_quality_passed':False})
            target,_=self.restore(migrated);del target.dual_rssm_migration
            old_hash=_hash_tensors(target.state_dict())
            with self.assertRaisesRegex(ValueError,'dual-RSSM'):load_renderer_student(target,root/'renderer.pt')
            self.assertEqual(old_hash,_hash_tensors(target.state_dict()))
            target.dual_rssm_migration=report
            nonflow={n:v.clone()for n,v in target.state_dict().items()if not n.startswith(('flow_renderer.','base_student.'))}
            metadata=load_renderer_student(target,root/'renderer.pt')
            self.assertTrue(target.config.enemy_rssm_enabled);self.assertEqual(target.flow_renderer.steps,1)
            self.assertFalse(metadata['metrics']['source_quality_passed'])
            self.assertTrue(all(torch.equal(v,target.state_dict()[n])for n,v in nonflow.items()))
            for field in ['latent_mean','fixed_noise']:
                invalid,_=self.restore(migrated)
                with torch.no_grad():getattr(invalid.flow_renderer,field).flatten()[0].add_(.01)
                with self.assertRaises(ValueError):load_renderer_student(invalid,root/'renderer.pt')

    def test_file_export_is_separate_immutable_atomic_and_reports_required_run_bindings(self):
        _,_,saved=self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'old/checkpoints/source.pt';source.parent.mkdir(parents=True)
            torch.save(saved,source);before=source.read_bytes()
            (source.parent.parent/'combat_training_plan.json').write_text('{"fixed":true}')
            destination=root/'rssm_dual/checkpoints/rssm_world.pt'
            report=prepare_dual_rssm_continuation(source,destination,runid='rssm_dual')
            self.assertEqual(source.read_bytes(),before);self.assertTrue(destination.exists())
            sidecar=json.loads(destination.with_suffix('.pt.migration.json').read_text())
            self.assertEqual(sidecar['source_checkpoint_sha256'],report['source_checkpoint_sha256'])
            self.assertEqual(len(report['copy_identically_to_new_run']),1)
            loaded=torch.load(destination,map_location='cpu',weights_only=False)
            self.assertEqual(loaded['training_steps'],4398);self.assertEqual(loaded['run_state']['run_id'],'rssm_dual')
            with self.assertRaises(FileExistsError):prepare_dual_rssm_continuation(source,destination)
            with self.assertRaises(FileExistsError):prepare_dual_rssm_continuation(source,source)


if __name__=='__main__':unittest.main()
