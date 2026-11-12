import copy
from dataclasses import replace
import json
from pathlib import Path
import random
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_renderer_student import TinyTeacher
from turkeyworld.base_student import build_base_student, export_base_student
from turkeyworld.models import SingleFrameUpscaler
from turkeyworld.renderer_student import build_renderer_student, export_renderer_student, _hash_tensors
from turkeyworld.rssm_preview_assets import (file_sha256, read_preview_assets_plan,
    preview_assets_cache_key, prepare_preview_assets, preserve_preview_rng, synchronize_preview_dynamics,
    resolve_preview_assets_plan)
from turkeyworld.train_rssm_dynamics import RSSMTraining
from turkeyworld.vjepa_validation import save_autoregressive_preview


class RSSMPreviewAssetsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        source = TinyTeacher().eval()
        source.config = replace(source.config, architecture='dreamer_rssm', navigation_only=False)
        base = build_base_student(source)
        export_base_student(source, base, self.root/'base.pt', training_steps=15200, metrics={'passed': False})
        source.base_student, source.base_student_bundle = base, torch.load(self.root/'base.pt', weights_only=False)
        student = build_renderer_student(source, steps=1, width=32, depth=1, heads=4, mlp_hidden=64)
        with torch.no_grad(): student.output.weight.fill_(.003)
        export_renderer_student(source, student, self.root/'renderer.pt', training_steps=15600,
            metrics={'refinement_updates':6000, 'source_quality_passed':False,'source_validation':{'release_passed':False}})
        del source.base_student, source.base_student_bundle
        sr = SingleFrameUpscaler(channels=4, blocks=1)
        with torch.no_grad(): sr.output[0].bias.fill_(.02)
        torch.save({'stage':'sr','config':sr.config,'model':sr.state_dict(), 'training_steps':38100,
            'refinement_updates':18100,'input_size':32,'output_size':128,'source_quality_passed':False}, self.root/'sr.pt')
        self.raw = {'version':1,'kind':'frozen_preview_assets_v1','allow_unqualified_codec':True,
            'renderer':{'checkpoint':str(self.root/'renderer.pt'),'file_sha256':file_sha256(self.root/'renderer.pt'),'refinement_updates':6000},
            'sr':{'checkpoint':str(self.root/'sr.pt'),'file_sha256':file_sha256(self.root/'sr.pt'),'selected_local_update':18100}}
        self.path = self.root/'assets.json'; self.path.write_text(json.dumps(self.raw), encoding='utf-8')
        self.source = source

    def prepare(self, plan=None):
        return prepare_preview_assets(self.source, plan or read_preview_assets_plan(self.path), 'cpu',
            clone_factory=lambda config:copy.deepcopy(self.source))

    def test_strict_updated_assets_are_separate_frozen_and_do_not_change_model_adam_or_rng(self):
        optimizer = torch.optim.Adam(self.source.vjepa_backend.core.parameters(), lr=.001)
        optimizer.zero_grad(); self.source.vjepa_backend.core(torch.ones(1,20)).square().mean().backward(); optimizer.step()
        old_adam = copy.deepcopy(optimizer.state_dict()); old_hash = _hash_tensors(self.source.state_dict())
        old_ids = [id(p) for p in self.source.parameters()]; old_modes = [m.training for m in self.source.modules()]
        torch_state = torch.get_rng_state().clone(); py_state = random.getstate(); np_state = np.random.get_state()
        client, sr, metadata, proof = self.prepare()
        self.assertEqual(old_hash, _hash_tensors(self.source.state_dict()))
        self.assertEqual(old_ids, [id(p) for p in self.source.parameters()]); self.assertEqual(old_modes,[m.training for m in self.source.modules()])
        self.assertEqual(self.source.flow_renderer.steps,3); self.assertFalse(hasattr(self.source,'base_student'))
        self.assertEqual(client.flow_renderer.steps,1); self.assertIsNot(client,self.source)
        bundle=torch.load(self.root/'renderer.pt',weights_only=False)
        self.assertEqual(_hash_tensors(client.flow_renderer.state_dict()),_hash_tensors(bundle['model']))
        self.assertEqual(_hash_tensors(client.base_student.state_dict()),_hash_tensors(bundle['base_student_bundle']['model']))
        self.assertTrue(all(not p.requires_grad for p in client.parameters())); self.assertTrue(all(not p.requires_grad for p in sr.parameters()))
        self.assertFalse(proof['renderer_source_quality_passed']); self.assertFalse(proof['gameplay_qualified'])
        self.assertFalse(proof['renderer_release_passed'])
        self.assertFalse(proof['sr_source_quality_passed']);self.assertEqual(proof['sr_refinement_updates'],18100)
        self.assertTrue(proof['training_model_tensors_bitexact']); self.assertEqual(metadata['training_steps'],38100)
        self.assertTrue(torch.equal(torch_state,torch.get_rng_state())); self.assertEqual(py_state,random.getstate())
        np.testing.assert_array_equal(np_state[1],np.random.get_state()[1])
        for key,state in old_adam['state'].items():
            for name,value in state.items(): torch.testing.assert_close(value,optimizer.state_dict()['state'][key][name],rtol=0,atol=0)

    def test_current_dynamics_refresh_keeps_updated_flow_base_codec_and_sr(self):
        client,sr,_,_=self.prepare(); flow_hash=_hash_tensors(client.flow_renderer.state_dict()); base_hash=_hash_tensors(client.base_student.state_dict())
        with torch.no_grad(): self.source.vjepa_backend.core.weight.add_(.125)
        synchronize_preview_dynamics(client,self.source)
        torch.testing.assert_close(client.vjepa_backend.core.weight,self.source.vjepa_backend.core.weight,rtol=0,atol=0)
        self.assertEqual(flow_hash,_hash_tensors(client.flow_renderer.state_dict())); self.assertEqual(base_hash,_hash_tensors(client.base_student.state_dict()))
        with torch.no_grad(): self.source.vjepa_backend.renderer.weight.add_(.001)
        with self.assertRaisesRegex(ValueError,'conditioning'): synchronize_preview_dynamics(client,self.source)

    def test_file_hash_tampering_bad_binding_and_untrained_sr_are_rejected(self):
        plan=read_preview_assets_plan(self.path)
        bad=copy.deepcopy(self.raw); bad['renderer']['file_sha256']='0'*64; self.path.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError,'SHA'): read_preview_assets_plan(self.path)
        self.path.write_text(json.dumps(self.raw))
        original_renderer=(self.root/'renderer.pt').read_bytes()
        bad_bundle=torch.load(self.root/'renderer.pt',weights_only=False);bad_bundle['conditioning_identity']['base_map_sha256']='bad'
        torch.save(bad_bundle,self.root/'renderer.pt');self.raw['renderer']['file_sha256']=file_sha256(self.root/'renderer.pt');self.path.write_text(json.dumps(self.raw))
        old=_hash_tensors(self.source.state_dict())
        with self.assertRaises(ValueError): self.prepare()
        self.assertEqual(old,_hash_tensors(self.source.state_dict()))
        (self.root/'renderer.pt').write_bytes(original_renderer);self.raw['renderer']['file_sha256']=file_sha256(self.root/'renderer.pt')
        sr=torch.load(self.root/'sr.pt',weights_only=False);sr['training_steps']=0;torch.save(sr,self.root/'sr.pt')
        self.assertNotEqual(plan['sr']['file_sha256'],file_sha256(self.root/'sr.pt'))
        self.raw['sr']['file_sha256']=file_sha256(self.root/'sr.pt');self.path.write_text(json.dumps(self.raw))
        with self.assertRaisesRegex(ValueError,'actually be trained'):self.prepare()
        self.assertEqual(old,_hash_tensors(self.source.state_dict()))

    def test_worker_cache_uses_asset_sha_and_plan_is_immutable_without_changing_legacy_args(self):
        trainer=RSSMTraining.__new__(RSSMTraining);trainer.args=SimpleNamespace(preview_assets_plan=self.path,renderer_checkpoint='original.pt',sr='original_sr.pt')
        trainer.device=torch.device('cpu');trainer.preview_model=None;trainer.state={'global_training_step':4250}
        real=prepare_preview_assets
        def prepare(model,plan,device):return real(model,plan,device,clone_factory=lambda config:copy.deepcopy(model))
        with patch('turkeyworld.rssm_preview_assets.prepare_preview_assets',side_effect=prepare) as create:
            first=trainer.initialize_preview_assets(self.source);client=trainer.preview_model
            with torch.no_grad():self.source.vjepa_backend.core.bias.add_(.25)
            trainer.state['global_training_step']=4251;second=trainer.initialize_preview_assets(self.source)
            self.assertEqual(create.call_count,1);self.assertIs(trainer.preview_model,client)
            self.assertEqual(second['current_world_step'],4251);self.assertEqual(first['renderer_refinement_updates'],6000)
            self.assertEqual(trainer.args.renderer_checkpoint,'original.pt');self.assertEqual(trainer.args.sr,'original_sr.pt')
            changed=copy.deepcopy(self.raw);changed['note']='different immutable plan';self.path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'Immutable'):trainer.initialize_preview_assets(self.source)
            self.assertIs(trainer.preview_model,client)
        plan=read_preview_assets_plan(self.path);new=copy.deepcopy(plan);new['sr']['file_sha256']='f'*64
        self.assertNotEqual(preview_assets_cache_key(plan,self.source.config),preview_assets_cache_key(new,self.source.config))

    def test_private_rng_and_five_row_chain_use_prediction_rgb_only(self):
        client,sr,metadata,_=self.prepare(); private=np.random.default_rng(17);saved=copy.deepcopy(private.bit_generator.state)
        with preserve_preview_rng('cpu',private):torch.rand(4);np.random.random();random.random();private.random()
        self.assertEqual(saved,private.bit_generator.state)
        generated=torch.full((4,3,32,32),.23);seen=[]
        hook=sr.register_forward_pre_hook(lambda module,args:seen.append(args[0].detach().clone()))
        preview={'context':np.zeros((1,32,32,3),np.uint8),'source':np.full((4,32,32,3),255,np.uint8),
            'generated':np.full((4,32,32,3),59,np.uint8),'generated_float':generated,
            'metadata':{'step':4251,'kind':'rssm_free_autoregressive_future','caption':'updated frozen preview','generated_finite':True}}
        try:report=save_autoregressive_preview(preview,self.root/'preview.png',step=4251,every=10,sr_model=sr,sr_metadata=metadata)
        finally:hook.remove()
        torch.testing.assert_close(seen[0],generated,rtol=0,atol=0)
        self.assertEqual(report['row_count'],5);self.assertEqual(report['sr_resolution'],[128,128]);self.assertEqual(report['upscaled_resolution'],[512,512])
        self.assertFalse(report['future_source_feedback']);self.assertFalse(report['sr_world_feedback'])

    def revision(self):
        payload=torch.load(self.root/'renderer.pt',weights_only=False)
        payload['model']['output.weight'].add_(.001)
        payload['training_steps']=16200;payload['metrics']['refinement_updates']=6600
        payload['base_student_bundle']['training_steps']=15800
        torch.save(payload,self.root/'renderer6600.pt')
        selected=copy.deepcopy(self.raw);selected['renderer'].update(checkpoint=str(self.root/'renderer6600.pt'),
            file_sha256=file_sha256(self.root/'renderer6600.pt'),refinement_updates=6600)
        selected_path=self.root/'assets6600.json';selected_path.write_text(json.dumps(selected),encoding='utf-8')
        revision={'version':1,'kind':'preview_assets_revision_v1','initial_plan_sha256':file_sha256(self.path),
            'selected_plan':str(selected_path),'selected_plan_sha256':file_sha256(selected_path)}
        revision_path=self.root/'revision.json';revision_path.write_text(json.dumps(revision),encoding='utf-8')
        return revision_path,revision,selected_path

    def test_explicit_preview6600_revision_preserves_original_pin_training_and_history_and_is_required_on_resume(self):
        initial_bytes=self.path.read_bytes();source_hash=_hash_tensors(self.source.state_dict())
        trainer=RSSMTraining.__new__(RSSMTraining)
        trainer.args=SimpleNamespace(preview_assets_plan=self.path,preview_assets_revision_plan=None,renderer_checkpoint='original.pt',sr='original_sr.pt')
        trainer.device=torch.device('cpu');trainer.preview_model=None;trainer.state={'global_training_step':4400}
        real=prepare_preview_assets
        def prepare(model,plan,device):return real(model,plan,device,clone_factory=lambda config:copy.deepcopy(model))
        with patch('turkeyworld.rssm_preview_assets.prepare_preview_assets',side_effect=prepare):
            original=trainer.initialize_preview_assets(self.source);original_client=trainer.preview_model
            revision_path,revision,selected_path=self.revision()
            trainer.args.preview_assets_revision_plan=revision_path;trainer.state['global_training_step']=4401
            torch_state=torch.get_rng_state().clone();updated=trainer.initialize_preview_assets(self.source)
            self.assertEqual(updated['renderer_refinement_updates'],6600);self.assertEqual(updated['renderer_training_steps'],16200)
            self.assertFalse(updated['renderer_source_quality_passed']);self.assertFalse(updated['sr_source_quality_passed'])
            self.assertIsNot(trainer.preview_model,original_client);self.assertTrue(torch.equal(torch_state,torch.get_rng_state()))
            self.assertEqual(self.path.read_bytes(),initial_bytes);self.assertEqual(_hash_tensors(self.source.state_dict()),source_hash)
            self.assertEqual(trainer.args.renderer_checkpoint,'original.pt');self.assertEqual(trainer.args.sr,'original_sr.pt')
            self.assertEqual(len(trainer.state['preview_assets_history']),2)
            self.assertEqual(trainer.state['preview_assets_history'][0]['plan_sha256'],original['plan_sha256'])
            self.assertEqual(trainer.state['preview_assets_revision']['initial_plan_sha256'],file_sha256(self.path))
            restored=RSSMTraining.__new__(RSSMTraining);restored.args=copy.copy(trainer.args);restored.device=torch.device('cpu')
            restored.preview_model=None;restored.state=copy.deepcopy(trainer.state)
            self.assertEqual(restored.initialize_preview_assets(self.source)['renderer_refinement_updates'],6600)
            self.assertEqual(len(restored.state['preview_assets_history']),2)
            client=restored.preview_model;state=copy.deepcopy(restored.state)
            restored.args.preview_assets_revision_plan=None
            with self.assertRaisesRegex(ValueError,'requires the saved'):restored.initialize_preview_assets(self.source)
            self.assertIs(restored.preview_model,client);self.assertEqual(restored.state,state)
            restored.args.preview_assets_revision_plan=revision_path
            changed=json.loads(selected_path.read_text());changed['note']='silent change';selected_path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'SHA'):restored.initialize_preview_assets(self.source)
            self.assertIs(restored.preview_model,client);self.assertEqual(restored.state,state)

    def test_revision_wrong_parent_hash_selected_hash_or_saved_identity_rejects(self):
        source_before=_hash_tensors(self.source.state_dict())
        path,revision,selected=self.revision()
        _,_,record=resolve_preview_assets_plan(self.path,path)
        for field,value in [('version',True),('initial_plan_sha256','0'*64),('selected_plan_sha256','f'*64),('selected_plan','relative.json')]:
            bad={**revision,field:value};path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):resolve_preview_assets_plan(self.path,path)
        path.write_text(json.dumps(revision))
        with self.assertRaises(ValueError):resolve_preview_assets_plan(self.path,None,record)
        with self.assertRaises(ValueError):resolve_preview_assets_plan(self.path,path,{**record,'revision_sha256':'different'})
        self.assertEqual(_hash_tensors(self.source.state_dict()),source_before)

    def chained_revision(self,parent_path,*,name='latest',phase=6000,total=22200):
        parent=json.loads(parent_path.read_text())
        old_plan=json.loads(Path(parent['selected_plan']).read_text())
        payload=torch.load(old_plan['renderer']['checkpoint'],weights_only=False)
        payload['model']['output.weight'].add_(.002)
        payload['training_steps']=total;payload['metrics']['refinement_updates']=phase
        payload['base_student_bundle']['training_steps']=total-400
        renderer=self.root/(name+'_renderer.pt');torch.save(payload,renderer)
        selected=copy.deepcopy(self.raw);selected['renderer'].update(checkpoint=str(renderer),
            file_sha256=file_sha256(renderer),refinement_updates=phase)
        selected_path=self.root/(name+'_assets.json');selected_path.write_text(json.dumps(selected))
        revision={'version':2,'kind':'preview_assets_revision_v2','initial_plan_sha256':file_sha256(self.path),
            'previous_revision_plan':str(parent_path),'previous_revision_sha256':file_sha256(parent_path),
            'selected_plan':str(selected_path),'selected_plan_sha256':file_sha256(selected_path)}
        path=self.root/(name+'_revision.json');path.write_text(json.dumps(revision))
        return path,revision,selected_path

    def test_v2_only_direct_parent_activation_or_exact_resume_and_no_ancestor_skipping(self):
        old_path,_,_=self.revision();_,_,old=resolve_preview_assets_plan(self.path,old_path)
        path,_,_=self.chained_revision(old_path)
        _,_,current=resolve_preview_assets_plan(self.path,path,old)
        self.assertEqual(current['kind'],'preview_assets_revision_v2')
        self.assertEqual(current['previous_revision_sha256'],old['revision_sha256'])
        self.assertEqual(resolve_preview_assets_plan(self.path,path,current)[2],current)
        for invalid in (None,{**old,'revision_sha256':'f'*64}):
            with self.assertRaisesRegex(ValueError,'immediate parent'):resolve_preview_assets_plan(self.path,path,invalid)
        future_path,_,_=self.chained_revision(path,name='future',phase=6100,total=22300)
        with self.assertRaisesRegex(ValueError,'immediate parent'):resolve_preview_assets_plan(self.path,future_path,old)
        _,_,future=resolve_preview_assets_plan(self.path,future_path,current)
        self.assertEqual(resolve_preview_assets_plan(self.path,future_path,future)[2],future)

    def test_v2_full_chain_rejects_tampered_ancestor_assets_missing_parent_cycles_and_wrong_roots(self):
        old_path,_,old_assets=self.revision();_,_,old=resolve_preview_assets_plan(self.path,old_path)
        path,raw,_=self.chained_revision(old_path)
        old_bytes=old_path.read_bytes();old_asset_bytes=old_assets.read_bytes()
        changed=json.loads(old_bytes);changed['note']='ancestor changed';old_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError,'SHA'):resolve_preview_assets_plan(self.path,path,old)
        old_path.write_bytes(old_bytes)
        changed=json.loads(old_asset_bytes);changed['note']='ancestor asset changed';old_assets.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError,'SHA'):resolve_preview_assets_plan(self.path,path,old)
        old_assets.write_bytes(old_asset_bytes);old_path.unlink()
        with self.assertRaisesRegex(ValueError,'SHA/path'):resolve_preview_assets_plan(self.path,path,old)
        old_path.write_bytes(old_bytes)
        for key,value,pattern in [('previous_revision_plan',str(path),'cycle'),
            ('previous_revision_plan','relative.json','SHA/path'),('previous_revision_sha256','f'*64,'SHA'),
            ('initial_plan_sha256','0'*64,'initial'),('version',True,'initial')]:
            path.write_text(json.dumps({**raw,key:value}))
            with self.assertRaisesRegex(ValueError,pattern):resolve_preview_assets_plan(self.path,path,old)
        path.write_text(json.dumps(raw));resolve_preview_assets_plan(self.path,path,old)

    def test_v2_worker_hot_and_cold_cache_preserve_training_adam_rng_curriculum_and_revision_history(self):
        old_path,_,_=self.revision()
        trainer=RSSMTraining.__new__(RSSMTraining)
        trainer.args=SimpleNamespace(preview_assets_plan=self.path,preview_assets_revision_plan=old_path,
            renderer_checkpoint='original.pt',sr='original_sr.pt')
        trainer.device=torch.device('cpu');trainer.preview_model=None
        trainer.state={'global_training_step':5330,'rssm_training':{'gate':{'current_horizon':128,'streak':0},
            'sampler':{'draws':1234}},'training_protocol':'unchanged'}
        optimizer=torch.optim.Adam(self.source.vjepa_backend.core.parameters(),lr=.001)
        self.source.vjepa_backend.core(torch.ones(1,20)).square().mean().backward();optimizer.step()
        adam=copy.deepcopy(optimizer.state_dict());weights=_hash_tensors(self.source.state_dict())
        ids=[id(p)for p in self.source.parameters()];training=copy.deepcopy(trainer.state['rssm_training'])
        real=prepare_preview_assets
        def prepare(model,plan,device):return real(model,plan,device,clone_factory=lambda config:copy.deepcopy(model))
        initial_bytes=self.path.read_bytes();old_bytes=old_path.read_bytes()
        with patch('turkeyworld.rssm_preview_assets.prepare_preview_assets',side_effect=prepare):
            trainer.initialize_preview_assets(self.source);old_client=trainer.preview_model
            path,_,_=self.chained_revision(old_path)
            trainer.args.preview_assets_revision_plan=path
            rng=torch.get_rng_state().clone();updated=trainer.initialize_preview_assets(self.source)
            self.assertEqual(updated['renderer_training_steps'],22200)
            self.assertEqual(updated['renderer_refinement_updates'],6000)
            self.assertEqual(updated['base_training_steps'],21800)
            self.assertFalse(updated['renderer_source_quality_passed']);self.assertFalse(updated['gameplay_qualified'])
            self.assertIsNot(old_client,trainer.preview_model)
            self.assertEqual(len(trainer.state['preview_assets_history']),3)
            self.assertEqual(len(trainer.state['preview_assets_revision_history']),2)
            self.assertEqual(trainer.state['rssm_training'],training)
            self.assertEqual(_hash_tensors(self.source.state_dict()),weights)
            self.assertEqual([id(p)for p in self.source.parameters()],ids)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for key,row in adam['state'].items():
                for name,value in row.items():torch.testing.assert_close(value,optimizer.state_dict()['state'][key][name],rtol=0,atol=0)
            self.assertEqual(adam['param_groups'],optimizer.state_dict()['param_groups'])
            self.assertEqual(trainer.args.renderer_checkpoint,'original.pt');self.assertEqual(trainer.args.sr,'original_sr.pt')
            self.assertEqual(self.path.read_bytes(),initial_bytes);self.assertEqual(old_path.read_bytes(),old_bytes)
            restored=RSSMTraining.__new__(RSSMTraining);restored.args=copy.copy(trainer.args)
            restored.device=torch.device('cpu');restored.preview_model=None;restored.state=copy.deepcopy(trainer.state)
            restored.initialize_preview_assets(self.source)
            self.assertEqual(len(restored.state['preview_assets_history']),3)
            self.assertEqual(len(restored.state['preview_assets_revision_history']),2)
            current=restored.preview_model;state=copy.deepcopy(restored.state)
            restored.args.preview_assets_revision_plan=old_path
            with self.assertRaises(ValueError):restored.initialize_preview_assets(self.source)
            self.assertIs(restored.preview_model,current);self.assertEqual(restored.state,state)


if __name__=='__main__':unittest.main()
