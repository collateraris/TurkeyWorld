"""Explicit resize lineage, real compact Base/RF loading, no training/GPU."""
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

import test_dual_rssm_migration as dual_tests
from turkeyworld.base_student import build_base_student, export_base_student
from turkeyworld.dual_rssm_migration import enable_dual_rssm_checkpoint
from turkeyworld.models import NeuralWorldModel, STATE_SCHEMA, load_world_checkpoint
from turkeyworld.renderer_student import (build_renderer_student, export_renderer_student,
    load_renderer_student, renderer_condition_identity, validate_dual_renderer_lineage,
    _hash_tensors)


def fingerprint(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()


class EnemyResizeRendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def fixture(self,root):
        helper=dual_tests.DualRSSMMigrationTests();legacy,_,saved=helper.fixture()
        dual,append=enable_dual_rssm_checkpoint(saved)
        small,_=helper.restore(dual)
        config=replace(small.config,enemy_rssm_deter_dim=400,enemy_rssm_hidden_dim=416)
        resized=NeuralWorldModel(config).eval()
        new=resized.state_dict()
        for key,value in small.state_dict().items():
            if not key.startswith('vjepa_backend.enemy_rssm.'):
                new[key].copy_(value)
        resized.rssm_migration=copy.deepcopy(small.rssm_migration)
        resized.dual_rssm_migration=copy.deepcopy(append)
        previous,target=asdict(small.config),asdict(config)
        report={'format_version':1,'kind':'resize_independent_enemy_rssm',
            'source_config':previous,'target_config':target,
            'source_config_sha256':fingerprint(previous),'target_config_sha256':fingerprint(target),
            'allowed_config_changes':['enemy_rssm_deter_dim','enemy_rssm_hidden_dim'],
            'player_and_shared_model_tensors_bitexact':True,
            'player_and_shared_adam_states_bitexact':True,'frozen_render_assets_bitexact':True,
            'source_renderer_conditioning_identity':renderer_condition_identity(small)}
        resized.enemy_rssm_resize=report
        base=build_base_student(legacy)
        export_base_student(legacy,base,root/'base.pt',training_steps=3)
        legacy.base_student=base;legacy.base_student_bundle=torch.load(root/'base.pt',weights_only=False)
        flow=build_renderer_student(legacy,steps=1,width=16,depth=1,heads=2,mlp_hidden=32)
        export_renderer_student(legacy,flow,root/'legacy_rf.pt',training_steps=7,
            metrics={'source_quality_passed':False})
        return legacy,small,resized,report

    def test_original_renderer_bundle_loads_only_through_verified_append_then_resize_without_nonflow_mutation(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);legacy,small,target,report=self.fixture(root)
            unchanged={k:v.clone()for k,v in target.state_dict().items()if not k.startswith(('flow_renderer.','base_student.'))}
            del target.enemy_rssm_resize
            before=_hash_tensors(target.state_dict())
            with self.assertRaisesRegex(ValueError,'resize lineage'):load_renderer_student(target,root/'legacy_rf.pt')
            self.assertEqual(before,_hash_tensors(target.state_dict()))
            target.enemy_rssm_resize=report
            metadata=load_renderer_student(target,root/'legacy_rf.pt')
            self.assertEqual((target.config.enemy_rssm_deter_dim,target.config.enemy_rssm_hidden_dim),(400,416))
            self.assertEqual(target.vjepa_backend.memory_size,target.core.memory_size+400+8*32)
            self.assertEqual(target.flow_renderer.steps,1)
            self.assertFalse(metadata['metrics']['source_quality_passed'])
            self.assertTrue(hasattr(target,'base_student'))
            self.assertTrue(all(torch.equal(v,target.state_dict()[k])for k,v in unchanged.items()))

    def test_already_dual_source_config_requires_the_exact_resize_source_and_rejects_forged_or_other_config_changes(self):
        with tempfile.TemporaryDirectory()as folder:
            legacy,small,target,report=self.fixture(Path(folder))
            validate_dual_renderer_lineage(target,asdict(small.config))
            before=_hash_tensors(target.state_dict())
            for bad in ('kind','missing_player','missing_adam','missing_frozen','fingerprint','identity','extra_change','wrong_target','wrong_source'):
                candidate=copy.deepcopy(report)
                if bad=='kind':candidate['kind']='unverified'
                elif bad=='missing_player':candidate['player_and_shared_model_tensors_bitexact']=False
                elif bad=='missing_adam':candidate['player_and_shared_adam_states_bitexact']=False
                elif bad=='missing_frozen':candidate['frozen_render_assets_bitexact']=False
                elif bad=='fingerprint':candidate['source_config_sha256']='wrong'
                elif bad=='identity':candidate['source_renderer_conditioning_identity']['fixed_noise_sha256']='wrong'
                elif bad=='extra_change':
                    candidate['target_config']['rssm_hidden_dim']+=8
                    candidate['target_config_sha256']=fingerprint(candidate['target_config'])
                elif bad=='wrong_target':
                    candidate['target_config']['enemy_rssm_deter_dim']+=4
                    candidate['target_config_sha256']=fingerprint(candidate['target_config'])
                else:
                    candidate['source_config']['enemy_rssm_hidden_dim']+=4
                    candidate['source_config_sha256']=fingerprint(candidate['source_config'])
                target.enemy_rssm_resize=candidate
                with self.subTest(bad=bad),self.assertRaises(ValueError):
                    validate_dual_renderer_lineage(target,asdict(small.config))
                self.assertEqual(before,_hash_tensors(target.state_dict()))
            target.enemy_rssm_resize=report
            # The original append lineage cannot be silently omitted or changed.
            target.dual_rssm_migration=None
            with self.assertRaises(ValueError):validate_dual_renderer_lineage(target,asdict(legacy.config))

    def test_loader_retains_resize_metadata_and_actual_asset_tampering_still_rejects(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);legacy,small,target,report=self.fixture(root)
            path=root/'world.pt'
            torch.save({'stage':'world','state_schema':dict(STATE_SCHEMA),'config':asdict(target.config),
                'model':target.state_dict(),'rssm_migration':target.rssm_migration,
                'dual_rssm_migration':target.dual_rssm_migration,'enemy_rssm_resize':report},path)
            restored,_=load_world_checkpoint(path)
            self.assertEqual(restored.enemy_rssm_resize,report)
            validate_dual_renderer_lineage(restored,asdict(legacy.config))
            with torch.no_grad():restored.flow_renderer.latent_mean.flatten()[0].add_(.01)
            before=_hash_tensors(restored.state_dict())
            with self.assertRaises(ValueError):load_renderer_student(restored,root/'legacy_rf.pt')
            self.assertEqual(before,_hash_tensors(restored.state_dict()))


if __name__=='__main__':unittest.main()
