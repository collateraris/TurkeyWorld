from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from test_renderer_student import TinyTeacher
from turkeyworld.renderer_student import _hash_tensors
from turkeyworld.rssm_preview_assets import prepare_preview_assets,file_sha256
from turkeyworld.rssm_preview_image_chain import PROFILE,validate_image_chain_plan


def profile_plan():
    return dict(inference_profile=PROFILE,preview_only=True,training_assets_changed=False,
        game_assets_changed=False,allow_unqualified_codec=True,plan_path='new-assets.json',plan_sha256='newplan',
        renderer=dict(checkpoint='renderer.pt',file_sha256='r',format_version=8,kind='decoder_rf_chain_bundle_v1',
            solver='euler',solver_steps=3,training_steps=32317,base_training_steps=31709,
            decoder_refinement_updates=4125,restorer_present=False),
        sr=dict(checkpoint='sr.pt',file_sha256='s',architecture='background_detail_sr',training_steps=3000,
            selected_local_update=3000,input_size=32,output_size=128,inference_inputs='RGB32_only',
            alpha_used=False,atlas_used=False,hypothesis_used=False,preferred_background_sha256='base'))


class ImageChainPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_explicit_chain_clone_keeps_training_ids_weights_modes_and_all_rng_untouched(self):
        source=TinyTeacher();source.combat_rssm_migration={'original_codec_lineage':'keep'}
        source_hash=_hash_tensors(source.state_dict());ids=[id(p)for p in source.parameters()]
        modes=[m.training for m in source.modules()];rng=torch.get_rng_state().clone();py=random.getstate();nr=np.random.get_state()
        plan=profile_plan();loaded=[]
        def renderer(client,path,**kwargs):
            self.assertIsNot(client,source);self.assertFalse(hasattr(client,'base_student'))
            self.assertEqual(client.combat_rssm_migration,source.combat_rssm_migration)
            self.assertIsNot(client.combat_rssm_migration,source.combat_rssm_migration)
            client.flow_renderer.kind='student';client.flow_renderer.steps=3
            client.flow_renderer._renderer_decoder_chain={'format_version':8}
            client.base_student=nn.Linear(2,2);client.base_student_bundle={'training_steps':31709}
            loaded.append(path);return{**plan['renderer'],'refinement_updates':3517}
        def sr(path,device):
            random.random();np.random.random();torch.rand(1)
            return nn.Linear(2,2),dict(training_steps=3000,background_detail_updates=3000)
        srmeta=dict(architecture='background_detail_sr',frozen_background_sha256='base',alpha_used=False,
            atlas_class_used=False,atlas_buffers_present=False,atlas_asset_required=False,hypothesis_used=False)
        with patch('turkeyworld.renderer_student.load_renderer_student',side_effect=renderer),\
             patch('turkeyworld.models.load_sr_checkpoint',side_effect=sr),\
             patch('turkeyworld.background_detail_sr.background_detail_metadata',return_value=srmeta):
            client,upscaler,metadata,proof=prepare_preview_assets(source,plan,'cpu',clone_factory=lambda config:deepcopy(source))
        self.assertEqual(loaded,['renderer.pt']);self.assertEqual(proof['renderer_solver_steps'],3)
        self.assertEqual(proof['renderer_format_version'],8);self.assertEqual(proof['decoder_refinement_updates'],4125)
        self.assertEqual(proof['sr_training_steps'],3000);self.assertFalse(proof['sr_atlas_used'])
        self.assertFalse(proof['gameplay_qualified']);self.assertFalse(proof['source_future_feedback'])
        self.assertTrue(all(not p.requires_grad for p in client.parameters()))
        self.assertTrue(all(not p.requires_grad for p in upscaler.parameters()))
        self.assertEqual(_hash_tensors(source.state_dict()),source_hash);self.assertEqual(ids,[id(p)for p in source.parameters()])
        self.assertEqual(modes,[m.training for m in source.modules()]);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(py,random.getstate());self.assertTrue(all(np.array_equal(a,b)for a,b in zip(nr,np.random.get_state())))

    def test_old_profile_cannot_silently_accept_chain_or_atlas_and_unknown_profiles_fail_closed(self):
        for mutation in(lambda p:p['renderer'].update(solver_steps=1),lambda p:p['renderer'].update(restorer_present=True),
                        lambda p:p['sr'].update(architecture='atlas_enemy_edsr'),lambda p:p['sr'].update(alpha_used=True),
                        lambda p:p.update(training_assets_changed=True),lambda p:p['sr'].update(selected_local_update=0)):
            plan=profile_plan();mutation(plan)
            with self.assertRaisesRegex(ValueError,'explicit complete'):validate_image_chain_plan(plan)
        plan=profile_plan();plan['inference_profile']='unknown'
        with self.assertRaisesRegex(ValueError,'Unknown explicit'):prepare_preview_assets(TinyTeacher(),plan,'cpu')

    def test_prepare_only_new_plan_preserves_MAIN_ladder_archive_and_only_changes_one_flag(self):
        path=Path(__file__).resolve().parents[1]/'training_runs/diagnostic_tools/prepare_image_chain_preview_revision_20261114.py'
        spec=importlib.util.spec_from_file_location('prepare_image_chain_preview_test',path);ops=importlib.util.module_from_spec(spec);spec.loader.exec_module(ops)
        with tempfile.TemporaryDirectory()as folder:
            run=Path(folder);(run/'checkpoints').mkdir();cp=run/'checkpoints/rssm_world.pt';cp.write_bytes(b'not deserialized checkpoint')
            renderer=run/'renderer.pt';renderer.write_bytes(b'complete selected image chain')
            sr=run/'sr.pt';sr.write_bytes(b'complete frozen new RGB SR')
            selected={**profile_plan(),'version':1,'kind':'frozen_preview_assets_v1'}
            selected['renderer'].update(checkpoint=str(renderer),file_sha256=file_sha256(renderer))
            selected['sr'].update(checkpoint=str(sr),file_sha256=file_sha256(sr))
            initial=run/'initial.json';initial.write_text(json.dumps(selected))
            revision=run/'old_revision.json';revision.write_text(json.dumps(dict(version=1,kind='preview_assets_revision_v1',
                initial_plan_sha256=file_sha256(initial),selected_plan=str(initial),selected_plan_sha256=file_sha256(initial))))
            from turkeyworld.rssm_preview_assets import resolve_preview_assets_plan
            _,_,record=resolve_preview_assets_plan(initial,revision)
            gate=dict(last_optimizer_step=6745,main_ladder_revision=dict(protocol='rssm_main18_full_bptt_loss_ladder_v1',
                last_optimizer_step=6745,source_optimizer_step=6686,updates=59,stage_updates=59,stage_streak=0,
                stage_index=0,current_horizon=16,sha256='same-original-main-plan'))
            state=dict(status='stopped',global_training_step=6745,rssm_training={'moe_training':gate},
                training_protocol={'main_ladder_revision':{'plan_sha256':'same-original-main-plan'}},preview_assets_revision=record)
            argv=['python','-m','turkeyworld.train_rssm_moe','--run-dir',str(run),'--main-ladder-plan','unchanged-main.json',
                '--preview-assets-revision-plan',str(revision)]
            (run/'status.json').write_text(json.dumps(state));(run/'launch_command.json').write_text(json.dumps({'argv':argv}))
            before={p:p.read_bytes()for p in[cp,initial,revision,run/'status.json',run/'launch_command.json']}
            receipt_path,receipt=ops.prepare(run,renderer_checkpoint=renderer,renderer_sha256=file_sha256(renderer),
                sr_checkpoint=sr,sr_sha256=file_sha256(sr),sr_step=3000,source_step=6745,
                source_checkpoint_sha256=file_sha256(cp),preferred_background_sha256='base')
            self.assertFalse(receipt['checkpoint_deserialized']);self.assertFalse(receipt['process_launched'])
            self.assertEqual(receipt['preserved_moe_gate'],gate);self.assertEqual(receipt['preserved_main_ladder']['updates'],59)
            self.assertEqual([i for i,(a,b)in enumerate(zip(argv,receipt['argv']))if a!=b],[argv.index('--preview-assets-revision-plan')+1])
            self.assertIn('--module',receipt['launch_argv']);self.assertTrue(receipt_path.is_file())
            for p,data in before.items():self.assertEqual(p.read_bytes(),data)
            same,_=ops.prepare(run,renderer_checkpoint=renderer,renderer_sha256=file_sha256(renderer),sr_checkpoint=sr,
                sr_sha256=file_sha256(sr),sr_step=3000,source_step=6745,source_checkpoint_sha256=file_sha256(cp),preferred_background_sha256='base')
            self.assertEqual(same,receipt_path)
            with self.assertRaisesRegex(ValueError,'Wait for'):
                ops.prepare(run,renderer_checkpoint=renderer,renderer_sha256=file_sha256(renderer),sr_checkpoint=sr,
                    sr_sha256=file_sha256(sr),sr_step=3000,source_step=6744,source_checkpoint_sha256=file_sha256(cp),preferred_background_sha256='base')


if __name__=='__main__':unittest.main()
