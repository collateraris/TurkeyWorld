from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

import torch


def operator():
    p=Path(__file__).resolve().parents[1]/'training_runs/diagnostic_tools/resume_image_chain_preview_20261114.py'
    spec=importlib.util.spec_from_file_location('resume_image_chain_ops_test',p)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def source_fixture(op):
    gate=dict(current_horizon=16,last_optimizer_step=6745,samplers={'16':{'PCG':{1:'typed-int-key'}}},
        main_ladder_revision=dict(source_optimizer_step=6686,updates=59,stage_updates=59,stage_streak=0,baseline_complete=True))
    q=dict(moe_training=gate,navigation_replay_revision={'sampler_state':{'draws':354},'enemy_sampler_state':{'draws':354}},
        player_navigation_revision={'baseline_complete':True},motion_correction_revision={'baseline_complete':True})
    before=dict(training_steps=6745,global_training_step=6745,model={str(i):torch.tensor(float(i))for i in range(1599)},
        optimizer={'state':{i:{'step':torch.tensor(6745.),'exp_avg':torch.tensor(.1)}for i in range(442)},
            'param_groups':[{'params':[],'lr':.001}for _ in range(7)]},optimizer_parameter_names=['same'],
        rng_state={'python':'same'},torch_rng_state=torch.tensor([1,2],dtype=torch.uint8),cuda_rng_state=['same'],
        rssm_training=deepcopy(q),config={'architecture':'dreamer_rssm'},
        run_state=dict(global_training_step=6745,rssm_training=q,training_protocol={'main_ladder':'original290d'},
            status='stopped',elapsed_seconds=100.,prediction_preview={'old':True}))
    revision={'revision_sha256':'newv5'};selected={'renderer':{'file_sha256':'renderer40f'},'sr':{'file_sha256':'src791'}}
    after=deepcopy(before);after['run_state'].update(status='running',elapsed_seconds=101.,
        preview_assets_revision=revision,prediction_preview={'new':True})
    preview=dict(step=6745,horizon=16,generated_rollout_transitions=16,generated_finite=True,
        future_source_feedback=False,source_held_out=True,current_stage_preview=True,row_count=5,
        sr_training_steps=3000,sr_world_feedback=False,renderer_solver_steps=3,
        caption='RF3 32317 Base31709 decoder4125 SR3000',preview_assets=dict(inference_profile=op.PROFILE,
            renderer_solver_steps=3,renderer_format_version=8,renderer=selected['renderer'],sr=selected['sr'],
            base_training_steps=31709,decoder_refinement_updates=4125,sr_training_steps=3000,
            restorer_present=False,sr_atlas_used=False,sr_hypothesis_used=False))
    return before,after,selected,revision,preview


class ImagePreviewResumeOperatorTests(unittest.TestCase):
    def test_actual_GPU_proof_requires_source_and_selected_file_pins_and_all_true_checks(self):
        op=operator();receipt=dict(source_step=6745,source_checkpoint_sha256=op.SOURCE_SHA,selected_plan_sha256='newplan')
        selected=dict(renderer={'file_sha256':'renderer40f'},sr={'file_sha256':'src791'})
        proof=dict(kind=op.PROOF_KIND,passed=True,source_step=6745,source_checkpoint_sha256=op.SOURCE_SHA,
            selected_plan_sha256='newplan',inference_profile=op.PROFILE,renderer_file_sha256='renderer40f',
            sr_file_sha256='src791',device='cuda',renderer_solver_steps=3,training_model_tensors_bitexact=True,
            checks={'actual_raw_clone':True,'finite':True,'RNG_preserved':True})
        self.assertIs(op.validate_gpu_proof(proof,receipt,selected),proof)
        for mutate in(lambda p:p.update(device='cpu'),lambda p:p.update(source_step=6744),
            lambda p:p.update(kind='background_detail_game_runtime_smoke_v1'),lambda p:p.update(sr_file_sha256='stripped_game_sr'),
            lambda p:p.update(checks={}),lambda p:p['checks'].update(finite=False)):
            bad=deepcopy(proof);mutate(bad)
            with self.assertRaisesRegex(ValueError,'actual raw-clone'):op.validate_gpu_proof(bad,receipt,selected)

    def test_full_typed_fingerprint_preserves_every_training_field_and_MAIN59_without_false_reset(self):
        op=operator();before,after,selected,revision,preview=source_fixture(op)
        good=op.verify_same_step(before,after,{},selected,revision,preview,immutable_inputs_unchanged=True)
        self.assertTrue(good['passed']);self.assertEqual(good['unexpected_changed_run_state_paths'],[])
        self.assertEqual(good['optimizer_updates_between_snapshots'],0);self.assertFalse(good['full450_validation_repeated'])
        for mutate in(lambda a:a['run_state']['rssm_training']['moe_training']['main_ladder_revision'].update(stage_updates=0),
            lambda a:a['run_state']['training_protocol'].update(main_ladder='rebound'),
            lambda a:a['optimizer']['state'][1]['exp_avg'].add_(.001),
            lambda a:a['model']['1'].add_(.001),
            lambda a:a['run_state']['rssm_training']['moe_training']['samplers']['16'].update(PCG={'1':'typed-int-key'}),
            lambda a:a.update(training_steps=6746)):
            bad=deepcopy(after);mutate(bad)
            result=op.verify_same_step(before,bad,{},selected,revision,preview,immutable_inputs_unchanged=True)
            self.assertFalse(result['passed'])
        bad=deepcopy(preview);bad['renderer_solver_steps']=1
        self.assertFalse(op.verify_same_step(before,after,{},selected,revision,bad,immutable_inputs_unchanged=True)['passed'])

    def test_only_preview_flag_changes_and_actual_entrypoint_keeps_earlyCUDA_bootstrap(self):
        op=operator();run=Path('test-run').resolve()
        old=['python','-X','utf8','-u','-m','turkeyworld.train_rssm_moe','--run-dir',str(run),
            '--main-ladder-plan','exact-old290d.json','--preview-assets-revision-plan','old-v4.json']
        new=op.PREPARE.revised_argv(old,'new-v5.json',run);boot=op.PREPARE.bootstrap_argv(new)
        self.assertEqual([i for i,(a,b)in enumerate(zip(old,new))if a!=b],[old.index('--preview-assets-revision-plan')+1])
        self.assertNotIn('-m',boot);self.assertIn('--module',boot)
        self.assertTrue(any('bootstrap_torch_cuda_20261114.py'in item for item in boot))
        self.assertIn('exact-old290d.json',boot);self.assertEqual(boot[boot.index('--module')+1],'turkeyworld.train_rssm_moe')
        self.assertNotIn('rssm_training',op.ALLOWED_OPERATIONAL);self.assertNotIn('training_protocol',op.ALLOWED_OPERATIONAL)


if __name__=='__main__':unittest.main()
