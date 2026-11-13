"""Measured phase clock and exact full-Adam SR continuation on real CPU SGD."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.decoder_restoration_protocol import (initial_restoration_schedule,
    validate_restoration_schedule,commit_restoration_update,next_restoration_blend_scale,
    bind_restoration_baseline,consider_restoration_gate)
from turkeyworld.enemy_decoder_migration import prepare_restoration_continuation
from turkeyworld.models import build_upscaler
from turkeyworld.train_semantic_sr import run
from test_enemy_decoder_training import GuidedPairs


def report(step,*,fg=.2,edge=.3,empty=.01,bg=.02):
    return dict(step=step,source_held_out=True,panel_fingerprint='panel',data_identity_sha256='data',
        restoration_raw=dict(raw_foreground_mae=fg,raw_foreground_edge_mae=edge,
                             raw_empty_mae=empty,raw_background_mae=bg))


def advance(schedule,step):
    while schedule['updates']<step:
        commit_restoration_update(schedule,step=schedule['updates']+1)


class RestorationProtocolTests(unittest.TestCase):
    def fixture(self):
        schedule=initial_restoration_schedule();baseline=report(0)
        bind_restoration_baseline(schedule,baseline,sha256='b'*64)
        return schedule,baseline

    def test_minimum200_quality_open_then400_complete_updates_preserve_exact_mid_ramp_resume(self):
        schedule,baseline=self.fixture();advance(schedule,100)
        early=consider_restoration_gate(schedule,report(100,fg=.1,edge=.1),baseline,step=100)
        self.assertFalse(early['passed']);self.assertEqual(schedule['blend_scale'],0.)
        advance(schedule,200)
        self.assertTrue(consider_restoration_gate(schedule,report(200,fg=.1,edge=.1),baseline,step=200)['passed'])
        self.assertEqual((schedule['phase'],schedule['blend_started_at'],schedule['blend_scale']),('blending',200,0.))
        self.assertEqual(next_restoration_blend_scale(schedule),1/400)
        advance(schedule,349)
        copied=json.loads(json.dumps(schedule))
        validate_restoration_schedule(copied,step=349,config_scale=149/400)
        advance(schedule,600);advance(copied,600)
        self.assertEqual(schedule,copied)
        self.assertEqual((schedule['phase'],schedule['blend_scale']),('restoration',1.))
        advance(schedule,620);self.assertEqual(schedule['blend_scale'],1.)

    def test_both_strict_raw_improvements_and_empty_background_guards_are_required(self):
        for changes in ({'fg':.2,'edge':.1},{'fg':.1,'edge':.3},
                        {'fg':.1,'edge':.1,'bg':.02001},{'fg':.1,'edge':.1,'empty':.01001}):
            schedule,baseline=self.fixture();advance(schedule,200)
            decision=consider_restoration_gate(schedule,report(200,**changes),baseline,step=200)
            self.assertFalse(decision['passed']);self.assertIsNone(schedule['blend_started_at'])
            self.assertEqual(schedule['blend_scale'],0.)

    def test_no_nonfinite_or_wrong_boundary_commit_and_no_reference_or_scale_tampering(self):
        schedule,baseline=self.fixture();advance(schedule,200);before=deepcopy(schedule)
        with self.assertRaises(ValueError):
            consider_restoration_gate(schedule,report(200,fg=float('nan')),baseline,step=200)
        self.assertEqual(schedule,before)
        with self.assertRaises(ValueError):commit_restoration_update(schedule,step=202)
        with self.assertRaises(ValueError):bind_restoration_baseline(schedule,baseline,sha256='c'*64)
        wrong=report(200);wrong['panel_fingerprint']='other'
        with self.assertRaises(ValueError):consider_restoration_gate(schedule,wrong,baseline,step=200)
        bad=deepcopy(schedule);bad['blend_scale']=1.
        with self.assertRaises(ValueError):validate_restoration_schedule(bad)
        bad=deepcopy(schedule);bad['settings']['warmup_minimum_updates']=0
        with self.assertRaises(ValueError):validate_restoration_schedule(bad)


def mocked_evaluate(model,data,panel,*,step,batch_size=8,baselines=None,baseline_report=None):
    batch=data.batch(len(panel),indices=panel)
    with torch.no_grad():model.forward_with_aux(batch['lr_npr'],batch['hypothesis'])
    result=report(step)
    result.update(kind='decoder_restoration_super_resolution_v3',selection_score=1-step*.01,
        relative_progress={'passed':False},training_goal={'passed':False},
        domains={'npr':{'models':{'current':{'summary':{'mae_mean':.1,'ssim_mean':.8}}}}},
        enemy_decoder={})
    return result,{}


class RestorationTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def initial(self,root):
        model=build_upscaler(dict(architecture='enemy_decoder_edsr',channels=4,blocks=16,
            semantic_channels=4,semantic_blocks=4,enemy_decoder_channels=8,
            enemy_decoder_stable=True,enemy_decoder_visual_first=True))
        names=list(dict(model.named_parameters()));params=dict(model.named_parameters())
        self.assertEqual(len(names),158)
        groups=[names[:74],names[74:102],names[102:149],names[149:]]
        optimizer=torch.optim.AdamW([{'params':[params[name]for name in row],'lr':5e-5}for row in groups])
        for index,parameter in enumerate(model.parameters()):
            parameter.grad=torch.full_like(parameter,.001+index*.00001)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        for index,parameter in enumerate(model.parameters()):optimizer.state[parameter]['step'].fill_(1+index%7)
        data=GuidedPairs(root,'train',hypotheses_root=root/'hyp',seed=173)
        sampler=data.sampler_state();sampler['draws']=44000
        source=dict(stage='sr',config=model.config,model=deepcopy(model.state_dict()),
            optimizer=deepcopy(optimizer.state_dict()),optimizer_parameter_names=groups,
            semantic_migration={'kind':'semantic'},enemy_decoder_migration={'kind':'decoder'},
            visual_first_migration={'kind':'visual'},training_steps=96200,
            semantic_updates=43100,decoder_updates=43100,visual_first_updates=43100,
            sampler=sampler,rng_numpy=np.random.default_rng(173).bit_generator.state,
            rng_torch=torch.get_rng_state(),rng_python=random.getstate(),rng_cuda=[],protocol={'old':'phase'})
        path=root/'parent.pt';initial=root/'initial.pt';torch.save(source,path)
        selected=root/'reference.pt'
        torch.save({k:source[k]for k in ('stage','config','model','training_steps')},selected)
        global_rng=torch.get_rng_state().clone();source_bytes=path.read_bytes()
        receipt=prepare_restoration_continuation(path,initial)
        self.assertEqual(source_bytes,path.read_bytes())
        torch.testing.assert_close(global_rng,torch.get_rng_state(),rtol=0,atol=0)
        migrated=torch.load(initial,weights_only=False)
        self.assertEqual(receipt['added_parameter_count'],0)
        self.assertEqual(receipt['old_Adam_states'],158)
        self.assertEqual(migrated['optimizer_parameter_names'],groups)
        self.assertEqual(migrated['sampler'],source['sampler'])
        self.assertEqual(migrated['optimizer']['param_groups'],source['optimizer']['param_groups'])
        for index,row in source['optimizer']['state'].items():
            for key,value in row.items():torch.testing.assert_close(value,migrated['optimizer']['state'][index][key],rtol=0,atol=0)
        for key,value in source['model'].items():torch.testing.assert_close(value,migrated['model'][key],rtol=0,atol=0)
        return initial,selected,source

    def args(self,root,initial,reference,updates):
        return argparse.Namespace(run_dir=root,data_root=root/'data',initial_checkpoint=initial,
            baseline38100=reference,baseline40100=reference,updates=updates,batch=4,
            validation_every=100,prediction_every=15,validation_frames=0,seed=173,device='cpu',
            enemy_decoder=True,visual_first=True,restoration=True,hypotheses_root=root/'hyp')

    def invoke(self,args):
        with patch('turkeyworld.enemy_decoder_data.EnemyDecoderData',GuidedPairs), \
                patch('turkeyworld.semantic_sr_validation.evaluate_semantic_sr',mocked_evaluate), \
                patch('turkeyworld.semantic_sr_validation.save_semantic_sr_preview',
                      side_effect=lambda p,path,every:dict(path=str(path),every=every)):
            run(args)

    def test_real_sgd_exact_resume_preserves_frozen102_moments_and_sampler_and_zero_display_phase(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);initial,reference,parent=self.initial(root)
            self.invoke(self.args(root/'continuous',initial,reference,2))
            self.invoke(self.args(root/'resumed',initial,reference,1))
            self.invoke(self.args(root/'resumed',initial,reference,2))
            a=torch.load(root/'continuous/checkpoints/sr_last.pt',weights_only=False)
            b=torch.load(root/'resumed/checkpoints/sr_last.pt',weights_only=False)
            for key in ('optimizer_parameter_names','sampler','rng_numpy','rng_python','restoration_schedule'):
                self.assertEqual(a[key],b[key])
            torch.testing.assert_close(a['rng_torch'],b['rng_torch'],rtol=0,atol=0)
            self.assertEqual((a['restoration_updates'],a['training_steps'],a['sampler']['draws']),(2,96202,44008))
            self.assertEqual(a['config']['restoration_blend_scale'],0.)
            self.assertEqual(a['restoration_schedule']['phase'],'standalone')
            for key,value in a['model'].items():torch.testing.assert_close(value,b['model'][key],rtol=0,atol=0)
            for index,row in a['optimizer']['state'].items():
                for key,value in row.items():torch.testing.assert_close(value,b['optimizer']['state'][index][key],rtol=0,atol=0)
            for names,group in zip(parent['optimizer_parameter_names'][:2],parent['optimizer']['param_groups'][:2]):
                for name,index in zip(names,group['params']):
                    torch.testing.assert_close(a['model'][name],parent['model'][name],rtol=0,atol=0)
                    for key,value in parent['optimizer']['state'][index].items():
                        torch.testing.assert_close(a['optimizer']['state'][index][key],value,rtol=0,atol=0)
            self.assertTrue(any(not torch.equal(value,a['model'][name])for name,value in parent['model'].items()
                                if name.startswith('enemy_decoder_')))
            status=json.loads((root/'resumed/status.json').read_text(encoding='utf-8'))
            self.assertEqual(status['restoration_training']['protocol'],'decoder_restoration_sr_v3')
            self.assertEqual(status['restoration_training']['blend_scale'],0.)
            self.assertEqual(status['semantic_sr_training']['protocol'],'decoder_restoration_super_resolution_v3')
            self.assertEqual(status['prediction_every'],15)
            self.assertFalse(status['automatic_deployment'])

    def test_strict_builder_rejects_a_saved_restoration_model_without_actual_blend_scale(self):
        with self.assertRaisesRegex(ValueError,'actual blend scale'):
            build_upscaler(dict(architecture='enemy_decoder_edsr',enemy_decoder_stable=True,
                enemy_decoder_visual_first=True,enemy_decoder_restoration=True))


if __name__=='__main__':unittest.main()
