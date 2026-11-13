"""Actual CPU optimizer commits across a full parent->decoder continuation."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.enemy_decoder_data import hr_box_and_visibility
from turkeyworld.enemy_decoder_migration import prepare_enemy_decoder_continuation
from turkeyworld.enemy_decoder_objectives import PROTOCOL
from turkeyworld.models import build_upscaler
from turkeyworld.train_semantic_sr import run,mix_enemy_hypotheses
import test_semantic_sr_training as semantic_training

TinyPairs=semantic_training.TinyPairs
preview=semantic_training.preview


class GuidedPairs(TinyPairs):
    def __init__(self,root,split,*,hypotheses_root,seed=173,device='cpu'):
        super().__init__(root,split,seed=seed,device=device)
        self.identity.update(hypothesis='frozen-causal-prior-fixture',sampler='actual-visible-empty-new-clock')
        state=torch.zeros(8,20);state[:,:2]=.25;state[:,3]=1.;state[:,4]=1.
        state[:,8]=.25+torch.tensor([1.,2.,3.,4.,5.,6.,7.,8.])/16
        state[:,9]=.25;state[:,12:14]=1.
        self.hypothesis=torch.cat((state,torch.ones(8,1)),1)
        self.source_animation_label=torch.tensor([0,1,2,3,4,5,6,0],dtype=torch.long)

    def batch(self,size,*,indices=None,rng=None):
        if indices is None:
            rng=self.rng if rng is None else rng
            indices=np.concatenate((rng.choice(6,size=size//2),rng.choice([6,7],size=size-size//2)))
            rng.shuffle(indices);indices=indices.tolist();self.draws+=size
        result=super().batch(size,indices=indices,rng=rng)
        result['hypothesis']=self.hypothesis[torch.as_tensor(indices)].clone()
        result['bbox_hr'],result['visible_labels']=hr_box_and_visibility(result['mask_hr'])
        result['animation_label']=self.source_animation_label[torch.as_tensor(indices)].clone()
        return result


def guided_evaluate(model,data,panel,*,step,batch_size=8,baselines=None,baseline_report=None):
    before=data.sampler_state();batch=data.batch(8,indices=panel)
    with torch.no_grad():
        prediction=model(batch['lr_npr'],batch['hypothesis'])
    if before!=data.sampler_state()or not torch.isfinite(prediction).all():
        raise AssertionError('Guided evaluation changed sampling or produced invalid RGB')
    row={'summary':{'mae_mean':.1,'ssim_mean':.8}}
    return {'kind':'enemy_decoder_super_resolution_v1','step':step,'data_identity_sha256':'guided-fixed-identity',
        'panel_fingerprint':'fixed-panel8','selection_score':1-step*.01,
        'relative_progress':{'passed':False},'training_goal':{'passed':False},
        'enemy_decoder':{'hypothesis_frames':8},'domains':{'npr':{'models':{'current':row}}}},{ }


class EnemyDecoderTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def migrated(self,root,*,stable=False):
        helper=semantic_training.SemanticTrainingTests()
        initial,selected=helper.initial(root)
        helper.invoke(helper.args(root/'semantic_parent',initial,selected,1))
        parent=root/'semantic_parent/checkpoints/sr_last.pt'
        migrated=root/'decoder_initial.pt'
        report=prepare_enemy_decoder_continuation(parent,migrated,seed=173,stable=stable)
        return migrated,selected,parent,report

    def args(self,root,initial,selected,updates,hypotheses_root):
        return argparse.Namespace(run_dir=root,data_root=root/'data',initial_checkpoint=initial,
            baseline38100=selected,baseline40100=selected,updates=updates,batch=4,
            validation_every=1,prediction_every=15,validation_frames=0,seed=173,device='cpu',
            enemy_decoder=True,hypotheses_root=hypotheses_root)

    def invoke(self,args):
        with patch('turkeyworld.enemy_decoder_data.EnemyDecoderData',GuidedPairs), \
                patch('turkeyworld.semantic_sr_validation.evaluate_semantic_sr',guided_evaluate), \
                patch('turkeyworld.semantic_sr_validation.save_semantic_sr_preview',preview):
            run(args)

    def test_real_two_sgd_updates_equal_stop_resume_all_named_adam_rng_and_new_sampler(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);initial,selected,parent,report=self.migrated(root)
            parent_payload=torch.load(parent,map_location='cpu',weights_only=False)
            migrated=torch.load(initial,map_location='cpu',weights_only=False)
            self.assertEqual(parent_payload['semantic_updates'],1)
            self.assertEqual(migrated['semantic_updates'],0)
            self.assertEqual(migrated['parent_sampler'],parent_payload['sampler'])
            self.assertTrue(report['old_Adam_and_RNG_preserved'])
            hypotheses=root/'hypotheses'
            self.invoke(self.args(root/'continuous',initial,selected,2,hypotheses))
            self.invoke(self.args(root/'resumed',initial,selected,1,hypotheses))
            self.invoke(self.args(root/'resumed',initial,selected,2,hypotheses))
            a=torch.load(root/'continuous/checkpoints/sr_last.pt',map_location='cpu',weights_only=False)
            b=torch.load(root/'resumed/checkpoints/sr_last.pt',map_location='cpu',weights_only=False)
            self.assertEqual(a['semantic_updates'],2);self.assertEqual(a['decoder_updates'],2)
            self.assertEqual(b['decoder_updates'],2)
            self.assertEqual(a['parent_training_steps'],parent_payload['training_steps'])
            self.assertEqual(a['training_steps'],parent_payload['training_steps']+2)
            self.assertEqual(a['sampler']['draws'],8)
            self.assertEqual(a['parent_sampler'],parent_payload['sampler'])
            for key in ('optimizer_parameter_names','sampler','rng_numpy','rng_python'):
                self.assertEqual(a[key],b[key])
            torch.testing.assert_close(a['rng_torch'],b['rng_torch'],rtol=0,atol=0)
            self.assertEqual(a['optimizer']['param_groups'],b['optimizer']['param_groups'])
            for name,value in a['model'].items():
                torch.testing.assert_close(value,b['model'][name],rtol=0,atol=0)
            for index,row in a['optimizer']['state'].items():
                for name,value in row.items():
                    torch.testing.assert_close(value,b['optimizer']['state'][index][name],rtol=0,atol=0)
            rows=[json.loads(line)for line in (root/'resumed/metrics.jsonl').read_text().splitlines()]
            self.assertEqual([row['step']for row in rows],[1,2])
            self.assertTrue(all(row['objective_protocol']==row['loss_protocol']==PROTOCOL for row in rows))
            self.assertTrue(all(row['actual_npr_fraction']==.75 for row in rows))
            self.assertTrue(all(row['mismatched_predicted_hypotheses']==1 for row in rows))
            self.assertTrue(all(row['actual_hypotheses_available']==4 for row in rows))
            self.assertTrue(all(row['global_presence_positive_count']==row['global_presence_negative_count']==2 for row in rows))
            self.assertTrue(all(row['future_Source_states_used_as_input']is False for row in rows))
            self.assertTrue(any(not torch.equal(value,a['model'][name])for name,value in migrated['model'].items()
                if name.startswith('enemy_decoder_')))
            status=json.loads((root/'resumed/status.json').read_text())
            self.assertEqual(status['semantic_sr_training']['protocol'],'enemy_decoder_super_resolution_v1')
            self.assertEqual(status['loss_protocol'],PROTOCOL)
            self.assertFalse(status['gameplay_qualified']);self.assertFalse(status['automatic_deployment'])

    def test_wrong_predicted_proposal_mixture_never_uses_source_labels_and_preserves_invalid_rows(self):
        hypotheses=torch.zeros(8,21);hypotheses[:6,-1]=1
        hypotheses[:6,8]=torch.arange(6)/16+.25
        before=hypotheses.clone()
        mixed,count=mix_enemy_hypotheses(hypotheses,np.random.default_rng(173))
        self.assertEqual(count,2)
        self.assertEqual(int((mixed!=before).any(1).sum()),2)
        self.assertTrue(torch.equal(hypotheses,before))
        self.assertTrue(torch.equal(mixed[6:],before[6:]))

    def test_stable_three_sgd_updates_preserve_all102_parent_weights_adam_and_resume_exactly(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory)
            # Keep sixteen parent residual blocks (102 named parameters), using
            # four channels so the complete real optimizer test stays on CPU.
            def full_parent_topology(config):
                return build_upscaler({**config,'blocks':16})
            with patch.object(semantic_training,'build_upscaler',side_effect=full_parent_topology):
                initial,selected,parent,report=self.migrated(root,stable=True)
            source=torch.load(parent,map_location='cpu',weights_only=False)
            migrated=torch.load(initial,map_location='cpu',weights_only=False)
            old_names=[name for group in source['optimizer_parameter_names']for name in group]
            self.assertEqual(len(old_names),102)
            self.assertEqual(len(source['optimizer']['state']),102)
            self.assertTrue(report['stable_RGB_parent_frozen'])
            self.assertEqual(report['frozen_parent_parameter_names'],old_names)
            self.assertEqual(report['added_parameter_count'],290854)
            hypotheses=root/'hypotheses'
            self.invoke(self.args(root/'stable_continuous',initial,selected,3,hypotheses))
            self.invoke(self.args(root/'stable_resumed',initial,selected,1,hypotheses))
            self.invoke(self.args(root/'stable_resumed',initial,selected,3,hypotheses))
            a=torch.load(root/'stable_continuous/checkpoints/sr_last.pt',map_location='cpu',weights_only=False)
            b=torch.load(root/'stable_resumed/checkpoints/sr_last.pt',map_location='cpu',weights_only=False)
            for name,value in source['model'].items():
                torch.testing.assert_close(value,a['model'][name],rtol=0,atol=0)
            self.assertEqual(a['optimizer']['param_groups'][:len(source['optimizer']['param_groups'])],
                source['optimizer']['param_groups'])
            for index,row in source['optimizer']['state'].items():
                for name,value in row.items():
                    torch.testing.assert_close(value,a['optimizer']['state'][index][name],rtol=0,atol=0)
            changed=[name for name,value in migrated['model'].items()if not torch.equal(value,a['model'][name])]
            self.assertTrue(changed)
            self.assertTrue(all(name.startswith('enemy_decoder_')for name in changed))
            self.assertIn('enemy_decoder_global_presence.weight',changed)
            for name,value in a['model'].items():
                torch.testing.assert_close(value,b['model'][name],rtol=0,atol=0)
            self.assertEqual(a['optimizer']['param_groups'],b['optimizer']['param_groups'])
            for index,row in a['optimizer']['state'].items():
                for name,value in row.items():
                    torch.testing.assert_close(value,b['optimizer']['state'][index][name],rtol=0,atol=0)
            for key in ('optimizer_parameter_names','sampler','rng_numpy','rng_python'):
                self.assertEqual(a[key],b[key])
            torch.testing.assert_close(a['rng_torch'],b['rng_torch'],rtol=0,atol=0)
            self.assertEqual((a['semantic_updates'],a['decoder_updates'],a['sampler']['draws']),(3,3,12))
            self.assertEqual(a['training_steps'],source['training_steps']+3)
            self.assertEqual(a['parent_sampler'],source['sampler'])
            status=json.loads((root/'stable_resumed/status.json').read_text())
            self.assertEqual(status['model_parameters']['trainable'],290854)
            self.assertTrue(status['enemy_decoder_training']['parent_rgb_frozen'])
            original=build_upscaler(source['config']).eval()
            original.load_state_dict(source['model'])
            stable=build_upscaler(a['config']).eval();stable.load_state_dict(a['model'])
            inputs=torch.linspace(.1,.7,3*32*32).reshape(1,3,32,32)
            with torch.no_grad():
                torch.testing.assert_close(stable(inputs),original(inputs),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
