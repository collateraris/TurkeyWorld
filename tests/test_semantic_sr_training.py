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

from turkeyworld.models import build_upscaler
from turkeyworld.sr_semantic_migration import prepare_semantic_sr_continuation
from turkeyworld.train_semantic_sr import run, mix_semantic_inputs, _choose_best


class TinyPairs:
    def __init__(self,root,split,*,seed=173,device='cpu'):
        self.identity={'split':split,'cache':'fixed-real-NPR-schema-fixture'}
        self.rng=np.random.default_rng(seed);self.draws=0;self.calls=[]
        self.low=torch.linspace(.1,.7,8*3*32*32).reshape(8,3,32,32)
        self.high=torch.linspace(.2,.8,8*3*128*128).reshape(8,3,128,128)
        self.mask=torch.zeros(8,1,32,32);self.mask[:6,:,10:12,15:17]=1
    def fixed_panel(self,count=None,seed=173):return list(range(8))
    def panel_fingerprint(self,panel):return 'fixed-panel8'
    def sampler_state(self):return {'draws':self.draws,'rng':deepcopy(self.rng.bit_generator.state)}
    def restore_sampler(self,state):self.draws=state['draws'];self.rng.bit_generator.state=state['rng']
    def batch(self,size,*,indices=None,rng=None):
        if indices is None:
            indices=(rng or self.rng).integers(8,size=size).tolist();self.draws+=size
        self.calls.append(list(indices));ids=torch.as_tensor(indices)
        return {'lr_npr':self.low[ids], 'lr_source':self.low[ids]*.8,
            'codec_lr':self.low[ids]*.6, 'hr_source':self.high[ids],
            'hr_background':self.high[ids]*.8,'mask_lr':self.mask[ids],
            'mask_hr':self.mask[ids].repeat_interleave(4,-1).repeat_interleave(4,-2),
            'cohort':['near','mid','far','guard'][:size], 'row_index':ids}


def evaluate(model,data,panel,*,step,batch_size=8,baselines=None,baseline_report=None):
    # Inference has no Source mask/state arguments, like production evaluation.
    before=data.sampler_state()
    with torch.no_grad():model(data.batch(8,indices=panel)['lr_npr'])
    assert before==data.sampler_state()
    row={'summary':{'mae_mean':.1,'ssim_mean':.8}}
    return {'step':step,'data_identity_sha256':'fixed-identity','panel_fingerprint':'fixed-panel8',
        'selection_score':1-step*.01,'relative_progress':{'passed':False},
        'training_goal':{'passed':False},'domains':{'npr':{'models':{'current':row}}}},{}


def preview(preview,path,*,every):return {'path':str(path),'step':0,'every':every}


class SemanticTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def initial(self,root):
        model=build_upscaler({'architecture':'edsr','channels':4,'blocks':1,'scale':4})
        opt=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=.01)
        (model(torch.full((1,3,32,32),.3))-.7).square().mean().backward();opt.step()
        payload=dict(stage='sr',config=model.config,model=model.state_dict(),optimizer=opt.state_dict(),
            training_steps=40100,completed_updates=2000,rng_numpy=np.random.default_rng(173).bit_generator.state,
            rng_torch=torch.get_rng_state(),rng_python=random.getstate(),rng_cuda=[],sampler={'draws':16000})
        full,selected=root/'full.pt',root/'selected.pt'
        torch.save(payload,full)
        torch.save({k:payload[k] for k in ('stage','config','model','training_steps')},selected)
        initial=root/'initial.pt';prepare_semantic_sr_continuation(full,initial,selected_checkpoint=selected)
        return initial,selected

    def args(self,root,initial,selected,updates):
        return argparse.Namespace(run_dir=root,data_root=root/'data',initial_checkpoint=initial,
            baseline38100=selected,baseline40100=selected,updates=updates,batch=4,
            validation_every=1,prediction_every=15,validation_frames=0,seed=173,device='cpu')

    def invoke(self,args):
        with patch('turkeyworld.sr_neural_pairs_data.SemanticSRData',TinyPairs), \
             patch('turkeyworld.semantic_sr_validation.evaluate_semantic_sr',evaluate), \
             patch('turkeyworld.semantic_sr_validation.save_semantic_sr_preview',preview):run(args)

    def test_three_quarter_actual_npr_and_genuine_guard_inputs(self):
        batch={'lr_npr':torch.ones(8,3,32,32)*.7,'lr_source':torch.ones(8,3,32,32)*.3,
               'codec_lr':torch.ones(8,3,32,32)*.1}
        inputs,kinds=mix_semantic_inputs(batch,np.random.default_rng(173))
        self.assertEqual(kinds.count('npr'),6);self.assertEqual(kinds.count('clean'),1);self.assertEqual(kinds.count('codec'),1)
        for index,kind in enumerate(kinds):
            self.assertAlmostEqual(float(inputs[index,0,0,0]),{'npr':.7,'clean':.3,'codec':.1}[kind],places=6)

    def test_actual_sgd_exact_resume_model_adam_sampler_rng(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);initial,selected=self.initial(root)
            self.invoke(self.args(root/'continuous',initial,selected,2))
            self.invoke(self.args(root/'resumed',initial,selected,1))
            self.invoke(self.args(root/'resumed',initial,selected,2))
            a=torch.load(root/'continuous/checkpoints/sr_last.pt',weights_only=False)
            b=torch.load(root/'resumed/checkpoints/sr_last.pt',weights_only=False)
            self.assertEqual(a['semantic_updates'],2);self.assertEqual(b['training_steps'],40102)
            self.assertEqual(a['sampler'],b['sampler']);self.assertEqual(a['rng_numpy'],b['rng_numpy'])
            torch.testing.assert_close(a['rng_torch'],b['rng_torch'],rtol=0,atol=0)
            self.assertEqual(a['rng_python'],b['rng_python'])
            self.assertEqual(a['optimizer']['param_groups'],b['optimizer']['param_groups'])
            for name,value in a['model'].items():torch.testing.assert_close(value,b['model'][name],rtol=0,atol=0)
            for index,row in a['optimizer']['state'].items():
                for name,value in row.items():torch.testing.assert_close(value,b['optimizer']['state'][index][name],rtol=0,atol=0)
            rows=[json.loads(line) for line in (root/'resumed/metrics.jsonl').read_text().splitlines()]
            self.assertEqual([row['step'] for row in rows],[1,2])
            self.assertTrue(all(row['actual_npr_fraction']==.75 for row in rows))

    def test_best_prefers_guarded_candidate_and_rejects_nonfinite(self):
        old={'step':1,'score':.1,'eligible':False}
        selected=_choose_best(old,{'selection_score':.2,'relative_progress':{'passed':True}},2)
        self.assertEqual(selected['step'],2)
        with self.assertRaises(ValueError):
            _choose_best(None,{'selection_score':float('nan'),'relative_progress':{'passed':True}},1)


if __name__=='__main__':unittest.main()
