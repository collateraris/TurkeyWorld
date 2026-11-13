import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from turkeyworld.models import build_upscaler
from turkeyworld.sr_enemy_patchgan import EnemyPatchDiscriminator
from turkeyworld.sr_patchgan_protocol import prepare_patchgan_continuation,scheduled_gan_weight,SETTINGS
from turkeyworld.train_sr_patchgan import patchgan_ab_step,run
from test_decoder_restoration_training import RestorationTrainingTests,advance,report
from turkeyworld.decoder_restoration_protocol import bind_restoration_baseline,consider_restoration_gate
from test_enemy_decoder_training import GuidedPairs


def equal(test,a,b):
    if torch.is_tensor(a):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        test.assertEqual(a.keys(),b.keys())
        for k in a:equal(test,a[k],b[k])
    elif isinstance(a,(tuple,list)):
        test.assertEqual(type(a),type(b));test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):equal(test,x,y)
    else:test.assertEqual(a,b)


class MatchedPairs(GuidedPairs):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.split=args[1]
    def batch(self,*args,**kwargs):
        result=super().batch(*args,**kwargs)
        result['state']=self.hypothesis[result['row_index'],:20].clone()
        return result


def evaluate(gan,control,data,panel,*,step,batch_size=8):
    # Actual forwards, but do not rerun the already independently tested evaluator.
    b=data.batch(8,indices=panel)
    with torch.no_grad():
        for model in (gan,control):model.forward_with_aux(b['lr_npr'],b['hypothesis'])
    return dict(kind='enemy_roi_patchgan_ab_validation_v1',step=step,
        panel_fingerprint=data.panel_fingerprint(panel),data_identity_sha256='data',
        temporal={'panel_fingerprint':'temporal'},arms={}),dict(metadata={})


class PatchGANTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def initial(self,root):
        initial,_,_=RestorationTrainingTests().initial(root)
        parent=torch.load(initial,weights_only=False)
        schedule=parent['restoration_schedule'];base=report(0)
        bind_restoration_baseline(schedule,base,sha256='a'*64);advance(schedule,200)
        consider_restoration_gate(schedule,report(200,fg=.1,edge=.1),base,step=200);advance(schedule,600)
        parent['config']['restoration_blend_scale']=1.
        parent['semantic_updates']=600;parent['training_steps']=parent['parent_training_steps']+600
        source=root/'restoration_parent.pt';torch.save(parent,source)
        target=root/'gan_initial.pt';before=source.read_bytes();rng=torch.get_rng_state().clone()
        receipt=prepare_patchgan_continuation(source,target)
        self.assertEqual(before,source.read_bytes());equal(self,rng,torch.get_rng_state())
        saved=torch.load(target,weights_only=False)
        for key in ('model','optimizer','sampler','rng_numpy','rng_torch','rng_python','rng_cuda'):
            equal(self,parent[key],saved[key])
        equal(self,saved['model'],saved['patchgan_state']['control_model'])
        equal(self,saved['optimizer'],saved['patchgan_state']['control_optimizer'])
        self.assertEqual(receipt['generator_new_parameters'],0)
        return target

    def test_schedule_is_small_gradual_and_counts_completed_updates(self):
        self.assertEqual(scheduled_gan_weight(100),0.)
        self.assertAlmostEqual(scheduled_gan_weight(101),.01/400)
        self.assertEqual(scheduled_gan_weight(300),.005)
        self.assertEqual(scheduled_gan_weight(500),.01)
        self.assertEqual(scheduled_gan_weight(900),.01)
        for value in (-1,True,1.5):
            with self.assertRaises(ValueError):scheduled_gan_weight(value)

    def twin_fixture(self,root):
        model=build_upscaler(dict(architecture='enemy_decoder_edsr',channels=4,blocks=1,
            semantic_channels=4,semantic_blocks=1,enemy_decoder_channels=4,
            enemy_decoder_stable=True,enemy_decoder_visual_first=True,
            enemy_decoder_restoration=True,restoration_blend_scale=1.))
        control=deepcopy(model);d=EnemyPatchDiscriminator()
        a=torch.optim.AdamW(model.parameters(),lr=5e-5);b=torch.optim.AdamW(control.parameters(),lr=5e-5)
        opt_d=torch.optim.Adam(d.parameters(),lr=.0002,betas=(0.,.99))
        data=MatchedPairs(root,'train',hypotheses_root=root/'hyp')
        batch=data.batch(8,indices=list(range(8)))
        return model,control,d,a,b,opt_d,batch

    def test_zero_weight_twins_match_and_enabled_GAN_is_gradient_capped(self):
        with tempfile.TemporaryDirectory()as folder:
            g,c,d,a,b,od,batch=self.twin_fixture(Path(folder))
            frozen={n:p.detach().clone()for n,p in g.named_parameters()if not p.requires_grad}
            row=patchgan_ab_step(g,c,d,a,b,od,batch,batch['lr_npr'],scheduled_weight=0.)
            equal(self,g.state_dict(),c.state_dict());equal(self,a.state_dict(),b.state_dict())
            self.assertTrue(row['discriminator_updated'])
            row=patchgan_ab_step(g,c,d,a,b,od,batch,batch['lr_npr'],scheduled_weight=.01)
            self.assertGreater(row['gan_weight_effective'],0.)
            self.assertLessEqual(row['gan_readout_weighted_gradient_ratio'],SETTINGS['gradient_ratio_limit']+1e-7)
            self.assertTrue(any(not torch.equal(p,c.state_dict()[n])for n,p in g.state_dict().items()))
            for n,p in frozen.items():equal(self,p,g.state_dict()[n])

    def test_actual_enabled_GAN_resume_preserves_both_generators_D_Adam_and_SN(self):
        with tempfile.TemporaryDirectory()as folder:
            fixture=self.twin_fixture(Path(folder));g,c,d,a,b,od,batch=fixture
            patchgan_ab_step(g,c,d,a,b,od,batch,batch['lr_npr'],scheduled_weight=.01)
            snap=deepcopy([g.state_dict(),c.state_dict(),d.state_dict(),a.state_dict(),b.state_dict(),od.state_dict()])
            for _ in range(2):patchgan_ab_step(g,c,d,a,b,od,batch,batch['lr_npr'],scheduled_weight=.01)
            expected=deepcopy([g.state_dict(),c.state_dict(),d.state_dict(),a.state_dict(),b.state_dict(),od.state_dict()])
            g.load_state_dict(snap[0]);c.load_state_dict(snap[1]);d.load_state_dict(snap[2])
            a.load_state_dict(snap[3]);b.load_state_dict(snap[4]);od.load_state_dict(snap[5])
            for _ in range(2):patchgan_ab_step(g,c,d,a,b,od,batch,batch['lr_npr'],scheduled_weight=.01)
            equal(self,expected,[g.state_dict(),c.state_dict(),d.state_dict(),a.state_dict(),b.state_dict(),od.state_dict()])

    def test_complete_worker_resume_preserves_shared_sampler_RNG_ledger_and_all_Adam(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);initial=self.initial(root)
            def invoke(path,updates):
                args=argparse.Namespace(run_dir=path,initial_checkpoint=initial,data_root=root/'data',
                    hypotheses_root=root/'hyp',seed=173,device='cpu',updates=updates)
                with patch('turkeyworld.train_sr_patchgan.EnemyDecoderData',MatchedPairs), \
                     patch('turkeyworld.sr_patchgan_validation.evaluate_patchgan_pair',evaluate), \
                     patch('turkeyworld.sr_patchgan_validation.save_patchgan_preview',
                           side_effect=lambda preview,path,every:dict(path=str(path),every=every)):
                    run(args)
            invoke(root/'continuous',2);invoke(root/'resumed',1);invoke(root/'resumed',2)
            a=torch.load(root/'continuous/checkpoints/sr_last.pt',weights_only=False)
            b=torch.load(root/'resumed/checkpoints/sr_last.pt',weights_only=False)
            for key in ('model','optimizer','patchgan_state','sampler','rng_numpy','rng_torch','rng_python','rng_cuda'):
                equal(self,a[key],b[key])
            self.assertEqual(a['patchgan_updates'],2)
            self.assertEqual(a['semantic_updates'],602)
            self.assertEqual(a['sampler']['draws'],44016)
            self.assertEqual(a['config']['restoration_blend_scale'],1.)
            rows=[json.loads(line)for line in(root/'resumed/metrics.jsonl').read_text().splitlines()]
            self.assertEqual([r['step']for r in rows],[1,2])
            self.assertTrue(all(r['gan_steps']==r['no_gan_steps']for r in rows))
            self.assertEqual(rows[-1]['input_domain_counts'],{'npr':6,'clean':1,'codec':1})


if __name__=='__main__':unittest.main()
