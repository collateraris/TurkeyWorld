"""Fair SR inputs, genuine temporal adjacency, motion-aware errors and no mutation."""
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from turkeyworld.sr_patchgan_validation import (build_temporal_panel,evaluate_patchgan_pair,
    preview_patchgan_pair,save_patchgan_preview,_extras,_arm_summary)


class AdjacentPairs:
    split='val'
    identity={'fingerprint':'actual-adjacent-Source-NPR-fixture'}
    def __init__(self):
        self.rng=np.random.default_rng(173);self.draws=0;self.calls=[]
        self.low=torch.full((8,3,32,32),.2)
        self.mask=torch.zeros(8,1,32,32)
        for i in range(8):
            self.low[i,:,0,0]=.6 if i%2 else .2
            if i<6:
                self.mask[i,:,10:12,3+i*2:5+i*2]=1
                self.low[i,0:1]+=self.mask[i]*.6
        self.hr=F.interpolate(self.low,scale_factor=4,mode='nearest')
        self.hr_mask=F.interpolate(self.mask,scale_factor=4,mode='nearest')
        self.bg=self.hr.clone();self.bg[:,0:1]-=self.hr_mask*.6
        self.state=torch.zeros(8,20);self.state[:,4]=1;self.state[:,12:14]=1
        self.state[:,8]=torch.tensor([1.,3.,7.,1.5,4.,8.,3.,3.])/16
        self.state[6:,12:14]=0
        self.rows=[]
        for i in range(8):
            group='a'if i<3 else'b'if i<6 else'c'
            self.rows.append(dict(split='val',dynamic=i!=7,source_sha256=group*64,
                source_path='/heldout/'+group+'.npz',source_seed=101+ord(group),
                frame=(10+i if i<3 else i+1 if i<6 else i+20),
                cohort=('near','mid','far')[i%3]if i<6 else'guard'))
    def panel_fingerprint(self,panel):return 'fixed_'+','.join(map(str,panel))
    def sampler_state(self):return deepcopy(dict(draws=self.draws,rng=self.rng.bit_generator.state))
    def restore_sampler(self,value):self.draws=value['draws'];self.rng.bit_generator.state=value['rng']
    def batch(self,size,*,indices):
        self.draws+=size;self.rng.random();np.random.random();random.random();torch.rand(1)
        self.calls.append(list(indices));ids=torch.tensor(indices)
        return dict(lr_npr=self.low[ids],lr_source=self.low[ids]*.99,codec_lr=self.low[ids]*.95,
            hr_source=self.hr[ids],hr_background=self.bg[ids],mask_lr=self.mask[ids],
            mask_hr=self.hr_mask[ids],state=self.state[ids],hypothesis=torch.zeros(size,21))


class FixtureSR(nn.Module):
    def __init__(self,flicker=0.):
        super().__init__();self.offset=nn.Parameter(torch.tensor(float(flicker)))
        self.config={'architecture':'enemy_decoder_edsr','enemy_decoder_restoration':True,'restoration_blend_scale':1.}
        self.child=nn.Identity();self.calls=[]
    def forward_with_aux(self,image,hypothesis=None):
        self.calls.append((image.detach().clone(),None if hypothesis is None else hypothesis.detach().clone()))
        torch.rand(1)  # The evaluator must preserve RNG even for future stochastic layers.
        target=F.interpolate(image,scale_factor=4,mode='nearest')
        alpha=F.interpolate((image[:,:1]>.7).float(),scale_factor=4,mode='nearest')
        flag=(image[:,0,0,0]>.4).float()[:,None,None,None]
        rgb=(target+self.offset*flag).clamp(0,1)
        visible=alpha.flatten(1).any(-1)[:,None]
        rgba=torch.cat((rgb,alpha),1)
        return dict(image=rgb,visual_route=visible,guided_route=torch.zeros_like(visible),
            visual_raw_warped_rgba=rgba,guided_raw_warped_rgba=rgba,
            visual_composite_alpha_hr=alpha,guided_route_weight=torch.zeros(len(image),1),
            enemy_composite_alpha_hr=torch.zeros_like(alpha),
            mask_logits_lr=(image[:,:1]>.7).float()*12-6,
            presence_logits=visible.float()*12-6)


def reconstruction(output,batch):
    return (output['image']-batch['hr_source'].detach()).abs().mean(),{}


class PatchGANValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_manifest_temporal_pairs_are_real_adjacent_same_episode_and_never_train_or_static(self):
        data=AdjacentPairs();panel=list(range(8));temporal=build_temporal_panel(data,panel)
        self.assertEqual(temporal['pair_count'],4)
        self.assertEqual([(p['previous_index'],p['current_index'])for p in temporal['pairs']],[(0,1),(1,2),(3,4),(4,5)])
        self.assertEqual(temporal['episode_count'],2)
        self.assertEqual(temporal['panel_fingerprint'],build_temporal_panel(data,panel)['panel_fingerprint'])
        self.assertNotIn(7,[p['current_index']for p in temporal['pairs']])
        data.rows[0]['split']='train'
        with self.assertRaisesRegex(ValueError,'TRAIN'):build_temporal_panel(data,panel)
        data.rows[0]['split']='val';data.rows[1]['source_seed']+=1
        self.assertEqual(build_temporal_panel(data,panel)['pair_count'],2)

    def test_source_motion_is_not_flicker_and_same_inputs_and_all_state_are_preserved(self):
        data=AdjacentPairs();gan,control=FixtureSR(.05),FixtureSR(0.)
        gan.train();gan.child.eval();control.train()
        modes=[[m.training for m in net.modules()]for net in(gan,control)]
        sampler=data.sampler_state();rng=torch.get_rng_state().clone();py=random.getstate();nprng=np.random.get_state()
        weights=[deepcopy(net.state_dict())for net in(gan,control)]
        report,preview=evaluate_patchgan_pair(gan,control,data,list(range(8)),step=100,
            batch_size=3,reconstruction_objective=reconstruction)
        self.assertEqual(report['kind'],'enemy_roi_patchgan_ab_validation_v1')
        self.assertEqual(report['temporal']['pair_count'],4)
        temporal=report['temporal']['arms']
        self.assertEqual(temporal['no_gan']['delta_mae'],0.)
        self.assertEqual(temporal['no_gan']['foreground_delta_mae'],0.)
        self.assertGreater(temporal['gan']['delta_mae'],.049)
        self.assertGreater(temporal['gan']['foreground_delta_mae'],.049)
        self.assertEqual(report['arms']['no_gan']['summary']['mae_mean'],0.)
        self.assertAlmostEqual(report['arms']['gan']['summary']['reconstruction_loss'],.025,places=6)
        self.assertEqual(report['arms']['gan']['summary']['reconstruction_frames'],8)
        self.assertEqual(report['arms']['no_gan']['summary']['silhouette_iou_mean'],1.)
        self.assertEqual(report['arms']['no_gan']['summary']['silhouette_frames'],6)
        self.assertEqual(report['arms']['no_gan']['summary']['empty_frames'],2)
        self.assertEqual(set(report['domains']),{'npr','clean','codec'})
        self.assertEqual(set(preview['cases']),{'near','mid','far','empty'})
        self.assertEqual(sampler,data.sampler_state());self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(py,random.getstate())
        self.assertTrue(all(np.array_equal(a,b)for a,b in zip(nprng,np.random.get_state())))
        self.assertEqual(modes,[[m.training for m in net.modules()]for net in(gan,control)])
        for index,net in enumerate((gan,control)):
            self.assertTrue(all(p.grad is None for p in net.parameters()))
            for name,value in weights[index].items():torch.testing.assert_close(value,net.state_dict()[name],rtol=0,atol=0)
        self.assertEqual(len(gan.calls),len(control.calls))
        for (a,hypa),(b,hypb)in zip(gan.calls,control.calls):
            torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(hypa,hypb,rtol=0,atol=0)
        self.assertFalse(report['source_input_allowed_for_sr']);self.assertFalse(report['gameplay_qualified'])
        json.dumps(report,allow_nan=False)

    def test_empty_layer_hallucination_is_measured_separately_from_correct_visible_silhouette(self):
        data=AdjacentPairs();batch=data.batch(8,indices=list(range(8)));model=FixtureSR()
        output=model.forward_with_aux(batch['lr_npr'],batch['hypothesis'])
        output['visual_route'][6:]=True
        output['visual_composite_alpha_hr'][6:,:,30:35,30:35]=1.
        rows=_extras(output,batch)
        for row in rows:
            row.update(psnr=30.,ssim=.9,mae=.01,foreground_mae=.01,background_mae=.01,
                foreground_edge_mae=.01,silhouette_edge_mae=.01,roi_crop_ssim=.9,roi_crop_mae=.01)
        summary=_arm_summary(rows,[True]*8)
        self.assertEqual(summary['empty_false_npc_rate'],1.)
        self.assertEqual(summary['empty_false_npc_count'],2)
        self.assertEqual(summary['empty_layer_false_positive_rate'],1.)
        self.assertGreater(summary['empty_alpha_leakage_mean'],0.)
        self.assertEqual(summary['silhouette_iou_mean'],1.)

    def test_no_temporal_support_is_null_and_shared_arms_or_tampered_panel_fail_closed(self):
        data=AdjacentPairs();gan,control=FixtureSR(),FixtureSR()
        for row in data.rows:row['dynamic']=False
        report,_=evaluate_patchgan_pair(gan,control,data,list(range(8)),step=0,
            reconstruction_objective=reconstruction,include_guards=False)
        self.assertFalse(report['temporal']['available'])
        self.assertEqual(report['temporal']['arms']['gan']['pair_count'],0)
        self.assertIsNone(report['temporal']['arms']['gan']['delta_mae'])
        with self.assertRaisesRegex(ValueError,'separate models'):
            evaluate_patchgan_pair(gan,gan,data,list(range(8)),step=1)
        bad=build_temporal_panel(data,list(range(8)));bad['panel_fingerprint']='changed'
        with self.assertRaisesRegex(ValueError,'temporal panel'):
            evaluate_patchgan_pair(gan,control,data,list(range(8)),step=1,temporal_panel=bad)
        control.config['restoration_blend_scale']=.5
        with self.assertRaisesRegex(ValueError,'same inference config'):
            evaluate_patchgan_pair(gan,control,data,list(range(8)),step=1)

    def test_four_case_light_preview_png_preserves_sampling_and_never_runs_full_validation(self):
        data=AdjacentPairs();gan,control=FixtureSR(.01),FixtureSR()
        initial,_=evaluate_patchgan_pair(gan,control,data,list(range(8)),step=0,
            reconstruction_objective=reconstruction,include_guards=False)
        gan.calls.clear();control.calls.clear();sampler=data.sampler_state();rng=torch.get_rng_state().clone()
        preview=preview_patchgan_pair(gan,control,data,list(range(8)),step=15,baseline_report=initial)
        self.assertEqual(sum(len(x[0])for x in gan.calls),4)
        self.assertEqual(sum(len(x[0])for x in control.calls),4)
        self.assertEqual(sampler,data.sampler_state());self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(preview['metadata']['preview_only'])
        self.assertEqual(preview['metadata']['reference_validation_step'],0)
        with tempfile.TemporaryDirectory()as temporary:
            metadata=save_patchgan_preview(preview,Path(temporary)/'preview.png')
            with Image.open(metadata['path'])as picture:self.assertEqual(picture.size,(1024,1120))
            self.assertEqual(metadata['row_count'],4);self.assertEqual(metadata['every'],15)

    def test_actual_restoration_generators_use_the_actual_content_loss_without_GAN_or_Source_inputs(self):
        from test_decoder_restoration_validation import DecoderRestorationValidationTests
        from test_enemy_decoder_validation import GuidedPairs
        model=DecoderRestorationValidationTests().model();control=deepcopy(model)
        data=GuidedPairs()
        report,_=evaluate_patchgan_pair(model,control,data,list(range(8)),step=0,batch_size=3)
        self.assertEqual(report['arms']['gan']['summary']['reconstruction_loss'],
                         report['arms']['no_gan']['summary']['reconstruction_loss'])
        self.assertGreater(report['arms']['gan']['summary']['reconstruction_loss'],0.)
        self.assertEqual(report['arms']['gan']['summary']['reconstruction_frames'],8)
        self.assertTrue(all(p.grad is None for net in(model,control)for p in net.parameters()))
        self.assertIsNone(data.hr.grad);self.assertIsNone(data.hr_mask.grad);self.assertIsNone(data.npr.grad)


if __name__=='__main__':unittest.main()
