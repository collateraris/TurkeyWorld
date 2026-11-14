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

from turkeyworld.background_detail_sr_validation import(
    PROTOCOL,evaluate_background_detail_sr,preview_background_detail_sr,
    save_background_detail_sr_preview,background_detail_candidate)
from turkeyworld.neural_enemy_restoration_validation import build_detail_animation_panel
from test_sr_patchgan_validation import AdjacentPairs


class BackgroundDetailFixture(nn.Module):
    def __init__(self):
        super().__init__();self.preferred_base=nn.Parameter(torch.tensor(0.),requires_grad=False)
        self.detail_residual=nn.Linear(1,1,bias=False);nn.init.zeros_(self.detail_residual.weight)
        self.child=nn.Identity();self.config={'architecture':'background_detail_sr','fixture':True};self.calls=[]
    def forward_with_aux(self,low):
        self.calls.append(low.detach().clone());torch.rand(1)
        base=(F.interpolate(low,scale_factor=4,mode='nearest')+self.preferred_base).clamp(0,1)
        residual=self.detail_residual.weight.reshape(1,1,1,1).expand_as(base)
        return dict(image=(base+residual).clamp(0,1),background=base,residual=residual)


def data_fixture():
    data=AdjacentPairs();data.state[0,14]=1;data.state[1,15]=.5
    data.state[2:4,12:14]=0;data.state[2,16]=.2;data.state[3,16]=1
    return data


def objective(output,batch):
    value=(output['image']-batch['hr_source'].detach()).abs().mean()
    return value,dict(objective_protocol=PROTOCOL,pixel=float(value),pixel_weighted=float(value))


def evaluate(model,data,panel,**kwargs):
    return evaluate_background_detail_sr(model,data,panel,objective_fn=objective,
        objective_weights={'pixel':1.},**kwargs)


class BackgroundDetailValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_same_forward_preferred_base_zeroinit_panels_metrics_and_state_are_preserved(self):
        model=BackgroundDetailFixture();model.train();model.child.eval();data=data_fixture()
        detail=build_detail_animation_panel(data);sampler=data.sampler_state();modes=[m.training for m in model.modules()]
        flags=[p.requires_grad for p in model.parameters()];weights=deepcopy(model.state_dict())
        random_before=random.getstate();torch_before=torch.get_rng_state().clone();numpy_before=np.random.get_state()
        report,preview=evaluate(model,data,list(range(8)),step=0,batch_size=3,detail_panel=detail)
        self.assertEqual(len(model.calls),5)  # primary8 and extra4, not a second base network
        self.assertEqual(set(report['models']),{'current','preferred_frozen_background'})
        for key in('ssim_mean','mae_mean','foreground_mae','background_mae','roi_crop_ssim_mean'):
            self.assertEqual(report['models']['current']['summary'][key],report['models']['preferred_frozen_background']['summary'][key])
        current=report['models']['current']['summary'];self.assertEqual(current['ssim_mean'],1.)
        self.assertEqual(current['low_frequency_change_mae'],0.);self.assertEqual(current['empty_paint_top64_mae'],0.)
        self.assertIsNone(current['alpha_iou']);self.assertEqual(current['alpha_frames'],0)
        self.assertEqual(report['detail_validation']['fingerprint'],detail['fingerprint'])
        self.assertEqual(report['detail_validation']['models']['current']['summary']['frames'],4)
        self.assertEqual(report['temporal']['pair_count'],4)
        self.assertTrue(report['preferred_base_preservation']['eligible'])
        self.assertFalse(report['gameplay_qualified']);self.assertFalse(report['automatic_deployment'])
        self.assertEqual(report['validation_objective']['loss'],0.)
        self.assertEqual(report['background_ablation']['additional_model_forwards'],0)
        self.assertEqual(preview['metadata']['row_count'],4);self.assertEqual(set(preview['cases']),{'near','mid','far','empty'})
        self.assertEqual(sampler,data.sampler_state());self.assertEqual(modes,[m.training for m in model.modules()])
        self.assertEqual(flags,[p.requires_grad for p in model.parameters()]);self.assertEqual(random_before,random.getstate())
        self.assertTrue(torch.equal(torch_before,torch.get_rng_state()))
        self.assertTrue(all(np.array_equal(a,b)for a,b in zip(numpy_before,np.random.get_state())))
        for key,value in model.state_dict().items():torch.testing.assert_close(weights[key],value,rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()));json.dumps(report,allow_nan=False)

    def test_selector_rejects_lost_base_details_empty_paint_and_missing_support_without_fake_quality(self):
        model=BackgroundDetailFixture();data=data_fixture();detail=build_detail_animation_panel(data)
        initial,_=evaluate(model,data,list(range(8)),step=0,detail_panel=detail)
        with torch.no_grad():model.detail_residual.weight.fill_(.08)
        report,_=evaluate(model,data,list(range(8)),step=12,detail_panel=detail,baseline_report=initial)
        candidate=background_detail_candidate(report,initial)
        self.assertFalse(candidate['eligible']);self.assertFalse(candidate['requirements']['main_ssim'])
        self.assertFalse(candidate['requirements']['empty_source_mae'])
        self.assertGreater(report['models']['current']['summary']['low_frequency_change_mae'],.07)
        self.assertGreater(report['models']['current']['summary']['empty_paint_top64_mae'],.07)
        unsupported=deepcopy(report);unsupported['models']['current']['cohorts']['empty']['mae_mean']=None
        self.assertFalse(background_detail_candidate(unsupported)['support_complete'])
        self.assertFalse(background_detail_candidate(unsupported)['eligible'])

    def test_actual_rgb_architecture_and_canonical_objective_use_same_frozen_base(self):
        from turkeyworld.background_detail_sr import BackgroundDetailSR
        from turkeyworld.background_detail_objectives import LOSS_WEIGHTS
        model=BackgroundDetailSR(channels=8,blocks=1,semantic_channels=8,semantic_blocks=1,
            detail_channels=8,detail_blocks=1)
        data=data_fixture();detail=build_detail_animation_panel(data)
        report,_=evaluate_background_detail_sr(model,data,list(range(8)),step=0,batch_size=4,detail_panel=detail)
        self.assertEqual(set(report['validation_objective']['components']),set(LOSS_WEIGHTS))
        self.assertAlmostEqual(report['validation_objective']['loss'],sum(value['weighted']
            for value in report['validation_objective']['components'].values()),places=6)
        for arm in report['models'].values():self.assertEqual(arm['summary']['residual_mean_abs'],0.)
        self.assertTrue(report['preferred_base_preservation']['eligible'])
        self.assertTrue(any(p.requires_grad for p in model.detail_residual.parameters()))

    def test_reference_protocol_panels_frozen_base_and_future_are_strict_and_preview_four_rows(self):
        model=BackgroundDetailFixture();data=data_fixture();detail=build_detail_animation_panel(data)
        baseline,_=evaluate(model,data,list(range(8)),step=0,detail_panel=detail)
        preview=preview_background_detail_sr(model,data,list(range(8)),step=15,baseline_report=baseline,detail_panel=detail)
        self.assertTrue(preview['metadata']['preview_only']);self.assertEqual(preview['metadata']['reference_validation_step'],0)
        with tempfile.TemporaryDirectory()as folder:
            saved=save_background_detail_sr_preview(preview,Path(folder)/'preview.png')
            with Image.open(saved['path'])as picture:self.assertEqual(picture.size,(1024,1120))
            self.assertIn('Выбранная замороженная',saved['row_labels'][2])
        for mutate in(lambda x:x.update(objective_protocol='neural_enemy_restoration_detail_v3'),
                       lambda x:x.update(step=16),lambda x:x.update(preferred_background_sha256='other'),
                       lambda x:x['detail_validation'].update(fingerprint='other')):
            invalid=deepcopy(baseline);mutate(invalid)
            with self.assertRaisesRegex(ValueError,'reference|future'):
                preview_background_detail_sr(model,data,list(range(8)),step=15,baseline_report=invalid,detail_panel=detail)
        changed=deepcopy(detail);changed['indices'][0]=changed['indices'][1]
        with self.assertRaises(ValueError):evaluate(model,data,list(range(8)),step=0,detail_panel=changed)
        with self.assertRaises(ValueError):evaluate(model,data,list(range(8)),step=0,detail_panel=None)
        with torch.no_grad():model.preferred_base.add_(.01)
        with self.assertRaisesRegex(ValueError,'preferred base'):
            preview_background_detail_sr(model,data,list(range(8)),step=15,baseline_report=baseline,detail_panel=detail)
        data.split='train'
        with self.assertRaisesRegex(ValueError,'held-out VAL'):evaluate(model,data,list(range(8)),step=0,detail_panel=detail)


if __name__=='__main__':unittest.main()
