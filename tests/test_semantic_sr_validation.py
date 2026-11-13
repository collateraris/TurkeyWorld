"""Actual NPR-only inference, same-tick HR, present-only metrics and fixed previews."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from turkeyworld.semantic_sr_validation import (evaluate_semantic_sr, save_semantic_sr_preview,
    preview_semantic_sr, _rows, _summary, _assess)
from turkeyworld.semantic_sr_objectives import semantic_sr_loss, LOSS_WEIGHTS


class PairedData:
    split = 'val'
    identity = dict(kind='actual_NPR32_true_Source128', fingerprint='fixed_npr_source_pairs')
    def __init__(self):
        self.rng = np.random.default_rng(19); self.draws = 0
        self.source = torch.full((8, 3, 32, 32), .2)
        self.mask = torch.zeros(8, 1, 32, 32)
        for index, extent in ((0, 9), (1, 5), (2, 3), (3, 1), (6, 3), (7, 2)):
            self.mask[index, :, 10:10+extent, 10:10+extent] = 1
        self.source[:, 0:1] += self.mask*.6
        self.npr = (self.source-.01).requires_grad_()
        # True HR has native high-frequency content absent from any LR resize.
        checker = ((torch.arange(128)[:, None]+torch.arange(128)[None, :]) % 2)*.02
        self.hr = (F.interpolate(self.source, scale_factor=4, mode='nearest')+checker).requires_grad_()
        self.hr_mask = self.mask.repeat_interleave(4, -1).repeat_interleave(4, -2).requires_grad_()
        self.background = torch.full_like(self.hr, .21)
        self.state = torch.zeros(8, 20); self.state[:, 4] = 1; self.state[:, 12:14] = 1
        self.state[:, 8] = torch.tensor([1., 3., 7., 8., 1., 6., 2., 3.])/16
        self.state[4, 12:14] = 0; self.state[6, 12:14] = 0; self.state[7, 4] = 0
    def panel_fingerprint(self, panel): return 'fixed_'+','.join(map(str, panel))
    def sampler_state(self): return copy.deepcopy(dict(rng=self.rng.bit_generator.state, draws=self.draws))
    def restore_sampler(self, state): self.rng.bit_generator.state=state['rng']; self.draws=state['draws']
    def batch(self, count, *, indices):
        ids = torch.tensor(indices); self.draws += count
        self.rng.random(); np.random.random(); random.random(); torch.rand(1)
        return dict(lr_npr=self.npr[ids], lr_source=self.source[ids], codec_lr=self.source[ids]-.005,
            hr_source=self.hr[ids], mask_lr=self.mask[ids], mask_hr=self.hr_mask[ids],
            hr_background=self.background[ids], state=self.state[ids])


class TinySR(nn.Module):
    def __init__(self, offset):
        super().__init__(); self.offset=nn.Parameter(torch.tensor(float(offset))); self.calls=[]
        self.branch=nn.Identity()
    def forward(self, image):
        self.calls.append(image.detach().clone())
        return (F.interpolate(image, scale_factor=4, mode='bilinear', align_corners=False)+self.offset).clamp(0, 1)


class TinySemantic(TinySR):
    def forward_with_aux(self, image):
        mask=(image[:, :1]>.6)
        return dict(image=self(image), mask_logits_lr=mask.float()*12-6,
            presence_logits=mask.flatten(1).any(-1).float()[:, None]*12-6)


class SemanticSRValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)
    def fixture(self):
        return PairedData(), TinySemantic(.01), {'sr38100':TinySR(.04), 'sr40100':TinySR(.02)}

    def test_domains_true_HR_distance_area_aux_metrics_and_all_rng_modes_preserved(self):
        data, model, references=self.fixture(); model.train(); model.branch.eval()
        before=data.sampler_state(); torch_rng=torch.get_rng_state().clone(); py=random.getstate(); np_rng=np.random.get_state()
        modes=[m.training for m in model.modules()]
        report, preview=evaluate_semantic_sr(model,data,list(range(8)),step=100,batch_size=4,baselines=references)
        self.assertEqual(before,data.sampler_state()); self.assertTrue(torch.equal(torch_rng,torch.get_rng_state()))
        self.assertEqual(py,random.getstate()); self.assertTrue(all(np.array_equal(a,b) for a,b in zip(np_rng,np.random.get_state())))
        self.assertEqual(modes,[m.training for m in model.modules()])
        self.assertEqual(set(report['domains']),{'npr','clean','codec'})
        self.assertTrue(torch.equal(model.calls[0],data.npr[:4].detach()))
        self.assertTrue(torch.equal(references['sr38100'].calls[0],data.npr[:4].detach()))
        rows=report['domains']['npr']['models']['current']['cohorts']
        self.assertEqual([rows[k]['frames'] for k in ('near','mid','far')],[1,1,2])
        self.assertEqual([rows[k]['frames'] for k in ('area_1_4','area_5_16','area_17_64','area_gt64')],[1,1,1,1])
        self.assertEqual(rows['empty']['aux_presence_false_positive_rate'],0)
        self.assertEqual(rows['visible_enemy']['aux_mask_iou_mean'],1)
        self.assertIsNotNone(rows['visible_enemy']['roi_crop_ssim_mean'])
        self.assertEqual(set(preview['cases']),{'near','mid','far','empty'})
        self.assertFalse(report['gameplay_qualified']); self.assertFalse(report['automatic_deployment'])
        self.assertTrue(all(p.grad is None for net in (model,*references.values()) for p in net.parameters()))
        self.assertIsNone(data.hr.grad); self.assertIsNone(data.state.grad); self.assertIsNone(data.npr.grad)
        json.dumps(report,allow_nan=False)

    def test_cached_frozen_baselines_and_pixels_need_exact_data_and_panel_identity(self):
        data, model, references=self.fixture()
        initial, _=evaluate_semantic_sr(model,data,list(range(8)),baselines=references)
        current, preview=evaluate_semantic_sr(model,data,list(range(8)),baseline_report=initial)
        for name in references:
            self.assertEqual(initial['domains']['npr']['models'][name],current['domains']['npr']['models'][name])
            self.assertEqual(preview['cases']['near']['predictions'][name].shape,(128,128,3))
        bad=copy.deepcopy(initial); bad['data_identity_sha256']='other'
        with self.assertRaisesRegex(ValueError,'different Source/NPR/panel'):
            evaluate_semantic_sr(model,data,list(range(8)),baseline_report=bad)
        with self.assertRaisesRegex(ValueError,'different Source/NPR/panel'):
            evaluate_semantic_sr(model,data,list(reversed(range(8))),baseline_report=initial)

    def test_foreground_mean_ignores_empty_frames_and_ROI_SSIM_is_actual_crop(self):
        truth=torch.zeros(2,3,128,128); prediction=truth.clone(); prediction[0,:,60:64,60:64]=1
        mask=torch.zeros(2,1,128,128); mask[0,:,60:64,60:64]=1
        rows=_rows(prediction,truth,mask,truth,torch.zeros(2,1,32,32))
        summary=_summary(rows,[True,True])
        self.assertEqual(summary['foreground_frames'],1); self.assertEqual(summary['foreground_mae'],1)
        self.assertEqual(summary['roi_crop_frames'],1)
        self.assertLess(summary['roi_crop_ssim_mean'],summary['ssim_mean'])

    def test_preview_five_full_rows_and_near_mid_far_same_Source_boxes_with_cached_references(self):
        data, model, references=self.fixture()
        report, preview=evaluate_semantic_sr(model,data,list(range(8)),step=100,baselines=references)
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'preview.png'
            metadata=save_semantic_sr_preview(preview,output)
            with Image.open(output) as image:self.assertEqual(image.size,(1024,2272))
            self.assertEqual(metadata['row_count'],5)
            self.assertEqual(metadata['ROI_zoom_rows'],['near','mid','far'])
            self.assertEqual(metadata['input_kind'],'npr_rgb')
            self.assertFalse(metadata['source_input_allowed_for_sr'])
            self.assertEqual(json.loads(output.with_suffix('.json').read_text())['panel_fingerprint'],report['panel_fingerprint'])

    def test_invalid_references_or_nonfinite_network_output_rejected_without_input_fallback(self):
        data, model, references=self.fixture()
        with self.assertRaisesRegex(ValueError,'two frozen SR references'):
            evaluate_semantic_sr(model,data,list(range(8)))
        with self.assertRaisesRegex(ValueError,'both38100 and40100'):
            evaluate_semantic_sr(model,data,list(range(8)),baselines={'sr38100':references['sr38100']})
        with torch.no_grad():model.offset.fill_(float('nan'))
        with self.assertRaisesRegex(FloatingPointError,'Invalid SR RGB128'):
            evaluate_semantic_sr(model,data,list(range(8)),baselines=references)

    def test_actual_semantic_EDSR_aux_shapes_and_native_RGB_forward_use_the_same_evaluator(self):
        from turkeyworld.semantic_sr import SemanticEDSRUpscaler
        from turkeyworld.models import EDSRUpscaler
        data=PairedData(); model=SemanticEDSRUpscaler(channels=8,blocks=1,
            semantic_channels=4,semantic_blocks=1)
        references={name:EDSRUpscaler(channels=8,blocks=1) for name in ('sr38100','sr40100')}
        report,_=evaluate_semantic_sr(model,data,np.arange(8),baselines=references,batch_size=4)
        rows=report['domains']['npr']['models']['current']['cohorts']
        self.assertEqual(rows['visible_enemy']['frames'],4)
        self.assertIsNotNone(rows['visible_enemy']['aux_mask_pixel_f1_mean'])
        self.assertEqual(rows['empty']['aux_presence_negative_frames'],2)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_light_preview_only_four_fixed_NPR_cases_restores_rng_and_marks_metrics_reference_age(self):
        data, model, references=self.fixture()
        baseline,_=evaluate_semantic_sr(model,data,list(range(8)),baselines=references)
        before=data.sampler_state(); rng=torch.get_rng_state().clone(); model.calls.clear()
        preview=preview_semantic_sr(model,data,list(range(8)),step=15,baseline_report=baseline)
        self.assertEqual(sum(len(x)for x in model.calls),4)
        self.assertEqual(before,data.sampler_state()); self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(preview['metadata']['preview_only']); self.assertEqual(preview['metadata']['step'],15)
        self.assertEqual(preview['metadata']['reference_validation_step'],0)
        self.assertEqual(set(preview['cases']),{'near','mid','far','empty'})

    def test_optional_codec_and_clean_guards_detect_RGB_or_background_forgetting(self):
        data, model, references=self.fixture()
        report,_=evaluate_semantic_sr(model,data,list(range(8)),baselines=references)
        for domain in ('clean','codec'):
            broken=copy.deepcopy(report)
            broken['domains'][domain]['models']['current']['summary']['mae_mean']=.9
            relative,_,_=_assess(broken)
            self.assertIn(domain+'_full_or_background_regression',relative['failure_reasons'])

    def test_validation_objective_is_exact_NPR_formula_with_image_weighted_partial_batch(self):
        data, model, references=self.fixture()
        manual=PairedData();expected_model=TinySemantic(.01)
        for item in (data,manual):
            with torch.no_grad():item.hr[6:].copy_((item.hr[6:]+.1).clamp(0,1))
        expected=[]
        with torch.no_grad():
            for first in range(0,8,3):
                ids=list(range(first,min(first+3,8)))
                batch=manual.batch(len(ids),indices=ids)
                output=expected_model.forward_with_aux(batch['lr_npr'].detach())
                loss,terms=semantic_sr_loss(output,batch)
                expected.append((len(ids),float(loss),terms))
        before=data.sampler_state();model.train();model.branch.eval()
        modes=[module.training for module in model.modules()]
        with patch('turkeyworld.semantic_sr_validation.semantic_sr_loss',wraps=semantic_sr_loss) as called:
            report,_=evaluate_semantic_sr(model,data,list(range(8)),baselines=references,batch_size=3)
        actual=report['validation_objective']
        self.assertEqual(called.call_count,3)  # NPR only, not clean/codec/reference forwards.
        self.assertEqual(len(model.calls),9)  # The existing forwards are reused, not repeated.
        self.assertEqual(actual['protocol'],'rgb_only_semantic_npr_sr_v1')
        self.assertEqual(actual['input_kind'],'npr_rgb');self.assertEqual(actual['frames'],8)
        self.assertEqual(actual['batch_size'],3);self.assertEqual(actual['batch_count'],3)
        self.assertEqual(actual['aggregation'],'image_count_weighted_mean_of_actual_batch_objectives')
        expected_loss=sum(count*loss for count,loss,_ in expected)/8
        self.assertAlmostEqual(actual['loss'],expected_loss,places=8)
        self.assertGreater(abs(expected_loss-sum(loss for _,loss,_ in expected)/3),1e-4)
        for component,weight in LOSS_WEIGHTS.items():
            expected_raw=sum(count*terms[component] for count,_,terms in expected)/8
            expected_weighted=sum(count*terms[component+'_weighted'] for count,_,terms in expected)/8
            self.assertAlmostEqual(actual['components'][component]['raw'],expected_raw,places=8)
            self.assertAlmostEqual(actual['components'][component]['weighted'],expected_weighted,places=8)
            self.assertEqual(actual['components'][component]['weight'],weight)
        self.assertAlmostEqual(actual['loss'],sum(row['weighted']for row in actual['components'].values()),places=6)
        self.assertTrue(actual['source_held_out'])
        self.assertFalse(actual['training_input_domain_difference']['same_input_mixture'])
        self.assertEqual(before,data.sampler_state())
        self.assertEqual(modes,[module.training for module in model.modules()])
        self.assertTrue(all(p.grad is None for net in (model,*references.values()) for p in net.parameters()))
        for value in (data.hr,data.hr_mask,data.npr,data.state):self.assertIsNone(value.grad)
        json.dumps(actual,allow_nan=False)


if __name__=='__main__': unittest.main()
