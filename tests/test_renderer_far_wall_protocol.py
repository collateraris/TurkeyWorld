"""Meaningful far-stratum normalization and renderer-only auxiliary gradients."""
import copy
import json
import unittest
from unittest.mock import patch

import torch

from test_renderer_enemy_objectives import tiny_model,batch
from turkeyworld.renderer_enemy_objectives import enemy_refinement_loss,far_wall_refinement_loss
from turkeyworld.renderer_far_wall_protocol import (far_wall_protocol,far_wall_loss,
    far_wall_preservation_assessment,far_wall_report_score)
from turkeyworld.renderer_refinement import frozen_fingerprint
from turkeyworld.renderer_wall_depth import DEPTH_BANDS,source_wall_depth_labels,depth_banded_wall_report


def synthetic_labels(area=1,empty_images=0):
    count=1+empty_images
    def row():return {name:torch.zeros(count,1,32,32)for name in
        ('wall','boundary','corner','thin_face','thin_height')}
    bands={name:row()for name,_,_ in DEPTH_BANDS}
    for row_value in bands.values():
        row_value['horizontal']=torch.zeros(count,1,32,31)
        row_value['vertical']=torch.zeros(count,1,31,32)
    for name,y in (('far_5_8',10),('thin_12_plus',20)):
        for key in ('wall','boundary','corner','thin_face','thin_height'):
            bands[name][key][0,0,y,:area]=1
    return {'bands':bands,'aggregates':{}}


class RendererFarWallProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_native_one_pixel_far_bands_equal_area_mass_and_absent_images_bands_do_not_dilute(self):
        for area in (1,2,16,31):
            for empty_images in (0,7):
                labels=synthetic_labels(area,empty_images)
                prediction=torch.zeros(1+empty_images,3,32,32,requires_grad=True)
                target=torch.zeros_like(prediction,requires_grad=True)
                with torch.no_grad():
                    prediction[0,:,10,:area]=.2;prediction[0,:,20,:area]=.8
                value,metrics=far_wall_loss(prediction,target,labels)
                self.assertAlmostEqual(float(value.detach()),.25,places=6)
                self.assertEqual(metrics['boundary_l1_present_bands'],2)
                self.assertEqual(metrics['wall_edge_l1_present_bands'],0)
                value.backward()
                self.assertAlmostEqual(float(prediction.grad.abs().sum()),.5,places=6)
                self.assertIsNone(target.grad);json.dumps(metrics,allow_nan=False)

    def test_v5_default_exact_and_v6_adds_only_manifest_far_terms_to_real_source_loss(self):
        model=tiny_model();source=batch((False,True))
        old,oldmetrics=enemy_refinement_loss(model,source,latent_weight=.1,small_enemy_rebalance=True)
        default,defaultmetrics=enemy_refinement_loss(model,source,latent_weight=.1,
            small_enemy_rebalance=True,far_wall_supervision=False)
        self.assertTrue(torch.equal(old,default));self.assertEqual(oldmetrics,defaultmetrics)
        loss,metrics=enemy_refinement_loss(model,source,latent_weight=.1,
            small_enemy_rebalance=True,far_wall_supervision=True)
        expected=sum(metrics[name+'_far_wall_weighted_loss']for name in ('base','rf1'))
        self.assertGreater(expected,0)
        self.assertAlmostEqual(float((loss-old).detach()),expected,places=6)
        self.assertEqual(metrics['far_wall_protocol'],far_wall_protocol())
        json.dumps(metrics,allow_nan=False)
        with self.assertRaises(ValueError):enemy_refinement_loss(model,source,far_wall_supervision=True)

    def test_far_only_auxiliary_never_calls_other_loss_families_or_encodes_target_and_trains_both_heads(self):
        model=tiny_model();source=batch((False,True))
        for key,value in source.items():
            if value.is_floating_point():source[key]=value.requires_grad_()
        frozen=frozen_fingerprint(model)
        codec=model.flow_renderer.codec;encode=codec.encode;encoded=[]
        def encode_base(value):
            self.assertFalse(torch.equal(value.detach(),source['target'].detach()))
            encoded.append(value)
            return encode(value)
        names=('balanced_image_l1','balanced_mse','decoded_ssim_loss','balanced_edge_l1',
               'foreground_contrastive_loss','balanced_base_regression_loss','source_object_crop_ssim')
        from contextlib import ExitStack
        with ExitStack()as stack:
            for name in names:stack.enter_context(patch('turkeyworld.renderer_enemy_objectives.'+name,
                side_effect=AssertionError('Auxiliary must be FAR-only')))
            stack.enter_context(patch.object(codec,'encode',side_effect=encode_base))
            loss,metrics=far_wall_refinement_loss(model,source)
        self.assertEqual(len(encoded),1)
        self.assertGreater(float(loss.detach()),0)
        loss.backward()
        for head in (model.base_student,model.flow_renderer):
            self.assertGreater(sum(float(p.grad.abs().sum())for p in head.parameters()if p.grad is not None),0)
        self.assertTrue(all(value.grad is None for value in source.values()if value.is_floating_point()))
        self.assertTrue(all(p.grad is None for p in model.parameters()if not p.requires_grad))
        self.assertEqual(frozen_fingerprint(model),frozen)
        self.assertEqual(metrics['far_wall_aux_examples'],2)
        self.assertFalse(metrics['far_wall_aux_future_source_feedback']);json.dumps(metrics,allow_nan=False)

    def test_exact_source_depth_report_guard_support_and_unmeasured_regions_are_honest(self):
        source=batch((False,True))
        target=source['target']
        arrays={'map':source['map'],'states':source['state'],'mask':source['object_mask'],
            'source':target,'base':target+.1,'rgb':target+.1,'codec_reference':target+.05}
        report=depth_banded_wall_report(arrays)
        gate=far_wall_preservation_assessment(report,report)
        self.assertTrue(gate['passed']);self.assertTrue(gate['measured_regions'])
        self.assertTrue(gate['unmeasured_regions'])
        bad=copy.deepcopy(report);bad['rf1']['bands']['far_5_8']['boundary_l1']+=.01
        self.assertFalse(far_wall_preservation_assessment(bad,report)['passed'])
        bad=copy.deepcopy(report);bad['base']['bands']['far_5_8']['boundary_images']+=1
        self.assertFalse(far_wall_preservation_assessment(bad,report)['passed'])
        self.assertGreater(far_wall_report_score(report),0)
        json.dumps(gate,allow_nan=False)
        protocol=far_wall_protocol();protocol['weights_per_head']['boundary_l1']=100
        self.assertEqual(far_wall_protocol()['weights_per_head']['boundary_l1'],.2)


if __name__=='__main__':unittest.main()
