from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import torch

from test_renderer_enemy_objectives import tiny_model, batch
from test_renderer_far_wall_protocol import synthetic_labels
from turkeyworld.renderer_geometry_objectives import (
    geometry_wall_protocol, geometry_wall_loss, geometry_wall_terms,
    geometry_refinement_loss, geometry_wall_report, geometry_preservation_assessment,
    geometry_report_score, _cross_boundary_contrast)
from turkeyworld.renderer_refinement import frozen_fingerprint


class RendererGeometryObjectivesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_near_and_tiny_far_regions_have_equal_mass_and_empty_images_do_not_dilute(self):
        for empty in (0, 7):
            labels=synthetic_labels(1, empty)
            far=labels['bands']['far_5_8']
            # Reuse a nonempty support as a near wall, leaving distant one pixel.
            labels['bands']['near_0_2']=deepcopy(far)
            for value in labels['bands']['far_5_8'].values(): value.zero_()
            prediction=torch.zeros(1+empty, 3, 32, 32, requires_grad=True)
            target=torch.zeros_like(prediction, requires_grad=True)
            with torch.no_grad():
                prediction[0,:,10,0]=.2; prediction[0,:,20,0]=.8
            loss, metrics=geometry_wall_loss(prediction, target, labels)
            # Every non-edge term sees the same two present bands, mean(.2,.8).
            expected=.5*sum(v for k,v in geometry_wall_protocol()['weights_per_head'].items()
                if k not in ('wall_edge_l1','cross_boundary_contrast'))
            self.assertAlmostEqual(float(loss.detach()), expected, places=6)
            self.assertEqual(metrics['wall_l1_present_bands'],2)
            loss.backward(); self.assertIsNone(target.grad)
            self.assertGreater(float(prediction.grad.abs().sum()),0)
            json.dumps(metrics,allow_nan=False)

    def test_cross_boundary_penalizes_flattened_and_swapped_edges_and_excludes_hidden_pixels(self):
        labels=synthetic_labels()['bands']['far_5_8']
        for value in labels.values(): value.zero_()
        labels['boundary'][0,0,10,10:12]=1
        labels['horizontal'][0,0,10,10]=1
        target=torch.zeros(1,3,32,32,requires_grad=True)
        with torch.no_grad(): target[:,:,:,11:]=1
        for left,right in ((.5,.5),(1.,0.),(0.,0.),(1.,1.)):
            prediction=target.detach().clone()
            prediction[:,:,10,10]=left; prediction[:,:,10,11]=right
            prediction.requires_grad_()
            loss,count,pixels=_cross_boundary_contrast(prediction,target,labels)
            self.assertGreater(float(loss.detach()),0)
            self.assertEqual((count,pixels),(1,1))
            loss.backward()
            self.assertGreater(float(prediction.grad[:,:,10,10:12].abs().sum()),0)
        loss,_,_=_cross_boundary_contrast(target,target,labels)
        self.assertEqual(float(loss.detach()),0)
        labels['horizontal'].zero_()
        loss,count,pixels=_cross_boundary_contrast(1-target,target,labels)
        self.assertEqual((float(loss.detach()),count,pixels),(0.,0,0))
        self.assertIsNone(target.grad)

    def test_actual_source_perfect_prediction_has_zero_loss_and_no_codec_or_source_gradient(self):
        model=tiny_model(); source=batch((False,True))
        for name,value in source.items():
            if value.is_floating_point(): source[name]=value.requires_grad_()
        before=frozen_fingerprint(model)
        codec=model.flow_renderer.codec; encode=codec.encode
        encoded=[]
        def encode_base(value):
            self.assertFalse(torch.equal(value.detach(),source['target'].detach()))
            encoded.append(True); return encode(value)
        with patch.object(codec,'encode',side_effect=encode_base), \
                patch('turkeyworld.engine.Game.render',side_effect=AssertionError('Recorded Source only')), \
                patch('turkeyworld.engine.Game.step',side_effect=AssertionError('No Source transitions')):
            loss,metrics=geometry_refinement_loss(model,source)
            loss.backward()
        self.assertEqual(len(encoded),1)
        for head in (model.base_student,model.flow_renderer):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in head.parameters() if p.grad is not None),0)
        self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
        self.assertTrue(all(v.grad is None for v in source.values() if v.is_floating_point()))
        self.assertEqual(before,frozen_fingerprint(model))
        self.assertTrue(metrics['geometry_aux_source_labels_only'])
        self.assertFalse(metrics['geometry_aux_future_source_feedback'])
        json.dumps(metrics,allow_nan=False)
        arrays={'map':source['map'],'states':source['state'],'mask':source['object_mask'],
            'source':source['target'],'base':source['target'],'rgb':source['target'],
            'codec_reference':source['target']}
        report=geometry_wall_report(arrays)
        self.assertEqual(geometry_report_score(report),0)

    def test_guard_rejects_missing_support_protocol_and_regressions_without_claiming_absent_bands(self):
        source=batch((False,True)); target=source['target']
        arrays={'map':source['map'],'states':source['state'],'mask':source['object_mask'],
            'source':target,'base':target,'rgb':target,'codec_reference':target}
        report=geometry_wall_report(arrays); gate=geometry_preservation_assessment(report,report)
        self.assertTrue(gate['passed']); self.assertTrue(gate['unmeasured_regions'])
        for change in ('loss','support','protocol'):
            bad=deepcopy(report)
            if change=='loss':
                band=next(k for k,v in bad['rf1']['bands'].items() if v['wall_images'])
                bad['rf1']['bands'][band]['wall_l1']+=.01
            elif change=='support': bad['base']['bands']['near_0_2']['wall_images']+=1
            else: bad['protocol']['version']=999
            self.assertFalse(geometry_preservation_assessment(bad,report)['passed'])
        json.dumps(report,allow_nan=False); json.dumps(gate,allow_nan=False)
        mutated=geometry_wall_protocol(); mutated['weights_per_head']['wall_l1']=123
        self.assertEqual(geometry_wall_protocol()['weights_per_head']['wall_l1'],.25)


if __name__=='__main__': unittest.main()
