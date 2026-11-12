"""Native small-object gradients, exact wall masks and explicit sampler phases."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_renderer_enemy_objectives import tiny_model, batch
from test_renderer_distance_audit import FakeParent
from turkeyworld.engine import Game
from turkeyworld.renderer_enemy_objectives import enemy_refinement_loss
from turkeyworld.renderer_enemy_validation import evaluate_enemy_renderer
from turkeyworld.renderer_refinement import frozen_fingerprint
from turkeyworld.renderer_small_enemy_data import SmallEnemyBalancedFrames
from turkeyworld.renderer_small_enemy_protocol import (separate_region_loss,native_crop_loss,
    native_area_report,native_area_assessment,native_preview_positions,small_enemy_protocol)
from turkeyworld.renderer_wall_supervision import source_wall_masks,wall_loss_terms,wall_preservation_assessment


class RendererSmallEnemyProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_native_1_8_31_pixels_have_equal_fg_mass_and_empty_rows_do_not_dilute_it(self):
        for area in (1,8,31):
            for empty_count in (0,7,31):
                p=torch.zeros(1+empty_count,3,32,32,requires_grad=True)
                t=torch.zeros_like(p,requires_grad=True)
                mask=torch.zeros(1+empty_count,1,32,32)
                mask[0].view(-1)[:area]=1
                with torch.no_grad(): p[0].view(3,-1)[:,:area]=.4
                mask.requires_grad_()
                regions=separate_region_loss(p,t,mask)
                self.assertAlmostEqual(float(regions['foreground'].detach()),.4,places=6)
                self.assertAlmostEqual(float(regions['loss'].detach()),.2,places=6)
                self.assertEqual(regions['foreground_images'],1)
                regions['loss'].backward()
                self.assertAlmostEqual(float(p.grad.abs().sum()),.5,places=6)
                self.assertIsNone(t.grad); self.assertIsNone(mask.grad)
        empty=torch.zeros(2,1,32,32)
        regions=separate_region_loss(torch.ones(2,3,32,32),torch.zeros(2,3,32,32),empty)
        self.assertEqual(regions['foreground_images'],0)
        self.assertEqual(float(regions['foreground']),0)
        self.assertEqual(float(regions['loss']),.5)

    def test_native_crop_sees_local_errors_without_resizing_or_distant_background(self):
        t=torch.zeros(2,3,32,32,requires_grad=True)
        mask=torch.zeros(2,1,32,32,requires_grad=True)
        with torch.no_grad():mask[0,0,15,16]=1
        p=torch.zeros_like(t,requires_grad=True)
        with torch.no_grad():p[0,:,15,16]=1
        loss,terms=native_crop_loss(p,t,mask)
        self.assertEqual(terms['native_crop_images'],1)
        changed=p.detach().clone();changed[:,:,:5]=1
        changed_loss,_=native_crop_loss(changed,t,mask)
        torch.testing.assert_close(loss.detach(),changed_loss,rtol=0,atol=0)
        loss.backward()
        self.assertGreater(float(p.grad[0,:,15,16].abs().sum()),0)
        self.assertEqual(float(p.grad[1].abs().sum()),0)
        self.assertIsNone(t.grad);self.assertIsNone(mask.grad)

    def test_source_wall_projection_matches_real_flat_source_render_and_excludes_npc_gun(self):
        game=Game(seed=2)
        for flash in (0.,.1,.6):
            state=game.state_vector();state[7]=flash
            game.set_state_vector(state)
            state=game.state_vector();game.enemies=[]
            game._textures[:]=np.array([240,0,240],np.uint8)
            # Supervision excludes any weapon alpha, including alpha so faint
            # that integer compositing leaves RGB unchanged. Make that same
            # raster support opaque for this independently rendered oracle.
            for sprite in game._guns:sprite[...,3]=(sprite[...,3]>0).astype(np.uint8)*255
            rgb=game.render(32,overlays=False)
            mask=torch.zeros(1,1,32,32)
            labels=source_wall_masks(torch.from_numpy(game.map.copy())[None],torch.from_numpy(state)[None],mask)
            expected=(rgb[...,0]==rgb[...,2])&(rgb[...,1]==0)&(rgb[...,0]>0)
            np.testing.assert_array_equal(labels['wall'][0,0].numpy().astype(bool),expected)
            self.assertGreater(int(labels['boundary'].sum()),0)
            mask[0,0,10:14,10:14]=1
            occluded=source_wall_masks(torch.from_numpy(game.map.copy())[None],torch.from_numpy(state)[None],mask)
            self.assertFalse(bool(occluded['wall'][0,0,10:14,10:14].any()))
            self.assertFalse(bool(occluded['boundary'][0,0,10:14,10:14].any()))
            # Restore an actual one-NPC state for the next Source decode.
            game=Game(seed=2)

    def test_wall_errors_normalize_present_frames_and_ignore_occluders(self):
        state=batch((False,))
        labels=source_wall_masks(state['map'],state['state'],state['object_mask'])
        t=torch.zeros(1,3,32,32,requires_grad=True)
        p=torch.ones_like(t,requires_grad=True)
        one=wall_loss_terms(p,t,labels)
        self.assertEqual(one['boundary_images'],1)
        self.assertAlmostEqual(float(one['boundary_l1'].detach()),1.)
        labels2={k:torch.cat([v,torch.zeros_like(v).expand(7,*v.shape[1:])]) for k,v in labels.items()}
        p2=torch.cat([p,torch.zeros(7,3,32,32)])
        t2=torch.zeros_like(p2)
        many=wall_loss_terms(p2,t2,labels2)
        torch.testing.assert_close(one['loss'],many['loss'],rtol=0,atol=0)
        one['loss'].backward()
        self.assertGreater(float(p.grad.abs().sum()),0);self.assertIsNone(t.grad)

    def test_exact_quota_rng_resume_and_no_val_or_fractional_area_sampling(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'train.npz'
            states=np.zeros((6,20),np.float32);states[:,12:14]=1
            states[:,8]=np.asarray([1,2,4,5,1,1])/16
            states[4,12:14]=0
            np.savez(path,states=states)
            parent=FakeParent(path);parent.split='train'
            areas=(1,8,31,90,12,0)
            for row,area in zip(parent.rows,areas):row['mask_fraction']=area/1024
            data=SmallEnemyBalancedFrames(parent)
            self.assertEqual(data.quota_buckets['tiny_1_16'],[0,1])
            self.assertEqual(data.quota_buckets['small_17_64'],[2])
            self.assertEqual(data.quota_buckets['far'],[3])
            for _ in range(10):data.batch(16)
            self.assertEqual(data.distance_counts,dict.fromkeys(('tiny_1_16','small_17_64','far','replay'),40))
            self.assertEqual(data.draws,160);self.assertEqual(parent.draws,40)
            saved=copy.deepcopy(data.sampler_state());json.dumps(saved,allow_nan=False)
            expected=data.batch(16)['indices']
            data.restore_sampler(saved)
            np.testing.assert_array_equal(expected,data.batch(16)['indices'])
            corrupted=copy.deepcopy(saved);corrupted['actual_area_counts']['empty']+=1
            with self.assertRaisesRegex(ValueError,'clocks'):data.restore_sampler(corrupted)
            parent.split='val'
            with self.assertRaisesRegex(ValueError,'TRAIN'):SmallEnemyBalancedFrames(parent)
            parent.split='train';parent.rows[0]['mask_fraction']=1.5/1024
            with self.assertRaisesRegex(ValueError,'exact native32'):SmallEnemyBalancedFrames(parent)

    def test_optin_real_base_rf_gradients_default_exact_and_frozen_model_and_labels(self):
        model=tiny_model();source=batch((False,True))
        for key,value in source.items():
            if value.is_floating_point():source[key]=value.requires_grad_()
        frozen=frozen_fingerprint(model)
        codec={name:value.clone() for name,value in model.flow_renderer.codec.state_dict().items()}
        default,old=enemy_refinement_loss(model,source,latent_weight=.1)
        explicit,unchanged=enemy_refinement_loss(model,source,latent_weight=.1,small_enemy_rebalance=False)
        self.assertTrue(torch.equal(default,explicit));self.assertEqual(old,unchanged)
        value,metrics=enemy_refinement_loss(model,source,latent_weight=.1,small_enemy_rebalance=True)
        self.assertTrue(torch.isfinite(value));json.dumps(metrics,allow_nan=False)
        self.assertTrue(metrics['source_wall_masks_are_loss_only'])
        self.assertGreater(metrics['base_native_crop_images'],0)
        value.backward()
        for head in (model.base_student,model.flow_renderer):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in head.parameters() if p.grad is not None),0)
        self.assertTrue(all(v.grad is None for v in source.values() if v.is_floating_point()))
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4);opt.step()
        self.assertEqual(frozen_fingerprint(model),frozen)
        self.assertTrue(all(torch.equal(value,model.flow_renderer.codec.state_dict()[name]) for name,value in codec.items()))
        self.assertTrue(all(p.grad is None for p in model.flow_renderer.codec.parameters()))

    def test_real_validation_reports_native_roi_walls_and_codec_and_keeps_rng_buffers(self):
        model=tiny_model();source=batch((False,True,False,True))
        class Data:
            split='val';rows=[{'category':key}for key in ('large','partial','corpse','empty_navigation')]
            def batch(self,n,indices):return {key:value[indices] for key,value in source.items()}
            def panel_fingerprint(self,panel):return 'same_val_panel'
        before={n:v.clone()for n,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
        report,arrays=evaluate_enemy_renderer(model,Data(),[0,1,2,3],batch_size=3,small_enemy_rebalance=True)
        self.assertEqual(report['panel_fingerprint'],'same_val_panel')
        self.assertIn('codec_reference',report['native_area_validation'])
        self.assertIn('boundary_l1',report['source_wall_validation']['codec_reference'])
        self.assertFalse(report['native_area_validation']['codec_reference_is_a_proven_lower_bound'])
        json.dumps(report,allow_nan=False)
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        self.assertTrue(all(torch.equal(v,model.state_dict()[n])for n,v in before.items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_small_metrics_do_not_hide_fg_regression_with_global_improvement(self):
        t=torch.zeros(4,3,32,32);m=torch.zeros(4,1,32,32);s=torch.zeros(4,20)
        s[:,12:14]=1;s[:,8]=.5
        m[0].view(-1)[:8]=1;m[1].view(-1)[:31]=1;m[2].view(-1)[:100]=1
        t=m.expand_as(t).clone()
        arrays={'source':t,'mask':m,'states':s,'base':t*.5,'rgb':t*.5,'codec_reference':t*.8}
        old=native_area_report(arrays)
        better=native_area_report({**arrays,'base':t*.75,'rgb':t*.75})
        self.assertTrue(native_area_assessment(better,old)['passed'])
        bad=copy.deepcopy(better);bad['rf1']['tiny_1_16']['foreground_mae']=.6
        self.assertFalse(native_area_assessment(bad,old)['passed'])
        bad=copy.deepcopy(better);bad['base']['empty']['full_image_mae']=.01
        self.assertFalse(native_area_assessment(bad,old)['passed'])
        self.assertEqual(native_preview_positions(m,s),[0,1,2,3])
        config=small_enemy_protocol();config['source_quotas']['far']=0
        self.assertEqual(small_enemy_protocol()['source_quotas']['far'],.25)

    def test_wall_guard_rejects_nonfinite_support_changes_and_hidden_boundary_regression(self):
        old={head:{'boundary_l1':.1,'wall_edge_l1':.1,'boundary_images':4,'wall_edge_images':4}
             for head in ('base','rf1')}
        better=copy.deepcopy(old);better['rf1']['boundary_l1']=.09
        self.assertTrue(wall_preservation_assessment(better,old)['passed'])
        for field,value in (('boundary_l1',.11),('wall_edge_l1',float('nan')),('boundary_images',0)):
            bad=copy.deepcopy(better);bad['rf1'][field]=value
            self.assertFalse(wall_preservation_assessment(bad,old)['passed'])


if __name__=='__main__':unittest.main()
