from copy import deepcopy
from functools import partial
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

from turkeyworld.enemy_animation import enemy_animation_index
from turkeyworld.neural_enemy_restoration_data import NeuralEnemyRestorationData
from turkeyworld.neural_enemy_restoration_validation import(
    evaluate_neural_enemy_restoration,preview_neural_enemy_restoration,
    save_neural_enemy_restoration_preview,_mask_rows,_summary,PROTOCOL,FOREGROUND_PROTOCOL,DETAIL_PROTOCOL,
    EMPTY_TOPK_PIXELS,build_detail_animation_panel)
from turkeyworld.small_enemy_data import digest
from test_enemy_decoder_data import paired_fixture
from test_sr_patchgan_validation import AdjacentPairs


class LearnedRGBFixture(nn.Module):
    def __init__(self):
        super().__init__();self.bias=nn.Parameter(torch.tensor(.0))
        self.config={'architecture':'neural_enemy_restoration_sr','fixture':True}
        self.child=nn.Identity();self.calls=[]
    def forward_with_aux(self,low):
        self.calls.append(len(low));rgb=F.interpolate(low,scale_factor=4,mode='nearest')
        image=(rgb+self.bias).clamp(0,1);alpha=(rgb[:,:1]>.7).float()
        return dict(image=image,background=rgb,enemy_rgb=image,alpha=alpha,
            alpha_logits=torch.logit(alpha.clamp(.001,.999)))


class FrozenAtlasFixture(nn.Module):
    def __init__(self):
        super().__init__();self.scale=nn.Parameter(torch.tensor(.9),requires_grad=False)
        self.config={'architecture':'frozen_atlas_fixture'};self.calls=[]
    def forward_with_aux(self,low,hypothesis=None):
        if hypothesis is not None:raise AssertionError('Reference did not need an RSSM hint')
        self.calls.append(len(low));rgb=F.interpolate(low,scale_factor=4,mode='nearest')
        return dict(image=(rgb*self.scale).clamp(0,1),atlas_composite_alpha_hr=(rgb[:,:1]>.7).float())


class MatchedBackgroundFixture(LearnedRGBFixture):
    def forward_with_aux(self,low):
        output=super().forward_with_aux(low);background=output['background'].clone()
        background[:,:1]-=.6*(background[:,:1]>.7).float()
        output['background']=background
        return output


def detail_data():
    data=AdjacentPairs();data.state[0,14]=1.;data.state[1,15]=.5
    data.state[2:4,12:14]=0;data.state[2,16]=.2;data.state[3,16]=1.
    return data


class Inventory120:
    split='val';identity={'kind':'explicit_VAL_animation_inventory_fixture'}
    def __init__(self):
        self.animation_labels=[3]*28+[4]*64+[5]*20+[6]*40
        self.rows=[{'split':'val','record':index}for index in range(len(self.animation_labels))]
        self.visible=[True]*len(self.rows);self.distance=[('near','mid','far')[index%3]for index in range(len(self.rows))]
    def panel_fingerprint(self,panel):return 'inventory_'+','.join(map(str,panel))


def actual_objective(output,batch):
    pixel=(output['image']-batch['hr_source'].detach()).abs().mean()
    alpha=F.binary_cross_entropy_with_logits(output['alpha_logits'],batch['mask_hr'].detach())
    loss=pixel+.2*alpha
    return loss,dict(pixel=float(pixel),alpha=float(alpha),pixel_weighted=float(pixel),alpha_weighted=float(.2*alpha))


def evaluate(model,reference,data,panel,**kwargs):
    return evaluate_neural_enemy_restoration(model,reference,data,panel,
        objective_fn=actual_objective,objective_weights={'pixel':1.,'alpha':.2},**kwargs)


class NeuralRestorationDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_actual_pair_cache_needs_no_hypothesis_and_balances_visible_distances_animation_and_empty(self):
        with tempfile.TemporaryDirectory()as folder:
            root=paired_fixture(folder);before=(root/'manifest.json').read_bytes()
            shards={path:digest(path)for path in root.rglob('*.npz')}
            data=NeuralEnemyRestorationData(root,'train');val=NeuralEnemyRestorationData(root,'val')
            base=data.base.sampler_state();start=data.sampler_state()
            batch=data.batch(8,indices=list(range(8)))
            self.assertNotIn('hypothesis',batch)
            self.assertEqual(batch['visible_labels'].shape,(8,1))
            self.assertTrue(torch.equal(batch['visible_labels'],batch['mask_hr'].bool().flatten(1).any(1)[:,None]))
            self.assertTrue(torch.equal(batch['animation_label'],enemy_animation_index(batch['state'])))
            self.assertEqual(start,data.sampler_state());self.assertEqual(base,data.base.sampler_state())
            picked=data.batch(120)['row_index'].tolist()
            self.assertEqual(data.case_counts,{'visible':60,'empty':60})
            self.assertEqual(data.subcounts['visible'],{'near':20,'mid':20,'far':20})
            self.assertEqual({key:sum(value.values())for key,value in data.animation_subcounts.items()},
                {'near':20,'mid':20,'far':20})
            data.restore_sampler(start);self.assertEqual(picked,data.batch(120)['row_index'].tolist())
            self.assertEqual(len(val.fixed_panel()),8)
            self.assertEqual(before,(root/'manifest.json').read_bytes())
            self.assertTrue(all(digest(path)==value for path,value in shards.items()))
            self.assertFalse(data.identity['hypothesis_sidecar_required'])
            self.assertTrue(data.identity['inference_rgb_only'])
            self.assertEqual(base,data.base.sampler_state())
            bad=deepcopy(data.sampler_state());bad['subcounts']['visible']['near']+=1
            with self.assertRaisesRegex(ValueError,'strata counters'):data.restore_sampler(bad)
            bad=deepcopy(data.sampler_state());phase=next(iter(bad['animation_subcounts']['near']))
            bad['animation_subcounts']['near'][phase]+=1
            with self.assertRaisesRegex(ValueError,'animation counters'):data.restore_sampler(bad)
            json.dumps(data.identity,allow_nan=False)

    def test_heldout_validation_measures_masks_raw_rgb_phases_temporal_and_preserves_all_state(self):
        model=LearnedRGBFixture();reference=FrozenAtlasFixture();data=AdjacentPairs()
        data.state[2,15]=.5;data.state[4,13]=0
        model.train();model.child.eval();reference.train()
        modes=[[module.training for module in net.modules()]for net in(model,reference)]
        gradflags=[[p.requires_grad for p in net.parameters()]for net in(model,reference)]
        sampler=data.sampler_state();rng=torch.get_rng_state().clone();py=random.getstate();nr=np.random.get_state()
        weights=[deepcopy(net.state_dict())for net in(model,reference)]
        report,preview=evaluate(model,reference,data,list(range(8)),step=100,batch_size=3)
        self.assertEqual(report['frames'],8);self.assertTrue(report['inference_rgb_only'])
        self.assertFalse(report['new_model_uses_fixed_atlas']);self.assertFalse(report['Source_or_RSSM_hint_as_input'])
        current=report['models']['current'];summary=current['summary']
        self.assertEqual(summary['mae_mean'],0.);self.assertAlmostEqual(summary['ssim_mean'],1.)
        self.assertEqual(summary['foreground_mae'],0.);self.assertEqual(summary['raw_rgb_foreground_mae'],0.)
        self.assertEqual(summary['alpha_iou'],1.);self.assertEqual(summary['alpha_pixel_recall'],1.)
        self.assertEqual(summary['empty_frames'],2);self.assertEqual(summary['empty_false_npc_rate'],0.)
        self.assertEqual(current['cohorts']['animation_hurt']['frames'],1)
        self.assertEqual(current['cohorts']['animation_death_start']['frames'],1)
        self.assertEqual(report['temporal']['pair_count'],4)
        self.assertEqual(report['temporal']['arms']['current']['delta_mae'],0.)
        objective=report['validation_objective']
        self.assertAlmostEqual(objective['loss'],sum(value['weighted']for value in objective['components'].values()),places=7)
        self.assertNotIn('reconstruction_loss',report['models']['frozen_atlas_parent']['summary'])
        self.assertEqual(preview['metadata']['row_count'],4)
        self.assertEqual(set(preview['cases']),{'near','mid','far','empty'})
        self.assertFalse(report['gameplay_qualified']);self.assertFalse(report['automatic_deployment'])
        self.assertEqual(modes,[[module.training for module in net.modules()]for net in(model,reference)])
        self.assertEqual(gradflags,[[p.requires_grad for p in net.parameters()]for net in(model,reference)])
        self.assertEqual(sampler,data.sampler_state());self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(py,random.getstate());self.assertTrue(all(np.array_equal(a,b)for a,b in zip(nr,np.random.get_state())))
        for net,saved in zip((model,reference),weights):
            self.assertTrue(all(p.grad is None for p in net.parameters()))
            for name,value in saved.items():torch.testing.assert_close(value,net.state_dict()[name],rtol=0,atol=0)
        json.dumps(report,allow_nan=False)

    def test_predicted_alpha_cannot_hide_raw_rgb_error_and_empty_masks_do_not_inflate_foreground_iou(self):
        data=AdjacentPairs();batch=data.batch(8,indices=list(range(8)))
        output=LearnedRGBFixture().forward_with_aux(batch['lr_npr'])
        good=_mask_rows(output,batch,current=True)
        bad={**output,'alpha':torch.zeros_like(output['alpha']),'enemy_rgb':torch.zeros_like(output['enemy_rgb'])}
        rows=_mask_rows(bad,batch,current=True)
        self.assertGreater(rows[0]['raw_rgb_foreground_mae'],good[0]['raw_rgb_foreground_mae'])
        self.assertEqual(rows[0]['alpha_pixel_recall'],0.)
        self.assertIsNone(rows[-1]['alpha_iou'])
        summary=_summary(rows,[True]*8)
        self.assertEqual(summary['alpha_iou_frames'],6)
        self.assertEqual(summary['raw_rgb_foreground_mae_frames'],6)
        self.assertEqual(summary['alpha_iou'],0.)
        false={**output,'alpha':torch.ones_like(output['alpha'])}
        self.assertEqual(_summary(_mask_rows(false,batch,current=True),[True]*8)['empty_false_npc_rate'],1.)

    def test_light_preview_reads_only_four_pinned_cases_and_validates_reference_panel_and_architecture(self):
        model=LearnedRGBFixture();reference=FrozenAtlasFixture();data=AdjacentPairs();panel=list(range(8))
        report,_=evaluate(model,reference,data,panel,step=0,batch_size=4)
        model.calls=[];reference.calls=[];sampler=data.sampler_state()
        preview=preview_neural_enemy_restoration(model,reference,data,panel,step=15,baseline_report=report,batch_size=3)
        self.assertEqual(sum(model.calls),4);self.assertEqual(sum(reference.calls),4)
        self.assertTrue(preview['metadata']['preview_only']);self.assertEqual(preview['metadata']['reference_validation_step'],0)
        self.assertEqual(sampler,data.sampler_state())
        with tempfile.TemporaryDirectory()as folder:
            path=Path(folder)/'preview.png';meta=save_neural_enemy_restoration_preview(preview,path)
            with Image.open(path)as image:self.assertEqual(image.size,(1024,1120))
            self.assertEqual(meta['every'],15);self.assertTrue(path.with_suffix('.json').exists())
        for field in('panel_fingerprint','frozen_reference_model_sha256','model_config_sha256'):
            wrong=deepcopy(report);wrong[field]='other'
            with self.assertRaisesRegex(ValueError,'frozen reference architecture or held-out panel'):
                preview_neural_enemy_restoration(model,reference,data,panel,step=16,baseline_report=wrong)
        future=deepcopy(report);future['step']=20
        with self.assertRaisesRegex(ValueError,'future step'):
            preview_neural_enemy_restoration(model,reference,data,panel,step=16,baseline_report=future)
        data.rows[0]['split']='train'
        with self.assertRaisesRegex(ValueError,'TRAIN'):evaluate(model,reference,data,panel,step=17)
        data=AdjacentPairs();data.split='train'
        with self.assertRaisesRegex(ValueError,'held-out VAL'):evaluate(model,reference,data,panel,step=18)

    def test_actual_restoration_model_and_objective_validate_without_atlas_in_current_inference(self):
        from turkeyworld.neural_enemy_restoration_sr import NeuralEnemyRestorationSR
        from turkeyworld.neural_enemy_restoration_objectives import LOSS_WEIGHTS
        model=NeuralEnemyRestorationSR(channels=8,blocks=1,semantic_channels=4,semantic_blocks=1,
            restoration_channels=8)
        reference=FrozenAtlasFixture();data=AdjacentPairs()
        flags=[p.requires_grad for p in model.parameters()]
        report,_=evaluate_neural_enemy_restoration(model,reference,data,list(range(8)),step=0,batch_size=4)
        self.assertEqual(set(report['validation_objective']['components']),set(LOSS_WEIGHTS))
        self.assertGreater(report['validation_objective']['loss'],0.)
        self.assertFalse(report['new_model_uses_fixed_atlas'])
        self.assertEqual(flags,[p.requires_grad for p in model.parameters()])
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_foreground_revision_reports_and_previews_carry_the_actual_protocol_and_cannot_reuse_v1_baseline(self):
        model=LearnedRGBFixture();reference=FrozenAtlasFixture();data=AdjacentPairs();panel=list(range(8))
        def revised_objective(output,batch,*,protocol):
            self.assertEqual(protocol,FOREGROUND_PROTOCOL)
            return actual_objective(output,batch)
        objective=partial(revised_objective,protocol=FOREGROUND_PROTOCOL)
        weights={'pixel':1.,'alpha':.2}
        report,preview=evaluate_neural_enemy_restoration(model,reference,data,panel,step=2308,
            objective_protocol=FOREGROUND_PROTOCOL,objective_fn=objective,objective_weights=weights)
        self.assertEqual(PROTOCOL,'neural_enemy_restoration_sr_v1')
        self.assertEqual(report['objective_protocol'],FOREGROUND_PROTOCOL)
        self.assertEqual(report['validation_objective']['protocol'],FOREGROUND_PROTOCOL)
        self.assertEqual(preview['metadata']['objective_protocol'],FOREGROUND_PROTOCOL)
        light=preview_neural_enemy_restoration(model,reference,data,panel,step=2323,
            baseline_report=report,objective_protocol=FOREGROUND_PROTOCOL)
        self.assertEqual(light['metadata']['objective_protocol'],FOREGROUND_PROTOCOL)
        self.assertEqual(light['metadata']['reference_validation_step'],2308)
        with tempfile.TemporaryDirectory()as folder:
            saved=save_neural_enemy_restoration_preview(light,Path(folder)/'preview.png')
            self.assertEqual(saved['objective_protocol'],FOREGROUND_PROTOCOL)
        old,_=evaluate(model,reference,data,panel,step=2308)
        with self.assertRaisesRegex(ValueError,'objective protocol'):
            evaluate_neural_enemy_restoration(model,reference,data,panel,step=2309,
                baseline_report=old,objective_protocol=FOREGROUND_PROTOCOL,objective_fn=objective,objective_weights=weights)
        with self.assertRaisesRegex(ValueError,'objective protocol'):
            preview_neural_enemy_restoration(model,reference,data,panel,step=2323,
                baseline_report=old,objective_protocol=FOREGROUND_PROTOCOL)
        with self.assertRaisesRegex(ValueError,'objective protocol'):
            preview_neural_enemy_restoration(model,reference,data,panel,step=2323,baseline_report=report)
        for kwargs in ({},{'objective_fn':objective},{'objective_weights':weights}):
            with self.assertRaisesRegex(ValueError,'explicit objective_fn and objective_weights'):
                evaluate_neural_enemy_restoration(model,reference,data,panel,step=0,
                    objective_protocol=FOREGROUND_PROTOCOL,**kwargs)
        def old_tagged_objective(output,batch):
            value,metrics=actual_objective(output,batch)
            return value,{**metrics,'objective_protocol':PROTOCOL,'loss_protocol':PROTOCOL}
        with self.assertRaisesRegex(ValueError,'measured loss objective protocol'):
            evaluate_neural_enemy_restoration(model,reference,data,panel,step=2308,
                objective_protocol=FOREGROUND_PROTOCOL,objective_fn=old_tagged_objective,objective_weights=weights)
        with self.assertRaisesRegex(ValueError,'Unknown.*objective protocol'):
            evaluate_neural_enemy_restoration(model,reference,data,panel,step=0,objective_protocol='unknown_v3')
        self.assertEqual(report['empty_topk_pixels'],64)
        self.assertIn('not_a_whole_enemy_detection',report['empty_localization_scope'])

    def test_localized_empty_blob_reports_max_top64_and_dense_pixels_despite_small_mean_opacity(self):
        data=AdjacentPairs();batch=data.batch(2,indices=[6,7])
        batch['hr_source']=torch.zeros_like(batch['hr_source']);batch['hr_background']=torch.zeros_like(batch['hr_background'])
        image=torch.zeros(2,3,128,128);image[0,:,20:28,30:38]=1
        alpha=torch.zeros(2,1,128,128);alpha[0,:,20:28,30:38]=1
        output=dict(image=image,background=torch.zeros_like(image),enemy_rgb=image,alpha=alpha,
            alpha_logits=torch.logit(alpha.clamp(.001,.999)))
        rows=_mask_rows(output,batch,current=True);row=rows[0]
        self.assertEqual(EMPTY_TOPK_PIXELS,64)
        self.assertLess(row['empty_alpha_leakage'],.004)
        self.assertEqual(row['empty_false_npc_rate'],1.)
        self.assertEqual(row['empty_soft_alpha_any_rate'],1.)
        self.assertEqual(row['empty_dense_false_npc_rate'],1.)
        self.assertEqual(row['empty_dense_alpha_pixels'],64)
        self.assertEqual(row['empty_dense_alpha_fraction'],64/(128*128))
        for key in ('empty_alpha_max','empty_alpha_top64_mean','empty_paint_max_mae','empty_paint_top64_mae'):
            self.assertEqual(row[key],1.)
        summary=_summary(rows,[True,True])
        self.assertEqual(summary['empty_target_frames'],2);self.assertEqual(summary['empty_alpha_frames'],2)
        self.assertEqual(summary['empty_dense_false_npc_count'],1)
        self.assertEqual(summary['empty_dense_alpha_pixels_total'],64)
        self.assertEqual(summary['empty_dense_alpha_pixels'],32.)
        self.assertEqual(summary['empty_alpha_top64_mean'],.5)
        self.assertEqual(summary['empty_paint_top64_mae'],.5)
        self.assertIsNone(summary['alpha_iou']);self.assertEqual(summary['alpha_iou_frames'],0)
        self.assertEqual(summary['empty_alpha_top64_mean_frames'],2)

    def test_soft_empty_opacity_dense_blobs_and_unavailable_background_have_distinct_support(self):
        data=AdjacentPairs();batch=data.batch(1,indices=[6]);image=torch.zeros(1,3,128,128)
        alpha=torch.zeros(1,1,128,128);alpha[0,0,8,8]=.1
        output=dict(image=image,enemy_rgb=image,alpha=alpha,alpha_logits=torch.logit(alpha.clamp(.001,.999)))
        row=_mask_rows(output,batch,current=True)[0]
        self.assertEqual(row['empty_false_npc_rate'],1.)
        self.assertEqual(row['empty_dense_false_npc_rate'],0.)
        self.assertEqual(row['empty_dense_alpha_pixels'],0)
        self.assertAlmostEqual(row['empty_alpha_max'],.1)
        self.assertAlmostEqual(row['empty_alpha_top64_mean'],.1/64)
        summary=_summary([row],[True])
        self.assertEqual(summary['empty_paint_top64_mae_frames'],0)
        self.assertIsNone(summary['empty_paint_top64_mae'])
        visible=data.batch(1,indices=[0]);positive=_mask_rows(output,visible,current=True)[0]
        only_positive=_summary([positive],[True])
        self.assertEqual(only_positive['empty_target_frames'],0)
        self.assertIsNone(only_positive['empty_dense_false_npc_rate'])
        self.assertIsNone(only_positive['empty_alpha_max'])
        self.assertEqual(only_positive['empty_alpha_top64_mean_frames'],0)

    def test_detail_panel_keeps_all_deaths_attack_and_only_caps_hurt_without_duplicates_or_train(self):
        data=Inventory120();receipt=build_detail_animation_panel(data,seed=173)
        self.assertEqual(receipt['frames'],120)
        self.assertEqual(len(set(receipt['indices'])),120)
        self.assertEqual(receipt['support']['selected_visible'],dict(idle=0,walk_first=0,walk_second=0,
            attack=28,hurt=32,death_start=20,death_late=40))
        self.assertIn('all_visible_deaths_and_attack',receipt['selection'])
        self.assertFalse(receipt['training_uses_panel']);self.assertTrue(receipt['source_held_out'])
        self.assertEqual(receipt,build_detail_animation_panel(data,seed=173))
        changed=build_detail_animation_panel(data,seed=174)
        self.assertNotEqual(receipt['selected_label_fingerprint'],changed['selected_label_fingerprint'])
        self.assertNotEqual(receipt['fingerprint'],changed['fingerprint'])
        data.rows[0]['split']='train'
        with self.assertRaisesRegex(ValueError,'never TRAIN'):build_detail_animation_panel(data)

    def test_detail_revision_matches_background_from_same_forward_and_keeps_primary_panel_objective(self):
        model=MatchedBackgroundFixture();reference=FrozenAtlasFixture();data=detail_data();panel=list(range(8))
        receipt=build_detail_animation_panel(data)
        old,_=evaluate(model,reference,data,panel,step=5308)
        model.calls=[];reference.calls=[];sampler=data.sampler_state();rng=torch.get_rng_state().clone()
        modes=[[module.training for module in net.modules()]for net in(model,reference)]
        report,preview=evaluate_neural_enemy_restoration(model,reference,data,panel,step=5308,
            objective_protocol=DETAIL_PROTOCOL,objective_fn=actual_objective,
            objective_weights={'pixel':1.,'alpha':.2},detail_panel=receipt,batch_size=3)
        self.assertEqual(report['frames'],8);self.assertEqual(report['panel_fingerprint'],old['panel_fingerprint'])
        self.assertAlmostEqual(report['validation_objective']['loss'],old['validation_objective']['loss'],places=7)
        self.assertEqual(set(report['models']),{'current','frozen_background','frozen_atlas_parent'})
        background=report['models']['frozen_background']['summary']
        self.assertGreater(background['foreground_mae'],report['models']['current']['summary']['foreground_mae'])
        self.assertEqual(background['alpha_frames'],0);self.assertIsNone(background['alpha_iou'])
        self.assertIsNone(background['empty_false_npc_rate'])
        self.assertEqual(report['background_ablation']['additional_model_forwards'],0)
        detail=report['detail_validation']
        self.assertEqual(detail['fingerprint'],receipt['fingerprint']);self.assertEqual(detail['panel_fingerprint'],receipt['fingerprint'])
        self.assertEqual(detail['indices'],receipt['indices']);self.assertFalse(detail['training_uses_panel'])
        self.assertTrue(detail['primary_validation_score_unchanged'])
        self.assertEqual(sum(model.calls),8+receipt['frames']);self.assertEqual(sum(reference.calls),8+receipt['frames'])
        self.assertEqual(preview['metadata']['row_count'],5)
        self.assertEqual(preview['metadata']['detail_animation_panel_fingerprint'],receipt['fingerprint'])
        self.assertIn('frozen_background',preview['cases']['near']['predictions'])
        model.calls=[];reference.calls=[]
        light=preview_neural_enemy_restoration(model,reference,data,panel,step=5323,baseline_report=report,
            objective_protocol=DETAIL_PROTOCOL,detail_panel=receipt)
        self.assertEqual(sum(model.calls),4);self.assertEqual(sum(reference.calls),4)
        with tempfile.TemporaryDirectory()as folder:
            path=Path(folder)/'preview.png';saved=save_neural_enemy_restoration_preview(light,path)
            with Image.open(path)as image:self.assertEqual(image.size,(1024,1400))
            self.assertEqual(saved['row_count'],5)
        self.assertEqual(sampler,data.sampler_state());self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(modes,[[module.training for module in net.modules()]for net in(model,reference)])
        json.dumps(report,allow_nan=False)

    def test_detail_revision_requires_pinned_actual_val_receipt_and_rejects_prior_objective_or_changed_labels(self):
        model=MatchedBackgroundFixture();reference=FrozenAtlasFixture();data=detail_data();panel=list(range(8))
        receipt=build_detail_animation_panel(data)
        kwargs=dict(objective_protocol=DETAIL_PROTOCOL,objective_fn=actual_objective,
            objective_weights={'pixel':1.,'alpha':.2})
        with self.assertRaisesRegex(ValueError,'explicit pinned unique VAL detail_panel'):
            evaluate_neural_enemy_restoration(model,reference,data,panel,step=5308,**kwargs)
        duplicate=deepcopy(receipt);duplicate['indices'].append(duplicate['indices'][0])
        with self.assertRaisesRegex(ValueError,'unique VAL detail_panel'):
            evaluate_neural_enemy_restoration(model,reference,data,panel,step=5308,detail_panel=duplicate,**kwargs)
        wrong=deepcopy(receipt);wrong['support']['selected_visible']['death_start']+=1
        with self.assertRaisesRegex(ValueError,'actual VAL Source identity or phase labels'):
            evaluate_neural_enemy_restoration(model,reference,data,panel,step=5308,detail_panel=wrong,**kwargs)
        old,_=evaluate(model,reference,data,panel,step=5308)
        with self.assertRaisesRegex(ValueError,'objective protocol'):
            evaluate_neural_enemy_restoration(model,reference,data,panel,step=5308,detail_panel=receipt,
                baseline_report=old,**kwargs)
        report,_=evaluate_neural_enemy_restoration(model,reference,data,panel,step=5308,detail_panel=receipt,**kwargs)
        wrong=deepcopy(report);wrong['detail_validation']['panel_fingerprint']='other'
        with self.assertRaisesRegex(ValueError,'pinned detail panel'):
            preview_neural_enemy_restoration(model,reference,data,panel,step=5323,detail_panel=receipt,
                objective_protocol=DETAIL_PROTOCOL,baseline_report=wrong)
        data.state[1,15]=0
        with self.assertRaisesRegex(ValueError,'actual VAL Source identity or phase labels'):
            evaluate_neural_enemy_restoration(model,reference,data,panel,step=5309,detail_panel=receipt,**kwargs)


if __name__=='__main__':unittest.main()
