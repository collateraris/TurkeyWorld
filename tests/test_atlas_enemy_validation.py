"""Actual atlas/reference evaluation with labels-only geometry and state isolation."""
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from turkeyworld.atlas_enemy_sr import AtlasEnemyEDSRUpscaler
from turkeyworld.enemy_decoder_sr import EnemyDecoderEDSRUpscaler
from turkeyworld.atlas_enemy_validation import (evaluate_atlas_enemy_sr,preview_atlas_enemy_sr,
    save_atlas_enemy_preview,_atlas_metrics,_summary)
from turkeyworld.atlas_enemy_objectives import LOSS_WEIGHTS
from test_atlas_enemy_objectives import fixture
from test_enemy_decoder_validation import GuidedPairs
from test_sr_patchgan_validation import AdjacentPairs


def networks():
    settings=dict(channels=8,blocks=1,semantic_channels=4,semantic_blocks=1,
        enemy_decoder_channels=4,enemy_decoder_stable=True,enemy_decoder_visual_first=True,
        enemy_decoder_restoration=True,restoration_blend_scale=1.)
    reference=EnemyDecoderEDSRUpscaler(**settings)
    model=AtlasEnemyEDSRUpscaler(**settings)
    state=model.state_dict();state.update(reference.state_dict());model.load_state_dict(state,strict=True)
    return model,reference


class AtlasEnemyValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_actual_model_objective_reference_and_all_rng_modes_sampler_source_grads_preserved(self):
        torch.manual_seed(173);model,reference=networks();data=GuidedPairs()
        model.train();model.enemy_decoder_visual.eval();reference.train()
        modes=[[m.training for m in net.modules()]for net in(model,reference)]
        sampler=data.sampler_state();rng=torch.get_rng_state().clone();py=random.getstate();nr=np.random.get_state()
        weights=[deepcopy(net.state_dict())for net in(model,reference)]
        report,preview=evaluate_atlas_enemy_sr(model,reference,data,list(range(8)),step=100,batch_size=3)
        self.assertEqual(report['kind'],'atlas_enemy_hybrid_validation_v1')
        self.assertEqual(set(report['models']),{'atlas','learned_reference'})
        objective=report['validation_objective']
        self.assertEqual(objective['protocol'],'atlas_enemy_hybrid_sr_v1')
        self.assertEqual((objective['frames'],objective['input_kind']),(8,'npr_rgb'))
        self.assertEqual(set(objective['components']),set(LOSS_WEIGHTS))
        self.assertAlmostEqual(objective['loss'],sum(row['weighted']for row in objective['components'].values()),places=5)
        summary=report['models']['atlas']['summary']
        self.assertEqual(summary['phase_frames'],6)
        self.assertEqual(summary['full_bbox_frames'],6)
        self.assertEqual(summary['visibility_frames'],6)
        self.assertGreater(summary['alpha_frames'],0)
        self.assertIn('alpha_iou_mean',summary)
        self.assertEqual(summary['empty_frames'],2)
        self.assertEqual(summary['reconstruction_loss'],objective['loss'])
        self.assertNotIn('reconstruction_loss',report['models']['learned_reference']['summary'])
        self.assertIn('not_matched_training_control',report['reference_scope'])
        self.assertFalse(report['temporal']['available'])
        self.assertIsNone(report['temporal']['arms']['atlas']['delta_mae'])
        self.assertEqual(preview['metadata']['row_count'],4)
        self.assertEqual(set(preview['cases']),{'near','mid','far','empty'})
        self.assertEqual(sampler,data.sampler_state());self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(py,random.getstate());self.assertTrue(all(np.array_equal(a,b)for a,b in zip(nr,np.random.get_state())))
        self.assertEqual(modes,[[m.training for m in net.modules()]for net in(model,reference)])
        for i,net in enumerate((model,reference)):
            for key,value in weights[i].items():torch.testing.assert_close(value,net.state_dict()[key],rtol=0,atol=0)
            self.assertTrue(all(p.grad is None for p in net.parameters()))
        self.assertIsNone(data.hr.grad);self.assertIsNone(data.hr_mask.grad);self.assertIsNone(data.npr.grad)
        json.dumps(report,allow_nan=False)

    def test_full_unclipped_geometry_correct_soft_alpha_and_binary_visibility_metrics_are_exact(self):
        output,batch,renderer,geometry=fixture(partial=True)
        intrinsic=renderer(geometry['full_bbox'],geometry['phase'])
        alpha=intrinsic[:,3:4]*batch['mask_hr']
        renderer.calls.clear()
        output.update(atlas_composite_alpha_hr=alpha,atlas_effective_alpha_hr=alpha)
        model=type('FixedLabelAsset',(),{'render_label_atlas':staticmethod(renderer)})()
        rows=_atlas_metrics(model,output,batch)
        for row in rows:
            self.assertEqual(row['full_bbox_l1_pixels'],0.)
            self.assertEqual(row['full_bbox_iou'],1.)
            self.assertEqual(row['phase_accuracy'],1.)
            self.assertEqual(row['alpha_iou_mean'],1.)
            self.assertEqual(row['visibility_iou_mean'],1.)
        # Binary opacity is NOT the correct target for a translucent PNG.
        opaque=dict(output,atlas_composite_alpha_hr=batch['mask_hr'])
        self.assertEqual(_atlas_metrics(model,opaque,batch)[0]['alpha_iou_mean'],.25)
        self.assertTrue(all(not grad_enabled for _,_,grad_enabled in renderer.calls))

    def test_adjacent_source_temporal_support_is_preserved_and_train_rows_are_rejected(self):
        model,reference=networks();data=AdjacentPairs()
        report,_=evaluate_atlas_enemy_sr(model,reference,data,list(range(8)),step=0,batch_size=4)
        self.assertEqual(report['temporal']['pair_count'],4)
        self.assertEqual(set(report['temporal']['arms']),{'atlas','learned_reference'})
        self.assertEqual(report['temporal']['arms']['atlas']['pair_count'],4)
        self.assertEqual(report['temporal']['arms']['atlas']['foreground_pairs'],4)
        self.assertIsNotNone(report['temporal']['arms']['atlas']['delta_mae'])
        self.assertTrue(report['temporal']['panel_fingerprint'])
        data.rows[0]['split']='train'
        with self.assertRaisesRegex(ValueError,'TRAIN'):
            evaluate_atlas_enemy_sr(model,reference,data,list(range(8)),step=1)
        data=GuidedPairs();data.split='train'
        with self.assertRaisesRegex(ValueError,'held-out VAL'):
            evaluate_atlas_enemy_sr(model,reference,data,list(range(8)),step=1)

    def test_frozen_reference_pin_light_four_case_preview_and_PNG_with_no_full_validation(self):
        model,reference=networks();data=GuidedPairs();panel=list(range(8))
        initial,_=evaluate_atlas_enemy_sr(model,reference,data,panel,step=0,batch_size=4)
        sampler=data.sampler_state();rng=torch.get_rng_state().clone()
        with patch.object(model,'forward_with_aux',wraps=model.forward_with_aux)as atlas_calls, \
                patch.object(reference,'forward_with_aux',wraps=reference.forward_with_aux)as ref_calls:
            preview=preview_atlas_enemy_sr(model,reference,data,panel,step=15,baseline_report=initial)
            self.assertEqual(sum(len(call.args[0])for call in atlas_calls.call_args_list),4)
            self.assertEqual(sum(len(call.args[0])for call in ref_calls.call_args_list),4)
        self.assertEqual(sampler,data.sampler_state());self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(preview['metadata']['preview_only'])
        self.assertEqual(preview['metadata']['reference_validation_step'],0)
        with tempfile.TemporaryDirectory()as temporary:
            metadata=save_atlas_enemy_preview(preview,Path(temporary)/'preview.png')
            with Image.open(metadata['path'])as image:self.assertEqual(image.size,(1024,1120))
            self.assertEqual(metadata['kind'],'atlas_enemy_hybrid_preview_v1')
            self.assertFalse(metadata['source_input_allowed_for_sr'])
        with torch.no_grad():next(reference.parameters()).add_(.001)
        with self.assertRaisesRegex(ValueError,'reference or held-out'):
            preview_atlas_enemy_sr(model,reference,data,panel,step=30,baseline_report=initial)
        with self.assertRaisesRegex(ValueError,'reference or held-out'):
            evaluate_atlas_enemy_sr(model,reference,data,panel,step=100,baseline_report=initial)


if __name__=='__main__':unittest.main()
