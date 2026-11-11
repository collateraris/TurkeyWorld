import copy
import unittest
from unittest.mock import patch

import torch

from turkeyworld.renderer_preservation_validation import (CRITICAL_COHORTS,base_preservation_assessment,
    validate_base_anchor_identity)


def report():
    row={'frames':18,'mae_mean':.1,'ssim_mean':.9,'foreground_frames':18,
         'background_frames':18,'foreground_mae':.1,'background_mae':.1}
    return {'preservation_cohorts':{name:{'base':dict(row),'rf1':dict(row)} for name in CRITICAL_COHORTS}}


class RendererReleaseGuardTests(unittest.TestCase):
    def test_explicit_codec_revision_cannot_reset_original_parent_or_change_source_panels(self):
        source={'world_sha256':'world','panel_fingerprint':'panel','hp0_panel_fingerprint':'hp0',
            'train':{'fingerprint':'train'},'val':{'fingerprint':'val'},'renderer_sha256':'old','codec_sha256':'old'}
        changed={**source,'renderer_sha256':'new','codec_sha256':'new'}
        validate_base_anchor_identity(source,changed)
        for key in ('world_sha256','panel_fingerprint','hp0_panel_fingerprint'):
            with self.assertRaises(ValueError):validate_base_anchor_identity(source,{**changed,key:'different'})
        with self.assertRaises(ValueError):validate_base_anchor_identity(source,{**changed,'val':{'fingerprint':'changed'}})
    def test_current_base_cannot_hide_regression_from_frozen_parent(self):
        parent=report();current=copy.deepcopy(parent)
        self.assertTrue(base_preservation_assessment(current,parent)['passed'])
        current['preservation_cohorts']['hp0']['base'].update(mae_mean=.2,foreground_mae=.2,background_mae=.2)
        current['preservation_cohorts']['hp0']['rf1'].update(mae_mean=.19,foreground_mae=.19,background_mae=.19)
        result=base_preservation_assessment(current,parent)
        self.assertTrue(result['rf_not_worse_than_current_base']['passed'])
        self.assertFalse(result['base_not_worse_than_frozen_parent']['passed'])
        self.assertFalse(result['passed'])

    def test_foreground_background_and_cohorts_cannot_compensate(self):
        parent=report();current=copy.deepcopy(parent)
        current['preservation_cohorts']['alive']['rf1'].update(mae_mean=.09,foreground_mae=.2,background_mae=0)
        result=base_preservation_assessment(current,parent)
        self.assertFalse(result['passed'])
        self.assertIn('alive:foreground_mae',result['rf_not_worse_than_current_base']['failures'])
        current=report();current['preservation_cohorts']['large']['rf1']['ssim_mean']=.8
        self.assertFalse(base_preservation_assessment(current,parent)['passed'])

    def test_missing_reference_or_required_cohort_never_qualifies_and_true_zeros_are_valid(self):
        current=report()
        self.assertFalse(base_preservation_assessment(current)['passed'])
        parent=copy.deepcopy(current);del current['preservation_cohorts']['hp0']
        self.assertFalse(base_preservation_assessment(current,parent)['passed'])
        current=report()
        for values in current['preservation_cohorts'].values():
            for row in values.values():row.update(mae_mean=0,foreground_mae=0,background_mae=0)
        self.assertTrue(base_preservation_assessment(current,current)['passed'])
        current['preservation_cohorts']['hp0']['rf1']['mae_mean']=float('nan')
        self.assertFalse(base_preservation_assessment(current,parent)['passed'])

    def test_validation_reuses_rf_pass_and_source_encoding_for_common_objective(self):
        from test_renderer_enemy_objectives import tiny_model,batch
        from turkeyworld.renderer_enemy_validation import evaluate_enemy_renderer
        torch.set_num_threads(2)
        model=tiny_model();values=batch((False,True,False,True))
        class Data:
            split='val'
            rows=[{'category':name} for name in ['large','corpse','empty_navigation','partial']]
            def batch(self,n,indices):return {key:value[indices] for key,value in values.items()}
            def panel_fingerprint(self,panel):return 'same_fixed_panel'
        data=Data()
        with (patch.object(model.flow_renderer,'latent_sample',wraps=model.flow_renderer.latent_sample) as sample,
                patch.object(model.flow_renderer.codec,'encode',wraps=model.flow_renderer.codec.encode) as encode):
            report,arrays=evaluate_enemy_renderer(model,data,[0,1,2,3],batch_size=2)
        self.assertEqual(sample.call_count,2)
        self.assertEqual(encode.call_count,4)  # Existing Base input + Source reference per batch.
        common=report['common_objective']
        self.assertEqual(common['extra_rf_forward_calls'],0)
        self.assertAlmostEqual(common['original_latent01_objective'],common['rgb_objective']+.1*common['source_latent_mse'])
        self.assertEqual(report['preservation_cohorts']['overall']['base']['frames'],4)
        self.assertFalse(report['base_preservation']['parent_reference_available'])
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__=='__main__':unittest.main()
