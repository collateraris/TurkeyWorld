from copy import deepcopy
import unittest

from turkeyworld.neural_enemy_detail_selection import detail_visual_candidate


def fixture():
    overall = dict(raw_rgb_foreground_ssim=.3, roi_crop_ssim_mean=.25,
        alpha_pixel_recall=.6, alpha_pixel_precision=.93)
    cohort = dict(frames=60, raw_rgb_foreground_ssim=.3,
        roi_crop_ssim_mean=.25, alpha_pixel_f1=.7)
    return dict(step=5308, objective_protocol='neural_enemy_restoration_detail_v3',
        panel_fingerprint='fixed', data_identity_sha256='data', frozen_reference_model_sha256='frozen',
        models={'current': {'cohorts': {'overall': overall,
            **{name: deepcopy(cohort) for name in ('near','mid','far')},
            'empty': {'empty_dense_false_npc_count':4, 'empty_dense_alpha_pixels_total':321}}}})


class DetailVisualSelectionTests(unittest.TestCase):
    def test_balanced_detail_improvement_is_eligible_independently_of_total_loss(self):
        baseline=fixture();current=deepcopy(baseline);current['step']+=100
        current['validation_objective']={'loss':99.}
        cohorts=current['models']['current']['cohorts']
        cohorts['overall'].update(raw_rgb_foreground_ssim=.35,roi_crop_ssim_mean=.3,alpha_pixel_recall=.7)
        cohorts['far']['raw_rgb_foreground_ssim']=.45
        result=detail_visual_candidate(current,baseline)
        self.assertTrue(result['eligible'])
        self.assertGreater(result['score'],detail_visual_candidate(baseline,baseline)['score'])
        self.assertFalse(result['gameplay_qualified'])

    def test_larger_background_SSIM_cannot_promote_hiding_enemy_or_returning_false_paint(self):
        baseline=fixture()
        for group,key,value in [('overall','raw_rgb_foreground_ssim',.29),
            ('overall','alpha_pixel_recall',.59),('overall','alpha_pixel_precision',.89),
            ('empty','empty_dense_false_npc_count',5),('empty','empty_dense_alpha_pixels_total',322)]:
            changed=deepcopy(baseline);changed['step']+=100
            c=changed['models']['current']['cohorts'];c['overall']['ssim_mean']=.99
            c[group][key]=value
            self.assertFalse(detail_visual_candidate(changed,baseline)['eligible'],key)

    def test_missing_distance_measurement_or_changed_panel_is_rejected(self):
        baseline=fixture()
        for mutation in ('panel','missing','nan'):
            changed=deepcopy(baseline)
            if mutation=='panel':changed['panel_fingerprint']='other'
            elif mutation=='missing':changed['models']['current']['cohorts']['far']['frames']=0
            else:changed['models']['current']['cohorts']['far']['raw_rgb_foreground_ssim']=float('nan')
            with self.assertRaises(ValueError):detail_visual_candidate(changed,baseline)


if __name__=='__main__':unittest.main()
