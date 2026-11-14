"""Physical image/mask selection for the detail phase, separate from loss."""
from copy import deepcopy
import math


POLICY = dict(kind='enemy_detail_visual_selection_v1',
    cohorts=['near', 'mid', 'far'], score_terms=['raw_rgb_foreground_ssim', 'roi_crop_ssim_mean', 'alpha_pixel_f1'],
    source_empty_dense_frames_may_increase=False, source_empty_dense_pixels_may_increase=False,
    source_overall_raw_RGB_SSIM_may_decrease=False, source_overall_ROI_SSIM_may_decrease=False,
    source_mask_recall_may_decrease=False, precision_tolerance=.03,
    gameplay_qualification=False, automatic_deployment=False)


def _number(mapping, key):
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError('Detail checkpoint selection needs actual finite ' + key)
    return float(value)


def detail_visual_candidate(report, baseline):
    """Balance distances and reject apparent gains obtained by hiding enemies."""
    if (report.get('objective_protocol') != 'neural_enemy_restoration_detail_v3'
            or baseline.get('objective_protocol') != report['objective_protocol']
            or any(report.get(key) != baseline.get(key) for key in (
                'panel_fingerprint', 'data_identity_sha256', 'frozen_reference_model_sha256'))
            or type(report.get('step')) is not int or report['step'] < baseline['step']):
        raise ValueError('Visual selection requires a matched detail-phase held-out baseline')
    current = report['models']['current']['cohorts']
    source = baseline['models']['current']['cohorts']
    checks = {}
    for key in ('empty_dense_false_npc_count', 'empty_dense_alpha_pixels_total'):
        checks[key] = _number(current['empty'], key) <= _number(source['empty'], key)
    for key in ('raw_rgb_foreground_ssim', 'roi_crop_ssim_mean', 'alpha_pixel_recall'):
        checks[key] = _number(current['overall'], key) >= _number(source['overall'], key)
    checks['alpha_pixel_precision'] = _number(current['overall'], 'alpha_pixel_precision') >= max(
        0., _number(source['overall'], 'alpha_pixel_precision') - POLICY['precision_tolerance'])
    values = []
    for cohort in POLICY['cohorts']:
        if _number(current[cohort], 'frames') <= 0:
            raise ValueError('Detail selection cannot invent an absent distance cohort')
        values.extend(_number(current[cohort], key) for key in POLICY['score_terms'])
    return dict(policy=deepcopy(POLICY), step=report['step'], score=sum(values)/len(values),
        eligible=all(checks.values()), guards=checks,
        baseline_step=baseline['step'], panel_fingerprint=report['panel_fingerprint'],
        objective_protocol=report['objective_protocol'], automatic_deployment=False,
        gameplay_qualified=False, selection_scope='primary_256_held_out_frames_not_training_loss')

