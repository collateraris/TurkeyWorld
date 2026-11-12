"""Explicit v6 depth-balanced distant geometry added to the unchanged v5 loss."""
from __future__ import annotations

import copy
import math

import torch

from .renderer_wall_depth import DEPTH_WALL_PROTOCOL,depth_banded_wall_terms


FAR_WALL_PROTOCOL = {
    'name':'one_enemy_native_area_far_geometry_renderer_v6','version':1,
    'parent_protocol':'one_enemy_native_area_balanced_renderer_v5',
    'source_labels':copy.deepcopy(DEPTH_WALL_PROTOCOL),
    'loss_bands':['far_5_8','distant_8_12','thin_12_plus'],
    'weights_per_head':{'boundary_l1':.2,'wall_edge_l1':.1,'corner_l1':.1,
                        'thin_face_l1':.1,'thin_height_l1':.1},
    'reduction':'Per-image present pixels, mean present images, then mean present disjoint far bands per term',
    'absent_regions':'Zero contribution; excluded from that term denominator, explicit support counts',
    'old_v5_objective_unchanged':True,'source_rgb_or_labels_condition_renderer':False,
    'new_parameters':0,'codec_and_current_feature_world_frozen':True,
    'auxiliary':{'batch_size':4,'far_only_weight':.25,'candidate_count':2048,'seed':2718,
                 'main_quota_sampler_unchanged':True,'private_auxiliary_rng_and_draw_clock':True},
}


def far_wall_protocol():
    return copy.deepcopy(FAR_WALL_PROTOCOL)


def far_wall_loss(prediction,target,labels):
    """Independent means stop near walls/empty strata diluting tiny far faces."""
    terms=depth_banded_wall_terms(prediction,target,labels)['bands']
    result={};total=prediction.sum()*0
    for field,weight in FAR_WALL_PROTOCOL['weights_per_head'].items():
        support=field.removesuffix('_l1')+'_images'
        active=[terms[band][field]for band in FAR_WALL_PROTOCOL['loss_bands']if terms[band][support]>0]
        value=torch.stack(active).mean()if active else prediction.sum()*0
        total=total+weight*value
        result[field]=float(value.detach());result[field+'_present_bands']=len(active)
    result['band_support']={band:{key:int(value)for key,value in terms[band].items()
        if key.endswith(('_images','_pixels'))}for band in FAR_WALL_PROTOCOL['loss_bands']}
    result['weighted_loss']=float(total.detach())
    return total,result


def far_wall_preservation_assessment(report,baseline,*,tolerance=.002):
    """Check only measured support; absent strata remain explicitly unmeasured."""
    failures=[];measured=[];unmeasured=[]
    for head in ('base','rf1'):
        for band in FAR_WALL_PROTOCOL['loss_bands']:
            current=report.get(head,{}).get('bands',{}).get(band,{})
            old=baseline.get(head,{}).get('bands',{}).get(band,{})
            for field in FAR_WALL_PROTOCOL['weights_per_head']:
                label=head+'_'+band+'_'+field
                support=field.removesuffix('_l1')+'_images'
                count,previous=current.get(support),old.get(support)
                if type(count)is not int or type(previous)is not int or count<0 or previous<0 or count!=previous:
                    failures.append(label+'_support_missing_or_changed');continue
                if not count:
                    unmeasured.append(label);continue
                a,b=current.get(field),old.get(field)
                if any(not isinstance(v,(int,float))or isinstance(v,bool)or not math.isfinite(v)for v in(a,b)):
                    failures.append(label+'_nonfinite');continue
                measured.append(label)
                if a>b+tolerance:failures.append(label+'_regressed')
    if not measured:failures.append('no_measured_far_wall_regions')
    return {'passed':not failures and bool(measured),'failure_reasons':failures,
            'measured_regions':measured,'unmeasured_regions':unmeasured,'absolute_mae_tolerance':tolerance,
            'scope':'Same Source far-depth strata and support; absent strata are not quality claims'}


def far_wall_report_score(report):
    score=0.
    for head in ('base','rf1'):
        for field,weight in FAR_WALL_PROTOCOL['weights_per_head'].items():
            support=field.removesuffix('_l1')+'_images'
            values=[report[head]['bands'][band][field]for band in FAR_WALL_PROTOCOL['loss_bands']
                    if report[head]['bands'][band][support]>0]
            if values:score+=weight*sum(values)/len(values)
    return score
