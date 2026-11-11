"""Loss-independent tiny-enemy progress and forgetting checks."""
import math
import numpy as np
import torch

from .renderer_enemy_validation import _summary


def small_enemy_cohorts(prediction, target, mask, states, *, native_mask=None):
    pixels=(mask if native_mask is None else native_mask).flatten(1).sum(-1)
    alive=(states[:,13]>=.5)&(states[:,12]>0)
    player_alive=states[:,4]>0
    distance=torch.linalg.vector_norm((states[:,8:10]-states[:,:2])*16,dim=-1)
    selected={'overall':torch.ones(len(states),dtype=torch.bool),
        'tiny':alive&player_alive&(pixels>=1)&(pixels<=16),
        'near':alive&player_alive&(pixels>16)&(distance<2),
        'corpse':~alive&(pixels>0),'hp0':states[:,4]<=0,
        'navigation':~alive&(pixels==0)&(states[:,4]>0)}
    return {key:_summary(prediction,target,mask,value.cpu())for key,value in selected.items()}


def relative_small_enemy_gate(report, baseline, *, predictions, compatibility=(),
                             improvement=.0005, guard_mae=.003, guard_ssim=.02):
    """Relative training progress; never an absolute Source/gameplay pass."""
    failures=[]
    def finite(value):return isinstance(value,(int,float))and not isinstance(value,bool)and math.isfinite(value)
    if (not report.get('panel_fingerprint') or report.get('panel_fingerprint')!=baseline.get('panel_fingerprint')
            or report.get('source_held_out')is not True or baseline.get('source_held_out')is not True):
        failures.append('heldout_panel_identity')
    for name in predictions:
        current=report.get('small_enemy_cohorts',{}).get(name,{})
        old=baseline.get('small_enemy_cohorts',{}).get(name,{})
        tiny,prior=current.get('tiny',{}),old.get('tiny',{})
        a,b=tiny.get('foreground_mae'),prior.get('foreground_mae')
        if (not tiny.get('frames') or tiny.get('frames')!=prior.get('frames')
                or not finite(a)or not finite(b)or a>=b-improvement):
            failures.append(name+'_tiny_foreground_not_improved')
        for cohort in ('overall','navigation','near','corpse','hp0'):
            now,before=current.get(cohort,{}),old.get(cohort,{})
            if not now.get('frames') or now.get('frames')!=before.get('frames'):
                failures.append(name+'_'+cohort+'_missing');continue
            mae,old_mae=now.get('mae_mean'),before.get('mae_mean')
            ssim,old_ssim=now.get('ssim_mean'),before.get('ssim_mean')
            if not all(finite(x)for x in (mae,old_mae,ssim,old_ssim)):
                failures.append(name+'_'+cohort+'_nonfinite')
            elif mae>old_mae+guard_mae or ssim<old_ssim-guard_ssim:
                failures.append(name+'_'+cohort+'_regression')
    for name in compatibility:
        current=report.get('small_enemy_cohorts',{}).get(name,{})
        old=baseline.get('small_enemy_cohorts',{}).get(name,{})
        for cohort,key in (('overall','mae_mean'),('tiny','foreground_mae')):
            a=current.get(cohort,{}).get(key);b=old.get(cohort,{}).get(key)
            if not finite(a)or not finite(b)or a>b+.005:
                failures.append(name+'_'+cohort+'_compatibility_regression')
    return {'passed':not failures,'failure_reasons':failures,'tiny_native_pixels':[1,16],
        'minimum_foreground_mae_improvement':improvement,
        'guard_mae_tolerance':guard_mae,'guard_ssim_tolerance':guard_ssim,
        'fixed_parent_rf_mae_tolerance':.005,'gameplay_qualified':False,
        'scope':'Relative tiny-Source reconstruction and forgetting checks; full Source criterion remains separate'}


def distance_preservation_gate(report,baseline):
    failures=[]
    if not report.get('panel_fingerprint')or report.get('panel_fingerprint')!=baseline.get('panel_fingerprint'):
        failures.append('historical_distance_panel_identity')
    for name in ('base','rf1'):
        for band in ('near','mid','far'):
            current=report.get(name,{}).get('categories',{}).get(band,{})
            old=baseline.get(name,{}).get('categories',{}).get(band,{})
            if current.get('frames')!=64 or old.get('frames')!=64:
                failures.append(name+'_'+band+'_missing64');continue
            for key,tolerance,lower in (('foreground_mae',.002,True),('mae_mean',.002,True),('ssim_mean',.01,False)):
                a,b=current.get(key),old.get(key)
                if not all(isinstance(x,(int,float))and not isinstance(x,bool)and math.isfinite(x)for x in (a,b)):
                    failures.append(name+'_'+band+'_'+key+'_nonfinite')
                elif (a>b+tolerance if lower else a<b-tolerance):failures.append(name+'_'+band+'_'+key+'_regression')
    return {'passed':not failures,'failure_reasons':failures,'scope':'Historical192 held-out Source distance preservation vs8000; no gameplay claim'}
