"""Explicit completed-checkpoint continuation into a new sampling protocol."""
import copy
import math

import torch

from .renderer_preservation_validation import validate_base_anchor_identity


def _identical(left,right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left.cpu(),right.detach().cpu())
    if isinstance(left,dict):
        return isinstance(right,dict) and left.keys()==right.keys() and all(_identical(value,right[key])for key,value in left.items())
    if isinstance(left,(list,tuple)):
        return type(left)==type(right) and len(left)==len(right) and all(_identical(a,b)for a,b in zip(left,right))
    return left==right


def restore_distance_parent(saved, model, optimizer, data, identity, configuration, frozen):
    """Preserve all trainable weights, Adam, private RNG and historical clocks.

    New phase fields are deliberately whitelisted; changing the underlying
    Source/codec/base/RF identity or the old objective is rejected.
    """
    if saved.get('frozen_fingerprint') != frozen:
        raise ValueError('Distance parent frozen renderer/world identity changed')
    parent_identity=copy.deepcopy(identity)
    parent_identity.pop('distance_sampling',None)
    parent_identity.pop('distance_panel_fingerprint',None)
    if saved.get('identity') != parent_identity:
        raise ValueError('Distance parent Source/renderer/panel identities changed')
    permitted={'run_dir','steps','batch','validation_batch','validate_every','preview_every','save_every','threads',
        'distance_balance','distance_panel','object_crop_weight','source_training_checkpoint'}
    old={k:v for k,v in saved['configuration'].items() if k not in permitted}
    current={k:v for k,v in configuration.items() if k not in permitted}
    if old != current:
        changed=sorted(k for k in old.keys()|current.keys() if old.get(k)!=current.get(k))
        raise ValueError('Distance continuation altered inherited objective/settings: '+','.join(changed))
    step=saved.get('step')
    if not isinstance(step,int) or isinstance(step,bool) or step<1 or configuration['steps']<=step:
        raise ValueError('Distance phase must extend a completed positive optimizer boundary')
    model.base_student.load_state_dict(saved['base'],strict=True)
    model.flow_renderer.load_state_dict(saved['flow'],strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    if not _identical(saved['optimizer'],optimizer.state_dict()):
        raise AssertionError('Distance continuation changed Adam moments/options')
    data.adopt_parent_sampler(saved['sampler'])
    for expected,actual in ((saved['base'],model.base_student.state_dict()),(saved['flow'],model.flow_renderer.state_dict())):
        if any(not torch.equal(value.cpu(),actual[key].detach().cpu()) for key,value in expected.items()):
            raise AssertionError('Distance continuation changed a parent parameter/buffer')
    return {'start_step':step,'parent_draws':data.parent_draws,'all_base_flow_codec_tensors_bitexact':True,
        'adam_preserved':True,'rng_restoration_required':True,'parent_best':saved.get('best'),
        'parent_mix_counts':dict(saved['sampler'].get('mix_counts',{})),
        'parent_case_counts':dict(saved['sampler'].get('case_counts',{})),
        'parent_streak':saved.get('streak',0),'parent_enemy_streak':saved.get('enemy_streak',0),
        'source_identity_preserved':True,'automatic_active_promotion':False}


def distance_candidate_assessment(report, baseline, *, min_improvement=.0005, mae_tolerance=.002):
    failures=[]
    if not report.get('panel_fingerprint') or report.get('panel_fingerprint')!=baseline.get('panel_fingerprint'):
        failures.append('distance_panel_identity_changed')
    for name in ('base','rf1'):
        for band in ('near','mid','far'):
            now=report.get(name,{}).get('categories',{}).get(band,{})
            old=baseline.get(name,{}).get('categories',{}).get(band,{})
            if now.get('frames')!=64 or old.get('frames')!=64:
                failures.append(name+'_'+band+'_missing64');continue
            a,b=now.get('foreground_mae'),old.get('foreground_mae')
            whole,old_whole=now.get('mae_mean'),old.get('mae_mean')
            if not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
                    for x in (a,b,whole,old_whole)):
                failures.append(name+'_'+band+'_nonfinite');continue
            if a>=b-min_improvement:
                failures.append(name+'_'+band+'_foreground_not_improved')
            if whole>old_whole+mae_tolerance:
                failures.append(name+'_'+band+'_full_image_regressed')
    return {'passed':not failures,'failures':failures,'minimum_foreground_improvement':min_improvement,
        'full_image_mae_tolerance':mae_tolerance,'scope':'Fixed192 GT static held-out queries; no gameplay claim'}
