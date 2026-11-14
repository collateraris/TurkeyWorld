"""Held-out details versus the exact preferred frozen RGB background.

Both measured arms come from the same RGB-only forward. Source and its masks
are labels only; there is no atlas, semantic hint, or additional reference net.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from torch.nn import functional as F

from .enemy_animation import enemy_animation_index
from .neural_enemy_restoration_validation import(
    _validate_detail_panel,_cohorts,_summary,_mask_rows,ROLES)
from .renderer_student import _hash_tensors
from .rssm_preview_assets import preserve_preview_rng
from .semantic_sr_validation import _rows,_pixels
from .sr_neural_pairs_data import fingerprint
from .sr_patchgan_validation import build_temporal_panel,_temporal_summary
from .training_pipeline import atomic_json,now,replace_file
from .visual_quality import strict_render_precision


PROTOCOL='background_detail_sr_v1'
KIND='background_detail_sr_validation_v1'
PREVIEW_KIND='background_detail_sr_preview_v1'
ARMS=('current','preferred_frozen_background')
BACKGROUND_ABLATION=dict(kind='preferred_frozen_background_same_forward_v1',same_inputs=True,
    output='display_clamped_model_background',additional_model_forwards=0,
    source_targets_are_labels_only=True)
ROW_LABELS=('Оригинал Source128','Вход нейрорендера32',
    'Выбранная замороженная основа128','Основа + обучаемые детали128')


def _validate_inputs(model,data,panel,step,batch_size,detail_panel):
    if type(step)is not int or step<0 or type(batch_size)is not int or batch_size<1:
        raise ValueError('Background detail validation requires completed local step and positive batch')
    if(getattr(model,'config',{}).get('architecture')!='background_detail_sr'
            or getattr(data,'split',None)!='val' or not panel or any(type(i)is not int for i in panel)
            or len(set(panel))!=len(panel)or any(i<0 or i>=len(data.rows)for i in panel)):
        raise ValueError('Background detail validation needs its RGB architecture and unique held-out VAL rows')
    return _validate_detail_panel(data,detail_panel)


def _identity(model,data,panel):
    frozen={key:value for key,value in model.state_dict().items()if not key.startswith('detail_residual.')}
    if not frozen:raise ValueError('Background detail model has no pinned preferred frozen base')
    return dict(data_identity_sha256=fingerprint(data.identity),panel_fingerprint=data.panel_fingerprint(panel),
        model_config_sha256=fingerprint(model.config),preferred_background_sha256=_hash_tensors(frozen))


def _check_report(report,identity,detail):
    measured=report.get('detail_validation',{})
    if(report.get('kind')!=KIND or report.get('objective_protocol')!=PROTOCOL
            or report.get('background_ablation')!=BACKGROUND_ABLATION
            or any(report.get(key)!=value for key,value in identity.items())
            or type(report.get('step'))is not int or report['step']<0
            or any(measured.get(key)!=value for key,value in detail.items())
            or measured.get('panel_fingerprint')!=detail['fingerprint']
            or measured.get('objective_protocol')!=PROTOCOL
            or any(measured.get('models',{}).get(arm,{}).get('summary',{}).get('frames')!=detail['frames']for arm in ARMS)):
        raise ValueError('Background detail reference changed protocol, preferred base or held-out panels')


def _forward(model,low):
    output=model.forward_with_aux(low)
    if not isinstance(output,dict)or any(key not in output for key in('image','background','residual')):
        raise ValueError('Background detail RGB forward must return image, background and residual')
    for key in('image','background','residual'):
        value=output[key]
        if(value.shape!=(len(low),3,128,128)or not bool(torch.isfinite(value).all())):
            raise FloatingPointError('Invalid background detail native128 '+key)
    if any(key in output for key in('alpha','atlas_composite_alpha_hr')):
        raise ValueError('Background detail inference must not be an alpha/atlas composite')
    base=output['background'].detach().float().clamp(0,1)
    current={**output,'image':output['image'].float().clamp(0,1),'background':base}
    return current,dict(current=current,preferred_frozen_background={'image':base,'background':base})


def _extra_rows(output,batch):
    image=output['image'].float();base=output['background'].detach().float();target=batch['hr_source'].detach().float()
    mask=batch['mask_hr'].detach().float();bg=(1-F.max_pool2d(mask,5,1,2))
    change=(image-base).abs();lowfreq=F.avg_pool2d(image-base,8,8).abs().flatten(1).mean(1)
    # This is an image-derived Source edge diagnostic, not a material/semantic
    # wall classifier. Excluding the actual NPC mask keeps its support honest.
    errors=[]
    for dim in(-1,-2):
        true=target.diff(dim=dim);pred=image.diff(dim=dim)
        support=(true.abs().mean(1,keepdim=True)>.04).float()
        adjacent=torch.minimum(bg.narrow(dim,0,bg.shape[dim]-1),bg.narrow(dim,1,bg.shape[dim]-1))
        support*=adjacent
        errors.append(((pred-true).abs()*support).flatten(1).sum(1)/(3*support.flatten(1).sum(1)).clamp_min(1))
    rows=_mask_rows(output,batch,current=False)
    for i,row in enumerate(rows):
        row.update(low_frequency_change_mae=float(lowfreq[i]),
            residual_mean_abs=float(change[i].mean()),residual_max_abs=float(change[i].max()),
            source_background_edge_mae=float((errors[0][i]+errors[1][i])*.5))
    return rows


def _summaries(rows,metadata):
    cohorts=_cohorts(metadata)
    def summary(selected):
        value=_summary(rows,selected);chosen=[row for row,keep in zip(rows,selected)if keep]
        for key in('low_frequency_change_mae','residual_mean_abs','residual_max_abs','source_background_edge_mae'):
            values=[row[key]for row in chosen if row.get(key)is not None]
            value[key]=float(np.mean(values))if values else None;value[key+'_frames']=len(values)
        return value
    return dict(summary=summary([True]*len(rows)),cohorts={key:summary(selected)for key,selected in cohorts.items()})


def _metadata(batch,indices):
    states=batch['state'].detach().cpu();phases=enemy_animation_index(states);mask=batch['mask_hr']
    return[dict(row_index=index,alive=bool(states[i,12]>0 and states[i,13]>=.5),
        player_alive=bool(states[i,4]>0),hr_visible=bool(mask[i].any()),
        native_area=int(batch['mask_lr'][i].sum()),animation_label=int(phases[i]),
        distance_tiles=float(((states[i,8:10]-states[i,:2])*16).norm()))for i,index in enumerate(indices)]


def _preview(cases,report,step):
    return dict(cases=cases,metadata=dict(kind=PREVIEW_KIND,objective_protocol=PROTOCOL,step=step,row_count=4,
        row_labels=list(ROW_LABELS),input_kind='npr_rgb',source_held_out=True,world_rollout=False,
        inference_rgb_only=True,Source_or_RSSM_hint_as_input=False,new_model_uses_fixed_atlas=False,
        source_input_allowed_for_sr=False,source_targets_are_labels_only=True,
        background_ablation=deepcopy(BACKGROUND_ABLATION),
        **{key:report[key]for key in('data_identity_sha256','panel_fingerprint','model_config_sha256','model_sha256','preferred_background_sha256')},
        detail_animation_panel_fingerprint=report['detail_validation']['fingerprint'],
        detail_animation_panel_frames=report['detail_validation']['frames']))


@torch.no_grad()
def evaluate_background_detail_sr(model,data,panel,*,step,batch_size=8,baseline_report=None,
                                  objective_fn=None,objective_weights=None,detail_panel=None):
    panel=list(panel);detail=_validate_inputs(model,data,panel,step,batch_size,detail_panel)
    if objective_fn is None:
        from .background_detail_objectives import background_detail_loss,LOSS_WEIGHTS
        objective_fn=background_detail_loss;objective_weights=LOSS_WEIGHTS if objective_weights is None else objective_weights
    if not callable(objective_fn)or not isinstance(objective_weights,dict)or not objective_weights:
        raise ValueError('Background detail validation needs its actual objective and weights')
    identity=_identity(model,data,panel)
    if baseline_report is not None:
        _check_report(baseline_report,identity,detail)
        if baseline_report['step']>step:raise ValueError('Background detail reference is from a future local step')
    device=next(model.parameters()).device;sampler=deepcopy(data.sampler_state())
    modes=[(module,module.training)for module in model.modules()];weights_before=_hash_tensors(model.state_dict())
    temporal=build_temporal_panel(data,panel);needed={i for pair in temporal['pairs']for i in(pair['previous_index'],pair['current_index'])}
    rows={arm:[]for arm in ARMS};images={arm:{}for arm in ARMS};targets={};masks={};metadata=[];cases={}
    total_loss=0.;components={key:dict(raw=0.,weighted=0.,weight=weight)for key,weight in objective_weights.items()}
    detail_rows={arm:[]for arm in ARMS};detail_metadata=[]
    try:
        model.eval()
        with preserve_preview_rng(device,getattr(data,'rng',None)),strict_render_precision(),torch.inference_mode(),torch.autocast(device.type,enabled=False):
            for selected,is_detail in((panel,False),(detail['indices'],True)):
                for first in range(0,len(selected),batch_size):
                    ids=selected[first:first+batch_size];batch=data.batch(len(ids),indices=ids)
                    low=batch['lr_npr'].detach();target=batch['hr_source'].detach();mask=batch['mask_hr'].detach()
                    if low.shape!=(len(ids),3,32,32)or target.shape!=(len(ids),3,128,128)or mask.shape!=(len(ids),1,128,128):
                        raise ValueError('Background detail requires exact cached NPR32/Source128/masks')
                    current,outputs=_forward(model,low);local=_metadata(batch,ids)
                    (detail_metadata if is_detail else metadata).extend(local)
                    picked={}
                    if not is_detail:
                        for role,flags in _cohorts(local).items():
                            if role not in ROLES or role in cases:continue
                            i=next((i for i,yes in enumerate(flags)if yes),None)
                            if i is not None:picked[role]=i;cases[role]=dict(row_index=ids[i],source=_pixels(target[i]),npr=_pixels(low[i]),predictions={})
                        for i,index in enumerate(ids):
                            if index in needed:targets[index]=target[i].cpu().clone();masks[index]=mask[i].bool().cpu().clone()
                    for arm,output in outputs.items():
                        values=_rows(output['image'],target,mask,batch['hr_background'],batch['mask_lr'])
                        for row,extra in zip(values,_extra_rows(output,batch)):row.update(extra)
                        (detail_rows if is_detail else rows)[arm].extend(values)
                        if not is_detail:
                            for i,index in enumerate(ids):
                                if index in needed:images[arm][index]=output['image'][i].cpu().clone()
                            for role,i in picked.items():cases[role]['predictions'][arm]=_pixels(output['image'][i])
                    if not is_detail:
                        loss,metrics=objective_fn(current,batch)
                        if any(metrics.get(key,PROTOCOL)!=PROTOCOL for key in('objective_protocol','loss_protocol')):
                            raise ValueError('Background detail validation objective protocol differs')
                        total_loss+=float(loss)*len(ids)
                        for key,value in components.items():value['raw']+=float(metrics[key])*len(ids);value['weighted']+=float(metrics[key+'_weighted'])*len(ids)
    finally:
        data.restore_sampler(sampler)
        for module,training in modes:module.training=training
    if _hash_tensors(model.state_dict())!=weights_before:raise AssertionError('Background detail validation changed model/base weights')
    models={arm:_summaries(rows[arm],metadata)for arm in ARMS}
    models['current']['summary'].update(reconstruction_loss=total_loss/len(panel),reconstruction_frames=len(panel))
    temporal['arms']={arm:_temporal_summary(temporal['pairs'],images[arm],targets,masks)for arm in ARMS}
    report=dict(kind=KIND,objective_protocol=PROTOCOL,step=step,frames=len(panel),model_sha256=weights_before,**identity,
        input_kind='npr_rgb',source_held_out=True,world_rollout=False,inference_rgb_only=True,
        source_input_allowed_for_sr=False,source_targets_are_labels_only=True,Source_or_RSSM_hint_as_input=False,
        new_model_uses_fixed_atlas=False,models=models,temporal=temporal,background_ablation=deepcopy(BACKGROUND_ABLATION),
        preferred_baseline_scope='exact_model_background_same_RGB_forward_not_Atlas_or_old_enemy_RGB_head',
        source_background_edge_scope='Source_RGB_edges_above_0.04_outside_5x5_dilated_true_NPC_mask_not_semantic_wall_labels',
        low_frequency_change_scope='per_frame_mean_abs_8x8_average_pool_of_image_minus_exact_preferred_background',
        preview_cases={role:case['row_index']for role,case in cases.items()},
        detail_validation={**deepcopy(detail),'panel_fingerprint':detail['fingerprint'],'objective_protocol':PROTOCOL,
            'models':{arm:_summaries(detail_rows[arm],detail_metadata)for arm in ARMS},
            'primary_validation_score_unchanged':True,'world_rollout':False,'background_ablation':deepcopy(BACKGROUND_ABLATION)},
        validation_objective=dict(protocol=PROTOCOL,loss=total_loss/len(panel),frames=len(panel),source_held_out=True,
            aggregation='image_count_weighted_mean_of_actual_batch_objectives',
            components={key:dict(raw=v['raw']/len(panel),weighted=v['weighted']/len(panel),weight=v['weight'])for key,v in components.items()}),
        gameplay_qualified=False,automatic_deployment=False,budget_completion_is_quality=False)
    summary=models['current']['summary']
    report['quality_goal']=dict(passed=summary['ssim_mean']>.91 and summary['mae_mean']*255<2.5,
        observed_ssim=summary['ssim_mean'],observed_mae_255=summary['mae_mean']*255,
        ssim_strictly_greater_than=.91,mae_255_strictly_less_than=2.5,gameplay_qualified=False)
    report['preferred_base_preservation']=background_detail_candidate(report,baseline_report)
    json.dumps(report,allow_nan=False)
    return report,_preview(cases,report,step)


def background_detail_candidate(report,baseline_report=None):
    if report.get('kind')!=KIND or report.get('objective_protocol')!=PROTOCOL:raise ValueError('Unrelated background detail visual candidate')
    if baseline_report is not None:
        for key in('objective_protocol','preferred_background_sha256','data_identity_sha256','panel_fingerprint'):
            if report.get(key)!=baseline_report.get(key):raise ValueError('Background detail visual baseline/panel changed')
        if report.get('detail_validation',{}).get('fingerprint')!=baseline_report.get('detail_validation',{}).get('fingerprint'):
            raise ValueError('Background detail visual animation panel changed')
    arms=report['models'];a=arms['current'];b=arms['preferred_frozen_background'];epsilon=1e-8
    def finite(x):return isinstance(x,(int,float))and np.isfinite(x)
    pairs=dict(main_ssim=(a['summary'].get('ssim_mean'),b['summary'].get('ssim_mean')),
        foreground_roi_ssim=(a['cohorts']['visible_enemy'].get('roi_crop_ssim_mean'),b['cohorts']['visible_enemy'].get('roi_crop_ssim_mean')),
        empty_source_mae=(a['cohorts']['empty'].get('mae_mean'),b['cohorts']['empty'].get('mae_mean')))
    supported=all(all(finite(x)for x in values)for values in pairs.values())
    checks={key:bool(all(finite(x)for x in pair)and(pair[0]<=pair[1]+epsilon if key=='empty_source_mae'else pair[0]+epsilon>=pair[1]))for key,pair in pairs.items()}
    return dict(kind='background_detail_preferred_base_preservation_v1',step=report['step'],
        score=float((pairs['main_ssim'][0]+pairs['foreground_roi_ssim'][0])*.5)if supported else None,
        score_definition='mean_of_primary_overall_SSIM_and_visible_enemy_ROI_SSIM; higher_is_better; preservation_eligibility_required',
        eligible=bool(supported and all(checks.values())),
        same_inputs_and_targets=True,numerical_tolerance=epsilon,requirements=checks,
        measurements={key:dict(current=values[0],preferred_base=values[1])for key,values in pairs.items()},
        support_complete=supported,gameplay_qualified=False,automatic_deployment=False)


@torch.no_grad()
def preview_background_detail_sr(model,data,panel,*,step,baseline_report,batch_size=8,detail_panel=None):
    panel=list(panel);detail=_validate_inputs(model,data,panel,step,batch_size,detail_panel)
    _check_report(baseline_report,_identity(model,data,panel),detail)
    if baseline_report['step']>step:raise ValueError('Background detail preview reference is from a future step')
    selected=[(role,baseline_report['preview_cases'][role])for role in ROLES if role in baseline_report['preview_cases']]
    if any(index not in panel for _,index in selected):raise ValueError('Background detail preview outside pinned primary panel')
    sampler=deepcopy(data.sampler_state());modes=[(module,module.training)for module in model.modules()];cases={}
    try:
        model.eval();device=next(model.parameters()).device
        with preserve_preview_rng(device,getattr(data,'rng',None)),strict_render_precision(),torch.inference_mode(),torch.autocast(device.type,enabled=False):
            for first in range(0,len(selected),batch_size):
                chosen=selected[first:first+batch_size];batch=data.batch(len(chosen),indices=[i for _,i in chosen]);_,outputs=_forward(model,batch['lr_npr'].detach())
                for j,(role,index)in enumerate(chosen):cases[role]=dict(row_index=index,source=_pixels(batch['hr_source'][j]),npr=_pixels(batch['lr_npr'][j]),predictions={arm:_pixels(output['image'][j])for arm,output in outputs.items()})
    finally:
        data.restore_sampler(sampler)
        for module,training in modes:module.training=training
    preview=_preview(cases,baseline_report,step);preview['metadata'].update(preview_only=True,reference_validation_step=baseline_report['step'],model_sha256=_hash_tensors(model.state_dict()))
    return preview


def save_background_detail_sr_preview(preview,path,*,every=15):
    metadata=preview['metadata']
    if(metadata.get('kind')!=PREVIEW_KIND or metadata.get('objective_protocol')!=PROTOCOL
            or metadata.get('row_count')!=4 or metadata.get('row_labels')!=list(ROW_LABELS)):
        raise ValueError('Background detail preview requires its four explicit rows')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);canvas=Image.new('RGB',(1024,1120),(14,22,31));draw=ImageDraw.Draw(canvas)
    font=None
    for candidate in('C:/Windows/Fonts/arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(candidate).is_file():font=ImageFont.truetype(candidate,14);break
    for row,label in enumerate(ROW_LABELS):
        draw.text((4,row*280+3),label,fill=(235,239,245),font=font)
        for column,role in enumerate(ROLES):
            case=preview['cases'].get(role)
            if case is None:continue
            pixels=(case['source'],case['npr'],case['predictions']['preferred_frozen_background'],case['predictions']['current'])[row]
            canvas.paste(Image.fromarray(np.asarray(pixels)).resize((256,256),Image.Resampling.NEAREST),(column*256,row*280+24))
            draw.text((column*256+4,row*280+13),role+' | VAL '+str(case['row_index']),fill=(220,225,235))
    temporary=path.with_name(path.stem+'.tmp.png');canvas.save(temporary,format='PNG');replace_file(temporary,path)
    result={**metadata,'path':str(path),'every':every,'generated_at':now(),'preview_image_size':[1024,1120],
        'caption':'Оригинал128 / нейрорендер32 / выбранная замороженная основа / основа с деталями'}
    atomic_json(path.with_suffix('.json'),result);return result
