"""Matched GAN/NoGAN SR evaluation, including genuine adjacent held-out frames.

The discriminator is absent. Both generators see exactly the same cached RGB
and optional predicted hypothesis. Source pixels/masks are measurement labels.
Temporal changes are compared with the true Source change, never with zero.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw
import torch

from .renderer_student import _hash_tensors
from .rssm_preview_assets import preserve_preview_rng
from .semantic_sr_validation import _rows,_summary,_cohorts,_pixels
from .training_pipeline import atomic_json,now,replace_file
from .visual_quality import strict_render_precision

PROTOCOL='enemy_roi_patchgan_ab_v1'
KIND='enemy_roi_patchgan_ab_validation_v1'
PREVIEW_KIND='enemy_roi_patchgan_ab_preview_v1'
ARMS=('gan','no_gan')
ROLES=('near','mid','far','empty')


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def build_temporal_panel(data,panel):
    """Use only proven adjacent selected VAL rows, not previous Source32 alone."""
    panel=list(map(int,panel))
    if getattr(data,'split',None)!='val' or len(set(panel))!=len(panel):
        raise ValueError('Temporal SR panel must contain unique held-out VAL indices')
    rows=getattr(data,'rows',None)
    if rows is None:
        pairs=[]
    else:
        lookup={};pairs=[]
        for index in panel:
            row=rows[index]
            if row.get('split')!='val':raise ValueError('Temporal SR cannot use a TRAIN row')
            if row.get('dynamic')is not True:continue
            if (type(row.get('frame'))is not int or row['frame']<0
                    or not isinstance(row.get('source_sha256'),str)
                    or not isinstance(row.get('source_path'),str)
                    or type(row.get('source_seed'))is not int):
                raise ValueError('Dynamic temporal row lacks exact Source episode/tick provenance')
            key=(row['source_sha256'],row['source_path'],row['source_seed'],row['frame'])
            if key in lookup:raise ValueError('Temporal Source episode/tick was selected twice')
            lookup[key]=index
        for key,index in sorted(lookup.items()):
            previous=(*key[:-1],key[-1]-1)
            if previous in lookup:
                pairs.append(dict(previous_index=lookup[previous],current_index=index,
                    source_sha256=key[0],source_path=key[1],source_seed=key[2],
                    previous_frame=key[-1]-1,current_frame=key[-1],
                    cohort=rows[index].get('cohort','unknown')))
    payload=dict(data_identity_sha256=_digest(data.identity),pairs=pairs)
    counts={}
    for pair in pairs:counts[pair['cohort']]=counts.get(pair['cohort'],0)+1
    fingerprint=_digest(payload)
    return dict(available=bool(pairs),pair_count=len(pairs),panel_fingerprint=fingerprint,
        fingerprint=fingerprint,cohort_counts=counts,pairs=pairs,
        episode_count=len({row['source_sha256']for row in pairs}),
        dt=1/30,source_held_out=True,
        scope='adjacent_cached_one_step_causal_NPR_not_autonomous_gameplay')


def _models(gan_model,control_model):
    models={'gan':gan_model,'no_gan':control_model}
    if gan_model is control_model or getattr(gan_model,'config',None)!=getattr(control_model,'config',None):
        raise ValueError('Matched SR arms must be separate models with the same inference config/blend scale')
    shared={p.data_ptr()for p in gan_model.parameters()if p.requires_grad}&{
        p.data_ptr()for p in control_model.parameters()if p.requires_grad}
    if shared:raise ValueError('GAN and NoGAN arms share mutable parameter storage')
    devices={p.device for net in models.values()for p in net.parameters()}
    if len(devices)!=1 or any(p.dtype!=torch.float32 for net in models.values()for p in net.parameters()):
        raise ValueError('Matched SR evaluation requires one device and FP32 weights')
    return models,next(iter(devices))


def _forward(net,low,batch):
    hypothesis=batch.get('hypothesis')
    result=net.forward_with_aux(low,None if hypothesis is None else hypothesis.detach())
    image=result['image']
    if (image.shape!=(len(low),3,128,128)or not bool(torch.isfinite(image).all())
            or bool(((image<0)|(image>1)).any())):
        raise FloatingPointError('Invalid matched SR RGB128 output')
    return result


def _edge_mean(image,mask):
    numerator=image.new_zeros(len(image));denominator=image.new_zeros(len(image))
    for axis in(-1,-2):
        a,b=[slice(None)]*4,[slice(None)]*4
        a[axis],b[axis]=slice(1,None),slice(None,-1);a,b=tuple(a),tuple(b)
        support=torch.maximum(mask[a],mask[b])
        numerator+=((image[a]-image[b]).abs()*support).flatten(1).sum(-1)
        denominator+=3*support.flatten(1).sum(-1)
    return numerator/denominator.clamp_min(1),denominator>0


@torch.no_grad()
def _extras(output,batch):
    image=output['image'].float();target=batch['hr_source'].detach().float()
    mask=batch['mask_hr'].detach().float();background=batch['hr_background'].detach().float()
    visual=output['visual_route'].detach().bool().reshape(-1)
    guided=output['guided_route'].detach().bool().reshape(-1)
    if bool((visual&guided).any()):raise ValueError('Inference painted both SR routes at once')
    warped=torch.where(guided[:,None,None,None],output['guided_raw_warped_rgba'],
                       output['visual_raw_warped_rgba']).float()
    raw=background+warped[:,3:4]*(warped[:,:3]-background)
    alpha=(output['visual_composite_alpha_hr'].float()
           +output['guided_route_weight'][:,:,None,None].float()*output['enemy_composite_alpha_hr'].float())
    if (alpha.shape!=mask.shape or warped.shape!=(len(mask),4,128,128)
            or not all(bool(torch.isfinite(value).all())for value in(raw,alpha,warped))):
        raise FloatingPointError('Invalid raw/effective SR restoration layers')
    area=mask.flatten(1).sum(-1);visible=area>0
    raw_fg=((raw-target).abs()*mask).flatten(1).sum(-1)/(3*area).clamp_min(1)
    raw_bg=((raw-background).abs()*(1-mask)).flatten(1).sum(-1)/(3*(1-mask).flatten(1).sum(-1)).clamp_min(1)
    raw_edge,_=_edge_mean(raw-target,mask)
    sharp,has_edge=_edge_mean(image,mask);truth_sharp,_=_edge_mean(target,mask)
    masks={name:values>=.5 for name,values in(('final',alpha),('raw',warped[:,3:4]))}
    shape={}
    for name,predicted in masks.items():
        truth=mask>=.5;intersection=(predicted&truth).flatten(1).sum(-1).float()
        union=(predicted|truth).flatten(1).sum(-1).float()
        total=predicted.flatten(1).sum(-1)+truth.flatten(1).sum(-1)
        shape[name]=(intersection/union.clamp_min(1),2*intersection/total.clamp_min(1))
    result=[]
    for i in range(len(mask)):
        present=bool(visible[i]);empty=not present
        result.append(dict(raw_foreground_mae=float(raw_fg[i])if present else None,
            raw_foreground_edge_mae=float(raw_edge[i])if present else None,
            raw_background_mae=float(raw_bg[i]),
            raw_empty_mae=float((raw[i]-background[i]).abs().mean())if empty else None,
            silhouette_iou_mean=float(shape['final'][0][i])if present else None,
            silhouette_f1_mean=float(shape['final'][1][i])if present else None,
            raw_silhouette_iou_mean=float(shape['raw'][0][i])if present else None,
            raw_silhouette_f1_mean=float(shape['raw'][1][i])if present else None,
            sharpness_mean=float(sharp[i])if has_edge[i] else None,
            source_sharpness_mean=float(truth_sharp[i])if has_edge[i] else None,
            sharpness_error_mean=float((sharp[i]-truth_sharp[i]).abs())if has_edge[i] else None,
            empty_false_npc_rate=float(visual[i]|guided[i])if empty else None,
            empty_layer_false_positive_rate=float((alpha[i]>=.05).any())if empty else None,
            empty_alpha_leakage_mean=float(alpha[i].mean())if empty else None,
            empty_mae=float((image[i]-target[i]).abs().mean())if empty else None))
    return result


def _arm_summary(rows,selected):
    summary=_summary(rows,selected)
    chosen=[row for row,keep in zip(rows,selected)if keep]
    keys=set().union(*(row.keys()for row in chosen)) if chosen else set()
    for key in keys:
        if key.startswith(('raw_','silhouette_','sharpness_','source_sharpness_','empty_')):
            values=[row[key]for row in chosen if row.get(key)is not None]
            summary[key]=float(np.mean(values))if values else None
    for count,key in(('silhouette_frames','silhouette_iou_mean'),
        ('sharpness_frames','sharpness_mean'),('empty_frames','empty_mae'),
        ('raw_foreground_frames','raw_foreground_mae'),
        ('raw_foreground_edge_frames','raw_foreground_edge_mae'),
        ('foreground_edge_frames','foreground_edge_mae')):
        summary[count]=sum(row.get(key)is not None for row in chosen)
    summary['empty_false_npc_count']=sum(row.get('empty_false_npc_rate')==1 for row in chosen)
    return summary


def _temporal_summary(pairs,images,targets,masks):
    rows=[]
    for pair in pairs:
        a,b=pair['previous_index'],pair['current_index']
        residual=(images[b]-images[a])-(targets[b]-targets[a])
        support=torch.maximum(masks[a],masks[b]).float()
        area=support.sum();foreground=float((residual.abs()*support).sum()/(3*area))if area else None
        edge,has=_edge_mean(residual[None],support[None])
        rows.append(dict(delta_mae=float(residual.abs().mean()),foreground_delta_mae=foreground,
            foreground_edge_delta_mae=float(edge[0])if has[0] else None,cohort=pair['cohort']))
    def summarize(selected):
        values=[row for row,keep in zip(rows,selected)if keep]
        result={'pair_count':len(values),'foreground_pairs':sum(row['foreground_delta_mae']is not None for row in values)}
        for key in('delta_mae','foreground_delta_mae','foreground_edge_delta_mae'):
            usable=[row[key]for row in values if row[key]is not None]
            result[key]=float(np.mean(usable))if usable else None
        return result
    return {**summarize([True]*len(rows)),
        'cohorts':{cohort:summarize([row['cohort']==cohort for row in rows])for cohort in(*ROLES[:3],'guard')}}


@torch.no_grad()
def evaluate_patchgan_pair(gan_model,control_model,data,panel,*,step,batch_size=8,
                          temporal_panel=None,reconstruction_objective=None,include_guards=True):
    """Evaluate matched generators; training-ledger/parent pins are attached by caller."""
    if type(step)is not int or step<0 or type(batch_size)is not int or batch_size<1:
        raise ValueError('A/B validation requires a complete nonnegative optimizer step')
    panel=list(map(int,panel))
    if getattr(data,'split',None)!='val' or not panel or len(set(panel))!=len(panel):
        raise ValueError('A/B SR validation requires unique held-out indices')
    models,device=_models(gan_model,control_model)
    temporal=build_temporal_panel(data,panel)
    if temporal_panel is not None and temporal_panel!=temporal:
        raise ValueError('Pinned temporal panel differs from actual adjacent held-out Source rows')
    needed={index for pair in temporal['pairs']for index in(pair['previous_index'],pair['current_index'])}
    if reconstruction_objective is None:
        from .decoder_restoration_objectives import decoder_restoration_loss
        reconstruction_objective=decoder_restoration_loss
    metadata=[];domains={};cases={};images={name:{}for name in ARMS};targets={};masks={}
    objective_sum=dict.fromkeys(ARMS,0.);objective_frames=0
    private=deepcopy(data.sampler_state());modes=[(m,m.training)for net in models.values()for m in net.modules()]
    hashes={name:_hash_tensors(net.state_dict())for name,net in models.items()}
    try:
        for net in models.values():net.eval()
        with preserve_preview_rng(device,getattr(data,'rng',None)),strict_render_precision(), \
                torch.inference_mode(),torch.autocast(device.type,enabled=False):
            for first in range(0,len(panel),batch_size):
                ids=panel[first:first+batch_size];batch=data.batch(len(ids),indices=ids)
                state=batch['state'].detach().cpu();target=batch['hr_source'].detach();mask=batch['mask_hr'].detach()
                if target.shape!=(len(ids),3,128,128)or mask.shape!=(len(ids),1,128,128):
                    raise ValueError('Need exact paired Source128 and visible masks')
                local=[]
                for i,index in enumerate(ids):
                    local.append(dict(row_index=index,alive=bool(state[i,13]>=.5 and state[i,12]>0),
                        player_alive=bool(state[i,4]>0),native_area=int(batch['mask_lr'][i].sum()),
                        hr_visible=bool(mask[i].any()),distance_tiles=float(torch.linalg.vector_norm(
                            (state[i,8:10]-state[i,:2])*16))))
                    if index in needed:targets[index]=target[i].float().cpu().clone();masks[index]=mask[i].bool().cpu().clone()
                metadata.extend(local);cohorts=_cohorts(local);picked={}
                for role in ROLES:
                    if role in cases:continue
                    selected=next((i for i,value in enumerate(cohorts[role])if value),None)
                    if selected is not None:
                        picked[role]=selected;cases[role]=dict(row_index=ids[selected],
                            source=_pixels(target[selected]),npr=_pixels(batch['lr_npr'][selected]),predictions={})
                inputs={'npr':batch['lr_npr'].detach()}
                if include_guards:
                    inputs['clean']=batch['lr_source'].detach()
                    if 'codec_lr'in batch:inputs['codec']=batch['codec_lr'].detach()
                for domain,low in inputs.items():
                    if low.shape!=(len(ids),3,32,32)or not bool(torch.isfinite(low).all()):
                        raise ValueError('A/B inference inputs must be finite native RGB32')
                    rows=domains.setdefault(domain,{name:[]for name in ARMS})
                    for name,net in models.items():
                        output=_forward(net,low,batch)
                        current=_rows(output['image'],target,mask,batch['hr_background'],batch['mask_lr'],output)
                        extra=_extras(output,batch)
                        for row,additional in zip(current,extra):row.update(additional)
                        rows[name].extend(current)
                        if domain=='npr':
                            objective,_=reconstruction_objective(output,batch)
                            if not bool(torch.isfinite(objective)):raise FloatingPointError('Nonfinite A/B reconstruction objective')
                            objective_sum[name]+=float(objective)*len(ids)
                            for i,index in enumerate(ids):
                                if index in needed:images[name][index]=output['image'][i].float().cpu().clone()
                            for role,i in picked.items():cases[role]['predictions'][name]=_pixels(output['image'][i])
                    if domain=='npr':objective_frames+=len(ids)
    finally:
        data.restore_sampler(private)
        for module,training in modes:module.training=training
    if any(_hash_tensors(net.state_dict())!=hashes[name]for name,net in models.items()):
        raise AssertionError('A/B validation changed a generator parameter or buffer')
    cohorts=_cohorts(metadata);results={}
    for domain,arms in domains.items():
        results[domain]={'arms':{name:dict(summary=_arm_summary(rows,[True]*len(rows)),
            cohorts={cohort:_arm_summary(rows,selected)for cohort,selected in cohorts.items()})for name,rows in arms.items()}}
    arms=results['npr']['arms']
    for name in ARMS:
        arms[name]['summary'].update(reconstruction_loss=objective_sum[name]/objective_frames,
                                    reconstruction_frames=objective_frames)
    temporal['arms']={name:_temporal_summary(temporal['pairs'],images[name],targets,masks)for name in ARMS}
    report=dict(kind=KIND,objective_protocol=PROTOCOL,step=step,frames=len(panel),input_kind='npr_rgb',
        source_held_out=True,world_rollout=False,source_input_allowed_for_sr=False,
        source_labels_are_measurement_only=True,source_future_as_input=False,
        inference_input='actual_NPR_RGB32_with_optional_causal_RSSM_hint',
        data_identity_sha256=_digest(data.identity),panel_fingerprint=data.panel_fingerprint(panel),
        arms=arms,domains=results,temporal=temporal,
        preview_cases={role:case['row_index']for role,case in cases.items()},
        effective_alpha_threshold=.5,empty_layer_alpha_threshold=.05,
        sharpness_scope='Source_foreground_gradient_energy_more_is_not_necessarily_better',
        raw_scope='pre_gain_RGBA_composite_on_Source_background_labels_not_inference',
        reconstruction_scope='same_restoration_content_objective_excludes_adversarial_loss',
        comparison_scope='same_held_out_inputs_generator_configs_and_inference_blend_scale',
        training_step_matching_verified_here=False,gameplay_qualified=False,automatic_deployment=False)
    report['gan_minus_no_gan']={key:arms['gan']['summary'][key]-value
        for key,value in arms['no_gan']['summary'].items()
        if type(value)in(float,int)and type(arms['gan']['summary'].get(key))in(float,int)}
    preview=_preview(cases,report,step)
    json.dumps(report,allow_nan=False)
    return report,preview


def _preview(cases,report,step):
    return dict(cases=cases,metadata=dict(kind=PREVIEW_KIND,objective_protocol=PROTOCOL,step=step,
        input_kind='npr_rgb',inference_input='actual_NPR_RGB32_with_optional_causal_RSSM_hint',
        panel_fingerprint=report['panel_fingerprint'],data_identity_sha256=report['data_identity_sha256'],
        source_held_out=True,world_rollout=False,source_input_allowed_for_sr=False,
        row_count=4,row_labels=['true Source128','actual NPR32','matched NoGAN SR128','GAN SR128']))


@torch.no_grad()
def preview_patchgan_pair(gan_model,control_model,data,panel,*,step,baseline_report,batch_size=8):
    """Four pinned cases only; no full validation or temporal recomputation."""
    if (baseline_report.get('kind')!=KIND or baseline_report.get('panel_fingerprint')!=data.panel_fingerprint(list(panel))
            or baseline_report.get('data_identity_sha256')!=_digest(data.identity) or getattr(data,'split',None)!='val'):
        raise ValueError('A/B preview belongs to another held-out data/panel identity')
    models,device=_models(gan_model,control_model);cases={}
    private=deepcopy(data.sampler_state());modes=[(m,m.training)for net in models.values()for m in net.modules()]
    chosen=[(role,baseline_report['preview_cases'][role])for role in ROLES if role in baseline_report['preview_cases']]
    try:
        for net in models.values():net.eval()
        with preserve_preview_rng(device,getattr(data,'rng',None)),strict_render_precision(), \
                torch.inference_mode(),torch.autocast(device.type,enabled=False):
            for first in range(0,len(chosen),batch_size):
                selected=chosen[first:first+batch_size];ids=[index for _,index in selected]
                batch=data.batch(len(ids),indices=ids);low=batch['lr_npr'].detach()
                outputs={name:_forward(net,low,batch)['image']for name,net in models.items()}
                for i,(role,index)in enumerate(selected):
                    cases[role]=dict(row_index=index,source=_pixels(batch['hr_source'][i]),npr=_pixels(low[i]),
                        predictions={name:_pixels(image[i])for name,image in outputs.items()})
    finally:
        data.restore_sampler(private)
        for module,training in modes:module.training=training
    result=_preview(cases,baseline_report,step)
    result['metadata'].update(preview_only=True,reference_validation_step=baseline_report['step'])
    return result


def save_patchgan_preview(preview,path,*,every=15):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    metadata=preview['metadata']
    if metadata.get('kind')!=PREVIEW_KIND or metadata.get('row_count')!=4:
        raise ValueError('Expected the explicit four-row matched GAN/NoGAN preview')
    cell,height=256,280;canvas=Image.new('RGB',(cell*4,height*4),(14,22,31));draw=ImageDraw.Draw(canvas)
    for row,label in enumerate(metadata['row_labels']):
        draw.text((4,row*height+4),label,fill=(235,239,245))
        for column,role in enumerate(ROLES):
            case=preview['cases'].get(role)
            if case is None:continue
            image=(case['source'],case['npr'],case['predictions']['no_gan'],case['predictions']['gan'])[row]
            canvas.paste(Image.fromarray(image).resize((cell,cell),Image.Resampling.NEAREST),
                         (column*cell,row*height+24))
            draw.text((column*cell+4,row*height+13),role+' | row '+str(case['row_index']),fill=(220,225,235))
    temporary=path.with_name(path.stem+'.tmp.png');canvas.save(temporary,format='PNG');replace_file(temporary,path)
    saved={**metadata,'path':str(path),'every':every,'generated_at':now(),'preview_image_size':[1024,1120],
           'caption':'Same actual NPR input and paired Source target; matched NoGAN versus GAN'}
    atomic_json(path.with_suffix('.json'),saved)
    return saved
