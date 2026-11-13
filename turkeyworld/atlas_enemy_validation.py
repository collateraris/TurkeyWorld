"""Held-out fixed-atlas SR versus a frozen learned-renderer reference.

This is an image-reference comparison, not a matched-training NoGAN experiment.
Only NPR RGB and optional cached predictions enter either neural forward.
Source state/masks and the pure fixed-asset label rasterizer measure geometry,
occlusion and soft texture alpha after the forwards have already completed.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw
import torch

from .atlas_enemy_objectives import atlas_enemy_loss,source_full_sprite_geometry,LOSS_WEIGHTS,PROTOCOL
from .enemy_decoder_sr import box_iou
from .renderer_student import _hash_tensors
from .rssm_preview_assets import preserve_preview_rng
from .semantic_sr_validation import _rows,_cohorts,_pixels
from .sr_patchgan_validation import (_arm_summary,_extras,_temporal_summary,_digest,
                                    build_temporal_panel)
from .training_pipeline import atomic_json,now,replace_file
from .visual_quality import strict_render_precision

KIND='atlas_enemy_hybrid_validation_v1'
PREVIEW_KIND='atlas_enemy_hybrid_preview_v1'
MODELS=('atlas','learned_reference')
ROLES=('near','mid','far','empty')


def _networks(model,reference):
    if model is reference or not callable(getattr(model,'render_label_atlas',None)):
        raise ValueError('Atlas validation needs separate atlas and learned-reference networks')
    networks={'atlas':model,'learned_reference':reference}
    devices={p.device for net in networks.values()for p in net.parameters()}
    if len(devices)!=1 or any(p.dtype!=torch.float32 for net in networks.values()for p in net.parameters()):
        raise ValueError('Atlas and frozen learned SR reference must use one device and FP32 parameters')
    return networks,next(iter(devices))


def _forward(model,low,batch):
    hint=batch.get('hypothesis')
    output=model.forward_with_aux(low,None if hint is None else hint.detach())
    image=output['image']
    if image.shape!=(len(low),3,128,128)or not bool(torch.isfinite(image).all())or bool(((image<0)|(image>1)).any()):
        raise FloatingPointError('Invalid atlas/reference RGB128 prediction')
    return output


@torch.no_grad()
def _atlas_metrics(model,output,batch):
    device=output['image'].device
    labels=source_full_sprite_geometry(batch['state'].detach().to(device))
    mask=batch['mask_hr'].detach().to(device).bool();visible=mask.flatten(1).any(-1)
    geometry_valid=visible&labels['geometry_valid']
    expected=model.render_label_atlas(labels['full_bbox'],labels['phase'],dead_index=labels['player_dead'])[:,3:4]*mask.float()
    actual=output['atlas_composite_alpha_hr'].float()
    effective=output['atlas_effective_alpha_hr'].float()
    if actual.shape!=mask.shape or not bool(torch.isfinite(actual).all()and torch.isfinite(effective).all()):
        raise ValueError('Atlas visibility/composite-alpha shape changed')
    alpha_valid=visible&(expected.flatten(1).sum(-1)>0)
    def soft_iou(prediction):
        return torch.minimum(prediction,expected).flatten(1).sum(-1)/torch.maximum(prediction,expected).flatten(1).sum(-1).clamp_min(1e-8)
    alpha_iou,effective_iou=soft_iou(actual),soft_iou(effective)
    predicted=output['atlas_full_bbox'].float()
    bbox_l1=(predicted-labels['full_bbox']).abs().mean(-1)*128
    iou=box_iou(predicted,labels['full_bbox'])
    phase=output['atlas_phase_logits'].argmax(-1)
    binary=output['atlas_visible_logits_hr']>=0
    visibility_iou=(binary&mask).flatten(1).sum(-1).float()/(binary|mask).flatten(1).sum(-1).clamp_min(1)
    result=[]
    for i in range(len(mask)):
        result.append(dict(full_bbox_l1_pixels=float(bbox_l1[i])if geometry_valid[i] else None,
            full_bbox_iou=float(iou[i])if geometry_valid[i] else None,
            phase_accuracy=float(phase[i]==labels['phase'][i])if visible[i] else None,
            alpha_iou_mean=float(alpha_iou[i])if alpha_valid[i] else None,
            effective_alpha_iou_mean=float(effective_iou[i])if alpha_valid[i] else None,
            visibility_iou_mean=float(visibility_iou[i])if visible[i] else None,
            empty_composite_fpr=float((actual[i]>=.05).any())if not visible[i] else None,
            source_visible_zero_label_alpha=int(visible[i]and not alpha_valid[i])))
    return result


def _summary(rows,selected):
    result=_arm_summary(rows,selected)
    chosen=[row for row,keep in zip(rows,selected)if keep]
    for name in('full_bbox_l1_pixels','full_bbox_iou','phase_accuracy','alpha_iou_mean',
                'effective_alpha_iou_mean','visibility_iou_mean','empty_composite_fpr'):
        values=[row[name]for row in chosen if row.get(name)is not None]
        if values:result[name]=float(np.mean(values))
        elif any(name in row for row in rows):result[name]=None
    for count,name in(('full_bbox_frames','full_bbox_l1_pixels'),('phase_frames','phase_accuracy'),
                       ('alpha_frames','alpha_iou_mean'),('visibility_frames','visibility_iou_mean')):
        result[count]=sum(row.get(name)is not None for row in chosen)
    result['source_visible_zero_label_alpha']=sum(row.get('source_visible_zero_label_alpha',0)for row in chosen)
    for name in sorted(set().union(*(row.keys() for row in chosen)) if chosen else ()):
        if name.startswith(('near_wall_', 'far_wall_')):
            values=[row[name] for row in chosen if row.get(name) is not None]
            result[name]=float(np.mean(values)) if values else None
            result[name+'_frames']=len(values)
    if 'empty_composite_fpr'not in result and 'empty_layer_false_positive_rate'in result:
        result['empty_composite_fpr']=result['empty_layer_false_positive_rate']
    return result


@torch.no_grad()
def evaluate_atlas_enemy_sr(model,reference,data,panel,*,step,batch_size=8,baseline_report=None,
                           objective_fn=None,objective_weights=None,objective_protocol=None,extra_metrics_fn=None):
    if type(step)is not int or step<0 or type(batch_size)is not int or batch_size<1:
        raise ValueError('Atlas validation needs a completed nonnegative step and positive batch')
    panel=list(map(int,panel))
    if getattr(data,'split',None)!='val' or not panel or len(set(panel))!=len(panel):
        raise ValueError('Atlas validation requires unique held-out VAL rows')
    networks,device=_networks(model,reference)
    data_sha=_digest(data.identity);panel_sha=data.panel_fingerprint(panel)
    reference_sha=_hash_tensors(reference.state_dict())
    reference_config_sha=_digest(getattr(reference,'config',{}))
    if baseline_report is not None and (baseline_report.get('kind')!=KIND
            or baseline_report.get('data_identity_sha256')!=data_sha
            or baseline_report.get('panel_fingerprint')!=panel_sha
            or baseline_report.get('learned_reference_model_sha256')!=reference_sha
            or baseline_report.get('learned_reference_config_sha256')!=reference_config_sha):
        raise ValueError('Atlas frozen reference or held-out panel differs from the initial report')
    temporal=build_temporal_panel(data,panel)
    needed={index for pair in temporal['pairs']for index in(pair['previous_index'],pair['current_index'])}
    metadata=[];rows={name:[]for name in MODELS};cases={}
    images={name:{}for name in MODELS};targets={};masks={}
    active_weights=LOSS_WEIGHTS if objective_weights is None else objective_weights
    active_protocol=PROTOCOL if objective_protocol is None else objective_protocol
    objective_sum=0.;components={name:{'raw':0.,'weighted':0.,'weight':weight}for name,weight in active_weights.items()}
    private=deepcopy(data.sampler_state());modes=[(m,m.training)for net in networks.values()for m in net.modules()]
    hashes={name:_hash_tensors(net.state_dict())for name,net in networks.items()}
    try:
        for net in networks.values():net.eval()
        with preserve_preview_rng(device,getattr(data,'rng',None)),strict_render_precision(), \
                torch.inference_mode(),torch.autocast(device.type,enabled=False):
            for first in range(0,len(panel),batch_size):
                ids=panel[first:first+batch_size];batch=data.batch(len(ids),indices=ids)
                low=batch['lr_npr'].detach();target=batch['hr_source'].detach();mask=batch['mask_hr'].detach()
                state=batch['state'].detach().cpu()
                if low.shape!=(len(ids),3,32,32)or target.shape!=(len(ids),3,128,128)or mask.shape!=(len(ids),1,128,128):
                    raise ValueError('Expected exact cached NPR32/Source128 and native visible masks')
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
                    selected=next((i for i,yes in enumerate(cohorts[role])if yes),None)
                    if selected is not None:
                        picked[role]=selected;cases[role]=dict(row_index=ids[selected],source=_pixels(target[selected]),
                            npr=_pixels(low[selected]),predictions={})
                for name,net in networks.items():
                    output=_forward(net,low,batch)
                    values=_rows(output['image'],target,mask,batch['hr_background'],batch['mask_lr'],output)
                    extras=_extras(output,batch)
                    for row,extra in zip(values,extras):row.update(extra)
                    if extra_metrics_fn is not None:
                        extra_rows=extra_metrics_fn(net,output,batch)
                        if len(extra_rows)!=len(values):raise ValueError('Extra SR validation metrics have wrong support')
                        for row,extra in zip(values,extra_rows):row.update(extra)
                    if name=='atlas':
                        if objective_fn is None:
                            loss,metrics=atlas_enemy_loss(output,batch,label_renderer=model.render_label_atlas)
                        else:loss,metrics=objective_fn(output,batch,label_renderer=model.render_label_atlas)
                        objective_sum+=float(loss)*len(ids)
                        for component,total in components.items():
                            total['raw']+=metrics[component]*len(ids)
                            total['weighted']+=metrics[component+'_weighted']*len(ids)
                        for row,extra in zip(values,_atlas_metrics(model,output,batch)):row.update(extra)
                    rows[name].extend(values)
                    for i,index in enumerate(ids):
                        if index in needed:images[name][index]=output['image'][i].float().cpu().clone()
                    for role,i in picked.items():cases[role]['predictions'][name]=_pixels(output['image'][i])
    finally:
        data.restore_sampler(private)
        for module,training in modes:module.training=training
    if any(_hash_tensors(net.state_dict())!=hashes[name]for name,net in networks.items()):
        raise AssertionError('Atlas validation changed atlas/reference weights or embedded asset buffers')
    cohorts=_cohorts(metadata)
    models={name:dict(summary=_summary(value,[True]*len(value)),
        cohorts={cohort:_summary(value,selected)for cohort,selected in cohorts.items()})for name,value in rows.items()}
    models['atlas']['summary'].update(reconstruction_loss=objective_sum/len(panel),reconstruction_frames=len(panel))
    temporal['arms']={name:_temporal_summary(temporal['pairs'],images[name],targets,masks)for name in MODELS}
    report=dict(kind=KIND,objective_protocol=active_protocol,step=step,frames=len(panel),input_kind='npr_rgb',
        source_held_out=True,world_rollout=False,source_input_allowed_for_sr=False,
        source_targets_are_labels_only=True,source_future_as_input=False,
        inference_input='actual_NPR_RGB32_with_optional_causal_RSSM_hint_and_embedded_fixed_atlas',
        data_identity_sha256=data_sha,panel_fingerprint=panel_sha,models=models,temporal=temporal,
        learned_reference_model_sha256=reference_sha,
        learned_reference_config_sha256=reference_config_sha,
        reference_scope='frozen_pre_migration_learned_SR_not_matched_training_control',
        full_bbox_target='Source128_full_projected_unclipped_square_not_visible_crop',
        alpha_target='native_fixed_Source_asset_soft_alpha_times_actual_Source_visible_mask',
        alpha_iou_scope='soft_min_max_IoU_actual_composite_alpha_positive_valid_Source_alpha_frames',
        visibility_iou_scope='binary_NN_visibility_vs_actual_Source_visible_mask_positive_frames',
        empty_composite_alpha_threshold=.05,
        preview_cases={role:case['row_index']for role,case in cases.items()},
        validation_objective=dict(protocol=active_protocol,loss=objective_sum/len(panel),frames=len(panel),
            input_kind='npr_rgb',source_held_out=True,batch_size=batch_size,
            aggregation='image_count_weighted_mean_of_actual_batch_objectives',
            loss_scope='atlas_heads_and_final_composite_only_reference_loss_not_comparable',
            components={name:dict(raw=value['raw']/len(panel),weighted=value['weighted']/len(panel),weight=value['weight'])
                        for name,value in components.items()}),
        gameplay_qualified=False,automatic_deployment=False)
    report['atlas_minus_learned_reference']={name:models['atlas']['summary'][name]-value
        for name,value in models['learned_reference']['summary'].items()
        if type(value)in(float,int)and type(models['atlas']['summary'].get(name))in(float,int)}
    json.dumps(report,allow_nan=False)
    return report,_preview(cases,report,step)


def _preview(cases,report,step):
    return dict(cases=cases,metadata=dict(kind=PREVIEW_KIND,objective_protocol=PROTOCOL,step=step,
        input_kind='npr_rgb',source_held_out=True,world_rollout=False,source_input_allowed_for_sr=False,
        source_targets_are_labels_only=True,
        inference_input='actual_NPR_RGB32_with_optional_causal_RSSM_hint_and_embedded_fixed_atlas',
        panel_fingerprint=report['panel_fingerprint'],data_identity_sha256=report['data_identity_sha256'],
        row_count=4,row_labels=['true Source128','actual NPR32','frozen learned SR128','new fixed-atlas SR128']))


@torch.no_grad()
def preview_atlas_enemy_sr(model,reference,data,panel,*,step,baseline_report,batch_size=8):
    if (baseline_report.get('kind')!=KIND or getattr(data,'split',None)!='val'
            or baseline_report.get('data_identity_sha256')!=_digest(data.identity)
            or baseline_report.get('panel_fingerprint')!=data.panel_fingerprint(list(panel))
            or baseline_report.get('learned_reference_model_sha256')!=_hash_tensors(reference.state_dict())
            or baseline_report.get('learned_reference_config_sha256')!=_digest(getattr(reference,'config',{}))):
        raise ValueError('Atlas light preview lost its frozen reference or held-out panel identity')
    networks,device=_networks(model,reference);cases={}
    private=deepcopy(data.sampler_state());modes=[(m,m.training)for net in networks.values()for m in net.modules()]
    selected=[(role,baseline_report['preview_cases'][role])for role in ROLES if role in baseline_report['preview_cases']]
    try:
        for net in networks.values():net.eval()
        with preserve_preview_rng(device,getattr(data,'rng',None)),strict_render_precision(), \
                torch.inference_mode(),torch.autocast(device.type,enabled=False):
            for first in range(0,len(selected),batch_size):
                chosen=selected[first:first+batch_size];batch=data.batch(len(chosen),indices=[index for _,index in chosen])
                low=batch['lr_npr'].detach();outputs={name:_forward(net,low,batch)['image']for name,net in networks.items()}
                for i,(role,index)in enumerate(chosen):
                    cases[role]=dict(row_index=index,source=_pixels(batch['hr_source'][i]),npr=_pixels(low[i]),
                        predictions={name:_pixels(image[i])for name,image in outputs.items()})
    finally:
        data.restore_sampler(private)
        for module,training in modes:module.training=training
    preview=_preview(cases,baseline_report,step)
    preview['metadata'].update(preview_only=True,reference_validation_step=baseline_report['step'])
    return preview


def save_atlas_enemy_preview(preview,path,*,every=15):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);metadata=preview['metadata']
    if metadata.get('kind')!=PREVIEW_KIND or metadata.get('row_count')!=4:
        raise ValueError('Expected the four-row fixed-atlas versus frozen learned SR preview')
    canvas=Image.new('RGB',(1024,1120),(14,22,31));draw=ImageDraw.Draw(canvas)
    for row,label in enumerate(metadata['row_labels']):
        draw.text((4,row*280+4),label,fill=(235,239,245))
        for column,role in enumerate(ROLES):
            case=preview['cases'].get(role)
            if case is None:continue
            image=(case['source'],case['npr'],case['predictions']['learned_reference'],case['predictions']['atlas'])[row]
            canvas.paste(Image.fromarray(image).resize((256,256),Image.Resampling.NEAREST),(column*256,row*280+24))
            draw.text((column*256+4,row*280+13),role+' | row '+str(case['row_index']),fill=(220,225,235))
    temporary=path.with_name(path.stem+'.tmp.png');canvas.save(temporary,format='PNG');replace_file(temporary,path)
    result={**metadata,'path':str(path),'every':every,'generated_at':now(),'preview_image_size':[1024,1120],
        'caption':'Same Source target/NPR input; frozen learned renderer versus NN-guided fixed enemy atlas'}
    atomic_json(path.with_suffix('.json'),result)
    return result
