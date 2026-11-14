"""Held-out RGB-only enemy reconstruction versus a separate frozen atlas SR.

The new network sees NPR RGB alone. Source RGB, masks, state and animation are
measurements/loss labels. The atlas belongs only to the explicit reference arm.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw
import torch

from .enemy_animation import ANIMATION_CLASSES,enemy_animation_index
from .renderer_student import _hash_tensors
from .rssm_preview_assets import preserve_preview_rng
from .semantic_sr_validation import _rows,_pixels
from .sr_neural_pairs_data import fingerprint
from .sr_patchgan_validation import build_temporal_panel,_temporal_summary
from .training_pipeline import atomic_json,now,replace_file
from .visual_quality import strict_render_precision


PROTOCOL='neural_enemy_restoration_sr_v1'
FOREGROUND_PROTOCOL='neural_enemy_restoration_foreground_v2'
DETAIL_PROTOCOL='neural_enemy_restoration_detail_v3'
OBJECTIVE_PROTOCOLS=(PROTOCOL,FOREGROUND_PROTOCOL,DETAIL_PROTOCOL)
DETAIL_PANEL_KIND='neural_enemy_restoration_detail_panel_v1'
KIND='neural_enemy_restoration_validation_v1'
PREVIEW_KIND='neural_enemy_restoration_preview_v1'
ROLES=('near','mid','far','empty')
MODELS=('current','frozen_atlas_parent')
EMPTY_TOPK_PIXELS=64
BACKGROUND_ABLATION=dict(kind='matched_frozen_background_v1',same_inputs=True,
    output='display_clamped_model_background',additional_model_forwards=0,
    source_targets_are_labels_only=True)


def _objective_protocol(value):
    if value not in OBJECTIVE_PROTOCOLS:raise ValueError('Unknown neural enemy restoration objective protocol')
    return value


def _detail_inventory(data):
    rows=getattr(data,'rows',None)
    if(getattr(data,'split',None)!='val'or not isinstance(rows,list)or not rows
            or any(row.get('split')!='val'for row in rows)):
        raise ValueError('Detail animation inventory requires actual held-out VAL rows, never TRAIN')
    count=len(rows);labels=getattr(data,'animation_labels',None);visible=getattr(data,'visible',None)
    distances=getattr(data,'distance',None)
    if(all(value is not None and len(value)==count for value in(labels,visible,distances))):
        inventory=[dict(row_index=index,animation_label=int(labels[index]),
            phase=ANIMATION_CLASSES[int(labels[index])],visible=bool(visible[index]),distance=distances[index])
            for index in range(count)]
    else:
        inventory=[];private=deepcopy(data.sampler_state())
        try:
            with preserve_preview_rng(getattr(data,'device','cpu'),getattr(data,'rng',None)),torch.no_grad():
                for first in range(0,count,32):
                    ids=list(range(first,min(first+32,count)));batch=data.batch(len(ids),indices=ids)
                    states=batch['state'].detach().cpu();mask=batch['mask_hr'].detach().cpu()
                    phases=enemy_animation_index(states)
                    if mask.shape!=(len(ids),1,128,128):raise ValueError('Detail inventory requires actual native128 masks')
                    for local,index in enumerate(ids):
                        distance=float(((states[local,8:10]-states[local,:2])*16).norm())
                        phase=int(phases[local]);inventory.append(dict(row_index=index,animation_label=phase,
                            phase=ANIMATION_CLASSES[phase],visible=bool(mask[local].any()),
                            distance='near'if distance<2 else'mid'if distance<5 else'far'))
        finally:data.restore_sampler(private)
    if any(row['distance']not in('near','mid','far')or not 0<=row['animation_label']<len(ANIMATION_CLASSES)for row in inventory):
        raise ValueError('Detail inventory has invalid Source phase/distance labels')
    return inventory


def build_detail_animation_panel(data,*,seed=173,max_per_phase=32):
    """Pin all visible death/attack rows plus balanced hurt rows from VAL only."""
    if type(seed)is not int or type(max_per_phase)is not int or max_per_phase<1:
        raise ValueError('Detail panel needs an integer seed and positive phase cap')
    inventory=_detail_inventory(data);rng=np.random.default_rng(seed)
    groups={name:[row['row_index']for row in inventory if row['visible']and row['phase']==name]
        for name in ANIMATION_CLASSES}
    indices=[index for name in('attack','death_start','death_late')for index in groups[name]]
    buckets={distance:rng.permutation([index for index in groups['hurt']if inventory[index]['distance']==distance]).tolist()
        for distance in('near','mid','far')};counts=dict.fromkeys(buckets,0)
    for _ in range(min(max_per_phase,len(groups['hurt']))):
        available=[distance for distance,values in buckets.items()if values]
        distance=min(available,key=lambda value:(counts[value],value));indices.append(int(buckets[distance].pop()));counts[distance]+=1
    indices=sorted(indices)
    if not indices:raise ValueError('No visible held-out attack/hurt/death rows for detail validation')
    if len(set(indices))!=len(indices):raise AssertionError('Detail phase selection duplicated a VAL row')
    selected=set(indices)
    def phase_counts(rows):return {name:sum(row['visible']and row['phase']==name for row in rows)for name in ANIMATION_CLASSES}
    support=dict(available_visible=phase_counts(inventory),selected_visible=phase_counts([inventory[i]for i in indices]),
        available_by_distance={distance:phase_counts([row for row in inventory if row['distance']==distance])for distance in('near','mid','far')},
        selected_by_distance={distance:phase_counts([row for row in inventory if row['row_index']in selected and row['distance']==distance])for distance in('near','mid','far')})
    value=dict(kind=DETAIL_PANEL_KIND,version=1,split='val',indices=indices,frames=len(indices),seed=seed,
        max_per_phase=max_per_phase,source_frames=len(inventory),data_identity_sha256=fingerprint(data.identity),
        source_panel_fingerprint=data.panel_fingerprint(list(range(len(inventory)))),
        inventory_fingerprint=fingerprint(inventory),selected_label_fingerprint=fingerprint([inventory[i]for i in indices]),
        support=support,source_held_out=True,training_uses_panel=False,
        selection='all_visible_deaths_and_attack_up_to_max_hurt')
    value['fingerprint']=fingerprint(value)
    return value


def _validate_detail_panel(data,receipt):
    if(not isinstance(receipt,dict)or receipt.get('kind')!=DETAIL_PANEL_KIND
            or not isinstance(receipt.get('indices'),list)or not receipt['indices']
            or any(type(index)is not int for index in receipt['indices'])
            or len(set(receipt['indices']))!=len(receipt['indices'])):
        raise ValueError('V3 requires an explicit pinned unique VAL detail_panel receipt')
    expected=build_detail_animation_panel(data,seed=receipt.get('seed'),max_per_phase=receipt.get('max_per_phase'))
    if json.dumps(receipt,sort_keys=True,allow_nan=False)!=json.dumps(expected,sort_keys=True,allow_nan=False):
        raise ValueError('Detail panel receipt differs from actual VAL Source identity or phase labels')
    return expected


def _networks(model,reference):
    if model is reference or getattr(model,'config',{}).get('architecture')!='neural_enemy_restoration_sr':
        raise ValueError('Learned enemy validation needs a distinct RGB-only restoration model')
    networks=dict(zip(MODELS,(model,reference)))
    parameters=[p for net in networks.values()for p in net.parameters()]
    devices={p.device for p in parameters}
    if len(devices)!=1 or any(p.dtype!=torch.float32 for p in parameters):
        raise ValueError('SR validation requires one device and FP32 model/reference parameters')
    return networks,next(iter(devices))


def _forward(model,low,*,current):
    # No batch or hypothesis is passed even if a caller happens to retain one.
    output=model.forward_with_aux(low)if current else model.forward_with_aux(low,None)
    image=output.get('image')
    if(not torch.is_tensor(image)or image.shape!=(len(low),3,128,128)
            or not bool(torch.isfinite(image).all())or bool(((image<0)|(image>1)).any())):
        raise FloatingPointError('SR validation requires finite bounded RGB128')
    if current:
        rgb,alpha,logits=output.get('enemy_rgb'),output.get('alpha'),output.get('alpha_logits')
        if(not torch.is_tensor(rgb)or rgb.shape!=image.shape
                or not torch.is_tensor(alpha)or alpha.shape!=(len(low),1,128,128)
                or not torch.is_tensor(logits)or logits.shape!=alpha.shape
                or not all(bool(torch.isfinite(value).all())for value in(rgb,alpha,logits))
                or bool(((rgb<0)|(rgb>1)).any())or bool(((alpha<0)|(alpha>1)).any())):
            raise FloatingPointError('RGB restoration needs independent finite bounded enemy RGB and alpha128')
    return output


def _with_matched_background(outputs,*,objective_protocol):
    if objective_protocol!=DETAIL_PROTOCOL:return outputs
    current=outputs['current'];background=current.get('background')
    if(not torch.is_tensor(background)or background.shape!=current['image'].shape
            or not bool(torch.isfinite(background).all())):
        raise FloatingPointError('V3 ablation requires the actual finite current-model background')
    value=background.detach().float().clamp(0,1)
    # The background is another measurement of the SAME forward, not another
    # SR model or learned restoration head. It has no predicted alpha mask.
    return dict(current=current,frozen_background={'image':value,'background':value},
        frozen_atlas_parent=outputs['frozen_atlas_parent'])


def _cohorts(metadata):
    names=('overall','visible_enemy','near','mid','far','empty','occluded','navigation','corpse','hp0',
        'subpixel_lr',*[f'animation_{name}'for name in ANIMATION_CLASSES])
    result={name:[]for name in names}
    for row in metadata:
        visible=row['hr_visible'];distance=row['distance_tiles']
        values=dict(overall=True,visible_enemy=visible,near=visible and distance<2,
            mid=visible and 2<=distance<5,far=visible and distance>=5,empty=not visible,
            occluded=not visible and row['alive']and row['player_alive'],
            navigation=not visible and not row['alive']and row['player_alive'],
            corpse=visible and not row['alive'],hp0=not row['player_alive'],
            subpixel_lr=visible and row['native_area']==0)
        values.update({f'animation_{name}':visible and row['animation_label']==index
            for index,name in enumerate(ANIMATION_CLASSES)})
        for name in names:result[name].append(bool(values[name]))
    return result


@torch.no_grad()
def _mask_rows(output,batch,*,current):
    mask=batch['mask_hr'].detach().float();truth=mask>=.5
    alpha=output.get('alpha')if current else output.get('atlas_composite_alpha_hr')
    if alpha is not None and(alpha.shape!=mask.shape or not bool(torch.isfinite(alpha).all())
            or bool(((alpha<0)|(alpha>1)).any())):
        raise FloatingPointError('Invalid measured composite alpha128')
    background=output.get('background')
    if background is None:background=output.get('restoration_base_rgb')
    if background is not None and(background.shape!=output['image'].shape or not bool(torch.isfinite(background).all())):
        raise FloatingPointError('Invalid provided frozen background for localized paint metrics')
    result=[]
    raw=None
    if current:
        # Correct target context outside the true mask isolates RGB detail from
        # alpha and from unsupervised raw-RGB background pixels.
        raw=batch['hr_source'].detach()+mask*(output['enemy_rgb'].float()-batch['hr_source'].detach())
        raw_rows=_rows(raw,batch['hr_source'],mask,batch['hr_background'],batch['mask_lr'])
    for index in range(len(mask)):
        visible=bool(truth[index].any());row={'target_empty':not visible}
        if alpha is not None:
            predicted=alpha[index]>=.5;expected=truth[index]
            tp=int((predicted&expected).sum());fp=int((predicted&~expected).sum());fn=int((~predicted&expected).sum())
            union=tp+fp+fn
            row.update(alpha_tp=tp,alpha_fp=fp,alpha_fn=fn,
                alpha_iou=tp/union if visible and union else None,
                alpha_pixel_precision=tp/(tp+fp)if tp+fp else None,
                alpha_pixel_recall=tp/(tp+fn)if tp+fn else None,
                empty_false_npc_rate=float(bool((alpha[index]>=.05).any()))if not visible else None,
                empty_alpha_leakage=float(alpha[index].mean())if not visible else None)
            if not visible:
                values=alpha[index].flatten().float();dense=values>=.5
                row.update(empty_soft_alpha_any_rate=float(bool((values>=.05).any())),
                    empty_dense_false_npc_rate=float(bool(dense.any())),
                    empty_dense_alpha_pixels=int(dense.sum()),empty_dense_alpha_fraction=float(dense.float().mean()),
                    empty_alpha_max=float(values.max()),
                    empty_alpha_top64_mean=float(values.topk(EMPTY_TOPK_PIXELS).values.mean()))
        if not visible and background is not None:
            paint=(output['image'][index].float()-background[index].detach().float()).abs().mean(0).flatten()
            row.update(empty_paint_max_mae=float(paint.max()),
                empty_paint_top64_mae=float(paint.topk(EMPTY_TOPK_PIXELS).values.mean()))
        if current:
            row.update(raw_rgb_foreground_mae=raw_rows[index]['foreground_mae'],
                raw_rgb_foreground_edge_mae=raw_rows[index]['foreground_edge_mae'],
                raw_rgb_foreground_ssim=raw_rows[index]['roi_crop_ssim'],
                raw_rgb_roi_crop_ssim=raw_rows[index]['roi_crop_ssim'])
        result.append(row)
    return result


def _summary(rows,selected):
    chosen=[row for row,keep in zip(rows,selected)if keep]
    result={'frames':len(chosen)}
    keys=('psnr','ssim','mae','foreground_mae','background_mae','foreground_edge_mae','silhouette_edge_mae',
        'roi_crop_ssim','roi_crop_mae','alpha_iou','raw_rgb_foreground_mae','raw_rgb_foreground_edge_mae',
        'raw_rgb_foreground_ssim','raw_rgb_roi_crop_ssim','empty_false_npc_rate','empty_alpha_leakage',
        'empty_soft_alpha_any_rate','empty_dense_false_npc_rate','empty_dense_alpha_pixels',
        'empty_dense_alpha_fraction','empty_alpha_max','empty_alpha_top64_mean',
        'empty_paint_max_mae','empty_paint_top64_mae')
    for key in keys:
        values=[row[key]for row in chosen if row.get(key)is not None]
        name=key+'_mean'if key in('psnr','ssim','mae','roi_crop_ssim','roi_crop_mae')else key
        result[name]=float(np.mean(values))if values else None
        result[key+'_frames']=len(values)
    measured=[row for row in chosen if 'alpha_tp'in row]
    tp=sum(row['alpha_tp']for row in measured);fp=sum(row['alpha_fp']for row in measured);fn=sum(row['alpha_fn']for row in measured)
    result.update(alpha_frames=len(measured),alpha_tp=tp,alpha_fp=fp,alpha_fn=fn,
        alpha_pixel_precision=tp/(tp+fp)if tp+fp else None,
        alpha_pixel_recall=tp/(tp+fn)if tp+fn else None,
        alpha_pixel_f1=2*tp/(2*tp+fp+fn)if 2*tp+fp+fn else None,
        empty_frames=sum(row.get('empty_false_npc_rate')is not None for row in chosen),
        empty_false_npc_count=sum(row.get('empty_false_npc_rate')==1 for row in chosen),
        empty_target_frames=sum(row.get('target_empty')is True for row in chosen),
        empty_alpha_frames=sum(row.get('empty_alpha_max')is not None for row in chosen),
        empty_paint_frames=sum(row.get('empty_paint_top64_mae')is not None for row in chosen),
        empty_dense_false_npc_count=sum(row.get('empty_dense_false_npc_rate')==1 for row in chosen),
        empty_dense_alpha_pixels_total=sum(row['empty_dense_alpha_pixels']for row in chosen
            if row.get('empty_dense_alpha_pixels')is not None))
    return result


def _identities(model,reference,data,panel):
    return dict(data_identity_sha256=fingerprint(data.identity),panel_fingerprint=data.panel_fingerprint(panel),
        frozen_reference_model_sha256=_hash_tensors(reference.state_dict()),
        frozen_reference_config_sha256=fingerprint(getattr(reference,'config',{})),
        model_config_sha256=fingerprint(getattr(model,'config',{})))


def _check_report(report,identity,*,objective_protocol=PROTOCOL,detail_panel=None):
    protocol=_objective_protocol(objective_protocol)
    if(report.get('kind')!=KIND or report.get('objective_protocol')!=protocol
            or type(report.get('step'))is not int or report['step']<0
            or any(report.get(name)!=value for name,value in identity.items())):
        raise ValueError('Restoration validation changed its objective protocol, frozen reference architecture or held-out panel')
    if protocol==DETAIL_PROTOCOL:
        detail=report.get('detail_validation')or{}
        if(not isinstance(detail_panel,dict)or report.get('background_ablation')!=BACKGROUND_ABLATION
                or detail.get('kind')!=DETAIL_PANEL_KIND or detail.get('objective_protocol')!=DETAIL_PROTOCOL
                or detail.get('panel_fingerprint')!=detail_panel.get('fingerprint')
                or any(detail.get(key)!=value for key,value in detail_panel.items())
                or not all(detail.get('models',{}).get(name,{}).get('summary',{}).get('frames')==detail_panel['frames']
                    for name in('current','frozen_background','frozen_atlas_parent'))):
            raise ValueError('V3 validation changed its pinned detail panel or matched background ablation')


def _preview(cases,report,step):
    protocol=_objective_protocol(report.get('objective_protocol'))
    result=dict(cases=cases,metadata=dict(kind=PREVIEW_KIND,objective_protocol=protocol,step=step,
        input_kind='npr_rgb',source_held_out=True,world_rollout=False,source_input_allowed_for_sr=False,
        source_targets_are_labels_only=True,inference_rgb_only=True,Source_or_RSSM_hint_as_input=False,
        new_model_uses_fixed_atlas=False,row_count=4,
        row_labels=['true Source128','actual final NPR32','frozen atlas baseline128','learned RGB-only enemy128'],
        panel_fingerprint=report['panel_fingerprint'],data_identity_sha256=report['data_identity_sha256'],
        frozen_reference_model_sha256=report['frozen_reference_model_sha256'],
        model_config_sha256=report['model_config_sha256']))
    if protocol==DETAIL_PROTOCOL:
        result['metadata'].update(row_count=5,background_ablation=deepcopy(BACKGROUND_ABLATION),
            detail_animation_panel_fingerprint=report['detail_validation']['fingerprint'],
            detail_animation_panel_frames=report['detail_validation']['frames'])
        result['metadata']['row_labels'].append('matched frozen background128')
    return result


def _measure_detail_panel(networks,data,receipt,*,batch_size):
    """Independent held-out animation metrics; never alters the primary score."""
    names=('current','frozen_background','frozen_atlas_parent');rows={name:[]for name in names};metadata=[]
    panel=receipt['indices']
    for first in range(0,len(panel),batch_size):
        ids=panel[first:first+batch_size];batch=data.batch(len(ids),indices=ids)
        low=batch['lr_npr'].detach();target=batch['hr_source'].detach();mask=batch['mask_hr'].detach()
        outputs=_with_matched_background({name:_forward(net,low,current=name=='current')
            for name,net in networks.items()},objective_protocol=DETAIL_PROTOCOL)
        states=batch['state'].detach().cpu();phases=enemy_animation_index(states)
        for i,index in enumerate(ids):
            metadata.append(dict(row_index=index,alive=bool(states[i,12]>0 and states[i,13]>=.5),
                player_alive=bool(states[i,4]>0),hr_visible=bool(mask[i].any()),
                native_area=int(batch['mask_lr'][i].sum()),animation_label=int(phases[i]),
                distance_tiles=float(((states[i,8:10]-states[i,:2])*16).norm())))
        for name,output in outputs.items():
            values=_rows(output['image'],target,mask,batch['hr_background'],batch['mask_lr'])
            for row,extra in zip(values,_mask_rows(output,batch,current=name=='current')):row.update(extra)
            rows[name].extend(values)
    cohorts=_cohorts(metadata)
    return {**deepcopy(receipt), 'panel_fingerprint':receipt['fingerprint'],
        'objective_protocol':DETAIL_PROTOCOL, 'models':{name:dict(summary=_summary(values,[True]*len(values)),
            cohorts={key:_summary(values,selected)for key,selected in cohorts.items()})for name,values in rows.items()},
        'primary_validation_score_unchanged':True,'world_rollout':False,
        'scope':'separate_existing_VAL_animation_panel_not_training_or_primary_score',
        'background_ablation':deepcopy(BACKGROUND_ABLATION)}


@torch.no_grad()
def evaluate_neural_enemy_restoration(model,reference,data,panel,*,step,batch_size=8,baseline_report=None,
                                      objective_fn=None,objective_weights=None,objective_protocol=PROTOCOL,
                                      detail_panel=None):
    if type(step)is not int or step<0 or type(batch_size)is not int or batch_size<1:
        raise ValueError('Restoration validation needs nonnegative completed step and positive batch')
    panel=list(map(int,panel))
    if getattr(data,'split',None)!='val'or not panel or len(set(panel))!=len(panel):
        raise ValueError('Restoration validation needs unique held-out VAL rows')
    objective_protocol=_objective_protocol(objective_protocol)
    if objective_protocol in(FOREGROUND_PROTOCOL,DETAIL_PROTOCOL)and(not callable(objective_fn)
            or not isinstance(objective_weights,dict)or not objective_weights):
        raise ValueError('Foreground/detail validation requires explicit objective_fn and objective_weights')
    detail_receipt=_validate_detail_panel(data,detail_panel)if objective_protocol==DETAIL_PROTOCOL else None
    if objective_fn is None:
        from .neural_enemy_restoration_objectives import neural_enemy_restoration_loss,LOSS_WEIGHTS
        objective_fn=neural_enemy_restoration_loss
        if objective_weights is None:objective_weights=LOSS_WEIGHTS
    weights=dict(objective_weights or {})
    networks,device=_networks(model,reference);identity=_identities(model,reference,data,panel)
    if baseline_report is not None:
        _check_report(baseline_report,identity,objective_protocol=objective_protocol,detail_panel=detail_receipt)
        if baseline_report['step']>step:raise ValueError('Restoration reference report is from a future step')
    temporal=build_temporal_panel(data,panel);needed={index for pair in temporal['pairs']for index in(pair['previous_index'],pair['current_index'])}
    model_names=('current','frozen_background','frozen_atlas_parent')if objective_protocol==DETAIL_PROTOCOL else MODELS
    metadata=[];rows={name:[]for name in model_names};images={name:{}for name in model_names};targets={};masks={};cases={};detail_validation=None
    objective_sum=0.;components={name:dict(raw=0.,weighted=0.,weight=weight)for name,weight in weights.items()}
    sampler=deepcopy(data.sampler_state());modes=[(module,module.training)for net in networks.values()for module in net.modules()]
    hashes={name:_hash_tensors(net.state_dict())for name,net in networks.items()}
    try:
        for net in networks.values():net.eval()
        with preserve_preview_rng(device,getattr(data,'rng',None)),strict_render_precision(),torch.inference_mode(),torch.autocast(device.type,enabled=False):
            for first in range(0,len(panel),batch_size):
                ids=panel[first:first+batch_size];batch=data.batch(len(ids),indices=ids)
                low=batch['lr_npr'].detach();target=batch['hr_source'].detach();mask=batch['mask_hr'].detach()
                if low.shape!=(len(ids),3,32,32)or target.shape!=(len(ids),3,128,128)or mask.shape!=(len(ids),1,128,128):
                    raise ValueError('Expected exact cached final NPR32, Source128 and native HR masks')
                outputs=_with_matched_background({name:_forward(net,low,current=name=='current')
                    for name,net in networks.items()},objective_protocol=objective_protocol)
                state=batch['state'].detach().cpu();animation=enemy_animation_index(state)
                local=[]
                for index,row_index in enumerate(ids):
                    local.append(dict(row_index=row_index,alive=bool(state[index,12]>0 and state[index,13]>=.5),
                        player_alive=bool(state[index,4]>0),hr_visible=bool(mask[index].any()),
                        native_area=int(batch['mask_lr'][index].sum()),animation_label=int(animation[index]),
                        distance_tiles=float(((state[index,8:10]-state[index,:2])*16).norm())))
                    if row_index in needed:targets[row_index]=target[index].float().cpu().clone();masks[row_index]=mask[index].bool().cpu().clone()
                metadata.extend(local);cohorts=_cohorts(local);picked={}
                for role in ROLES:
                    if role in cases:continue
                    index=next((i for i,yes in enumerate(cohorts[role])if yes),None)
                    if index is not None:
                        picked[role]=index;cases[role]=dict(row_index=ids[index],source=_pixels(target[index]),npr=_pixels(low[index]),predictions={})
                for name,output in outputs.items():
                    values=_rows(output['image'],target,mask,batch['hr_background'],batch['mask_lr'])
                    for value,extra in zip(values,_mask_rows(output,batch,current=name=='current')):value.update(extra)
                    rows[name].extend(values)
                    for index,row_index in enumerate(ids):
                        if row_index in needed:images[name][row_index]=output['image'][index].float().cpu().clone()
                    for role,index in picked.items():
                        cases[role]['predictions'][name]=_pixels(output['image'][index])
                        if name=='current':
                            cases[role]['enemy_rgb']=_pixels(output['enemy_rgb'][index])
                            cases[role]['alpha']=_pixels(output['alpha'][index].expand(3,-1,-1))
                loss,metrics=objective_fn(outputs['current'],batch)
                if any(metrics.get(key,objective_protocol)!=objective_protocol
                       for key in('objective_protocol','loss_protocol')):
                    raise ValueError('Restoration measured loss objective protocol differs from selected validation protocol')
                objective_sum+=float(loss)*len(ids)
                for name,total in components.items():
                    total['raw']+=metrics[name]*len(ids);total['weighted']+=metrics[name+'_weighted']*len(ids)
            if detail_receipt is not None:
                detail_validation=_measure_detail_panel(networks,data,detail_receipt,batch_size=batch_size)
    finally:
        data.restore_sampler(sampler)
        for module,training in modes:module.training=training
    if any(_hash_tensors(net.state_dict())!=hashes[name]for name,net in networks.items()):
        raise AssertionError('Restoration validation changed model/reference parameters or buffers')
    cohorts=_cohorts(metadata)
    models={name:dict(summary=_summary(values,[True]*len(values)),
        cohorts={cohort:_summary(values,selected)for cohort,selected in cohorts.items()})for name,values in rows.items()}
    models['current']['summary'].update(reconstruction_loss=objective_sum/len(panel),reconstruction_frames=len(panel))
    temporal['arms']={name:_temporal_summary(temporal['pairs'],images[name],targets,masks)for name in model_names}
    summary=models['current']['summary'];ssim=summary['ssim_mean'];mae=summary['mae_mean']
    report=dict(kind=KIND,objective_protocol=objective_protocol,step=step,frames=len(panel),input_kind='npr_rgb',
        source_held_out=True,world_rollout=False,source_input_allowed_for_sr=False,source_targets_are_labels_only=True,
        Source_or_RSSM_hint_as_input=False,source_future_as_input=False,inference_rgb_only=True,new_model_uses_fixed_atlas=False,
        **identity,models=models,temporal=temporal,
        reference_scope='frozen_atlas_parent_same_actual_NPR_inputs_not_a_matched_training_control',
        mask_scope='actual_SourceHR_visible_enemy_pixels_including_hit_and_death',
        raw_RGB_metric_scope='true-mask_RGB_error_without_predicted_alpha_gating; SSIM_in_Source_mask_bbox_crop_with_true_target_context_outside_mask',
        empty_alpha_threshold=.05,empty_dense_alpha_threshold=.5,empty_topk_pixels=EMPTY_TOPK_PIXELS,
        empty_localization_scope='per_true_empty_native128_frame; soft/dense_opacity_presence_is_not_a_whole_enemy_detection; paint_is_per_pixel_mean_abs_RGB_change_from_provided_background',
        preview_cases={role:case['row_index']for role,case in cases.items()},
        validation_objective=dict(protocol=objective_protocol,loss=objective_sum/len(panel),frames=len(panel),input_kind='npr_rgb',
            source_held_out=True,batch_size=batch_size,aggregation='image_count_weighted_mean_of_actual_batch_objectives',
            components={name:dict(raw=value['raw']/len(panel),weighted=value['weighted']/len(panel),weight=value['weight'])for name,value in components.items()}),
        quality_goal=dict(passed=ssim>.91 and mae*255<2.5,ssim_strictly_greater_than=.91,
            mae_255_strictly_less_than=2.5,observed_ssim=ssim,observed_mae_255=mae*255,
            gameplay_qualified=False,budget_completion_is_quality=False),gameplay_qualified=False,automatic_deployment=False)
    if detail_validation is not None:
        report.update(detail_validation=detail_validation,background_ablation=deepcopy(BACKGROUND_ABLATION))
    json.dumps(report,allow_nan=False)
    return report,_preview(cases,report,step)


@torch.no_grad()
def preview_neural_enemy_restoration(model,reference,data,panel,*,step,baseline_report,batch_size=8,
                                     objective_protocol=PROTOCOL,detail_panel=None):
    if type(step)is not int or step<0 or type(batch_size)is not int or batch_size<1:
        raise ValueError('Invalid light restoration preview clock/batch')
    panel=list(map(int,panel));networks,device=_networks(model,reference)
    if getattr(data,'split',None)!='val'or not panel or len(set(panel))!=len(panel):raise ValueError('Preview requires unique held-out VAL rows')
    objective_protocol=_objective_protocol(objective_protocol)
    detail_receipt=_validate_detail_panel(data,detail_panel)if objective_protocol==DETAIL_PROTOCOL else None
    _check_report(baseline_report,_identities(model,reference,data,panel),objective_protocol=objective_protocol,
        detail_panel=detail_receipt)
    if baseline_report['step']>step:raise ValueError('Restoration reference report is from a future step')
    selected=[(role,baseline_report['preview_cases'][role])for role in ROLES if role in baseline_report['preview_cases']]
    if any(index not in panel for _,index in selected):raise ValueError('Preview case is outside its pinned VAL panel')
    sampler=deepcopy(data.sampler_state());modes=[(module,module.training)for net in networks.values()for module in net.modules()];cases={}
    try:
        for net in networks.values():net.eval()
        with preserve_preview_rng(device,getattr(data,'rng',None)),strict_render_precision(),torch.inference_mode(),torch.autocast(device.type,enabled=False):
            for first in range(0,len(selected),batch_size):
                chosen=selected[first:first+batch_size];batch=data.batch(len(chosen),indices=[index for _,index in chosen]);low=batch['lr_npr'].detach()
                outputs=_with_matched_background({name:_forward(net,low,current=name=='current')
                    for name,net in networks.items()},objective_protocol=objective_protocol)
                for index,(role,row_index)in enumerate(chosen):
                    current=outputs['current'];cases[role]=dict(row_index=row_index,source=_pixels(batch['hr_source'][index]),
                        npr=_pixels(low[index]),predictions={name:_pixels(output['image'][index])for name,output in outputs.items()},
                        enemy_rgb=_pixels(current['enemy_rgb'][index]),alpha=_pixels(current['alpha'][index].expand(3,-1,-1)))
    finally:
        data.restore_sampler(sampler)
        for module,training in modes:module.training=training
    preview=_preview(cases,baseline_report,step)
    preview['metadata'].update(preview_only=True,reference_validation_step=baseline_report['step'])
    return preview


def save_neural_enemy_restoration_preview(preview,path,*,every=15):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);meta=preview['metadata']
    protocol=_objective_protocol(meta.get('objective_protocol'));row_count=5 if protocol==DETAIL_PROTOCOL else 4
    if meta.get('kind')!=PREVIEW_KIND or meta.get('row_count')!=row_count:raise ValueError('Restoration preview row count differs from its objective protocol')
    canvas=Image.new('RGB',(1024,row_count*280),(14,22,31));draw=ImageDraw.Draw(canvas)
    for row,label in enumerate(meta['row_labels']):
        draw.text((4,row*280+4),label,fill=(235,239,245))
        for column,role in enumerate(ROLES):
            case=preview['cases'].get(role)
            if case is None:continue
            images=[case['source'],case['npr'],case['predictions']['frozen_atlas_parent'],case['predictions']['current']]
            if row_count==5:images.append(case['predictions']['frozen_background'])
            pixels=images[row]
            canvas.paste(Image.fromarray(np.asarray(pixels)).resize((256,256),Image.Resampling.NEAREST),(column*256,row*280+24))
            draw.text((column*256+4,row*280+13),role+' | row '+str(case['row_index']),fill=(220,225,235))
    temporary=path.with_name(path.stem+'.tmp.png');canvas.save(temporary,format='PNG');replace_file(temporary,path)
    result={**meta,'path':str(path),'every':every,'generated_at':now(),'preview_image_size':[1024,row_count*280],
        'caption':'Actual final NPR32 to learned RGB/alpha128; same Source128 and frozen atlas baseline'}
    atomic_json(path.with_suffix('.json'),result)
    return result
