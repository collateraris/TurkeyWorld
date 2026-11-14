"""Held-out raw Base and frozen-codec condition-quality reports; no RF calls."""
from contextlib import contextmanager
from copy import deepcopy
import math

import torch

from .base_geometry_bundle import PROTOCOL
from .decoder_rf_adaptation_validation import _identity,_plain,_preservation,_geometry_preservation
from .renderer_enemy_validation import renderer_metrics
from .renderer_geometry_objectives import geometry_wall_terms,geometry_wall_protocol
from .renderer_moe_objectives import moe_conditions
from .renderer_student import _hash_tensors
from .renderer_wall_depth import source_wall_depth_labels
from .renderer_wall_supervision import source_wall_masks,wall_loss_terms
from .staged_validation import _isolated_validation

KIND='base_geometry_validation_v1'
PREVIEW_KIND='base_geometry_GT_query'
BRANCHES=('base_old','base_new','base_codec_old','base_codec_new','source_codec')


@contextmanager
def _readonly(model,reference,data):
    if getattr(data,'split',None) not in ('val','test'):
        raise ValueError('Base geometry validation requires held-out Source val/test data')
    if (any(p.requires_grad and not n.startswith('base_student.') for n,p in model.named_parameters())
            or any(p.requires_grad for p in reference.parameters())
            or any(getattr(n.flow_renderer,'conditional_restorer',None) is not None for n in (model,reference))):
        raise ValueError('Only current Base may train; no restorer or trainable RF/codec/world is allowed')
    flows=[_hash_tensors(n.flow_renderer.state_dict()) for n in (model,reference)]
    if len(set(flows)) != 1:
        raise ValueError('Base comparisons need byte-identical frozen RF/codec/private/noise/normalization')
    hashes=[(_hash_tensors(n.flow_renderer.state_dict()),_hash_tensors(n.base_student.state_dict())) for n in (model,reference)]
    sampler=deepcopy(data.sampler_state()) if callable(getattr(data,'sampler_state',None)) else None
    try:
        with _isolated_validation(model,data,173),_isolated_validation(reference,data,173):yield flows[0]
    finally:
        if sampler is not None:data.restore_sampler(sampler)
        if hashes != [(_hash_tensors(n.flow_renderer.state_dict()),_hash_tensors(n.base_student.state_dict())) for n in (model,reference)]:
            raise RuntimeError('Read-only Base validation changed weights or buffers')


def _collect(model,reference,data,panel,batch_size,objective_fn=None):
    from .base_geometry_objectives import base_geometry_forward
    parts={k:[] for k in (*BRANCHES,'source','mask','background','states','map','dynamic')}
    frames,total,sums=0,0.,{}
    for first in range(0,len(panel),batch_size):
        ids=panel[first:first+batch_size];batch=data.batch(len(ids),indices=ids)
        current=moe_conditions(model,batch);old=moe_conditions(reference,batch)
        raw,_=base_geometry_forward(model,current)
        previous,_=base_geometry_forward(reference,old)
        codec=model.flow_renderer.codec
        def reconstruct(rgb):return codec.decode(codec.encode(rgb)['quantized'])
        values=dict(base_old=previous,base_new=raw,base_codec_old=reconstruct(previous),
            base_codec_new=reconstruct(raw),source_codec=reconstruct(current['target'].detach()),
            source=current['target'],mask=current['object_mask'],background=current['background'],
            states=current['state'],map=current['map'],dynamic=current['dynamic'])
        if any(not bool(torch.isfinite(v).all()) for v in values.values()):
            raise FloatingPointError('Nonfinite Base/codec validation output')
        for name,value in values.items():parts[name].append(value.detach().cpu())
        if objective_fn is not None:
            value,metrics=objective_fn(model,batch,kind='main');loss=float(value.detach())
            if not math.isfinite(loss):raise FloatingPointError('Nonfinite held-out Base MAIN objective')
            total+=len(ids)*loss;frames+=len(ids)
            def visit(item,path=()):
                if isinstance(item,dict):
                    for k,v in item.items():visit(v,(*path,k))
                elif type(item)in(int,float):
                    if not math.isfinite(item):raise FloatingPointError('Nonfinite MAIN component')
                    support=path[-1].endswith(('_images','_pixels'))
                    sums[path]=sums.get(path,0.)+item*(1 if support else len(ids))
            visit(metrics)
    arrays={k:torch.cat(v)for k,v in parts.items()};arrays['categories']=[str(data.rows[int(i)]['category'])for i in panel]
    arrays['row_indices']=list(panel);components={}
    for path,value in sums.items():
        row=components
        for key in path[:-1]:row=row.setdefault(key,{})
        row[path[-1]]=int(value)if path[-1].endswith(('_images','_pixels'))else value/frames
    return arrays,dict(loss=total/frames if frames else None,frames=frames,components=components)


def _measure(arrays):
    labels=source_wall_masks(arrays['map'],arrays['states'],arrays['mask']);models={}
    for name in BRANCHES:
        raw=renderer_metrics(arrays[name],arrays['source'],arrays['mask'],arrays['background'],arrays['categories'],arrays['dynamic'])
        wall=_plain(wall_loss_terms(arrays[name],arrays['source'],labels))
        summary=dict(raw['overall'],wall_boundary_l1=wall['boundary_l1']if wall['boundary_images']else None,
                     wall_boundary_frames=wall['boundary_images'])
        models[name]=dict(summary=summary,overall=deepcopy(summary),cohorts=raw['categories'],
            visible_enemy=raw['visible_enemy'],wall=wall,static_zero_features=raw['static_zero_features'],
            dynamic_past_prior_features=raw['dynamic_past_prior_features'])
    return models


def evaluate_base_geometry(model,reference_model,data,panel,geometry_panel=None,*,step,
                           batch_size=8,baseline=None,objective_fn=None):
    from .base_geometry_objectives import base_geometry_refinement_loss
    panel=[int(i)for i in panel];geometry_panel=None if geometry_panel is None else[int(i)for i in geometry_panel]
    if (type(step)is not int or step<0 or type(batch_size)is not int or batch_size<1
            or not panel or len(set(panel))!=len(panel)
            or geometry_panel is not None and(not geometry_panel or len(set(geometry_panel))!=len(geometry_panel))):
        raise ValueError('Base validation needs unique nonempty panels and positive batch size')
    identity,identity_sha=_identity(data);panel_sha=data.panel_fingerprint(panel)
    geo_sha=data.panel_fingerprint(geometry_panel)if geometry_panel is not None else None
    codec_sha=_hash_tensors(model.flow_renderer.codec.state_dict())
    with _readonly(model,reference_model,data)as flow_sha:
        if baseline is not None and any(baseline.get(k)!=v for k,v in dict(kind=KIND,protocol=PROTOCOL,
            panel_fingerprint=panel_sha,geometry_panel_fingerprint=geo_sha,data_identity_sha256=identity_sha,
            codec_sha256=codec_sha,frozen_flow_sha256=flow_sha).items()):
            raise ValueError('Base baseline must use identical held-out panels/data/frozen RF and current codec')
        arrays,objective=_collect(model,reference_model,data,panel,batch_size,objective_fn or base_geometry_refinement_loss)
        models=_measure(arrays);geometry=None
        if geometry_panel is not None:
            a,_=_collect(model,reference_model,data,geometry_panel,batch_size)
            labels=source_wall_depth_labels(a['map'],a['states'],a['mask'])
            measured={name:dict(bands=_plain(geometry_wall_terms(a[name],a['source'],labels)))for name in BRANCHES}
            geometry=dict(frames=len(geometry_panel),panel_fingerprint=geo_sha,protocol=geometry_wall_protocol(),
                models=measured,raw_base_preservation=_geometry_preservation(measured['base_new'],measured['base_old']),
                codec_base_preservation=_geometry_preservation(measured['base_codec_new'],measured['base_codec_old']))
    if baseline is not None and any(models[k]!=baseline['models'][k]for k in('base_old','base_codec_old','source_codec')):
        raise ValueError('Frozen old Base/current codec reference changed from the initial report')
    objective.update(protocol=PROTOCOL,input_kind='expanded_Base_raw_RGB_frozen_encoder_latent_target',
        aggregation='batch_image_weighted',source_held_out=True,main_only=True,
        matches_training_formula=True,excludes_far_geometry_auxiliary=True,Source_Q_image_branch=False)
    report=dict(kind=KIND,protocol=PROTOCOL,step=step,frames=len(panel),native_resolution=32,
        panel_fingerprint=panel_sha,geometry_panel_fingerprint=geo_sha,data_identity=identity,
        data_identity_sha256=identity_sha,codec_sha256=codec_sha,frozen_flow_sha256=flow_sha,
        models=models,summary=models['base_new']['summary'],geometry=geometry,
        raw_base_preservation=_preservation(models['base_new']['summary'],models['base_old']['summary']),
        codec_base_preservation=_preservation(models['base_codec_new']['summary'],models['base_codec_old']['summary']),
        source_preservation=dict(passed=True,frozen_Source_codec=True),
        val_loss=objective['loss'],validation_objective=objective,validation_objective_matches_main_training=True,
        best_metric='models.base_new.summary.ssim_mean',source_held_out=True,renderer_only=True,
        source_sampler_unchanged=True,RF_forward_calls=0,RF_tensors_unchanged=True,full_codec_tensors_unchanged=True,
        restorer_active=False,Source_Q_image_branch=False,Source_RGB_into_Base_conditioning=False,
        future_source_rgb_in_prior=False,future_source_pose_in_prior=False,world_rollout=False,
        autonomous_rollout=False,sr_quality_evaluated=False,gameplay_qualified=False,automatic_active_promotion=False,
        passed=False,quality_passed=False,
        scope='Native32 raw Base geometry and frozen E/Q/D condition quality; no RF or restorer forward',
        budget_completion_is_quality=False,RF_adapted_to_new_Base=False)
    if baseline is not None:report['baseline_step']=baseline['step']
    arrays.update(panel_fingerprint=panel_sha,step=step)
    return report,arrays


def save_base_geometry_preview(arrays,path,step):
    from pathlib import Path
    from PIL import Image,ImageDraw
    chosen=[]
    for category in('large','partial','distant','empty_navigation'):
        index=next((i for i,c in enumerate(arrays['categories'])if c==category and i not in chosen),None)
        if index is not None:chosen.append(index)
    chosen +=[i for i in range(len(arrays['categories']))if i not in chosen][:4-len(chosen)]
    rows=[('source','SOURCE32'),('base_old','FROZEN BASE29209'),('base_new','NEW EXPANDED BASE'),
          ('base_codec_old','OLD BASE E/Q → D1725'),('base_codec_new','NEW BASE E/Q → SAME D1725')]
    image=Image.new('RGB',(192*len(chosen),217*5),(12,17,25));draw=ImageDraw.Draw(image)
    for row,(key,title)in enumerate(rows):
        for col,index in enumerate(chosen):
            pixels=arrays[key][index].detach().float().clamp(0,1).mul(255).round().byte().permute(1,2,0).numpy()
            image.paste(Image.fromarray(pixels).resize((192,192),Image.Resampling.NEAREST),(col*192,row*217+25))
            draw.text((col*192+3,row*217+3),f'{title} g{step} | {arrays["categories"][index]}',fill='white')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp.png');image.save(temporary);temporary.replace(path)
    return dict(kind=PREVIEW_KIND,protocol=PROTOCOL,step=step,panel_fingerprint=arrays['panel_fingerprint'],
        row_count=5,rows=[k for k,_ in rows],row_labels=[v for _,v in rows],
        row_indices=[arrays['row_indices'][i]for i in chosen],columns=[arrays['categories'][i]for i in chosen],
        native_resolution=32,source_held_out=True,renderer_only=True,world_rollout=False,
        future_source_rgb_in_prior=False,future_source_pose_in_prior=False,
        restorer_active=False,RF_forward_calls=0,sr_quality_evaluated=False,gameplay_qualified=False,
        enlargement='nearest_neighbor_display_only')
