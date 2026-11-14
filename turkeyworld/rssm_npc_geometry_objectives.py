"""Opt-in NPC footprint/swept-wall loss, without inference corrections."""
from copy import deepcopy
import json
import torch
from .trajectory_objectives import circle_wall_penalty

NPC_NAVIGATION_PROTOCOL=dict(version=1,protocol='rssm_npc_own_aligned_geometry_v1',
    geometry=dict(version=1,radius=.25,wall_weight=.25,endpoint_fraction=.5,path_fraction=.5,
        samples_per_leg=3,axis_order='x_then_y',normalization='squared_penetration_over_radius_squared',
        source_future_feedback=False,inference_projection=False),
    capture=dict(every=16,action_horizon=32,prediction_horizon=16,memory_provenance='actualownprefix'),
    recorded_velocity_direction=False,own_aligned_local_active=True)


def normalize_npc_navigation_protocol(protocol):
    if protocol is None:return None
    if not isinstance(protocol,dict)or json.dumps(protocol,sort_keys=True,allow_nan=False)!=json.dumps(NPC_NAVIGATION_PROTOCOL,sort_keys=True):
        raise ValueError('NPC navigation requires its explicit own-aligned geometry protocol')
    return deepcopy(protocol)


def prepare_npc_geometry(previous_source,target,grid,*,npc_navigation_protocol):
    protocol=normalize_npc_navigation_protocol(npc_navigation_protocol)
    if protocol is None:return None
    if(previous_source.ndim!=3 or previous_source.shape[-1]!=20 or target.shape!=previous_source.shape
            or not previous_source.shape[0]or not previous_source.shape[1]
            or grid.shape!=(len(target),16,16)
            or not all(bool(torch.isfinite(value).all())for value in(previous_source,target,grid))):
        raise ValueError('NPC geometry requires finite full Source tapes and matching maps')
    alive=((previous_source[...,12]>0)&(previous_source[...,13]>=.5)
        &(target[...,12]>0)&(target[...,13]>=.5)).detach()
    return dict(config=protocol['geometry'],alive=alive,denominator=alive.sum().clamp_min(1),
        grid=grid.detach(),horizon=target.shape[1],source_alive_transitions=int(alive.sum()))


def npc_geometry_terms(predictions,own_previous,supervision,*,first=0,predicted_respawns=None):
    if(supervision is None or predictions.ndim!=3 or predictions.shape[-1]!=20
            or predictions.shape[0]!=supervision['alive'].shape[0]or predictions.shape[1]<1
            or own_previous.shape!=(len(predictions),20)or type(first)is not int or first<0
            or first+predictions.shape[1]>supervision['horizon']
            or not bool(torch.isfinite(predictions).all())or not bool(torch.isfinite(own_previous).all())):
        raise ValueError('NPC geometry requires a finite own chunk and its numerical predecessor')
    batch,horizon=predictions.shape[:2]
    if predicted_respawns is None:
        respawn=torch.zeros((batch,horizon),dtype=torch.bool,device=predictions.device)
    else:
        if(predicted_respawns.shape!=(batch,horizon,1)or predicted_respawns.dtype!=torch.bool
                or predicted_respawns.device!=predictions.device):
            raise ValueError('NPC swept loss requires actual bool respawn flags[B,H,1]')
        respawn=predicted_respawns[...,0].detach()
    cfg=supervision['config'];points=predictions[...,8:10].float()*16
    old=torch.cat([own_previous.detach()[:,None,8:10].float(),predictions[:,:-1,8:10].float()],1)*16
    elbow=torch.stack([points[...,0],old[...,1]],-1)
    fractions=torch.arange(1,cfg['samples_per_leg']+1,device=points.device,dtype=points.dtype)/(cfg['samples_per_leg']+1)
    samples=torch.cat([old[:,:,None]+fractions[None,None,:,None]*(elbow-old)[:,:,None],elbow[:,:,None],
        elbow[:,:,None]+fractions[None,None,:,None]*(points-elbow)[:,:,None]],2)
    endpoint=circle_wall_penalty(points,supervision['grid'],cfg['radius'])
    swept=circle_wall_penalty(samples,supervision['grid'],cfg['radius']).mean(2)
    source=supervision['alive'][:,first:first+horizon];path_mask=source&~respawn
    denominator=supervision['denominator'];scale=cfg['radius']**2
    endpoint_raw=(endpoint*source).sum()/denominator;swept_raw=(swept*path_mask).sum()/denominator
    endpoint_loss=cfg['wall_weight']*cfg['endpoint_fraction']*endpoint_raw/scale
    path_loss=cfg['wall_weight']*cfg['path_fraction']*swept_raw/scale;loss=endpoint_loss+path_loss
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite NPC geometry objective')
    own_present=(predictions[...,12]>0)&(predictions[...,13]>=.5)
    metrics=dict(npc_wall_endpoint_penetration_squared=float(endpoint_raw.detach()),
        npc_wall_path_penetration_squared=float(swept_raw.detach()),npc_wall_endpoint_loss=float(endpoint_loss.detach()),
        npc_wall_path_loss=float(path_loss.detach()),npc_wall_loss=float(loss.detach()),
        npc_wall_overlap_transitions=int(((endpoint.detach()>1e-12)&source).sum()),
        npc_wall_path_overlap_transitions=int(((swept.detach()>1e-12)&path_mask).sum()),
        npc_wall_predicted_spawn_path_exclusions=int((source&respawn).sum()),
        npc_wall_source_alive_model_missing_transitions=int((source&~own_present).sum()),
        npc_wall_source_alive_transitions=supervision['source_alive_transitions'],
        npc_wall_axis_order='x_then_y',npc_wall_radius=.25,npc_wall_radius_squared_normalized=True,
        npc_wall_source_future_feedback=False,npc_wall_inference_projection=False)
    return loss,metrics


def add_npc_geometry_metrics(total,part):
    additive=('npc_wall_endpoint_penetration_squared','npc_wall_path_penetration_squared',
        'npc_wall_endpoint_loss','npc_wall_path_loss','npc_wall_loss','npc_wall_overlap_transitions',
        'npc_wall_path_overlap_transitions','npc_wall_predicted_spawn_path_exclusions',
        'npc_wall_source_alive_model_missing_transitions')
    for key,value in part.items():total[key]=total.get(key,0)+value if key in additive else value
    return total
