"""Opt-in train-only player footprint and X-then-Y swept-wall supervision.

The map is observed static input. Source states determine lifetime masks only;
all penalized coordinates are the autonomous prediction, with no state repair.
"""
from copy import deepcopy

import torch

from .trajectory_objectives import circle_wall_penalty


PLAYER_GEOMETRY_PROTOCOL = dict(version=1, radius=.20, wall_weight=.25,
    endpoint_fraction=.5, path_fraction=.5, samples_per_leg=3,
    axis_order='x_then_y', normalization='squared_penetration_over_radius_squared',
    source_future_feedback=False, inference_projection=False)


def player_geometry_protocol():return deepcopy(PLAYER_GEOMETRY_PROTOCOL)


def player_navigation_tbptt(protocol):
    if protocol is not None and not isinstance(protocol,dict):raise ValueError('Player navigation protocol must be a mapping')
    value=16 if protocol is None else protocol.get('tbptt',16)
    if type(value)is not int or value not in(16,64,450):raise ValueError('Player navigation TBPTT must be explicit16/64/450')
    return value


def capture_player_chunks(callback,captures,predictions,actions,grid,*,first):
    """Capture every16 ticks independently of the differentiable graph length."""
    if callback is None:return
    for tick,state,memory in captures:
        offset=tick-first
        callback(tick,state,memory,actions[:,tick:tick+32],grid,predictions[:,offset:offset+16])


def normalize_player_navigation_geometry(protocol):
    if protocol is None:return None
    if not isinstance(protocol,dict):raise ValueError('Player navigation protocol must be a mapping')
    geometry=protocol.get('geometry',protocol)
    if (not isinstance(geometry,dict)or set(geometry)!=set(PLAYER_GEOMETRY_PROTOCOL)
            or any(type(geometry[k])is not type(v)or geometry[k]!=v for k,v in PLAYER_GEOMETRY_PROTOCOL.items())):
        raise ValueError('Player geometry requires its exact radius-normalized X-then-Y protocol')
    return deepcopy(geometry)


def prepare_player_geometry(previous_source,target,grid,*,player_navigation_protocol):
    config=normalize_player_navigation_geometry(player_navigation_protocol)
    if config is None:return None
    if(previous_source.ndim!=3 or previous_source.shape[-1]!=20 or target.shape!=previous_source.shape
            or not previous_source.shape[0]or not previous_source.shape[1]
            or grid.shape!=(len(target),16,16)
            or not all(bool(torch.isfinite(v).all())for v in(previous_source,target,grid))):
        raise ValueError('Player geometry needs full finite Source tapes and matching known16x16 maps')
    alive=((previous_source[...,4].detach()>0)&(target[...,4].detach()>0)).detach()
    return dict(config=config,alive=alive,denominator=alive.sum().clamp_min(1),
        grid=grid.detach(),horizon=target.shape[1],source_alive_transitions=int(alive.sum()))


def player_geometry_terms(predictions,own_previous,supervision,*,first=0):
    """Return one TBPTT chunk's additive loss and detached scalar diagnostics.

    The full-tape denominator is prepared once, so summing16-step chunks is
    identical to measuring the same numerical trajectory in one call.
    """
    if (supervision is None or predictions.ndim!=3 or predictions.shape[-1]!=20
            or predictions.shape[0]!=supervision['alive'].shape[0]or predictions.shape[1]<1
            or own_previous.shape!=(len(predictions),20)
            or type(first)is not int or first<0
            or first+predictions.shape[1]>supervision['horizon']
            or not bool(torch.isfinite(predictions).all())or not bool(torch.isfinite(own_previous).all())):
        raise ValueError('Player geometry requires a finite own trajectory chunk and actual numerical predecessor')
    cfg=supervision['config'];points=predictions[...,:2].float()*16
    old=torch.cat((own_previous.detach()[:,None,:2].float(),predictions[:,:-1,:2].float()),1)*16
    elbow=torch.stack((points[...,0],old[...,1]),-1)
    fractions=torch.arange(1,cfg['samples_per_leg']+1,device=points.device,dtype=points.dtype)/(cfg['samples_per_leg']+1)
    x_leg=old[:,:,None]+fractions[None,None,:,None]*(elbow-old)[:,:,None]
    y_leg=elbow[:,:,None]+fractions[None,None,:,None]*(points-elbow)[:,:,None]
    samples=torch.cat((x_leg,elbow[:,:,None],y_leg),2)
    endpoint=circle_wall_penalty(points,supervision['grid'],cfg['radius'])
    swept=circle_wall_penalty(samples,supervision['grid'],cfg['radius']).mean(2)
    mask=supervision['alive'][:,first:first+points.shape[1]].to(endpoint.dtype)
    denominator=supervision['denominator']
    endpoint_raw=(endpoint*mask).sum()/denominator
    swept_raw=(swept*mask).sum()/denominator
    scale=cfg['radius']**2
    endpoint_loss=cfg['wall_weight']*cfg['endpoint_fraction']*endpoint_raw/scale
    swept_loss=cfg['wall_weight']*cfg['path_fraction']*swept_raw/scale
    total=endpoint_loss+swept_loss
    if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite player geometry objective')
    metrics=dict(player_wall_endpoint_penetration_squared=float(endpoint_raw.detach()),
        player_wall_path_penetration_squared=float(swept_raw.detach()),
        player_wall_endpoint_loss=float(endpoint_loss.detach()),player_wall_path_loss=float(swept_loss.detach()),
        player_wall_loss=float(total.detach()),
        player_wall_overlap_transitions=int(((endpoint.detach()>1e-12)&mask.bool()).sum()),
        player_wall_path_overlap_transitions=int(((swept.detach()>1e-12)&mask.bool()).sum()),
        player_wall_source_alive_transitions=supervision['source_alive_transitions'],
        player_wall_axis_order='x_then_y',player_wall_radius=.20,
        player_wall_radius_squared_normalized=True,player_wall_inference_projection=False,
        player_wall_source_future_feedback=False)
    return total,metrics


def add_player_geometry_metrics(total,part):
    """Combine additive chunk metrics without multiplying full-tape counts."""
    for key,value in part.items():
        if key in('player_wall_endpoint_penetration_squared','player_wall_path_penetration_squared',
                   'player_wall_endpoint_loss','player_wall_path_loss','player_wall_loss',
                   'player_wall_overlap_transitions','player_wall_path_overlap_transitions'):
            total[key]=total.get(key,0)+value
        else:total[key]=value
    return total
