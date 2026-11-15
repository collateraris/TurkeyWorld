"""Versioned long-rollout enemy velocity supervision and hard TRAIN selection."""
from copy import deepcopy
import math

import numpy as np
import torch
from torch.nn import functional as F


SETTINGS=dict(protocol='rssm_enemy_drift_v1',version=1,
    velocity_weight=.05,speed_weight=.05,direction_weight=.01,velocity_field_weight=.025,
    hz=30.,motion_epsilon=.05,hard_sampling=True,wall_margin_cells=.5,
    turn_degrees=30.,sampling_order=['near_wall','turn','uniform','uniform'],
    Source_labels_only=True,Source_future_feedback=False)


def validate_settings(settings):
    if settings != SETTINGS:raise ValueError('Enemy drift settings changed without a new revision')
    return deepcopy(settings)


def enemy_velocity_terms(prediction,initial,target,*,respawns=None,settings=SETTINGS):
    validate_settings(settings)
    own_previous=torch.cat((initial[:,None],prediction[:,:-1]),1)
    source_previous=torch.cat((initial[:,None],target[:,:-1]),1).detach()
    alive=((source_previous[...,12]>0)&(source_previous[...,13]>=.5)
        &(target[...,12]>0)&(target[...,13]>=.5)).detach()
    if respawns is not None:alive=alive & ~respawns.squeeze(-1).detach()
    velocity=(prediction[...,8:10].float()-own_previous[...,8:10].float())*(16*settings['hz'])
    reference=(target[...,8:10].detach().float()-source_previous[...,8:10].float())*(16*settings['hz'])
    def mean(values,mask=alive):return (values*mask).sum()/mask.sum().clamp_min(1)
    speed=velocity.norm(dim=-1);source_speed=reference.norm(dim=-1)
    vector=mean(F.smooth_l1_loss(velocity,reference,reduction='none').mean(-1))
    magnitude=mean(F.smooth_l1_loss(speed,source_speed,reduction='none'))
    moving=alive & (source_speed>settings['motion_epsilon'])
    direction=mean(1-F.cosine_similarity(velocity,reference,dim=-1,eps=settings['motion_epsilon']),moving)
    field=mean(F.smooth_l1_loss(prediction[...,10:12].float()*2,reference,reduction='none').mean(-1))
    loss=(settings['velocity_weight']*vector+settings['speed_weight']*magnitude
        +settings['direction_weight']*direction+settings['velocity_field_weight']*field)
    metrics=dict(enemy_long_velocity_loss=float(vector.detach()),enemy_long_speed_loss=float(magnitude.detach()),
        enemy_long_direction_loss=float(direction.detach()),enemy_long_velocity_field_loss=float(field.detach()),
        enemy_long_velocity_weighted=float(loss.detach()),enemy_long_valid_transitions=int(alive.sum()),
        enemy_long_speed_mean=float(mean(speed).detach()),enemy_long_reference_speed_mean=float(mean(source_speed).detach()))
    return loss,metrics


def hard_episode_classes(states,grid,*,horizon,settings=SETTINGS):
    """Inspect recorded TRAIN labels only; never query Source or alter rollout."""
    s=np.asarray(states[:horizon+1],dtype=np.float32)
    xy=s[:,8:10]*16
    walls=np.argwhere(np.asarray(grid)!=0)[:,::-1].astype(np.float32)
    near=False
    if len(walls):
        delta=np.maximum(np.maximum(walls[None]-xy[:,None],xy[:,None]-(walls[None]+1)),0)
        near=bool((np.linalg.norm(delta,axis=-1).min(axis=1)<(.25+settings['wall_margin_cells'])).any())
    v=np.diff(xy,axis=0)*settings['hz'];speed=np.linalg.norm(v,axis=-1)
    good=(speed[:-1]>settings['motion_epsilon'])&(speed[1:]>settings['motion_epsilon'])
    cosine=(v[:-1]*v[1:]).sum(-1)/np.maximum(speed[:-1]*speed[1:],1e-12)
    turn=bool((good & (cosine<math.cos(math.radians(settings['turn_degrees'])))).any())
    return dict(near_wall=near,turn=turn)
