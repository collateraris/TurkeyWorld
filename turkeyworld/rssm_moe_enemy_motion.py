"""Quiet enemy Source cohort, with opt-in direct player navigation supervision.

The original controls and full own P/E/Combat memory are preserved. No Source
oracle, attack/spawn objective, future observation or state repair is used.
The legacy default stays enemy-only; the explicit player revision adds XY and
train-only geometry without changing any NPC/router denominator or weight.
"""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F

from .gradient_safety import finite_gradient_norm
from .rssm_combat_protocol import environment_profile
from .rssm_coordinate_gradients import COORDINATE_GRADIENT_PROTOCOL,integrate_coordinate_gradients
from .rssm_moe_protocol import MOE_HORIZONS,SETTINGS
from .rssm_moe_training import _router_terms
from .rssm_objectives import _batch_data,_initialize,_observed_count,_valid_result
from .rssm_player_geometry_objectives import(prepare_player_geometry,player_geometry_terms,add_player_geometry_metrics,
    player_navigation_tbptt,capture_player_chunks)
from .rssm_npc_geometry_objectives import prepare_npc_geometry,npc_geometry_terms,add_npc_geometry_metrics


PROTOCOL='rssm_moe_quiet_enemy_motion_v1'
VELOCITY_WEIGHT=.1
DIRECTION_WEIGHT=.05


def enemy_motion_backward(model,batch,*,context=1,observed_context=None,
                          expected_horizon=None,enemy_weight=16/54,router_weight=16/64,
                          progress_callback=None,check_gradients=True,player_weight=0.,
                          player_navigation_protocol=None,player_contact_callback=None,
                          npc_navigation_protocol=None):
    for key,value in (('enemy_weight',enemy_weight),('router_weight',router_weight),('player_weight',player_weight)):
        if type(value)not in(int,float)or not math.isfinite(value)or value<0:
            raise ValueError(key+' must be finite and nonnegative')
    states,frames,actions,target,grid=_batch_data(batch,context)
    observed=_observed_count(context,observed_context)
    actors,horizon=actions.shape[:2]
    if (not 1<=actors<=64 or horizon not in MOE_HORIZONS
            or expected_horizon is not None and expected_horizon!=horizon):
        raise ValueError('Enemy-motion cohort requires the exact active MoE horizon and 1..64 actors')
    source=torch.cat((states[:,:context],target),1).detach()
    if (bool(actions[...,3].ne(0).any())or bool((source[...,4]<=0).any())
            or bool((source[...,12]<=0).any())or bool((source[...,13]<.5).any())
            or not torch.equal(source[...,4],source[:,:1,4].expand(-1,source.shape[1]))
            or not torch.equal(source[...,12],source[:,:1,12].expand(-1,source.shape[1]))):
        raise ValueError('Enemy-motion Source tapes must be no-fire, continuing-alive and HP-stable')
    previous_source=torch.cat((states[:,context-1:context],target[:,:-1]),1).detach()
    if player_navigation_protocol is None and player_weight:
        raise ValueError('Quiet-player supervision requires its explicit navigation revision')
    geometry=prepare_player_geometry(previous_source,target,grid,player_navigation_protocol=player_navigation_protocol)
    npc_geometry=prepare_npc_geometry(previous_source,target,grid,npc_navigation_protocol=npc_navigation_protocol)
    tbptt=player_navigation_tbptt(player_navigation_protocol)
    geometry_metrics={};raw_player_xy=raw_player_wall=0.
    npc_geometry_metrics={};raw_npc_wall=0.
    velocity_weight=VELOCITY_WEIGHT if npc_geometry is None else 0.
    direction_weight=DIRECTION_WEIGHT if npc_geometry is None else 0.
    target_velocity=(target[...,8:10].detach()-previous_source[...,8:10])*16*model.config.simulation_hz
    speed=target_velocity.norm(dim=-1)
    direction_valid=speed>=.05
    denominator=2*actors*horizon
    raw=dict(xy=0.,velocity=0.,direction=0.,role=0.,balance=0.)
    lost=spawns=0
    modes=[(module,module.training)for module in model.modules()]
    try:
        with environment_profile(model,navigation_only=False):
            with torch.no_grad():fmap=model.encode_map(grid).detach()
            memory=_initialize(model,frames,states,fmap,context,observed)
            shape=tuple(memory.shape);current=states[:,context-1]
            for first in range(0,horizon,tbptt):
                if first:current,memory=current.detach(),memory.detach()
                chunk_state,chunk_memory=current,memory
                last=min(first+tbptt,horizon)
                predictions=[];full_predictions=[];velocities=[];role=[];balance=[];captures=[];respawns=[]
                for tick in range(first,last):
                    if player_contact_callback is not None and tick%16==0 and horizon-tick>=4:
                        captures.append((tick,current,memory))
                    old=current
                    result=model.transition(current.detach(),actions[:,tick],fmap,memory)
                    _valid_result(result,shape,'Quiet enemy-motion own prior')
                    routers=_router_terms(result)
                    result=integrate_coordinate_gradients(current,result,COORDINATE_GRADIENT_PROTOCOL)
                    current,memory=result['state'],result['memory']
                    predictions.append(current[:,8:10])
                    if geometry is not None or npc_geometry is not None or player_contact_callback is not None:full_predictions.append(current)
                    if npc_geometry is not None:respawns.append(result['respawn_flags'])
                    velocities.append((current[:,8:10]-old[:,8:10])*16*model.config.simulation_hz)
                    role.append(routers['router_role_loss']/horizon)
                    balance.append(routers['router_balance_loss']/horizon)
                    lost+=int(((current[:,12]<=0)|(current[:,13]<.5)).sum())
                    if 'respawn_flags'in result:spawns+=int(result['respawn_flags'].sum())
                prediction=torch.stack(predictions,1).float()
                velocity=torch.stack(velocities,1).float()
                local_velocity=velocity if npc_geometry is None else velocity.detach()
                xy=F.smooth_l1_loss(prediction*16,target[:,first:last,8:10].detach().float()*16,
                                   reduction='sum')/denominator
                local=F.smooth_l1_loss(local_velocity,target_velocity[:,first:last],reduction='sum')/denominator
                selected=direction_valid[:,first:last]
                cosine=F.cosine_similarity(local_velocity,target_velocity[:,first:last],dim=-1,eps=.05)
                direction=((1-cosine)*selected).sum()/direction_valid.sum().clamp_min(1)
                r,b=sum(role),sum(balance)
                loss=enemy_weight*(xy+velocity_weight*local+direction_weight*direction)
                loss=loss+router_weight*(SETTINGS['router_role_weight']*r+SETTINGS['router_balance_weight']*b)
                predicted=torch.stack(full_predictions,1)if full_predictions else None
                capture_player_chunks(player_contact_callback,captures,predicted,actions,grid,first=first)
                if npc_geometry is not None:
                    wall,wall_metrics=npc_geometry_terms(predicted,chunk_state,npc_geometry,first=first,
                        predicted_respawns=torch.stack(respawns,1))
                    loss=loss+enemy_weight*wall;raw_npc_wall+=float(wall.detach())
                    add_npc_geometry_metrics(npc_geometry_metrics,wall_metrics)
                if geometry is not None:
                    player_xy=F.smooth_l1_loss(predicted[...,:2].float()*16,target[:,first:last,:2].detach().float()*16,
                        reduction='sum')/denominator
                    wall,wall_metrics=player_geometry_terms(predicted,chunk_state,geometry,first=first)
                    loss=loss+player_weight*(player_xy+wall)
                    raw_player_xy+=float(player_xy.detach());raw_player_wall+=float(wall.detach())
                    add_player_geometry_metrics(geometry_metrics,wall_metrics)
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite enemy-motion objective')
                loss.backward()
                if check_gradients:finite_gradient_norm(model.named_parameters(),context='Quiet enemy-motion MoE')
                for key,value in (('xy',xy),('velocity',local),('direction',direction),('role',r),('balance',b)):
                    raw[key]+=float(value.detach())
                if progress_callback:progress_callback(last,horizon)
    finally:
        for module,mode in modes:module.training=mode
    raw_enemy=raw['xy']+velocity_weight*raw['velocity']+direction_weight*raw['direction']+raw_npc_wall
    enemy=float(enemy_weight)*raw_enemy
    role=float(router_weight)*SETTINGS['router_role_weight']*raw['role']
    balance=float(router_weight)*SETTINGS['router_balance_weight']*raw['balance']
    player=float(player_weight)*(raw_player_xy+raw_player_wall)
    report=dict(enemy_motion_objective_protocol=PROTOCOL,enemy_motion_enemy_loss=enemy,
        enemy_motion_router_role_loss=role,enemy_motion_router_balance_loss=balance,
        enemy_motion_loss=enemy+role+balance+player,
        enemy_motion_raw_enemy_loss=raw_enemy,
        enemy_motion_raw_xy_loss=raw['xy'],enemy_motion_raw_velocity_loss=raw['velocity'],
        enemy_motion_raw_direction_loss=raw['direction'],enemy_motion_raw_router_role_loss=raw['role'],
        enemy_motion_raw_router_balance_loss=raw['balance'],enemy_motion_enemy_weight=float(enemy_weight),
        enemy_motion_router_weight=float(router_weight),enemy_motion_velocity_weight=velocity_weight,
        enemy_motion_direction_weight=direction_weight,enemy_motion_actor_count=actors,
        enemy_motion_horizon=horizon,enemy_motion_forward_transitions=actors*horizon,
        enemy_motion_observed_context_frames=observed,enemy_motion_valid_npc_transitions=actors*horizon,
        enemy_motion_direction_transitions=int(direction_valid.sum()),enemy_motion_predicted_lost_npc_steps=lost,
        enemy_motion_predicted_respawns=spawns,enemy_motion_source_oracle_calls=0,
        enemy_motion_combat_loss=0.,enemy_motion_player_loss=player,enemy_motion_future_rgb_encoded=0,
        enemy_motion_memory_resets_after_initialization=0,enemy_motion_source_future_feedback=False,
        source_future_feedback=False,enemy_motion_tbptt=tbptt)
    if geometry is not None:
        report.update(enemy_motion_player_xy_loss=float(player_weight)*raw_player_xy,
            enemy_motion_player_wall_loss=float(player_weight)*raw_player_wall,
            enemy_motion_raw_player_xy_loss=raw_player_xy,enemy_motion_raw_player_wall_loss=raw_player_wall,
            enemy_motion_player_weight=float(player_weight),enemy_motion_player_geometry=geometry_metrics,
            player_geometry_active=True)
    if npc_geometry is not None:
        report.update(enemy_motion_enemy_xy_loss=float(enemy_weight)*raw['xy'],
            enemy_motion_npc_wall_loss=float(enemy_weight)*raw_npc_wall,
            enemy_motion_raw_npc_wall_loss=raw_npc_wall,enemy_motion_npc_geometry=npc_geometry_metrics,
            npc_geometry_active=True,npc_navigation_objective_protocol=npc_navigation_protocol['protocol'],
            enemy_motion_recorded_local_supervision_active=False,
            enemy_motion_recorded_velocity_direction_diagnostic_only=True,
            enemy_motion_own_aligned_local_supervision_expected=True)
    return report
