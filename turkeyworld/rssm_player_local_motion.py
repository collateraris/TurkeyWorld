"""Opt-in, player-only Source movement labels after an own neural rollout.

The reference game moves the player before NPC events. Only that one movement
step is supervised. The Source query disables NPCs on its label side; the
network's full own state, enemy conditioning and RSSM memory are never edited.
"""
from copy import deepcopy
import json
import math

import numpy as np
import torch
from torch.nn import functional as F

from .engine import Game
from .motion_recovery_training import pass_fraction_targets
from .runtime_checks import circle_is_clear


PLAYER_LOCAL_MOTION_PROTOCOL = dict(
    protocol='rssm_player_local_motion_recovery_v1',version=1,
    quiet_nonzero_prefix_branches=2,max_contact_branches=4,max_source_probes=32,
    quiet_source_probe_budget=16,
    preferred_outcomes=['x_free_y_blocked','x_blocked_y_free','both_free'],
    source_scope='exact_single_player_movement_step',source_NPCs_disabled_only_for_labels=True,
    player_body_must_be_legal=True,player_must_be_alive=True,
    free_fraction=.95,blocked_fraction=.05,
    axis_class_mass=dict(free=2.,blocked=1.,partial=1.),
    normalization='equal_present_axis_mass_then_normalized_present_class_mass',
    local_velocity_normalizer_cells_s=2.4,
    free_progress_min_fraction=.9,free_progress_weight=.5,
    cumulative_residual='contiguous_valid_own_steps_only',
    wall_weight_increase=False,Source_future_inputs=False,own_state_correction=False,
    memory_reset_after_prefix=False)


def normalize_player_local_motion_protocol(value):
    if value is None:return None
    if(not isinstance(value,dict)or set(value)!=set(PLAYER_LOCAL_MOTION_PROTOCOL)
            or json.dumps(value,sort_keys=True,allow_nan=False)!=json.dumps(PLAYER_LOCAL_MOTION_PROTOCOL,sort_keys=True)):
        raise ValueError('Player local motion needs its exact explicit recovery protocol')
    return deepcopy(PLAYER_LOCAL_MOTION_PROTOCOL)


def source_player_motion_step(state,action,grid,*,dt=1/30):
    """Return only an exact XY movement label, never a corrected world state.

    No NPC field is validated or restored into the Source. The authoritative
    player's position/action/heading is used unchanged, including turn, diagonal
    normalization and Source X-then-Y substeps. Invalid or dead player bodies
    are rejected. This function must be called after the neural rollout.
    """
    previous=np.asarray(state,np.float32);control=np.asarray(action,np.float32)
    tilemap=np.asarray(grid)
    if(previous.shape!=(20,)or control.shape!=(4,)or tilemap.shape!=(16,16)
            or not np.isfinite(previous).all()or not np.isfinite(control).all()
            or not np.isfinite(tilemap).all()or type(dt)not in(int,float)
            or not math.isfinite(dt)or not 0<dt<=.25):
        raise ValueError('invalid_player_motion_query_shape_or_values')
    if not np.array_equal(tilemap,tilemap.astype(np.uint8)):
        raise ValueError('invalid_player_motion_map_values')
    if previous[4]<=0:raise ValueError('dead_player_motion_source')
    if not circle_is_clear(tilemap,previous[:2]*16,.2):
        raise ValueError('invalid_player_motion_body_geometry')
    source=Game(seed=0,endless=True,enemies_enabled=False)
    source.map=tilemap.astype(np.uint8,copy=True);source.map_height,source.map_width=source.map.shape
    # Set only the observed player adapter. Calling set_state_vector would
    # incorrectly reject a legal player because an unrelated NPC is in a wall.
    # Disabled Source NPC placeholders do not participate in player movement.
    source.player.update(x=float(previous[0]*16),y=float(previous[1]*16),
        angle=math.atan2(float(previous[2]),float(previous[3])),
        hp=float(np.clip(previous[4],0,1)*100),
        ammo=int(round(float(np.clip(previous[5],0,1)*99))),
        cooldown=float(np.clip(previous[6],0,1)*source.FIRE_INTERVAL),
        shot_flash=float(np.clip(previous[7],0,1)*source.FLASH_DURATION))
    if not np.allclose(source.state_vector()[:8],previous[:8],rtol=0,atol=2e-7):
        raise ValueError('noncanonical_player_motion_source_no_repair')
    source.time=max(0.,float(previous[-1]*300))
    source.step(control,dt=float(dt))
    player=source.player
    target=previous.copy()
    target[:2]=np.asarray([player['x']/16,player['y']/16],np.float32)
    # Only XY is a supervised label. All other copied values are placeholders
    # for the existing pass-fraction interface, not future world-state targets.
    return dict(previous=previous.copy(),target=target,
        delta_xy=(target[:2]-previous[:2]).copy(),supervised_state_indices=[0,1],
        source_scope='exact_single_player_movement_step',source_NPC_geometry_ignored=True,
        own_NPC_body_invalid=not circle_is_clear(tilemap,previous[8:10]*16,.25),
        Source_future_feedback=False,own_state_corrected=False)


def player_motion_supervised_terms(previous,prediction,actions,target,pass_logits,*,
                                   player_local_motion_protocol,simulation_hz=30.):
    """Balanced one-step pass/velocity and free-axis progress, with diagnostics.

    All tensors correspond to already completed own steps with legal player
    labels. Neither target fields nor the preceding state receive gradients.
    Free/blocked/partial masses are2/1/1 within each present axis; X/Y then have
    equal mass. Both orientations of wall sliding are reported separately.
    """
    cfg=normalize_player_local_motion_protocol(player_local_motion_protocol)
    if cfg is None:raise ValueError('Player local terms require explicit recovery settings')
    if(previous.ndim!=2 or previous.shape[-1]!=20 or prediction.shape!=previous.shape
            or target.shape!=previous.shape or actions.shape!=(len(previous),4)
            or pass_logits.shape!=(len(previous),2)or not len(previous)
            or type(simulation_hz)not in(int,float)or not math.isfinite(simulation_hz)or simulation_hz<=0
            or not all(bool(torch.isfinite(v).all())for v in(previous,prediction,target,actions,pass_logits))):
        raise ValueError('Player local terms need finite matching own one-step tensors')
    old=previous.detach().float();truth=target.detach().float();control=actions.detach().float()
    predicted=prediction.float();logits=pass_logits.float();dt=1/simulation_hz
    fraction,active,command=pass_fraction_targets(old,truth,control,dt=dt)
    groups=dict(free=active&(fraction>=cfg['free_fraction']),blocked=active&(fraction<=cfg['blocked_fraction']))
    groups['partial']=active&~groups['free']&~groups['blocked']
    counts={name:mask.sum(0)for name,mask in groups.items()}
    present_mass=sum((count>0).float()*cfg['axis_class_mass'][name]for name,count in counts.items())
    present_axes=(present_mass>0).sum().clamp_min(1)
    weights=torch.zeros_like(fraction)
    for name,mask in groups.items():
        weights=weights+mask.float()*cfg['axis_class_mass'][name]/counts[name].clamp_min(1)/present_mass.clamp_min(1)/present_axes
    weights=weights.detach()
    delta=predicted[:,:2]-old[:,:2];expected=truth[:,:2]-old[:,:2]
    scale=16*simulation_hz;normalizer=cfg['local_velocity_normalizer_cells_s']
    bce=F.binary_cross_entropy_with_logits(logits,fraction,reduction='none')
    velocity=F.smooth_l1_loss(delta*(scale/normalizer),expected*(scale/normalizer),reduction='none')
    free=groups['free'];free_count=free.sum(0);free_axes=(free_count>0).sum().clamp_min(1)
    free_weights=free.float()/free_count.clamp_min(1)/free_axes
    progress=delta*command.sign()
    missing=(cfg['free_progress_min_fraction']*command.abs()-progress).clamp_min(0)*(scale/normalizer)
    progress_error=F.smooth_l1_loss(missing,torch.zeros_like(missing),reduction='none')
    terms=dict(pass_bce=(bce*weights).sum(),axis_velocity=(velocity*weights).sum(),
        free_progress=cfg['free_progress_weight']*(progress_error*free_weights).sum())
    error=(delta.detach()-expected).abs()*scale
    false=(free&(delta.detach().abs()<.2*command.abs()))
    missed=groups['blocked']&(delta.detach().abs()>.2*command.abs())
    metrics=dict(protocol=cfg['protocol'],source_label_count=len(old),
        axis_group_counts={name:[int(n)for n in count.tolist()]for name,count in counts.items()},
        axis_class_mass=deepcopy(cfg['axis_class_mass']),axis_weight_mass=[float(v)for v in weights.sum(0)],
        false_stop_axes=int(false.sum()),missed_block_axes=int(missed.sum()),
        axis_velocity_mae_cells_per_second={name:float((error[:,axis]*active[:,axis]).sum()/active[:,axis].sum().clamp_min(1))
            for axis,name in enumerate(('x','y'))},axis_components={})
    for axis,name in enumerate(('x','y')):
        metrics['axis_components'][name]={}
        for outcome,mask in groups.items():
            selected=mask[:,axis];count=selected.sum().clamp_min(1)
            metrics['axis_components'][name][outcome]=dict(count=int(selected.sum()),
                weight_mass=float(weights[:,axis][selected].sum()),
                BCE=float((bce[:,axis].detach()*selected).sum()/count),
                normalized_velocity_loss=float((velocity[:,axis].detach()*selected).sum()/count),
                velocity_MAE_cells_s=float((error[:,axis]*selected).sum()/count))
    for name,free_axis,blocked_axis in(('x_free_y_blocked',0,1),('x_blocked_y_free',1,0)):
        mask=free[:,free_axis]&groups['blocked'][:,blocked_axis];count=mask.sum().clamp_min(1)
        metrics[name]=dict(count=int(mask.sum()),
            free_velocity_MAE_cells_s=float((error[:,free_axis]*mask).sum()/count),
            blocked_velocity_MAE_cells_s=float((error[:,blocked_axis]*mask).sum()/count),
            false_free_stop_count=int((false[:,free_axis]&mask).sum()),
            blocked_leak_count=int((missed[:,blocked_axis]&mask).sum()))
    return terms,metrics
