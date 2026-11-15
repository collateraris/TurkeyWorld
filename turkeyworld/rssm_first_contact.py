"""First wrong own-player transition labels, computed after the MAIN rollout.
No extra actors, state correction, observations, optimizer or RNG draws.
"""
from copy import deepcopy
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
from .runtime_checks import circle_is_clear,segment_is_clear
from .rssm_player_local_motion import source_player_motion_step,player_motion_supervised_terms

SETTINGS=dict(protocol='rssm_first_wrong_contact_v1',version=1,max_source_probes_per_stream=32,
    error_threshold_tiles=.015,weight=.5,xy_weight=2.,enemy_weight=.25,enemy_velocity_threshold=.3,enemy_direction_weight=.2,Source_labels_only=True,Source_future_feedback=False)

def validate_settings(settings):
    if settings!=SETTINGS:raise ValueError('First-contact settings changed without a revision')
    return deepcopy(settings)

def _clear(grid,a,b):
    elbow=np.array([b[0],a[1]])
    return circle_is_clear(grid,b,.2) and segment_is_clear(grid,a,elbow,.2) and segment_is_clear(grid,elbow,b,.2)

def first_contact_terms(prediction,initial,actions,grid,pass_logits,*,protocol,settings=SETTINGS):
    validate_settings(settings)
    previous=torch.cat((initial[:,None],prediction[:,:-1]),1).detach()
    own=previous.float().cpu().numpy();future=prediction.detach().float().cpu().numpy()
    controls=actions.detach().float().cpu().numpy();maps=grid.detach().cpu().numpy()
    pools=[];skips=Counter()
    for actor in range(len(own)):
        crossing=[];stopping=[];near=[]
        for tick in range(own.shape[1]):
            a=own[actor,tick,:2]*16;b=future[actor,tick,:2]*16
            if own[actor,tick,4]<=0 or not circle_is_clear(maps[actor],a,.2):break
            if not _clear(maps[actor],a,b):crossing.append(tick)
            elif np.linalg.norm(controls[actor,tick,:2])>.2 and np.linalg.norm(b-a)<.015:stopping.append(tick)
            elif tick%4==0 and not circle_is_clear(maps[actor],a,.4):near.append(tick)
        # A crossing is a direct physics fault. Stopped/near-wall points are
        # diagnostic candidates, not automatically called errors.
        pools.append(list(dict.fromkeys(crossing[:1]+stopping+near)))
    found={};probes=0
    while probes<settings['max_source_probes_per_stream'] and any(pools):
        for actor,pool in enumerate(pools):
            if not pool or actor in found:pool.clear();continue
            tick=pool.pop(0);probes+=1
            try:label=source_player_motion_step(own[actor,tick],controls[actor,tick],maps[actor])
            except ValueError as error:skips[str(error)]+=1;continue
            error=np.max(np.abs((future[actor,tick,:2]-label['target'][:2])*16))
            if error>=settings['error_threshold_tiles']:found[actor]=(tick,label);pool.clear()
            if probes>=settings['max_source_probes_per_stream']:break
    metrics=dict(protocol=settings['protocol'],Source_future_feedback=False,source_probes=probes,
        source_skip_counts=dict(skips),selected_actors=len(found),first_error_ticks=[v[0] for v in found.values()],
        independent_actors=0,additional_model_transitions=0,own_states_corrected=0)
    if not found:return prediction.sum()*0,{**metrics,'loss':0.,'xy_loss':0.}
    actors=list(found);ticks=[found[a][0] for a in actors]
    old=previous[actors,ticks];pred=prediction[actors,ticks];control=actions[actors,ticks]
    target=torch.as_tensor(np.stack([found[a][1]['target'] for a in actors]),device=pred.device,dtype=pred.dtype)
    terms,diagnostics=player_motion_supervised_terms(old,pred,control,target,pass_logits[actors,ticks],player_local_motion_protocol=protocol,simulation_hz=30.)
    xy=F.smooth_l1_loss(pred[:,:2]*16,target[:,:2]*16)
    loss=settings['weight']*(settings['xy_weight']*xy+sum(terms.values()))
    metrics.update(loss=float(loss.detach()),xy_loss=float(xy.detach()),axis_bce=float(terms['pass_bce'].detach()),axis_velocity=float(terms['axis_velocity'].detach()),diagnostics=diagnostics)
    return loss,metrics


def first_enemy_terms(prediction,initial,actions,grid,*,settings=SETTINGS):
    """First wrong own-NPC velocity/obstacle transition from the same MAIN tape."""
    validate_settings(settings)
    from .rssm_enemy_local_recovery import _source_label
    previous=torch.cat((initial[:,None],prediction[:,:-1]),1).detach()
    own=previous.float().cpu().numpy();future=prediction.detach().float().cpu().numpy()
    controls=actions.detach().float().cpu().numpy();maps=grid.detach().cpu().numpy()
    pools=[];skips=Counter()
    for actor in range(len(own)):
        legal=[];fault=[]
        for tick,state in enumerate(own[actor]):
            if state[13]<.5 or state[12]<=0:break
            if not circle_is_clear(maps[actor],state[8:10]*16,.25):break
            if not circle_is_clear(maps[actor],state[:2]*16,.2):break
            if not segment_is_clear(maps[actor],state[8:10]*16,future[actor,tick,8:10]*16,.25):fault.append(tick)
            if tick%4==0:legal.append(tick)
        pools.append(list(dict.fromkeys(fault[:1]+legal)))
    probes=0;found={}
    while probes<settings['max_source_probes_per_stream'] and any(pools):
        for actor,pool in enumerate(pools):
            if not pool or actor in found:pool.clear();continue
            tick=pool.pop(0);probes+=1
            try:label=_source_label(own[actor,tick],controls[actor,tick],maps[actor],30.)
            except ValueError as error:skips[str(error)]+=1;continue
            if label['target'][13]<.5:continue
            expected=(label['target'][8:10]-own[actor,tick,8:10])*480
            predicted=(future[actor,tick,8:10]-own[actor,tick,8:10])*480
            if np.linalg.norm(predicted-expected)>=settings['enemy_velocity_threshold']:found[actor]=(tick,label);pool.clear()
            if probes>=settings['max_source_probes_per_stream']:break
    metrics=dict(protocol=settings['protocol'],source_probes=probes,source_skip_counts=dict(skips),selected_actors=len(found),first_error_ticks=[v[0] for v in found.values()],Source_future_feedback=False,own_states_corrected=0)
    if not found:return prediction.sum()*0,{**metrics,'loss':0.}
    actors=list(found);ticks=[found[a][0] for a in actors];old=previous[actors,ticks];pred=prediction[actors,ticks]
    target=torch.as_tensor(np.stack([found[a][1]['target'] for a in actors]),device=pred.device,dtype=pred.dtype)
    actual=(pred[:,8:10]-old[:,8:10])*480;expected=(target[:,8:10]-old[:,8:10])*480
    velocity=F.smooth_l1_loss(actual,expected);moving=expected.norm(dim=-1)>.05
    direction=(1-F.cosine_similarity(actual[moving],expected[moving],dim=-1)).mean() if bool(moving.any()) else velocity*0
    loss=settings['enemy_weight']*(velocity+settings['enemy_direction_weight']*direction)
    metrics.update(loss=float(loss.detach()),velocity_loss=float(velocity.detach()),direction_loss=float(direction.detach()))
    return loss,metrics
