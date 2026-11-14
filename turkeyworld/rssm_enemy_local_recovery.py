"""Opt-in, bounded own-memory NPC recovery with post-rollout Source targets.

The Source never corrects an input, h/z, pose or spawn. It labels complete
neural rollouts afterwards. These local physical targets differ deliberately
from the unchanged recorded long-trajectory XY objective.
"""
from collections import Counter,defaultdict
from contextlib import nullcontext
from copy import deepcopy
import json
import math
import numpy as np
import torch
from torch.nn import functional as F

from .engine import Game
from .gradient_safety import finite_gradient_norm
from .rssm_combat_branches import _source_step_at_own_state
from .rssm_combat_protocol import environment_profile
from .rssm_contact_branches import capture_contact_candidate
from .rssm_coordinate_gradients import COORDINATE_GRADIENT_PROTOCOL,integrate_coordinate_gradients
from .rssm_objectives import _valid_result
from .runtime_checks import circle_is_clear,segment_is_clear

ENEMY_LOCAL_RECOVERY_PROTOCOL=dict(version=1,protocol='rssm_enemy_own_local_recovery_v1',
    max_branches=4,max_source_probes=32,horizons=[4,8,16,32],
    preferred_stream_quotas=dict(combat=2,enemy_motion=2),capture_every=16,
    weight=.5,velocity_weight=1.,speed_weight=1.,direction_weight=.25,trajectory_weight=.1,
    first_step_fraction=.5,motion_epsilon_cells_s=.05,local_error_cells_s=.2,
    normalization='present_first_rest_stream_and_motion_classes',
    Source_labels_after_own_rollout=True,Source_future_feedback=False,
    state_or_memory_correction=False,predicted_spawn_local_velocity_excluded=True)
INVALID_SOURCE_ERRORS=frozenset(('invalid_direct_own_state_geometry',
    'noncanonical_direct_own_state_no_repair','Cannot restore an entity inside a wall'))


def normalize_enemy_local_protocol(protocol):
    if protocol is None:return None
    if not isinstance(protocol,dict)or json.dumps(protocol,sort_keys=True,allow_nan=False)!=json.dumps(ENEMY_LOCAL_RECOVERY_PROTOCOL,sort_keys=True):
        raise ValueError('Enemy recovery requires its explicit bounded own-Source protocol')
    return deepcopy(protocol)


def capture_enemy_local_candidate(first,state,memory,actions,grid,prediction,*,stream):
    if stream not in('combat','enemy_motion'):raise ValueError('Enemy candidates require combat or quiet enemy stream')
    result=capture_contact_candidate(first,state,memory,actions,grid,prediction)
    result.update(stream=stream,navigation_only=False)
    return result


def _alive(state):return state[...,12]>0 and state[...,13]>=.5


def _npc_path_clear(grid,previous,current):
    # Be explicit even though the shared runtime helper currently follows the
    # same convention: the direct diagonal is not the engine's movement.
    elbow=np.array([current[0],previous[1]])
    return segment_is_clear(grid,previous,elbow,.25)and segment_is_clear(grid,elbow,current,.25)


def _candidate_rows(candidates,horizon):
    rows=[];faults={};skip=Counter()
    for item in sorted(candidates,key=lambda value:(value.get('stream',''),value['step'])):
        if item.get('stream')not in('combat','enemy_motion'):continue
        if(item.get('memory_provenance')!='actualownprefix'or item.get('new_observed_rgb_frames')!=0
                or item.get('navigation_only',False)is not False):
            raise ValueError('Enemy recovery requires genuine detached own-prefix state/full h/z')
        state=item['state'].detach().float().cpu().numpy();grid=item['map'].detach().cpu().numpy()
        prediction=item['prediction'].detach().float().cpu().numpy()
        for actor in range(len(state)):
            key=(item['stream'],actor);previous=state[actor]
            for index,current in enumerate(prediction[actor]):
                # A genuine predicted revival may teleport; its endpoint must
                # clear the map, while its across-map chord is not motion.
                revived=not _alive(previous)and _alive(current)
                bad_player=not circle_is_clear(grid[actor],current[:2]*16,.2)
                bad_npc=_alive(current)and(not circle_is_clear(grid[actor],current[8:10]*16,.25)
                    or not revived and not _npc_path_clear(grid[actor],previous[8:10]*16,current[8:10]*16))
                if bad_player or bad_npc:
                    faults.setdefault(key,item['step']+index+1);break
                previous=current
            if item['actions'].shape[1]<horizon:skip['terminal_prefix_too_short']+=1;continue
            if not _alive(state[actor])or state[actor,4]<=0:skip['own_initial_not_living']+=1;continue
            rows.append(dict(item=item,actor=actor,key=key,step=item['step'],state=state[actor],
                grid=grid[actor],prediction=prediction[actor]))
    eligible=[]
    for row in rows:
        fault=faults.get(row['key']);row['first_fault']=fault
        row['pre_fault']=fault is not None and row['step']<fault<=row['step']+len(row['prediction'])
        if fault is not None and row['step']>=fault:skip['prefix_after_first_geometry_fault']+=1
        else:eligible.append(row)
    return eligible,faults,skip


def _source_label(state,action,grid,hz):
    return _source_step_at_own_state(np.asarray(state,np.float32),np.asarray(action,np.float32),
        grid,{'canonical_atol':2e-7},1/hz)


def _motion_class(velocity,epsilon=.05):
    x,y=np.abs(velocity)>epsilon
    return 'both_moving'if x and y else'x_moving_y_stopped'if x else'x_stopped_y_moving'if y else'stationary'


def select_enemy_local_branches(candidates,*,horizon,simulation_hz=30.,protocol=ENEMY_LOCAL_RECOVERY_PROTOCOL):
    config=normalize_enemy_local_protocol(protocol)
    if config is None:return [],dict(source_probes=0)
    if horizon not in config['horizons']:raise ValueError('Enemy recovery needs H4/8/16/32')
    rows,faults,skips=_candidate_rows(candidates,horizon);probes=[];attempts=0
    streams=[name for name in('combat','enemy_motion')if any(r['item']['stream']==name for r in rows)]
    budget=config['max_source_probes']//max(1,len(streams))
    for stream in streams:
        selected=[row for row in rows if row['item']['stream']==stream]
        # Probe initial physics and retained-memory pre-contact states, with
        # actor diversity. At most16 probes per stream,32 in total.
        initial=sorted((r for r in selected if r['step']==0),key=lambda r:r['actor'])
        later=sorted((r for r in selected if r['step']>0),
            key=lambda r:(not r['pre_fault'],r['first_fault']or 10**6,-r['step'],r['actor']))
        ordered=initial[:budget//2];per_actor=Counter(r['key']for r in ordered)
        for row in later+initial[budget//2:]:
            if len(ordered)>=budget:break
            if per_actor[row['key']]>=2:continue
            ordered.append(row);per_actor[row['key']]+=1
        for row in ordered:
            attempts+=1;action=row['item']['actions'][row['actor'],0].detach().float().cpu().numpy()
            try:label=_source_label(row['state'],action,row['grid'],simulation_hz)
            except ValueError as error:
                if str(error)not in INVALID_SOURCE_ERRORS:raise
                skips[str(error)]+=1;continue
            if not _alive(label['previous'])or not _alive(label['target']):skips['Source_first_tick_not_continuing']+=1;continue
            velocity=(label['target'][8:10]-label['previous'][8:10])*(16*simulation_hz)
            predicted=(row['prediction'][0,8:10]-row['state'][8:10])*(16*simulation_hz)
            probes.append(dict(row,source_first=label,cohort=_motion_class(velocity),
                first_velocity_error=float(np.linalg.norm(predicted-velocity))))
    def priority(row):return(not row['pre_fault'],-row['first_velocity_error'],row['first_fault']or 10**6,-row['step'],row['actor'])
    chosen=[];used=set()
    for stream in streams:
        available=sorted((r for r in probes if r['item']['stream']==stream),key=priority)
        for initial in(True,False):
            matching=[r for r in available if(r['step']==0)==initial and r['key']not in used]
            if not matching:matching=[r for r in available if r['key']not in used]
            if not matching:matching=[r for r in available if not any(r is old for old in chosen)]
            if not initial:
                represented={r['cohort']for r in chosen}
                # A stopped/one-axis Source target must not lose every slot
                # to a larger vector error on unrestricted two-axis motion.
                matching.sort(key=lambda r:(r['cohort']in represented,r['cohort']!='stationary',*priority(r)))
            if matching and len(chosen)<config['max_branches']:
                chosen.append(matching[0]);used.add(matching[0]['key'])
    for row in sorted(probes,key=priority):
        if len(chosen)>=config['max_branches']:break
        if not any(row is old for old in chosen):chosen.append(row)
    return chosen,dict(source_probes=attempts,legal_source_probes=len(probes),source_skip_counts=dict(skips),
        first_geometry_fault_actors=len(faults),selected_prefix_steps=[r['step']for r in chosen],
        selected_streams=[r['item']['stream']for r in chosen],selected_cohorts=dict(Counter(r['cohort']for r in chosen)),
        selected_nonzero_prefixes=sum(r['step']>0 for r in chosen),selected_initial_prefixes=sum(r['step']==0 for r in chosen),
        selected_first_velocity_errors=[r['first_velocity_error']for r in chosen])


def _class_weights(records,classes,*,eligible=None):
    """Equal first/rest, stream and present target-class mass; sums to1."""
    result=np.zeros(len(records),np.float32)
    included=list(range(len(records)))if eligible is None else list(eligible)
    phases=sorted({records[i]['tick']==0 for i in included})
    for phase in phases:
        phase_ids=[i for i in included if(records[i]['tick']==0)==phase]
        streams=sorted({records[i]['stream']for i in phase_ids})
        for stream in streams:
            indices=[i for i in phase_ids if records[i]['stream']==stream];present=sorted({classes[i]for i in indices})
            for category in present:
                group=[i for i in indices if classes[i]==category]
                result[group]=1/(len(phases)*len(streams)*len(present)*len(group))
    return result


def _source_trajectory(row,horizon,hz):
    """Autonomous Source reference rooted at own prefix, never a model input."""
    game=Game(seed=0,endless=True,enemies_enabled=True);game.map=np.asarray(row['grid'],np.uint8).copy()
    game.map_height,game.map_width=game.map.shape;game.set_state_vector(row['state'])
    if not np.allclose(game.state_vector(),row['state'],rtol=0,atol=2e-7):raise ValueError('Noncanonical Source recovery trajectory anchor')
    labels=[game.state_vector().copy()]
    for action in row['item']['actions'][row['actor'],:horizon].detach().float().cpu().numpy():
        game.step(action,dt=1/hz);labels.append(game.state_vector().copy())
    return np.asarray(labels,np.float32)


def enemy_local_recovery_backward(model,candidates=None,*,horizon=16,protocol=None,amp=None,check_gradients=True):
    config=normalize_enemy_local_protocol(protocol)
    metrics=dict(enemy_local_loss=0.,enemy_local_forward_transitions=0,enemy_local_valid_source_labels=0,
        enemy_local_active=config is not None,enemy_local_horizon=horizon,optimizer_steps=0,renderer_calls=0,
        source_future_feedback=False,state_resets_after_prefix=0,memory_resets_after_prefix=0)
    if config is None:return metrics
    chosen,selection=select_enemy_local_branches(candidates or [],horizon=horizon,
        simulation_hz=model.config.simulation_hz,protocol=config)
    metrics.update(enemy_local_protocol=config['protocol'],selection=selection,branches=len(chosen))
    if not chosen:return metrics
    autocast=nullcontext if amp is None else amp;records=[];trajectories=[];hz=model.config.simulation_hz
    for branch,row in enumerate(chosen):
        actor=row['actor'];item=row['item'];current=item['state'][actor:actor+1].detach()
        memory=item['memory'][actor:actor+1].detach();grid=item['map'][actor:actor+1].detach()
        controls=item['actions'][actor:actor+1,:horizon].detach();points=[]
        with environment_profile(model,navigation_only=False),autocast():
            with torch.no_grad():features=model.encode_map(grid).detach()
            shape=tuple(memory.shape)
            for tick in range(horizon):
                raw=model.transition(current.detach(),controls[:,tick],features,memory)
                _valid_result(raw,shape,'Own enemy recovery branch')
                respawn=raw.get('respawn_flags')
                if not torch.is_tensor(respawn)or respawn.shape!=(1,1)or respawn.dtype!=torch.bool:
                    raise ValueError('Enemy local branches require actual predicted respawn flags')
                record=dict(stream=item['stream'],tick=tick,branch=branch,previous=current.detach(),
                    predicted=raw['state'],respawn=bool(respawn[0,0]),action=controls[:,tick],grid=grid)
                if 'proposed_npc_displacement'in raw:record['proposal']=raw['proposed_npc_displacement']
                records.append(record)
                integrated=integrate_coordinate_gradients(current,raw,COORDINATE_GRADIENT_PROTOCOL)
                current,memory=integrated['state'],integrated['memory'];points.append(current)
            trajectories.append(torch.stack(points,1))
    # Only after ALL own neural transitions: query exact own conditional local
    # targets. Illegal states are skipped, never projected or repaired.
    direct=[];skips=Counter();spawn_excluded=0
    for record in records:
        if record['respawn']:spawn_excluded+=1;continue
        previous=record['previous'][0].detach().float().cpu().numpy()
        try:label=_source_label(previous,record['action'][0].detach().float().cpu().numpy(),
            record['grid'][0].detach().cpu().numpy(),hz)
        except ValueError as error:
            if str(error)not in INVALID_SOURCE_ERRORS:raise
            skips[str(error)]+=1;continue
        if not _alive(label['previous'])or not _alive(label['target']):skips['Source_not_continuing_npc']+=1;continue
        direct.append((record,label))
    zero=trajectories[0].sum()*0;velocity_loss=speed_loss=direction_loss=zero
    speeds={};counts={};pred_cap_fraction=None;proposal_metrics={}
    if direct:
        device=direct[0][0]['predicted'].device
        target=torch.from_numpy(np.stack([(r['target'][8:10]-r['previous'][8:10])*(16*hz)for _,r in direct])).to(device)
        predicted=torch.cat([(r['predicted'][:,8:10]-r['previous'][:,8:10])*(16*hz)for r,_ in direct]).float()
        source_speed=target.norm(dim=-1);own_speed=predicted.norm(dim=-1)
        row_records=[r for r,_ in direct];classes=[_motion_class(v,config['motion_epsilon_cells_s'])for v in target.detach().cpu().numpy()]
        for axis in range(2):
            axis_classes=['moving'if abs(v)>config['motion_epsilon_cells_s']else'stopped'for v in target[:,axis].detach().cpu().numpy()]
            weights=torch.from_numpy(_class_weights(row_records,axis_classes)).to(device)
            velocity_loss=velocity_loss+.5*(F.smooth_l1_loss(predicted[:,axis],target[:,axis],reduction='none')*weights).sum()
        moving=(source_speed>config['motion_epsilon_cells_s']).detach()
        weights=torch.from_numpy(_class_weights(row_records,['moving'if yes else'stationary'for yes in moving.cpu().tolist()])).to(device)
        speed_loss=(F.smooth_l1_loss(own_speed,source_speed,reduction='none')*weights).sum()
        direction_weights=torch.from_numpy(_class_weights(row_records,classes,eligible=np.flatnonzero(moving.cpu().numpy()))).to(device)
        direction_error=1-F.cosine_similarity(predicted,target,dim=-1,eps=config['motion_epsilon_cells_s'])
        direction_loss=(direction_error*direction_weights).sum()
        counts=dict(Counter(classes));speeds=dict(Source_mean=float(source_speed.mean()),own_mean=float(own_speed.detach().mean()),
            speed_MAE=float((own_speed.detach()-source_speed).abs().mean()),
            velocity_vector_MAE=float((predicted.detach()-target).norm(dim=-1).mean()),
            direction_cosine_error=float((direction_error.detach()*moving).sum()/moving.sum().clamp_min(1)))
        pred_cap_fraction=float((own_speed.detach()>=1.1999).float().mean())
        proposals=[r['proposal'].detach().float().reshape(-1,2)*(16*hz)for r,_ in direct if 'proposal'in r]
        if proposals:
            magnitude=torch.cat(proposals).norm(dim=-1)
            proposal_metrics=dict(mean=float(magnitude.mean()),p95=float(torch.quantile(magnitude,.95)),
                over_cap_fraction=float((magnitude>1.2).float().mean()))
    # A small own-anchored Source trajectory term retains short XY recovery;
    # Source lives mask it, regardless of predicted missing/dead/spawn flags.
    xy=[]
    if config['trajectory_weight']:
        for row,prediction in zip(chosen,trajectories):
            source=torch.from_numpy(_source_trajectory(row,horizon,hz)).to(prediction.device)
            alive=(source[:-1,12]>0)&(source[:-1,13]>=.5)&(source[1:,12]>0)&(source[1:,13]>=.5)
            error=F.smooth_l1_loss(prediction[0,:,8:10].float()*16,source[1:,8:10]*16,reduction='none').sum(-1)
            xy.append((error*alive).sum()/(2*alive.sum().clamp_min(1)))
    trajectory_loss=sum(xy)/len(xy)if xy else zero
    loss=config['weight']*(config['velocity_weight']*velocity_loss+config['speed_weight']*speed_loss
        +config['direction_weight']*direction_loss+config['trajectory_weight']*trajectory_loss)
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite own-aligned enemy recovery loss')
    loss.backward()
    if check_gradients:finite_gradient_norm(model.named_parameters(),context='Own-aligned enemy recovery')
    metrics.update(enemy_local_loss=float(loss.detach()),enemy_local_forward_transitions=len(records),
        enemy_local_valid_source_labels=len(direct),velocity_loss=float(velocity_loss.detach()),speed_loss=float(speed_loss.detach()),
        direction_loss=float(direction_loss.detach()),trajectory_loss=float(trajectory_loss.detach()),
        predicted_spawn_velocity_exclusions=spawn_excluded,source_skip_counts=dict(skips),Source_motion_cohorts=counts,
        direct_Source_queries_attempted=len(records)-spawn_excluded,
        Source_trajectory_transitions=horizon*len(chosen)if config['trajectory_weight']else 0,
        total_Source_step_queries=selection['source_probes']+len(records)-spawn_excluded
            +(horizon*len(chosen)if config['trajectory_weight']else 0),
        speed_metrics=speeds,typed_speed_cap_fraction=pred_cap_fraction,proposed_speed_metrics=proposal_metrics,
        weight=config['weight'],normalization=config['normalization'],first_step_fraction=config['first_step_fraction'],
        first_tick_Source_labels=sum(r['tick']==0 for r,_ in direct),nonzero_prefix_branches=selection['selected_nonzero_prefixes'],
        full_own_memory_preserved=True,new_observed_rgb_frames_after_prefix=0,backward_calls=1,
        global_recorded_XY_objective_changed=False,branch_horizon_graph=horizon,Source_labels_after_own_rollout=True)
    return metrics
