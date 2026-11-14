"""Balanced short contacts and own-memory pre-contact branches for player learning.

The static map and recorded TRAIN outcomes select examples only. Source labels
are queried after an autonomous scout/branch, never supplied as neural inputs.
No collision correction, renderer, posterior or optimizer is used here.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .contact_objectives import CONTACT_PROTOCOL_V2,swept_player_wall_loss
from .engine import Game
from .gradient_safety import finite_gradient_norm
from .motion_recovery_training import pass_fraction_targets
from .rssm_combat_protocol import environment_profile
from .rssm_contact_branches import capture_contact_candidate
from .rssm_coordinate_gradients import COORDINATE_GRADIENT_PROTOCOL,integrate_coordinate_gradients
from .rssm_kinematic_losses import KINEMATIC_PROTOCOL_V1,prepare_kinematic_supervision
from .rssm_objectives import _batch_data,_initialize,_valid_result
from .rssm_player_local_motion import(PLAYER_LOCAL_MOTION_PROTOCOL,
    normalize_player_local_motion_protocol,source_player_motion_step,player_motion_supervised_terms)
from .runtime_checks import circle_is_clear,segment_is_clear
from .training_pipeline import atomic_json

PROTOCOL='rssm_player_short_contact_replay_v1'
COHORTS=('both_free','x_free_y_blocked','x_blocked_y_free','both_blocked','partial')
HORIZONS=(4,8,16,32)
MAX_BRANCHES=4
MAX_SOURCE_PROBES=32
TBPTT=64


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def _cohorts(previous,target,actions,dt=1/30):
    fractions,active,_=pass_fraction_targets(previous,target,actions,dt=dt)
    free=active&(fractions>=.95);blocked=active&(fractions<=.05)
    full=active.all(-1)
    masks=dict(both_free=full&free.all(-1),x_free_y_blocked=full&free[:,0]&blocked[:,1],
        x_blocked_y_free=full&blocked[:,0]&free[:,1],both_blocked=full&blocked.all(-1))
    masks['partial']=active.any(-1)&~torch.stack(list(masks.values())).any(0)
    return {name:mask.detach().cpu().numpy()for name,mask in masks.items()}


class PlayerContactReplaySampler:
    """Four balanced TRAIN actors, each with one genuine observation +32 controls."""
    def __init__(self,data,*,seed=174175,cache_path=None):
        if getattr(data,'split',None)!='train'or not data.episodes:
            raise ValueError('Player contact replay accepts explicit nonempty TRAIN only')
        if type(seed)is not int or seed<0:raise ValueError('Contact replay requires a nonnegative private seed')
        self.data,self.seed=data,seed;self.rng=np.random.default_rng(seed)
        groups={name:[]for name in COHORTS};records=[];counts=Counter()
        for index,episode in enumerate(data.episodes):
            path=Path(episode['path'])
            with np.load(path,allow_pickle=False)as saved:
                states=np.asarray(saved['states'],np.float32);actions=np.asarray(saved['actions'],np.float32)
                if str(saved['split'])!='train':raise ValueError('Contact replay refuses mixed TRAIN/VAL source')
                if len(actions)<32:continue
                states,actions=states[:33],actions[:32]
                if(states.shape!=(33,20)or actions.shape!=(32,4)or not np.isfinite(states).all()
                        or not np.isfinite(actions).all()or np.any(states[:,12:14]!=0)or np.any(actions[:,3]!=0)):
                    raise ValueError('Contact data must contain canonical no-enemy/no-fire32-step Source clips')
                masks=_cohorts(torch.from_numpy(states[:-1]),torch.from_numpy(states[1:]),torch.from_numpy(actions))
            # One first occurrence per episode/cohort avoids overweighting a
            # long free stretch compared with a single blocked transition.
            for name,mask in masks.items():
                indices=np.flatnonzero(mask);counts[name]+=len(indices)
                if len(indices):groups[name].append((index,int(indices[0])+1))
            stat=path.stat()
            with path.open('rb')as stream:file_sha=hashlib.file_digest(stream,'sha256').hexdigest()
            records.append(dict(index=index,name=path.name,bytes=stat.st_size,sha256=file_sha))
        if any(not groups[name]for name in COHORTS[:4]):
            raise ValueError('Contact replay needs all four free/X-block/Y-block/both-block Source cohorts')
        self.groups={name:rows for name,rows in groups.items()if rows};self.cycle=tuple(self.groups)
        self.identity=dict(protocol=PROTOCOL,split='train',context=1,horizon=32,seed=seed,
            source_fingerprint=_digest(records),source_episodes=len(records),
            cohort_episodes={name:len(rows)for name,rows in self.groups.items()},source_axis_outcome_counts=dict(counts),
            event_selection='first_occurrence_per_episode_and_cohort',Source_future_inputs=False)
        self.identity_sha256=_digest(self.identity)
        self.draws=self.round_robin=0;self.cohort_draws={name:0 for name in self.groups}
        if cache_path is not None:
            path=Path(cache_path);value=dict(identity=self.identity,identity_sha256=self.identity_sha256)
            if path.exists()and json.loads(path.read_text(encoding='utf-8'))!=value:
                raise ValueError('Pinned contact sampler TRAIN inventory changed')
            if not path.exists():atomic_json(path,value)

    def state_dict(self):
        return dict(protocol=PROTOCOL,identity_sha256=self.identity_sha256,seed=self.seed,
            draws=self.draws,round_robin=self.round_robin,cohort_draws=deepcopy(self.cohort_draws),
            rng=deepcopy(self.rng.bit_generator.state))

    def restore_state(self,saved):
        if(saved.get('protocol')!=PROTOCOL or saved.get('identity_sha256')!=self.identity_sha256
                or saved.get('seed')!=self.seed or type(saved.get('draws'))is not int or saved['draws']<0
                or saved['draws']%4 or saved.get('round_robin')!=saved['draws']
                or set(saved.get('cohort_draws',{}))!=set(self.groups)
                or any(type(n)is not int or n<0 for n in saved['cohort_draws'].values())
                or sum(saved['cohort_draws'].values())!=saved['draws']):
            raise ValueError('Contact sampler complete four-actor ledger or identity changed')
        complete,remainder=divmod(saved['draws'],len(self.cycle))
        expected={name:complete+(index<remainder)for index,name in enumerate(self.cycle)}
        if any(saved['cohort_draws'][name]!=expected[name]for name in self.groups):
            raise ValueError('Contact sampler cohort rotation changed')
        rng=np.random.default_rng();rng.bit_generator.state=deepcopy(saved['rng'])
        self.rng.bit_generator.state=deepcopy(rng.bit_generator.state)
        self.draws=saved['draws'];self.round_robin=saved['round_robin'];self.cohort_draws=deepcopy(saved['cohort_draws'])

    def batch(self,actors=4):
        if type(actors)is not int or actors!=4:raise ValueError('Contact replay uses exactly four actors per complete update')
        before=self.state_dict()
        try:
            pairs=[];cohorts=[];offsets=[]
            for _ in range(actors):
                name=self.cycle[self.round_robin%len(self.cycle)];rows=self.groups[name]
                episode,event=rows[int(self.rng.integers(len(rows)))];pairs.append((episode,0));offsets.append(event);cohorts.append(name)
                self.round_robin+=1;self.draws+=1;self.cohort_draws[name]+=1
            batch=self.data.batch(actors,context=1,horizon=32,indices=np.asarray(pairs,np.int64))
            batch['contact_event_offsets']=offsets;batch['contact_cohorts']=cohorts
            receipt=dict(protocol=PROTOCOL,actors=4,pairs=[list(p)for p in pairs],cohorts=cohorts,
                event_offsets=offsets,context=1,horizon=32,private_pcg_draws=self.draws,
                identity_sha256=self.identity_sha256,source_future_feedback=False)
            return batch,receipt
        except BaseException:self.restore_state(before);raise


def capture_player_contact_candidate(step,state,memory,actions,grid,prediction,*,navigation_only=True,stream='navigation'):
    if type(navigation_only)is not bool or not isinstance(stream,str)or not stream:
        raise ValueError('Own contact candidate requires an explicit environment profile and stream')
    result=capture_contact_candidate(step,state,memory,actions,grid,prediction)
    result.update(navigation_only=navigation_only,stream=stream)
    return result


def _axis_path_clear(grid,old,new):
    elbow=np.asarray([new[0],old[1]])
    return(circle_is_clear(grid,new,.2)and segment_is_clear(grid,old,elbow,.2)
           and segment_is_clear(grid,elbow,new,.2))


def _flatten(candidates):
    rows=[];first_wall={}
    for item in sorted(candidates,key=lambda x:(x.get('stream','navigation'),x['step'])):
        if(item.get('memory_provenance')!='actualownprefix'or item.get('new_observed_rgb_frames')!=0
                or type(item.get('navigation_only'))is not bool):
            raise ValueError('Contact replay accepts only genuine own state/h/z snapshots')
        states=item['state'].detach().float().cpu().numpy();maps=item['map'].detach().cpu().numpy()
        predictions=item['prediction'].detach().float().cpu().numpy()
        for actor in range(len(states)):
            key=(item['stream'],actor);old=states[actor,:2]*16
            for offset,new in enumerate(predictions[actor]):
                if not _axis_path_clear(maps[actor],old,new[:2]*16):
                    first_wall.setdefault(key,item['step']+offset+1);break
                old=new[:2]*16
            rows.append(dict(item=item,actor=actor,key=key,step=item['step'],state_numpy=states[actor].copy(),
                map_numpy=maps[actor].copy(),prediction_numpy=predictions[actor].copy()))
    for row in rows:
        wall=first_wall.get(row['key']);row['first_wall_step']=wall
        row['pre_wall']=wall is not None and row['step']<wall<=row['step']+len(row['prediction_numpy'])
        event=row['item'].get('source_event_offsets')
        row['pre_recorded_contact']=bool(event is not None and row['step']<event[row['actor']]<=row['step']+len(row['prediction_numpy']))
    return [r for r in rows if r['first_wall_step']is None or r['step']<r['first_wall_step']],first_wall


def _game_at(state,grid,*,navigation_only):
    if(not np.isfinite(state).all()or state[4]<=0 or not circle_is_clear(grid,state[:2]*16,.2)
            or not navigation_only and state[13]>=.5 and state[12]>0 and not circle_is_clear(grid,state[8:10]*16,.25)):
        raise ValueError('invalid_contact_own_state_geometry')
    game=Game(seed=0,endless=True,enemies_enabled=not navigation_only)
    game.map=np.asarray(grid,np.uint8).copy();game.map_height,game.map_width=game.map.shape
    game.set_state_vector(state)
    if not np.allclose(game.state_vector(),state,rtol=0,atol=2e-7):raise ValueError('noncanonical_contact_own_state_no_repair')
    return game


def _source_branch(row,horizon,dt):
    game=_game_at(row['state_numpy'],row['map_numpy'],navigation_only=row['item']['navigation_only'])
    actions=row['item']['actions'][row['actor'],:horizon].detach().float().cpu().numpy()
    labels=[row['state_numpy'].copy()]
    for action in actions:game.step(action,dt=dt);labels.append(game.state_vector().copy())
    labels=np.asarray(labels,np.float32)
    groups=_cohorts(torch.from_numpy(labels[:-1]),torch.from_numpy(labels[1:]),torch.from_numpy(actions),dt)
    cohort='partial'
    # Prefer the first real collision, then free motion. This is selection of
    # a training label after the scout, never a feature fed to the model.
    for name in('x_free_y_blocked','x_blocked_y_free','both_blocked','both_free','partial'):
        if groups[name].any():cohort=name;break
    return dict(row,labels=labels,cohort=cohort,horizon=len(actions))


def select_player_contact_branches(candidates,*,horizon,simulation_hz=30.,player_local_motion_protocol=None):
    if horizon not in HORIZONS:raise ValueError('Contact horizon must be4/8/16/32')
    local=normalize_player_local_motion_protocol(player_local_motion_protocol)
    if local is not None:
        return _select_player_local_branches(candidates,horizon=horizon,simulation_hz=simulation_hz,protocol=local)
    rows,first_wall=_flatten(candidates);skips=Counter();probed=[]
    def priority(row):
        event=row['item'].get('source_event_offsets')
        # Match the selected H, not a fixed16-tick preview. For H4 this selects
        # the last own prefix before the recorded contact, never an early free
        # window whose event falls outside the branch.
        event_in_horizon=event is not None and row['step']<event[row['actor']]<=row['step']+horizon
        return(not row['pre_wall'],not event_in_horizon,not row['pre_recorded_contact'],
            row['step'],row['item']['stream'],row['actor'])
    # Reserve both data sources. Stream-local actor IDs prevent a scout wall
    # at tick8 from incorrectly excluding a main-path actor at tick48.
    sets=[[r for r in rows if r['item']['stream']=='contact_short'],[r for r in rows if r['item']['stream']!='contact_short']]
    budgets=[16,16]if all(sets)else[MAX_SOURCE_PROBES if group else 0 for group in sets]
    attempts=0
    for group,budget in zip(sets,budgets):
        ordered=sorted(group,key=priority);full=[r for r in ordered if r['item']['actions'].shape[1]>=horizon]
        partial=[r for r in ordered if r['item']['actions'].shape[1]<horizon]
        for row in(full+partial)[:budget]:
            attempts+=1
            try:probed.append(_source_branch(row,horizon,1/simulation_hz))
            except ValueError as error:skips[str(error)]+=1
    probed.sort(key=priority)
    selected=[];used=set()
    for name in COHORTS[:4]+('partial',):
        same=[r for r in probed if r['cohort']==name]
        if same and len(selected)<MAX_BRANCHES:
            row=next((r for r in same if r['key']not in used),same[0]);selected.append(row);used.add(row['key'])
    for row in probed:
        if len(selected)>=MAX_BRANCHES:break
        if not any(row is old for old in selected):selected.append(row)
    return selected,dict(source_probes=attempts,legal_source_probes=len(probed),source_skip_counts=dict(skips),
        first_wall_actor_count=len(first_wall),selected_cohort_counts=dict(Counter(r['cohort']for r in selected)),
        selected_prefix_steps=[r['step']for r in selected],
        selected_branch_horizons=[r['horizon']for r in selected],
        selected_streams=[r['item']['stream']for r in selected],
        selected_profiles=[r['item']['navigation_only']for r in selected],
        selected_pre_wall_count=sum(r['pre_wall']for r in selected),
        axis_outcome_coverage_complete=all(any(r['cohort']==name for r in selected)for name in COHORTS[:4]))


def _player_axis_hint(row,dt):
    """Map-only sampling hint; the actual label is still checked by Source."""
    from .structured_state import player_command_delta
    state=row['state_numpy'];actions=row['item']['actions']
    if not actions.shape[1]:return 'partial'
    control=actions[row['actor'],0].detach().float().cpu()
    command=player_command_delta(torch.from_numpy(state[None]),control[None],dt=dt)[0].numpy()*16
    active=np.abs(command)>=16e-5
    if not active.all():return 'partial'
    old=state[:2]*16;grid=row['map_numpy'];elbow=old.copy();elbow[0]+=command[0]
    x_free=circle_is_clear(grid,elbow,.2)
    if not x_free:elbow[0]=old[0]
    end=elbow.copy();end[1]+=command[1]
    y_free=circle_is_clear(grid,end,.2)
    return ('both_free'if x_free and y_free else'x_free_y_blocked'if x_free else
            'x_blocked_y_free'if y_free else'both_blocked')


def _select_player_local_branches(candidates,*,horizon,simulation_hz,protocol):
    rows,first_wall=_flatten(candidates);dt=1/simulation_hz;skips=Counter()
    preferred=tuple(protocol['preferred_outcomes'])+('both_blocked','partial')
    def priority(row):
        gap=row['first_wall_step']-row['step']if row['pre_wall']else 1000000
        return(not row['pre_wall'],gap,row['item']['actions'].shape[1]<horizon,
               row['step'],row['item']['stream'],row['actor'])
    def ordered(pool):
        buckets={name:[]for name in preferred}
        for row in sorted(pool,key=priority):
            try:hint=_player_axis_hint(row,dt)
            except(ValueError,RuntimeError):hint='partial'
            buckets[hint].append(row)
        result=[]
        while any(buckets.values()):
            for name in preferred:
                if buckets[name]:result.append(buckets[name].pop(0))
        return result
    quiet=[r for r in rows if r['item']['stream']=='enemy_motion'and r['step']>0]
    others=[r for r in rows if not(r['item']['stream']=='enemy_motion'and r['step']>0)]
    quiet_budget=min(len(quiet),protocol['quiet_source_probe_budget'])
    other_budget=min(len(others),protocol['max_source_probes']-quiet_budget)
    quiet_budget=min(len(quiet),protocol['max_source_probes']-other_budget)
    probed=[];attempts=0;quiet_probes=0;ignored_npc=0
    for pool,budget,quiet_pool in((quiet,quiet_budget,True),(others,other_budget,False)):
        for row in ordered(pool)[:budget]:
            attempts+=1;quiet_probes+=int(quiet_pool)
            try:
                if not row['item']['actions'].shape[1]:raise ValueError('empty_player_motion_controls')
                action=row['item']['actions'][row['actor'],0].detach().float().cpu().numpy()
                label=source_player_motion_step(row['state_numpy'],action,row['map_numpy'],dt=dt)
                masks=_cohorts(torch.from_numpy(label['previous'][None]),torch.from_numpy(label['target'][None]),
                    torch.from_numpy(action[None]),dt)
                cohort=next((name for name in preferred if masks[name].any()),'partial')
                probed.append(dict(row,cohort=cohort,horizon=min(horizon,row['item']['actions'].shape[1]),
                    selection_player_label=label))
                ignored_npc+=int(label['own_NPC_body_invalid'])
            except ValueError as error:skips[str(error)]+=1
    selected=[];used_keys=set()
    def add(pool,count):
        for _ in range(count):
            available=[r for r in pool if not any(r is old for old in selected)]
            if not available:break
            present={r['cohort']for r in selected}
            # Capture near the first observed own wall, then preserve both
            # sliding orientations/free motion and prefer distinct own actors.
            row=min(available,key=lambda r:(not r['pre_wall'],r['cohort']in present,
                preferred.index(r['cohort']),r['key']in used_keys,priority(r)))
            selected.append(row);used_keys.add(row['key'])
    legal_quiet=[r for r in probed if r['item']['stream']=='enemy_motion'and r['step']>0]
    add(legal_quiet,protocol['quiet_nonzero_prefix_branches'])
    quiet_reserved=len(selected)
    add(probed,protocol['max_contact_branches']-len(selected))
    quiet_selected=sum(r['item']['stream']=='enemy_motion'and r['step']>0 for r in selected)
    return selected,dict(source_probes=attempts,legal_source_probes=len(probed),source_skip_counts=dict(skips),
        first_wall_actor_count=len(first_wall),selected_cohort_counts=dict(Counter(r['cohort']for r in selected)),
        selected_prefix_steps=[r['step']for r in selected],selected_branch_horizons=[r['horizon']for r in selected],
        selected_streams=[r['item']['stream']for r in selected],
        selected_profiles=[r['item']['navigation_only']for r in selected],
        selected_pre_wall_count=sum(r['pre_wall']for r in selected),
        axis_outcome_coverage_complete=all(any(r['cohort']==name for r in selected)for name in COHORTS[:4]),
        preferred_axis_outcome_coverage_complete=all(any(r['cohort']==name for r in selected)for name in protocol['preferred_outcomes']),
        player_local_motion_protocol=protocol['protocol'],player_only_Source_labels=True,
        quiet_nonzero_source_probes=quiet_probes,quiet_nonzero_legal_probes=len(legal_quiet),
        quiet_nonzero_requested=protocol['quiet_nonzero_prefix_branches'],quiet_nonzero_selected=quiet_selected,
        quiet_nonzero_reserved=quiet_reserved,
        quiet_nonzero_shortfall=max(0,protocol['quiet_nonzero_prefix_branches']-quiet_selected),
        quiet_nonzero_missing_reason=('no_legal_nonzero_own_quiet_prefix'if quiet_selected<protocol['quiet_nonzero_prefix_branches']else None),
        selection_player_labels_despite_invalid_NPC=ignored_npc,
        quiet_nonzero_distinct_actors=len({r['key']for r in selected if r['item']['stream']=='enemy_motion'and r['step']>0}))


def _scout(model,batch,amp):
    states,frames,actions,_,grid=_batch_data(batch,1)
    if actions.shape!=(4,32,4)or bool((states[...,12:14]!=0).any()):
        raise ValueError('Short contact scout requires four no-enemy32-transition TRAIN examples')
    autocast=nullcontext if amp is None else amp
    snapshots={};predictions=[]
    with environment_profile(model,navigation_only=True),torch.no_grad(),autocast():
        features=model.encode_map(grid).detach();memory=_initialize(model,frames,states,features,1,1)
        current=states[:,0];shape=tuple(memory.shape)
        for tick in range(32):
            if tick%4==0:snapshots[tick]=(current.detach().clone(),memory.detach().clone())
            result=model.transition(current,actions[:,tick],features,memory);_valid_result(result,shape,'Contact scout')
            current,memory=result['state'],result['memory'];predictions.append(current.detach())
    candidates=[]
    # Future recorded event offsets are consulted only after every scout
    # transition finished. The actual network saw one initial RGB/state only.
    events=batch.get('contact_event_offsets')
    for tick,(state,memory)in snapshots.items():
        item=capture_player_contact_candidate(tick,state,memory,actions[:,tick:tick+32],grid,
            torch.stack(predictions[tick:min(32,tick+16)],1),navigation_only=True,stream='contact_short')
        if events is not None:item['source_event_offsets']=list(events)
        candidates.append(item)
    return candidates


def contact_replay_backward(model,batch,*,candidates=None,horizon=16,weight=.25,amp=None,check_gradients=True,
                            player_local_motion_protocol=None):
    """Add only player gradients, without an optimizer step or state correction."""
    if horizon not in HORIZONS or type(weight)not in(int,float)or not math.isfinite(weight)or weight<0:
        raise ValueError('Player contact replay needs finite weight and4/8/16/32 horizon')
    local=normalize_player_local_motion_protocol(player_local_motion_protocol)
    if local is not None:
        return _player_local_contact_backward(model,batch,candidates=candidates,horizon=horizon,weight=weight,
            amp=amp,check_gradients=check_gradients,protocol=local)
    scout=_scout(model,batch,amp)
    chosen,selection=select_player_contact_branches([*(candidates or []),*scout],horizon=horizon,
        simulation_hz=model.config.simulation_hz)
    metrics=dict(protocol=PROTOCOL,player_contact_loss=0.,player_contact_horizon=horizon,
        scout_forward_transitions=128,branches=len(chosen),state_resets_after_prefix=0,memory_resets_after_prefix=0,
        prior_initial_memory_preserved=True,new_observed_rgb_frames_after_prefix=0,
        source_future_feedback=False,Source_labels_after_own_rollout=True,optimizer_steps=0,renderer_calls=0,
        tbptt=TBPTT,player_contact_tbptt=TBPTT,
        **selection)
    if not chosen:return metrics
    autocast=nullcontext if amp is None else amp;counts=Counter(r['cohort']for r in chosen)
    records=[];trajectory=[];wall=[];offset_errors=defaultdict(list);dt=1/model.config.simulation_hz
    config={**CONTACT_PROTOCOL_V2,'wall_margin_tiles':.02}
    # Each selected source trajectory is rooted at its OWN numerical prefix.
    # Group normalization gives equal present outcome mass across branches.
    for row in chosen:
        actor=row['actor'];item=row['item'];length=row['horizon']
        initial=item['state'][actor:actor+1].detach();memory=item['memory'][actor:actor+1].detach()
        grid=item['map'][actor:actor+1].detach();actions=item['actions'][actor:actor+1,:length].detach()
        targets=torch.from_numpy(row['labels']).to(device=initial.device,dtype=initial.dtype)
        branch_weight=1/(len(counts)*counts[row['cohort']]);current=initial;points=[];previous=[]
        with environment_profile(model,navigation_only=item['navigation_only']),autocast():
            with torch.no_grad():features=model.encode_map(grid).detach()
            shape=tuple(memory.shape)
            for tick in range(length):
                # Own-prefix scout history is detached once at entry. All
                # H<=32 branch transitions retain memory/coordinate credit,
                # within the newly authorized TBPTT64 limit.
                previous.append(current)
                raw=model.transition(current.detach(),actions[:,tick],features,memory)
                _valid_result(raw,shape,'Own contact branch')
                if raw.get('player_pass_logits')is None:raise ValueError('Contact branches require actual axis-pass logits')
                integrated=integrate_coordinate_gradients(current,raw,COORDINATE_GRADIENT_PROTOCOL)
                # Retain only the two outputs used by the local losses;
                # unrelated enemy/combat/readout outputs need no extra graph
                # references across four complete short branches.
                records.append(dict(raw={'state':raw['state'],'player_pass_logits':raw['player_pass_logits']},
                    previous=current.detach(),action=actions[:,tick],grid=grid,
                    navigation_only=item['navigation_only']))
                current,memory=integrated['state'],integrated['memory'];points.append(current)
            prediction=torch.stack(points,1);own_previous=torch.stack(previous,1)
            alive=(targets[:-1,4]>0)&(targets[1:,4]>0);denominator=alive.sum().clamp_min(1)
            xy=F.smooth_l1_loss(prediction[0,:,:2].float()*16,targets[1:,:2].detach().float()*16,reduction='none').sum(-1)
            trajectory.append(branch_weight*(xy*alive).sum()/(2*denominator))
            wall.append(swept_player_wall_loss(own_previous[:,:,:2],prediction[:,:,:2],
                targets[None,:-1,:2],targets[None,1:,:2],grid,
                branch_weight*alive[None].float()/denominator,contact_protocol=config))
            for offset in HORIZONS:
                if offset<=length:offset_errors[offset].append(float(((prediction[0,offset-1,:2].detach()-targets[offset,:2])*16).norm()))
    # The Source is consulted only AFTER all selected neural branches finished.
    # A legal own state provides a local label; invalid states are never fixed.
    direct=[];skips=Counter()
    for record in records:
        old=record['previous'][0].detach().float().cpu().numpy();grid=record['grid'][0].detach().cpu().numpy()
        action=record['action'][0].detach().float().cpu().numpy()
        try:
            game=_game_at(old,grid,navigation_only=record['navigation_only']);game.step(action,dt=dt)
            direct.append((record,old.copy(),game.state_vector().copy(),action.copy()))
        except ValueError as error:skips[str(error)]+=1
    zero=trajectory[0]*0;pass_loss=velocity_loss=zero
    groups={};velocity_mae={};false_stops=missed_blocks=0
    if direct:
        device=records[0]['previous'].device
        tensor=lambda values:torch.from_numpy(np.asarray(values,np.float32)).to(device)
        old=tensor([row[1]for row in direct]);target=tensor([row[2]for row in direct]);actions=tensor([row[3]for row in direct])
        supervision=prepare_kinematic_supervision(old[:,None],target[:,None],actions[:,None],
            kinematic_protocol=KINEMATIC_PROTOCOL_V1,simulation_hz=model.config.simulation_hz)
        fraction,active,_=pass_fraction_targets(old,target,actions,dt=dt);weights=supervision['player_axis_weight'][:,0]
        logits=torch.cat([row[0]['raw']['player_pass_logits']for row in direct]).float()
        delta=torch.cat([row[0]['raw']['state'][:,:2]-row[0]['previous'][:,:2]for row in direct]).float()
        expected=target[:,:2]-old[:,:2]
        pass_loss=(F.binary_cross_entropy_with_logits(logits,fraction,reduction='none')*weights).sum()
        velocity_loss=(F.smooth_l1_loss(delta*(16/(2.4*dt)),expected*(16/(2.4*dt)),reduction='none')*weights).sum()
        groups={name:[int(v)for v in value.tolist()]for name,value in supervision['player_axis_group_counts'].items()}
        error=(delta.detach()-expected).abs()*(16/dt)
        for axis,name in enumerate(('x','y')):
            mask=active[:,axis];velocity_mae[name]=float((error[:,axis]*mask).sum()/mask.sum().clamp_min(1))
        commands=pass_fraction_targets(old,target,actions,dt=dt)[2].abs()
        actual=delta.detach().abs();free=active&(fraction>=.95);blocked=active&(fraction<=.05)
        false_stops=int((free&(actual<.2*commands)).sum());missed_blocks=int((blocked&(actual>.2*commands)).sum())
    xy=sum(trajectory);geometry=sum(wall);loss=weight*(xy+geometry+pass_loss+velocity_loss)
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite own-memory player contact loss')
    loss.backward()
    if check_gradients:finite_gradient_norm(model.named_parameters(),context='Player contact replay')
    metrics.update(player_contact_loss=float(loss.detach()),xy_loss=float(xy.detach()),wall_loss=float(geometry.detach()),
        axis_bce_loss=float(pass_loss.detach()),axis_velocity_loss=float(velocity_loss.detach()),weight=float(weight),
        forward_transitions=len(records),direct_source_attempts=len(records),direct_source_valid=len(direct),
        direct_source_skip_counts=dict(skips),axis_group_counts=groups,axis_velocity_mae_cells_per_second=velocity_mae,
        false_stop_axes=false_stops,missed_block_axes=missed_blocks,backward_calls=1,
        xy_offsets_tiles_mean={str(key):sum(values)/len(values)for key,values in offset_errors.items()})
    return metrics


def _player_local_contact_backward(model,batch,*,candidates,horizon,weight,amp,check_gradients,protocol):
    """Recover own player motion without fabricating a no-NPC future tape."""
    scout=_scout(model,batch,amp)
    chosen,selection=select_player_contact_branches([*(candidates or []),*scout],horizon=horizon,
        simulation_hz=model.config.simulation_hz,player_local_motion_protocol=protocol)
    metrics=dict(protocol='rssm_player_short_contact_replay_v2',player_contact_loss=0.,
        player_contact_horizon=horizon,scout_forward_transitions=128,branches=len(chosen),
        state_resets_after_prefix=0,memory_resets_after_prefix=0,prior_initial_memory_preserved=True,
        new_observed_rgb_frames_after_prefix=0,source_future_feedback=False,Source_labels_after_own_rollout=True,
        optimizer_steps=0,renderer_calls=0,tbptt=TBPTT,player_contact_tbptt=TBPTT,
        player_local_motion_settings=deepcopy(protocol),**selection)
    if not chosen:return metrics
    autocast=nullcontext if amp is None else amp;dt=1/model.config.simulation_hz
    branch_counts=Counter(row['cohort']for row in chosen);records=[]
    # Finish every own rollout before consulting direct Source movement labels.
    # The full original NPC state and actual h/z remain neural inputs.
    for branch,row in enumerate(chosen):
        actor=row['actor'];item=row['item'];length=row['horizon']
        current=item['state'][actor:actor+1].detach();memory=item['memory'][actor:actor+1].detach()
        grid=item['map'][actor:actor+1].detach();actions=item['actions'][actor:actor+1,:length].detach()
        branch_weight=1/(len(branch_counts)*branch_counts[row['cohort']])
        with environment_profile(model,navigation_only=item['navigation_only']),autocast():
            with torch.no_grad():features=model.encode_map(grid).detach()
            shape=tuple(memory.shape)
            for tick in range(length):
                old=current.detach()
                raw=model.transition(old,actions[:,tick],features,memory)
                _valid_result(raw,shape,'Player-only own recovery branch')
                if raw.get('player_pass_logits')is None:raise ValueError('Player recovery requires axis-pass logits')
                records.append(dict(branch=branch,tick=tick,weight=branch_weight,previous=old,
                    prediction=raw['state'],pass_logits=raw['player_pass_logits'],
                    action=actions[:,tick],grid=grid))
                integrated=integrate_coordinate_gradients(current,raw,COORDINATE_GRADIENT_PROTOCOL)
                current,memory=integrated['state'],integrated['memory']
    direct=[];skips=Counter()
    for record in records:
        old=record['previous'][0].detach().float().cpu().numpy()
        control=record['action'][0].detach().float().cpu().numpy()
        grid=record['grid'][0].detach().cpu().numpy()
        try:
            label=source_player_motion_step(old,control,grid,dt=dt)
            direct.append((record,label))
        except ValueError as error:skips[str(error)]+=1
    metrics.update(forward_transitions=len(records),direct_source_attempts=len(records),
        direct_source_valid=len(direct),direct_source_skip_counts=dict(skips),
        direct_Source_scope='exact_single_player_movement_step',
        direct_labels_despite_invalid_NPC=sum(int(label['own_NPC_body_invalid'])for _,label in direct),
        xy_loss_semantics='contiguous_integrated_exact_local_delta_error',
        fabricated_no_NPC_future_trajectory=False,xy_offsets_tiles_mean={})
    if not direct:
        metrics.update(backward_calls=0,axis_group_counts={},false_stop_axes=0,missed_block_axes=0)
        return metrics
    device=records[0]['previous'].device;dtype=records[0]['previous'].dtype
    old=torch.cat([r['previous']for r,_ in direct]).detach()
    prediction=torch.cat([r['prediction']for r,_ in direct])
    actions=torch.cat([r['action']for r,_ in direct]).detach()
    logits=torch.cat([r['pass_logits']for r,_ in direct])
    target=torch.as_tensor(np.stack([label['target']for _,label in direct]),device=device,dtype=dtype).detach()
    local,diagnostics=player_motion_supervised_terms(old,prediction,actions,target,logits,
        player_local_motion_protocol=protocol,simulation_hz=model.config.simulation_hz)
    # Accumulate one-step displacement residuals only over contiguous valid
    # own ticks. Invalid/dead player steps break the chain instead of obtaining
    # an invented Source target. No label is fed back into the neural trajectory.
    by_branch=defaultdict(list)
    for index,(record,label)in enumerate(direct):by_branch[record['branch']].append((index,record))
    zero=prediction.sum()*0;trajectory=zero;sample_weights=[];offsets=defaultdict(list)
    residual=(prediction[:,:2]-old[:,:2]-(target[:,:2]-old[:,:2]))*16
    for branch,items in by_branch.items():
        segments=[];segment=[];last_tick=None
        for index,record in items:
            if last_tick is not None and record['tick']!=last_tick+1:
                segments.append(segment);segment=[]
            segment.append(index);last_tick=record['tick']
        if segment:segments.append(segment)
        branch_weight=items[0][1]['weight'];count=len(items)
        for segment in segments:
            cumulative=residual[segment].cumsum(0)
            trajectory=trajectory+branch_weight*F.smooth_l1_loss(cumulative,torch.zeros_like(cumulative),reduction='sum')/(2*count)
            for offset in HORIZONS:
                if offset<=len(segment):offsets[offset].append(float(cumulative[offset-1].detach().norm()))
    for record,_ in direct:sample_weights.append(record['weight']/len(by_branch[record['branch']]))
    grids=torch.cat([record['grid']for record,_ in direct]).detach()
    config={**CONTACT_PROTOCOL_V2,'wall_margin_tiles':.02}
    wall=swept_player_wall_loss(old[:,None,:2],prediction[:,None,:2],old[:,None,:2],target[:,None,:2],
        grids,torch.as_tensor(sample_weights,device=device,dtype=dtype)[:,None],contact_protocol=config)
    loss=weight*(trajectory+wall+sum(local.values()))
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite player local recovery objective')
    loss.backward()
    if check_gradients:finite_gradient_norm(model.named_parameters(),context='Player local motion recovery')
    metrics.update(player_contact_loss=float(loss.detach()),xy_loss=float(trajectory.detach()),wall_loss=float(wall.detach()),
        axis_bce_loss=float(local['pass_bce'].detach()),axis_velocity_loss=float(local['axis_velocity'].detach()),
        free_progress_loss=float(local['free_progress'].detach()),weight=float(weight),backward_calls=1,
        axis_group_counts=diagnostics['axis_group_counts'],
        axis_velocity_mae_cells_per_second=diagnostics['axis_velocity_mae_cells_per_second'],
        false_stop_axes=diagnostics['false_stop_axes'],missed_block_axes=diagnostics['missed_block_axes'],
        player_local_motion=diagnostics,
        player_local_cumulative_residual_tiles_mean={str(k):sum(v)/len(v)for k,v in offsets.items()})
    return metrics
