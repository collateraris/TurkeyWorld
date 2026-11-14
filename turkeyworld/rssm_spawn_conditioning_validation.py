"""Independent read-only evidence for own-clock/candidate spawn conditioning."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .spawn import SPAWN_ANCHORS
from .rssm_spawn_curriculum import file_sha,source_recovery_labels
from .staged_validation import _isolated_validation


def _legal_reference(state,floor):
    player=np.asarray(state[:2],np.float64)*16
    npc=np.asarray(state[8:10],np.float64)*16
    alive=bool(state[13]>=.5 and state[12]>0)
    legal=[]
    for y in range(16):
        for x in range(16):
            point=np.asarray([x+.5,y+.5])
            allowed=bool(floor[y,x]) and float(((point-player)**2).sum())>=16
            allowed=allowed and not(alive and float(((point-npc)**2).sum())<1)
            if allowed:legal.append((x,y))
    flags=[anchor in legal for anchor in SPAWN_ANCHORS]
    fallback=[(x+.5,y+.5)for x,y in legal if(x,y)not in SPAWN_ANCHORS]
    center=np.mean(fallback,axis=0)/16 if fallback else np.zeros(2)
    return flags+[bool(fallback)],center


def audit_spawn_conditioning_inputs(module,*,expect_zero=False):
    """8 phases ×10 safety cases, 24 phase edges, and three death-clock edges.

    The reference enumerates ordinary floor/distance safety independently in
    Python. It neither supplies ranks/logits nor chooses a runtime spawn.
    """
    from .spawn import decode_spawn_positions
    device=next(module.parameters()).device
    states,maps,expected_ticks,expected_legality,expected_centers,cases=[],[],[],[],[],[]
    for phase in range(8):
        for case in range(10):
            state=np.zeros(20,np.float32);state[:2]=np.asarray([8.5,8.5])/16
            state[4]=1;state[8:10]=np.asarray([2.5,2.5])/16;state[16]=89/18
            tick=phase*90+45;state[19]=(tick-1)/9000
            floor=np.ones((16,16),bool)
            if case==1:floor[2,2]=False
            if case==2:state[:2]=np.asarray([2.5,2.5])/16
            if case==3:state[12:14]=1
            if case==4:floor[:]=False
            if case==5:
                floor[:]=False
                for x,y in SPAWN_ANCHORS:floor[y,x]=True
            if case==6:floor[:]=False;floor[1,1]=True
            if case==7:state[:2]=np.asarray([6.5,2.5])/16
            if case==8:state[:2]=np.asarray([6.499,2.5])/16
            if case==9:state[8:10]=np.asarray([3.5,2.5])/16;state[12:14]=1
            flags,center=_legal_reference(state,floor)
            states.append(state);maps.append(floor);expected_ticks.append(tick)
            expected_legality.append(flags);expected_centers.append(center);cases.append(dict(phase=phase,case=case))
    own=torch.tensor(np.asarray(states),device=device)
    floor=torch.tensor(np.asarray(maps),device=device)
    def forward(state,walkable):
        return module(state,torch.zeros(len(state),module.width,device=device),
                      torch.zeros(len(state),2,module.width,device=device),walkable,dt=1/30)
    with torch.no_grad():
        out=forward(own,floor)
        boundaries=[];desired=[]
        for phase in range(1,9):
            for delta in(-1,0,1):
                tick=phase*90+delta;row=own[0].clone();row[19]=(tick-1)/9000
                boundaries.append(row);desired.append(tick)
        edge=forward(torch.stack(boundaries),floor[:1].expand(len(boundaries),-1,-1))
        timers=own[:3].clone();timers[:,16]=timers.new_tensor([88/18,89/18,90/18])
        _,available=decode_spawn_positions(torch.zeros(3,1,256,device=device),timers,floor[:3],1/30)
    clock_checks=dict(next_tick_exact=out['next_tick'].cpu().tolist()==expected_ticks,
        eight_phases=out['phase_index'].cpu().tolist()==[c['phase']for c in cases],
        one_hot_exact=bool(torch.equal(out['phase_one_hot'],torch.nn.functional.one_hot(out['phase_index'],8))),
        phase_edges_24=all(int(t)==wanted and int(p)==(wanted//90)%8 for t,p,wanted in
                          zip(edge['next_tick'],edge['phase_index'],desired)),
        within_phase_exact=bool(torch.allclose(edge['within_phase'].double().cpu(),
            torch.tensor([(t%90)/90 for t in desired],dtype=torch.float64),rtol=0,atol=5e-8)),
        candidate_legal_python_parity=out['candidate_legal'].cpu().tolist()==expected_legality,
        fallback_mean_python_parity=bool(torch.allclose(out['candidate_xy'][:,8].double().cpu(),
            torch.tensor(np.asarray(expected_centers)),rtol=0,atol=1e-7)),
        candidate_geometry_independent_of_clock=all(torch.equal(out['candidate_features'][:10],out['candidate_features'][p*10:(p+1)*10])for p in range(1,8)),
        respawn_boundary_88_89_90=available[:,0].cpu().tolist()==[False,True,True],
        finite=all(bool(torch.isfinite(v).all())for v in out.values()if torch.is_tensor(v)))
    if expect_zero:clock_checks['zero_initial_residual']=bool((out['anchor_residual']==0).all())
    return dict(kind='independent_spawn_conditioning_input_audit_v1',passed=all(clock_checks.values()),
        checks=clock_checks,synthetic_cases=len(cases),phase_boundary_cases=len(desired),
        source_future_feedback=False,external_engine_calls=0,Source_rank_or_choice_supplied=False,
        cases=[dict(c,next_tick=int(out['next_tick'][i]),legal=out['candidate_legal'][i].cpu().tolist())for i,c in enumerate(cases)])


def build_spawn_event_panel(data,*,context=1,allow_train=False):
    allowed=('train','val','test') if allow_train is True else('val','test')
    if getattr(data,'split',None)not in allowed or type(context)is not int or context<1:
        raise ValueError('Spawn event panels require held-out data and positive observed context')
    rows,files=[],[]
    for episode,info in enumerate(data.episodes):
        path=Path(info['path'])
        with np.load(path,allow_pickle=False)as saved:
            labels=source_recovery_labels(saved['states'],saved['attack_flags'],float(saved['dt']))
        files.append(dict(episode=path.name,sha256=file_sha(path)))
        for tick in np.flatnonzero(labels['spawn']):
            if tick>=context-1:rows.append(dict(episode=episode,start=int(tick)-context+1,
                transition=int(tick),anchor=int(labels['anchor'][tick]),clock_phase=int(labels['clock_phase'][tick])))
    if not rows:raise ValueError('Held-out Source panel contains no actual revivals')
    spec=dict(context=context,source_files=files,events=rows,kind='all_long_val_spawn_events_v1')
    return dict(rows=rows,source_files=files,context=context,source_split=data.split,
                fingerprint=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest())


def audit_Source_player_step_legality(data,panel):
    """Compare recorded birth position against previous and next player XY.

    This is diagnostic only; neither candidate mask nor the Source ordering is
    changed. Original Game.step respawns before moving the player.
    """
    examples=[];previous_illegal=target_illegal=only_previous_illegal=only_target_illegal=0
    by_episode={}
    for row in panel['rows']:by_episode.setdefault(row['episode'],[]).append(row)
    for episode,rows in by_episode.items():
        with np.load(data.episodes[episode]['path'],allow_pickle=False)as saved:
            states=np.asarray(saved['states']);floor=np.asarray(saved['map'])==0
            for row in rows:
                tick=row['transition'];old,new=states[tick],states[tick+1]
                point=new[8:10].astype(np.float64)*16
                cell=np.floor(point).astype(int)
                previous_distance=float(np.linalg.norm(point-old[:2].astype(np.float64)*16))
                target_distance=float(np.linalg.norm(point-new[:2].astype(np.float64)*16))
                valid_floor=0<=cell[0]<16 and 0<=cell[1]<16 and bool(floor[cell[1],cell[0]])
                previous_valid=valid_floor and previous_distance>=4
                target_valid=valid_floor and target_distance>=4
                previous_illegal+=int(not previous_valid);target_illegal+=int(not target_valid)
                only_previous_illegal+=int(not previous_valid and target_valid)
                only_target_illegal+=int(previous_valid and not target_valid)
                if previous_valid!=target_valid:
                    examples.append(dict(**row,previous_player_distance_tiles=previous_distance,
                        target_player_distance_tiles=target_distance,previous_legal=previous_valid,target_legal=target_valid))
    return dict(events=len(panel['rows']),panel_fingerprint=panel['fingerprint'],
        Source_order='clock_and_dead_timers_then_respawn_then_player_move',
        illegal_using_previous_player=previous_illegal,illegal_using_target_player=target_illegal,
        illegal_previous_but_legal_target=only_previous_illegal,
        legal_previous_but_illegal_target=only_target_illegal,examples=examples,
        changed_rules=False,source_future_feedback=False)


def spawn_confusion(result,targets):
    logits=result.get('spawn_anchor_logits')
    if not torch.is_tensor(logits)or logits.shape!=(len(targets),1,9)or not bool(torch.isfinite(logits).all()):
        raise ValueError('Spawn confusion requires finite actual nine-class anchor logits')
    labels=torch.tensor([r['anchor']for r in targets],device=logits.device)
    predicted=logits[:,0].argmax(-1)
    confusion=torch.bincount(labels*9+predicted,minlength=81).reshape(9,9)
    report=dict(events=len(targets),raw_anchor_confusion=confusion.cpu().tolist(),
        raw_anchor_top1_accuracy=float((labels==predicted).double().mean()),
        raw_anchor_CE=float(torch.nn.functional.cross_entropy(logits[:,0].float(),labels)),
        per_phase={},source_future_feedback=False)
    legal=result.get('spawn_anchor_legal')
    if torch.is_tensor(legal):
        if legal.shape!=logits.shape or legal.dtype!=torch.bool:raise ValueError('Spawn legality shape differs')
        selected=logits[:,0].masked_fill(~legal[:,0],-torch.finfo(logits.dtype).max).argmax(-1)
        report.update(legal_anchor_top1_accuracy=float((labels==selected).double().mean()),
            Source_anchor_own_legal_count=int(legal[:,0].gather(1,labels[:,None]).sum()))
    for phase in range(8):
        ids=[i for i,r in enumerate(targets)if r['clock_phase']==phase]
        report['per_phase'][str(phase)]=dict(events=len(ids),raw_anchor_top1_accuracy=(
            float((labels[ids]==predicted[ids]).double().mean())if ids else None))
    return report


def evaluate_spawn_conditioning_events(model,data,panel,*,batch_size=32):
    """Teacher-forced one-step event queries; future states serve labels only."""
    rows=panel['rows'];context=panel['context'];outputs={};sampler=(deepcopy(data.sampler_state())
        if callable(getattr(data,'sampler_state',None))else None)
    try:
        with _isolated_validation(model,data,173):
            for first in range(0,len(rows),batch_size):
                chosen=rows[first:first+batch_size];indices=np.asarray([[r['episode'],r['start']]for r in chosen],np.int64)
                batch=data.batch(len(chosen),context=context,horizon=1,indices=indices)
                fmap=model.encode_map(batch['map'])
                memory=model.initialize_from_image(batch['context_frames'],batch['states'][:,:context],fmap)
                result=model.transition(batch['states'][:,context-1],batch['actions'][:,0],fmap,memory)
                for key in('spawn_anchor_logits','spawn_anchor_legal'):
                    if key in result:outputs.setdefault(key,[]).append(result[key].detach().cpu())
        result={k:torch.cat(v)for k,v in outputs.items()}
        report=spawn_confusion(result,rows)
        report.update(kind='spawn_conditioning_fixed_Source_event_comparison_v1',
            panel_fingerprint=panel['fingerprint'],context=context,source_held_out=True,
            source_episode_count=len(panel['source_files']),observed_scope='causal past Source bootstrap then one own prior step',
            future_rgb_inputs=0,runtime_engine_calls=0,autonomous450_evaluated=False)
        return report
    finally:
        if sampler is not None:data.restore_sampler(sampler)
