"""Player-only long-rollout diagnostics and conservative whole-checkpoint selection."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np

PROTOCOL='rssm_player_navigation_validation_v1'
OFFSETS=(1,4,8,16,32,64,128,256,450)
PANELS={'navigation_context1':'navigation/one_frame450',
        'navigation_context2':'navigation/long450','combat_context1':'combat/trajectory'}


def _stats(values):
    values=np.asarray(values,dtype=np.float64)
    return dict(count=int(values.size),mean=float(values.mean())if values.size else None,
                p95=float(np.percentile(values,95))if values.size else None,
                maximum=float(values.max())if values.size else None)


def _clear(points,grid,radius=.20):
    """Vectorized equivalent of runtime_checks.circle_is_clear; no model query."""
    points=np.asarray(points,dtype=np.float64);shape=points.shape[:-1];flat=points.reshape(-1,2)
    cells=np.argwhere(np.asarray(grid)!=0)[:,::-1].astype(np.float64)
    result=np.ones(len(flat),dtype=bool)
    for first in range(0,len(flat),2048):
        p=flat[first:first+2048]
        clear=np.isfinite(p).all(-1)&(p[:,0]-radius>=0)&(p[:,1]-radius>=0)&(p[:,0]+radius<16)&(p[:,1]+radius<16)
        if len(cells):
            nearest=np.maximum(cells[None],np.minimum(p[:,None],cells[None]+1))
            distance=((p[:,None]-nearest)**2).sum(-1)
            clear&=~(distance<radius**2-1e-8).any(-1)
        result[first:first+len(p)]=clear
    return result.reshape(shape)


def _path_clear(previous,current,grid):
    previous,current=np.asarray(previous),np.asarray(current)
    elbow=np.stack((current[...,0],previous[...,1]),-1)
    largest=float(np.max(np.abs(current-previous)))
    count=max(2,min(4096,int(np.ceil(largest/.04))+1))
    fraction=np.linspace(0,1,count).reshape(*([1]*(current.ndim-1)),count,1)
    first=previous[...,None,:]+fraction*(elbow-previous)[...,None,:]
    second=elbow[...,None,:]+fraction*(current-elbow)[...,None,:]
    return _clear(first,grid).all(-1)&_clear(second,grid).all(-1)


def player_panel_metrics(predicted,reference,maps,*,panel_fingerprint):
    predicted,reference=(np.asarray(x,dtype=np.float64)for x in(predicted,reference))
    maps=np.asarray(maps)
    if(predicted.shape!=reference.shape or predicted.ndim!=3 or len(predicted)<1 or predicted.shape[1:]!=(451,20)
            or maps.shape!=(len(predicted),16,16)or not panel_fingerprint
            or not np.isfinite(predicted).all()or not np.isfinite(reference).all()or not np.isfinite(maps).all()):
        raise ValueError('Player selection requires finite matched held-out450 states and maps')
    alive=(reference[:,:-1,4]>0)&(reference[:,1:,4]>0)
    if not alive.any():raise ValueError('Player selection requires nonzero Source-alive panel support')
    error=np.linalg.norm((predicted[:,1:,:2]-reference[:,1:,:2])*16,axis=-1)
    walls=np.zeros_like(alive);paths=np.zeros_like(alive);source_walls=np.zeros_like(alive)
    first=[]
    for actor,grid in enumerate(maps):
        walls[actor]=~_clear(predicted[actor,1:,:2]*16,grid)
        paths[actor]=~_path_clear(predicted[actor,:-1,:2]*16,predicted[actor,1:,:2]*16,grid)
        source_walls[actor]=~_clear(reference[actor,1:,:2]*16,grid)
        hits=np.flatnonzero(walls[actor]&alive[actor]);first.append(int(hits[0]+1)if len(hits)else None)
    count=int(alive.sum());wall_count=int((walls&alive).sum());path_count=int((paths&alive).sum())
    digest=hashlib.sha256(np.ascontiguousarray(reference).tobytes()+np.ascontiguousarray(maps).tobytes()).hexdigest()
    return dict(protocol=PROTOCOL,panel_fingerprint=panel_fingerprint,source_reference_sha256=digest,
        horizon=450,actors=len(predicted),valid_source_player_transitions=count,
        source_life_mask_only=True,predicted_dead_does_not_hide_error=True,
        drift=_stats(error[alive]),final=_stats(error[:,-1][alive[:,-1]]),
        offsets={str(offset):_stats(error[:,offset-1][alive[:,offset-1]])for offset in OFFSETS},
        player_wall_steps=wall_count,player_wall_path_steps=path_count,
        player_wall_fraction=wall_count/max(1,count),player_wall_path_fraction=path_count/max(1,count),
        source_player_wall_steps=int((source_walls&alive).sum()),first_player_wall_step=first,
        source_alive_model_dead_steps=int((alive&(predicted[:,1:,4]<=0)).sum()),
        _errors=error,_alive=alive,_walls=walls,_paths=paths)


def annotate_player_navigation_validation(report,revision,directory):
    """Reuse the actual saved autonomous panels; perform no additional forward."""
    directory=Path(directory);panels={};all_errors=[];final_errors=[];wall_count=path_count=valid_count=0
    for key,relative in PANELS.items():
        root=directory/relative
        metadata=json.loads((root/'heldout_trajectory_validation.json').read_text(encoding='utf-8'))
        panel=metadata.get('panel',{})
        if(metadata.get('source_held_out')is not True or metadata.get('source_future_feedback')is not False
                or metadata.get('future_rgb_encoded')is not False or metadata.get('horizon')!=450
                or metadata.get('all_states_finite')is not True
                or metadata.get('all_memory_finite_and_shape_valid')is not True
                or panel.get('split')!='val'or panel.get('horizon')!=450
                or type(panel.get('context'))is not int
                or metadata.get('context')!=panel.get('context')
                or panel.get('context')!=(2 if key=='navigation_context2'else 1)
                or metadata.get('fingerprint')!=hashlib.sha256(json.dumps(panel,sort_keys=True).encode()).hexdigest()):
            raise ValueError('Player ranking requires actual finite autonomous held-out450 panels')
        with np.load(root/'heldout_trajectory_states.npz',allow_pickle=False)as saved:
            required={'predicted','reference','maps','indices','actions','dt'}
            if not required.issubset(saved.files):raise ValueError('Incomplete saved player validation panel')
            predicted,reference=saved['predicted'],saved['reference']
            if(predicted.ndim!=3 or predicted.shape[1:]!=(451,20)or reference.shape!=predicted.shape):
                raise ValueError('Saved player validation requires matched full450 state tapes')
            count=len(predicted);windows=panel.get('windows',[])
            indices=saved['indices'];actions=saved['actions'];dt=saved['dt']
            if(metadata.get('count')!=count or count<1 or len(windows)!=count
                    or indices.shape!=(count,2)or indices.dtype.kind not in 'iu'
                    or actions.shape!=(count,450,4)or dt.shape!=(count,)
                    or not np.isfinite(actions).all()or not np.isfinite(dt).all()
                    or not np.allclose(dt,1/30,rtol=0,atol=1e-7)
                    or not np.array_equal(predicted[:,0],reference[:,0])
                    or any(window.get('episode_index')!=int(indices[row,0])
                        or window.get('start')!=int(indices[row,1])
                        or not isinstance(window.get('episode'),str)or not window['episode']
                        or (indices[row]<0).any()for row,window in enumerate(windows))):
                raise ValueError('Saved player states do not match their held-out panel metadata')
            value=player_panel_metrics(saved['predicted'],saved['reference'],saved['maps'],
                                       panel_fingerprint=metadata['fingerprint'])
        if value['source_player_wall_steps']:
            raise ValueError('Reference player is inside a wall in the held-out panel')
        errors,alive=value.pop('_errors'),value.pop('_alive');value.pop('_walls');value.pop('_paths')
        all_errors.extend(errors[alive]);final_errors.extend(errors[:,-1][alive[:,-1]])
        wall_count+=value['player_wall_steps'];path_count+=value['player_wall_path_steps'];valid_count+=value['valid_source_player_transitions']
        panels[key]=value
    navigation=report.get('navigation',{});contacts={}
    for key,item in(('short',navigation.get('short',{})),('contact',navigation.get('contact_generalization',{}))):
        metrics=item.get('metrics',{})
        contacts[key]={name:metrics.get(name)for name in('false_stop_rate','missed_block_rate','sliding_velocity_error_tiles_s')}
    drift,final=_stats(all_errors),_stats(final_errors)
    score=drift['mean']/.1+(final['mean']or 0.)/.25+(final['p95']or 0.)/.5+10*(wall_count+path_count)/max(1,valid_count)
    value=dict(protocol=PROTOCOL,step=report['step'],revision_plan_sha256=revision['sha256'],
        source_step=revision['source_optimizer_step'],source_held_out=True,source_future_feedback=False,
        horizon=450,offsets=list(OFFSETS),panels=panels,player_drift=drift,player_final_drift=final,
        player_wall_steps=wall_count,player_wall_path_steps=path_count,valid_player_transitions=valid_count,
        player_wall_fraction=wall_count/max(1,valid_count),player_wall_path_fraction=path_count/max(1,valid_count),
        contact_axis_errors=contacts,score=float(score),
        ranking='lower normalized drift/final/p95 plus wall rates; no wall/path/mean-drift regression',
        training_loss_is_not_selection_metric=True,train_H450_goal_unchanged=True,automatic_runtime_deployment=False)
    report['player_navigation_validation']=value
    from .training_pipeline import atomic_json
    atomic_json(directory/'player_navigation_validation.json',value)
    return value


def player_navigation_improves(candidate,previous):
    if(candidate.get('protocol')!=PROTOCOL or candidate.get('horizon')!=450
            or not candidate.get('revision_plan_sha256')or candidate.get('valid_player_transitions',0)<1
            or not isinstance(candidate.get('score'),(int,float))or not math.isfinite(candidate['score'])):
        raise ValueError('Invalid player navigation selection score or Source support')
    if previous is None:return True
    for key in('protocol','revision_plan_sha256','valid_player_transitions','horizon'):
        if candidate.get(key)!=previous.get(key):raise ValueError('Player selection changed its revision or panel support')
    for name in PANELS:
        for key in('panel_fingerprint','source_reference_sha256','actors'):
            if candidate['panels'][name][key]!=previous['panels'][name][key]:
                raise ValueError('Player selection changed its held-out Source panel')
    if(candidate['player_wall_steps']>previous['player_wall_steps']
            or candidate['player_wall_path_steps']>previous['player_wall_path_steps']
            or candidate['player_drift']['mean']>previous['player_drift']['mean']+1e-12):return False
    return candidate['score']<previous['score']-1e-12


def publish_best_player_navigation(checkpoint,run_dir,*,report,validation_report):
    from .training_pipeline import atomic_json,replace_file
    run_dir=Path(run_dir);selector=run_dir/'best_player_navigation.json'
    previous=json.loads(selector.read_text(encoding='utf-8'))if selector.exists()else None
    score=report['player_navigation_validation']
    if not player_navigation_improves(score,previous['score']if previous else None):return previous
    checkpoint,validation_report=Path(checkpoint),Path(validation_report)
    saved_report=json.loads(validation_report.read_text(encoding='utf-8'))
    if(saved_report.get('step')!=report['step']or saved_report.get('player_navigation_validation')!=score
            or score.get('step')!=report['step']):
        raise ValueError('Player best checkpoint requires the exact matching saved validation report')
    def sha(path):
        with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    checkpoint_sha=sha(checkpoint);report_sha=sha(validation_report)
    directory=run_dir/'checkpoints/player_navigation_best';directory.mkdir(parents=True,exist_ok=True)
    candidate=directory/f"world_g{report['step']:06d}.pt"
    if candidate.exists()and sha(candidate)!=checkpoint_sha:
        raise ValueError('Existing player best candidate contains different checkpoint bytes at the same step')
    # Validate on CPU only for a genuine candidate. Ranking never substitutes
    # isolated player tensors into a different model/Adam/RNG boundary.
    import torch
    snapshot=torch.load(checkpoint,map_location='cpu',weights_only=False)
    try:
        step=report['step'];optimizer=snapshot.get('optimizer',{});run_state=snapshot.get('run_state')
        if(not isinstance(snapshot.get('model'),dict)or not snapshot['model']
                or not isinstance(optimizer,dict)or not optimizer.get('state')or not optimizer.get('param_groups')
                or snapshot.get('training_steps')!=step or snapshot.get('global_training_step')!=step
                or not isinstance(run_state,dict)or run_state.get('global_training_step')!=step):
            raise ValueError('Player best requires a complete model/Adam checkpoint at the validated optimizer step')
    finally:del snapshot
    if not candidate.exists():
        try:os.link(checkpoint,candidate)
        except OSError:shutil.copy2(checkpoint,candidate)
    if sha(candidate)!=checkpoint_sha:
        raise ValueError('Player best checkpoint changed during publication')
    alias=run_dir/'checkpoints/best_player_navigation.pt';temporary=alias.with_name(alias.name+'.tmp')
    if temporary.exists():temporary.unlink()
    try:os.link(candidate,temporary)
    except OSError:shutil.copy2(candidate,temporary)
    replace_file(temporary,alias)
    value=dict(protocol=PROTOCOL,step=report['step'],score=score,checkpoint=str(candidate.resolve()),
        checkpoint_sha256=checkpoint_sha,validation_report=str(validation_report.resolve()),
        validation_report_sha256=report_sha,contains_full_model_and_Adam=True,
        combined_quality_passed=bool(report.get('passed')),automatic_runtime_deployment=False)
    atomic_json(selector,value);return value
