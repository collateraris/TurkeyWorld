"""Advance pure-XY horizon after ten consecutive completed losses below0.25."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .rssm_drift_only_protocol import DRIFT_ONLY_PROTOCOL, frozen_balance_values, _digest


HORIZONS = (16,32,64,96,128,160,192,224,256,288,320,352,384,416,448,450)
LOSS_GATE_PROTOCOL = 'rssm_xy_loss_gated_horizon_v1'
SETTINGS = {'version':1, 'horizons':list(HORIZONS), 'threshold':.25, 'comparison':'strict_less_than',
    'required_consecutive_passes':10,
    'loss_source':'completed_optimizer_update.train_loss_including_main_and_late_xy',
    'stage_advances_per_update_max':1, 'microbatch_size':64, 'gradient_accumulation_steps':4,
    'effective_batch_size':256, 'forced_long_replay':False, 'ema_learning':False,
    'validation_horizon':450, 'optimizer_reset':False, 'source_future_feedback':False}


def loss_gated_curriculum_plan(state, *, source_checkpoint, source_checkpoint_sha256, base_plan_sha256):
    q=state['rssm_training']
    return {'version':1, 'curriculum_protocol':LOSS_GATE_PROTOCOL, 'settings':deepcopy(SETTINGS),
        'source_checkpoint':str(source_checkpoint), 'source_checkpoint_sha256':source_checkpoint_sha256,
        'start_optimizer_step':state['global_training_step'], 'base_plan_sha256':base_plan_sha256,
        'coordinate_plan_sha256':q['coordinate_curriculum']['sha256'],
        'replay_plan_sha256':q['replay_curriculum']['sha256'],
        'drift_only_plan_sha256':q['drift_only_revision']['sha256'],
        'frozen_ema_values_sha256':_digest(frozen_balance_values(q['loss_balance_state']))}


def _boundary(state):
    from .rssm_drift_only_protocol import _boundary as drift_boundary
    q,step=drift_boundary(state)
    if (state.get('objective_protocol')!=DRIFT_ONLY_PROTOCOL
            or q.get('drift_only_revision',{}).get('last_optimizer_step')!=step):
        raise ValueError('Loss-gated horizons preserve a complete pure-XY optimizer boundary')
    return q,step


def loss_gated_training_horizon(state):
    gate=state['rssm_training']['loss_gated_curriculum']
    index=gate.get('stage_index')
    if (type(index)is not int or not 0<=index<len(HORIZONS)
            or gate.get('current_horizon')!=HORIZONS[index]):
        raise ValueError('Loss-gated stage and next training horizon disagree')
    horizon=HORIZONS[index]
    return {'horizon':horizon, 'curriculum_horizon':horizon, 'forced_long_replay':False,
            'ema_bank':'long450'if horizon==450 else'short'}


def loss_gated_coordinate_expectation(state, completed_coordinate_updates):
    """Retain pre-gate history and add only horizons actually trained here."""
    gate=state['rssm_training']['loss_gated_curriculum']
    if completed_coordinate_updates-gate['coordinate_updates_at_start']!=gate['updates']:
        raise ValueError('Loss gate and old coordinate clocks disagree')
    histogram=deepcopy(gate['coordinate_horizon_counts_at_start'])
    for horizon,count in gate['horizon_update_counts'].items():
        if count:histogram[horizon]=histogram.get(horizon,0)+count
    return histogram,sum(int(key)*count for key,count in histogram.items()),gate['current_horizon']


def bind_loss_gated_curriculum(state, plan, *, path, sha256, source_checkpoint_sha256,
                                loaded_checkpoint_sha256, base_plan_sha256):
    q,step=_boundary(state)
    expected=loss_gated_curriculum_plan(state,source_checkpoint=plan.get('source_checkpoint',''),
        source_checkpoint_sha256=source_checkpoint_sha256,base_plan_sha256=base_plan_sha256)
    expected['start_optimizer_step']=plan.get('start_optimizer_step')
    raw=Path(path).read_bytes()
    if (not sha256 or not source_checkpoint_sha256 or not base_plan_sha256 or not plan.get('source_checkpoint')
            or type(plan.get('start_optimizer_step'))is not int or not 0<=plan['start_optimizer_step']<=step
            or json.dumps(plan,sort_keys=True)!=json.dumps(expected,sort_keys=True)
            or hashlib.sha256(raw).hexdigest()!=sha256 or json.loads(raw.decode('utf-8'))!=plan):
        raise ValueError('Loss gate source, threshold, stage path or parent identities changed')
    gate=q.get('loss_gated_curriculum')
    if gate is None:
        if plan['start_optimizer_step']!=step or loaded_checkpoint_sha256!=source_checkpoint_sha256:
            raise ValueError('First loss-gated bind requires the exact full saved pure-XY boundary')
        gate={'version':1,'curriculum_protocol':LOSS_GATE_PROTOCOL,'sha256':sha256,'plan':str(path),
            'start_optimizer_step':step,'last_optimizer_step':step,'updates':0,
            'initial_loaded_checkpoint_sha256':loaded_checkpoint_sha256,
            'settings':deepcopy(SETTINGS),'stage_index':0,'current_horizon':16,'next_horizon':16,
            'stage_updates':0,'gate_advances':0,'last_train_loss':None,'last_training_horizon':None,
            'consecutive_below_threshold':0,'goal_met':False,
            'last_gate_passed':False,'last_stage_advanced':False,'advance_history':[],
            'horizon_update_counts':{str(horizon):0 for horizon in HORIZONS},
            'main_transitions':0,'main_examples':0,'baseline_complete':False,
            'coordinate_updates_at_start':q['coordinate_curriculum']['updates'],
            'coordinate_horizon_counts_at_start':deepcopy(q['coordinate_curriculum']['horizon_update_counts']),
            'replay_updates_at_start':q['replay_curriculum']['updates'],
            'replay_forced_long_updates_at_start':q['replay_curriculum']['forced_long_replay_updates'],
            'replay_long450_updates_at_start':q['replay_curriculum']['long450_updates'],
            'replay_short_updates_at_start':q['replay_curriculum']['short_updates']}
        q['loss_gated_curriculum']=gate
        q['coordinate_curriculum']['next_horizon']=16
        state.setdefault('curriculum_history',[]).append({'last_step':step,
            'reason':'user_requested_pureXY_horizon_gate_on_full_optimizer_train_loss',
            'new_curriculum_protocol':LOSS_GATE_PROTOCOL,'forced_long_replay_removed':True,
            'old_global_coordinate_replay_clocks_preserved':True,'optimizer_reset':False})
        state.update(loss=None,latest_metrics=None,loss_groups=None)
    else:
        count=step-plan['start_optimizer_step']
        if (gate.get('sha256')!=sha256 or gate.get('start_optimizer_step')!=plan['start_optimizer_step']
                or gate.get('last_optimizer_step')!=step or gate.get('updates')!=count or count<0
                or gate.get('initial_loaded_checkpoint_sha256')!=source_checkpoint_sha256
                or gate.get('settings')!=SETTINGS or gate.get('gate_advances')!=gate.get('stage_index')
                or len(gate.get('advance_history',[]))!=gate.get('stage_index')
                or q['coordinate_curriculum']['updates']-gate.get('coordinate_updates_at_start',-1)!=count
                or q['replay_curriculum']['updates']-gate.get('replay_updates_at_start',-1)!=count
                or q['replay_curriculum']['forced_long_replay_updates']!=gate.get('replay_forced_long_updates_at_start')
                or set(gate.get('horizon_update_counts',{}))!={str(h)for h in HORIZONS}
                or any(type(n)is not int or n<0 for n in gate['horizon_update_counts'].values())
                or sum(gate['horizon_update_counts'].values())!=count
                or gate.get('main_examples')!=count*256
                or gate.get('main_transitions')!=sum(int(h)*n for h,n in gate['horizon_update_counts'].items())*256):
            raise ValueError('Saved loss-gated decisions, phase clock or actual horizon counts changed')
        loss_gated_training_horizon(state)
        if gate.get('next_horizon')!=gate['current_horizon']:
            raise ValueError('Loss-gated preview and next optimizer horizon disagree')
        history=gate['advance_history']
        for index,entry in enumerate(history):
            if (entry.get('from_horizon')!=HORIZONS[index] or entry.get('to_horizon')!=HORIZONS[index+1]
                    or type(entry.get('step'))is not int or not gate['start_optimizer_step']<entry['step']<=step
                    or index and entry['step']<=history[index-1]['step']
                    or type(entry.get('train_loss'))not in(int,float)or not math.isfinite(entry['train_loss'])
                    or not 0<=entry['train_loss']<SETTINGS['threshold']
                    or entry.get('qualifying_streak')!=SETTINGS['required_consecutive_passes']
                    or type(entry.get('completed_updates_at_previous_stage'))is not int
                    or entry['completed_updates_at_previous_stage']<SETTINGS['required_consecutive_passes']
                    or gate['horizon_update_counts'][str(HORIZONS[index])]!=entry['completed_updates_at_previous_stage']):
                raise ValueError('Saved horizon advance lacks its strict finite completed-update decision')
        durations=sum(entry['completed_updates_at_previous_stage']for entry in history)
        if (type(gate.get('stage_updates'))is not int or gate['stage_updates']!=count-durations
                or gate['horizon_update_counts'][str(gate['current_horizon'])]!=gate['stage_updates']
                or any(gate['horizon_update_counts'][str(h)]for h in HORIZONS[gate['stage_index']+1:])
                or q['replay_curriculum']['long450_updates']-gate['replay_long450_updates_at_start']!=gate['horizon_update_counts']['450']
                or q['replay_curriculum']['short_updates']-gate['replay_short_updates_at_start']!=count-gate['horizon_update_counts']['450']):
            raise ValueError('Saved horizon duration or actual short/450 accounting changed')
        streak=gate.get('consecutive_below_threshold')
        maximum=SETTINGS['required_consecutive_passes'] if gate['stage_index']==len(HORIZONS)-1 else SETTINGS['required_consecutive_passes']-1
        if (type(streak)is not int or not 0<=streak<=maximum or streak>gate['stage_updates']
                or gate.get('goal_met') is not (gate['current_horizon']==450 and streak>=SETTINGS['required_consecutive_passes'])
                or count and (type(gate.get('last_train_loss'))not in(int,float)
                    or not math.isfinite(gate['last_train_loss'])
                    or gate.get('last_gate_passed')is not (gate['last_train_loss']<SETTINGS['threshold'])
                    or (not gate['last_gate_passed'] and streak!=0)
                    or (gate['last_gate_passed'] and not gate['last_stage_advanced'] and streak==0))):
            raise ValueError('Saved strict loss-gate consecutive streak or goal flag changed')
    state['horizon_curriculum_protocol']=LOSS_GATE_PROTOCOL
    state['training_protocol']['loss_gated_curriculum']={'plan_sha256':sha256,**deepcopy(SETTINGS)}
    state['training_protocol']['forced_long_replay_active']=False
    state['training_protocol']['coordinate_horizon_curriculum']['active']=False
    state['training_protocol']['coordinate_horizon_curriculum']['history_only']=True
    return gate


def commit_loss_gated_update(state, *, step, horizon, train_loss, microbatches=4,
                              microbatch_size=64, effective_batch_size=256):
    q,actual=_boundary(state);gate=q['loss_gated_curriculum'];before=gate['stage_index']
    if (type(step)is not int or step!=actual or step!=gate['last_optimizer_step']+1
            or step-gate['start_optimizer_step']!=gate['updates']+1
            or type(horizon)is not int or horizon!=HORIZONS[before]
            or any(type(value)is not int for value in(microbatches,microbatch_size,effective_batch_size))
            or (microbatches,microbatch_size,effective_batch_size)!=(4,64,256)
            or type(train_loss)not in(int,float)or not math.isfinite(train_loss)or train_loss<0
            or q['coordinate_curriculum']['updates']-gate['coordinate_updates_at_start']!=gate['updates']+1
            or q['replay_curriculum']['updates']-gate['replay_updates_at_start']!=gate['updates']+1):
        raise ValueError('Loss gate decides once after a complete finite64x4 pure-XY optimizer commit')
    passed=train_loss<SETTINGS['threshold']
    streak=min(SETTINGS['required_consecutive_passes'],gate['consecutive_below_threshold']+1) if passed else 0
    advanced=streak>=SETTINGS['required_consecutive_passes'] and before<len(HORIZONS)-1
    gate['updates']+=1;gate['last_optimizer_step']=step
    gate['horizon_update_counts'][str(horizon)]+=1
    gate['main_transitions']+=horizon*256;gate['main_examples']+=256
    gate['stage_updates']+=1
    if advanced:
        gate['stage_index']+=1;gate['gate_advances']+=1
        gate['advance_history'].append({'step':step,'train_loss':float(train_loss),
            'from_horizon':horizon,'to_horizon':HORIZONS[gate['stage_index']],
            'qualifying_streak':streak,
            'completed_updates_at_previous_stage':gate['stage_updates']})
        gate['stage_updates']=0
        streak=0
    gate.update(current_horizon=HORIZONS[gate['stage_index']],next_horizon=HORIZONS[gate['stage_index']],
        last_train_loss=float(train_loss),last_training_horizon=horizon,
        consecutive_below_threshold=streak,
        goal_met=HORIZONS[gate['stage_index']]==450 and streak>=SETTINGS['required_consecutive_passes'],
        last_gate_passed=passed,last_stage_advanced=advanced)
    q['coordinate_curriculum']['next_horizon']=gate['current_horizon']
    return gate
