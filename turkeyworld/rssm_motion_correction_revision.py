"""Pinned local-motion supervision, retaining the full450 training and goal."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

PROTOCOL='rssm_player_enemy_local_motion_repair_v3'
HORIZONS=(4,8,16,32)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def file_sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def motion_correction_settings():
    from .rssm_player_contact_replay import PLAYER_LOCAL_MOTION_PROTOCOL
    from .rssm_npc_geometry_objectives import NPC_NAVIGATION_PROTOCOL
    from .rssm_enemy_local_recovery import ENEMY_LOCAL_RECOVERY_PROTOCOL
    return dict(version=1,horizon=450,tbptt=450,
        main_batch=dict(microbatch_size=16,gradient_accumulation_steps=1,effective_batch_size=16,
            combat=4,enemy_motion=8,navigation=4,combat_auxiliary=16,combat_late=2),
        player_local=deepcopy(PLAYER_LOCAL_MOTION_PROTOCOL),npc_navigation=deepcopy(NPC_NAVIGATION_PROTOCOL),
        enemy_local=deepcopy(ENEMY_LOCAL_RECOVERY_PROTOCOL),
        local_horizons=list(HORIZONS),local_horizon_schedule='rotating_latin_square_16_updates',
        player_horizon_offset=0,enemy_horizon_offset=2,player_contact_actors=4,
        Source_future_feedback=False,renderer_frozen=True,optimizer_reset=False,
        architecture_change=False,old_sampler_reset=False,goal_changed=False,
        H450_threshold=1.,H450_required_consecutive=10,H450_max_updates=None,
        validation_every=250,preview_every=10,fresh_physical_baseline_before_SGD=True)


def local_motion_horizons(updates):
    if type(updates)is not int or updates<0:raise ValueError('Local schedule requires exact completed update count')
    index=(updates%4+(updates//4)%4)%4
    return HORIZONS[index],HORIZONS[(index+2)%4]


def _check_source_state(state):
    from .rssm_moe_protocol import moe_training_horizon
    from .rssm_moe_batch_revision import moe_batch_settings
    from .rssm_h450_goal import validate_h450_goal
    from .rssm_player_navigation_revision import FULL_HORIZON_PROTOCOL
    step=state['global_training_step'];q=state['rssm_training'];gate=q['moe_training']
    batch=moe_batch_settings(state);player=q.get('player_navigation_revision',{})
    goal=validate_h450_goal(gate['final_horizon_goal'],current_step=step)
    if(moe_training_horizon(state)!=450 or player.get('protocol')!=FULL_HORIZON_PROTOCOL
            or player.get('baseline_complete')is not True or player.get('settings',{}).get('tbptt')!=450
            or gate['last_optimizer_step']!=step
            or tuple(batch[k]for k in('microbatch_size','gradient_accumulation_steps','effective_batch_size'))!=(16,1,16)
            or batch['micro_layouts']!=[dict(combat=4,enemy_motion=8,navigation=4)]
            or goal['settings']['threshold']!=1. or goal['settings']['required_consecutive_passes']!=10
            or goal['settings']['final_stage_max_updates']is not None):
        raise ValueError('Motion repair requires unchanged450/B16/4combat8quiet4NAV and TOTAL<1x10')
    return step,q,gate,player,goal


def motion_correction_plan(saved,*,source_checkpoint,source_checkpoint_sha256):
    state=saved.get('run_state',saved);step,q,gate,player,goal=_check_source_state(state)
    source=Path(source_checkpoint).resolve()
    if(q.get('motion_correction_revision')is not None or not source.is_file()
            or file_sha(source)!=source_checkpoint_sha256 or goal['streak']!=0 or goal['goal_met']):
        raise ValueError('Motion repair activation needs an immutable unmodified streak0 source')
    return dict(kind=PROTOCOL,version=1,settings=motion_correction_settings(),
        source_checkpoint=str(source),source_checkpoint_sha256=source_checkpoint_sha256,
        source_optimizer_step=step,source_moe_updates=gate['updates'],
        source_player_navigation_updates=player['updates'],source_contact_draws=player['contact_draws'],
        source_gate_sha256=digest(gate),source_goal_sha256=digest(goal),
        source_player_revision_sha256=digest(player),source_navigation_replay_sha256=digest(q['navigation_replay_revision']),
        parent_player_plan_sha256=player['sha256'],batch_plan_sha256=gate['batch_revision']['sha256'],
        previous_TOTAL_not_comparable=True,all_model_Adam_RNG_and_existing_samplers_preserved=True)


def _validate_record(record,step):
    if(not isinstance(record,dict)or record.get('protocol')!=PROTOCOL
            or record.get('settings')!=motion_correction_settings()
            or type(record.get('updates'))is not int or record['updates']<0
            or record.get('last_optimizer_step')!=step
            or record['source_optimizer_step']+record['updates']!=step
            or not isinstance(record.get('baseline_complete'),bool)
            or record.get('baseline_step')!=(record['source_optimizer_step']if record['baseline_complete']else None)
            or record['updates']and not record['baseline_complete']):
        raise ValueError('Motion correction complete update or initial baseline ledger changed')
    return record


def bind_motion_correction_revision(state,plan,*,path,sha256,loaded_checkpoint_sha256):
    raw=Path(path).read_bytes();step,q,gate,player,goal=_check_source_state(state)
    if(hashlib.sha256(raw).hexdigest()!=sha256 or json.loads(raw.decode('utf-8'))!=plan
            or plan.get('kind')!=PROTOCOL or plan.get('version')!=1
            or plan.get('settings')!=motion_correction_settings()
            or file_sha(plan['source_checkpoint'])!=plan['source_checkpoint_sha256']
            or plan.get('parent_player_plan_sha256')!=player['sha256']
            or plan.get('batch_plan_sha256')!=gate['batch_revision']['sha256']
            or plan.get('previous_TOTAL_not_comparable')is not True
            or plan.get('all_model_Adam_RNG_and_existing_samplers_preserved')is not True):
        raise ValueError('Motion correction immutable plan or parent source changed')
    record=q.get('motion_correction_revision')
    if record is None:
        if(step!=plan['source_optimizer_step']or loaded_checkpoint_sha256!=plan['source_checkpoint_sha256']
                or digest(gate)!=plan['source_gate_sha256']or digest(goal)!=plan['source_goal_sha256']
                or digest(player)!=plan['source_player_revision_sha256']
                or digest(q['navigation_replay_revision'])!=plan['source_navigation_replay_sha256']
                or goal['streak']!=0 or goal['goal_met']):
            raise ValueError('Motion correction requires the exact completed source and all old sampler clocks')
        record=dict(protocol=PROTOCOL,version=1,plan_path=str(Path(path).resolve()),sha256=sha256,
            source_optimizer_step=step,last_optimizer_step=step,updates=0,
            source_moe_updates=gate['updates'],source_player_navigation_updates=player['updates'],
            source_contact_draws=player['contact_draws'],source_checkpoint_sha256=plan['source_checkpoint_sha256'],
            source_gate_sha256=plan['source_gate_sha256'],source_goal_sha256=plan['source_goal_sha256'],
            settings=deepcopy(plan['settings']),baseline_complete=False,baseline_step=None,
            enemy_branch_transitions=0,enemy_valid_local_labels=0)
        q['motion_correction_revision']=record
        state.setdefault('objective_history',[]).append(dict(step=step,kind=PROTOCOL,
            source_loss=state.get('loss'),source_latest_metrics=deepcopy(state.get('latest_metrics')),
            source_loss_groups=deepcopy(state.get('loss_groups')),settings=deepcopy(plan['settings']),
            previous_TOTAL_comparable=False,reason='free_player_axes_own_NPC_velocity_and_NPC_geometry'))
    elif(record.get('sha256')!=sha256 or record.get('source_checkpoint_sha256')!=plan['source_checkpoint_sha256']
            or record.get('source_optimizer_step')!=plan['source_optimizer_step']
            or record.get('source_gate_sha256')!=plan['source_gate_sha256']
            or record.get('source_goal_sha256')!=plan['source_goal_sha256']):
        raise ValueError('Saved motion correction revision changed its source identity')
    _validate_record(record,step)
    if(gate['updates']!=record['source_moe_updates']+record['updates']
            or player['updates']!=record['source_player_navigation_updates']+record['updates']
            or player['contact_draws']!=record['source_contact_draws']+4*record['updates']):
        raise ValueError('Motion correction and existing main/contact ledgers diverged')
    state.setdefault('training_protocol',{})['motion_correction']=dict(protocol=PROTOCOL,
        plan_sha256=sha256,**deepcopy(plan['settings']))
    state['active_objective_formula_revision']=PROTOCOL
    return record


def mark_motion_correction_baseline(state,*,step):
    record=state['rssm_training']['motion_correction_revision'];_validate_record(record,state['global_training_step'])
    if record['updates']or step!=record['source_optimizer_step']or step!=state['global_training_step']:
        raise ValueError('Initial motion baseline must precede new SGD')
    record['baseline_complete']=True;record['baseline_step']=step;return record


def commit_motion_correction_revision(state,*,step,metrics):
    record=state['rssm_training']['motion_correction_revision'];_validate_record(record,step-1)
    q=state['rssm_training'];player=q['player_navigation_revision'];gate=q['moe_training']
    loss=metrics.get('enemy_local_loss')
    if(not record['baseline_complete']or state['global_training_step']!=step
            or gate['last_optimizer_step']!=step or player['last_optimizer_step']!=step
            or gate['updates']!=record['source_moe_updates']+record['updates']+1
            or player['contact_draws']!=record['source_contact_draws']+4*(record['updates']+1)
            or metrics.get('objective_formula_revision')!=PROTOCOL
            or metrics.get('motion_correction_loss_revision')!=PROTOCOL
            or type(loss)not in(int,float)or not math.isfinite(loss)or loss<0):
        raise ValueError('Motion repair counts only full real optimizer commits with its own aligned loss')
    record['updates']+=1;record['last_optimizer_step']=step
    for target,key in(('enemy_branch_transitions','enemy_local_forward_transitions'),
                       ('enemy_valid_local_labels','enemy_local_valid_source_labels')):
        count=metrics.get(key,0)
        if type(count)is not int or count<0:raise ValueError('Invalid actual NPC local exposure')
        record[target]+=count
    return _validate_record(record,step)
