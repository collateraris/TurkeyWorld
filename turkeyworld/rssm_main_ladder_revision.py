"""Explicit MAIN18/full-BPTT curriculum over immutable H450 parent history.

Only completed optimizer TOTAL measurements advance this new independent clock.
Old player/motion/batch plans remain the source contracts, never rewritten.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

PROTOCOL='rssm_main18_full_bptt_loss_ladder_v1'
HORIZONS=(*range(16,449,32),450)


def digest(value):
    return hashlib.sha256(json.dumps(_canonical_json(value),sort_keys=True,allow_nan=False).encode()).hexdigest()


def _canonical_json(value):
    """Plan bytes use JSON keys; checkpoint snapshots retain their Python types."""
    return json.loads(json.dumps(value,allow_nan=False))


def file_sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def settings():
    return dict(version=1,horizons=list(HORIZONS),threshold=1.,required_consecutive_passes=10,
        comparison='strict_less_than',metric='completed_optimizer_update.raw_TOTAL',stage_max_updates=None,
        advance_condition='ten_consecutive_completed_raw_TOTAL_strictly_below_one',
        full_BPTT_current_MAIN_horizon=True,tbptt='current_MAIN_horizon',
        main_batch=dict(microbatch_size=18,gradient_accumulation_steps=1,effective_batch_size=18,
            combat=6,enemy_motion=6,navigation=6),auxiliary_actors=16,late_actors=2,
        contact_actors=4,enemy_local_actors=4,local_horizon_cap='min_selected_and_MAIN',
        source_support_is_gate_requirement=False,EMA_is_gate_measurement=False,
        historical_losses_reinterpreted=False,weights_reset=False,optimizer_reset=False,
        RNG_reset=False,samplers_reset=False,renderer_frozen=True,validation_every=250,preview_every=10)


def batch_settings():
    from .rssm_moe_batch_revision import MICRO16_SINGLE_SETTINGS
    value=deepcopy(MICRO16_SINGLE_SETTINGS)
    value.update(microbatch_size=18,effective_batch_size=18,
        micro_layouts=[dict(combat=6,enemy_motion=6,navigation=6)],
        combat_main_actors_per_update=6,enemy_motion_main_actors_per_update=6,
        navigation_main_actors_per_update=6,combat_rng_draws_per_microbatch=18,
        main_episode_rng_draws=18,discarded_combat_rng_draws_per_update=12,
        player_router_micro_weight='actor_count_divided_by_18',
        navigation_fraction=1/3,enemy_motion_fraction=1/3,
        historical_navigation_actor_counts_per_update=[40,10,5,4],
        historical_enemy_motion_actor_counts_per_update=[64,16,8],
        historical_effective_batch_sizes=[256,64,32,16])
    return value


def record_for_gate(gate):return gate.get('main_ladder_revision')


def main_ladder_plan(saved,*,source_checkpoint,source_checkpoint_sha256):
    from .rssm_moe_protocol import _validate_gate
    from .rssm_motion_correction_revision import PROTOCOL as MOTION,_validate_record
    from .rssm_player_navigation_revision import FULL_HORIZON_PROTOCOL
    from .rssm_moe_batch_revision import moe_batch_settings
    state=saved.get('run_state',saved);q=state['rssm_training'];gate=q['moe_training'];step=state['global_training_step']
    _validate_gate(gate,step)
    motion=q.get('motion_correction_revision',{});player=q.get('player_navigation_revision',{})
    _validate_record(motion,step)
    if(record_for_gate(gate)is not None or gate['current_horizon']!=450
            or motion.get('protocol')!=MOTION or not motion['baseline_complete']
            or player.get('protocol')!=FULL_HORIZON_PROTOCOL or not player['baseline_complete']
            or moe_batch_settings(state)['micro_layouts']!=[dict(combat=4,enemy_motion=8,navigation=4)]
            or file_sha(source_checkpoint)!=source_checkpoint_sha256):
        raise ValueError('MAIN ladder needs the exact complete H450/B16 motion-repair source')
    return _canonical_json(dict(kind=PROTOCOL,version=1,settings=settings(),batch_settings=batch_settings(),
        source_checkpoint=str(Path(source_checkpoint).resolve()),source_checkpoint_sha256=source_checkpoint_sha256,
        source_optimizer_step=step,source_gate_sha256=digest(gate),source_rssm_training=deepcopy(q),
        source_training_protocol=deepcopy(state.get('training_protocol',{})),
        parent_player_plan_sha256=player['sha256'],parent_motion_plan_sha256=motion['sha256'],
        old_plans_are_immutable_history=True,all_weights_Adam_RNG_and_samplers_preserved=True,
        fresh_stage_counter=True,user_requested_MAIN18_equal_six=True))


def legacy_binding_state(state,plan):
    """Validate old plans against their pinned source contracts on every resume."""
    if record_for_gate(state['rssm_training']['moe_training'])is None:return state
    result=deepcopy(state)
    result['global_training_step']=plan['source_optimizer_step']
    # Source samplers contain integer anchor keys. Read them from the exact
    # pickle checkpoint snapshot, never from their JSON plan representation.
    result['rssm_training']=deepcopy(record_for_gate(state['rssm_training']['moe_training'])['source_rssm_training'])
    result['training_protocol']=deepcopy(plan['source_training_protocol'])
    return result


def _validate_plan(plan):
    from .rssm_moe_protocol import _validate_gate
    q=plan.get('source_rssm_training',{});source=q.get('moe_training',{})
    if(plan.get('kind')!=PROTOCOL or plan.get('version')!=1 or plan.get('settings')!=settings()
            or plan.get('batch_settings')!=batch_settings()
            or plan.get('source_gate_sha256')!=digest(source)
            or plan.get('parent_player_plan_sha256')!=q.get('player_navigation_revision',{}).get('sha256')
            or plan.get('parent_motion_plan_sha256')!=q.get('motion_correction_revision',{}).get('sha256')
            or any(plan.get(k)is not True for k in ('old_plans_are_immutable_history',
                'all_weights_Adam_RNG_and_samplers_preserved','fresh_stage_counter','user_requested_MAIN18_equal_six'))):
        raise ValueError('MAIN ladder immutable source, batch or full-BPTT criterion changed')
    _validate_gate(source,plan['source_optimizer_step'])


def bind_main_ladder_revision(state,plan,*,path,sha256,loaded_checkpoint_sha256):
    from .rssm_h450_goal import initial_h450_goal
    raw=Path(path).read_bytes();_validate_plan(plan)
    if(hashlib.sha256(raw).hexdigest()!=sha256 or json.loads(raw.decode('utf-8'))!=plan
            or file_sha(plan['source_checkpoint'])!=plan['source_checkpoint_sha256']):
        raise ValueError('MAIN ladder plan bytes or pinned full source changed')
    q=state['rssm_training'];gate=q['moe_training'];record=record_for_gate(gate);step=state['global_training_step']
    if record is None:
        if(step!=plan['source_optimizer_step']or loaded_checkpoint_sha256!=plan['source_checkpoint_sha256']
                or digest(gate)!=plan['source_gate_sha256']or digest(q)!=digest(plan['source_rssm_training'])):
            raise ValueError('MAIN ladder activation requires exact complete source and sampler clocks')
        record=dict(protocol=PROTOCOL,version=1,plan_path=str(Path(path).resolve()),sha256=sha256,
            source_optimizer_step=step,last_optimizer_step=step,updates=0,
            source_checkpoint_sha256=plan['source_checkpoint_sha256'],source_gate_sha256=plan['source_gate_sha256'],
            source_gate=deepcopy(gate),source_rssm_training=deepcopy(q),settings=settings(),
            stage_index=0,stage_updates=0,stage_streak=0,stage_history=[],
            qualifying_optimizer_steps=[],qualifying_totals=[],
            horizon_update_counts={str(h):0 for h in HORIZONS},main_examples=0,main_transitions=0,
            baseline_complete=False,baseline_step=None)
        gate['main_ladder_revision']=record
        gate.update(current_horizon=16,next_horizon=16,stage_index=0,stage_updates=0,
            consecutive_below_threshold=0,gate_advances=0,last_gate_passed=False,last_stage_advanced=False,
            goal_met=False,effective_stage_budget=None,mandatory_stage_budget=False,
            loss_streak_can_advance=True,budget_exhausted=False,curriculum_completed=False,
            completion_reason=None,last_advance_reason=None,
            final_horizon_goal=initial_h450_goal(source_optimizer_step=step,plan_sha256=sha256))
    elif(record.get('sha256')!=sha256 or record.get('source_gate_sha256')!=plan['source_gate_sha256']
            or record.get('source_checkpoint_sha256')!=plan['source_checkpoint_sha256']
            or digest(record.get('source_rssm_training'))!=digest(plan['source_rssm_training'])):
        raise ValueError('Saved MAIN ladder source lineage changed')
    validate_gate(gate,step)
    validate_continued_ledgers(state)
    publish_contract(state)
    return record


def publish_contract(state):
    gate=state['rssm_training']['moe_training'];r=record_for_gate(gate)
    if r is None:return
    state['horizon_curriculum_protocol']=PROTOCOL
    state['active_objective_formula_revision']=PROTOCOL
    state['h450_total_loss_goal']=deepcopy(gate['final_horizon_goal'])
    state.setdefault('training_protocol',{}).update(main_ladder_revision=dict(protocol=PROTOCOL,
        plan_sha256=r['sha256'],**settings()),tbptt=gate['current_horizon'],
        active_objective_formula_revision=PROTOCOL,microbatch_size=18,gradient_accumulation_steps=1,
        effective_batch_size=18,horizons=list(HORIZONS),threshold=1.,required_consecutive_passes=10)


def validate_gate(gate,step):
    from .rssm_moe_protocol import _validate_gate,SETTINGS
    from .rssm_h450_goal import validate_h450_goal
    r=record_for_gate(gate)
    if(not isinstance(r,dict)or r.get('protocol')!=PROTOCOL or r.get('settings')!=settings()
            or r.get('source_gate_sha256')!=digest(r.get('source_gate'))
            or r.get('source_rssm_training',{}).get('moe_training')!=r.get('source_gate')):
        raise ValueError('MAIN ladder source history or settings changed')
    source=r['source_gate'];start=r['source_optimizer_step'];n=r['updates'];index=r['stage_index']
    _validate_gate(source,start)
    if(type(n)is not int or n<0 or start+n!=step or r['last_optimizer_step']!=step
            or type(index)is not int or not 0<=index<len(HORIZONS)
            or gate.get('settings')!=SETTINGS or gate['updates']!=source['updates']+n
            or gate['last_optimizer_step']!=step or gate['start_optimizer_step']!=source['start_optimizer_step']
            or gate['sha256']!=source['sha256']or gate['batch_revision']!=source['batch_revision']
            or gate['advance_policy']!=source['advance_policy']):
        raise ValueError('MAIN ladder complete optimizer clock or immutable parent records changed')
    counts=r['horizon_update_counts'];history=r['stage_history'];streak=r['stage_streak']
    if(set(counts)!={str(h)for h in HORIZONS}or any(type(c)is not int or c<0 for c in counts.values())
            or sum(counts.values())!=n or len(history)!=index or any(counts[str(h)]for h in HORIZONS[index+1:])
            or r['stage_updates']!=counts[str(HORIZONS[index])]
            or type(streak)is not int or not 0<=streak<=(10 if index==len(HORIZONS)-1 else 9)
            or streak>r['stage_updates']):
        raise ValueError('MAIN ladder stage counters or fresh qualifying streak changed')
    steps=r.get('qualifying_optimizer_steps');losses=r.get('qualifying_totals')
    if(not isinstance(steps,list)or not isinstance(losses,list)or len(steps)!=streak or len(losses)!=streak
            or any(type(s)is not int or not start<s<=step for s in steps)
            or any(b!=a+1 for a,b in zip(steps,steps[1:]))
            or any(type(v)not in(int,float)or not math.isfinite(v)or not 0<=v<1 for v in losses)
            or streak and steps[-1]!=step):
        raise ValueError('MAIN ladder lacks consecutive raw TOTAL evidence')
    prior=start
    for i,e in enumerate(history):
        if(e.get('from_horizon')!=HORIZONS[i]or e.get('to_horizon')!=HORIZONS[i+1]
                or e.get('qualifying_streak')!=10 or e.get('advance_reason')!='loss_streak'
                or not prior<e.get('step',-1)<=step or e.get('completed_updates_at_previous_stage')!=counts[str(HORIZONS[i])]
                or e['completed_updates_at_previous_stage']<10 or not 0<=e.get('train_loss',2)<1
                or e.get('qualifying_optimizer_steps')!=list(range(e['step']-9,e['step']+1))
                or len(e.get('qualifying_totals',[]))!=10
                or any(type(v)not in(int,float)or not math.isfinite(v)or not 0<=v<1 for v in e['qualifying_totals'])):
            raise ValueError('MAIN ladder advance lacks ten completed strict TOTAL measurements')
        prior=e['step']
    examples=18*n;transitions=18*sum(int(h)*c for h,c in counts.items())
    expected_counts={h:source['horizon_update_counts'][h]+counts.get(h,0)for h in source['horizon_update_counts']}
    if(r['main_examples']!=examples or r['main_transitions']!=transitions
            or gate['main_examples']!=source['main_examples']+examples
            or gate['main_transitions']!=source['main_transitions']+transitions
            or gate['horizon_update_counts']!=expected_counts
            or gate['advance_history']!=source['advance_history']+history
            or gate['current_horizon']!=HORIZONS[index]or gate['next_horizon']!=HORIZONS[index]
            or gate['stage_index']!=index or gate['stage_updates']!=r['stage_updates']
            or gate['consecutive_below_threshold']!=streak or gate['gate_advances']!=index):
        raise ValueError('MAIN ladder actual horizon/MAIN18 exposure or historical path changed')
    goal=validate_h450_goal(gate['final_horizon_goal'],current_step=step)
    done=index==len(HORIZONS)-1 and streak==10
    if(goal['plan_sha256']!=r['sha256']or goal['source_optimizer_step']!=start
            or goal['h450_updates']!=counts['450']or goal['goal_met']is not done
            or gate['goal_met']is not done or gate['curriculum_completed']is not done
            or gate['budget_exhausted']is not False or gate['completion_reason']!=('loss_streak'if done else None)):
        raise ValueError('MAIN ladder final H450 goal or no-cap completion changed')
    return r


def mark_main_ladder_baseline(state,*,step):
    r=validate_gate(state['rssm_training']['moe_training'],state['global_training_step'])
    if r['updates']or step!=r['source_optimizer_step']:raise ValueError('MAIN ladder baseline must precede SGD')
    r.update(baseline_complete=True,baseline_step=step)


def validate_continued_ledgers(state):
    """The old component clocks/PCGs continue; only their active MAIN contract changes."""
    from .rssm_player_navigation_revision import _validate_record as player_record
    from .rssm_motion_correction_revision import _validate_record as motion_record
    q=state['rssm_training'];gate=q['moe_training'];r=validate_gate(gate,state['global_training_step'])
    old=r['source_rssm_training'];n=r['updates']
    for key,validator in(('player_navigation_revision',player_record),('motion_correction_revision',motion_record)):
        actual=q[key];source=old[key];validator(actual,state['global_training_step'])
        if(actual['updates']!=source['updates']+n or actual['sha256']!=source['sha256']
                or actual['settings']!=source['settings']):
            raise ValueError('MAIN ladder lost its unchanged parent player/motion clocks')
    if q['player_navigation_revision']['contact_draws']!=old['player_navigation_revision']['contact_draws']+4*n:
        raise ValueError('MAIN ladder contact sampler reset or exposure changed')
    nav=q['navigation_replay_revision'];source=old['navigation_replay_revision']
    expected=replay_totals(nav,state['global_training_step'],gate)
    if(any(nav.get(key)!=value for key,value in expected.items())
            or nav.get('sampler_state',{}).get('draws')!=expected['navigation_main_actors']
            or nav.get('enemy_sampler_state',{}).get('draws')!=expected['enemy_motion_main_actors']):
        raise ValueError('MAIN18 navigation/enemy sampler piecewise exposure changed')
    return r


def commit_main_ladder_update(state,*,step,horizon,train_loss,microbatches,microbatch_size,
                              effective_batch_size,loss_gate_eligible=True):
    from .rssm_h450_goal import commit_h450_goal
    gate=state['rssm_training']['moe_training'];r=validate_gate(gate,step-1)
    if(not r['baseline_complete']or state['global_training_step']!=step or gate['goal_met']
            or horizon!=gate['current_horizon']or any(type(v)is not int for v in(microbatches,microbatch_size,effective_batch_size))
            or (microbatches,microbatch_size,effective_batch_size)!=(1,18,18)
            or type(train_loss)not in(int,float)or not math.isfinite(train_loss)or train_loss<0):
        raise ValueError('MAIN ladder requires one finite complete MAIN18 raw TOTAL optimizer commit')
    old=r['stage_index'];passed=train_loss<1.;streak=r['stage_streak']+1 if passed else 0
    if passed:
        r['qualifying_optimizer_steps'].append(step);r['qualifying_totals'].append(float(train_loss))
    else:r.update(qualifying_optimizer_steps=[],qualifying_totals=[])
    r.update(updates=r['updates']+1,last_optimizer_step=step,stage_updates=r['stage_updates']+1,
        main_examples=r['main_examples']+18,main_transitions=r['main_transitions']+18*horizon)
    r['horizon_update_counts'][str(horizon)]+=1
    advanced=streak==10 and old<len(HORIZONS)-1
    if advanced:
        r['stage_history'].append(dict(step=step,from_horizon=horizon,to_horizon=HORIZONS[old+1],
            train_loss=float(train_loss),qualifying_streak=10,advance_reason='loss_streak',
            completed_updates_at_previous_stage=r['stage_updates'],
            qualifying_optimizer_steps=deepcopy(r['qualifying_optimizer_steps']),
            qualifying_totals=deepcopy(r['qualifying_totals'])))
        r.update(stage_index=old+1,stage_updates=0);streak=0
        r.update(qualifying_optimizer_steps=[],qualifying_totals=[])
    r['stage_streak']=streak
    source=r['source_gate'];done=r['stage_index']==len(HORIZONS)-1 and streak==10
    gate.update(updates=source['updates']+r['updates'],last_optimizer_step=step,
        stage_index=r['stage_index'],stage_updates=r['stage_updates'],gate_advances=r['stage_index'],
        current_horizon=HORIZONS[r['stage_index']],next_horizon=HORIZONS[r['stage_index']],
        consecutive_below_threshold=streak,last_train_loss=float(train_loss),last_training_horizon=horizon,
        last_gate_passed=passed,last_stage_advanced=advanced,last_advance_reason='loss_streak'if advanced else None,
        last_loss_gate_eligible=True,last_source_support_gate_eligible=loss_gate_eligible,
        main_examples=source['main_examples']+r['main_examples'],main_transitions=source['main_transitions']+r['main_transitions'],
        advance_history=deepcopy(source['advance_history'])+deepcopy(r['stage_history']),
        goal_met=done,curriculum_completed=done,completion_reason='loss_streak'if done else None)
    gate['horizon_update_counts'][str(horizon)]+=1
    gate['final_horizon_goal']=commit_h450_goal(gate['final_horizon_goal'],step=step,horizon=horizon,total_loss=train_loss)
    validate_gate(gate,step);publish_contract(state)
    return gate


def runtime_player_settings(parent,*,horizon,plan_sha256):
    from .rssm_player_navigation_revision import player_navigation_settings
    expected=player_navigation_settings(tbptt=450,microbatch_size=16)
    if(parent!=expected or horizon not in HORIZONS or not isinstance(plan_sha256,str)or len(plan_sha256)!=64):
        raise ValueError('MAIN ladder runtime requires pinned parent objective and current full-BPTT horizon')
    value=deepcopy(parent);value.update(tbptt=horizon,player_weights=dict(combat=1/3,navigation=1/3,enemy_motion=1/3),
        main_batch_settings=dict(microbatch_size=18,gradient_accumulation_steps=1,effective_batch_size=18,
            combat=6,navigation=6,enemy_motion=6,combat_auxiliary=16,combat_late=2),
        main_ladder_override=dict(protocol=PROTOCOL,plan_sha256=plan_sha256,horizon=horizon))
    return value


def validate_runtime_player_settings(value):
    from .rssm_player_navigation_revision import player_navigation_settings
    override=value.get('main_ladder_override',{})
    return value==runtime_player_settings(player_navigation_settings(tbptt=450,microbatch_size=16),
        horizon=override.get('horizon'),plan_sha256=override.get('plan_sha256'))


def replay_totals(record,step,gate):
    r=record_for_gate(gate);source=r['source_rssm_training']['navigation_replay_revision'];n=step-r['source_optimizer_step']
    if n<0 or n!=r['updates']:raise ValueError('MAIN18 replay clock diverged')
    result=dict(updates=source['updates']+n)
    for key,per in(('navigation_main_actors',6),('enemy_motion_main_actors',6),('combat_main_actors',6),('discarded_combat_rng_draws',12)):
        result[key]=source[key]+n*per
    return result
