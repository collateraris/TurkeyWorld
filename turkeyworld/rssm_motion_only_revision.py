"""Independent motion-only rounds; old combat/MAIN18 ledgers are opaque history.

This module changes metadata and exposure clocks only. Model tensors, named
Adam groups, global RNG and private sampler RNGs are supplied unchanged by the
standalone runner. No legacy training gate is advanced by this revision.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import re

from .rssm_main_ladder_revision import digest,file_sha

PROTOCOL='rssm_motion_only_main16_v1'
HORIZONS=(16,48,80,112,144,180)
STAGE_UPDATES=30
ROUND_UPDATES=len(HORIZONS)*STAGE_UPDATES
CORPUS_PROTOCOL='rssm_motion_only_Source_cycle_corpus_v1'
HISTORY_KEYS=('moe_training','navigation_replay_revision','player_navigation_revision',
    'motion_correction_revision','combat_rssm_revision','respawn_timing_revision')


def _json(value):return json.loads(json.dumps(value,allow_nan=False))
def _sha(value):return isinstance(value,str)and re.fullmatch('[0-9a-f]{64}',value)is not None


def motion_only_settings():
    return dict(version=1,horizons=list(HORIZONS),updates_per_stage=STAGE_UPDATES,
        world_updates_per_round=ROUND_UPDATES,advance_condition='exact_completed_optimizer_updates',
        loss_is_advance_condition=False,loss_threshold=None,required_loss_streak=None,
        final_stage_result='world_round_complete_not_quality_pass',next_round_initial_horizon=16,
        target_game_steps=180,simulation_hz=30,target_game_seconds=6.,
        full_BPTT_current_MAIN_horizon=True,tbptt='current_MAIN_horizon',context=1,
        MAIN=dict(microbatch_size=16,gradient_accumulation_steps=1,effective_batch_size=16,
            navigation=8,enemy_motion=8,combat=0),
        player_weights=dict(navigation=.5,enemy_motion=.5),enemy_weights=dict(enemy_motion=1.),
        combat_loss_enabled=False,combat_examples=0,combat_auxiliary_actors=0,combat_late_actors=0,
        combat_episode_RNG_draws=0,combat_sampler_RNG_advanced=False,
        combat_router_tasks_enabled=False,coupled_combat_memory_motion_gradients=True,
        independent_contact_actors=0,independent_contact_RNG_draws=0,
        local_branches='reuse_actual_MAIN_prefixes_only',max_player_local_branches=4,max_enemy_local_branches=4,
        source_scouts=0,local_horizon_cap='min_selected_and_MAIN',
        renderer_frozen=True,architecture_change=False,weights_reset=False,optimizer_reset=False,
        global_RNG_reset=False,private_sampler_RNG_reset=False,parent_ledgers_history_only=True,
        source_future_feedback=False,validation_every=250,target_validation_horizon=180,preview_every=10,
        historical_450_validation_is_quality_gate=False,
        sampling_start='state0_action0',sampling_context=1,sampling_episode_RNG_only=True,
        replay_with_latest_corpus=dict(original=4,latest_policy_Source=4),
        replay_without_latest_corpus=dict(original=8),
        cycle_order=['policy_controls_real_Source','append_immutable_TRAIN_corpus',
            'world_motion_round_180_updates','policy_rebinds_new_world_and_trains'],
        corpus_append_protocol=CORPUS_PROTOCOL,validation_corpus_frozen=True)


def batch_settings():
    return dict(microbatch_size=16,gradient_accumulation_steps=1,effective_batch_size=16,
        micro_layouts=[dict(combat=0,navigation=8,enemy_motion=8)],
        navigation_main_actors_per_update=8,enemy_motion_main_actors_per_update=8,
        combat_main_actors_per_update=0,combat_rng_draws_per_microbatch=0,
        combat_auxiliary_actors_per_microbatch=0,combat_late_actors_per_microbatch=0,
        discarded_combat_rng_draws_per_update=0,enemy_normalization_denominator_per_update=8,
        player_router_micro_weight='actor_count_divided_by_16',
        historical_navigation_actor_counts_per_update=[40,10,5,4,6],
        historical_enemy_motion_actor_counts_per_update=[64,16,8,6])


def motion_only_plan(saved,*,source_checkpoint,source_checkpoint_sha256,navdir,enemydir,corpus_identity=None):
    from .rssm_main_ladder_revision import validate_gate,validate_continued_ledgers
    state=saved.get('run_state',saved);q=state['rssm_training'];gate=q['moe_training'];step=state['global_training_step']
    validate_gate(gate,step);validate_continued_ledgers(state)
    if(q.get('motion_only_revision')is not None or gate['current_horizon']!=16
            or not _sha(source_checkpoint_sha256)or file_sha(source_checkpoint)!=source_checkpoint_sha256):
        raise ValueError('Motion-only activation needs the exact complete MAIN18/H16 source')
    source=Path(source_checkpoint).resolve();nav=Path(navdir).resolve();enemy=Path(enemydir).resolve()
    if not nav.is_dir()or not enemy.is_dir():raise ValueError('Motion-only replay roots must exist')
    return _json(dict(kind=PROTOCOL,version=1,settings=motion_only_settings(),batch_settings=batch_settings(),
        source_checkpoint=str(source),source_checkpoint_sha256=source_checkpoint_sha256,source_optimizer_step=step,
        source_rssm_training=deepcopy(q),source_rssm_training_sha256=digest(q),
        source_training_protocol=deepcopy(state.get('training_protocol',{})),
        navigation_directory=str(nav),enemy_motion_directory=str(enemy),
        initial_corpus_identity=deepcopy(corpus_identity),
        all_weights_named_Adam_RNG_and_lineage_preserved=True,new_formula_history_separate=True))


def _new_gate(source_step,round_index):
    return dict(round_index=round_index,source_optimizer_step=source_step,last_optimizer_step=source_step,
        updates=0,stage_index=0,current_horizon=HORIZONS[0],stage_updates=0,
        horizon_update_counts={str(h):0 for h in HORIZONS},advance_history=[],completed=False,
        completion_reason=None,quality_passed=False,last_loss=None)


def record(state):return state['rssm_training']['motion_only_revision']


def bind_motion_only(state,plan,*,path,loaded_sha,sha256=None):
    raw=Path(path).read_bytes();actual=file_sha(path);sha256=actual if sha256 is None else sha256
    if(actual!=sha256 or json.loads(raw.decode('utf-8'))!=plan or plan.get('kind')!=PROTOCOL
            or plan.get('settings')!=motion_only_settings()or plan.get('batch_settings')!=batch_settings()
            or digest(plan.get('source_rssm_training'))!=plan.get('source_rssm_training_sha256')
            or file_sha(plan['source_checkpoint'])!=plan['source_checkpoint_sha256']):
        raise ValueError('Motion-only immutable plan or exact source bytes changed')
    q=state['rssm_training'];r=q.get('motion_only_revision');step=state['global_training_step']
    if r is None:
        if(step!=plan['source_optimizer_step']or loaded_sha!=plan['source_checkpoint_sha256']
                or digest(q)!=plan['source_rssm_training_sha256']):
            raise ValueError('Motion-only activation must preserve the complete raw source ledger')
        old=deepcopy(q);replay=old['navigation_replay_revision']
        nav=deepcopy(replay['sampler_state']);enemy=deepcopy(replay['enemy_sampler_state'])
        r=dict(protocol=PROTOCOL,version=1,sha256=sha256,plan_path=str(Path(path).resolve()),
            source_optimizer_step=step,last_optimizer_step=step,updates=0,
            source_checkpoint_sha256=plan['source_checkpoint_sha256'],source_rssm_training=old,
            source_rssm_training_sha256=digest(old),settings=motion_only_settings(),
            nav_sampler_state=nav,enemy_sampler_state=enemy,initial_nav_draws=nav['draws'],initial_enemy_draws=enemy['draws'],
            gate=_new_gate(step,0),round_history=[],corpus_revisions=[],main_examples=0,main_transitions=0,
            reused_player_local_branches=0,reused_enemy_local_branches=0,local_Source_queries=0,
            independent_contact_draws=0,combat_draws=0,combat_examples=0,
            baseline_complete=False,baseline_step=None)
        q['motion_only_revision']=r
        state.setdefault('objective_history',[]).append(dict(step=step,kind=PROTOCOL,
            source_loss=state.get('loss'),source_latest_metrics=deepcopy(state.get('latest_metrics')),
            source_loss_groups=deepcopy(state.get('loss_groups')),previous_TOTAL_comparable=False))
        state.update(loss=None,latest_metrics=None,loss_groups=None)
        if plan.get('initial_corpus_identity')is not None:attach_corpus(state,plan['initial_corpus_identity'])
    elif(r.get('sha256')!=sha256 or r.get('source_checkpoint_sha256')!=plan['source_checkpoint_sha256']
            or digest(r.get('source_rssm_training'))!=plan['source_rssm_training_sha256']):
        raise ValueError('Saved motion-only source lineage changed')
    validate_motion_only(state);publish_contract(state)
    return r


bind_motion_only_revision=bind_motion_only


def _validate_gate(gate):
    index=gate.get('stage_index');updates=gate.get('updates');counts=gate.get('horizon_update_counts',{})
    if(type(index)is not int or not 0<=index<len(HORIZONS)or type(updates)is not int or not 0<=updates<=ROUND_UPDATES
            or gate.get('current_horizon')!=HORIZONS[index]or set(counts)!={str(h)for h in HORIZONS}
            or any(type(n)is not int or not 0<=n<=STAGE_UPDATES for n in counts.values())
            or sum(counts.values())!=updates or gate.get('stage_updates')!=counts[str(HORIZONS[index])]
            or any(counts[str(h)]!=STAGE_UPDATES for h in HORIZONS[:index])
            or any(counts[str(h)]!=0 for h in HORIZONS[index+1:])
            or type(gate.get('round_index'))is not int or gate['round_index']<0
            or gate.get('last_optimizer_step')!=gate['source_optimizer_step']+updates):
        raise ValueError('Motion-only exact30-stage or round optimizer accounting changed')
    done=updates==ROUND_UPDATES
    if(gate.get('completed')is not done or gate.get('quality_passed')is not False
            or gate.get('completion_reason')!=('world_round_complete'if done else None)
            or not done and gate['stage_updates']>=STAGE_UPDATES or len(gate.get('advance_history',[]))!=index):
        raise ValueError('Motion-only round completion must not be loss/quality gated')
    for i,row in enumerate(gate['advance_history']):
        if(row.get('from_horizon')!=HORIZONS[i]or row.get('to_horizon')!=HORIZONS[i+1]
                or row.get('stage_updates')!=STAGE_UPDATES or row.get('reason')!='fixed_stage_updates'
                or row.get('step')!=gate['source_optimizer_step']+STAGE_UPDATES*(i+1)):
            raise ValueError('Motion-only horizon advance changed its exact update boundary')


def validate_motion_only(state):
    r=record(state);step=state['global_training_step'];source=r.get('source_rssm_training',{})
    if(r.get('protocol')!=PROTOCOL or r.get('settings')!=motion_only_settings()
            or digest(source)!=r.get('source_rssm_training_sha256')or type(r.get('updates'))is not int or r['updates']<0
            or r['source_optimizer_step']+r['updates']!=step or r.get('last_optimizer_step')!=step):
        raise ValueError('Motion-only revision source, settings or complete clock changed')
    for key in HISTORY_KEYS:
        if key in source and key in state['rssm_training']and digest(source[key])!=digest(state['rssm_training'][key]):
            raise ValueError('Motion-only training rewrote an opaque parent training ledger')
    rounds=[*r.get('round_history',[]),r['gate']]
    for i,gate in enumerate(rounds):
        _validate_gate(gate)
        if(gate['round_index']!=i or gate['source_optimizer_step']!=r['source_optimizer_step']+ROUND_UPDATES*i
                or i<len(rounds)-1 and not gate['completed']):raise ValueError('Motion-only serial round history changed')
    if(sum(g['updates']for g in rounds)!=r['updates']or r['gate']['last_optimizer_step']!=step
            or r['main_examples']!=16*r['updates']
            or r['main_transitions']!=16*sum(int(h)*n for g in rounds for h,n in g['horizon_update_counts'].items())
            or r['nav_sampler_state']['draws']!=r['initial_nav_draws']+8*r['updates']
            or r['enemy_sampler_state']['draws']!=r['initial_enemy_draws']+8*r['updates']
            or r['independent_contact_draws']!=0 or r['combat_draws']!=0 or r['combat_examples']!=0):
        raise ValueError('Motion-only MAIN16/8+8 or frozen combat/contact exposure changed')
    return r


def current_horizon(state):return validate_motion_only(state)['gate']['current_horizon']


def mark_motion_only_baseline(state,*,step):
    r=validate_motion_only(state)
    if r['updates']or step!=r['source_optimizer_step']:raise ValueError('Motion-only baseline must precede new SGD')
    r.update(baseline_complete=True,baseline_step=step)


def commit_motion_only(state,*,step,loss,nav_sampler_state,enemy_sampler_state,metrics):
    r=record(state);previous=dict(state,global_training_step=step-1);validate_motion_only(previous)
    if(state['global_training_step']!=step or not r['baseline_complete']or r['gate']['completed']
            or type(loss)not in(int,float)or not math.isfinite(loss)or loss<0
            or nav_sampler_state.get('draws')!=r['nav_sampler_state']['draws']+8
            or enemy_sampler_state.get('draws')!=r['enemy_sampler_state']['draws']+8):
        raise ValueError('Motion-only commit needs one complete finite16-actor update and8+8 private draws')
    if(any(metrics.get(key,0)!=0 for key in('moe_combat_loss','combat_loss','main_combat_actor_count',
            'combat_auxiliary_actor_count','combat_late_actor_count','main_combat_episode_rng_draw_count',
            'combat_batch_draws','extra_contact_fragment_count','contact_scout_transitions',
            'direct_combat_supervision_calls','teacher_calls','renderer_calls'))
            or metrics.get('main_navigation_actor_count')!=8 or metrics.get('main_enemy_motion_actor_count')!=8
            or metrics.get('main_independent_actor_count')!=16
            or metrics.get('microbatch_size')!=16 or metrics.get('effective_batch_size')!=16
            or metrics.get('gradient_accumulation_steps')!=1 or metrics.get('completed_microbatches')!=1
            or metrics.get('optimizer_steps')!=1
            or metrics.get('combat_role_router_supervision',False)is not False
            or metrics.get('Source_future_feedback',False)is not False):
        raise ValueError('Motion-only update contained combat examples/loss or wrong actor counts')
    gate=r['gate'];horizon=gate['current_horizon'];old=gate['stage_index']
    if(any(metrics.get(key)!=horizon for key in('main_horizon','moe_main_horizon','nav_tbptt',
            'enemy_motion_tbptt','moe_main_tbptt'))
            or metrics.get('objective_formula_revision')!=PROTOCOL
            or metrics.get('motion_only_loss_revision')!=PROTOCOL):
        raise ValueError('Motion-only MAIN16 must actually backpropagate the complete currentH')
    streams=metrics.get('motion_only_streams',[])
    if(len(streams)!=2 or {row.get('stream')for row in streams}!={'navigation','enemy_motion'}
            or any(row.get('actors')!=8 or row.get('horizon')!=horizon or row.get('tbptt')!=horizon
                or row.get('forward_transitions')!=8*horizon
                or row.get('memory_detaches_after_initialization')!=0 for row in streams)):
        raise ValueError('Motion-only full-BPTT stream work receipt changed')
    local=metrics.get('player_contact_replay',{});enemy_local=metrics.get('enemy_local_recovery',{})
    increments=dict(reused_player_local_branches=local.get('branches',metrics.get('player_contact_branches',0)),
        reused_enemy_local_branches=enemy_local.get('branches',metrics.get('enemy_local_branches',0)),
        local_Source_queries=metrics.get('motion_only_Source_queries',
            local.get('direct_source_attempts',0)+local.get('selection',{}).get('source_probes',0)
            +enemy_local.get('total_Source_step_queries',enemy_local.get('selection',{}).get('source_probes',0))))
    if(any(type(value)is not int or value<0 for value in increments.values())
            or increments['reused_player_local_branches']>4 or increments['reused_enemy_local_branches']>4):
        raise ValueError('Invalid motion-only actual MAIN-prefix branch/query count')
    for actual,previous in((nav_sampler_state,r['nav_sampler_state']),(enemy_sampler_state,r['enemy_sampler_state'])):
        if(set(actual)!=set(previous)or any(actual[key]!=previous[key]for key in actual if key not in('draws','rng'))):
            raise ValueError('Motion-only sampler identity must retain its raw parent protocol')
    # Validate every receipt before mutating the completed-update ledger. A rejected
    # partial/foreign update must leave a retryable exact pre-commit boundary.
    gate['updates']+=1;gate['stage_updates']+=1;gate['horizon_update_counts'][str(horizon)]+=1
    gate.update(last_optimizer_step=step,last_loss=float(loss))
    if gate['stage_updates']==STAGE_UPDATES:
        if old<len(HORIZONS)-1:
            gate['advance_history'].append(dict(step=step,from_horizon=horizon,to_horizon=HORIZONS[old+1],
                stage_updates=STAGE_UPDATES,reason='fixed_stage_updates',loss_diagnostic=float(loss)))
            gate.update(stage_index=old+1,current_horizon=HORIZONS[old+1],stage_updates=0)
        else:gate.update(completed=True,completion_reason='world_round_complete')
    r.update(updates=r['updates']+1,last_optimizer_step=step,nav_sampler_state=deepcopy(nav_sampler_state),
        enemy_sampler_state=deepcopy(enemy_sampler_state),main_examples=r['main_examples']+16,
        main_transitions=r['main_transitions']+16*horizon)
    for total,value in increments.items():r[total]+=value
    validate_motion_only(state);publish_contract(state)
    return r


def attach_corpus(state,receipt):
    r=record(state)
    if(not isinstance(receipt,dict)or not _sha(receipt.get('policy_checkpoint_sha256'))
            or not _sha(receipt.get('world_checkpoint_sha256'))or receipt.get('split')!='train'
            or receipt.get('validation_unchanged')is not True
            or not receipt.get('episodes')or not Path(receipt.get('nav_train_directory','')).is_dir()
            or not Path(receipt.get('enemy_train_directory','')).is_dir()):
        raise ValueError('A cycle corpus needs actual TRAIN records, frozen VAL and pinned policy/world')
    for row in receipt['episodes']:
        path=Path(row['path']).resolve()
        roots=(Path(receipt[key]).resolve()for key in('nav_train_directory','enemy_train_directory'))
        if(row.get('split')!='train'or not any(path.is_relative_to(root)for root in roots)
                or not _sha(row.get('sha256'))or file_sha(path)!=row['sha256']):
            raise ValueError('A policy-collected Source TRAIN episode changed or leaks VAL')
    identity=digest(receipt)
    if any(row['sha256']==identity for row in r['corpus_revisions']):raise ValueError('A cycle corpus cannot be appended twice')
    r['corpus_revisions'].append(dict(sha256=identity,receipt=_json(receipt),attached_at_optimizer_step=state['global_training_step']))
    return r['corpus_revisions'][-1]


def begin_motion_round(state,corpus_receipt):
    r=validate_motion_only(state)
    if r['gate']['updates']==0 and r['updates']==0:
        attach_corpus(state,corpus_receipt);publish_contract(state);return r['gate']
    if not r['gate']['completed']:raise ValueError('A serial cycle cannot skip unfinished motion horizons')
    attach_corpus(state,corpus_receipt)
    r['round_history'].append(deepcopy(r['gate']))
    r['gate']=_new_gate(state['global_training_step'],len(r['round_history']))
    validate_motion_only(state);publish_contract(state)
    return r['gate']


def publish_contract(state):
    r=record(state);gate=r['gate'];state.update(objective_protocol=PROTOCOL,loss_protocol=PROTOCOL,
        active_objective_formula_revision=PROTOCOL,horizon_curriculum_protocol=PROTOCOL)
    state.setdefault('training_protocol',{}).update(motion_only_revision=dict(protocol=PROTOCOL,
        plan_sha256=r['sha256'],**motion_only_settings()),microbatch_size=16,gradient_accumulation_steps=1,
        effective_batch_size=16,tbptt=gate['current_horizon'],active_objective_formula_revision=PROTOCOL)
    return r


def runtime_player_settings(parent,*,horizon,plan_sha256):
    from .rssm_player_geometry_objectives import PLAYER_GEOMETRY_PROTOCOL
    if(horizon not in HORIZONS or not _sha(plan_sha256)or parent.get('geometry')!=PLAYER_GEOMETRY_PROTOCOL):
        raise ValueError('Motion-only player geometry or exact current full-BPTT horizon changed')
    result=deepcopy(parent);result.update(tbptt=horizon,player_weights=dict(navigation=.5,enemy_motion=.5,combat=0.),
        main_batch_settings=dict(microbatch_size=16,gradient_accumulation_steps=1,effective_batch_size=16,
            combat=0,navigation=8,enemy_motion=8,combat_auxiliary=0,combat_late=0),
        motion_only_override=dict(protocol=PROTOCOL,plan_sha256=plan_sha256,horizon=horizon))
    return result


def legacy_binding_state(state,plan):
    r=record(state);result=deepcopy(state);result['global_training_step']=r['source_optimizer_step']
    result['rssm_training']=deepcopy(r['source_rssm_training'])
    result['training_protocol']=deepcopy(plan['source_training_protocol'])
    return result
