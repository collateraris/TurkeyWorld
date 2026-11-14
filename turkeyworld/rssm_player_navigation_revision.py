"""Pinned player-contact objective revision with an unchanged H450 curriculum."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

PROTOCOL='rssm_player_navigation_repair_v1'
FULL_HORIZON_PROTOCOL='rssm_player_navigation_repair_tbptt450_batch16_v2'
SEED=174175
HORIZONS=(4,8,16,32)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def file_sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def _sha(value):return isinstance(value,str)and re.fullmatch('[0-9a-f]{64}',value)is not None


def player_navigation_settings(*,tbptt=64,microbatch_size=32):
    from .rssm_player_geometry_objectives import PLAYER_GEOMETRY_PROTOCOL
    if(type(tbptt)is not int or type(microbatch_size)is not int
            or(tbptt,microbatch_size)not in((64,32),(450,16))):
        raise ValueError('Player repair needs explicit TBPTT64/B32 or TBPTT450/B16')
    settings=dict(version=1,tbptt=64,geometry=deepcopy(PLAYER_GEOMETRY_PROTOCOL),
        player_xy_reduction='actor_weighted_means_all_three_main_streams',
        player_weights=dict(combat=19/32,navigation=5/32,enemy_motion=8/32),
        NPC_weights_unchanged=True,combat_weights_unchanged=True,main_actor_quotas_unchanged=True,
        contact=dict(seed=SEED,actors_per_microbatch=4,horizons=list(HORIZONS),weight=.25,
            context=1,source_scout_horizon=32,source_probe_budget=32,
            candidate_streams=['combat','navigation','enemy_motion'],
            whole_own_memory_preserved=True,TBPTT=64),
        Source_future_feedback=False,world_or_renderer_training=False,optimizer_reset=False,
        architecture_change=False,old_sampler_reset=False,horizon_reset=False,
        H450_threshold=1.,H450_required_consecutive=10,H450_max_updates=None,
        validation_every=250,preview_every=10,best_selection_independent_of_train_goal=True)
    if(tbptt,microbatch_size)==(450,16):
        settings.update(version=2,tbptt=450,
            player_weights=dict(combat=4/16,navigation=4/16,enemy_motion=8/16),
            NPC_weights_unchanged=False,NPC_actor_weight_normalization_unchanged=True,
            main_actor_quotas_unchanged=False,
            main_batch_settings=dict(microbatch_size=16,gradient_accumulation_steps=1,
                effective_batch_size=16,combat=4,navigation=4,enemy_motion=8,
                combat_auxiliary=16,combat_late=2),
            full_horizon_credit_without_memory_detach=True)
    return settings


def player_navigation_protocol(settings):
    """Validate a canonical objective independently of mutable UI metadata."""
    for tbptt,batch,protocol in((64,32,PROTOCOL),(450,16,FULL_HORIZON_PROTOCOL)):
        if digest(settings)==digest(player_navigation_settings(tbptt=tbptt,microbatch_size=batch)):
            return protocol
    raise ValueError('Unknown player-navigation objective revision')


def player_navigation_settings_for_plan(plan):
    settings=plan.get('settings');protocol=player_navigation_protocol(settings)
    version=1 if protocol==PROTOCOL else 2
    if(plan.get('kind')!=protocol or type(plan.get('version'))is not int or plan['version']!=version):
        raise ValueError('Player navigation settings and immutable protocol/version differ')
    return deepcopy(settings)


def _state(saved):return saved.get('run_state',saved)


def player_navigation_plan(saved,*,source_checkpoint,source_checkpoint_sha256,
                           contact_root,contact_sampler_identity,tbptt=64,microbatch_size=32):
    """Create a complete-boundary plan; model/optimizer/RNG are never touched."""
    from .rssm_moe_protocol import moe_training_horizon
    from .rssm_moe_batch_revision import moe_batch_settings,MICRO16_SINGLE_PROTOCOL
    settings=player_navigation_settings(tbptt=tbptt,microbatch_size=microbatch_size)
    protocol=player_navigation_protocol(settings)
    state=_state(saved);step=state.get('global_training_step')
    root=Path(contact_root).resolve();source=Path(source_checkpoint).resolve()
    if(type(step)is not int or step<1 or not _sha(source_checkpoint_sha256)
            or not source.is_file()or file_sha(source)!=source_checkpoint_sha256
            or state.get('rssm_training',{}).get('player_navigation_revision')is not None
            or not isinstance(contact_sampler_identity,dict)or not contact_sampler_identity):
        raise ValueError('Player navigation needs an immutable complete source and TRAIN sampler identity')
    horizon=moe_training_horizon(state);gate=state['rssm_training']['moe_training']
    batch=moe_batch_settings(state)
    goal=gate.get('final_horizon_goal')or{}
    if(horizon!=450 or gate.get('consecutive_below_threshold')!=0 or goal.get('streak')!=0
            or goal.get('goal_met')is not False
            or tuple(batch[k]for k in('microbatch_size','gradient_accumulation_steps','effective_batch_size'))!=(microbatch_size,1,microbatch_size)):
        raise ValueError('Player repair requires its explicit H450 batch with streak0; no pass reset is inferred')
    batch_record=gate.get('batch_revision')
    if protocol==FULL_HORIZON_PROTOCOL and(not isinstance(batch_record,dict)
            or batch_record.get('kind')!=MICRO16_SINGLE_PROTOCOL or batch_record.get('updates')!=0
            or batch_record.get('source_optimizer_step')!=step
            or batch_record.get('source_checkpoint_sha256')!=source_checkpoint_sha256):
        raise ValueError('Full-horizon player repair requires the fresh exact B32x1-to-B16x1 source binding')
    plan=dict(kind=protocol,version=settings['version'],settings=settings,
        source_checkpoint=str(source),source_checkpoint_sha256=source_checkpoint_sha256,
        source_optimizer_step=step,source_moe_updates=gate['updates'],source_horizon=horizon,
        source_gate_sha256=digest(gate),source_H450_goal_sha256=digest(goal),
        moe_training_plan_sha256=gate['sha256'],
        batch_settings={k:batch[k]for k in('microbatch_size','gradient_accumulation_steps','effective_batch_size')},
        contact_root=str(root),contact_train_directory=str(root/'short/rgb/train'),
        contact_manifest_sha256=file_sha(root/'manifest.json'),
        contact_sampler_identity=deepcopy(contact_sampler_identity),
        contact_sampler_identity_sha256=digest(contact_sampler_identity),
        previous_TOTAL_not_reinterpreted=True,model_optimizer_and_existing_RNG_unchanged=True)
    if protocol==FULL_HORIZON_PROTOCOL:
        plan.update(batch_revision_sha256=batch_record['sha256'],
            source_pre_batch_gate_sha256=batch_record['source_gate_sha256'],
            source_pre_batch_protocol=batch_record['source_gate']['batch_protocol'])
    return plan


def _validate_record(record,step):
    if not isinstance(record,dict):raise ValueError('Missing player navigation ledger')
    protocol=player_navigation_protocol(record.get('settings'))
    if(record.get('protocol')!=protocol
            or type(record.get('version'))is not int or record['version']!=record['settings']['version']
            or type(record.get('source_optimizer_step'))is not int
            or record['source_optimizer_step']<1
            or type(record.get('updates'))is not int or record['updates']<0
            or type(record.get('last_optimizer_step'))is not int
            or record.get('last_optimizer_step')!=step
            or record['source_optimizer_step']+record['updates']!=step
            or type(record.get('contact_draws'))is not int
            or record.get('contact_draws')!=4*record['updates']
            or not isinstance(record.get('baseline_complete'),bool)
            or record.get('baseline_step')!=(record['source_optimizer_step']if record['baseline_complete']else None)
            or type(record.get('contact_generated_transitions'))is not int
            or record['contact_generated_transitions']<0
            or record['updates']and(not record['baseline_complete']or record.get('contact_sampler_state')is None)
            or record.get('contact_sampler_state')is not None and(
                not isinstance(record['contact_sampler_state'],dict)
                or record['contact_sampler_state'].get('draws')!=record['contact_draws'])):
        raise ValueError('Player navigation complete-update, sampler or baseline ledger changed')
    return record


def bind_player_navigation_revision(state,plan,*,path,sha256,loaded_checkpoint_sha256):
    from .rssm_moe_batch_revision import moe_batch_settings,MICRO16_SINGLE_PROTOCOL
    settings=player_navigation_settings_for_plan(plan);protocol=plan['kind']
    raw=Path(path).read_bytes()
    root=Path(plan.get('contact_root',''))
    source=Path(plan.get('source_checkpoint',''))
    if(not _sha(sha256)or hashlib.sha256(raw).hexdigest()!=sha256
            or json.loads(raw.decode('utf-8'))!=plan
            or not source.is_absolute()or not source.is_file()
            or file_sha(source)!=plan.get('source_checkpoint_sha256')
            or not root.is_absolute()or file_sha(root/'manifest.json')!=plan.get('contact_manifest_sha256')
            or Path(plan.get('contact_train_directory','')).resolve()!=root/'short/rgb/train'
            or digest(plan.get('contact_sampler_identity'))!=plan.get('contact_sampler_identity_sha256')
            or plan.get('previous_TOTAL_not_reinterpreted')is not True
            or plan.get('model_optimizer_and_existing_RNG_unchanged')is not True):
        raise ValueError('Player navigation immutable plan, source or contact data changed')
    step=state.get('global_training_step');q=state['rssm_training'];gate=q['moe_training']
    size=32 if protocol==PROTOCOL else 16
    expected_batch=dict(microbatch_size=size,gradient_accumulation_steps=1,effective_batch_size=size)
    actual_batch=moe_batch_settings(state)
    if plan.get('batch_settings')!=expected_batch or any(actual_batch[k]!=v for k,v in expected_batch.items()):
        raise ValueError('Player navigation revision cannot change its explicitly pinned batch')
    if protocol==FULL_HORIZON_PROTOCOL:
        batch_record=gate.get('batch_revision')or{}
        if(batch_record.get('kind')!=MICRO16_SINGLE_PROTOCOL
                or batch_record.get('sha256')!=plan.get('batch_revision_sha256')
                or batch_record.get('source_gate_sha256')!=plan.get('source_pre_batch_gate_sha256')
                or batch_record.get('source_gate',{}).get('batch_protocol')!=plan.get('source_pre_batch_protocol')
                or batch_record.get('source_checkpoint_sha256')!=plan.get('source_checkpoint_sha256')):
            raise ValueError('Full-horizon player repair lost its explicit smaller-batch source lineage')
    if gate['sha256']!=plan.get('moe_training_plan_sha256')or gate['last_optimizer_step']!=step:
        raise ValueError('Player navigation cannot alter the established MoE plan clock')
    record=q.get('player_navigation_revision')
    if record is None:
        if(step!=plan.get('source_optimizer_step')or loaded_checkpoint_sha256!=plan.get('source_checkpoint_sha256')
                or gate.get('updates')!=plan.get('source_moe_updates')or gate.get('current_horizon')!=450
                or digest(gate)!=plan.get('source_gate_sha256')
                or digest(gate.get('final_horizon_goal'))!=plan.get('source_H450_goal_sha256')
                or gate.get('consecutive_below_threshold')!=0 or gate['final_horizon_goal'].get('streak')!=0):
            raise ValueError('Player repair activation needs the exact complete streak0 source; no reset allowed')
        record=dict(protocol=protocol,version=plan['version'],plan_path=str(Path(path).resolve()),sha256=sha256,
            source_optimizer_step=step,last_optimizer_step=step,updates=0,source_moe_updates=gate['updates'],
            source_checkpoint_sha256=plan['source_checkpoint_sha256'],source_gate_sha256=plan['source_gate_sha256'],
            source_H450_goal_sha256=plan['source_H450_goal_sha256'],settings=deepcopy(plan['settings']),
            contact_sampler_identity_sha256=plan['contact_sampler_identity_sha256'],
            contact_sampler_state=None,contact_draws=0,contact_generated_transitions=0,
            baseline_complete=False,baseline_step=None)
        q['player_navigation_revision']=record
        state.setdefault('objective_history',[]).append(dict(step=step,kind=protocol,
            source_loss=state.get('loss'),source_latest_metrics=deepcopy(state.get('latest_metrics')),
            settings=deepcopy(plan['settings']),reason='repair_player_sliding_contacts_and_reuse_quiet_player_labels',
            previous_TOTAL_comparable=False,source_future_feedback=False))
    elif(record.get('protocol')!=protocol or record.get('sha256')!=sha256 or record.get('source_checkpoint_sha256')!=plan['source_checkpoint_sha256']
            or record.get('source_optimizer_step')!=plan['source_optimizer_step']
            or record.get('source_gate_sha256')!=plan['source_gate_sha256']
            or record.get('source_H450_goal_sha256')!=plan['source_H450_goal_sha256']
            or record.get('contact_sampler_identity_sha256')!=plan['contact_sampler_identity_sha256']):
        raise ValueError('Saved player navigation revision identity changed')
    _validate_record(record,step)
    if gate['updates']!=record['source_moe_updates']+record['updates']:
        raise ValueError('Player repair and existing MoE optimizer ledgers diverged')
    state.setdefault('training_protocol',{})['player_navigation_repair']=dict(
        protocol=protocol,plan_sha256=sha256,**deepcopy(settings))
    state['active_objective_formula_revision']=protocol
    return record


def mark_player_navigation_baseline(state,*,step):
    record=state['rssm_training']['player_navigation_revision']
    _validate_record(record,state['global_training_step'])
    if record['updates']or step!=record['source_optimizer_step']or state['global_training_step']!=step:
        raise ValueError('Fresh player baseline must precede every optimizer update')
    record['baseline_complete']=True;record['baseline_step']=step
    return record


def commit_player_navigation_revision(state,*,step,contact_sampler_state,metrics):
    record=state['rssm_training']['player_navigation_revision']
    _validate_record(record,step-1)
    if state['global_training_step']!=step or state['rssm_training']['moe_training']['last_optimizer_step']!=step:
        raise ValueError('Player repair counts only a complete committed MoE optimizer boundary')
    if not record['baseline_complete']:raise ValueError('Player repair must measure its initial own450 baseline before SGD')
    value=metrics.get('player_contact_loss')
    if(type(value)not in(int,float)or not math.isfinite(value)or value<0
            or metrics.get('player_navigation_loss_revision')!=record['protocol']
            or not isinstance(contact_sampler_state,dict)
            or contact_sampler_state.get('draws')!=record['contact_draws']+4):
        raise ValueError('Player navigation needs finite contact loss and exact objective tag')
    transitions=metrics.get('player_contact_generated_transitions',0)
    if type(transitions)is not int or transitions<0:raise ValueError('Invalid contact exposure counter')
    record['updates']+=1;record['last_optimizer_step']=step
    record['contact_draws']+=4;record['contact_sampler_state']=deepcopy(contact_sampler_state)
    record['contact_generated_transitions']+=transitions
    return _validate_record(record,step)
