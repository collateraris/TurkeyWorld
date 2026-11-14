"""Final H450 loss goal over an immutable v5 curriculum history."""
from copy import deepcopy
import json
from pathlib import Path

from .rssm_combat_memory_protocol import digest_json, file_sha
from .rssm_h450_goal import h450_goal_spec, initial_h450_goal, validate_h450_goal

PROTOCOL = 'rssm_moe_h450_total_loss_goal_v6'
MANDATORY = {'240':100,'368':100}
ADVANCE = 'H450_total_loss_streak_else_mandatory_selected_stage_budget_else_loss_streak_OR_completed_stage_budget'


def h450_loss_goal_plan(state, *, source_checkpoint):
    from .rssm_moe_budget import budget_plan, TARGETED_HORIZON_PROTOCOL
    plan = budget_plan(state,source_checkpoint=source_checkpoint)
    previous = state['rssm_training']['moe_training'].get('advance_policy') or {}
    if previous.get('kind') != TARGETED_HORIZON_PROTOCOL:
        raise ValueError('H450 goal must preserve the exact existing v5 selected-horizon policy')
    plan.update(version=6,kind=PROTOCOL,budget_base=20,budget_increment=0,
        threshold=.5,required_consecutive_passes=5,comparison='strict_less_than',
        mandatory_horizon_updates=deepcopy(MANDATORY),mandatory_horizons_disable_loss_gate=True,
        advance_condition=ADVANCE,final_stage_continues=True,final_stage_max_updates=None,
        final_stage_goal=h450_goal_spec(),source_policy_sha256=previous['sha256'],
        completion_status='completed_loss_gate',loss_gate_completion_status='completed_loss_gate',
        streak_initialization='preserve_nonfinal_stage_streak_fresh_H450_goal')
    return plan


def _settings_valid(policy):
    return (type(policy.get('version')) is int and policy['version']==6
        and policy.get('kind')==PROTOCOL and type(policy.get('budget_base')) is int
        and policy['budget_base']==20 and type(policy.get('budget_increment')) is int
        and policy['budget_increment']==0 and policy.get('threshold')==.5
        and type(policy.get('required_consecutive_passes')) is int and policy['required_consecutive_passes']==5
        and policy.get('comparison')=='strict_less_than' and policy.get('mandatory_horizon_updates')==MANDATORY
        and all(type(n) is int for n in policy['mandatory_horizon_updates'].values())
        and policy.get('mandatory_horizons_disable_loss_gate') is True
        and policy.get('final_stage_max_updates','missing') is None
        and policy.get('final_stage_goal')==h450_goal_spec())


def validate_h450_policy(gate,step):
    from .rssm_moe_budget import TARGETED_HORIZON_PROTOCOL
    from .rssm_moe_protocol import _validate_gate
    policy = gate['advance_policy']
    previous = policy.get('source_gate')
    if (not _settings_valid(policy) or not isinstance(previous,dict)
            or type(policy.get('source_optimizer_step')) is not int
            or type(policy.get('source_moe_updates')) is not int
            or type(policy.get('updates')) is not int or policy['updates']<0
            or policy['source_optimizer_step']+policy['updates']!=step
            or policy['source_moe_updates']+policy['updates']!=gate['updates']
            or policy.get('last_optimizer_step')!=step
            or previous.get('last_optimizer_step')!=policy['source_optimizer_step']
            or previous.get('updates')!=policy['source_moe_updates']
            or digest_json(previous)!=policy.get('source_gate_sha256')
            or previous.get('sha256')!=gate.get('sha256')
            or (previous.get('advance_policy') or {}).get('kind')!=TARGETED_HORIZON_PROTOCOL
            or (previous.get('advance_policy') or {}).get('sha256')!=policy.get('previous_policy_sha256')
            or gate['advance_history'][:len(previous['advance_history'])]!=previous['advance_history']
            or any(gate['horizon_update_counts'].get(h,-1)<n for h,n in previous['horizon_update_counts'].items())):
        raise ValueError('H450 policy changed its immutable v5 source or completed update clocks')
    _validate_gate(previous,policy['source_optimizer_step'])
    goal = validate_h450_goal(gate.get('final_horizon_goal'),current_step=step)
    if (goal['plan_sha256']!=policy['sha256'] or goal['source_optimizer_step']!=policy['source_optimizer_step']
            or goal['updates']!=policy['updates']
            or goal['h450_updates']!=gate['horizon_update_counts']['450']-previous['horizon_update_counts']['450']):
        raise ValueError('H450 goal does not match the exact policy and actual final-horizon commits')
    if policy['updates']:
        if goal['last_total_loss']!=gate['last_train_loss'] or goal['last_horizon']!=gate['last_training_horizon']:
            raise ValueError('H450 goal did not measure the actual last committed TOTAL/horizon')
    if gate['current_horizon']==450 and (gate['consecutive_below_threshold']!=goal['streak']
                                       or gate['goal_met'] is not goal['goal_met']):
        raise ValueError('Final curriculum streak differs from the measured H450 goal')


def validate_h450_completion(gate):
    from .rssm_moe_budget import stage_budget,stage_requires_full_budget
    policy = gate['advance_policy']
    goal = gate['final_horizon_goal']
    done = goal['goal_met']
    cap = stage_budget(gate['stage_index'],policy)
    mandatory = stage_requires_full_budget(gate['stage_index'],policy)
    if (gate.get('budget_exhausted') is not False or gate.get('curriculum_completed') is not done
            or gate.get('completion_reason')!=('loss_streak' if done else None)
            or gate.get('effective_stage_budget')!=cap or gate.get('mandatory_stage_budget') is not mandatory
            or gate.get('loss_streak_can_advance') is not (not mandatory)
            or cap is not None and gate['stage_updates']>=cap):
        raise ValueError('H450 unbounded completion or intermediate stage budget changed')


def bind_h450_policy(state,plan,*,path,sha256,loaded_checkpoint_sha256):
    from .rssm_moe_budget import update_completion
    from .rssm_moe_protocol import _validate_gate
    gate = state['rssm_training']['moe_training']
    step = state['global_training_step']
    _validate_gate(gate,step)
    current = gate.get('advance_policy') or {}
    replacing = current.get('kind')!=PROTOCOL
    previous = deepcopy(gate if replacing else current['source_gate'])
    source_state = dict(global_training_step=previous['last_optimizer_step'],rssm_training=dict(moe_training=previous))
    expected = h450_loss_goal_plan(source_state,source_checkpoint=plan['source_checkpoint'])
    if (json.loads(Path(path).read_text(encoding='utf-8'))!=plan or file_sha(path)!=sha256 or plan!=expected
            or file_sha(plan['source_checkpoint'])!=plan['source_checkpoint_sha256']):
        raise ValueError('H450 goal plan changed its exact policy, criterion or source checkpoint')
    if replacing:
        if (step!=plan['source_optimizer_step'] or loaded_checkpoint_sha256!=plan['source_checkpoint_sha256']
                or digest_json(gate)!=plan['source_gate_sha256']):
            raise ValueError('H450 goal activation requires its complete pinned optimizer boundary')
        policy = dict(version=6,kind=PROTOCOL,plan=str(Path(path).resolve()),sha256=sha256,
            budget_base=20,budget_increment=0,threshold=.5,required_consecutive_passes=5,
            comparison='strict_less_than',mandatory_horizon_updates=deepcopy(MANDATORY),
            mandatory_horizons_disable_loss_gate=True,final_stage_max_updates=None,final_stage_goal=h450_goal_spec(),
            source_optimizer_step=step,source_moe_updates=gate['updates'],last_optimizer_step=step,updates=0,
            source_gate=previous,source_gate_sha256=plan['source_gate_sha256'],
            previous_policy_sha256=plan['source_policy_sha256'])
        gate['advance_policy']=policy
        gate['final_horizon_goal']=initial_h450_goal(source_optimizer_step=step,plan_sha256=sha256)
        if gate['current_horizon']==450:
            gate.update(consecutive_below_threshold=0,last_gate_passed=False,goal_met=False)
        update_completion(gate)
    else:
        policy = current
        if (current['sha256']!=sha256 or current['source_optimizer_step']!=plan['source_optimizer_step']
                or current['source_gate_sha256']!=plan['source_gate_sha256']
                or current['previous_policy_sha256']!=plan['source_policy_sha256']):
            raise ValueError('Saved H450 goal requires its same immutable continuation plan')
    _validate_gate(gate,step)
    state['horizon_curriculum_protocol']=PROTOCOL
    state.setdefault('training_protocol',{})['moe_stage_budget']=dict(
        kind=PROTOCOL,version=6,plan_sha256=sha256,budget_base=20,budget_increment=0,stage_index_origin=0,
        advance_condition=ADVANCE,final_stage_continues=True,threshold=.5,required_consecutive_passes=5,
        comparison='strict_less_than',mandatory_horizon_updates=deepcopy(MANDATORY),
        mandatory_horizons_disable_loss_gate=True,final_stage_max_updates=None,final_stage_goal=h450_goal_spec())
    return policy
