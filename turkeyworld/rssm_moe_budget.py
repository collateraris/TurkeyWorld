"""Versioned per-stage budgets and loss gates with immutable parent history."""
from copy import deepcopy
import json
from pathlib import Path

from .rssm_combat_memory_protocol import digest_json, file_sha

BUDGET_PROTOCOL = 'rssm_moe_loss_or_stage_budget_v2'
FLAT_CAP_PROTOCOL = 'rssm_moe_flat_stage_cap_v3'
SHORT_CAP_PROTOCOL = 'rssm_moe_flat_stage_cap_v4'
CAP100_PROTOCOL = FLAT_CAP_PROTOCOL  # Existing preparation callers may keep this alias.
FLAT_PROTOCOLS = (FLAT_CAP_PROTOCOL, SHORT_CAP_PROTOCOL)


def loss_gate_settings(gate):
    policy = gate.get('advance_policy') or {}
    if policy.get('kind') == SHORT_CAP_PROTOCOL:
        return policy['threshold'], policy['required_consecutive_passes']
    return .25, 10


def stage_budget(index, policy=None):
    if type(index) is not int or not 0 <= index <= 30:
        raise ValueError('MoE stage index must be zero-based0..30')
    if policy is not None and policy.get('kind') in FLAT_PROTOCOLS:
        return policy['budget_base']
    return 50+15*index


def curriculum_completion(gate):
    """Flat policies terminate the final stage; legacy v2 remains continuous."""
    policy = gate.get('advance_policy')
    if not isinstance(policy, dict) or policy.get('kind') not in FLAT_PROTOCOLS:
        return None
    if gate.get('goal_met'):
        return 'completed_loss_gate'
    if gate.get('stage_index') == 30 and gate.get('stage_updates', 0) >= policy['budget_base']:
        return 'completed_budget'
    return None


def update_completion(gate):
    policy = gate.get('advance_policy')
    if not isinstance(policy, dict) or policy.get('kind') not in FLAT_PROTOCOLS:
        return
    completion = curriculum_completion(gate)
    gate.update(budget_exhausted=completion == 'completed_budget',
        curriculum_completed=completion is not None,
        completion_reason={'completed_budget': 'stage_budget',
                           'completed_loss_gate': 'loss_streak'}.get(completion))


def validate_completion(gate):
    policy = gate.get('advance_policy')
    if not isinstance(policy, dict) or policy.get('kind') not in FLAT_PROTOCOLS:
        return
    completion = curriculum_completion(gate)
    expected = (completion == 'completed_budget', completion is not None,
                {'completed_budget': 'stage_budget', 'completed_loss_gate': 'loss_streak'}.get(completion))
    if tuple(gate.get(k) for k in ('budget_exhausted', 'curriculum_completed', 'completion_reason')) != expected:
        raise ValueError('Saved MoE final-stage completion or budget exhaustion changed')
    # An already overdue source is historical; no additional step may run there.
    cap = policy['budget_base']
    maximum = max(cap, policy['source_gate']['stage_updates']) if gate['stage_index'] == policy['source_gate']['stage_index'] else cap
    if gate['stage_updates'] > maximum or gate['stage_index'] < 30 and gate['stage_updates'] >= cap:
        raise ValueError('Saved MoE stage exceeded its explicit flat update cap')


def validate_policy(gate, step):
    policy = gate.get('advance_policy')
    if policy is None:return
    if (not isinstance(policy,dict) or policy.get('kind') not in (BUDGET_PROTOCOL, *FLAT_PROTOCOLS)
            or (policy.get('kind') == CAP100_PROTOCOL and
                (type(policy.get('budget_base')) is not int or policy['budget_base'] not in (50, 100)
                 or policy.get('budget_increment') != 0))
            or (policy.get('kind') == SHORT_CAP_PROTOCOL and
                (policy.get('version') != 4 or type(policy.get('budget_base')) is not int
                 or policy['budget_base'] not in (20,25) or policy.get('budget_increment') != 0
                 or policy.get('threshold') != .5 or type(policy.get('required_consecutive_passes')) is not int
                 or policy['required_consecutive_passes'] != 5 or policy.get('comparison') != 'strict_less_than'))
            or (policy.get('kind') == BUDGET_PROTOCOL and
                (policy.get('budget_base'), policy.get('budget_increment')) != (50, 15))
            or type(policy.get('source_optimizer_step')) is not int
            or type(policy.get('source_moe_updates')) is not int
            or type(policy.get('updates')) is not int or policy['updates'] < 0
            or policy['source_optimizer_step']+policy['updates'] != step
            or policy['source_moe_updates']+policy['updates'] != gate['updates']
            or policy.get('last_optimizer_step') != step):
        raise ValueError('Saved MoE budget policy, completed update clock or zero-based formula changed')
    if policy['kind'] in FLAT_PROTOCOLS:
        from .rssm_moe_protocol import _validate_gate
        previous = policy.get('source_gate')
        if (policy.get('version') != (4 if policy['kind'] == SHORT_CAP_PROTOCOL else 3)
                or not isinstance(previous, dict)
                or policy['kind'] == CAP100_PROTOCOL and (previous.get('advance_policy') or {}).get('kind') in FLAT_PROTOCOLS
                or digest_json(previous) != policy.get('source_gate_sha256')
                or previous.get('updates') != policy['source_moe_updates']
                or previous.get('last_optimizer_step') != policy['source_optimizer_step']
                or previous.get('sha256') != gate.get('sha256')
                or (previous.get('advance_policy') or {}).get('sha256') != policy.get('previous_policy_sha256')
                or gate.get('advance_history', [])[:len(previous.get('advance_history', []))] != previous.get('advance_history')
                or any(gate.get('horizon_update_counts', {}).get(h, -1) < n
                       for h, n in previous.get('horizon_update_counts', {}).items())):
            raise ValueError('MoE flat-cap parent policy, immutable history or source gate changed')
        _validate_gate(previous, policy['source_optimizer_step'])


def validate_budget_advance(gate, entry, index):
    policy=gate.get('advance_policy')
    # Historical v2 entries retain their actual 50+15*N policy after v3 binds.
    if isinstance(policy, dict) and policy.get('kind') in FLAT_PROTOCOLS and entry.get('advance_policy_sha256') != policy.get('sha256'):
        previous = policy['source_gate']
        if entry not in previous.get('advance_history', []):
            return False
        return validate_budget_advance(previous, entry, index)
    return (isinstance(policy,dict) and policy.get('kind') in (BUDGET_PROTOCOL, *FLAT_PROTOCOLS)
        and entry.get('advance_policy_sha256')==policy.get('sha256')
        and entry.get('step',-1)>=policy['source_optimizer_step']
        and entry.get('stage_budget')==stage_budget(index, policy)
        and type(entry.get('completed_updates_at_previous_stage')) is int
        and entry['completed_updates_at_previous_stage']>=stage_budget(index, policy)
        and (policy['kind'] not in FLAT_PROTOCOLS or entry.get('activation_boundary') is True
             or entry['completed_updates_at_previous_stage'] == policy['budget_base'])
        and type(entry.get('qualifying_streak')) is int
        and 0<=entry['qualifying_streak']<=loss_gate_settings(gate)[1])


def validate_loss_advance(gate, entry):
    """Validate each historical advance using the policy that actually made it."""
    policy = gate.get('advance_policy') or {}
    if policy.get('kind') in FLAT_PROTOCOLS and entry.get('advance_policy_sha256') != policy.get('sha256'):
        previous = policy['source_gate']
        return entry in previous.get('advance_history', []) and validate_loss_advance(previous, entry)
    threshold, required = loss_gate_settings(gate)
    loss = entry.get('train_loss')
    return (entry.get('advance_reason') in (None, 'loss_streak')
            and type(loss) in (float, int) and 0 <= loss < threshold
            and entry.get('qualifying_streak') == required
            and type(entry.get('completed_updates_at_previous_stage')) is int
            and entry['completed_updates_at_previous_stage'] >= required)


def budget_plan(state, *, source_checkpoint):
    from .rssm_moe_protocol import moe_training_horizon
    moe_training_horizon(state)
    return dict(version=2,kind=BUDGET_PROTOCOL,source_optimizer_step=state['global_training_step'],
        source_checkpoint=str(Path(source_checkpoint).resolve()),source_checkpoint_sha256=file_sha(source_checkpoint),
        source_gate_sha256=digest_json(state['rssm_training']['moe_training']),
        moe_training_plan_sha256=state['rssm_training']['moe_training']['sha256'],
        budget_base=50,budget_increment=15,stage_index_origin=0,
        threshold=.25,required_consecutive_passes=10,comparison='strict_less_than',
        advance_condition='loss_streak_OR_completed_stage_budget',
        preserve_completed_updates=True,weights_reset=False,optimizer_reset=False,
        final_stage_continues=True,budget_does_not_qualify_gameplay=True)


def flat_cap_plan(state, *, source_checkpoint, cap=50):
    """Pin a new policy at the exact full boundary without resetting the stage."""
    if type(cap) is not int or cap not in (50, 100):
        raise ValueError('Flat stage cap must be an explicit supported integer50 or100')
    plan = budget_plan(state, source_checkpoint=source_checkpoint)
    gate = state['rssm_training']['moe_training']
    previous = gate.get('advance_policy')
    if previous is not None and previous.get('kind') != BUDGET_PROTOCOL:
        raise ValueError('Flat-cap activation requires the original gate or its historical v2 policy')
    plan.update(version=3, kind=CAP100_PROTOCOL, budget_base=cap, budget_increment=0,
        final_stage_continues=False, final_stage_max_updates=cap,
        source_policy_sha256=previous.get('sha256') if previous else None,
        completion_status='completed_budget', loss_gate_completion_status='completed_loss_gate')
    return plan


def cap100_plan(state, *, source_checkpoint):
    return flat_cap_plan(state, source_checkpoint=source_checkpoint, cap=100)


def short_cap_plan(state, *, source_checkpoint, cap=25):
    """Continue a complete boundary with20/25 updates OR five losses<.5."""
    if type(cap) is not int or cap not in (20,25):
        raise ValueError('Short stage cap must be an explicit supported integer20 or25')
    plan = budget_plan(state, source_checkpoint=source_checkpoint)
    previous = state['rssm_training']['moe_training'].get('advance_policy') or {}
    plan.update(version=4, kind=SHORT_CAP_PROTOCOL, budget_base=cap, budget_increment=0,
        threshold=.5, required_consecutive_passes=5, final_stage_continues=False,
        final_stage_max_updates=cap, source_policy_sha256=previous.get('sha256'),
        completion_status='completed_budget', loss_gate_completion_status='completed_loss_gate',
        streak_initialization='new_policy_zero_no_reinterpretation_of_historical_losses')
    return plan


def bind_budget_policy(state, plan, *, path, sha256, loaded_checkpoint_sha256):
    from .rssm_moe_protocol import _validate_gate, MOE_HORIZONS
    raw=json.loads(Path(path).read_text(encoding='utf-8'))
    short = plan.get('kind') == SHORT_CAP_PROTOCOL and plan.get('version') == 4
    flat = short or plan.get('kind') == CAP100_PROTOCOL and plan.get('version') == 3
    if (raw!=plan or file_sha(path)!=sha256
            or not (flat or plan.get('kind') == BUDGET_PROTOCOL and plan.get('version') == 2)
            or flat and (type(plan.get('budget_base')) is not int
                         or plan['budget_base'] not in ((20,25) if short else (50,100))
                         or (plan.get('budget_increment'),plan.get('stage_index_origin')) != (0,0))
            or not flat and (plan.get('budget_base'),plan.get('budget_increment'),plan.get('stage_index_origin'))!=(50,15,0)
            or (plan.get('threshold'),plan.get('required_consecutive_passes'))!=((.5,5) if short else (.25,10))
            or short and plan.get('streak_initialization') != 'new_policy_zero_no_reinterpretation_of_historical_losses'
            or plan.get('comparison') != 'strict_less_than'
            or plan.get('advance_condition')!='loss_streak_OR_completed_stage_budget'
            or any(plan.get(k)is not True for k in ('preserve_completed_updates','budget_does_not_qualify_gameplay'))
            or plan.get('final_stage_continues') is not (not flat)
            or flat and (plan.get('final_stage_max_updates') != plan['budget_base']
                         or plan.get('completion_status') != 'completed_budget'
                         or plan.get('loss_gate_completion_status') != 'completed_loss_gate')
            or any(plan.get(k)is not False for k in ('weights_reset','optimizer_reset'))
            or file_sha(plan['source_checkpoint'])!=plan['source_checkpoint_sha256']):
        raise ValueError('MoE immutable loss-or-budget policy or full source changed')
    gate=state['rssm_training']['moe_training'];step=state['global_training_step']
    _validate_gate(gate,step)
    if gate['sha256']!=plan['moe_training_plan_sha256']:
        raise ValueError('Stage budget cannot replace the established objective plan')
    policy=gate.get('advance_policy')
    replacing = (flat and policy is not None and policy.get('kind') == BUDGET_PROTOCOL
                 or short and policy is not None and policy.get('sha256') != sha256)
    if policy is None or replacing:
        if (step!=plan['source_optimizer_step'] or loaded_checkpoint_sha256!=plan['source_checkpoint_sha256']
                or digest_json(gate)!=plan['source_gate_sha256']):
            raise ValueError('Stage-budget activation requires its exact complete source optimizer/gate')
        previous=deepcopy(gate)
        if flat and plan.get('source_policy_sha256') != (policy or {}).get('sha256'):
            raise ValueError('Flat-cap activation changed the exact historical parent budget policy')
        policy=dict(version=4 if short else 3 if flat else 2,kind=plan['kind'],plan=str(Path(path).resolve()),sha256=sha256,
            budget_base=plan['budget_base'],budget_increment=plan['budget_increment'],source_optimizer_step=step,
            source_moe_updates=gate['updates'],last_optimizer_step=step,updates=0,source_gate=previous)
        if flat:
            policy.update(source_gate_sha256=plan['source_gate_sha256'],
                          previous_policy_sha256=plan['source_policy_sha256'])
        if short:
            policy.update(threshold=.5, required_consecutive_passes=5, comparison='strict_less_than')
            gate.update(consecutive_below_threshold=0, last_gate_passed=False,
                        last_stage_advanced=False, goal_met=False)
        gate['advance_policy']=policy
        # The user counts updates already completed on the current stage.
        # An overdue stage advances once at this existing complete boundary.
        old=gate['stage_index']
        if old<len(MOE_HORIZONS)-1 and gate['stage_updates']>=stage_budget(old, policy):
            gate['advance_history'].append(dict(step=step,train_loss=gate['last_train_loss'],
                from_horizon=MOE_HORIZONS[old],to_horizon=MOE_HORIZONS[old+1],
                qualifying_streak=gate['consecutive_below_threshold'],
                completed_updates_at_previous_stage=gate['stage_updates'],advance_reason='stage_budget',
                stage_budget=stage_budget(old, policy),advance_policy_sha256=sha256,activation_boundary=True))
            gate.update(stage_index=old+1,gate_advances=old+1,stage_updates=0,
                current_horizon=MOE_HORIZONS[old+1],next_horizon=MOE_HORIZONS[old+1],
                consecutive_below_threshold=0,last_stage_advanced=True,
                last_advance_reason='stage_budget',goal_met=False)
        update_completion(gate)
    elif (policy.get('sha256')!=sha256 or policy.get('source_optimizer_step')!=plan['source_optimizer_step']
          or policy.get('kind') != plan['kind']
          or (policy.get('budget_base'), policy.get('budget_increment')) != (plan['budget_base'], plan['budget_increment'])
          or flat and
          (policy.get('source_gate_sha256') != plan['source_gate_sha256']
           or policy.get('previous_policy_sha256') != plan['source_policy_sha256'])
          or short and (policy.get('threshold'), policy.get('required_consecutive_passes')) != (.5,5)):
        raise ValueError('Saved stage-budget policy identity changed')
    _validate_gate(gate,step)
    state['horizon_curriculum_protocol']=plan['kind']
    state.setdefault('training_protocol',{})['moe_stage_budget']={
        'kind':plan['kind'],'plan_sha256':sha256,'budget_base':plan['budget_base'],'budget_increment':plan['budget_increment'],
        'stage_index_origin':0,'advance_condition':'loss_streak_OR_completed_stage_budget',
        'final_stage_continues':not flat,
        **({'threshold':.5, 'required_consecutive_passes':5, 'comparison':'strict_less_than'} if short else {})}
    return policy
