"""Explicit three-task MoE phase and total-loss horizon gate.

This metadata-only protocol never resets tensors, Adam, RNG or old plans.
Legacy phase ledgers are archived rather than committed by this new phase.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re


MOE_PROTOCOL = 'rssm_dual_transformer_moe_three_tasks_v1'
MOE_HORIZONS = (4, 8, 16, *range(32, 449, 16), 450)
SETTINGS = {
    'version': 1, 'objective_protocol': MOE_PROTOCOL,
    'horizons': list(MOE_HORIZONS), 'threshold': .25,
    'comparison': 'strict_less_than', 'required_consecutive_passes': 10,
    'loss_source': 'completed_optimizer_update.total_player_enemy_combat_router_loss',
    'task_weights': {'player': 1., 'enemy': 1., 'combat': 1.},
    'local_navigation_weight': .1,
    'router_role_weight': .01, 'router_balance_weight': .001,
    'combat_reduction': 'sum_of_fixed_semantic_components_with_whole_query_denominators',
    'microbatch_size': 64, 'gradient_accumulation_steps': 4,
    'effective_batch_size': 256, 'tbptt': 16,
    'combat_branch_max_horizon': 16, 'forced_long_replay': False,
    'late_own_prefix_actors': 8, 'late_own_prefix_weight': .25,
    'latent_cycle': ['argmax', 'argmax', 'sample', 'argmax'],
    'ema_normalization': False, 'posterior_reconstruction_KL': False,
    'validation_horizon': 450, 'validation_every': 250, 'preview_every': 10,
    'source_future_feedback': False, 'optimizer_reset': False,
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def moe_training_plan(state, *, source_checkpoint, source_checkpoint_sha256,
                      architecture_migration_sha256, data_roots=None, source_identities=None):
    step = state.get('global_training_step')
    if (type(step) is not int or step < 0 or not source_checkpoint
            or not _sha(source_checkpoint_sha256) or not _sha(architecture_migration_sha256)):
        raise ValueError('MoE phase requires a complete pinned migrated optimizer boundary')
    return {'version': 1, 'kind': MOE_PROTOCOL, 'settings': deepcopy(SETTINGS),
            'source_checkpoint': str(source_checkpoint),
            'source_checkpoint_sha256': source_checkpoint_sha256,
            'architecture_migration_sha256': architecture_migration_sha256,
            'data_roots': deepcopy({} if data_roots is None else data_roots),
            'source_identities': deepcopy({} if source_identities is None else source_identities),
            'start_optimizer_step': step}


def _validate_gate(gate, step):
    if gate.get('main_ladder_revision')is not None:
        from .rssm_main_ladder_revision import validate_gate
        validate_gate(gate,step)
        return
    from .rssm_moe_budget import (validate_policy, validate_budget_advance, validate_completion,
        validate_loss_advance, loss_gate_settings, SHORT_CAP_PROTOCOL, H450_GOAL_PROTOCOL, stage_requires_full_budget)
    from .rssm_moe_batch_revision import validate_moe_batch_accounting
    start, updates = gate.get('start_optimizer_step'), gate.get('updates')
    if (type(start) is not int or type(updates) is not int or updates < 0
            or start + updates != step or gate.get('last_optimizer_step') != step
            or gate.get('settings') != SETTINGS):
        raise ValueError('Saved MoE phase optimizer clock or fixed settings changed')
    validate_policy(gate,step)
    threshold, required = loss_gate_settings(gate)
    index = gate.get('stage_index')
    counts, history = gate.get('horizon_update_counts', {}), gate.get('advance_history', [])
    if (type(index) is not int or not 0 <= index < len(MOE_HORIZONS)
            or gate.get('current_horizon') != MOE_HORIZONS[index]
            or gate.get('next_horizon') != MOE_HORIZONS[index]
            or set(counts) != {str(h) for h in MOE_HORIZONS}
            or any(type(n) is not int or n < 0 for n in counts.values())
            or sum(counts.values()) != updates or len(history) != index
            or gate.get('gate_advances') != index
            or any(counts[str(h)] for h in MOE_HORIZONS[index + 1:])):
        raise ValueError('Saved MoE stage path or completed-horizon counts changed')
    previous_step = start
    for i, entry in enumerate(history):
        loss = entry.get('train_loss')
        budget_advance=entry.get('advance_reason')=='stage_budget'
        valid_reason=(validate_budget_advance(gate,entry,i) if budget_advance else
                      validate_loss_advance(gate,entry))
        if (entry.get('from_horizon') != MOE_HORIZONS[i]
                or entry.get('to_horizon') != MOE_HORIZONS[i + 1]
                or type(entry.get('step')) is not int or not previous_step < entry['step'] <= step
                or type(loss) not in (float, int) or not math.isfinite(loss) or loss<0
                or type(entry.get('loss_gate_eligible', True)) is not bool
                or not budget_advance and not entry.get('loss_gate_eligible', True)
                or not valid_reason
                or type(entry.get('completed_updates_at_previous_stage')) is not int
                or entry['completed_updates_at_previous_stage'] < max(1, entry.get('qualifying_streak', 0))
                or counts[str(MOE_HORIZONS[i])] != entry['completed_updates_at_previous_stage']):
            raise ValueError('Saved MoE advance lacks its versioned qualifying losses or pinned stage budget')
        previous_step = entry['step']
    stage_updates = updates - sum(row['completed_updates_at_previous_stage'] for row in history)
    streak = gate.get('consecutive_below_threshold')
    mandatory = stage_requires_full_budget(index, gate.get('advance_policy'))
    main_examples, main_transitions = validate_moe_batch_accounting(gate, step)
    if (gate.get('stage_updates') != stage_updates or counts[str(MOE_HORIZONS[index])] != stage_updates
            or type(streak) is not int or not 0 <= streak <= (required if mandatory or index == len(MOE_HORIZONS) - 1 else required-1)
            or streak > stage_updates
            or gate.get('goal_met') is not (not mandatory and index == len(MOE_HORIZONS) - 1 and streak == required)
            or gate.get('main_examples') != main_examples
            or gate.get('main_transitions') != main_transitions):
        raise ValueError('Saved MoE streak, goal or actual transition accounting changed')
    if updates:
        loss = gate.get('last_train_loss')
        eligible = gate.get('last_loss_gate_eligible', True)
        policy = gate.get('advance_policy') or {}
        activation = (policy.get('kind') == SHORT_CAP_PROTOCOL
                      or policy.get('kind')==H450_GOAL_PROTOCOL and gate['current_horizon']==450) and policy.get('updates') == 0
        decision_threshold,_ = loss_gate_settings(gate,horizon=gate.get('last_training_horizon',gate['current_horizon']))
        if (type(loss) not in (float, int) or not math.isfinite(loss) or loss < 0
                or type(eligible) is not bool
                or not activation and (gate.get('last_gate_passed') is not (eligible and loss < decision_threshold)
                or (not eligible or loss >= decision_threshold) and streak != 0
                or eligible and loss < decision_threshold and not gate.get('last_stage_advanced') and streak == 0)):
            raise ValueError('Saved MoE total-loss decision changed')
    validate_completion(gate)


def bind_moe_training(state, plan, *, path, sha256, loaded_checkpoint_sha256):
    path = Path(path)
    raw = path.read_bytes()
    expected = moe_training_plan({'global_training_step': plan.get('start_optimizer_step')},
        source_checkpoint=plan.get('source_checkpoint'),
        source_checkpoint_sha256=plan.get('source_checkpoint_sha256'),
        architecture_migration_sha256=plan.get('architecture_migration_sha256'),
        data_roots=plan.get('data_roots'), source_identities=plan.get('source_identities'))
    if (plan != expected or not _sha(sha256) or hashlib.sha256(raw).hexdigest() != sha256
            or json.loads(raw.decode('utf-8')) != plan):
        raise ValueError('MoE immutable phase plan, source or total-loss settings changed')
    step = state.get('global_training_step')
    if type(step) is not int or step < plan['start_optimizer_step']:
        raise ValueError('MoE phase cannot precede its complete optimizer boundary')
    q = state.setdefault('rssm_training', {})
    gate = q.get('moe_training')
    if gate is None:
        if step != plan['start_optimizer_step'] or loaded_checkpoint_sha256 != plan['source_checkpoint_sha256']:
            raise ValueError('New MoE phase must load its exact full migrated source checkpoint')
        state.setdefault('moe_training_history', []).append({
            'last_step': step, 'rssm_training': deepcopy(q),
            'objective_protocol': state.get('objective_protocol'),
            'horizon_curriculum_protocol': state.get('horizon_curriculum_protocol'),
            'training_protocol': deepcopy(state.get('training_protocol', {}))})
        gate = {'version': 1, 'objective_protocol': MOE_PROTOCOL,
            'plan': str(path.resolve()), 'sha256': sha256, 'settings': deepcopy(SETTINGS),
            'source_checkpoint_sha256': plan['source_checkpoint_sha256'],
            'architecture_migration_sha256': plan['architecture_migration_sha256'],
            'start_optimizer_step': step, 'last_optimizer_step': step, 'updates': 0,
            'stage_index': 0, 'current_horizon': 4, 'next_horizon': 4,
            'consecutive_below_threshold': 0, 'stage_updates': 0,
            'gate_advances': 0, 'advance_history': [],
            'horizon_update_counts': {str(h): 0 for h in MOE_HORIZONS},
            'main_examples': 0, 'main_transitions': 0,
            'last_train_loss': None, 'last_gate_passed': False,
            'last_stage_advanced': False, 'goal_met': False}
        q['moe_training'] = gate
        state.update(loss=None, latest_metrics=None, loss_groups=None)
    elif (gate.get('sha256') != sha256 or gate.get('source_checkpoint_sha256') != plan['source_checkpoint_sha256']
            or gate.get('architecture_migration_sha256') != plan['architecture_migration_sha256']
            or gate.get('start_optimizer_step') != plan['start_optimizer_step']):
        raise ValueError('Saved MoE phase source identity changed')
    _validate_gate(gate, step)
    state['objective_protocol'] = MOE_PROTOCOL
    state['loss_protocol'] = MOE_PROTOCOL
    state['horizon_curriculum_protocol'] = MOE_PROTOCOL
    q.update(objective_protocol=MOE_PROTOCOL, index_reservation_only=False,
             legacy_phase_ledgers_history_only=True)
    protocol = state.setdefault('training_protocol', {})
    protocol['moe_training'] = {'plan_sha256': sha256, **deepcopy(SETTINGS)}
    protocol.update(auxiliary_source_training_active=True, combat_head_training_active=True,
        posterior_training_active=False, memory_overshooting_active=False,
        kinematic_training_active=True, index_reservation_only=False,
        combat_branch_training_scope='three_tasks_exact_own_priors_current_H',
        forced_long_replay_active=False, ema_normalization=False,
        legacy_phase_ledgers_history_only=True,
        active_task_heads=['player_navigation', 'enemy_navigation', 'combat'])
    return gate


def moe_training_horizon(state):
    gate = state['rssm_training']['moe_training']
    _validate_gate(gate, state['global_training_step'])
    return gate['current_horizon']


def commit_moe_update(state, *, step, horizon, train_loss, microbatches=4,
                      microbatch_size=64, effective_batch_size=256, loss_gate_eligible=True):
    gate = state['rssm_training']['moe_training']
    if gate.get('main_ladder_revision')is not None:
        from .rssm_main_ladder_revision import commit_main_ladder_update
        return commit_main_ladder_update(state,step=step,horizon=horizon,train_loss=train_loss,
            microbatches=microbatches,microbatch_size=microbatch_size,effective_batch_size=effective_batch_size,
            loss_gate_eligible=loss_gate_eligible)
    from .rssm_moe_budget import (curriculum_completion, update_completion, loss_gate_settings,
                                 stage_requires_full_budget, H450_GOAL_PROTOCOL)
    from .rssm_moe_batch_revision import batch_settings_for_gate, commit_moe_batch_revision
    batch = batch_settings_for_gate(gate)
    if curriculum_completion(gate) is not None:
        raise ValueError('Completed flat-cap MoE curriculum cannot perform another optimizer commit')
    if (type(step) is not int or step != state.get('global_training_step')
            or step != gate['last_optimizer_step'] + 1
            or type(horizon) is not int or horizon != gate['current_horizon']
            or any(type(v) is not int for v in (microbatches, microbatch_size, effective_batch_size))
            or (microbatches, microbatch_size, effective_batch_size) !=
                (batch['gradient_accumulation_steps'], batch['microbatch_size'], batch['effective_batch_size'])
            or type(train_loss) not in (float, int) or not math.isfinite(train_loss) or train_loss < 0
            or type(loss_gate_eligible) is not bool):
        raise ValueError('MoE gate requires one finite complete pinned-batch total-loss optimizer commit')
    old = gate['stage_index']
    final_goal_policy=(gate.get('advance_policy') or {}).get('kind')==H450_GOAL_PROTOCOL
    actual_source_eligible=loss_gate_eligible
    if final_goal_policy and horizon==450:
        # This user criterion measures the actual TOTAL; Source-support
        # diagnostics remain separate and do not silently add a requirement.
        loss_gate_eligible=True
    threshold, required = loss_gate_settings(gate)
    passed = loss_gate_eligible and train_loss < threshold
    streak = min(required, gate['consecutive_below_threshold'] + 1) if passed else 0
    gate['updates'] += 1; gate['last_optimizer_step'] = step
    gate['horizon_update_counts'][str(horizon)] += 1
    gate['main_examples'] += effective_batch_size
    gate['main_transitions'] += horizon * effective_batch_size
    gate['stage_updates'] += 1
    policy=gate.get('advance_policy')
    budget_passed=False
    if policy is not None:
        from .rssm_moe_budget import stage_budget
        policy['updates']+=1;policy['last_optimizer_step']=step
        cap=stage_budget(old, policy)
        budget_passed=cap is not None and gate['stage_updates']>=cap
    mandatory = stage_requires_full_budget(old, policy)
    loss_passed = streak == required and not mandatory
    advanced = (loss_passed or budget_passed) and old < len(MOE_HORIZONS) - 1
    if advanced:
        gate['advance_history'].append({'step': step, 'train_loss': float(train_loss),
            'from_horizon': horizon, 'to_horizon': MOE_HORIZONS[old + 1],
            'qualifying_streak': streak, 'completed_updates_at_previous_stage': gate['stage_updates'],
            **({'loss_gate_eligible': False} if not loss_gate_eligible else {}),
            **({'advance_reason':'loss_streak' if loss_passed else 'stage_budget',
                'stage_budget':stage_budget(old, policy),'advance_policy_sha256':policy['sha256']}if policy else{})})
        gate['stage_index'] += 1; gate['gate_advances'] += 1
        gate['stage_updates'] = 0; streak = 0
    gate.update(current_horizon=MOE_HORIZONS[gate['stage_index']],
        next_horizon=MOE_HORIZONS[gate['stage_index']], consecutive_below_threshold=streak,
        last_train_loss=float(train_loss), last_training_horizon=horizon,
        last_gate_passed=passed, last_stage_advanced=advanced,
        goal_met=not mandatory and gate['stage_index'] == len(MOE_HORIZONS) - 1 and streak == required)
    if not loss_gate_eligible or 'last_loss_gate_eligible' in gate:
        gate['last_loss_gate_eligible'] = loss_gate_eligible
    if policy is not None:
        gate['last_advance_reason']=gate['advance_history'][-1]['advance_reason'] if advanced else None
    if final_goal_policy:
        from .rssm_h450_goal import commit_h450_goal
        gate['final_horizon_goal']=commit_h450_goal(gate['final_horizon_goal'],step=step,
            horizon=horizon,total_loss=train_loss)
        gate['last_source_support_gate_eligible']=actual_source_eligible
    update_completion(gate)
    commit_moe_batch_revision(state, step=step, horizon=horizon)
    _validate_gate(gate, step)
    return gate
