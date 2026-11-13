"""Three-task own-prior MoE learning, with an explicit total for its gate.

No posterior, future image encoder, neural renderer, EMA, optimizer or scheduler
is run here. Counterfactual Source labels are queried only after each own branch
has finished; they never replace a neural state or its recurrent memory.
"""
from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
import math

import numpy as np
import torch
from torch.nn import functional as F

from .gradient_safety import finite_gradient_norm
from .rssm_coordinate_gradients import (COORDINATE_GRADIENT_PROTOCOL,
    normalize_coordinate_gradient_protocol, integrate_coordinate_gradients)
from .rssm_drift_only_objective import _source_masks, _xy_errors
from .rssm_objectives import _batch_data, _observed_count, _initialize, _valid_result
from .rssm_moe_protocol import MOE_PROTOCOL, MOE_HORIZONS, SETTINGS


COMBAT_COMPONENTS = ('combat_events', 'combat_target', 'combat_attack',
    'combat_damage_npc', 'combat_damage_player', 'combat_death_npc', 'combat_spawn')


def _router_terms(result):
    diagnostics = result.get('rssm_fusion_moe')
    if not isinstance(diagnostics, dict):
        raise ValueError('Three-task objective requires the active shared Transformer MoE')
    result = {}
    for name in ('router_role_loss', 'router_balance_loss'):
        value = diagnostics.get(name)
        if not torch.is_tensor(value) or value.numel() != 1 or not bool(torch.isfinite(value)):
            raise ValueError('Missing finite differentiable MoE router diagnostic: ' + name)
        result[name] = value.float()
    return result


def _direct_task_terms(results, previous, actions, grid, *, simulation_hz):
    """All branch ticks, exact conditional Source labels and global denominators."""
    from .rssm_combat_branches import _source_step_at_own_state
    from .rssm_combat_objectives import prepare_combat_supervision, combat_supervised_terms
    from .rssm_kinematic_losses import KINEMATIC_PROTOCOL_V1
    from .motion_recovery_training import pass_fraction_targets, balanced_pass_weights
    count, horizon = actions.shape[:2]
    if not 1 <= horizon <= 16 or count * horizon > 1024:
        raise ValueError('MoE direct combat branch exceeds its bounded64x16 query budget')
    poses = torch.stack(previous, 1).detach().float().cpu().numpy()
    controls, maps = actions.detach().float().cpu().numpy(), grid.detach().cpu().numpy()
    selected, labels, skipped = [], [], Counter()
    spawn_anchors = {str(index): 0 for index in range(8)}
    spawn_anchors['fallback'] = 0
    from .spawn import SPAWN_ANCHORS
    per_tick = [dict(valid=0, hit=0, attack=0, spawn=0) for _ in range(horizon)]
    for tick in range(horizon):
        for actor in range(count):
            try:
                label = _source_step_at_own_state(poses[actor, tick], controls[actor, tick],
                    maps[actor], {'canonical_atol': 2e-7}, 1 / simulation_hz)
            except ValueError as error:
                skipped[str(error)] += 1
                continue
            selected.append((tick, actor)); labels.append(label)
            row = per_tick[tick]; row['valid'] += 1
            row['hit'] += int(label['hit'] >= 0); row['attack'] += int(label['attack'][0] > .5)
            revived = (not (label['previous'][13] >= .5 and label['previous'][12] > 0)
                       and label['target'][13] >= .5 and label['target'][12] > 0)
            row['spawn'] += int(revived)
            if revived:
                center = np.asarray(label['target'][8:10])*16
                index = next((i for i, cell in enumerate(SPAWN_ANCHORS)
                    if np.allclose(center, np.asarray(cell)+.5, rtol=0, atol=2e-6)), None)
                spawn_anchors[str(index) if index is not None else 'fallback'] += 1
    if not selected:
        raise ValueError('MoE combat branch has no legal actual-own-state Source supervision')
    device = actions.device
    def tensor(values, dtype=torch.float32):
        return torch.as_tensor(np.asarray(values), dtype=dtype, device=device).detach()
    old, target = tensor([r['previous'] for r in labels]), tensor([r['target'] for r in labels])
    control = torch.stack([actions[actor, tick] for tick, actor in selected]).detach()
    batch = {'states': torch.stack((old, target), 1), 'target_states': target[:, None],
        'actions': control[:, None], 'events': tensor([r['events'] for r in labels])[:, None],
        'hit_ids': tensor([r['hit'] for r in labels], torch.long)[:, None],
        'attack_flags': tensor([r['attack'] for r in labels])[:, None]}
    fields = ('state', 'event_logits', 'target_logits', 'attack_logits', 'shot', 'respawn_flags',
              'player_pass_logits', 'spawn_logits', 'spawn_anchor_logits', 'spawn_fallback_logits')
    prior = {key: torch.cat([results[tick][key][actor:actor + 1] for tick, actor in selected])
             for key in fields if key in results[0]}
    supervision = prepare_combat_supervision(batch, context=1, simulation_hz=simulation_hz,
        npc_motion_weight=1., target_supervision='shot_distance_balanced',
        kinematic_protocol=KINEMATIC_PROTOCOL_V1)
    terms, diagnostics = combat_supervised_terms(prior, old, target, control, supervision=supervision, index=0)
    fraction, active, _ = pass_fraction_targets(old, target, control, dt=1 / simulation_hz)
    weights = balanced_pass_weights(fraction, active)
    pass_loss = (F.binary_cross_entropy_with_logits(prior['player_pass_logits'].float(),
                                                   fraction, reduction='none') * weights).sum()
    player = .5 * (pass_loss + terms['combat_kinematic_player_axes'])
    enemy = (terms['combat_npc_motion'] + terms['combat_kinematic_npc_speed']
             + terms['combat_kinematic_npc_direction']) / 3
    present = {'combat_events': True, 'combat_target': bool(supervision['target_shot'].any()),
        'combat_attack': bool(supervision['attack_positive_count'] + supervision['attack_negative_count']),
        'combat_damage_npc': bool(supervision['old_alive_count']),
        'combat_damage_player': bool(supervision['player_alive_count']),
        'combat_death_npc': bool(supervision['killable_count']), 'combat_spawn': bool(supervision['spawn_count'])}
    combat = sum(terms[name] for name in COMBAT_COMPONENTS)
    metrics = {'direct_queries_attempted': count * horizon, 'direct_queries_valid': len(selected),
        'direct_invalid_skips': dict(skipped), 'direct_per_tick_counts': per_tick,
        'direct_spawn_anchor_counts': spawn_anchors,
        'direct_source_labels_after_own_rollout': True, 'source_future_feedback': False,
        'direct_present_combat_components': [key for key in COMBAT_COMPONENTS if present[key]],
        'player_pass_loss': float(pass_loss.detach()),
        **{key: float(value.detach()) for key, value in terms.items()},
        **{key: float(value) for key, value in diagnostics.items()}}
    return {'player': player, 'enemy': enemy, 'combat': combat}, metrics


def _own_branch_backward(model, initial, memory, grid, actions, *, amp, weight=1., check_gradients=True):
    autocast = nullcontext if amp is None else amp
    with torch.no_grad(), autocast():
        features = model.encode_map(grid).detach()
    current, shape = initial.detach(), tuple(memory.shape)
    memory = memory.detach()
    previous, results = [], []
    with autocast():
        for tick in range(actions.shape[1]):
            previous.append(current.detach())
            result = model.transition(current.detach(), actions[:, tick], features, memory)
            _valid_result(result, shape, 'MoE direct own branch')
            _router_terms(result)
            results.append(result); current, memory = result['state'], result['memory']
        tasks, metrics = _direct_task_terms(results, previous, actions, grid,
                                             simulation_hz=model.config.simulation_hz)
        value = weight * (SETTINGS['local_navigation_weight'] * (tasks['player'] + tasks['enemy'])
                          + tasks['combat'])
    if not bool(torch.isfinite(value)):
        raise FloatingPointError('Nonfinite three-task MoE direct own-branch loss')
    value.backward()
    if check_gradients:
        finite_gradient_norm(model.named_parameters(), context='Three-task MoE own branch')
    return {'player': weight * SETTINGS['local_navigation_weight'] * float(tasks['player'].detach()),
            'enemy': weight * SETTINGS['local_navigation_weight'] * float(tasks['enemy'].detach()),
            'combat': weight * float(tasks['combat'].detach()),
            'horizon': int(actions.shape[1]), 'actors': len(initial),
            'memory_resets_after_initialization': 0, **metrics}


def moe_dynamics_backward(model, batch, auxiliary_batch, *, context=1, observed_context=None,
                          auxiliary_context=1, expected_horizon=None, chunk_size=16,
                          amp=None, progress_callback=None,
                          coordinate_gradient_protocol=COORDINATE_GRADIENT_PROTOCOL,
                          check_gradients=True, main_player_weight=1., main_router_weight=1., main_enemy_weight=1.):
    """Backpropagate three tasks plus router regularization; never step Adam.

    Router losses are per-step Bx3 means, divided by full H before adding TBPTT
    chunks. The total includes all tasks and regularization, and is the sole
    curriculum gate loss. Late combat uses an actual saved main prefix h/z.
    """
    for name,value in (('main_player_weight',main_player_weight),('main_router_weight',main_router_weight),('main_enemy_weight',main_enemy_weight)):
        if isinstance(value,bool)or not isinstance(value,(int,float))or not math.isfinite(value)or not 0<value<=1:
            raise ValueError('MoE '+name+' must be finite in(0,1]')
    if type(chunk_size) is not int or chunk_size != 16:
        raise ValueError('MoE training preserves16-step TBPTT')
    coordinate = normalize_coordinate_gradient_protocol(coordinate_gradient_protocol)
    if coordinate is None:
        raise ValueError('MoE navigation retains integrated XY gradients')
    states, frames, actions, target, grid = _batch_data(batch, context)
    observed = _observed_count(context, observed_context)
    horizon = actions.shape[1]
    if horizon not in MOE_HORIZONS or expected_horizon is not None and expected_horizon != horizon:
        raise ValueError('MoE training must use the exact current4..450 stage horizon')
    autocast = nullcontext if amp is None else amp
    with torch.no_grad(), autocast():
        features = model.encode_map(grid).detach()
    with autocast():
        memory = _initialize(model, frames, states, features, context, observed)
    current = states[:, context - 1]
    previous_source = torch.cat((current[:, None], target[:, :-1]), 1)
    player_mask, enemy_mask = _source_masks(previous_source, target)
    masks = {'player': player_mask, 'enemy': enemy_mask}
    denominator = {name: (2 * mask.sum()).clamp_min(1) for name, mask in masks.items()}
    totals = dict.fromkeys(('player', 'enemy', 'combat', 'router_role', 'router_balance'), 0.)
    late_start = horizon - min(horizon, 16)
    late_prefix = None
    shape = tuple(memory.shape)
    for first in range(0, horizon, 16):
        last = min(horizon, first + 16)
        if first:
            current, memory = current.detach(), memory.detach()
        predictions, role, balance = [], [], []
        with autocast():
            for tick in range(first, last):
                if tick == late_start:
                    count = min(len(states), SETTINGS['late_own_prefix_actors'])
                    late_prefix = (current[:count].detach(), memory[:count].detach(),
                                   grid[:count], actions[:count, tick:])
                result = model.transition(current.detach(), actions[:, tick], features, memory)
                _valid_result(result, shape, 'Three-task MoE own prior')
                routers = _router_terms(result)
                role.append(routers['router_role_loss'] / horizon)
                balance.append(routers['router_balance_loss'] / horizon)
                result = integrate_coordinate_gradients(current, result, coordinate)
                current, memory = result['state'], result['memory']
                predictions.append(current)
            errors = _xy_errors(torch.stack(predictions, 1), target[:, first:last])
            parts = {name: (value * masks[name][:, first:last]).sum() / denominator[name]
                     for name, value in errors.items()}
            parts['player'] = parts['player'] * main_player_weight
            parts['enemy'] = parts['enemy'] * main_enemy_weight
            parts['router_role'] = main_router_weight * (SETTINGS['router_role_weight'] * sum(role))
            parts['router_balance'] = main_router_weight * (SETTINGS['router_balance_weight'] * sum(balance))
            loss = sum(parts.values())
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite three-task MoE autonomous objective')
        loss.backward()
        if check_gradients:
            finite_gradient_norm(model.named_parameters(), context='Three-task MoE own prior')
        for name, value in parts.items():
            totals[name] += float(value.detach())
        if progress_callback:
            progress_callback(last, horizon)
    main_totals=dict(totals)
    aux_states, aux_frames, aux_actions, _, aux_grid = _batch_data(auxiliary_batch, auxiliary_context)
    expected_aux = min(horizon, SETTINGS['combat_branch_max_horizon'])
    if aux_actions.shape[1] != expected_aux or len(aux_states) > 64:
        raise ValueError('MoE balanced combat auxiliary must match min(currentH,16), at most64 actors')
    with torch.no_grad(), autocast():
        aux_features = model.encode_map(aux_grid).detach()
    with autocast():
        aux_memory = _initialize(model, aux_frames, aux_states, aux_features,
                                 auxiliary_context, auxiliary_context)
    auxiliary = _own_branch_backward(model, aux_states[:, auxiliary_context - 1], aux_memory,
        aux_grid, aux_actions, amp=amp, check_gradients=check_gradients)
    late = _own_branch_backward(model, *late_prefix, amp=amp,
        weight=SETTINGS['late_own_prefix_weight'], check_gradients=check_gradients)
    for name in ('player', 'enemy', 'combat'):
        totals[name] += auxiliary[name] + late[name]
    metrics = {'loss_protocol': MOE_PROTOCOL, 'objective_protocol': MOE_PROTOCOL,
        'train_loss': sum(totals.values()), 'train_loss_without_kl_floor': sum(totals.values()),
        'moe_player_loss': totals['player'], 'moe_enemy_loss': totals['enemy'],
        'moe_combat_loss': totals['combat'],
        'moe_router_role_loss': totals['router_role'], 'moe_router_balance_loss': totals['router_balance'],
        'moe_router_loss': totals['router_role'] + totals['router_balance'],
        'moe_main_horizon': horizon, 'rssm_prior_horizon': horizon,
        'rssm_prior_future_rgb_inputs': 0, 'rssm_source_posterior_calls': 0,
        'rssm_renderer_calls': 0, 'rssm_optimizer_steps': 0, 'rssm_ema_updates': 0,
        'rssm_kl_minimum_floor': 0., 'source_future_feedback': False,
        'moe_main_player_loss': main_totals['player'],'moe_main_enemy_loss':main_totals['enemy'],
        'moe_main_router_role_loss':main_totals['router_role'],'moe_main_router_balance_loss':main_totals['router_balance'],
        'moe_main_player_weight':float(main_player_weight),'moe_main_router_weight':float(main_router_weight),
        'moe_main_enemy_weight':float(main_enemy_weight),
        'moe_late_prefix_start': late_start, 'moe_auxiliary': auxiliary,
        'moe_late_own_prefix': late, 'moe_fixed_task_weights': dict(SETTINGS['task_weights'])}
    metrics.update(moe_auxiliary_direct_per_step_counts=auxiliary['direct_per_tick_counts'],
        moe_late_direct_per_step_counts=late['direct_per_tick_counts'],
        moe_auxiliary_spawn_anchor_counts=auxiliary['direct_spawn_anchor_counts'],
        moe_late_spawn_anchor_counts=late['direct_spawn_anchor_counts'])
    for name in COMBAT_COMPONENTS:
        metrics['moe_'+name+'_loss'] = auxiliary[name]+SETTINGS['late_own_prefix_weight']*late[name]
    metrics['loss_groups'] = summarize_moe_metrics([metrics])
    return metrics


def summarize_moe_metrics(reports, *, step=None):
    """Fixed task weights, no adaptive normalization; router is added once."""
    if not reports:
        raise ValueError('Cannot summarize an incomplete MoE optimizer update')
    fields = {'player': 'moe_player_loss', 'enemy': 'moe_enemy_loss', 'combat': 'moe_combat_loss',
              'router': 'moe_router_loss'}
    groups = {}
    for name, key in fields.items():
        values = [row[key] for row in reports]
        if any(type(value) not in (float, int) or not math.isfinite(value) or value < 0 for value in values):
            raise ValueError('MoE loss report contains an invalid task value')
        value = sum(values) / len(values)
        groups[name] = {'raw': value, 'normalized': value, 'weighted': value,
                        'scale': 1., 'weight': 1., 'kl_floor': 0.}
    total = sum(row['weighted'] for row in groups.values())
    groups.update(total=total, kl_floor=0., normalization='fixed_three_task_no_EMA',
        loss_protocol=MOE_PROTOCOL, step=step,
        interaction=dict(groups['combat']), interaction_is_combat_alias_not_added_twice=True)
    return groups


def build_moe_combat_sampler(data, horizon, *, cache_path=None):
    """Fresh event windows for the actual4/8/16 auxiliary horizon, TRAIN only."""
    from .rssm_spawn_curriculum import RecoveryCombatSampler
    from .rssm_moe_protocol import _digest
    if type(horizon) is not int or horizon not in MOE_HORIZONS or data.split != 'train':
        raise ValueError('MoE auxiliary sampler needs a current stage and explicit TRAIN data')
    actual = min(horizon, 16)
    sampler = RecoveryCombatSampler(data, context=1, horizon=actual, cache_path=cache_path,
        target_distance_balanced=True, attack_positive_balanced=True)
    sampler.moe_identity = {'horizon': actual, 'context': 1,
        'source_fingerprint': sampler.base.base.fingerprint,
        'summary': sampler.summary(batch_size=64), 'split': 'train'}
    sampler.moe_identity['fingerprint'] = _digest(sampler.moe_identity)
    return sampler


def sample_moe_combat_batch(sampler, data, rng, *, horizon, batch_size=64):
    """Return (ordinary Source batch, native-JSON quota/row receipt)."""
    import copy
    import json
    if (data.split != 'train' or type(batch_size) is not int or batch_size != 64
            or type(horizon) is not int or horizon not in MOE_HORIZONS
            or sampler.horizon != min(horizon, 16) or sampler.context != 1):
        raise ValueError('MoE event quotas must cover the actual64x min(currentH,16) Source tape')
    pairs = sampler.sample(64, rng)
    batch = data.batch(64, context=1, horizon=sampler.horizon, indices=pairs)
    receipt = {**copy.deepcopy(sampler.last_sample),
        'moe_horizon': horizon, 'auxiliary_horizon': sampler.horizon,
        'source_identity': copy.deepcopy(sampler.moe_identity),
        'pairs': pairs.tolist(), 'index_reservation_only': False,
        'used_aux_fragments': 64, 'training_cases_used': True}
    json.dumps(receipt, allow_nan=False)
    return batch, receipt
