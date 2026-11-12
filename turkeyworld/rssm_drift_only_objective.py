"""Temporary XY-only training with own state/h-z and no auxiliary objectives.

Only absolute player/NPC positions enter a differentiable loss. Future Source
poses and counterfactual Source branches are labels, never neural conditions.
No posterior, decoder, KL, state-field, motion, contact or combat loss runs.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch
from torch.nn import functional as F

from .gradient_safety import finite_gradient_norm
from .rssm_coordinate_gradients import (COORDINATE_GRADIENT_PROTOCOL,
    normalize_coordinate_gradient_protocol, integrate_coordinate_gradients)
from .rssm_late_prefix_recovery import normalize_late_prefix_protocol, continue_own_prefix_candidates
from .rssm_loss_gated_curriculum import HORIZONS
from .rssm_objectives import _batch_data, _observed_count, _initialize, _valid_result


LOSS_PROTOCOL = 'rssm_coordinate_drift_only_v1'
NORMALIZATION = 'none_drift_only_fixed_weights'
DISABLED_OBJECTIVES = ('state_fields', 'posterior', 'reconstruction', 'KL', 'memory_overshooting',
    'velocity', 'heading', 'pass_BCE', 'walls', 'combat', 'spawn', 'events')


def _source_masks(previous, target):
    player = (previous[..., 4] > 0) & (target[..., 4] > 0)
    old_npc = (previous[..., 12] > 0) & (previous[..., 13] >= .5)
    new_npc = (target[..., 12] > 0) & (target[..., 13] >= .5)
    return player, old_npc & new_npc


def _xy_errors(prediction, target):
    return {'player': F.smooth_l1_loss(prediction[..., :2].float()*16,
                target[..., :2].detach().float()*16, reduction='none').sum(-1),
            'enemy': F.smooth_l1_loss(prediction[..., 8:10].float()*16,
                target[..., 8:10].detach().float()*16, reduction='none').sum(-1)}


def drift_only_branches_backward(model, candidates, *, combat_branch_protocol,
                                 coordinate_gradient_protocol, late_prefix_protocol=None,
                                 amp=None, check_gradients=True):
    """Existing legal own-prefix selection, followed by XY-only16-step losses."""
    from .rssm_combat_branches import normalize_combat_branch_protocol, select_combat_branches
    protocol = normalize_combat_branch_protocol(combat_branch_protocol)
    if protocol is None:
        return {'loss': 0., 'player_xy_loss': 0., 'enemy_xy_loss': 0., 'branches': 0}
    if getattr(model.config, 'navigation_only', True):
        raise ValueError('Drift-only branches require the one-enemy profile')
    selected, selection = select_combat_branches(candidates, combat_branch_protocol=protocol,
        dt=1/model.config.simulation_hz,
        **({'late_prefix_protocol': late_prefix_protocol} if late_prefix_protocol is not None else {}))
    metrics = {**selection, 'branches': len(selected), 'loss': 0.,
        'player_xy_loss': 0., 'enemy_xy_loss': 0., 'xy_only': True,
        'new_observed_rgb_frames': 0, 'source_future_inputs': 0, 'memory_resets': 0,
        'state_resets': 0, 'optimizer_steps': 0, 'direct_source_queries': 0,
        'combat_loss_calls': 0, 'memory_provenance': 'actualownprefix'}
    if not selected:
        return metrics
    initial = torch.stack([row['item']['state'][row['actor']] for row in selected]).detach()
    memory = torch.stack([row['item']['memory'][row['actor']] for row in selected]).detach()
    actions = torch.stack([row['item']['actions'][row['actor'], :protocol['horizon']]
                           for row in selected]).detach()
    grid = torch.stack([row['item']['map'][row['actor']] for row in selected]).detach()
    # Source replay already happened after own-prefix capture in the selector.
    import numpy as np
    target = torch.as_tensor(np.stack([row['labels'] for row in selected]),
                             device=initial.device, dtype=torch.float32).detach()
    player, enemy = _source_masks(target[:, :-1], target[:, 1:])
    offsets = protocol['offsets']
    chosen_offsets = torch.tensor([value-1 for value in offsets], device=initial.device)
    masks = {'player': player, 'enemy': enemy}
    denominators = {name: (2*mask[:, chosen_offsets].sum()).clamp_min(1)
                    for name, mask in masks.items()}
    autocast = nullcontext if amp is None else amp
    with torch.no_grad(), autocast():
        features = model.encode_map(grid).detach()
    current, expected_shape = initial, tuple(memory.shape)
    terms = {}
    with autocast():
        for index in range(protocol['horizon']):
            result = model.transition(current.detach(), actions[:, index], features, memory)
            _valid_result(result, expected_shape, f'drift-only own branch {index+1}')
            result = integrate_coordinate_gradients(current, result, coordinate_gradient_protocol)
            current, memory = result['state'], result['memory']
            if index+1 in offsets:
                error = _xy_errors(current, target[:, index+1])
                for name in masks:
                    term = (error[name]*masks[name][:, index]).sum()/denominators[name]
                    terms[name] = terms.get(name, 0)+term
        weight = protocol['weight']*protocol['xy_weight']
        weighted = {name: weight*term for name, term in terms.items()}
        loss = sum(weighted.values())
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite XY-only own-prefix branch loss')
    loss.backward()
    if check_gradients:
        finite_gradient_norm(model.named_parameters(), context='XY-only own-prefix branches')
    metrics.update(loss=float(loss.detach()), player_xy_loss=float(weighted['player'].detach()),
        enemy_xy_loss=float(weighted['enemy'].detach()), source_xy_offsets=list(offsets),
        player_source_xy_count=int(player[:, chosen_offsets].sum()),
        npc_source_xy_count=int(enemy[:, chosen_offsets].sum()), xy_weight=weight)
    return metrics


def drift_only_backward(model, batch, *, context=1, observed_context=None,
                        expected_horizon=None, chunk_size=16, amp=None,
                        progress_callback=None, coordinate_gradient_protocol=COORDINATE_GRADIENT_PROTOCOL,
                        late_prefix_protocol=None, late_prefix_batch=None,
                        combat_branch_protocol=None, check_gradients=True):
    """Backpropagate own XY trajectories; never step Adam or any EMA.

    Main player and enemy weights are1. Each uses2 times its full-tape Source
    continuing-alive count. Exact Source respawn/death transitions are omitted;
    later lives and missing predicted actors remain supervised. Branch XY uses
    the explicit existing weight x xy_weight only; other protocol losses are
    deliberately neither computed nor backpropagated.
    """
    coordinate = normalize_coordinate_gradient_protocol(coordinate_gradient_protocol)
    if coordinate is None or chunk_size != 16 or type(chunk_size) is not int:
        raise ValueError('Drift-only training requires integrated XY and16-step TBPTT')
    states, frames, actions, target, grid = _batch_data(batch, context)
    observed = _observed_count(context, observed_context)
    horizon = actions.shape[1]
    if horizon not in HORIZONS or expected_horizon is not None and expected_horizon != horizon:
        raise ValueError('Drift-only main horizon must follow the approved16..450 curriculum path')
    late = normalize_late_prefix_protocol(late_prefix_protocol)
    if late_prefix_batch is not None and late is None:
        raise ValueError('Late-prefix controls require their explicit protocol')
    if late is not None and combat_branch_protocol is None:
        raise ValueError('Late prefixes require selected XY-only branches')
    if late is not None and horizon < late['scout_horizon']:
        scout_count = min(len(states), late['scout_batch_size'])
        if (not isinstance(late_prefix_batch, dict) or set(late_prefix_batch) != {'actions', 'map'}
                or late_prefix_batch['actions'].shape != (scout_count, late['scout_horizon'], 4)
                or late_prefix_batch['map'].shape != grid[:scout_count].shape
                or not torch.equal(late_prefix_batch['map'], grid[:scout_count])
                or not torch.equal(late_prefix_batch['actions'][:, :horizon].detach().float(), actions[:scout_count])):
            raise ValueError('Late-prefix controls/maps must match the exact first8 main pairs')
    from .rssm_combat_branches import normalize_combat_branch_protocol, capture_combat_candidate
    branch = normalize_combat_branch_protocol(combat_branch_protocol)
    if late is not None and branch['version'] != 2:
        raise ValueError('Late XY-only prefixes require branch protocol version2')
    autocast = nullcontext if amp is None else amp
    with torch.no_grad(), autocast():
        features = model.encode_map(grid).detach()
    with autocast():
        memory = _initialize(model, frames, states, features, context, observed)
    initial = states[:, context-1]
    previous_source = torch.cat([initial[:, None], target[:, :-1]], 1)
    player_mask, enemy_mask = _source_masks(previous_source, target)
    masks = {'player': player_mask, 'enemy': enemy_mask}
    denominators = {name: (2*mask.sum()).clamp_min(1) for name, mask in masks.items()}
    current, expected_shape = initial, tuple(memory.shape)
    totals, candidates, chunks = {'player': 0., 'enemy': 0.}, [], 0
    for first in range(0, horizon, chunk_size):
        last = min(horizon, first+chunk_size)
        if first:
            current, memory = current.detach(), memory.detach()
        own_state, own_memory = current.detach(), memory.detach()
        predictions = []
        with autocast():
            for index in range(first, last):
                result = model.transition(current.detach(), actions[:, index], features, memory)
                _valid_result(result, expected_shape, f'drift-only own prior {index+1}')
                result = integrate_coordinate_gradients(current, result, coordinate)
                current, memory = result['state'], result['memory']
                predictions.append(current)
            prediction = torch.stack(predictions, 1)
            errors = _xy_errors(prediction, target[:, first:last])
            parts = {name: (error*masks[name][:, first:last]).sum()/denominators[name]
                     for name, error in errors.items()}
            loss = sum(parts.values())
        if branch is not None:
            candidates.append(capture_combat_candidate(first, own_state, own_memory,
                actions[:, first:first+branch['horizon']], grid, prediction))
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite XY-only autonomous loss')
        loss.backward()
        chunks += 1
        if check_gradients:
            finite_gradient_norm(model.named_parameters(), context=f'XY-only own backward {last}/{horizon}')
        for name, value in parts.items():
            totals[name] += float(value.detach())
        if progress_callback:
            progress_callback(last, horizon)
    scout = {}
    if late is not None and horizon < late['scout_horizon']:
        extra, scout = continue_own_prefix_candidates(model, current[:scout_count], memory[:scout_count],
            features[:scout_count], grid[:scout_count], late_prefix_batch['actions'],
            start=horizon, late_prefix_protocol=late, amp=amp)
        candidates.extend(extra)
    branch_metrics = (drift_only_branches_backward(model, candidates,
        combat_branch_protocol=branch, coordinate_gradient_protocol=coordinate,
        late_prefix_protocol=late, amp=amp, check_gradients=check_gradients) if branch is not None else {})
    actor = {'player': totals['player']+branch_metrics.get('player_xy_loss', 0.),
             'enemy': totals['enemy']+branch_metrics.get('enemy_xy_loss', 0.), 'interaction': 0.}
    total = sum(actor.values())
    groups = {name: {'raw': value, 'normalized': value, 'weighted': value,
                    'scale': 1., 'weight': 1., 'effective_weight': 1., 'kl_floor': 0.,
                    'active': value > 0} for name, value in actor.items()}
    groups.update(total=total, kl_floor=0., normalization=NORMALIZATION, loss_protocol=LOSS_PROTOCOL)
    return {'train_loss': total, 'train_loss_without_kl_floor': total, 'loss_groups': groups,
        'loss_protocol': LOSS_PROTOCOL, 'rssm_drift_only': True,
        'rssm_xy_player': totals['player'], 'rssm_xy_npc': totals['enemy'],
        'rssm_main_loss': sum(totals.values()), 'rssm_kl_minimum_floor': 0.,
        'rssm_prior_horizon': horizon, 'rssm_prior_tbptt_chunks': chunks,
        'rssm_prior_context_frames': observed, 'rssm_prior_numerical_resets': 0,
        'rssm_prior_future_rgb_inputs': 0, 'rssm_posterior_feedback_to_prior': False,
        'rssm_source_posterior_calls': 0, 'rssm_decoder_calls': 0, 'rssm_renderer_calls': 0,
        'rssm_optimizer_steps': 0, 'rssm_ema_updates': 0, 'rssm_numeric_state_feedback': 'detached',
        'rssm_integrated_xy_gradients': True,
        'rssm_player_xy_supervised_transitions': int(player_mask.sum()),
        'rssm_npc_xy_supervised_transitions': int(enemy_mask.sum()),
        'rssm_disabled_objectives': list(DISABLED_OBJECTIVES),
        **{'rssm_late_prefix_scout_'+name: value for name, value in scout.items()},
        **{'rssm_combat_branch_'+name: value for name, value in branch_metrics.items()}}
