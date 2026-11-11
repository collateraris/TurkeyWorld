"""Source-dominant Dreamer-style RSSM learning without JEPA or an EMA target.

The prior-only trajectory completes first. A separate posterior stream may
observe Source RGB for representation/KL training, but its numerical state or
latent memory never corrects the autonomous trajectory. Optimizer ownership
and its one-update boundary remain with the caller.
"""
from __future__ import annotations

from contextlib import nullcontext
import math

import torch
from torch.nn import functional as F

from .gradient_safety import finite_gradient_norm
from .contact_objectives import (normalize_contact_protocol, prepare_contact_targets,
    differentiable_contact_terms, swept_player_wall_loss)
from .motion_recovery_training import (pass_fraction_targets, balanced_pass_weights,
    PASS_WEIGHT, RAW_DELTA_WEIGHT, SCALED_MOTION_WEIGHT, _local)
from .rssm import balanced_kl
from .spatial_entities import entity_motion_basis
from .structured_state import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
from .train import state_loss
from .trajectory_objectives import trajectory_state_losses


def _batch_data(batch, context):
    values = {name: batch[name].detach() for name in ('states', 'frames', 'actions', 'target_states', 'map')}
    states, frames, actions, targets, grid = (values[name] for name in values)
    if (not isinstance(context, int) or isinstance(context, bool) or context < 1
            or states.ndim != 3 or states.shape[-1] != STATE_DIM
            or actions.ndim != 3 or actions.shape[2] != 4 or actions.shape[1] < 1
            or len(states) != len(actions) or states.shape[1] < context+actions.shape[1]
            or frames.ndim != 5 or frames.shape[:2] != states.shape[:2] or frames.shape[2] != 3
            or targets.shape != (len(states), actions.shape[1], STATE_DIM) or grid.shape != (len(states), 16, 16)):
        raise ValueError('RSSM objectives require complete Source context/future/control/map sequences')
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise FloatingPointError('Nonfinite Source RSSM training inputs')
    return states.float(), frames, actions.float(), targets.float(), grid


def _observed_count(context, observed):
    value = context if observed is None else observed
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= context:
        raise ValueError('RSSM observed context must contain genuine1..context past frames')
    return value


def _initialize(model, frames, states, features, context, observed):
    return model.initialize_from_image(frames[:, context-observed:context],
                                       states[:, context-observed:context], features)


def _valid_result(result, expected_shape, where):
    if (result['state'].ndim != 2 or result['state'].shape[-1] != STATE_DIM
            or not torch.is_tensor(result['memory']) or tuple(result['memory'].shape) != expected_shape):
        raise ValueError(f'Invalid RSSM state/packed memory shape at {where}')
    for key in ('state', 'memory', 'rssm_prior_logits', 'rssm_posterior_logits', 'rssm_readout', 'features', 'rssm_reconstruction'):
        if key in result and not bool(torch.isfinite(result[key]).all()):
            raise FloatingPointError(f'Nonfinite RSSM {key} at {where}')


def _teacher_targets(teacher, states, frames, actions, grid, context, observed):
    modes = [(module, module.training) for module in teacher.modules()]
    rows = {name: [] for name in ('state', 'readout', 'features')}
    try:
        teacher.eval()
        with torch.no_grad():
            features = teacher.encode_map(grid)
            memory = _initialize(teacher, frames, states, features, context, observed)
            current = states[:, context-1]
            for index in range(actions.shape[1]):
                result = teacher.transition(current, actions[:, index], features, memory)
                current, memory = result['state'], result['memory']
                pooled = result['latent_prediction'].mean(1).to(teacher.vjepa_backend.readout_neck[1].weight.dtype)
                pooled = pooled+teacher.vjepa_backend.readout_neck(pooled)
                rows['state'].append(current.detach())
                rows['readout'].append(pooled.detach().float())
                rows['features'].append(result['features'].detach().float())
    finally:
        for module, mode in modes:
            module.training = mode
    return {name: torch.stack(values, 1) for name, values in rows.items()}


def _auxiliary_view(batch, context, horizon):
    return {**batch, 'states': batch['states'][:, :context+horizon],
        'frames': batch['frames'][:, :context+horizon], 'actions': batch['actions'][:, :horizon],
        'target_states': batch['target_states'][:, :horizon]}


def _kl_settings(model):
    config = getattr(model, 'config', None)
    return {'free_nats': getattr(config, 'rssm_free_nats', 1.),
        'dynamics_scale': getattr(config, 'rssm_dynamics_scale', 1.),
        'representation_scale': getattr(config, 'rssm_representation_scale', .1),
        'unimix': getattr(config, 'rssm_unimix', .01)}


def _posterior_backward(model, batch, *, context, observed, autocast, chunk_size, check_gradients,
                        contact_protocol=None):
    states, frames, actions, targets, grid = _batch_data(batch, context)
    batch_size, horizon = actions.shape[:2]
    if horizon > 16:
        raise ValueError('Separate RSSM Source posterior/motion supervision is bounded to16 steps')
    previous = states[:, context-1:context-1+horizon]
    labels, active, commands = pass_fraction_targets(previous.flatten(0, 1), targets.flatten(0, 1), actions.flatten(0, 1))
    labels, active, commands = (value.reshape(batch_size, horizon, 2) for value in (labels, active, commands))
    label_weights = balanced_pass_weights(labels, active)
    moving = (previous[:, :, 4] > 0) & actions[:, :, :2].ne(0).any(-1)
    denominator = (2*moving.sum()).clamp_min(1)
    totals = dict.fromkeys(('posterior_kl', 'posterior_reconstruction', 'posterior_state',
                           'pass_bce', 'raw_delta', 'scaled_motion', 'physical_motion', 'endpoint_margin', 'contact_wall'), 0.)
    contact = (prepare_contact_targets(previous, targets, actions, contact_protocol=contact_protocol)
               if contact_protocol is not None else None)
    kl_settings = _kl_settings(model)
    kl_metrics, resets, calls = {}, 0, 0
    free, blocked = active & (labels >= .9), active & (labels <= .1)
    false_stops = missed_blocks = 0
    physical_false_stops = physical_missed_blocks = 0
    physical_free = active & (labels >= .95)
    physical_blocked = active & (labels <= .05)
    velocity_error = 0.
    with torch.no_grad(), autocast():
        features = model.encode_map(grid).detach()
    with autocast():
        memory = _initialize(model, frames, states, features, context, observed)
    expected_shape = tuple(memory.shape)
    for first in range(0, horizon, chunk_size):
        last = min(horizon, first+chunk_size)
        if first: memory = memory.detach()
        with autocast():
            parts = {}
            for index in range(first, last):
                post = model.vjepa_backend.observe_step(previous[:, index], actions[:, index], features,
                    memory, frames[:, context+index])
                _valid_result(post, expected_shape, f'Source posterior {index+1}')
                if 'prior_result' not in post:
                    raise ValueError('Posterior observation must retain the actual pre-RGB prior_result')
                prior = post['prior_result']
                _valid_result(prior, expected_shape, f'Source pre-RGB prediction {index+1}')
                memory = post['memory']
                logits, raw = prior['player_pass_logits'].float(), prior['raw_player_delta'].float()
                kl, details = balanced_kl(post['rssm_posterior_logits'], post['rssm_prior_logits'], **kl_settings)
                rgb = post['rssm_reconstruction'].float()
                target_rgb = frames[:, context+index].float()
                if rgb.shape != target_rgb.shape:
                    raise ValueError('RSSM posterior reconstruction must match genuine Source native RGB')
                reconstruction = F.l1_loss(rgb, target_rgb)+.1*F.mse_loss(rgb, target_rgb)
                expected_delta = targets[:, index, :2]-previous[:, index, :2]
                accepted_delta = prior['state'][:, :2]-previous[:, index, :2]
                values = {'posterior_kl': kl/horizon, 'posterior_reconstruction': reconstruction/horizon,
                    'posterior_state': 2*state_loss(prior['state'], targets[:, index])/horizon}
                if contact_protocol is None:
                    basis = entity_motion_basis(previous[:, index], actions[:, index])
                    raw_loss = F.smooth_l1_loss(_local(raw*200, basis), _local(expected_delta*200, basis), reduction='none')
                    motion_loss = F.smooth_l1_loss(_local(accepted_delta*200, basis), _local(expected_delta*200, basis), reduction='none')
                    values.update(pass_bce=(F.binary_cross_entropy_with_logits(logits, labels[:, index], reduction='none')*label_weights[:, index]).sum(),
                        raw_delta=(raw_loss.sum(-1)*moving[:, index]).sum()/denominator,
                        scaled_motion=(motion_loss.sum(-1)*moving[:, index]).sum()/denominator)
                else:
                    values.update(differentiable_contact_terms(prior, previous[:, index], targets[:, index],
                        actions[:, index], grid, contact_protocol=contact_protocol, supervision=contact, index=index))
                for name, value in values.items(): parts[name] = parts.get(name, 0)+value
                for name, value in details.items(): kl_metrics[name] = kl_metrics.get(name, 0.)+float(value.detach())/horizon
                with torch.no_grad():
                    probability = logits.sigmoid()
                    false_stops += int(((probability < .5) & free[:, index]).sum())
                    missed_blocks += int(((probability >= .5) & blocked[:, index]).sum())
                    velocity_error += float(((accepted_delta-expected_delta).norm(dim=-1)*200*moving[:, index]).sum())
                    threshold = .2 if contact_protocol is None else contact_protocol['physical_error_fraction']
                    physical_false_stops += int(((accepted_delta.abs() < threshold*commands[:, index].abs()) & physical_free[:, index]).sum())
                    physical_missed_blocks += int(((accepted_delta.abs() > threshold*commands[:, index].abs()) & physical_blocked[:, index]).sum())
            loss = parts['posterior_kl']+parts['posterior_reconstruction']+parts['posterior_state']
            if contact_protocol is None:
                loss = loss+PASS_WEIGHT*parts['pass_bce']+RAW_DELTA_WEIGHT*parts['raw_delta']+SCALED_MOTION_WEIGHT*parts['scaled_motion']
            else:
                loss = loss+sum(contact_protocol[name+'_weight']*parts[name]
                    for name in ('pass_bce', 'physical_motion', 'endpoint_margin', 'contact_wall'))
        if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite RSSM posterior/motion objective')
        loss.backward(); calls += 1
        if check_gradients: finite_gradient_norm(model.named_parameters(), context=f'RSSM posterior backward {last}/{horizon}')
        for name, value in parts.items(): totals[name] += float(value.detach())
    total = totals['posterior_kl']+totals['posterior_reconstruction']+totals['posterior_state']
    if contact_protocol is None:
        total = total+PASS_WEIGHT*totals['pass_bce']+RAW_DELTA_WEIGHT*totals['raw_delta']+SCALED_MOTION_WEIGHT*totals['scaled_motion']
    else:
        total = total+sum(contact_protocol[name+'_weight']*totals[name]
            for name in ('pass_bce', 'physical_motion', 'endpoint_margin', 'contact_wall'))
    return {'loss': total, **totals, **{'kl_'+name: value for name, value in kl_metrics.items()},
        'source_windows': batch_size, 'source_observed_frames': observed, 'source_horizon': horizon,
        'posterior_rgb_frames': batch_size*horizon, 'source_queries_before_target_rgb': True,
        'source_backward_calls': calls, 'source_memory_numerical_resets': resets,
        'false_stop_rate': false_stops/max(int(free.sum()), 1),
        'missed_block_rate': missed_blocks/max(int(blocked.sum()), 1),
        'classifier_false_stop_rate': false_stops/max(int(free.sum()), 1),
        'classifier_missed_block_rate': missed_blocks/max(int(blocked.sum()), 1),
        'physical_false_stop_rate': physical_false_stops/max(int(physical_free.sum()), 1),
        'physical_missed_block_rate': physical_missed_blocks/max(int(physical_blocked.sum()), 1),
        'physical_error_fraction': .2 if contact_protocol is None else contact_protocol['physical_error_fraction'],
        'relative_velocity_error': velocity_error/max(int(moving.sum()), 1),
        'free_axes': int(free.sum()), 'blocked_axes': int(blocked.sum())}


def rssm_dynamics_backward(model, batch, *, phase='source', teacher=None, auxiliary_batch=None,
                           context=2, observed_context=None, auxiliary_context=None,
                           auxiliary_observed_context=None, chunk_size=16, teacher_weight=.1,
                           amp=None, progress_callback=None, expected_horizon=None, check_gradients=True,
                           contact_protocol=None):
    """Backpropagate own prior trajectory, then separate posterior/Source labels.

    Distillation uses1..16 own teacher/student transitions plus Source anchors.
    Source learning defaults to450 own transitions. No optimizer step, EMA,
    renderer call, GT state replacement or future RGB input to that trajectory.
    """
    if (phase not in {'source', 'distill'} or phase == 'distill' and teacher is None
            or phase == 'source' and teacher is not None or not 0 <= teacher_weight <= .25
            or not isinstance(chunk_size, int) or chunk_size < 1 or amp is not None and not callable(amp)):
        raise ValueError('Invalid RSSM phase/teacher/chunk/weight/AMP contract')
    states, frames, actions, targets, grid = _batch_data(batch, context)
    contact_protocol = normalize_contact_protocol(contact_protocol)
    observed = _observed_count(context, observed_context)
    horizon = actions.shape[1]
    expected = (450 if phase == 'source' else horizon) if expected_horizon is None else expected_horizon
    if horizon != expected or phase == 'distill' and horizon > 16:
        raise ValueError('RSSM source learning requires450 steps; distillation is bounded to16')
    autocast = nullcontext if amp is None else amp
    teacher_values = (_teacher_targets(teacher, states, frames, actions, grid, context, observed)
                      if phase == 'distill' else None)
    with torch.no_grad(), autocast(): features = model.encode_map(grid).detach()
    with autocast(): memory = _initialize(model, frames, states, features, context, observed)
    current, initial = states[:, context-1], states[:, context-1]
    expected_shape = tuple(memory.shape)
    source_npc = targets[:, :, GLOBAL_DIM:-2].reshape(len(states), horizon, ENTITY_SLOTS, ENTITY_DIM)
    initial_npc = initial[:, GLOBAL_DIM:-2].reshape(len(states), ENTITY_SLOTS, ENTITY_DIM)
    alive = (source_npc[..., 5] >= .5) & (source_npc[..., 4] > 0)
    initial_alive = (initial_npc[..., 5] >= .5) & (initial_npc[..., 4] > 0)
    previous_alive = torch.cat([initial_alive[:, None], alive[:, :-1]], 1)
    stable = initial_alive[:, None] & alive & (((~previous_alive & alive).long().cumsum(1)) == 0)
    player_alive = (initial[:, None, 4] > 0) & (targets[:, :, 4] > 0)
    player_denominator, npc_denominator = (2*player_alive.sum()).clamp_min(1), (2*stable.sum()).clamp_min(1)
    source_previous = torch.cat([initial[:, None], targets[:, :-1]], 1)
    main_contact = (prepare_contact_targets(source_previous, targets, actions, contact_protocol=contact_protocol)
                    if contact_protocol is not None else None)
    totals = dict.fromkeys(('state', 'xy_player', 'xy_npc', 'wall_player', 'wall_npc', 'teacher_state', 'teacher_readout', 'teacher_features'), 0.)
    chunks = 0
    for first in range(0, horizon, chunk_size):
        last = min(horizon, first+chunk_size)
        if first: current, memory = current.detach(), memory.detach()
        own_previous = current.detach()
        predicted, readouts, appearances = [], [], []
        with autocast():
            for index in range(first, last):
                result = model.transition(current, actions[:, index], features, memory)
                _valid_result(result, expected_shape, f'own prior {index+1}')
                current, memory = result['state'], result['memory']
                predicted.append(current)
                if teacher_values is not None:
                    readouts.append(result['rssm_readout'].float()); appearances.append(result['features'].float())
            prediction = torch.stack(predicted, 1)
            fraction = (last-first)/horizon
            previous_source = initial if first == 0 else targets[:, first-1]
            wall, _ = trajectory_state_losses(prediction, previous_source, targets[:, first:last], grid,
                offsets=range(1, last-first+1), xy_weight=0., previous_prediction=own_previous)
            npc_prediction = prediction[:, :, GLOBAL_DIM:-2].reshape(len(states), last-first, ENTITY_SLOTS, ENTITY_DIM)
            player_error = F.smooth_l1_loss(prediction[:, :, :2].float()*16, targets[:, first:last, :2]*16, reduction='none').sum(-1)
            npc_error = F.smooth_l1_loss(npc_prediction[..., :2].float()*16, source_npc[:, first:last, :, :2]*16, reduction='none').sum(-1)
            parts = {'state': fraction*10*state_loss(prediction.flatten(0, 1), targets[:, first:last].flatten(0, 1)),
                'xy_player': (player_error*player_alive[:, first:last]).sum()/player_denominator,
                'xy_npc': (npc_error*stable[:, first:last]).sum()/npc_denominator,
                'wall_player': fraction*wall['ar_wall_player'], 'wall_npc': fraction*wall['ar_wall_npc']}
            if contact_protocol is not None:
                parts['xy_player'] = contact_protocol['autonomous_xy_weight']*parts['xy_player']
                parts['xy_npc'] = contact_protocol['autonomous_xy_weight']*parts['xy_npc']
                own_old = torch.cat([own_previous[:, None, :2], prediction[:, :-1, :2]], 1)
                parts['wall_player'] = contact_protocol['autonomous_wall_weight']*swept_player_wall_loss(
                    own_old, prediction[:, :, :2], source_previous[:, first:last, :2], targets[:, first:last, :2],
                    grid, main_contact['sample_weight'][:, first:last], contact_protocol=contact_protocol)
            if teacher_values is not None:
                readout, feature = torch.stack(readouts, 1), torch.stack(appearances, 1)
                reference = teacher_values['readout'][:, first:last].detach()
                parts.update(teacher_state=fraction*teacher_weight*state_loss(prediction.flatten(0, 1), teacher_values['state'][:, first:last].flatten(0, 1)),
                    teacher_readout=fraction*teacher_weight*F.l1_loss(F.layer_norm(readout.float(), (readout.shape[-1],), eps=1e-6),
                                                                  F.layer_norm(reference.float(), (reference.shape[-1],), eps=1e-6)),
                    teacher_features=fraction*teacher_weight*.25*F.l1_loss(feature, teacher_values['features'][:, first:last]))
            value = sum(parts.values())
        if not bool(torch.isfinite(value)): raise FloatingPointError('Nonfinite autonomous RSSM objective')
        value.backward(); chunks += 1
        if check_gradients: finite_gradient_norm(model.named_parameters(), context=f'RSSM own prior backward {last}/{horizon}')
        for name, part in parts.items(): totals[name] += float(part.detach())
        if progress_callback: progress_callback(last, horizon)
    if auxiliary_batch is None:
        aux_context = context
        auxiliary_batch = _auxiliary_view(batch, context, min(horizon, 16))
    else:
        aux_context = context if auxiliary_context is None else auxiliary_context
    aux_observed = _observed_count(aux_context, auxiliary_observed_context)
    auxiliary = _posterior_backward(model, auxiliary_batch, context=aux_context, observed=aux_observed,
        autocast=autocast, chunk_size=min(chunk_size, 4), check_gradients=check_gradients, contact_protocol=contact_protocol)
    total = sum(totals.values())+auxiliary['loss']
    kl_settings = _kl_settings(model)
    floor = kl_settings['free_nats']*(kl_settings['dynamics_scale']+kl_settings['representation_scale'])
    return {'train_loss': total, 'train_loss_without_kl_floor': total-floor,
        **{'rssm_'+name: value for name, value in totals.items()},
        **{'rssm_aux_'+name: value for name, value in auxiliary.items()},
        'rssm_phase': phase, 'rssm_prior_horizon': horizon, 'rssm_prior_tbptt_chunks': chunks,
        'rssm_prior_context_frames': observed, 'rssm_prior_numerical_resets': 0,
        'rssm_prior_future_rgb_inputs': 0, 'rssm_posterior_feedback_to_prior': False,
        'rssm_teacher_used': teacher_values is not None, 'rssm_teacher_weight': teacher_weight if teacher_values is not None else 0.,
        'rssm_ema_used': False, 'rssm_renderer_calls': 0, 'rssm_optimizer_steps': 0,
        'rssm_main_loss': sum(totals.values()), 'rssm_auxiliary_loss': auxiliary['loss'],
        'rssm_kl_minimum_floor': floor, 'rssm_contact_protocol_version': 1 if contact_protocol is None else 2}
