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
    for key in ('state', 'memory', 'rssm_prior_logits', 'rssm_posterior_logits', 'rssm_readout', 'features', 'rssm_reconstruction',
                'enemy_rssm_prior_logits','enemy_rssm_posterior_logits','enemy_rssm_h','enemy_rssm_z'):
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


def _npc_xy_mask(initial_alive, source_alive, *, combat=False):
    previous_alive = torch.cat([initial_alive[:, None], source_alive[:, :-1]], 1)
    if combat:
        # The single persistent ID continues after every respawn. Only the
        # exact Source teleport/revival tick is excluded from motion tracking.
        return previous_alive & source_alive
    return initial_alive[:, None] & source_alive & (((~previous_alive & source_alive).long().cumsum(1)) == 0)


def _posterior_backward(model, batch, *, context, observed, autocast, chunk_size, check_gradients,
                        contact_protocol=None, combat_protocol=None, loss_groups=None):
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
    dual_enemy=bool(getattr(getattr(model,'config',None),'enemy_rssm_enabled',False))
    if dual_enemy:totals['enemy_posterior_kl']=0.
    contact = (prepare_contact_targets(previous, targets, actions, contact_protocol=contact_protocol)
               if contact_protocol is not None else None)
    combat_supervision = None
    combat_diagnostics = {}
    combat_names = ('combat_events', 'combat_target', 'combat_attack', 'combat_damage_npc',
                    'combat_damage_player', 'combat_death_npc', 'combat_spawn', 'combat_npc_motion')
    if combat_protocol is not None and combat_protocol.get('kinematic_protocol') is not None:
        combat_names += ('combat_kinematic_player_axes', 'combat_kinematic_npc_speed', 'combat_kinematic_npc_direction')
    if combat_protocol is not None:
        from .rssm_combat_objectives import prepare_combat_supervision, combat_supervised_terms
        combat_supervision = prepare_combat_supervision(batch, context, **combat_protocol)
        totals.update(dict.fromkeys(combat_names, 0.))
    kl_settings = _kl_settings(model)
    kl_floor = kl_settings['free_nats']*(kl_settings['dynamics_scale']+kl_settings['representation_scale'])
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
            task_parts = {}
            spawn_part = None
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
                if dual_enemy:
                    if 'enemy_rssm_prior_logits'not in post or 'enemy_rssm_posterior_logits'not in post:
                        raise ValueError('Dual RSSM posterior must expose both causal enemy prior and observed posterior')
                    enemy_settings={**kl_settings,'unimix':model.config.enemy_rssm_unimix}
                    enemy_kl,enemy_details=balanced_kl(post['enemy_rssm_posterior_logits'],post['enemy_rssm_prior_logits'],**enemy_settings)
                    values['enemy_posterior_kl']=enemy_kl/horizon
                    for name,value in enemy_details.items():
                        kl_metrics['enemy_'+name]=kl_metrics.get('enemy_'+name,0.)+float(value.detach())/horizon
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
                if combat_supervision is not None:
                    combat_terms, diagnostics = combat_supervised_terms(prior, previous[:, index], targets[:, index],
                        actions[:, index], supervision=combat_supervision, index=index)
                    values.update(combat_terms)
                    for name, value in diagnostics.items():
                        combat_diagnostics[name] = combat_diagnostics.get(name, 0.)+float(value)
                if loss_groups is not None:
                    from .rssm_loss_balance import split_state_loss, combat_loss_groups
                    separated = split_state_loss(prior['state'], targets[:, index])
                    task_values = {name: 2*value/horizon for name, value in separated.items()}
                    task_values['player'] = task_values['player']+values['posterior_kl']+values['posterior_reconstruction']
                    if dual_enemy:
                        task_values['enemy'] = task_values['enemy']+values['enemy_posterior_kl']
                    if contact_protocol is None:
                        task_values['player'] = (task_values['player']+PASS_WEIGHT*values['pass_bce']
                            +RAW_DELTA_WEIGHT*values['raw_delta']+SCALED_MOTION_WEIGHT*values['scaled_motion'])
                    else:
                        task_values['player'] = task_values['player']+sum(contact_protocol[name+'_weight']*values[name]
                            for name in ('pass_bce', 'physical_motion', 'endpoint_margin', 'contact_wall'))
                    if combat_supervision is not None:
                        for name, value in combat_loss_groups(combat_terms).items():
                            task_values[name] = task_values[name]+value
                        value = combat_terms['combat_spawn']
                        task_values['enemy'] = task_values['enemy']-value
                        spawn_part = value if spawn_part is None else spawn_part+value
                    for name, value in task_values.items():
                        task_parts[name] = task_parts.get(name, 0)+value
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
            if dual_enemy:loss=loss+parts['enemy_posterior_kl']
            if contact_protocol is None:
                loss = loss+PASS_WEIGHT*parts['pass_bce']+RAW_DELTA_WEIGHT*parts['raw_delta']+SCALED_MOTION_WEIGHT*parts['scaled_motion']
            else:
                loss = loss+sum(contact_protocol[name+'_weight']*parts[name]
                    for name in ('pass_bce', 'physical_motion', 'endpoint_margin', 'contact_wall'))
            if combat_supervision is not None:
                loss = loss+sum(parts[name] for name in combat_names)
            if loss_groups is not None:
                floors = {'player': kl_floor*(last-first)/horizon}
                if dual_enemy: floors['enemy'] = floors['player']
                loss = loss_groups.combine(task_parts, floors=floors)
                if spawn_part is not None:
                    loss = loss+loss_groups.combine({'enemy': spawn_part}, component='spawn')
        if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite RSSM posterior/motion objective')
        loss.backward(); calls += 1
        if check_gradients: finite_gradient_norm(model.named_parameters(), context=f'RSSM posterior backward {last}/{horizon}')
        for name, value in parts.items(): totals[name] += float(value.detach())
    total = totals['posterior_kl']+totals['posterior_reconstruction']+totals['posterior_state']
    if dual_enemy:total+=totals['enemy_posterior_kl']
    if contact_protocol is None:
        total = total+PASS_WEIGHT*totals['pass_bce']+RAW_DELTA_WEIGHT*totals['raw_delta']+SCALED_MOTION_WEIGHT*totals['scaled_motion']
    else:
        total = total+sum(contact_protocol[name+'_weight']*totals[name]
            for name in ('pass_bce', 'physical_motion', 'endpoint_margin', 'contact_wall'))
    if combat_supervision is not None:
        total += sum(totals[name] for name in combat_names)
        combat_diagnostics.update(combat_source_shot_count=float(batch['events'][..., 0].sum()),
            combat_source_player_hit_count=float(batch['events'][..., 3].sum()),
            combat_source_player_death_count=float(((previous[..., 4] > 0) & (targets[..., 4] <= 0)).sum()))
    return {'loss': total, **totals, **{'kl_'+name: value for name, value in kl_metrics.items()},
        **combat_diagnostics, 'combat_objectives_active': combat_supervision is not None,
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
                           contact_protocol=None, contact_branch_protocol=None, combat_protocol=None,
                           numeric_state_feedback='attached', combat_branch_protocol=None, loss_groups=None,
                           coordinate_gradient_protocol=None, memory_overshooting_protocol=None,
                           late_prefix_protocol=None, late_prefix_batch=None):
    """Backpropagate own prior trajectory, then separate posterior/Source labels.

    Distillation uses1..16 own teacher/student transitions plus Source anchors.
    Source learning defaults to450 own transitions. No optimizer step, EMA,
    renderer call, GT state replacement or future RGB input to that trajectory.
    """
    if numeric_state_feedback not in ('attached', 'detached'):
        raise ValueError('Unknown numeric-state gradient feedback protocol')
    from .rssm_coordinate_gradients import normalize_coordinate_gradient_protocol, integrate_coordinate_gradients
    coordinate_gradient_protocol = normalize_coordinate_gradient_protocol(coordinate_gradient_protocol)
    if coordinate_gradient_protocol is not None and numeric_state_feedback != 'detached':
        raise ValueError('XY identity credit requires detached neural state conditioning')
    if loss_groups is not None and (phase != 'source' or loss_groups.calls != 0):
        raise ValueError('Grouped losses require a fresh Source microbatch accumulator')
    own_combat_protocol = None
    combat_candidates = []
    if combat_branch_protocol is not None:
        from .rssm_combat_branches import (normalize_combat_branch_protocol,
            capture_combat_candidate, combat_branches_backward)
        own_combat_protocol = normalize_combat_branch_protocol(combat_branch_protocol)
        if phase != 'source': raise ValueError('Own combat branches require the Source phase')
    if (phase not in {'source', 'distill'} or phase == 'distill' and teacher is None
            or phase == 'source' and teacher is not None or not 0 <= teacher_weight <= .25
            or not isinstance(chunk_size, int) or chunk_size < 1 or amp is not None and not callable(amp)):
        raise ValueError('Invalid RSSM phase/teacher/chunk/weight/AMP contract')
    states, frames, actions, targets, grid = _batch_data(batch, context)
    contact_protocol = normalize_contact_protocol(contact_protocol)
    branch_protocol, branch_candidates = None, []
    if contact_branch_protocol is not None:
        from .rssm_contact_branches import (normalize_contact_branch_protocol,
            capture_contact_candidate, contact_branches_backward)
        branch_protocol = normalize_contact_branch_protocol(contact_branch_protocol)
        if phase != 'source':
            raise ValueError('Own-memory contact branches belong only to Source training')
    observed = _observed_count(context, observed_context)
    horizon = actions.shape[1]
    expected = (450 if phase == 'source' else horizon) if expected_horizon is None else expected_horizon
    if horizon != expected or phase == 'distill' and horizon > 16:
        raise ValueError('RSSM source learning requires450 steps; distillation is bounded to16')
    late_settings = None
    if late_prefix_protocol is not None:
        from .rssm_late_prefix_recovery import normalize_late_prefix_protocol, continue_own_prefix_candidates
        late_settings = normalize_late_prefix_protocol(late_prefix_protocol)
        if phase != 'source' or own_combat_protocol is None or own_combat_protocol['version'] != 2:
            raise ValueError('Late own prefixes require actual Source combat branches version2')
        if numeric_state_feedback != 'detached':
            raise ValueError('Late own-prefix recovery preserves detached numerical conditioning')
    if late_prefix_batch is not None and late_settings is None:
        raise ValueError('Late-prefix controls require their explicit protocol')
    if late_settings is not None and horizon < late_settings['scout_horizon']:
        scout_count = min(len(states), late_settings['scout_batch_size'])
        if (not isinstance(late_prefix_batch, dict) or set(late_prefix_batch) != {'actions', 'map'}
                or late_prefix_batch['actions'].shape != (scout_count, late_settings['scout_horizon'], 4)
                or late_prefix_batch['map'].shape != grid[:scout_count].shape
                or not torch.equal(late_prefix_batch['map'], grid[:scout_count])
                or not torch.equal(late_prefix_batch['actions'][:, :horizon].detach().float(), actions[:scout_count])):
            raise ValueError('Late-prefix controls/maps must match the exact first8 main Source pairs')
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
    stable = _npc_xy_mask(initial_alive, alive, combat=combat_protocol is not None)
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
        branch_start = (current.detach(), memory.detach()) if branch_protocol is not None or own_combat_protocol is not None else None
        predicted, readouts, appearances = [], [], []
        with autocast():
            for index in range(first, last):
                # Own numerical values remain identical. The optional combat
                # protocol cuts only the unstable typed-state Jacobian; latent
                # h/z memory still carries temporal gradients within TBPTT.
                transition_state = current.detach() if numeric_state_feedback == 'detached' else current
                result = model.transition(transition_state, actions[:, index], features, memory)
                _valid_result(result, expected_shape, f'own prior {index+1}')
                result = integrate_coordinate_gradients(current, result, coordinate_gradient_protocol)
                current, memory = result['state'], result['memory']
                predicted.append(current)
                if teacher_values is not None:
                    readouts.append(result['rssm_readout'].float()); appearances.append(result['features'].float())
            prediction = torch.stack(predicted, 1)
            if branch_protocol is not None:
                branch_candidates.append(capture_contact_candidate(first, branch_start[0], branch_start[1],
                    actions[:, first:min(horizon, first+branch_protocol['horizon'])], grid, prediction))
            if own_combat_protocol is not None:
                combat_candidates.append(capture_combat_candidate(first, branch_start[0], branch_start[1],
                    actions[:, first:min(horizon, first+own_combat_protocol['horizon'])], grid, prediction))
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
            if loss_groups is not None:
                from .rssm_loss_balance import split_state_loss
                separated = split_state_loss(prediction.flatten(0, 1), targets[:, first:last].flatten(0, 1))
                tasks = {name: fraction*10*term for name, term in separated.items()}
                tasks['player'] = tasks['player']+parts['xy_player']+parts['wall_player']
                tasks['enemy'] = tasks['enemy']+parts['xy_npc']+parts['wall_npc']
                value = loss_groups.combine(tasks)
        if not bool(torch.isfinite(value)): raise FloatingPointError('Nonfinite autonomous RSSM objective')
        value.backward(); chunks += 1
        if check_gradients: finite_gradient_norm(model.named_parameters(), context=f'RSSM own prior backward {last}/{horizon}')
        for name, part in parts.items(): totals[name] += float(part.detach())
        if progress_callback: progress_callback(last, horizon)
    late_scout_metrics = {}
    if late_settings is not None and horizon < late_settings['scout_horizon']:
        extra_candidates, late_scout_metrics = continue_own_prefix_candidates(model,
            current[:scout_count], memory[:scout_count], features[:scout_count], grid[:scout_count],
            late_prefix_batch['actions'], start=horizon, late_prefix_protocol=late_settings, amp=amp)
        combat_candidates.extend(extra_candidates)
    branch_metrics = (contact_branches_backward(model, branch_candidates,
        contact_branch_protocol=branch_protocol, contact_protocol=contact_protocol,
        amp=amp, check_gradients=check_gradients,
        **({'loss_groups': loss_groups} if loss_groups is not None else {})) if branch_protocol is not None else {})
    combat_branch_metrics = (combat_branches_backward(model, combat_candidates,
        combat_branch_protocol=own_combat_protocol, amp=amp, check_gradients=check_gradients,
        coordinate_gradient_protocol=coordinate_gradient_protocol,
        kinematic_protocol=(combat_protocol or {}).get('kinematic_protocol'),
        **({'late_prefix_protocol': late_settings} if late_settings is not None else {}),
        **({'contact_protocol': contact_protocol} if own_combat_protocol.get('version') == 2 else {}),
        **({'loss_groups': loss_groups} if loss_groups is not None else {}))
        if own_combat_protocol is not None else {})
    if auxiliary_batch is None:
        aux_context = context
        auxiliary_batch = _auxiliary_view(batch, context, min(horizon, 16))
    else:
        aux_context = context if auxiliary_context is None else auxiliary_context
    aux_observed = _observed_count(aux_context, auxiliary_observed_context)
    auxiliary = _posterior_backward(model, auxiliary_batch, context=aux_context, observed=aux_observed,
        autocast=autocast, chunk_size=min(chunk_size, 4), check_gradients=check_gradients,
        contact_protocol=contact_protocol, combat_protocol=combat_protocol, loss_groups=loss_groups)
    memory_metrics = {}
    if memory_overshooting_protocol is not None:
        from .rssm_memory_overshooting import memory_overshooting_backward
        memory_metrics = memory_overshooting_backward(model, auxiliary_batch, context=aux_context,
            observed_context=aux_observed, protocol=memory_overshooting_protocol,
            coordinate_gradient_protocol=coordinate_gradient_protocol,
            loss_groups=loss_groups, check_gradients=check_gradients)
    total = (sum(totals.values())+auxiliary['loss']+branch_metrics.get('loss', 0.)
        +combat_branch_metrics.get('loss', 0.)+memory_metrics.get('loss', 0.))
    kl_settings = _kl_settings(model)
    floor = kl_settings['free_nats']*(kl_settings['dynamics_scale']+kl_settings['representation_scale'])
    if getattr(getattr(model,'config',None),'enemy_rssm_enabled',False):floor*=2
    balanced = loss_groups.report() if loss_groups is not None else None
    if balanced is not None and not math.isclose(balanced['kl_floor'], floor, rel_tol=1e-6, abs_tol=1e-6):
        raise AssertionError('Grouped KL floor accounting omitted or duplicated an observation objective')
    actual = balanced['total'] if balanced is not None else total
    return {'train_loss': actual, 'train_loss_without_kl_floor': actual-floor,
        **({'train_loss_raw': total, 'loss_groups': balanced} if balanced is not None else {}),
        **{'rssm_'+name: value for name, value in totals.items()},
        **{'rssm_aux_'+name: value for name, value in auxiliary.items()},
        **{'rssm_contact_branch_'+name: value for name, value in branch_metrics.items()},
        **{'rssm_combat_branch_'+name: value for name, value in combat_branch_metrics.items()},
        **{'rssm_memory_overshooting_'+name: value for name, value in memory_metrics.items()},
        **{'rssm_late_prefix_scout_'+name: value for name, value in late_scout_metrics.items()},
        'rssm_integrated_xy_gradients': coordinate_gradient_protocol is not None,
        'rssm_phase': phase, 'rssm_prior_horizon': horizon, 'rssm_prior_tbptt_chunks': chunks,
        'rssm_prior_context_frames': observed, 'rssm_prior_numerical_resets': 0,
        'rssm_prior_future_rgb_inputs': 0, 'rssm_posterior_feedback_to_prior': False,
        'rssm_teacher_used': teacher_values is not None, 'rssm_teacher_weight': teacher_weight if teacher_values is not None else 0.,
        'rssm_ema_used': False, 'rssm_renderer_calls': 0, 'rssm_optimizer_steps': 0,
        'rssm_main_loss': sum(totals.values()), 'rssm_auxiliary_loss': auxiliary['loss'],
        'rssm_npc_xy_supervised_transitions': int(stable.sum()),
        'rssm_numeric_state_feedback': numeric_state_feedback,
        'rssm_kl_minimum_floor': floor, 'rssm_contact_protocol_version': 1 if contact_protocol is None else 2}
