"""MAIN16 navigation/quiet-enemy backward, with no combat data or objective.

Eight navigation and eight quiet-enemy actors supply all independent examples.
Short recovery branches reuse their actual own-prefix state, complete P/E/C
h/z and controls. No teacher/AUX/late combat batch, renderer or optimizer runs.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import math

import numpy as np
import torch
from torch.nn import functional as F

from .contact_objectives import CONTACT_PROTOCOL_V2, swept_player_wall_loss
from .gradient_safety import finite_gradient_norm
from .rssm_combat_protocol import environment_profile
from .rssm_coordinate_gradients import COORDINATE_GRADIENT_PROTOCOL, integrate_coordinate_gradients
from .rssm_enemy_local_recovery import (capture_enemy_local_candidate, enemy_local_recovery_backward,
    normalize_enemy_local_protocol)
from .rssm_objectives import _batch_data, _initialize, _observed_count, _valid_result
from .rssm_migration import FROZEN_PREFIXES
from .rssm_player_contact_replay import capture_player_contact_candidate, select_player_contact_branches
from .rssm_player_geometry_objectives import (prepare_player_geometry, player_geometry_terms)
from .rssm_npc_geometry_objectives import prepare_npc_geometry, npc_geometry_terms
from .rssm_player_local_motion import (normalize_player_local_motion_protocol, source_player_motion_step,
    player_motion_supervised_terms)

PROTOCOL = 'rssm_motion_only_main16_v1'
HORIZONS = (16, 48, 80, 112, 144, 180)
COMBAT_READOUT_PREFIXES = ('vjepa_backend.hit_target_head.', 'vjepa_backend.attack_head.',
    'vjepa_backend.spatial_combat_head.', 'vjepa_backend.enemy_rssm.attack_head.',
    'vjepa_backend.combat_rssm.task_head.', 'vjepa_backend.fusion_moe.combat_head.',
    'vjepa_backend.five_role_fusion.combat_head.')
OBJECTIVE_SETTINGS = dict(protocol=PROTOCOL, main_navigation=8, main_enemy_motion=8,
    independent_examples=16, player_weights=dict(navigation=.5, enemy_motion=.5), enemy_weight=1.,
    router_weights=dict(navigation=.5, enemy_motion=.5, role=.01, balance=.001),
    router_base_roles=[0, 1], router_five_roles=[0, 1, 3], router_player_modes=[0, 1],
    router_balance='noncombat_probability_importance_CV_squared',
    full_BPTT_current_horizon=True, capture_every=16,
    player_local_weight=.25, local_branches_from_MAIN_only=True,
    independent_contact_scouts=0, additional_fragment_draws=0,
    combat_loss=False, combat_MAIN_actors=0, combat_AUX_actors=0, combat_late_actors=0,
    teacher_calls=0, renderer_calls=0, Source_future_feedback=False,
    all_three_memories_retained=True, shared_memory_motion_gradients=True,
    pure_combat_readout_prefixes=list(COMBAT_READOUT_PREFIXES))


def configure_motion_only_trainable(model):
    """Call AFTER legacy named-Adam restore/configure; preserve all group/state IDs.

    Frozen task readouts have grad=None, so Adam cannot decay or advance them.
    The complete Combat recurrent memory remains a shared motion representation.
    """
    frozen = []
    for name, parameter in model.named_parameters():
        if name.startswith(FROZEN_PREFIXES):
            parameter.requires_grad_(False)
            parameter.grad = None
        if name.startswith(COMBAT_READOUT_PREFIXES):
            parameter.requires_grad_(False)
            parameter.grad = None
            frozen.append(name)
    return dict(protocol=PROTOCOL, pure_combat_readouts_frozen=frozen,
        frozen_readout_parameters=sum(p.numel() for n, p in model.named_parameters() if n in frozen),
        all_three_memories_retained=True, shared_combat_memory_may_receive_motion_gradients=True,
        optimizer_groups_rebuilt=False, optimizer_states_reset=False)


def _restricted_router(logits, roles):
    if not torch.is_tensor(logits) or logits.ndim != 3 or logits.shape[1] != logits.shape[2]:
        raise ValueError('Motion routers require actual square role/expert logits')
    index = torch.as_tensor(roles, device=logits.device)
    selected = logits.index_select(1, index).index_select(2, index).float()
    labels = torch.arange(len(roles), device=logits.device)[None].expand(len(logits), -1)
    role = F.cross_entropy(selected.reshape(-1, len(roles)), labels.reshape(-1))
    importance = selected.softmax(-1).mean((0, 1))
    balance = (len(roles)*importance.square().sum()-1).clamp_min(0)
    return role, balance


def motion_router_terms(result):
    """Exclude combat query/target columns instead of reusing summed old losses."""
    base = result.get('rssm_fusion_moe')
    if not isinstance(base, dict) or base.get('router_logits') is None:
        raise ValueError('Motion-only objective needs unmerged raw three-role router logits')
    if base['router_logits'].shape[1:] != (3, 3):
        raise ValueError('Expected the canonical three-role router')
    role, balance = _restricted_router(base['router_logits'], (0, 1))
    five = base.get('five_role_fusion', result.get('five_role_fusion'))
    if five is not None:
        if five['router_logits'].shape[1:] != (5, 5):
            raise ValueError('Expected the canonical five-role router')
        extra_role, extra_balance = _restricted_router(five['router_logits'], (0, 1, 3))
        role, balance = role+extra_role, balance+extra_balance
        modes, targets = five.get('player_mode_logits'), five.get('player_mode_targets')
        if modes is None or targets is None or modes.shape != (len(modes), 3) or targets.shape != (len(modes),):
            raise ValueError('Motion-only routing needs actual causal player-mode labels')
        eligible = (targets == 0) | (targets == 1)
        mode_loss = (F.cross_entropy(modes[eligible, :2].float(), targets[eligible])
            if bool(eligible.any()) else modes.sum()*0)
        role = role+.25*mode_loss
    if not bool(torch.isfinite(role)) or not bool(torch.isfinite(balance)):
        raise FloatingPointError('Nonfinite motion-only router supervision')
    return role, balance


def _validate_source(states, actions, target, *, stream, horizon):
    if len(actions) != 8 or actions.shape[1:] != (horizon, 4) or bool(actions[..., 3].ne(0).any()):
        raise ValueError('Each motion-only stream requires exactly8 no-fire actors at currentH')
    tape = torch.cat((states, target), 1)
    if (bool((tape[..., 4] <= 0).any())
            or not torch.equal(tape[..., 4], tape[:, :1, 4].expand(-1, tape.shape[1]))):
        raise ValueError('Motion-only Source players must stay alive with unchangedHP')
    if stream == 'navigation':
        if bool(tape[..., 12:14].ne(0).any()):
            raise ValueError('Navigation Source must contain no enemy')
    elif (bool((tape[..., 12] <= 0).any()) or bool((tape[..., 13] < .5).any())
            or not torch.equal(tape[..., 12], tape[:, :1, 12].expand(-1, tape.shape[1]))):
        raise ValueError('Quiet-enemy Source requires one continuing alive HP-stable enemy')


def _stream_backward(model, batch, *, stream, horizon, context, observed_context,
                     player_navigation_protocol, npc_navigation_protocol, captures, enemy_captures,
                     check_gradients, progress_callback):
    states, frames, actions, target, grid = _batch_data(batch, context)
    observed = _observed_count(context, observed_context)
    _validate_source(states[:, :context], actions, target, stream=stream, horizon=horizon)
    previous = torch.cat((states[:, context-1:context], target[:, :-1]), 1)
    alive = (previous[..., 4] > 0) & (target[..., 4] > 0)
    player_geometry = prepare_player_geometry(previous, target, grid,
        player_navigation_protocol=player_navigation_protocol)
    if player_geometry is None:
        raise ValueError('Motion-only training requires the explicit unchanged player geometry protocol')
    npc_geometry = (prepare_npc_geometry(previous, target, grid, npc_navigation_protocol=npc_navigation_protocol)
        if stream == 'enemy_motion' else None)
    if stream == 'enemy_motion' and npc_geometry is None:
        raise ValueError('Motion-only NPC training requires the explicit own-aligned geometry protocol')
    modes = [(module, module.training) for module in model.modules()]
    try:
        with environment_profile(model, navigation_only=stream == 'navigation'):
            with torch.no_grad(): fmap = model.encode_map(grid).detach()
            memory = _initialize(model, frames, states, fmap, context, observed)
            shape = tuple(memory.shape)
            current = states[:, context-1]
            initial = current
            points, respawns, router_roles, router_balances, prefix = [], [], [], [], []
            for tick in range(horizon):
                if tick % 16 == 0 and horizon-tick >= 4:
                    prefix.append((tick, current, memory))
                result = model.transition(current.detach(), actions[:, tick], fmap, memory)
                _valid_result(result, shape, 'Motion-only full-current-H prior')
                if stream == 'navigation' and (bool(result['state'][:, 12:14].ne(0).any())
                        or bool(result['respawn_flags'].any())):
                    raise ValueError('Navigation prediction produced an enemy/respawn')
                role, balance = motion_router_terms(result)
                result = integrate_coordinate_gradients(current, result, COORDINATE_GRADIENT_PROTOCOL)
                current, memory = result['state'], result['memory']
                points.append(current)
                respawns.append(result['respawn_flags'])
                router_roles.append(role/horizon); router_balances.append(balance/horizon)
                if progress_callback: progress_callback(stream, tick+1, horizon)
            prediction = torch.stack(points, 1)
            for tick, own_state, own_memory in prefix:
                controls = actions[:, tick:tick+32]
                future = prediction[:, tick:tick+16]
                captures.append(capture_player_contact_candidate(tick, own_state, own_memory,
                    controls, grid, future, navigation_only=stream == 'navigation', stream=stream))
                if stream == 'enemy_motion':
                    enemy_captures.append(capture_enemy_local_candidate(tick, own_state, own_memory,
                        controls, grid, future, stream='enemy_motion'))
            player_xy = (F.smooth_l1_loss(prediction[..., :2].float()*16,
                target[..., :2].detach().float()*16, reduction='none').sum(-1)*alive).sum()/(2*alive.sum()).clamp_min(1)
            player_wall, player_wall_metrics = player_geometry_terms(prediction, initial, player_geometry)
            enemy_xy = prediction.sum()*0; enemy_wall = enemy_xy; npc_wall_metrics = {}
            if npc_geometry is not None:
                npc_alive = npc_geometry['alive']
                enemy_xy = (F.smooth_l1_loss(prediction[..., 8:10].float()*16,
                    target[..., 8:10].detach().float()*16, reduction='none').sum(-1)*npc_alive).sum()/(2*npc_alive.sum()).clamp_min(1)
                enemy_wall, npc_wall_metrics = npc_geometry_terms(prediction, initial, npc_geometry,
                    predicted_respawns=torch.stack(respawns, 1))
            role, balance = sum(router_roles), sum(router_balances)
            player = .5*(player_xy+player_wall)
            enemy = enemy_xy+enemy_wall
            router = .5*(.01*role+.001*balance)
            loss = player+enemy+router
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite motion-only long objective')
            loss.backward()
            if check_gradients: finite_gradient_norm(model.named_parameters(), context='Motion-only main full-H')
            return dict(stream=stream, actors=8, horizon=horizon, tbptt=horizon,
                player=float(player.detach()), enemy=float(enemy.detach()), router=float(router.detach()),
                raw_player_xy=float(player_xy.detach()), raw_player_wall=float(player_wall.detach()),
                raw_enemy_xy=float(enemy_xy.detach()), raw_enemy_wall=float(enemy_wall.detach()),
                raw_router_role=float(role.detach()), raw_router_balance=float(balance.detach()),
                player_geometry=player_wall_metrics, npc_geometry=npc_wall_metrics,
                forward_transitions=8*horizon, observed_context_frames=observed,
                memory_detaches_after_initialization=0, state_resets_after_initialization=0,
                future_RGB_encoded=0, Source_future_feedback=False,
                recorded_NPC_velocity_direction_gradients=False,
                full_three_memory_shape=list(shape))
    finally:
        for module, mode in modes: module.training = mode


def _player_local_backward(model, candidates, *, horizon, protocol, check_gradients):
    """Existing recovery objective, using MAIN prefixes instead of extra scouts."""
    protocol = normalize_player_local_motion_protocol(protocol)
    if protocol is None: raise ValueError('Motion-only training requires current player local-motion supervision')
    chosen, selection = select_player_contact_branches(candidates, horizon=horizon,
        simulation_hz=model.config.simulation_hz, player_local_motion_protocol=protocol)
    metrics = dict(player_contact_loss=0., player_contact_horizon=horizon, scout_forward_transitions=0,
        independent_contact_fragments=0, branches=len(chosen), optimizer_steps=0, renderer_calls=0,
        Source_labels_after_own_rollout=True, source_future_feedback=False,
        state_resets_after_prefix=0, memory_resets_after_prefix=0, selection=deepcopy(selection), **selection)
    if not chosen: return metrics
    counts = Counter(row['cohort'] for row in chosen)
    records = []
    for branch, row in enumerate(chosen):
        item, actor = row['item'], row['actor']
        current = item['state'][actor:actor+1].detach()
        memory = item['memory'][actor:actor+1].detach()
        grid = item['map'][actor:actor+1].detach()
        actions = item['actions'][actor:actor+1, :row['horizon']].detach()
        branch_weight = 1/(len(counts)*counts[row['cohort']])
        with environment_profile(model, navigation_only=item['navigation_only']):
            with torch.no_grad(): fmap = model.encode_map(grid).detach()
            for tick in range(row['horizon']):
                old = current.detach()
                raw = model.transition(old, actions[:, tick], fmap, memory)
                _valid_result(raw, tuple(memory.shape), 'MAIN-prefix player recovery')
                records.append(dict(branch=branch, tick=tick, weight=branch_weight, previous=old,
                    prediction=raw['state'], pass_logits=raw['player_pass_logits'], action=actions[:, tick], grid=grid))
                integrated = integrate_coordinate_gradients(current, raw, COORDINATE_GRADIENT_PROTOCOL)
                current, memory = integrated['state'], integrated['memory']
    direct, skipped = [], Counter()
    for record in records:
        try:
            label = source_player_motion_step(record['previous'][0].detach().float().cpu().numpy(),
                record['action'][0].detach().float().cpu().numpy(), record['grid'][0].detach().cpu().numpy(),
                dt=1/model.config.simulation_hz)
            direct.append((record, label))
        except ValueError as error: skipped[str(error)] += 1
    metrics.update(forward_transitions=len(records), direct_source_valid=len(direct),
        direct_source_attempts=len(records), direct_source_skip_counts=dict(skipped),
        direct_Source_scope='exact_single_player_movement_step', new_observed_rgb_frames_after_prefix=0)
    if not direct: return metrics
    old = torch.cat([record['previous'] for record, _ in direct]).detach()
    prediction = torch.cat([record['prediction'] for record, _ in direct])
    actions = torch.cat([record['action'] for record, _ in direct]).detach()
    logits = torch.cat([record['pass_logits'] for record, _ in direct])
    target = torch.as_tensor(np.stack([label['target'] for _, label in direct]), device=old.device, dtype=old.dtype)
    local, diagnostics = player_motion_supervised_terms(old, prediction, actions, target, logits,
        player_local_motion_protocol=protocol, simulation_hz=model.config.simulation_hz)
    by_branch = defaultdict(list)
    for index, (record, _) in enumerate(direct): by_branch[record['branch']].append((index, record))
    residual = (prediction[:, :2]-target[:, :2])*16
    trajectory = prediction.sum()*0
    sample_weights = []
    for items in by_branch.values():
        segments, segment, previous_tick = [], [], None
        for index, record in items:
            if previous_tick is not None and record['tick'] != previous_tick+1:
                segments.append(segment); segment = []
            segment.append(index); previous_tick = record['tick']
        if segment: segments.append(segment)
        weight = items[0][1]['weight']
        for segment in segments:
            cumulative = residual[segment].cumsum(0)
            trajectory = trajectory+weight*F.smooth_l1_loss(cumulative,
                torch.zeros_like(cumulative), reduction='sum')/(2*len(items))
    for record, _ in direct: sample_weights.append(record['weight']/len(by_branch[record['branch']]))
    grids = torch.cat([record['grid'] for record, _ in direct]).detach()
    wall = swept_player_wall_loss(old[:, None, :2], prediction[:, None, :2], old[:, None, :2],
        target[:, None, :2], grids, torch.as_tensor(sample_weights, device=old.device, dtype=old.dtype)[:, None],
        contact_protocol={**CONTACT_PROTOCOL_V2, 'wall_margin_tiles': .02})
    loss = .25*(trajectory+wall+sum(local.values()))
    if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite MAIN-prefix player local objective')
    loss.backward()
    if check_gradients: finite_gradient_norm(model.named_parameters(), context='Motion-only player local recovery')
    metrics.update(player_contact_loss=float(loss.detach()), xy_loss=float(trajectory.detach()),
        wall_loss=float(wall.detach()), axis_bce_loss=float(local['pass_bce'].detach()),
        axis_velocity_loss=float(local['axis_velocity'].detach()), free_progress_loss=float(local['free_progress'].detach()),
        player_local_motion=diagnostics, backward_calls=1, full_own_memory_preserved=True)
    return metrics


def motion_only_backward(model, navigation_batch, enemy_batch, contact_batch=None, *, horizon,
                         context=1, observed_context=None, player_navigation_protocol,
                         npc_navigation_protocol, player_local_protocol, enemy_local_protocol,
                         player_local_horizon=16, enemy_local_horizon=16,
                         check_gradients=True, progress_callback=None):
    """Accumulate two independent group means and MAIN-prefix local gradients.

    The caller owns zero_grad, clipping, Adam and the exact20-update stage gate.
    No random batch draw or independent recovery fragment occurs here.
    """
    if contact_batch is not None:
        raise ValueError('Motion-only phase uses MAIN-prefix recovery, not an extra contact batch')
    if type(horizon) is not int or horizon not in HORIZONS:
        raise ValueError('Motion-only horizon must be one of16,48,80,112,144,180')
    if not isinstance(player_navigation_protocol, dict) or player_navigation_protocol.get('tbptt') != horizon:
        raise ValueError('Motion-only main BPTT must equal the full current horizon')
    if any(type(value) is not int or value not in (4, 8, 16, 32) or value > horizon
            for value in (player_local_horizon, enemy_local_horizon)):
        raise ValueError('Motion local branches require4/8/16/32 bounded by currentH')
    if normalize_enemy_local_protocol(enemy_local_protocol) is None:
        raise ValueError('Motion-only training requires current own-aligned NPC local recovery')
    # Reject an invalid second stream before the first stream can accumulate
    # gradients. A failed update must never leave half a valid batch to commit.
    for stream, batch in (('navigation', navigation_batch), ('enemy_motion', enemy_batch)):
        states, _, actions, target, _ = _batch_data(batch, context)
        _validate_source(states[:, :context], actions, target, stream=stream, horizon=horizon)
    configure_motion_only_trainable(model)
    captures, enemy_captures = [], []
    streams = []
    for stream, batch in (('navigation', navigation_batch), ('enemy_motion', enemy_batch)):
        streams.append(_stream_backward(model, batch, stream=stream, horizon=horizon, context=context,
            observed_context=observed_context, player_navigation_protocol=player_navigation_protocol,
            npc_navigation_protocol=npc_navigation_protocol, captures=captures, enemy_captures=enemy_captures,
            check_gradients=check_gradients, progress_callback=progress_callback))
    player_local = _player_local_backward(model, captures, horizon=player_local_horizon,
        protocol=player_local_protocol, check_gradients=check_gradients)
    enemy_local = enemy_local_recovery_backward(model, candidates=enemy_captures,
        horizon=enemy_local_horizon, protocol=enemy_local_protocol, check_gradients=check_gradients)
    player = sum(row['player'] for row in streams)+player_local['player_contact_loss']
    enemy = sum(row['enemy'] for row in streams)+enemy_local['enemy_local_loss']
    router = sum(row['router'] for row in streams)
    total = player+enemy+router
    source_queries = (player_local.get('direct_source_attempts', 0)+player_local.get('source_probes', 0)
        +enemy_local.get('total_Source_step_queries', enemy_local.get('selection', {}).get('source_probes', 0)))
    groups = {name: dict(raw=value, weight=1., weighted=value) for name, value in (
        ('player', player), ('enemy', enemy), ('combat', 0.), ('router', router))}
    groups['total'] = total
    return dict(protocol=PROTOCOL, objective_formula_revision=PROTOCOL, motion_only_loss_revision=PROTOCOL,
        train_loss=total, player_loss=player, enemy_loss=enemy, combat_loss=0.,
        moe_player_loss=player, moe_enemy_loss=enemy, moe_combat_loss=0., moe_router_loss=router,
        loss_groups=groups, main_horizon=horizon, moe_main_horizon=horizon,
        microbatch_size=16, effective_batch_size=16, gradient_accumulation_steps=1, completed_microbatches=1,
        main_navigation_actor_count=8, main_enemy_motion_actor_count=8, main_combat_actor_count=0,
        main_independent_actor_count=16, combat_auxiliary_actor_count=0, combat_late_actor_count=0,
        extra_contact_fragment_count=0, contact_scout_transitions=0, combat_batch_draws=0,
        motion_only_Source_queries=source_queries,
        teacher_calls=0, renderer_calls=0, optimizer_steps=0, Source_future_feedback=False,
        direct_combat_supervision_calls=0, combat_role_router_supervision=False,
        all_three_memories_retained=True, pure_combat_readouts_frozen=True,
        shared_combat_memory_may_receive_motion_gradients=True,
        nav_tbptt=horizon, enemy_motion_tbptt=horizon, moe_main_tbptt=horizon,
        motion_only_streams=streams, player_contact_replay=player_local, enemy_local_recovery=enemy_local,
        enemy_local_loss=enemy_local['enemy_local_loss'],
        enemy_local_valid_source_labels=enemy_local.get('enemy_local_valid_source_labels', 0),
        player_local_horizon=player_local_horizon, enemy_local_horizon=enemy_local_horizon,
        objective_settings=deepcopy(OBJECTIVE_SETTINGS))
