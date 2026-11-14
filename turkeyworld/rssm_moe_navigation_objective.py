"""No-enemy main-batch navigation replay for the three-memory MoE model.

Only genuine initial observations and subsequent controls condition a rollout.
Future Source coordinates are loss labels; no Source oracle, posterior, combat
auxiliary, renderer or optimizer is called by this module.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from .gradient_safety import finite_gradient_norm
from .rssm_combat_protocol import environment_profile
from .rssm_coordinate_gradients import COORDINATE_GRADIENT_PROTOCOL, integrate_coordinate_gradients
from .rssm_moe_protocol import MOE_HORIZONS, SETTINGS
from .rssm_moe_training import _router_terms
from .rssm_objectives import _batch_data, _initialize, _observed_count, _valid_result
from .rssm_player_geometry_objectives import(prepare_player_geometry,player_geometry_terms,add_player_geometry_metrics,
    player_navigation_tbptt,capture_player_chunks)


PROTOCOL = 'rssm_moe_no_enemy_main_navigation_v1'


def navigation_replay_backward(model, batch, *, context=1, observed_context=None,
                               expected_horizon=None, weight=10/64,
                               progress_callback=None, check_gradients=True,
                               player_navigation_protocol=None,player_contact_callback=None):
    """Accumulate weighted player XY/router gradients without stepping Adam.

    ``weight`` applies to both the player mean and the already fixed .01/.001
    router weights. Enemy/combat losses are deliberately absent: the unchanged
    combat main/auxiliary streams retain those groups and their denominators.
    Packed P/E/Combat memory continues through every own transition, with graph
    detaches at legacy16 or explicitly revised64/450-step boundaries. Numerical
    states are never fixed. Contact snapshots still occur every16 transitions.
    """
    if (not isinstance(weight, (int, float)) or isinstance(weight, bool)
            or not math.isfinite(weight) or weight < 0):
        raise ValueError('Navigation replay weight must be finite and nonnegative')
    states, frames, actions, target, grid = _batch_data(batch, context)
    observed = _observed_count(context, observed_context)
    actors, horizon = actions.shape[:2]
    if (not 1 <= actors <= 64 or horizon not in MOE_HORIZONS
            or expected_horizon is not None and expected_horizon != horizon):
        raise ValueError('Navigation replay requires the exact active MoE horizon and 1..64 actors')
    for label in (states, target):
        if bool((label[..., 12].ne(0) | label[..., 13].ne(0)).any()):
            raise ValueError('No-enemy replay rejects living NPC or positive-HP Source labels')
    previous_source = torch.cat((states[:, context-1:context], target[:, :-1]), 1)
    alive = (previous_source[..., 4] > 0) & (target[..., 4] > 0)
    denominator = (2*alive.sum()).clamp_min(1)
    geometry=prepare_player_geometry(previous_source,target,grid,player_navigation_protocol=player_navigation_protocol)
    tbptt=player_navigation_tbptt(player_navigation_protocol)
    geometry_metrics={};raw_xy=raw_wall=0.
    raw_player = raw_role = raw_balance = 0.
    modes = [(module, module.training) for module in model.modules()]
    try:
        with environment_profile(model, navigation_only=True):
            with torch.no_grad():
                fmap = model.encode_map(grid).detach()
            memory = _initialize(model, frames, states, fmap, context, observed)
            shape = tuple(memory.shape)
            if len(shape) != 2 or shape[0] != actors or not bool(torch.isfinite(memory).all()):
                raise FloatingPointError('Invalid initialized no-enemy replay memory')
            current = states[:, context-1]
            for first in range(0, horizon, tbptt):
                if first:
                    current, memory = current.detach(), memory.detach()
                chunk_state,chunk_memory=current,memory
                last = min(first+tbptt, horizon)
                predictions,full_predictions,role_terms,balance_terms,captures=[],[],[],[],[]
                for tick in range(first, last):
                    if player_contact_callback is not None and tick%16==0 and horizon-tick>=4:
                        captures.append((tick,current,memory))
                    result = model.transition(current.detach(), actions[:, tick], fmap, memory)
                    _valid_result(result, shape, 'No-enemy MoE navigation replay')
                    if result['state'].shape != (actors, 20):
                        raise ValueError('No-enemy replay changed its actor/state dimensions')
                    routers = _router_terms(result)
                    if bool((result['state'][:, 12].ne(0) | result['state'][:, 13].ne(0)).any()):
                        raise ValueError('No-enemy replay transition produced a living NPC/HP')
                    respawn = result.get('respawn_flags')
                    if not torch.is_tensor(respawn) or bool(respawn.any()):
                        raise ValueError('No-enemy replay must not produce a respawn')
                    result = integrate_coordinate_gradients(current, result, COORDINATE_GRADIENT_PROTOCOL)
                    current, memory = result['state'], result['memory']
                    predictions.append(current[:, :2])
                    if geometry is not None or player_contact_callback is not None:full_predictions.append(current)
                    role_terms.append(routers['router_role_loss']/horizon)
                    balance_terms.append(routers['router_balance_loss']/horizon)
                errors = F.smooth_l1_loss(torch.stack(predictions, 1).float()*16,
                    target[:, first:last, :2].detach().float()*16, reduction='none').sum(-1)
                player = (errors*alive[:, first:last]).sum()/denominator
                raw_xy+=float(player.detach())
                predicted=torch.stack(full_predictions,1)if full_predictions else None
                capture_player_chunks(player_contact_callback,captures,predicted,actions,grid,first=first)
                if geometry is not None:
                    wall,wall_metrics=player_geometry_terms(predicted,chunk_state,geometry,first=first)
                    player=player+wall;raw_wall+=float(wall.detach())
                    add_player_geometry_metrics(geometry_metrics,wall_metrics)
                role, balance = sum(role_terms), sum(balance_terms)
                loss = weight*(player + SETTINGS['router_role_weight']*role
                               + SETTINGS['router_balance_weight']*balance)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite no-enemy navigation replay loss')
                loss.backward()
                if check_gradients:
                    finite_gradient_norm(model.named_parameters(), context='No-enemy MoE replay')
                raw_player += float(player.detach())
                raw_role += float(role.detach())
                raw_balance += float(balance.detach())
                if progress_callback:
                    progress_callback(last, horizon)
    finally:
        # Preserve per-core argmax/sample mode even when a failed caller/helper
        # modified a child mode. Environment configs are restored by its context.
        for module, mode in modes:
            module.training = mode
    player = float(weight)*raw_player
    role = float(weight)*SETTINGS['router_role_weight']*raw_role
    balance = float(weight)*SETTINGS['router_balance_weight']*raw_balance
    report={'nav_objective_protocol': PROTOCOL,
        'nav_player_loss': player, 'nav_router_role_loss': role,
        'nav_router_balance_loss': balance, 'nav_loss': player+role+balance,
        'nav_raw_player_loss': raw_player, 'nav_raw_router_role_loss': raw_role,
        'nav_raw_router_balance_loss': raw_balance, 'nav_weight': float(weight),
        'nav_actor_count': int(actors), 'nav_horizon': int(horizon),
        'nav_forward_transitions': int(actors*horizon), 'nav_observed_context_frames': int(observed),
        'nav_player_axis_label_count': int(2*alive.sum().item()),
        'nav_memory_resets_after_initialization': 0, 'nav_npc_spawns': 0,
        'nav_source_oracle_calls': 0, 'nav_future_rgb_encoded': 0,
        'nav_enemy_loss': 0., 'nav_combat_loss': 0.,
        'source_future_feedback': False, 'nav_source_future_feedback': False,
        'nav_tbptt': tbptt, 'nav_environment_profile': 'navigation_no_enemies_v1'}
    if geometry is not None:
        report.update(nav_player_xy_loss=float(weight)*raw_xy,nav_player_wall_loss=float(weight)*raw_wall,
            nav_raw_player_xy_loss=raw_xy,nav_raw_player_wall_loss=raw_wall,
            nav_player_geometry=geometry_metrics,player_geometry_active=True)
    return report
