"""Globally normalized one-enemy combat labels for pre-RGB RSSM priors.

The caller supplies a separate Source-pose prior query. These helpers never
observe an image, call an engine/model, or replace the main rollout's state or
memory. Terms can be summed into independently detached posterior chunks.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from .aligned_training import motion_error_parts, spawn_supervision_statistics
from .combat_objectives import (DEFAULT_DAMAGE_WEIGHT, DEFAULT_DEATH_WEIGHT,
                                combat_transition_losses)
from .structured_state import ENTITY_DIM, ENTITY_SLOTS, GLOBAL_DIM, STATE_DIM
from .rssm_kinematic_losses import prepare_kinematic_supervision, kinematic_supervised_terms


def prepare_combat_supervision(batch, context=1, *, simulation_hz=30., none_class_weight=.2,
                               attack_pos_weight=4., npc_motion_weight=5., spawn_weight=1.,
                               damage_weight=DEFAULT_DAMAGE_WEIGHT, death_weight=DEFAULT_DEATH_WEIGHT,
                               game_training_v5=True, target_supervision='legacy', kinematic_protocol=None):
    """Prepare loss-only Source masks/denominators over the complete B×H tape."""
    if not isinstance(context, int) or isinstance(context, bool) or context < 1:
        raise ValueError('Combat context must be a positive integer')
    if target_supervision not in ('legacy', 'shot_distance_balanced'):
        raise ValueError('Unknown combat target supervision protocol')
    settings = dict(simulation_hz=simulation_hz, none_class_weight=none_class_weight,
                    attack_pos_weight=attack_pos_weight, npc_motion_weight=npc_motion_weight,
                    spawn_weight=spawn_weight, damage_weight=damage_weight, death_weight=death_weight)
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
           or not math.isfinite(value) or value < 0 for value in settings.values()) or simulation_hz <= 0:
        raise ValueError('Combat settings must be finite and nonnegative with positive simulation rate')
    states, actions, targets = (batch[name].detach().float() for name in ('states', 'actions', 'target_states'))
    if actions.ndim != 3:
        raise ValueError('Combat actions require[B,H,4]')
    count, horizon = actions.shape[:2]
    if (count < 1 or not 1 <= horizon <= 16 or states.ndim != 3 or states.shape[-1] != STATE_DIM
            or states.shape[0] != count or states.shape[1] < context+horizon or actions.shape[-1] != 4
            or targets.shape != (count, horizon, STATE_DIM)
            or not all(bool(torch.isfinite(value).all()) for value in (states, actions, targets))):
        raise ValueError('Combat supervision requires finite genuine Source context and1..16 transitions')
    previous = states[:, context-1:context-1+horizon]
    if not torch.equal(targets, states[:, context:context+horizon]):
        raise ValueError('Combat target states do not align with their Source frame/tick sequence')
    hits = batch['hit_ids'].detach().long()
    attacks, events = (batch[name].detach().float() for name in ('attack_flags', 'events'))
    if (hits.shape != (count, horizon) or attacks.shape != (count, horizon, ENTITY_SLOTS)
            or events.shape != (count, horizon, 4) or bool(((hits < -1) | (hits >= ENTITY_SLOTS)).any())
            or not all(bool(torch.isfinite(value).all()) for value in (attacks, events))
            or any(bool(((value < 0) | (value > 1)).any()) for value in (attacks, events))):
        raise ValueError('Combat event/hit/attack labels must match the one-enemy Source tape')
    old = previous[..., GLOBAL_DIM:-2].reshape(count, horizon, ENTITY_SLOTS, ENTITY_DIM)
    new = targets[..., GLOBAL_DIM:-2].reshape_as(old)
    old_alive = (old[..., 5] >= .5) & (old[..., 4] > 0)
    next_alive = (new[..., 5] >= .5) & (new[..., 4] > 0)
    player_alive = previous[..., 4] > 0
    continuing = old_alive & next_alive
    revived = ~old_alive & next_alive
    shot = events[..., 0] >= .5
    killable = old_alive & (old[..., 4] <= .35+1e-6) & shot[..., None]
    source_range = ((old[..., :2]-targets[..., None, :2])*16).square().sum(-1) <= 1.+1e-6
    eligible = (old_alive & next_alive & player_alive[..., None]
                & (old[..., 6]-(1/simulation_hz)/1.2 <= 1e-6) & source_range)
    positive = attacks > .5
    negative = eligible & ~positive
    positive_count, negative_count = positive.sum(), negative.sum()
    class_mass = attack_pos_weight*(positive_count > 0).float()+(negative_count > 0).float()
    supervision = {**settings, 'game_training_v5': bool(game_training_v5), 'context': context,
            'batch_size': count, 'horizon': horizon, 'previous': previous, 'target': targets,
            'actions': actions, 'hit_ids': hits, 'attack_flags': attacks, 'events': events,
            'old_alive': old_alive, 'player_alive': player_alive, 'continuing': continuing,
            'revived': revived, 'killable': killable,
            'old_alive_count': old_alive.sum(), 'player_alive_count': player_alive.sum(),
            'continuing_count': continuing.sum(), 'spawn_count': revived.sum(), 'killable_count': killable.sum(),
            'attack_positive_count': positive_count, 'attack_negative_count': negative_count,
            'attack_class_mass': class_mass.clamp_min(1e-6)}
    if kinematic_protocol is not None:
        supervision['kinematic_supervision'] = prepare_kinematic_supervision(previous, targets, actions,
            kinematic_protocol=kinematic_protocol, simulation_hz=simulation_hz)
    if target_supervision == 'shot_distance_balanced':
        hit = shot & (hits >= 0)
        miss = shot & (hits < 0)
        if bool(((hits >= 0) != (events[..., 1] >= .5)).any()) or bool(((hits >= 0) & ~shot).any()):
            raise ValueError('Source target hits require a real shot and matching recorded hit event')
        # Source fires after moving the player, before moving NPCs. Only
        # loss-side labels use these positions; no future pose enters a prior.
        selected = hits.clamp_min(0)[..., None, None].expand(count, horizon, 1, 2)
        hit_xy = old[..., :2].gather(2, selected).squeeze(2)
        distance = ((hit_xy-targets[..., :2])*16).norm(dim=-1)
        groups = {'near': hit & (distance < 2),
                  'mid': hit & (distance >= 2) & (distance < 5),
                  'far': hit & (distance >= 5)}
        group_counts = {name: mask.sum() for name, mask in groups.items()}
        supervision.update(target_supervision=target_supervision, target_shot=shot,
            target_hit=hit, target_fired_miss=miss, target_hit_groups=groups,
            target_hit_group_counts=group_counts, target_fired_miss_count=miss.sum(),
            target_present_hit_groups=sum((n > 0).float() for n in group_counts.values()).clamp_min(1),
            target_present_classes=((hit.any()).float()+(miss.any()).float()).clamp_min(1))
    return supervision


def combat_supervised_terms(prior, previous, target, action, *, supervision, index):
    """Return (additive differentiable terms, detached count diagnostics).

    All terms already have whole-tape B×H normalization: do not divide by H
    again. In particular, one rare spawn/attack keeps its mass across chunks.
    """
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < supervision['horizon']:
        raise ValueError('Combat timestep is outside the prepared Source tape')
    for supplied, expected, name in ((previous, supervision['previous'][:, index], 'previous'),
            (target, supervision['target'][:, index], 'target'),
            (action, supervision['actions'][:, index], 'action')):
        if supplied.shape != expected.shape or not torch.equal(supplied.detach().float(), expected):
            raise ValueError('Combat query is not aligned with its exact Source '+name)
    previous, target, action = (value.detach().float() for value in (previous, target, action))
    losses = combat_transition_losses(prior, previous, target,
        supervision['hit_ids'][:, index], supervision['attack_flags'][:, index], supervision['events'][:, index],
        action=action, dt=1/supervision['simulation_hz'], endless=True,
        none_class_weight=supervision['none_class_weight'], attack_pos_weight=supervision['attack_pos_weight'],
        damage_weight=supervision['damage_weight'], death_weight=supervision['death_weight'],
        game_training_v5=supervision['game_training_v5'])
    horizon = supervision['horizon']
    terms = {'combat_events': losses['events']/horizon, 'combat_target': losses['target']/horizon}
    target_diagnostics = {}
    if supervision.get('target_supervision') == 'shot_distance_balanced':
        labels = supervision['hit_ids'][:, index]
        labels = torch.where(labels < 0, torch.full_like(labels, ENTITY_SLOTS), labels)
        error = F.cross_entropy(prior['target_logits'].float(), labels, reduction='none')
        hit = sum((error*mask[:, index]).sum()/supervision['target_hit_group_counts'][name].clamp_min(1)
                  for name, mask in supervision['target_hit_groups'].items())/supervision['target_present_hit_groups']
        miss = (error*supervision['target_fired_miss'][:, index]).sum()/supervision['target_fired_miss_count'].clamp_min(1)
        mass = supervision['target_present_classes']
        terms['combat_target'] = (hit+miss)/mass
        target_diagnostics = {
            'combat_target_shot_count': supervision['target_shot'][:, index].sum().detach(),
            'combat_target_hit_count': supervision['target_hit'][:, index].sum().detach(),
            'combat_target_fired_miss_count': supervision['target_fired_miss'][:, index].sum().detach(),
            'combat_target_nonshot_ignored_count': (~supervision['target_shot'][:, index]).sum().detach(),
            'combat_target_hit_component': (hit/mass).detach(),
            'combat_target_fired_miss_component': (miss/mass).detach(),
            **{'combat_target_hit_'+name+'_count': mask[:, index].sum().detach()
               for name, mask in supervision['target_hit_groups'].items()}}
    if supervision['game_training_v5']:
        attack = (supervision['attack_pos_weight']*losses['attack_positive_sum']
                    /supervision['attack_positive_count'].clamp_min(1)
                  +losses['attack_negative_sum']/supervision['attack_negative_count'].clamp_min(1))
        terms['combat_attack'] = attack/supervision['attack_class_mass']
    else:
        terms['combat_attack'] = losses['attack']/horizon
    counts = {'old_alive': supervision['old_alive'][:, index].sum(),
              'player_alive': supervision['player_alive'][:, index].sum(),
              'continuing': supervision['continuing'][:, index].sum(),
              'killable': supervision['killable'][:, index].sum()}
    terms['combat_damage_npc'] = losses['damage_npc']*counts['old_alive']/supervision['old_alive_count'].clamp_min(1)
    terms['combat_damage_player'] = losses['damage_player']*counts['player_alive']/supervision['player_alive_count'].clamp_min(1)
    terms['combat_death_npc'] = losses['death_npc']*counts['killable']/supervision['killable_count'].clamp_min(1)
    _, npc = motion_error_parts(prior['state'], previous, target,
        simulation_hz=supervision['simulation_hz'], local_motion=True, action=action)
    terms['combat_npc_motion'] = (supervision['npc_motion_weight']*npc*counts['continuing']
                                /supervision['continuing_count'].clamp_min(1))
    spawn = spawn_supervision_statistics(prior, previous, target)
    terms['combat_spawn'] = supervision['spawn_weight']*spawn['spawn_sum']/supervision['spawn_count'].clamp_min(1)
    kinematic_diagnostics = {}
    if 'kinematic_supervision' in supervision:
        kinematic_terms, kinematic_diagnostics = kinematic_supervised_terms(prior['state'],
            supervision=supervision['kinematic_supervision'], index=index)
        terms.update(kinematic_terms)
    if not all(bool(torch.isfinite(value)) for value in terms.values()):
        raise FloatingPointError('Nonfinite aligned RSSM combat objective')
    diagnostics = {**target_diagnostics, **kinematic_diagnostics, 'combat_source_hit_count': (supervision['hit_ids'][:, index] >= 0).sum().detach(),
        'combat_attack_positive_count': (supervision['attack_flags'][:, index] > .5).sum().detach(),
        'combat_attack_negative_count': losses.get('attack_negative_count', prior['state'].new_zeros(())).detach(),
        'combat_death_opportunity_count': counts['killable'].detach(),
        'combat_spawn_event_count': spawn['spawn_count'].detach(),
        'combat_continuing_npc_count': counts['continuing'].detach(),
        'combat_spawn_anchor_ce': (spawn['spawn_anchor_sum']/supervision['spawn_count'].clamp_min(1)).detach(),
        'combat_spawn_fallback_ce': (spawn['spawn_fallback_sum']/supervision['spawn_count'].clamp_min(1)).detach()}
    return terms, diagnostics
