"""HP-derived life descriptors and aligned combat-only training objectives.

These descriptors are nominal damage equivalents, not observed historical hit
counts. Source augmentation can initialize injured/dead NPCs, and terminal HP
does not reveal overkill. No new state field, engine call or future observation
is introduced. This module never changes runtime state or predicts sprite yaw.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from .structured_state import (SHOT_DAMAGE, MELEE_DAMAGE, FIRE_INTERVAL,
                               ATTACK_INTERVAL, HURT_DURATION)


NPC_INITIAL_HP = .60             # Source enemies start/respawn at60HP, not100.
PLAYER_INITIAL_HP = 1.
DEFAULT_DAMAGE_WEIGHT = 1.
DEFAULT_DEATH_WEIGHT = .25
COMBAT_COUNT_FEATURE_DIM = 6
COMBAT_FEATURE_DIM = COMBAT_COUNT_FEATURE_DIM
COMBAT_COUNT_FEATURE_NAMES = ('nominal_hits_received', 'nominal_hits_fraction',
    'continuous_damage_equivalent', 'remaining_hits_to_death', 'hp_ladder_valid', 'alive')
COMBAT_COUNT_FEATURE_SLICES = {name: slice(index, index+1)
                              for index, name in enumerate(COMBAT_COUNT_FEATURE_NAMES)}


def _check_state(state):
    if not torch.is_tensor(state) or state.ndim != 2 or state.shape[1] != 90 or not state.is_floating_point():
        raise ValueError('Combat state must be floating point[B,90]')


def combat_count_features(state):
    """Return[B,9,6], player then8NPCs, with straight-through nominal rounding.

    NPC60→25→0HP means nominal0→1→2hits; respawn60HP resets to0.
    Player100HP with8HP melee means nominal13hits to death. HP35 source
    augmentation is not on the NPC60−35n ladder and is explicitly flagged.
    Even a ladder-valid HP cannot prove actual prior hits in augmented data.
    Discrete life/validity flags have no surrogate boolean gradients; count
    derivatives use continuous HP damage equivalents only.
    """
    _check_state(state)
    dtype = torch.float64 if state.dtype == torch.float64 else torch.float32
    values = state.to(dtype=dtype)
    npc = values[:, 8:88].reshape(-1, 8, 10)
    hp = torch.cat([values[:, 4:5], npc[:, :, 4]], -1)
    alive = torch.cat([values[:, 4:5] > 0, (npc[:, :, 5] >= .5) & (npc[:, :, 4] > 0)], -1)
    initial = hp.new_tensor([PLAYER_INITIAL_HP]+[NPC_INITIAL_HP]*8)[None]
    damage = hp.new_tensor([MELEE_DAMAGE]+[SHOT_DAMAGE]*8)[None]
    maximum = (initial/damage).ceil()
    continuous = ((initial-hp)/damage).clamp_min(0)
    continuous = torch.minimum(continuous, maximum)
    rounded = continuous.round()
    hard = torch.where(alive, rounded, maximum.expand_as(hp))
    nominal = hard.detach()+(continuous-continuous.detach())
    remaining_soft = hp.clamp_min(0)/damage
    remaining_hard = torch.where(alive, (remaining_soft-1e-6).ceil().clamp_min(0), torch.zeros_like(hp))
    remaining = remaining_hard.detach()+(remaining_soft-remaining_soft.detach())
    reconstructed = (initial-damage*hard).clamp_min(0)
    valid = ((hp-reconstructed).abs() <= 2e-5) & (hp >= -1e-6) & (hp <= initial+1e-6)
    # A dead flag with positive HP is malformed, not a known damaged life.
    valid = valid & (alive | (hp <= 1e-6))
    return torch.stack([nominal, nominal/maximum, continuous, remaining,
                         valid.to(dtype), alive.to(dtype)], -1)


def _shot_gate(result, previous, action, dt, endless):
    if 'shot' in result:
        return result['shot'].detach().to(previous.dtype)
    if action is not None:
        if action.shape != (previous.shape[0], 4):
            raise ValueError('Combat action must have shape[B,4]')
        ready = previous[:, 6]-dt/FIRE_INTERVAL <= 1e-6
        ammo = (previous[:, 5]*99).round() >= 1
        return ((action[:, 3] >= .5) & (previous[:, 4] > 0) & ready & (ammo | bool(endless))).to(previous.dtype)
    if 'events' in result:
        return result['events'][:, 0].detach().to(previous.dtype)
    raise ValueError('Combat probability objectives require an own predicted shot gate or controls')


def _attack_count_distribution(probabilities):
    """Exact differentiable Poisson-binomial count distribution[B,9]."""
    distribution = probabilities.new_ones((len(probabilities), 1))
    for probability in probabilities.unbind(-1):
        p = probability[:, None]
        zero = torch.zeros_like(distribution[:, :1])
        distribution = torch.cat([distribution*(1-p), zero], -1)+torch.cat([zero, distribution*p], -1)
    return distribution


def combat_image_state(result, previous, target_pose, *, action=None, dt=1/30, endless=True):
    """Auxiliary GT-pose render condition with own hard combat in the forward pass.

    All labels/poses and hard accepted state are detached. Only own combat-head
    probabilities supply derivatives, including at lethal HP clamping and hard
    life/hurt decisions. This tensor is a training-only render condition and
    must never replace the autoregressive world state. Counts remain HP-derived
    nominal equivalents, not a new per-life history ledger. Respawned slots have
    their physical accepted values but no combat surrogate on the spawn tick.
    """
    _check_state(previous); _check_state(target_pose); _check_state(result['state'])
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('Combat timestep must be positive and finite')
    previous = previous.detach().float()
    pose = target_pose.detach().float()
    hard = result['state'].detach().float()
    targets = result['target_logits'].float()
    attacks = result['attack_logits'].float()
    batch = len(previous)
    if targets.shape != (batch, 9) or attacks.shape != (batch, 8):
        raise ValueError('Combat heads must provide target_logits[B,9] and attack_logits[B,8]')
    old = previous[:, 8:88].reshape(batch, 8, 10)
    accepted = hard[:, 8:88].reshape_as(old)
    base = pose[:, 8:88].reshape_as(old)
    old_alive = (old[:, :, 5] >= .5) & (old[:, :, 4] > 0)
    respawn = result.get('respawn_flags', torch.zeros_like(old_alive)).detach().bool()
    valid = old_alive & ~respawn
    shot = _shot_gate(result, previous, action, dt, endless)
    hit_probability = targets.softmax(-1)[:, :8]*shot[:, None]*valid.to(targets.dtype)
    lethal_probability = hit_probability*(old[:, :, 4] <= SHOT_DAMAGE+1e-6)
    npc_hp = old[:, :, 4].clamp_min(0)-hit_probability*old[:, :, 4].clamp(0, SHOT_DAMAGE)
    alive = old_alive.to(targets.dtype)-lethal_probability
    decayed_hurt = (old[:, :, 7].clamp(0, 1)-dt/HURT_DURATION).clamp_min(0)
    hurt = decayed_hurt+(1-decayed_hurt)*hit_probability
    # A new death sets its clock to zero; old corpse clocks remain physical.
    # The differentiable life flag carries newly-dead sprite visibility.
    death = old[:, :, 8].clamp_min(0)*(1-lethal_probability)
    separation = (old[:, :, :2]-hard[:, None, :2])*16
    eligible = (valid & (accepted[:, :, 5] >= .5) & (previous[:, 4:5] > 0)
                & (old[:, :, 6]-dt/ATTACK_INTERVAL <= 1e-6)
                & (separation.square().sum(-1) <= 1.))
    attack_probability = attacks.sigmoid()*eligible.to(attacks.dtype)
    distribution = _attack_count_distribution(attack_probability)
    nominal_damage = torch.arange(9, device=hard.device, dtype=hard.dtype)[None]*MELEE_DAMAGE
    actual_damage = torch.minimum(nominal_damage, previous[:, 4:5].clamp_min(0))
    player_hp = previous[:, 4]-(distribution*actual_damage).sum(-1)

    def physical_forward(value, surrogate, mask=None):
        if mask is not None:
            surrogate = surrogate*mask.to(surrogate.dtype)
        return value+(surrogate-surrogate.detach())

    entities = torch.cat([base[:, :, :4],
        physical_forward(accepted[:, :, 4], npc_hp, valid)[:, :, None],
        physical_forward(accepted[:, :, 5], alive, valid)[:, :, None],
        accepted[:, :, 6:7],
        physical_forward(accepted[:, :, 7], hurt, valid)[:, :, None],
        physical_forward(accepted[:, :, 8], death, valid)[:, :, None],
        base[:, :, 9:10]], -1)
    globals_ = torch.cat([pose[:, :4], physical_forward(hard[:, 4], player_hp)[:, None],
                          pose[:, 5:7], hard[:, 7:8]], -1)
    return torch.cat([globals_, entities.flatten(1), pose[:, 88:90]], -1)


def combat_transition_losses(result, previous, target, hit_ids, attack_flags, events, *,
                              action=None, dt=1/30, endless=True,
                              none_class_weight=.2, attack_pos_weight=4.,
                              damage_weight=DEFAULT_DAMAGE_WEIGHT,
                              death_weight=DEFAULT_DEATH_WEIGHT):
    """Additive aligned loss terms; labels never enter a prediction/state path.

    Existing events/target/attack BCE/CE weights are unchanged. New damage terms
    compare expected actual HP loss in one-hit units, including lethal clipping.
    NPC death uses stable one-vs-rest categorical log probabilities when an own
    legal shot can kill a previously living slot. No gradient is fabricated
    through hard alive/death flags. Prior-dead and respawned slots are excluded.
    Player death is intentionally not given a separate ambiguous boolean loss;
    its clipped damage expectation already trains the attacking heads.
    """
    _check_state(previous); _check_state(target); _check_state(result['state'])
    if (not math.isfinite(dt) or dt <= 0 or any(not math.isfinite(value) or value < 0
            for value in (none_class_weight, attack_pos_weight, damage_weight, death_weight))):
        raise ValueError('Combat timestep/weights must be finite and nonnegative')
    previous, target = previous.detach().float(), target.detach().float()
    targets = result['target_logits'].float()
    attacks = result['attack_logits'].float()
    batch = previous.shape[0]
    if targets.shape != (batch, 9) or attacks.shape != (batch, 8):
        raise ValueError('Combat heads must provide target_logits[B,9] and attack_logits[B,8]')
    zero = targets.sum()*0+attacks.sum()*0
    losses = {'events': zero, 'target': zero, 'attack': zero,
              'damage_npc': zero, 'damage_player': zero, 'death_npc': zero}
    if 'event_logits' in result:
        losses['events'] = F.binary_cross_entropy_with_logits(result['event_logits'], events.detach())
    labels = torch.where(hit_ids < 0, torch.full_like(hit_ids, 8), hit_ids).detach().long()
    class_weights = targets.new_ones(9); class_weights[-1] = none_class_weight
    losses['target'] = F.cross_entropy(targets, labels, weight=class_weights, reduction='none').mean()
    losses['attack'] = F.binary_cross_entropy_with_logits(attacks, attack_flags.detach(),
        pos_weight=attacks.new_full((8,), attack_pos_weight))
    old = previous[:, 8:88].reshape(batch, 8, 10)
    expected = target[:, 8:88].reshape_as(old)
    predicted = result['state'][:, 8:88].reshape_as(old)
    old_alive = (old[:, :, 5] >= .5) & (old[:, :, 4] > 0)
    if 'respawn_flags' in result:
        old_alive = old_alive & ~result['respawn_flags'].detach().bool()
    old_hp = old[:, :, 4].clamp_min(0)
    shot = _shot_gate(result, previous, action, dt, endless)
    probabilities = targets.softmax(-1)[:, :8]*shot[:, None]*old_alive.to(targets.dtype)
    expected_npc_damage = probabilities*old_hp.clamp_max(SHOT_DAMAGE)
    observed_npc_damage = (old_hp-expected[:, :, 4]).clamp_min(0)
    npc_error = F.smooth_l1_loss(expected_npc_damage/SHOT_DAMAGE,
                                observed_npc_damage/SHOT_DAMAGE, reduction='none')
    losses['damage_npc'] = damage_weight*(npc_error*old_alive).sum()/old_alive.sum().clamp_min(1)

    # Reuse disclosed accepted-state topology/range/cooldown rules, without
    # querying source collision/visibility or using target attack labels here.
    player_alive = previous[:, 4] > 0
    separation = (old[:, :, :2]-result['state'][:, None, :2])*16
    ready = old[:, :, 6]-dt/ATTACK_INTERVAL <= 1e-6
    eligible = (old_alive & (predicted[:, :, 5].detach() >= .5) & player_alive[:, None]
                & ready & (separation.detach().square().sum(-1) <= 1.))
    attack_probabilities = attacks.sigmoid()*eligible.to(attacks.dtype)
    distribution = _attack_count_distribution(attack_probabilities)
    nominal = torch.arange(9, device=previous.device, dtype=previous.dtype)[None]*MELEE_DAMAGE
    capped = torch.minimum(nominal, previous[:, 4:5].clamp_min(0))
    expected_player_damage = (distribution*capped).sum(-1)
    observed_player_damage = (previous[:, 4]-target[:, 4]).clamp_min(0)
    player_error = F.smooth_l1_loss(expected_player_damage/MELEE_DAMAGE,
                                   observed_player_damage/MELEE_DAMAGE, reduction='none')
    losses['damage_player'] = damage_weight*(player_error*player_alive).sum()/player_alive.sum().clamp_min(1)

    killable = old_alive & (old_hp <= SHOT_DAMAGE+1e-6) & (shot[:, None] > .5)
    death_labels = (expected[:, :, 5] < .5).to(targets.dtype)
    log_z = targets.logsumexp(-1, keepdim=True)
    log_p = targets[:, :8]-log_z
    diagonal = torch.arange(9, device=targets.device)[None, :] == torch.arange(8, device=targets.device)[:, None]
    log_not_p = targets[:, None, :].masked_fill(diagonal[None], -torch.inf).logsumexp(-1)-log_z
    death_error = -death_labels*log_p-(1-death_labels)*log_not_p
    losses['death_npc'] = death_weight*(death_error*killable).sum()/killable.sum().clamp_min(1)
    return losses
