"""Disclosed event-based state bookkeeping for the ``event_v1`` world model.

There is no engine import, wall query or raycast here. Networks propose XY
motion, one shot target (eight persistent IDs plus ``none``), and eight attack
logits. This module applies the game's explicit kinematic and combat rules so
small regression errors cannot silently consume HP, ammunition or life flags.
It does not guarantee that a learned move avoids walls or a hit sees its target.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

GLOBAL_DIM = 8
ENTITY_DIM = 10
ENTITY_SLOTS = 1
STATE_DIM = GLOBAL_DIM + ENTITY_DIM*ENTITY_SLOTS + 2
STATE_SCHEMA_VERSION = "single_enemy_v1"
ACTION_DIM = 4
MAP_EXTENT = 16.0
SHOT_DAMAGE = 0.35
MELEE_DAMAGE = 0.08
FIRE_INTERVAL = 0.22
FLASH_DURATION = 0.10
ATTACK_INTERVAL = 1.20
HURT_DURATION = 0.15
DEATH_DURATION = 0.60
RESPAWN_DELAY = 3.0
PLAYER_SPEED = 2.4
NPC_SPEED_BOUND = 1.2
TURN_SPEED = 2.2
EVENT_NAMES = ("shot", "hit", "kill", "player_hit")
STRUCTURED_STATE_VERSION = "event_v1"


def _bounded_delta(previous_xy, candidate_xy, speed, dt, *, training=False):
    """Hard speed cap with an optional straight-through training derivative.

    Radial clipping otherwise has zero radial derivative for an overspeed
    proposal, so a stopped or slower NPC target cannot teach it to slow down.
    The forward displacement and inference rule remain the same hard bound.
    """
    delta = candidate_xy.clamp(0, 1) - previous_xy
    tiny = max(torch.finfo(delta.dtype).tiny, 1e-6)
    norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    factor = ((speed * dt / MAP_EXTENT) / norm.clamp_min(tiny)).clamp_max(1)
    bounded = delta * factor
    return bounded.detach() + (delta - delta.detach()) if training else bounded


def _straight_through(hard, soft, training):
    return hard + (soft - soft.detach()) if training else hard


def structured_transition(previous, candidate, action, target_logits,
                          attack_logits, dt=1 / 30, training=False,
                          endless=False, spawn_positions=None, spawn_valid=None, respawn_enabled=True):
    """Return a coherent 20D state and accepted events without a simulator.

    Args:
        previous, candidate: float tensors [B,20]. Only candidate player/NPC XY
            coordinates are used. Heading, health, flags, ammunition, timers,
            animation phase, kills and time use the rules stated in this file.
        action: [B,4] = forward, strafe, turn, fire. Turn is integrated exactly;
            zero movement inputs preserve player XY regardless of candidate.
            Endless mode also bounds each proposed XY component by the signed
            command after turning. Walls may reduce a component; the network
            cannot invent movement along an uncommanded or opposite axis.
        target_logits: [B,2]; class0 identifies the only NPC, class1 means no hit.
            Argmax chooses a target; choosing an already dead target is rejected.
        attack_logits: [B,1]; >=0 proposes an attack. Range, life, terminal state
            and cooldown guards must also hold. Occlusion remains learned.
        dt: positive finite seconds, normally 1/30; rules are otherwise rate aware.
        training: preserve hard forward decisions but use straight-through
            gradients through NPC speed clipping and softmax/sigmoid damage.
            Train the two heads with their target-CE/attack-BCE labels;
            ``events`` itself is hard-valued.

    Return values include state[B,20], events[B,4], event_probs/event_logits,
    accepted_target[B] (-1 for no accepted hit), accepted_attacks[B,1] bool,
    shot/hit/kill/player_hit[B] bool, npc_damage[B,1] and player_damage[B].
    Half/bfloat16 inputs are promoted to authoritative float32 output state;
    network logits can have a different floating dtype on the same device.
    Damage values are normalized by 100 HP; player_damage is nominal summed
    attack damage and may exceed the HP actually remaining before clamping.
    """
    if not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    if previous.ndim != 2 or previous.shape[-1] != STATE_DIM:
        raise ValueError("previous must have shape [B,20]")
    batch = previous.shape[0]
    for name, value, shape in (
        ("candidate", candidate, (batch, STATE_DIM)),
        ("action", action, (batch, ACTION_DIM)),
        ("target_logits", target_logits, (batch, ENTITY_SLOTS + 1)),
        ("attack_logits", attack_logits, (batch, ENTITY_SLOTS)),
    ):
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if value.device != previous.device:
            raise ValueError(f"{name} must share previous's device")
        if not value.is_floating_point():
            raise ValueError(f"{name} must be floating point")
    if not previous.is_floating_point():
        raise ValueError("previous must be floating point")
    # NN weights/logits may use FP16, but cumulative state must not. In half
    # precision time += 1/9000 stalls near t=0.25 (75 seconds), and small pose
    # increments disappear. Return authoritative FP32 state for half inputs;
    # callers must retain it rather than cast it back to network precision.
    dtype = torch.float32 if previous.dtype in (torch.float16, torch.bfloat16) else previous.dtype
    previous, candidate, action, target_logits, attack_logits = (
        tensor.to(dtype=dtype) for tensor in (previous, candidate, action, target_logits, attack_logits)
    )
    old = previous[:, GLOBAL_DIM:-2].reshape(batch, ENTITY_SLOTS, ENTITY_DIM)
    proposed = candidate[:, GLOBAL_DIM:-2].reshape_as(old)
    original_alive = (old[:, :, 5] >= 0.5) & (old[:, :, 4] > 0)
    respawn = torch.zeros_like(original_alive)
    if endless and respawn_enabled and spawn_positions is not None:
        if spawn_positions.shape != (batch, ENTITY_SLOTS, 2):
            raise ValueError("spawn_positions must have shape [B,1,2]")
        if spawn_valid is None or spawn_valid.shape != (batch, ENTITY_SLOTS):
            raise ValueError("spawn_valid must have shape [B,1]")
        respawn = (~original_alive & spawn_valid.bool()
                   & (old[:, :, 8] * DEATH_DURATION + dt >= RESPAWN_DELAY - 1e-6)
                   & (previous[:, 4:5] > 0))
        fresh = torch.cat([spawn_positions.to(dtype=dtype).clamp(0, 1),
                           torch.zeros_like(old[:, :, 2:4]),
                           torch.full_like(old[:, :, 4:5], .60),
                           torch.ones_like(old[:, :, 5:6]), torch.zeros_like(old[:, :, 6:])], -1)
        old = torch.where(respawn[:, :, None], fresh, old)
        # A new slot starts at its decoded center; learned motion resumes on
        # the following tick. It is never bounded from its previous corpse.
        proposed = torch.cat([torch.where(respawn[:, :, None], old[:, :, :2], proposed[:, :, :2]),
                              proposed[:, :, 2:]], -1)
    # Valid event_v1 states have exact binary life flags. These guards also
    # prevent malformed dead-flag/positive-HP inputs from resurrecting an ID.
    old_alive = (old[:, :, 5] >= 0.5) & (old[:, :, 4] > 0)
    old_hp = torch.where(old_alive, old[:, :, 4].clamp(0, 1), torch.zeros_like(old[:, :, 4]))
    player_hp_before = previous[:, 4].clamp(0, 1)
    player_alive = player_hp_before > 0
    controls = torch.cat([action[:, :3].clamp(-1, 1), action[:, 3:4].clamp(0, 1)], dim=-1)

    # Player pose: learned bounded XY, exact disclosed angular integration.
    movement_requested = controls[:, :2].ne(0).any(dim=-1) & player_alive
    player_delta = _bounded_delta(previous[:, :2], candidate[:, :2], PLAYER_SPEED, dt)
    player_xy = previous[:, :2] + player_delta * movement_requested[:, None].to(dtype)
    heading = previous[:, 2:4]
    norm = torch.linalg.vector_norm(heading, dim=-1, keepdim=True)
    safe = norm.clamp_min(max(torch.finfo(dtype).tiny, 1e-6))
    fallback = torch.stack([torch.zeros_like(norm[:, 0]), torch.ones_like(norm[:, 0])], dim=-1)
    heading = torch.where(norm > 1e-6, heading / safe, fallback)
    angle_step = controls[:, 2] * (TURN_SPEED * dt)
    turn_sin, turn_cos = angle_step.sin(), angle_step.cos()
    turned = torch.stack([heading[:, 0] * turn_cos + heading[:, 1] * turn_sin,
                          heading[:, 1] * turn_cos - heading[:, 0] * turn_sin], dim=-1)
    turned = turned / torch.linalg.vector_norm(turned, dim=-1, keepdim=True).clamp_min(1e-6)
    new_heading = torch.where(player_alive[:, None], turned, previous[:, 2:4])
    if endless:
        # A disclosed kinematic bound, not a collision query. The learned XY
        # proposal still determines how much commanded movement is blocked.
        # Per-axis limits already imply the Euclidean player speed limit.
        movement = controls[:, :2]
        magnitude = torch.linalg.vector_norm(movement, dim=-1, keepdim=True).clamp_min(1.)
        forward, strafe = (movement / magnitude).unbind(-1)
        commanded = torch.stack([new_heading[:, 1]*forward-new_heading[:, 0]*strafe,
                                 new_heading[:, 0]*forward+new_heading[:, 1]*strafe], -1)
        commanded = commanded * (PLAYER_SPEED * dt / MAP_EXTENT)
        proposed_delta = candidate[:, :2].clamp(0, 1)-previous[:, :2]
        zero = torch.zeros_like(commanded)
        constrained = torch.maximum(torch.minimum(proposed_delta, torch.maximum(commanded, zero)),
                                     torch.minimum(commanded, zero))
        # Preserve gradients to an incorrectly signed/oversized NN proposal.
        # Without this straight-through term clipping can strand it at zero.
        player_delta = (proposed_delta + (constrained-proposed_delta).detach()
                        if training else constrained)
        player_xy = previous[:, :2]+player_delta*movement_requested[:, None].to(dtype)

    fire_cd = (previous[:, 6].clamp(0, 1) - dt / FIRE_INTERVAL).clamp_min(0)
    flash = (previous[:, 7].clamp(0, 1) - dt / FLASH_DURATION).clamp_min(0)
    # Ammunition is an integer count stored in a normalized field. Subtracting
    # 1/99 from a float repeatedly strands the last bullet below its threshold.
    ammo_count = (previous[:, 5].clamp(0, 1) * 99).round()
    shot = ((controls[:, 3] >= 0.5) & player_alive &
            ((ammo_count >= 1) | bool(endless)) & (fire_cd <= 1e-6))
    shot_float = shot.to(dtype)
    ammo = (ammo_count if endless else ammo_count - shot_float) / 99
    fire_cd = torch.where(shot, torch.ones_like(fire_cd), fire_cd)
    flash = torch.where(shot, torch.ones_like(flash), flash)

    # A shot has exactly one categorical target, or none. Rejected target IDs
    # do not turn into hits on a different enemy. No geometry oracle is queried.
    target_probs = target_logits.softmax(dim=-1)
    chosen = target_logits.argmax(dim=-1)
    hard_target = F.one_hot(chosen, ENTITY_SLOTS + 1).to(dtype)
    selected = _straight_through(hard_target, target_probs, training)
    valid_target = old_alive.to(dtype)
    hit_weights = selected[:, :ENTITY_SLOTS] * valid_target * shot_float[:, None]
    hard_hit_weights = hard_target[:, :ENTITY_SLOTS] * valid_target * shot_float[:, None]
    hit = hard_hit_weights.sum(dim=-1) > 0
    accepted_target = torch.where(hit, chosen, torch.full_like(chosen, -1))
    npc_damage = SHOT_DAMAGE * hit_weights
    npc_hp = (old_hp - npc_damage).clamp_min(0)
    alive = old_alive & (npc_hp.detach() > 0)
    newly_dead = old_alive & ~alive
    kill = newly_dead.any(dim=-1)

    # NPC motion occurs after the shot. Killed or previously dead slots keep
    # their old location; terminal player states also stop NPC movement.
    npc_delta = _bounded_delta(old[:, :, :2], proposed[:, :, :2], NPC_SPEED_BOUND, dt,
                               training=training)
    # A freshly respawned slot holds its center and phase for this tick, as
    # does the deterministic source. Learned motion starts on the next tick.
    npc_moving = alive & player_alive[:, None] & ~respawn
    npc_delta = npc_delta * npc_moving[:, :, None].to(dtype)
    npc_xy = old[:, :, :2] + npc_delta
    # Schema stores vx/2 and vy/2, while XY is normalized by map extent.
    npc_velocity = npc_delta * (MAP_EXTENT / (2 * dt))

    hurt = (old[:, :, 7].clamp(0, 1) - dt / HURT_DURATION).clamp_min(0)
    hurt = torch.where(hard_hit_weights > 0, torch.ones_like(hurt), hurt)
    death_limit = RESPAWN_DELAY / DEATH_DURATION if endless else 1.0
    death = torch.where(~old_alive,
                        (old[:, :, 8].clamp(0, death_limit) + dt / DEATH_DURATION).clamp_max(death_limit),
                        old[:, :, 8].clamp(0, death_limit))
    death = torch.where(newly_dead, torch.zeros_like(death), death)
    phase = torch.where(npc_moving, torch.remainder(old[:, :, 9] + 2.6 * dt, 1), old[:, :, 9])
    attack_cd = (old[:, :, 6].clamp(0, 1) - dt / ATTACK_INTERVAL).clamp_min(0)

    # The reference game's melee range is evaluated BEFORE NPC movement but
    # AFTER player movement. Visibility is supplied by the learned attack head.
    separation = (old[:, :, :2] - player_xy[:, None, :]) * MAP_EXTENT
    in_range = separation.square().sum(dim=-1) <= 1.0
    eligible_attack = alive & player_alive[:, None] & (attack_cd <= 1e-6) & in_range
    attack_probs = attack_logits.sigmoid()
    accepted_attacks = (attack_logits >= 0) & eligible_attack
    attack_weights = _straight_through(accepted_attacks.to(dtype),
                                      attack_probs * eligible_attack.to(dtype), training)
    player_damage = MELEE_DAMAGE * attack_weights.sum(dim=-1)
    player_hp = (player_hp_before - player_damage).clamp_min(0)
    attack_cd = torch.where(accepted_attacks, torch.ones_like(attack_cd), attack_cd)
    player_hit = accepted_attacks.any(dim=-1)

    globals_ = torch.cat([player_xy, new_heading, player_hp[:, None], ammo[:, None],
                          fire_cd[:, None], flash[:, None]], dim=-1)
    entities = torch.cat([npc_xy, npc_velocity, npc_hp[:, :, None], alive.to(dtype)[:, :, None],
                          attack_cd[:, :, None], hurt[:, :, None], death[:, :, None],
                          phase[:, :, None]], dim=-1)
    kills = (previous[:, -2].clamp_min(0) + newly_dead.to(dtype).sum(-1) / ENTITY_SLOTS
             if endless else (~alive).to(dtype).sum(dim=-1) / ENTITY_SLOTS)
    # Time is normalized by 300 but deliberately NOT clipped at 300 seconds.
    elapsed = previous[:, -1].clamp_min(0) + dt / 300
    state = torch.cat([globals_, entities.flatten(1), kills[:, None], elapsed[:, None]], dim=-1)
    events = torch.stack([shot, hit, kill, player_hit], dim=-1).to(dtype)

    # Diagnostics/loss-compatible probabilities derived from the same legal
    # action and eligibility masks. There is no independent event classifier.
    hit_probability = shot_float * (target_probs[:, :ENTITY_SLOTS] * valid_target).sum(dim=-1)
    killable = old_alive & (old_hp <= SHOT_DAMAGE)
    kill_probability = shot_float * (target_probs[:, :ENTITY_SLOTS] * killable.to(dtype)).sum(dim=-1)
    attack_probability = attack_probs * eligible_attack.to(dtype)
    player_hit_probability = 1 - (1 - attack_probability).prod(dim=-1)
    event_probs = torch.stack([shot_float, hit_probability, kill_probability, player_hit_probability], dim=-1)
    eps = max(torch.finfo(dtype).eps, 1e-6)
    event_logits = torch.logit(event_probs.clamp(eps, 1 - eps))
    return {
        "state": state, "events": events, "event_probs": event_probs,
        "event_logits": event_logits, "target_logits": target_logits,
        "attack_logits": attack_logits, "target_probs": target_probs,
        "attack_probs": attack_probs, "accepted_target": accepted_target,
        "accepted_attacks": accepted_attacks, "shot": shot, "hit": hit,
        "kill": kill, "player_hit": player_hit, "npc_damage": npc_damage,
        "player_damage": player_damage, "damage": npc_damage,
        "newly_dead": newly_dead, "respawn_flags": respawn,
        "player_displacement": player_xy - previous[:, :2],
        "npc_displacement": npc_delta,
    }
