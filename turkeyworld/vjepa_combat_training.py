"""H180 event and motion supervision for the single-enemy V-JEPA prior."""
import torch
from torch.nn import functional as F

from .structured_state import ATTACK_INTERVAL, MELEE_DAMAGE, SHOT_DAMAGE
from .trajectory_objectives import circle_wall_penalty

from .vjepa_training_profiles import THREE_MODE, THREE_MODE_ACTORS

STREAM_ACTORS = {
    "motion_only": {"navigation": 8, "enemy_motion": 8},
    THREE_MODE: THREE_MODE_ACTORS,
}
EVENT_HEADS = tuple("vjepa_backend." + name + "." for name in
                    ("hit_target_head", "attack_head", "spatial_combat_head"))


def dynamics_parameter_trainable(name, three_mode):
    if name.startswith("vjepa_backend.core.encoder."):
        return (name.startswith("vjepa_backend.core.encoder.norm.") or
                name.startswith("vjepa_backend.core.encoder.blocks.") and int(name.split(".")[4]) >= 16)
    return (name.startswith(("vjepa_backend.core.predictor.", "vjepa_backend.readout_neck.",
                             "vjepa_backend.delta_head.", "vjepa_backend.map_projection.",
                             "vjepa_backend.spatial_visual_attention.", "vjepa_backend.spatial_motion_"))
            or three_mode and name.startswith(EVENT_HEADS))


def training_profile(requested, old):
    current = old.get("training_profile", "motion_only")
    profile = requested or current
    if profile not in STREAM_ACTORS or current not in STREAM_ACTORS:
        raise ValueError("Unknown dynamics training profile")
    if current == THREE_MODE and profile != current:
        raise ValueError("Resuming a three-mode optimizer requires its recorded training profile")
    return profile


def masked_mean(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1)


def balanced_mean(value, positive, negative):
    classes = positive.any().float() + negative.any().float()
    return (masked_mean(value, positive) + masked_mean(value, negative)) / classes.clamp_min(1)


def combat_npc_motion(predicted, target, initial, grid, wall_weight):
    """Compare matching lives only. A respawn teleport is never a velocity label."""
    previous = torch.cat((initial[:, None], target[:, :-1]), 1)
    own_previous = torch.cat((initial[:, None], predicted[:, :-1]), 1)
    own_alive, alive = predicted[..., 13] >= .5, target[..., 13] >= .5
    old_alive, old_own_alive = previous[..., 13] >= .5, own_previous[..., 13] >= .5
    source_respawn = (~old_alive & alive).long()
    own_respawn = (~old_own_alive & own_alive).long()
    same_generation = source_respawn.cumsum(1) == own_respawn.cumsum(1)
    continuing = (same_generation & alive & own_alive & old_alive & old_own_alive
                  & (previous[..., 4] > 0) & (own_previous[..., 4] > 0))
    xy = F.smooth_l1_loss(predicted[..., 8:10]*16, target[..., 8:10]*16, reduction="none").mean(-1)
    velocity = F.smooth_l1_loss((predicted[..., 8:10] - own_previous[..., 8:10])*480,
                                (target[..., 8:10] - previous[..., 8:10])*480, reduction="none").mean(-1)
    walls = circle_wall_penalty(predicted[..., 8:10]*16, grid, .25) / .25**2
    return masked_mean(xy, continuing) + .25*masked_mean(velocity, continuing) + wall_weight*masked_mean(walls, own_alive)


def combat_objective(target_logits, attack_logits, event_probs, batch, predicted):
    """Label the prior's raw head logits and its expected damage on the recorded commands.

    Conditional masks use original states only on the loss side. One comparison
    per fired shot prevents non-firing ticks from swamping hit supervision.
    """
    target, events = batch["target_states"], batch["events"]
    previous = torch.cat((batch["states"][:, :1], target[:, :-1]), 1)
    hit_id = batch["hit_ids"]
    shot, hit = events[..., 0] > .5, events[..., 1] > .5
    attack = batch["attack_flags"][..., 0] > .5
    if (hit_id.shape != shot.shape or target_logits.shape != (*shot.shape, 2)
            or attack_logits.shape != (*shot.shape, 1) or event_probs.shape != events.shape):
        raise ValueError("Combat labels must align with each autonomous transition")
    if not torch.equal(hit_id >= 0, hit) or not torch.equal(attack, events[..., 3] > .5):
        raise ValueError("Combat events disagree with recorded hit/attack identities")
    classes = torch.where(hit_id < 0, 1, hit_id).long()
    ce = F.cross_entropy(target_logits.float().transpose(1, 2), classes, reduction="none")
    target_loss = balanced_mean(ce, hit, shot & ~hit)
    separation = ((previous[..., 8:10] - target[..., :2])*16).norm(dim=-1)
    ready = ((previous[..., 4] > 0) & (previous[..., 13] >= .5) & (target[..., 13] >= .5)
             & (previous[..., 14] <= (1/30)/ATTACK_INTERVAL + 1e-6))
    nearby_negative = ready & (separation <= 1.5) & ~attack
    bce = F.binary_cross_entropy_with_logits(attack_logits.squeeze(-1).float(), attack.float(), reduction="none")
    attack_loss = balanced_mean(bce, attack, nearby_negative)
    # Soft expected damage from actual legally gated prior probabilities is
    # differentiable even though the production state update uses discrete events.
    npc_damage_unit = previous[..., 12].clamp(0, SHOT_DAMAGE)/SHOT_DAMAGE
    player_damage_unit = previous[..., 4].clamp(0, MELEE_DAMAGE)/MELEE_DAMAGE
    npc_delta = (previous[..., 12] - target[..., 12]).clamp_min(0)/SHOT_DAMAGE
    player_delta = (previous[..., 4] - target[..., 4]).clamp_min(0)/MELEE_DAMAGE
    hit_error = (event_probs[..., 1]*npc_damage_unit - npc_delta).square()
    attack_error = (event_probs[..., 3]*player_damage_unit - player_delta).square()
    damage_loss = .5*(balanced_mean(hit_error, hit, shot & ~hit)
                      + balanced_mean(attack_error, attack, nearby_negative))
    hp_mae = (predicted[..., [4, 12]].detach() - target[..., [4, 12]]).abs().mean()
    return {
        "target_ce": target_loss, "attack_bce": attack_loss, "damage_loss": damage_loss,
        "hp_mae": hp_mae, "combat_total": target_loss + attack_loss + .25*damage_loss,
    }
