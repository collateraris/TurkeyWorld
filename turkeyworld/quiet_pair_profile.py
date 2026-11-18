"""Per-tick eligibility and checkpoint provenance of learned quiet player/NPC motion."""
import hashlib

import torch

from .learned_axis_collision import MODE_NAV, PREFIX as NAV_PREFIX, PROTOCOL as NAV_PROTOCOL, head_digest
from .structured_state import MAP_EXTENT, PLAYER_SPEED

MODE = "learned_axis_quiet_pair"
PLAYER_PREFIX = "vjepa_backend.quiet_player_head."
NPC_PREFIX = "vjepa_backend.quiet_pursuit_head."
PREFIXES = (PLAYER_PREFIX, NPC_PREFIX)
PROTOCOL = "learned_quiet_pair_v1"


def quiet_mask(state, action, dt):
    """Decide from the current state whether the upcoming tick is free of combat.

    The typed transition checks melee after moving the player. Outside the sum
    of melee range and maximum player travel, melee cannot occur on this tick.
    """
    distance_squared = (state[:, 8:10] - state[:, :2]).square().sum(-1) * MAP_EXTENT**2
    alive = (state[:, 4] > 0) & (state[:, 12] > 0) & (state[:, 13] >= 0.5)
    clear_timers = (state[:, 6:8] == 0).all(1) & (state[:, 14:17] == 0).all(1)
    return (action[:, 3] < 0.5) & alive & clear_timers & (distance_squared > (1 + PLAYER_SPEED * dt) ** 2)


def _fingerprint(weights):
    digest = hashlib.sha256()
    for key, tensor in sorted(weights.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _weights(player, npc):
    return {
        prefix + name: value
        for prefix, head in zip(PREFIXES, (player, npc))
        for name, value in head.state_dict().items()
    }


def with_quiet_heads(saved, player, npc):
    config = saved["config"]
    if config.get("architecture") != "vjepa2_ac" or config.get("player_motion_mode") != MODE_NAV:
        raise ValueError("Expected the previously qualified V-JEPA NAV-only parent")
    nav = saved["local_axis_head_migration"]
    if not (nav["target_mode"] == MODE_NAV and nav["parent_adam_unchanged"] and nav["no_reference_future"]):
        raise ValueError("Previous navigation migration proof is missing")
    if any(key.startswith(PREFIXES) for key in saved["model"]):
        raise ValueError("Quiet pair already exists")
    added = {name: value.detach().cpu().clone() for name, value in _weights(player, npc).items()}
    if not all(torch.isfinite(value).all() for value in added.values()):
        raise ValueError("Invalid quiet weights")
    proof = dict(
        protocol=PROTOCOL,
        source_step=saved["training_steps"],
        source_mode=MODE_NAV,
        target_mode=MODE,
        head_sha256=_fingerprint(added),
        immutable_nav_sha256=nav["head_sha256"],
        added_keys=sorted(added),
        previous_tensors=len(saved["model"]),
        parent_adam_unchanged=True,
        parent_rng_and_clocks_unchanged=True,
        no_reference_future=True,
    )
    return {
        **saved,
        "config": {**config, "player_motion_mode": MODE},
        "model": {**saved["model"], **added},
        "quiet_pair_migration": proof,
    }


def verified_quiet_migration(model, saved):
    receipt = saved.get("quiet_pair_migration", {})
    nav = saved.get("local_axis_head_migration", {})
    if (
        model.config.player_motion_mode != MODE
        or model.config.architecture != "vjepa2_ac"
        or not isinstance(receipt, dict)
        or not isinstance(nav, dict)
    ):
        return False
    backend = model.vjepa_backend
    weights = _weights(backend.quiet_player_head, backend.quiet_pursuit_head)
    retained = len(saved["model"]) - len(weights)
    nav_keys = sorted(NAV_PREFIX + name for name in backend.local_axis_head.state_dict())
    nav_digest = head_digest(backend.local_axis_head)
    guarantees = ("parent_adam_unchanged", "parent_rng_and_clocks_unchanged", "no_reference_future")
    return (
        receipt.get("protocol") == PROTOCOL
        and receipt.get("source_mode") == MODE_NAV
        and receipt.get("target_mode") == MODE
        and receipt.get("added_keys") == sorted(weights)
        and receipt.get("head_sha256") == _fingerprint(weights)
        and receipt.get("previous_tensors") == retained
        and receipt.get("immutable_nav_sha256") == nav.get("head_sha256") == nav_digest
        and nav.get("protocol") == NAV_PROTOCOL
        and nav.get("source_mode") == "delta"
        and nav.get("target_mode") == MODE_NAV
        and nav.get("added_weight_keys") == nav_keys
        and nav.get("retained_original_weight_keys") == retained - len(nav_keys)
        and type(nav.get("source_step")) is int
        and type(receipt.get("source_step")) is int
        and nav["source_step"] <= receipt["source_step"] <= saved["training_steps"]
        and all(ledger.get(key) is True for ledger in (nav, receipt) for key in guarantees)
    )


def compatible_policy_refresh(previous, model, saved):
    """Permit NAV-only → quiet pair and delta → verified NAV → quiet pair."""
    if previous.player_motion_mode not in ("delta", MODE_NAV) or previous.architecture != "vjepa2_ac":
        return False
    if any(
        value != getattr(model.config, key)
        for key, value in vars(previous).items()
        if key != "player_motion_mode"
    ):
        return False
    return verified_quiet_migration(model, saved)


def verify_frozen_resume(model, saved, frozen, current_hash):
    from .renderer_student import _hash_tensors

    if not verified_quiet_migration(model, saved):
        raise ValueError("Invalid quiet player/NPC motion migration or weights")
    previous_hash = saved.get("vjepa_sr_training", {}).get("frozen_parameters_sha256")
    if previous_hash == current_hash:
        return True
    if saved["training_steps"] == saved["quiet_pair_migration"]["source_step"]:
        original = {key: value for key, value in frozen.items() if not key.startswith(PREFIXES)}
        if previous_hash == _hash_tensors(original):
            return True
    raise ValueError("Preexisting frozen weights changed during quiet-pair motion migration")
