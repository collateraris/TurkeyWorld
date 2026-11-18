"""Attach TRAIN-only quiet-motion heads to a new file at a full world optimizer boundary.

To rebase onto a later boundary, pass the evaluated checkpoint with --validated.
The parent supplies every existing model/optimizer tensor, clock and RNG state.
"""
import argparse
import os
import tempfile
from pathlib import Path

import torch

from .learned_axis_collision import LocalAxisCollisionHead, MODE_NAV, PREFIX, head_digest
from .quiet_pair_profile import MODE as PAIR_MODE, with_quiet_heads
from .training_pipeline import atomic_json


def apply(parent, npc_artifact, player_artifact=None, validated=None, expect_step=None):
    from .learned_quiet_pursuit import QuietPursuitHead

    step = parent["training_steps"]
    config = parent["config"]
    if step != parent.get("global_training_step") or (expect_step is not None and step != expect_step):
        raise ValueError("Not the requested saved optimizer boundary")
    if (
        parent.get("stage") != "world"
        or config["player_motion_mode"] != MODE_NAV
        or not parent.get("vjepa_sr_training", {}).get("frozen_parameters_sha256")
    ):
        raise ValueError("Incomplete quiet migration parent")
    if (
        parent.get("training_counters", {}).get("pending_microbatches", 0)
        or not parent.get("optimizer", {}).get("state")
        or not parent.get("optimizer_parameter_names")
    ):
        raise ValueError("Partial or uninitialized optimizer")

    previous = parent["local_axis_head_migration"]
    nav = LocalAxisCollisionHead()
    nav_weights = {
        key[len(PREFIX) :]: value for key, value in parent["model"].items() if key.startswith(PREFIX)
    }
    nav.load_state_dict(nav_weights, strict=True)
    if head_digest(nav) != previous.get("head_sha256"):
        raise ValueError("Base NAV head changed")

    npc = QuietPursuitHead()
    npc_receipt = npc_artifact["train_receipt"]
    if npc_receipt.get("training_uses_heldout") is not False:
        raise ValueError("Expected TRAIN-only NPC")
    npc.load_state_dict(npc_artifact["model"], strict=True)
    player = LocalAxisCollisionHead()
    if player_artifact is None:
        player.load_state_dict(nav.state_dict())
        player_receipt = dict(
            training_uses_heldout=False, reuses_qualified_nav_head=True, head_sha256=head_digest(nav)
        )
    else:
        player_receipt = player_artifact["train_receipt"]
        if player_receipt.get("training_uses_heldout") is not False:
            raise ValueError("Expected TRAIN-only player")
        player.load_state_dict(player_artifact["model"], strict=True)
        if player_receipt.get("head_sha256", head_digest(player)) != head_digest(player):
            raise ValueError("Quiet player weights differ from training receipt")

    result = with_quiet_heads(parent, player, npc)
    result["quiet_pair_training"] = {"player": player_receipt, "npc": npc_receipt}
    if validated is not None:
        proof = validated.get("quiet_pair_migration", {})
        current = result["quiet_pair_migration"]
        if (
            validated["training_steps"] > step
            or validated["config"] != {**config, "player_motion_mode": PAIR_MODE}
            or validated["state_schema"] != parent["state_schema"]
            or validated["vjepa_sr_feedback"] != parent["vjepa_sr_feedback"]
            or proof.get("head_sha256") != current["head_sha256"]
            or proof.get("immutable_nav_sha256") != previous["head_sha256"]
            or proof.get("added_keys") != current["added_keys"]
            or validated.get("quiet_pair_training") != result["quiet_pair_training"]
        ):
            raise ValueError("Validated candidate does not match this parent, gates or receipt")
        if not all(
            torch.equal(validated["model"][key], result["model"][key]) for key in current["added_keys"]
        ):
            raise ValueError("Validated quiet head differs")
    for key in (
        "optimizer",
        "optimizer_parameter_names",
        "vjepa_sr_training",
        "rng_torch",
        "rng_cuda",
        "ema_target_encoder",
        "local_axis_head_migration",
    ):
        if result[key] is not parent[key]:
            raise AssertionError("Unexpected mutation of " + key)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "npc", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--player", type=Path)
    parser.add_argument("--validated", type=Path)
    parser.add_argument("--expect-step", type=int)
    args = parser.parse_args()
    output = args.output.absolute()
    sources = [p for p in (args.base, args.npc, args.player, args.validated) if p is not None]
    if output.exists() or output.is_symlink() or any(output.resolve() == path.resolve() for path in sources):
        raise ValueError("Output must be a new checkpoint file")

    def identities():
        stats = (p.stat() for p in sources)
        return [(s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns) for s in stats]

    def load(path):
        if path is None:
            return None
        return torch.load(path, weights_only=False, mmap=True, map_location="cpu")

    pinned = identities()
    result = apply(load(args.base), load(args.npc), load(args.player), load(args.validated), args.expect_step)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=output.name + ".", suffix=".tmp", dir=output.parent, delete=False
    ) as file:
        temporary = Path(file.name)
    try:
        torch.save(result, temporary)
        if identities() != pinned:
            raise OSError("Input checkpoint changed during attachment")
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {
        **result["quiet_pair_migration"],
        "base": str(args.base),
        "validated": str(args.validated) if args.validated else None,
        "updates": result["vjepa_sr_training"]["updates"],
        "retained_adam_slots": len(result["optimizer"]["state"]),
    }
    atomic_json(output.with_suffix(output.suffix + ".attachment.json"), receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    main()
