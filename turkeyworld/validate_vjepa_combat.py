"""Fixed, held-out 30 Hz V-JEPA combat rollouts using production SR feedback."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .models import load_world_checkpoint
from .rssm_combat_protocol import environment_profile
from .small_enemy_data import digest
from .structured_state import EVENT_NAMES, MAP_EXTENT
from .training_pipeline import atomic_json, now
from .vjepa_combat_training import combat_objective
from .vjepa_policy_world import VJEPAImaginationAdapter
from .workspace_paths import ROOT


VAL = ROOT / "training_runs/combat_source_20261111/data/full_game/rgb/val"
HORIZON = 180
CONTEXT = 4
CASES = ("duel", "melee_then_shoot", "miss_then_hit", "blocked", "kill_respawn", "return")
STARTS = (3, 3, 3, 3, 3, 3, 82, 27, 91, 104, 36, 127)
PANEL_SHA256 = "ca15d8709ed1f5e4dbf2d5631f1b41108fd134892d369439cef61b79ff1afc1b"


def load_panel():
    windows = []
    identities = []
    for i, start in enumerate(STARTS):
        seed = 95100000 + i
        path = VAL / f"episode_{seed:09d}.npz"
        case = CASES[i % len(CASES)]
        identity = dict(file=path.name, start=start, case=case, sha256=digest(path))
        with np.load(path, allow_pickle=False) as ep:
            if (
                ep["case"].item() != case
                or int(ep["seed"]) != seed
                or ep["split"].item() != "val"
                or ep["stage"].item() != "full_game"
                or not ep["enemies_enabled"].item()
                or abs(float(ep["dt"]) - 1 / 30) > 1e-7
            ):
                raise ValueError(f"Unexpected Source provenance in {path}")
            end = start + HORIZON
            states = ep["states"][start - CONTEXT + 1 : end + 1]
            actions = ep["actions"][start:end]
            events = ep["events"][start:end]
            hit_ids = ep["hit_ids"][start:end]
            attack_flags = ep["attack_flags"][start:end]
            grid = ep["map"]
            if (
                states.shape != (CONTEXT + HORIZON, 20)
                or actions.shape != (HORIZON, 4)
                or events.shape != (HORIZON, 4)
                or grid.shape != (16, 16)
            ):
                raise ValueError(f"Incomplete Source window in {path}")
        identities.append(identity)
        windows.append((states, actions, events, grid, hit_ids, attack_flags))
    fingerprint = hashlib.sha256(
        json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if fingerprint != PANEL_SHA256:
        raise ValueError("The preregistered held-out combat panel has changed")
    return windows, identities


def generations(initial_alive, alive):
    """Number each new live NPC relative to the slot which exists at context end."""
    previous = np.concatenate([initial_alive[:, None], alive[:, :-1]], axis=1)
    return np.cumsum(alive & ~previous, axis=1)


def event_scores(predicted, reference):
    tp = int(np.count_nonzero(predicted & reference))
    fp = int(np.count_nonzero(predicted & ~reference))
    fn = int(np.count_nonzero(~predicted & reference))
    return dict(
        reference=tp + fn, predicted=tp + fp, tp=tp, fp=fp, fn=fn,
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    )


def summarize(pred, truth, events, ref_events, initial):
    if not (np.isfinite(pred).all() and np.isfinite(events).all()):
        raise ValueError("Nonfinite forecast")
    hard = events.astype(bool)
    reference = ref_events.astype(bool)
    scores = {name: event_scores(hard[..., i], reference[..., i]) for i, name in enumerate(EVENT_NAMES)}

    pred_alive, true_alive = pred[..., 13] >= 0.5, truth[..., 13] >= 0.5
    initial_alive = initial[:, 13] >= 0.5
    pred_generation = generations(initial_alive, pred_alive)
    true_generation = generations(initial_alive, true_alive)
    comparable = pred_alive & true_alive & (pred_generation == true_generation)
    pred_spawn = np.diff(pred_generation, axis=1, prepend=0).astype(bool)
    true_spawn = np.diff(true_generation, axis=1, prepend=0).astype(bool)
    respawn = event_scores(pred_spawn, true_spawn)
    matched = pred_spawn & true_spawn
    predicted_xy = pred[..., 8:10][matched] * MAP_EXTENT
    reference_xy = truth[..., 8:10][matched] * MAP_EXTENT
    respawn["same_tile"] = int((np.floor(predicted_xy) == np.floor(reference_xy)).all(axis=-1).sum())
    respawn["xy_mean_tiles_on_matched_ticks"] = (
        float(np.linalg.norm(predicted_xy - reference_xy, axis=-1).mean()) if matched.any() else None
    )
    # HP errors include dead actors and disagreement about whether a slot is alive.
    hp = {
        name: {
            "mean_abs": float(np.mean(np.abs(pred[..., i] - truth[..., i]))),
            "H180_abs": float(np.mean(np.abs(pred[:, -1, i] - truth[:, -1, i]))),
        }
        for name, i in [("player", 4), ("npc", 12)]
    }
    position = {}
    for name, offset, valid in [("player", 0, np.ones_like(comparable)), ("npc_same_life", 8, comparable)]:
        err = np.linalg.norm(
            (pred[..., offset : offset + 2] - truth[..., offset : offset + 2]) * MAP_EXTENT, axis=-1
        )
        position[name] = {
            "mean_path_tiles": float(err[valid].mean()) if valid.any() else None,
            "H180_tiles": float(err[:, -1][valid[:, -1]].mean()) if valid[:, -1].any() else None,
            "ticks_scored": int(valid.sum()),
            "endpoints_scored": int(valid[:, -1].sum()),
        }
    return dict(
        windows=len(pred),
        ticks=int(np.prod(pred.shape[:2])),
        exact_tick_events=scores,
        normalized_hp=hp,
        xy=position,
        alive_disagreement_ticks=int(np.count_nonzero(pred_alive != true_alive)),
        reference_respawns=respawn["reference"],
        predicted_respawns=respawn["predicted"],
        exact_tick_respawns=respawn,
    )


def check_stop(stop_file):
    if stop_file is not None and stop_file.exists():
        raise InterruptedError("Combat validation stopped before report publication")


@torch.inference_mode()
def evaluate(world, device, batch_size, stop_file=None):
    check_stop(stop_file)
    windows, identities = load_panel()
    states, actions, events, grids, hit_ids, attack_flags = map(np.stack, zip(*windows))
    model, payload = load_world_checkpoint(world, "cpu")
    step = int(payload["training_steps"])
    del payload
    adapter = VJEPAImaginationAdapter(model.to(device))
    if (
        not adapter.feedback_renderer
        or adapter.feedback_settings["Source_future_feedback"]
        or int(model.config.simulation_hz) != 30
    ):
        raise ValueError("Combat qualification requires production 30 Hz SR feedback")
    outputs = {k: [] for k in ("state", "events", "target_logits", "attack_logits", "event_probs")}
    with environment_profile(model, navigation_only=False):
        for begin in range(0, len(states), batch_size):
            check_stop(stop_file)
            group = slice(begin, begin + batch_size)
            past = torch.as_tensor(states[group, :CONTEXT], dtype=torch.float32, device=device)
            grid = torch.as_tensor(grids[group], dtype=torch.long, device=device)
            control = torch.as_tensor(actions[group], dtype=torch.float32, device=device)
            current = adapter.initialize(past, grid, past.new_zeros(len(past), CONTEXT, 3, 32, 32))
            traces = {k: [] for k in outputs}
            for tick in range(HORIZON):
                if tick % 30 == 0:
                    check_stop(stop_file)
                current, info = adapter.transition_with_info(current, control[:, tick])
                for k in traces:
                    traces[k].append(info[k])
            for k in outputs:
                outputs[k].append(torch.stack(traces[k], dim=1).cpu())
    outputs = {k: torch.cat(batches, dim=0) for k, batches in outputs.items()}
    pred, accepted = outputs["state"].numpy(), outputs["events"].numpy()
    truth, initial = states[:, CONTEXT:], states[:, CONTEXT - 1]
    report = summarize(pred, truth, accepted, events, initial)
    # These labels are attached only after all autonomous predictions have been generated.
    batch = {
        "states": torch.from_numpy(initial[:, None]),
        "target_states": torch.from_numpy(truth),
        "hit_ids": torch.from_numpy(hit_ids),
        "attack_flags": torch.from_numpy(attack_flags),
        "events": torch.from_numpy(events),
    }
    loss = combat_objective(
        outputs["target_logits"], outputs["attack_logits"], outputs["event_probs"], batch, outputs["state"]
    )
    report["combat_objective"] = {k: float(v) for k, v in loss.items()}
    per_window = [
        {
            **row,
            **summarize(
                pred[i : i + 1], truth[i : i + 1], accepted[i : i + 1], events[i : i + 1], initial[i : i + 1]
            ),
        }
        for i, row in enumerate(identities)
    ]
    return dict(
        protocol="vjepa_combat_real_VAL_H180_v1",
        step=step,
        context=CONTEXT,
        horizon=HORIZON,
        hz=30,
        source_split="val",
        generated_at=now(),
        feedback_mode="generated_Base_SR",
        source_future_feedback=False,
        source_rgb_input=False,
        panel_sha256=PANEL_SHA256,
        source="combat_source_20261111/data/full_game/rgb/val",
        summary=report,
        per_window=per_window,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stop-file", type=Path, help="Abort without publishing a partial report")
    parser.add_argument("--batch-size", type=int, choices=range(1, 13), default=4)
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        report = evaluate(args.world, args.device, args.batch_size, args.stop_file)
        check_stop(args.stop_file)
    except InterruptedError as exc:
        parser.exit(130, str(exc) + "\n")
    atomic_json(args.output, report)
    print(json.dumps({"step": report["step"], **report["summary"]}, ensure_ascii=False))
