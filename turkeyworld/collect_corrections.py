"""Collect training-only oracle branches from states actually visited by a model.

A fixed source action tape is recorded before the model starts. The neural
trajectory is never repaired: each query creates a separate reference Game,
canonicalizes a COPY of the student state, and labels a short independent branch.
Invalid geometry is rejected and logged, without correcting the neural state.
This is a DAgger-like data query, not a five-minute quality certificate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .engine import Game
from .evaluate_rollout import build_action_tape
from .geometry_targets import build_geometry_targets
from .models import load_world_checkpoint

EVENT_NAMES = ("shot", "hit", "kill", "player_hit")


def query_branch(student_state, tilemap, actions, seed=0, dt=1 / 30,
                 size=128, rgb=False, geometry=False):
    """Return ``(payload, audit)`` or ``(None, rejection)`` without input mutation.

    ``states[0]`` is the canonical state the oracle ACTUALLY uses, not the
    unrounded model output. ``student_state`` retains that original output for
    audit. Every accepted payload is one independent episode; never concatenate
    branches into a continuous training trajectory.
    """
    original = np.array(student_state, dtype=np.float32, copy=True)
    action_tape = np.array(actions, dtype=np.float32, copy=True)
    grid = np.array(tilemap, copy=True)
    if original.shape != (90,):
        return None, {"accepted": False, "reason": "invalid_state", "detail": "expected 90 values"}
    if not np.isfinite(original).all():
        return None, {"accepted": False, "reason": "nonfinite", "detail": "student state contains NaN/infinity"}
    if grid.shape != (16, 16) or not np.isfinite(grid).all():
        raise ValueError("tilemap must be finite with shape [16,16]")
    if action_tape.ndim != 2 or action_tape.shape[1] != 4 or len(action_tape) < 1:
        raise ValueError("actions must contain at least one [4] control")
    if not np.isfinite(action_tape).all():
        raise ValueError("action tape must be finite")
    if not math.isfinite(dt) or not 0 < dt <= .25:
        raise ValueError("oracle dt must lie in (0,0.25]")
    oracle = Game(seed=seed)
    oracle.map = grid.astype(np.uint8, copy=True)
    oracle.map_height, oracle.map_width = grid.shape
    try:
        oracle.set_state_vector(original)
    except (ValueError, IndexError, OverflowError) as exc:
        return None, {"accepted": False, "reason": "invalid_state", "detail": str(exc)}
    canonical = oracle.state_vector().copy()
    difference = float(np.max(np.abs(canonical - original)))
    if canonical[4] <= 0:
        return None, {"accepted": False, "reason": "terminal", "detail": "player already dead",
                      "canonicalization_max_abs_diff": difference}
    initial_json = json.dumps(oracle.get_state(), ensure_ascii=False)
    states = [canonical]
    frames = [oracle.render(size=size, overlays=False)] if rgb else None
    geometries = [build_geometry_targets(oracle.map, canonical)] if geometry else None
    accepted_actions, event_rows, hit_ids, attack_flags = [], [], [], []
    for action in action_tape:
        event = oracle.step(action, dt=dt)
        accepted_actions.append(action.copy())
        event_rows.append([float(bool(event.get(name, False))) for name in EVENT_NAMES])
        hit_ids.append(int(event.get("hit_id", -1)))
        attacks = np.zeros(8, dtype=np.float32)
        attacks[event.get("attacker_ids", [])] = 1.
        attack_flags.append(attacks)
        states.append(oracle.state_vector().copy())
        if frames is not None:
            frames.append(oracle.render(size=size, overlays=False))
        if geometries is not None:
            geometries.append(build_geometry_targets(oracle.map, states[-1]))
        if states[-1][4] <= 0:
            break
    payload = {
        "format_version": np.asarray(3),
        "map": oracle.map.astype(np.int8),
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(accepted_actions, dtype=np.float32),
        "events": np.asarray(event_rows, dtype=np.float32),
        "hit_ids": np.asarray(hit_ids, dtype=np.int64),
        "attack_flags": np.asarray(attack_flags, dtype=np.float32),
        "initial_state": np.asarray(initial_json),
        "student_state": original,
        "canonicalization_max_abs_diff": np.asarray(difference, dtype=np.float32),
        "seed": np.asarray(seed), "dt": np.asarray(dt),
        "policy": np.asarray("fixed_tape_student_visited_oracle_branch"),
        "model_trajectory_corrected": np.asarray(False),
        "independent_branch": np.asarray(True),
    }
    if frames is not None:
        payload["frames"] = np.asarray(frames, dtype=np.uint8)
    if geometries is not None:
        for key in ("depth", "visibility", "entity_forward"):
            payload[key] = np.asarray([g[key] for g in geometries], dtype=np.float32)
    return payload, {"accepted": True, "transitions": len(accepted_actions),
                     "canonicalization_max_abs_diff": difference,
                     "source_terminal": bool(states[-1][4] <= 0)}


def _save_branch(path, payload):
    """Exclusive creation prevents overwriting a previously collected episode."""
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **payload)


def collect_corrections(model, output, steps=600, episodes=1, every=64,
                        horizon=16, size=128, rgb=False, geometry=False,
                        seed=0, max_branches=32, scenario="mixed", metadata=None):
    if min(steps, episodes, every, horizon, size, max_branches) < 1:
        raise ValueError("counts and sizes must be positive")
    if scenario not in {"explore", "mixed"}:
        raise ValueError("scenario must be explore or mixed")
    dt = 1 / float(getattr(model.config, "simulation_hz", 30))
    if not math.isfinite(dt) or not 0 < dt <= .25:
        raise ValueError("checkpoint timestep is unsupported by the reference oracle")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / f"corrections_seed_{seed:06d}.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite {summary_path}; choose a new --seed or output directory")
    parameter = next(model.parameters())
    device, network_dtype = parameter.device, parameter.dtype
    state_dtype = torch.float32 if getattr(model.config, "transition_mode", "residual_v0") == "event_v1" else network_dtype
    model.eval()
    counts = {"attempted_branches": 0, "valid_branches": 0,
              "rejected_invalid_states": 0, "rejected_nonfinite": 0,
              "skipped_terminal_branches": 0, "neural_steps": 0,
              "neural_episodes_completed": 0, "saved_transitions": 0}
    records, paths = [], []
    canonicalization_max_diff = 0.
    started = time.perf_counter()
    for episode in range(episodes):
        episode_seed = seed + episode
        # All source controls exist before the neural episode begins. Oracle
        # branches below never supply a replacement control, state or GRU value.
        tape = build_action_tape(episode_seed, scenario, steps, dt, shadow_reference=True)
        state = torch.tensor(tape["initial_state"], device=device, dtype=state_dtype)[None]
        grid = torch.tensor(tape["map"], device=device, dtype=torch.long)[None]
        completed = True
        with torch.inference_mode():
            map_features = model.encode_map(grid)
            memory = model.initial_memory(1, device, network_dtype)
            for index, action in enumerate(tape["actions"]):
                original = state[0].float().cpu().numpy().copy()
                if not np.isfinite(original).all() or not bool(torch.isfinite(memory).all()):
                    counts["rejected_nonfinite"] += 1
                    records.append({"episode_seed": episode_seed, "step": index,
                                    "accepted": False, "reason": "nonfinite_model_state_or_memory"})
                    completed = False
                    break
                if index % every == 0 and counts["valid_branches"] < max_branches:
                    counts["attempted_branches"] += 1
                    payload, audit = query_branch(original, tape["map"],
                                                   tape["actions"][index:index+horizon],
                                                   seed=episode_seed, dt=dt, size=size,
                                                   rgb=rgb, geometry=geometry)
                    record = {"episode_seed": episode_seed, "step": index, **audit}
                    if payload is None:
                        counter = {"invalid_state": "rejected_invalid_states", "nonfinite": "rejected_nonfinite",
                                   "terminal": "skipped_terminal_branches"}[audit["reason"]]
                        counts[counter] += 1
                    else:
                        path = output / f"correction_seed_{episode_seed:06d}_step_{index:06d}.npz"
                        payload.update({"collection_episode_seed": np.asarray(episode_seed),
                                        "student_step": np.asarray(index),
                                        "scenario": np.asarray(scenario),
                                        "world_training_steps": np.asarray(int((metadata or {}).get("training_steps", 0)))})
                        _save_branch(path, payload)
                        paths.append(str(path.resolve()))
                        record["file"] = str(path.resolve())
                        counts["valid_branches"] += 1
                        counts["saved_transitions"] += audit["transitions"]
                        canonicalization_max_diff = max(canonicalization_max_diff, audit["canonicalization_max_abs_diff"])
                    records.append(record)
                # Use the ORIGINAL model state and memory, not the canonical
                # state, endpoint, event, frame or memory from the oracle branch.
                control = torch.tensor(action, device=device, dtype=state_dtype)[None]
                prediction = model.transition(state, control, map_features, memory)
                state, memory = prediction["state"], prediction["memory"]
                counts["neural_steps"] += 1
                if not bool(torch.isfinite(state).all()) or not bool(torch.isfinite(memory).all()):
                    counts["rejected_nonfinite"] += 1
                    records.append({"episode_seed": episode_seed, "step": index+1,
                                    "accepted": False, "reason": "nonfinite_model_prediction"})
                    completed = False
                    break
            if completed:
                counts["neural_episodes_completed"] += 1
    report = {
        "mode": "training_only_student_visited_oracle_branch_collection",
        **counts, "requested_neural_episodes": episodes,
        "steps_per_neural_episode": steps, "query_every_steps": every,
        "branch_horizon": horizon, "max_saved_branches": max_branches,
        "dt": dt, "scenario": scenario, "device": str(device),
        "states_only": not rgb, "geometry_targets": geometry,
        "canonicalization_max_abs_diff": canonicalization_max_diff,
        "model_trajectory_corrected": False,
        "oracle_feedback_to_model_state_or_memory": False,
        "action_tape_recorded_before_neural_episode": True,
        "each_branch_saved_as_separate_episode": True,
        "world_training_steps": int((metadata or {}).get("training_steps", 0)),
        "wall_seconds": time.perf_counter()-started,
        "files": paths, "branch_audit": records,
        "quality_status": "Data queries only. Neither successful collection nor branch count establishes gameplay accuracy or five-minute stability.",
    }
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--every", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--size", type=int, default=128)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--rgb", action="store_true", help="Store real RGB branch frames as well as state labels")
    mode.add_argument("--states-only", dest="rgb", action="store_false", help="Default: omit RGB to keep correction collection small")
    parser.set_defaults(rgb=False)
    parser.add_argument("--geometry", action="store_true", help="Add training-only depth and object-center visibility labels")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-branches", type=int, default=32)
    parser.add_argument("--scenario", choices=("explore", "mixed"), default="mixed")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if min(args.steps, args.episodes, args.every, args.horizon, args.size, args.max_branches, args.threads) < 1:
        parser.error("counts and sizes must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if not args.world.is_file():
        parser.error("world checkpoint not found; train a model first")
    torch.set_num_threads(args.threads)
    model, metadata = load_world_checkpoint(args.world, args.device)
    collect_corrections(model, args.output, args.steps, args.episodes, args.every,
                        args.horizon, args.size, args.rgb, args.geometry, args.seed,
                        args.max_branches, args.scenario, metadata)


if __name__ == "__main__":
    main()
