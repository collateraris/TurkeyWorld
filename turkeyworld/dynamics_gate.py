"""Held-out short dynamics gate; source labels measure, never repair, rollout."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
import hashlib
import json

import numpy as np
import torch

from .runtime_checks import circle_is_clear, segment_is_clear


DEFAULT_THRESHOLDS = {"pose_error_tiles_mean": .15, "player_pose_error_tiles_mean": .15,
    "player_velocity_relative_error": .35, "hit_f1": .7, "hit_recall": .7,
    "attack_f1": .6, "collision_fraction": .01,
    "minimum_moving_transitions": 1, "minimum_hit_positives": 1,
    "minimum_attack_positives": 1, "minimum_source_player_tiles": 2}


@contextmanager
def _evaluation_context(model, device, seed):
    modes = [(module, module.training) for module in model.modules()]
    precision = torch.get_float32_matmul_precision()
    tf32, cudnn_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        model.eval()
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.random.fork_rng(devices=devices), torch.inference_mode(), torch.autocast(device.type, enabled=False):
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            yield
    finally:
        for module, training in modes:
            module.training = training
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def _fingerprint(model, data, panel, context, horizon, seed, thresholds):
    digest = hashlib.sha256()
    for episode_index, start in panel:
        episode = data.episodes[episode_index]
        first = start + context - 1
        for value in (episode["frames"][start:start + context],
                      episode["states"][start:start + context + horizon],
                      episode["actions"][first:first + horizon], episode["events"][first:first + horizon],
                      episode["attack_flags"][first:first + horizon], episode["map"]):
            array = np.ascontiguousarray(value)
            digest.update(str((array.dtype, array.shape)).encode())
            digest.update(array.tobytes())
    config = model.config
    config = asdict(config) if is_dataclass(config) else vars(config)
    specification = {"panel": panel, "context": context, "horizon": horizon, "seed": seed,
        "thresholds": thresholds, "model_config": config, "content": digest.hexdigest(),
        "protocol": "own-ar-dynamics-v1", "precision": "fp32-tf32-off"}
    return hashlib.sha256(json.dumps(specification, sort_keys=True, default=str).encode()).hexdigest()


def _confusion(predicted, target):
    predicted, target = np.asarray(predicted, dtype=bool), np.asarray(target, dtype=bool)
    return np.array([np.count_nonzero(predicted & target), np.count_nonzero(predicted & ~target),
                     np.count_nonzero(~predicted & target), np.count_nonzero(~predicted & ~target)], dtype=np.int64)


def _f1(values):
    true, false, missed, _ = map(int, values)
    return 2 * true / (2 * true + false + missed) if 2 * true + false + missed else 0.


def _collision(tilemap, previous, current):
    if not np.isfinite(current).all():
        return True
    old = previous[8:88].reshape(8, 10)
    new = current[8:88].reshape(8, 10)
    for before, after, radius, revived in [(previous[:2], current[:2], .2, False)] + [
            (old[index, :2], new[index, :2], .25,
             old[index, 5] < .5 and new[index, 5] >= .5) for index in range(8)]:
        if not circle_is_clear(tilemap, after * 16, radius):
            return True
        if not revived and not segment_is_clear(tilemap, before * 16, after * 16, radius):
            return True
    return False


def evaluate_dynamics_gate(model, data, panel=None, *, count=64, context=2, horizon=4,
                           device=None, batch_size=1, seed=73, thresholds=None):
    """Measure own-AR pose/motion/combat/collisions on a fixed held-out panel.

    Attack F1 is micro-F1 of accepted per-NPC attacks (raw attack logits only
    as an explicitly reported fallback). Empty-positive or stationary source
    panels cannot qualify. Geometry is read-only measurement, not a model
    input/correction. No future source image is rendered or encoded.
    """
    if getattr(data, "split", None) not in {"val", "test"}:
        raise ValueError("Dynamics qualification requires a held-out val/test split")
    if min(count, context, horizon, batch_size) < 1:
        raise ValueError("Dynamics panel counts must be positive")
    limits = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        if set(thresholds) - set(limits):
            raise ValueError("Unknown dynamics-gate thresholds")
        limits.update(thresholds)
    if not all(np.isfinite(value) and value >= 0 for value in limits.values()):
        raise ValueError("Dynamics thresholds must be finite and nonnegative")
    panel = data.fixed_panel(count=count, context=context, horizon=horizon, seed=seed) if panel is None else panel
    values = np.asarray(panel)
    if values.ndim != 2 or values.shape[1] != 2 or not len(values) or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Panel must contain integer (episode,start) windows")
    panel = [(int(episode), int(start)) for episode, start in values]
    if len(set(panel)) != len(panel):
        raise ValueError("Dynamics panel windows must be unique")
    for episode, start in panel:
        if (not 0 <= episode < len(data.episodes) or start < 0
                or start + context + horizon > len(data.episodes[episode]["frames"])):
            raise ValueError("Dynamics window crosses an episode boundary")
    device = torch.device(device) if device is not None else next(model.parameters()).device
    if any(parameter.is_floating_point() and parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("Dynamics qualification requires FP32 model parameters")
    fingerprint = _fingerprint(model, data, panel, context, horizon, seed, limits)
    poses, player_poses, velocity_errors = [], [], []
    hit_counts, attack_counts = np.zeros(4, np.int64), np.zeros(4, np.int64)
    collision_count = invalid_count = missing_predictions = transitions = 0
    moving_count = source_hits = source_attacks = 0
    source_distance = 0.
    source_tiles = set()
    prediction_protocol = set()
    with _evaluation_context(model, device, seed):
        for first in range(0, len(panel), batch_size):
            chosen = panel[first:first + batch_size]
            batch = data.batch(len(chosen), context=context, horizon=horizon, indices=chosen)
            grid = batch["map"].to(device)
            maps = grid.cpu().numpy()
            features = model.encode_map(grid)
            source_context = batch["states"][:, :context].to(device).float()
            memory = model.initialize_from_image(batch["context_frames"].to(device).float(), source_context, features)
            state = source_context[:, -1].clone()
            invalid = not bool(torch.isfinite(memory).all())
            for index in range(horizon):
                before = state.detach().double().cpu().numpy().copy()
                if not invalid:
                    result = model.transition(state, batch["actions"][:, index].to(device).float(), features, memory)
                    state, memory = result["state"], result["memory"]
                    invalid = not bool(torch.isfinite(state).all() and torch.isfinite(memory).all())
                if invalid:
                    after = np.full_like(before, np.nan)
                    predicted_hit, predicted_attack = np.zeros(len(chosen), bool), np.zeros((len(chosen), 8), bool)
                else:
                    after = state.detach().double().cpu().numpy()
                    if "events" in result and torch.is_tensor(result["events"]):
                        missing_predictions += int((~torch.isfinite(result["events"]).all(-1)).sum())
                        predicted_hit = result["events"][:, 1].detach().cpu().numpy() >= .5
                    elif "hit" in result:
                        predicted_hit = result["hit"].detach().cpu().numpy().astype(bool)
                    else:
                        missing_predictions += len(chosen)
                        predicted_hit = np.zeros(len(chosen), bool)
                    if "accepted_attacks" in result:
                        predicted_attack = result["accepted_attacks"].detach().cpu().numpy().astype(bool)
                        prediction_protocol.add("accepted_attack_slots")
                    elif "attack_logits" in result:
                        missing_predictions += int((~torch.isfinite(result["attack_logits"]).all(-1)).sum())
                        predicted_attack = result["attack_logits"].detach().cpu().numpy() >= 0
                        prediction_protocol.add("raw_attack_head_slots")
                    else:
                        missing_predictions += len(chosen)
                        predicted_attack = np.zeros((len(chosen), 8), bool)
                source_previous = batch["previous_states"][:, index].double().cpu().numpy()
                source_next = batch["target_states"][:, index].double().cpu().numpy()
                reference_hit = batch["events"][:, index, 1].cpu().numpy() >= .5
                reference_attack = batch["attack_flags"][:, index].cpu().numpy() >= .5
                source_hits += int(reference_hit.sum())
                source_attacks += int(reference_attack.sum())
                hit_counts += _confusion(predicted_hit, reference_hit)
                attack_counts += _confusion(predicted_attack, reference_attack)
                for row in range(len(chosen)):
                    transitions += 1
                    finite = np.isfinite(after[row]).all()
                    invalid_count += int(not finite)
                    collision_count += int(_collision(maps[row], before[row], after[row]))
                    source_tiles.add(tuple(np.floor(source_next[row, :2] * 16).astype(int)))
                    source_delta = (source_next[row, :2] - source_previous[row, :2]) * 16
                    distance = float(np.linalg.norm(source_delta))
                    source_distance += distance
                    timestep = float(batch["dt"][row])
                    if distance / timestep > .1:
                        moving_count += 1
                        predicted_delta = (after[row, :2] - before[row, :2]) * 16
                        velocity_errors.append(float(np.linalg.norm(predicted_delta - source_delta) / max(distance, 1e-8))
                                               if finite else 1e6)
                    if finite:
                        actor_error = np.concatenate([(after[row, :2] - source_next[row, :2])[None],
                            (after[row, 8:88].reshape(8, 10)[:, :2] - source_next[row, 8:88].reshape(8, 10)[:, :2])]) * 16
                        distances = np.linalg.norm(actor_error, axis=-1)
                        poses.extend(distances.tolist())
                        player_poses.append(float(distances[0]))
                    else:
                        poses.extend([1e6] * 9)
                        player_poses.append(1e6)
    mean = lambda values: float(np.mean(values)) if values else 1e6
    recall = int(hit_counts[0]) / source_hits if source_hits else 0.
    metrics = {"pose_error_tiles_mean": mean(poses), "player_pose_error_tiles_mean": mean(player_poses),
        "player_velocity_relative_error": mean(velocity_errors), "hit_f1": _f1(hit_counts),
        "hit_recall": recall, "attack_f1": _f1(attack_counts),
        "collision_fraction": collision_count / max(1, transitions)}
    counts = {"windows": len(panel), "transitions": transitions, "source_moving_transitions": moving_count,
        "source_hit_positives": source_hits, "source_attack_positives": source_attacks,
        "source_unique_player_tiles": len(source_tiles), "source_player_distance_tiles": source_distance,
        "collision_transitions": collision_count, "nonfinite_transitions": invalid_count,
        "missing_predictions": missing_predictions,
        "hit": dict(zip(("tp", "fp", "fn", "tn"), map(int, hit_counts))),
        "attack": dict(zip(("tp", "fp", "fn", "tn"), map(int, attack_counts)))}
    checks = {name: metrics[name] < limits[name] for name in ("pose_error_tiles_mean", "player_pose_error_tiles_mean",
        "player_velocity_relative_error", "collision_fraction")}
    checks.update({name: metrics[name] > limits[name] for name in ("hit_f1", "hit_recall", "attack_f1")})
    checks.update(source_motion=moving_count >= limits["minimum_moving_transitions"],
                  source_hits=source_hits >= limits["minimum_hit_positives"],
                  source_attacks=source_attacks >= limits["minimum_attack_positives"],
                  source_coverage=len(source_tiles) >= limits["minimum_source_player_tiles"],
                  finite_predictions=invalid_count == 0, complete_predictions=missing_predictions == 0)
    score = (metrics["pose_error_tiles_mean"] / max(limits["pose_error_tiles_mean"], 1e-8)
        + metrics["player_velocity_relative_error"] / max(limits["player_velocity_relative_error"], 1e-8)
        + 1 - metrics["hit_f1"] + 1 - metrics["hit_recall"] + 1 - metrics["attack_f1"]
        + metrics["collision_fraction"] / max(limits["collision_fraction"], 1e-8)
        + 10 * sum(not checks[name] for name in ("source_motion", "source_hits", "source_attacks", "source_coverage",
                                                "finite_predictions", "complete_predictions")))
    return {"passed": all(checks.values()), "score": float(score), "metrics": metrics,
        "counts": counts, "thresholds": limits, "checks": checks,
        "failure_reasons": [name for name, passed in checks.items() if not passed],
        "fingerprint": fingerprint, "panel_fingerprint": fingerprint, "horizon": horizon, "context": context,
        "panel": [[episode, start] for episode, start in panel], "source_held_out": True,
        "future_ground_truth_feedback": False, "attack_measurement": sorted(prediction_protocol),
        "scope": "Short held-out dynamics only; not a 9000-step gameplay qualification"}
