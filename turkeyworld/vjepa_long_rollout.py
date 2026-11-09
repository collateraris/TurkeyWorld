"""Own-state V-JEPA rollout training and return-to-combat memory probes.

The source state and image initialize a segment once. Every subsequent model
transition receives the preceding prediction and carried memory. Recorded
future states, events and optional images are loss/diagnostic targets only.
Truncated backpropagation detaches graphs, never numerical values. A callback
can persist the returned resume state at any completed chunk boundary.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .long_train import collision_barrier, map_signed_distance
from .train import combat_label_arrays, image_loss, state_loss


CURRICULUM_HORIZONS = (32, 128, 512, 2048, 9000)
CURRICULUM_SCHEDULE = (32,) * 8 + (128,) * 4 + (512,) * 2 + (2048,) * 2 + (9000,)


def curriculum_horizon(index, total_updates, horizons=CURRICULUM_HORIZONS):
    """Zero-based progress through the explicit 32..9000-step curriculum."""
    if total_updates < 1 or not 0 <= index < total_updates or not horizons or min(horizons) < 1:
        raise ValueError("Expected a valid update index and positive horizons")
    return int(horizons[min(len(horizons) - 1, index * len(horizons) // total_updates)])


def _field(episode, name, default=None):
    if isinstance(episode, Mapping) or hasattr(episode, "files"):
        return episode[name] if name in episode else default
    return getattr(episode, name, default)


def _tree_map(value, function):
    if torch.is_tensor(value):
        return function(value)
    if isinstance(value, dict):
        return {key: _tree_map(item, function) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree_map(item, function) for item in value)
    if isinstance(value, list):
        return [_tree_map(item, function) for item in value]
    return value


def _detached(value, cpu=False):
    return _tree_map(value, lambda item: item.detach().cpu().clone() if cpu else item.detach())


def _tree_difference(first, second):
    if torch.is_tensor(first):
        return float((first.detach() - second.detach().to(first.device)).abs().max()) if first.numel() else 0.
    if isinstance(first, dict):
        return max((_tree_difference(first[key], second[key]) for key in first), default=0.)
    if isinstance(first, (tuple, list)):
        return max((_tree_difference(a, b) for a, b in zip(first, second)), default=0.)
    return 0.


def _episode_arrays(episode):
    states = np.asarray(_field(episode, "states"), dtype=np.float32)
    actions = np.asarray(_field(episode, "actions"), dtype=np.float32)
    events = np.asarray(_field(episode, "events"), dtype=np.float32)
    grid = np.asarray(_field(episode, "grid", _field(episode, "map")), dtype=np.int64)
    count = len(actions)
    if (count < 1 or states.shape != (count + 1, 90) or actions.shape != (count, 4)
            or events.shape != (count, 4) or grid.shape != (16, 16)):
        raise ValueError("Expected states[T+1,90], actions/events[T,4], map[16,16]")
    if not all(np.isfinite(value).all() for value in (states, actions, events)):
        raise ValueError("Episode contains non-finite values")
    labels = {"states": states, "actions": actions}
    for key in ("hit_ids", "attack_flags"):
        value = _field(episode, key)
        if value is not None:
            labels[key] = value
    hits, attacks, source = combat_label_arrays(labels)
    return states, actions, events, grid, hits, attacks, source


def _episode_key(states, actions, grid, prefix_start):
    digest = hashlib.sha256()
    for array in (grid, states, actions, np.asarray(prefix_start, dtype=np.int64)):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def source_initial_image(episode, image_size=128, prefix_start=0, *, endless=False):
    """Read/render one source image before rollout; never advances the engine.

    State-only corpora contain no RGB. Their one initialization frame can be
    reconstructed from the recorded initial state/map. Future engine states or
    frames are never queried by this helper or the training rollout.
    """
    frames = _field(episode, "frames")
    if frames is not None:
        return np.asarray(frames[prefix_start]).copy()
    initial = _field(episode, "initial_rgb") if prefix_start == 0 else None
    if initial is not None:
        return np.asarray(initial).copy()
    path = _field(episode, "path")
    saved_state = None
    if path and Path(path).is_file():
        with np.load(path, allow_pickle=False) as saved:
            if prefix_start == 0 and "initial_rgb" in saved.files:
                return np.asarray(saved["initial_rgb"]).copy()
            if prefix_start == 0 and "frames" in saved.files:
                return np.asarray(saved["frames"][0]).copy()
            if prefix_start == 0 and "initial_state" in saved.files:
                saved_state = json.loads(str(saved["initial_state"]))
            if "endless" in saved.files:
                endless = bool(saved["endless"])
    from .engine import Game
    states, _, _, grid, _, _, _ = _episode_arrays(episode)
    game = Game(seed=int(_field(episode, "seed", 0)), endless=endless)
    game.map = grid.astype(np.uint8).copy()
    game.map_height, game.map_width = grid.shape
    if saved_state is not None:
        game.set_state(saved_state)
    else:
        game.set_state_vector(states[prefix_start])
    return game.render(size=max(32, image_size), overlays=False)


def _rgb_tensor(image, image_size, device):
    value = torch.as_tensor(np.asarray(image).copy() if not torch.is_tensor(image) else image,
                            device=device)
    if value.ndim == 3 and value.shape[-1] == 3:
        value = value.permute(2, 0, 1)[None]
    elif value.ndim == 3 and value.shape[0] == 3:
        value = value[None]
    elif value.ndim == 4 and value.shape[-1] == 3:
        value = value.permute(0, 3, 1, 2)
    if value.ndim not in (4, 5) or value.shape[-3] != 3 or value.shape[0] != 1:
        raise ValueError("Initial RGB must be [H,W,3], [1,3,H,W] or [1,T,3,H,W]")
    value = value.float() / 255 if value.dtype == torch.uint8 else value.float()
    if not torch.isfinite(value).all() or value.min() < 0 or value.max() > 1:
        raise ValueError("Initial RGB must be uint8 or normalized finite [0,1]")
    if value.shape[-2:] != (image_size, image_size):
        shape = value.shape
        value = F.interpolate(value.reshape(-1, 3, *shape[-2:]), (image_size, image_size),
                              mode="bilinear", align_corners=False, antialias=True)
        value = value.reshape(*shape[:-2], image_size, image_size)
    return value


def frustum_visibility(states):
    """Cheap training diagnostics when exact recorded visibility is unavailable.

    This labels front/back/frustum only; it makes no wall-occlusion claim.
    It is not a neural runtime input or an engine visibility correction.
    """
    states = torch.as_tensor(states, dtype=torch.float32)
    entities = states[:, 8:88].reshape(-1, 8, 10)
    delta = entities[:, :, :2] - states[:, None, :2]
    sine, cosine = states[:, 2:3], states[:, 3:4]
    forward = delta[:, :, 0] * cosine + delta[:, :, 1] * sine
    lateral = -delta[:, :, 0] * sine + delta[:, :, 1] * cosine
    return (forward > .12 / 16) & (lateral.abs() <= forward * math.tan(math.radians(34)))


def object_memory_metrics(predicted_states, target_states, visibility=None):
    """Slot-indexed HP/life/identity retention when objects leave and reappear.

    Reappearance requires a prior sighting followed by an invisible interval.
    Legitimate reference respawns are allowed; resurrection is counted only
    while the reference slot remains dead. Ambiguous overlapping source slots
    are excluded from position-based identity association.
    """
    predicted = torch.as_tensor(predicted_states, dtype=torch.float32).detach().cpu()
    targets = torch.as_tensor(target_states, dtype=torch.float32).detach().cpu()
    if predicted.shape != targets.shape or predicted.ndim != 2 or predicted.shape[1] != 90:
        raise ValueError("Expected matching [T+1,90] predicted/target states")
    visible = frustum_visibility(targets) if visibility is None else torch.as_tensor(visibility).cpu() > .5
    if visible.shape != (len(targets), 8):
        raise ValueError("Visibility must align as [T+1,8]")
    p = predicted[:, 8:88].reshape(-1, 8, 10)
    t = targets[:, 8:88].reshape(-1, 8, 10)
    seen = visible.to(torch.int64).cummax(0).values.bool()
    reappeared = visible[1:] & ~visible[:-1] & seen[:-1]
    hidden = ~visible[1:] & seen[:-1]
    hp_error = (p[1:, :, 4] - t[1:, :, 4]).abs() * 100
    alive_match = (p[1:, :, 5] >= .5) == (t[1:, :, 5] >= .5)
    position_error = torch.linalg.vector_norm((p[1:, :, :2] - t[1:, :, :2]) * 16, dim=-1)
    dead = t[1:, :, 5] < .5
    resurrection = (p[:-1, :, 5] < .5) & (p[1:, :, 5] >= .5) & dead
    target_respawns = (t[:-1, :, 5] < .5) & (t[1:, :, 5] >= .5)
    distances = torch.cdist(p[1:, :, :2], t[1:, :, :2]) * 16
    target_distances = torch.cdist(t[1:, :, :2], t[1:, :, :2]) * 16
    target_distances = target_distances + torch.eye(8)[None] * 1000
    unambiguous = target_distances.min(-1).values > .10
    identity_mask = reappeared & unambiguous
    associated = distances.argmin(-1) == torch.arange(8)[None]

    def mean(value, mask):
        return float(value[mask].float().mean()) if bool(mask.any()) else None

    return {"visibility_source": "recorded" if visibility is not None else "frustum_only",
            "hidden_object_samples": int(hidden.sum()), "reappearance_samples": int(reappeared.sum()),
            "hidden_hp_mae": mean(hp_error, hidden), "hidden_alive_accuracy": mean(alive_match, hidden),
            "reappearance_hp_mae": mean(hp_error, reappeared),
            "reappearance_alive_accuracy": mean(alive_match, reappeared),
            "reappearance_position_error_tiles": mean(position_error, reappeared),
            "reappearance_object_id_samples": int(identity_mask.sum()),
            "reappearance_object_id_accuracy": mean(associated, identity_mask),
            "dead_object_samples": int(dead.sum()), "dead_object_retention_accuracy": mean(p[1:, :, 5] < .5, dead),
            "false_resurrections": int(resurrection.sum()), "reference_respawns": int(target_respawns.sum()),
            "object_slots_keep_fixed_ids": True}


def autoregressive_losses(result, previous_prediction, target_previous, target_next,
                          target_events, hit_ids, attack_flags, *, simulation_hz=30.,
                          hidden_mask=None, signed_distance=None, supervise_local=True):
    """Trajectory-target losses, never alternate teacher-forced transitions.

    Recorded event labels describe the reference trajectory. They are tracking
    targets, not asserted oracle labels for an already drifted predicted pose.
    The model receives none of these targets in its recurrent transition.
    """
    prediction = result["state"]
    zero = prediction.new_zeros(())
    losses = {"state": state_loss(prediction, target_next), "events": zero,
              "target": zero, "attack": zero, "spawn": zero, "memory": zero,
              "collision": zero}
    if "event_logits" in result:
        losses["events"] = F.binary_cross_entropy_with_logits(result["event_logits"].float(), target_events)
    elif "events" in result:
        losses["events"] = F.mse_loss(result["events"].float(), target_events)
    if "target_logits" in result:
        labels = torch.where(hit_ids < 0, torch.full_like(hit_ids, 8), hit_ids)
        weights = prediction.new_ones(9)
        weights[-1] = .2
        losses["target"] = F.cross_entropy(result["target_logits"].float(), labels,
                                            weight=weights, reduction="none").mean()
    if "attack_logits" in result:
        losses["attack"] = F.binary_cross_entropy_with_logits(result["attack_logits"].float(), attack_flags,
                                                               pos_weight=prediction.new_full((8,), 4.))
    dt = 1 / simulation_hz
    losses["motion"] = F.smooth_l1_loss((prediction[:, :2] - previous_prediction[:, :2]) * (16 / (2.4 * dt)),
                                       (target_next[:, :2] - target_previous[:, :2]) * (16 / (2.4 * dt)))
    pred = prediction[:, 8:88].reshape(-1, 8, 10)
    prev = previous_prediction[:, 8:88].reshape(-1, 8, 10)
    old = target_previous[:, 8:88].reshape(-1, 8, 10)
    target = target_next[:, 8:88].reshape(-1, 8, 10)
    continuing = (old[:, :, 5] >= .5) & (target[:, :, 5] >= .5)
    npc_motion = F.smooth_l1_loss((pred[:, :, :2] - prev[:, :, :2]) * (16 / (1.2 * dt)),
                                (target[:, :, :2] - old[:, :, :2]) * (16 / (1.2 * dt)), reduction="none")
    losses["motion"] = losses["motion"] + (npc_motion * continuing[:, :, None]).sum() / (continuing.sum() * 2).clamp_min(1)
    hp_alive = F.smooth_l1_loss(pred[:, :, 4:6], target[:, :, 4:6], reduction="none").mean(-1)
    # Invisible slots retain their identity, health and life state too.
    weights = 1. + (hidden_mask.float() if hidden_mask is not None else torch.zeros_like(hp_alive))
    losses["memory"] = (hp_alive * weights).mean() + F.relu(pred[:, :, 5] - target[:, :, 5]).mean()
    if "spawn_logits" in result:
        revived = (old[:, :, 5] < .5) & (target[:, :, 5] >= .5)
        tiles = (target[:, :, :2] * 16).long().clamp(0, 15)
        labels = tiles[:, :, 0] + 16 * tiles[:, :, 1]
        spawn = F.cross_entropy(result["spawn_logits"].float().reshape(-1, 256), labels.flatten(), reduction="none").reshape(-1, 8)
        losses["spawn"] = (spawn * revived).sum() / revived.sum().clamp_min(1)
    if signed_distance is not None:
        losses["collision"] = collision_barrier(prediction, signed_distance)
    if not supervise_local:
        # Local event/displacement labels are valid at the reference pose.
        # The main free trajectory keeps only trajectory/memory/geometry loss.
        for key in ('events', 'target', 'attack', 'spawn', 'motion'):
            losses[key] = zero
    return losses


def train_autoregressive_episode(model, episode, optimizer=None, *, horizon=None,
                                chunk_size=32, prefix_start=0, device=None, initial_rgb=None,
                                resume=None, on_chunk=None, grad_clip=1., amp_dtype=None,
                                collision_weight=1., memory_weight=2., motion_weight=5.,
                                event_weight=1., state_weight=10., render_every=0,
                                render_weight=1., capture_trace=False, aligned_supervision=False,
                                gradient_accumulation_steps=1):
    """Train/evaluate a continuous own-prediction rollout, with resumable TBPTT.

    ``horizon`` counts transitions from ``prefix_start``. An explicit horizon
    exceeding the recording raises, rather than quietly claiming 9000 steps.
    Callback payloads contain a checkpointable ``resume`` only after an optimizer
    flush (or in evaluation). Every chunk detaches its graph; optimizer groups
    average chunk losses, including correctly sized final partial groups.
    Returning False within a group defers stopping until its next safe flush.
    Restoring a resume never reinitializes from source state, images or memory.
    The caller owns model/optimizer checkpoint restoration before resuming.
    """
    states, actions, events, grid, hits, attacks, label_source = _episode_arrays(episode)
    count = len(actions)
    if not 0 <= prefix_start < count or chunk_size < 1 or not math.isfinite(grad_clip) or grad_clip <= 0:
        raise ValueError("Invalid prefix, chunk size or gradient limit")
    if not isinstance(gradient_accumulation_steps, int) or gradient_accumulation_steps < 1:
        raise ValueError('Gradient accumulation steps must be a positive integer')
    horizon = count - prefix_start if horizon is None else int(horizon)
    if horizon < 1 or prefix_start + horizon > count:
        raise ValueError("Requested autonomous horizon does not fit the continuous episode")
    if render_every < 0 or min(collision_weight, memory_weight, motion_weight, event_weight, state_weight, render_weight) < 0:
        raise ValueError("Loss weights and render cadence must be nonnegative")
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    if amp_dtype is None and device.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    simulation_hz = float(getattr(model.config, "simulation_hz", 30.))
    dt = _field(episode, "dt")
    if dt is not None and abs(float(dt) - 1 / simulation_hz) > 1e-6:
        raise ValueError("Episode timestep differs from the model timestep")
    signature = _episode_key(states, actions, grid, prefix_start)
    tensor_states = torch.as_tensor(states, device=device)
    tensor_actions = torch.as_tensor(actions, device=device)
    tensor_events = torch.as_tensor(events, device=device)
    tensor_hits = torch.as_tensor(hits, device=device, dtype=torch.long)
    tensor_attacks = torch.as_tensor(attacks, device=device)
    map_tensor = torch.as_tensor(grid[None], device=device, dtype=torch.long)
    signed_distance = torch.as_tensor(map_signed_distance(grid), device=device)[None, None] if collision_weight else None
    visibility = _field(episode, "visibility")
    visible = frustum_visibility(states) if visibility is None else torch.as_tensor(visibility) > .5
    if visible.shape != (count + 1, 8):
        raise ValueError("Visibility must align with all episode states")
    visible_device = visible.to(device)
    frames = _field(episode, "frames")
    if render_every and frames is None:
        raise ValueError("RGB rollout supervision requested, but the recording has no future frames")
    names = ("state", "motion", "memory", "events", "target", "attack", "spawn", "collision", "image", "loss")
    sums = {name: 0. for name in names}
    if resume is not None:
        saved_accumulation = int(resume.get('gradient_accumulation_steps', 1))
        if (saved_accumulation != gradient_accumulation_steps
                or int(resume.get('accumulation_microstep', 0)) != 0
                or (saved_accumulation > 1 and 'chunk_size' in resume and int(resume['chunk_size']) != chunk_size)):
            raise ValueError('Resume accumulation protocol differs or contains partial gradients')
        if gradient_accumulation_steps > 1 and resume.get('horizon') != horizon:
            raise ValueError('Resume accumulation protocol uses a different horizon')
        if bool(resume.get('aligned_supervision', False)) != bool(aligned_supervision):
            raise ValueError('Resume supervision protocol differs from this rollout')
        if resume.get("format_version") != 1 or resume.get("episode_key") != signature or resume.get("prefix_start") != prefix_start:
            raise ValueError("Resume state belongs to a different episode/prefix or format")
        offset = int(resume["offset"])
        if not 0 <= offset <= horizon:
            raise ValueError("Resume offset is outside this horizon")
        current = torch.as_tensor(resume["state"], device=device).detach()
        memory = _tree_map(resume["memory"], lambda value: value.to(device).detach())
        if aligned_supervision and 'auxiliary_memory' not in resume:
            raise ValueError('Aligned rollout resume is missing its separate auxiliary memory')
        auxiliary_memory = _tree_map(resume.get('auxiliary_memory'), lambda value: value.to(device).detach())
        history = [torch.as_tensor(resume["predicted_states"], device="cpu").detach().clone()]
        if current.shape != (1, 90) or history[0].shape != (offset + 1, 90):
            raise ValueError("Invalid carried state/history dimensions")
        sums.update(resume.get("metric_sums", {}))
        if "rng_torch" in resume:
            torch.random.set_rng_state(resume["rng_torch"].cpu())
        if device.type == "cuda" and resume.get("rng_cuda"):
            torch.cuda.set_rng_state_all(resume["rng_cuda"])
        initializations = 0
    else:
        offset = 0
        current = tensor_states[prefix_start:prefix_start + 1].clone()
        memory = None
        auxiliary_memory = None
        history = [current.detach().cpu().clone()]
        initializations = 1
        if initial_rgb is None:
            initial_rgb = source_initial_image(episode, int(model.config.image_size), prefix_start,
                                              endless=bool(getattr(model.config, "endless", False)))
        initial_rgb = _rgb_tensor(initial_rgb, int(model.config.image_size), device)
    started_offset = offset
    prior_mode = model.training
    training = optimizer is not None
    model.train(training)
    optimizer_steps = map_calls = 0
    accumulation_microstep, accumulation_group_size = 0, 0
    accumulation_group_loss = 0.
    accumulation_groups = []
    trace, boundaries = [], []
    interrupted = False
    stop_requested = False
    started = time.perf_counter()
    previous_end_state = previous_end_memory = None

    def resume_payload():
        if accumulation_microstep:
            raise ValueError('Partial accumulated gradients cannot be checkpointed')
        return {"format_version": 1, "episode_key": signature, "prefix_start": prefix_start,
                "aligned_supervision": bool(aligned_supervision),
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'accumulation_microstep': 0, 'chunk_size': chunk_size,
                "offset": offset, "horizon": horizon, "state": _detached(current, cpu=True),
                "memory": _detached(memory, cpu=True), "predicted_states": torch.cat(history).clone(),
                "auxiliary_memory": _detached(auxiliary_memory, cpu=True),
                "metric_sums": dict(sums), "rng_torch": torch.random.get_rng_state().clone(),
                "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}

    try:
        while offset < horizon:
            length = min(chunk_size, horizon - offset)
            # Only a graph boundary. Never replace carried values with labels.
            current, memory = _detached(current), _detached(memory)
            auxiliary_memory = _detached(auxiliary_memory)
            boundary = {"offset": offset, "length": length, "state_reset": False, "memory_reset": False,
                        "state_has_grad_fn": current.grad_fn is not None,
                        "state_carry_max_abs_difference": _tree_difference(current, previous_end_state) if previous_end_state is not None else 0.,
                        "memory_carry_max_abs_difference": _tree_difference(memory, previous_end_memory) if previous_end_memory is not None else 0.}
            boundaries.append(boundary)
            entry = {"offset": offset, "input_state": _detached(current, cpu=True),
                     "input_memory": _detached(memory, cpu=True)} if capture_trace else None
            if training and accumulation_microstep == 0:
                remaining_chunks = math.ceil((horizon-offset)/chunk_size)
                accumulation_group_size = min(gradient_accumulation_steps, remaining_chunks)
                accumulation_group_loss = 0.
                optimizer.zero_grad(set_to_none=True)
            optimizer_updated = False
            model.train(training)
            context = torch.enable_grad() if training else torch.no_grad()
            autocast = torch.autocast(device.type, dtype=amp_dtype) if amp_dtype is not None else nullcontext()
            chunk_predictions = []
            with context, autocast:
                # Static map is re-encoded after parameter updates; no GT state
                # or visual observation is re-injected into model memory.
                map_features = model.encode_map(map_tensor)
                map_calls += 1
                if memory is None:
                    memory = model.initialize_from_image(initial_rgb, current, map_features)
                    if aligned_supervision:
                        auxiliary_memory = _tree_map(memory, lambda value: value.clone())
                    if entry is not None:
                        entry["input_memory"] = _detached(memory, cpu=True)
                totals = {name: current.new_zeros(()) for name in names if name != "loss"}
                image_count = 0
                for within in range(length):
                    index = prefix_start + offset + within
                    result = model.transition(current, tensor_actions[index:index + 1], map_features, memory)
                    losses = autoregressive_losses(result, current, tensor_states[index:index + 1],
                        tensor_states[index + 1:index + 2], tensor_events[index:index + 1],
                        tensor_hits[index:index + 1], tensor_attacks[index:index + 1],
                        simulation_hz=simulation_hz, hidden_mask=~visible_device[index + 1:index + 2],
                        signed_distance=signed_distance, supervise_local=not aligned_supervision)
                    auxiliary = None
                    if aligned_supervision:
                        # This separate stream receives exact previous GT
                        # states, while its memory is carried across chunks.
                        # Neither stream's targets enter main state/memory.
                        auxiliary = model.transition(tensor_states[index:index + 1],
                            tensor_actions[index:index + 1], map_features, auxiliary_memory)
                        auxiliary_memory = auxiliary['memory']
                        local = autoregressive_losses(auxiliary, tensor_states[index:index + 1],
                            tensor_states[index:index + 1], tensor_states[index + 1:index + 2],
                            tensor_events[index:index + 1], tensor_hits[index:index + 1],
                            tensor_attacks[index:index + 1], simulation_hz=simulation_hz,
                            signed_distance=signed_distance)
                        for key in ('events', 'target', 'attack', 'spawn', 'motion'):
                            losses[key] = local[key]
                        losses['state'] = losses['state']+.25*local['state']
                        losses['collision'] = losses['collision']+.25*local['collision']
                    for key, value in losses.items():
                        totals[key] = totals[key] + value
                    if render_every and (offset + within) % render_every == 0:
                        render_state = tensor_states[index + 1:index + 2] if aligned_supervision else result['state']
                        render_features = auxiliary['features'] if aligned_supervision else result['features']
                        rgb = model.render(render_state, render_features, map_features)
                        target = _rgb_tensor(frames[index + 1], int(model.config.image_size), device)
                        totals["image"] = totals["image"] + image_loss(rgb, target)
                        image_count += 1
                    current, memory = result["state"], result["memory"]
                    chunk_predictions.append(current.detach())
                normalized = {key: value / (max(1, image_count) if key == "image" else length)
                              for key, value in totals.items()}
                loss = (state_weight * normalized["state"] + motion_weight * normalized["motion"]
                        + memory_weight * normalized["memory"] + event_weight * (normalized["events"] + normalized["target"] + normalized["attack"])
                        + .5 * normalized["spawn"] + collision_weight * normalized["collision"] + render_weight * normalized["image"])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite autonomous rollout loss; carried state was not reset")
                if training:
                    accumulation_group_loss += float(loss.detach())/accumulation_group_size
                    (loss/accumulation_group_size).backward()
                    accumulation_microstep += 1
                    if accumulation_microstep == accumulation_group_size:
                        nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], grad_clip,
                                                 error_if_nonfinite=True)
                        optimizer.step()
                        optimizer_steps += 1
                        optimizer_updated = True
            for key, value in normalized.items():
                sums[key] += float(value.detach()) * length
            sums["loss"] += float(loss.detach()) * length
            history.append(torch.cat(chunk_predictions).cpu())
            offset += length
            current, memory = _detached(current), _detached(memory)
            auxiliary_memory = _detached(auxiliary_memory)
            previous_end_state, previous_end_memory = current.clone(), _tree_map(memory, lambda value: value.clone())
            if entry is not None:
                entry.update(output_state=_detached(current, cpu=True), output_memory=_detached(memory, cpu=True))
                trace.append(entry)
            reported_microstep = accumulation_microstep
            if optimizer_updated:
                accumulation_groups.append(accumulation_group_size)
                accumulation_microstep = 0
            checkpoint_safe = not training or optimizer_updated
            if on_chunk is not None:
                payload = {"offset": offset, "steps": length, "horizon": horizon,
                           "optimizer_steps": optimizer_steps, "metrics": {key: value / offset for key, value in sums.items()},
                           'optimizer_updated': optimizer_updated, 'checkpoint_safe': checkpoint_safe,
                           'accumulation_microstep': reported_microstep,
                           'accumulation_steps': gradient_accumulation_steps,
                           'accumulation_group_size': accumulation_group_size if training else 0,
                           'optimizer_group_loss': accumulation_group_loss if optimizer_updated else None,
                           "chunk_metrics": {**{key: float(value.detach()) for key, value in normalized.items()}, "loss": float(loss.detach())},
                           "resume": resume_payload() if checkpoint_safe else None, "boundary": boundary,
                           "state": current, "memory": memory}
                if on_chunk(payload) is False:
                    stop_requested = True
                if stop_requested and checkpoint_safe:
                    interrupted = offset < horizon
                    break
    finally:
        model.train(prior_mode)
    predictions = torch.cat(history)
    memory_report = object_memory_metrics(predictions, states[prefix_start:prefix_start + offset + 1],
        None if visibility is None else np.asarray(visibility)[prefix_start:prefix_start + offset + 1])
    return {"metrics": {key: value / max(1, offset) for key, value in sums.items()},
            "memory_metrics": memory_report, "final_state": current, "final_memory": memory,
            "resume": resume_payload(), "trace": trace,
            "declaration": {"episode_key": signature, "prefix_start": prefix_start, "horizon": horizon,
                            "rollout_steps": offset, "steps_this_call": offset - started_offset,
                            "chunk_size": chunk_size, "chunk_boundaries": boundaries,
                            "optimizer_steps": optimizer_steps, "map_encoder_calls": map_calls,
                            'gradient_accumulation_steps': gradient_accumulation_steps,
                            'accumulation_group_sizes': accumulation_groups,
                            'pending_accumulation_microsteps': accumulation_microstep,
                            "ground_truth_initializations_this_call": initializations,
                            "source_image_initializations_this_call": initializations,
                            "future_state_feedback": False, "future_image_feedback": False,
                            "teacher_forced_auxiliary_transitions": offset-started_offset if aligned_supervision else 0,
                            "local_labels_at_reference_pose_only": bool(aligned_supervision),
                            "state_carried_across_chunks": True, "memory_carried_across_chunks": True,
                            "event_labels": label_source, "event_supervision": "reference_trajectory_tracking_targets",
                            "seconds": time.perf_counter() - started, "interrupted": interrupted,
                            "stop_requested": stop_requested}}


class ReturnCombatPolicy:
    """Damage an object, leave its view, return to shoot, then revisit its corpse.

    The reference engine alone decides hits, health, death and legal respawns.
    No scene or health edits are made after the ordinary game initialization.
    """
    def __init__(self, seed=0, hz=30.):
        from .prepare_training_data import SurvivalPolicy
        self.hz, self.turn_sign = hz, 1. if seed % 2 else -1.
        self.survival = SurvivalPolicy(seed, hz)

    def __call__(self, game, index):
        # Durations are expressed in 30Hz reference ticks.
        tick = int(index * 30 / self.hz)
        if tick == 0:
            return np.array([0., 0., 0., 1.], np.float32)
        if tick <= 60:
            return np.array([1., 0., 0., 0.], np.float32)
        if tick <= 103:
            return np.array([0., 0., self.turn_sign, 0.], np.float32)
        if tick <= 112:
            enemy, player = game.enemies[0], game.player
            desired = math.atan2(enemy["y"] - player["y"], enemy["x"] - player["x"])
            error = (desired - player["angle"] + math.pi) % (2 * math.pi) - math.pi
            turn = float(np.clip(error / (2.2 / self.hz), -1, 1))
            fire = abs(error - turn * 2.2 / self.hz) < .035
            return np.array([0., 0., turn, float(fire)], np.float32)
        if tick <= 144:
            return np.array([1., 0., 0., 0.], np.float32)
        if tick <= 187:
            return np.array([0., 0., self.turn_sign, 0.], np.float32)
        if tick <= 204:
            return np.array([0., 0., 0., 1.], np.float32)
        return self.survival(game, index)


def _source_visibility(game):
    p = game.player
    c, s = math.cos(p["angle"]), math.sin(p["angle"])
    result = np.zeros(8, np.float32)
    for i, enemy in enumerate(game.enemies):
        dx, dy = enemy["x"] - p["x"], enemy["y"] - p["y"]
        forward, lateral = dx * c + dy * s, -dx * s + dy * c
        result[i] = float(forward > .12 and abs(lateral) <= forward * math.tan(math.radians(34))
                          and game._line_clear(p["x"], p["y"], enemy["x"], enemy["y"]))
    return result


def _atomic_replace(temporary, destination):
    for attempt in range(40):
        try:
            Path(temporary).replace(destination)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.05)


def generate_return_episode(path, *, seed=73, split="train", steps=512, hz=30.,
                            image_size=128, endless=True):
    """Write a faithful targeted state corpus with exactly one source RGB image."""
    from .engine import EVENT_NAMES, Game
    if steps < 1 or hz <= 0 or split not in {"train", "val"}:
        raise ValueError("Invalid return-probe length, timestep or split")
    path = Path(path)
    if path.is_file():
        with np.load(path, allow_pickle=False) as saved:
            if (str(saved["split"]) != split or int(saved["seed"]) != seed
                    or int(saved["requested_transitions"]) != steps or bool(saved["endless"]) != endless):
                raise ValueError("Existing return episode belongs to another configuration")
            return {"path": str(path), "transitions": len(saved["actions"]), "resumed": True}
    game = Game(seed=seed, endless=endless)
    policy = ReturnCombatPolicy(seed, hz)
    initial_state = json.dumps(game.get_state())
    initial_rgb = game.render(size=image_size, overlays=False)
    states = [game.state_vector()]
    visibility = [_source_visibility(game)]
    actions, events, hits, attacks, respawns = [], [], [], [], []
    for index in range(steps):
        action = policy(game, index)
        old_alive = np.array([enemy["alive"] for enemy in game.enemies])
        event = game.step(action, dt=1 / hz)
        actions.append(action)
        events.append([float(bool(event[name])) for name in EVENT_NAMES])
        hits.append(event.get("hit_id", -1))
        attacking = np.zeros(8, np.float32)
        attacking[event.get("attacker_ids", [])] = 1.
        attacks.append(attacking)
        respawns.append((~old_alive & np.array([enemy["alive"] for enemy in game.enemies])).astype(np.float32))
        states.append(game.state_vector())
        visibility.append(_source_visibility(game))
        if game.player["hp"] <= 0:
            break
    states, visibility = np.asarray(states), np.asarray(visibility)
    diagnostic = object_memory_metrics(states, states, visibility)
    annotations = {key: diagnostic[key] for key in ("hidden_object_samples", "reappearance_samples", "reference_respawns")}
    payload = {"format_version": np.asarray(4), "probe_format_version": np.asarray(1),
               "map": game.map.astype(np.int8), "states": states,
               "actions": np.asarray(actions, np.float32), "events": np.asarray(events, np.float32),
               "hit_ids": np.asarray(hits, np.int64), "attack_flags": np.asarray(attacks, np.float32),
               "respawn_flags": np.asarray(respawns, np.float32), "visibility": visibility,
               "initial_rgb": initial_rgb, "initial_state": np.asarray(initial_state),
               "seed": np.asarray(seed), "split": np.asarray(split), "endless": np.asarray(endless),
               "policy": np.asarray("damage_leave_return_shoot_revisit"), "dt": np.asarray(1 / hz),
               "requested_transitions": np.asarray(steps), "probe_annotations": np.asarray(json.dumps(annotations))}
    # Existing long loader expects paired geometry fields. Exact visibility is
    # preserved for memory probes; depth is real training-oracle geometry too.
    from .geometry_targets import build_geometry_targets
    payload["depth"] = np.stack([build_geometry_targets(game.map, state)["depth"] for state in states])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    _atomic_replace(temporary, path)
    return {"path": str(path), "transitions": len(actions), "resumed": False,
            "split": split, "seed": seed, "terminal": bool(states[-1, 4] <= 0),
            "reappearance_samples": diagnostic["reappearance_samples"],
            "events": dict(zip(EVENT_NAMES, np.asarray(events).sum(0).astype(int).tolist()))}


def prepare_return_corpus(directory, *, seed=73, train_episodes=8, val_episodes=2,
                          steps=512, hz=30., image_size=128, endless=True):
    """Prepare disjoint seeded train/val probe files without resets inside them."""
    root = Path(directory)
    records = []
    for split, count, offset in (("train", train_episodes, 100000), ("val", val_episodes, 200000)):
        for index in range(count):
            records.append(generate_return_episode(root / split / f"return_{index:03d}.npz",
                seed=seed + offset + index, split=split, steps=steps, hz=hz,
                image_size=image_size, endless=endless))
    return {"directory": str(root), "train_episodes": train_episodes, "val_episodes": val_episodes,
            "seeds_disjoint": True, "recordings": records, "reference_resets_inside_recordings": 0}


def _corpus_files(directory, *, endless):
    files = []
    for path in sorted(Path(directory).glob("*.npz")):
        with np.load(path, allow_pickle=False) as saved:
            if "split" in saved.files and str(saved["split"]) != "train":
                raise ValueError("Training directory includes a non-training episode")
            if "endless" in saved.files and bool(saved["endless"]) != endless:
                raise ValueError("Corpus endless rules differ from the model")
            files.append({"path": str(path.resolve()), "transitions": len(saved["actions"]),
                          "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns})
    return files


def train_curriculum(model, long_directory, optimizer=None, device=None, chunk_size=32,
                     resume=None, on_chunk=None, seed=73, return_directory=None, *,
                     schedule=CURRICULUM_SCHEDULE, amp_dtype=None, capture_trace=False,
                     aligned_supervision=False, gradient_accumulation_steps=1):
    """Run the requested curriculum for either teacher or student.

    Each task is selected only from recordings with enough transitions for
    its horizon. Short return probes alternate with long recordings
    for the short horizons. Validation probes are generated but never sampled.
    ``on_chunk`` receives ``resume_state``, real cumulative metrics, transition
    counts, current horizon and a measured ETA. Only optimizer flushes have a
    checkpointable resume_state; a mid-group stop request waits for that flush.
    """
    from .long_train import load_episode
    schedule = tuple(int(horizon) for horizon in schedule)
    if not schedule or min(schedule) < 1 or chunk_size < 1:
        raise ValueError("Curriculum horizons and chunk size must be positive")
    if not isinstance(gradient_accumulation_steps, int) or gradient_accumulation_steps < 1:
        raise ValueError('Gradient accumulation steps must be a positive integer')
    long_directory = Path(long_directory).resolve()
    return_directory = Path(return_directory).resolve() if return_directory else long_directory.parents[1] / "return"
    endless = bool(getattr(model.config, "endless", False))
    corpus_report = prepare_return_corpus(return_directory, seed=seed, endless=endless,
        hz=float(getattr(model.config, "simulation_hz", 30.)), image_size=int(model.config.image_size))
    long_files = _corpus_files(long_directory, endless=endless)
    return_files = _corpus_files(return_directory / "train", endless=endless)
    if not long_files:
        raise ValueError(f"No continuous training recordings in {long_directory}")
    descriptors = {"long": long_files, "return": return_files}
    fingerprint = hashlib.sha256(json.dumps(descriptors, sort_keys=True).encode()).hexdigest()
    rng = np.random.default_rng(seed)
    plan = []
    for index, horizon in enumerate(schedule):
        use_return = horizon <= 512 and index % 2 == 0
        kind = "return" if use_return and any(item["transitions"] >= horizon for item in return_files) else "long"
        eligible = [item for item in descriptors[kind] if item["transitions"] >= horizon]
        if not eligible:
            raise ValueError(f"No genuine continuous {horizon}-step {kind} recording; cannot shorten this task")
        chosen = eligible[int(rng.integers(len(eligible)))]
        prefix = 0 if kind == "return" or horizon == 9000 else int(rng.integers(chosen["transitions"] - horizon + 1))
        plan.append({"kind": kind, "path": chosen["path"], "horizon": horizon, "prefix_start": prefix})
    total = sum(schedule)
    if resume is not None:
        if (int(resume.get('gradient_accumulation_steps', 1)) != gradient_accumulation_steps
                or int(resume.get('accumulation_microstep', 0)) != 0
                or (gradient_accumulation_steps > 1 and 'chunk_size' in resume and int(resume['chunk_size']) != chunk_size)):
            raise ValueError('Curriculum resume accumulation protocol differs or contains partial gradients')
        if (resume.get("format_version") != 1 or resume.get("dataset_fingerprint") != fingerprint
                or resume.get("seed") != seed or tuple(resume.get("schedule", ())) != schedule
                or resume.get("plan") != plan):
            raise ValueError("Curriculum resume does not match the immutable dataset, seed or schedule")
        task_index = int(resume["task_index"])
        task_resume = resume.get("task_resume")
        finished_sums = dict(resume.get("finished_metric_sums", {}))
    else:
        task_index, task_resume, finished_sums = 0, None, {}
    if not 0 <= task_index <= len(plan):
        raise ValueError("Invalid resumed task index")
    completed_before = sum(schedule[:task_index])
    start_completed = completed_before + (int(task_resume["offset"]) if task_resume else 0)
    started = time.perf_counter()
    last_state = None
    last_progress_metrics = None
    reports = []
    interrupted = False
    optimizer_steps = 0

    def wrapper_resume(index, inner):
        return {"format_version": 1, "dataset_fingerprint": fingerprint, "seed": seed,
                "schedule": list(schedule), "plan": plan, "task_index": index,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'accumulation_microstep': 0, 'chunk_size': chunk_size,
                "task_resume": inner, "finished_metric_sums": dict(finished_sums)}

    while task_index < len(plan):
        task = plan[task_index]
        episode = load_episode(task["path"], load_frames=False)

        def callback(progress):
            nonlocal last_state, last_progress_metrics
            done = completed_before + progress["offset"]
            metrics = {key: (finished_sums.get(key, 0.) + value * progress["offset"]) / max(1, done)
                       for key, value in progress["metrics"].items()}
            last_state = wrapper_resume(task_index, progress['resume']) if progress['checkpoint_safe'] else None
            last_progress_metrics = metrics
            advanced = done - start_completed
            eta = (time.perf_counter() - started) / advanced * (total - done) if advanced > 0 else None
            payload = {"resume_state": last_state, "metrics": metrics,
                       'optimizer_updated': progress['optimizer_updated'],
                       'checkpoint_safe': progress['checkpoint_safe'],
                       'accumulation_microstep': progress['accumulation_microstep'],
                       'accumulation_steps': gradient_accumulation_steps,
                       'accumulation_group_size': progress['accumulation_group_size'],
                       'optimizer_group_loss': progress['optimizer_group_loss'],
                       'optimizer_steps': optimizer_steps+progress['optimizer_steps'],
                       "completed_transitions": done, "total_transitions": total,
                       "horizon": task["horizon"], "eta_seconds": eta,
                       "task_index": task_index, "task_count": len(plan), "episode_kind": task["kind"],
                       "chunk": progress, "state": progress["state"], "memory": progress["memory"]}
            return on_chunk(payload) if on_chunk is not None else None

        report = train_autoregressive_episode(model, episode, optimizer, horizon=task["horizon"],
            prefix_start=task["prefix_start"], chunk_size=chunk_size, device=device,
            resume=task_resume, on_chunk=callback, amp_dtype=amp_dtype, capture_trace=capture_trace,
            aligned_supervision=aligned_supervision, gradient_accumulation_steps=gradient_accumulation_steps)
        optimizer_steps += report['declaration']['optimizer_steps']
        reports.append({"task": task, "metrics": report["metrics"], "memory_metrics": report["memory_metrics"],
                        "declaration": report["declaration"]})
        if report["declaration"]["interrupted"]:
            interrupted = True
            break
        for key, value in report["metrics"].items():
            finished_sums[key] = finished_sums.get(key, 0.) + value * task["horizon"]
        completed_before += task["horizon"]
        task_index += 1
        task_resume = None
        last_state = wrapper_resume(task_index, None)
        if report["declaration"].get("stop_requested") and task_index < len(plan):
            interrupted = True
            break
    done = (completed_before + int(last_state["task_resume"]["offset"]) if interrupted and last_state["task_resume"] else completed_before)
    metrics = last_progress_metrics if interrupted else {key: value / max(1, done) for key, value in finished_sums.items()}
    return {"metrics": metrics, "resume_state": last_state or wrapper_resume(task_index, task_resume),
            "completed_transitions": done, "total_transitions": total,
            "tasks": reports, "return_corpus": corpus_report,
            "declaration": {"schedule": list(schedule), "dataset_fingerprint": fingerprint,
                            "autonomous_9000_step_task_completed": not interrupted and 9000 in schedule and task_index == len(plan),
                            "target_rollout_steps": max(schedule),
                            "autonomous_target_rollout_completed": not interrupted and task_index == len(plan),
                            "teacher_forced_future_inputs": 0, "chunk_size": chunk_size,
                            "aligned_auxiliary_supervision": bool(aligned_supervision),
                            'gradient_accumulation_steps': gradient_accumulation_steps,
                            'optimizer_steps': optimizer_steps,
                            "validation_episodes_used_for_training": 0, "interrupted": interrupted,
                            "seconds": time.perf_counter() - started}}
