"""Stateful TBPTT preparation for continuous, long world-model rollouts.

`--updates` counts continuous segments, not optimizer steps. A segment begins
from one recorded state, runs a closed-loop no-gradient burn-in, then carries
both predicted state and recurrent memory across optimizer/chunk boundaries.
Only their graphs are detached. Static-map features are re-encoded per chunk.
RGB is optional; the default trains dynamics without loading image arrays.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .models import (ACTION_DIM, ENTITY_DIM, ENTITY_SLOTS, EVENT_DIM, GLOBAL_DIM,
                     STATE_DIM, STATE_SCHEMA, NeuralWorldModel, WorldModelConfig,
                     load_world_checkpoint)
from .train import (combat_label_arrays, discover_files, image_loss,
                    normalize_world_config, state_loss, supervised_event_losses)


@dataclass
class LongEpisode:
    path: Path
    states: np.ndarray
    actions: np.ndarray
    events: np.ndarray
    grid: np.ndarray
    hit_ids: np.ndarray
    attack_flags: np.ndarray
    label_source: str
    dt: float | None = None
    frames: np.ndarray | None = None
    depth: np.ndarray | None = None
    visibility: np.ndarray | None = None

    @property
    def transitions(self):
        return len(self.actions)


def load_episode(path, load_frames=False):
    """Load one continuous episode; no RGB field is required or synthesized."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"states", "actions", "events", "map"}
        if not required.issubset(data.files):
            raise ValueError(f"{path.name}: missing {sorted(required.difference(data.files))}")
        states = np.asarray(data["states"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        events = np.asarray(data["events"], dtype=np.float32)
        grid = np.asarray(data["map"], dtype=np.int64)
        steps = len(actions)
        if steps < 1 or states.shape != (steps + 1, STATE_DIM):
            raise ValueError(f"{path.name}: expected nonempty states [T+1,{STATE_DIM}]")
        if actions.shape != (steps, ACTION_DIM) or events.shape != (steps, EVENT_DIM):
            raise ValueError(f"{path.name}: expected actions/events [T,4]")
        if grid.shape != (16, 16):
            raise ValueError(f"{path.name}: expected map [16,16]")
        if not all(np.isfinite(array).all() for array in (states, actions, events)):
            raise ValueError(f"{path.name}: non-finite states, actions or events")
        hits, attacks, source = combat_label_arrays(data)
        frames = None
        if load_frames and "frames" in data.files:
            frames = np.asarray(data["frames"], dtype=np.uint8)
            if frames.ndim != 4 or len(frames) != steps + 1 or frames.shape[-1] != 3:
                raise ValueError(f"{path.name}: frames must align as [T+1,H,W,3]")
        depth = visibility = None
        if ("depth" in data.files) != ("visibility" in data.files):
            raise ValueError(f"{path.name}: depth and visibility must be paired")
        if "depth" in data.files:
            depth = np.asarray(data["depth"], dtype=np.float32)
            visibility = np.asarray(data["visibility"], dtype=np.float32)
            if depth.shape != (steps + 1, 32) or visibility.shape != (steps + 1, ENTITY_SLOTS):
                raise ValueError(f"{path.name}: expected depth [T+1,32], visibility [T+1,8]")
        dt = float(data["dt"]) if "dt" in data.files else None
        if dt is not None and (not np.isfinite(dt) or dt <= 0):
            raise ValueError(f"{path.name}: invalid dt")
        return LongEpisode(path, states, actions, events, grid, hits, attacks, source,
                           dt, frames, depth, visibility)


def map_signed_distance(grid):
    """Approximate tile-center signed distances, measured in world tile units.

    This differentiable training regularizer is not an exact collision solver.
    Actual geometry validity is evaluated separately without state correction.
    """
    occupied = np.asarray(grid) != 0
    y, x = np.mgrid[:16, :16]
    points = np.stack([x, y], axis=-1).astype(np.float32)
    blocked, free = points[occupied], points[~occupied]
    def nearest(locations):
        if not len(locations):
            return np.full((16, 16), 16., dtype=np.float32)
        return np.sqrt(((points[:, :, None, :] - locations[None, None, :, :]) ** 2).sum(-1)).min(-1)
    positive = nearest(blocked) - .5
    negative = -(nearest(free) - .5)
    return np.where(occupied, negative, positive).astype(np.float32)


def collision_barrier(state, signed_distance):
    """Soft learned-trajectory penalty; never moves/clamps actors out of walls."""
    entities = state[:, GLOBAL_DIM:-2].reshape(-1, ENTITY_SLOTS, ENTITY_DIM)
    positions = torch.cat([state[:, None, :2], entities[:, :, :2]], dim=1)
    sample_grid = (positions * 2 - 1).unsqueeze(2)
    distances = F.grid_sample(signed_distance.expand(state.shape[0], -1, -1, -1),
                              sample_grid, mode="bilinear", padding_mode="border",
                              align_corners=False)[:, 0, :, 0]
    radii = state.new_tensor([.20] + [.25] * ENTITY_SLOTS)[None]
    wall_penalty = F.relu(radii + .03 - distances).square()
    # Border sampling has zero gradient outside the map; explicit bounds give
    # a useful training gradient there too. This does not alter inference.
    world_positions = positions * 16
    bounds = (F.relu(radii[..., None] - world_positions).square()
              + F.relu(world_positions - (16 - radii[..., None])).square()).sum(-1)
    weights = torch.cat([torch.ones_like(state[:, :1]), entities[:, :, 5].clamp(0, 1)], dim=-1)
    return ((wall_penalty + bounds) * weights).sum() / weights.sum().clamp_min(1)


def _image_target(frame, size, device):
    tensor = torch.as_tensor(frame.copy(), device=device).permute(2, 0, 1)[None].float() / 255
    if tensor.shape[-2:] != (size, size):
        tensor = F.interpolate(tensor, (size, size), mode="bilinear", align_corners=False, antialias=True)
    return tensor


def curriculum_horizon(index, updates, rollout, curriculum):
    if not curriculum:
        return rollout
    stage = min(len(curriculum) - 1, index * len(curriculum) // max(1, updates))
    return min(rollout, curriculum[stage])


def choose_segment(episodes, rng, horizon, burn_in, start_mode="random", event_fraction=.25):
    """Sample whole continuous segments, optionally covering rare hit/attacks.

    Positive-event starts alter the training distribution. Validation never
    uses this sampler, and classifier probabilities are not assumed calibrated.
    """
    event_pool = []
    for episode in episodes:
        positive = np.flatnonzero((episode.hit_ids >= 0) | (episode.attack_flags > .5).any(-1))
        event_pool.extend((episode, int(index)) for index in positive)
    if event_pool and rng.random() < event_fraction:
        episode, event_index = rng.choice(event_pool)
        burn = min(burn_in, event_index)
        length = min(horizon, episode.transitions - burn)
        earliest = max(0, event_index - burn - length + 1)
        latest = min(event_index - burn, episode.transitions - burn - length)
        prefix = rng.randint(earliest, latest)
        return episode, prefix, burn, {"mode": "positive_event", "event_index": event_index}
    episode = rng.choice(episodes)
    length = min(horizon, max(1, episode.transitions - min(burn_in, episode.transitions - 1)))
    burn = min(burn_in, max(0, episode.transitions - length))
    max_prefix = episode.transitions - length - burn
    prefix = rng.randint(0, max_prefix) if start_mode == "random" else 0
    return episode, prefix, burn, {"mode": "ordinary_continuous", "event_index": None}


def run_segment(model, episode, *, optimizer=None, teacher=None, device="cpu",
                prefix_start=0, rollout=256, chunk=32, burn_in=32,
                render_every=0, geometry_every=32, collision_weight=1.0,
                distill_weight=.1, none_class_weight=.2, attack_pos_weight=4., capture_trace=False):
    """Train/evaluate one uninterrupted segment with real stateful TBPTT.

    A numerical state is initialized exactly once from `states[prefix_start]`.
    No ground-truth state is injected into the main rollout after that point.
    The separate GT-state auxiliary stream supplies correctly posed event and
    geometry labels. It also carries its own recurrent memory across chunks.
    """
    if min(rollout, chunk) < 1 or min(prefix_start, burn_in, render_every, geometry_every) < 0:
        raise ValueError("invalid rollout/chunk/prefix/burn-in/render cadence")
    if prefix_start >= episode.transitions:
        raise ValueError("prefix starts after the last transition")
    available = episode.transitions - prefix_start
    burn_in = min(burn_in, max(0, available - 1))
    rollout = min(rollout, available - burn_in)
    device = torch.device(device)
    training = optimizer is not None
    model.train(training)
    if teacher is not None:
        teacher.eval()
    states = torch.as_tensor(episode.states, device=device)
    actions = torch.as_tensor(episode.actions, device=device)
    events = torch.as_tensor(episode.events, device=device)
    hit_ids = torch.as_tensor(episode.hit_ids, device=device)
    attack_flags = torch.as_tensor(episode.attack_flags, device=device)
    grid = torch.as_tensor(episode.grid[None], device=device)
    sdf = torch.as_tensor(map_signed_distance(episode.grid), device=device)[None, None]
    current_state = states[prefix_start:prefix_start + 1].clone()
    memory = auxiliary_memory = teacher_memory = None
    teacher_map = None
    map_encode_calls = 0
    with torch.no_grad():
        if teacher is not None:
            teacher_map = teacher.encode_map(grid)
        if burn_in:
            burn_map = model.encode_map(grid)
            map_encode_calls += 1
            for step in range(prefix_start, prefix_start + burn_in):
                action = actions[step:step + 1]
                result = model.transition(current_state, action, burn_map, memory)
                auxiliary = model.transition(states[step:step + 1], action, burn_map, auxiliary_memory)
                if teacher is not None:
                    teach = teacher.transition(current_state, action, teacher_map, teacher_memory)
                    teacher_memory = teach["memory"]
                current_state, memory, auxiliary_memory = result["state"], result["memory"], auxiliary["memory"]
    metrics = {key: 0. for key in ["state", "events", "target", "attack", "collision",
                                  "distill", "image", "depth", "visibility", "loss"]}
    label_steps = image_steps = geometry_steps = optimizer_steps = detach_boundaries = 0
    boundaries, trace = [], []
    diagnostics = {name: torch.zeros((), dtype=torch.int64, device=device) for name in ["negative_hit_steps", "false_hit_events",
                    "negative_target_steps", "false_target_predictions",
                    "negative_attack_slots", "false_attack_head_predictions",
                    "false_accepted_attacks"]}
    previous_end_state = previous_end_memory = None
    segment_started = time.perf_counter()
    first_step = prefix_start + burn_in
    for offset in range(0, rollout, chunk):
        length = min(chunk, rollout - offset)
        current_state = current_state.detach()
        memory = None if memory is None else memory.detach()
        auxiliary_memory = None if auxiliary_memory is None else auxiliary_memory.detach()
        teacher_memory = None if teacher_memory is None else teacher_memory.detach()
        if offset:
            detach_boundaries += 1
        boundary = {"offset": offset, "length": length, "state_has_grad_fn": current_state.grad_fn is not None,
                    "memory_has_grad_fn": memory is not None and memory.grad_fn is not None,
                    "state_reset": False, "memory_reset": False}
        if previous_end_state is not None:
            boundary["state_carry_max_abs_difference"] = float((current_state - previous_end_state).abs().max())
            boundary["memory_carry_max_abs_difference"] = float((memory - previous_end_memory).abs().max())
        boundaries.append(boundary)
        trace_item = None
        if capture_trace:
            trace_item = {"input_state": current_state.detach().cpu().clone(),
                          "input_memory": None if memory is None else memory.detach().cpu().clone()}
        if training:
            optimizer.zero_grad(set_to_none=True)
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            # Recompute AFTER the preceding optimizer step. Never reuse a
            # freed graph or stale trainable map encoding across TBPTT chunks.
            map_features = model.encode_map(grid)
            map_encode_calls += 1
            totals = {key: current_state.new_zeros(()) for key in metrics if key != "loss"}
            local_images = local_geometry = 0
            for within in range(length):
                step = first_step + offset + within
                action = actions[step:step + 1]
                result = model.transition(current_state, action, map_features, memory)
                totals["state"] = totals["state"] + state_loss(result["state"], states[step + 1:step + 2])
                auxiliary = model.transition(states[step:step + 1], action, map_features, auxiliary_memory)
                auxiliary_memory = auxiliary["memory"]
                losses = supervised_event_losses(auxiliary, hit_ids[step:step + 1],
                                                  attack_flags[step:step + 1], events[step:step + 1],
                                                  none_class_weight, attack_pos_weight)
                for name, loss in losses.items():
                    totals[name] = totals[name] + loss
                with torch.no_grad():
                    negative_hit = events[step, 1] < .5
                    predicted_hit = (auxiliary["events"][0, 1] > .5 if "events" in auxiliary else
                                     auxiliary["event_logits"][0, 1] > 0)
                    diagnostics["negative_hit_steps"] += negative_hit.long()
                    diagnostics["false_hit_events"] += (negative_hit & predicted_hit).long()
                    if "target_logits" in auxiliary:
                        negative_target = hit_ids[step] < 0
                        diagnostics["negative_target_steps"] += negative_target.long()
                        diagnostics["false_target_predictions"] += (negative_target &
                            (auxiliary["target_logits"].argmax(-1)[0] < ENTITY_SLOTS)).long()
                    if "attack_logits" in auxiliary:
                        negative_attacks = attack_flags[step] < .5
                        diagnostics["negative_attack_slots"] += negative_attacks.sum()
                        diagnostics["false_attack_head_predictions"] += (
                            negative_attacks & (auxiliary["attack_logits"][0] >= 0)).sum()
                        if "accepted_attacks" in auxiliary:
                            diagnostics["false_accepted_attacks"] += (
                                negative_attacks & auxiliary["accepted_attacks"][0]).sum()
                if collision_weight:
                    totals["collision"] = totals["collision"] + collision_barrier(result["state"], sdf)
                if teacher is not None:
                    with torch.no_grad():
                        teach = teacher.transition(current_state.detach(), action, teacher_map, teacher_memory)
                        teacher_memory = teach["memory"]
                    kd = state_loss(result["state"], teach["state"])
                    kd = kd + .1 * F.mse_loss(result["features"], teach["features"])
                    if "target_logits" in result and "target_logits" in teach:
                        kd = kd + F.kl_div(F.log_softmax(result["target_logits"], -1),
                                           F.softmax(teach["target_logits"], -1), reduction="batchmean")
                    if "attack_logits" in result and "attack_logits" in teach:
                        kd = kd + F.binary_cross_entropy_with_logits(result["attack_logits"], teach["attack_logits"].sigmoid())
                    totals["distill"] = totals["distill"] + kd
                relative_step = offset + within
                if render_every and episode.frames is not None and relative_step % render_every == 0:
                    rgb = model.render(result["state"], result["features"], map_features)
                    target = _image_target(episode.frames[step + 1], model.config.image_size, device)
                    totals["image"] = totals["image"] + image_loss(rgb, target)
                    local_images += 1
                if (geometry_every and episode.depth is not None
                        and getattr(model.config, "architecture", "legacy_cnn") == "camera_transformer"
                        and relative_step % geometry_every == 0):
                    geometry = model.geometry(states[step + 1:step + 2], map_features)
                    target_depth = torch.as_tensor(episode.depth[step + 1:step + 2], device=device)
                    target_visibility = torch.as_tensor(episode.visibility[step + 1:step + 2], device=device)
                    totals["depth"] = totals["depth"] + F.smooth_l1_loss(
                        torch.log1p(geometry["depth"].clamp_min(0)), torch.log1p(target_depth.clamp_min(0)))
                    totals["visibility"] = totals["visibility"] + F.binary_cross_entropy_with_logits(
                        geometry["visibility_logits"], target_visibility)
                    local_geometry += 1
                current_state, memory = result["state"], result["memory"]
            normalized = {name: value / (max(1, local_images) if name == "image" else
                                          max(1, local_geometry) if name in {"depth", "visibility"} else length)
                          for name, value in totals.items()}
            loss = (10 * normalized["state"] + normalized["events"] + normalized["target"]
                    + normalized["attack"] + collision_weight * normalized["collision"]
                    + distill_weight * normalized["distill"] + normalized["image"]
                    + .2 * (normalized["depth"] + normalized["visibility"]))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite TBPTT loss; checkpoint not overwritten")
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                optimizer_steps += 1
        for name, value in normalized.items():
            metrics[name] += float(value.detach()) * length
        metrics["loss"] += float(loss.detach()) * length
        label_steps += length
        image_steps += local_images
        geometry_steps += local_geometry
        previous_end_state, previous_end_memory = current_state.detach().clone(), memory.detach().clone()
        if trace_item is not None:
            trace_item.update(output_state=previous_end_state.cpu(), output_memory=previous_end_memory.cpu())
            trace.append(trace_item)
    metrics = {key: value / label_steps for key, value in metrics.items()}
    diagnostics = {name: int(value.detach().cpu()) for name, value in diagnostics.items()}
    declaration = {
        "episode": str(episode.path), "label_source": episode.label_source,
        "prefix_start": prefix_start, "ground_truth_initializations": 1,
        "burn_in_steps": burn_in, "burn_in_mode": "closed_loop_no_gradient",
        "rollout_steps": rollout, "chunk_size": chunk, "optimizer_steps": optimizer_steps,
        "chunk_count": len(boundaries), "detach_boundaries": detach_boundaries,
        "state_carried_across_chunks": True, "memory_carried_across_chunks": True,
        "map_encoder_calls": map_encode_calls, "map_reencoded_each_chunk": True,
        "event_auxiliary_stream": "recorded_state_and_own_carried_memory",
        "rgb_available": episode.frames is not None, "rendered_supervision_steps": image_steps,
        "geometry_supervision_steps": geometry_steps, "chunk_boundaries": boundaries,
        "event_diagnostics": diagnostics,
        "negative_hit_false_positive_rate": (diagnostics["false_hit_events"] / diagnostics["negative_hit_steps"]
                                               if diagnostics["negative_hit_steps"] else None),
        "negative_target_false_positive_rate": (diagnostics["false_target_predictions"] / diagnostics["negative_target_steps"]
                                                  if diagnostics["negative_target_steps"] else None),
        "negative_attack_head_false_positive_rate": (diagnostics["false_attack_head_predictions"] / diagnostics["negative_attack_slots"]
                                                       if diagnostics["negative_attack_slots"] else None),
        "seconds": time.perf_counter() - segment_started,
    }
    return {"metrics": metrics, "declaration": declaration, "trace": trace,
            "final_state": current_state.detach(), "final_memory": memory.detach()}


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--kind", choices=["teacher", "student"], default="student")
    parser.add_argument("--architecture", choices=["camera_transformer", "legacy_cnn"], default="camera_transformer")
    parser.add_argument("--transition-mode", choices=["event_v1", "residual_v0"], default="event_v1")
    parser.add_argument("--image-size", type=int, choices=[128, 256], default=128)
    parser.add_argument("--hz", type=float, default=30.)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--rollout", type=int, default=256)
    parser.add_argument("--burn-in", type=int, default=32)
    parser.add_argument("--updates", type=int, default=1000,
                        help="Total segment-update target, including segments already completed when resuming")
    parser.add_argument("--curriculum", default="16,64,256",
                        help="Comma-separated horizon caps, evenly scheduled over --updates; empty disables")
    parser.add_argument("--render-every", type=int, default=0,
                        help="0 disables RGB entirely; otherwise supervise every Nth rollout step if frames exist")
    parser.add_argument("--geometry-every", type=int, default=32)
    parser.add_argument("--collision-weight", type=float, default=1.)
    parser.add_argument("--distill-weight", type=float, default=.1)
    parser.add_argument("--event-fraction", type=float, default=.25,
                        help="Nominal fraction of training segments covering positive hit/attack labels")
    parser.add_argument("--none-class-weight", type=float, default=.2)
    parser.add_argument("--attack-pos-weight", type=float, default=4.)
    parser.add_argument("--teacher")
    parser.add_argument("--resume")
    parser.add_argument("--freeze-map", action="store_true",
                        help="Freeze the shared map encoder, useful for state-only continuation of RGB-trained weights")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--start-mode", choices=["random", "initial"], default="random")
    parser.add_argument("--val-fraction", type=float, default=.2)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--output", default="checkpoints/student_long.pt")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if min(args.chunk, args.rollout, args.updates, args.threads, args.save_every, args.log_every) < 1:
        raise ValueError("counts must be positive")
    if min(args.burn_in, args.render_every, args.geometry_every, args.collision_weight, args.distill_weight) < 0:
        raise ValueError("cadences, burn-in and loss weights cannot be negative")
    if not 0 <= args.val_fraction < 1 or args.hz <= 0:
        raise ValueError("val-fraction must be in [0,1); hz must be positive")
    if not 0 <= args.event_fraction <= 1 or min(args.none_class_weight, args.attack_pos_weight) <= 0:
        raise ValueError("event-fraction must be in [0,1] and class weights must be positive")
    if args.teacher and args.kind != "student":
        raise ValueError("teacher distillation is supported for student training")
    curriculum = [int(value) for value in args.curriculum.split(",") if value.strip()]
    if any(value < 1 for value in curriculum) or curriculum != sorted(curriculum):
        raise ValueError("curriculum must contain increasing positive horizon caps")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    paths = discover_files(args.data)
    rng.shuffle(paths)
    n_val = max(1, round(len(paths) * args.val_fraction)) if len(paths) > 1 and args.val_fraction > 0 else 0
    val_paths, train_paths = paths[:n_val], paths[n_val:]
    # State-only episodes are deliberately inexpensive to keep resident. RGB is
    # loaded only when an explicit nonzero sparse-render cadence requests it.
    episodes = [load_episode(path, load_frames=args.render_every > 0) for path in train_paths]
    validation_episodes = [load_episode(path, load_frames=args.render_every > 0) for path in val_paths]
    for episode in episodes + validation_episodes:
        if episode.dt is not None and not np.isclose(episode.dt, 1 / args.hz, rtol=1e-4, atol=1e-7):
            raise ValueError(f"{episode.path.name}: dt={episode.dt} disagrees with --hz={args.hz}")
        if episode.dt is None:
            warnings.warn(f"{episode.path.name}: no dt metadata; assuming --hz={args.hz}", RuntimeWarning)
        if args.render_every and episode.frames is None:
            print(json.dumps({"episode": episode.path.name, "rgb_loss": "disabled_missing_frames"}), flush=True)
    config = WorldModelConfig(kind=args.kind, image_size=args.image_size, architecture=args.architecture,
                              transition_mode=args.transition_mode, simulation_hz=args.hz)
    model = NeuralWorldModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    teacher = None
    if args.teacher:
        teacher, teacher_metadata = load_world_checkpoint(args.teacher, device)
        teacher.requires_grad_(False)
        if teacher_metadata.get("training_steps", 0) < 1:
            raise ValueError("teacher has no supervised training steps")
        if not np.isclose(getattr(teacher.config, "simulation_hz", 30.), args.hz):
            raise ValueError("teacher simulation_hz differs from the student/data timestep")
    completed_segments = total_optimizer_steps = event_sampling_segments = 0
    declarations = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("stage") != "world" or normalize_world_config(checkpoint.get("config", {})) != asdict(config):
            raise ValueError("resume world architecture/transition/configuration mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        total_optimizer_steps = int(checkpoint.get("training_steps", 0))
        completed_segments = int(checkpoint.get("segment_updates", 0))
        event_sampling_segments = int(checkpoint.get("event_sampling_segments", 0))
        declarations = list(checkpoint.get("training_declarations", []))[-100:]
        if "sampler_random_state" in checkpoint:
            rng.setstate(checkpoint["sampler_random_state"])
        if "torch_random_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_random_state"].cpu())
    if args.freeze_map:
        backend = getattr(model, "camera_backend", model)
        backend.map_encoder.requires_grad_(False)
    if completed_segments >= args.updates:
        raise ValueError("--updates must exceed the checkpoint's completed segment_updates")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    last_result = validation = None
    started = time.perf_counter()
    print(json.dumps({"trainer": "stateful_tbptt", "config": asdict(config), "segment_update_target": args.updates,
                      "optimizer_steps_per_segment": "ceil(actual_rollout/chunk)",
                      "curriculum": curriculum, "chunk": args.chunk, "burn_in": args.burn_in,
                      "render_every": args.render_every, "train_episodes": [path.name for path in train_paths],
                      "validation_episodes": [path.name for path in val_paths],
                      "event_fraction_nominal": args.event_fraction,
                      "none_class_weight": args.none_class_weight, "attack_pos_weight": args.attack_pos_weight,
                      "calibration": "Event oversampling/class weights change the training distribution; probabilities are not calibrated.",
                      "map_encoder_frozen": args.freeze_map,
                      "shared_renderer_warning": ("State-only updates can change shared map/geometry features. After dynamics training, fit RGB with train --renderer-only before SR/evaluation."
                                                  if args.render_every == 0 else None),
                      "validation_sampling": "initial held-out trajectory, without positive-event oversampling"}), flush=True)

    def save_checkpoint(completed):
        payload = {"format_version": 3, "stage": "world", "trainer": "stateful_tbptt",
                   "config": asdict(config), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "training_steps": total_optimizer_steps, "segment_updates": completed,
                   "event_sampling_segments": event_sampling_segments,
                   "event_fraction_realized": event_sampling_segments / max(1, completed),
                   "event_probability_calibration": "Uncalibrated after positive-event oversampling and class reweighting.",
                   "state_schema": STATE_SCHEMA, "last_metrics": last_result["metrics"],
                   "last_tbptt_declaration": last_result["declaration"],
                   "training_declarations": declarations, "validation": validation,
                   "training_files": [str(path.resolve()) for path in train_paths],
                   "validation_files": [str(path.resolve()) for path in val_paths],
                   "training_arguments": vars(args), "sampler_random_state": rng.getstate(),
                   "torch_random_state": torch.get_rng_state(),
                   "quality_status": "Unverified: long-rollout training preparation is not evidence of 300-second gameplay accuracy."}
        temporary = output.with_suffix(output.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(output)

    for index in range(completed_segments, args.updates):
        horizon = curriculum_horizon(index, args.updates, args.rollout, curriculum)
        episode, prefix, burn, sampling = choose_segment(episodes, rng, horizon, args.burn_in,
                                                        args.start_mode, args.event_fraction)
        last_result = run_segment(model, episode, optimizer=optimizer, teacher=teacher, device=device,
                                  prefix_start=prefix, rollout=horizon, chunk=args.chunk, burn_in=burn,
                                  render_every=args.render_every, geometry_every=args.geometry_every,
                                  collision_weight=args.collision_weight, distill_weight=args.distill_weight,
                                  none_class_weight=args.none_class_weight, attack_pos_weight=args.attack_pos_weight)
        last_result["declaration"]["segment_sampling"] = sampling
        event_sampling_segments += int(sampling["mode"] == "positive_event")
        total_optimizer_steps += last_result["declaration"]["optimizer_steps"]
        declarations.append(last_result["declaration"])
        # Keep bounded checkpoint metadata for prolonged training runs.
        declarations = declarations[-100:]
        if (index + 1) % args.log_every == 0 or index == completed_segments:
            print(json.dumps({"segment_updates": index + 1, "optimizer_steps": total_optimizer_steps,
                              "event_fraction_nominal": args.event_fraction,
                              "event_fraction_realized": event_sampling_segments / (index + 1),
                              "curriculum_horizon": horizon, **last_result["metrics"],
                              "tbptt": last_result["declaration"]}), flush=True)
        if (index + 1) % args.save_every == 0:
            save_checkpoint(index + 1)
    if validation_episodes:
        episode = validation_episodes[0]
        val_result = run_segment(model, episode, teacher=None, device=device, prefix_start=0,
                                 rollout=args.rollout, chunk=args.chunk, burn_in=args.burn_in,
                                 render_every=args.render_every, geometry_every=args.geometry_every,
                                 collision_weight=args.collision_weight,
                                 none_class_weight=args.none_class_weight, attack_pos_weight=args.attack_pos_weight)
        validation = {"metrics": val_result["metrics"], "tbptt": val_result["declaration"],
                      "scope": "One held-out continuous segment; not a gameplay success certificate."}
        print(json.dumps({"validation": validation}), flush=True)
    else:
        print("Validation disabled: supply at least two episode files for a held-out trajectory.", flush=True)
    save_checkpoint(args.updates)
    print(json.dumps({"saved": str(output), "segment_updates": args.updates,
                      "optimizer_steps": total_optimizer_steps, "seconds": time.perf_counter() - started}), flush=True)


if __name__ == "__main__":
    main()
