"""Supervised short-unroll world-model training and true-pair x4 SR training.

Example: python -m turkeyworld.train --data data --kind student --steps 2000
         python -m turkeyworld.train --data data_hr --stage sr --steps 2000
Training checkpoints are not evidence of five-minute accuracy or a target FPS.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .models import (ACTION_DIM, ENTITY_DIM, ENTITY_SLOTS, EVENT_DIM, GLOBAL_DIM,
                     STATE_DIM, STATE_SCHEMA, NeuralWorldModel, SingleFrameUpscaler,
                     WorldModelConfig, load_world_checkpoint, parameter_counts)


def normalize_world_config(config):
    """Preserve the architecture/transition semantics of older checkpoints."""
    result = dict(config)
    result.setdefault("architecture", "legacy_cnn")
    result.setdefault("transition_mode", "residual_v0")
    result.setdefault("simulation_hz", 30.0)
    result.setdefault("endless", False)
    result.setdefault("navigation_only", False)
    result.setdefault('player_motion_mode', 'delta')
    for key, value in asdict(WorldModelConfig()).items():
        if key.startswith('rssm_'):
            result.setdefault(key, value)
    result.setdefault("renderer_mode", "direct")
    result.setdefault("flow_kind", "")
    result.setdefault("spatial_entities", False)
    result.setdefault("spatial_combat", False)
    result.setdefault("game_training_v5", False)
    result.setdefault("flow_patch_size", 4)
    result.setdefault("flow_block_conditioning", False)
    result.setdefault("flow_steps", 0)
    for name in ("flow_width", "flow_depth", "flow_heads", "flow_adapter_width", "flow_adapter_depth", "flow_adapter_heads"):
        result.setdefault(name, 0)
    result.setdefault("flow_mlp_ratio", 4.)
    result.setdefault("flow_adapter_mlp_ratio", 4.)
    return result


def infer_combat_targets(states):
    """Weak legacy labels from HP/cooldown deltas; exact collector IDs are better.

    A single hitscan shot damages at most one slot. Attack attribution uses an
    entity's cooldown reset when player HP decreases; old recordings cannot
    resolve arbitrary simultaneous attacks as reliably as explicit flags.
    """
    entities = np.asarray(states, dtype=np.float32)[:, GLOBAL_DIM:-2].reshape(-1, ENTITY_SLOTS, ENTITY_DIM)
    damage = entities[:-1, :, 4] - entities[1:, :, 4]
    has_hit = damage.max(-1) > 1e-5
    hit_ids = np.where(has_hit, damage.argmax(-1), -1).astype(np.int64)
    player_damaged = np.asarray(states[:-1, 4]) > np.asarray(states[1:, 4]) + 1e-5
    resets = entities[1:, :, 6] > entities[:-1, :, 6] + 1e-5
    attacks = (resets & player_damaged[:, None]).astype(np.float32)
    return hit_ids, attacks


def combat_label_arrays(data, warn=True):
    """Return target IDs and attack flags, preserving exact labels if present."""
    fields = data.files if hasattr(data, "files") else data
    missing = "hit_ids" not in fields or "attack_flags" not in fields
    if missing:
        if warn:
            warnings.warn("Legacy episode has no complete hit_ids/attack_flags: inferring weak combat labels from HP/cooldown changes. Recollect for exact labels.",
                          RuntimeWarning, stacklevel=2)
        inferred_hit, inferred_attacks = infer_combat_targets(data["states"])
    hits = np.asarray(data["hit_ids"] if "hit_ids" in fields else inferred_hit, dtype=np.int64)
    attacks = np.asarray(data["attack_flags"] if "attack_flags" in fields else inferred_attacks, dtype=np.float32)
    count = len(data["actions"])
    if hits.shape != (count,) or np.any((hits < -1) | (hits >= ENTITY_SLOTS)):
        raise ValueError("hit_ids must be [T], using -1 for no target and 0..7 for slots")
    if attacks.shape != (count, ENTITY_SLOTS) or np.any((attacks < 0) | (attacks > 1)):
        raise ValueError("attack_flags must be [T,8] in [0,1]")
    return hits, attacks, "inferred_legacy" if missing else "collector_exact"


def supervised_event_losses(result, hit_ids, attack_flags, events,
                            none_class_weight=.2, attack_pos_weight=4.):
    """Use on a GT-state auxiliary transition, not a drifted one-step pose."""
    zero = result["state"].new_zeros(())
    losses = {"events": zero, "target": zero, "attack": zero}
    if "event_logits" in result:
        losses["events"] = F.binary_cross_entropy_with_logits(result["event_logits"], events)
    if "target_logits" in result:
        labels = torch.where(hit_ids < 0, torch.full_like(hit_ids, ENTITY_SLOTS), hit_ids)
        class_weights = result["target_logits"].new_ones(ENTITY_SLOTS + 1)
        class_weights[-1] = none_class_weight
        # Avoid the weighted-mean denominator cancelling the none-class weight
        # in the batch-one long trainer.
        losses["target"] = F.cross_entropy(result["target_logits"], labels.long(),
                                            weight=class_weights, reduction="none").mean()
    if "attack_logits" in result:
        positive_weight = result["attack_logits"].new_full((ENTITY_SLOTS,), attack_pos_weight)
        losses["attack"] = F.binary_cross_entropy_with_logits(result["attack_logits"], attack_flags,
                                                              pos_weight=positive_weight)
    return losses


class EpisodeWindows(Dataset):
    """Windows never cross an episode. Splits, when requested, use whole files.

    Each NPZ is one episode. Keep at most two decoded episodes per worker;
    compressed NPZ is suitable for a small prototype, not a petabyte dataset.
    """
    def __init__(self, paths, horizon=4, require_hr=False):
        self.paths = [Path(p) for p in paths]
        self.horizon = horizon
        self.cache = OrderedDict()
        self.windows = []
        self.has_geometry = None
        for episode, path in enumerate(self.paths):
            with np.load(path, allow_pickle=False) as data:
                required = {"states", "actions", "frames", "map", "events"}
                if require_hr:
                    required.add("hr_frames")
                missing = required.difference(data.files)
                if missing:
                    raise ValueError(f"{path.name}: missing fields {sorted(missing)}")
                states, actions, frames, events = (data[k] for k in ["states", "actions", "frames", "events"])
                transitions = len(actions)
                if states.shape != (transitions + 1, STATE_DIM):
                    raise ValueError(f"{path.name}: expected states [T+1,{STATE_DIM}]")
                if actions.shape != (transitions, ACTION_DIM) or events.shape != (transitions, EVENT_DIM):
                    raise ValueError(f"{path.name}: expected actions [T,4], events [T,4]")
                if len(frames) != transitions + 1 or frames.ndim != 4 or frames.shape[-1] != 3:
                    raise ValueError(f"{path.name}: expected frames [T+1,H,W,3]")
                if data["map"].shape != (16, 16):
                    raise ValueError(f"{path.name}: expected map [16,16]")
                combat_label_arrays(data)
                if ("depth" in data.files) != ("visibility" in data.files):
                    raise ValueError(f"{path.name}: depth and visibility must be present together")
                present = "depth" in data.files
                if self.has_geometry is None:
                    self.has_geometry = present
                elif present != self.has_geometry:
                    raise ValueError("Do not mix episodes with and without geometry labels in one data directory")
                if "depth" in data.files:
                    if data["depth"].shape != (transitions + 1, 32):
                        raise ValueError(f"{path.name}: expected depth [T+1,32]")
                    if data["visibility"].shape != (transitions + 1, ENTITY_SLOTS):
                        raise ValueError(f"{path.name}: expected visibility [T+1,8]")
                if require_hr:
                    hr = data["hr_frames"]
                    if len(hr) != transitions + 1 or hr.ndim != 4 or hr.shape[-1] != 3:
                        raise ValueError(f"{path.name}: hr_frames must align with frames")
                    if tuple(hr.shape[1:3]) != (frames.shape[1] * 4, frames.shape[2] * 4):
                        raise ValueError(f"{path.name}: HR must be exactly four times LR in each dimension")
                for start in range(0, transitions - horizon + 1, horizon):
                    self.windows.append((episode, start))
        if not self.windows:
            raise ValueError(f"No episode contains a complete horizon of {horizon} transitions")
        self.require_hr = require_hr

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode, start = self.windows[index]
        if episode not in self.cache:
            with np.load(self.paths[episode], allow_pickle=False) as data:
                fields = ["states", "actions", "frames", "map", "events"]
                if self.require_hr:
                    fields.append("hr_frames")
                if "depth" in data.files:
                    fields.extend(["depth", "visibility"])
                self.cache[episode] = {key: data[key].copy() for key in fields}
                hits, attacks, _ = combat_label_arrays(data, warn=False)
                self.cache[episode]["hit_ids"] = hits
                self.cache[episode]["attack_flags"] = attacks
            while len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(episode)
        data = self.cache[episode]
        end = start + self.horizon
        result = {
            "states": torch.from_numpy(data["states"][start:end + 1].astype(np.float32)),
            "actions": torch.from_numpy(data["actions"][start:end].astype(np.float32)),
            "events": torch.from_numpy(data["events"][start:end].astype(np.float32)),
            "frames": torch.from_numpy(data["frames"][start:end + 1].copy()),
            "map": torch.from_numpy(data["map"].astype(np.int64)),
            "hit_ids": torch.from_numpy(data["hit_ids"][start:end].copy()),
            "attack_flags": torch.from_numpy(data["attack_flags"][start:end].copy()),
        }
        if self.require_hr:
            result["hr_frames"] = torch.from_numpy(data["hr_frames"][start:end + 1].copy())
        if "depth" in data:
            result["depth"] = torch.from_numpy(data["depth"][start:end + 1].astype(np.float32))
            result["visibility"] = torch.from_numpy(data["visibility"][start:end + 1].astype(np.float32))
        return result


def prepare_images(frames, size, device):
    """Resizing real HR targets is permitted; inventing HR from LR is not."""
    b, t, h, w, c = frames.shape
    images = frames.to(device, non_blocking=True).permute(0, 1, 4, 2, 3).float().div_(255)
    images = images.reshape(b * t, c, h, w)
    if (h, w) != (size, size):
        images = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    return images.reshape(b, t, c, size, size)


def image_loss(prediction, target):
    base = F.l1_loss(prediction, target)
    dx = F.l1_loss(prediction[..., :, 1:] - prediction[..., :, :-1],
                   target[..., :, 1:] - target[..., :, :-1])
    dy = F.l1_loss(prediction[..., 1:, :] - prediction[..., :-1, :],
                   target[..., 1:, :] - target[..., :-1, :])
    return base + 0.1 * (dx + dy)


def state_loss(prediction, target):
    weights = torch.ones(STATE_DIM, device=prediction.device, dtype=prediction.dtype)
    weights[:4] = 4  # pose and heading affect every rendered surface
    weights[4:6] = 2
    per_entity = torch.ones(ENTITY_DIM, device=prediction.device, dtype=prediction.dtype)
    per_entity[:2] = 3
    per_entity[4:6] = 4  # HP and persistent life state
    weights[GLOBAL_DIM:-2] = per_entity.repeat(ENTITY_SLOTS)
    return ((prediction - target).square() * weights).mean()


def world_loss(model, batch, device, teacher=None, distill_weight=0.1,
               depth_weight=0.2, visibility_weight=0.2,
               none_class_weight=.2, attack_pos_weight=4., renderer_only=False):
    states = batch["states"].to(device)
    actions = batch["actions"].to(device)
    events = batch["events"].to(device)
    maps = batch["map"].to(device)
    frames = prepare_images(batch["frames"], model.config.image_size, device)
    map_features = model.encode_map(maps)
    if renderer_only:
        if getattr(model.config, "transition_mode", "residual_v0") != "event_v1":
            raise ValueError("renderer-only training requires event_v1 state-only rendering")
        loss = states.new_zeros(())
        for time_index in range(actions.shape[1]):
            # Event-v1 RGB is a function of the explicit state and map. Use the
            # actual labeled pose, keeping dynamics and shared geometry fixed.
            rgb = model.render(states[:, time_index + 1], states.new_zeros(states.shape[0], 128), map_features)
            loss = loss + image_loss(rgb, frames[:, time_index + 1])
        loss = loss / actions.shape[1]
        return loss, {"image": loss}
    teacher_map = None
    if teacher is not None:
        with torch.no_grad():
            teacher_map = teacher.encode_map(maps)
    memory = teacher_memory = auxiliary_memory = None
    current_state = states[:, 0]
    totals = {key: torch.zeros((), device=device) for key in
              ["state", "events", "target", "attack", "image", "distill", "depth", "visibility"]}
    horizon = actions.shape[1]
    # This is genuine short free unrolling: predictions and recurrent memory
    # feed the next step, with gradients across this complete short window.
    for time_index in range(horizon):
        result = model.transition(current_state, actions[:, time_index], map_features, memory)
        rgb = model.render(result["state"], result["features"], map_features)
        totals["state"] = totals["state"] + state_loss(result["state"], states[:, time_index + 1])
        # Categorical event labels describe the recorded state. A separate
        # recurrent GT-state stream calibrates heads without stale pose labels.
        auxiliary_result = model.transition(states[:, time_index], actions[:, time_index],
                                            map_features, auxiliary_memory)
        auxiliary_memory = auxiliary_result["memory"]
        auxiliary_losses = supervised_event_losses(auxiliary_result,
                            batch["hit_ids"][:, time_index].to(device),
                            batch["attack_flags"][:, time_index].to(device), events[:, time_index],
                            none_class_weight, attack_pos_weight)
        for name, value in auxiliary_losses.items():
            totals[name] = totals[name] + value
        totals["image"] = totals["image"] + image_loss(rgb, frames[:, time_index + 1])
        if (getattr(model.config, "architecture", "legacy_cnn") == "camera_transformer"
                and "depth" in batch):
            # Calibrate geometric heads at the actual labeled pose. A visibility
            # label from a recorded pose must not become a one-step oracle label
            # at a different, drifted student pose. RGB/state losses above are
            # whole-trajectory objectives from their common initial state.
            auxiliary = model.geometry(states[:, time_index + 1], map_features)
            depth_target = batch["depth"][:, time_index + 1].to(device)
            visibility_target = batch["visibility"][:, time_index + 1].to(device)
            totals["depth"] = totals["depth"] + F.smooth_l1_loss(
                torch.log1p(auxiliary["depth"].clamp_min(0)),
                torch.log1p(depth_target.clamp_min(0)))
            totals["visibility"] = totals["visibility"] + F.binary_cross_entropy_with_logits(
                auxiliary["visibility_logits"], visibility_target)
        if teacher is not None:
            # Teacher sees the SAME student-visited explicit state, rather than
            # an unrelated teacher trajectory. Real-data supervision remains.
            with torch.no_grad():
                teach = teacher.transition(current_state.detach(), actions[:, time_index], teacher_map, teacher_memory)
                teacher_rgb = teacher.render(teach["state"], teach["features"], teacher_map)
                teacher_memory = teach["memory"]
                if teacher_rgb.shape[-2:] != rgb.shape[-2:]:
                    teacher_rgb = F.interpolate(teacher_rgb, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
            kd = state_loss(result["state"], teach["state"])
            kd = kd + 0.1 * F.mse_loss(result["features"], teach["features"])
            kd = kd + image_loss(rgb, teacher_rgb)
            totals["distill"] = totals["distill"] + kd
        current_state, memory = result["state"], result["memory"]
    totals = {key: value / horizon for key, value in totals.items()}
    total = (10 * totals["state"] + totals["events"] + totals["target"] + totals["attack"] + totals["image"]
             + distill_weight * totals["distill"] + depth_weight * totals["depth"]
             + visibility_weight * totals["visibility"])
    return total, totals


def sr_loss(model, batch, size, device):
    low = prepare_images(batch["frames"], size, device)[:, 1:]
    high = prepare_images(batch["hr_frames"], size * 4, device)[:, 1:]
    low = low.flatten(0, 1)
    high = high.flatten(0, 1)
    loss = image_loss(model(low), high)
    return loss, {"true_hr_image": loss}


def discover_files(path):
    path = Path(path)
    paths = [path] if path.is_file() else sorted(path.rglob("*.npz"))
    if not paths:
        raise ValueError(f"No NPZ episodes found in {path}")
    return paths


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="One episode NPZ or a directory of episodes")
    parser.add_argument("--stage", choices=["world", "sr"], default="world")
    parser.add_argument("--kind", choices=["teacher", "student"], default="student")
    parser.add_argument("--architecture", choices=["camera_transformer", "legacy_cnn"],
                        default="camera_transformer")
    parser.add_argument("--transition-mode", choices=["event_v1", "residual_v0"], default="event_v1")
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--image-size", type=int, choices=[128, 256], default=256)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--teacher", help="Trained world checkpoint, used only for student distillation")
    parser.add_argument("--distill-weight", type=float, default=0.1)
    parser.add_argument("--depth-weight", type=float, default=0.2)
    parser.add_argument("--visibility-weight", type=float, default=0.2)
    parser.add_argument("--none-class-weight", type=float, default=.2)
    parser.add_argument("--attack-pos-weight", type=float, default=4.)
    parser.add_argument("--resume", help="Resume a checkpoint of the same stage and configuration")
    parser.add_argument("--renderer-only", action="store_true",
                        help="With event_v1 --resume: fit RGB at GT states, freezing dynamics/map/geometry")
    parser.add_argument("--output", default="checkpoints/student.pt")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Hold out whole episode files; requires at least two files")
    parser.add_argument("--log-every", type=int, default=10)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if min(args.steps, args.batch, args.horizon, args.threads) < 1:
        raise ValueError("steps, batch, horizon and threads must be positive")
    if not 0 <= args.val_fraction < 1:
        raise ValueError("val-fraction must be in [0,1)")
    if args.teacher and (args.stage != "world" or args.kind != "student"):
        raise ValueError("--teacher is supported for student world training")
    if args.renderer_only and (args.stage != "world" or args.transition_mode != "event_v1" or not args.resume or args.teacher):
        raise ValueError("--renderer-only requires --stage world --transition-mode event_v1 --resume and no --teacher")
    if min(args.none_class_weight, args.attack_pos_weight, args.hz) <= 0:
        raise ValueError("class weights and hz must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    paths = discover_files(args.data)
    geometry_flags = []
    for path in paths:
        with np.load(path, allow_pickle=False) as episode:
            geometry_flags.append("depth" in episode.files or "visibility" in episode.files)
    if any(geometry_flags) and not all(geometry_flags):
        raise ValueError("Do not mix episodes with and without geometry labels in one data directory")
    shuffled = paths.copy()
    random.Random(args.seed).shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * args.val_fraction)) if len(shuffled) > 1 and args.val_fraction > 0 else 0
    val_paths, train_paths = shuffled[:n_val], shuffled[n_val:]
    require_hr = args.stage == "sr"
    train_set = EpisodeWindows(train_paths, args.horizon, require_hr=require_hr)
    if args.stage == "world" and args.architecture == "camera_transformer" and not train_set.has_geometry:
        print("Geometry supervision is disabled for these older episodes. Recollect data to train wall-depth and visibility heads.", flush=True)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=False)
    val_loader = None
    if val_paths:
        val_loader = DataLoader(EpisodeWindows(val_paths, args.horizon, require_hr=require_hr), batch_size=args.batch)
    else:
        print("Validation disabled: add a separate episode to hold out an entire trajectory.", flush=True)
    if args.stage == "world":
        config = WorldModelConfig(kind=args.kind, image_size=args.image_size,
                                  architecture=args.architecture, transition_mode=args.transition_mode,
                                  simulation_hz=args.hz)
        model = NeuralWorldModel(config).to(device)
        config_dict = asdict(config)
    else:
        model = SingleFrameUpscaler().to(device)
        config_dict = model.config
    teacher = None
    if args.teacher:
        teacher, teacher_ckpt = load_world_checkpoint(args.teacher, device)
        teacher.requires_grad_(False)
        if teacher_ckpt.get("training_steps", 0) < 1:
            raise ValueError("teacher checkpoint contains no supervised training steps")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        resumed_config = dict(checkpoint.get("config", {}))
        if args.stage == "world":
            resumed_config = normalize_world_config(resumed_config)
        if checkpoint.get("stage") != args.stage or resumed_config != config_dict:
            raise ValueError("resume stage/configuration mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["training_steps"]
    if args.renderer_only:
        render_names = {"render_condition", "screen_position", "render_blocks", "render_norm",
                        "rgb128", "rgb256", "render_input"}
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(any(part in render_names for part in name.split(".")))
        if not any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("No renderer parameters matched this architecture")
    print(json.dumps({"stage": args.stage, "config": config_dict,
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "train_episodes": [p.name for p in train_paths],
                      "validation_episodes": [p.name for p in val_paths],
                      "horizon": args.horizon, "device": str(device),
                      "training_phase": "renderer_only_gt_state" if args.renderer_only else args.stage,
                      "event_loss_weights": {"none_class_weight": args.none_class_weight,
                                             "attack_pos_weight": args.attack_pos_weight},
                      "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}, ensure_ascii=False), flush=True)
    iterator = iter(train_loader)
    started = time.perf_counter()
    last_metrics = {}
    model.train()
    for local_step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        if args.stage == "world":
            loss, metrics = world_loss(model, batch, device, teacher, args.distill_weight,
                                       args.depth_weight, args.visibility_weight,
                                       args.none_class_weight, args.attack_pos_weight, args.renderer_only)
        else:
            loss, metrics = sr_loss(model, batch, args.image_size, device)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite loss; checkpoint not overwritten")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_metrics = {key: float(value.detach()) for key, value in metrics.items()}
        last_metrics["loss"] = float(loss.detach())
        if local_step == 1 or local_step % args.log_every == 0 or local_step == args.steps:
            print(json.dumps({"step": start_step + local_step, **last_metrics}), flush=True)
    validation = None
    if val_loader is not None:
        model.eval()
        values = []
        with torch.inference_mode():
            for batch in val_loader:
                loss, _ = (world_loss(model, batch, device,
                                      depth_weight=args.depth_weight,
                                      visibility_weight=args.visibility_weight,
                                      none_class_weight=args.none_class_weight,
                                      attack_pos_weight=args.attack_pos_weight,
                                      renderer_only=args.renderer_only) if args.stage == "world"
                           else sr_loss(model, batch, args.image_size, device))
                values.append(float(loss))
        validation = sum(values) / len(values)
        print(json.dumps({"held_out_short_unroll_loss": validation,
                          "note": "This does not evaluate five-minute free gameplay."}), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1, "stage": args.stage, "config": config_dict,
        "training_phase": "renderer_only_gt_state" if args.renderer_only else args.stage,
        "event_loss_weights": {"none_class_weight": args.none_class_weight,
                               "attack_pos_weight": args.attack_pos_weight},
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "training_steps": start_step + args.steps, "state_schema": STATE_SCHEMA,
        "training_files": [str(p.resolve()) for p in train_paths],
        "validation_files": [str(p.resolve()) for p in val_paths],
        "last_metrics": last_metrics, "validation_loss": validation,
        "quality_status": "Unverified: training steps alone do not establish gameplay accuracy.",
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(output)
    print(json.dumps({"saved": str(output), "training_seconds": time.perf_counter() - started,
                      "upscaler_loss_enabled": args.stage == "sr"}), flush=True)


if __name__ == "__main__":
    main()
