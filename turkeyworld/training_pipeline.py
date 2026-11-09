"""Persistent teacher Flow Matching -> student Rectified Flow experiment.

Writes real progress, atomic resumable checkpoints and independent 300-second
evaluations. Training completion and five-minute gameplay qualification are
different outcomes. This runner never declares success from step counts alone.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import json
import hashlib
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from .models import NeuralWorldModel, WorldModelConfig, STATE_SCHEMA, load_world_checkpoint
from .train import image_loss, state_loss, supervised_event_losses, normalize_world_config
from .long_train import collision_barrier, map_signed_distance


def now():
    return datetime.now(timezone.utc).isoformat()


def model_fingerprint(model):
    configuration = asdict(model.config)
    # New optional fields must not invalidate previously published125M
    # checkpoint fingerprints when their original default behavior is used.
    for key, default in (('vjepa_readout_mlp_ratio', 4), ('vjepa_gradient_checkpointing', False),
                         ('flow_mlp_hidden', 0), ('renderer_codec', 'pixel'), ('renderer_codec_config', None)):
        if configuration.get(key) == default:
            configuration.pop(key, None)
    digest = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode())
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def replace_file(temporary, destination):
    """Windows readers may briefly keep a report or checkpoint open."""
    for attempt in range(40):
        try:
            os.replace(temporary, destination)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.05)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    replace_file(temporary, path)


class MemoryData:
    """Decode once, avoiding repeated compressed-episode reads at every update."""
    def __init__(self, directory, device, limit=0):
        paths = sorted(Path(directory).glob("*.npz"))
        if limit:
            paths = paths[:limit]
        if not paths:
            raise ValueError(f"No training episodes in {directory}")
        fields = {key: [] for key in ("old", "new", "actions", "events", "hits", "attacks", "images", "depth", "visibility")}
        self.episode_ranges = []
        offset = 0
        for path in paths:
            with np.load(path, allow_pickle=False) as episode:
                count = len(episode["actions"])
                self.episode_ranges.append({"path": str(path.resolve()), "start": offset, "stop": offset + count})
                offset += count
                fields["old"].append(episode["states"][:-1])
                fields["new"].append(episode["states"][1:])
                for key, source in [("actions", "actions"), ("events", "events"), ("hits", "hit_ids"), ("attacks", "attack_flags")]:
                    fields[key].append(episode[source])
                for key, source in [("images", "frames"), ("depth", "depth"), ("visibility", "visibility")]:
                    fields[key].append(episode[source][1:])
                grid = episode["map"].copy()
        self.data = {key: np.concatenate(values) for key, values in fields.items()}
        self.grid = torch.as_tensor(grid, dtype=torch.long, device=device)[None]
        self.device, self.count, self.files = device, len(self.data["old"]), len(paths)
        self.positive = np.flatnonzero((self.data["hits"] >= 0) | (self.data["attacks"] > .5).any(-1))
        old_alive = self.data["old"][:, 8:88].reshape(-1, 8, 10)[:, :, 5] > .5
        new_alive = self.data["new"][:, 8:88].reshape(-1, 8, 10)[:, :, 5] > .5
        self.respawns = np.flatnonzero((~old_alive & new_alive).any(-1))
        self.sdf = torch.as_tensor(map_signed_distance(grid), device=device)[None, None]

    def batch(self, count, rng, indices=None, images=True):
        if indices is None:
            indices = rng.integers(self.count, size=count)
            if len(self.positive):
                indices[:count // 3] = rng.choice(self.positive, size=count // 3)
            if len(self.respawns):
                indices[-max(1, count // 8):] = rng.choice(self.respawns, size=max(1, count // 8))
        result = {}
        for key, values in self.data.items():
            if key == "images" and not images:
                continue
            tensor = torch.from_numpy(values[indices]).to(self.device)
            result[key] = (tensor.permute(0, 3, 1, 2).float() / 255 if key == "images"
                           else tensor.long() if key == "hits" else tensor.float())
        result["map"] = self.grid.expand(len(indices), -1, -1)
        return result


def transition_losses(model, batch, map_features, teacher=None, teacher_features=None):
    old, target = batch["old"], batch["new"]
    result = model.transition(old, batch["actions"], map_features)
    losses = supervised_event_losses(result, batch["hits"], batch["attacks"], batch["events"])
    losses["state"] = state_loss(result["state"], target)
    dt = 1 / model.config.simulation_hz
    losses["player_motion"] = F.smooth_l1_loss((result["state"][:, :2] - old[:, :2]) * (16 / (2.4 * dt)),
                                               (target[:, :2] - old[:, :2]) * (16 / (2.4 * dt)))
    previous_entities = old[:, 8:88].reshape(-1, 8, 10)
    target_entities = target[:, 8:88].reshape(-1, 8, 10)
    predicted_entities = result["state"][:, 8:88].reshape(-1, 8, 10)
    continuing = (previous_entities[:, :, 5] > .5) & (target_entities[:, :, 5] > .5)
    movement = F.smooth_l1_loss((predicted_entities[:, :, :2] - previous_entities[:, :, :2]) * (16 / (1.2 * dt)),
                                (target_entities[:, :, :2] - previous_entities[:, :, :2]) * (16 / (1.2 * dt)), reduction="none")
    losses["npc_motion"] = (movement * continuing[:, :, None]).sum() / (continuing.sum() * 2).clamp_min(1)
    if "spawn_logits" in result:
        revived = (previous_entities[:, :, 5] < .5) & (target_entities[:, :, 5] > .5)
        labels = (target_entities[:, :, 0] * 16).long().clamp(0, 15) + 16 * (target_entities[:, :, 1] * 16).long().clamp(0, 15)
        values = F.cross_entropy(result["spawn_logits"].reshape(-1, 256), labels.reshape(-1), reduction="none").reshape(-1, 8)
        losses["spawn"] = (values * revived).sum() / revived.sum().clamp_min(1)
    geometry = model.geometry(target, map_features)
    losses["depth"] = F.smooth_l1_loss(torch.log1p(geometry["depth"].clamp_min(0)), torch.log1p(batch["depth"].clamp_min(0)))
    losses["visibility"] = F.binary_cross_entropy_with_logits(geometry["visibility_logits"], batch["visibility"])
    if teacher is not None:
        with torch.no_grad():
            teach = teacher.transition(old, batch["actions"], teacher_features)
        losses["distill"] = F.kl_div(F.log_softmax(result["target_logits"], -1), F.softmax(teach["target_logits"], -1), reduction="batchmean")
        losses["distill"] = losses["distill"] + F.binary_cross_entropy_with_logits(result["attack_logits"], teach["attack_logits"].sigmoid())
    return result, losses


class Experiment:
    def __init__(self, args):
        self.args = args
        self.root = args.run_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.device = torch.device(args.device)
        self.rng = np.random.default_rng(args.seed)
        self.plan = [("teacher", args.teacher_steps), ("teacher_temporal", args.temporal_updates),
                     ("reflow", args.reflow_pairs), ("student", args.student_steps),
                     ("student_temporal", args.temporal_updates), ("evaluation", 9000)]
        self.measured = {}
        self._adversarial_trainer = None
        self._adversarial_phase = None
        self.state = {"run_id": self.root.name, "status": "running", "started_at": now(),
                      "evaluation": {"status": "pending", "target_seconds": 300, "passed": False},
                      "message": "Подготовка памяти для обучения", "model": "Teacher CFM / Student Rectified Flow",
                      "estimated_total_range_seconds": [3600, 7200]}
        if getattr(args, "teacher_until_quality", False):
            self.state.update(teacher_until_quality=True, estimated_total_range_seconds=None,
                              eta_seconds=None, eta_scope="До достижения визуального порога; срок пока неизвестен")
        self.update("initializing", 0, 1, "Загрузка данных")

    def update(self, stage, current, total, label, **fields):
        self.state.update(stage=stage, stage_label=label, current=current, total=total,
                          progress=None if total is None else current / max(1, total), unit="шагов", updated_at=now(),
                          elapsed_seconds=round(time.time() - self.started, 1), **fields)
        atomic_json(self.root / "status.json", self.state)

    def metric(self, value):
        value = {"timestamp": now(), **value}
        with (self.root / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def save(self, model, optimizer, phase, step, metrics, extra=None):
        output = self.root / "checkpoints" / f"{phase}.pt"
        output.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {"format_version": 1, "stage": "world", "config": asdict(model.config),
                      "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "training_steps": step, "pipeline_phase": phase, "state_schema": STATE_SCHEMA,
                      "last_metrics": metrics, "quality_status": "Unverified until independent 300-second evaluation passes.",
                      "created_at": now(), "rng_numpy": self.rng.bit_generator.state,
                      "rng_torch": torch.random.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else []}
        if self.state.get("teacher_fingerprint"):
            checkpoint["teacher_fingerprint"] = self.state["teacher_fingerprint"]
        if self._adversarial_trainer is not None and self._adversarial_phase == phase:
            checkpoint["adversarial_training"] = self._adversarial_trainer.state_dict()
        if extra:
            checkpoint.update(extra)
        temporary = output.with_suffix(".pt.tmp")
        torch.save(checkpoint, temporary)
        replace_file(temporary, output)
        self.state["checkpoint"] = {"name": str(output), "step": step, "created_at": checkpoint["created_at"]}
        return output

    def preview(self, model, batch, phase, step):
        model.eval()
        strict_preview = getattr(self.args, "teacher_until_quality", False) and model.config.kind == "teacher"
        if strict_preview:
            from .visual_quality import strict_render_precision
        context = strict_render_precision() if strict_preview else self.amp()
        with torch.inference_mode(), context:
            features = model.encode_map(batch["map"][:1])
            state = batch["new"][:1]
            base = model.render_base(state, state.new_zeros(1, 128), features)
            predicted = model.flow_renderer.sample(base, state)
        truth = batch["images"][0].detach().float().cpu().permute(1, 2, 0).numpy()
        values = [truth, base[0].detach().float().cpu().permute(1, 2, 0).numpy(), predicted[0].detach().float().cpu().permute(1, 2, 0).numpy()]
        canvas = Image.new("RGB", (384, 150), (14, 22, 31))
        draw = ImageDraw.Draw(canvas)
        for index, (array, label) in enumerate(zip(values, ("Source", "Condition", f"{phase} {step}"))):
            canvas.paste(Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8)).resize((128, 128)), (index * 128, 22))
            draw.text((index * 128 + 3, 5), label, fill="white")
        temporary = self.root / "preview.tmp.png"
        canvas.save(temporary)
        replace_file(temporary, self.root / "preview.png")
        model.train()

    def amp(self, cache_enabled=True):
        return torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=cache_enabled) if self.device.type == "cuda" else nullcontext()

    def optimizer_state_for_restore(self, checkpoint, model, optimizer):
        return checkpoint['optimizer']

    def restore(self, model, optimizer, phase):
        path = self.root / "checkpoints" / f"{phase}.pt"
        if not path.exists():
            return 0
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if phase.startswith("student") and checkpoint.get("teacher_fingerprint") != self.state.get("teacher_fingerprint"):
            return 0  # A changed teacher requires a new student, not stale distillation.
        if normalize_world_config(checkpoint["config"]) != asdict(model.config):
            raise ValueError(f"Configuration mismatch in {path}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(self.optimizer_state_for_restore(checkpoint, model, optimizer))
        self.rng.bit_generator.state = checkpoint["rng_numpy"]
        torch.random.set_rng_state(checkpoint["rng_torch"].cpu())
        if self.device.type == "cuda" and checkpoint.get("rng_cuda"):
            torch.cuda.set_rng_state_all([value.cpu() for value in checkpoint["rng_cuda"]])
        return checkpoint["training_steps"]

    def _publish_model_sizes(self, model):
        sizes = self.state.setdefault("model_parameters", {})
        kind = model.config.kind
        sizes[kind + "_total"] = sum(p.numel() for p in model.parameters())
        sizes[kind + "_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if getattr(self.args, "teacher_size", "current") == "75m":
            sizes["target_teacher_total"] = 75000000
        if self._adversarial_trainer is not None:
            sizes["discriminator_total"] = self._adversarial_trainer.parameter_count

    def _setup_adversarial(self, phase):
        if not getattr(self.args, "adversarial", False):
            return
        if self._adversarial_trainer is not None and self._adversarial_phase == phase:
            return
        from .adversarial_training import AdversarialTrainer
        path = self.root / "checkpoints" / f"{phase}.pt"
        saved = None
        if path.exists():
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            same_teacher = (not phase.startswith("student") or
                            checkpoint.get("teacher_fingerprint") == self.state.get("teacher_fingerprint"))
            if same_teacher:
                saved = checkpoint.get("adversarial_training")
        rng_devices = [self.device.index or torch.cuda.current_device()] if self.device.type == "cuda" else []
        # Reconstructing a saved critic must not consume the restored flow RNG.
        with torch.random.fork_rng(devices=rng_devices) if saved else nullcontext():
            trainer = AdversarialTrainer(self.device,
                weight=getattr(self.args, "adversarial_weight", .02),
                feature_weight=getattr(self.args, "adversarial_feature_weight", .05),
                warmup_steps=getattr(self.args, "adversarial_warmup", 64),
                ramp_steps=getattr(self.args, "adversarial_ramp", 256),
                r1_gamma=getattr(self.args, "adversarial_r1_gamma", 1.),
                r1_interval=getattr(self.args, "adversarial_r1_interval", 16),
                lr=getattr(self.args, "adversarial_lr", 1e-4))
            if saved:
                trainer.load_state_dict(saved)
        self._adversarial_trainer, self._adversarial_phase = trainer, phase
        self.state["adversarial"] = {"enabled": True, "weight": 0., "step": trainer.updates,
                                     "warmup_steps": trainer.config["warmup_steps"],
                                     "ramp_steps": trainer.config["ramp_steps"]}

    def _adversarial_loss(self, predicted, real, condition, state, **accumulation):
        if self._adversarial_trainer is None:
            return predicted.new_zeros(()), {}
        loss, metrics = self._adversarial_trainer.generator_loss(predicted, real, condition, state, **accumulation)
        self.state["adversarial"] = {**metrics, "active": bool(metrics.get("enabled")), "enabled": True,
            "discriminator_training": True, "generator_adversarial_active": bool(metrics.get("enabled")),
            "max_weight": self._adversarial_trainer.config['weight'],
            "max_feature_weight": self._adversarial_trainer.config['feature_weight']}
        flattened = {"adversarial_" + key: value for key, value in metrics.items()
                     if isinstance(value, (int, float))}
        return loss, flattened

    def quality_thresholds(self):
        return {"psnr": getattr(self.args, "quality_psnr", 35.),
                "ssim": getattr(self.args, "quality_ssim", .98),
                "mae": getattr(self.args, "quality_mae", 2 / 255),
                "required_fraction": getattr(self.args, "quality_fraction", .95)}

    def _teacher_quality_step(self, model, optimizer, data):
        """Keep FM supervision and optimize the actual complete Heun sample too."""
        batch = data.batch(self.args.batch, self.rng,
                           indices=self.rng.integers(data.count, size=self.args.batch))
        optimizer.zero_grad(set_to_none=True)
        with self.amp():
            features = model.encode_map(batch["map"])
            state = batch["new"]
            base = model.render_base(state, state.new_zeros(len(state), 128), features)
            flow, flow_metrics = model.flow_renderer.loss(batch["images"], base, state)
            base_loss = image_loss(base, batch["images"])
            count = min(len(state), getattr(self.args, "quality_endpoint_batch", 1))
            # Small differentiable sampling batch keeps all 24 Heun velocity
            # evaluations within VRAM. Source is a loss target, never an input.
            predicted = model.flow_renderer.sample(base[:count], state[:count])
            endpoint = image_loss(predicted, batch["images"][:count])
            mse = F.mse_loss(predicted.float(), batch["images"][:count])
            loss = flow + 2 * base_loss + 4 * endpoint + 2 * mse
        adversarial, adversarial_metrics = self._adversarial_loss(
            predicted, batch["images"][:count], base[:count], state[:count])
        loss = loss + adversarial
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite teacher refinement loss; last checkpoint preserved")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step()
        return {"train_loss": float(loss.detach()), "flow": float(flow.detach()),
                "base_image": float(base_loss.detach()), "sample_image": float(endpoint.detach()),
                "sample_mse": float(mse.detach()), **{k: float(v) for k, v in flow_metrics.items()},
                **adversarial_metrics}

    def fit_teacher_until_quality(self, model, data, validation, phase="teacher_quality", reset_streak=False):
        """No step/time cap: only a checkpoint-backed held-out visual gate exits."""
        from .visual_quality import select_validation_indices, evaluate_visual_quality
        renderer_parts = {"render_condition", "screen_position", "render_blocks", "render_norm",
                          "rgb128", "rgb256", "render_input"}
        trainability = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("flow_renderer.") or
                                     any(part in renderer_parts for part in name.split(".")))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=self.args.lr / 3)
        checkpoint_path = self.root / "checkpoints" / f"{phase}.pt"
        completed = self.restore(model, optimizer, phase)
        self._setup_adversarial(phase)
        self._publish_model_sizes(model)
        saved_gate = {}
        if checkpoint_path.exists():
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            saved_gate = saved.get("teacher_quality_gate", {})
        if reset_streak:
            saved_gate = {}
        indices = select_validation_indices(validation, getattr(self.args, "quality_frames", 256), self.args.seed)
        thresholds = self.quality_thresholds()
        required = getattr(self.args, "quality_passes", 3)
        every = getattr(self.args, "quality_every", 500)
        best_score = float(saved_gate.get("best_score", 1e9))
        stale_checks = int(saved_gate.get("stale_checks", 0))
        streak, fingerprint = 0, None
        step, metrics = completed, {}
        last_gate = {}
        started = time.perf_counter()
        try:
            while True:
                if (self.root / "stop_training").exists():
                    self.save(model, optimizer, phase, step, metrics, {"teacher_quality_gate": last_gate})
                    self.state["status"] = "stopped"
                    self.update(phase, step, None, "Учитель остановлен пользователем", eta_seconds=None)
                    raise SystemExit(0)
                if step == completed or step % every == 0:
                    result = evaluate_visual_quality(model, validation, indices,
                                                     batch_size=min(8, self.args.batch), device=self.device,
                                                     amp=False, thresholds=thresholds)
                    current_fingerprint = result["panel"]["fingerprint"]
                    if fingerprint is None:
                        fingerprint = current_fingerprint
                        # A stored streak is useful only for this exact panel,
                        # policy and checkpoint; repeating a resume is no new pass.
                        if (saved_gate.get("fingerprint") == fingerprint
                                and saved_gate.get("last_checked_step", -1) <= completed
                                and saved_gate.get("phase") == phase):
                            streak = int(saved_gate.get("consecutive_passes", 0))
                            if saved_gate.get("last_checked_step") == completed:
                                streak = max(0, streak - int(bool(saved_gate.get("passed"))))
                    elif fingerprint != current_fingerprint:
                        fingerprint, streak = current_fingerprint, 0
                    streak = streak + 1 if result["passed"] else 0
                    summary = result["summary"]
                    score = summary["mae_mean"] + .5 * (1 - summary["ssim_mean"])
                    improved = score < best_score * .995
                    best_score, stale_checks = (score, 0) if improved else (best_score, stale_checks + 1)
                    if stale_checks >= 6:
                        for group in optimizer.param_groups:
                            group["lr"] = max(1e-6, group["lr"] * .5)
                        stale_checks = 0
                    last_gate = {"phase": phase, "fingerprint": fingerprint, "last_checked_step": step,
                                 "consecutive_passes": streak, "required_passes": required,
                                 "passed": bool(result["passed"]), "best_score": best_score,
                                 "stale_checks": stale_checks, "thresholds": thresholds}
                    quality = {**summary, "status": "passed" if streak >= required else "training",
                               "matching_fraction": summary["pass_fraction"], "required_fraction": thresholds["required_fraction"],
                               "consecutive_passes": streak, "required_passes": required,
                               "target_psnr": thresholds["psnr"], "target_ssim": thresholds["ssim"],
                               "target_mae": thresholds["mae"], "frame_count": summary["frames"],
                               "last_checked_step": step, "fingerprint": fingerprint}
                    self.state["teacher_quality"] = quality
                    self.save(model, optimizer, phase, step, metrics, {"teacher_quality_gate": last_gate})
                    atomic_json(self.root / f"{phase}_report.json", {**result, "gate": last_gate})
                    self.metric({"stage": phase, "step": step, "teacher_quality": quality})
                    self.preview(model, validation.batch(1, self.rng, indices=indices[:1]), phase, step)
                    self.update(phase, step, None, "Учитель: визуальная проверка",
                                eta_seconds=None, estimated_total_range_seconds=None,
                                eta_scope="До достижения порога качества; срок неизвестен")
                    if streak >= required:
                        return quality
                self.update(phase, step, None, "Учитель: до почти идентичных кадров",
                            eta_seconds=None, estimated_total_range_seconds=None,
                            eta_scope="До достижения PSNR/SSIM/MAE на отложенных кадрах; срок неизвестен",
                            message="Студент ждёт: учитель продолжает обучение до трёх успешных визуальных проверок подряд")
                model.train()
                metrics = self._teacher_quality_step(model, optimizer, data)
                step += 1
                if step % self.args.log_every == 0 or step == completed + 1:
                    seconds_per_step = (time.perf_counter() - started) / (step - completed)
                    self.metric({"stage": phase, "step": step, **metrics, "seconds_per_step": seconds_per_step,
                                 "learning_rate": optimizer.param_groups[0]["lr"]})
                    self.state["loss"] = metrics["train_loss"]
                    self.state["throughput"] = {"value": round(1 / seconds_per_step, 2), "unit": "шагов/с"}
                if step % self.args.save_every == 0:
                    self.save(model, optimizer, phase, step, metrics, {"teacher_quality_gate": last_gate})
        finally:
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(trainability[name])

    def joint(self, model, phase, steps, data, validation, teacher=None, pairs=None):
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr)
        completed = self.restore(model, optimizer, phase)
        self._setup_adversarial(phase)
        self._publish_model_sizes(model)
        if completed >= steps:
            return
        self.update(phase, completed, steps, "Учитель: Flow Matching" if phase == "teacher" else "Студент: Rectified Flow",
                    message="Обучение динамики, боя, геометрии и векторного поля генерации кадров")
        model.train()
        if teacher is not None:
            teacher.eval()
        started = time.perf_counter()
        metrics = {}
        for step in range(completed + 1, steps + 1):
            if (self.root / "stop_training").exists():
                self.save(model, optimizer, phase, step - 1, metrics)
                self.state["status"] = "stopped"
                self.update(phase, step - 1, steps, "Остановлено пользователем")
                raise SystemExit(0)
            batch = data.batch(self.args.batch, self.rng)
            optimizer.zero_grad(set_to_none=True)
            with self.amp():
                features = model.encode_map(batch["map"])
                with torch.no_grad():
                    teacher_features = None if teacher is None else teacher.encode_map(batch["map"])
                result, losses = transition_losses(model, batch, features, teacher, teacher_features)
                losses["collision"] = collision_barrier(result["state"], data.sdf.expand(len(batch["old"]), -1, -1, -1))
                base = model.render_base(batch["new"], batch["new"].new_zeros(len(batch["new"]), 128), features)
                losses["base_image"] = image_loss(base, batch["images"])
                losses["flow"], flow_metrics = model.flow_renderer.loss(batch["images"], base, batch["new"])
                if pairs is not None:
                    indices = self.rng.integers(len(pairs["states"]), size=self.args.batch)
                    pair_state = torch.from_numpy(pairs["states"][indices]).to(self.device)
                    pair_noise = torch.from_numpy(pairs["noise"][indices]).to(self.device).float()
                    endpoint = torch.from_numpy(pairs["endpoints"][indices]).to(self.device).float() / 255
                    pair_base = model.render_base(pair_state, pair_state.new_zeros(len(pair_state), 128), features)
                    losses["reflow"], _ = model.flow_renderer.loss(endpoint, pair_base, pair_state, noise=pair_noise, target_endpoint=endpoint)
                weights = {"state": 10, "player_motion": 5, "npc_motion": 5, "events": 1, "target": 2,
                           "attack": 2, "spawn": .5, "depth": .2, "visibility": .2, "distill": .2,
                           "collision": 1, "base_image": 1, "flow": 1, "reflow": 1}
                loss = sum(value * weights.get(key, 1) for key, value in losses.items())
                adversarial_sample = None
                if self._adversarial_trainer is not None:
                    count = min(len(batch["new"]), getattr(self.args, "quality_endpoint_batch", 1))
                    adversarial_sample = model.flow_renderer.sample(base[:count], batch["new"][:count])
            adversarial_metrics = {}
            if adversarial_sample is not None:
                adversarial_loss, adversarial_metrics = self._adversarial_loss(
                    adversarial_sample, batch["images"][:count], base[:count], batch["new"][:count])
                loss = loss + adversarial_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at {phase} {step}; previous checkpoint preserved")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if step == completed + 1 or step % self.args.log_every == 0:
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                seconds_per_step = (time.perf_counter() - started) / (step - completed)
                self.measured[phase] = seconds_per_step
                metrics = {key: float(value.detach()) for key, value in losses.items()}
                metrics.update({key: float(value) for key, value in flow_metrics.items()})
                metrics.update(adversarial_metrics)
                metrics["optimization_loss"] = float(loss.detach())
                metrics["train_loss"] = sum(float(value.detach()) * weights.get(key, 1)
                                            for key, value in losses.items() if key not in {"distill", "reflow"})
                self.metric({"stage": phase, "step": step, **metrics, "seconds_per_step": seconds_per_step})
                estimate = (steps - step) * seconds_per_step
                self.update(phase, step, steps, "Учитель: Flow Matching" if phase == "teacher" else "Студент: Rectified Flow",
                            eta_seconds=round(estimate), eta_scope="текущий этап; общий прогноз уточняется после каждого измерения",
                            throughput={"value": round(1 / seconds_per_step, 2), "unit": "шагов/с"}, loss=metrics["train_loss"])
            if step % self.args.save_every == 0 or step == steps or step == completed + 1:
                self.save(model, optimizer, phase, step, metrics)
            if step % self.args.validate_every == 0 or step == steps:
                model.eval()
                with torch.inference_mode(), self.amp():
                    val = validation.batch(min(32, self.args.batch), self.rng,
                                           indices=self.rng.integers(validation.count, size=min(32, self.args.batch)))
                    fmap = model.encode_map(val["map"])
                    val_result, values = transition_losses(model, val, fmap)
                    values["collision"] = collision_barrier(val_result["state"], validation.sdf.expand(len(val["old"]), -1, -1, -1))
                    val_base = model.render_base(val["new"], val["new"].new_zeros(len(val["new"]), 128), fmap)
                    flow_loss, _ = model.flow_renderer.loss(val["images"], val_base, val["new"])
                    values["base_image"], values["flow"] = image_loss(val_base, val["images"]), flow_loss
                    value = float(sum(item * weights.get(key, 1) for key, item in values.items()))
                self.metric({"stage": phase, "step": step, "val_loss": value})
                self.preview(model, val, phase, step)

    def temporal(self, model, phase):
        from .long_train import load_episode, run_segment
        episodes = [load_episode(path) for path in sorted((self.root / "data" / "long" / "train").glob("*.npz"))]
        # Preserve the trained image/flow path and shared map/geometry reader.
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(not any(part in name for part in ("flow_renderer", "render_", "rgb", "screen_", "map_encoder", "geometry_reader")))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=self.args.lr / 3)
        completed = self.restore(model, optimizer, phase)
        started = time.perf_counter()
        for index in range(completed, self.args.temporal_updates):
            if (self.root / "stop_training").exists():
                self.save(model, optimizer, phase, index, {})
                raise SystemExit(0)
            horizon = (16, 32, 64, 128)[min(3, index * 4 // max(1, self.args.temporal_updates))]
            episode = episodes[int(self.rng.integers(len(episodes)))]
            prefix = int(self.rng.integers(max(1, episode.transitions - horizon - 8)))
            self.update(phase, index, self.args.temporal_updates, "Непрерывная динамика и память",
                        message=f"Свободный rollout {horizon} шагов без подстановки истинных состояний")
            with self.amp(cache_enabled=False):
                report = run_segment(model, episode, optimizer=optimizer, device=self.device,
                                     prefix_start=prefix, rollout=horizon, chunk=16, burn_in=8,
                                     render_every=0, geometry_every=0, collision_weight=1.)
            metrics = report["metrics"]
            seconds_per_update = (time.perf_counter() - started) / (index - completed + 1)
            self.metric({"stage": phase, "step": index + 1, "train_loss": metrics["loss"], **metrics,
                         "rollout": horizon, "seconds_per_update": seconds_per_update})
            self.update(phase, index + 1, self.args.temporal_updates, "Непрерывная динамика и память",
                        eta_seconds=round((self.args.temporal_updates - index - 1) * seconds_per_update))
            if (index + 1) % 10 == 0 or index + 1 == self.args.temporal_updates:
                self.save(model, optimizer, phase, index + 1, metrics)
        for parameter in model.parameters():
            parameter.requires_grad_(True)

    def reflow(self, teacher, data):
        path = self.root / "reflow_pairs.npz"
        fingerprint = model_fingerprint(teacher)
        self.state["teacher_fingerprint"] = fingerprint
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if "teacher_fingerprint" in saved.files and str(saved["teacher_fingerprint"]) == fingerprint:
                    return {key: saved[key] for key in saved.files}
        teacher.eval()
        states, noises, endpoints = [], [], []
        for first in range(0, self.args.reflow_pairs, self.args.batch):
            count = min(self.args.batch, self.args.reflow_pairs - first)
            batch = data.batch(count, self.rng)
            with torch.inference_mode(), self.amp():
                features = teacher.encode_map(batch["map"])
                state = batch["new"]
                base = teacher.render_base(state, state.new_zeros(count, 128), features)
                noise = torch.randn(count, 3, teacher.config.image_size, teacher.config.image_size, device=self.device).half().float()
                endpoint = teacher.flow_renderer.sample(base, state, noise=noise)
            states.append(state.cpu().numpy())
            noises.append(noise.half().cpu().numpy())
            endpoints.append((endpoint.float().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8))
            self.update("reflow", first + count, self.args.reflow_pairs, "Подготовка reflow-пар учителя",
                        message="Одни и те же noise и teacher endpoint связываются для обучения студента")
        pairs = {"states": np.concatenate(states), "noise": np.concatenate(noises), "endpoints": np.concatenate(endpoints),
                 "teacher_fingerprint": np.asarray(fingerprint)}
        temporary = path.with_name("reflow_pairs.tmp.npz")
        np.savez_compressed(temporary, **pairs)
        replace_file(temporary, path)
        return pairs

    def run(self):
        while True:
            status_path = self.root / "data" / "status.json"
            if status_path.exists():
                collector = json.loads(status_path.read_text(encoding="utf-8"))
                if collector["status"] == "complete":
                    break
                if collector["status"] == "failed":
                    raise RuntimeError("Data collection failed; inspect data/status.json")
            self.update("data", 0, 1, "Ожидание данных")
            time.sleep(5)
        atomic_json(self.root / "experiment_config.json", vars(self.args) | {"run_dir": str(self.root)})
        data = MemoryData(self.root / "data" / "rgb" / "train", self.device)
        validation = MemoryData(self.root / "data" / "rgb" / "val", self.device)
        self.state["dataset"] = {"episodes": sum(1 for _ in (self.root / "data").rglob("*.npz")),
                                 "frames": data.count + validation.count,
                                 "size_bytes": sum(p.stat().st_size for p in (self.root / "data").rglob("*.npz")),
                                 "ready": True, "rgb_transitions": data.count + validation.count,
                                 "long_episodes": sum(1 for _ in (self.root / "data" / "long").rglob("*.npz"))}
        expanded = getattr(self.args, "teacher_size", "current") == "75m"
        if expanded:
            initial_path = self.root / "checkpoints" / "teacher_75m_initial.pt"
            if not initial_path.exists():
                from .model_expansion import expand_teacher_checkpoint
                candidates = [self.root / "checkpoints" / name for name in
                              ("teacher_final_quality.pt", "teacher_quality.pt", "teacher.pt")]
                source = next((path for path in candidates if path.exists()), None)
                if source is None:
                    raise ValueError("The 75M profile needs a trained teacher checkpoint for expansion")
                migration = expand_teacher_checkpoint(source, initial_path)
                atomic_json(self.root / "teacher_75m_migration.json", migration)
            teacher, initial = load_world_checkpoint(initial_path, self.device)
            total = sum(p.numel() for p in teacher.parameters())
            if abs(total - 75000000) > 1000000:
                raise ValueError(f"Expanded checkpoint is not the selected 75M profile: {total}")
            self.state["model_expansion"] = initial.get("expansion", {})
            self.state["model"] = "Teacher 75M Flow Matching + adversarial / Student Rectified Flow"
        else:
            teacher = NeuralWorldModel(WorldModelConfig(kind="teacher", image_size=128, endless=True,
                                                        renderer_mode="flow_matching", flow_steps=12)).to(self.device)
            self.joint(teacher, "teacher", self.args.teacher_steps, data, validation)
        self._publish_model_sizes(teacher)
        quality_phase = "teacher_quality_75m" if expanded else "teacher_quality"
        temporal_phase = "teacher_temporal_75m" if expanded else "teacher_temporal"
        final_quality_phase = "teacher_final_quality_75m" if expanded else "teacher_final_quality"
        if getattr(self.args, "teacher_until_quality", False):
            self.fit_teacher_until_quality(teacher, data, validation, phase=quality_phase)
        self.temporal(teacher, temporal_phase)
        if getattr(self.args, "teacher_until_quality", False):
            # A separate checkpoint preserves the newly trained dynamics rather
            # than restoring the earlier pre-temporal quality checkpoint.
            self.fit_teacher_until_quality(teacher, data, validation, phase=final_quality_phase)
        teacher.requires_grad_(False)
        pairs = self.reflow(teacher, data)
        student = NeuralWorldModel(WorldModelConfig(kind="student", image_size=128, endless=True,
                                                    renderer_mode="rectified_flow", flow_steps=4)).to(self.device)
        self.joint(student, "student", self.args.student_steps, data, validation, teacher, pairs)
        self.temporal(student, "student_temporal")
        model_path = self.root / "checkpoints" / "student_temporal.pt"
        report_path = self.root / "evaluation_300s.json"
        self.update("evaluation", 0, 9000, "Независимая проверка 300 секунд",
                    evaluation={"status": "running", "duration_seconds": 0, "target_seconds": 300, "passed": False})
        command = [sys.executable, "-X", "utf8", "-u", "-m", "turkeyworld.evaluate_rollout", "--world", str(model_path),
                   "--seconds", "300", "--episodes", "1", "--scenario", "mixed", "--device", str(self.device),
                   "--render-every", "30", "--min-render-psnr", "18", "--continue-on-failure", "--output", str(report_path)]
        with (self.root / "evaluation.stdout.log").open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                if "evaluation_step" in item:
                    self.update("evaluation", item["evaluation_step"], 9000, "Независимая проверка 300 секунд")
            code = process.wait()
        if not report_path.exists():
            raise RuntimeError(f"Evaluation failed (exit {code}); inspect evaluation.stdout.log")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        episodes = report.get("episodes", report.get("reports", []))
        passed = bool(episodes and all(e.get("300s_active_pass", False) for e in episodes))
        checked_seconds = min((e.get("checked_simulated_seconds", 0) for e in episodes), default=0)
        self.state["status"] = "complete" if passed else "needs_improvement"
        self.update("complete", 1, 1, "Проверка пройдена" if passed else "Обучение закончено; проверка требует улучшений",
                    evaluation={"status": "passed" if passed else "failed", "target_seconds": 300,
                                "duration_seconds": checked_seconds, "passed": passed,
                                "details": "Подробный независимый отчёт: evaluation_300s.json"},
                    message="Модель готова к ручной проверке" if passed else "Веса сохранены. Пять минут корректной игры не подтверждены; ошибки перечислены в отчёте.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--teacher-steps", type=int, default=12000)
    parser.add_argument("--teacher-size", choices=["current", "75m"], default="current",
                        help="75m expands an existing trained teacher to 75.2M parameters and refines until quality passes")
    parser.add_argument("--adversarial", action="store_true")
    parser.add_argument("--adversarial-weight", type=float, default=.02)
    parser.add_argument("--adversarial-feature-weight", type=float, default=.05)
    parser.add_argument("--adversarial-warmup", type=int, default=64)
    parser.add_argument("--adversarial-ramp", type=int, default=256)
    parser.add_argument("--adversarial-r1-gamma", type=float, default=1.)
    parser.add_argument("--adversarial-r1-interval", type=int, default=16)
    parser.add_argument("--adversarial-lr", type=float, default=1e-4)
    parser.add_argument("--teacher-until-quality", action="store_true",
                        help="After bootstrap, train without a step/time cap until the held-out final-image gate passes")
    parser.add_argument("--quality-frames", type=int, default=256)
    parser.add_argument("--quality-every", type=int, default=500)
    parser.add_argument("--quality-passes", type=int, default=3)
    parser.add_argument("--quality-psnr", type=float, default=35.)
    parser.add_argument("--quality-ssim", type=float, default=.98)
    parser.add_argument("--quality-mae", type=float, default=2/255)
    parser.add_argument("--quality-fraction", type=float, default=.95)
    parser.add_argument("--quality-endpoint-batch", type=int, default=1)
    parser.add_argument("--student-steps", type=int, default=16000)
    parser.add_argument("--temporal-updates", type=int, default=400)
    parser.add_argument("--reflow-pairs", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--validate-every", type=int, default=250)
    args = parser.parse_args(argv)
    if args.teacher_size == "75m":
        args.teacher_until_quality = True
    if min(args.teacher_steps, args.student_steps, args.temporal_updates, args.reflow_pairs,
           args.batch, args.log_every, args.save_every, args.validate_every,
           args.quality_frames, args.quality_every, args.quality_passes, args.quality_endpoint_batch) < 1:
        parser.error("All counts must be positive")
    if (not math.isfinite(args.quality_psnr) or args.quality_psnr <= 0
            or not 0 < args.quality_ssim <= 1 or not 0 < args.quality_mae <= 1
            or not 0 < args.quality_fraction <= 1):
        parser.error("Quality thresholds must be finite, positive and in range")
    if (min(args.adversarial_weight, args.adversarial_feature_weight, args.adversarial_r1_gamma,
            args.adversarial_warmup) < 0 or min(args.adversarial_ramp, args.adversarial_r1_interval,
                                               args.adversarial_lr) <= 0
            or not all(math.isfinite(value) for value in (args.adversarial_weight,
                args.adversarial_feature_weight, args.adversarial_r1_gamma, args.adversarial_lr))):
        parser.error("Adversarial weights/warmup must be nonnegative, and rates/intervals positive")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    experiment = Experiment(args)
    try:
        from .power_guard import KeepAwake
        with KeepAwake() as awake:
            experiment.state['power_guard'] = {'active': awake.active,
                'screen_lock_allowed': True, 'display_required': False, 'error': awake.error}
            experiment.run()
    except Exception as error:
        experiment.state["status"] = "failed"
        experiment.update(experiment.state.get("stage", "initializing"), experiment.state.get("current", 0),
                          experiment.state.get("total", 1), "Ошибка процесса", message=str(error))
        raise


if __name__ == "__main__":
    main()
