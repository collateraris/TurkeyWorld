"""Resumable CPU expansion of source-oracle game datasets, with hardlink reuse.

New clips balance movement, wall contact, combat, misses, kills, respawns,
return-memory, delayed attacks and TRAIN-ONLY perturbed reference states.
Perturbations are explicitly not claimed to be model-visited DAgger states.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np

from .engine import EVENT_NAMES
from .prepare_training_data import (_atomic_json, _episode_stats, _generate_job,
                                    _require_episode_behavior, current_behavior_version)


TRAIN_POLICIES = ("movement", "collision", "combat", "misses", "kill", "respawn", "return_memory", "attack", "perturbed")
VALIDATION_POLICIES = TRAIN_POLICIES[:-1]


def _frame_shape(path, field="frames"):
    with zipfile.ZipFile(path) as archive, archive.open(field+".npy") as stream:
        version = np.lib.format.read_magic(stream)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        return reader(stream)[0]


def _describe(path, output, kind, split, reused=False):
    with np.load(path, allow_pickle=False) as saved:
        behavior_version = _require_episode_behavior(saved, current_behavior_version(), path)
        if "split" in saved.files and str(saved["split"]) != split:
            raise ValueError(f"Refusing cross-split source reuse: {path}")
        seed = int(saved["seed"])
        count = int(saved["requested_transitions"]) if "requested_transitions" in saved.files else len(saved["actions"])
        return {"path": str(path), "relative_path": str(path.relative_to(output)).replace("\\", "/"),
                "behavior_version": behavior_version,
                "rgb_time_span": float(saved["rgb_time_span"]) if "rgb_time_span" in saved.files else 0.,
                "job_id": f"{kind}/{split}/{seed}", "kind": kind, "split": split, "seed": seed,
                "frames": count, "hz": 1/float(saved["dt"]), "policy": str(saved["policy"]), "reused": reused}


def _reuse(source, output, targets, size, long_frames, rgb_time_span=0., hr_size=0):
    jobs, modes = [], {"hardlink": 0, "copy": 0, "existing": 0}
    if source is None or not Path(source).is_dir():
        return jobs, modes
    for kind in ("rgb", "long"):
        for split in ("train", "val"):
            target = targets[f"{kind}_{split}"]
            selected = 0
            for path in sorted((Path(source)/kind/split).glob("*.npz")):
                if selected >= target:
                    break
                with np.load(path, allow_pickle=False) as saved:
                    _require_episode_behavior(saved, current_behavior_version(), path)
                    if kind == "rgb" and (float(saved["rgb_time_span"]) if "rgb_time_span" in saved.files else 0.) != rgb_time_span:
                        raise ValueError("Source RGB time span differs from the requested corpus")
                    if "split" in saved.files and str(saved["split"]) != split:
                        raise ValueError("Original corpus split metadata is inconsistent")
                    if "endless" in saved.files and not bool(saved["endless"]):
                        continue
                    if kind == "long" and len(saved["actions"]) < long_frames:
                        continue
                    if kind == "rgb" and tuple(_frame_shape(path)[1:]) != (size, size, 3):
                        continue
                    if kind == "rgb" and hr_size and ("hr_frames" not in saved.files
                            or tuple(_frame_shape(path, "hr_frames")[1:]) != (hr_size, hr_size, 3)):
                        continue
                destination = output/kind/split/path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    modes["existing"] += 1
                else:
                    try:
                        os.link(path, destination)
                        modes["hardlink"] += 1
                    except OSError:
                        shutil.copy2(path, destination)
                        modes["copy"] += 1
                jobs.append(_describe(destination, output, kind, split, reused=True))
                selected += 1
    return jobs, modes


def _safe_generate(job):
    for attempt in range(4):
        try:
            return _generate_job(dict(job, initialization_attempt=attempt))
        except RuntimeError as error:
            if "Warmup ended in death" not in str(error) or attempt == 3:
                raise


def _append_target_growth(output, jobs, config):
    """Append new immutable jobs; never rebuild or renumber an existing plan."""
    targets, seed = config["targets"], config["seed"]
    references = [job["path"] for job in jobs if job["kind"] == "rgb" and job["split"] == "train" and job["reused"]]
    extra = []
    for kind, split in (("rgb", "val"), ("rgb", "train"), ("long", "val"), ("long", "train")):
        existing = [job for job in jobs if job["kind"] == kind and job["split"] == split]
        target = targets[f"{kind}_{split}"]
        if kind == "rgb" and target < len(existing):
            raise ValueError("Expansion target shrank below an already committed append plan")
        band_seed = seed+(int(split == "val")+2*int(kind == "long"))*1_000_000
        next_index = max((job["seed"]-band_seed for job in existing if not job["reused"]), default=-1)+1
        choices = TRAIN_POLICIES if split == "train" else VALIDATION_POLICIES
        for offset in range(max(0, target-len(existing))):
            index = next_index+offset
            if index >= 1_000_000:
                raise ValueError("Append would cross a reserved disjoint seed band")
            job_seed = band_seed+index
            relative = f"{kind}/{split}/episode_{job_seed:010d}.npz"
            policy = choices[index % len(choices)] if kind == "rgb" else "survival"
            extra.append({"path": str(output/relative), "relative_path": relative,
                "behavior_version": config["behavior_version"],
                "rgb_time_span": config.get("rgb_time_span", 0.) if kind == "rgb" else 0.,
                "job_id": f"{kind}/{split}/{job_seed}", "kind": kind, "split": split, "seed": job_seed,
                "frames": config["clip_frames"] if kind == "rgb" else config["long_frames"], "policy": policy,
                "size": config["size"], "hr_size": config.get("hr_size", 0) if kind == "rgb" else 0,
                "hz": config["hz"], "endless": True, "long_geometry": False,
                "return_offset": (0, 48, 80, 96, 128, 160, 176, 192)[index % 8],
                "delayed_attack": policy == "attack", "reused": False,
                "reference_path": references[index % len(references)] if references and policy == "perturbed" else None})
    return extra


def _add_record(aggregate, record, long_frames):
    groups, policies = aggregate["groups"], aggregate["policies"]
    key = f"{record['kind']}/{record['split']}"
    group = groups.setdefault(key, {"files": 0, "transitions": 0, "size_bytes": 0,
        "events": {name: 0 for name in EVENT_NAMES}, "eligible_9000": 0, "eligible_requested_length": 0,
        "hit_events_after_first_action": 0, "attack_events_after_first_action": 0,
        "moving_transitions": 0, "blocked_movement_steps": 0, "terminal_episodes": 0,
        "respawns": 0, "reused_files": 0})
    group["files"] += 1
    for name in ("transitions", "size_bytes", "hit_events_after_first_action", "attack_events_after_first_action",
                 "moving_transitions", "blocked_movement_steps", "respawns"):
        group[name] += record.get(name, 0)
    group["eligible_9000"] += int(record["transitions"] >= 9000)
    group["eligible_requested_length"] += int(record["transitions"] >= long_frames)
    group["terminal_episodes"] += int(record["terminal"])
    group["reused_files"] += int(record.get("reused", False))
    aggregate["reused"] += int(record.get("reused", False))
    for name in EVENT_NAMES:
        group["events"][name] += record["events"][name]
    policy = policies.setdefault(key+"/"+record["policy"], {"files": 0, "transitions": 0,
        "hit": 0, "kill": 0, "player_hit": 0, "missed_shots": 0,
        "blocked_movement_steps": 0, "hit_events_after_first_action": 0, "attack_events_after_first_action": 0})
    policy["files"] += 1
    for name in ("transitions", "missed_shots", "blocked_movement_steps", "hit_events_after_first_action", "attack_events_after_first_action"):
        policy[name] += record.get(name, 0)
    for name in ("hit", "kill", "player_hit"):
        policy[name] += record["events"][name]


def _summarize(records, jobs, started, newly_finished, errors, targets=None, long_frames=9000, aggregate=None):
    if aggregate is None:
        aggregate = {"groups": {}, "policies": {}, "reused": 0}
        for record in records.values():
            _add_record(aggregate, record, long_frames)
        aggregate["planned_rgb_training_transitions"] = sum(job["frames"] for job in jobs if job["kind"] == "rgb" and job["split"] == "train")
    groups, policies, reused = aggregate["groups"], aggregate["policies"], aggregate["reused"]
    elapsed, remaining = time.perf_counter()-started, len(jobs)-len(records)
    qualified = targets is None or all(groups.get(f"long/{split}", {}).get("eligible_requested_length", 0) >= targets[f"long_{split}"]
                                      for split in ("train", "val"))
    status = "failed" if errors and not remaining else "complete" if not remaining and qualified else "running"
    return {"status": status, "state": status, "completed": len(records), "total": len(jobs),
        "behavior_version": current_behavior_version(),
        "completed_files": len(records), "planned_files": len(jobs), "remaining_files": remaining,
        "reused_files": reused, "new_files": len(records)-reused,
        "elapsed_seconds": round(elapsed, 2), "eta_seconds": round(elapsed*remaining/newly_finished) if newly_finished else None,
        "groups": groups, "policies": policies, "errors": errors,
        "full_length_long_targets_satisfied": qualified,
        "ready_train_minimum": (groups.get("rgb/train", {}).get("files", 0) >= min(8192, (targets or {}).get("rgb_train", 8192))
            and groups.get("rgb/val", {}).get("files", 0) >= min(128, (targets or {}).get("rgb_val", 128))),
        "planned_rgb_training_transitions": aggregate.get("planned_rgb_training_transitions", 0),
        "readiness": {"minimum_ready": groups.get("rgb/train", {}).get("files", 0) >= 512 and groups.get("rgb/val", {}).get("files", 0) >= 128,
            "rgb_train_files": groups.get("rgb/train", {}).get("files", 0), "rgb_val_files": groups.get("rgb/val", {}).get("files", 0),
            "new_rgb_train_files": groups.get("rgb/train", {}).get("files", 0)-groups.get("rgb/train", {}).get("reused_files", 0)},
        "augmentation_provenance": "TRAIN-ONLY valid perturbed reference states; not claimed to be model-visited or DAgger.",
        "runtime_engine_feedback": False, "quality_claim": "Source data coverage only; no neural gameplay quality guarantee."}


def prepare_expansion(output, *, source=None, rgb_train=8192, rgb_val=512,
                      long_train=256, long_val=32, clip_frames=32, long_frames=9000,
                      size=128, hz=30, seed=10_000_000, workers=8, behavior_version=None, rgb_time_span=0., hr_size=0):
    behavior_version = current_behavior_version(behavior_version)
    if not np.isfinite(rgb_time_span) or rgb_time_span < 0:
        raise ValueError("RGB time span must be finite and nonnegative")
    if hr_size not in (0, size*4):
        raise ValueError("Paired HR size must be zero or exactly four times native RGB")
    output = Path(output).resolve()
    source = Path(source).resolve() if source else None
    output.mkdir(parents=True, exist_ok=True)
    targets = dict(rgb_train=rgb_train, rgb_val=rgb_val, long_train=long_train, long_val=long_val)
    if min(targets.values()) < 0 or max(targets.values()) >= 1_000_000 or min(workers, clip_frames, long_frames) < 1:
        raise ValueError("Invalid expanded dataset counts or lengths")
    config = {"targets": targets, "source": str(source) if source else None,
              "clip_frames": clip_frames, "long_frames": long_frames, "size": size, "hz": hz, "seed": seed,
              "behavior_version": behavior_version, "rgb_time_span": float(rgb_time_span)}
    if hr_size:
        config['hr_size'] = hr_size
    manifest_path, plan_path = output/"manifest.json", output/"expansion_plan.json"
    if plan_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        old_config = manifest["config"]
        old_config.setdefault("rgb_time_span", 0.)
        if ({key: value for key, value in old_config.items() if key != "targets"}
                != {key: value for key, value in config.items() if key != "targets"}
                or any(targets[key] < old_config["targets"][key] for key in targets)):
            raise ValueError("Expansion configuration changed; choose another output directory")
        jobs = json.loads(plan_path.read_text(encoding="utf-8"))
        if any(job.get("behavior_version") != behavior_version for job in jobs):
            raise ValueError("Expansion plan behavior version changed; choose another output directory")
        extra = _append_target_growth(output, jobs, config)
        if extra:
            jobs.extend(extra)
            # Plan first, manifest second. A retry after a manifest write
            # interruption sees these exact committed seeds and adds no twins.
            _atomic_json(plan_path, jobs)
        if old_config != config:
            manifest.setdefault("growth_history", []).append({"from_targets": old_config["targets"],
                "to_targets": dict(targets), "added_planned_files": len(extra),
                "committed_at": datetime.now(timezone.utc).isoformat(), "mode": "append_only"})
            manifest["config"], manifest["targets"] = config, dict(targets)
            _atomic_json(manifest_path, manifest)
    else:
        reused_jobs, modes = _reuse(source, output, targets, size, long_frames, rgb_time_span, hr_size)
        references = [job["path"] for job in reused_jobs if job["kind"] == "rgb" and job["split"] == "train"]
        jobs = list(reused_jobs)
        # Expanded validation is generated before bulk training clips.
        for kind, split in (("rgb", "val"), ("rgb", "train"), ("long", "val"), ("long", "train")):
            reused_count = sum(job["kind"] == kind and job["split"] == split for job in reused_jobs)
            band = int(split == "val") + 2*int(kind == "long")
            choices = TRAIN_POLICIES if split == "train" else VALIDATION_POLICIES
            for index in range(targets[f"{kind}_{split}"]-reused_count):
                job_seed = seed+band*1_000_000+index
                relative = f"{kind}/{split}/episode_{job_seed:010d}.npz"
                policy = choices[index % len(choices)] if kind == "rgb" else "survival"
                jobs.append({"path": str(output/relative), "relative_path": relative,
                    "behavior_version": behavior_version,
                    "rgb_time_span": float(rgb_time_span) if kind == "rgb" else 0.,
                    "job_id": f"{kind}/{split}/{job_seed}", "kind": kind, "split": split, "seed": job_seed,
                    "frames": clip_frames if kind == "rgb" else long_frames, "policy": policy,
                    "size": size, "hr_size": hr_size if kind == "rgb" else 0,
                    "hz": hz, "endless": True, "long_geometry": False,
                    "return_offset": (0, 48, 80, 96, 128, 160, 176, 192)[index % 8],
                    "delayed_attack": policy == "attack", "reused": False,
                    "reference_path": references[index % len(references)] if references and policy == "perturbed" else None})
        manifest = {"format_version": 4, "expansion_version": 1, "behavior_version": behavior_version, "config": config,
            "targets": targets, "reuse": modes, "paths": {"world_training": "rgb/train", "world_validation": "rgb/val",
                "long_training": "long/train", "long_validation": "long/val"},
            "seed_rule": f"Preserved inherited split seeds; fresh disjoint RGBtrain/RGBval/longtrain/longval bands at{seed},+1M,+2M,+3M.",
            "train_policies": list(TRAIN_POLICIES), "validation_policies": list(VALIDATION_POLICIES),
            "rgb_time_initialization": "Seeded initial clock offset before warmup; valid training states, not elapsed survival; long recordings start at zero",
            "oracle_augmentation": {"training_only": True, "model_visited": False,
                "source": "valid perturbed reference initial states", "labels": "exact Game.step subsequent transitions"}}
        _atomic_json(manifest_path, manifest)
        _atomic_json(plan_path, jobs)
    records, pending, errors = {}, [], []
    for job in jobs:
        if Path(job["path"]).is_file():
            record = _episode_stats(Path(job["path"]), job)
            record["reused"] = bool(job["reused"])
            records[job["job_id"]] = record
        else:
            pending.append(job)
    aggregate = {"groups": {}, "policies": {}, "reused": 0}
    for record in records.values():
        _add_record(aggregate, record, long_frames)
    aggregate["planned_rgb_training_transitions"] = sum(job["frames"] for job in jobs if job["kind"] == "rgb" and job["split"] == "train")
    started, newly_finished, last_write = time.perf_counter(), 0, 0.

    def write_status(force=False):
        nonlocal last_write
        if not force and time.perf_counter()-last_write < 2:
            return
        status = _summarize(records, jobs, started, newly_finished, errors, targets, long_frames, aggregate)
        _atomic_json(output/"status.json", status)
        manifest["actual"] = status
        _atomic_json(manifest_path, manifest)
        last_write = time.perf_counter()

    write_status(True)

    def completed(job, result=None, error=None):
        nonlocal newly_finished
        if error is not None:
            entry = {"job_id": job["job_id"], "error": repr(error)}
            errors.append(entry)
        else:
            result["reused"] = False
            records[job["job_id"]] = result
            _add_record(aggregate, result, long_frames)
            newly_finished += 1
            entry = result
        with (output/"progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry)+"\n")
        if newly_finished % 16 == 0 or error:
            print(json.dumps({"completed": len(records), "total": len(jobs), "new": newly_finished,
                              "last_job": job["job_id"], "error": entry.get("error")}), flush=True)
        write_status()

    def execute(batch):
        if workers == 1:
            for job in batch:
                try:
                    result = _safe_generate(job)
                except Exception as error:
                    completed(job, error=error)
                else:
                    completed(job, result)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                pending_jobs = iter(batch)
                futures = {}
                def refill():
                    while len(futures) < workers*4:
                        job = next(pending_jobs, None)
                        if job is None:
                            break
                        futures[pool.submit(_safe_generate, job)] = job
                refill()
                while futures:
                    finished, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in finished:
                        job = futures.pop(future)
                        try:
                            result = future.result()
                        except Exception as error:
                            completed(job, error=error)
                        else:
                            completed(job, result)
                    refill()

    execute(pending)
    # Preserve genuine early-death recordings, but do not count them as full
    # 9000-step examples. Add fresh independent source episodes until the
    # requested full-length train/validation quotas are actually satisfied.
    for qualification_round in range(32):
        summary = _summarize(records, jobs, started, newly_finished, errors, targets, long_frames, aggregate)
        extra = []
        for split in ("train", "val"):
            needed = targets[f"long_{split}"]-summary["groups"].get(f"long/{split}", {}).get("eligible_requested_length", 0)
            if needed <= 0:
                continue
            band_seed = seed+(2+int(split == "val"))*1_000_000
            next_index = max((job["seed"]-band_seed for job in jobs if job["kind"] == "long" and job["split"] == split and not job["reused"]), default=-1)+1
            for offset in range(needed):
                job_seed = band_seed+next_index+offset
                relative = f"long/{split}/episode_{job_seed:010d}.npz"
                extra.append({"path": str(output/relative), "relative_path": relative,
                    "behavior_version": behavior_version,
                    "rgb_time_span": 0.,
                    "job_id": f"long/{split}/{job_seed}", "kind": "long", "split": split,
                    "seed": job_seed, "frames": long_frames, "policy": "survival", "size": size,
                    "hr_size": 0, "hz": hz, "endless": True, "long_geometry": False, "reused": False,
                    "qualification_round": qualification_round+1})
        if not extra or errors:
            break
        jobs.extend(extra)
        _atomic_json(plan_path, jobs)
        write_status(True)
        execute(extra)
    else:
        errors.append({"error": "Unable to meet genuine full-length long-recording quotas after32 rounds"})
    write_status(True)
    if errors:
        raise RuntimeError(f"{len(errors)} expansion jobs failed; inspect status.json/progress.jsonl")
    return _summarize(records, jobs, started, newly_finished, errors, targets, long_frames, aggregate)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--rgb-train", type=int, default=8192)
    parser.add_argument("--rgb-val", type=int, default=512)
    parser.add_argument("--rgb-time-span", type=float, default=0., help="Seeded RGB-only initial game-time range; long recordings still start at zero")
    parser.add_argument("--long-train", type=int, default=256)
    parser.add_argument("--long-val", type=int, default=32)
    parser.add_argument("--clip-frames", type=int, default=32)
    parser.add_argument("--long-frames", type=int, default=9000)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--hr-size", type=int, default=0, help="Genuine paired source RGB, either0 or native size times4")
    parser.add_argument("--hz", type=int, default=30)
    parser.add_argument("--seed", type=int, default=10_000_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--behavior-version", help="Assert engine behavior version; incompatible source/resume data are rejected")
    args = vars(parser.parse_args(argv))
    keep_awake = None
    if os.name == "nt":
        import ctypes
        keep_awake = ctypes.windll.kernel32.SetThreadExecutionState
        keep_awake(0x80000001)
    try:
        print(json.dumps(prepare_expansion(**args), indent=2), flush=True)
    finally:
        if keep_awake:
            keep_awake(0x80000000)


if __name__ == "__main__":
    main()
