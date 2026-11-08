"""Batch-one model throughput, with honest untrained/checkpoint status.

No window, display transfer, gameplay validity or 300-second accuracy is measured.
Every reported CUDA interval uses events and synchronization after the frame.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .models import (GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS, STATE_DIM,
                     NeuralWorldModel, SingleFrameUpscaler, WorldModelConfig,
                     load_world_checkpoint, load_sr_checkpoint, parameter_counts)


def summary(values):
    data = np.asarray(values, dtype=np.float64)
    return {"mean_ms": float(data.mean()), "p50_ms": float(np.percentile(data, 50)),
            "p95_ms": float(np.percentile(data, 95))}


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="Optional trained world checkpoint; absent means random weights")
    parser.add_argument("--sr-checkpoint", help="Optional trained SR checkpoint; absent means bilinear-initialized SR")
    parser.add_argument("--kind", choices=["student", "teacher"], default="student")
    parser.add_argument("--architecture", choices=["camera_transformer", "legacy_cnn"],
                        default=None, help="Architecture for random weights; old checkpoints retain their own architecture")
    parser.add_argument("--image-size", type=int, choices=[128, 256], default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--no-sr", action="store_true")
    parser.add_argument("--episode", help="Optional real NPZ: use its map and initial state")
    parser.add_argument("--output-json", help="Save the same report printed to stdout")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.frames < 1 or args.warmup < 0 or args.threads < 1:
        raise ValueError("frames and threads must be positive; warmup cannot be negative")
    torch.set_num_threads(args.threads)
    torch.manual_seed(7)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but this PyTorch installation cannot use it")
    if device.type == "cpu" and args.dtype != "fp32":
        raise ValueError("Use fp32 for the portable CPU benchmark; mixed precision is a GPU option")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    checkpoint = sr_checkpoint = None
    if args.checkpoint:
        model, checkpoint = load_world_checkpoint(args.checkpoint, device)
        if args.architecture is not None and args.architecture != model.config.architecture:
            raise ValueError("architecture disagrees with the checkpoint")
        if args.image_size is not None and args.image_size != model.config.image_size:
            raise ValueError("image-size disagrees with the checkpoint; benchmark its trained configuration")
    else:
        model = NeuralWorldModel(WorldModelConfig(kind=args.kind, image_size=args.image_size or 256,
                                                 architecture=args.architecture or "camera_transformer")).to(device)
    upscaler = None
    if not args.no_sr:
        if args.sr_checkpoint:
            upscaler, sr_checkpoint = load_sr_checkpoint(args.sr_checkpoint, device)
        else:
            upscaler = SingleFrameUpscaler().to(device)
        upscaler = upscaler.eval().to(dtype=dtype)
    model = model.eval().to(dtype=dtype)
    state = torch.zeros(1, STATE_DIM, device=device, dtype=dtype)
    state[:, :2] = 0.4
    state[:, 3:6] = 1
    for slot in range(ENTITY_SLOTS):
        offset = GLOBAL_DIM + slot * ENTITY_DIM
        state[:, offset:offset + 2] = 0.5
        state[:, offset + 4:offset + 6] = 1
    grid = torch.zeros(1, 16, 16, device=device, dtype=torch.long)
    grid[:, 0, :] = grid[:, -1, :] = 1
    grid[:, :, 0] = grid[:, :, -1] = 1
    if args.episode:
        with np.load(args.episode, allow_pickle=False) as episode:
            state = torch.as_tensor(episode["states"][0:1], device=device, dtype=dtype)
            grid = torch.as_tensor(episode["map"][None], device=device, dtype=torch.long)
    action = torch.zeros(1, 4, device=device, dtype=dtype)
    action[:, 0] = 0.2
    memory = None
    timings = {key: [] for key in ["dynamics", "renderer", "upscaler", "model_frame", "host_frame"]}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        map_started = time.perf_counter()
        map_features = model.encode_map(grid)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        map_ms = (time.perf_counter() - map_started) * 1000
        for index in range(args.warmup + args.frames):
            host_started = time.perf_counter()
            if device.type == "cuda":
                markers = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
                markers[0].record()
            else:
                markers = [time.perf_counter()]
            result = model.transition(state, action, map_features, memory)
            if device.type == "cuda":
                markers[1].record()
            else:
                markers.append(time.perf_counter())
            rgb = model.render(result["state"], result["features"], map_features)
            if device.type == "cuda":
                markers[2].record()
            else:
                markers.append(time.perf_counter())
            output = upscaler(rgb) if upscaler is not None else rgb
            if device.type == "cuda":
                markers[3].record()
                markers[3].synchronize()
                elapsed = [markers[i].elapsed_time(markers[i + 1]) for i in range(3)]
                frame_ms = markers[0].elapsed_time(markers[3])
            else:
                markers.append(time.perf_counter())
                elapsed = [(markers[i + 1] - markers[i]) * 1000 for i in range(3)]
                frame_ms = (markers[3] - markers[0]) * 1000
            host_ms = (time.perf_counter() - host_started) * 1000
            state, memory = result["state"], result["memory"]
            if index >= args.warmup:
                for key, value in zip(["dynamics", "renderer", "upscaler"], elapsed):
                    timings[key].append(value)
                timings["model_frame"].append(frame_ms)
                timings["host_frame"].append(host_ms)
    report = {
        "scope": "Batch-one inference throughput only; no gameplay-quality or display measurement.",
        "world_weights": "UNTRAINED_RANDOM" if checkpoint is None else "SUPERVISED_CHECKPOINT_QUALITY_UNVERIFIED",
        "world_training_steps": 0 if checkpoint is None else checkpoint.get("training_steps", 0),
        "sr_weights": ("DISABLED" if upscaler is None else
                       "UNTRAINED_BILINEAR_INITIALIZATION" if sr_checkpoint is None else "TRUE_PAIR_TRAINED_QUALITY_UNVERIFIED"),
        "sr_training_steps": 0 if sr_checkpoint is None else sr_checkpoint.get("training_steps", 0),
        "config": asdict(model.config), "parameters": parameter_counts(model, upscaler),
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "CPU",
        "torch_version": torch.__version__, "dtype": args.dtype, "cpu_threads": args.threads,
        "warmup_frames": args.warmup, "measured_frames": args.frames,
        "output_shape": list(output.shape), "map_encode_once_ms": map_ms,
        "stages": {key: summary(value) for key, value in timings.items()},
        "model_only_fps_from_mean": 1000 / float(np.mean(timings["model_frame"])),
        "host_inference_fps_from_mean": 1000 / float(np.mean(timings["host_frame"])),
        "within_30fps_p95_model_budget": bool(np.percentile(timings["model_frame"], 95) <= 1000 / 30),
        "within_60fps_p95_model_budget": bool(np.percentile(timings["model_frame"], 95) <= 1000 / 60),
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else None,
        "limitations": ["No CPU readback, window presentation or input latency is included.",
                        "A small training-step count does not prove accuracy or five-minute stability.",
                        "Static map encoding is cached; recurrent world state persists per frame.",
                        "SR receives only current LR RGB; it has no temporal reconstruction guarantees."],
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    print(encoded)
    if args.output_json:
        destination = Path(args.output_json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
