"""Strict held-out codec reconstruction measurements and source/recon previews.

Source pixels are legitimate INPUTS to this first-stage image codec. This
evaluation does not measure autonomous world prediction and must not be used
as evidence of future-RGB conditioning being allowed in the world model.
Code usage is measured over the complete fixed panel, not averaged per batch
and not inferred from the quantizer's training-history usage EMA.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from .visual_quality import _thresholds, image_quality_metrics, strict_render_precision, summarize_visual_quality


DEFAULT_CODEC_THRESHOLDS = {"psnr": 40., "ssim": .99, "mae": 1 / 255,
                            "required_fraction": .95}


def _uint8(images):
    return (images.detach().float().nan_to_num(nan=0., posinf=1., neginf=0.).clamp(0, 1)
            .mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy())


def _edge_error(predicted, source):
    values = []
    if predicted.shape[-1] > 1:
        values.append(((predicted[..., 1:] - predicted[..., :-1])
                       - (source[..., 1:] - source[..., :-1])).abs().flatten(1).mean(1))
    if predicted.shape[-2] > 1:
        values.append(((predicted[..., 1:, :] - predicted[..., :-1, :])
                       - (source[..., 1:, :] - source[..., :-1, :])).abs().flatten(1).mean(1))
    return torch.stack(values).mean(0) if values else predicted.new_zeros(len(predicted))


def _finite_value(value, fallback=1e6):
    value = float(value)
    return value if math.isfinite(value) else float(fallback)


def evaluate_vqgan(model, data, panel=None, *, count=128, batch_size=4, device=None,
                   thresholds=None, seed=73, step=0, minimum_active_codes=3):
    """Return (JSON report, CPU preview) for 128 adjacent held-out pairs.

    Default readiness requires 40dB/.99/1-of255 jointly on 95% of 256 frames,
    every reconstruction finite/in range, valid finite quantization and at
    least 3 distinct panel codes. The trainer owns consecutive-check persistence.
    All module modes, CPU/selected-CUDA/Python/NumPy/data RNG and precision
    policy are restored even on failure. No training usage buffers are updated.
    """
    if min(count, batch_size, minimum_active_codes) < 1:
        raise ValueError("Codec panel, batch and active-code counts must be positive")
    if getattr(data, "split", None) != "val":
        raise ValueError("Codec qualification requires the explicit held-out val split")
    parameters = list(model.parameters())
    if any(parameter.is_floating_point() and parameter.dtype != torch.float32 for parameter in parameters):
        raise ValueError("Codec qualification requires strict FP32 model parameters")
    device = torch.device(device if device is not None else parameters[0].device if parameters else data.device)
    configuration = getattr(model, "config", None)
    configuration = (asdict(configuration) if is_dataclass(configuration) else dict(configuration)
                     if isinstance(configuration, dict) else vars(configuration) if configuration is not None else {})
    n_embed = int(configuration.get("n_embed", 0))
    if n_embed < 1:
        raise ValueError("Codec configuration must declare a positive n_embed")
    limits = dict(DEFAULT_CODEC_THRESHOLDS)
    limits.update(thresholds or {})
    limits = _thresholds(limits)
    pairs = np.asarray(data.fixed_panel(count=count, seed=seed) if panel is None else panel)
    if (pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs)
            or not np.issubdtype(pairs.dtype, np.integer)):
        raise ValueError("Codec panel must contain integer (episode,start) pairs")
    panel = [(int(episode), int(start)) for episode, start in pairs]
    if len(set(panel)) != len(panel):
        raise ValueError("Codec validation panel pairs must be unique")
    for episode, start in panel:
        if not 0 <= episode < len(data.episodes) or not 0 <= start < len(data.episodes[episode]["frames"]) - 1:
            raise ValueError("Codec validation pair crosses an episode boundary")
    modes = [(module, module.training) for module in model.modules()]
    numpy_state, python_state = np.random.get_state(), random.getstate()
    generators = []
    for owner in (data, getattr(data, "data", None)):
        rng = getattr(owner, "rng", None)
        if isinstance(rng, np.random.Generator) and not any(rng is other for other, _ in generators):
            generators.append((rng, copy.deepcopy(rng.bit_generator.state)))
    cuda_devices = ([device.index if device.index is not None else torch.cuda.current_device()]
                    if device.type == "cuda" else [])
    per_frame, per_pair, images = [], [], []
    histogram = torch.zeros(n_embed, dtype=torch.int64)
    invalid_indices, quantization_finite = 0, True
    source_hash = hashlib.sha256()
    started = time.perf_counter()
    try:
        model.eval()
        with strict_render_precision(), torch.random.fork_rng(devices=cuda_devices), \
                torch.inference_mode(), torch.autocast(device.type, enabled=False):
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            for first in range(0, len(panel), batch_size):
                chosen = panel[first:first + batch_size]
                batch = data.batch(len(chosen), indices=chosen)
                frames = batch["frames"].to(device=device, dtype=torch.float32)
                if (frames.ndim != 5 or frames.shape[:3] != (len(chosen), 2, 3)
                        or not bool(torch.isfinite(frames).all())
                        or bool((frames < 0).any() or (frames > 1).any())):
                    raise ValueError("Codec validation requires clean source RGB [B,2,3,H,W] in[0,1]")
                source = frames.flatten(0, 1)
                output = model(source)
                reconstruction = output["reconstruction"].float()
                if reconstruction.shape != source.shape:
                    raise ValueError("Codec reconstruction/source image dimensions differ")
                prequant, quantized = output["prequant"], output["quantized"]
                if prequant.shape != quantized.shape or len(prequant) != len(source):
                    raise ValueError("Codec prequant/quantized feature dimensions differ")
                quantization = (prequant.double() - quantized.double()).square().flatten(1).mean(1)
                quantization_finite &= bool(torch.isfinite(quantization).all())
                codes = output["indices"]
                if codes.ndim != 3 or len(codes) != len(source) or codes.is_floating_point():
                    raise ValueError("Codec indices must be integer [frames,latent_h,latent_w]")
                codes = codes.detach().cpu().long().flatten()
                valid = (codes >= 0) & (codes < n_embed)
                invalid_indices += int((~valid).sum())
                histogram += torch.bincount(codes[valid], minlength=n_embed)
                quality = {name: values.cpu().tolist() for name, values in
                           image_quality_metrics(reconstruction, source).items()}
                edges = _edge_error(reconstruction.double(), source.double()).cpu().tolist()
                quantization = quantization.cpu().tolist()
                paired = reconstruction.reshape_as(frames)
                temporal = (((paired[:, 1].double() - paired[:, 0].double())
                             - (frames[:, 1].double() - frames[:, 0].double()))
                            .abs().flatten(1).mean(1).cpu().tolist())
                events = batch["events"].detach().cpu().numpy()
                states = batch.get("states")
                states = states.detach().cpu().numpy() if states is not None else None
                for row, (episode, start) in enumerate(chosen):
                    # Per-pair order makes the fingerprint independent of batch size.
                    for name, value in (("frames", frames[row].cpu().numpy()), ("events", events[row])):
                        array = np.ascontiguousarray(value)
                        source_hash.update(name.encode())
                        source_hash.update(str((array.shape, array.dtype)).encode())
                        source_hash.update(array.tobytes())
                    if states is not None:
                        source_hash.update(np.ascontiguousarray(states[row]).tobytes())
                    fired = bool(np.asarray(events[row]).reshape(-1, 4)[:, 0].max() > .5)
                    per_pair.append({"pair": first + row, "episode_index": episode, "start": start,
                        "shot_transition": fired, "temporal_reconstruction_error": _finite_value(temporal[row])})
                    for offset in range(2):
                        index = row * 2 + offset
                        state_flash = states is not None and bool(states[row, offset, 7] > 0)
                        per_frame.append({"pair": first + row, "episode_index": episode,
                            "episode": Path(data.episodes[episode]["path"]).name,
                            "start": start, "offset": offset, "source_frame": start + offset,
                            "recorded_flash": state_flash or (fired and offset == 1),
                            "edge_l1": _finite_value(edges[index]),
                            "quantization_mse": _finite_value(quantization[index]),
                            **{name: values[index] for name, values in quality.items()}})
                        if len(images) < 4:
                            images.append((source[index].detach().cpu(), reconstruction[index].detach().cpu()))
    finally:
        for module, training in modes:
            module.training = training
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        for generator, saved in generators:
            generator.bit_generator.state = saved
    gate = summarize_visual_quality(per_frame, limits)
    for frame, passed in zip(per_frame, gate.pop("frame_passed")):
        frame["passed"] = passed
    active = int((histogram > 0).sum())
    total_tokens = int(histogram.sum())
    probabilities = histogram.double() / max(1, total_tokens)
    perplexity = float((-(probabilities * probabilities.clamp_min(1e-300).log()).sum()).exp()) if total_tokens else 0.
    codebook = {"size": n_embed, "active_codes": active, "dead_codes": n_embed - active,
        "live_fraction": active / n_embed, "dead_fraction": 1 - active / n_embed,
        "perplexity": perplexity, "histogram": histogram.tolist(), "total_tokens": total_tokens,
        "minimum_active_codes": minimum_active_codes, "invalid_indices": invalid_indices,
        "scope": "fixed_validation_panel"}
    summary = gate["summary"]
    summary.update(edge_l1=float(np.mean([frame["edge_l1"] for frame in per_frame])),
        quantization_mse=float(np.mean([frame["quantization_mse"] for frame in per_frame])),
        temporal_reconstruction_error=float(np.mean([pair["temporal_reconstruction_error"] for pair in per_pair])),
        codebook_active_codes=active, codebook_perplexity=perplexity,
        codebook_usage=codebook["live_fraction"], flash_frames=sum(frame["recorded_flash"] for frame in per_frame))
    checks = {"visual_quality": gate["passed"], "all_frames_finite": summary["nonfinite_frames"] == 0,
        "all_frames_in_range": summary["out_of_range_frames"] == 0,
        "quantization_finite": quantization_finite, "valid_code_indices": invalid_indices == 0,
        "codebook_broad_usage": active >= minimum_active_codes}
    specification = {"pairs": [list(pair) for pair in panel], "split": "val", "seed": seed,
        "model_config": configuration, "thresholds": limits, "minimum_active_codes": minimum_active_codes,
        "source_content_sha256": source_hash.hexdigest(), "render_precision": "fp32", "tf32": False,
        "metric_version": "vqgan_codec_gaussian_ssim_v1"}
    specification["fingerprint"] = hashlib.sha256(json.dumps(specification, sort_keys=True,
        separators=(",", ":"), default=str).encode()).hexdigest()
    flash = [frame for frame in per_frame if frame["recorded_flash"]]
    report = {**gate, "passed": all(checks.values()), "visual_passed": gate["passed"],
        "checks": checks, "failure_reasons": [name for name, passed in checks.items() if not passed],
        "codebook": codebook, "per_frame": per_frame, "per_pair": per_pair, "panel": specification,
        "fingerprint": specification["fingerprint"], "step": int(step), "pairs": len(panel),
        "seconds": time.perf_counter() - started, "source_held_out": True,
        "kind": "vqgan_codec_reconstruction", "source_input_allowed_for_codec": True,
        "world_rollout": False, "render_precision": "fp32",
        "score": summary["mae_mean"] + .5 * (1 - summary["ssim_mean"]),
        "counts": {"pairs": len(panel), "frames": len(per_frame), "tokens": total_tokens,
                   "shot_pairs": sum(pair["shot_transition"] for pair in per_pair), "flash_frames": len(flash)},
        "flash_summary": summarize_visual_quality(flash, limits)["summary"] if flash else None,
        "scope": "Held-out source-image codec reconstruction; not autonomous world or five-minute qualification"}
    source = torch.stack([image[0] for image in images])
    reconstructed = torch.stack([image[1] for image in images])
    preview = {"source": _uint8(source), "reconstruction": _uint8(reconstructed),
        "difference": _uint8((reconstructed - source).abs()),
        "metadata": {"kind": "vqgan_codec_reconstruction", "step": int(step), "every": 500,
            "generated_at": datetime.now(timezone.utc).isoformat(), "source_held_out": True,
            "source_input_allowed_for_codec": True, "world_rollout": False, "split": "val",
            "reconstruction_finite": bool(torch.isfinite(reconstructed).all()),
            "reconstruction_in_range": bool(torch.isfinite(reconstructed).all()
                and (reconstructed >= 0).all() and (reconstructed <= 1).all()),
            "panel_fingerprint": specification["fingerprint"], "difference_display_gain": 4,
            "caption": "Held-out SOURCE input / VQGAN reconstruction / absolute RGB difference x4; codec only"}}
    return report, preview


def _replace(temporary, destination):
    for attempt in range(40):
        try:
            os.replace(temporary, destination)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.05)


def save_vqgan_preview(preview, path, *, step=None):
    """Write 512x430: first 4 source / reconstruction / absolute-difference rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(preview["metadata"])
    if step is not None:
        metadata["step"] = int(step)
    canvas = Image.new("RGB", (512, 430), (14, 22, 31))
    draw = ImageDraw.Draw(canvas)
    gain = float(metadata.get("difference_display_gain", 4))
    invalid = not metadata.get("reconstruction_finite", True) or not metadata.get("reconstruction_in_range", True)
    labels = (f"SOURCE input | held-out | step {metadata['step']}",
              "VQGAN RECONSTRUCTION" + (" | INVALID" if invalid else "") + " | source codec input",
              f"ABSOLUTE DIFFERENCE x{gain:g}")
    for row, (key, label) in enumerate(zip(("source", "reconstruction", "difference"), labels)):
        top = row * 142
        draw.text((4, top + 3), label, fill=(235, 239, 245))
        for column, frame in enumerate(preview[key][:4]):
            pixels = np.asarray(frame, dtype=np.uint8)
            if key == "difference":
                pixels = np.clip(pixels.astype(np.float32) * gain, 0, 255).astype(np.uint8)
            image = Image.fromarray(pixels).resize((128, 128), Image.Resampling.NEAREST)
            canvas.paste(image, (column * 128, top + 16))
    temporary = path.with_name(path.stem + f".{os.getpid()}.tmp.png")
    canvas.save(temporary, format="PNG")
    _replace(temporary, path)
    metadata.update(path=str(path), image_size=[512, 430])
    sidecar = path.with_suffix(".json")
    temporary_json = sidecar.with_name(sidecar.name + f".{os.getpid()}.tmp")
    temporary_json.write_text(json.dumps(metadata, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    _replace(temporary_json, sidecar)
    return metadata
