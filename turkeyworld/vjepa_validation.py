"""Held-out autoregressive V-JEPA image measurements and reproducible previews.

Only the initial context RGB/state and the prerecorded future controls enter
the model. Every future state and visual memory is predicted, including for
the complete configured flow sampler. Future source RGB is a measurement
target; it is never encoded or used to repair the generated trajectory.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from .visual_quality import _thresholds, image_quality_metrics, summarize_visual_quality


def _replace(temporary, destination):
    for attempt in range(40):
        try:
            os.replace(temporary, destination)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.05)


def _uint8(images):
    return (images.detach().float().nan_to_num(nan=0., posinf=1., neginf=0.)
            .clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy())


def _panel_specification(model, data, panel, context, horizon, seed, limits):
    configuration = getattr(model, "config", None)
    configuration = (asdict(configuration) if is_dataclass(configuration)
                     else dict(vars(configuration)) if configuration is not None else {})
    content = hashlib.sha256()
    windows = []
    for episode_index, start in panel:
        episode = data.episodes[episode_index]
        for name, value in (("frames", episode["frames"][start:start + context + horizon]),
                            ("states", episode["states"][start:start + context + horizon]),
                            ("actions", episode["actions"][start + context - 1:start + context - 1 + horizon]),
                            ("map", episode["map"])):
            array = np.ascontiguousarray(value)
            content.update(name.encode())
            content.update(str(array.dtype).encode())
            content.update(str(array.shape).encode())
            content.update(array.tobytes())
        windows.append({"episode_index": episode_index, "start": start,
                        "episode": Path(episode["path"]).name})
    flow = getattr(model, "flow_renderer", None)
    noise = getattr(flow, "fixed_noise", None)
    specification = {"windows": windows, "context": context, "horizon": horizon,
                     "split": data.split, "seed": seed, "thresholds": limits,
                     "model_config": configuration, "sampling_steps": getattr(flow, "steps", None),
                     "flow_noise_sha256": (hashlib.sha256(noise.detach().cpu().float().numpy().tobytes()).hexdigest()
                                           if noise is not None else None),
                     "source_content_sha256": content.hexdigest(),
                     "render_precision": "fp32", "tf32": False,
                     "metric_version": "jepa-free-rollout-gaussian-ssim-v1"}
    encoded = json.dumps(specification, sort_keys=True, separators=(",", ":"), default=str).encode()
    return {**specification, "fingerprint": hashlib.sha256(encoded).hexdigest()}


def evaluate_autoregressive(model, data, panel=None, *, count=64, context=2, horizon=4,
                            batch_size=1, device=None, thresholds=None, seed=73, step=0,
                            preview_every=500):
    """Return (JSON-safe report, first-window CPU preview arrays/metadata).

    The default64 disjoint held-out windows produce256 future-image metrics.
    Batching1/2 windows limits memory use. Training mode flags, RNG states,
    global matmul precision/TF32 flags and the data RNG are restored afterward.
    Source initialization uses all context frames/states exactly once/window;
    subsequent transitions receive only predicted state/memory plus actions.
    """
    if min(count, context, horizon, batch_size) < 1:
        raise ValueError("Panel/context/horizon/batch counts must be positive")
    if getattr(data, "split", None) not in {"val", "test"}:
        raise ValueError("Autoregressive qualification requires an explicit held-out val/test split")
    device = torch.device(device) if device is not None else next(model.parameters()).device
    if any(parameter.is_floating_point() and parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("Strict FP32 validation requires FP32 model parameters")
    limits = _thresholds(thresholds)
    panel = data.fixed_panel(count=count, context=context, horizon=horizon, seed=seed) if panel is None else panel
    pairs = np.asarray(panel)
    if (pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs)
            or not np.issubdtype(pairs.dtype, np.integer)):
        raise ValueError("Panel must contain nonempty integer (episode,start) pairs")
    panel = [(int(episode), int(start)) for episode, start in pairs]
    if len(set(panel)) != len(panel):
        raise ValueError("Validation panel windows must be unique")
    for episode, start in panel:
        if (not 0 <= episode < len(data.episodes) or start < 0
                or start + context + horizon > len(data.episodes[episode]["frames"])):
            raise ValueError("Validation panel crosses an episode boundary")
    specification = _panel_specification(model, data, panel, context, horizon, seed, limits)
    modes = [(module, module.training) for module in model.modules()]
    precision = torch.get_float32_matmul_precision()
    cuda_tf32, cudnn_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    data_rng = getattr(data, "rng", None)
    data_rng_state = copy.deepcopy(data_rng.bit_generator.state) if isinstance(data_rng, np.random.Generator) else None
    cuda_devices = ([device.index if device.index is not None else torch.cuda.current_device()]
                    if device.type == "cuda" else [])
    per_frame, preview = [], None
    started = time.perf_counter()
    try:
        model.eval()
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode(), \
                torch.autocast(device.type, enabled=False):
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            for first in range(0, len(panel), batch_size):
                chosen = panel[first:first + batch_size]
                batch = data.batch(len(chosen), context=context, horizon=horizon,
                                   rng=np.random.default_rng(seed), indices=chosen)
                context_rgb = batch["context_frames"].to(device=device, dtype=torch.float32)
                context_states = batch["states"][:, :context].to(device=device, dtype=torch.float32)
                actions = batch["actions"].to(device=device, dtype=torch.float32)
                source = batch["target_frames"].to(device=device, dtype=torch.float32)
                grid = batch["map"].to(device=device, dtype=torch.long)
                map_features = model.encode_map(grid)
                memory = model.initialize_from_image(context_rgb, context_states, map_features)
                state = context_states[:, -1].clone()
                generated = []
                invalid = not bool(torch.isfinite(memory).all())
                for offset in range(horizon):
                    if not invalid:
                        result = model.transition(state, actions[:, offset], map_features, memory)
                        state, memory = result["state"], result["memory"]
                        invalid = not bool(torch.isfinite(state).all() and torch.isfinite(memory).all())
                    if invalid:
                        image = source[:, offset].new_full(source[:, offset].shape, float("nan"))
                    else:
                        image = model.render(state, result["features"], map_features).float()
                    if image.shape != source[:, offset].shape:
                        raise ValueError("Generated/source RGB dimensions differ")
                    generated.append(image)
                    measured = image_quality_metrics(image, source[:, offset])
                    values = {name: value.detach().cpu().tolist() for name, value in measured.items()}
                    for row, (episode_index, start) in enumerate(chosen):
                        episode = data.episodes[episode_index]
                        object_values = {}
                        if 'target_object_masks' in batch:
                            from .object_losses import region_error_metrics
                            object_values = {name: float(value) for name, value in region_error_metrics(
                                image[row:row+1], source[row:row+1, offset],
                                batch['target_object_masks'][row:row+1, offset]).items()}
                        per_frame.append({"window": first + row, "episode_index": episode_index,
                                          "episode": Path(episode["path"]).name,
                                          "start": start, "future_offset": offset + 1,
                                          "source_frame": start + context + offset,
                                          **{name: value[row] for name, value in values.items()}, **object_values})
                if preview is None:
                    future = torch.stack(generated, dim=1)[0]
                    preview = {"context": _uint8(context_rgb[0]), "source": _uint8(source[0]),
                               "generated": _uint8(future),
                               "generated_float": future.detach().float().cpu(),
                               "metadata": {"step": int(step), "every": int(preview_every),
                                            "context": context, "horizon": horizon,
                                            "generated_at": datetime.now(timezone.utc).isoformat(),
                                            "source_held_out": True, "split": data.split,
                                            "kind": "free_rollout", "window": specification["windows"][0],
                                            "generated_finite": bool(torch.isfinite(future).all()),
                                            "caption": "Context RGB only; predicted future state/memory; source future is target only"}}
                    object_frames = [frame for frame in per_frame if frame['window'] == 0 and frame.get('npc_present')]
                    if object_frames:
                        preview['metadata'].update(npc_rgb_mae=float(np.mean([frame['npc_rgb_mae'] for frame in object_frames])),
                            background_rgb_mae=float(np.mean([frame['background_rgb_mae'] for frame in object_frames])),
                            npc_pixel_fraction=float(np.mean([frame['npc_pixel_fraction'] for frame in object_frames])))
    finally:
        for module, training in modes:
            module.training = training
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = cuda_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)
        if data_rng_state is not None:
            data_rng.bit_generator.state = data_rng_state
    per_frame.sort(key=lambda frame: (frame["window"], frame["future_offset"]))
    gate = summarize_visual_quality(per_frame, limits)
    for frame, passed in zip(per_frame, gate.pop("frame_passed")):
        frame["passed"] = passed
    report = {**gate, "per_frame": per_frame, "panel": specification,
              "seconds": time.perf_counter() - started,
              "render_precision": "fp32", "kind": "free_rollout", "context": context,
              "horizon": horizon, "windows": len(panel), "step": int(step),
              "source_held_out": True, "future_source_feedback": False,
              "scope": "Held-out short autoregressive RGB; not a five-minute gameplay qualification"}
    return report, preview


def save_autoregressive_preview(preview, path, *, step=None, every=500, sr_model=None, sr_metadata=None):
    """Save three legacy rows, or five rows with prediction-only SR and bilinear."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(preview["metadata"])
    if step is not None:
        metadata["step"] = int(step)
    metadata["every"] = int(every)
    rows = [("context", f"CONTEXT input | step {metadata['step']} | held-out", preview["context"]),
            ("source", "SOURCE future: measurement targets only", preview["source"]),
            ("generated", "MODEL: free autoregressive future" +
             (" | INVALID" if not metadata.get("generated_finite", True) else ""), preview["generated"])]
    native = tuple(np.asarray(preview["generated"]).shape[1:3])
    metadata["native_resolution"] = list(native)
    cell, stride, canvas_size = 128, 142, (512, 430)
    if sr_model is not None:
        if native not in ((32, 32), (64, 64)):
            raise ValueError("Preview SR requires native32/64 generated RGB")
        trained = (sr_metadata or {}).get("training_steps", 0)
        if not isinstance(trained, (int, float)) or isinstance(trained, bool) or not np.isfinite(trained) or trained < 1:
            raise ValueError("Preview SR requires positive training_steps")
        for key, expected in (("input_size", native[0]), ("output_size", native[0]*4)):
            if key in (sr_metadata or {}) and sr_metadata[key] != expected:
                raise ValueError("Preview SR checkpoint resolution differs from generated RGB")
        generated = preview.get("generated_float")
        if generated is None:
            generated = torch.as_tensor(np.asarray(preview["generated"])).permute(0, 3, 1, 2).float()/255
        else:
            generated = generated.detach().float()
        if generated.ndim != 4 or tuple(generated.shape[1:]) != (3, *native):
            raise ValueError("Preview generated_float must contain matching native predictions")
        parameter = next(sr_model.parameters(), None)
        device = parameter.device if parameter is not None else torch.device('cpu')
        dtype = parameter.dtype if parameter is not None else torch.float32
        with torch.inference_mode():
            sr = sr_model(generated[:4].to(device=device, dtype=dtype)).float()
            middle, final = native[0]*4, native[0]*16
            if tuple(sr.shape[1:]) != (3, middle, middle) or not bool(torch.isfinite(sr).all()):
                raise ValueError("Preview SR must produce finite RGB at exactly x4 native size")
            enlarged = torch.nn.functional.interpolate(sr, size=(final, final), mode='bilinear', align_corners=False)
        rows.extend([("sr", f"LEARNED SR x4: model predictions only | {middle}x{middle}", _uint8(sr)),
                     ("bilinear", f"BILINEAR x4: same SR row | {final}x{final}", _uint8(enlarged))])
        cell, stride, canvas_size = final, final+24, (final*4, (final+24)*5+4)
        metadata.update(sr_resolution=[middle, middle], upscaled_resolution=[final, final],
            sr_training_steps=int(trained), sr_source='generated_free_rollout_rgb',
            future_source_feedback=False, sr_world_feedback=False,
            upscale_stages=[{'method': 'neural_x4', 'input_resolution': list(native),
                             'output_resolution': [middle, middle], 'scale': 4},
                            {'method': 'bilinear', 'input_resolution': [middle, middle],
                             'output_resolution': [final, final], 'scale': 4, 'align_corners': False}])
    canvas = Image.new("RGB", canvas_size, (14, 22, 31))
    draw = ImageDraw.Draw(canvas)
    row_metadata = []
    for row, (key, label, frames) in enumerate(rows):
        top = row*stride
        draw.text((4, top + 3), label, fill=(235, 239, 245))
        actual = list(np.asarray(frames).shape[1:3])
        row_metadata.append({'key': key, 'label': label, 'native_resolution': actual,
                             'display_resolution': [cell, cell],
                             'display_resize': 'none' if actual == [cell, cell] else 'nearest'})
        for column, frame in enumerate(frames[:4]):
            image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
            if image.size != (cell, cell):
                image = image.resize((cell, cell), Image.Resampling.NEAREST)
            canvas.paste(image, (column*cell, top+16))
    temporary = path.with_name(path.stem + f".{os.getpid()}.tmp.png")
    canvas.save(temporary, format="PNG")
    _replace(temporary, path)
    metadata.update(image_size=list(canvas_size), rows=row_metadata, row_count=len(rows),
                    path=str(path), caption=preview["metadata"]["caption"])
    sidecar = path.with_suffix(".json")
    temporary_json = sidecar.with_name(sidecar.name + f".{os.getpid()}.tmp")
    temporary_json.write_text(json.dumps(metadata, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    _replace(temporary_json, sidecar)
    return metadata
