"""Fixed-pose validation of final generated RGB against held-out source RGB.

This measures rendering, not free-running dynamics or five-minute gameplay.
Only the map and recorded numerical state enter the model. The source image is
used afterward as a measurement target. Conditional/flow-training losses and
the intermediate patch decoder are never substituted for final generated RGB.
"""
from __future__ import annotations

from .engine import GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS, STATE_DIM

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

DEFAULT_THRESHOLDS = {"psnr": 35., "ssim": .98, "mae": 2/255,
                      "required_fraction": .95}
METRIC_VERSION = 'gaussian_ssim_v1'


@contextmanager
def strict_render_precision():
    """Temporarily use full FP32 matmuls/convolutions, restoring caller flags."""
    policy = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        # The policy setter and legacy matmul flag are coupled. Restore the
        # exact high/medium/highest policy last, including a saved 'medium'.
        torch.set_float32_matmul_precision(policy)


def _thresholds(overrides=None):
    values = dict(DEFAULT_THRESHOLDS)
    if overrides:
        unknown = set(overrides)-set(values)
        if unknown:
            raise ValueError(f"Unknown visual thresholds: {sorted(unknown)}")
        values.update({key: float(value) for key, value in overrides.items()})
    if (not all(math.isfinite(value) for value in values.values())
            or not -1 <= values['ssim'] <= 1 or values['mae'] < 0
            or not 0 <= values['required_fraction'] <= 1):
        raise ValueError("Visual thresholds must be finite and in their valid ranges")
    return values


def _episode_ranges(data):
    ranges = getattr(data, 'episode_ranges', None)
    if ranges is None:
        # Compatibility for older fixed-length MemoryData instances only.
        files = int(getattr(data, 'files', 1))
        if files < 1 or data.count % files:
            raise ValueError("Exact episode_ranges are required for unequal episode lengths")
        length = data.count//files
        ranges = [{'path': f'episode_{index}', 'start': index*length,
                   'stop': (index+1)*length} for index in range(files)]
    result = []
    cursor = 0
    for item in ranges:
        start, stop = int(item['start']), int(item['stop'])
        if start != cursor or stop <= start or stop > data.count:
            raise ValueError("episode_ranges must partition data in contiguous file order")
        result.append({'path': str(item['path']), 'start': start, 'stop': stop})
        cursor = stop
    if not result or cursor != data.count:
        raise ValueError("episode_ranges do not cover the complete validation data")
    return result


def _entity_scores(states):
    """A fixed GT-state coverage proxy; it never inspects predictions or RGB."""
    state = np.asarray(states)
    enemies = state[:, GLOBAL_DIM:-2].reshape(-1, ENTITY_SLOTS, ENTITY_DIM)
    delta = (enemies[:, :, :2]-state[:, None, :2])*16
    forward = delta[:, :, 0]*state[:, None, 3]+delta[:, :, 1]*state[:, None, 2]
    lateral = -delta[:, :, 0]*state[:, None, 2]+delta[:, :, 1]*state[:, None, 3]
    visible = (forward > .12) & (np.abs(lateral) <= forward*math.tan(math.radians(34)))
    alive = enemies[:, :, 5] >= .5
    proximity = (visible/(1+np.linalg.norm(delta, axis=-1))).sum(-1)
    return ((visible & alive).sum(-1) + 1.5*(visible & ~alive).sum(-1)
            + .5*proximity + .5*(enemies[:, :, 7] > 0).any(-1)
            + .5*(state[:, 7] > 0))


def select_validation_indices(data, count=256, seed=73):
    """Balance held-out files, then alternate uniform and entity-rich time bins.

    The seed and selected indices are persisted by the caller. Disjoint time
    bins prevent duplicates and prediction-independent selection prevents
    repeatedly replacing difficult examples with easier ones during training.
    """
    if count < 1 or data.count < 1:
        raise ValueError("Validation panel and data must be nonempty")
    ranges = _episode_ranges(data)
    count = min(int(count), int(data.count))
    rng = np.random.default_rng(seed)
    quotas = np.zeros(len(ranges), dtype=int)
    capacities = np.array([item['stop']-item['start'] for item in ranges])
    if count < len(ranges):
        chosen = np.floor((np.arange(count)+.5)*len(ranges)/count).astype(int)
        quotas[chosen] = 1
    else:
        remaining = count
        while remaining:
            available = np.flatnonzero(quotas < capacities)
            take = min(remaining, len(available))
            if take < len(available):
                available = available[np.floor((np.arange(take)+.5)*len(available)/take).astype(int)]
            quotas[available[:take]] += 1
            remaining -= take
    selected = []
    for item, quota in zip(ranges, quotas):
        if not quota:
            continue
        bins = np.array_split(np.arange(item['start'], item['stop']), quota)
        for number, candidates in enumerate(bins):
            if number % 2 == 0:
                chosen = int(rng.choice(candidates))
            else:
                scores = _entity_scores(data.data['new'][candidates])
                tied = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0, atol=1e-9))
                chosen = int(candidates[int(rng.choice(tied))])
            selected.append(chosen)
    return np.array(sorted(selected), dtype=np.int64)


def image_quality_metrics(prediction, target):
    """Per-image PSNR/MAE and standard Gaussian-window RGB SSIM.

    Inputs are [B,3,H,W] RGB in [0,1]. SSIM uses an 11x11 Gaussian, sigma1.5,
    valid convolution, channel/pixel averaging and C1=.01², C2=.03². Metrics
    use double precision, independently of neural inference autocast. Exact
    identity is represented by finite PSNR120dB for JSON/logging stability.
    """
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 3:
        raise ValueError("Expected matching RGB tensors [B,3,H,W]")
    if min(prediction.shape[-2:]) < 11:
        raise ValueError("Gaussian SSIM requires image dimensions of at least11")
    if not bool(torch.isfinite(target).all()) or bool(((target < 0) | (target > 1)).any()):
        raise ValueError("Held-out source RGB must be finite and in [0,1]")
    finite = torch.isfinite(prediction).flatten(1).all(-1)
    in_range = finite & ((prediction >= 0) & (prediction <= 1)).flatten(1).all(-1)
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        first = torch.where(torch.isfinite(prediction), prediction, torch.zeros_like(prediction)).double()
        second = target.double()
        difference = first-second
        mse = difference.square().flatten(1).mean(-1)
        mae = difference.abs().flatten(1).mean(-1)
        psnr = -10*torch.log10(mse.clamp_min(1e-12))
        axis = torch.arange(11, device=first.device, dtype=first.dtype)-5
        gaussian = torch.exp(-axis.square()/(2*1.5**2))
        gaussian /= gaussian.sum()
        window = (gaussian[:, None]*gaussian[None]).expand(3, 1, 11, 11).contiguous()
        mu1, mu2 = F.conv2d(first, window, groups=3), F.conv2d(second, window, groups=3)
        var1 = (F.conv2d(first.square(), window, groups=3)-mu1.square()).clamp_min(0)
        var2 = (F.conv2d(second.square(), window, groups=3)-mu2.square()).clamp_min(0)
        covariance = F.conv2d(first*second, window, groups=3)-mu1*mu2
        c1, c2 = .01**2, .03**2
        ssim = (((2*mu1*mu2+c1)*(2*covariance+c2)) /
                ((mu1.square()+mu2.square()+c1)*(var1+var2+c2)))
        ssim = ssim.flatten(1).mean(-1).clamp(-1, 1)
        # Invalid network output fails explicitly; no NaN/Infinity reaches JSON.
        psnr = torch.where(finite, psnr, torch.full_like(psnr, -120.))
        mae = torch.where(finite, mae, torch.ones_like(mae))
        ssim = torch.where(finite, ssim, -torch.ones_like(ssim))
    return {'psnr': psnr, 'ssim': ssim, 'mae': mae, 'finite': finite, 'in_range': in_range}


def summarize_visual_quality(per_frame, thresholds=None):
    """Apply all three per-frame criteria and the panel-level mean criteria."""
    limits = _thresholds(thresholds)
    if not per_frame:
        raise ValueError("Cannot qualify an empty validation panel")
    passed = []
    for frame in per_frame:
        passed.append(bool(frame.get('finite', True) and frame.get('in_range', True)
                           and frame['psnr'] >= limits['psnr'] and frame['ssim'] >= limits['ssim']
                           and frame['mae'] <= limits['mae']))
    psnr, ssim, mae = [np.array([frame[key] for frame in per_frame], dtype=float)
                       for key in ('psnr', 'ssim', 'mae')]
    required = math.ceil(limits['required_fraction']*len(per_frame)-1e-12)
    summary = {'frames': len(per_frame), 'passing_frames': sum(passed), 'required_frames': required,
               'pass_fraction': float(np.mean(passed)),
               'psnr_mean': float(psnr.mean()), 'psnr_p05': float(np.percentile(psnr, 5)), 'psnr_min': float(psnr.min()),
               'ssim_mean': float(ssim.mean()), 'ssim_p05': float(np.percentile(ssim, 5)), 'ssim_min': float(ssim.min()),
               'mae_mean': float(mae.mean()), 'mae_p95': float(np.percentile(mae, 95)), 'mae_max': float(mae.max()),
               'nonfinite_frames': sum(not frame.get('finite', True) for frame in per_frame),
               'out_of_range_frames': sum(not frame.get('in_range', True) for frame in per_frame)}
    qualified = bool(sum(passed) >= required and summary['ssim_mean'] >= limits['ssim']
                     and summary['mae_mean'] <= limits['mae'] and not summary['nonfinite_frames'])
    return {'passed': qualified, 'thresholds': limits, 'summary': summary, 'frame_passed': passed}


def _panel_spec(model, data, indices, thresholds, *, amp=False):
    config = getattr(model, 'config', None)
    config = asdict(config) if is_dataclass(config) else dict(vars(config)) if config is not None else {}
    grid = data.grid.detach().cpu().numpy() if torch.is_tensor(data.grid) else np.asarray(data.grid)
    content = hashlib.sha256()
    for value in (np.asarray(indices, dtype=np.int64), data.data['new'][indices], data.data['images'][indices], grid):
        array = np.ascontiguousarray(value)
        content.update(str(array.dtype).encode())
        content.update(str(array.shape).encode())
        content.update(array.tobytes())
    flow = getattr(model, 'flow_renderer', None)
    noise_hash = None
    if flow is not None and hasattr(flow, 'fixed_noise'):
        noise_hash = hashlib.sha256(flow.fixed_noise.detach().cpu().float().numpy().tobytes()).hexdigest()
    specification = {'indices': [int(value) for value in indices], 'model_config': config,
                     'thresholds': thresholds, 'source_content_sha256': content.hexdigest(),
                     'metric_version': METRIC_VERSION,
                     'render_precision': 'bf16_autocast' if amp else 'fp32',
                     'float32_matmul_precision': torch.get_float32_matmul_precision() if amp else 'highest',
                     'tf32_matmul': bool(torch.backends.cuda.matmul.allow_tf32) if amp else False,
                     'tf32_cudnn': bool(torch.backends.cudnn.allow_tf32) if amp else False,
                     'inference_noise_sha256': noise_hash,
                     'sampling_steps': getattr(flow, 'steps', None),
                     'selection': 'Balanced episode files; disjoint seeded uniform/entity-rich time bins'}
    encoded = json.dumps(specification, sort_keys=True, separators=(',', ':'), default=str).encode()
    return {**specification, 'fingerprint': hashlib.sha256(encoded).hexdigest()}


def evaluate_visual_quality(model, data, indices, batch_size=8, device='cpu', amp=False, thresholds=None):
    """Measure actual runtime decoder output on an immutable held-out panel.

    FP32 rendering is the default. Teacher Flow Matching uses its complete
    configured12-step Heun sampler with the cached runtime noise field. Model
    mode flags and PyTorch RNG states are restored, including mixed train/eval
    submodule modes. No source pixels enter encode_map or model.render.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or not len(indices) or len(np.unique(indices)) != len(indices):
        raise ValueError("Validation indices must be a nonempty unique one-dimensional panel")
    if np.any(indices < 0) or np.any(indices >= data.count) or batch_size < 1:
        raise ValueError("Validation indices/batch size are outside their valid ranges")
    device = torch.device(device)
    if amp and device.type not in {'cpu', 'cuda'}:
        raise ValueError("Autocast is supported only on CPU/CUDA here")
    config = getattr(model, 'config', None)
    flow = getattr(model, 'flow_renderer', None)
    if (getattr(config, 'kind', None) == 'teacher'
            and getattr(config, 'renderer_mode', None) == 'flow_matching'
            and getattr(flow, 'steps', None) != 12):
        raise ValueError("Teacher visual qualification requires the complete12-step Heun sampler")
    limits = _thresholds(thresholds)
    panel = _panel_spec(model, data, indices, limits, amp=amp)
    modes = [(module, module.training) for module in model.modules()]
    ranges = _episode_ranges(data)
    ends = np.array([item['stop'] for item in ranges])
    per_frame = []
    started = time.perf_counter()
    rng_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        model.eval()
        precision = nullcontext() if amp else strict_render_precision()
        with precision, torch.random.fork_rng(devices=rng_devices), torch.inference_mode():
            for first in range(0, len(indices), batch_size):
                chosen = indices[first:first+batch_size]
                batch = data.batch(len(chosen), np.random.default_rng(0), indices=chosen)
                state, source = batch['new'].to(device), batch['images'].to(device)
                grid = batch['map'].to(device)
                context = torch.autocast(device.type, dtype=torch.bfloat16) if amp else nullcontext()
                with context:
                    features = model.encode_map(grid)
                    generated = model.render(state, state.new_zeros(len(chosen), 128), features)
                metrics = image_quality_metrics(generated.float(), source.float())
                values = {key: value.detach().cpu().tolist() for key, value in metrics.items()}
                for offset, index in enumerate(chosen):
                    episode = ranges[int(np.searchsorted(ends, index, side='right'))]
                    per_frame.append({'index': int(index), 'episode': episode['path'],
                                      'source_frame': int(index-episode['start']+1),
                                      **{key: values[key][offset] for key in values}})
    finally:
        for module, training in modes:
            module.training = training
    gate = summarize_visual_quality(per_frame, limits)
    for frame, passed in zip(per_frame, gate.pop('frame_passed')):
        frame['passed'] = passed
    return {**gate, 'per_frame': per_frame, 'panel': panel,
            'seconds': time.perf_counter()-started, 'render_precision': 'bf16_autocast' if amp else 'fp32',
            'scope': 'Held-out fixed-pose final-RGB rendering only; this does not qualify free dynamics or300-second gameplay',
            'ssim_definition': 'Gaussian11x11 sigma1.5; valid convolution; C1=.01^2 C2=.03^2; RGB channel/pixel mean'}
