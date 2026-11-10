"""Held-out renderer checks at Source poses, independent of world dynamics."""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .engine import STATE_DIM
from .staged_validation import _data, _image_rows, _image_summary, _isolated_validation, _panel
from .training_pipeline import atomic_json


RENDERER_THRESHOLDS = {'rgb_ssim_min': .90, 'rgb_mae_max': .03}


def _validate(model, data, context, count, batch_size, seed):
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
           for value in (context, count, batch_size)):
        raise ValueError('Renderer context/count/batch size must be positive integers')
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError('Renderer seed must be an integer')
    if any(parameter.is_floating_point() and parameter.dtype != torch.float32
           for parameter in model.parameters()):
        raise ValueError('Renderer component validation requires FP32 model parameters')
    if not hasattr(model, 'flow_renderer'):
        raise ValueError('Renderer component validation requires the full Flow renderer')
    device = next(model.parameters()).device
    data = _data(data, 'navigation', device, None)
    panel = _panel(data, context, 1, count, seed)
    if not panel:
        raise ValueError('No held-out navigation windows contain the requested target pose')
    return device, data, panel


def _queries(model, data, panel, context, batch_size, seed):
    """Render target coordinates directly; RGB/context never enter a network."""
    device = next(model.parameters()).device
    generated_rows, base_rows, identities, examples = [], [], [], []
    with _isolated_validation(model, data, seed):
        for first in range(0, len(panel), batch_size):
            pairs = panel[first:first+batch_size]
            batch = {name: value.to(device) if torch.is_tensor(value) else value
                     for name, value in data.batch(len(pairs), context=context, horizon=1, indices=pairs).items()}
            state = batch['target_states'][:, 0].float()
            target = batch['target_frames'][:, 0].float()
            mask, background = batch['target_object_masks'][:, 0], batch['target_background_frames'][:, 0].float()
            if state.shape != (len(pairs), STATE_DIM) or not bool(torch.isfinite(state).all()):
                raise ValueError('Expected finite Source target poses, not predicted coordinates')
            maps = model.encode_map(batch['map'])
            base = model.render_base(state, state.new_zeros(len(state), 128), maps).float()
            prediction = model.flow_renderer.sample(base, state, map_condition=batch['map']).float()
            if prediction.shape != target.shape or base.shape != target.shape:
                raise ValueError('Generated/Base/Source renderer RGB dimensions differ')
            generated_rows.extend(_image_rows(prediction, target, mask, background))
            base_rows.extend(_image_rows(base, target, mask, background))
            for row, (episode, start) in enumerate(pairs):
                identities.append({'episode_index': int(episode), 'episode': Path(data.episodes[episode]['path']).name,
                    'start': int(start), 'source_frame': int(start+context), 'split': data.split})
                if len(examples) < 4:
                    examples.append(tuple(value[row].detach().float().cpu().clone() for value in (base, target, prediction)))
    return generated_rows, base_rows, identities, examples


def evaluate_renderer_component(model, data, *, context=2, count=32, batch_size=4,
                                seed=173, output_dir=None, step=0):
    """Measure only full generated RGB at held-out ground-truth coordinates.

    No transition, visual encoder/predictor or EMA is called. The zeros128D
    appearance input matches the explicitly isolated renderer training phase.
    """
    _, data, panel = _validate(model, data, context, count, batch_size, seed)
    generated, base, identities, _ = _queries(model, data, panel, context, batch_size, seed)
    metrics = _image_summary(generated)
    metrics.update({'base_'+name: value for name, value in _image_summary(base).items()})
    failures = []
    if not metrics['rgb_finite']:
        failures.append('Generated RGB is non-finite, out of range or empty')
    if not metrics['base_rgb_finite']:
        failures.append('Base RGB is non-finite, out of range or empty')
    if metrics['rgb_ssim'] is None or not np.isfinite(metrics['rgb_ssim']) or metrics['rgb_ssim'] < .90:
        failures.append('rgb_ssim below .90 or unavailable')
    if metrics['rgb_mae'] is None or not np.isfinite(metrics['rgb_mae']) or metrics['rgb_mae'] > .03:
        failures.append('rgb_mae above .03 or unavailable')
    path = Path(output_dir)/'renderer_component_validation.json' if output_dir is not None else None
    report = {'version': 'renderer_component_source_gt_v1', 'stage': 'navigation',
        'component_phase': 'renderer', 'coordinate_source': 'source_gt', 'step': int(step),
        'passed': not failures, 'metrics': metrics, 'thresholds': dict(RENDERER_THRESHOLDS),
        'failures': failures, 'report_path': str(path.resolve()) if path is not None else None,
        'source_held_out': True, 'split': data.split, 'renderer_only': True, 'world_rollout': False,
        'source_gt_state_is_input': True, 'source_rgb_is_model_input': False,
        'future_source_feedback': False, 'engine_corrections': 0, 'transitions_evaluated': 0,
        'context': context, 'horizon': 1, 'requested_count': count, 'actual_windows': len(panel),
        'render_precision': 'fp32_tf32_disabled', 'seed': seed, 'panel': [list(pair) for pair in panel],
        'per_frame': [{**identity, **image, **{'base_'+key: value for key, value in coarse.items()}}
                      for identity, image, coarse in zip(identities, generated, base)],
        'scope': 'Renderer at held-out Source navigation poses; dynamics and gameplay are not evaluated'}
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, report)
    return report


def renderer_component_preview(model, data, step, *, context=2, seed=173):
    """Four direct GT-pose image queries, formatted for the existing preview UI."""
    _, data, panel = _validate(model, data, context, 4, 4, seed)
    _, _, identities, examples = _queries(model, data, panel, context, 4, seed)
    # Small fixture pools may contain fewer than four disjoint clips. The
    # repeated display columns are disclosed; no extra validation is claimed.
    displayed = [examples[index % len(examples)] for index in range(4)]
    base, target, generated = (torch.stack([example[index] for example in displayed]) for index in range(3))
    def rgb(value):
        return value.nan_to_num().clamp(0, 1).mul(255).round().byte().permute(0, 2, 3, 1).numpy()
    return {'context': rgb(base), 'source': rgb(target), 'generated': rgb(generated),
        'generated_float': generated,
        'metadata': {'step': int(step), 'every': 10, 'context': 0, 'horizon': 0,
            'query_sampling_context': context, 'generated_at': datetime.now(timezone.utc).isoformat(),
            'source_held_out': True, 'split': data.split, 'kind': 'static_scene_query',
            'component_phase': 'renderer', 'staged_stage': 'navigation', 'renderer_only': True,
            'world_rollout': False, 'coordinate_source': 'source_gt', 'source_gt_state_is_input': True,
            'source_rgb_is_model_input': False, 'future_source_feedback': False,
            'generated_finite': bool(torch.isfinite(generated).all()), 'unique_windows': len(examples),
            'display_windows': [identities[index % len(identities)] for index in range(4)],
            'caption': 'Held-out navigation Source poses; renderer only; no world transition',
            'row_labels': {'context': 'BASE: held-out GT navigation coordinates',
                'source': 'SOURCE: held-out navigation image targets',
                'generated': 'MODEL: renderer at the same GT pose; dynamics not evaluated'}}}
