"""Held-out, same-code comparisons for decoder-only RF3 adaptation.

Source RGB is a reconstruction target and a separate frozen-E/Q diagnostic
branch. It never conditions the RF endpoint. This is native32 rendering
validation, not super-resolution or autonomous gameplay qualification.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .decoder_rf_adaptation_objectives import (
    PROTOCOL, TRAINABLE_PREFIXES, decoder_rf_adaptation_protocol,
    decoder_rf_adaptation_loss, generate_decoder_training_codes)
from .renderer_enemy_validation import renderer_metrics
from .renderer_geometry_objectives import geometry_wall_terms, geometry_wall_protocol
from .renderer_student import _hash_tensors
from .renderer_wall_depth import source_wall_depth_labels
from .renderer_wall_supervision import source_wall_masks, wall_loss_terms
from .staged_validation import _isolated_validation


KIND = 'decoder_rf_adaptation_validation_v1'
PREVIEW_KIND = 'decoder_rf_adaptation_preview_v1'
BRANCHES = ('rf_old', 'rf_new', 'source_old', 'source_new')
PRESERVATION = dict(mae_tolerance=.002, ssim_tolerance=.01,
                    foreground_mae_tolerance=.002, background_mae_tolerance=.002,
                    wall_boundary_tolerance=.002, geometry_tolerance=.002)
IMPROVEMENT = dict(npc_foreground_improvement=.0005, visible_frames_min=8)


def _plain(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _identity(data):
    identity = _plain(deepcopy(getattr(data, 'identity', {})))
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return identity, digest


def _fixed_codec_state(codec):
    return {name: value for name, value in codec.state_dict().items()
            if not name.startswith(('decoder.', 'post_quant_conv.'))}


@contextmanager
def _isolated_decoder_validation(model, reference_codec, data):
    """Also preserve grad flags and private sampler ledgers, not just RNG."""
    if getattr(data, 'split', None) not in ('val', 'test'):
        raise ValueError('Decoder adaptation validation requires held-out val/test data')
    illegal = [n for n, p in model.named_parameters()
               if p.requires_grad and not n.startswith(TRAINABLE_PREFIXES)]
    if illegal:
        raise ValueError('Decoder validation requires frozen RF/Base/world/E/Q; only D may train')
    if any(p.requires_grad for p in reference_codec.parameters()):
        raise ValueError('Reference codec must be a frozen pre-adaptation clone')
    codec = model.flow_renderer.codec
    if (codec.get_config() != reference_codec.get_config()
            or next(codec.parameters()).device != next(reference_codec.parameters()).device
            or _hash_tensors(_fixed_codec_state(codec)) !=
               _hash_tensors(_fixed_codec_state(reference_codec))):
        raise ValueError('Old/new decoders must share exact frozen encoder, codebook and device')
    fixed_hash = _hash_tensors(_fixed_codec_state(codec))
    reference_hash = _hash_tensors(reference_codec.state_dict())
    flags = [(p, p.requires_grad) for p in model.parameters()]
    reference_modes = [(m, m.training) for m in reference_codec.modules()]
    sampler = (deepcopy(data.sampler_state()) if callable(getattr(data, 'sampler_state', None))
               else None)
    try:
        reference_codec.eval()
        with _isolated_validation(model, data, 173):
            yield reference_hash
    finally:
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
        for module, mode in reference_modes:
            module.training = mode
        if sampler is not None:
            data.restore_sampler(sampler)
        if _hash_tensors(reference_codec.state_dict()) != reference_hash:
            raise RuntimeError('Validation changed the frozen reference codec')
        if _hash_tensors(_fixed_codec_state(codec)) != fixed_hash:
            raise RuntimeError('Validation changed the frozen encoder or quantizer')


class _ObjectiveMeans:
    def __init__(self):
        self.frames = 0
        self.loss = 0.
        self.sums = {}

    def add(self, loss, metrics, count):
        value = float(loss.detach())
        if not math.isfinite(value):
            raise FloatingPointError('Nonfinite decoder validation objective')
        self.loss += count * value
        self.frames += count

        def visit(value, path=()):
            if isinstance(value, dict):
                for key, item in value.items():
                    visit(item, (*path, key))
            elif type(value) in (int, float):
                if not math.isfinite(value):
                    raise FloatingPointError('Nonfinite decoder objective component')
                support = path[-1].endswith(('_images', '_pixels'))
                self.sums[path] = self.sums.get(path, 0.) + value * (1 if support else count)
        visit(metrics['branches'])

    def components(self):
        result = {}
        for path, value in self.sums.items():
            row = result
            for key in path[:-1]:
                row = row.setdefault(key, {})
            support = path[-1].endswith(('_images', '_pixels'))
            row[path[-1]] = int(value) if support else value / self.frames
        return result


def _collect(model, reference_codec, data, panel, batch_size, *, objective):
    parts = {key: [] for key in (*BRANCHES, 'source', 'mask', 'background',
                                'states', 'map', 'dynamic')}
    means = _ObjectiveMeans()
    for first in range(0, len(panel), batch_size):
        ids = panel[first:first + batch_size]
        batch = data.batch(len(ids), indices=ids)
        prepared = generate_decoder_training_codes(model, batch)
        rf_codes, source_codes, conditions = prepared
        codec = model.flow_renderer.codec
        predictions = dict(rf_old=reference_codec.decode(rf_codes),
            rf_new=codec.decode(rf_codes), source_old=reference_codec.decode(source_codes),
            source_new=codec.decode(source_codes))
        target = conditions['target']
        for name, value in predictions.items():
            if value.shape != target.shape or not bool(torch.isfinite(value).all()):
                raise FloatingPointError('Nonfinite or wrongly shaped decoder validation RGB')
            parts[name].append(value.detach().cpu())
        for name, key in (('source', 'target'), ('mask', 'object_mask'),
                          ('background', 'background'), ('states', 'state'),
                          ('map', 'map'), ('dynamic', 'dynamic')):
            parts[name].append(conditions[key].detach().cpu())
        if objective:
            loss, metrics = decoder_rf_adaptation_loss(model, prepared=prepared)
            means.add(loss, metrics, len(ids))
    arrays = {key: torch.cat(value) for key, value in parts.items()}
    arrays['categories'] = [str(data.rows[int(index)]['category']) for index in panel]
    arrays['row_indices'] = [int(index) for index in panel]
    return arrays, means


def _models(arrays):
    labels = source_wall_masks(arrays['map'], arrays['states'], arrays['mask'])
    result = {}
    for branch in BRANCHES:
        metrics = renderer_metrics(arrays[branch], arrays['source'], arrays['mask'],
            arrays['background'], arrays['categories'], arrays['dynamic'])
        wall = _plain(wall_loss_terms(arrays[branch], arrays['source'], labels))
        summary = dict(metrics['overall'], wall_boundary_l1=(wall['boundary_l1']
            if wall['boundary_images'] else None), wall_boundary_frames=wall['boundary_images'])
        result[branch] = dict(summary=summary, overall=deepcopy(summary),
            cohorts=metrics['categories'], visible_enemy=metrics['visible_enemy'], wall=wall,
            static_zero_features=metrics['static_zero_features'],
            dynamic_past_prior_features=metrics['dynamic_past_prior_features'])
    return result


def _preservation(current, old):
    failures, unmeasured = [], []
    for count in ('frames', 'foreground_frames', 'background_frames', 'wall_boundary_frames'):
        if current.get(count) != old.get(count):
            failures.append(count + '_support_changed')
    for field, tolerance in (('mae_mean', PRESERVATION['mae_tolerance']),
            ('foreground_mae', PRESERVATION['foreground_mae_tolerance']),
            ('background_mae', PRESERVATION['background_mae_tolerance']),
            ('wall_boundary_l1', PRESERVATION['wall_boundary_tolerance'])):
        a, b = current.get(field), old.get(field)
        if a is None or b is None:
            unmeasured.append(field)
        elif a > b + tolerance:
            failures.append(field + '_regressed')
    if current['ssim_mean'] < old['ssim_mean'] - PRESERVATION['ssim_tolerance']:
        failures.append('ssim_regressed')
    return dict(passed=not failures, failure_reasons=failures,
                unmeasured=unmeasured, criteria=deepcopy(PRESERVATION))


def _geometry(arrays):
    labels = source_wall_depth_labels(arrays['map'], arrays['states'], arrays['mask'])
    return {name: dict(bands=_plain(geometry_wall_terms(arrays[name], arrays['source'], labels)))
            for name in BRANCHES}


def _geometry_preservation(current, old):
    failures, measured, unmeasured = [], [], []
    for band in geometry_wall_protocol()['loss_bands']:
        for field in geometry_wall_protocol()['weights_per_head']:
            support = field.removesuffix('_l1') + '_images'
            a, b = current['bands'][band], old['bands'][band]
            label = band + '_' + field
            if a[support] != b[support]:
                failures.append(label + '_support_changed')
            elif not a[support]:
                unmeasured.append(label)
            else:
                measured.append(label)
                if a[field] > b[field] + PRESERVATION['geometry_tolerance']:
                    failures.append(label + '_regressed')
    return dict(passed=bool(measured) and not failures, failure_reasons=failures,
                measured=measured, unmeasured=unmeasured,
                absolute_loss_tolerance=PRESERVATION['geometry_tolerance'])


def evaluate_decoder_adaptation(model, reference_codec, data, panel, *, step,
                                batch_size=8, geometry_panel=None, baseline=None):
    """Evaluate both decoders on identical detached RF-Q and Source-E/Q codes."""
    panel = [int(index) for index in panel]
    geometry_panel = None if geometry_panel is None else [int(index) for index in geometry_panel]
    if (type(step) is not int or step < 0 or type(batch_size) is not int or batch_size < 1
            or not panel or len(set(panel)) != len(panel)
            or geometry_panel is not None and (not geometry_panel or len(set(geometry_panel)) != len(geometry_panel))):
        raise ValueError('Decoder validation needs a nonempty unique panel, step>=0 and positive batch size')
    identity, identity_sha = _identity(data)
    panel_sha = data.panel_fingerprint(panel)
    geometry_sha = data.panel_fingerprint(geometry_panel) if geometry_panel is not None else None
    with _isolated_decoder_validation(model, reference_codec, data) as reference_hash:
        if baseline is not None:
            expected = dict(kind=KIND, protocol=PROTOCOL, panel_fingerprint=panel_sha,
                geometry_panel_fingerprint=geometry_sha, data_identity_sha256=identity_sha,
                reference_codec_sha256=reference_hash, solver_steps=3)
            if any(baseline.get(key) != value for key, value in expected.items()):
                raise ValueError('Decoder baseline requires identical data, panels, RF3 and reference codec')
        arrays, means = _collect(model, reference_codec, data, panel, batch_size, objective=True)
        models = _models(arrays)
        geometry = None
        if geometry_panel is not None:
            geometry_arrays, _ = _collect(model, reference_codec, data, geometry_panel,
                                          batch_size, objective=False)
            branches = _geometry(geometry_arrays)
            geometry = dict(frames=len(geometry_panel), panel_fingerprint=geometry_sha,
                protocol=geometry_wall_protocol(), models=branches,
                source_preservation=_geometry_preservation(branches['source_new'], branches['source_old']),
                rf_preservation=_geometry_preservation(branches['rf_new'], branches['rf_old']))
    source = _preservation(models['source_new']['summary'], models['source_old']['summary'])
    rf = _preservation(models['rf_new']['summary'], models['rf_old']['summary'])
    failures = list(rf['failure_reasons'])
    now, old = models['rf_new']['summary'], models['rf_old']['summary']
    if now['foreground_frames'] < IMPROVEMENT['visible_frames_min']:
        failures.append('insufficient_visible_NPC_frames')
    elif now['foreground_mae'] >= old['foreground_mae'] - IMPROVEMENT['npc_foreground_improvement']:
        failures.append('npc_foreground_not_improved')
    for cohort, current in models['rf_new']['cohorts'].items():
        if cohort.startswith('empty'):
            previous = models['rf_old']['cohorts'][cohort]
            if (current['mae_mean'] > previous['mae_mean'] + PRESERVATION['mae_tolerance']
                    or current['ssim_mean'] < previous['ssim_mean'] - PRESERVATION['ssim_tolerance']):
                failures.append(cohort + '_regressed')
    if geometry is not None:
        if not geometry['source_preservation']['passed']:
            source['failure_reasons'].append('same_panel_Source_code_geometry_regressed')
            source['passed'] = False
        if not geometry['rf_preservation']['passed']:
            failures.append('same_panel_RF_code_geometry_regressed')
    improvement = dict(validated_improvement=not failures, failure_reasons=failures,
                       criteria={**deepcopy(PRESERVATION), **IMPROVEMENT})
    quality = bool(now['ssim_mean'] > .9 and now['mae_mean'] < .03
                   and now['foreground_frames'] >= IMPROVEMENT['visible_frames_min']
                   and now['foreground_mae'] is not None and now['foreground_mae'] < .05)
    report = dict(kind=KIND, protocol=PROTOCOL, step=step, frames=len(panel), solver='euler', solver_steps=3,
        panel_fingerprint=panel_sha, geometry_panel_fingerprint=geometry_sha,
        data_identity=identity, data_identity_sha256=identity_sha,
        reference_codec_sha256=reference_hash, reference_codec_tensors_unchanged=True,
        encoder_quantizer_tensors_unchanged=True, source_sampler_unchanged=True,
        source_held_out=True, renderer_only=True, native_resolution=32,
        scope='Native32 GT-pose Source rendering; frozen RF3 and causal past-prior/zero features',
        Source_RGB_into_RF_conditioning=False, future_source_rgb_in_prior=False,
        future_source_pose_in_prior=False, source_geometry_and_masks_labels_only=True,
        sr_quality_evaluated=False, autonomous_rollout=False, world_rollout=False, gameplay_qualified=False,
        automatic_active_promotion=False, models=models, geometry=geometry,
        source_preservation=source, rf_improvement=improvement, val_loss=means.loss / means.frames,
        validation_objective=dict(protocol=PROTOCOL,
            input_kind='RF3_quantized_and_Source_quantized_latents', loss=means.loss / means.frames,
            frames=means.frames, decoder_examples=2 * means.frames,
            batch_size=batch_size, branch_weights=dict(rf_quantized=.5, source_quantized=.5),
            components=means.components(), weights=decoder_rf_adaptation_protocol()['weights'],
            aggregation='batch_image_weighted',
            aggregation_details='Actual batch objective mean weighted by Source image count; tail weighted by its size. Component means use the same weighting; support counts are summed.',
            source_held_out=True, matches_training_formula=True),
        passed=quality, quality_passed=quality,
        thresholds=dict(ssim_strict_min=.9, mae_strict_max=.03, npc_foreground_strict_max=.05,
                        visible_frames_min=IMPROVEMENT['visible_frames_min'], applies_to='rf_new native32'),
        experimental_selection=dict(validated_improvement=improvement['validated_improvement']
            and source['passed'], requires_Source_code_preservation=True))
    if baseline is not None:
        if any(models[name] != baseline['models'][name] for name in ('rf_old', 'source_old')):
            raise ValueError('Frozen RF/reference outputs changed from the pinned decoder baseline')
        report['baseline_step'] = baseline['step']
    arrays.update(panel_fingerprint=panel_sha, step=step)
    return report, arrays


def save_decoder_adaptation_preview(arrays, path, step):
    """Five native32 rows; enlargement is nearest-neighbor display only."""
    from PIL import Image, ImageDraw

    categories = arrays['categories']
    chosen = []
    for alternatives in (('large', 'near'), ('medium', 'partial', 'mid'),
                         ('distant', 'far'), ('empty_navigation', 'empty_hidden', 'empty')):
        index = next((i for name in alternatives for i, category in enumerate(categories)
                      if category == name and i not in chosen), None)
        if index is not None:
            chosen.append(index)
    chosen += [i for i in range(len(categories)) if i not in chosen][:4 - len(chosen)]
    rows = [('source', 'SOURCE32'), ('rf_old', 'OLD D | RF3 CODES'),
            ('rf_new', 'ADAPTED D | RF3 CODES'), ('source_old', 'OLD D | SOURCE E/Q'),
            ('source_new', 'ADAPTED D | SOURCE E/Q')]
    size, caption = 192, 25
    image = Image.new('RGB', (size * len(chosen), (size + caption) * len(rows)), (12, 17, 25))
    draw = ImageDraw.Draw(image)
    for row, (key, title) in enumerate(rows):
        for column, index in enumerate(chosen):
            pixels = arrays[key][index].detach().float().clamp(0, 1).mul(255).round().byte()
            frame = Image.fromarray(pixels.permute(1, 2, 0).cpu().numpy())
            image.paste(frame.resize((size, size), Image.Resampling.NEAREST),
                        (column * size, row * (size + caption) + caption))
            draw.text((column * size + 3, row * (size + caption) + 3),
                      f'{title} g{step} | {categories[index]}', fill='white')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.png')
    image.save(temporary)
    temporary.replace(path)
    return dict(kind=PREVIEW_KIND, protocol=PROTOCOL, step=step, path=str(path),
        rows=[key for key, _ in rows], row_count=5, columns=[categories[i] for i in chosen],
        row_indices=[arrays['row_indices'][i] for i in chosen],
        panel_fingerprint=arrays['panel_fingerprint'], native_resolution=32,
        source_held_out=True, Source_RGB_into_RF_conditioning=False,
        source_input_allowed_for_rf=False, sr_quality_evaluated=False,
        gameplay_qualified=False, enlargement='nearest_neighbor_display_only')
