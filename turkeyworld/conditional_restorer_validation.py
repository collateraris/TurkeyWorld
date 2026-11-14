"""Same-code held-out comparison of a frozen codec and its new restorer."""
from contextlib import contextmanager
from copy import deepcopy

import torch

from .conditional_restorer_objectives import (
    PROTOCOL, TRAINABLE_PREFIXES, conditional_restorer_protocol,
    generate_conditional_restorer_inputs, conditional_restorer_predictions,
    conditional_restorer_loss)
from .decoder_rf_adaptation_validation import (
    BRANCHES, PRESERVATION, IMPROVEMENT, _identity, _models, _preservation,
    _geometry, _geometry_preservation, _ObjectiveMeans,
    save_decoder_adaptation_preview)
from .renderer_geometry_objectives import geometry_wall_protocol
from .renderer_student import _hash_tensors
from .staged_validation import _isolated_validation


KIND = 'conditional_restorer_validation_v1'
PREVIEW_KIND = 'conditional_restorer_preview_v1'


@contextmanager
def _isolated_restorer_validation(model, reference_codec, data):
    if getattr(data, 'split', None) not in ('val', 'test'):
        raise ValueError('Conditional-restorer validation requires held-out val/test Source data')
    restorer = getattr(model.flow_renderer, 'conditional_restorer', None)
    if not isinstance(restorer, torch.nn.Module):
        raise ValueError('Conditional-restorer validation requires the explicit new module')
    if any(p.requires_grad and not n.startswith(TRAINABLE_PREFIXES)
           for n, p in model.named_parameters()):
        raise ValueError('Only the conditional restorer may train; full decoder/RF/E/Q must be frozen')
    if any(p.requires_grad for p in reference_codec.parameters()):
        raise ValueError('Conditional-restorer reference codec must be entirely frozen')
    codec = model.flow_renderer.codec
    reference_hash = _hash_tensors(reference_codec.state_dict())
    if (codec.get_config() != reference_codec.get_config()
            or next(codec.parameters()).device != next(reference_codec.parameters()).device
            or _hash_tensors(codec.state_dict()) != reference_hash):
        raise ValueError('Conditional restoration requires the exact frozen parent decoder/E/Q codec')
    # The whole pretrained D is covered, unlike decoder-adaptation validation.
    restorer_hash = _hash_tensors(restorer.state_dict())
    flags = [(p, p.requires_grad) for p in model.parameters()]
    reference_modes = [(module, module.training) for module in reference_codec.modules()]
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
        if (_hash_tensors(reference_codec.state_dict()) != reference_hash
                or _hash_tensors(codec.state_dict()) != reference_hash):
            raise RuntimeError('Validation changed the frozen parent codec, including D/post_quant_conv')
        if _hash_tensors(restorer.state_dict()) != restorer_hash:
            raise RuntimeError('Read-only validation changed conditional-restorer weights or buffers')


def _collect_restorer(model, reference_codec, data, panel, batch_size, *, objective):
    parts = {key: [] for key in (*BRANCHES, 'source', 'mask', 'background',
                                'states', 'map', 'dynamic')}
    means = _ObjectiveMeans()
    for first in range(0, len(panel), batch_size):
        ids = panel[first:first + batch_size]
        prepared = generate_conditional_restorer_inputs(model, data.batch(len(ids), indices=ids))
        predictions = conditional_restorer_predictions(model, prepared)
        # Use the independently frozen pre-phase reference for both old arms.
        predictions['rf_old'] = reference_codec.decode(prepared[0])
        predictions['source_old'] = reference_codec.decode(prepared[1])
        conditions = prepared[2]
        for name, value in predictions.items():
            if value.shape != conditions['target'].shape or not bool(torch.isfinite(value).all()):
                raise FloatingPointError('Nonfinite native32 conditional-restorer comparison')
            parts[name].append(value.detach().cpu())
        for name, key in (('source', 'target'), ('mask', 'object_mask'),
                          ('background', 'background'), ('states', 'state'),
                          ('map', 'map'), ('dynamic', 'dynamic')):
            parts[name].append(conditions[key].detach().cpu())
        if objective:
            loss, metrics = conditional_restorer_loss(model, prepared=prepared)
            means.add(loss, metrics, len(ids))
    arrays = {key: torch.cat(value) for key, value in parts.items()}
    arrays['categories'] = [str(data.rows[int(index)]['category']) for index in panel]
    arrays['row_indices'] = [int(index) for index in panel]
    return arrays, means


def evaluate_conditional_restorer(model, reference_codec, data, panel, *, step,
                                  batch_size=8, geometry_panel=None, baseline=None):
    """Four branches on shared frozen RF-Q / Source-EQ codes and Source labels.

    Old arms are the complete frozen parent codec. New arms add only the
    conditional restorer using codes, old decoded RGB, state20 and raw map16.
    """
    panel = [int(index) for index in panel]
    geometry_panel = None if geometry_panel is None else [int(index) for index in geometry_panel]
    if (type(step) is not int or step < 0 or type(batch_size) is not int or batch_size < 1
            or not panel or len(set(panel)) != len(panel)
            or geometry_panel is not None and (not geometry_panel or len(set(geometry_panel)) != len(geometry_panel))):
        raise ValueError('Conditional-restorer validation requires unique nonempty panels and positive batch size')
    identity, identity_sha = _identity(data)
    panel_sha = data.panel_fingerprint(panel)
    geometry_sha = data.panel_fingerprint(geometry_panel) if geometry_panel is not None else None
    with _isolated_restorer_validation(model, reference_codec, data) as reference_hash:
        if baseline is not None:
            expected = dict(kind=KIND, protocol=PROTOCOL, panel_fingerprint=panel_sha,
                geometry_panel_fingerprint=geometry_sha, data_identity_sha256=identity_sha,
                reference_codec_sha256=reference_hash, solver_steps=3)
            if any(baseline.get(key) != value for key, value in expected.items()):
                raise ValueError('Conditional-restorer baseline must use identical data, panels and frozen parent codec')
        arrays, means = _collect_restorer(model, reference_codec, data, panel, batch_size, objective=True)
        models = _models(arrays)
        geometry = None
        if geometry_panel is not None:
            geometry_arrays, _ = _collect_restorer(model, reference_codec, data, geometry_panel,
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
            source['passed'] = False
            source['failure_reasons'].append('same_panel_Source_code_geometry_regressed')
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
        full_codec_tensors_unchanged=True, decoder_post_quant_conv_tensors_unchanged=True,
        encoder_quantizer_tensors_unchanged=True, conditional_restorer_tensors_unchanged_by_validation=True,
        source_sampler_unchanged=True, source_held_out=True, renderer_only=True, native_resolution=32,
        scope='Native32 GT-pose Source rendering; frozen parent RF3 and full codec plus independent conditional restorer',
        restorer_inputs=['quantized_codes', 'frozen_decoded_RGB32', 'state20', 'raw_map16x16'],
        Source_RGB_direct_restorer_input=False, Source_RGB_into_RF_conditioning=False,
        future_source_rgb_in_prior=False, future_source_pose_in_prior=False,
        source_geometry_and_masks_labels_only=True, sr_quality_evaluated=False,
        autonomous_rollout=False, world_rollout=False, gameplay_qualified=False, automatic_active_promotion=False,
        models=models, geometry=geometry, source_preservation=source, rf_improvement=improvement,
        val_loss=means.loss / means.frames,
        validation_objective=dict(protocol=PROTOCOL,
            input_kind='frozen_RF3_or_Source_quantized_codes_RGB_state_map', loss=means.loss / means.frames,
            frames=means.frames, restorer_examples=2 * means.frames, decoder_examples=2 * means.frames,
            batch_size=batch_size, branch_weights=dict(rf_quantized=.5, source_quantized=.5),
            components=means.components(), weights=conditional_restorer_protocol()['weights'],
            aggregation='batch_image_weighted',
            aggregation_details='Actual paired objective mean weighted by Source image count; unequal tail weighted by its size. Support counts are summed.',
            source_held_out=True, matches_training_formula=True),
        passed=quality, quality_passed=quality,
        thresholds=dict(ssim_strict_min=.9, mae_strict_max=.03, npc_foreground_strict_max=.05,
                        visible_frames_min=IMPROVEMENT['visible_frames_min'], applies_to='rf_new native32'),
        experimental_selection=dict(validated_improvement=improvement['validated_improvement']
            and source['passed'], requires_Source_code_preservation=True))
    if baseline is not None:
        if any(models[name] != baseline['models'][name] for name in ('rf_old', 'source_old')):
            raise ValueError('Frozen RF/parent codec outputs changed from the conditional-restorer baseline')
        report['baseline_step'] = baseline['step']
    arrays.update(panel_fingerprint=panel_sha, step=step)
    return report, arrays


def save_conditional_restorer_preview(arrays, path, step):
    """Reuse the existing same-code five-row layout with truthful titles."""
    from pathlib import Path
    from PIL import Image, ImageDraw

    metadata = save_decoder_adaptation_preview(arrays, path, step)
    titles = ['SOURCE32', 'FROZEN D | RF3 CODES', 'RESTORER | RF3 CODES',
              'FROZEN D | SOURCE E/Q', 'RESTORER | SOURCE E/Q']
    # The inherited pixels/layout are unchanged; this profile trains no D.
    path = Path(path)
    with Image.open(path) as original:
        image = original.convert('RGB')
    draw = ImageDraw.Draw(image)
    for row, title in enumerate(titles):
        top = row * 217
        draw.rectangle((0, top, image.width - 1, top + 24), fill=(12, 17, 25))
        for column, category in enumerate(metadata['columns']):
            draw.text((column * 192 + 3, top + 3), f'{title} g{step} | {category}', fill='white')
    temporary = path.with_name(path.name + '.tmp.png')
    image.save(temporary)
    temporary.replace(path)
    metadata.update(kind=PREVIEW_KIND, protocol=PROTOCOL,
        row_labels=titles,
        restoration='conditional_restorer_after_frozen_decoder',
        Source_RGB_direct_restorer_input=False)
    return metadata
