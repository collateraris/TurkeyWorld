"""Decoder-only overlays on an immutable, strictly validated expanded RF3 bundle.

The original codec qualification applies only to the nested original codec. The
new decoder is always explicitly unqualified; E/Q and RF conditioning are exact.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from .renderer_rf5_bundle import (
    _atomic_save, _parent_input, _private_teacher_copy, _sha,
    parent_bundle_fingerprint, load_rf5_renderer_payload,
)
from .renderer_student import _hash_tensors, renderer_condition_identity


FORMAT_VERSION = 3
STAGE = 'decoder_rf_adaptation'
KIND = 'decoder_rf_adaptation_bundle_v1'
PROTOCOL = 'decoder_rf_source_paired_adaptation_v1'
DECODER_PREFIXES = ('codec.decoder.', 'codec.post_quant_conv.')
ENCODER_QUANTIZER_PREFIXES = ('codec.encoder.', 'codec.quant_conv.', 'codec.quantize.')


def _decoder(state):
    return {key: value for key, value in state.items() if key.startswith(DECODER_PREFIXES)}


def _frozen(state):
    return {key: value for key, value in state.items() if not key.startswith(DECODER_PREFIXES)}


def _encoder_quantizer(state):
    return {key: value for key, value in state.items() if key.startswith(ENCODER_QUANTIZER_PREFIXES)}


def _spec(state):
    return {key: {'shape': list(value.shape), 'dtype': str(value.dtype)}
            for key, value in sorted(state.items())}


def _parent_header(parent):
    from .renderer_rf5_bundle import KIND as RF_KIND, PROTOCOL as RF_PROTOCOL
    if (not isinstance(parent, dict) or parent.get('format_version') != 2
            or parent.get('stage') != 'renderer_rf5_expanded' or parent.get('kind') != RF_KIND
            or parent.get('objective_protocol') != RF_PROTOCOL
            or parent.get('solver') != 'euler' or type(parent.get('solver_steps')) is not int
            or parent['solver_steps'] != 3 or not isinstance(parent.get('model'), dict)
            or not _sha(parent.get('model_sha256'))
            or not _sha(parent.get('base_student_identity'))
            or type(parent.get('training_steps')) is not int or parent['training_steps'] < 1):
        raise ValueError('Decoder adaptation requires an intact, explicit expanded RF3 format2 parent')
    decoder = _decoder(parent['model'])
    if (not decoder or not any(key.startswith('codec.decoder.') for key in decoder)
            or not any(key.startswith('codec.post_quant_conv.') for key in decoder)):
        raise ValueError('RF3 parent lacks complete decoder and post-quantization weights')


def _flow_contract(model, parent):
    flow = model.flow_renderer
    actual_config = getattr(flow, '_renderer_student_config', None)
    # The strict parent loader already validates dynamics-only descendants.
    # Preserve its established live RSSM/V-JEPA metadata allowance here.
    dynamic = ('rssm_', 'vjepa_', 'enemy_rssm_', 'combat_rssm_')
    frozen_config = lambda value: {key: item for key, item in value.items()
        if not key.startswith(dynamic) and key not in {'architecture', 'player_motion_mode'}}
    if (flow.kind != 'student' or flow.mode != 'rectified_flow'
            or flow.steps != parent['solver_steps']
            or not isinstance(actual_config, dict)
            or frozen_config(actual_config) != frozen_config(parent['config'])
            or not hasattr(model, 'base_student')
            or _hash_tensors(model.base_student.state_dict()) != parent['base_student_identity']
            or getattr(flow, '_renderer_base_identity', None) != parent['base_student_identity']):
        raise ValueError('Decoder adaptation changed the parent RF3 solver/config/Base binding')
    state = flow.state_dict()
    if (set(state) != set(parent['model'])
            or _hash_tensors(_frozen(state)) != _hash_tensors(_frozen(parent['model']))
            or _spec(_decoder(state)) != _spec(_decoder(parent['model']))):
        raise ValueError('Decoder adaptation changed RF, E/Q, geometry, normalization or allowed shapes')
    return state


def _check_override(override, parent):
    expected = _decoder(parent['model'])
    if (not isinstance(override, dict) or set(override) != set(expected)
            or not all(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all())
                       for value in override.values())
            or _spec(override) != _spec(expected)):
        raise ValueError('Decoder override must contain every exact decoder/post_quant_conv tensor, and no others')


def _apply_override(flow, override):
    # Only these two modules are copied. Frozen RF/E/Q tensor versions and
    # identities are not changed by a whole-codec/whole-renderer load_state_dict.
    for prefix, module in (('codec.decoder.', flow.codec.decoder),
                           ('codec.post_quant_conv.', flow.codec.post_quant_conv)):
        module.load_state_dict({key[len(prefix):]: value for key, value in override.items()
                                if key.startswith(prefix)}, strict=True)


def export_decoder_adaptation(model, parent_bundle, path, *, decoder_updates,
                              source_model=None, metrics=None, provenance=None):
    """Export only D/postquant; RF/Base counters remain fixed at the RF3 parent."""
    parent, source_sha, source_path = _parent_input(parent_bundle, provenance)
    _parent_header(parent)
    if type(decoder_updates) is not int or decoder_updates < 0:
        raise ValueError('Decoder update count must be an explicit nonnegative integer')
    state = _flow_contract(model, parent)
    override = {key: value.detach().cpu().clone() for key, value in _decoder(state).items()}
    _check_override(override, parent)
    # Preserve the original teacher for immutable Combat/RSSM/codec lineage.
    # The working model already contains the replaced compact codec/new D.
    private = _private_teacher_copy(model if source_model is None else source_model)
    load_rf5_renderer_payload(private, parent, checkpoint=source_path or '<RF3-parent>',
        allow_unqualified_codec=getattr(model.flow_renderer, '_renderer_allow_unqualified_codec', False))
    actual = renderer_condition_identity(model)
    expected = renderer_condition_identity(private)
    if any(actual[key] != value for key, value in expected.items() if key != 'codec_sha256'):
        raise ValueError('Decoder adaptation changed validated Base/map/appearance/normalization conditions')
    payload = dict(format_version=FORMAT_VERSION, stage=STAGE, kind=KIND,
        objective_protocol=PROTOCOL, parent_bundle=deepcopy(parent),
        parent_bundle_sha256=parent_bundle_fingerprint(parent),
        parent_source_sha256=source_sha, parent_source_checkpoint=source_path,
        config=deepcopy(parent['config']), source_teacher_config=deepcopy(parent['source_teacher_config']),
        solver=parent['solver'], solver_steps=parent['solver_steps'],
        solver_profile=deepcopy(parent['solver_profile']),
        training_steps=parent['training_steps'], renderer_training_steps=parent['training_steps'],
        parent_renderer_training_steps=parent['training_steps'],
        refinement_updates=parent['refinement_updates'],
        base_training_steps=parent['base_training_steps'],
        decoder_refinement_updates=decoder_updates,
        decoder_override=override, decoder_override_sha256=_hash_tensors(override),
        decoder_override_spec=_spec(override), allowed_prefixes=list(DECODER_PREFIXES),
        frozen_flow_sha256=_hash_tensors(_frozen(state)),
        encoder_quantizer_sha256=_hash_tensors(_encoder_quantizer(state)),
        adapted_flow_sha256=_hash_tensors(state),
        adapted_codec_sha256=_hash_tensors(model.flow_renderer.codec.state_dict()),
        codec_config=deepcopy(model.flow_renderer.codec.get_config()),
        base_student_identity=parent['base_student_identity'],
        source_codec_reference='unchanged_nested_RF3_parent_decoder',
        decoder_quality_qualified=False, source_quality_passed=False,
        original_codec_qualification_applies_to_new_decoder=False,
        experimental_preview=True, gameplay_qualified=False, automatic_promotion=False,
        Source_future_inputs=False, metrics=deepcopy(metrics or {}), provenance=deepcopy(provenance or {}))
    _atomic_save(payload, path)
    return {key: value for key, value in payload.items()
            if key not in {'parent_bundle', 'decoder_override'}}


def load_decoder_adaptation_payload(teacher, payload, *, checkpoint,
                                     require_trained=True, allow_unqualified_codec=False):
    """Strict parent first, private D replacement second, renderer-only commit last."""
    if (not isinstance(payload, dict) or payload.get('format_version') != FORMAT_VERSION
            or payload.get('stage') != STAGE or payload.get('kind') != KIND
            or payload.get('objective_protocol') != PROTOCOL
            or not _sha(payload.get('parent_source_sha256'))
            or payload.get('allowed_prefixes') != list(DECODER_PREFIXES)
            or payload.get('decoder_quality_qualified') is not False
            or payload.get('source_quality_passed') is not False
            or payload.get('original_codec_qualification_applies_to_new_decoder') is not False
            or payload.get('experimental_preview') is not True
            or payload.get('gameplay_qualified') is not False
            or payload.get('automatic_promotion') is not False
            or payload.get('Source_future_inputs') is not False
            or payload.get('source_codec_reference') != 'unchanged_nested_RF3_parent_decoder'):
        raise ValueError('Expected an explicitly unqualified decoder-only RF3 adaptation bundle')
    parent = payload.get('parent_bundle')
    _parent_header(parent)
    if (payload.get('parent_bundle_sha256') != parent_bundle_fingerprint(parent)
            or payload.get('config') != parent['config']
            or payload.get('source_teacher_config') != parent['source_teacher_config']
            or payload.get('base_student_identity') != parent['base_student_identity']
            or payload.get('solver') != parent['solver']
            or type(payload.get('solver_steps')) is not int or payload['solver_steps'] != parent['solver_steps']
            or payload.get('solver_profile') != parent['solver_profile']
            or any(type(payload.get(key)) is not int or payload[key] != parent['training_steps']
                   for key in ('training_steps', 'renderer_training_steps', 'parent_renderer_training_steps'))
            or payload.get('refinement_updates') != parent['refinement_updates']
            or payload.get('base_training_steps') != parent['base_training_steps']
            or type(payload.get('decoder_refinement_updates')) is not int
            or payload['decoder_refinement_updates'] < 0):
        raise ValueError('Decoder bundle changed its exact parent, solver, Base or frozen RF clock')
    override = payload.get('decoder_override')
    _check_override(override, parent)
    if (payload.get('decoder_override_sha256') != _hash_tensors(override)
            or payload.get('decoder_override_spec') != _spec(override)
            or payload.get('frozen_flow_sha256') != _hash_tensors(_frozen(parent['model']))
            or payload.get('encoder_quantizer_sha256') != _hash_tensors(_encoder_quantizer(parent['model']))):
        raise ValueError('Decoder override or immutable RF/E/Q hashes differ')
    private = _private_teacher_copy(teacher)
    load_rf5_renderer_payload(private, parent,
        checkpoint=payload.get('parent_source_checkpoint') or '<RF3-parent>',
        require_trained=require_trained, allow_unqualified_codec=allow_unqualified_codec)
    flow = private.flow_renderer
    if payload.get('codec_config') != flow.codec.get_config():
        raise ValueError('Decoder adaptation changed codec/latent/codebook geometry')
    _apply_override(flow, override)
    _flow_contract(private, parent)
    if (_hash_tensors(flow.state_dict()) != payload.get('adapted_flow_sha256')
            or _hash_tensors(flow.codec.state_dict()) != payload.get('adapted_codec_sha256')):
        raise ValueError('Adapted decoder/codec/full-flow hashes differ')
    flow._check_codec()
    flow.eval()
    flow._renderer_decoder_adaptation = {key: deepcopy(value) for key, value in payload.items()
        if key not in {'parent_bundle', 'decoder_override'}}
    flow._renderer_decoder_parent_bundle = deepcopy(parent)
    if hasattr(teacher, 'base_student'):
        if _hash_tensors(teacher.base_student.state_dict()) != parent['base_student_identity']:
            raise ValueError('Loaded compact Base differs from decoder-adaptation parent')
        private.base_student = teacher.base_student
    metadata = deepcopy(flow._renderer_decoder_adaptation)
    metadata['checkpoint'] = str(Path(checkpoint).resolve())
    teacher.flow_renderer = flow
    teacher.base_student = private.base_student
    teacher.base_student_metadata = private.base_student_metadata
    teacher.base_student_bundle = private.base_student_bundle
    teacher.config = private.config
    teacher.renderer_student_metadata = metadata
    return metadata
