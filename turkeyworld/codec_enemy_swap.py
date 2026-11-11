"""Prepare an explicit compact-codec replacement for approved current RF1 weights.

This helper does not deploy, attach, export, train or modify either input model.
The original teacher's Base/map/appearance/codec identity remains the source of
truth. Only the new private Flow's compact E/D tensors and their provenance
change; quantizer, latent normalization, fixed noise and Base stay identical.
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch

from .base_student import prepare_base_student
from .renderer_student import (EXPERIMENTAL_CODEC_REASON, _hash_tensors, _require_teacher,
    _shared_identity, _student_config, _validate_compact_codec, codec_tensor_fingerprint,
    renderer_condition_identity, validate_renderer_source_lineage)
from .vqgan import VQGAN


def _candidate(original_source, path_or_payload, allow_unqualified_codec):
    payload = (torch.load(Path(path_or_payload), map_location='cpu', weights_only=False)
               if isinstance(path_or_payload, (str, Path)) else path_or_payload)
    if (not isinstance(payload, dict) or payload.get('format_version') != 1
            or not isinstance(payload.get('model'), dict)):
        raise ValueError('Codec replacement requires a standard trained compact-codec payload')
    state = payload['model']
    if not all(torch.is_tensor(value) and bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError('Replacement codec tensors must be finite')
    with torch.random.fork_rng(devices=[]):
        codec = VQGAN(payload.get('codec_config', {})).eval().requires_grad_(False)
    try:
        codec.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError('Replacement codec tensors differ from its declared configuration') from error
    provenance = {key: copy.deepcopy(value) for key, value in payload.items() if key != 'model'}
    gate = provenance.get('quality_gate')
    if allow_unqualified_codec and isinstance(gate, dict) and gate.get('passed') is False:
        provenance['acceptance'] = {'scope':'experimental_preview','quality_gate_passed':False,
            'reason':EXPERIMENTAL_CODEC_REASON,'allow_unqualified_codec':True}
    _validate_compact_codec(original_source, codec, provenance,
                            allow_unqualified_codec=allow_unqualified_codec)
    actual = codec_tensor_fingerprint(codec)
    if 'codec_fingerprint' in provenance and provenance['codec_fingerprint'] != actual:
        raise ValueError('Replacement codec fingerprint differs from its actual E/Q/D tensors')
    provenance.update(codec_fingerprint=actual, parameters=sum(p.numel() for p in codec.parameters()))
    if isinstance(path_or_payload, (str, Path)):
        provenance['checkpoint'] = str(Path(path_or_payload).resolve())
    return codec, provenance


def prepare_codec_enemy_swap(original_source, working, codec_checkpoint_or_payload, *,
                            allow_unqualified_codec=False):
    """Return(new_flow, metadata), validating all dependencies before any attachment.

    `original_source` must retain the original teacher codec and copied renderer
    assets. `working` must already contain a validated compact Base/RF1 bundle.
    Both disk checkpoints and standard in-memory codec payloads are accepted.
    The caller may assign the returned Flow for isolated validation/adaptation;
    final export must still use the unchanged original_source, never working.
    """
    if not isinstance(allow_unqualified_codec, bool):
        raise ValueError('allow_unqualified_codec must be an explicit boolean')
    _require_teacher(original_source)
    old = _require_teacher(working)
    if old.kind != 'student' or old.steps != 1 or old.mode != 'rectified_flow':
        raise ValueError('Enemy codec replacement requires the current one-Euler RF student')
    _student_config(old, original_source)
    approved = getattr(working, 'renderer_student_metadata', None)
    if not isinstance(approved, dict) or not isinstance(approved.get('source_teacher_config'), dict):
        raise ValueError('Current RF needs its validated original-source bundle metadata')
    validate_renderer_source_lineage(original_source, approved['source_teacher_config'])
    original_identity = renderer_condition_identity(original_source)
    if approved.get('conditioning_identity') != original_identity:
        raise ValueError('Current renderer bundle belongs to different original-source assets')
    current_identity = renderer_condition_identity(working)
    if any(current_identity[key] != original_identity[key]
           for key in ('base_map_sha256','appearance_sha256')):
        raise ValueError('Working Base/map/appearance differs from the original source')
    shared = _shared_identity(old)
    if any(shared[key] != original_identity[key]
           for key in ('latent_normalization_sha256','fixed_noise_sha256')):
        raise ValueError('Current RF changed original latent normalization or fixed noise')
    old_provenance = getattr(old, '_renderer_codec_distillation', None)
    if not isinstance(old_provenance, dict):
        raise ValueError('Current RF must have approved compact-codec provenance')
    _validate_compact_codec(original_source, old.codec, old_provenance,
        allow_unqualified_codec=getattr(old, '_renderer_allow_unqualified_codec', False))
    base_bundle = getattr(old, '_renderer_base_bundle', None)
    if not isinstance(base_bundle, dict) or not hasattr(working, 'base_student'):
        raise ValueError('Current RF needs an approved unchanged compact Base')
    approved_base, _, _ = prepare_base_student(original_source, base_bundle)
    base_identity = _hash_tensors(working.base_student.state_dict())
    if (base_identity != _hash_tensors(approved_base.state_dict())
            or base_identity != getattr(old, '_renderer_base_identity', None)):
        raise ValueError('Current compact Base changed after renderer bundle validation')
    previous_state = old.state_dict()
    if not all(bool(torch.isfinite(value).all()) for value in previous_state.values()):
        raise ValueError('Current RF weights must be finite')
    codec, provenance = _candidate(original_source, codec_checkpoint_or_payload, allow_unqualified_codec)
    if codec.get_config() != old.codec.get_config():
        raise ValueError('Enemy fine-tuning may change compact E/D weights, not codec architecture')
    parameter = next(old.codec.parameters())
    codec.to(device=parameter.device, dtype=parameter.dtype).eval().requires_grad_(False)
    candidate = copy.deepcopy(old)
    candidate.codec = codec
    candidate._renderer_codec_distillation = provenance
    candidate._renderer_allow_unqualified_codec = allow_unqualified_codec
    candidate._renderer_student_config = copy.deepcopy(old._renderer_student_config)
    candidate._renderer_student_config['renderer_codec_config'] = codec.get_config()
    candidate._codec_student_bundle = {**copy.deepcopy(provenance),
        'model':{key:value.detach().cpu().clone() for key,value in codec.state_dict().items()}}
    candidate._codec_student_identity = _shared_identity(candidate)
    _student_config(candidate, original_source)
    _validate_compact_codec(original_source, candidate.codec, provenance,
                            allow_unqualified_codec=allow_unqualified_codec)
    candidate_state = candidate.state_dict()
    noncodec = [key for key in previous_state if not key.startswith('codec.')]
    if any(key not in candidate_state or not torch.equal(previous_state[key], candidate_state[key])
           for key in noncodec):
        raise AssertionError('Compact-codec replacement changed non-codec RF tensors')
    if _hash_tensors(working.base_student.state_dict()) != base_identity:
        raise AssertionError('Compact-codec preparation changed the live Base')
    metadata = {'kind':'approved_compact_codec_replacement','automatic_deployment':False,
        'original_source_conditioning_identity':original_identity,
        'parent_renderer_checkpoint':approved.get('checkpoint'),
        'parent_renderer_training_steps':approved.get('training_steps'),
        'previous_codec_sha256':shared['codec_sha256'],
        'student_codec_identity':copy.deepcopy(candidate._codec_student_identity),
        'codec_distillation':copy.deepcopy(provenance),
        'codec_quality_qualified':provenance['quality_gate']['passed'],
        'experimental_preview':provenance['quality_gate']['passed'] is False,
        'rf_noncodec_tensors_bitexact':True,'noncodec_rf_tensor_count':len(noncodec),
        'base_identity':base_identity,'base_bitexact':True,'quantizer_bitexact':True,
        'latent_normalization_bitexact':True,'fixed_noise_bitexact':True,
        'requires_full_renderer_revalidation':True,'gameplay_qualified':False}
    candidate._codec_enemy_swap = copy.deepcopy(metadata)
    return candidate, metadata
