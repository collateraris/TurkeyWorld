"""Explicit RF5-expanded bundles over an immutable, strictly validated RF1 parent.

The parent config remains an honest RF1 source contract. No WorldModelConfig
field is inferred or rewritten to make an unflagged legacy bundle accept RF5.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import time

import torch

from .renderer_rf5_experts import (
    SETTINGS, rf5_expert_metadata, upgrade_rf5_experts,
    validate_rf5_expert_metadata,
)


FORMAT_VERSION = 2
STAGE = 'renderer_rf5_expanded'
KIND = 'renderer_rf5_expanded_bundle_v1'
PROTOCOL = 'renderer_rf_expanded_experts_v1'
SOLVER_STEPS = (3, 5)


def _solver_profile(steps):
    if type(steps) is not int or steps not in SOLVER_STEPS:
        raise ValueError('Expanded RF requires an explicit integer of three or five Euler steps')
    return {'method': 'euler', 'steps': steps,
            'velocity_evaluation_times': [index / steps for index in range(steps)]}


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def parent_bundle_fingerprint(payload):
    """Canonical logical hash of all nested parent values, including tensor bytes."""
    digest = hashlib.sha256()

    def visit(value):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(json.dumps(['tensor', str(tensor.dtype), list(tensor.shape)],
                                     separators=(',', ':')).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            digest.update(b'dict[')
            for key in sorted(value, key=lambda item: (type(item).__name__, str(item))):
                visit(key)
                visit(value[key])
            digest.update(b']')
        elif isinstance(value, (list, tuple)):
            digest.update(b'list[' if isinstance(value, list) else b'tuple[')
            for item in value:
                visit(item)
            digest.update(b']')
        elif value is None or type(value) in (str, int, float, bool):
            digest.update(json.dumps([type(value).__name__, value], allow_nan=False,
                                     separators=(',', ':')).encode())
        else:
            raise ValueError(f'Unsupported renderer parent metadata type: {type(value).__name__}')
    visit(payload)
    return digest.hexdigest()


def _parent_input(parent, provenance=None):
    provenance = provenance or {}
    if isinstance(parent, (str, Path)):
        source = Path(parent)
        with source.open('rb') as stream:
            payload = torch.load(stream, map_location='cpu', weights_only=False)
            stream.seek(0)
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(4 << 20), b''):
                digest.update(block)
        fingerprint = digest.hexdigest()
        if (provenance.get('parent_source_sha256') is not None
                and provenance['parent_source_sha256'] != fingerprint):
            raise ValueError('RF5 parent source file SHA differs from its pinned provenance')
        return payload, fingerprint, str(source.resolve())
    if not isinstance(parent, dict) or not _sha(provenance.get('parent_source_sha256')):
        raise ValueError('An in-memory RF5 parent requires its explicit parent_source_sha256 file pin')
    return parent, provenance['parent_source_sha256'], provenance.get('parent_source_checkpoint')


def _validate_parent_header(parent):
    from .renderer_moe import FIVE_KIND, FIVE_SCHEMA
    if (not isinstance(parent, dict) or parent.get('format_version') != 1
            or parent.get('stage') != 'renderer_student'
            or parent.get('solver') != 'euler' or type(parent.get('solver_steps')) is not int
            or parent['solver_steps'] != 1
            or parent.get('config', {}).get('flow_steps') != 1
            or type(parent.get('training_steps')) is not int or parent['training_steps'] < 1
            or parent.get('renderer_output_moe', {}).get('kind') != FIVE_KIND
            or parent.get('renderer_output_moe', {}).get('flow', {}).get('role_schema') != FIVE_SCHEMA
            or not isinstance(parent.get('base_student_bundle'), dict)
            or not _sha(parent.get('base_student_identity'))):
        raise ValueError('RF5 expanded bundle requires the intact trained five-role RF1/Base parent')


def _fixed_state(state):
    return {name: value for name, value in state.items()
            if name.startswith(('codec.', 'geometry_opacity.'))
            or name in {'codec_loaded', 'latent_mean', 'latent_std', 'fixed_noise'}}


def _base_steps(parent):
    value = parent['base_student_bundle'].get('training_steps')
    if type(value) is not int or value < 1:
        raise ValueError('RF5 parent lacks its positive trained compact Base clock')
    return value


def _validate_state_against_parent(state, parent):
    from .renderer_student import _hash_tensors
    if (not isinstance(state, dict) or not state
            or not all(isinstance(v, torch.Tensor) and bool(torch.isfinite(v).all())
                       for v in state.values())):
        raise ValueError('RF5 bundle requires full finite tensor weights')
    fixed = _fixed_state(state)
    expected = _fixed_state(parent['model'])
    if set(fixed) != set(expected) or _hash_tensors(fixed) != _hash_tensors(expected):
        raise ValueError('RF5 training changed parent codec, private opacity, normalization or fixed noise')
    return _hash_tensors(fixed)


def _atomic_save(payload, path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f'.{os.getpid()}.tmp')
    try:
        torch.save(payload, temporary)
        for attempt in range(21):
            try:
                os.replace(temporary, output)
                break
            except PermissionError:
                if attempt == 20:
                    raise
                time.sleep(.1)
    finally:
        temporary.unlink(missing_ok=True)


def _private_teacher_copy(model):
    # MoE diagnostic caches can retain non-leaf training tensors. They are not
    # model state and cannot be deep-copied by PyTorch. Discard them in the copy
    # through deepcopy's memo, without modifying even the live cache itself.
    memo = {id(module.last_aux): None for module in model.modules()
            if getattr(module, 'last_aux', None) is not None}
    return deepcopy(model, memo)


def export_rf5_renderer(model, parent_bundle, path, *, training_steps, local_updates,
                        base_training_steps=29209, metrics=None, provenance=None,
                        source_model=None):
    """Write RF-only refinement; local0 is explicitly unqualified initialization."""
    from .renderer_student import _hash_tensors
    parent, source_sha, source_path = _parent_input(parent_bundle, provenance)
    _validate_parent_header(parent)
    if (type(local_updates) is not int or local_updates < 0
            or type(training_steps) is not int
            or training_steps != parent['training_steps'] + local_updates
            or type(base_training_steps) is not int or base_training_steps != _base_steps(parent)):
        raise ValueError('RF5 local/cumulative/Base clocks differ from the immutable parent')
    flow = model.flow_renderer
    if (flow.kind != 'student' or flow.mode != 'rectified_flow'
            or type(flow.steps) is not int or flow.steps not in SOLVER_STEPS
            or not hasattr(model, 'base_student')
            or _hash_tensors(model.base_student.state_dict()) != parent['base_student_identity']
            or getattr(flow, '_renderer_base_identity', None) != parent['base_student_identity']
            or getattr(flow, '_renderer_student_config', {}).get('flow_steps') != 1):
        raise ValueError('Expanded RF export needs Euler3/5 and its unchanged RF1 parent/Base contract')
    expansion = rf5_expert_metadata(flow.output_moe)
    state = {name: value.detach().cpu().clone() for name, value in flow.state_dict().items()}
    fixed_sha = _validate_state_against_parent(state, parent)
    # Validate the original dependency chain against the current teacher privately.
    # This cannot reset the live renderer's trained RF5 weights or its optimizer.
    from .renderer_student import _load_renderer_student_payload, renderer_condition_identity
    # A migrated teacher's immutable lineage refers to its ORIGINAL codec.
    # Re-validating that lineage against the already-replaced compact codec is
    # invalid. Callers with codec replacement provide the unchanged source
    # teacher, captured before the first strict parent load.
    private = _private_teacher_copy(model if source_model is None else source_model)
    _load_renderer_student_payload(private, parent, checkpoint=source_path or '<RF5-parent>',
        allow_unqualified_codec=getattr(flow, '_renderer_allow_unqualified_codec', False))
    if renderer_condition_identity(model) != renderer_condition_identity(private):
        raise ValueError('Expanded RF working Base/map/appearance/codec conditions differ from its validated parent')
    payload = {
        'format_version': FORMAT_VERSION, 'stage': STAGE, 'kind': KIND,
        'objective_protocol': PROTOCOL, 'solver': 'euler', 'solver_steps': flow.steps,
        'solver_profile': _solver_profile(flow.steps),
        'parent_bundle': deepcopy(parent),
        'parent_bundle_sha256': parent_bundle_fingerprint(parent),
        'parent_source_sha256': source_sha, 'parent_source_checkpoint': source_path,
        'config': deepcopy(parent['config']),
        'source_teacher_config': deepcopy(parent['source_teacher_config']),
        'conditioning_identity': deepcopy(parent['conditioning_identity']),
        'model': state, 'model_sha256': _hash_tensors(state),
        'expansion': expansion, 'base_student_identity': parent['base_student_identity'],
        'parent_fixed_assets_sha256': fixed_sha,
        'parent_renderer_training_steps': parent['training_steps'],
        'training_steps': training_steps, 'local_updates': local_updates,
        'refinement_updates': local_updates, 'base_training_steps': base_training_steps,
        'metrics': deepcopy(metrics or {}), 'provenance': deepcopy(provenance or {}),
        'experimental_preview': True, 'source_quality_passed': False,
        'gameplay_qualified': False, 'automatic_promotion': False,
        'quality_status': 'RF5 expanded experts; image/gameplay qualification pending',
    }
    _atomic_save(payload, path)
    return {k: v for k, v in payload.items() if k not in {'model', 'parent_bundle'}}


def load_rf5_renderer_payload(teacher, payload, *, checkpoint,
                              require_trained=True, allow_unqualified_codec=False):
    """Check a private copy fully before committing only renderer-related fields."""
    from .renderer_student import _hash_tensors, _load_renderer_student_payload
    if not isinstance(allow_unqualified_codec, bool):
        raise ValueError('allow_unqualified_codec must be an explicit boolean')
    if (not isinstance(payload, dict) or payload.get('format_version') != FORMAT_VERSION
            or payload.get('stage') != STAGE or payload.get('kind') != KIND
            or payload.get('objective_protocol') != PROTOCOL
            or payload.get('solver') != 'euler' or type(payload.get('solver_steps')) is not int
            or payload['solver_steps'] not in SOLVER_STEPS
            or payload.get('solver_profile') != _solver_profile(payload['solver_steps'])
            or not _sha(payload.get('parent_source_sha256'))
            or payload.get('experimental_preview') is not True
            or payload.get('source_quality_passed') is not False
            or payload.get('gameplay_qualified') is not False
            or payload.get('automatic_promotion') is not False):
        raise ValueError('Expected an explicitly unqualified RF5 expanded renderer bundle')
    parent = payload.get('parent_bundle')
    _validate_parent_header(parent)
    if (payload.get('parent_bundle_sha256') != parent_bundle_fingerprint(parent)
            or payload.get('config') != parent['config']
            or payload.get('source_teacher_config') != parent['source_teacher_config']
            or payload.get('conditioning_identity') != parent['conditioning_identity']
            or payload.get('base_student_identity') != parent['base_student_identity']
            or payload.get('base_training_steps') != _base_steps(parent)
            or payload.get('parent_renderer_training_steps') != parent['training_steps']):
        raise ValueError('RF5 bundle changed its immutable RF1 parent or Base binding')
    local = payload.get('local_updates')
    if (type(local) is not int or local < 0 or payload.get('refinement_updates') != local
            or type(payload.get('training_steps')) is not int
            or payload['training_steps'] != parent['training_steps'] + local
            or require_trained and payload['training_steps'] < 1):
        raise ValueError('RF5 bundle has inconsistent completed update clocks')
    validate_rf5_expert_metadata(payload.get('expansion'))
    state = payload.get('model')
    fixed_sha = _validate_state_against_parent(state, parent)
    if (payload.get('model_sha256') != _hash_tensors(state)
            or payload.get('parent_fixed_assets_sha256') != fixed_sha):
        raise ValueError('RF5 model/frozen-asset tensor hashes differ')
    private = _private_teacher_copy(teacher)
    _load_renderer_student_payload(private, parent,
        checkpoint=payload.get('parent_source_checkpoint') or '<RF5-parent>',
        allow_unqualified_codec=allow_unqualified_codec)
    flow = private.flow_renderer
    upgrade_rf5_experts(flow.output_moe)
    flow.load_state_dict(state, strict=True)
    if payload['expansion'] != rf5_expert_metadata(flow.output_moe):
        raise ValueError('RF5 expansion names/counts differ from the constructed architecture')
    flow.steps = payload['solver_steps']  # After RF1 chain and expanded state validate.
    flow.eval()
    flow._renderer_rf5_expansion = deepcopy(payload['expansion'])
    flow._renderer_rf5_parent_bundle = deepcopy(parent)
    flow._renderer_rf5_parent_source_sha256 = payload['parent_source_sha256']
    flow._renderer_rf5_protocol = PROTOCOL
    # Reuse already loaded identical frozen objects; no non-renderer is replaced.
    if hasattr(teacher, 'base_student'):
        if _hash_tensors(teacher.base_student.state_dict()) != parent['base_student_identity']:
            raise ValueError('Loaded compact Base differs from RF5 parent')
        private.base_student = teacher.base_student
    if _hash_tensors(teacher.flow_renderer.codec.state_dict()) == _hash_tensors(flow.codec.state_dict()):
        flow.codec = teacher.flow_renderer.codec
    metadata = {k: deepcopy(v) for k, v in payload.items() if k not in {'model', 'parent_bundle'}}
    metadata['checkpoint'] = str(Path(checkpoint).resolve())
    teacher.flow_renderer = flow
    teacher.base_student = private.base_student
    teacher.base_student_metadata = private.base_student_metadata
    teacher.base_student_bundle = private.base_student_bundle
    teacher.config = private.config
    teacher.renderer_student_metadata = metadata
    return metadata
