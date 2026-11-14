"""A strict restorer-only format4 overlay over an immutable format3 decoder parent."""
from copy import deepcopy
from pathlib import Path

import torch

from .conditional_decoder_restorer import (
    conditional_decoder_restorer_metadata, load_conditional_decoder_restorer,
)
from .decoder_rf_adaptation_bundle import load_decoder_adaptation_payload
from .renderer_rf5_bundle import (
    _atomic_save, _parent_input, _private_teacher_copy, _sha, parent_bundle_fingerprint,
)
from .renderer_student import _hash_tensors, renderer_condition_identity


FORMAT_VERSION = 4
STAGE = 'conditional_rf_restorer'
KIND = 'conditional_restorer_bundle_v1'
PROTOCOL = 'conditional_rf_restorer_paired_v1'
PREFIX = 'conditional_restorer.'
RUNTIME = 'RF3_denormalize_quantize_decode_then_current_state_raw_map_restorer'


def _parent_state(parent):
    from .decoder_rf_adaptation_bundle import KIND as DECODER_KIND, PROTOCOL as DECODER_PROTOCOL
    if (not isinstance(parent, dict) or parent.get('format_version') != 3
            or parent.get('stage') != 'decoder_rf_adaptation'
            or parent.get('kind') != DECODER_KIND
            or parent.get('objective_protocol') != DECODER_PROTOCOL
            or type(parent.get('solver_steps')) is not int or parent['solver_steps'] != 3
            or not isinstance(parent.get('parent_bundle', {}).get('model'), dict)
            or not isinstance(parent.get('decoder_override'), dict)
            or not _sha(parent.get('adapted_flow_sha256'))
            or not _sha(parent.get('base_student_identity'))):
        raise ValueError('Conditional restorer needs its exact validated format3 decoder/RF3 parent')
    state = {**parent['parent_bundle']['model'], **parent['decoder_override']}
    if _hash_tensors(state) != parent['adapted_flow_sha256']:
        raise ValueError('Conditional-restorer parent full RF/E/Q/D state binding differs')
    return state


def _frozen_state(flow):
    return {key: value for key, value in flow.state_dict().items() if not key.startswith(PREFIX)}


def _flow_contract(model, parent):
    flow = model.flow_renderer
    source = _parent_state(parent)
    raw = getattr(flow, '_renderer_student_config', None)
    dynamic = ('rssm_', 'vjepa_', 'enemy_rssm_', 'combat_rssm_')
    frozen = lambda cfg: {key: value for key, value in cfg.items()
        if not key.startswith(dynamic) and key not in {'architecture', 'player_motion_mode'}}
    if (flow.kind != 'student' or flow.mode != 'rectified_flow'
            or flow.steps != parent['solver_steps'] or not flow.uses_wall_conditioning
            or not isinstance(raw, dict) or frozen(raw) != frozen(parent['config'])
            or not hasattr(model, 'base_student')
            or _hash_tensors(model.base_student.state_dict()) != parent['base_student_identity']
            or getattr(flow, '_renderer_base_identity', None) != parent['base_student_identity']
            or set(_frozen_state(flow)) != set(source)
            or _hash_tensors(_frozen_state(flow)) != parent['adapted_flow_sha256']
            or _hash_tensors(flow.codec.state_dict()) != parent['adapted_codec_sha256']
            or flow.codec.config.image_size != 32 or flow.codec.config.embedding_dim != 16
            or flow.codec.config.latent_size != 8):
        raise ValueError('Conditional restorer changed frozen RF3/E/Q/D/Base/geometry/codec conditions')


def _clock_fields(parent):
    return {key: deepcopy(parent[key]) for key in (
        'training_steps', 'renderer_training_steps', 'parent_renderer_training_steps',
        'refinement_updates', 'base_training_steps', 'decoder_refinement_updates')}


def export_conditional_restorer(model, parent_bundle, path, *, restorer_updates,
                                source_model=None, metrics=None, provenance=None):
    """Train-clock applies only to the added module; all parent clocks stay fixed."""
    parent, source_sha, source_path = _parent_input(parent_bundle, provenance)
    _parent_state(parent)
    if type(restorer_updates) is not int or restorer_updates < 0:
        raise ValueError('Restorer update count must be a nonnegative integer')
    _flow_contract(model, parent)
    module = getattr(model.flow_renderer, 'conditional_restorer', None)
    metadata = conditional_decoder_restorer_metadata(module)
    state = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
    if not all(bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError('Conditional restorer weights must be finite')
    private = _private_teacher_copy(model if source_model is None else source_model)
    load_decoder_adaptation_payload(private, parent, checkpoint=source_path or '<decoder-parent>',
        allow_unqualified_codec=getattr(model.flow_renderer, '_renderer_allow_unqualified_codec', False))
    if renderer_condition_identity(model) != renderer_condition_identity(private):
        raise ValueError('Restorer working conditioning differs from the strictly validated original source')
    payload = dict(format_version=FORMAT_VERSION, stage=STAGE, kind=KIND,
        objective_protocol=PROTOCOL, parent_bundle=deepcopy(parent),
        parent_bundle_sha256=parent_bundle_fingerprint(parent), parent_source_sha256=source_sha,
        parent_source_checkpoint=source_path, config=deepcopy(parent['config']),
        source_teacher_config=deepcopy(parent['source_teacher_config']),
        solver=parent['solver'], solver_steps=parent['solver_steps'],
        solver_profile=deepcopy(parent['solver_profile']), **_clock_fields(parent),
        restorer_updates=restorer_updates, restorer_refinement_updates=restorer_updates,
        restorer_config=module.get_config(), restorer_metadata=metadata,
        restorer_state=state, restorer_state_sha256=_hash_tensors(state),
        frozen_parent_flow_sha256=parent['adapted_flow_sha256'],
        frozen_parent_codec_sha256=parent['adapted_codec_sha256'],
        base_student_identity=parent['base_student_identity'],
        runtime_integration=RUNTIME, Source_code_decode_automatically_restored=False,
        Source_future_inputs=False, parent_codec_qualification_applies_to_restorer=False,
        restorer_quality_qualified=False, source_quality_passed=False, experimental_preview=True,
        gameplay_qualified=False, automatic_promotion=False,
        metrics=deepcopy(metrics or {}), provenance=deepcopy(provenance or {}))
    _atomic_save(payload, path)
    return {key: value for key, value in payload.items() if key not in {'parent_bundle', 'restorer_state'}}


def load_conditional_restorer_payload(teacher, payload, *, checkpoint,
                                      require_trained=True, allow_unqualified_codec=False):
    if (not isinstance(payload, dict) or payload.get('format_version') != FORMAT_VERSION
            or payload.get('stage') != STAGE or payload.get('kind') != KIND
            or payload.get('objective_protocol') != PROTOCOL
            or not _sha(payload.get('parent_source_sha256'))
            or not isinstance(payload.get('restorer_config'), dict)
            or not isinstance(payload.get('restorer_metadata'), dict)
            or payload['restorer_metadata'].get('config') != payload['restorer_config']
            or payload.get('runtime_integration') != RUNTIME
            or payload.get('Source_code_decode_automatically_restored') is not False
            or payload.get('Source_future_inputs') is not False
            or payload.get('parent_codec_qualification_applies_to_restorer') is not False
            or payload.get('restorer_quality_qualified') is not False
            or payload.get('source_quality_passed') is not False
            or payload.get('experimental_preview') is not True
            or payload.get('gameplay_qualified') is not False
            or payload.get('automatic_promotion') is not False):
        raise ValueError('Expected an explicitly unqualified conditional-restorer format4 bundle')
    parent = payload.get('parent_bundle')
    _parent_state(parent)
    if (payload.get('parent_bundle_sha256') != parent_bundle_fingerprint(parent)
            or payload.get('config') != parent['config']
            or payload.get('source_teacher_config') != parent['source_teacher_config']
            or any(payload.get(key) != value for key, value in _clock_fields(parent).items())
            or payload.get('solver') != parent['solver']
            or type(payload.get('solver_steps')) is not int or payload['solver_steps'] != 3
            or payload.get('solver_profile') != parent['solver_profile']
            or payload.get('base_student_identity') != parent['base_student_identity']
            or payload.get('frozen_parent_flow_sha256') != parent['adapted_flow_sha256']
            or payload.get('frozen_parent_codec_sha256') != parent['adapted_codec_sha256']
            or type(payload.get('restorer_updates')) is not int or payload['restorer_updates'] < 0
            or type(payload.get('restorer_refinement_updates')) is not int
            or payload.get('restorer_refinement_updates') != payload['restorer_updates']):
        raise ValueError('Conditional restorer changed its parent, frozen clocks or solver')
    state = payload.get('restorer_state')
    if (not isinstance(state, dict) or not all(isinstance(value, torch.Tensor)
            and bool(torch.isfinite(value).all()) for value in state.values())
            or payload.get('restorer_state_sha256') != _hash_tensors(state)):
        raise ValueError('Conditional restorer needs complete finite hash-bound weights')
    module = load_conditional_decoder_restorer(payload.get('restorer_config'), state,
                                               metadata=payload.get('restorer_metadata'))
    private = _private_teacher_copy(teacher)
    load_decoder_adaptation_payload(private, parent,
        checkpoint=payload.get('parent_source_checkpoint') or '<decoder-parent>',
        require_trained=require_trained, allow_unqualified_codec=allow_unqualified_codec)
    _flow_contract(private, parent)
    parameter = private.flow_renderer.image_input.weight
    module.to(device=parameter.device, dtype=parameter.dtype).eval()
    if conditional_decoder_restorer_metadata(module) != payload['restorer_metadata']:
        raise ValueError('Restorer loaded precision or module state differs from its strict metadata')
    private.flow_renderer.conditional_restorer = module
    _flow_contract(private, parent)
    private.flow_renderer._renderer_conditional_restorer = {
        key: deepcopy(value) for key, value in payload.items() if key not in {'parent_bundle', 'restorer_state'}}
    private.flow_renderer._renderer_restorer_parent_bundle = deepcopy(parent)
    if hasattr(teacher, 'base_student'):
        if _hash_tensors(teacher.base_student.state_dict()) != parent['base_student_identity']:
            raise ValueError('Already loaded Base differs from the restorer parent')
        private.base_student = teacher.base_student
    metadata = deepcopy(private.flow_renderer._renderer_conditional_restorer)
    metadata['checkpoint'] = str(Path(checkpoint).resolve())
    teacher.flow_renderer = private.flow_renderer
    teacher.base_student = private.base_student
    teacher.base_student_metadata = private.base_student_metadata
    teacher.base_student_bundle = private.base_student_bundle
    teacher.config = private.config
    teacher.renderer_student_metadata = metadata
    return metadata
