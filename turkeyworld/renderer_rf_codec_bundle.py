"""RF-only format5 overlay on the exact adapted-decoder format3 parent."""
from copy import deepcopy
from pathlib import Path

import torch

from .decoder_rf_adaptation_bundle import load_decoder_adaptation_payload, _spec
from .renderer_rf5_bundle import (
    _atomic_save, _parent_input, _private_teacher_copy, _sha, parent_bundle_fingerprint)
from .renderer_student import _hash_tensors, renderer_condition_identity


FORMAT_VERSION = 5
STAGE = 'renderer_rf_codec_geometry'
KIND = 'renderer_rf_codec_geometry_bundle_v1'
PROTOCOL = 'renderer_rf_codec_geometry_v1'
FROZEN_PREFIXES = ('codec.', 'geometry_opacity.')


def _parent_flow(parent):
    from .conditional_restorer_bundle import _parent_state
    return _parent_state(parent)


def _names(flow):
    if (getattr(flow, 'conditional_restorer', None) is not None
            or any('conditional_restorer' in name for name,_ in flow.named_modules())):
        raise ValueError('RF/codec geometry bundles must not contain a restorer')
    return sorted(name for name, _ in flow.named_parameters()
                  if not name.startswith(FROZEN_PREFIXES))


def _frozen(state, names):
    allowed = set(names)
    return {name: value for name, value in state.items() if name not in allowed}


def _contract(model, parent):
    flow, source = model.flow_renderer, _parent_flow(parent)
    names = _names(flow)
    state = flow.state_dict()
    raw = getattr(flow, '_renderer_student_config', None)
    dynamic = ('rssm_', 'vjepa_', 'enemy_rssm_', 'combat_rssm_')
    frozen_config = lambda cfg: {k:v for k,v in cfg.items()
        if not k.startswith(dynamic) and k not in {'architecture', 'player_motion_mode'}}
    if (flow.kind != 'student' or flow.mode != 'rectified_flow' or flow.steps != 3
            or not flow.uses_wall_conditioning or not isinstance(raw, dict)
            or frozen_config(raw) != frozen_config(parent['config'])
            or set(state) != set(source) or not names or any(n not in source for n in names)
            or _hash_tensors(_frozen(state, names)) != _hash_tensors(_frozen(source, names))
            or _hash_tensors(flow.codec.state_dict()) != parent['adapted_codec_sha256']
            or not hasattr(model, 'base_student')
            or _hash_tensors(model.base_student.state_dict()) != parent['base_student_identity']
            or getattr(flow, '_renderer_base_identity', None) != parent['base_student_identity']):
        raise ValueError('RF-only continuation changed codec/D/E/Q/Base/private geometry/noise/normalization')
    return state, names


def export_rf_codec_renderer(model, parent_bundle, path, *, local_updates,
                             source_model=None, metrics=None, provenance=None):
    """Only RF named parameters may differ; all decoder-parent clocks stay explicit."""
    parent, source_sha, source_path = _parent_input(parent_bundle, provenance)
    if type(local_updates) is not int or local_updates < 0:
        raise ValueError('RF continuation needs a nonnegative completed-update count')
    state, names = _contract(model, parent)
    source = _parent_flow(parent)
    override = {name:state[name].detach().cpu().clone() for name in names}
    if (not all(bool(torch.isfinite(v).all()) for v in override.values())
            or _spec(override) != _spec({name:source[name] for name in names})):
        raise ValueError('RF override needs all finite exact-shaped named RF parameters')
    private = _private_teacher_copy(model if source_model is None else source_model)
    load_decoder_adaptation_payload(private, parent, checkpoint=source_path or '<D1725-parent>',
        allow_unqualified_codec=getattr(model.flow_renderer, '_renderer_allow_unqualified_codec', False))
    if renderer_condition_identity(private) != renderer_condition_identity(model):
        raise ValueError('RF continuation changed the validated Source/map/Base/codec conditioning')
    payload = dict(format_version=FORMAT_VERSION, stage=STAGE, kind=KIND,
        objective_protocol=PROTOCOL, parent_bundle=deepcopy(parent),
        parent_bundle_sha256=parent_bundle_fingerprint(parent),
        parent_source_sha256=source_sha, parent_source_checkpoint=source_path,
        config=deepcopy(parent['config']), source_teacher_config=deepcopy(parent['source_teacher_config']),
        solver='euler', solver_steps=3, solver_profile=deepcopy(parent['solver_profile']),
        parent_renderer_training_steps=parent['training_steps'],
        training_steps=parent['training_steps']+local_updates,
        renderer_training_steps=parent['training_steps']+local_updates,
        local_updates=local_updates, refinement_updates=local_updates,
        parent_refinement_updates=parent['refinement_updates'],
        decoder_refinement_updates=parent['decoder_refinement_updates'],
        base_training_steps=parent['base_training_steps'],
        rf_parameter_names=['flow_renderer.'+name for name in names],
        rf_override=override, rf_override_spec=_spec(override), rf_override_sha256=_hash_tensors(override),
        frozen_flow_sha256=_hash_tensors(_frozen(source, names)),
        model_sha256=_hash_tensors(state), codec_sha256=parent['adapted_codec_sha256'],
        base_student_identity=parent['base_student_identity'], restorer_present=False,
        Source_future_inputs=False, codec_frozen=True, decoder_frozen=True,
        source_quality_passed=False, gameplay_qualified=False, automatic_promotion=False,
        experimental_preview=True, metrics=deepcopy(metrics or {}), provenance=deepcopy(provenance or {}))
    _atomic_save(payload, path)
    return {k:v for k,v in payload.items() if k not in {'parent_bundle', 'rf_override'}}


def load_rf_codec_renderer_payload(teacher, payload, *, checkpoint,
                                   require_trained=True, allow_unqualified_codec=False):
    if (not isinstance(payload, dict) or payload.get('format_version') != FORMAT_VERSION
            or payload.get('stage') != STAGE or payload.get('kind') != KIND
            or payload.get('objective_protocol') != PROTOCOL
            or not _sha(payload.get('parent_source_sha256'))
            or payload.get('restorer_present') is not False
            or payload.get('Source_future_inputs') is not False
            or payload.get('codec_frozen') is not True or payload.get('decoder_frozen') is not True
            or payload.get('source_quality_passed') is not False
            or payload.get('gameplay_qualified') is not False
            or payload.get('automatic_promotion') is not False
            or payload.get('experimental_preview') is not True):
        raise ValueError('Expected an explicitly unqualified no-restorer RF/codec format5 bundle')
    parent = payload.get('parent_bundle')
    source = _parent_flow(parent)
    local = payload.get('local_updates')
    if (type(local) is not int or local < 0
            or payload.get('parent_bundle_sha256') != parent_bundle_fingerprint(parent)
            or payload.get('config') != parent['config']
            or payload.get('source_teacher_config') != parent['source_teacher_config']
            or payload.get('solver') != 'euler' or type(payload.get('solver_steps')) is not int
            or payload['solver_steps'] != 3 or payload.get('solver_profile') != parent['solver_profile']
            or any(type(payload.get(k)) is not int or payload[k] != parent['training_steps']+local
                   for k in ('training_steps', 'renderer_training_steps'))
            or payload.get('parent_renderer_training_steps') != parent['training_steps']
            or payload.get('refinement_updates') != local
            or payload.get('parent_refinement_updates') != parent['refinement_updates']
            or payload.get('decoder_refinement_updates') != parent['decoder_refinement_updates']
            or payload.get('base_training_steps') != parent['base_training_steps']
            or payload.get('base_student_identity') != parent['base_student_identity']
            or payload.get('codec_sha256') != parent['adapted_codec_sha256']):
        raise ValueError('RF/codec bundle changed frozen parent identity, solver or independent clocks')
    private = _private_teacher_copy(teacher)
    load_decoder_adaptation_payload(private, parent,
        checkpoint=payload.get('parent_source_checkpoint') or '<D1725-parent>',
        require_trained=require_trained, allow_unqualified_codec=allow_unqualified_codec)
    flow = private.flow_renderer
    names = _names(flow)
    override = payload.get('rf_override')
    expected = {n:source[n] for n in names}
    if (payload.get('rf_parameter_names') != ['flow_renderer.'+n for n in names]
            or not isinstance(override, dict) or set(override) != set(names)
            or not all(torch.is_tensor(v) and bool(torch.isfinite(v).all()) for v in override.values())
            or _spec(override) != _spec(expected) or payload.get('rf_override_spec') != _spec(expected)
            or payload.get('rf_override_sha256') != _hash_tensors(override)
            or payload.get('frozen_flow_sha256') != _hash_tensors(_frozen(source, names))):
        raise ValueError('RF override changes its exact named-parameter allowlist, spec or state hash')
    # The legacy missing-normalization compatibility hook inserts mean0/std1
    # for a partial RF-only state. Preserve the complete validated parent when
    # applying named RF parameters, including its non-unit normalization.
    flow.load_state_dict({**flow.state_dict(),**override}, strict=True)
    _contract(private, parent)
    if _hash_tensors(flow.state_dict()) != payload.get('model_sha256'):
        raise ValueError('RF/codec assembled full-flow hash differs')
    flow._check_codec()
    flow.eval()
    metadata = {k:deepcopy(v) for k,v in payload.items() if k not in {'parent_bundle', 'rf_override'}}
    metadata['checkpoint'] = str(Path(checkpoint).resolve())
    flow._renderer_rf_codec = deepcopy(metadata)
    flow._renderer_rf_codec_parent_bundle = deepcopy(parent)
    if hasattr(teacher, 'base_student'):
        if _hash_tensors(teacher.base_student.state_dict()) != parent['base_student_identity']:
            raise ValueError('Existing compact Base differs from the RF/codec parent')
        private.base_student = teacher.base_student
    teacher.flow_renderer = flow
    teacher.base_student = private.base_student
    teacher.base_student_metadata = private.base_student_metadata
    teacher.base_student_bundle = private.base_student_bundle
    teacher.config = private.config
    teacher.renderer_student_metadata = metadata
    return metadata
