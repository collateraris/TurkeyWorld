"""Explicit expanded-Base binding over an unchanged RF3/current-codec parent."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import torch

from .base_student import BaseStudentConfig, prepare_base_student
from .conditional_restorer_bundle import _parent_state
from .decoder_rf_adaptation_bundle import load_decoder_adaptation_payload
from .renderer_moe import output_moe_metadata, renderer_moe_metadata, validate_renderer_moe_bundle
from .renderer_rf5_bundle import (
    _atomic_save, _parent_input, _private_teacher_copy, _sha, parent_bundle_fingerprint)
from .renderer_student import _hash_tensors, renderer_condition_identity

FORMAT_VERSION = 6
STAGE = 'base_expanded_geometry'
KIND = 'base_expanded_geometry_bundle_v1'
PROTOCOL = 'base_expanded_geometry_v1'


def _flow_contract(model, parent):
    state = _parent_state(parent)
    flow = model.flow_renderer
    raw = getattr(flow,'_renderer_student_config',None)
    dynamic = ('rssm_','vjepa_','enemy_rssm_','combat_rssm_')
    frozen_config = lambda cfg:{k:v for k,v in cfg.items()
        if not k.startswith(dynamic) and k not in {'architecture','player_motion_mode'}}
    if (flow.kind != 'student' or flow.mode != 'rectified_flow' or flow.steps != 3
            or not isinstance(raw,dict) or frozen_config(raw) != frozen_config(parent['config'])
            or getattr(flow, 'conditional_restorer', None) is not None
            or set(flow.state_dict()) != set(state)
            or _hash_tensors(flow.state_dict()) != parent['adapted_flow_sha256']
            or _hash_tensors(flow.codec.state_dict()) != parent['adapted_codec_sha256']):
        raise ValueError('Base phase must preserve every RF/codec/private-geometry/noise/normalization tensor')


def _base_bundle(head, original, updates):
    config = asdict(head.config)
    old = asdict(BaseStudentConfig(**original['config']))
    if (config.get('output_moe_expanded_enabled') is not True
            or {k:v for k,v in config.items() if k != 'output_moe_expanded_enabled'} !=
               {k:v for k,v in old.items() if k != 'output_moe_expanded_enabled'}):
        raise ValueError('Only the explicitly expanded Base-expert configuration may change')
    state = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
    if not all(bool(torch.isfinite(v).all()) for v in state.values()):
        raise ValueError('Expanded Base requires complete finite tensors')
    result = deepcopy(original)
    result.update(config=config, model=state, model_sha256=_hash_tensors(state),
        training_steps=original['training_steps']+updates,
        head_parameters=sum(p.numel() for p in head.parameters()), output_moe=output_moe_metadata(head),
        quality_status='Expanded Base geometry experiment; RF not adapted to its changed RGB condition')
    return result


def _binding(parent, base):
    return dict(kind='explicit_expanded_Base_condition_binding_v1',
        parent_base_identity=parent['base_student_identity'], new_base_identity=base['model_sha256'],
        frozen_RF_identity=parent['adapted_flow_sha256'], RF_weights_changed=False,
        RF_adapted_to_new_Base=False, RF_quality_qualified=False, automatic_promotion=False)


def export_base_geometry(model, parent_bundle, path, *, local_updates,
                          source_model=None, metrics=None, provenance=None):
    parent, source_sha, source_path = _parent_input(parent_bundle, provenance)
    if type(local_updates) is not int or local_updates < 0:
        raise ValueError('Base phase needs a nonnegative completed-update count')
    _flow_contract(model, parent)
    pristine = _private_teacher_copy(model if source_model is None else source_model)
    private = _private_teacher_copy(pristine)
    load_decoder_adaptation_payload(private,parent,checkpoint=source_path or '<D-parent>',
        allow_unqualified_codec=getattr(model.flow_renderer,'_renderer_allow_unqualified_codec',False))
    if renderer_condition_identity(model) != renderer_condition_identity(private):
        raise ValueError('Expanded Base changed its frozen world/map/appearance/codec inputs')
    base = _base_bundle(model.base_student,private.base_student_bundle,local_updates)
    # Source lineage is checked against the original world, before replacing its
    # codec with D1725. This avoids reinterpreting a descendant as its teacher.
    prepared,_,_ = prepare_base_student(pristine,base)
    header = renderer_moe_metadata(model.flow_renderer,prepared)
    validate_renderer_moe_bundle(model.flow_renderer,base,header)
    payload = dict(format_version=FORMAT_VERSION,stage=STAGE,kind=KIND,objective_protocol=PROTOCOL,
        parent_bundle=deepcopy(parent),parent_bundle_sha256=parent_bundle_fingerprint(parent),
        parent_source_sha256=source_sha,parent_source_checkpoint=source_path,
        config=deepcopy(parent['config']),source_teacher_config=deepcopy(parent['source_teacher_config']),
        solver='euler',solver_steps=3,solver_profile=deepcopy(parent['solver_profile']),
        training_steps=parent['training_steps'],renderer_training_steps=parent['training_steps'],
        refinement_updates=parent['refinement_updates'],
        decoder_refinement_updates=parent['decoder_refinement_updates'],
        parent_base_training_steps=private.base_student_bundle['training_steps'],
        base_training_steps=base['training_steps'],base_refinement_updates=local_updates,local_updates=local_updates,
        base_student_bundle=base,base_student_identity=base['model_sha256'],
        parent_base_student_identity=parent['base_student_identity'],renderer_output_moe=header,
        frozen_flow_sha256=parent['adapted_flow_sha256'],codec_sha256=parent['adapted_codec_sha256'],
        RF_base_condition_binding=_binding(parent,base),restorer_present=False,
        source_quality_passed=False,Base_quality_qualified=False,RF_quality_qualified=False,
        gameplay_qualified=False,automatic_promotion=False,Source_future_inputs=False,
        metrics=deepcopy(metrics or {}),provenance=deepcopy(provenance or {}))
    _atomic_save(payload,path)
    return {k:v for k,v in payload.items() if k not in {'parent_bundle','base_student_bundle'}}


def load_base_geometry_payload(teacher,payload,*,checkpoint,require_trained=True,allow_unqualified_codec=False):
    if (not isinstance(payload,dict) or payload.get('format_version') != FORMAT_VERSION
            or payload.get('stage') != STAGE or payload.get('kind') != KIND
            or payload.get('objective_protocol') != PROTOCOL or not _sha(payload.get('parent_source_sha256'))
            or any(payload.get(k) is not False for k in ('restorer_present','source_quality_passed',
                'Base_quality_qualified','RF_quality_qualified','gameplay_qualified','automatic_promotion','Source_future_inputs'))):
        raise ValueError('Expected an explicitly unqualified expanded-Base format6 bundle')
    parent = payload.get('parent_bundle');_parent_state(parent)
    local = payload.get('local_updates')
    if (type(local) is not int or local < 0
            or payload.get('parent_bundle_sha256') != parent_bundle_fingerprint(parent)
            or payload.get('config') != parent['config']
            or payload.get('source_teacher_config') != parent['source_teacher_config']
            or payload.get('solver') != 'euler' or type(payload.get('solver_steps')) is not int
            or payload['solver_steps'] != 3 or payload.get('solver_profile') != parent['solver_profile']
            or any(payload.get(k) != parent['training_steps'] for k in ('training_steps','renderer_training_steps'))
            or payload.get('refinement_updates') != parent['refinement_updates']
            or payload.get('decoder_refinement_updates') != parent['decoder_refinement_updates']
            or payload.get('parent_base_student_identity') != parent['base_student_identity']
            or payload.get('frozen_flow_sha256') != parent['adapted_flow_sha256']
            or payload.get('codec_sha256') != parent['adapted_codec_sha256']
            or payload.get('base_refinement_updates') != local):
        raise ValueError('Expanded Base changed its exact parent, RF/D clocks, solver or frozen flow')
    # Prepare the new Base while the pristine teacher lineage is still intact.
    base = payload.get('base_student_bundle')
    head,base_meta,base = prepare_base_student(teacher,base)
    private = _private_teacher_copy(teacher)
    load_decoder_adaptation_payload(private,parent,checkpoint=payload.get('parent_source_checkpoint') or '<D-parent>',
        require_trained=require_trained,allow_unqualified_codec=allow_unqualified_codec)
    canonical = _base_bundle(head,private.base_student_bundle,local)
    if parent_bundle_fingerprint(base) != parent_bundle_fingerprint(canonical):
        raise ValueError('Expanded Base contains fields outside its allowed parent continuation')
    if (payload.get('parent_base_training_steps') != private.base_student_bundle['training_steps']
            or payload.get('base_training_steps') != canonical['training_steps']
            or payload.get('base_student_identity') != canonical['model_sha256']
            or payload.get('RF_base_condition_binding') != _binding(parent,canonical)):
        raise ValueError('Expanded Base clock/hash or explicit unadapted RF binding differs')
    validate_renderer_moe_bundle(private.flow_renderer,canonical,payload.get('renderer_output_moe'))
    _flow_contract(private,parent)
    private.base_student=head;private.base_student_metadata=base_meta;private.base_student_bundle=canonical
    private.flow_renderer._renderer_base_identity=canonical['model_sha256']
    private.flow_renderer._renderer_base_bundle=deepcopy(canonical)
    private.flow_renderer._renderer_Base_RF_unadapted_binding=deepcopy(payload['RF_base_condition_binding'])
    metadata={k:deepcopy(v) for k,v in payload.items() if k not in {'parent_bundle','base_student_bundle'}}
    metadata['checkpoint']=str(Path(checkpoint).resolve())
    private.flow_renderer._renderer_base_geometry=deepcopy(metadata)
    teacher.flow_renderer=private.flow_renderer;teacher.base_student=head
    teacher.base_student_metadata=base_meta;teacher.base_student_bundle=canonical;teacher.config=private.config
    teacher.renderer_student_metadata=metadata
    return metadata
