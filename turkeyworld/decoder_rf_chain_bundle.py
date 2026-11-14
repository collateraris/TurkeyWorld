"""Format8: decoder/post-quant overlay over the exact selected RF/Base chain."""
from copy import deepcopy
from pathlib import Path

import torch

from .decoder_rf_adaptation_bundle import (
    DECODER_PREFIXES, _decoder, _frozen, _encoder_quantizer, _spec,
    _flow_contract as _decoder_contract, _check_override, _apply_override)
from .renderer_rf5_bundle import _atomic_save,_parent_input,_private_teacher_copy,_sha,parent_bundle_fingerprint
from .renderer_student import _hash_tensors,renderer_condition_identity


FORMAT_VERSION=8
STAGE='decoder_rf_chain_adaptation'
KIND='decoder_rf_chain_bundle_v1'
PROTOCOL='decoder_rf_source_paired_adaptation_v1'


def _parent(parent):
    from .renderer_rf_chain_bundle import FORMAT_VERSION as RF_VERSION,KIND as RF_KIND,rf_chain_state
    if (not isinstance(parent,dict)or parent.get('format_version')!=RF_VERSION
            or parent.get('kind')!=RF_KIND or parent.get('solver')!='euler'
            or type(parent.get('solver_steps'))is not int or parent['solver_steps']!=3
            or parent.get('restorer_present')is not False):
        raise ValueError('Decoder chain requires its strict no-restorer selected RF3 format7 parent')
    state=rf_chain_state(parent)
    return {**parent,'model':state}


def _contract(model,parent):
    if getattr(model.flow_renderer,'conditional_restorer',None)is not None:
        raise ValueError('Decoder chain must not attach a restorer')
    return _decoder_contract(model,_parent(parent))


def export_decoder_rf_chain(model,parent_bundle,path,*,local_updates,source_model=None,metrics=None,provenance=None):
    parent,source_sha,source_path=_parent_input(parent_bundle,provenance)
    view=_parent(parent)
    if type(local_updates)is not int or local_updates<0:raise ValueError('Decoder chain local updates must be nonnegative')
    state=_contract(model,parent)
    override={name:value.detach().cpu().clone()for name,value in _decoder(state).items()}
    _check_override(override,view)
    from .renderer_rf_chain_bundle import load_rf_chain_payload
    private=_private_teacher_copy(model if source_model is None else source_model)
    load_rf_chain_payload(private,parent,checkpoint=source_path or '<selected-RF-parent>',
        allow_unqualified_codec=getattr(model.flow_renderer,'_renderer_allow_unqualified_codec',False))
    before,after=renderer_condition_identity(private),renderer_condition_identity(model)
    if any(after[key]!=value for key,value in before.items()if key!='codec_sha256'):
        raise ValueError('Decoder chain changed the frozen Source/map/appearance/noise conditioning')
    parent_updates=parent['decoder_refinement_updates']
    payload=dict(format_version=FORMAT_VERSION,stage=STAGE,kind=KIND,objective_protocol=PROTOCOL,
        parent_bundle=deepcopy(parent),parent_bundle_sha256=parent_bundle_fingerprint(parent),
        parent_source_sha256=source_sha,parent_source_checkpoint=source_path,
        config=deepcopy(parent['config']),source_teacher_config=deepcopy(parent['source_teacher_config']),
        solver='euler',solver_steps=3,solver_profile=deepcopy(parent['solver_profile']),
        training_steps=parent['training_steps'],renderer_training_steps=parent['training_steps'],
        parent_renderer_training_steps=parent['training_steps'],refinement_updates=parent['refinement_updates'],
        rf_chain_local_updates=parent['rf_chain_local_updates'],
        base_training_steps=parent['base_training_steps'],base_refinement_updates=parent['base_refinement_updates'],
        decoder_parent_updates=parent_updates,decoder_local_updates=local_updates,local_updates=local_updates,
        decoder_refinement_updates=parent_updates+local_updates,
        decoder_override=override,decoder_override_sha256=_hash_tensors(override),decoder_override_spec=_spec(override),
        allowed_prefixes=list(DECODER_PREFIXES),frozen_flow_sha256=_hash_tensors(_frozen(state)),
        encoder_quantizer_sha256=_hash_tensors(_encoder_quantizer(state)),adapted_flow_sha256=_hash_tensors(state),
        adapted_codec_sha256=_hash_tensors(model.flow_renderer.codec.state_dict()),
        codec_config=deepcopy(model.flow_renderer.codec.get_config()),base_student_identity=parent['base_student_identity'],
        source_codec_reference='unchanged_selected_RF3_parent_decoder',restorer_present=False,
        decoder_quality_qualified=False,source_quality_passed=False,
        original_codec_qualification_applies_to_new_decoder=False,experimental_preview=True,
        gameplay_qualified=False,automatic_promotion=False,Source_future_inputs=False,
        metrics=deepcopy(metrics or {}),provenance=deepcopy(provenance or {}))
    _atomic_save(payload,path)
    return {key:value for key,value in payload.items()if key not in {'parent_bundle','decoder_override'}}


def load_decoder_rf_chain_payload(teacher,payload,*,checkpoint,require_trained=True,allow_unqualified_codec=False):
    if (not isinstance(payload,dict)or payload.get('format_version')!=FORMAT_VERSION
            or payload.get('stage')!=STAGE or payload.get('kind')!=KIND
            or payload.get('objective_protocol')!=PROTOCOL or not _sha(payload.get('parent_source_sha256'))
            or payload.get('allowed_prefixes')!=list(DECODER_PREFIXES)
            or any(payload.get(key)is not False for key in ('restorer_present','decoder_quality_qualified',
                'source_quality_passed','original_codec_qualification_applies_to_new_decoder',
                'gameplay_qualified','automatic_promotion','Source_future_inputs'))
            or payload.get('experimental_preview')is not True
            or payload.get('source_codec_reference')!='unchanged_selected_RF3_parent_decoder'):
        raise ValueError('Expected an explicitly unqualified no-restorer decoder chain format8 bundle')
    parent=payload.get('parent_bundle');view=_parent(parent)
    local=payload.get('decoder_local_updates')
    if (type(local)is not int or local<0 or payload.get('local_updates')!=local
            or payload.get('parent_bundle_sha256')!=parent_bundle_fingerprint(parent)
            or payload.get('config')!=parent['config']or payload.get('source_teacher_config')!=parent['source_teacher_config']
            or payload.get('solver')!='euler' or type(payload.get('solver_steps'))is not int or payload['solver_steps']!=3
            or payload.get('solver_profile')!=parent['solver_profile']
            or any(type(payload.get(key))is not int or payload[key]!=parent['training_steps']
                   for key in ('training_steps','renderer_training_steps','parent_renderer_training_steps'))
            or payload.get('refinement_updates')!=parent['refinement_updates']
            or payload.get('rf_chain_local_updates')!=parent['rf_chain_local_updates']
            or payload.get('base_training_steps')!=parent['base_training_steps']
            or payload.get('base_refinement_updates')!=parent['base_refinement_updates']
            or payload.get('decoder_parent_updates')!=parent['decoder_refinement_updates']
            or payload.get('decoder_refinement_updates')!=parent['decoder_refinement_updates']+local
            or payload.get('base_student_identity')!=parent['base_student_identity']):
        raise ValueError('Decoder chain changed its exact selected parent, solver or independent clocks')
    override=payload.get('decoder_override');_check_override(override,view)
    if (payload.get('decoder_override_sha256')!=_hash_tensors(override)
            or payload.get('decoder_override_spec')!=_spec(override)
            or payload.get('frozen_flow_sha256')!=_hash_tensors(_frozen(view['model']))
            or payload.get('encoder_quantizer_sha256')!=_hash_tensors(_encoder_quantizer(view['model']))):
        raise ValueError('Decoder chain overlay or frozen RF/E/Q fingerprints differ')
    from .renderer_rf_chain_bundle import load_rf_chain_payload
    private=_private_teacher_copy(teacher)
    load_rf_chain_payload(private,parent,checkpoint=payload.get('parent_source_checkpoint')or'<selected-RF-parent>',
        require_trained=require_trained,allow_unqualified_codec=allow_unqualified_codec)
    flow=private.flow_renderer
    if payload.get('codec_config')!=flow.codec.get_config():raise ValueError('Decoder chain changed codec geometry')
    _apply_override(flow,override);_contract(private,parent)
    if (_hash_tensors(flow.state_dict())!=payload.get('adapted_flow_sha256')
            or _hash_tensors(flow.codec.state_dict())!=payload.get('adapted_codec_sha256')):
        raise ValueError('Decoder chain assembled full-flow/codec fingerprints differ')
    flow._check_codec();flow.eval()
    metadata={key:deepcopy(value)for key,value in payload.items()if key not in {'parent_bundle','decoder_override'}}
    metadata['checkpoint']=str(Path(checkpoint).resolve())
    flow._renderer_decoder_chain=deepcopy(metadata);flow._renderer_decoder_chain_parent_bundle=deepcopy(parent)
    if hasattr(teacher,'base_student'):
        if _hash_tensors(teacher.base_student.state_dict())!=parent['base_student_identity']:
            raise ValueError('Existing expanded Base differs from the selected decoder-chain parent')
        private.base_student=teacher.base_student
    teacher.flow_renderer=flow;teacher.base_student=private.base_student
    teacher.base_student_metadata=private.base_student_metadata;teacher.base_student_bundle=private.base_student_bundle
    teacher.config=private.config;teacher.renderer_student_metadata=metadata
    return metadata
