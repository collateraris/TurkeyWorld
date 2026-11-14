"""Explicit RF3/decoder-chain + RGB detail SR on a private current-world clone.

The training model, old migration records, Adam and all sampler/RNG streams
remain untouched. Every renderer loader still validates its original lineage;
the selected image bundle is loaded once into a pristine training-model copy.
"""
from copy import deepcopy
from dataclasses import asdict

import torch

from .rssm_preview_assets import preserve_preview_rng,synchronize_preview_dynamics


PROFILE='rf3_decoder_chain_background_detail_preview_v1'
MIGRATION_NAMES=('rssm_migration','dual_rssm_migration','enemy_rssm_resize','rssm_fusion_migration',
    'combat_rssm_migration','rssm_five_migration','rssm_five_expanded_migration',
    'rssm_spawn_conditioning_migration')


def validate_image_chain_plan(plan):
    renderer=plan.get('renderer',{});sr=plan.get('sr',{})
    if(plan.get('inference_profile')!=PROFILE or plan.get('preview_only')is not True
            or plan.get('training_assets_changed')is not False or plan.get('game_assets_changed')is not False
            or plan.get('allow_unqualified_codec')is not True
            or renderer.get('format_version')!=8 or renderer.get('kind')!='decoder_rf_chain_bundle_v1'
            or renderer.get('solver')!='euler' or renderer.get('solver_steps')!=3
            or renderer.get('restorer_present')is not False
            or type(renderer.get('training_steps'))is not int or renderer['training_steps']<=0
            or type(renderer.get('base_training_steps'))is not int or renderer['base_training_steps']<=0
            or type(renderer.get('decoder_refinement_updates'))is not int or renderer['decoder_refinement_updates']<=0
            or sr.get('architecture')!='background_detail_sr' or sr.get('inference_inputs')!='RGB32_only'
            or sr.get('alpha_used')is not False or sr.get('atlas_used')is not False
            or sr.get('hypothesis_used')is not False or sr.get('input_size')!=32 or sr.get('output_size')!=128
            or type(sr.get('training_steps'))is not int or sr['training_steps']<=0
            or sr.get('selected_local_update')!=sr['training_steps']):
        raise ValueError('RF3/decoder/background-detail preview requires its explicit complete image-only plan')
    return renderer,sr


def prepare_image_chain_preview_assets(model,plan,device,*,clone_factory=None):
    from .models import NeuralWorldModel,WorldModelConfig,load_sr_checkpoint
    from .renderer_student import _hash_tensors,load_renderer_student
    from .background_detail_sr import background_detail_metadata
    requested_renderer,requested_sr=validate_image_chain_plan(plan)
    if model.config.image_size!=32:raise ValueError('Selected RGB detail SR requires native32 current dynamics')
    device=torch.device(device);factory=clone_factory or(lambda config:NeuralWorldModel(WorldModelConfig(**asdict(config))))
    source_hash=_hash_tensors(model.state_dict());parameter_ids=tuple(id(p)for p in model.parameters())
    with preserve_preview_rng(device):
        client=factory(model.config);client.load_state_dict(model.state_dict(),strict=True)
        for name in MIGRATION_NAMES:
            if hasattr(model,name):setattr(client,name,deepcopy(getattr(model,name)))
        # Do not reuse the previous inference clone: its compact codec/Base are
        # descendants, while migration proofs refer to the raw training assets.
        renderer=load_renderer_student(client,requested_renderer['checkpoint'],allow_unqualified_codec=True)
        if(any(renderer.get(key)!=requested_renderer.get(key)for key in
                ('format_version','kind','solver','solver_steps','training_steps','base_training_steps','decoder_refinement_updates'))
                or renderer.get('restorer_present')is not False
                or client.flow_renderer.kind!='student' or client.flow_renderer.steps!=3
                or getattr(client.flow_renderer,'conditional_restorer',None)is not None
                or not isinstance(getattr(client.flow_renderer,'_renderer_decoder_chain',None),dict)):
            raise ValueError('Selected preview is not the exact no-restorer format8 RF3 decoder chain')
        sr,sr_metadata=load_sr_checkpoint(requested_sr['checkpoint'],'cpu')
        metadata=background_detail_metadata(sr)
        if(sr_metadata.get('training_steps')!=requested_sr['training_steps']
                or sr_metadata.get('background_detail_updates')!=requested_sr['selected_local_update']
                or metadata.get('architecture')!='background_detail_sr'
                or metadata.get('frozen_background_sha256')!=requested_sr.get('preferred_background_sha256')
                or metadata.get('alpha_used')is not False or metadata.get('atlas_class_used')is not False
                or metadata.get('atlas_buffers_present')is not False or metadata.get('atlas_asset_required')is not False
                or metadata.get('hypothesis_used')is not False
                or any(key in sr_metadata for key in('sr_moe_binding','atlas_rgba','atlas_asset'))):
            raise ValueError('Preview SR differs from the pinned trained standalone RGB background/detail model')
        if not hasattr(client,'base_student')or client.base_student_bundle.get('training_steps')!=requested_renderer['base_training_steps']:
            raise ValueError('Selected preview expanded Base clock differs from the decoder chain')
        client.to(device).eval().requires_grad_(False);sr.to(device).eval().requires_grad_(False)
        copied=synchronize_preview_dynamics(client,model)
    if _hash_tensors(model.state_dict())!=source_hash or tuple(id(p)for p in model.parameters())!=parameter_ids:
        raise AssertionError('Image-only preview clone mutated current training model/parameter identities')
    proof=dict(kind='frozen_preview_assets_v1',inference_profile=PROFILE,plan_path=plan['plan_path'],
        plan_sha256=plan['plan_sha256'],renderer=deepcopy(plan['renderer']),sr=deepcopy(plan['sr']),
        renderer_training_steps=renderer['training_steps'],renderer_refinement_updates=renderer.get('refinement_updates'),
        renderer_solver_steps=3,renderer_format_version=8,base_training_steps=renderer['base_training_steps'],
        decoder_refinement_updates=renderer['decoder_refinement_updates'],restorer_present=False,
        sr_training_steps=sr_metadata['training_steps'],sr_refinement_updates=sr_metadata['background_detail_updates'],
        sr_architecture='background_detail_sr',sr_input_kind='generated_current_world_RGB32_only',
        sr_hypothesis_used=False,sr_alpha_used=False,sr_atlas_used=False,
        sr_source_quality_passed=sr_metadata.get('source_quality_passed',False),
        codec_quality_qualified=False,renderer_source_quality_passed=False,renderer_release_passed=False,
        gameplay_qualified=False,all_preview_parameters_frozen=True,training_model_tensors_bitexact=True,
        training_parameter_ids_preserved=True,training_optimizer_untouched=True,current_world_tensors_copied=copied,
        training_model_tensor_sha256=source_hash,source_future_feedback=False,automatic_active_promotion=False,
        verification_scope='Raw current-dynamics clone; image bundles replaced only privately; current dynamics copied on every preview')
    return client,sr,sr_metadata,proof
