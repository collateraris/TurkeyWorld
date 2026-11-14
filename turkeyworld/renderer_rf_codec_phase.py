"""Exact full-Adam RF3 continuation with a separately pinned frozen decoder.

The current codec may differ from the RF parent ONLY in D/post_quant_conv.
This module neither trains a restorer nor changes any optimizer/sampler clock.
"""
from copy import deepcopy
import math

import torch

from .renderer_native_phase import _same
from .renderer_refinement import frozen_fingerprint
from .renderer_rf5_experts import rf5_expert_metadata
from .renderer_rf5_phase import _canonical_names
from .renderer_student import _hash_tensors


PROTOCOL = 'renderer_rf_codec_geometry_v1'
PARENT_PROTOCOL = 'renderer_rf_expanded_experts_v1'
FROZEN_FLOW_PREFIXES = ('codec.', 'geometry_opacity.')
ALLOWED_CODEC_OVERLAY = ('codec.decoder.', 'codec.post_quant_conv.')


def rf_codec_protocol():
    return dict(name=PROTOCOL, version=1, parent_protocol=PARENT_PROTOCOL,
        solver='euler', solver_steps=3, conditional_restorer_allowed=False,
        trainable='existing_RF_trunk_router_and_five_expanded_experts_only',
        Base_frozen=True, world_map_appearance_frozen=True, full_codec_frozen=True,
        encoder_quant_conv_codebook_frozen=True, decoder_post_quant_conv_frozen=True,
        private_opacity_frozen=True, latent_normalization_frozen=True, fixed_noise_frozen=True,
        allowed_parent_codec_changes=list(ALLOWED_CODEC_OVERLAY),
        old_optimizer_groups_preserved=6, old_Adam_moments_and_options_preserved=True,
        added_parameters=0, fresh_optimizer_groups=0, local_counter_starts_zero=True,
        frozen_RF_parent_clock_continues=True, Base_clock_unchanged=True,
        Source_targets_are_labels_only=True, Source_future_feedback=False,
        GAN_active=False, automatic_active_promotion=False, gameplay_qualified=False)


# Explicit alias for callers which use the established renderer_* naming.
renderer_rf_codec_protocol = rf_codec_protocol


def _validate_model(model):
    flow = getattr(model, 'flow_renderer', None)
    if (not isinstance(getattr(model, 'base_student', None), torch.nn.Module)
            or flow is None or getattr(flow, 'kind', None) != 'student'
            or getattr(flow, 'mode', None) != 'rectified_flow'
            or type(getattr(flow, 'steps', None)) is not int or flow.steps != 3
            or not isinstance(getattr(flow, 'codec', None), torch.nn.Module)
            or not isinstance(getattr(flow, 'geometry_opacity', None), torch.nn.Module)):
        raise ValueError('RF-codec continuation requires actual RF3, compact Base, codec and private opacity')
    if (getattr(flow, 'conditional_restorer', None) is not None
            or any('conditional_restorer' in name.split('.') for name, _ in model.named_modules())):
        raise ValueError('RF-codec training must not attach or invoke a conditional restorer')
    return rf5_expert_metadata(flow.output_moe)


def rf_codec_trainable_names(model):
    """Exact RF weight allowlist, independent of temporary inference grad flags."""
    _validate_model(model)
    return sorted('flow_renderer.' + name for name, _ in model.flow_renderer.named_parameters()
                  if not name.startswith(FROZEN_FLOW_PREFIXES))


def rf_codec_parameter_counts(model):
    flow = model.flow_renderer
    allowed = set(rf_codec_trainable_names(model))
    return dict(base=sum(p.numel() for p in model.base_student.parameters()),
        rf_total_without_codec=sum(p.numel() for n, p in flow.named_parameters() if not n.startswith('codec.')),
        rf_trainable=sum(p.numel() for n, p in model.named_parameters() if n in allowed),
        rf_trainable_tensors=len(allowed), codec_frozen=sum(p.numel() for p in flow.codec.parameters()),
        private_opacity_frozen=sum(p.numel() for p in flow.geometry_opacity.parameters()),
        total_model=sum(p.numel() for p in model.parameters()),
        trainable_model=sum(p.numel() for p in model.parameters() if p.requires_grad),
        conditional_restorer_parameters=0)


def rf_codec_frozen_fingerprint(model):
    """Hash EVERY buffer and every parameter outside the exact RF allowlist."""
    actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if actual != set(rf_codec_trainable_names(model)):
        raise ValueError('Configure RF-codec training before measuring its whole frozen fingerprint')
    return frozen_fingerprint(model)


def configure_rf_codec_training(model):
    metadata = _validate_model(model)
    model.eval().requires_grad_(False)
    flow = model.flow_renderer
    flow.train()
    for name, parameter in flow.named_parameters():
        parameter.requires_grad_(not name.startswith(FROZEN_FLOW_PREFIXES))
    flow.codec.eval().requires_grad_(False)
    flow.geometry_opacity.eval().requires_grad_(False)
    model.base_student.eval().requires_grad_(False)
    names = rf_codec_trainable_names(model)
    return dict(protocol=PROTOCOL, solver='euler', solver_steps=3,
        trainable_parameter_names=names, parameters=rf_codec_parameter_counts(model),
        expert_architecture=metadata, full_codec_frozen=True, Base_frozen=True,
        world_map_appearance_frozen=True, private_opacity_frozen=True,
        conditional_restorer_enabled=False, frozen_fingerprint=rf_codec_frozen_fingerprint(model))


def _expected_groups(model, metadata):
    previous, renamed, expanded = _canonical_names(model, metadata)
    return [[renamed.get(name, name) for name in group] for group in previous] + [expanded]


def adopt_rf_codec_optimizer(model, saved_full_RF):
    """Adopt all six original groups/moments, permitting only the frozen D overlay.

    No tensor names, moments, group options, model values, samplers or random
    streams are changed. Inherited frozen Base/private states remain present.
    """
    metadata = _validate_model(model)
    saved = saved_full_RF
    if (not isinstance(saved, dict)
            or not {'base', 'flow', 'optimizer', 'optimizer_parameter_names', 'step',
                    'renderer_training_steps', 'base_training_steps', 'phase'} <= saved.keys()
            or any(type(saved[key]) is not int or saved[key] < 1
                   for key in ('step', 'renderer_training_steps', 'base_training_steps'))
            or saved['phase'].get('protocol') != PARENT_PROTOCOL):
        raise ValueError('RF-codec adoption requires the complete trained RF3 parent and its real clocks')
    names = _expected_groups(model, metadata)
    if saved['optimizer_parameter_names'] != names:
        raise ValueError('RF-codec parent names/order differ from the six complete canonical Adam groups')
    flat = [name for group in names for name in group]
    parameters = dict(model.named_parameters())
    wanted = {name for name in parameters if name.startswith('base_student.')
              or name.startswith('flow_renderer.') and not name.startswith('flow_renderer.codec.')}
    if len(flat) != len(set(flat)) or set(flat) != wanted:
        raise ValueError('RF-codec parent parameter names must cover all old Base/RF/private weights exactly once')
    if {name for name, p in parameters.items() if p.requires_grad} != set(rf_codec_trainable_names(model)):
        raise ValueError('Configure RF-only training before adopting its unchanged named Adam')
    changed_decoder = []
    for head, old, current in (('base_student', saved['base'], model.base_student.state_dict()),
                                ('flow_renderer', saved['flow'], model.flow_renderer.state_dict())):
        if set(old) != set(current):
            raise ValueError('RF-codec adoption cannot add, remove or rename model tensors')
        for name, value in old.items():
            target = current[name]
            if (not torch.is_tensor(value) or not torch.is_tensor(target)
                    or value.shape != target.shape or value.dtype != target.dtype
                    or not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(target).all())):
                raise ValueError('RF-codec parent/current model tensors must retain finite shapes and dtypes')
            if not _same(value, target):
                if head != 'flow_renderer' or not name.startswith(ALLOWED_CODEC_OVERLAY):
                    raise ValueError('Only frozen codec.decoder/post_quant_conv may differ from the RF parent')
                changed_decoder.append(head + '.' + name)
    optimizer_state = saved['optimizer']
    groups, states = optimizer_state.get('param_groups', []), optimizer_state.get('state', {})
    ids = [index for group in groups for index in group.get('params', [])]
    if (len(groups) != 6 or any(len(group.get('params', [])) != len(group_names)
            for group, group_names in zip(groups, names))
            or ids != list(range(len(flat))) or set(states) != set(ids)):
        raise ValueError('RF-codec adoption requires all six ordered groups and every existing Adam state')
    for index, name in zip(ids, flat):
        state, parameter = states[index], parameters[name]
        age = state.get('step')
        if (not torch.is_tensor(age) or age.numel() != 1 or not bool(torch.isfinite(age).all())
                or float(age) < 1 or float(age) != int(float(age))
                or any(not torch.is_tensor(state.get(key)) or state[key].shape != parameter.shape
                       or state[key].dtype != parameter.dtype or not bool(torch.isfinite(state[key]).all())
                       for key in ('exp_avg', 'exp_avg_sq'))
                or bool((state['exp_avg_sq'] < 0).any())):
            raise ValueError('RF-codec adoption cannot reset an Adam age or change its moment shape/dtype/value validity')
    configured = [dict(**deepcopy({key: value for key, value in group.items() if key != 'params'}),
                       params=[parameters[name] for name in group_names])
                  for group, group_names in zip(groups, names)]
    optimizer = torch.optim.AdamW(configured)
    optimizer.load_state_dict(deepcopy(optimizer_state))
    if not _same(optimizer.state_dict(), optimizer_state):
        raise AssertionError('RF-codec optimizer adoption changed an old state or group option')
    source_codec = {name: value for name, value in saved['flow'].items() if name.startswith('codec.')}
    current_codec = {name: value for name, value in model.flow_renderer.state_dict().items() if name.startswith('codec.')}
    frozen_names = [name for name in flat if not parameters[name].requires_grad]
    proof = dict(kind=PROTOCOL, version=1, source_step=saved['step'], start_step=0,
        parent_renderer_training_steps=saved['renderer_training_steps'],
        frozen_base_training_steps=saved['base_training_steps'], solver='euler', solver_steps=3,
        old_Adam_states=len(states), old_optimizer_groups=6,
        all_old_Adam_states_options_bitexact=True, optimizer_parameter_names=deepcopy(names),
        old_parameter_name_map={name: name for name in flat}, added_parameter_names=[],
        added_parameters=0, fresh_Adam_states=0, fresh_optimizer_groups=0,
        frozen_optimizer_parameter_names=frozen_names,
        old_RF_Base_private_tensors_bitexact=True, codec_E_Q_and_buffers_bitexact=True,
        allowed_codec_tensor_changes=list(ALLOWED_CODEC_OVERLAY),
        decoder_overlay_changed_tensor_count=len(changed_decoder),
        decoder_overlay_changed_tensor_names=changed_decoder,
        source_codec_sha256=_hash_tensors(source_codec), current_codec_sha256=_hash_tensors(current_codec),
        parameters=rf_codec_parameter_counts(model),
        trainable_parameter_names=rf_codec_trainable_names(model),
        frozen_fingerprint=rf_codec_frozen_fingerprint(model),
        conditional_restorer_enabled=False, sampler_and_global_RNG_unchanged_by_optimizer_builder=True,
        automatic_active_promotion=False, gameplay_qualified=False)
    return optimizer, proof
