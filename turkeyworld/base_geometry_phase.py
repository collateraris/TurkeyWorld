"""Expand/train only compact Base while preserving the complete RF Adam ledger."""
from copy import deepcopy
import math
from types import SimpleNamespace

import torch

from .base_expanded_experts import base_expanded_expert_metadata
from .renderer_native_phase import _same
from .renderer_refinement import frozen_fingerprint
from .renderer_rf5_experts import rf5_expert_metadata
from .renderer_rf_codec_phase import _validate_model, _expected_groups, ALLOWED_CODEC_OVERLAY


PROTOCOL = 'base_expanded_geometry_v1'
PARENT_PROTOCOL = 'renderer_rf_expanded_experts_v1'


def base_geometry_protocol():
    return dict(name=PROTOCOL, version=1, trainable='all_compact_Base_parameters_only',
        old_Base_parameter_tensors=146, added_Base_parameter_tensors=80,
        RF_frozen=True, full_codec_frozen=True, decoder_post_quant_conv_frozen=True,
        encoder_quant_conv_codebook_frozen=True, world_map_appearance_frozen=True,
        private_opacity_frozen=True, latent_normalization_frozen=True, fixed_noise_frozen=True,
        RF_forward_active=False, conditional_restorer_allowed=False,
        old_optimizer_groups_preserved=6, fresh_optimizer_group=7, new_optimizer_lr=1e-4,
        old_Adam_moments_and_options_preserved=True, local_counter_starts_zero=True,
        Source_targets_are_labels_only=True, Source_future_feedback=False,
        automatic_active_promotion=False, gameplay_qualified=False)


def base_geometry_trainable_names(model):
    _validate_model(model)
    base_expanded_expert_metadata(model.base_student)
    return sorted('base_student.'+name for name, _ in model.base_student.named_parameters())


def base_geometry_parameter_counts(model):
    metadata = base_expanded_expert_metadata(model.base_student)
    return dict(base=sum(p.numel() for p in model.base_student.parameters()),
        base_trainable=sum(p.numel() for p in model.base_student.parameters() if p.requires_grad),
        base_parameter_tensors=len(list(model.base_student.parameters())),
        parameters_per_expert=metadata['parameters_per_expert'],
        added_parameters=metadata['added_parameters'], added_parameter_tensors=80,
        RF_frozen=sum(p.numel() for n, p in model.flow_renderer.named_parameters() if not n.startswith('codec.')),
        codec_frozen=sum(p.numel() for p in model.flow_renderer.codec.parameters()),
        private_opacity_frozen=sum(p.numel() for p in model.flow_renderer.geometry_opacity.parameters()),
        total_model=sum(p.numel() for p in model.parameters()),
        trainable_model=sum(p.numel() for p in model.parameters() if p.requires_grad))


def base_geometry_frozen_fingerprint(model):
    allowed = set(base_geometry_trainable_names(model))
    if {name for name, p in model.named_parameters() if p.requires_grad} != allowed:
        raise ValueError('Configure Base-only training before measuring its frozen fingerprint')
    return frozen_fingerprint(model)


def configure_base_geometry_training(model):
    _validate_model(model)
    metadata = base_expanded_expert_metadata(model.base_student)
    model.eval().requires_grad_(False)
    model.base_student.train().requires_grad_(True)
    return dict(protocol=PROTOCOL, trainable_parameter_names=base_geometry_trainable_names(model),
        parameters=base_geometry_parameter_counts(model), expert_architecture=metadata,
        RF_frozen=True, codec_frozen=True, private_opacity_frozen=True, world_frozen=True,
        RF_forward_active=False, conditional_restorer_enabled=False,
        frozen_fingerprint=base_geometry_frozen_fingerprint(model))


class _OldBaseParameters:
    def __init__(self, head, names):
        self.parameters_by_name = dict(head.named_parameters())
        self.names = names

    def named_parameters(self):
        return ((old, self.parameters_by_name[current]) for old, current in self.names.items())


def adopt_base_geometry_optimizer(model, saved_full_RF):
    """Remap every old Base parameter, keep all six groups, append80 fresh."""
    _validate_model(model)
    base = model.base_student
    metadata = base_expanded_expert_metadata(base)
    saved = saved_full_RF
    if (not isinstance(saved, dict) or not {'base', 'flow', 'optimizer', 'optimizer_parameter_names',
            'step', 'renderer_training_steps', 'base_training_steps', 'phase'} <= saved.keys()
            or saved['phase'].get('protocol') != PARENT_PROTOCOL
            or any(type(saved[key]) is not int or saved[key] < 1
                   for key in ('step', 'renderer_training_steps', 'base_training_steps'))):
        raise ValueError('Expanded Base requires the complete trained RF parent and optimizer clocks')
    local_map = metadata['legacy_parameter_name_map']
    remap = {'base_student.'+old: 'base_student.'+new for old, new in local_map.items()}
    added = ['base_student.'+name for name in metadata['added_parameter_names']]
    view = SimpleNamespace(base_student=_OldBaseParameters(base, local_map), flow_renderer=model.flow_renderer)
    expected = _expected_groups(view, rf5_expert_metadata(model.flow_renderer.output_moe))
    if saved['optimizer_parameter_names'] != expected:
        raise ValueError('Expanded Base parent must retain all six canonical old name groups')
    mapped = [[remap.get(name, name) for name in group] for group in expected]
    flat = [name for group in mapped for name in group]
    parameters = dict(model.named_parameters())
    wanted = {name for name in parameters if name.startswith('base_student.')
              or name.startswith('flow_renderer.') and not name.startswith('flow_renderer.codec.')}
    if (len(local_map) != 146 or len(added) != 80 or len(set(flat+added)) != len(flat)+80
            or set(flat+added) != wanted):
        raise ValueError('Expanded Base mapping must preserve146 old and append80 new tensors without duplicates')
    if {name for name, p in parameters.items() if p.requires_grad} != set(base_geometry_trainable_names(model)):
        raise ValueError('Only Base may train during expanded Base optimizer adoption')
    current = {'base_student.'+name: value for name, value in base.state_dict().items()}
    current.update({'flow_renderer.'+name: value for name, value in model.flow_renderer.state_dict().items()})
    old = {'base_student.'+name: value for name, value in saved['base'].items()}
    old.update({'flow_renderer.'+name: value for name, value in saved['flow'].items()})
    source = {remap.get(name, name): value for name, value in old.items()}
    if len(source) != len(old) or set(current) != set(source) | set(added):
        raise ValueError('Expanded Base cannot silently add/remove/rename a parent state tensor')
    decoder_changes = []
    for name, value in source.items():
        target = current[name]
        if (value.shape != target.shape or value.dtype != target.dtype
                or not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(target).all())):
            raise ValueError('Expanded Base parent tensors must keep finite shapes and dtypes')
        if not _same(value, target):
            if not name.startswith(tuple('flow_renderer.'+prefix for prefix in ALLOWED_CODEC_OVERLAY)):
                raise ValueError('Expanded Base initial parent weights changed outside the explicit frozen D overlay')
            decoder_changes.append(name)
    if (any(not bool(torch.isfinite(current[name]).all()) for name in added)
            or any(bool(current[name].count_nonzero()) for name in added if '.output_projection.' in name)):
        raise ValueError('Expanded Base fresh capacity must be finite with exactly zero residual outputs')
    source_optimizer = saved['optimizer']
    groups, states = source_optimizer.get('param_groups', []), source_optimizer.get('state', {})
    ids = [index for group in groups for index in group.get('params', [])]
    if (len(groups) != 6 or any(len(group.get('params', [])) != len(names)
                              for group, names in zip(groups, expected))
            or ids != list(range(len(flat))) or set(states) != set(ids)):
        raise ValueError('Expanded Base must preserve every existing ordered Adam state in all six groups')
    for index, name in zip(ids, flat):
        state, parameter = states[index], parameters[name]
        age = state.get('step')
        if (not torch.is_tensor(age) or age.numel() != 1 or not bool(torch.isfinite(age).all())
                or float(age) < 1 or float(age) != int(float(age))
                or any(not torch.is_tensor(state.get(key)) or state[key].shape != parameter.shape
                       or state[key].dtype != parameter.dtype or not bool(torch.isfinite(state[key]).all())
                       for key in ('exp_avg', 'exp_avg_sq'))
                or bool((state['exp_avg_sq'] < 0).any())):
            raise ValueError('Expanded Base old Adam ages/moments must remain complete finite and shape-correct')
    configured = [dict(**deepcopy({key: value for key, value in group.items() if key != 'params'}),
                       params=[parameters[name] for name in names]) for group, names in zip(groups, mapped)]
    optimizer = torch.optim.AdamW(configured)
    optimizer.load_state_dict(deepcopy(source_optimizer))
    fresh = deepcopy({key: value for key, value in groups[0].items() if key != 'params'})
    fresh.update(params=[parameters[name] for name in added], lr=base_geometry_protocol()['new_optimizer_lr'])
    optimizer.add_param_group(fresh)
    result = optimizer.state_dict()
    if (not _same(result['state'], states) or not _same(result['param_groups'][:6], groups)
            or set(result['param_groups'][6]['params']) & set(result['state'])):
        raise AssertionError('Expanded Base optimizer reset old state or initialized fresh moments')
    frozen = [name for name in flat if not parameters[name].requires_grad]
    proof = dict(kind=PROTOCOL, version=1, source_step=saved['step'], start_step=0,
        parent_renderer_training_steps=saved['renderer_training_steps'],
        parent_base_training_steps=saved['base_training_steps'], old_Adam_states=len(states),
        old_optimizer_groups=6, all_old_Adam_states_options_bitexact=True,
        optimizer_parameter_names=deepcopy(mapped)+[added],
        old_parameter_name_map={name: remap.get(name, name) for group in expected for name in group},
        added_parameter_names=added, added_parameter_tensors=80,
        added_parameters=sum(parameters[name].numel() for name in added), fresh_Adam_states=0,
        fresh_optimizer_group=7, frozen_optimizer_parameter_names=frozen,
        old_RF_Base_private_tensors_bitexact=True, codec_E_Q_and_buffers_bitexact=True,
        allowed_codec_tensor_changes=list(ALLOWED_CODEC_OVERLAY), decoder_overlay_changed_tensor_names=decoder_changes,
        zero_residual_initialization=True, Base_only_trainable=True, RF_forward_active=False,
        parameters=base_geometry_parameter_counts(model), trainable_parameter_names=base_geometry_trainable_names(model),
        frozen_fingerprint=base_geometry_frozen_fingerprint(model),
        sampler_and_global_RNG_unchanged_by_optimizer_builder=True,
        automatic_active_promotion=False, gameplay_qualified=False)
    return optimizer, proof
