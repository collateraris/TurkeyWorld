"""Function-preserving expansion of the trained conditional RGB restorer.

The complete trained64-wide module remains the legacy branch.  A separate
224-wide map/state attention branch corrects its restored RGB.  Its final
convolution starts at zero, preserving the learned parent function exactly.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace

import torch
from torch import nn

from .conditional_decoder_restorer import (
    ConditionalDecoderRestorer, ConditionalDecoderRestorerConfig, _state_sha256)


KIND = 'expanded_conditional_decoder_restorer_v1'
SCHEMA = 'legacy_plus_zero_wide_RF_RGB_current_state_map_v1'
VERSION = 1


def _legacy_config():
    return asdict(ConditionalDecoderRestorerConfig())


def _expansion_config():
    return asdict(ConditionalDecoderRestorerConfig(width=224, heads=8, hidden=288, init_seed=174))


@dataclass(frozen=True)
class ExpandedConditionalDecoderRestorerConfig:
    architecture: str = KIND
    legacy_config: dict = field(default_factory=_legacy_config)
    expansion_config: dict = field(default_factory=_expansion_config)

    def __post_init__(self):
        if self.architecture != KIND or not isinstance(self.legacy_config, dict) or not isinstance(self.expansion_config, dict):
            raise ValueError('Expanded restorer requires its explicit architecture and two complete branch configs')
        old = ConditionalDecoderRestorerConfig(**self.legacy_config)
        new = ConditionalDecoderRestorerConfig(**self.expansion_config)
        if ((old.width, old.heads, old.hidden, old.depth) != (64, 4, 128, 2)
                or (new.width, new.heads, new.hidden, new.depth) != (224, 8, 288, 2)):
            raise ValueError('Expanded restorer requires legacy64/4/128 and independent224/8/288 attention branches')
        object.__setattr__(self, 'legacy_config', asdict(old))
        object.__setattr__(self, 'expansion_config', asdict(new))


class ExpandedConditionalDecoderRestorer(nn.Module):
    """Keep trained legacy RGB, then add a separately learned wide correction."""
    def __init__(self, legacy=None, config=None, *, seed=None):
        super().__init__()
        if config is None:
            config = ExpandedConditionalDecoderRestorerConfig(
                legacy_config=legacy.get_config() if legacy is not None else _legacy_config())
        elif isinstance(config, dict):
            config = ExpandedConditionalDecoderRestorerConfig(**config)
        if not isinstance(config, ExpandedConditionalDecoderRestorerConfig):
            raise TypeError('Expected an explicit expanded conditional restorer config')
        if seed is not None:
            branch = asdict(replace(ConditionalDecoderRestorerConfig(**config.expansion_config), init_seed=seed))
            config = replace(config, expansion_config=branch)
        self.config = config
        if legacy is None:
            legacy = ConditionalDecoderRestorer(config.legacy_config)
        if type(legacy) is not ConditionalDecoderRestorer or legacy.get_config() != config.legacy_config:
            raise ValueError('Expand only the exact trained native64 conditional restorer direct parent')
        if (sum(p.numel() for p in legacy.parameters()), len(list(legacy.parameters()))) != (247539, 65):
            raise ValueError('Expanded restorer parent must contain all247539 parameters /65 tensors')
        self.legacy = legacy
        # Both constructors isolate CPU initialization; moving a new branch to
        # the existing dtype/device performs no stochastic draw.
        self.expansion = ConditionalDecoderRestorer(config.expansion_config)
        parameter = next(legacy.parameters())
        self.expansion.to(device=parameter.device, dtype=parameter.dtype)
        self.expansion.train(legacy.training)
        self.expansion.requires_grad_(any(p.requires_grad for p in legacy.parameters()))
        # Preserve each legacy child mode/grad flag; do not recursively .train
        # or .requires_grad_ the existing learned module during construction.
        self.training = legacy.training
        if (sum(p.numel() for p in self.expansion.parameters()),
                sum(p.numel() for p in self.parameters())) != (1750259, 1997798):
            raise RuntimeError('Expanded restorer real neural parameter count changed')

    def get_config(self):
        return deepcopy(asdict(self.config))

    def forward_with_aux(self, codes, rgb, state, raw_map):
        old = self.legacy.forward_with_aux(codes, rgb, state, raw_map)
        fresh = self.expansion.forward_with_aux(codes, old['image'], state, raw_map)
        return dict(image=fresh['image'], delta=fresh['image'] - rgb,
            residual_logits=fresh['residual_logits'], screen_tokens=fresh['screen_tokens'],
            legacy_image=old['image'], legacy_delta=old['delta'], expansion_delta=fresh['delta'])

    def forward(self, codes, rgb, state, raw_map):
        return self.forward_with_aux(codes, rgb, state, raw_map)['image']


def expanded_conditional_restorer_metadata(module, *, include_state_hash=True):
    if not isinstance(module, ExpandedConditionalDecoderRestorer):
        raise TypeError('Expected the actual expanded conditional restorer')
    result = dict(kind=KIND, version=VERSION, schema=SCHEMA, config=module.get_config(),
        parameters=sum(p.numel() for p in module.parameters()),
        parameter_tensors=len(list(module.parameters())), legacy_parameters=247539,
        legacy_parameter_tensors=65, added_parameters=1750259, added_parameter_tensors=65,
        branch_names=['legacy', 'expansion'], branch_widths=dict(legacy=64, expansion=224),
        branch_heads=dict(legacy=4, expansion=8), branch_hidden=dict(legacy=128, expansion=288),
        attention_blocks_per_branch=2, screen_tokens=64, cross_attention_memory_tokens=259,
        dropout=0., initialization='trained_legacy_plus_zero_RGB_output_wide_correction',
        expansion_RGB_input='current_legacy_restored_RGB', rgb_amplitude=1.,
        input_scope='current_quantized_RF_codes_decoded_RGB_current_state_raw_material_map',
        Source_RGB_inputs=False, Source_masks_inputs=False, future_Source_inputs=False,
        Source_raycast_or_occlusion_oracle=False, learned_RGB_residual=True,
        masks_or_gates_required=False, dead_padding_parameters=False)
    if include_state_hash:
        result['state_sha256'] = _state_sha256(module.state_dict())
    return result


def validate_expanded_conditional_restorer_state(module, state, *, metadata=None):
    if not isinstance(module, ExpandedConditionalDecoderRestorer) or not isinstance(state, dict):
        raise ValueError('Expanded restorer state requires its exact module and complete tensor dictionary')
    expected = module.state_dict()
    if set(state) != set(expected) or any(not torch.is_tensor(value)
            or value.shape != expected[name].shape or value.dtype != expected[name].dtype
            or not bool(torch.isfinite(value).all()) for name, value in state.items()):
        raise ValueError('Expanded restorer state has missing/extra, wrong-shape/dtype or nonfinite tensors')
    state_sha = _state_sha256(state)
    if metadata is not None:
        architecture = expanded_conditional_restorer_metadata(module, include_state_hash=False)
        if (not isinstance(metadata, dict) or set(metadata) != set(architecture) | {'state_sha256'}
                or any(metadata.get(key) != value for key, value in architecture.items())
                or metadata.get('state_sha256') != state_sha):
            raise ValueError('Expanded restorer architecture/config/input/count or state binding differs')
    return dict(passed=True, state_sha256=state_sha, parameters=1997798, parameter_tensors=130)


def expand_conditional_restorer(legacy, *, seed=174):
    """Return a private wrapper and explicit old-name/new-name migration receipt.

    No model attachment, optimizer mutation, checkpoint write, parent tensor,
    legacy flag or RNG change is performed.  Existing65 Adam entries can be
    renamed through the returned map; expansion65 entries start fresh.
    """
    from .conditional_decoder_restorer import conditional_decoder_restorer_metadata
    if type(legacy) is not ConditionalDecoderRestorer:
        raise ValueError('Restorer capacity migration requires its direct unexpanded parent')
    old_metadata = conditional_decoder_restorer_metadata(legacy)
    names = {name: 'legacy.' + name for name, _ in legacy.named_parameters()}
    states = {name: 'legacy.' + name for name in legacy.state_dict()}
    expanded = ExpandedConditionalDecoderRestorer(legacy=legacy, seed=seed)
    if _state_sha256(legacy.state_dict()) != old_metadata['state_sha256']:
        raise AssertionError('Expanded restorer constructor changed the learned parent')
    added = ['expansion.' + name for name, _ in expanded.expansion.named_parameters()]
    report = dict(kind='append_conditional_restorer_capacity_v1', version=1,
        source_metadata=old_metadata, expanded_metadata=expanded_conditional_restorer_metadata(expanded),
        source_parameters=247539, source_parameter_tensors=65,
        added_parameters=1750259, added_parameter_tensors=65,
        combined_parameters=1997798, combined_parameter_tensors=130,
        old_parameter_name_map=names, old_state_name_map=states,
        old_model_tensor_name_map=states, old_name_map=names, added_parameter_names=added,
        old_prefix='legacy.', new_prefix='expansion.', parent_tensors_bitexact=True,
        parent_parameter_objects_preserved=True, zero_output_initialization=True,
        initial_function_preserved=True, old_optimizer_states_must_be_renamed_not_reset=True,
        new_optimizer_states_fresh=True, Source_future_inputs=False)
    return expanded, report
