"""Append real five-million-parameter attention capacity to the learned2M restorer."""
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace

import torch
from torch import nn

from .conditional_decoder_restorer import (
    ConditionalDecoderRestorer, ConditionalDecoderRestorerConfig, _state_sha256)
from .conditional_restorer_expansion import (
    ExpandedConditionalDecoderRestorer, ExpandedConditionalDecoderRestorerConfig)


KIND = 'large_conditional_decoder_restorer_v1'
SCHEMA = 'trained_2M_plus_zero_wide_7M_current_state_map_v1'
VERSION = 1


def _previous_config():
    return asdict(ExpandedConditionalDecoderRestorerConfig())


def _expansion_config():
    return asdict(ConditionalDecoderRestorerConfig(width=384, heads=8, hidden=720, init_seed=175))


@dataclass(frozen=True)
class LargeConditionalDecoderRestorerConfig:
    architecture: str = KIND
    previous_config: dict = field(default_factory=_previous_config)
    expansion_config: dict = field(default_factory=_expansion_config)

    def __post_init__(self):
        if (self.architecture != KIND or not isinstance(self.previous_config, dict)
                or not isinstance(self.expansion_config, dict)):
            raise ValueError('Large restorer requires its explicit7M architecture and complete branch configs')
        previous = ExpandedConditionalDecoderRestorerConfig(**self.previous_config)
        new = ConditionalDecoderRestorerConfig(**self.expansion_config)
        if (new.width, new.heads, new.hidden, new.depth) != (384, 8, 720, 2):
            raise ValueError('Large restorer requires independent384/8/720 attention capacity')
        object.__setattr__(self, 'previous_config', asdict(previous))
        object.__setattr__(self, 'expansion_config', asdict(new))


class LargeConditionalDecoderRestorer(nn.Module):
    """The trained2M function followed by a zero-initialized, trainable5M correction."""
    def __init__(self, previous=None, config=None, *, seed=None):
        super().__init__()
        if config is None:
            config = LargeConditionalDecoderRestorerConfig(
                previous_config=previous.get_config() if previous is not None else _previous_config())
        elif isinstance(config, dict):
            config = LargeConditionalDecoderRestorerConfig(**config)
        if not isinstance(config, LargeConditionalDecoderRestorerConfig):
            raise TypeError('Expected an explicit LargeConditionalDecoderRestorerConfig')
        if seed is not None:
            branch = asdict(replace(ConditionalDecoderRestorerConfig(**config.expansion_config), init_seed=seed))
            config = replace(config, expansion_config=branch)
        self.config = config
        if previous is None:
            previous = ExpandedConditionalDecoderRestorer(config=config.previous_config)
        if type(previous) is not ExpandedConditionalDecoderRestorer or previous.get_config() != config.previous_config:
            raise ValueError('Large restorer capacity must continue its exact learned2M direct parent')
        if (sum(p.numel() for p in previous.parameters()), len(list(previous.parameters()))) != (1997798, 130):
            raise ValueError('Large restorer parent must contain all1,997,798 parameters /130 tensors')
        self.previous = previous
        self.expansion = ConditionalDecoderRestorer(config.expansion_config)
        parameter = next(previous.parameters())
        self.expansion.to(device=parameter.device, dtype=parameter.dtype)
        self.expansion.train(previous.training)
        self.expansion.requires_grad_(any(p.requires_grad for p in previous.parameters()))
        self.training = previous.training
        if (sum(p.numel() for p in self.expansion.parameters()),
                sum(p.numel() for p in self.parameters())) != (5002515, 7000313):
            raise RuntimeError('Large restorer real neural parameter count changed')

    def get_config(self):
        return deepcopy(asdict(self.config))

    def forward_with_aux(self, codes, rgb, state, raw_map):
        previous = self.previous.forward_with_aux(codes, rgb, state, raw_map)
        fresh = self.expansion.forward_with_aux(codes, previous['image'], state, raw_map)
        return dict(image=fresh['image'], delta=fresh['image'] - rgb,
            residual_logits=fresh['residual_logits'], screen_tokens=fresh['screen_tokens'],
            previous_image=previous['image'], previous_delta=previous['delta'],
            expansion_delta=fresh['delta'])

    def forward(self, codes, rgb, state, raw_map):
        return self.forward_with_aux(codes, rgb, state, raw_map)['image']


def large_conditional_restorer_metadata(module, *, include_state_hash=True):
    if not isinstance(module, LargeConditionalDecoderRestorer):
        raise TypeError('Expected the actual7M conditional restorer')
    result = dict(kind=KIND, version=VERSION, schema=SCHEMA, config=module.get_config(),
        capacity_variant='expanded_7m_v1', parameters=sum(p.numel() for p in module.parameters()),
        parameter_tensors=len(list(module.parameters())), previous_parameters=1997798,
        previous_parameter_tensors=130, added_parameters=5002515, added_parameter_tensors=65,
        branch_names=['previous', 'expansion'], expansion_width=384, expansion_heads=8,
        expansion_hidden=720, attention_blocks_added=2, previous_attention_blocks=4,
        screen_tokens=64, cross_attention_memory_tokens=259, dropout=0.,
        initialization='trained_2M_plus_zero_RGB_output_5M_correction',
        expansion_RGB_input='current_previous_2M_restored_RGB', rgb_amplitude=1.,
        input_scope='current_quantized_RF_codes_decoded_RGB_current_state_raw_material_map',
        Source_RGB_inputs=False, Source_masks_inputs=False, future_Source_inputs=False,
        Source_raycast_or_occlusion_oracle=False, learned_RGB_residual=True,
        masks_or_gates_required=False, dead_padding_parameters=False)
    if include_state_hash:
        result['state_sha256'] = _state_sha256(module.state_dict())
    return result


def validate_large_conditional_restorer_state(module, state, *, metadata=None):
    if not isinstance(module, LargeConditionalDecoderRestorer) or not isinstance(state, dict):
        raise ValueError('Large restorer requires its exact module and complete195-tensor state')
    expected = module.state_dict()
    if set(state) != set(expected) or any(not torch.is_tensor(value)
            or value.shape != expected[name].shape or value.dtype != expected[name].dtype
            or not bool(torch.isfinite(value).all()) for name, value in state.items()):
        raise ValueError('Large restorer state has missing/extra, wrong-shape/dtype or nonfinite tensors')
    state_sha = _state_sha256(state)
    if metadata is not None:
        architecture = large_conditional_restorer_metadata(module, include_state_hash=False)
        if (not isinstance(metadata, dict) or set(metadata) != set(architecture) | {'state_sha256'}
                or any(metadata.get(key) != value for key, value in architecture.items())
                or metadata.get('state_sha256') != state_sha):
            raise ValueError('Large restorer architecture/config/input/count or state binding differs')
    return dict(passed=True, state_sha256=state_sha, parameters=7000313, parameter_tensors=195)


def expand_conditional_restorer_7m(previous, *, seed=175):
    """Return a zero-output7M wrapper plus complete130-name Adam/state mapping."""
    from .conditional_decoder_restorer import conditional_decoder_restorer_metadata
    if type(previous) is not ExpandedConditionalDecoderRestorer:
        raise ValueError('7M capacity migration requires the direct learned2M restorer parent')
    old_metadata = conditional_decoder_restorer_metadata(previous)
    names = {name: 'previous.' + name for name, _ in previous.named_parameters()}
    states = {name: 'previous.' + name for name in previous.state_dict()}
    large = LargeConditionalDecoderRestorer(previous=previous, seed=seed)
    if _state_sha256(previous.state_dict()) != old_metadata['state_sha256']:
        raise AssertionError('7M constructor changed its learned2M parent')
    added = ['expansion.' + name for name, _ in large.expansion.named_parameters()]
    report = dict(kind='append_conditional_restorer_7M_capacity_v1', version=1,
        source_metadata=old_metadata, expanded_metadata=large_conditional_restorer_metadata(large),
        source_parameters=1997798, source_parameter_tensors=130,
        added_parameters=5002515, added_parameter_tensors=65,
        combined_parameters=7000313, combined_parameter_tensors=195,
        old_parameter_name_map=names, old_state_name_map=states,
        old_model_tensor_name_map=states, old_name_map=names, added_parameter_names=added,
        old_prefix='previous.', new_prefix='expansion.', parent_tensors_bitexact=True,
        parent_parameter_objects_preserved=True, zero_output_initialization=True,
        initial_function_preserved=True, old_optimizer_states_must_be_renamed_not_reset=True,
        new_optimizer_states_fresh=True, Source_future_inputs=False)
    return large, report
