"""Append private RF expert capacity without changing the trained initial function.

This module changes neither solver steps nor the compact Base. The caller owns
the explicit RF5 configuration, checkpoint migration and optimizer name mapping.
"""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


KIND = 'renderer_rf5_expanded_private_experts_v1'
VERSION = 1
ROLE_SCHEMA = 'near_far_enemy_motion_enemy_combat_player_combat_v2'
ROLES = ('near_geometry_player', 'far_geometry', 'enemy_motion',
         'enemy_combat', 'player_combat')
PHYSICAL_ROLE_INDICES = (0, 3, 1, 4, 2)
SETTINGS = {
    'kind': KIND, 'version': VERSION, 'role_schema': ROLE_SCHEMA,
    'roles': list(ROLES), 'experts': 5,
    'legacy_width': 32, 'legacy_hidden': 64, 'legacy_heads': 4,
    'private_width': 256, 'private_heads': 8, 'private_hidden': 416,
    'private_transformer_blocks_per_expert': 1,
    'legacy_parameters_per_expert': 8544,
    'new_parameters_per_expert': 494528,
    'parameters_per_expert': 503072,
    'expert_parameters': 2515360,
    'added_parameters': 2472640,
    'routing': 'existing_five_role_dense_soft_routing_unchanged',
    'initialization': 'legacy_plus_zero_projected_private_residual',
    'private_initialization_seeds': [173, 174, 175, 176, 177],
    'Base_changed': False, 'Source_future_inputs': False,
}


def validate_rf5_expert_metadata(metadata):
    """Validate the immutable architecture specification, independently of weights."""
    if not isinstance(metadata, dict) or any(
            type(metadata.get(k)) is not type(v) or metadata[k] != v
            for k, v in SETTINGS.items()):
        raise ValueError('RF5 expanded expert metadata differs from its explicit v1 profile')
    return True


def _validate_legacy(layer):
    if (not isinstance(layer, nn.TransformerEncoderLayer)
            or layer.self_attn.embed_dim != 32 or layer.self_attn.num_heads != 4
            or not layer.self_attn.batch_first or not layer.norm_first
            or layer.linear1.in_features != 32 or layer.linear1.out_features != 64
            or layer.linear2.in_features != 64 or layer.linear2.out_features != 32
            or layer.activation_relu_or_gelu != 2
            or layer.self_attn.dropout != 0.
            or any(getattr(layer, name).p != 0. for name in ('dropout', 'dropout1', 'dropout2'))
            or sum(p.numel() for p in layer.parameters()) != 8544):
        raise ValueError('RF5 expansion requires the intact RF d32/FF64 private Transformer')


class RF5PrivateExpert(nn.Module):
    """The original expert plus a zero-initialized, independently learned residual."""

    def __init__(self, legacy: nn.TransformerEncoderLayer, *, seed=173):
        super().__init__()
        _validate_legacy(legacy)
        if type(seed) is not int or seed < 0:
            raise ValueError('RF5 expert initialization seed must be a nonnegative integer')
        self.initialization_seed = seed
        self.legacy = legacy  # Keep the actual old Parameter objects for migration.
        parameter = next(legacy.parameters())
        devices = [parameter.device.index] if parameter.is_cuda else []
        factory = {'device': parameter.device, 'dtype': parameter.dtype}
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if parameter.is_cuda:
                torch.cuda.default_generators[parameter.device.index].manual_seed(seed)
            self.private_input = nn.Linear(32, 256, **factory)
            self.private_transformer = nn.TransformerEncoderLayer(
                256, 8, dim_feedforward=416, dropout=0., activation='gelu',
                batch_first=True, norm_first=True, **factory)
            self.private_output = nn.Linear(256, 32, **factory)
            nn.init.zeros_(self.private_output.weight)
            nn.init.zeros_(self.private_output.bias)
        # Do not recursively change the existing layer's possibly mixed modes.
        self.training = legacy.training
        for module in (self.private_input, self.private_transformer, self.private_output):
            module.train(legacy.training)
        if sum(p.numel() for p in self.parameters()) != SETTINGS['parameters_per_expert']:
            raise RuntimeError('RF5 private expert parameter count changed')

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        if src.ndim != 3 or src.shape[-1] != 32:
            raise ValueError('RF5 expert expects batch-first screen tokens[B,N,32]')
        old = self.legacy(src, src_mask=src_mask,
                          src_key_padding_mask=src_key_padding_mask, is_causal=is_causal)
        new = self.private_transformer(self.private_input(src), src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask, is_causal=is_causal)
        return old + self.private_output(new)


def _validate_owner(moe):
    if (getattr(moe, 'role_schema', None) != ROLE_SCHEMA
            or getattr(moe, 'expert_count', None) != 5
            or getattr(moe, 'expert_width', None) != 32
            or getattr(moe, 'width', None) != 256
            or getattr(moe, 'output_dim', None) != 64
            or getattr(moe, 'grid', None) != 4
            or not isinstance(getattr(moe, 'experts', None), nn.ModuleList)
            or len(moe.experts) != 5
            or not isinstance(getattr(moe, 'split_router', None), nn.Linear)):
        raise ValueError('RF5 expanded experts require the complete five-role RF32/grid4 head')


def rf5_expert_metadata(moe):
    """Return JSON-native architecture facts and relative optimizer-name mappings."""
    _validate_owner(moe)
    if not all(isinstance(expert, RF5PrivateExpert) for expert in moe.experts):
        raise ValueError('RF5 expert metadata requires all five expanded experts')
    result = deepcopy(SETTINGS)
    old_to_new, added = {}, []
    for index, expert in enumerate(moe.experts):
        _validate_legacy(expert.legacy)
        if expert.initialization_seed != SETTINGS['private_initialization_seeds'][index]:
            raise ValueError('RF5 expert initialization seed metadata differs')
        for name, _ in expert.legacy.named_parameters():
            old_to_new[f'experts.{index}.{name}'] = f'experts.{index}.legacy.{name}'
        for name, _ in expert.named_parameters():
            if not name.startswith('legacy.'):
                added.append(f'experts.{index}.{name}')
    result.update(
        logical_role_expert_indices=list(PHYSICAL_ROLE_INDICES),
        legacy_parameter_name_map=old_to_new,
        added_parameter_names=added,
        added_parameter_tensors=len(added),
        trainable_expert_parameters=sum(p.numel() for e in moe.experts
                                       for p in e.parameters() if p.requires_grad),
        trainable_moe_parameters=sum(p.numel() for p in moe.parameters() if p.requires_grad),
        moe_parameters=sum(p.numel() for p in moe.parameters()),
    )
    validate_rf5_expert_metadata(result)
    return result


def upgrade_rf5_experts(moe):
    """Wrap exactly the five RF experts, preserving old parameters, modes and RNG.

    Returned parameter names are relative to ``moe``. All preexisting parameters
    outside ``experts`` retain their names. Their Adam entries need no renaming.
    The caller must map each legacy expert name to its ``.legacy`` destination.
    """
    _validate_owner(moe)
    wrapped = [isinstance(expert, RF5PrivateExpert) for expert in moe.experts]
    if any(wrapped):
        if not all(wrapped):
            raise ValueError('Cannot expand a partially upgraded RF5 expert head')
        return rf5_expert_metadata(moe)
    for expert in moe.experts:
        _validate_legacy(expert)
    # Construct all five before changing the caller, so validation/construction
    # errors cannot leave a partially replaced ModuleList.
    replacements = [RF5PrivateExpert(expert, seed=173 + index)
                    for index, expert in enumerate(moe.experts)]
    for index, expert in enumerate(replacements):
        moe.experts[index] = expert
    moe.last_aux = None
    return rf5_expert_metadata(moe)
