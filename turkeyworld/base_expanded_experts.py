"""Five ~300k Base experts with full-query conditioning and zero activation.

The old 8D private experts and shared routing/output remain the actual trained
modules. Each new Transformer reads the original 144D screen query directly.
"""
from copy import deepcopy
from dataclasses import replace

import torch
from torch import nn


KIND = 'renderer_base_expanded_private_experts_v1'
VERSION = 1
ROLE_SCHEMA = 'near_far_enemy_motion_enemy_combat_player_combat_v2'
ROLES = ('near_geometry_player', 'far_geometry', 'enemy_motion', 'enemy_combat', 'player_combat')
ROLE_INDICES = (0, 3, 1, 4, 2)
SETTINGS = dict(kind=KIND, version=VERSION, role_schema=ROLE_SCHEMA, roles=list(ROLES),
    experts=5, legacy_width=8, legacy_heads=2, legacy_hidden=16,
    query_context_width=144, private_width=192, private_heads=6, private_hidden=314,
    private_transformer_blocks_per_expert=1, legacy_parameters_per_expert=600,
    new_parameters_per_expert=299458, parameters_per_expert=300058,
    expert_parameters=1500290, added_parameters=1497290, added_parameter_tensors=80,
    private_initialization_seeds=[173, 174, 175, 176, 177],
    private_input='full_current_Base_query_before_shared_8D_projection',
    routing='existing_five_role_hierarchy_and_shared_output_unchanged',
    initialization='legacy_plus_zero_projected_private_full_query_residual',
    dropout=0., norm_first=True, activation='gelu', RF_changed=False,
    Source_future_inputs=False, runtime_depth_oracle=False)


def validate_base_expanded_expert_metadata(metadata):
    if not isinstance(metadata, dict) or any(
            type(metadata.get(key)) is not type(value) or metadata[key] != value
            for key, value in SETTINGS.items()):
        raise ValueError('Expanded Base expert architecture metadata differs from its explicit profile')
    return True


def _validate_legacy(layer):
    if (not isinstance(layer, nn.TransformerEncoderLayer)
            or layer.self_attn.embed_dim != 8 or layer.self_attn.num_heads != 2
            or not layer.self_attn.batch_first or not layer.norm_first
            or layer.linear1.in_features != 8 or layer.linear1.out_features != 16
            or layer.linear2.in_features != 16 or layer.linear2.out_features != 8
            or layer.activation_relu_or_gelu != 2 or layer.self_attn.dropout != 0.
            or any(getattr(layer, name).p != 0. for name in ('dropout', 'dropout1', 'dropout2'))
            or sum(parameter.numel() for parameter in layer.parameters()) != 600):
        raise ValueError('Base expansion requires the intact 8D/2-head/FF16 private Transformer')


class BaseExpandedExpert(nn.Module):
    """Keep the trained 8D expert, add a private full-144D-query correction."""
    def __init__(self, legacy, *, seed=173):
        super().__init__()
        _validate_legacy(legacy)
        if type(seed) is not int or seed < 0:
            raise ValueError('Expanded Base expert requires a nonnegative integer seed')
        self.legacy = legacy
        self.initialization_seed = seed
        parameter = next(legacy.parameters())
        devices = [parameter.device.index] if parameter.is_cuda else []
        factory = dict(device=parameter.device, dtype=parameter.dtype)
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if parameter.is_cuda:
                torch.cuda.default_generators[parameter.device.index].manual_seed(seed)
            self.input_projection = nn.Linear(144, 192, **factory)
            self.private_transformer = nn.TransformerEncoderLayer(192, 6, dim_feedforward=314,
                dropout=0., activation='gelu', batch_first=True, norm_first=True, **factory)
            self.output_projection = nn.Linear(192, 8, **factory)
            nn.init.zeros_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)
        self.training = legacy.training
        new_trainable = any(value.requires_grad for value in legacy.parameters())
        for module in (self.input_projection, self.private_transformer, self.output_projection):
            module.train(legacy.training).requires_grad_(new_trainable)
        if sum(parameter.numel() for parameter in self.parameters()) != SETTINGS['parameters_per_expert']:
            raise RuntimeError('Expanded Base expert parameter count changed')

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False, *, query_context=None):
        if (src.ndim != 3 or src.shape[-1] != 8 or query_context is None
                or query_context.shape != (*src.shape[:2], 144)
                or query_context.device != src.device):
            raise ValueError('Expanded Base expert needs projected[B,N,8] and full query_context[B,N,144]')
        old = self.legacy(src, src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask, is_causal=is_causal)
        context = query_context.to(self.input_projection.weight.dtype)
        fresh = self.private_transformer(self.input_projection(context), src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask, is_causal=is_causal)
        return old + self.output_projection(fresh).to(old.dtype)


def _validate_head(head):
    from .base_student import CompactBaseHead
    from .renderer_moe import RendererTokenMoE
    moe = getattr(head, 'output_moe', None)
    if (not isinstance(head, CompactBaseHead) or not isinstance(moe, RendererTokenMoE)
            or not head.config.output_moe_enabled or head.config.pixel_residual
            or head.config.width != 144 or head.config.output_moe_role_schema != ROLE_SCHEMA
            or head.config.output_moe_experts != 5
            or (moe.width, moe.expert_width, moe.output_dim, moe.grid, moe.expert_count) != (144, 8, 12, 16, 5)
            or moe.role_schema != ROLE_SCHEMA or len(moe.experts) != 5
            or not isinstance(getattr(moe, 'split_router', None), nn.Linear)):
        raise ValueError('Base expansion requires its complete five-role 144D/grid16 compact head')
    return moe


def base_expanded_expert_metadata(head):
    """Architecture and optimizer mappings, relative to the complete Base head."""
    moe = _validate_head(head)
    if (not head.config.output_moe_expanded_enabled or not getattr(moe, 'base_expanded_enabled', False)
            or not all(isinstance(expert, BaseExpandedExpert) for expert in moe.experts)):
        raise ValueError('Expanded Base experts require their explicit flag and all five private branches')
    added, name_map = [], {}
    for index, expert in enumerate(moe.experts):
        _validate_legacy(expert.legacy)
        if expert.initialization_seed != SETTINGS['private_initialization_seeds'][index]:
            raise ValueError('Expanded Base expert initialization seed differs')
        for name, _ in expert.named_parameters():
            if not name.startswith('legacy.'):
                added.append(f'output_moe.experts.{index}.{name}')
    for name, _ in head.named_parameters():
        if name in added:
            continue
        old_name = name
        for index in range(5):
            old_name = old_name.replace(f'output_moe.experts.{index}.legacy.', f'output_moe.experts.{index}.')
        name_map[old_name] = name
    if len(added) != 80:
        raise RuntimeError('Expanded Base must append exactly 80 parameter tensors')
    result = dict(deepcopy(SETTINGS), logical_role_expert_indices=list(ROLE_INDICES),
        legacy_parameter_name_map=name_map, added_parameter_names=added,
        base_parameters=sum(parameter.numel() for parameter in head.parameters()),
        base_parameter_tensors=len(list(head.parameters())),
        moe_parameters=sum(parameter.numel() for parameter in moe.parameters()),
        expert_parameters_actual=sum(parameter.numel() for expert in moe.experts for parameter in expert.parameters()))
    validate_base_expanded_expert_metadata(result)
    return result


def upgrade_base_experts(head):
    """Append the new capacity without replacing any existing Parameter object."""
    moe = _validate_head(head)
    wrapped = [isinstance(expert, BaseExpandedExpert) for expert in moe.experts]
    if any(wrapped):
        if not all(wrapped):
            raise ValueError('Cannot expand a partially upgraded Base expert head')
        return base_expanded_expert_metadata(head)
    for expert in moe.experts:
        _validate_legacy(expert)
    new_config = replace(head.config, output_moe_expanded_enabled=True)
    if sum(parameter.numel() for parameter in head.parameters()) + SETTINGS['added_parameters'] > 2_500_000:
        raise ValueError('Expanded Base exceeds its explicit 2.5M parameter budget')
    replacements = [BaseExpandedExpert(expert, seed=seed)
                    for expert, seed in zip(moe.experts, SETTINGS['private_initialization_seeds'])]
    for index, expert in enumerate(replacements):
        moe.experts[index] = expert
    head.config = new_config
    moe.base_expanded_enabled = True
    moe.last_aux = None
    return base_expanded_expert_metadata(head)
