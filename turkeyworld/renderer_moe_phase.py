"""Append renderer experts while preserving the complete parent Adam/history."""
from copy import deepcopy
import hashlib
import json
import re

import torch

from .renderer_native_phase import _same, restore_renderer_native_rng


PROTOCOL = {
    'name': 'one_enemy_renderer_token_moe_v1', 'version': 1,
    'roles': ['scene_walls', 'enemy', 'combat'], 'routing': 'soft_per_token',
    'experts_per_head': 3, 'expert_transformer_blocks': 1,
    'expert_widths': {'base': 8, 'rf1': 32},
    'router_role_weight': .01, 'router_balance_weight': .001,
    'new_optimizer_lr': .0001, 'main_batch': 16,
    'far_auxiliary_batch': 4, 'far_auxiliary_weight': .25,
    'geometry_auxiliary_batch': 8, 'geometry_auxiliary_weight': 1.,
    'feature_policy': 'frozen_current_world_previous_RGB_state_action_prior_by_Source_environment',
    'navigation_profile': 'explicit_Source_row_metadata_not_predicted_life_flags',
    'static_features': 'zero128', 'source_role_masks': 'training_labels_only',
    'parent_Adam_all_groups_preserved': True, 'private_opacity_frozen': True,
    'codec_world_map_SR_frozen': True, 'gan_active': False,
    'local_counter_starts_zero': True, 'automatic_active_promotion': False,
    'gameplay_qualified': False,
}
NEW_PREFIX = 'output_moe.'


def renderer_moe_protocol():
    return deepcopy(PROTOCOL)


def identity_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def renderer_parameter_groups(model):
    base = [p for n, p in model.base_student.named_parameters() if not n.startswith(NEW_PREFIX)]
    flow = [p for n, p in model.flow_renderer.named_parameters()
            if not n.startswith(('codec.', 'geometry_opacity.', NEW_PREFIX))]
    private = list(model.flow_renderer.geometry_opacity.parameters())
    added = [p for module in (model.base_student, model.flow_renderer)
             for n, p in module.named_parameters() if n.startswith(NEW_PREFIX)]
    if not added or len(private) != 4 or any(p.requires_grad for p in private):
        raise ValueError('Renderer MoE requires appended experts and four frozen private-opacity tensors')
    return [base, flow, private, added]


def build_renderer_moe_optimizer(model, saved):
    params = renderer_parameter_groups(model)
    old = saved['optimizer']['param_groups']
    if len(old) != 3 or any(len(group['params']) != len(wanted)
                             for group, wanted in zip(old, params[:3])):
        raise ValueError('Renderer MoE must retain the three original ordered optimizer groups')
    groups = [{**{k: deepcopy(v) for k, v in old_group.items() if k != 'params'}, 'params': wanted}
              for old_group, wanted in zip(old, params[:3])]
    groups.append({'params': params[3], 'lr': PROTOCOL['new_optimizer_lr']})
    return torch.optim.AdamW(groups, weight_decay=.01)


def restore_renderer_moe_parent(saved, model, optimizer, data, far_data, geometry_data,
                                identity, configuration, frozen, *, parent_checkpoint_sha256):
    required = {'base', 'flow', 'optimizer', 'step', 'sampler', 'far_wall_aux_sampler',
                'geometry_wall_aux_sampler', 'identity', 'configuration'}
    if (not required <= saved.keys() or type(saved['step']) is not int or saved['step'] < 1
            or not re.fullmatch('[0-9a-f]{64}', parent_checkpoint_sha256)
            or identity.get('renderer_moe_protocol') != PROTOCOL
            or identity.get('parent_identity_sha256') != identity_digest(saved['identity'])
            or configuration.get('inherited_parent_configuration') != saved['configuration']
            or identity.get('codec_sha256') != saved['identity']['codec_sha256']
            or identity.get('feature_world_sha256') == saved['identity'].get('world_sha256')):
        raise ValueError('Renderer MoE needs an explicit full parent and distinct pinned current feature world')
    # Data/panels remain exact even though the newly frozen feature world changes.
    for key in ('train', 'val', 'panel', 'panel_fingerprint', 'distance_sampling',
                'far_wall_aux_pool', 'geometry_wall_aux_pool', 'geometry_wall_panel',
                'geometry_wall_fingerprint', 'native_area_guard_panel', 'native_area_guard_fingerprint'):
        if key in saved['identity'] and not _same(identity.get(key), saved['identity'][key]):
            raise ValueError('Renderer MoE changed an inherited Source identity or panel: ' + key)
    current_states = {'base': model.base_student.state_dict(), 'flow': model.flow_renderer.state_dict()}
    additions = {}
    for head, current in current_states.items():
        additions[head] = {k: v.detach().clone() for k, v in current.items() if k.startswith(NEW_PREFIX)}
        old = saved[head]
        if (not additions[head] or set(current) != set(old) | set(additions[head])
                or any(not torch.is_tensor(v) or v.shape != current[k].shape or v.dtype != current[k].dtype
                       or not bool(torch.isfinite(v).all()) for k, v in old.items())):
            raise ValueError('Renderer MoE changed an existing Base/RF architecture')
    params = renderer_parameter_groups(model)
    original = saved['optimizer']; groups = original['param_groups']
    ids = [i for group in groups for i in group['params']]
    current_optimizer = optimizer.state_dict()
    if (len(ids) != len(set(ids)) or set(ids) != set(original['state'])
            or len(current_optimizer['param_groups']) != 4
            or any([id(p) for p in group['params']] != [id(p) for p in wanted]
                   for group, wanted in zip(optimizer.param_groups, params))):
        raise ValueError('Renderer MoE lost an inherited moment or canonical parameter group')
    for old_group, new_group, wanted in zip(groups, current_optimizer['param_groups'], params):
        if (len(old_group['params']) != len(wanted) or not _same(
                {k: v for k, v in old_group.items() if k != 'params'},
                {k: v for k, v in new_group.items() if k != 'params'})):
            raise ValueError('Renderer MoE changed an inherited Adam option')
        for index, parameter in zip(old_group['params'], wanted):
            moment = original['state'][index]
            if (not {'step', 'exp_avg', 'exp_avg_sq'} <= moment.keys()
                    or float(moment['step']) <= 0
                    or any(moment[key].shape != parameter.shape or not bool(torch.isfinite(moment[key]).all())
                           for key in ('exp_avg', 'exp_avg_sq'))):
                raise ValueError('Renderer MoE inherited Adam state has wrong shape or age')
    new_group = deepcopy(current_optimizer['param_groups'][3])
    if (set(new_group['params']) & set(ids) or new_group['lr'] != PROTOCOL['new_optimizer_lr']):
        raise ValueError('New renderer experts require a separate fresh Adam group')
    for head, module in (('base', model.base_student), ('flow', model.flow_renderer)):
        module.load_state_dict({**current_states[head], **saved[head]}, strict=True)
    optimizer.load_state_dict({'state': deepcopy(original['state']),
                               'param_groups': deepcopy(groups) + [new_group]})
    for stream, key in ((data, 'sampler'), (far_data, 'far_wall_aux_sampler'),
                        (geometry_data, 'geometry_wall_aux_sampler')):
        stream.restore_sampler(deepcopy(saved[key]))
        if not _same(stream.sampler_state(), saved[key]):
            raise AssertionError('Renderer MoE changed an inherited Source sampler')
    for head, module in (('base', model.base_student), ('flow', model.flow_renderer)):
        now = module.state_dict()
        if (not _same(saved[head], {k: v for k, v in now.items() if not k.startswith(NEW_PREFIX)})
                or not _same(additions[head], {k: v for k, v in now.items() if k.startswith(NEW_PREFIX)})):
            raise AssertionError('Renderer MoE adoption changed source weights or new initialization')
    if not _same(optimizer.state_dict()['state'], original['state']):
        raise AssertionError('Renderer MoE adoption changed inherited Adam moments')
    rng = restore_renderer_native_rng(saved, device=next(model.parameters()).device)
    return dict(kind=PROTOCOL['name'], version=1, start_step=0, parent_step=saved['step'],
        parent_checkpoint_sha256=parent_checkpoint_sha256, identity=deepcopy(identity),
        parent_feature_identity=deepcopy(saved['identity'].get('frozen_feature_snapshot')),
        feature_world_changed_explicitly=True, all_old_Base_RF_private_tensors_bitexact=True,
        all_old_Adam_states_options_bitexact=True, inherited_Adam_states=len(ids),
        new_Adam_states_at_start=0, added_parameter_tensors=len(params[3]),
        added_parameters=sum(p.numel() for p in params[3]), all_three_Source_samplers_bitexact=True,
        frozen_current_feature_fingerprint=frozen, **rng,
        private_opacity_frozen=True, automatic_active_promotion=False, gameplay_qualified=False)
