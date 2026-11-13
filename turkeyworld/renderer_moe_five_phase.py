"""Explicit three-to-five renderer expert continuation with named Adam adoption."""
from copy import deepcopy
import math
import re

import torch

from .renderer_moe_phase import identity_digest
from .renderer_native_phase import _same, restore_renderer_native_rng


ROLES = ['near_geometry_player', 'far_geometry', 'enemy_motion', 'enemy_combat', 'player_combat']
ROLE_SCHEMA = 'near_far_enemy_motion_enemy_combat_player_combat_v2'
PROTOCOL = dict(
    name='one_enemy_renderer_token_moe_five_v1', version=1, roles=ROLES,
    role_schema=ROLE_SCHEMA, routing='hierarchical_soft_per_token',
    parent_roles=['scene_walls', 'enemy', 'combat'], logical_to_physical_experts=[0, 3, 1, 4, 2],
    experts_per_head=5, expert_transformer_blocks=1, expert_widths=dict(base=8, rf1=32),
    near_far_boundary_tiles=5., near_far_comparison='near_strict_less_than',
    sky_floor_depth='same_Source_camera_column_first_wall_depth',
    source_role_masks_loss_only=True, source_role_masks='training_labels_only',
    enemy_combat_labels='current_hurt_dead_hp0_attack_pose_reset_or_recorded_combat_case',
    player_combat_labels='Source_weapon_during_flash_recorded_shot_hurt_HP_drop;full_frame_only_player_HP0',
    idle_player_weapon_role='near_geometry_player_regardless_wall_column_depth',
    router_role_weight=.01, router_balance_weight=.001, new_optimizer_lr=.0001,
    main_batch=16, far_auxiliary_batch=4, far_auxiliary_weight=.25,
    geometry_auxiliary_batch=8, geometry_auxiliary_weight=1.,
    feature_policy='frozen_current_world_previous_RGB_state_action_prior_by_Source_environment',
    navigation_profile='explicit_Source_row_metadata_not_predicted_life_flags', static_features='zero128',
    parent_Adam_all_groups_preserved=True, private_opacity_frozen=True,
    codec_world_map_SR_frozen=True, feature_world_changed=False, gan_active=False,
    local_counter_starts_zero=True, automatic_active_promotion=False, gameplay_qualified=False)


def renderer_moe_five_protocol():
    return deepcopy(PROTOCOL)


def original_renderer_configuration(configuration):
    """Find the original RGB/data settings without rewriting parent provenance."""
    value = configuration
    seen = set()
    while isinstance(value, dict) and id(value) not in seen:
        seen.add(id(value))
        if {'combat_data_root', 'appearance_data_root'} <= value.keys():
            return deepcopy(value)
        value = value.get('inherited_parent_configuration')
    raise ValueError('Five-role continuation needs the original pinned Source/RGB configuration')


def is_added_name(name):
    return (name.startswith(('output_moe.experts.3.', 'output_moe.experts.4.',
                             'output_moe.split_router.')))


def renderer_five_parameter_groups(model):
    named = [('base_student.' + n, p) for n, p in model.base_student.named_parameters()]
    flow = [('flow_renderer.' + n, p) for n, p in model.flow_renderer.named_parameters()]
    groups = [[], [], [], [], []]
    names = [[], [], [], [], []]
    for head, values in (('base_student.', named), ('flow_renderer.', flow)):
        for full_name, parameter in values:
            local = full_name[len(head):]
            if head == 'flow_renderer.' and local.startswith('codec.'):
                continue
            if is_added_name(local): index = 4
            elif local.startswith('output_moe.'): index = 3
            elif head == 'flow_renderer.' and local.startswith('geometry_opacity.'): index = 2
            else: index = 0 if head == 'base_student.' else 1
            groups[index].append(parameter); names[index].append(full_name)
    if (any(not group for group in groups) or len(groups[2]) != 4
            or any(p.requires_grad for p in groups[2])
            or len({id(p) for group in groups for p in group}) != sum(map(len, groups))):
        raise ValueError('Five-role renderer needs four unchanged parent groups and a fresh appended group')
    return groups, names


def build_renderer_moe_five_optimizer(model, saved):
    params, _ = renderer_five_parameter_groups(model)
    old = saved['optimizer']['param_groups']
    if len(old) != 4 or any(len(a['params']) != len(b) for a, b in zip(old, params[:4])):
        raise ValueError('Five-role renderer must preserve four complete ordered parent Adam groups')
    groups = [{**deepcopy({k: v for k, v in old_group.items() if k != 'params'}), 'params': wanted}
              for old_group, wanted in zip(old, params[:4])]
    groups.append(dict(params=params[4], lr=PROTOCOL['new_optimizer_lr']))
    return torch.optim.AdamW(groups, weight_decay=.01)


def restore_renderer_moe_five_parent(saved, model, optimizer, data, far_data, geometry_data,
                                     identity, configuration, frozen, *, parent_checkpoint_sha256):
    """Adopt old mixed-age moments; new cloned experts start with fresh moments."""
    required = {'base', 'flow', 'optimizer', 'step', 'sampler', 'far_wall_aux_sampler',
                'geometry_wall_aux_sampler', 'identity', 'configuration', 'frozen_fingerprint',
                'renderer_training_steps', 'base_training_steps'}
    if (not required <= saved.keys() or type(saved['step']) is not int or saved['step'] < 1
            or not re.fullmatch('[0-9a-f]{64}', parent_checkpoint_sha256)
            or identity.get('renderer_moe_protocol') != PROTOCOL
            or identity.get('parent_identity_sha256') != identity_digest(saved['identity'])
            or configuration.get('inherited_parent_configuration') != saved['configuration']
            or saved['identity'].get('renderer_moe_protocol', {}).get('name') != 'one_enemy_renderer_token_moe_v1'
            or saved['frozen_fingerprint'] != frozen):
        raise ValueError('Five-role continuation requires the exact complete parent, identity and frozen tensors')
    allowed = {'renderer_moe_protocol', 'parent_identity_sha256'}
    if ({k: v for k, v in identity.items() if k not in allowed}
            != {k: v for k, v in saved['identity'].items() if k not in allowed}):
        raise ValueError('Five-role continuation changed a Source/feature-world/codec/panel asset identity')
    params, names = renderer_five_parameter_groups(model)
    if saved.get('optimizer_parameter_names', names[:4]) != names[:4]:
        raise ValueError('Parent Adam parameter names/order changed')
    for head, module in (('base', model.base_student), ('flow', model.flow_renderer)):
        current = module.state_dict(); old = saved[head]
        additions = {k for k in current if is_added_name(k)}
        if (not additions or set(current) != set(old) | additions
                or any(not _same(value, current[name]) for name, value in old.items())
                or any(not bool(torch.isfinite(current[k]).all()) for k in additions)):
            raise ValueError('Five-role migration changed an old tensor or has invalid added tensors')
        for clone, source in ((3, 0), (4, 1)):
            for name, value in current.items():
                prefix = 'output_moe.experts.' + str(clone) + '.'
                if name.startswith(prefix) and not _same(value, old[name.replace(prefix,
                        'output_moe.experts.' + str(source) + '.', 1)]):
                    raise ValueError('New renderer expert is not the declared exact parent clone')
        if any(bool(current[k].count_nonzero()) for k in additions if k.startswith('output_moe.split_router.')):
            raise ValueError('Initial hierarchical splits must be zero for function-preserving adoption')
    old_optimizer = saved['optimizer']; old_groups = old_optimizer['param_groups']
    old_ids = [i for g in old_groups for i in g['params']]
    current = optimizer.state_dict(); new_groups = current['param_groups']
    if (len(old_groups) != 4 or len(new_groups) != 5 or len(set(old_ids)) != len(old_ids)
            or set(old_ids) != set(old_optimizer['state'])
            or any([id(p) for p in g['params']] != [id(p) for p in wanted]
                   for g, wanted in zip(optimizer.param_groups, params))):
        raise ValueError('Five-role adoption lost parent moments or canonical optimizer groups')
    adopted = {}
    for old_group, new_group, wanted in zip(old_groups, new_groups[:4], params[:4]):
        if (len(old_group['params']) != len(wanted) or not _same(
                {k: v for k, v in old_group.items() if k != 'params'},
                {k: v for k, v in new_group.items() if k != 'params'})):
            raise ValueError('Five-role adoption changed parent Adam options/order')
        for old_id, new_id, parameter in zip(old_group['params'], new_group['params'], wanted):
            moment = old_optimizer['state'][old_id]
            if (not {'step', 'exp_avg', 'exp_avg_sq'} <= moment.keys()
                    or not math.isfinite(float(moment['step'])) or float(moment['step']) <= 0
                    or any(moment[k].shape != parameter.shape or not bool(torch.isfinite(moment[k]).all())
                           for k in ('exp_avg', 'exp_avg_sq'))):
                raise ValueError('Invalid parent Adam moment shape/age')
            adopted[new_id] = deepcopy(moment)
    optimizer.load_state_dict(dict(state=adopted, param_groups=deepcopy(new_groups)))
    for stream, key in ((data, 'sampler'), (far_data, 'far_wall_aux_sampler'),
                        (geometry_data, 'geometry_wall_aux_sampler')):
        stream.restore_sampler(deepcopy(saved[key]))
        if not _same(stream.sampler_state(), saved[key]):
            raise AssertionError('Five-role adoption changed a parent Source sampler')
    if not _same(optimizer.state_dict()['state'], adopted):
        raise AssertionError('Five-role adoption changed an inherited Adam tensor')
    rng = restore_renderer_native_rng(saved, device=next(model.parameters()).device)
    return dict(kind=PROTOCOL['name'], version=1, start_step=0, parent_step=saved['step'],
        parent_checkpoint_sha256=parent_checkpoint_sha256, identity=deepcopy(identity),
        parent_renderer_training_steps=saved['renderer_training_steps'],
        parent_base_training_steps=saved['base_training_steps'],
        feature_world_changed_explicitly=False, all_old_Base_RF_private_tensors_bitexact=True,
        all_old_Adam_states_options_bitexact=True, inherited_Adam_states=len(old_ids),
        optimizer_parameter_names=names, new_Adam_states_at_start=0,
        added_parameter_tensors=len(params[4]), added_parameters=sum(p.numel() for p in params[4]),
        all_three_Source_samplers_bitexact=True, frozen_current_feature_fingerprint=frozen,
        private_opacity_frozen=True, automatic_active_promotion=False, gameplay_qualified=False, **rng)


def validate_renderer_moe_five_resume(saved, phase, *, identity, configuration, frozen, plan_sha256,
                                      parent_checkpoint_sha256):
    """Reject a different phase, counter reset or stale optimizer-name ledger before loading."""
    step = saved.get('step')
    expected_names = phase.get('optimizer_parameter_names')
    if (type(step) is not int or not 0 <= step <= configuration['steps']
            or phase.get('kind') != PROTOCOL['name']
            or phase.get('parent_checkpoint_sha256') != parent_checkpoint_sha256
            or phase.get('identity') != identity or phase.get('configuration') != configuration
            or phase.get('frozen_current_feature_fingerprint') != frozen
            or saved.get('identity') != identity or saved.get('configuration') != configuration
            or saved.get('frozen_fingerprint') != frozen
            or saved.get('renderer_moe_phase') != phase
            or saved.get('renderer_moe_plan_sha256') != plan_sha256
            or saved.get('parent_step') != phase.get('parent_step')
            or saved.get('renderer_training_steps') != phase.get('parent_renderer_training_steps', -1) + step
            or saved.get('base_training_steps') != phase.get('parent_base_training_steps', -1) + step
            or saved.get('optimizer_parameter_names') != expected_names
            or len(expected_names or []) != 5
            or len(saved.get('optimizer', {}).get('param_groups', [])) != 5
            or any(len(a['params']) != len(b)
                   for a, b in zip(saved['optimizer']['param_groups'], expected_names))):
        raise ValueError('Five-role resume changed its immutable phase, parent, counters or optimizer-name ledger')
