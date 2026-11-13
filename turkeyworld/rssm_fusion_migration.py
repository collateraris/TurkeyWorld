"""Append a zero-residual shared MoE to two existing RSSMs, without rollback.

The source full checkpoint is immutable. Every old tensor, Adam ID/group/state,
RNG payload, numerical memory layout and curriculum history is retained. Only
the new fusion module and its initially empty optimizer group are added.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import torch

from .models import NeuralWorldModel, WorldModelConfig, STATE_SCHEMA, parameter_counts
from .rssm_migration import configure_rssm_training
from .training_pipeline import atomic_json, replace_file


FUSION_PREFIX = 'vjepa_backend.fusion_moe.'
FUSION_DEFAULTS = {'rssm_fusion_moe_enabled': True, 'rssm_fusion_moe_width': 256,
    'rssm_fusion_moe_depth': 2, 'rssm_fusion_moe_heads': 4, 'rssm_fusion_moe_expert_hidden': 512,
    'rssm_fusion_moe_experts': 3, 'rssm_fusion_moe_top_k': 2, 'rssm_fusion_moe_role_bias': 2.}
RESIDUAL_KEYS = ('player_pass_residual', 'enemy_motion_residual', 'interaction_target_residual',
                 'interaction_attack_residual', 'interaction_spawn_tile_residual',
                 'interaction_spawn_anchor_residual')


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _config_sha(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def enable_rssm_fusion_checkpoint(saved, *, seed=173, run_id=None, config_overrides=None):
    """Return a separate full payload and explicit append-only migration proof."""
    from .rssm_fusion_moe import RSSMFusionMoE, RSSMFusionMoEConfig
    from .renderer_student import renderer_condition_identity
    if type(seed) is not int or seed < 0:
        raise ValueError('Fusion initialization requires a nonnegative integer seed')
    if not isinstance(saved, dict) or saved.get('stage') != 'world':
        raise ValueError('Fusion migration requires a complete dual-RSSM world checkpoint')
    old_config = WorldModelConfig(**saved['config'])
    if (old_config.kind != 'teacher' or old_config.architecture != 'dreamer_rssm'
            or not old_config.enemy_rssm_enabled or old_config.rssm_fusion_moe_enabled
            or old_config.player_motion_mode != 'axis_fraction'
            or not old_config.spatial_entities or not old_config.spatial_combat or not old_config.endless):
        raise ValueError('Expected the unchanged spatial dual RSSM before shared fusion')
    if saved.get('state_schema') != STATE_SCHEMA:
        raise ValueError('Shared fusion cannot change the permanent one-enemy state schema')
    step = saved.get('global_training_step', saved.get('training_steps'))
    run_state = saved.get('run_state')
    if (type(step) is not int or step < 0 or saved.get('training_steps') != step
            or not isinstance(run_state, dict) or run_state.get('global_training_step') != step
            or saved.get('training_counters', {}).get('pending_microbatches', 0)):
        raise ValueError('Fusion migration requires a complete saved optimizer boundary')
    state = saved.get('model')
    if (not isinstance(state, dict) or not state or any(not torch.is_tensor(v)
            or v.device.type != 'cpu' or v.is_floating_point() and not bool(torch.isfinite(v).all())
            for v in state.values()) or any(k.startswith(FUSION_PREFIX) for k in state)):
        raise ValueError('Need finite CPU source tensors without an existing or partial fusion module')
    overrides = dict(config_overrides or {})
    if set(overrides)-FUSION_DEFAULTS.keys() or overrides.get('rssm_fusion_moe_enabled', True) is not True:
        raise ValueError('Fusion migration may override only its new explicit module dimensions')
    config = {**asdict(old_config), **FUSION_DEFAULTS, **overrides}
    new_config = WorldModelConfig(**config)
    with torch.random.fork_rng(devices=[]), torch.device('meta'):
        previous = NeuralWorldModel(old_config)
        target = NeuralWorldModel(new_config)
    expected_old = previous.state_dict()
    if (state.keys() != expected_old.keys() or any(v.shape != expected_old[n].shape
            or v.dtype != expected_old[n].dtype for n, v in state.items())):
        raise ValueError('Existing dual-RSSM tensor namespace, shapes or dtypes changed')
    width = old_config.vjepa_width or 512
    fusion_config = RSSMFusionMoEConfig(width=new_config.rssm_fusion_moe_width,
        depth=new_config.rssm_fusion_moe_depth, heads=new_config.rssm_fusion_moe_heads,
        expert_hidden=new_config.rssm_fusion_moe_expert_hidden, experts=new_config.rssm_fusion_moe_experts,
        top_k=new_config.rssm_fusion_moe_top_k, role_bias=new_config.rssm_fusion_moe_role_bias)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        fusion = RSSMFusionMoE(fusion_config, player_memory_dim=target.vjepa_backend.core.memory_size,
            enemy_memory_dim=target.vjepa_backend.enemy_rssm.memory_size, map_width=width, spatial_width=width)
    target.vjepa_backend.fusion_moe = fusion
    expected_added = set(target.state_dict())-set(state)
    if not expected_added or any(not name.startswith(FUSION_PREFIX) for name in expected_added):
        raise ValueError('Only the appended shared fusion module may be new')
    missing, unexpected = target.load_state_dict(dict(state), strict=False, assign=True)
    if unexpected or set(missing) != expected_added:
        raise ValueError('Fusion append cannot omit or replace an existing world tensor')
    migrated = target.state_dict()
    for name, value in state.items():
        if not torch.equal(value, migrated[name]):
            raise AssertionError('Fusion migration changed an existing tensor: '+name)
        migrated[name] = value
    for name in expected_added:
        migrated[name] = migrated[name].detach().clone()
    with torch.no_grad():
        fusion.eval()
        result = fusion(torch.zeros(2, target.vjepa_backend.core.memory_size),
            torch.zeros(2, target.vjepa_backend.enemy_rssm.memory_size), torch.zeros(2, 20),
            torch.zeros(2, 4), torch.zeros(2, width), torch.zeros(2, 2, width))
        if any(bool(result[key].count_nonzero()) for key in RESIDUAL_KEYS):
            raise AssertionError('Shared fusion must begin with exactly zero task residuals')
    configure_rssm_training(previous)
    canonical_old = [name for name, value in previous.named_parameters() if value.requires_grad]
    configure_rssm_training(target)
    canonical_target = [name for name, value in target.named_parameters() if value.requires_grad]
    added = [name for name, _ in target.named_parameters() if name.startswith(FUSION_PREFIX)]
    if canonical_target != canonical_old+added:
        raise ValueError('Appending fusion must preserve every existing canonical optimizer name/order')
    optimizer, names = saved.get('optimizer'), saved.get('optimizer_parameter_names')
    if (names != canonical_old or not isinstance(optimizer, dict)
            or not isinstance(optimizer.get('state'), dict) or not optimizer.get('param_groups')):
        raise ValueError('The complete canonical old Adam state/name map is required')
    groups = optimizer['param_groups']
    old_ids = [key for group in groups for key in group.get('params', [])]
    if old_ids != list(range(len(names))) or any(type(key) is not int for key in old_ids) or set(optimizer['state'])-set(old_ids):
        raise ValueError('Old Adam group IDs must map exactly to unchanged canonical parameter names')
    for key, name in enumerate(names):
        for label, value in optimizer['state'].get(key, {}).items():
            if torch.is_tensor(value) and (value.device.type != 'cpu'
                    or label == 'step' and (value.numel() != 1 or float(value) < 0 or float(value) > step)
                    or label != 'step' and value.shape != state[name].shape
                    or value.is_floating_point() and not bool(torch.isfinite(value).all())):
                raise ValueError('Invalid retained Adam moment or age: '+name+'/'+label)
    for group in groups:
        if 'param_names' in group and group['param_names'] != [names[key] for key in group['params']]:
            raise ValueError('Old optimizer group parameter names changed')
    new_group = copy.deepcopy({k: v for k, v in groups[0].items() if k not in {'params', 'param_names'}})
    new_group['params'] = list(range(len(names), len(names)+len(added)))
    if 'param_names' in groups[0]:
        new_group['param_names'] = added
    if target.vjepa_backend.memory_size != previous.vjepa_backend.memory_size:
        raise AssertionError('Fusion may not change either recurrent memory layout')
    counts = parameter_counts(target)
    identity = renderer_condition_identity(target)
    report = {'format_version': 1, 'kind': 'append_shared_fusion_moe', 'source_step': step, 'seed': seed,
        'source_config': asdict(old_config), 'target_config': config,
        'source_config_sha256': _config_sha(asdict(old_config)), 'target_config_sha256': _config_sha(config),
        'allowed_config_changes': sorted(k for k in config if config[k] != asdict(old_config)[k]),
        'old_model_tensors_bitexact': True, 'old_tensor_count': len(state),
        'old_parameter_order_is_prefix': True, 'added_parameter_names': added,
        'added_parameter_count': sum(p.numel() for p in fusion.parameters()),
        'added_tensor_count': len(expected_added), 'zero_residual_initialization': True,
        'player_memory_size': target.vjepa_backend.core.memory_size,
        'enemy_memory_size': target.vjepa_backend.enemy_rssm.memory_size,
        'combined_memory_size': target.vjepa_backend.memory_size,
        'old_optimizer_groups_preserved': len(groups), 'old_adam_states_preserved': len(optimizer['state']),
        'new_optimizer_groups': 1, 'new_optimizer_state_entries': 0,
        'old_mixed_Adam_ages_preserved_not_reset_to_global': True,
        'frozen_render_assets_bitexact': True, 'source_renderer_conditioning_identity': identity,
        'model_parameter_counts': counts, 'rng_payloads_preserved': True,
        'all_old_curriculum_sampler_EMA_history_retained': True,
        'no_new_recurrent_memory_or_future_RGB_input': True,
        'zero_residual_forward_equivalence_covered_by_backend_tests': True,
        'gameplay_qualified': False, 'automatic_active_promotion': False}
    result_payload = {**saved, 'config': config, 'model': migrated,
        'optimizer_parameter_names': names+added,
        'optimizer': {**optimizer, 'state': dict(optimizer['state']),
                      'param_groups': [copy.deepcopy(group) for group in groups]+[new_group]},
        'rssm_fusion_migration': report, 'run_state': copy.deepcopy(run_state)}
    if run_id is not None:
        result_payload['run_state']['run_id'] = str(run_id)
    result_payload['run_state']['rssm_fusion_migration'] = report
    result_payload['run_state']['world_dynamics_protocol'] = 'dual_rssm_shared_fusion_moe_v1'
    # Retain old active/inactive loss/curriculum ledgers verbatim. The new
    # all-task objective is bound by a separate operator-owned immutable plan.
    result_payload['rssm_training'] = copy.deepcopy(saved.get('rssm_training', run_state.get('rssm_training', {})))
    result_payload['run_state']['model_parameters'] = {**run_state.get('model_parameters', {}), **counts}
    result_payload['run_state'].setdefault('objective_history', []).append({
        'kind': 'append_shared_fusion_moe_architecture', 'step': step,
        'added_parameters': report['added_parameter_count'], 'memory_size_unchanged': report['combined_memory_size'],
        'old_objective_and_curricula_retained_as_history': True, 'gameplay_qualified': False})
    return result_payload, report


def prepare_rssm_fusion_continuation(source, destination, *, seed=173, run_id=None, config_overrides=None):
    """Write a separate CPU checkpoint and proof; never launch or mutate source."""
    source, destination = Path(source).resolve(strict=True), Path(destination).resolve()
    if source == destination or destination.exists():
        raise FileExistsError('Choose a new separate shared-fusion checkpoint destination')
    original_sha = file_sha(source)
    saved = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    result, report = enable_rssm_fusion_checkpoint(saved, seed=seed, run_id=run_id, config_overrides=config_overrides)
    if file_sha(source) != original_sha:
        raise ValueError('Source changed while migrating; use the immutable completed archive')
    report.update(source_checkpoint=str(source), source_checkpoint_sha256=original_sha,
                  checkpoint=str(destination), source_plan_sha256=saved.get('plan_sha256'))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name+f'.{os.getpid()}.tmp')
    try:
        torch.save(result, temporary)
        replace_file(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    report['checkpoint_sha256'] = file_sha(destination)
    atomic_json(destination.with_suffix(destination.suffix+'.fusion.json'), report)
    return report
