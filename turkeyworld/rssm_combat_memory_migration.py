"""Append persistent Combat RSSM memory to a complete trained MoE checkpoint.

Existing tensors, named Adam groups/states, RNG, source clocks and the active
31-stage gate are retained. Only the new combat module and an empty Adam group
are added; packed P/E memory remains the unchanged prefix of a fresh episode.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, fields
import hashlib
import json
import os
from pathlib import Path

import torch

from .models import NeuralWorldModel, WorldModelConfig, STATE_SCHEMA, parameter_counts
from .rssm_migration import configure_rssm_training
from .training_pipeline import atomic_json, replace_file


COMBAT_PREFIX = 'vjepa_backend.combat_rssm.'
COMBAT_MEMORY_PROTOCOL = 'append_persistent_combat_rssm'


def file_sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def _config_sha(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def _combat_config(config):
    from .combat_rssm import CombatRSSMConfig
    available = {field.name for field in fields(CombatRSSMConfig)}
    return CombatRSSMConfig(**{key.removeprefix('combat_rssm_'): value
        for key, value in asdict(config).items()
        if key.startswith('combat_rssm_') and key.removeprefix('combat_rssm_') in available})


def enable_combat_memory_checkpoint(saved, *, seed=173, run_id=None, config_overrides=None):
    """Return an independent full payload and strict append-only CPU proof."""
    from .combat_rssm import CombatRSSM
    from .renderer_student import renderer_condition_identity
    if type(seed) is not int or seed < 0:
        raise ValueError('Combat memory initialization requires a nonnegative integer seed')
    if not isinstance(saved, dict) or saved.get('stage') != 'world' or saved.get('state_schema') != STATE_SCHEMA:
        raise ValueError('Combat memory requires a complete one-enemy world checkpoint')
    old_config = WorldModelConfig(**saved['config'])
    if (old_config.kind != 'teacher' or old_config.architecture != 'dreamer_rssm'
            or not old_config.enemy_rssm_enabled or not old_config.rssm_fusion_moe_enabled
            or old_config.combat_rssm_enabled or not old_config.spatial_entities
            or not old_config.spatial_combat or not old_config.endless):
        raise ValueError('Expected the existing dual RSSMs plus trained shared MoE, before Combat memory')
    step = saved.get('global_training_step', saved.get('training_steps'))
    run = saved.get('run_state')
    if (type(step) is not int or step < 0 or saved.get('training_steps') != step
            or not isinstance(run, dict) or run.get('global_training_step') != step
            or saved.get('training_counters', {}).get('pending_microbatches', 0)):
        raise ValueError('Combat memory requires a complete saved optimizer boundary')
    state = saved.get('model')
    if (not isinstance(state, dict) or not state or any(not torch.is_tensor(value)
            or value.device.type != 'cpu' or value.is_floating_point() and not bool(torch.isfinite(value).all())
            for value in state.values()) or any(name.startswith(COMBAT_PREFIX) for name in state)):
        raise ValueError('Expected finite CPU source tensors without a partial Combat module')
    if not isinstance(saved.get('rssm_fusion_migration'), dict):
        raise ValueError('Existing shared MoE migration lineage must be retained')
    from .rssm_moe_protocol import moe_training_horizon
    try:
        moe_training_horizon(run)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('Combat memory must retain a complete current31-stage MoE gate') from error
    overrides = dict(config_overrides or {})
    config = asdict(old_config)
    if (any(not name.startswith('combat_rssm_') or name not in config for name in overrides)
            or overrides.get('combat_rssm_enabled', True) is not True):
        raise ValueError('Only the new Combat RSSM configuration may change')
    config.update(combat_rssm_enabled=True)
    config.update(overrides)
    new_config = WorldModelConfig(**config)
    with torch.random.fork_rng(devices=[]), torch.device('meta'):
        previous = NeuralWorldModel(old_config)
        target = NeuralWorldModel(new_config)
    expected_old = previous.state_dict()
    if (set(expected_old) != set(state) or any(value.shape != expected_old[name].shape
            or value.dtype != expected_old[name].dtype for name, value in state.items())):
        raise ValueError('Existing P/E/MoE namespace, shape or dtype differs from its source configuration')
    previous.load_state_dict(dict(state), strict=True, assign=True)
    width = old_config.vjepa_width or 512
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        combat = CombatRSSM(_combat_config(new_config), width=width,
            observation_dim=old_config.rssm_embed_dim,
            fusion_width=old_config.rssm_fusion_moe_width)
    target.vjepa_backend.combat_rssm = combat
    expected_added = set(target.state_dict()) - set(state)
    if not expected_added or any(not name.startswith(COMBAT_PREFIX) for name in expected_added):
        raise ValueError('Only the appended Combat module may add model tensors')
    missing, unexpected = target.load_state_dict(dict(state), strict=False, assign=True)
    if unexpected or set(missing) != expected_added:
        raise ValueError('Combat memory cannot omit or replace any trained P/E/MoE tensor')
    migrated = target.state_dict()
    for name, value in state.items():
        if not torch.equal(value, migrated[name]):
            raise AssertionError('Combat append changed an old tensor: ' + name)
        migrated[name] = value
    for name in expected_added:
        migrated[name] = migrated[name].detach().clone()
    configure_rssm_training(previous)
    policy = configure_rssm_training(target)
    old_canonical = [name for name, value in previous.named_parameters() if value.requires_grad]
    new_canonical = [name for name, value in target.named_parameters() if value.requires_grad]
    names = saved.get('optimizer_parameter_names')
    optimizer = saved.get('optimizer')
    if (not isinstance(names, list) or len(names) != len(set(names)) or set(names) != set(old_canonical)
            or not isinstance(optimizer, dict) or not isinstance(optimizer.get('state'), dict)
            or not isinstance(optimizer.get('param_groups'), list) or not optimizer['param_groups']):
        raise ValueError('The complete named old Adam namespace/groups are required')
    added = [name for name in new_canonical if name not in set(names)]
    if (not added or set(new_canonical) != set(names + added)
            or any(not name.startswith(COMBAT_PREFIX) for name in added)):
        raise ValueError('Only the appended Combat parameters may join Adam')
    groups = optimizer['param_groups']
    old_ids = [identifier for group in groups for identifier in group.get('params', [])]
    if (old_ids != list(range(len(names))) or any(type(identifier) is not int for identifier in old_ids)
            or set(optimizer['state']) - set(old_ids)):
        raise ValueError('Old Adam IDs must map exactly to saved parameter-name order')
    for identifier, name in enumerate(names):
        for label, value in optimizer['state'].get(identifier, {}).items():
            if torch.is_tensor(value) and (value.device.type != 'cpu'
                    or label == 'step' and (value.numel() != 1 or not 0 <= float(value) <= step)
                    or label != 'step' and value.shape != state[name].shape
                    or value.is_floating_point() and not bool(torch.isfinite(value).all())):
                raise ValueError('Invalid retained Adam moment or mixed age: ' + name + '/' + label)
    for group in groups:
        if 'param_names' in group and group['param_names'] != [names[i] for i in group['params']]:
            raise ValueError('Old named Adam group identity changed')
    group = copy.deepcopy({key: value for key, value in groups[0].items() if key not in {'params', 'param_names'}})
    group['params'] = list(range(len(names), len(names) + len(added)))
    if 'param_names' in groups[0]:
        group['param_names'] = list(added)
    with torch.no_grad():
        combat.eval()
        initial = combat.initial(2)
        for sample in (False, True):
            before_rng = torch.get_rng_state().clone()
            prior = combat.prior(initial, torch.zeros(2, 20), torch.zeros(2, 4),
                torch.zeros(2, 2), torch.zeros(2, width), torch.zeros(2, 2, width),
                player_context=torch.zeros(2, width), enemy_context=torch.zeros(2, width), sample=sample)
            if any(bool(prior[name].count_nonzero()) for name in
                    ('target_residual', 'attack_residual', 'spawn_tile_residual', 'spawn_anchor_residual')):
                raise AssertionError('Combat task additions must start at exact zero')
            if bool(combat.fusion_context(prior['memory']).count_nonzero()):
                raise AssertionError('Combat to existing MoE query must start at exact zero')
            if not torch.equal(before_rng, torch.get_rng_state()):
                raise AssertionError('New Combat sampling changed the established actor RNG stream')
    old_memory = previous.vjepa_backend.memory_size
    combat_memory = combat.memory_size
    if target.vjepa_backend.memory_size != old_memory + combat_memory:
        raise AssertionError('Only one appended Combat h/z segment may change packed memory')
    counts = parameter_counts(target)
    combat_count = sum(parameter.numel() for parameter in combat.parameters())
    identity = renderer_condition_identity(target)
    if identity != renderer_condition_identity(previous):
        raise AssertionError('Combat memory changed frozen image conditioning identity')
    changed = sorted(name for name, value in config.items() if value != asdict(old_config)[name])
    report = {'format_version': 1, 'kind': COMBAT_MEMORY_PROTOCOL, 'source_step': step, 'seed': seed,
        'source_config': asdict(old_config), 'target_config': config,
        'source_config_sha256': _config_sha(asdict(old_config)), 'target_config_sha256': _config_sha(config),
        'allowed_config_changes': changed, 'old_model_tensors_bitexact': True,
        'old_tensor_count': len(state), 'old_optimizer_parameter_names_preserved': True,
        'old_parameter_order_is_prefix': new_canonical[:len(names)] == names,
        'added_parameter_names': added, 'added_parameter_count': combat_count,
        'added_tensor_count': len(expected_added), 'combat_parameter_count': combat_count,
        'zero_residual_initialization': True, 'zero_combat_to_existing_MoE_query': True,
        'source_memory_size': old_memory, 'old_memory_size': old_memory,
        'source_combined_memory_size': old_memory,
        'combined_memory_size': target.vjepa_backend.memory_size,
        'player_memory_size': previous.vjepa_backend.core.memory_size,
        'enemy_memory_size': previous.vjepa_backend.enemy_rssm.memory_size,
        'combat_memory_size': combat_memory, 'combat_deterministic_dim': new_config.combat_rssm_deter_dim,
        'combat_stoch': new_config.combat_rssm_stoch, 'combat_classes': new_config.combat_rssm_classes,
        'memory_schema': {'version': 2, 'order': ['player', 'enemy', 'combat'],
            'old_prefix_size': old_memory, 'appended_combat_size': combat_memory,
            'live_memory_migrated': False, 'policy': 'Fresh episode initialization; never splice a live game'},
        'old_optimizer_groups_preserved': len(groups), 'old_adam_states_preserved': len(optimizer['state']),
        'new_optimizer_groups': 1, 'new_optimizer_state_entries': 0,
        'old_mixed_Adam_ages_preserved_not_reset_to_global': True,
        'source_renderer_conditioning_identity': identity, 'frozen_render_assets_bitexact': True,
        'rng_payloads_preserved': True, 'current_moe_gate_streak_and_all_history_preserved': True,
        'no_objective_or_horizon_restart': True, 'model_parameter_counts': counts,
        'new_branch_RNG_isolated_by_backend': True,
        'gameplay_qualified': False, 'automatic_active_promotion': False}
    new_run = copy.deepcopy(run)
    if run_id is not None:
        new_run['run_id'] = str(run_id)
    new_run['combat_rssm_migration'] = report
    new_run['world_dynamics_protocol'] = 'triple_rssm_shared_fusion_moe_v1'
    new_run['model_parameters'] = {**run.get('model_parameters', {}), **counts}
    q = new_run.setdefault('rssm_training', {})
    q.update(combat_rssm_enabled=True, combat_parameter_count=combat_count,
        combat_deterministic_dim=new_config.combat_rssm_deter_dim,
        combat_stoch=new_config.combat_rssm_stoch, combat_classes=new_config.combat_rssm_classes,
        combat_memory_size=combat_memory, memory_size=target.vjepa_backend.memory_size,
        parameter_count=counts['dynamics_including_map'], dynamics_trainable=policy['dynamics_trainable'])
    result = {**saved, 'config': config, 'model': migrated,
        'optimizer_parameter_names': names + added,
        'optimizer': {**optimizer, 'state': dict(optimizer['state']),
                      'param_groups': [copy.deepcopy(value) for value in groups] + [group]},
        'combat_rssm_migration': report, 'run_state': new_run, 'rssm_training': copy.deepcopy(q)}
    return result, report


def prepare_combat_memory_continuation(source, destination, *, seed=173, run_id=None, config_overrides=None):
    """Write a separate checkpoint/proof; never stop a worker or overwrite source."""
    source, destination = Path(source).resolve(strict=True), Path(destination).resolve()
    if source == destination or destination.exists():
        raise FileExistsError('Combat memory requires a new separate checkpoint destination')
    original_sha = file_sha(source)
    saved = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    result, report = enable_combat_memory_checkpoint(saved, seed=seed, run_id=run_id, config_overrides=config_overrides)
    if file_sha(source) != original_sha:
        raise ValueError('Source changed during migration; use a stopped immutable boundary')
    report.update(source_checkpoint=str(source), source_checkpoint_sha256=original_sha,
                  checkpoint=str(destination), source_plan_sha256=saved.get('plan_sha256'))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f'.{os.getpid()}.tmp')
    try:
        torch.save(result, temporary)
        replace_file(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    report['checkpoint_sha256'] = file_sha(destination)
    atomic_json(destination.with_suffix(destination.suffix + '.combat-memory.json'), report)
    return report
