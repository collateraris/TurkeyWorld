"""CPU-only checkpoint migration: retain the player RSSM, append enemy memory.

The source is never overwritten. Existing model/Adam/RNG/curriculum payloads
are retained; only the new enemy module and a fresh Adam group are appended.
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


ENEMY_PREFIX = 'vjepa_backend.enemy_rssm.'
ENEMY_DEFAULTS = {'enemy_rssm_enabled': True, 'enemy_rssm_deter_dim': 384,
    'enemy_rssm_hidden_dim': 384, 'enemy_rssm_stoch': 8, 'enemy_rssm_classes': 32,
    'enemy_rssm_embed_dim': 128, 'enemy_rssm_action_dim': 128,
    'enemy_rssm_blocks': 4, 'enemy_rssm_unimix': .01}


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _config_sha(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def enable_dual_rssm_checkpoint(saved, seed=173, *, run_id=None):
    """Return a new complete payload and proof without modifying the source."""
    from .enemy_rssm import EnemyRSSM, EnemyRSSMConfig
    from .renderer_student import renderer_condition_identity
    if type(seed) is not int or seed < 0:
        raise ValueError('Dual migration requires a nonnegative integer seed')
    if not isinstance(saved, dict) or saved.get('stage') != 'world':
        raise ValueError('Dual migration requires a complete world checkpoint')
    old_config = WorldModelConfig(**saved['config'])
    if (old_config.kind != 'teacher' or old_config.architecture != 'dreamer_rssm'
            or old_config.enemy_rssm_enabled or not old_config.spatial_entities or not old_config.spatial_combat):
        raise ValueError('Expected the existing single-memory spatial RSSM teacher')
    if saved.get('state_schema') != STATE_SCHEMA:
        raise ValueError('Dual migration must preserve the exact one-enemy state schema')
    step = saved.get('global_training_step', saved.get('training_steps'))
    if (type(step) is not int or step < 0 or saved.get('training_steps') != step
            or saved.get('training_counters', {}).get('pending_microbatches', 0)):
        raise ValueError('Dual migration requires a completed optimizer boundary')
    state = saved.get('model')
    if (not isinstance(state, dict) or not state or any(not torch.is_tensor(value)
            or value.device.type != 'cpu' for value in state.values())):
        raise ValueError('Load all source model tensors on CPU before migration')
    if any(name.startswith(ENEMY_PREFIX) for name in state):
        raise ValueError('Source contains an unconfigured/partial enemy RSSM')
    if any(value.is_floating_point() and not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError('Source checkpoint contains nonfinite model tensors')
    config = {**asdict(old_config), **ENEMY_DEFAULTS}
    with torch.random.fork_rng(devices=[]), torch.device('meta'):
        previous = NeuralWorldModel(old_config)
        target = NeuralWorldModel(WorldModelConfig(**config))
    if not hasattr(target.vjepa_backend, 'enemy_rssm') or target.vjepa_backend.enemy_rssm is None:
        raise RuntimeError('Dual RSSM backend is not installed')
    expected_old = previous.state_dict()
    if set(expected_old) != set(state) or any(expected_old[key].shape != value.shape
            or expected_old[key].dtype != value.dtype for key, value in state.items()):
        raise ValueError('Source keys/shapes/dtypes differ from the declared legacy model')
    old_parameters = list(dict(previous.named_parameters()))
    new_parameters = list(dict(target.named_parameters()))
    added = [name for name in new_parameters if name not in old_parameters]
    if (new_parameters[:len(old_parameters)] != old_parameters or not added
            or any(not name.startswith(ENEMY_PREFIX) for name in added)):
        raise ValueError('Dual backend must append only enemy RSSM parameters after legacy names')
    expected_added = set(target.state_dict())-set(state)
    if any(not name.startswith(ENEMY_PREFIX) for name in expected_added):
        raise ValueError('Dual model changed an existing module outside the enemy prefix')
    # Construct only the new small module on CPU. Direct RSSM block parameters
    # receive their real constructor initialization, never generic reset guesses.
    enemy_config = EnemyRSSMConfig(**{name.removeprefix('enemy_rssm_'): value
        for name, value in ENEMY_DEFAULTS.items() if name != 'enemy_rssm_enabled'})
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        target.vjepa_backend.enemy_rssm = EnemyRSSM(enemy_config, width=old_config.vjepa_width,
            observation_dim=old_config.rssm_embed_dim)
    missing, unexpected = target.load_state_dict(dict(state), strict=False, assign=True)
    if unexpected or set(missing) != expected_added:
        raise ValueError('Only the newly initialized enemy RSSM may be missing')
    migrated = target.state_dict()
    for name, value in state.items():
        if not torch.equal(migrated[name], value):
            raise AssertionError('Dual migration changed an existing tensor: '+name)
        migrated[name] = value
    for name in expected_added:
        migrated[name] = migrated[name].detach().clone()
    configure_rssm_training(previous)
    canonical_old = [name for name, value in previous.named_parameters() if value.requires_grad]
    optimizer = saved.get('optimizer')
    names = saved.get('optimizer_parameter_names')
    if (not isinstance(optimizer, dict) or not isinstance(optimizer.get('state'), dict)
            or not isinstance(names, list) or names != canonical_old):
        raise ValueError('Source Adam must retain the exact canonical legacy parameter names')
    groups = optimizer.get('param_groups')
    if not isinstance(groups, list) or not groups:
        raise ValueError('Source Adam requires existing named parameter groups')
    old_ids = [identifier for group in groups for identifier in group.get('params', [])]
    if (old_ids != list(range(len(names))) or any(type(identifier) is not int for identifier in old_ids)
            or set(optimizer['state'])-set(old_ids)):
        raise ValueError('Adam IDs do not map exactly to the preserved legacy names')
    for identifier, name in enumerate(names):
        for key, value in optimizer['state'].get(identifier, {}).items():
            if torch.is_tensor(value) and (value.device.type != 'cpu'
                    or (key == 'step' and value.numel() != 1)
                    or (key != 'step' and value.shape != state[name].shape)
                    or value.is_floating_point() and not bool(torch.isfinite(value).all())):
                raise ValueError('Invalid source Adam moment: '+name+'/'+key)
    for group in groups:
        if 'param_names' in group and group['param_names'] != [names[i] for i in group['params']]:
            raise ValueError('Saved Adam group names differ from the global name/ID map')
    new_ids = list(range(len(names), len(names)+len(added)))
    fresh_group = copy.deepcopy({key: value for key, value in groups[0].items() if key not in {'params', 'param_names'}})
    fresh_group['params'] = new_ids
    if 'param_names' in groups[0]:
        fresh_group['param_names'] = added
    result = {**saved, 'config': config, 'model': migrated,
        'optimizer': {**optimizer, 'state': dict(optimizer['state']),
            'param_groups': [copy.deepcopy(group) for group in groups]+[fresh_group]},
        'optimizer_parameter_names': names+added}
    policy = configure_rssm_training(target)
    parameters = dict(target.named_parameters())
    counts = parameter_counts(target)
    identity = renderer_condition_identity(target)
    report = {'format_version': 1, 'kind': 'append_independent_enemy_rssm', 'architecture': 'dreamer_rssm',
        'source_architecture': 'dreamer_rssm', 'source_enemy_rssm_enabled': False, 'enemy_rssm_enabled': True,
        'source_config': asdict(old_config), 'target_config': config, 'source_step': step, 'seed': seed,
        'source_config_sha256': _config_sha(asdict(old_config)), 'target_config_sha256': _config_sha(config),
        'allowed_config_changes': sorted(key for key in config if config[key] != asdict(old_config)[key]),
        'source_renderer_conditioning_identity': identity, 'frozen_render_assets_bitexact': True,
        'old_model_tensors_bitexact': True, 'old_tensor_count': len(state),
        'old_parameter_order_is_prefix': True, 'added_parameter_names': added,
        'added_parameter_count': sum(parameters[name].numel() for name in added),
        'model_parameter_counts': counts, 'player_memory_size': previous.vjepa_backend.memory_size,
        'enemy_memory_size': target.vjepa_backend.enemy_rssm.memory_size,
        'combined_memory_size': target.vjepa_backend.memory_size,
        'old_optimizer_groups_preserved': len(groups), 'old_adam_states_preserved': len(optimizer['state']),
        'new_optimizer_groups': 1, 'new_optimizer_state_entries': 0,
        'optimizer_parameter_groups': [[result['optimizer_parameter_names'][i] for i in group['params']]
            for group in result['optimizer']['param_groups']],
        'rng_payloads_preserved': True, 'global_curriculum_and_sampler_clocks_preserved': True,
        'inference_functions_bitexact_claimed': False, 'gameplay_qualified': False}
    result['dual_rssm_migration'] = report
    run_state = saved.get('run_state')
    if not isinstance(run_state, dict) or run_state.get('global_training_step') != step:
        raise ValueError('Source run state must identify the same completed optimizer step')
    result['run_state'] = copy.deepcopy(run_state)
    if run_id is not None:
        result['run_state']['run_id'] = str(run_id)
    result['run_state']['dual_rssm_migration'] = report
    # All previous clocks/history objects remain numerically identical. The
    # new architecture marker is separate from the established batch protocol.
    result['run_state']['world_dynamics_protocol'] = 'dual_player_enemy_rssm_v1'
    training = result['run_state'].setdefault('rssm_training', {})
    training.update(enemy_rssm_enabled=True, player_memory_size=report['player_memory_size'],
        enemy_memory_size=report['enemy_memory_size'], memory_size=report['combined_memory_size'],
        enemy_deterministic_dim=config['enemy_rssm_deter_dim'], enemy_stoch=config['enemy_rssm_stoch'],
        enemy_classes=config['enemy_rssm_classes'], enemy_parameter_count=report['added_parameter_count'],
        parameter_count=counts['dynamics_including_map'], dynamics_trainable=policy['dynamics_trainable'])
    result['rssm_training'] = copy.deepcopy(training)
    result['run_state']['model_parameters'] = {**result['run_state'].get('model_parameters', {}), **counts}
    result['run_state'].setdefault('objective_history', []).append({'kind': 'dual_rssm_architecture_migration',
        'step': step, 'world_dynamics_protocol': 'dual_player_enemy_rssm_v1',
        'loss_protocol_retained': run_state.get('loss_protocol'), 'added_enemy_parameters': report['added_parameter_count'],
        'old_history_retained': True, 'gameplay_qualified': False})
    return result, report


def prepare_dual_rssm_continuation(source, destination, run_id=None, seed=173, *, runid=None):
    """Write a separate checkpoint and audit sidecar; never touch processes."""
    source, destination = Path(source).resolve(strict=True), Path(destination).resolve()
    if source == destination or destination.exists():
        raise FileExistsError('Choose a new separate dual-RSSM checkpoint destination')
    if run_id is not None and runid is not None and run_id != runid:
        raise ValueError('Conflicting destination run IDs')
    source_sha = _file_sha(source)
    saved = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    payload, report = enable_dual_rssm_checkpoint(saved, seed, run_id=run_id or runid)
    if _file_sha(source) != source_sha:
        raise RuntimeError('Source checkpoint changed during CPU migration; use its immutable archive')
    report.update(source_checkpoint=str(source), source_checkpoint_sha256=source_sha,
        checkpoint=str(destination), source_run_dir=str(source.parent.parent),
        destination_run_dir=str(destination.parent.parent), source_plan_sha256=saved.get('plan_sha256'))
    bindings = []
    for name in ('combat_training_plan.json', 'launch_command.json', 'combat_recovery_revision.json',
                 'batch128_revision.json', 'preview_assets_batch128_g4398.json'):
        path = source.parent.parent/name
        if path.is_file():
            bindings.append({'relative_path': name, 'source': str(path), 'sha256': _file_sha(path)})
    report['copy_identically_to_new_run'] = bindings
    report['run_binding_note'] = 'Preserve original plan/revision paths and SHAs; copy the run plan/launcher, change only explicit dual-run destination arguments.'
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name+f'.{os.getpid()}.tmp')
    try:
        torch.save(payload, temporary)
        replace_file(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    report['checkpoint_sha256'] = _file_sha(destination)
    atomic_json(destination.with_suffix(destination.suffix+'.migration.json'), report)
    return report
