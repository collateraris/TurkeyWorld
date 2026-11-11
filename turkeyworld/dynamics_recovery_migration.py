"""Append an axis-pass head to a complete stopped world checkpoint on CPU."""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import torch
from torch import nn

from .models import NeuralWorldModel, STATE_SCHEMA, WorldModelConfig
from .training_pipeline import atomic_json, replace_file
from .vqgan_pipeline import process_alive


HEAD_PREFIX = 'vjepa_backend.player_pass_head.'


def _fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def enable_axis_fraction_checkpoint(saved, seed=173):
    """Return (payload, report); preserve old tensors and every saved moment.

    Only new head parameters are materialized. All old model tensors and Adam
    state tensors keep their original CPU storage, including mmap storage.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError('Migration seed must be a nonnegative integer')
    if not isinstance(saved, dict) or saved.get('stage') != 'world':
        raise ValueError('Dynamics recovery requires a complete world checkpoint')
    old_config = WorldModelConfig(**saved['config'])
    if (old_config.kind != 'teacher' or old_config.architecture != 'vjepa2_ac'
            or not old_config.spatial_entities or not old_config.endless
            or old_config.transition_mode != 'event_v1'):
        raise ValueError('Dynamics recovery requires a spatial endless V-JEPA teacher')
    if saved.get('state_schema', STATE_SCHEMA) != STATE_SCHEMA:
        raise ValueError('Dynamics recovery cannot change the state schema')
    if saved.get('training_counters', {}).get('pending_microbatches', 0):
        raise ValueError('Only a completed optimizer boundary can be migrated')
    source = saved.get('model')
    if (not isinstance(source, dict) or not source
            or any(not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                   for value in source.values())):
        raise ValueError('Load the source model tensors on CPU before migration')
    if old_config.player_motion_mode == 'delta' and any(name.startswith(HEAD_PREFIX) for name in source):
        raise ValueError('Partial player pass head without axis_fraction configuration')
    config = asdict(old_config)
    config['player_motion_mode'] = 'axis_fraction'
    with torch.random.fork_rng(devices=[]), torch.device('meta'):
        old_model = NeuralWorldModel(old_config)
        model = NeuralWorldModel(WorldModelConfig(**config))
    old_names = list(dict(old_model.named_parameters()))
    parameters = dict(model.named_parameters())
    if list(parameters)[:len(old_names)] != old_names:
        raise ValueError('New player head changed the legacy parameter ordering')
    expected = set(model.state_dict())-set(source)
    if old_config.player_motion_mode == 'axis_fraction' and expected:
        raise ValueError('Already migrated checkpoint is missing its head parameters')
    missing, unexpected = model.load_state_dict(dict(source), strict=False, assign=True)
    if unexpected or set(missing) != expected or any(not name.startswith(HEAD_PREFIX) for name in missing):
        raise ValueError('Only the new player pass head may be missing: '+str((missing, unexpected)))
    if missing:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            head = model.vjepa_backend.player_pass_head
            head.to_empty(device='cpu')
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    module.reset_parameters()
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
    state = model.state_dict()
    for name, value in source.items():
        if state[name].shape != value.shape or state[name].dtype != value.dtype or not torch.equal(state[name], value):
            raise ValueError('Migration changed an existing tensor: '+name)
        state[name] = value
    for name in missing:
        state[name] = state[name].detach().clone()
    parameters = dict(model.named_parameters())
    added = [name for name in parameters if name not in old_names]
    result = {**saved, 'config': config, 'model': state}
    optimizer = saved.get('optimizer')
    states_preserved = 0
    if optimizer is not None:
        groups = optimizer.get('param_groups', [])
        if len(groups) != 1 or not isinstance(optimizer.get('state'), dict):
            raise ValueError('Migration requires one named optimizer group')
        group = groups[0]
        ids = group.get('params', [])
        names = saved.get('optimizer_parameter_names')
        names = names[0] if isinstance(names, (list, tuple)) and len(names) == 1 else group.get('param_names')
        if (not isinstance(names, (list, tuple)) or len(names) != len(ids)
                or len(set(names)) != len(names) or len(set(ids)) != len(ids)
                or any(not isinstance(identifier, int) or isinstance(identifier, bool) for identifier in ids)
                or any(name not in old_names for name in names)
                or set(optimizer['state'])-set(ids)
                or group.get('param_names') is not None and list(group['param_names']) != list(names)):
            raise ValueError('Saved optimizer names/IDs do not identify the old parameters')
        for name, identifier in zip(names, ids):
            for key, value in optimizer['state'].get(identifier, {}).items():
                if isinstance(value, torch.Tensor) and ((key == 'step' and value.numel() != 1)
                        or key != 'step' and value.shape != source[name].shape):
                    raise ValueError('Saved optimizer moment shape differs: '+name+'/'+key)
        new_ids = list(range(max(ids, default=-1)+1, max(ids, default=-1)+1+len(added)))
        new_names = list(names)+added
        new_group = copy.deepcopy({key: value for key, value in group.items() if key not in {'params', 'param_names'}})
        new_group.update(params=list(ids)+new_ids, param_names=new_names)
        result['optimizer'] = {**optimizer, 'state': dict(optimizer['state']), 'param_groups': [new_group]}
        result['optimizer_parameter_names'] = [new_names]
        states_preserved = len(optimizer['state'])
    count = sum(parameter.numel() for parameter in parameters.values())
    report = {'status': 'already_enabled' if old_config.player_motion_mode == 'axis_fraction' else 'expanded',
        'source_player_motion_mode': old_config.player_motion_mode, 'player_motion_mode': 'axis_fraction',
        'source_step': int(saved.get('global_training_step', saved.get('training_steps', 0))),
        'seed': seed, 'added_parameters': added,
        'added_parameter_count': sum(parameters[name].numel() for name in added),
        'teacher_parameters': count, 'old_tensors_preserved': len(source),
        'old_parameter_order_is_prefix': True, 'optimizer_state_entries_preserved': states_preserved,
        'new_optimizer_state_entries': 0, 'untrained_initial_pass_fraction': .5,
        'teacher_config_fingerprint': _fingerprint(config)}
    result['dynamics_recovery_migration'] = saved.get('dynamics_recovery_migration', report)
    if isinstance(saved.get('initialization'), dict):
        result['initialization'] = {**saved['initialization'], 'player_motion_mode': 'axis_fraction',
            'teacher_parameters': count, 'teacher_config_fingerprint': report['teacher_config_fingerprint']}
    return result, report


def migrate_dynamics_checkpoint(source, destination, *, seed=173, overwrite=False):
    """Write a separate checkpoint/report; never overwrite the source archive."""
    source, destination = Path(source).resolve(strict=True), Path(destination).resolve()
    if source == destination or destination.exists() and not overwrite:
        raise ValueError('Choose a separate new destination or explicitly allow destination replacement')
    for root in {source.parent.parent, destination.parent.parent}:
        for name in ('training.pid', 'pipeline.pid', 'controller.pid'):
            path = root/name
            if path.exists() and process_alive(int(path.read_text(encoding='ascii').strip())):
                raise RuntimeError('Stop the world/supervisor before dynamics recovery migration')
    saved = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    payload, report = enable_axis_fraction_checkpoint(saved, seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name+f'.{os.getpid()}.tmp')
    try:
        torch.save(payload, temporary)
        replace_file(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    report = {**report, 'source_checkpoint': str(source), 'checkpoint': str(destination)}
    atomic_json(destination.with_suffix(destination.suffix+'.migration.json'), report)
    return report
