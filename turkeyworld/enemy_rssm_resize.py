"""Expand only the enemy RSSM of a completed CPU world checkpoint.

The player/shared model and its Adam state remain bit exact. Enemy tensors
retain all old channels; recurrent group/gate and concatenation offsets are
mapped explicitly. Changed enemy tensors receive fresh Adam state. This is
an architecture warm start, never a function-preserving enlargement.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, is_dataclass
import os
from pathlib import Path

import torch

from .dual_rssm_migration import ENEMY_PREFIX, _config_sha, _file_sha
from .enemy_rssm import EnemyRSSM, EnemyRSSMConfig
from .models import NeuralWorldModel, WorldModelConfig, STATE_SCHEMA, parameter_counts
from .rssm_migration import configure_rssm_training
from .training_pipeline import atomic_json, replace_file


ENEMY_20M_OVERRIDES = {'enemy_rssm_deter_dim': 1344, 'enemy_rssm_hidden_dim': 1408}
ALLOWED_CONFIG_CHANGES = tuple(ENEMY_20M_OVERRIDES)


def _enemy_config(config):
    return EnemyRSSMConfig(**{key.removeprefix('enemy_rssm_'): value
        for key, value in asdict(config).items()
        if key.startswith('enemy_rssm_') and key != 'enemy_rssm_enabled'})


def _h_indices(old, new):
    """Old deterministic units keep their within-group identities."""
    return torch.cat([torch.arange(old.deter_dim//old.blocks)+group*(new.deter_dim//new.blocks)
        for group in range(old.blocks)])


def _copy_overlap(name, source, target, old, new):
    """Copy all old values with semantic h/gate/concatenation coordinates."""
    if source.ndim != target.ndim or any(a > b for a, b in zip(source.shape, target.shape)):
        raise ValueError('Enemy resize must retain every old tensor coordinate: '+name)
    if source.shape == target.shape:
        return source, {'method': 'bitexact', 'copied_elements': source.numel()}
    axes = [torch.arange(size) for size in source.shape]
    h = _h_indices(old, new)
    old_group, new_group = old.deter_dim//old.blocks, new.deter_dim//new.blocks
    if name in ('core.deter_input.0.weight', 'core.prior_head.0.weight'):
        axes[1] = h
    elif name == 'core.recurrent_norm.weight':
        axes[0] = h
    elif name == 'core.recurrent_hidden.weight':
        axes[2] = torch.cat([torch.arange(old_group), *[
            torch.arange(old.hidden_dim)+new_group+part*new.hidden_dim for part in range(3)]])
    elif name in ('core.recurrent_gates.weight', 'core.recurrent_gates.bias'):
        axes[1] = torch.cat([torch.arange(old_group)+part*new_group for part in range(3)])
    elif name in ('core.posterior_head.0.weight', 'readout.0.weight'):
        tail = source.shape[1]-old.deter_dim
        axes[1] = torch.cat([h, torch.arange(tail)+new.deter_dim])
    # All other expanded axes are hidden units or per-group local units.
    # Their prefix coordinates have unchanged meanings.
    with torch.no_grad():
        target[tuple(torch.meshgrid(*axes, indexing='ij'))] = source
    return target, {'method': 'semantic_channel_overlap', 'copied_elements': source.numel(),
        'new_elements': target.numel()-source.numel(),
        'source_shape': list(source.shape), 'target_shape': list(target.shape),
        'destination_axis_indices': [axis.tolist() for axis in axes]}


def resize_enemy_rssm_checkpoint(saved, seed=173, *, run_id=None, enemy_config=None):
    """Return a new payload/report, preserving the supplied checkpoint.

    ``enemy_config`` may override deterministic/hidden widths only. Defaults
    yield exactly20,000,323 enemy parameters at shared width/observation512.
    All saved parameter names/IDs and group options are retained. Shape-
    changed enemy parameters deliberately have no Adam state until next step.
    """
    from .renderer_student import renderer_condition_identity

    if type(seed) is not int or seed < 0:
        raise ValueError('Enemy resize requires a nonnegative integer seed')
    if not isinstance(saved, dict) or saved.get('stage') != 'world' or saved.get('state_schema') != STATE_SCHEMA:
        raise ValueError('Enemy resize requires a complete one-enemy world checkpoint')
    old_config = WorldModelConfig(**saved['config'])
    if (old_config.kind != 'teacher' or old_config.architecture != 'dreamer_rssm'
            or not old_config.enemy_rssm_enabled or not old_config.spatial_combat or not old_config.spatial_entities):
        raise ValueError('Expected the existing dual spatial/combat RSSM teacher')
    step = saved.get('global_training_step', saved.get('training_steps'))
    counters = saved.get('training_counters', {})
    run_state = saved.get('run_state')
    if (type(step) is not int or step < 0 or saved.get('training_steps') != step
            or counters.get('pending_microbatches', 0)
            or not isinstance(run_state, dict) or run_state.get('global_training_step') != step):
        raise ValueError('Resize requires a completed optimizer boundary with matching clocks')
    state = saved.get('model')
    if not isinstance(state, dict) or not state or any(not torch.is_tensor(v)
            or v.device.type != 'cpu' or v.is_floating_point() and not bool(torch.isfinite(v).all())
            for v in state.values()):
        raise ValueError('Source model tensors must be finite CPU values')
    config = asdict(old_config)
    overrides = dict(ENEMY_20M_OVERRIDES)
    if enemy_config is not None:
        incoming = asdict(enemy_config) if is_dataclass(enemy_config) else dict(enemy_config)
        overrides = {(key if key.startswith('enemy_rssm_') else 'enemy_rssm_'+key): value
            for key, value in incoming.items()}
    for key, value in overrides.items():
        if key not in config or key not in ALLOWED_CONFIG_CHANGES and value != config[key]:
            raise ValueError('Only enemy deterministic/hidden widths may change: '+key)
        config[key] = value
    new_config = WorldModelConfig(**config)
    old, new = _enemy_config(old_config), _enemy_config(new_config)
    if (new.deter_dim < old.deter_dim or new.hidden_dim < old.hidden_dim
            or (new.deter_dim, new.hidden_dim) == (old.deter_dim, old.hidden_dim)):
        raise ValueError('Enemy resize must enlarge widths without removing old units')

    with torch.random.fork_rng(devices=[]), torch.device('meta'):
        previous = NeuralWorldModel(old_config)
        target = NeuralWorldModel(new_config)
    expected_old, expected_new = previous.state_dict(), target.state_dict()
    if (set(expected_old) != set(state) or set(expected_new) != set(state)
            or any(expected_old[k].shape != v.shape or expected_old[k].dtype != v.dtype for k, v in state.items())):
        raise ValueError('Source keys/shapes/dtypes differ from the declared dual model')
    for name, value in state.items():
        if not name.startswith(ENEMY_PREFIX) and (expected_new[name].shape != value.shape
                or expected_new[name].dtype != value.dtype):
            raise ValueError('Enemy resize changed a player/shared tensor: '+name)
    configure_rssm_training(previous)
    policy = configure_rssm_training(target)
    names = saved.get('optimizer_parameter_names')
    canonical_old = [name for name, p in previous.named_parameters() if p.requires_grad]
    canonical_new = [name for name, p in target.named_parameters() if p.requires_grad]
    optimizer = saved.get('optimizer', {})
    groups = optimizer.get('param_groups')
    if (names != canonical_old or canonical_new != canonical_old or not isinstance(groups, list)
            or not groups or not isinstance(optimizer.get('state'), dict)):
        raise ValueError('Source Adam must retain the canonical named dual parameters')
    identifiers = [identifier for group in groups for identifier in group.get('params', [])]
    if (identifiers != list(range(len(names))) or any(type(i) is not int for i in identifiers)
            or set(optimizer['state'])-set(identifiers)):
        raise ValueError('Adam parameter IDs differ from the exact name order')
    for group in groups:
        if 'param_names' in group and group['param_names'] != [names[i] for i in group['params']]:
            raise ValueError('Adam group parameter names differ')
    for identifier, values in optimizer['state'].items():
        for key, value in values.items():
            if torch.is_tensor(value) and (value.device.type != 'cpu'
                    or (key == 'step' and value.numel() != 1)
                    or (key != 'step' and value.shape != state[names[identifier]].shape)
                    or value.is_floating_point() and not bool(torch.isfinite(value).all())):
                raise ValueError('Invalid source Adam state: '+names[identifier]+'/'+key)

    # Only the resized enemy is constructed on CPU. The large established world
    # remains meta until its original tensors are assigned without copying.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        enemy = EnemyRSSM(new, width=old_config.vjepa_width, observation_dim=old_config.rssm_embed_dim)
    migrated, tensor_report = dict(state), {}
    fresh_names, exact_enemy_names = [], []
    for short_name, value in enemy.state_dict().items():
        name = ENEMY_PREFIX+short_name
        copied, detail = _copy_overlap(short_name, state[name], value, old, new)
        migrated[name] = copied
        tensor_report[name] = detail
        (exact_enemy_names if detail['method'] == 'bitexact' else fresh_names).append(name)
    fresh_ids = {i for i, name in enumerate(names) if name in fresh_names}
    new_optimizer = {**optimizer, 'param_groups': copy.deepcopy(groups),
        'state': {i: values for i, values in optimizer['state'].items() if i not in fresh_ids}}
    target.load_state_dict(migrated, strict=True, assign=True)
    counts = parameter_counts(target)
    identity = renderer_condition_identity(target)
    enemy_count = sum(p.numel() for p in target.vjepa_backend.enemy_rssm.parameters())
    old_enemy_count = sum(p.numel() for p in previous.vjepa_backend.enemy_rssm.parameters())
    preserved_names = [name for name in state if not name.startswith(ENEMY_PREFIX)]
    if any(not torch.equal(state[name], migrated[name]) for name in preserved_names):
        raise AssertionError('Enemy resize changed established player/shared weights')
    report = {'format_version': 1, 'kind': 'resize_independent_enemy_rssm', 'seed': seed,
        'source_step': step, 'source_config': asdict(old_config), 'target_config': config,
        'source_config_sha256': _config_sha(asdict(old_config)), 'target_config_sha256': _config_sha(config),
        'allowed_config_changes': [key for key in config if config[key] != asdict(old_config)[key]],
        'source_renderer_conditioning_identity': identity,
        'player_and_shared_model_tensors_bitexact': True, 'frozen_render_assets_bitexact': True,
        'player_and_shared_adam_states_bitexact': True,
        'preserved_player_and_shared_tensors': len(preserved_names),
        'source_enemy_parameter_count': old_enemy_count, 'enemy_parameter_count': enemy_count,
        'player_memory_size': target.vjepa_backend.core.memory_size,
        'source_enemy_memory_size': old.memory_size, 'enemy_memory_size': new.memory_size,
        'combined_memory_size': target.vjepa_backend.memory_size,
        'enemy_tensor_mappings': tensor_report, 'enemy_exact_tensor_names': exact_enemy_names,
        'enemy_resized_parameter_names': fresh_names, 'enemy_adam_reset_parameter_ids': sorted(fresh_ids),
        'enemy_adam_reset_existing_states': len(set(optimizer['state']) & fresh_ids),
        'enemy_adam_policy': 'Preserve exact-shape states; resized tensors start fresh Adam at their next update',
        'optimizer_groups_and_ids_bitexact': True, 'optimizer_state_entries_preserved': len(new_optimizer['state']),
        'rng_payloads_preserved': True, 'global_curriculum_and_sampler_clocks_preserved': True,
        'runtime_memory_policy': 'New episode initialization; packed enemy memory shape changes',
        'inference_functions_bitexact_claimed': False, 'gameplay_qualified': False,
        'model_parameter_counts': counts}
    result = {**saved, 'config': config, 'model': migrated, 'optimizer': new_optimizer,
        'enemy_rssm_resize': report, 'run_state': copy.deepcopy(run_state)}
    result['run_state']['enemy_rssm_resize'] = report
    if run_id is not None:
        result['run_state']['run_id'] = str(run_id)
    training = result['run_state'].setdefault('rssm_training', {})
    training.update(enemy_rssm_enabled=True, enemy_parameter_count=enemy_count,
        enemy_memory_size=new.memory_size, memory_size=target.vjepa_backend.memory_size,
        enemy_deterministic_dim=new.deter_dim, enemy_hidden_dim=new.hidden_dim,
        enemy_stoch=new.stoch, enemy_classes=new.classes,
        parameter_count=counts['dynamics_including_map'], dynamics_trainable=policy['dynamics_trainable'])
    result['rssm_training'] = copy.deepcopy(training)
    result['run_state']['model_parameters'] = {**result['run_state'].get('model_parameters', {}), **counts}
    result['run_state']['model'] = (f'Dual RSSM ({counts["dynamics_including_map"]:,} dynamics parameters; '
        f'enemy {enemy_count:,}) + frozen Base / RF / VQGAN')
    result['run_state'].setdefault('objective_history', []).append({
        'kind': 'enemy_rssm_width_expansion', 'step': step, 'enemy_parameter_count': enemy_count,
        'player_and_shared_weights_and_adam_preserved': True, 'enemy_changed_adam_fresh': True,
        'gameplay_qualified': False})
    return result, report


def prepare_enemy_rssm_resize(source, destination, *, seed=173, run_id=None, enemy_config=None):
    """Atomically write to a new path, leaving source/processes untouched."""
    source, destination = Path(source).resolve(strict=True), Path(destination).resolve()
    if source == destination or destination.exists():
        raise FileExistsError('Enemy resize requires a new separate checkpoint destination')
    source_sha = _file_sha(source)
    saved = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    payload, report = resize_enemy_rssm_checkpoint(saved, seed, run_id=run_id, enemy_config=enemy_config)
    if _file_sha(source) != source_sha:
        raise RuntimeError('Source checkpoint changed during resize; use its stopped immutable archive')
    report.update(source_checkpoint=str(source), source_checkpoint_sha256=source_sha,
        checkpoint=str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name+f'.{os.getpid()}.tmp')
    try:
        torch.save(payload, temporary)
        replace_file(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    report['checkpoint_sha256'] = _file_sha(destination)
    atomic_json(destination.with_suffix('.resize.json'), report)
    return report
