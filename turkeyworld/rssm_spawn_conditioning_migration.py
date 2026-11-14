"""Append causal spawn conditioning without restarting any dynamics ledger."""
from copy import deepcopy
from dataclasses import asdict
import os
from pathlib import Path
import re

import torch

from .models import NeuralWorldModel, WorldModelConfig, STATE_SCHEMA, parameter_counts
from .renderer_native_phase import _same
from .rssm_five_migration import file_sha, report_sha
from .rssm_migration import configure_rssm_training
from .training_pipeline import atomic_json, replace_file


PROTOCOL = 'rssm_current_state_spawn_conditioning_v1'
PREFIX = 'vjepa_backend.spawn_conditioning.'
KIND = 'append_current_state_spawn_conditioning'
PLAN_KIND = 'rssm_spawn_conditioning_revision_v1'


def enable_spawn_conditioning_checkpoint(saved, *, seed=173, source_checkpoint_sha256=None):
    """Keep every old tensor/name/Adam ID and append one fresh parameter group."""
    from .renderer_student import renderer_condition_identity, _hash_tensors
    from .rssm_moe_protocol import moe_training_horizon
    from .rssm_spawn_conditioning import RSSMSpawnConditioning, spawn_conditioning_metadata
    if (type(seed) is not int or seed < 0 or not isinstance(saved, dict)
            or saved.get('stage') != 'world' or saved.get('state_schema') != STATE_SCHEMA
            or not re.fullmatch('[0-9a-f]{64}', source_checkpoint_sha256 or '')):
        raise ValueError('Spawn conditioning requires an immutable complete world checkpoint and seed')
    old_config = WorldModelConfig(**saved['config'])
    step, run = saved.get('training_steps'), saved.get('run_state')
    if (not old_config.rssm_five_expanded_enabled or old_config.rssm_spawn_conditioning_enabled
            or not old_config.combat_rssm_enabled or not old_config.game_training_v5
            or type(step) is not int or step < 1 or saved.get('global_training_step') != step
            or not isinstance(run, dict) or run.get('global_training_step') != step
            or saved.get('training_counters', {}).get('pending_microbatches', 0)):
        raise ValueError('Spawn append requires the existing expanded five-expert optimizer boundary')
    moe_training_horizon(run)
    state = saved.get('model')
    if (not isinstance(state, dict) or not state or any(n.startswith(PREFIX) for n in state)
            or any(not torch.is_tensor(v) or v.device.type != 'cpu'
                   or not bool(torch.isfinite(v).all()) for v in state.values())):
        raise ValueError('Spawn append needs finite complete CPU parent tensors and no partial adapter')
    config = {**asdict(old_config), 'rssm_spawn_conditioning_enabled': True}
    with torch.random.fork_rng(devices=[]), torch.device('meta'):
        old = NeuralWorldModel(old_config)
        target = NeuralWorldModel(WorldModelConfig(**config))
    expected = old.state_dict()
    if (set(expected) != set(state) or any(expected[n].shape != v.shape or expected[n].dtype != v.dtype
                                          for n, v in state.items())):
        raise ValueError('Spawn parent namespace does not match its explicit configuration')
    old.load_state_dict(dict(state), strict=True, assign=True)
    adapter = RSSMSpawnConditioning(width=old_config.vjepa_width, hidden=64, seed=seed)
    target.vjepa_backend.spawn_conditioning = adapter
    missing, unexpected = target.load_state_dict(dict(state), strict=False, assign=True)
    added = [PREFIX+name for name, _ in adapter.named_parameters()]
    if unexpected or set(missing) != set(added) or any(not name.startswith(PREFIX) for name in missing):
        raise ValueError('Spawn migration may append only the declared causal adapter tensors')
    new_state = target.state_dict()
    if any(not _same(value, new_state[name]) for name, value in state.items()):
        raise AssertionError('Spawn conditioning changed an existing model tensor')
    configure_rssm_training(old); policy = configure_rssm_training(target)
    canonical_old = [n for n,p in old.named_parameters() if p.requires_grad]
    canonical_new = [n for n,p in target.named_parameters() if p.requires_grad]
    names = saved.get('optimizer_parameter_names')
    expanded = saved.get('rssm_five_expanded_migration', {})
    previous_added = set(expanded.get('added_parameter_names', []))
    if (expanded.get('revision_sha256') != report_sha(expanded)
            or names != expanded.get('optimizer_parameter_names') or set(names) != set(canonical_old)
            or [n for n in canonical_old if n not in previous_added] != [n for n in names if n not in previous_added]
            or [n for n in canonical_new if n not in set(added)] != canonical_old):
        raise ValueError('Spawn append must preserve the complete ordered expanded-parent optimizer names')
    optimizer = saved.get('optimizer', {})
    groups, states = optimizer.get('param_groups', []), optimizer.get('state', {})
    ids = [i for group in groups for i in group.get('params', [])]
    if len(groups) != 6 or ids != list(range(len(names))) or set(states) != set(ids):
        raise ValueError('Spawn append requires every old Adam state in all six ordered groups')
    for index, name in enumerate(names):
        moment = states[index]
        age = moment.get('step')
        if (not torch.is_tensor(age) or age.numel() != 1 or not bool(torch.isfinite(age))
                or float(age) < 1 or float(age) != int(float(age))
                or any(not torch.is_tensor(moment.get(k)) or moment[k].shape != state[name].shape
                       or moment[k].dtype != state[name].dtype or not bool(torch.isfinite(moment[k]).all())
                       for k in ('exp_avg', 'exp_avg_sq'))
                or bool((moment['exp_avg_sq'] < 0).any())):
            raise ValueError('Spawn append cannot reset an old Adam age or omit/change a finite moment')
    architecture = spawn_conditioning_metadata(adapter)
    identity = renderer_condition_identity(old)
    if (renderer_condition_identity(target) != identity
            or old.vjepa_backend.memory_size != target.vjepa_backend.memory_size):
        raise AssertionError('Spawn conditioning changed renderer assets or recurrent memory')
    fresh = deepcopy({k:v for k,v in groups[0].items() if k not in ('params','param_names')})
    fresh['params'] = list(range(len(names), len(names)+len(added)))
    all_names = list(names)+added
    report = dict(format_version=1, kind=KIND, protocol=PROTOCOL, source_step=step,
        source_checkpoint_sha256=source_checkpoint_sha256, source_config=asdict(old_config), target_config=config,
        allowed_config_changes=['rssm_spawn_conditioning_enabled'],
        added_default_config_fields=sorted(set(asdict(old_config))-set(saved['config'])),
        old_model_tensors_bitexact=True, old_model_tensor_count=len(state),
        old_model_tensor_name_map={name:name for name in state}, source_model_sha256=_hash_tensors(state),
        added_parameter_names=added, added_parameter_count=sum(p.numel() for p in adapter.parameters()),
        added_tensor_count=len(added), memory_size=old.vjepa_backend.memory_size,
        frozen_render_assets_bitexact=True, zero_residual_initialization=True,
        source_renderer_conditioning_identity=identity, architecture=architecture,
        source_optimizer_parameter_names=list(names), optimizer_parameter_names=all_names,
        old_Adam_states=len(states), old_optimizer_groups=6, new_optimizer_groups=1,
        new_optimizer_state_entries=0, all_old_Adam_states_options_bitexact=True,
        all_old_RNG_sampler_gate_history_bitexact=True, objective_formula_changed=False,
        spawn_supervision='existing_normalized_Source_respawn_anchor_CE_and_fallback_CE',
        H450_total_loss_definition_unchanged=True, model_parameter_counts=parameter_counts(target),
        Source_future_inputs=False, automatic_active_promotion=False, gameplay_qualified=False)
    report['revision_sha256'] = report_sha(report)
    newrun = deepcopy(run)
    newrun.update(rssm_spawn_conditioning_migration=report, rssm_spawn_conditioning_training=dict(
        protocol=PROTOCOL, enabled=True, migration_sha256=report['revision_sha256'],
        source_step=step, current_step=step, architecture=architecture,
        added_parameters=report['added_parameter_count'], added_parameter_tensors=len(added)))
    newrun.setdefault('training_protocol', {})['spawn_conditioning'] = dict(
        protocol=PROTOCOL, migration_sha256=report['revision_sha256'], source_step=step)
    payload = {**saved, 'config':config,
        'model':{n:v.detach().clone() if n in added else v for n,v in new_state.items()},
        'optimizer_parameter_names':all_names,
        'optimizer':{**optimizer, 'state':deepcopy(states), 'param_groups':deepcopy(groups)+[fresh]},
        'run_state':newrun, 'rssm_spawn_conditioning_migration':report}
    return payload, report


def prepare_spawn_conditioning_continuation(source, destination, *, seed=173):
    source, destination = Path(source).resolve(strict=True), Path(destination).resolve()
    if source == destination or destination.exists():
        raise FileExistsError('Spawn migration requires a separate immutable destination')
    source_sha = file_sha(source)
    saved = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    payload, report = enable_spawn_conditioning_checkpoint(saved, seed=seed, source_checkpoint_sha256=source_sha)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name+f'.{os.getpid()}.tmp')
    try:
        with temporary.open('wb') as stream:
            torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
        if file_sha(source) != source_sha: raise ValueError('Source checkpoint changed during spawn migration')
        replace_file(temporary, destination)
    finally: temporary.unlink(missing_ok=True)
    receipt = dict(report, source_checkpoint=str(source), checkpoint=str(destination), checkpoint_sha256=file_sha(destination))
    atomic_json(destination.with_suffix(destination.suffix+'.spawn-conditioning.json'), receipt)
    return receipt


def bind_spawn_conditioning_plan(state, plan, *, path, sha256, report):
    if (report.get('revision_sha256') != report_sha(report)
            or plan.get('kind') != PLAN_KIND or plan.get('version') != 1
            or plan.get('migration_sha256') != report['revision_sha256']
            or plan.get('source_checkpoint_sha256') != report['source_checkpoint_sha256']
            or plan.get('source_step') != report['source_step'] or plan.get('architecture') != report['architecture']
            or not re.fullmatch('[0-9a-f]{64}', sha256 or '')):
        raise ValueError('Spawn conditioning plan differs from the immutable append migration')
    record = dict(protocol=PROTOCOL, plan_path=str(path), plan_sha256=sha256,
        migration_sha256=report['revision_sha256'], source_step=report['source_step'])
    existing = state.get('training_protocol', {}).get('spawn_conditioning', {})
    if existing and any(existing.get(k) != v for k,v in record.items() if k in existing):
        raise ValueError('Saved spawn conditioning revision pin cannot change on resume')
    state.setdefault('training_protocol', {})['spawn_conditioning'] = record
    state['rssm_spawn_conditioning_training'].update(plan_path=str(path), plan_sha256=sha256)
    return record
