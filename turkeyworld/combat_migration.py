"""Add combat/spawn conditioning to a stopped spatial175M world in place.

Old tensors, named Adam moments, EMA/critic payloads, budgets, curriculum start
and saved RNG survive unchanged. Only missing, explicitly whitelisted branches
are initialized. Checkpoint backups are exclusive and completion commits last.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import torch

from .models import NeuralWorldModel, WorldModelConfig
from .training_pipeline import atomic_json, now, replace_file
from .vqgan_pipeline import process_alive


COMBAT_PREFIXES = (
    'vjepa_backend.spatial_combat_head.',
    'vjepa_backend.renderer.spatial_combat_counts.',
    'vjepa_backend.spatial_spawn_query.',
    'vjepa_backend.spatial_spawn_entity_key.',
    'vjepa_backend.spatial_spawn_entity_value.',
    'vjepa_backend.spatial_spawn_output.',
    'vjepa_backend.spatial_spawn_map_key.',
    'flow_renderer.spatial_condition.',
    'flow_renderer.residual_adapter.spatial_condition.',
    'flow_renderer.wall_condition.',
    'flow_renderer.residual_adapter.wall_condition.',
)


def config_fingerprint(config):
    """Default-false flags do not change legacy configuration identities."""
    value = dict(config)
    for flag in ('spatial_entities', 'spatial_combat', 'game_training_v5', 'flow_block_conditioning'):
        if not value.get(flag, False):
            value.pop(flag, None)
    if value.get('flow_patch_size', 4) == 4:
        value.pop('flow_patch_size', None)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def enable_combat_checkpoint(saved, seed=173):
    """Return a CPU payload/report with all old tensors and payload objects kept.

    The old optimizer name list intentionally remains unchanged. The existing
    name-aware restore attaches its saved moments to old parameters and leaves
    the new parameters' optimizer states empty until their first update.
    """
    if not isinstance(saved, dict) or saved.get('stage') != 'world':
        raise ValueError('Combat migration requires a world-model checkpoint')
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError('Combat migration seed must be a nonnegative integer')
    source_config = WorldModelConfig(**saved['config'])
    if (source_config.architecture != 'vjepa2_ac' or source_config.transition_mode != 'event_v1'
            or source_config.kind != 'teacher' or source_config.image_size != 32
            or not source_config.spatial_entities):
        raise ValueError('Combat migration requires a native32 spatial V-JEPA event_v1 teacher')
    source = saved.get('model')
    if not isinstance(source, dict) or not source:
        raise ValueError('Combat migration requires the complete old model state')
    if any(not torch.is_tensor(value) or value.device.type != 'cpu' for value in source.values()):
        raise ValueError('Load the complete source checkpoint on CPU before combat migration')
    optimizer = saved.get('optimizer')
    if optimizer is not None:
        groups = optimizer.get('param_groups', [])
        if (len(groups) != 1 or (saved.get('optimizer_parameter_names') is None
                               and groups[0].get('param_names') is None)):
            raise ValueError('Combat optimizer preservation requires one explicitly named saved parameter group')
    config = asdict(source_config)
    config['spatial_combat'] = True
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        torch.random.default_generator.manual_seed(seed)
        model = NeuralWorldModel(WorldModelConfig(**config)).eval()
    missing, unexpected = model.load_state_dict(source, strict=False)
    if unexpected or any(not name.startswith(COMBAT_PREFIXES) for name in missing):
        raise ValueError(f'Old checkpoint has missing or incompatible non-combat parameters: {missing}, {unexpected}')
    if source_config.spatial_combat and missing:
        raise ValueError('Already-combat checkpoint is missing combat parameters')
    if not source_config.spatial_combat and any(name.startswith(COMBAT_PREFIXES) for name in source):
        raise ValueError('Partial combat parameters without their configuration flag are ambiguous')
    state, parameters = model.state_dict(), dict(model.named_parameters())
    for name, tensor in source.items():
        if state[name].shape != tensor.shape or not torch.equal(state[name], tensor):
            raise ValueError('Combat migration changed an existing tensor: '+name)
        state[name] = tensor  # Preserve original storage and dtype.
    for name in missing:
        state[name] = state[name].detach().clone()
    count = sum(parameter.numel() for parameter in parameters.values())
    profile = {'teacher_parameters': count, 'teacher_config_fingerprint': config_fingerprint(config),
               'spatial_entities': True, 'spatial_entity_count': 9, 'spatial_combat': True}
    report = {'status': 'already_enabled' if source_config.spatial_combat else 'expanded',
        'seed': seed, 'spatial_entities': True, 'spatial_entity_count': 9, 'spatial_combat': True,
        'source_spatial_combat': bool(source_config.spatial_combat),
        'source_step': int(saved.get('training_steps', 0)),
        'start_optimizer_step': int(saved.get('global_training_step', saved.get('training_steps', 0))),
        'source_parameters': sum(parameter.numel() for name, parameter in parameters.items() if name in source),
        'generator_parameters': count, 'added_parameters': [name for name in parameters if name in missing],
        'added_parameter_count': sum(parameter.numel() for name, parameter in parameters.items() if name in missing),
        'old_tensors_preserved': len(source), 'optimizer_preserved': optimizer is not None,
        'source_config_fingerprint': config_fingerprint(asdict(source_config)),
        'teacher_config_fingerprint': profile['teacher_config_fingerprint'],
        'allowed_source_sampling_upgrade': True,
        'preserved_payload_fields': sorted(set(saved)-{'model', 'config', 'initialization'})}
    result = {**saved, 'config': config, 'model': state}
    if isinstance(saved.get('initialization'), dict):
        result['initialization'] = {**saved['initialization'], **profile}
    if 'combat_conditioning_migration' not in saved:
        result['combat_conditioning_migration'] = {
            'format_version': 1, 'source_spatial_combat': bool(source_config.spatial_combat),
            'allowed_source_sampling_upgrade': True, 'source_step': report['source_step'],
            'start_optimizer_step': report['start_optimizer_step'], 'spatial_entities': True,
            'spatial_combat': True, 'seed': seed, 'generator_parameters': count,
            'teacher_config_fingerprint': profile['teacher_config_fingerprint']}
    return result, report


def _read_json(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def _backup_once(path):
    backup = path.with_name(path.stem+'_before_combat'+path.suffix)
    if backup.exists():
        return backup
    try:
        os.link(path, backup)
    except FileExistsError:
        pass
    except OSError:
        with path.open('rb') as source, backup.open('xb') as target:
            shutil.copyfileobj(source, target)
        shutil.copystat(path, backup)
    return backup


def _stage_checkpoint(payload, destination):
    descriptor, temporary = tempfile.mkstemp(prefix=destination.stem+'.combat.', suffix='.tmp', dir=destination.parent)
    os.close(descriptor)
    path = Path(temporary)
    try:
        torch.save(payload, path)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _configuration(payload):
    config = asdict(WorldModelConfig(**payload['config']))
    if (not config['spatial_entities'] or config['image_size'] != 32 or config['kind'] != 'teacher'
            or config['architecture'] != 'vjepa2_ac' or config['transition_mode'] != 'event_v1'):
        raise ValueError('Both world/initial sources must be native32 nine-participant V-JEPA teachers')
    return config


def _marker_fingerprints(config, raw_config):
    # Prior spatial migration versions serialized default-false combat either
    # explicitly or by omission. Accept both actual legacy identities.
    legacy = dict(config)
    if not legacy.get('spatial_entities', False):
        legacy.pop('spatial_entities', None)
    return {config_fingerprint(config), config_fingerprint({**config, 'spatial_combat': False}),
        hashlib.sha256(json.dumps(legacy, sort_keys=True).encode()).hexdigest(),
        hashlib.sha256(json.dumps({**legacy, 'spatial_combat': False}, sort_keys=True).encode()).hexdigest(),
        hashlib.sha256(json.dumps(raw_config, sort_keys=True).encode()).hexdigest()}


def migrate_combat_run(run_dir, sr_checkpoint, seed=173, target_seconds=30.):
    """Migrate stopped175M main/initial checkpoints, publishing completion last."""
    root, sr_path = Path(run_dir).resolve(), Path(sr_checkpoint).resolve()
    if (not isinstance(target_seconds, (int, float)) or isinstance(target_seconds, bool)
            or not math.isfinite(target_seconds) or target_seconds <= 0
            or not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        raise ValueError('Combat migration duration must be finite/positive and seed nonnegative')
    for name in ('training.pid', 'controller.pid'):
        path = root/name
        if path.exists() and process_alive(int(path.read_text(encoding='ascii').strip())):
            raise RuntimeError('Stop the world/controller before combat checkpoint migration')
    main = root/'checkpoints'/'teacher175vq_world.pt'
    initial = root/'checkpoints'/'teacher_vjepa175vq_initial.pt'
    initialization = root/'initialization.json'
    journal = root/'combat_conditioning_migration.json'
    previous = _read_json(journal, {})
    if previous and (previous.get('run_dir') != str(root) or previous.get('target_seconds') != target_seconds
                     or previous.get('seed') != seed or previous.get('sr_checkpoint') != str(sr_path)):
        raise ValueError('Existing combat migration belongs to a different run/duration/seed/SR checkpoint')
    if not main.is_file() or not initial.is_file() or not initialization.is_file():
        raise ValueError('Stopped175M world, its initial checkpoint and initialization marker are required')
    if previous.get('status') == 'complete':
        for path, phase in ((main, 'teacher175vq_world'), (initial, 'teacher175vq_initial')):
            payload = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
            config = _configuration(payload)
            if (payload.get('pipeline_phase') != phase or not config['spatial_combat']
                    or config_fingerprint(config) != previous.get('teacher_config_fingerprint')):
                raise ValueError('Completed combat migration no longer matches its checkpoint configuration')
            del payload
        marker = _read_json(initialization)
        if (not marker.get('spatial_combat') or marker.get('teacher_parameters') != previous.get('generator_parameters')
                or marker.get('teacher_config_fingerprint') != previous.get('teacher_config_fingerprint')):
            raise ValueError('Completed combat migration no longer matches its initialization marker')
        return previous
    spatial_marker = _read_json(root/'spatial_entities_migration.json', {})
    if (spatial_marker.get('status') != 'complete' or not spatial_marker.get('spatial_entities')
            or spatial_marker.get('spatial_entity_count') != 9):
        raise ValueError('The prior nine-participant spatial migration must already be complete')
    source = torch.load(main, map_location='cpu', weights_only=False)
    origin = torch.load(initial, map_location='cpu', weights_only=False)
    if (source.get('stage') != 'world' or source.get('pipeline_phase') != 'teacher175vq_world'
            or int(source.get('training_steps', 0)) < 1 or origin.get('stage') != 'world'
            or origin.get('pipeline_phase') != 'teacher175vq_initial'
            or int(origin.get('training_steps', -1)) != 0 or int(origin.get('global_training_step', 0)) != 0):
        raise ValueError('Wrong world/initial source phase or step; refusing another training stage')
    main_config, initial_config = (_configuration(payload) for payload in (source, origin))
    comparison = [{key: value for key, value in config.items() if key != 'spatial_combat'}
                  for config in (main_config, initial_config)]
    if comparison[0] != comparison[1]:
        raise ValueError('World and initial checkpoint configurations do not belong to the same run')
    if not previous and (main_config['spatial_combat'] or initial_config['spatial_combat']):
        raise ValueError('An unrecorded already-combat checkpoint cannot start a new migration transaction')
    marker = _read_json(initialization)
    acceptable = _marker_fingerprints(initial_config, origin['config'])
    if previous.get('teacher_config_fingerprint'):
        acceptable.add(previous['teacher_config_fingerprint'])
    if (not isinstance(marker, dict) or marker.get('teacher_size') != '175m_vqgan'
            or not marker.get('spatial_entities') or marker.get('spatial_entity_count') != 9
            or marker.get('teacher_config_fingerprint') not in acceptable):
        raise ValueError('Initialization marker/configuration does not identify this spatial teacher175 source')
    if previous.get('status') == 'pending' and (
            int(source.get('training_steps', 0)) != previous.get('source_step')
            or int(source.get('global_training_step', source.get('training_steps', 0))) != previous.get('start_optimizer_step')):
        raise ValueError('World checkpoint advanced while combat migration was pending')
    sr = torch.load(sr_path, map_location='cpu', weights_only=False)
    if (sr.get('stage') != 'sr' or int(sr.get('training_steps', 0)) < 1 or sr.get('input_size') != 32
            or sr.get('output_size') != 128 or sr.get('config', {}).get('scale') != 4):
        raise ValueError('Updated SR must be trained x4 at the world native32 resolution')
    sr_quality = {'training_steps': int(sr['training_steps']), 'target_met': sr.get('target_met') is True}
    del sr
    upgraded, main_report = enable_combat_checkpoint(source, seed)
    upgraded_initial, initial_report = enable_combat_checkpoint(origin, seed)
    if main_report['teacher_config_fingerprint'] != initial_report['teacher_config_fingerprint']:
        raise ValueError('World/initial expanded combat configurations must match')
    profile = {'teacher_parameters': main_report['generator_parameters'],
        'teacher_config_fingerprint': main_report['teacher_config_fingerprint'],
        'spatial_entities': True, 'spatial_entity_count': 9, 'spatial_combat': True}
    upgraded_marker = {**marker, **profile}
    staged = []
    try:
        staged.append(_stage_checkpoint(upgraded, main))
        staged.append(_stage_checkpoint(upgraded_initial, initial))
        backups = {str(path): str(_backup_once(path)) for path in (main, initial, initialization)}
        config = _read_json(root/'experiment_config175vq.json', {})
        source_report = previous.get('world_report', main_report)
        report = {**previous, 'format_version': 1, 'status': 'pending', 'run_dir': str(root),
            **{key: value for key, value in profile.items() if key != 'teacher_parameters'},
            'source_spatial_combat': bool(source_report['source_spatial_combat']),
            'target_seconds': float(target_seconds), 'sr_checkpoint': str(sr_path), 'sr_quality': sr_quality,
            'source_step': source_report['source_step'], 'start_optimizer_step': source_report['start_optimizer_step'],
            'seed': seed, 'generator_parameters': main_report['generator_parameters'],
            'training_subset_manifest': config.get('training_subset_manifest'), 'backups': backups,
            'files': {'world': str(main), 'initial': str(initial), 'initialization': str(initialization)},
            'world_report': source_report, 'initial_report': previous.get('initial_report', initial_report),
            'teacher_budget_preserved': True, 'motion_curriculum_start_preserved': True,
            'allowed_source_sampling_upgrade': True, 'created_at': previous.get('created_at', now())}
        atomic_json(journal, report)
        replace_file(staged[1], initial)
        replace_file(staged[0], main)
        atomic_json(initialization, upgraded_marker)
        report.update(status='complete', completed_at=now())
        atomic_json(journal, report)
        return report
    finally:
        for path in staged:
            path.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--sr-checkpoint', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=173)
    parser.add_argument('--target-seconds', type=float, default=30.)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    print(json.dumps(migrate_combat_run(args.run_dir, args.sr_checkpoint, args.seed, args.target_seconds),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
