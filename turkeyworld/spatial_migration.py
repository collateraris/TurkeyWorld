"""Add spatial branches to a stopped world without resetting learned training.

Old model tensors, optimizer IDs/names, EMA, critic, counters and saved RNG are
retained. Only new zero-output spatial branches and their configuration differ.
The file transaction keeps exclusive backups and publishes completion last.
"""
from dataclasses import asdict
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch

from .models import NeuralWorldModel, WorldModelConfig
from .training_pipeline import atomic_json, now, replace_file
from .vqgan_pipeline import process_alive


SPATIAL_PREFIXES = ('vjepa_backend.spatial_encoder.', 'vjepa_backend.spatial_visual_attention.',
    'vjepa_backend.spatial_motion_context.', 'vjepa_backend.spatial_motion_head.',
    'vjepa_backend.renderer.spatial_encoder.', 'vjepa_backend.renderer.spatial_render.')


def config_fingerprint(config):
    value = dict(config)
    if not value.get('spatial_combat', False):
        value.pop('spatial_combat', None)
    if not value.get('spatial_entities', False):
        value.pop('spatial_entities', None)
    for flag in ('game_training_v5', 'flow_block_conditioning'):
        if not value.get(flag, False):
            value.pop(flag, None)
    if value.get('flow_patch_size', 4) == 4:
        value.pop('flow_patch_size', None)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def enable_spatial_checkpoint(saved, seed=173):
    """Return a CPU payload/report; every old weight and saved moment survives.

    Existing optimizer_parameter_names intentionally stay OLD: the regular
    name-aware optimizer restore maps their moments and starts new states empty.
    Model initialization consumes neither caller CPU RNG nor CUDA RNG.
    """
    if not isinstance(saved, dict) or saved.get('stage') != 'world':
        raise ValueError('Spatial migration requires a world-model checkpoint')
    source_config = WorldModelConfig(**saved['config'])
    if source_config.architecture != 'vjepa2_ac' or source_config.transition_mode != 'event_v1':
        raise ValueError('Spatial migration requires V-JEPA event_v1 dynamics')
    source = saved.get('model')
    if not isinstance(source, dict) or not source:
        raise ValueError('Spatial migration requires the complete old model state')
    if any(not torch.is_tensor(value) or value.device.type != 'cpu' for value in source.values()):
        raise ValueError('Load the complete source checkpoint on CPU before spatial migration')
    optimizer = saved.get('optimizer')
    if optimizer is not None:
        groups = optimizer.get('param_groups', [])
        if len(groups) != 1 or (saved.get('optimizer_parameter_names') is None and groups[0].get('param_names') is None):
            raise ValueError('Spatial optimizer preservation requires one explicitly named saved parameter group')
    config = asdict(source_config)
    config['spatial_entities'] = True
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        torch.random.default_generator.manual_seed(seed)
        model = NeuralWorldModel(WorldModelConfig(**config)).eval()
    missing, unexpected = model.load_state_dict(source, strict=False)
    if unexpected or any(not name.startswith(SPATIAL_PREFIXES) for name in missing):
        raise ValueError(f'Old checkpoint has missing or incompatible non-spatial parameters: {missing}, {unexpected}')
    if source_config.spatial_entities and missing:
        raise ValueError('Already-spatial checkpoint is missing spatial parameters')
    if not source_config.spatial_entities and any(name.startswith(SPATIAL_PREFIXES) for name in source):
        raise ValueError('Partial spatial parameters without their configuration flag are ambiguous')
    state = model.state_dict()
    parameters = dict(model.named_parameters())
    for name, tensor in source.items():
        if state[name].shape != tensor.shape or not torch.equal(state[name], tensor):
            raise ValueError('Spatial migration changed an existing tensor: '+name)
        state[name] = tensor  # Keep original storage/dtype, not a second large copy.
    for name in missing:
        state[name] = state[name].detach().clone()
    count = sum(parameter.numel() for parameter in parameters.values())
    profile = {'teacher_parameters': count, 'teacher_config_fingerprint': config_fingerprint(config),
               'spatial_entities': True, 'spatial_entity_count': 9}
    report = {'status': 'already_enabled' if source_config.spatial_entities else 'expanded',
        'seed': int(seed), 'spatial_entities': True, 'spatial_entity_count': 9,
        'source_step': int(saved.get('training_steps', 0)),
        'start_optimizer_step': int(saved.get('global_training_step', saved.get('training_steps', 0))),
        'source_parameters': sum(parameter.numel() for name, parameter in parameters.items() if name in source),
        'generator_parameters': count, 'added_parameters': [name for name in parameters if name in missing],
        'added_parameter_count': sum(parameter.numel() for name, parameter in parameters.items() if name in missing),
        'old_tensors_preserved': len(source), 'optimizer_preserved': optimizer is not None,
        'source_config_fingerprint': config_fingerprint(asdict(source_config)),
        'teacher_config_fingerprint': profile['teacher_config_fingerprint'],
        'preserved_payload_fields': sorted(set(saved)-{'model', 'config', 'initialization'})}
    result = {**saved, 'config': config, 'model': state}
    if isinstance(saved.get('initialization'), dict):
        result['initialization'] = {**saved['initialization'], **profile}
    return result, report


def _read_json(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def _backup_once(path):
    backup = path.with_name(path.stem+'_before_spatial'+path.suffix)
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
    descriptor, temporary = tempfile.mkstemp(prefix=destination.stem+'.spatial.', suffix='.tmp', dir=destination.parent)
    os.close(descriptor)
    path = Path(temporary)
    try:
        torch.save(payload, path)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def migrate_spatial_run(run_dir, target_seconds=30., seed=173, sr_checkpoint=None):
    """Migrate both stopped175M resume/initial checkpoints, commit marker last."""
    root = Path(run_dir).resolve()
    if not 0 < target_seconds < float('inf'):
        raise ValueError('Target duration must be finite and positive')
    for name in ('training.pid', 'controller.pid'):
        pid = root/name
        if pid.exists() and process_alive(int(pid.read_text(encoding='ascii').strip())):
            raise RuntimeError('Stop the world/controller before spatial checkpoint migration')
    main = root/'checkpoints'/'teacher175vq_world.pt'
    initial = root/'checkpoints'/'teacher_vjepa175vq_initial.pt'
    initialization = root/'initialization.json'
    journal = root/'spatial_entities_migration.json'
    previous = _read_json(journal, {})
    if previous and (previous.get('run_dir') != str(root) or previous.get('target_seconds') != target_seconds
                     or previous.get('seed') != seed):
        raise ValueError('Existing spatial migration belongs to a different run/duration/seed')
    if previous.get('status') == 'complete':
        if (previous.get('run_dir') != str(root) or previous.get('target_seconds') != target_seconds
                or not main.is_file() or not initial.is_file()):
            raise ValueError('Completed spatial migration differs from the requested run/duration')
        if sr_checkpoint is not None and previous.get('sr_checkpoint') != str(Path(sr_checkpoint).resolve()):
            raise ValueError('Completed spatial migration already selected a different SR checkpoint')
        for path, phase in ((main, 'teacher175vq_world'), (initial, 'teacher175vq_initial')):
            payload = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
            if payload.get('pipeline_phase') != phase or payload.get('config', {}).get('spatial_entities') is not True:
                raise ValueError('Completed spatial migration no longer matches its checkpoint configuration')
            del payload
        return previous
    if not main.is_file() or not initial.is_file() or not initialization.is_file():
        raise ValueError('Stopped175M world, its initial checkpoint and initialization marker are required')
    source = torch.load(main, map_location='cpu', weights_only=False)
    origin = torch.load(initial, map_location='cpu', weights_only=False)
    if (source.get('pipeline_phase') != 'teacher175vq_world' or int(source.get('training_steps', 0)) < 1
            or origin.get('pipeline_phase') != 'teacher175vq_initial'
            or int(origin.get('training_steps', -1)) != 0 or int(origin.get('global_training_step', 0)) != 0):
        raise ValueError('Wrong world/initial source phase or step; refusing to overwrite another stage')
    main_config, initial_config = (asdict(WorldModelConfig(**payload['config'])) for payload in (source, origin))
    for config in (main_config, initial_config):
        config.pop('spatial_entities')
    if main_config != initial_config:
        raise ValueError('World and initial checkpoint configurations do not belong to the same run')
    if main_config['kind'] != 'teacher':
        raise ValueError('Stopped teacher175 run cannot be replaced with a student checkpoint')
    marker = _read_json(initialization)
    origin_config = asdict(WorldModelConfig(**origin['config']))
    legacy_config = {**origin_config, 'spatial_entities': False}
    acceptable_fingerprints = {config_fingerprint(origin_config), config_fingerprint(legacy_config),
                               previous.get('teacher_config_fingerprint')}
    if (not isinstance(marker, dict) or marker.get('teacher_size') != '175m_vqgan'
            or marker.get('teacher_config_fingerprint') not in acceptable_fingerprints):
        raise ValueError('Initialization marker/configuration does not identify this teacher source')
    if previous.get('status') == 'pending' and int(source.get('training_steps', 0)) != previous.get('source_step'):
        raise ValueError('World checkpoint advanced while spatial migration was pending')
    sr_path = Path(sr_checkpoint).resolve() if sr_checkpoint is not None else (
        Path(__file__).resolve().parents[1]/'training_runs'/'sr32x4_quality_20261109'/'checkpoints'/'sr_for_game.pt')
    sr = torch.load(sr_path, map_location='cpu', weights_only=False)
    if (sr.get('stage') != 'sr' or int(sr.get('training_steps', 0)) < 1
            or sr.get('input_size') != main_config['image_size']
            or sr.get('output_size') != main_config['image_size']*4 or sr.get('config', {}).get('scale') != 4):
        raise ValueError('Updated SR must be trained x4 at the world native resolution')
    del sr
    upgraded, main_report = enable_spatial_checkpoint(source, seed)
    upgraded_initial, initial_report = enable_spatial_checkpoint(origin, seed)
    profile = {'teacher_parameters': main_report['generator_parameters'],
        'teacher_config_fingerprint': main_report['teacher_config_fingerprint'],
        'spatial_entities': True, 'spatial_entity_count': 9}
    upgraded_marker = {**marker, **profile}
    staged = []
    try:
        staged.append(_stage_checkpoint(upgraded, main))
        staged.append(_stage_checkpoint(upgraded_initial, initial))
        backups = {str(path): str(_backup_once(path)) for path in (main, initial, initialization)}
        config = _read_json(root/'experiment_config175vq.json', {})
        report = {**previous, 'format_version': 1, 'status': 'pending', 'run_dir': str(root),
            'spatial_entities': True, 'spatial_entity_count': 9, 'target_seconds': float(target_seconds),
            'sr_checkpoint': str(sr_path), 'source_step': main_report['source_step'],
            'start_optimizer_step': main_report['start_optimizer_step'], 'seed': int(seed),
            'generator_parameters': main_report['generator_parameters'],
            'teacher_config_fingerprint': main_report['teacher_config_fingerprint'],
            'training_subset_manifest': config.get('training_subset_manifest'), 'backups': backups,
            'files': {'world': str(main), 'initial': str(initial), 'initialization': str(initialization)},
            'world_report': previous.get('world_report', main_report),
            'initial_report': previous.get('initial_report', initial_report),
            'teacher_budget_preserved': True, 'created_at': previous.get('created_at', now())}
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
    parser.add_argument('--target-seconds', type=float, default=30.)
    parser.add_argument('--seed', type=int, default=173)
    parser.add_argument('--sr-checkpoint', type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    print(json.dumps(migrate_spatial_run(args.run_dir, args.target_seconds, args.seed,
                                      args.sr_checkpoint), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
