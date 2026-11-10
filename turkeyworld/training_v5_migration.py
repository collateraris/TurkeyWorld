"""Upgrade stopped game training while preserving compatible weights/moments."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch
from torch.nn import functional as F

from .combat_migration import config_fingerprint
from .models import NeuralWorldModel, WorldModelConfig
from .training_pipeline import atomic_json, now, replace_file
from .vqgan_pipeline import process_alive


NEW_PREFIXES = ('vjepa_backend.spatial_spawn_anchor_head.',
    'flow_renderer.block_spatial_condition.', 'flow_renderer.block_wall_condition.',
    'flow_renderer.residual_adapter.block_spatial_condition.',
    'flow_renderer.residual_adapter.block_wall_condition.')
PATCH_KEYS = {prefix+suffix for prefix in ('flow_renderer.', 'flow_renderer.residual_adapter.')
    for suffix in ('image_input.weight', 'position', 'output.weight', 'output.bias')}


def resize_patch_tensor(name, source, destination, config):
    """Deterministic patch4->2 transfer; no exact-function claim is made."""
    if name not in PATCH_KEYS:
        raise ValueError('Unsupported resized model tensor: '+name)
    channels = int(config['renderer_codec_config']['embedding_dim'])
    old_patch, new_patch = 4, int(config['flow_patch_size'])
    if new_patch != 2:
        raise ValueError('This migration transfers patch4 to patch2 only')
    value = source.float()
    if name.endswith('image_input.weight'):
        if source.shape[-2:] != (old_patch, old_patch) or destination.shape[-2:] != (new_patch, new_patch):
            raise ValueError('Unexpected input patch tensor: '+name)
        # Sum each2x2 subkernel: preserves the constant-input response and
        # approximately preserves variance for independent latent inputs.
        value = F.avg_pool2d(value, 2)*4
    elif name.endswith('position'):
        old_grid, new_grid = int(source.shape[1]**.5), int(destination.shape[1]**.5)
        if old_grid**2 != source.shape[1] or new_grid**2 != destination.shape[1]:
            raise ValueError('Position tokens must form square grids')
        value = value.reshape(1, old_grid, old_grid, source.shape[-1]).permute(0, 3, 1, 2)
        value = F.interpolate(value, (new_grid, new_grid), mode='bilinear', align_corners=False)
        value = value.permute(0, 2, 3, 1).reshape_as(destination)
    elif name.endswith('output.weight'):
        width = source.shape[1]
        if source.shape[0] != channels*old_patch**2:
            raise ValueError('Unexpected output patch channels: '+name)
        value = value.reshape(channels, old_patch, old_patch, width).permute(0, 3, 1, 2)
        value = F.avg_pool2d(value, 2).permute(0, 2, 3, 1).reshape_as(destination)
    elif name.endswith('output.bias'):
        value = F.avg_pool2d(value.reshape(channels, 1, old_patch, old_patch), 2).reshape_as(destination)
    else:
        raise ValueError('Unsupported resized Flow parameter: '+name)
    if value.shape != destination.shape or not bool(torch.isfinite(value).all()):
        raise ValueError('Patch transfer produced an invalid tensor: '+name)
    return value.to(source.dtype)


def upgrade_training_checkpoint(saved, seed=173):
    """Keep all unchanged tensors and all non-patch Adam/EMA/GAN/RNG state."""
    if saved.get('stage') != 'world' or not isinstance(saved.get('model'), dict):
        raise ValueError('Training v5 migration requires a complete world checkpoint')
    old_config = WorldModelConfig(**saved['config'])
    if (old_config.architecture != 'vjepa2_ac' or old_config.kind != 'teacher'
            or old_config.image_size != 32 or not old_config.spatial_combat
            or old_config.renderer_codec != 'vqgan'):
        raise ValueError('Training v5 migration requires the native32 spatial-combat teacher')
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError('Seed must be a nonnegative integer')
    counters = saved.get('training_counters', {})
    if counters.get('pending_microbatches', 0):
        raise ValueError('Save a completed accumulation group before migration')
    config = asdict(old_config)
    config.update(game_training_v5=True, flow_patch_size=2, flow_block_conditioning=True)
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        torch.manual_seed(seed)
        model = NeuralWorldModel(WorldModelConfig(**config)).eval()
    desired = model.state_dict()
    source = saved['model']
    if set(source)-set(desired):
        raise ValueError('Migration unexpectedly removes model tensors')
    new_names = sorted(set(desired)-set(source))
    if any(not name.startswith(NEW_PREFIXES) for name in new_names):
        raise ValueError('Unrecognized new model tensors: '+str(new_names))
    result_state, resized, preserved = {}, [], []
    for name, target in desired.items():
        if name not in source:
            result_state[name] = target.detach().clone()
        elif source[name].shape == target.shape:
            result_state[name] = source[name]
            preserved.append(name)
        else:
            if old_config.game_training_v5:
                raise ValueError('Already-v5 checkpoint has incompatible shapes: '+name)
            result_state[name] = resize_patch_tensor(name, source[name], target, config)
            resized.append(name)
    if old_config.game_training_v5 and new_names:
        raise ValueError('Already-v5 checkpoint is missing v5 weights')
    model.load_state_dict(result_state, strict=True)
    result = {**saved, 'config': config, 'model': result_state}
    reset_moments, kept_moments = [], []
    if 'optimizer' in saved:
        optimizer = saved['optimizer']
        groups = optimizer.get('param_groups', [])
        names = saved.get('optimizer_parameter_names')
        if len(groups) != 1 or not names or len(names) != 1 or len(names[0]) != len(groups[0]['params']):
            raise ValueError('Migration requires explicitly named optimizer state')
        moments = dict(optimizer['state'])
        for name, identifier in zip(names[0], groups[0]['params']):
            if name in resized:
                moments.pop(identifier, None)
                reset_moments.append(name)
            elif identifier in moments:
                kept_moments.append(name)
        result['optimizer'] = {**optimizer, 'state': moments}
    count = sum(parameter.numel() for parameter in model.parameters())
    fingerprint = config_fingerprint(config)
    profile = {'teacher_parameters': count, 'teacher_config_fingerprint': fingerprint,
        'game_training_v5': True, 'flow_patch_size': 2, 'flow_block_conditioning': True}
    if isinstance(saved.get('initialization'), dict):
        result['initialization'] = {**saved['initialization'], **profile}
    migration = {'format_version': 1, 'source_step': int(saved.get('global_training_step', 0)),
        'game_training_v5': True, 'flow_patch_size': 2, 'flow_image_tokens': 16,
        'flow_block_conditioning': True, 'teacher_config_fingerprint': fingerprint,
        'generator_parameters': count, 'seed': seed, 'behavior_validation_every': 100}
    result['training_v5_migration'] = migration
    report = {**migration, 'status': 'already_enabled' if old_config.game_training_v5 else 'upgraded',
        'preserved_tensor_count': len(preserved), 'resized_tensors': resized, 'new_tensors': new_names,
        'preserved_optimizer_states': len(kept_moments), 'reset_patch_optimizer_states': reset_moments,
        'source_config_fingerprint': config_fingerprint(asdict(old_config)),
        'preserved_payload_fields': sorted(set(saved)-{'config', 'model', 'optimizer', 'initialization'})}
    return result, report


def _backup(path):
    target = path.with_name(path.stem+'_before_v5'+path.suffix)
    if not target.exists():
        try:
            os.link(path, target)
        except FileExistsError:
            pass
        except OSError:
            with path.open('rb') as source, target.open('xb') as destination:
                shutil.copyfileobj(source, destination)
    return target


def _stage(payload, destination):
    handle, name = tempfile.mkstemp(prefix=destination.stem+'.v5.', suffix='.tmp', dir=destination.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        torch.save(payload, temporary)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def migrate_training_run(run_dir, seed=173):
    """Exclusive backups, resumable transaction, completion marker committed last."""
    root = Path(run_dir).resolve()
    parent_path = root/'pipeline_parent.json'
    pid_paths = [root/'training.pid']
    if parent_path.exists():
        parent = Path(json.loads(parent_path.read_text(encoding='utf-8'))['run_dir'])
        pid_paths.append(parent/'training.pid')
    for path in pid_paths:
        if path.exists() and process_alive(int(path.read_text().strip())):
            raise RuntimeError('Stop training/controller before migration: '+str(path))
    marker_path = root/'training_v5_migration.json'
    previous = json.loads(marker_path.read_text(encoding='utf-8')) if marker_path.exists() else {}
    main = root/'checkpoints/teacher175vq_world.pt'
    initial = root/'checkpoints/teacher_vjepa175vq_initial.pt'
    initialization = root/'initialization.json'
    if not all(path.is_file() for path in (main, initial, initialization)):
        raise ValueError('Stopped world, initial checkpoint and initialization marker are required')
    if previous.get('status') == 'complete':
        current = torch.load(main, map_location='cpu', weights_only=False, mmap=True)
        origin = torch.load(initial, map_location='cpu', weights_only=False, mmap=True)
        expected = previous['teacher_config_fingerprint']
        for payload in (current, origin):
            if not payload['config'].get('game_training_v5') or config_fingerprint(payload['config']) != expected:
                raise ValueError('Completed v5 marker no longer matches checkpoints')
        return previous
    backups = {str(path): str(_backup(path)) for path in (main, initial, initialization)}
    # On recovery always use immutable pre-v5 sources, never double-transform.
    saved = torch.load(backups[str(main)], map_location='cpu', weights_only=False, mmap=True)
    original = torch.load(backups[str(initial)], map_location='cpu', weights_only=False, mmap=True)
    if config_fingerprint(asdict(WorldModelConfig(**saved['config']))) != config_fingerprint(asdict(WorldModelConfig(**original['config']))):
        raise ValueError('World and initial checkpoint configurations differ')
    upgraded, report = upgrade_training_checkpoint(saved, seed)
    upgraded_initial, initial_report = upgrade_training_checkpoint(original, seed)
    profile = {key: report[key] for key in ('generator_parameters', 'teacher_config_fingerprint',
        'game_training_v5', 'flow_patch_size', 'flow_block_conditioning')}
    old_marker = json.loads(Path(backups[str(initialization)]).read_text(encoding='utf-8'))
    if old_marker.get('teacher_config_fingerprint') != report['source_config_fingerprint']:
        raise ValueError('Initialization marker does not match the pre-v5 checkpoint')
    new_marker = {**old_marker, **profile, 'teacher_parameters': profile['generator_parameters']}
    pending = {**report, 'status': 'pending', 'created_at': now(), 'run_dir': str(root),
        'backups': backups, 'initial_report': initial_report}
    atomic_json(marker_path, pending)
    staged = []
    try:
        staged.append(_stage(upgraded, main))
        staged.append(_stage(upgraded_initial, initial))
        # Release all source memory mappings before replacing either hardlink
        # on Windows. Report/journal objects contain no tensor references.
        del saved, original, upgraded, upgraded_initial
        gc.collect()
        replace_file(staged[0], main)
        replace_file(staged[1], initial)
        atomic_json(initialization, new_marker)
        completed = {**pending, 'status': 'complete', 'completed_at': now()}
        atomic_json(marker_path, completed)
        return completed
    finally:
        for path in staged:
            path.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=173)
    args = parser.parse_args(argv)
    print(json.dumps(migrate_training_run(args.run_dir, args.seed), indent=2))


if __name__ == '__main__':
    main()
