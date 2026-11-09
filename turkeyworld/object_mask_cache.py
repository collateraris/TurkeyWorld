"""Offline visible-NPC labels; training only reads the resulting sidecars."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import numpy as np


MASK_KIND = 'visible_npc_rgb_difference'


def source_identity(path):
    path = Path(path)
    stat = path.stat()
    return {'episode': path.name, 'source_size': stat.st_size, 'source_mtime_ns': stat.st_mtime_ns}


def _cache_metadata(saved, source_path, expected_split):
    metadata = json.loads(str(saved['metadata']))
    if metadata.get('mask_kind') != MASK_KIND or metadata.get('format_version') != 1:
        raise ValueError('Unsupported NPC mask cache')
    identity = source_identity(source_path)
    if any(metadata.get(key) != value for key, value in identity.items()):
        raise ValueError('NPC mask cache belongs to a changed source episode')
    if expected_split is not None and metadata.get('split') != expected_split:
        raise ValueError('NPC mask cache split mismatch')
    shape = tuple(int(value) for value in saved['shape'])
    if len(shape) != 3 or min(shape) < 1 or saved['mask_bits'].shape != (*shape[:2], (shape[2]+7)//8):
        raise ValueError('Invalid NPC mask dimensions')
    return metadata, shape


def read_object_masks(path, source_path, expected_split=None):
    """Read labels without importing/calling the source engine or decoding RGB."""
    with np.load(path, allow_pickle=False) as saved:
        metadata, shape = _cache_metadata(saved, source_path, expected_split)
        mask = np.unpackbits(saved['mask_bits'], axis=-1, count=shape[2]).astype(np.bool_)
    return mask, metadata


def read_object_backgrounds(path, source_path, expected_split=None):
    """Read offline NPC-removed RGB labels; never query/import the engine."""
    with np.load(path, allow_pickle=False) as saved:
        metadata, shape = _cache_metadata(saved, source_path, expected_split)
        if metadata.get('background_frames_available') is not True or 'background_rgb' not in saved.files:
            raise ValueError('NPC cache has no counterfactual background frames; upgrade it offline')
        frames = saved['background_rgb']
        if frames.dtype != np.uint8 or frames.shape != (*shape, 3):
            raise ValueError('Invalid counterfactual background RGB shape/dtype')
    return frames, metadata


def _generate_masks(job):
    # This import is intentionally confined to OFFLINE annotation generation.
    from .engine import Game, BEHAVIOR_VERSION
    path, output = Path(job['source']), Path(job['output'])
    identity = source_identity(path)
    include_backgrounds = bool(job.get('include_backgrounds', False))
    previous_mask = None
    if output.exists():
        mask, metadata = read_object_masks(output, path, job['split'])
        if metadata.get('behavior_version') != BEHAVIOR_VERSION:
            raise ValueError('NPC mask behavior version mismatch')
        if not include_backgrounds:
            return metadata
        if metadata.get('background_frames_available') is True:
            read_object_backgrounds(output, path, job['split'])
            return metadata
        previous_mask = mask
    with np.load(path, allow_pickle=False) as data:
        behavior = str(data['behavior_version'])
        if behavior != BEHAVIOR_VERSION or str(data['split']) != job['split']:
            raise ValueError('NPC mask source behavior/split mismatch')
        frames, actions = data['frames'], data['actions']
        game = Game(seed=int(data['seed']), endless=bool(data['endless']))
        game.set_state(json.loads(str(data['initial_state'])))
        dt = float(data['dt'])
        if frames.shape[0] != len(actions)+1 or frames.shape[1] != frames.shape[2]:
            raise ValueError('NPC mask source dimensions mismatch')
        masks = np.empty(frames.shape[:3], dtype=np.bool_)
        backgrounds = np.empty_like(frames) if include_backgrounds else None
        for index in range(len(frames)):
            full = game.render(size=frames.shape[1], overlays=False)
            if not np.array_equal(full, frames[index]):
                raise ValueError(f'Offline source replay differs from recorded RGB: {path.name} frame {index}')
            enemies = game.enemies
            try:
                game.enemies = []
                background = game.render(size=frames.shape[1], overlays=False)
            finally:
                game.enemies = enemies
            masks[index] = np.any(full != background, axis=-1)
            if backgrounds is not None:
                backgrounds[index] = background
            if index < len(actions):
                game.step(actions[index], dt=dt)
        if previous_mask is not None and not np.array_equal(previous_mask, masks):
            raise ValueError('Existing NPC masks differ from exact replay; refusing background upgrade')
        metadata = {**identity, 'format_version': 1, 'mask_kind': MASK_KIND,
            'background_frames_available': include_backgrounds,
            'behavior_version': behavior, 'split': job['split'], 'verified_source_frames': len(frames),
            'source_frames_sha256': hashlib.sha256(frames.tobytes()).hexdigest(),
            'foreground_pixels': int(masks.sum()),
            'frames_with_npc_pixels': int(masks.any(axis=(1, 2)).sum()),
            'mean_foreground_fraction': float(masks.mean())}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name+f'.{os.getpid()}.tmp')
    payload = {'mask_bits': np.packbits(masks, axis=-1), 'shape': np.asarray(masks.shape),
               'metadata': np.asarray(json.dumps(metadata))}
    if backgrounds is not None:
        payload['background_rgb'] = backgrounds
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **payload)
    os.replace(temporary, output)
    return metadata


def prepare_object_masks(data_root, output, training_subset_manifest, workers=8, include_backgrounds=False):
    from .engine import BEHAVIOR_VERSION
    from .training_subset import load_training_subset
    from .training_pipeline import atomic_json
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    if workers < 1:
        raise ValueError('Mask workers must be positive')
    subset = load_training_subset(training_subset_manifest, data_root/'rgb'/'train')
    jobs = []
    for split in ('train', 'val'):
        paths = ([data_root/'rgb'/split/name for name in subset['filenames']] if split == 'train'
                 else sorted((data_root/'rgb'/split).glob('*.npz')))
        for path in paths:
            jobs.append({'source': str(path), 'output': str(output/split/path.name), 'split': split,
                         'include_backgrounds': bool(include_backgrounds)})
    report = {'format_version': 1, 'mask_kind': MASK_KIND, 'status': 'running',
        'source_data_root': str(data_root), 'training_subset_fingerprint': subset['fingerprint'],
        'behavior_version': BEHAVIOR_VERSION, 'total_episodes': len(jobs), 'completed_episodes': 0,
        'runtime_engine_calls': False, 'source_replay': 'Offline only, every RGB frame verified bit-exact',
        'background_frames_available': False,
        'records': []}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        for key in ('source_data_root', 'training_subset_fingerprint', 'behavior_version', 'mask_kind'):
            if previous.get(key) != report[key]:
                raise ValueError('Existing NPC mask cache belongs to a different experiment')
    atomic_json(manifest_path, report)
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for record in pool.map(_generate_masks, jobs, chunksize=1):
                record = dict(record, background_frames_available=bool(record.get('background_frames_available', False)))
                report['records'].append(record)
                report['completed_episodes'] += 1
                if report['completed_episodes'] % 64 == 0:
                    atomic_json(manifest_path, report)
                    print(json.dumps({'object_mask_episodes': report['completed_episodes'], 'total': len(jobs)}), flush=True)
    except Exception as error:
        report.update(status='failed', error=str(error))
        atomic_json(manifest_path, report)
        raise
    report['status'] = 'complete'
    report['background_frames_available'] = bool(report['records']) and all(
        record['background_frames_available'] for record in report['records'])
    report['verified_source_frames'] = sum(record['verified_source_frames'] for record in report['records'])
    atomic_json(manifest_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--training-subset-manifest', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--include-backgrounds', action='store_true', help='Offline cache/upgrade of exact NPC-removed RGB targets')
    args = parser.parse_args(argv)
    options = vars(args)
    if not options['include_backgrounds']:
        options.pop('include_backgrounds')  # Preserve the legacy CLI/API default call.
    report = prepare_object_masks(**options)
    print(json.dumps({key: report[key] for key in ('status', 'completed_episodes', 'verified_source_frames')}))


if __name__ == '__main__':
    main()
