"""Frozen current-world NPR pairs and causal hints with recorded Source profiles.

Only Source provenance chooses navigation/combat. Current/future life or HP
values never choose the environment. Target state is a render query, while
both appearance and RSSM hypotheses read only genuine previous observations.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np
import torch

from .cache_enemy_decoder_hypotheses import preserve_rng, rng_fingerprint, initialize_cache_device
from .cache_sr_neural_pairs import build_cache, _pixels
from .visual_quality import strict_render_precision
from .enemy_decoder_data import HYPOTHESIS_PROTOCOL
from .renderer_student import _hash_tensors
from .rssm_combat_protocol import environment_profile
from .small_enemy_data import digest
from .sr_neural_pairs_data import SemanticSRData, _atomic_json, fingerprint


ENVIRONMENT_POLICY = 'recorded_Source_header_or_path_environment_profiles_v1'


@lru_cache(maxsize=4096)
def _source_header(path, size, modified):
    del size, modified  # stat values deliberately participate in the immutable cache key.
    result = {}
    with np.load(path, allow_pickle=False) as arrays:
        for key in ('navigation_only', 'nav_only', 'enemies_enabled', 'stage'):
            if key in arrays:
                result[key] = arrays[key].tolist()
        if 'snapshot_states' in arrays:
            snapshots = json.loads(str(arrays['snapshot_states'].item()))
            # Only environment metadata is retained; no pose/life/HP becomes a condition.
            result['snapshot_environments'] = [value.get('enemies_enabled') for value in snapshots]
    return result


def source_navigation_only(record):
    """Resolve Source environment metadata without looking at any entity state."""
    values = []
    for key in ('navigation_only', 'nav_only', 'source_navigation_only'):
        if key in record:
            if type(record[key]) is not bool:
                raise ValueError('Recorded navigation profile must be an explicit boolean')
            values.append(record[key])
    if 'enemies_enabled' in record:
        if type(record['enemies_enabled']) is not bool:
            raise ValueError('Recorded enemies_enabled must be a boolean')
        values.append(not record['enemies_enabled'])
    if values and len(set(values)) != 1:
        raise ValueError('Contradictory recorded Source navigation metadata')
    explicit = values[0] if values else None
    raw_path = record.get('source_path')
    header = {}
    if raw_path and Path(raw_path).is_file():
        stat = Path(raw_path).stat()
        header = _source_header(str(Path(raw_path).resolve()), stat.st_size, stat.st_mtime_ns)
    frame = int(record.get('frame', 0))
    header_values = []
    for key in ('navigation_only', 'nav_only', 'enemies_enabled'):
        if key in header:
            value = header[key]
            if isinstance(value, list): value = value[frame]
            if type(value) is not bool:
                raise ValueError('Source NPZ environment flags must be booleans')
            header_values.append(not value if key == 'enemies_enabled' else value)
    if 'snapshot_environments' in header:
        environments = header['snapshot_environments']
        value = environments[frame]
        if value is not None:
            if type(value) is not bool:
                raise ValueError('Source snapshot has invalid environment metadata')
            if record.get('dynamic') and frame > 0 and environments[frame-1] != value:
                raise ValueError('Dynamic Source pair crosses an environment switch')
            # Independent snapshots may override their file's default profile.
            # A dynamic pair is accepted only when the previous profile agrees.
            header_values = [not value]
    if len(set(header_values)) > 1 or explicit is not None and header_values and explicit != header_values[0]:
        raise ValueError('Contradictory recorded Source navigation metadata')
    if explicit is not None: return explicit
    if header_values: return header_values[0]
    path = str(raw_path or '').replace('\\', '/').lower()
    stage = str(record.get('source_stage', record.get('stage', header.get('stage', '')))).lower()
    if '/navigation/' in path or 'navigation' in stage or record.get('source_kind') == 'empty_navigation':
        return True
    if any('/'+part+'/' in path or part in stage for part in ('full_game', 'moving1', 'static_enemy')):
        return False
    raise ValueError('Recorded Source environment provenance is required; stratum/life flags are insufficient')


def _profiles(records, count):
    if records is None or len(records) != count:
        raise ValueError('Each cached row requires its exact Source provenance record')
    return [source_navigation_only(record) for record in records]


def _causal_prior(model, past, records):
    device = next(model.parameters()).device
    dynamic = past['dynamic'].to(device=device)
    if dynamic.ndim != 1 or dynamic.dtype != torch.bool or not len(dynamic):
        raise ValueError('Genuine-past dynamic flags must be boolean[B]')
    count = len(dynamic)
    profiles = torch.tensor(_profiles(records, count), device=device, dtype=torch.bool)
    features = torch.zeros(count, 128, device=device)
    predicted = torch.zeros(count, 20, device=device)
    for navigation in (False, True):
        ids = torch.nonzero(dynamic & (profiles == navigation), as_tuple=False).flatten()
        if not len(ids):
            continue
        previous = past['previous_state'].to(device=device, dtype=torch.float32)[ids]
        image = past['previous_rgb'].to(device=device, dtype=torch.float32)[ids]
        actions = past['action'].to(device=device, dtype=torch.float32)[ids]
        grid = past['map'].to(device=device, dtype=torch.long)[ids]
        if (previous.shape != (len(ids), 20) or image.shape != (len(ids), 3, 32, 32)
                or actions.shape != (len(ids), 4) or grid.shape != (len(ids), 16, 16)
                or not all(bool(torch.isfinite(x).all()) for x in (previous, image, actions))
                or bool(((image < 0) | (image > 1)).any())):
            raise ValueError('Dynamic hypotheses require genuine finite previous RGB/state/action/map')
        with environment_profile(model, navigation_only=navigation):
            encoded = model.encode_map(grid)
            memory = model.initialize_from_image(image, previous, encoded)
            prior = model.transition(previous, actions, encoded, memory)
        value, state = prior['features'].detach().float(), prior['state'].detach().float()
        if (value.shape != (len(ids), 128) or state.shape != (len(ids), 20)
                or not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(state).all())):
            raise FloatingPointError('Frozen profiled prior returned invalid state or appearance')
        features[ids], predicted[ids] = value, state
    return features, predicted, dynamic, profiles


@torch.inference_mode()
def render_profiled_cache_batch(model, batch, *, records, include_codec=True):
    """GT pose/map is a query; no current RGB/HR/mask enters the neural prior."""
    device = next(model.parameters()).device
    past = {key: batch[key] for key in ('dynamic', 'previous_rgb', 'previous_state', 'action', 'map')}
    with preserve_rng(device):
        features, predicted, dynamic, profiles = _causal_prior(model, past, records)
        state = batch['state'].to(device=device, dtype=torch.float32)
        grid = batch['map'].to(device=device, dtype=torch.long)
        if state.shape != (len(dynamic), 20) or not bool(torch.isfinite(state).all()):
            raise ValueError('NPR queries require finite recorded target state20')
        generated = np.empty((len(dynamic), 32, 32, 3), dtype=np.uint8)
        for navigation in (False, True):
            ids = torch.nonzero(profiles == navigation, as_tuple=False).flatten()
            if not len(ids):
                continue
            with environment_profile(model, navigation_only=navigation), strict_render_precision():
                encoded = model.encode_map(grid[ids])
                generated[ids.cpu().numpy()] = _pixels(model.render(state[ids], features[ids], encoded), 32)
        result = {'lr_npr': generated, 'pred_state': predicted.cpu().numpy().astype(np.float32),
                  'valid': dynamic.cpu().numpy().astype(np.bool_)}
        if include_codec:
            with strict_render_precision():
                result['codec_lr'] = _pixels(model.flow_renderer.codec(
                    batch['lr_source'].to(device=device, dtype=torch.float32))['reconstruction'], 32)
        return result


@torch.inference_mode()
def predict_profiled_hypotheses(model, batch, *, records):
    """This deliberately does not accept or read target pose/RGB/mask labels."""
    device = next(model.parameters()).device
    past = {key: batch[key] for key in ('dynamic', 'previous_rgb', 'previous_state', 'action', 'map')}
    with preserve_rng(device):
        _, predicted, dynamic, _ = _causal_prior(model, past, records)
    return {'pred_state': predicted.cpu().numpy().astype(np.float32),
            'valid': dynamic.cpu().numpy().astype(np.bool_)}


def _hypotheses(data_root, output, model, *, world_sha256, world_step, batch_size, seed):
    root, data_root = Path(output).resolve(), Path(data_root).resolve()
    if root == data_root or root in data_root.parents:
        raise ValueError('Hypothesis sidecar must not overwrite paired inputs')
    base_path = data_root/'manifest.json'; base_bytes = base_path.read_bytes()
    base = json.loads(base_bytes)
    readers = {s: SemanticSRData(data_root, s, seed=seed) for s in ('train', 'val')}
    if readers['train'].scene_keys & readers['val'].scene_keys:
        raise ValueError('TRAIN and VAL Source scene overlap')
    identity = dict(protocol=HYPOTHESIS_PROTOCOL, base_manifest_sha256=digest(base_path),
        base_data_sha256=base['data_sha256'], world_sha256=world_sha256, world_step=world_step, seed=seed,
        bootstrap_horizon=1, past_fields=['previous_rgb', 'previous_state', 'action', 'map'],
        future_state_or_rgb_conditioning=False, static_hypothesis_policy='zero_state_invalid',
        observation_policy='one_real_previous_Source_observation_then_frozen_prior',
        source_environment_policy=ENVIRONMENT_POLICY, autonomous_450_step_claim=False, optimizer_steps=0)
    plan = fingerprint(identity); root.mkdir(parents=True, exist_ok=True)
    path = root/'manifest.json'
    if path.exists():
        old = json.loads(path.read_text(encoding='utf-8'))
        if old.get('fingerprint') != plan:
            raise ValueError('Fresh profiled hypothesis identity changed')
        if old.get('status') == 'complete':
            if any(digest(root/e['path']) != e['sha256'] for e in old['shards']):
                raise ValueError('Completed hypothesis shard changed')
            return old
    lookup = {split: {(row['shard'], row['offset']): i for i, row in enumerate(data.rows)}
              for split, data in readers.items()}
    modes = {m: m.training for m in model.modules()}
    flags = {p: p.requires_grad for p in model.parameters()}
    before = _hash_tensors(model.state_dict()); device = next(model.parameters()).device
    rng = rng_fingerprint(device); model.eval().requires_grad_(False)
    shards = []; completed = valid_count = 0; started = time.perf_counter()
    def publish(status, verified=False):
        report = {**identity, 'fingerprint': plan, 'status': status, 'stage': 'data',
            'current': completed, 'total': len(base['records']), 'shards': shards,
            'valid_dynamic_rows': valid_count, 'invalid_static_rows': completed-valid_count,
            'elapsed_seconds': time.perf_counter()-started, 'base_arrays_unchanged': True,
            'frozen_model_tensors_bitexact': verified, 'rng_state_unchanged': verified,
            'gameplay_qualified': False, 'automatic_deployment': False}
        report['data_sha256'] = fingerprint({'identity': identity, 'shards': shards})
        _atomic_json(path, report)
        _atomic_json(root/'status.json', {k: v for k, v in report.items() if k != 'shards'})
        return report
    publish('caching')
    try:
        for entry in base['shards']:
            name, split, count = entry['path'], entry['split'], entry['rows']
            if digest(data_root/name) != entry['sha256']:
                raise ValueError('Immutable paired shard changed')
            states = []; validity = []
            for start in range(0, count, batch_size):
                ids = [lookup[split][(name, j)] for j in range(start, min(count, start+batch_size))]
                batch = readers[split].batch(len(ids), indices=ids)
                result = predict_profiled_hypotheses(model, batch,
                    records=[readers[split].rows[i] for i in ids])
                states.append(result['pred_state']); validity.append(result['valid'])
            arrays = dict(pred_state=np.concatenate(states), valid=np.concatenate(validity))
            target = root/name; target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                with np.load(target, allow_pickle=False) as z:
                    if set(z.files) != set(arrays) or any(not np.array_equal(z[k], v) for k, v in arrays.items()):
                        raise ValueError('Existing fresh hypothesis sidecar differs')
            else:
                temporary = target.with_suffix('.tmp.npz')
                np.savez_compressed(temporary, **arrays); temporary.replace(target)
            shards.append(dict(path=name, base_shard=name, base_sha256=entry['sha256'],
                sha256=digest(target), rows=count, split=split))
            completed += count; valid_count += int(arrays['valid'].sum()); publish('caching')
        if base_bytes != base_path.read_bytes() or any(digest(data_root/e['path']) != e['sha256'] for e in base['shards']):
            raise AssertionError('Source paired inputs changed during hypothesis caching')
        if before != _hash_tensors(model.state_dict()) or rng != rng_fingerprint(device):
            raise AssertionError('Frozen world/RNG changed during profiled hypothesis caching')
        if any(p.grad is not None for p in model.parameters()):
            raise AssertionError('Unexpected frozen model gradient')
        return publish('complete', True)
    except BaseException:
        publish('failed'); raise
    finally:
        for m, mode in modes.items(): m.training = mode
        for p, enabled in flags.items(): p.requires_grad_(enabled)


def build_sr_moe_cache(source_root, output, model, *, cache_identity, world_sha256,
                       world_step=None, hypotheses_output=None, batch_size=8, seed=173,
                       include_codec=True):
    source_manifest = json.loads((Path(source_root)/'manifest.json').read_text(encoding='utf-8'))
    profiles = [source_navigation_only(record) for record in source_manifest['records']]
    identity = {**cache_identity, 'world_sha256': world_sha256, 'world_step': world_step,
        'source_environment_policy': ENVIRONMENT_POLICY,
        'source_environment_fingerprint': fingerprint(profiles),
        'source_environment_counts': {'navigation': int(sum(profiles)), 'combat': int(len(profiles)-sum(profiles))},
        'hypothesis_policy': 'actual_past_profiled_frozen_prior_one_step'}
    paired = build_cache(source_root, output, model, cache_identity=identity,
        batch_size=batch_size, seed=seed, include_codec=include_codec,
        render_batch_fn=render_profiled_cache_batch)
    hypotheses_output = Path(hypotheses_output) if hypotheses_output is not None else Path(output).parent/'hypotheses'
    hypotheses = _hypotheses(output, hypotheses_output, model, world_sha256=world_sha256,
        world_step=world_step, batch_size=batch_size, seed=seed)
    return {'paired': paired, 'hypotheses': hypotheses}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'output', 'world', 'renderer'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--hypotheses-output', type=Path)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--seed', type=int, default=173)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--allow-unqualified-codec', action='store_true')
    args = parser.parse_args(argv)
    device = initialize_cache_device(args.device)  # CUDA initialization precedes every torch.load on Windows.
    from .models import load_world_checkpoint
    from .renderer_student import load_renderer_student, codec_tensor_fingerprint
    model, world = load_world_checkpoint(args.world, 'cpu')
    renderer = load_renderer_student(model, args.renderer, allow_unqualified_codec=args.allow_unqualified_codec)
    model.float().to(device).eval().requires_grad_(False)
    identity = dict(world_sha256=digest(args.world), world_step=int(world['training_steps']),
        renderer_sha256=digest(args.renderer), renderer_step=int(renderer['training_steps']),
        renderer_refinement_updates=renderer.get('metrics', {}).get('refinement_updates'),
        codec_tensor_sha256=codec_tensor_fingerprint(model.flow_renderer.codec),
        future_rgb_features=False, paired_same_pose_map_tick=True)
    result = build_sr_moe_cache(args.source_root, args.output, model, cache_identity=identity,
        world_sha256=identity['world_sha256'], world_step=identity['world_step'],
        hypotheses_output=args.hypotheses_output, batch_size=args.batch, seed=args.seed)
    print(json.dumps({'status': 'complete', 'frames': result['paired']['current'],
        'paired_fingerprint': result['paired']['data_sha256'],
        'hypotheses_fingerprint': result['hypotheses']['fingerprint']}))


if __name__ == '__main__':
    main()
