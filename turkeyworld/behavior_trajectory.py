"""Held-out, state-only free trajectories; no future target enters the model."""

from .engine import GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS, STATE_DIM
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
import torch

from turkeyworld.runtime_checks import circle_is_clear, PLAYER_RADIUS, NPC_RADIUS
from turkeyworld.visual_quality import strict_render_precision


def _stats(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    return {'mean': float(finite.mean()) if len(finite) else None,
            'p95': float(np.percentile(finite, 95)) if len(finite) else None,
            'n': len(finite), 'nonfinite': int(len(values)-len(finite))}


def evaluate_state_trajectories(model, data, output_dir, context=2, horizon=16,
                                count=32, batch_size=4, seed=173):
    """Measure paired free AR state drift, independently of the image generator.

    One window per distinct uniformly stratified held-out episode; no overlap.
    NPC primary errors include false-missing predictions rather than dropping
    them. A separate common-alive metric makes its narrower mask explicit.
    Source slots which respawn after the context are excluded from paired NPC
    geometry/velocity errors, but remain in all life/death/confusion counts.
    Wall checks are measurement-only CPU circle-vs-tile tests, never feedback.
    """
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in (context, horizon, count, batch_size)):
        raise ValueError('Context/horizon/count/batch_size must be positive integers')
    if data.split not in ('val', 'test'):
        raise ValueError('Trajectory validation requires an explicit held-out val/test split')
    rng = np.random.default_rng(seed)
    eligible = sorted((Path(ep['path']).name, index, len(ep['frames']))
        for index, ep in enumerate(data.episodes) if len(ep['frames']) >= context+horizon)
    if len(eligible) < count:
        raise ValueError('Need count distinct eligible held-out episodes for nonoverlapping windows')
    chosen = [eligible[int(rng.choice(group))] for group in np.array_split(np.arange(len(eligible)), count)]
    panel = [(index, int(rng.integers(frames-context-horizon+1))) for _, index, frames in chosen]
    windows = [{'episode': name, 'episode_index': index, 'start': start}
               for (name, index, _), (_, start) in zip(chosen, panel)]
    content, identities = hashlib.sha256(), []
    for (index, start), window in zip(panel, windows):
        episode = data.episodes[index]
        path = Path(episode['path'])
        identities.append({'episode': path.name, 'bytes': path.stat().st_size,
                           'mtime_ns': path.stat().st_mtime_ns})
        for name, value in (('context_rgb', episode['frames'][start:start+context]),
                ('states', episode['states'][start:start+context+horizon]),
                ('actions', episode['actions'][start+context-1:start+context-1+horizon]), ('map', episode['map'])):
            array = np.ascontiguousarray(value)
            content.update(name.encode()+array.dtype.str.encode()+str(array.shape).encode()+array.tobytes())
    specification = {'version': 'heldout_free_state_v1', 'split': data.split, 'windows': windows,
        'context': context, 'horizon': horizon, 'seed': seed, 'source_content_sha256': content.hexdigest(),
        'episode_identity_sha256': hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()}
    fingerprint = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    device = next(model.parameters()).device
    modes = [(module, module.training) for module in model.modules()]
    caches = [(module, name, getattr(module, name).detach().clone()) for module in model.modules()
              for name in ('_spawn_walkable', '_map_condition') if torch.is_tensor(getattr(module, name, None))]
    prediction, reference, grids, controls, timesteps = [], [], [], [], []
    started = time.perf_counter()
    memory_shape = None
    try:
        model.eval()
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=cuda_devices), strict_render_precision(),\
                torch.autocast(device.type, enabled=False), torch.inference_mode():
            for first in range(0, count, batch_size):
                pairs = np.asarray(panel[first:first+batch_size], dtype=np.int64)
                batch = data.batch(len(pairs), context=context, horizon=horizon, indices=pairs)
                batch = {name: value.to(device) if torch.is_tensor(value) else value for name, value in batch.items()}
                fmap = model.encode_map(batch['map'])
                current = batch['states'][:, context-1].float()
                memory = model.initialize_from_image(batch['context_frames'], batch['states'][:, :context], fmap)
                if memory.shape[0] != len(pairs) or not bool(torch.isfinite(memory).all()):
                    raise FloatingPointError('Invalid initialized visual memory')
                expected_memory_shape = tuple(memory.shape[1:])
                memory_shape = expected_memory_shape
                path = [current.cpu().numpy().copy()]
                for offset in range(horizon):
                    result = model.transition(current, batch['actions'][:, offset], fmap, memory)
                    current, memory = result['state'], result['memory']
                    if (current.shape != (len(pairs), STATE_DIM) or not bool(torch.isfinite(current).all())
                            or memory.shape[0] != len(pairs) or tuple(memory.shape[1:]) != expected_memory_shape
                            or not bool(torch.isfinite(memory).all())):
                        raise FloatingPointError(f'Invalid predicted state/memory at future offset{offset+1}')
                    path.append(current.cpu().numpy().copy())
                prediction.append(np.stack(path, 1))
                reference.append(batch['states'][:, context-1:context+horizon].cpu().numpy())
                grids.append(batch['map'].cpu().numpy()); controls.append(batch['actions'].cpu().numpy())
                timesteps.append(batch['dt'].cpu().numpy())
    finally:
        for module, mode in modes:
            module.training = mode
        for module, name, value in caches:
            setattr(module, name, value)
    predicted, source = np.concatenate(prediction), np.concatenate(reference)
    tilemaps, actions, dt = np.concatenate(grids), np.concatenate(controls), np.concatenate(timesteps)
    if not np.allclose(dt, 1/float(model.config.simulation_hz), atol=1e-7, rtol=0):
        raise ValueError('Model timestep must match the recorded held-out source')
    pred_npc, true_npc = (array[:, :, GLOBAL_DIM:-2].reshape(count, horizon+1, ENTITY_SLOTS, ENTITY_DIM) for array in (predicted, source))
    pred_alive = (pred_npc[..., 5] >= .5) & (pred_npc[..., 4] > 0)
    true_alive = (true_npc[..., 5] >= .5) & (true_npc[..., 4] > 0)
    respawn_seen = np.zeros((count, ENTITY_SLOTS), dtype=bool)
    first_collision = [None]*count
    offsets = {}
    for offset in range(1, horizon+1):
        before, now = predicted[:, offset-1], predicted[:, offset]
        true_before, true_now = source[:, offset-1], source[:, offset]
        respawn = ~true_alive[:, offset-1] & true_alive[:, offset]
        respawn_seen |= respawn
        stable = true_alive[:, 0] & true_alive[:, offset-1] & true_alive[:, offset] & ~respawn_seen
        common = stable & pred_alive[:, offset]
        npc_distance = np.linalg.norm((pred_npc[:, offset, :, :2]-true_npc[:, offset, :, :2])*16, axis=-1)
        pdelta, tdelta = (now[:, :2]-before[:, :2])*16, (true_now[:, :2]-true_before[:, :2])*16
        ndelta = (pred_npc[:, offset, :, :2]-pred_npc[:, offset-1, :, :2])*16
        ntarget = (true_npc[:, offset, :, :2]-true_npc[:, offset-1, :, :2])*16
        heading_denominator = np.linalg.norm(now[:, 2:4], axis=-1)*np.linalg.norm(true_now[:, 2:4], axis=-1)
        cosine = (now[:, 2:4]*true_now[:, 2:4]).sum(-1)/np.maximum(heading_denominator, 1e-12)
        player_collisions = npc_collisions = 0
        for row in range(count):
            collisions = []
            if not circle_is_clear(tilemaps[row], now[row, :2]*16, PLAYER_RADIUS):
                player_collisions += 1; collisions.append('player')
            for slot in np.flatnonzero(pred_alive[row, offset]):
                if not circle_is_clear(tilemaps[row], pred_npc[row, offset, slot, :2]*16, NPC_RADIUS):
                    npc_collisions += 1; collisions.append(f'npc{slot}')
            if collisions and first_collision[row] is None:
                first_collision[row] = {'offset': offset, 'participants': collisions}
        offsets[str(offset)] = {
            'player_xy_error_tiles': _stats(np.linalg.norm((now[:, :2]-true_now[:, :2])*16, axis=-1)),
            'player_heading_error_degrees': _stats(np.degrees(np.arccos(np.clip(cosine, -1, 1)))),
            'player_displacement_error_tiles': _stats(np.linalg.norm(pdelta-tdelta, axis=-1)),
            'player_velocity_error_tiles_s': _stats(np.linalg.norm(pdelta-tdelta, axis=-1)/dt),
            'player_speed_tiles_s': _stats(np.linalg.norm(pdelta, axis=-1)/dt),
            'reference_player_speed_tiles_s': _stats(np.linalg.norm(tdelta, axis=-1)/dt),
            'npc_xy_error_tiles': _stats(npc_distance[stable]),
            'npc_common_alive_xy_error_tiles': _stats(npc_distance[common]),
            'npc_displacement_error_tiles': _stats(np.linalg.norm(ndelta-ntarget, axis=-1)[stable]),
            'npc_velocity_error_tiles_s': _stats((np.linalg.norm(ndelta-ntarget, axis=-1)/dt[:, None])[stable]),
            'npc_velocity_field_error_tiles_s': _stats(np.linalg.norm(
                2*(pred_npc[:, offset, :, 2:4]-true_npc[:, offset, :, 2:4]), axis=-1)[stable]),
            'npc_speed_tiles_s': _stats(np.linalg.norm(2*pred_npc[:, offset, :, 2:4], axis=-1)[stable]),
            'reference_npc_speed_tiles_s': _stats(np.linalg.norm(2*true_npc[:, offset, :, 2:4], axis=-1)[stable]),
            'npc_alive_false_negative': int((true_alive[:, offset] & ~pred_alive[:, offset]).sum()),
            'npc_alive_false_positive': int((~true_alive[:, offset] & pred_alive[:, offset]).sum()),
            'npc_death_event_mismatches': int(((pred_alive[:, offset-1] & ~pred_alive[:, offset]) !=
                                             (true_alive[:, offset-1] & ~true_alive[:, offset])).sum()),
            'player_alive_mismatches': int(((now[:, 4] > 0) != (true_now[:, 4] > 0)).sum()),
            'player_death_event_mismatches': int((((before[:, 4] > 0) & (now[:, 4] <= 0)) !=
                                                 ((true_before[:, 4] > 0) & (true_now[:, 4] <= 0))).sum()),
            'source_respawns_this_offset': int(respawn.sum()), 'source_respawned_slots_excluded': int(respawn_seen.sum()),
            'predicted_respawns_this_offset': int((~pred_alive[:, offset-1] & pred_alive[:, offset]).sum()),
            'player_wall_collisions': player_collisions, 'living_npc_wall_collisions': npc_collisions}
    report = {'panel': specification, 'fingerprint': fingerprint, 'source_held_out': True,
        'source_future_feedback': False, 'future_rgb_encoded': False, 'count': count,
        'horizon': horizon, 'context': context, 'render_precision': 'fp32_tf32_disabled',
        'all_states_finite': bool(np.isfinite(predicted).all()), 'all_memory_finite_and_shape_valid': True,
        'memory_shape_per_sample': list(memory_shape), 'elapsed_seconds': time.perf_counter()-started,
        'npc_geometry_mask': 'source initially alive and still continuing, no source respawn since context; own missing retained',
        'offsets': offsets, 'key_offsets': {str(k): offsets[str(k)] for k in (1, 4, 8, 16) if k <= horizon},
        'first_wall_collision': [{'episode': window['episode'], 'start': window['start'], 'collision': collision}
                                 for window, collision in zip(windows, first_collision)],
        'windows_with_wall_collision': sum(value is not None for value in first_collision)}
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'heldout_trajectory_states.npz', predicted=predicted, reference=source,
                        actions=actions, maps=tilemaps, indices=np.asarray(panel), dt=dt)
    (output/'heldout_trajectory_validation.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return report
