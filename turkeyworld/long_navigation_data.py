"""Continuous navigation recordings in the explicit no-enemy Source profile.

Player physics are ordinary Game.step. One valid initialization precedes the
whole episode; no reset, state repair, clock edit or hidden actor suppression
occurs inside a trajectory. The reserved inactive NPC slot keeps state20.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .engine import BEHAVIOR_VERSION, ENTITY_SLOTS, STATE_DIM, Game
from .prepare_training_data import _atomic_json
from .staged_data import CASES, _safe_pose

ENVIRONMENT_PROFILE = 'navigation_no_enemies_v1'
DATA_VERSION = 1


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def action_for(case, index, leg, hold):
    action = np.zeros(4, np.float32)
    if case == 'turn':
        action[2] = .75
    elif case == 'slide':
        action[:2] = (1., .6*(1 if index//leg % 2 == 0 else -1))
    elif case == 'wall_stop':
        # Real approach/blocked hold, retreat, and approach again.
        position = index % (2*leg+hold)
        action[0] = 1. if position < leg+hold else -1.
    else:
        action[1 if case == 'strafe' else 0] = 1. if index//leg % 2 == 0 else -1.
    return action


def record_episode(seed, case, transitions=451, size=32):
    if case not in CASES or transitions < 1 or size != 32:
        raise ValueError('Require a known navigation case, positive transitions and native32')
    game = Game(seed=seed, endless=True, enemies_enabled=False)
    rng = np.random.default_rng(seed+173)
    leg = int(rng.integers(80, 121) if case == 'straight' else rng.integers(56, 91)
              if case == 'strafe' else rng.integers(24, 49))
    hold = int(rng.integers(25, 56))
    if case in ('wall_stop', 'slide', 'turn'):
        point, angle = _safe_pose(game, rng, case, 32)
    else:
        point, angle = _safe_pose(game, rng, 'short_return', 2*leg)
        if case == 'strafe': angle -= math.pi/2
    initial = game.get_state()
    initial['player'].update(x=float(point[0]), y=float(point[1]), angle=float(angle))
    game.set_state(initial)
    initial_json = json.dumps(game.get_state())
    if game.enemies_enabled or json.loads(initial_json).get('enemies_enabled') is not False:
        raise ValueError('Source engine did not retain its explicit no-enemy profile')
    states, images, actions, events, hits, attacks, respawns = [], [], [], [], [], [], []
    def observe():
        states.append(game.state_vector())
        images.append(game.render(size=32, overlays=False))
    observe()
    for index in range(transitions):
        action = action_for(case, index, leg, hold)
        before = np.asarray([enemy['alive'] for enemy in game.enemies], bool)
        event = game.step(action, dt=1/30)
        actions.append(action); events.append(Game.event_vector(event)); hits.append(event['hit_id'])
        flags = np.zeros(ENTITY_SLOTS, np.float32); flags[event['attacker_ids']] = 1.
        attacks.append(flags)
        respawns.append(~before & np.asarray([enemy['alive'] for enemy in game.enemies], bool))
        observe()
        if game.done:
            raise RuntimeError('No-enemy navigation terminated unexpectedly; no repairs are permitted')
    states, frames = np.asarray(states, np.float32), np.asarray(images, np.uint8)
    if states.shape != (transitions+1, STATE_DIM) or np.any(states[:, 12:14]):
        raise RuntimeError('Expected exact state20 and a genuinely inactive reserved NPC slot')
    return {'map': game.map.astype(np.int8), 'states': states, 'frames': frames,
        'actions': np.asarray(actions, np.float32), 'events': np.asarray(events, np.float32),
        'hit_ids': np.asarray(hits, np.int64), 'attack_flags': np.asarray(attacks, np.float32),
        'respawn_flags': np.asarray(respawns, np.float32),
        # In this documented environment there are no NPC pixels to remove.
        'object_masks': np.zeros(frames.shape[:3], bool), 'background_frames': frames.copy(),
        'initial_state': np.asarray(initial_json), 'initial_game_time': np.asarray(0.),
        'segment_steps': np.asarray(leg), 'wall_hold_steps': np.asarray(hold),
        'active_ids': np.zeros(0, np.int64)}


def _record(path, config, job, reused):
    with np.load(path, allow_pickle=False) as saved:
        if (str(saved['long_navigation_fingerprint']) != config['fingerprint']
                or str(saved['environment_profile']) != ENVIRONMENT_PROFILE
                or str(saved['behavior_version']) != BEHAVIOR_VERSION
                or str(saved['split']) != job['split'] or int(saved['seed']) != job['seed']
                or str(saved['movement_case']) != job['case'] or bool(saved['enemies_enabled'])
                or saved['states'].shape != (config['transitions']+1, STATE_DIM)
                or saved['actions'].shape != (config['transitions'], 4)
                or saved['attack_flags'].shape != (config['transitions'], ENTITY_SLOTS)
                or not np.isfinite(saved['states']).all() or np.any(saved['states'][:, 12:14])
                or saved['object_masks'].any() or saved['respawn_flags'].any()):
            raise ValueError('Existing long navigation source differs from the immutable plan')
        displacement = np.linalg.norm(np.diff(saved['states'][:, :2], axis=0)*16, axis=-1)
        controls = np.linalg.norm(saved['actions'][:, :2], axis=-1)
        blocked = int(((controls > .1) & (displacement < 1e-5)).sum())
    return {**job, 'file': str(path), 'sha256': file_hash(path), 'bytes': path.stat().st_size,
        'transitions': config['transitions'], 'observations': config['transitions']+1,
        'eligible_450': config['transitions'] >= 451, 'blocked_motion_steps': blocked, 'reused': reused}


def _job(arguments):
    root, config, job = arguments
    path = Path(root)/job['relative_path']
    if path.exists(): return _record(path, config, job, True)
    payload = record_episode(job['seed'], job['case'], config['transitions'])
    payload.update(format_version=np.asarray(4), long_navigation_data_version=np.asarray(DATA_VERSION),
        staged_data_version=np.asarray(2), behavior_version=np.asarray(BEHAVIOR_VERSION),
        environment_profile=np.asarray(ENVIRONMENT_PROFILE), enemies_enabled=np.asarray(False),
        state_schema_version=np.asarray('single_enemy_v1'), state_dim=np.asarray(STATE_DIM),
        entity_slots=np.asarray(ENTITY_SLOTS), split=np.asarray(job['split']), stage=np.asarray('navigation'),
        movement_case=np.asarray(job['case']), policy=np.asarray('long_'+job['case']),
        dataset_kind=np.asarray('world_transitions'), world_transition_compatible=np.asarray(True),
        dt=np.asarray(1/30), endless=np.asarray(True), seed=np.asarray(job['seed']),
        source_frame_count=np.asarray(config['transitions']+1), requested_transitions=np.asarray(config['transitions']),
        frame_source=np.asarray('reference_engine'), background_frame_source=np.asarray('reference_engine_no_enemy_profile'),
        initialization_source=np.asarray('seeded_valid_continuous_start'),
        engine_semantics_modified=np.asarray(False), player_physics_modified=np.asarray(False), ai_freeze=np.asarray(False),
        resets_inside_trajectory=np.asarray(0), state_repairs=np.asarray(0),
        long_navigation_fingerprint=np.asarray(config['fingerprint']), staged_config_fingerprint=np.asarray(config['fingerprint']))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream: np.savez_compressed(stream, **payload)
    os.replace(temporary, path)
    return _record(path, config, job, False)


def collect_long_navigation(root, train_clips=64, val_clips=16, *, transitions=451, seed=90_000_000, workers=4):
    counts = (train_clips, val_clips)
    if (any(not isinstance(n, int) or isinstance(n, bool) or not 0 <= n < 1_000_000 for n in counts)
            or sum(counts) < 1 or transitions < 1 or workers < 1 or seed < 0):
        raise ValueError('Invalid counts, episode length, seed or worker count')
    root = Path(root).resolve()
    config = {'version': DATA_VERSION, 'environment_profile': ENVIRONMENT_PROFILE,
        'enemies_enabled': False, 'behavior_version': BEHAVIOR_VERSION, 'state_dim': STATE_DIM,
        'entity_slots': ENTITY_SLOTS, 'train_clips': train_clips, 'val_clips': val_clips,
        'transitions': transitions, 'size': 32, 'hz': 30, 'seed': seed, 'cases': list(CASES)}
    config['fingerprint'] = fingerprint(config)
    manifest = root/'manifest.json'
    if root.exists() and any(root.iterdir()) and not manifest.exists():
        raise ValueError('Refusing a nonempty destination without its matching long navigation manifest')
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))['config'] != config:
        raise ValueError('Immutable long navigation configuration changed')
    root.mkdir(parents=True, exist_ok=True)
    jobs = [{'split': split, 'index': index, 'seed': seed+offset+index, 'case': CASES[index % len(CASES)],
             'relative_path': f'rgb/{split}/episode_{seed+offset+index:010d}.npz'}
            for split, count, offset in (('train', train_clips, 0), ('val', val_clips, 1_000_000)) for index in range(count)]
    records, started = [], time.perf_counter()
    _atomic_json(manifest, {'status': 'planned', 'config': config, 'jobs': jobs})
    def publish(status):
        result = {'status': status, 'config': config, 'completed_files': len(records), 'total_files': len(jobs),
            'elapsed_seconds': time.perf_counter()-started, 'world_transitions': len(records)*transitions,
            'groups': dict(Counter(row['split'] for row in records)),
            'updated_at': datetime.now(timezone.utc).isoformat()}
        _atomic_json(root/'status.json', result)
        print(json.dumps(result), flush=True)
        return result
    publish('running')
    try:
        arguments = [(str(root), config, job) for job in jobs]
        pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
        try:
            for row in pool.map(_job, arguments) if pool is not None else map(_job, arguments):
                records.append(row)
                with (root/'progress.jsonl').open('a', encoding='utf-8') as stream: stream.write(json.dumps(row)+'\n')
                publish('running')
        finally:
            if pool is not None: pool.shutdown(wait=True)
        result = publish('complete')
        _atomic_json(manifest, {**result, 'records': records,
            'source_profile': 'Explicit persistent no-enemy Game profile; normal player Game.step for every action',
            'resets_inside_trajectories': 0, 'state_repairs': 0,
            'content_fingerprint': fingerprint([{k: row[k] for k in ('relative_path', 'sha256')} for row in records])})
        return result
    except BaseException:
        publish('failed'); raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--train-clips', type=int, default=64)
    parser.add_argument('--val-clips', type=int, default=16)
    parser.add_argument('--transitions', type=int, default=451)
    parser.add_argument('--seed', type=int, default=90_000_000)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(argv)
    collect_long_navigation(**vars(args))


if __name__ == '__main__': main()
