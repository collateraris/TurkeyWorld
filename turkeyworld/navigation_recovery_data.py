"""Immutable recovery episodes from the ordinary no-enemy Source environment.

Counterfactual means separate continuous Source branches sharing one valid
initial pose. Nothing is reset, clipped or repaired within an episode. Valid
neural poses may initialize TRAIN branches only; invalid poses are rejected.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .engine import BEHAVIOR_VERSION, Game, STATE_DIM
from .prepare_training_data import _atomic_json


VERSION = 1
PROFILE = 'navigation_no_enemies_v1'
ACTION_NAMES = ('idle', 'forward', 'backward', 'strafe_right', 'strafe_left',
                'diagonal_forward', 'diagonal_back', 'turn_right', 'turn_left',
                'forward_turn_right', 'back_turn_left', 'strafe_turn_right',
                'strafe_turn_left', 'diagonal_turn_left', 'diagonal_turn_right', 'out_back')
ACTIONS = np.asarray(((0, 0, 0, 0), (1, 0, 0, 0), (-1, 0, 0, 0),
    (0, 1, 0, 0), (0, -1, 0, 0), (1, 1, 0, 0), (-1, -1, 0, 0),
    (0, 0, .75, 0), (0, 0, -.75, 0), (.7, 0, .75, 0), (-.7, 0, -.75, 0),
    (0, .7, .6, 0), (0, -.7, -.6, 0), (.7, -.5, -.5, 0), (-.7, .5, .5, 0), (1, 0, 0, 0)), np.float32)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def action_for(branch, step, *, long=False, episode=0):
    index = (step//30+episode) % len(ACTIONS) if long else branch
    if index == 15:
        return ACTIONS[1 if step % 32 < 16 else 2].copy()
    return ACTIONS[index].copy()


def source_axis_labels(states, actions, dt=1/30):
    """Audit-only intended and realized worldXY motion; never model inputs."""
    previous, current = np.asarray(states[:-1], float), np.asarray(states[1:], float)
    controls = np.array(actions[:, :2], dtype=float, copy=True)
    controls /= np.maximum(1, np.linalg.norm(controls, axis=-1, keepdims=True))
    angle = np.arctan2(previous[:, 2], previous[:, 3])+np.asarray(actions[:, 2])*2.2*dt
    command = np.stack([np.cos(angle)*controls[:, 0]-np.sin(angle)*controls[:, 1],
                        np.sin(angle)*controls[:, 0]+np.cos(angle)*controls[:, 1]], -1)*2.4*dt
    actual = (current[:, :2]-previous[:, :2])*16
    active = np.abs(command) > 1e-5
    fraction = np.divide(actual, command, out=np.zeros_like(actual), where=active)
    free, blocked = active & (fraction >= .95), active & (np.abs(fraction) <= .05)
    partial = active & ~free & ~blocked
    return command.astype(np.float32), fraction.astype(np.float32), active, {
        'free_axes': int(free.sum()), 'blocked_axes': int(blocked.sum()), 'partial_axes': int(partial.sum()),
        'active_axes': int(active.sum()), 'moving_turn_steps': int(((np.linalg.norm(actions[:, :2], axis=-1) > 0)
                                                                  & (np.abs(actions[:, 2]) > 0)).sum()),
        'negative_turn_steps': int((actions[:, 2] < 0).sum()), 'positive_turn_steps': int((actions[:, 2] > 0).sum()),
        'idle_steps': int((np.abs(actions[:, :3]).sum(-1) == 0).sum())}


def valid_own_poses(path):
    if path is None:
        return [], {'path': None, 'accepted': 0, 'rejected': 0}
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as source:
        values = source['predicted'].copy()
    values = values.reshape(-1, STATE_DIM)
    game = Game(seed=0, endless=True, enemies_enabled=False)
    accepted, rejected = [], 0
    for offset, vector in enumerate(values):
        try:
            game.set_state_vector(vector)
            if not np.allclose(game.state_vector(), vector, atol=2e-5, rtol=0):
                raise ValueError('State adapter would clip or repair this pose')
        except (ValueError, TypeError):
            rejected += 1
            continue
        accepted.append({'state': vector.tolist(), 'kind': 'own_valid', 'own_offset': offset})
    return accepted, {'path': str(path), 'sha256': digest(path), 'accepted': len(accepted), 'rejected': rejected}


def _sample_pose(game, rng, kind):
    floor = np.argwhere(game.map == 0)
    for _ in range(3000):
        row, column = floor[int(rng.integers(len(floor)))]
        point = np.asarray([column+.5, row+.5], float)
        walls = [np.asarray([dx, dy], float) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                 if game.map[row+dy, column+dx]]
        if kind in ('wall', 'slide', 'corner'):
            if not walls:
                continue
            toward = walls[int(rng.integers(len(walls)))]
            point += toward*.29
            if kind == 'corner':
                other = next((value for value in walls if abs(value @ toward) < .1), None)
                if other is None:
                    continue
                point += other*.29
            angle = math.atan2(toward[1], toward[0])+(math.pi/4 if kind == 'slide' else 0)+float(rng.uniform(-.12, .12))
        else:
            point += rng.uniform(-.18, .18, 2)
            if not all(game._walkable(*(point+offset), game.PLAYER_RADIUS) for offset in
                       ((0, 0), (.9, 0), (-.9, 0), (0, .9), (0, -.9))):
                continue
            angle = float(rng.uniform(-math.pi, math.pi))
        if not game._walkable(*point, game.PLAYER_RADIUS):
            continue
        state = game.get_state()
        state['player'].update(x=float(point[0]), y=float(point[1]), angle=angle)
        game.set_state(state)
        return {'state': game.state_vector().tolist(), 'kind': kind}
    raise RuntimeError(f'Could not initialize a legal Source {kind} pose')


def create_plan(config):
    own, own_report = valid_own_poses(config.get('own_trajectory'))
    game = Game(seed=0, endless=True, enemies_enabled=False)
    poses, jobs = {}, []
    for stage in ('short', 'long'):
        for split in ('train', 'val'):
            count = config[f'{stage}_{split}']
            groups = math.ceil(count/16) if stage == 'short' else count
            rng = np.random.default_rng(config['seed']+(0 if split == 'train' else 1_000_000)+(0 if stage == 'short' else 100_000))
            for group in range(groups):
                pose_id = f'{stage}/{split}/pose_{group:05d}'
                if split == 'train' and group == 0:
                    default = Game(seed=0, endless=True, enemies_enabled=False)
                    pose = {'state': default.state_vector().tolist(), 'kind': 'default_spawn'}
                elif split == 'train' and own and group % 4 == 1:
                    pose = own[int(rng.integers(len(own)))].copy()
                else:
                    pose = _sample_pose(game, rng, ('free', 'wall', 'slide', 'corner')[group % 4])
                poses[pose_id] = pose
                for branch in range(16 if stage == 'short' else 1):
                    index = group*16+branch if stage == 'short' else group
                    if index >= count:
                        break
                    seed = config['seed']+(0 if split == 'train' else 1_000_000)+(0 if stage == 'short' else 100_000)+index
                    jobs.append({'stage': stage, 'split': split, 'index': index, 'seed': seed, 'pose_id': pose_id,
                        'branch': branch, 'frames': 32 if stage == 'short' else 451,
                        'relative_path': f'{stage}/rgb/{split}/episode_{seed:010d}.npz'})
    keys = {split: {tuple(np.round(pose['state'][:4], 7)) for name, pose in poses.items() if f'/{split}/' in name}
            for split in ('train', 'val')}
    if keys['train'] & keys['val']:
        raise ValueError('Train/val initial pose groups overlap')
    return {'config': config, 'fingerprint': fingerprint(config), 'poses': poses, 'jobs': jobs,
            'own_pose_audit': own_report, 'train_val_initial_pose_overlap': 0}


def record_episode(job, pose):
    game = Game(seed=job['seed'], endless=True, enemies_enabled=False)
    game.set_state_vector(np.asarray(pose['state'], np.float32))
    if not np.allclose(game.state_vector(), pose['state'], atol=2e-5, rtol=0):
        raise ValueError('Initial Source state would require a repair')
    initial = json.dumps(game.get_state())
    states, frames, actions, events, hits, attacks = [], [], [], [], [], []
    def observe():
        states.append(game.state_vector().copy())
        frames.append(game.render(size=32, overlays=False))
    observe()
    for step in range(job['frames']):
        controls = action_for(job['branch'], step, long=job['stage'] == 'long', episode=job['index'])
        result = game.step(controls, dt=1/30)
        actions.append(controls)
        events.append(Game.event_vector(result))
        hits.append(result['hit_id'])
        attacks.append([float(bool(result['attacker_ids']))])
        observe()
        if game.done:
            raise RuntimeError('No-enemy Source terminated; no repairs are allowed')
    states, frames, actions = np.asarray(states, np.float32), np.asarray(frames, np.uint8), np.asarray(actions, np.float32)
    command, fractions, active, audit = source_axis_labels(states, actions)
    return {'map': game.map.astype(np.int8), 'states': states, 'frames': frames, 'actions': actions,
        'events': np.asarray(events, np.float32), 'hit_ids': np.asarray(hits, np.int64),
        'attack_flags': np.asarray(attacks, np.float32), 'respawn_flags': np.zeros((len(actions), 1), np.float32),
        'object_masks': np.zeros(frames.shape[:3], bool), 'background_frames': frames.copy(),
        'initial_state': np.asarray(initial), 'axis_command_tiles': command,
        'axis_pass_targets': fractions, 'axis_command_active': active}, audit


def _job(arguments):
    root, plan, job = arguments
    path = Path(root)/job['relative_path']
    reused = path.exists()
    if not reused:
        payload, audit = record_episode(job, plan['poses'][job['pose_id']])
        payload.update(format_version=np.asarray(4), navigation_recovery_version=np.asarray(VERSION),
            navigation_recovery_fingerprint=np.asarray(plan['fingerprint']), behavior_version=np.asarray(BEHAVIOR_VERSION),
            environment_profile=np.asarray(PROFILE), enemies_enabled=np.asarray(False), state_dim=np.asarray(20),
            state_schema_version=np.asarray('single_enemy_v1'), entity_slots=np.asarray(1),
            split=np.asarray(job['split']), stage=np.asarray('navigation'), movement_case=np.asarray(plan['poses'][job['pose_id']]['kind']),
            dataset_kind=np.asarray('world_transitions'), world_transition_compatible=np.asarray(True), dt=np.asarray(1/30),
            endless=np.asarray(True), seed=np.asarray(job['seed']), pose_id=np.asarray(job['pose_id']),
            action_branch=np.asarray(job['branch']), initialization_source=np.asarray(plan['poses'][job['pose_id']]['kind']),
            source_frame_count=np.asarray(job['frames']+1), frame_source=np.asarray('reference_engine'),
            state_repairs=np.asarray(0), resets_inside_trajectory=np.asarray(0))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **payload)
        os.replace(temporary, path)
    with np.load(path, allow_pickle=False) as saved:
        if (str(saved['navigation_recovery_fingerprint']) != plan['fingerprint'] or str(saved['split']) != job['split']
                or str(saved['pose_id']) != job['pose_id'] or saved['states'].shape != (job['frames']+1, 20)
                or saved['actions'].shape != (job['frames'], 4) or str(saved['behavior_version']) != BEHAVIOR_VERSION):
            raise ValueError('Existing recovery record differs from its immutable plan')
        _, _, _, audit = source_axis_labels(saved['states'], saved['actions'])
    return {**job, 'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size, 'reused': reused, 'audit': audit}


def collect(root, *, short_train=512, short_val=128, long_train=64, long_val=16,
            workers=1, seed=130_000_000, own_trajectory=None):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = {'version': VERSION, 'behavior_version': BEHAVIOR_VERSION, 'profile': PROFILE, 'seed': seed,
              'short_train': short_train, 'short_val': short_val, 'long_train': long_train, 'long_val': long_val,
              'own_trajectory': str(Path(own_trajectory).resolve()) if own_trajectory else None,
              'own_sha256': digest(own_trajectory) if own_trajectory else None, 'size': 32}
    if min(short_train, short_val, long_train, long_val) < 0 or workers < 1:
        raise ValueError('Counts must be nonnegative and workers positive')
    plan_path = root/'plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        if plan['config'] != config:
            raise ValueError('Existing recovery plan is immutable')
    else:
        plan = create_plan(config)
        _atomic_json(plan_path, plan)
    started, records = time.perf_counter(), []
    tasks = [(str(root), plan, job) for job in plan['jobs']]
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        iterator = executor.map(_job, tasks) if executor is not None else map(_job, tasks)
        for record in iterator:
            records.append(record)
            _atomic_json(root/'status.json', {'status': 'running', 'completed': len(records), 'total': len(tasks),
                'elapsed_seconds': time.perf_counter()-started, 'last_file': record['path']})
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    totals = {name: sum(record['audit'][name] for record in records) for name in
              ('free_axes', 'blocked_axes', 'partial_axes', 'active_axes', 'moving_turn_steps',
               'negative_turn_steps', 'positive_turn_steps', 'idle_steps')}
    manifest = {'status': 'complete', 'config': config, 'fingerprint': plan['fingerprint'], 'records': records,
        'completed': len(records), 'total': len(tasks), 'audit': totals,
        'train_val_initial_pose_overlap': 0, 'own_pose_audit': plan['own_pose_audit'],
        'elapsed_seconds': time.perf_counter()-started}
    _atomic_json(root/'manifest.json', manifest)
    _atomic_json(root/'status.json', {key: value for key, value in manifest.items() if key != 'records'})
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--own-trajectory', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=130_000_000)
    for flag, default in (('short-train', 512), ('short-val', 128), ('long-train', 64), ('long-val', 16)):
        parser.add_argument('--'+flag, type=int, default=default)
    args = parser.parse_args(argv)
    result = collect(args.root, short_train=args.short_train, short_val=args.short_val,
        long_train=args.long_train, long_val=args.long_val, workers=args.workers,
        seed=args.seed, own_trajectory=args.own_trajectory)
    print(json.dumps({key: value for key, value in result.items() if key != 'records'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
