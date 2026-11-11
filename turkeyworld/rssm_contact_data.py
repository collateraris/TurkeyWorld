"""Immutable offline Source contact branches with distinct train/held-out poses.

The engine is used only while collecting labels. Neural poses must come from
explicitly TRAIN rollouts and survive an unmodified Source state round-trip.
No trajectory is repaired, reset or supplied with neural predictions as labels.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .engine import BEHAVIOR_VERSION, Game, STATE_DIM
from .navigation_recovery_data import digest, fingerprint, source_axis_labels
from .prepare_training_data import _atomic_json


VERSION = 1
PROFILE = 'navigation_no_enemies_v1'
CLEARANCES = (.005, .02, .04, .08, .16, .30)
CASE_NAMES = ('free', 'blocked', 'slide', 'turn', 'idle', 'partial')
BRANCH_NAMES = ('forward', 'backward', 'strafe_right', 'strafe_left',
                'diagonal_alternating', 'move_turn_positive',
                'move_turn_negative', 'back_away_idle_reapproach')


def action_for(branch, step, *, long=False, episode=0):
    if long:
        branch, step = (branch+step//32) % 8, step % 32
    controls = ((1, 0, 0, 0), (-1, 0, 0, 0), (0, 1, 0, 0), (0, -1, 0, 0),
                (1, 1 if step < 16 else -1, 0, 0), (.7, .35, .75, 0),
                (.7, -.35, -.75, 0))
    if branch < 7:
        return np.asarray(controls[branch], np.float32)
    # A genuine reverse/idle/return sequence inside every short clip.
    return np.asarray((-1, 0, 0, 0) if step < 8 else
                      (1, 0, 0, 0) if 12 <= step < 28 else (0, 0, 0, 0), np.float32)


def contact_cases(states, actions):
    command, fractions, active, audit = source_axis_labels(states, actions)
    free_axis = active & (fractions >= .9)
    blocked_axis = active & (fractions <= .1)
    moving = active.any(-1)
    cases = np.stack((moving & ((~active) | free_axis).all(-1),
        moving & ((~active) | blocked_axis).all(-1),
        free_axis.any(-1) & blocked_axis.any(-1),
        np.abs(actions[:, 2]) > 1e-8,
        (np.abs(actions[:, :3]).sum(-1) == 0),
        (active & (fractions > .1) & (fractions < .9)).any(-1)), -1)
    audit.update({name+'_steps': int(cases[:, index].sum()) for index, name in enumerate(CASE_NAMES)})
    return command, fractions, active, cases, audit


def first16_cohorts(cases):
    """Membership and dominant group use Source future labels, not NN input."""
    counts = {name: int(cases[:16, index].sum()) for index, name in enumerate(CASE_NAMES)}
    members = [name for name in ('free', 'blocked', 'slide', 'partial') if counts[name]]
    if counts['turn'] or counts['idle']:
        members.append('turn_idle')
    dominant = next((name for name in ('slide', 'partial', 'blocked', 'turn_idle', 'free')
                     if name in members), 'turn_idle')
    return dominant, members, counts


def _metadata_scalar(source, name):
    if name not in source:
        return None
    value = source[name]
    if value.shape != ():
        raise ValueError('Own rollout provenance '+name+' must be a scalar')
    return str(value.item())


def _source_pose(game, vector):
    value = np.asarray(vector, np.float32)
    if (value.shape != (STATE_DIM,) or not np.isfinite(value).all()
            or value[4] <= 0 or value[12] > 0 or value[13] > 0):
        raise ValueError('Own pose must be finite, alive and have no active enemy')
    game.set_state_vector(value)
    if not np.allclose(game.state_vector(), value, atol=2e-6, rtol=0):
        raise ValueError('Source would clip or repair the supplied initial state')
    return game.state_vector().copy()


def valid_own_poses(path):
    """Read TRAIN predictions[B,T,20] only; reject invalid states, never repair."""
    path = Path(path).resolve(strict=True)
    with np.load(path, allow_pickle=False) as source:
        panel_text = _metadata_scalar(source, 'panel')
        panel = json.loads(panel_text) if panel_text is not None else {}
        if not isinstance(panel, dict):
            raise ValueError('Own rollout panel provenance must be a JSON object')
        declarations = [_metadata_scalar(source, name) for name in ('split', 'source_split')]
        if 'split' in panel:
            declarations.append(panel['split'])
        declarations = [value for value in declarations if value is not None]
        if not declarations or any(value != 'train' for value in declarations):
            raise ValueError('Own contact poses require explicit, consistent TRAIN provenance; held-out inputs are forbidden')
        key = 'predicted' if 'predicted' in source else 'predictions'
        values = np.asarray(source[key], np.float32)
        if values.ndim not in (2, 3) or values.shape[-1] != STATE_DIM:
            raise ValueError('Own predictions must have shape[N,20] or[B,T,20]')
        values = values.reshape(-1, STATE_DIM)
    game = Game(seed=0, endless=True, enemies_enabled=False)
    poses, accepted, rejected, duplicates, seen = [], 0, 0, 0, set()
    for offset, vector in enumerate(values):
        try:
            pose = _source_pose(game, vector)
        except (ValueError, TypeError):
            rejected += 1
            continue
        accepted += 1
        # Time-only changes in a stopped trajectory are not distinct poses.
        key = tuple(np.round(pose[:2]*16, 4))
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        poses.append({'state': pose.tolist(), 'kind': 'own_valid_train',
                      'own_offset': offset, 'own_source': str(path)})
    return poses, {'path': str(path), 'sha256': digest(path), 'panel': panel,
        'source_split': 'train', 'accepted_states': accepted, 'rejected_states': rejected,
        'duplicate_xy_states': duplicates, 'distinct_xy_poses': len(poses)}


def _excluded_positions(root):
    if root is None:
        return {'train': [], 'val': []}
    root = Path(root)
    plan = json.loads((root/'plan.json').read_text(encoding='utf-8'))
    positions = {'train': [], 'val': []}
    for name, pose in plan['poses'].items():
        split = next((part for part in name.split('/') if part in positions), None)
        if split is None:
            raise ValueError('Excluded data plan has a pose with no explicit split')
        positions[split].append(np.asarray(pose['state'][:2], float)*16)
    return positions


def _clear_of(point, forbidden, distance):
    return not forbidden or bool((np.linalg.norm(np.asarray(forbidden)-point, axis=-1) >= distance-1e-7).all())


def _sample_pose(game, rng, ordinal, forbidden, radius, used_xy):
    floor = np.argwhere(game.map == 0)
    clearance = CLEARANCES[ordinal % len(CLEARANCES)]
    wanted_normal = (ordinal//len(CLEARANCES)) % 4
    normals = np.asarray(((1, 0), (-1, 0), (0, 1), (0, -1)))
    for _ in range(20000):
        row, column = floor[int(rng.integers(len(floor)))]
        toward = normals[wanted_normal]
        if not game.map[row+toward[1], column+toward[0]]:
            continue
        tangent = np.asarray((-toward[1], toward[0]))
        point = np.asarray((column+.5, row+.5))+toward*(.5-game.PLAYER_RADIUS-clearance)
        point += tangent*float(rng.uniform(-.25, .25))
        kind = 'wall_clearance'
        if ordinal % 5 == 0:
            candidates = [normal for normal in normals if abs(normal @ toward) == 0
                          and game.map[row+normal[1], column+normal[0]]]
            if candidates:
                other = candidates[int(rng.integers(len(candidates)))]
                point = np.asarray((column+.5, row+.5))+toward*(.5-game.PLAYER_RADIUS-clearance)
                point += other*(.5-game.PLAYER_RADIUS-CLEARANCES[(ordinal+1) % len(CLEARANCES)])
                kind = 'corner_clearance'
        xy_key = tuple(np.round(point, 4))
        if xy_key in used_xy or not _clear_of(point, forbidden, radius):
            continue
        if not game._walkable(*point, game.PLAYER_RADIUS):
            continue
        heading_offset = (0., math.pi/4, -math.pi/4, math.pi, math.pi/2, -math.pi/2)[(ordinal//24) % 6]
        # Keep an exact axis-aligned subset: otherwise tiny tangential motion
        # turns every straight wall stop into a two-axis slide cohort.
        jitter = 0. if ordinal % 3 == 0 else float(rng.uniform(-.04, .04))
        angle = math.atan2(toward[1], toward[0])+heading_offset+jitter
        saved = game.get_state()
        saved['player'].update(x=float(point[0]), y=float(point[1]), angle=angle)
        game.set_state(saved)
        used_xy.add(xy_key)
        return {'state': game.state_vector().tolist(), 'kind': kind,
            'requested_wall_clearance_tiles': clearance, 'wall_normal': toward.tolist(),
            'heading_offset_radians': heading_offset}
    raise RuntimeError('Could not find a legal contact pose outside held-out neighborhoods; use a new seed or smaller requested corpus')


def create_plan(config):
    own, own_reports = [], []
    for path in config['own_trajectories']:
        poses, report = valid_own_poses(path)
        own.extend(poses)
        own_reports.append(report)
    excluded = _excluded_positions(config['exclude_data_root'])
    game = Game(seed=0, endless=True, enemies_enabled=False)
    default = {'state': game.state_vector().tolist(), 'kind': 'default_spawn_train'}
    default_xy = np.asarray(default['state'][:2])*16
    own_xy = [np.asarray(pose['state'][:2])*16 for pose in own]
    radius = config['exclusion_radius']
    poses, jobs, points = {}, [], {'train': [], 'val': []}
    own_used, used_xy = 0, set()
    for split in ('val', 'train'):
        rng = np.random.default_rng(config['seed']+(1_000_000 if split == 'val' else 0))
        forbidden = (excluded['train']+own_xy+[default_xy] if split == 'val'
                     else excluded['val']+points['val'])
        ordinal = 0
        for stage in ('short', 'long'):
            count = config[stage+'_'+split]
            groups = math.ceil(count/8) if stage == 'short' else count
            for group in range(groups):
                pose_id = f'{stage}/{split}/pose_{group:05d}'
                pose = None
                if split == 'train' and stage == 'short' and group == 0 and _clear_of(default_xy, forbidden, radius):
                    pose = default
                elif split == 'train' and own and group % 4 == 1:
                    for candidate in rng.permutation(len(own)):
                        option = own[int(candidate)]
                        xy = np.asarray(option['state'][:2])*16
                        key = tuple(np.round(xy, 4))
                        if key not in used_xy and _clear_of(xy, forbidden, radius):
                            pose = option
                            own_used += 1
                            break
                if pose is None:
                    pose = _sample_pose(game, rng, ordinal, forbidden, radius, used_xy)
                xy = np.asarray(pose['state'][:2])*16
                used_xy.add(tuple(np.round(xy, 4)))
                points[split].append(xy)
                poses[pose_id] = pose
                ordinal += 1
                for branch in range(8 if stage == 'short' else 1):
                    index = group*8+branch if stage == 'short' else group
                    if index >= count:
                        break
                    episode_seed = config['seed']+(1_000_000 if split == 'val' else 0)+(100_000 if stage == 'long' else 0)+index
                    jobs.append({'stage': stage, 'split': split, 'index': index, 'seed': episode_seed,
                        'pose_id': pose_id, 'branch': branch if stage == 'short' else group % 8,
                        'frames': 32 if stage == 'short' else 451,
                        'relative_path': f'{stage}/rgb/{split}/episode_{episode_seed:010d}.npz'})
    separation = min((float(np.linalg.norm(a-b)) for a in points['train'] for b in points['val']), default=None)
    if separation is not None and separation < radius-1e-5:
        raise AssertionError('Contact train/validation neighborhoods overlap')
    return {'config': config, 'fingerprint': fingerprint(config), 'poses': poses, 'jobs': jobs,
        'own_pose_audit': own_reports, 'own_training_poses_used': own_used,
        'train_val_initial_pose_overlap': 0, 'train_val_min_initial_xy_distance_tiles': separation,
        'exclusion_audit': {'old_train_poses': len(excluded['train']), 'old_val_poses': len(excluded['val']),
            'radius_tiles': radius, 'old_validation_files_modified': False}}


def record_episode(job, pose):
    game = Game(seed=job['seed'], endless=True, enemies_enabled=False)
    _source_pose(game, pose['state'])
    initial = json.dumps(game.get_state())
    states, frames, actions, events = [game.state_vector().copy()], [game.render(size=32, overlays=False)], [], []
    for step in range(job['frames']):
        action = action_for(job['branch'], step, long=job['stage'] == 'long', episode=job['index'])
        result = game.step(action, dt=1/30)
        if game.done:
            raise RuntimeError('Source no-enemy contact episode terminated; no repair is allowed')
        states.append(game.state_vector().copy())
        frames.append(game.render(size=32, overlays=False))
        actions.append(action)
        events.append(Game.event_vector(result))
    states, frames, actions = np.asarray(states, np.float32), np.asarray(frames, np.uint8), np.asarray(actions, np.float32)
    command, fractions, active, cases, audit = contact_cases(states, actions)
    cohort, members, _ = first16_cohorts(cases)
    starts = np.arange(max(0, len(actions)-16+1), dtype=np.int32)
    windows = np.asarray([cases[start:start+16].any(0) for start in starts], bool)
    return {'map': game.map.astype(np.int8), 'states': states, 'frames': frames, 'actions': actions,
        'events': np.asarray(events, np.float32), 'hit_ids': np.full(len(actions), -1, np.int64),
        'attack_flags': np.zeros((len(actions), 1), np.float32), 'respawn_flags': np.zeros((len(actions), 1), np.float32),
        'object_masks': np.zeros(frames.shape[:3], bool), 'background_frames': frames.copy(),
        'initial_state': np.asarray(initial), 'axis_command_tiles': command,
        'axis_pass_targets': fractions, 'axis_command_active': active,
        'contact_case_names': np.asarray(CASE_NAMES), 'contact_cases': cases,
        'contact_first16_cohort': np.asarray(cohort), 'contact_sample_cohorts': np.asarray(members),
        'contact_window_starts_h16': starts, 'contact_window_cases_h16': windows}, audit


def _replace(source, destination):
    for attempt in range(101):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 100:
                raise
            time.sleep(.02)


def _job(arguments):
    root, pose, job, identity = arguments
    path = Path(root)/job['relative_path']
    # Source serializes heading through atan2/sin/cos. Near a cardinal angle,
    # its second component can change by ~1e-17 on this legal round-trip.
    # Keep the immutable plan, and require an EXACT match to Source's own
    # canonical initial vector rather than broadly tolerating altered files.
    expected_initial = _source_pose(Game(seed=job['seed'], endless=True, enemies_enabled=False), pose['state'])
    reused = path.exists()
    if not reused:
        payload, _ = record_episode(job, pose)
        payload.update(format_version=np.asarray(4), navigation_recovery_version=np.asarray(1),
            navigation_recovery_fingerprint=np.asarray(identity), rssm_contact_version=np.asarray(VERSION),
            behavior_version=np.asarray(BEHAVIOR_VERSION), environment_profile=np.asarray(PROFILE),
            enemies_enabled=np.asarray(False), state_dim=np.asarray(STATE_DIM), entity_slots=np.asarray(1),
            state_schema_version=np.asarray('single_enemy_v1'), split=np.asarray(job['split']),
            stage=np.asarray('navigation'), movement_case=np.asarray(pose['kind']),
            dataset_kind=np.asarray('world_transitions'), world_transition_compatible=np.asarray(True),
            dt=np.asarray(1/30), endless=np.asarray(True), seed=np.asarray(job['seed']),
            pose_id=np.asarray(job['pose_id']), action_branch=np.asarray(job['branch']),
            action_branch_name=np.asarray(BRANCH_NAMES[job['branch']]), initialization_source=np.asarray(pose['kind']),
            source_frame_count=np.asarray(job['frames']+1), frame_source=np.asarray('reference_engine'),
            state_repairs=np.asarray(0), resets_inside_trajectory=np.asarray(0))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
        try:
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **payload)
            _replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    with np.load(path, allow_pickle=False) as saved:
        if (str(saved['navigation_recovery_fingerprint']) != identity
                or str(saved['split']) != job['split'] or str(saved['pose_id']) != job['pose_id']
                or saved['states'].shape != (job['frames']+1, STATE_DIM)
                or saved['actions'].shape != (job['frames'], 4)
                or str(saved['behavior_version']) != BEHAVIOR_VERSION
                or not np.array_equal(saved['states'][0], expected_initial)):
            raise ValueError('Existing contact episode differs from its immutable plan')
        _, _, _, cases, audit = contact_cases(saved['states'], saved['actions'])
        if not np.array_equal(cases, saved['contact_cases']):
            raise ValueError('Existing contact labels differ from genuine Source movement')
        cohort, members, first_counts = first16_cohorts(cases)
    return {**job, 'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size,
        'reused': reused, 'audit': audit, 'initialization_source': pose['kind'],
        'cohort': cohort, 'sample_cohorts': members, 'first16_case_counts': first_counts}


def collect(root, *, short_train=2048, short_val=256, long_train=0, long_val=0,
            workers=4, seed=140_000_000, own_trajectories=(), exclude_data_root=None, exclusion_radius=.25):
    root = Path(root).resolve()
    counts = (short_train, short_val, long_train, long_val)
    if (any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts)
            or not isinstance(workers, int) or isinstance(workers, bool) or workers < 1
            or not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
            or not math.isfinite(exclusion_radius) or exclusion_radius <= 0):
        raise ValueError('Contact counts/seed must be nonnegative integers, workers positive and exclusion radius positive')
    own_paths = [Path(path).resolve(strict=True) for path in own_trajectories]
    excluded = Path(exclude_data_root).resolve(strict=True) if exclude_data_root else None
    config = {'version': VERSION, 'behavior_version': BEHAVIOR_VERSION, 'profile': PROFILE,
        'seed': seed, 'short_train': short_train, 'short_val': short_val,
        'long_train': long_train, 'long_val': long_val, 'size': 32,
        'own_trajectories': [str(path) for path in own_paths],
        'own_sha256': [digest(path) for path in own_paths],
        'exclude_data_root': str(excluded) if excluded else None,
        'excluded_plan_sha256': digest(excluded/'plan.json') if excluded else None,
        'exclusion_radius': float(exclusion_radius), 'clearances_tiles': list(CLEARANCES),
        'branches': list(BRANCH_NAMES), 'contact_cases': list(CASE_NAMES)}
    plan_path = root/'plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        if plan['config'] != config:
            raise ValueError('Existing contact plan is immutable; use a new root')
    else:
        plan = create_plan(config)
        root.mkdir(parents=True, exist_ok=True)
        _atomic_json(plan_path, plan)
    started, records = time.perf_counter(), []
    tasks = [(str(root), plan['poses'][job['pose_id']], job, plan['fingerprint']) for job in plan['jobs']]
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        iterator = executor.map(_job, tasks) if executor else map(_job, tasks)
        for record in iterator:
            records.append(record)
            elapsed = time.perf_counter()-started
            _atomic_json(root/'status.json', {'status': 'running', 'completed': len(records), 'total': len(tasks),
                'elapsed_seconds': elapsed, 'eta_seconds': elapsed/len(records)*(len(tasks)-len(records)),
                'last_file': record['relative_path'], 'fingerprint': plan['fingerprint']})
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
    audit_keys = set().union(*(record['audit'].keys() for record in records)) if records else set()
    groups = {}
    for stage in ('short', 'long'):
        for split in ('train', 'val'):
            selected = [record for record in records if record['stage'] == stage and record['split'] == split]
            pose_ids = {record['pose_id'] for record in selected}
            groups[stage+'/'+split] = {'clips': len(selected), 'distinct_initial_poses': len(pose_ids),
                'transitions': sum(record['frames'] for record in selected),
                'cohort_counts': {name: sum(record['cohort'] == name for record in selected)
                    for name in ('free', 'blocked', 'slide', 'turn_idle', 'partial')},
                'sample_cohort_counts': {name: sum(name in record['sample_cohorts'] for record in selected)
                    for name in ('free', 'blocked', 'slide', 'turn_idle', 'partial')},
                'audit': {key: sum(record['audit'].get(key, 0) for record in selected) for key in sorted(audit_keys)}}
    manifest = {key: value for key, value in plan.items() if key not in ('poses', 'jobs')}
    manifest.update(status='complete', records=records, completed=len(records), total=len(tasks), groups=groups,
        audit={key: sum(record['audit'].get(key, 0) for record in records) for key in sorted(audit_keys)},
        elapsed_seconds=time.perf_counter()-started,
        source_labels='continuous offline Source32; no enemy; no state repair or within-episode reset',
        heldout_policy='Initial XY neighborhoods separated; existing held-out data unchanged; own poses TRAIN only')
    _atomic_json(root/'manifest.json', manifest)
    _atomic_json(root/'status.json', {key: value for key, value in manifest.items() if key != 'records'})
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--own-trajectory', type=Path, action='append', default=[])
    parser.add_argument('--exclude-data-root', type=Path)
    parser.add_argument('--exclusion-radius', type=float, default=.25)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=140_000_000)
    for flag, default in (('short-train', 2048), ('short-val', 256), ('long-train', 0), ('long-val', 0)):
        parser.add_argument('--'+flag, type=int, default=default)
    args = parser.parse_args(argv)
    result = collect(args.root, short_train=args.short_train, short_val=args.short_val,
        long_train=args.long_train, long_val=args.long_val, workers=args.workers, seed=args.seed,
        own_trajectories=args.own_trajectory, exclude_data_root=args.exclude_data_root,
        exclusion_radius=args.exclusion_radius)
    print(json.dumps({key: value for key, value in result.items() if key != 'records'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
