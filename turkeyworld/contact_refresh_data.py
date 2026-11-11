"""Paired wall-relative Source labels from TRAIN neural poses and legal Source augmentation.

Existing corpora are read-only. Every newly recorded branch is a separate
continuous Source episode; invalid neural states are rejected, never repaired.
Replay files retain their original bytes, fingerprints and explicit provenance.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .engine import BEHAVIOR_VERSION, Game, STATE_DIM
from .navigation_recovery_data import digest, fingerprint
from .prepare_training_data import _atomic_json
from .rssm_contact_data import (PROFILE, CASE_NAMES, _clear_of, _excluded_positions,
    _replace, _sample_pose, _source_pose, contact_cases, first16_cohorts, valid_own_poses)


VERSION = 2
BRANCH_NAMES = ('approach_wall', 'tangent_positive', 'tangent_negative', 'away_from_wall',
    'diagonal_slide_positive', 'diagonal_slide_negative', 'approach_turn_positive',
    'approach_turn_negative', 'block_release_reapproach', 'idle_turn_return')
COHORTS = ('free', 'blocked', 'slide', 'turn_idle')
NORMALS = np.asarray(((1., 0.), (-1., 0.), (0., 1.), (0., -1.)))


def wall_pair_geometry(game, vector, *, free_probe_steps=8):
    """Offline map/Source checks, never an inference collision-correction API."""
    point = np.asarray(vector[:2], float)*16
    walls = np.argwhere(game.map != 0)[:, ::-1].astype(float)
    choices = []
    for normal in NORMALS:
        axis = int(abs(normal[1]))
        other = 1-axis
        lateral = np.maximum(np.maximum(walls[:, other]-point[other],
            point[other]-(walls[:, other]+1)), 0)
        valid = lateral < game.PLAYER_RADIUS
        extension = np.sqrt(np.maximum(game.PLAYER_RADIUS**2-lateral**2, 0))
        distance = (walls[:, axis]-extension-point[axis] if normal[axis] > 0
                    else point[axis]-(walls[:, axis]+1+extension))
        valid &= (distance >= -2e-6) & (distance <= .35)
        if valid.any():
            choices.append((float(distance[valid].min()), normal))
    for clearance, normal in sorted(choices, key=lambda pair: pair[0]):
        tangent = np.asarray((-normal[1], normal[0]))
        for amplitude in (1., .75, .5, .4):
            points = [point+direction*(.08*amplitude*step)
                      for direction in (tangent, -tangent, -normal)
                      for step in range(1, free_probe_steps+1)]
            if all(game._walkable(*value, game.PLAYER_RADIUS) for value in points):
                return {'wall_normal': normal.tolist(), 'wall_clearance_tiles': max(0., clearance),
                    'paired_movement_scale': amplitude, 'free_probe_steps': free_probe_steps,
                    'geometry_source': 'offline_static_map_and_Source_walkability'}
    return None


def action_for(branch, step, normal, angle):
    """World-relative direction -> ego controls AFTER the known signed turn."""
    normal = np.asarray(normal, float)
    tangent = np.asarray((-normal[1], normal[0]))
    turn = 0.
    if branch == 0:
        direction = normal
    elif branch == 1:
        direction = tangent
    elif branch == 2:
        direction = -tangent
    elif branch == 3:
        direction = -normal
    elif branch in (4, 5):
        direction = normal+(tangent if branch == 4 else -tangent)
        direction /= np.linalg.norm(direction)
    elif branch in (6, 7):
        sign = 1 if branch == 6 else -1
        direction, turn = normal+.35*sign*tangent, .75*sign
        direction /= np.linalg.norm(direction)
    elif branch == 8:
        direction = normal if step < 8 or 16 <= step < 24 else -normal if step < 16 else np.zeros(2)
    elif branch == 9:
        direction = np.zeros(2) if step < 4 or step >= 28 else -normal if step < 12 else normal if step < 20 else tangent
        turn = .6 if 4 <= step < 12 else -.6 if 12 <= step < 20 else 0.
    else:
        raise ValueError('Paired Source action branch must be in0..9')
    post_angle = angle+turn*2.2/30
    cosine, sine = math.cos(post_angle), math.sin(post_angle)
    forward = cosine*direction[0]+sine*direction[1]
    strafe = -sine*direction[0]+cosine*direction[1]
    return np.asarray((forward, strafe, turn, 0.), np.float32)


def _cohort(cases, actions):
    dominant, members, counts = first16_cohorts(cases)
    # Moving-turn counterexamples must remain available as a dedicated bucket.
    if np.abs(actions[:16, 2]).any() or counts['idle']:
        dominant = 'turn_idle'
    if dominant == 'partial':
        dominant = 'slide' if counts['slide'] else 'blocked' if counts['blocked'] else 'free'
    return dominant, members, counts


def _yaw_pose(game, pose, angle, kind):
    value = np.asarray(pose['state'], np.float32).copy()
    value[2:4] = (math.sin(angle), math.cos(angle))
    value = _source_pose(game, value)
    return {**pose, 'state': value.tolist(), 'kind': kind, 'source_yaw_counterfactual': True,
        'source_anchor_kind': pose['kind'], 'yaw_target_radians': angle}


def _own_inputs(path):
    poses, report = valid_own_poses(path)
    game = Game(seed=0, endless=True, enemies_enabled=False)
    with np.load(path, allow_pickle=False) as saved:
        key = 'maps' if 'maps' in saved else 'map' if 'map' in saved else None
        if key is not None and (saved[key].shape[-2:] != (16, 16) or not np.all(saved[key] == game.map)):
            raise ValueError('Own rollout map differs from the Source map used for labels')
    panel = report['panel']
    if any(panel.get(name, False) not in (False, 0) for name in
           ('future_source_feedback', 'engine_corrections', 'state_resets', 'memory_resets')):
        raise ValueError('Own TRAIN predictions must not contain Source feedback or resets')
    return poses, report


def create_plan(config):
    own, own_report = _own_inputs(config['own_trajectory'])
    game = Game(seed=0, endless=True, enemies_enabled=False)
    excluded = {'train': [], 'val': []}
    for root in config['exclude_data_roots']:
        current = _excluded_positions(root)
        for split in excluded:
            excluded[split].extend(current[split])
    source_root = Path(config['source_contact_root'])
    old_manifest = json.loads((source_root/'manifest.json').read_text(encoding='utf-8'))
    old_plan = json.loads((source_root/'plan.json').read_text(encoding='utf-8'))
    if old_manifest.get('status') != 'complete':
        raise ValueError('Replay Source contact corpus is incomplete')
    train_records = [record for record in old_manifest['records'] if record['stage'] == 'short' and record['split'] == 'train']
    rng = np.random.default_rng(config['seed'])
    buckets = {key: [] for key in COHORTS}
    for index in rng.permutation(len(train_records)):
        record = train_records[int(index)]
        path = (source_root/record['relative_path']).resolve(strict=True)
        if source_root not in path.parents:
            raise ValueError('Replay path escaped its Source corpus')
        point = np.asarray(old_plan['poses'][record['pose_id']]['state'][:2])*16
        if not _clear_of(point, excluded['val'], config['exclusion_radius']):
            continue
        bucket = record.get('cohort')
        if bucket not in buckets:
            raise ValueError('Replay record has no supported Source cohort')
        buckets[bucket].append(record)
    replay = []
    while len(replay) < config['replay_clips']:
        added = False
        for bucket in buckets.values():
            if bucket and len(replay) < config['replay_clips']:
                replay.append(bucket.pop())
                added = True
        if not added:
            raise ValueError('Not enough distinct TRAIN replay clips outside all held-out neighborhoods')
    replay_points = [np.asarray(old_plan['poses'][record['pose_id']]['state'][:2])*16 for record in replay]
    own_points = [np.asarray(pose['state'][:2])*16 for pose in own]
    poses, points, used = {}, {'train': [], 'val': []}, set()
    ordinal = 0
    def generated(split, forbidden):
        nonlocal ordinal
        for _ in range(10000):
            candidate = _sample_pose(game, rng, ordinal, forbidden, config['exclusion_radius'], used)
            ordinal += 1
            pair = wall_pair_geometry(game, candidate['state'])
            if pair is not None:
                angle = (math.pi-.005) if ordinal % 4 == 0 else (-math.pi+.005) if ordinal % 4 == 1 else None
                if angle is not None:
                    candidate = _yaw_pose(game, candidate, angle, 'independent_Source_wall_augmentation')
                candidate.update(pair)
                candidate['kind'] = 'independent_Source_wall_augmentation'
                candidate['source_augmentation'] = True
                return candidate
        raise RuntimeError('No legal paired Source wall pose outside held-out neighborhoods')
    val_forbidden = excluded['train']+own_points+replay_points
    for index in range(config['val_poses']):
        pose = generated('val', val_forbidden)
        poses[f'refresh/val/pose_{index:05d}'] = pose
        points['val'].append(np.asarray(pose['state'][:2])*16)
    forbidden = excluded['val']+points['val']
    selected = []
    for index in rng.permutation(len(own)):
        pose = own[int(index)]
        point = np.asarray(pose['state'][:2])*16
        if not _clear_of(point, forbidden, config['exclusion_radius']):
            continue
        pair = wall_pair_geometry(game, pose['state'])
        if pair is None:
            continue
        selected.append({**pose, **pair, 'kind': 'unchanged_valid_TRAIN_own'})
        if len(selected) >= config['train_poses']:
            break
    if config['allow_source_augmentation']:
        # Retain raw own states, add explicitly labeled yaw counterfactuals,
        # then fill legal independent Source walls. Invalid own states never
        # become an anchor and are never moved out of a wall.
        raw_limit = min(len(selected), max(1, config['train_poses']//2))
        selected = selected[:raw_limit]
        yaw_count = min(len(selected), config['train_poses']//4)
        for index in range(yaw_count):
            angle = (math.pi-.005) if index % 2 == 0 else (-math.pi+.005)
            pose = _yaw_pose(game, selected[index], angle, 'Source_yaw_counterfactual_at_valid_TRAIN_own_XY')
            pose.update(wall_pair_geometry(game, pose['state']))
            selected.append(pose)
    if len(selected) < config['train_poses'] and not config['allow_source_augmentation']:
        raise ValueError(f'Only{len(selected)} legal paired own poses available; fewer poses or explicit Source augmentation is required')
    for pose in selected:
        used.add(tuple(np.round(np.asarray(pose['state'][:2])*16, 4)))
    while len(selected) < config['train_poses']:
        selected.append(generated('train', forbidden))
    selected = selected[:config['train_poses']]
    for index, pose in enumerate(selected):
        poses[f'refresh/train/pose_{index:05d}'] = pose
        points['train'].append(np.asarray(pose['state'][:2])*16)
    jobs = []
    for split, count in (('train', config['train_poses']), ('val', config['val_poses'])):
        for index in range(count):
            for branch in range(10):
                seed = config['seed']+(1_000_000 if split == 'val' else 0)+index*10+branch
                jobs.append({'stage': 'short', 'split': split, 'index': index*10+branch, 'seed': seed,
                    'frames': 32, 'branch': branch, 'pose_id': f'refresh/{split}/pose_{index:05d}',
                    'relative_path': f'short/rgb/{split}/episode_{seed:010d}.npz', 'origin': 'new_Source'})
    for index, record in enumerate(replay):
        relative = Path(record['relative_path'])
        jobs.append({**record, 'relative_path': 'short/rgb/train/replay_'+relative.name,
            'origin': 'replay_v1_TRAIN', 'source_relative_path': record['relative_path'],
            'source_sha256': record['sha256'], 'source_pose_id': record['pose_id'],
            'pose_id': 'replay/train/'+record['pose_id'].replace('/', '_'), 'index': len(jobs)})
    minimum = min((float(np.linalg.norm(a-b)) for a in points['train']+replay_points for b in points['val']), default=None)
    kinds = {kind: sum(pose['kind'] == kind for name, pose in poses.items() if '/train/' in name)
             for kind in sorted({pose['kind'] for pose in selected})}
    return {'config': config, 'fingerprint': fingerprint(config), 'poses': poses, 'jobs': jobs,
        'own_pose_audit': own_report, 'new_train_pose_kinds': kinds,
        'old_validation_neighborhoods_excluded': len(excluded['val']),
        'old_train_neighborhoods_excluded_from_newval': len(excluded['train']),
        'train_val_min_initial_xy_distance_tiles': minimum,
        'heldout_inputs_used_for_train': False, 'old_data_modified': False}


def record_episode(job, pose):
    game = Game(seed=job['seed'], endless=True, enemies_enabled=False)
    _source_pose(game, pose['state'])
    initial = json.dumps(game.get_state())
    states, frames, actions, events = [game.state_vector().copy()], [game.render(size=32, overlays=False)], [], []
    for step in range(32):
        action = action_for(job['branch'], step, pose['wall_normal'], game.player['angle'])
        action[:2] *= pose['paired_movement_scale']
        result = game.step(action, dt=1/30)
        if game.done:
            raise RuntimeError('Source paired episode terminated; no repairs or resets are allowed')
        states.append(game.state_vector().copy())
        frames.append(game.render(size=32, overlays=False))
        actions.append(action)
        events.append(Game.event_vector(result))
    states, frames, actions = np.asarray(states, np.float32), np.asarray(frames, np.uint8), np.asarray(actions, np.float32)
    command, fractions, active, cases, audit = contact_cases(states, actions)
    cohort, members, first_counts = _cohort(cases, actions)
    starts = np.arange(17, dtype=np.int32)
    return {'map': game.map.astype(np.int8), 'states': states, 'frames': frames, 'actions': actions,
        'events': np.asarray(events, np.float32), 'hit_ids': np.full(32, -1, np.int64),
        'attack_flags': np.zeros((32, 1), np.float32), 'respawn_flags': np.zeros((32, 1), np.float32),
        'object_masks': np.zeros(frames.shape[:3], bool), 'background_frames': frames.copy(),
        'initial_state': np.asarray(initial), 'axis_command_tiles': command,
        'axis_pass_targets': fractions, 'axis_command_active': active,
        'contact_case_names': np.asarray(CASE_NAMES), 'contact_cases': cases,
        'contact_first16_cohort': np.asarray(cohort), 'contact_sample_cohorts': np.asarray(members),
        'contact_window_starts_h16': starts,
        'contact_window_cases_h16': np.asarray([cases[start:start+16].any(0) for start in starts], bool)}, audit


def _job(arguments):
    root, config, identity, pose, job = arguments
    path = Path(root)/job['relative_path']
    reused = path.exists()
    if job['origin'] == 'replay_v1_TRAIN':
        source = (Path(config['source_contact_root'])/job['source_relative_path']).resolve(strict=True)
        if digest(source) != job['source_sha256']:
            raise ValueError('Original replay file changed after the refresh plan was pinned')
        storage = 'existing'
        if not reused:
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, path)
                storage = 'hardlink'
            except OSError:
                temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
                try:
                    shutil.copy2(source, temporary)
                    _replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
                storage = 'copy'
        if digest(path) != job['source_sha256']:
            raise ValueError('Refresh replay differs from its exact original bytes')
    else:
        expected = _source_pose(Game(seed=job['seed'], endless=True, enemies_enabled=False), pose['state'])
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
                action_branch_name=np.asarray(BRANCH_NAMES[job['branch']]),
                initialization_source=np.asarray(pose['kind']), source_frame_count=np.asarray(33),
                frame_source=np.asarray('reference_engine'), state_repairs=np.asarray(0),
                resets_inside_trajectory=np.asarray(0), paired_wall_normal=np.asarray(pose['wall_normal'], np.float32),
                paired_movement_scale=np.asarray(pose['paired_movement_scale']),
                source_augmentation=np.asarray(bool(pose.get('source_augmentation') or pose.get('source_yaw_counterfactual'))))
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
            try:
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream, **payload)
                _replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        storage = 'existing' if reused else 'new_Source'
    with np.load(path, allow_pickle=False) as saved:
        if (str(saved['split']) != job['split'] or saved['states'].shape != (33, STATE_DIM)
                or saved['actions'].shape != (32, 4) or str(saved['behavior_version']) != BEHAVIOR_VERSION):
            raise ValueError('Refresh record differs from its immutable Source schema/split')
        if job['origin'] != 'replay_v1_TRAIN' and (
                str(saved['navigation_recovery_fingerprint']) != identity
                or str(saved['pose_id']) != job['pose_id']
                or not np.array_equal(saved['states'][0], expected)):
            raise ValueError('Refresh record differs from its exact Source-canonical plan')
        _, _, _, cases, audit = contact_cases(saved['states'], saved['actions'])
        if not np.array_equal(cases, saved['contact_cases']):
            raise ValueError('Refresh case labels disagree with Source movement')
        cohort, members, counts = _cohort(cases, saved['actions'])
        # Replay keeps its original sampling declaration; never rewrite its
        # shared/hardlinked bytes to match a new preferred label convention.
        if job['origin'] == 'replay_v1_TRAIN':
            cohort = str(saved['contact_first16_cohort'])
    return {**job, 'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size,
        'reused': reused, 'storage': storage, 'audit': audit, 'cohort': cohort,
        'sample_cohorts': members, 'first16_case_counts': counts}


def collect(root, *, own_trajectory, source_contact_root, train_poses=384, replay_clips=1024,
            val_poses=0, exclude_data_roots=(), workers=4, seed=150_000_000,
            exclusion_radius=.25, allow_source_augmentation=False):
    counts = (train_poses, replay_clips, val_poses)
    if (any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts)
            or not isinstance(workers, int) or isinstance(workers, bool) or workers < 1
            or not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
            or not isinstance(allow_source_augmentation, bool)
            or not math.isfinite(exclusion_radius) or exclusion_radius <= 0):
        raise ValueError('Invalid immutable refresh counts, worker count, seed or exclusion radius')
    root, own = Path(root).resolve(), Path(own_trajectory).resolve(strict=True)
    source = Path(source_contact_root).resolve(strict=True)
    excluded = list(dict.fromkeys(str(Path(path).resolve(strict=True)) for path in exclude_data_roots))
    if str(source) not in excluded:
        excluded.append(str(source))
    if any(root == protected or protected in root.parents or root in protected.parents
           for protected in [source]+[Path(path) for path in excluded]):
        raise ValueError('Refresh output must not overwrite any original corpus')
    config = {'version': VERSION, 'profile': PROFILE, 'behavior_version': BEHAVIOR_VERSION,
        'seed': seed, 'train_poses': train_poses, 'replay_clips': replay_clips, 'val_poses': val_poses,
        'own_trajectory': str(own), 'own_sha256': digest(own), 'source_contact_root': str(source),
        'source_manifest_sha256': digest(source/'manifest.json'),
        'exclude_data_roots': excluded, 'excluded_plan_sha256': [digest(Path(path)/'plan.json') for path in excluded],
        'exclusion_radius': float(exclusion_radius), 'allow_source_augmentation': allow_source_augmentation,
        'image_size': 32, 'transitions_per_clip': 32, 'branches': list(BRANCH_NAMES)}
    plan_path = root/'plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        if plan['config'] != config:
            raise ValueError('Refresh plan is immutable; use a new output root')
        manifest_path = root/'manifest.json'
        if manifest_path.exists():
            finished = json.loads(manifest_path.read_text(encoding='utf-8'))
            if finished.get('status') == 'complete':
                if finished.get('config') != config or finished.get('fingerprint') != plan['fingerprint']:
                    raise ValueError('Completed refresh identity differs from its immutable plan')
                for record in finished['records']:
                    path = (root/record['relative_path']).resolve(strict=True)
                    if root not in path.parents or digest(path) != record['sha256']:
                        raise ValueError('Completed refresh data changed or escaped its output root')
                return {**finished, 'verified_existing': True}
    else:
        plan = create_plan(config)
        root.mkdir(parents=True, exist_ok=True)
        _atomic_json(plan_path, plan)
    started, records = time.perf_counter(), []
    tasks = [(str(root), config, plan['fingerprint'], plan['poses'].get(job['pose_id']), job) for job in plan['jobs']]
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        iterator = executor.map(_job, tasks) if executor else map(_job, tasks)
        for record in iterator:
            records.append(record)
            elapsed = time.perf_counter()-started
            _atomic_json(root/'status.json', {'status': 'running', 'completed': len(records), 'total': len(tasks),
                'elapsed_seconds': elapsed, 'eta_seconds': elapsed/len(records)*(len(tasks)-len(records)),
                'fingerprint': plan['fingerprint'], 'last_file': record['relative_path']})
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
    groups = {}
    for split in ('train', 'val'):
        selected = [record for record in records if record['split'] == split]
        groups['short/'+split] = {'clips': len(selected), 'transitions': len(selected)*32,
            'new_Source_clips': sum(record['origin'] == 'new_Source' for record in selected),
            'exact_replay_clips': sum(record['origin'] == 'replay_v1_TRAIN' for record in selected),
            'cohort_counts': {cohort: sum(record['cohort'] == cohort for record in selected) for cohort in COHORTS}}
    keys = set().union(*(record['audit'] for record in records)) if records else set()
    manifest = {key: value for key, value in plan.items() if key not in ('poses', 'jobs')}
    manifest.update(status='complete', completed=len(records), total=len(tasks), records=records, groups=groups,
        audit={key: sum(record['audit'].get(key, 0) for record in records) for key in sorted(keys)},
        elapsed_seconds=time.perf_counter()-started,
        label_source='offline continuous Source32; wall-relative recorded controls; no runtime engine corrections',
        replay_source='TRAIN only; original exact bytes; never relabeled or rewritten')
    _atomic_json(root/'manifest.json', manifest)
    _atomic_json(root/'status.json', {key: value for key, value in manifest.items() if key != 'records'})
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--own-trajectory', type=Path, required=True)
    parser.add_argument('--source-contact-root', type=Path, required=True)
    parser.add_argument('--exclude-data-root', type=Path, action='append', default=[])
    parser.add_argument('--train-poses', type=int, default=384)
    parser.add_argument('--replay-clips', type=int, default=1024)
    parser.add_argument('--val-poses', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=150_000_000)
    parser.add_argument('--exclusion-radius', type=float, default=.25)
    parser.add_argument('--allow-source-augmentation', action='store_true')
    args = parser.parse_args(argv)
    report = collect(args.root, own_trajectory=args.own_trajectory, source_contact_root=args.source_contact_root,
        train_poses=args.train_poses, replay_clips=args.replay_clips, val_poses=args.val_poses,
        exclude_data_roots=args.exclude_data_root, workers=args.workers, seed=args.seed,
        exclusion_radius=args.exclusion_radius, allow_source_augmentation=args.allow_source_augmentation)
    print(json.dumps({key: value for key, value in report.items() if key != 'records'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
