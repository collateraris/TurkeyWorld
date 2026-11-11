"""Unmodified Source melee negatives and natural corner-rounding positives.

Only the initial valid player/NPC poses vary. All32 controls are zero; Source AI
and collision/attack logic run normally. Render-only NPC ablations are restored
before each step and are stored solely as foreground/background labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .engine import BEHAVIOR_VERSION, ENTITY_SLOTS, EVENT_NAMES, Game, STATE_DIM
from .geometry_targets import build_geometry_targets
from .prepare_training_data import _atomic_json
from .staged_data import _observe


PROTOCOL = 'source_corner_melee_hard_negatives_v1'


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_pose_plan(train_clips=256, val_clips=64, *, seed=173, pose_guard=.05):
    """Choose separated legal camera/NPC poses, without changing Source rules."""
    if (any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in (train_clips, val_clips))
            or not isinstance(pose_guard, (float, int)) or not math.isfinite(pose_guard) or pose_guard <= 0):
        raise ValueError('Positive train/val counts and finite positive pose guard are required')
    game = Game(endless=True)
    corners = [(x, y, sx, sy) for y in range(1, 16) for x in range(1, 16)
        for sx in (-1, 1) for sy in (-1, 1)
        if game.map[y if sy > 0 else y-1, x if sx > 0 else x-1] != 0]
    # Hold out complete wall vertices: nearby poses from one corner never
    # straddle TRAIN/VAL. Within each split, all four-coordinate poses differ.
    corners = [(x, y, sx, sy) for x, y, sx, sy in corners
        if game.map[y if sy > 0 else y-1, x-1 if sx > 0 else x] == 0
        and game.map[y-1 if sy > 0 else y, x if sx > 0 else x-1] == 0]
    rng = np.random.default_rng(seed)
    vertices = sorted({tuple(corner[:2]) for corner in corners})
    rng.shuffle(vertices)
    val_count = max(1, min(len(vertices)-1, math.ceil(len(vertices)*val_clips/(train_clips+val_clips))))
    heldout_vertices = set(vertices[:val_count])
    result, all_poses = [], []
    for split, count, base in (('train', train_clips, 96000000), ('val', val_clips, 97000000)):
        choices = [corner for corner in corners if (tuple(corner[:2]) in heldout_vertices) == (split == 'val')]
        poses, anchors = [], []
        for _ in range(max(200000, count*1500)):
            x, y, sx, sy = choices[int(rng.integers(len(choices)))]
            a, b, d, f = rng.uniform([.22, .22, .27, .27], [.4, .47, .47, .4])
            player = np.array([x-sx*a, y+sy*b]); enemy = np.array([x+sx*d, y-sy*f])
            if (not .80 <= np.linalg.norm(enemy-player) <= .995
                    or not game._walkable(*player, Game.PLAYER_RADIUS)
                    or not game._walkable(*enemy, Game.ENEMY_RADIUS)
                    or game._line_clear(*enemy, *player)):
                continue
            pose = np.r_[player, enemy]
            if all_poses and np.linalg.norm(np.asarray(all_poses)-pose, axis=1).min() < pose_guard:
                continue
            if split == 'val':
                difference = np.asarray([row['pose'] for row in result])-pose
                if (np.linalg.norm(difference[:, :2], axis=1).min() < pose_guard
                        or np.linalg.norm(difference[:, 2:], axis=1).min() < pose_guard):
                    continue
            poses.append(pose.tolist()); anchors.append([x, y, sx, sy]); all_poses.append(pose.tolist())
            if len(poses) == count: break
        if len(poses) != count:
            raise ValueError('Insufficient independently separated legal corner poses')
        for index, source in enumerate(rng.permutation(count)):
            result.append(dict(split=split, seed=base+index, pose=poses[int(source)], corner=anchors[int(source)]))
    return result


def _record(path, job):
    with np.load(path, allow_pickle=False) as z:
        steps = len(z['actions'])
        if (str(z['collection_protocol']) != job['fingerprint'] or int(z['seed']) != job['seed']
                or str(z['split']) != job['split'] or str(z['behavior_version']) != BEHAVIOR_VERSION
                or z['states'].shape != (33, STATE_DIM) or steps != 32
                or not np.array_equal(z['initial_pose_tiles'], job['pose'])
                or not bool(z['source_hard_negative_flags'][0, 0])):
            raise ValueError('Existing corner clip differs from its immutable Source plan')
        counts = dict(zip(EVENT_NAMES, z['events'].sum(0).astype(int).tolist()))
        return dict(relative_path=job['relative_path'], stage='moving1', split=job['split'],
            seed=job['seed'], case='melee_wall_hard_negative', transitions=steps, observations=steps+1,
            events=counts, hard_negative_count=int(z['source_hard_negative_flags'].sum()),
            eligible_attack_positive_count=int((z['source_melee_eligible'] & (z['attack_flags'] > .5)).sum()),
            initial_hard_negative=True, later_positive=bool(z['attack_flags'][1:].any()),
            terminal=bool(z['states'][-1, 4] <= 0), eligible_450=False,
            sha256=_digest(path), size_bytes=path.stat().st_size)


def collect_job(job):
    path = Path(job['path'])
    if path.exists():
        return _record(path, job)
    game = Game(seed=job['seed'], endless=True)
    p, e = np.asarray(job['pose'], np.float64).reshape(2, 2)
    initial = game.get_state()
    initial['player'].update(x=float(p[0]), y=float(p[1]),
        angle=math.atan2(e[1]-p[1], e[0]-p[0]), hp=100.)
    initial['enemies'][0].update(x=float(e[0]), y=float(e[1]), hp=60., alive=True,
        vx=0., vy=0., attack_cooldown=0., hurt_timer=0., death_timer=0.)
    game.set_state(initial)
    initial_json = json.dumps(game.get_state())
    values = {key: [] for key in ('states', 'frames', 'object_masks', 'background_frames',
        'depth', 'visibility', 'actions', 'events', 'hit_ids', 'attack_flags', 'respawn_flags',
        'source_melee_eligible', 'source_los_clear', 'source_hard_negative_flags')}

    def observe():
        state = game.state_vector()
        rgb, mask, background = _observe(game, 32)
        geometry = build_geometry_targets(game.map, state)
        for key, value in dict(states=state, frames=rgb, object_masks=mask,
                background_frames=background, depth=geometry['depth'], visibility=geometry['visibility']).items():
            values[key].append(value)

    observe()
    for _ in range(32):
        old = game.state_vector()
        clear = game._line_clear(game.enemies[0]['x'], game.enemies[0]['y'],
                                 game.player['x'], game.player['y'])
        action = np.zeros(4, np.float32)
        event = game.step(action, dt=1/30)
        new = game.state_vector()
        eligible = bool(old[13] >= .5 and old[12] > 0 and new[13] >= .5 and new[12] > 0
            and old[4] > 0 and old[14]-1/(30*1.2) <= 1e-6
            and np.linalg.norm((old[8:10]-new[:2])*16) <= 1.+1e-6)
        attack = bool(0 in event['attacker_ids'])
        for key, value in dict(actions=action, events=Game.event_vector(event),
            hit_ids=event['hit_id'], attack_flags=[attack], respawn_flags=[old[13] < .5 and new[13] >= .5],
            source_melee_eligible=[eligible], source_los_clear=[clear],
            source_hard_negative_flags=[eligible and not clear and not attack]).items():
            values[key].append(value)
        observe()
    payload = {key: np.asarray(value) for key, value in values.items()}
    payload.update(map=game.map.astype(np.int8), initial_state=np.asarray(initial_json),
        initial_pose_tiles=np.asarray(job['pose'], np.float64), wall_corner=np.asarray(job['corner']),
        seed=np.asarray(job['seed']), split=np.asarray(job['split']), stage=np.asarray('moving1'),
        case=np.asarray('melee_wall_hard_negative'), combat_case=np.asarray('melee_wall_hard_negative'),
        dt=np.asarray(1/30), hz=np.asarray(30), behavior_version=np.asarray(BEHAVIOR_VERSION),
        state_dim=np.asarray(STATE_DIM), entity_slots=np.asarray(ENTITY_SLOTS),
        state_schema_version=np.asarray('single_enemy_v1'), format_version=np.asarray(4),
        world_transition_compatible=np.asarray(True), enemies_enabled=np.asarray(True),
        navigation_only=np.asarray(False), policy=np.asarray('stationary_source_corner_rounding'),
        collection_protocol=np.asarray(job['fingerprint']), collection_kind=np.asarray(PROTOCOL))
    for key in ('states', 'actions', 'events', 'attack_flags', 'respawn_flags', 'depth', 'visibility'):
        payload[key] = payload[key].astype(np.float32)
    payload['hit_ids'] = payload['hit_ids'].astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **payload); temporary.replace(path)
    return _record(path, job)


def collect_dataset(output, *, train_clips=256, val_clips=64, workers=4, seed=173, pose_guard=.05):
    root = Path(output).resolve()
    config = dict(protocol=PROTOCOL, behavior_version=BEHAVIOR_VERSION, state_dim=STATE_DIM,
        entity_slots=ENTITY_SLOTS, train_clips=train_clips, val_clips=val_clips,
        frames=32, image_size=32, hz=30, seed=seed, pose_guard_tiles=pose_guard)
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    manifest_path = root/'manifest.json'
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding='utf8'))
        if existing['fingerprint'] != fingerprint:
            raise ValueError('Immutable hard-negative configuration changed')
        if existing['status'] == 'complete':
            for row in existing['records']:
                if _digest(root/row['relative_path']) != row['sha256']:
                    raise ValueError('Immutable hard-negative Source content changed')
            return existing
    root.mkdir(parents=True, exist_ok=True)
    plan = build_pose_plan(train_clips, val_clips, seed=seed, pose_guard=pose_guard)
    for job in plan:
        job.update(fingerprint=fingerprint, relative_path=f"moving1/rgb/{job['split']}/episode_{job['seed']:09d}.npz")
        job['path'] = str(root/job['relative_path'])
    started, records = time.perf_counter(), []

    def publish(status):
        groups = {}
        for row in records:
            group = groups.setdefault(row['split'], dict(files=0, transitions=0, hard_negatives=0,
                eligible_attack_positives=0, episodes_with_later_positive=0, events=Counter()))
            group['files'] += 1; group['transitions'] += row['transitions']
            group['hard_negatives'] += row['hard_negative_count']
            group['eligible_attack_positives'] += row['eligible_attack_positive_count']
            group['episodes_with_later_positive'] += int(row['later_positive']); group['events'].update(row['events'])
        report = dict(format_version=1, status=status, fingerprint=fingerprint, config=config,
            behavior_version=BEHAVIOR_VERSION, state_dim=STATE_DIM, entity_slots=ENTITY_SLOTS, image_size=32, hz=30,
            completed_files=len(records), total_files=len(plan), groups=groups,
            records=sorted(records, key=lambda row: row['relative_path']),
            pose_plan=plan, source_physics_unchanged=True, initial_state_changes_only=True,
            source_future_feedback=False, training_validation_seeds_disjoint=True,
            cross_split_player_and_npc_min_separation_tiles=pose_guard,
            all_four_coordinate_poses_min_separation_tiles=pose_guard,
            eligibility_definition='Source previous NPC and next player; alive, cooldown-dt<=0, distance<=1',
            hard_negative_definition='eligible, Source centre LOS false, Source accepted attack false',
            elapsed_seconds=time.perf_counter()-started, updated_at=datetime.now(timezone.utc).isoformat())
        _atomic_json(manifest_path, report)
        _atomic_json(root/'status.json', {key: value for key, value in report.items() if key not in ('records', 'pose_plan')})
        return report

    publish('collecting')
    try:
        if workers == 1:
            for job in plan:
                records.append(collect_job(job))
                if len(records) % 16 == 0: publish('collecting')
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for future in as_completed([executor.submit(collect_job, job) for job in plan]):
                    records.append(future.result())
                    if len(records) % 16 == 0: publish('collecting')
    except BaseException:
        publish('failed'); raise
    return publish('complete')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-clips', type=int, default=256)
    parser.add_argument('--val-clips', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=173)
    args = parser.parse_args(argv)
    if args.workers < 1: parser.error('workers must be positive')
    report = collect_dataset(**vars(args))
    print(json.dumps({'status': report['status'], 'files': report['completed_files'], 'groups': report['groups']}), flush=True)


if __name__ == '__main__':
    main()
