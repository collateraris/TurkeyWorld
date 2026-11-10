"""Resumable native32 curriculum data from unchanged reference-engine physics.

Reduced-NPC episodes use valid dead-slot initialization, never frozen AI or
disabled respawn rules. Independent static pictures are explicitly not world
transitions. Exact NPC-difference masks/backgrounds are offline labels only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .engine import BEHAVIOR_VERSION, ENTITY_SLOTS, STATE_DIM, EVENT_NAMES, Game
from .geometry_targets import build_geometry_targets
from .prepare_training_data import _atomic_json, ClipPolicy, SurvivalPolicy


STAGES = ('navigation', 'static_enemy', 'moving1', 'full_game')
ACTIVE_COUNTS = dict(zip(STAGES, (0, 1, 1, 1)))
CASES = ('straight', 'strafe', 'turn', 'wall_stop', 'slide', 'short_return')
MOVING_CASES = CASES+('combat', 'misses', 'attack', 'respawn')
STAGED_DATA_VERSION = 2
_POSE_CACHE = {}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _safe_pose(game, rng, case, frames):
    cache_key = (game.map.tobytes(), case, frames)
    if cache_key in _POSE_CACHE:
        choices = _POSE_CACHE[cache_key]
        point, angle = choices[int(rng.integers(len(choices)))]
        return point.copy(), angle
    floor = np.argwhere(game.map == 0)[:, ::-1].astype(float)+.5
    directions = ((1., 0.), (0., 1.), (-1., 0.), (0., -1.))
    candidates = []
    if case in ('wall_stop', 'slide'):
        for point in floor:
            for direction in directions:
                normal = np.asarray(direction)
                wall = np.floor(point+normal).astype(int)
                if not (0 <= wall[0] < 16 and 0 <= wall[1] < 16 and game.map[wall[1], wall[0]]):
                    continue
                pose = point+.15*normal
                if game._walkable(*pose, game.PLAYER_RADIUS):
                    candidates.append((pose, math.atan2(normal[1], normal[0])))
    else:
        distance = 0. if case == 'turn' else frames*2.4/30*(.5 if case == 'short_return' else 1.)
        for point in floor:
            for direction in directions:
                normal = np.asarray(direction)
                if all(game._walkable(*(point+normal*value), game.PLAYER_RADIUS)
                       for value in np.linspace(0, distance, 9)):
                    angle = math.atan2(normal[1], normal[0])
                    candidates.append((point, angle-math.pi/2 if case == 'strafe' else angle))
    if not candidates:
        raise ValueError('The source map has no valid initialization for '+case)
    _POSE_CACHE[cache_key] = candidates
    return candidates[int(rng.integers(len(candidates)))]


def _initialize(game, job, rng, snapshot=0):
    stage, index = job['stage'], job['index']
    active = list(range(ACTIVE_COUNTS[stage]))
    if stage == 'full_game':
        # Seeds must change genuine trajectories, not merely their filenames:
        # the deterministic engine plus old survival policy has only two
        # seed-parity routes from its fixed reset. Vary the valid START pose.
        state = game.get_state()
        corners = np.array([[1.75, 1.75], [14.25, 1.75], [14.25, 14.25], [1.75, 14.25]])
        for _ in range(64):
            edge = int(rng.integers(4)); fraction = float(rng.uniform(.05, .95))
            point = corners[edge]+fraction*(corners[(edge+1) % 4]-corners[edge])
            if np.linalg.norm(point-np.array([game.enemies[0]['x'], game.enemies[0]['y']])) >= 3.:
                break
        state['player'].update(x=float(point[0]), y=float(point[1]), angle=float(rng.uniform(-math.pi, math.pi)))
        game.set_state(state)
        return list(range(ENTITY_SLOTS)), 0.
    state = game.get_state()
    static = stage == 'static_enemy'
    case = 'turn' if static else 'straight' if job['case'] in ('combat', 'misses', 'attack', 'respawn') else job['case']
    point, angle = _safe_pose(game, rng, case, job['frames'])
    state['player'].update(x=float(point[0]), y=float(point[1]),
                           angle=float(rng.uniform(-math.pi, math.pi) if static else angle))
    floor = np.argwhere(game.map == 0)[:, ::-1].astype(float)+.5
    placed = []
    for ident in active+[slot for slot in range(ENTITY_SLOTS) if slot not in active]:
        choices = [xy for xy in floor if np.linalg.norm(xy-point) >= 1.2
                   and all(np.linalg.norm(xy-other) >= .75 for other in placed)]
        if ident == (active[0] if active else -1):
            visible = [xy for xy in choices if np.linalg.norm(xy-point) <= 5.
                       and game._line_clear(*point, *xy)]
            forward = np.array([math.cos(angle), math.sin(angle)])
            ahead = [xy for xy in visible if np.dot(xy-point, forward) > .5]
            choices = visible if static else ahead or visible or choices
        if not choices:
            raise ValueError('No valid source NPC initialization')
        xy = choices[int(rng.integers(len(choices)))]; placed.append(xy)
        if job['case'] == 'attack':
            close = [point+.86*np.array([math.cos(theta), math.sin(theta)])
                     for theta in rng.uniform(-math.pi, math.pi, 32)]
            close = [candidate for candidate in close if game._walkable(*candidate, game.ENEMY_RADIUS)
                     and game._line_clear(*point, *candidate)]
            if close:
                xy = close[0]
        enemy = state['enemies'][ident]
        if ident not in active:
            # Source corpses remain rendered after their animation. A valid
            # dead corpse at the camera's START point stays at/behind its
            # near clip for all six short navigation controls, without any
            # render override, AI freeze, out-of-map pose or state correction.
            xy = point.copy()
        enemy.update(x=float(xy[0]), y=float(xy[1]), vx=0., vy=0.,
                     hp=60. if ident in active else 0., alive=ident in active,
                     death_timer=0. if ident in active else .6,
                     attack_cooldown=0., hurt_timer=0., animation_phase=0.)
        if job['case'] == 'attack':
            # A valid initial cooldown keeps the first accepted melee event
            # AFTER the two observed context frames, not hidden at action0.
            enemy['attack_cooldown'] = .13
        if static and ident in active:
            enemy.update(hp=float(rng.choice([60., 60., 25.])), animation_phase=float(rng.random()),
                         hurt_timer=float(rng.choice([0., 0., .075])))
            delta = xy-point
            state['player']['angle'] = math.atan2(delta[1], delta[0])+float(rng.uniform(-.65, .65))
        elif job['case'] in ('combat', 'misses', 'attack', 'respawn'):
            delta = xy-point
            state['player']['angle'] = math.atan2(delta[1], delta[0])
            if job['case'] == 'respawn':
                enemy.update(hp=0., alive=False, death_timer=2.8)
    span = job['rgb_time_span']
    offset = int(rng.integers(max(1, math.ceil(span*30))))/30 if span else 0.
    state['time'] = offset
    game.set_state(state)
    return active, offset


def _action(case, index, frames):
    if case == 'straight' or case == 'wall_stop': return np.array([1., 0., 0., 0.], np.float32)
    if case == 'strafe': return np.array([0., 1., 0., 0.], np.float32)
    if case == 'turn': return np.array([0., 0., .75, 0.], np.float32)
    if case == 'slide': return np.array([1., .6, 0., 0.], np.float32)
    if case == 'short_return': return np.array([1. if index < frames//2 else -1., 0., 0., 0.], np.float32)
    raise ValueError('Unknown staged movement case '+case)


def _observe(game, size):
    full = game.render(size=size, overlays=False)
    enemies = game.enemies
    try:
        game.enemies = []  # Render-only ablation; restored before any step.
        background = game.render(size=size, overlays=False)
    finally:
        game.enemies = enemies
    return full, np.any(full != background, axis=-1), background


def _dynamic(job, attempt=0):
    seed = job['seed']+attempt
    game = Game(seed=seed, endless=True)
    rng = np.random.default_rng(seed+173)
    active, offset = _initialize(game, job, rng)
    initial = json.dumps(game.get_state())
    initial_time = game.time
    states, frames, masks, backgrounds, depths, visibility = [], [], [], [], [], []
    actions, events, hits, attacks, respawns = [], [], [], [], []
    policy = (SurvivalPolicy(seed, 30) if job['stage'] == 'full_game' else
              ClipPolicy(seed, job['case'], 30, job['frames']) if job['case'] in ('combat', 'misses', 'attack', 'respawn') else None)
    if job['stage'] == 'full_game':
        p = game.player
        edge = int(np.argmin([abs(p['y']-1.75), abs(p['x']-14.25), abs(p['y']-14.25), abs(p['x']-1.75)]))
        policy.index = (edge+1) % 4 if policy.direction > 0 else edge

    def observe():
        state = game.state_vector()
        full, mask, background = _observe(game, job['size'])
        geometry = build_geometry_targets(game.map, state)
        states.append(state); frames.append(full); masks.append(mask); backgrounds.append(background)
        depths.append(geometry['depth']); visibility.append(geometry['visibility'])

    observe()
    for index in range(job['frames']):
        action = policy(game, index) if policy else _action(job['case'], index, job['frames'])
        before = np.asarray([enemy['alive'] for enemy in game.enemies])
        event = game.step(action, dt=1/30)
        actions.append(action); events.append(Game.event_vector(event)); hits.append(event.get('hit_id', -1))
        flags = np.zeros(ENTITY_SLOTS, np.float32); flags[event.get('attacker_ids', [])] = 1
        attacks.append(flags); respawns.append(~before & np.asarray([enemy['alive'] for enemy in game.enemies]))
        observe()
        if game.done:
            break
    if job['stage'] != 'full_game':
        inactive = [slot for slot in range(ENTITY_SLOTS) if slot not in active]
        entities = np.asarray(states)[:, 8:-2].reshape(-1, ENTITY_SLOTS, 10)
        if inactive and (np.any(entities[:, inactive, 5] >= .5) or np.any(np.asarray(respawns)[:, inactive])):
            raise AssertionError('A disabled staged slot revived inside a short clip')
    return dict(map=game.map.astype(np.int8), states=np.asarray(states, np.float32),
        frames=np.asarray(frames, np.uint8), actions=np.asarray(actions, np.float32),
        events=np.asarray(events, np.float32), hit_ids=np.asarray(hits, np.int64),
        attack_flags=np.asarray(attacks, np.float32), respawn_flags=np.asarray(respawns, np.float32),
        depth=np.asarray(depths, np.float32), visibility=np.asarray(visibility, np.float32),
        object_masks=np.asarray(masks, bool), background_frames=np.asarray(backgrounds, np.uint8),
        initial_state=np.asarray(initial), initial_time_offset=np.asarray(offset),
        initial_game_time=np.asarray(initial_time),
        active_ids=np.asarray(active, np.int64), seed=np.asarray(seed), initialization_attempt=np.asarray(attempt))


def _snapshots(job):
    game = Game(seed=job['seed'], endless=True)
    rng = np.random.default_rng(job['seed']+173)
    frames, states, masks, backgrounds, snapshots, ids = [], [], [], [], [], []
    for index in range(job['frames']):
        game.reset(job['seed'])
        active, _ = _initialize(game, job, rng, snapshot=index)
        full, mask, background = _observe(game, job['size'])
        frames.append(full); masks.append(mask); backgrounds.append(background)
        states.append(game.state_vector()); snapshots.append(game.get_state()); ids.append(active[0])
    return dict(map=game.map.astype(np.int8), frames=np.asarray(frames, np.uint8),
        states=np.asarray(states, np.float32), object_masks=np.asarray(masks, bool),
        background_frames=np.asarray(backgrounds, np.uint8), snapshot_states=np.asarray(json.dumps(snapshots)),
        snapshot_active_ids=np.asarray(ids, np.int64), seed=np.asarray(job['seed']))


def _stats(path, job):
    with np.load(path, allow_pickle=False) as saved:
        for key, value in (('stage', job['stage']), ('split', job['split']),
                           ('behavior_version', BEHAVIOR_VERSION), ('staged_config_fingerprint', job['fingerprint'])):
            if key not in saved.files or str(saved[key]) != value:
                raise ValueError('Existing staged file provenance differs: '+str(path))
        static = job['stage'] == 'static_enemy'
        attempt = int(saved['initialization_attempt']) if 'initialization_attempt' in saved.files else 0
        expected_seed = job['seed']+attempt if job['stage'] == 'full_game' else job['seed']
        if (int(saved['seed']) != expected_seed or not 0 <= attempt < job['full_attempts']
                or int(saved['state_dim']) != STATE_DIM or int(saved['entity_slots']) != ENTITY_SLOTS):
            raise ValueError('Existing staged seed/state schema differs')
        if bool(saved['world_transition_compatible']) == static:
            raise ValueError('Staged transition/snapshot claim mismatch')
        frames, states = saved['frames'], saved['states']
        count = len(frames)
        if (frames.dtype != np.uint8 or frames.shape[1:] != (32, 32, 3) or states.shape != (count, STATE_DIM)
                or saved['object_masks'].shape != (count, 32, 32) or saved['object_masks'].dtype != bool
                or saved['background_frames'].shape != frames.shape or saved['background_frames'].dtype != np.uint8
                or not np.isfinite(states).all()):
            raise ValueError('Existing staged RGB/state/label schema differs')
        transitions = 0 if static else len(saved['actions'])
        if static and any(key in saved.files for key in ('actions', 'events', 'hit_ids')):
            raise ValueError('Static snapshots must not claim world transitions')
        if not static and (transitions+1 != count or saved['actions'].shape != (transitions, 4)):
            raise ValueError('Invalid staged transition alignment')
        if (static and count != job['frames'] or not static and
                (transitions != job['frames'] and (job['stage'] != 'full_game' or states[-1, 4] > 0))):
            raise ValueError('Incomplete staged file without a genuine terminal episode')
        totals = {name: int(saved['events'][:, index].sum()) for index, name in enumerate(EVENT_NAMES)} if not static else None
        return {'job_id': job['job_id'], 'stage': job['stage'], 'split': job['split'],
            'relative_path': job['relative_path'], 'seed': int(saved['seed']), 'job_seed': job['seed'],
            'case': job['case'], 'transitions': transitions, 'snapshots': count if static else 0,
            'observations': count, 'size_bytes': path.stat().st_size,
            'completed_requested_length': count == job['frames'] if static else transitions == job['frames'],
            'eligible_450': bool(not static and transitions >= 451 and states[-1, 4] > 0),
            'terminal': bool(not static and states[-1, 4] <= 0), 'events': totals,
            'respawns': 0 if static else int(saved['respawn_flags'].sum()),
            'frames_with_npc_pixels': int(saved['object_masks'].any(axis=(1, 2)).sum()),
            'mean_foreground_fraction': float(saved['object_masks'].mean())}


def generate_job(job):
    """Generate one independent file, or validate/reuse its existing bytes."""
    path = Path(job['path'])
    if path.exists():
        return {**_stats(path, job), 'reused': True}
    static = job['stage'] == 'static_enemy'
    if static:
        payload = _snapshots(job)
    else:
        payload = _dynamic(job)
        if job['stage'] == 'full_game':
            for attempt in range(1, job['full_attempts']):
                if len(payload['actions']) == job['frames']:
                    break
                candidate = _dynamic(job, attempt)
                if len(candidate['actions']) > len(payload['actions']):
                    payload = candidate
    policy = ('survival' if job['stage'] == 'full_game' else 'collision'
              if job['case'] in ('wall_stop', 'slide') else 'return_memory'
              if job['case'] == 'short_return' else 'movement')
    payload.update(format_version=np.asarray(4), staged_data_version=np.asarray(STAGED_DATA_VERSION),
        behavior_version=np.asarray(BEHAVIOR_VERSION),
        stage=np.asarray(job['stage']), split=np.asarray(job['split']), policy=np.asarray(policy),
        movement_case=np.asarray(job['case']), dt=np.asarray(1/30), endless=np.asarray(True),
        warmup_steps=np.asarray(0), requested_transitions=np.asarray(0 if static else job['frames']),
        dataset_kind=np.asarray('renderer_snapshots' if static else 'world_transitions'),
        world_transition_compatible=np.asarray(not static),
        staged_config_fingerprint=np.asarray(job['fingerprint']),
        rgb_time_span=np.asarray(0. if job['stage'] == 'full_game' else job['rgb_time_span']),
        frame_source=np.asarray('reference_engine'), background_frame_source=np.asarray('reference_engine_npc_ablation'),
        initialization_source=np.asarray('seeded_valid_continuous_start' if job['stage'] == 'full_game' else 'staged_valid_initial_state'),
        disabled_slots_initialization_only=np.asarray(job['stage'] == 'navigation'),
        state_dim=np.asarray(STATE_DIM), entity_slots=np.asarray(ENTITY_SLOTS),
        state_schema_version=np.asarray('single_enemy_v1'),
        engine_semantics_modified=np.asarray(False), ai_freeze=np.asarray(False), model_visited=np.asarray(False))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.npz.'+str(os.getpid())+'.tmp')
    try:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {**_stats(path, job), 'reused': False}


def _plan(root, config):
    jobs = []
    for stage in config['stages']:
        for split in ('train', 'val'):
            for index in range(config['counts'][stage][split]):
                seed = config['seed']+STAGES.index(stage)*2_000_000+(1_000_000 if split == 'val' else 0)
                seed += index*(16 if stage == 'full_game' else 1)
                kind = 'snapshots' if stage == 'static_enemy' else 'rgb'
                relative = f'{stage}/{kind}/{split}/episode_{seed:010d}.npz'
                jobs.append({**config, 'stage': stage, 'split': split, 'index': index, 'seed': seed,
                    'case': 'static_view' if stage == 'static_enemy' else 'survival' if stage == 'full_game' else
                            (MOVING_CASES if stage == 'moving1' else CASES)[index % len(MOVING_CASES if stage == 'moving1' else CASES)],
                    'frames': config['full_frames'] if stage == 'full_game' else config['frames'],
                    'relative_path': relative, 'path': str(root/relative), 'job_id': f'{stage}/{split}/{index}'})
    return jobs


def collect_staged(root, train_clips=None, val_clips=64, *, stages=None, navigation_clips=None,
                   static_clips=None, moving_clips=None, full_train_clips=0, full_val_clips=0,
                   frames=32, full_frames=451, size=32, seed=70_000_000, workers=4,
                   rgb_time_span=120., full_attempts=4):
    """Collect per-stage/split files. Full-game clips are optional and genuine.

    Short counts apply per phase; static counts mean files of `frames`
    independent snapshots. Full-game frames means actual source transitions.
    """
    root = Path(root).resolve()
    if (not 1 <= frames <= 32 or size != 32 or workers < 1 or full_frames < 1 or
            not 1 <= full_attempts <= 16 or not math.isfinite(rgb_time_span) or rgb_time_span < 0):
        raise ValueError('Require native32, 1..32 short transitions, positive workers/full frames and valid time span')
    if not isinstance(seed, int) or seed < 0:
        raise ValueError('Seed must be a nonnegative integer')
    selected = list(STAGES[:-1] if stages is None else stages)
    if (full_train_clips or full_val_clips) and 'full_game' not in selected:
        selected.append('full_game')
    if not selected or len(set(selected)) != len(selected) or any(stage not in STAGES for stage in selected):
        raise ValueError('Unknown or duplicated staged phase')
    selected.sort(key=STAGES.index)
    defaults = {'navigation': 1000, 'static_enemy': 1000, 'moving1': 2000}
    overrides = {'navigation': navigation_clips, 'static_enemy': static_clips,
                 **{stage: moving_clips for stage in STAGES[2:-1]}}
    counts = {}
    for stage in selected:
        train = full_train_clips if stage == 'full_game' else overrides[stage]
        if train is None: train = train_clips if train_clips is not None else defaults[stage]
        counts[stage] = {'train': train, 'val': full_val_clips if stage == 'full_game' else val_clips}
        if any(not isinstance(n, int) or isinstance(n, bool) or not 0 <= n <= 50_000 for n in counts[stage].values()):
            raise ValueError('Per-stage/split clip count must be an integer in0..50000')
    config = {'staged_data_version': STAGED_DATA_VERSION, 'behavior_version': BEHAVIOR_VERSION,
        'state_schema_version': 'single_enemy_v1', 'state_dim': STATE_DIM, 'entity_slots': ENTITY_SLOTS,
        'navigation_corpse_pose': 'player_start_nearclip', 'attack_initial_cooldown': .13,
        'stages': selected, 'counts': counts, 'frames': frames, 'full_frames': full_frames,
        'size': size, 'hz': 30, 'seed': seed, 'rgb_time_span': float(rgb_time_span), 'full_attempts': full_attempts}
    config['fingerprint'] = _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path, status_path = root/'manifest.json', root/'status.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8'))['config'] != config:
        raise ValueError('Immutable staged data configuration differs; choose a new root')
    jobs = _plan(root, config)
    records, started = {}, time.perf_counter()

    def publish(status='running', error=None):
        groups = {}
        for row in records.values():
            group = groups.setdefault(row['stage']+'/'+row['split'], Counter())
            for key in ('transitions', 'snapshots', 'observations', 'size_bytes', 'respawns', 'eligible_450'):
                group[key] += row[key]
            group['files'] += 1
        generated = sum(not record['reused'] for record in records.values())
        elapsed = time.perf_counter()-started
        result = {'format_version': 1, 'status': status, 'updated_at': _now(), 'config': config,
            'completed_files': len(records), 'total_files': len(jobs), 'new_files': generated,
            'reused_files': len(records)-generated, 'groups': groups, 'elapsed_seconds': elapsed,
            'eta_seconds': (len(jobs)-len(records))*elapsed/generated if generated else None,
            'world_transition_examples': sum(row['transitions'] for row in records.values()),
            'renderer_snapshot_examples': sum(row['snapshots'] for row in records.values()), 'error': error}
        _atomic_json(status_path, result)
        return result

    _atomic_json(manifest_path, {'format_version': 1, 'status': 'planned', 'config': config,
        'jobs': [{'job_id': job['job_id'], 'relative_path': job['relative_path'], 'seed': job['seed']} for job in jobs]})
    publish()

    def record(row):
        records[row['job_id']] = row
        with (root/'progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row)+'\n')
        summary = publish()
        print(json.dumps({key: summary[key] for key in ('completed_files', 'total_files', 'world_transition_examples',
                                                       'renderer_snapshot_examples', 'eta_seconds')}), flush=True)

    try:
        if workers == 1:
            for job in jobs: record(generate_job(job))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                iterator, pending = iter(jobs), {}
                for _ in range(min(len(jobs), workers*2)):
                    job = next(iterator); pending[pool.submit(generate_job, job)] = job
                while pending:
                    ready, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in ready:
                        pending.pop(future); record(future.result())
                        job = next(iterator, None)
                        if job is not None: pending[pool.submit(generate_job, job)] = job
        result = publish('complete')
        _atomic_json(manifest_path, {**result, 'records': sorted(records.values(), key=lambda row: row['job_id']),
            'source_engine_step': 'Normal Game.step; no AI freeze, health repair or reset inside any trajectory',
            'static_snapshots': 'Independent source poses; not world transitions',
            'selection': 'Use explicit stage/split paths; no legacy nine-policy quota claim'})
        return result
    except BaseException as error:
        publish('failed', repr(error))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--train-clips', type=int)
    parser.add_argument('--val-clips', type=int, default=64)
    parser.add_argument('--navigation-clips', type=int)
    parser.add_argument('--static-clips', type=int)
    parser.add_argument('--moving-clips', type=int)
    parser.add_argument('--full-train-clips', type=int, default=0)
    parser.add_argument('--full-val-clips', type=int, default=0)
    parser.add_argument('--full-frames', type=int, default=451)
    parser.add_argument('--full-attempts', type=int, default=4)
    parser.add_argument('--stages', nargs='+', choices=STAGES)
    parser.add_argument('--frames', type=int, default=32)
    parser.add_argument('--size', type=int, choices=[32], default=32)
    parser.add_argument('--seed', type=int, default=70_000_000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--rgb-time-span', type=float, default=120.)
    print(json.dumps(collect_staged(**vars(parser.parse_args(argv))), indent=2))


if __name__ == '__main__':
    main()
