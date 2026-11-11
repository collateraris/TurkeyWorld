"""Collect one-NPC combat and detailed navigation from unchanged Source physics.

Only valid initial states and player controls vary. No healing, resets, frozen
AI or corrected transitions occur inside an episode. TRAIN/VAL seeds differ.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .collect import PatrolPolicy
from .engine import BEHAVIOR_VERSION, ENTITY_SLOTS, EVENT_NAMES, Game, STATE_DIM
from .geometry_targets import build_geometry_targets
from .power_guard import KeepAwake
from .staged_data import _initialize, _observe
from .training_pipeline import atomic_json, now


COMBAT_CASES = ('duel', 'melee_then_shoot', 'miss_then_hit', 'blocked', 'kill_respawn', 'return')
NAV_CASES = ('corner', 'slide', 'return', 'wall_stop', 'turn', 'strafe')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _initial_combat(game, job, rng):
    case = job['case']
    source_case = ('attack' if case in ('melee_then_shoot', 'terminal') else
                   'respawn' if case == 'kill_respawn' and job['index'] % 2 else 'combat')
    _initialize(game, {'stage': 'moving1', 'index': job['index'], 'case': source_case,
        'frames': 32, 'rgb_time_span': 60.}, rng)
    state = game.get_state()
    if case == 'terminal':
        state['player']['hp'] = 8.
        state['enemies'][0]['attack_cooldown'] = .6
    if case == 'blocked':
        player = np.array([state['player']['x'], state['player']['y']])
        floor = np.argwhere(game.map == 0)[:, ::-1]+.5
        choices = [p for p in floor if 1.2 < np.linalg.norm(p-player) < 5.
                   and not game._line_clear(*player, *p)]
        if choices:
            xy = choices[int(rng.integers(len(choices)))]
            state['enemies'][0].update(x=float(xy[0]), y=float(xy[1]), vx=0., vy=0.)
    enemy = state['enemies'][0]
    state['player']['angle'] = math.atan2(enemy['y']-state['player']['y'], enemy['x']-state['player']['x'])
    game.set_state(state)


def _initial_navigation(game, job, rng):
    case = job['case']
    candidates = []
    for _ in range(1200):
        floor = np.argwhere(game.map == 0)
        row, column = floor[int(rng.integers(len(floor)))]
        point = np.array([column+rng.uniform(.22, .78), row+rng.uniform(.22, .78)])
        if not game._walkable(*point, game.PLAYER_RADIUS):
            continue
        walls = []
        for angle in (0., math.pi/2, math.pi, -math.pi/2):
            direction = np.array([math.cos(angle), math.sin(angle)])
            blocked = next((d for d in (.25, .4, .6, .8, 1.)
                if not game._walkable(*(point+d*direction), game.PLAYER_RADIUS)), 2.)
            if blocked < 1.1:
                walls.append((angle, blocked))
        if case == 'corner':
            pairs = [(a, b) for a in walls for b in walls if a[1] <= .6 and b[1] <= .6
                     and abs(abs(math.atan2(math.sin(a[0]-b[0]), math.cos(a[0]-b[0])))-math.pi/2) < .01]
            if not pairs:
                continue
            a, b = pairs[int(rng.integers(len(pairs)))]
            angle = math.atan2(math.sin(a[0])+math.sin(b[0]), math.cos(a[0])+math.cos(b[0]))
        elif case in ('slide', 'wall_stop'):
            if not walls:
                continue
            angle = walls[int(rng.integers(len(walls)))][0]
        else:
            angle = float(rng.uniform(-math.pi, math.pi))
        candidates.append((point, angle+float(rng.uniform(-.06, .06))))
        break
    if not candidates:
        raise ValueError('Could not initialize a valid detailed-navigation '+case)
    point, angle = candidates[0]
    state = game.get_state()
    state['player'].update(x=float(point[0]), y=float(point[1]), angle=angle)
    state['time'] = float(rng.integers(1800))/30
    game.set_state(state)


def control(game, case, tick, patrol):
    if case in NAV_CASES and not game.enemies_enabled:
        if case == 'turn': return np.array([0., 0., .6, 0.], np.float32)
        if case == 'strafe': return np.array([0., .8, 0., 0.], np.float32)
        if case == 'slide': return np.array([1., .7, 0., 0.], np.float32)
        if case == 'corner': return np.array([.8, .3 if tick < 16 else -.3, 0., 0.], np.float32)
        if case == 'return': return np.array([.8 if tick < 16 else -.8, 0., 0., 0.], np.float32)
        return np.array([1., 0., 0., 0.], np.float32)
    if case == 'terminal':
        return np.zeros(4, np.float32)
    alive = [e for e in game.enemies if e['alive']]
    if not alive:
        return patrol(game, tick)
    enemy = alive[0]; player = game.player
    delta = np.array([enemy['x']-player['x'], enemy['y']-player['y']])
    distance = float(np.linalg.norm(delta))
    error = math.atan2(math.sin(math.atan2(delta[1], delta[0])-player['angle']),
                       math.cos(math.atan2(delta[1], delta[0])-player['angle']))
    offset = .32 if case == 'miss_then_hit' and tick % 100 < 30 else 0.
    turn = np.clip((error-offset)*5, -1, 1)
    clear = game._line_clear(player['x'], player['y'], enemy['x'], enemy['y'])
    wait = case == 'melee_then_shoot' and tick < 50
    fire = float(not wait and (abs(error-offset) < .09 or case == 'blocked' and tick < 30))
    if wait:
        forward, side = 0., 0.
    elif case == 'return':
        forward, side = (.65 if tick % 100 < 50 else -.65), .15
    elif not clear:
        move = patrol(game, tick)
        forward, side = float(move[0]), float(move[1])
    else:
        forward = .7 if distance > 2.5 and abs(error) < .35 else -.4 if distance < 1.1 else 0.
        side = .18*math.sin(tick*.05)
    return np.array([forward, side, turn, fire], np.float32)


def collect_job(job):
    path = Path(job['path'])
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if str(saved['collection_protocol']) != job['protocol'] or int(saved['seed']) != job['seed']:
                raise ValueError('Existing combat corpus file has different provenance')
        return record(path, job)
    enabled = job['stage'] != 'navigation'
    game = Game(seed=job['seed'], endless=True, enemies_enabled=enabled)
    rng = np.random.default_rng(job['seed'])
    (_initial_combat if enabled else _initial_navigation)(game, job, rng)
    initial = json.dumps(game.get_state())
    patrol = PatrolPolicy()
    states, frames, masks, backgrounds, depth, visibility = [], [], [], [], [], []
    actions, events, hit_ids, attack_flags, respawn_flags = [], [], [], [], []
    def observe():
        state = game.state_vector()
        rgb, mask, bg = _observe(game, 32)
        geometry = build_geometry_targets(game.map, state)
        states.append(state); frames.append(rgb); masks.append(mask); backgrounds.append(bg)
        depth.append(geometry['depth']); visibility.append(geometry['visibility'])
    observe()
    for tick in range(job['frames']):
        action = control(game, job['case'], tick, patrol)
        before = np.array([enemy['alive'] for enemy in game.enemies])
        event = game.step(action, dt=1/30)
        flags = np.zeros(ENTITY_SLOTS, np.float32)
        flags[event.get('attacker_ids', [])] = 1.
        actions.append(action); events.append(Game.event_vector(event)); hit_ids.append(event.get('hit_id', -1))
        attack_flags.append(flags)
        respawn_flags.append(~before & np.array([enemy['alive'] for enemy in game.enemies]))
        observe()
        if game.done:
            break
    payload = dict(map=game.map.astype(np.int8), states=np.asarray(states, np.float32),
        frames=np.asarray(frames, np.uint8), actions=np.asarray(actions, np.float32),
        events=np.asarray(events, np.float32), hit_ids=np.asarray(hit_ids, np.int64),
        attack_flags=np.asarray(attack_flags, np.float32), respawn_flags=np.asarray(respawn_flags, np.float32),
        object_masks=np.asarray(masks, bool), background_frames=np.asarray(backgrounds, np.uint8),
        depth=np.asarray(depth, np.float32), visibility=np.asarray(visibility, np.float32),
        initial_state=np.asarray(initial), seed=np.asarray(job['seed']), split=np.asarray(job['split']),
        stage=np.asarray(job['stage']), case=np.asarray(job['case']), combat_case=np.asarray(job['case']),
        enemies_enabled=np.asarray(enabled), navigation_only=np.asarray(not enabled),
        dt=np.asarray(1/30), hz=np.asarray(30), behavior_version=np.asarray(BEHAVIOR_VERSION),
        state_dim=np.asarray(STATE_DIM), entity_slots=np.asarray(ENTITY_SLOTS),
        state_schema_version=np.asarray('single_enemy_v1'),
        world_transition_compatible=np.asarray(True), collection_protocol=np.asarray(job['protocol']),
        format_version=np.asarray(4), policy=np.asarray('source_combat_detail_v1'))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **payload); temporary.replace(path)
    return record(path, job)


def record(path, job):
    with np.load(path, allow_pickle=False) as saved:
        steps = len(saved['actions'])
        assert saved['states'].shape == (steps+1, 20) and saved['frames'].shape == (steps+1, 32, 32, 3)
        assert np.isfinite(saved['states']).all() and np.isfinite(saved['actions']).all()
        counts = dict(zip(EVENT_NAMES, saved['events'].sum(0).astype(int).tolist()))
        counts['respawn'] = int(saved['respawn_flags'].sum())
        eligible = steps >= 451 and saved['states'][-1, 4] > 0
        terminal = bool(saved['states'][-1, 4] <= 0)
    return {'stage': job['stage'], 'split': job['split'], 'case': job['case'],
        'seed': job['seed'], 'relative_path': job['relative_path'], 'transitions': steps,
        'observations': steps+1, 'events': counts, 'eligible_450': bool(eligible),
        'terminal': terminal, 'sha256': digest(path), 'size_bytes': path.stat().st_size}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--short-train', type=int, default=512); parser.add_argument('--short-val', type=int, default=128)
    parser.add_argument('--long-train', type=int, default=256); parser.add_argument('--long-val', type=int, default=64)
    parser.add_argument('--nav-train', type=int, default=256); parser.add_argument('--nav-val', type=int, default=64)
    args = parser.parse_args(argv); root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy(); config['output'] = str(root)
    protocol = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    jobs = []
    for stage, train, val, frames, offset in [('moving1', args.short_train, args.short_val, 32, 0),
            ('full_game', args.long_train, args.long_val, 451, 100000),
            ('navigation', args.nav_train, args.nav_val, 32, 200000)]:
        for split, count, base in [('train', train, 94000000), ('val', val, 95000000)]:
            cases = NAV_CASES if stage == 'navigation' else COMBAT_CASES+('terminal',) if stage == 'moving1' else COMBAT_CASES
            for index in range(count):
                seed = base+offset+index; relative = f'{stage}/rgb/{split}/episode_{seed:09d}.npz'
                jobs.append(dict(stage=stage, split=split, index=index, seed=seed, frames=frames,
                    case=cases[index % len(cases)], relative_path=relative, path=str(root/relative), protocol=protocol))
    old = root/'manifest.json'
    if old.exists() and json.loads(old.read_text(encoding='utf-8'))['fingerprint'] != protocol:
        raise ValueError('Immutable combat corpus configuration differs')
    started = time.perf_counter(); records = []
    def publish(status):
        groups = {}
        for row in records:
            key = row['stage']+'/'+row['split']; item = groups.setdefault(key, {'files':0,'transitions':0,'eligible_450':0,'terminal':0,'events':Counter()})
            item['files'] += 1; item['transitions'] += row['transitions']; item['eligible_450'] += int(row['eligible_450'])
            item['terminal'] += int(row['terminal']); item['events'].update(row['events'])
        result = dict(format_version=1, status=status, fingerprint=protocol, config=config,
            behavior_version=BEHAVIOR_VERSION, state_dim=20, entity_slots=1, image_size=32, hz=30,
            completed_files=len(records), total_files=len(jobs), groups=groups,
            elapsed_seconds=time.perf_counter()-started, updated_at=now(),
            train_val_seed_overlap=False, source_physics_unchanged=True,
            autonomous_training_feedback=False, records=sorted(records,key=lambda x:x['relative_path']))
        atomic_json(root/'manifest.json',result)
        atomic_json(root/'status.json',{k:v for k,v in result.items() if k!='records'})
    publish('collecting')
    with KeepAwake(), ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(collect_job, job) for job in jobs]
        try:
            for future in as_completed(futures):
                records.append(future.result())
                if len(records)%32 == 0: publish('collecting')
        except BaseException:
            publish('failed'); raise
    publish('complete')
    print(json.dumps({'status':'complete','files':len(records),'seconds':time.perf_counter()-started}),flush=True)


if __name__ == '__main__': main()
