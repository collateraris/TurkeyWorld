"""Immutable TRAIN-only, full-speed Source corners and pinned contact replay.

Both concave arena corners and convex obstacle corners are genuine Source
geometry. Two perpendicular circle-clearances must be at most .08 tiles.
No free-tangent eligibility filter, pose repair, heldout copy or runtime oracle
is used. Source is called only while recording these offline labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
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
from .rssm_contact_data import (CASE_NAMES, PROFILE, _clear_of, _excluded_positions,
    _replace, _source_pose, contact_cases, first16_cohorts)

VERSION = 1
BRANCH_NAMES = ('inward_both', 'inward_x_outward_y', 'outward_x_inward_y',
    'outward_both', 'move_turn_positive', 'move_turn_negative',
    'block_release_reapproach', 'idle_away_turn_return')
AXIS_OUTCOMES = ('free_free', 'free_blocked', 'blocked_free', 'blocked_blocked')
QUADRANTS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
CLEARANCES = (.005, .01, .02, .04, .06, .075)


def corner_geometry(game, state, maximum=.08):
    """Exact static circle-sweep distances along worldX/Y (offline only)."""
    point = np.asarray(state[:2], dtype=float)*16
    if not game._walkable(*point, game.PLAYER_RADIUS):
        return None
    walls = np.argwhere(game.map != 0)[:, ::-1].astype(float)
    distances = []
    for normal in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        axis, other = int(abs(normal[1])), 1-int(abs(normal[1]))
        lateral = np.maximum(np.maximum(walls[:, other]-point[other], point[other]-walls[:, other]-1), 0)
        extension = np.sqrt(np.maximum(game.PLAYER_RADIUS**2-lateral**2, 0))
        distance = (walls[:, axis]-extension-point[axis] if normal[axis] > 0
                    else point[axis]-walls[:, axis]-1-extension)
        valid = (lateral < game.PLAYER_RADIUS) & (distance >= -2e-6)
        distances.append(float(distance[valid].min()) if valid.any() else 99.)
    if min(distances[:2]) > maximum+2e-6 or min(distances[2:]) > maximum+2e-6:
        return None
    sx = 1 if distances[0] <= distances[1] else -1
    sy = 1 if distances[2] <= distances[3] else -1
    return {'corner_normals': [sx, sy], 'corner_clearances_tiles':
        [distances[0 if sx > 0 else 1], distances[2 if sy > 0 else 3]],
        'cardinal_clearances_tiles': distances}


def axis_outcomes(fractions, active, length=16):
    """Actual Source outcomes; only transitions with BOTH active axes qualify."""
    fractions, active = np.asarray(fractions[:length]), np.asarray(active[:length], bool)
    both = active.all(-1)
    free, blocked = fractions >= .95, np.abs(fractions) <= .05
    masks = (both & free.all(-1), both & free[:, 0] & blocked[:, 1],
             both & blocked[:, 0] & free[:, 1], both & blocked.all(-1))
    counts = {key: int(mask.sum()) for key, mask in zip(AXIS_OUTCOMES, masks)}
    counts['partial_or_unclassified_both_axes'] = int((both & ~np.logical_or.reduce(masks)).sum())
    counts['single_or_inactive_axes'] = int((~both).sum())
    return [key for key in AXIS_OUTCOMES if counts[key]], counts


def action_for(branch, step, normals, angle):
    """Full-speed desired world direction mapped AFTER the typed signed turn."""
    sx, sy = normals
    directions = ((sx, sy), (sx, -sy), (-sx, sy), (-sx, -sy))
    turn = 0.
    if branch < 4:
        direction = np.asarray(directions[branch], dtype=float)
    elif branch in (4, 5):
        turn = .75 if branch == 4 else -.75
        # Both slide orders occur before and after a signed turn.
        direction = np.asarray(directions[1 if (step//8+branch) % 2 else 2], dtype=float)
    elif branch == 6:
        direction = np.asarray(directions[0 if step < 8 or 16 <= step < 24 else 3], dtype=float)
        if step >= 24: direction *= 0
    elif branch == 7:
        direction = np.asarray(directions[3 if step < 12 else 0 if step < 20 else 1], dtype=float)
        if step < 4 or step >= 28: direction *= 0
        turn = .6 if 4 <= step < 12 else -.6 if 12 <= step < 20 else 0.
    else:
        raise ValueError('Corner Source branch must be in0..7')
    norm = np.linalg.norm(direction)
    if norm: direction /= norm
    post = angle+turn*2.2/30
    c, s = math.cos(post), math.sin(post)
    return np.asarray((c*direction[0]+s*direction[1], -s*direction[0]+c*direction[1], turn, 0), np.float32)


def _pose(game, point, angle, kind, **provenance):
    state = game.get_state()
    state['player'].update(x=float(point[0]), y=float(point[1]), angle=float(angle))
    game.set_state(state)
    vector = _source_pose(game, game.state_vector())
    geometry = corner_geometry(game, vector)
    if geometry is None:
        return None
    return {'state': vector.tolist(), 'kind': kind, **geometry, **provenance}


def create_plan(config):
    game = Game(seed=0, endless=True, enemies_enabled=False)
    forbidden = []
    for root in config['exclude_data_roots']:
        forbidden.extend(_excluded_positions(root)['val'])
    buckets = {q: {'legacy': [], 'concave': [], 'convex': []} for q in QUADRANTS}
    seen = set()
    def add(pose, kind):
        if pose is None: return
        xy = np.asarray(pose['state'][:2])*16
        key = tuple(np.round(xy, 5))
        if key in seen or not _clear_of(xy, forbidden, config['exclusion_radius']): return
        seen.add(key)
        buckets[tuple(pose['corner_normals'])][kind].append(pose)
    # Reuse only declared TRAIN states, and recheck ALL heldout neighborhoods.
    for root in config['reuse_train_roots']:
        old = json.loads((Path(root)/'plan.json').read_text(encoding='utf-8'))
        for name, candidate in old['poses'].items():
            if '/train/' not in name or 'corner' not in candidate['kind']: continue
            vector = _source_pose(game, candidate['state'])
            geometry = corner_geometry(game, vector)
            if geometry:
                add({'state': vector.tolist(), 'kind': 'reused_Source_TRAIN_corner', **geometry,
                    'source_train_root': root, 'source_train_pose_id': name}, 'legacy')
    headings = (math.pi-.005, -math.pi+.005, 0., math.pi/2, -math.pi/2,
                math.pi/4, -math.pi/4, 3*math.pi/4, -3*math.pi/4)
    ordinal = 0
    for row, col in np.argwhere(game.map == 0):
        for sx, sy in QUADRANTS:
            if not (game.map[row, col+sx] and game.map[row+sy, col]): continue
            for cx in CLEARANCES:
                for cy in CLEARANCES:
                    point = np.asarray((col+.5+sx*(.3-cx), row+.5+sy*(.3-cy)))
                    add(_pose(game, point, headings[ordinal % len(headings)], 'Source_concave_corner_augmentation',
                              source_floor_cell=[int(col), int(row)], requested_clearances_tiles=[cx, cy]), 'concave')
                    ordinal += 1
    # A rounded visible obstacle corner also has two perpendicular approach
    # constraints, and permits new quadrants outside reserved arena corners.
    for row, col in np.argwhere(game.map != 0):
        for sx, sy in QUADRANTS:
            vertex = np.asarray((col+(sx < 0), row+(sy < 0)), dtype=float)
            for gap in (.0025, .005, .01, .02, .04):
                for approach in (math.pi/6, math.pi/4, math.pi/3):
                    point = vertex-np.asarray((sx*math.cos(approach), sy*math.sin(approach)))*(.2+gap)
                    if not game._walkable(*point, game.PLAYER_RADIUS): continue
                    add(_pose(game, point, headings[ordinal % len(headings)], 'Source_convex_obstacle_corner_augmentation',
                              source_wall_vertex=vertex.tolist(), radial_clearance_tiles=gap,
                              corner_approach_radians=approach), 'convex')
                    ordinal += 1
    rng = np.random.default_rng(config['seed'])
    available = {str(q): {k: len(v) for k,v in kinds.items()} for q,kinds in buckets.items()}
    for kinds in buckets.values():
        for values in kinds.values(): rng.shuffle(values)
    selected = []
    for i in range(config['train_poses']):
        quadrant = QUADRANTS[i % 4]
        kinds = buckets[quadrant]
        choice = 'legacy' if kinds['legacy'] else ('concave' if (i//4) % 2 == 0 else 'convex')
        if not kinds[choice]: choice = 'convex' if kinds['convex'] else 'concave'
        if not kinds[choice]:
            raise ValueError('Not enough distinct legal corner poses in quadrant '+str(quadrant)+
                             ' outside the unchanged heldout exclusion radius; do not relax it silently')
        selected.append(kinds[choice].pop())
    poses, jobs = {}, []
    for index, pose in enumerate(selected):
        pose_id = f'corner/train/pose_{index:05d}'
        poses[pose_id] = pose
        for branch in range(8):
            seed = config['seed']+index*8+branch
            jobs.append({'stage':'short','split':'train','index':index*8+branch,'seed':seed,
                'frames':32,'branch':branch,'pose_id':pose_id,'origin':'new_Source_corner','pool':'corner',
                'relative_path':f'short/rgb/train/episode_{seed:010d}.npz'})
    if config['replay_root']:
        source = Path(config['replay_root'])
        replay = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
        if replay.get('status') != 'complete': raise ValueError('Replay corpus must be complete')
        for record in replay['records']:
            if record['split'] != 'train' or record['stage'] != 'short':
                raise ValueError('Composite replay must contain ONLY short TRAIN episodes')
            original = (source/record['relative_path']).resolve(strict=True)
            if source not in original.parents or digest(original) != record['sha256']:
                raise ValueError('Replay source changed or escaped its immutable root')
            with np.load(original, allow_pickle=False) as saved:
                if str(saved['split']) != 'train' or not _clear_of(saved['states'][0,:2]*16, forbidden, config['exclusion_radius']):
                    raise ValueError('Replay source violates TRAIN split or heldout initial neighborhood exclusion')
            jobs.append({'stage':'short','split':'train','index':len(jobs),'seed':record['seed'],
                'frames':32,'branch':record['branch'],'pose_id':'replay/'+record['pose_id'],
                'origin':'exact_replay_TRAIN','pool':'singlewall',
                'relative_path':'short/rgb/train/replay_'+original.name,
                'source_relative_path':record['relative_path'],'source_sha256':record['sha256'],
                'source_pose_id':record['pose_id']})
    minimum = min((float(np.linalg.norm(np.asarray(p['state'][:2])*16-q)) for p in selected for q in forbidden), default=None)
    return {'config':config,'fingerprint':fingerprint(config),'poses':poses,'jobs':jobs,
        'available_corner_geometry':available,'selected_corner_kinds':dict(Counter(p['kind'] for p in selected)),
        'selected_corner_quadrants':dict(Counter(str(tuple(p['corner_normals'])) for p in selected)),
        'old_val_pose_neighborhoods_excluded':len(forbidden),'minimum_initial_xy_distance_to_old_val_tiles':minimum,
        'heldout_inputs_used_for_train':False,'old_data_modified':False,'validation_clips':0}


def record_episode(job, pose):
    game = Game(seed=job['seed'], endless=True, enemies_enabled=False)
    _source_pose(game, pose['state'])
    initial = json.dumps(game.get_state())
    states, images, actions, events = [game.state_vector().copy()], [game.render(size=32, overlays=False)], [], []
    for step in range(32):
        action = action_for(job['branch'], step, pose['corner_normals'], game.player['angle'])
        event = game.step(action, dt=1/30)
        if game.done: raise RuntimeError('Source corner terminated; no reset/repair is allowed')
        states.append(game.state_vector().copy());images.append(game.render(size=32, overlays=False))
        actions.append(action);events.append(Game.event_vector(event))
    states, images, actions = np.asarray(states,np.float32), np.asarray(images,np.uint8), np.asarray(actions,np.float32)
    command, fractions, active, cases, audit = contact_cases(states,actions)
    cohort, members, first_counts = first16_cohorts(cases)
    if cases[:16,3:5].any(): cohort = 'turn_idle'
    if cohort == 'partial': cohort = 'slide' if first_counts['slide'] else 'blocked' if first_counts['blocked'] else 'free'
    outcomes, counts = axis_outcomes(fractions,active)
    starts = np.arange(17,dtype=np.int32)
    return {'map':game.map.astype(np.int8),'states':states,'frames':images,'actions':actions,
        'events':np.asarray(events,np.float32),'hit_ids':np.full(32,-1,np.int64),
        'attack_flags':np.zeros((32,1),np.float32),'respawn_flags':np.zeros((32,1),np.float32),
        'object_masks':np.zeros(images.shape[:3],bool),'background_frames':images.copy(),
        'initial_state':np.asarray(initial),'axis_command_tiles':command,'axis_pass_targets':fractions,
        'axis_command_active':active,'contact_case_names':np.asarray(CASE_NAMES),'contact_cases':cases,
        'contact_first16_cohort':np.asarray(cohort),'contact_sample_cohorts':np.asarray(members),
        'contact_window_starts_h16':starts,'contact_window_cases_h16':np.asarray([cases[i:i+16].any(0) for i in starts]),
        'corner_axis_outcomes':np.asarray(outcomes,dtype='U32'),
        'corner_axis_outcome_names':np.asarray(list(counts)),
        'corner_axis_outcome_counts_first16':np.asarray(list(counts.values()),np.int32)}, audit


def _job(arguments):
    root, config, identity, pose, job = arguments
    path = Path(root)/job['relative_path'];reused = path.exists()
    if job['pool'] == 'singlewall':
        source = Path(config['replay_root'])/job['source_relative_path']
        if digest(source) != job['source_sha256']: raise ValueError('Pinned TRAIN replay changed')
        storage = 'existing'
        if not reused:
            path.parent.mkdir(parents=True,exist_ok=True)
            try: os.link(source,path);storage='hardlink'
            except OSError:
                temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
                try: shutil.copy2(source,temporary);_replace(temporary,path)
                finally: temporary.unlink(missing_ok=True)
                storage='copy'
        if digest(path) != job['source_sha256']: raise ValueError('Composite replay bytes differ')
    else:
        expected = _source_pose(Game(seed=job['seed'],endless=True,enemies_enabled=False),pose['state'])
        if not reused:
            payload,_ = record_episode(job,pose)
            payload.update(format_version=np.asarray(4),navigation_recovery_version=np.asarray(1),
                navigation_recovery_fingerprint=np.asarray(identity),rssm_contact_version=np.asarray(3),
                rssm_corner_version=np.asarray(VERSION),behavior_version=np.asarray(BEHAVIOR_VERSION),
                environment_profile=np.asarray(PROFILE),enemies_enabled=np.asarray(False),state_dim=np.asarray(STATE_DIM),
                entity_slots=np.asarray(1),state_schema_version=np.asarray('single_enemy_v1'),split=np.asarray('train'),
                stage=np.asarray('navigation'),movement_case=np.asarray('corner'),dataset_kind=np.asarray('world_transitions'),
                world_transition_compatible=np.asarray(True),dt=np.asarray(1/30),endless=np.asarray(True),seed=np.asarray(job['seed']),
                pose_id=np.asarray(job['pose_id']),action_branch=np.asarray(job['branch']),action_branch_name=np.asarray(BRANCH_NAMES[job['branch']]),
                initialization_source=np.asarray(pose['kind']),source_frame_count=np.asarray(33),frame_source=np.asarray('reference_engine'),
                state_repairs=np.asarray(0),resets_inside_trajectory=np.asarray(0),corner_normals=np.asarray(pose['corner_normals']),
                corner_clearances_tiles=np.asarray(pose['corner_clearances_tiles']))
            path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
            try:
                with temporary.open('wb') as stream: np.savez_compressed(stream,**payload)
                _replace(temporary,path)
            finally: temporary.unlink(missing_ok=True)
        storage='existing' if reused else 'new_Source'
    with np.load(path,allow_pickle=False) as saved:
        if (str(saved['split'])!='train' or saved['states'].shape!=(33,STATE_DIM) or saved['actions'].shape!=(32,4)
            or str(saved['behavior_version'])!=BEHAVIOR_VERSION): raise ValueError('Corner/composite immutable schema differs')
        if job['pool']=='corner' and (str(saved['navigation_recovery_fingerprint'])!=identity
            or str(saved['pose_id'])!=job['pose_id'] or not np.array_equal(saved['states'][0],expected)):
            raise ValueError('Existing corner episode differs from exact Source-canonical immutable plan')
        _,fractions,active,cases,audit = contact_cases(saved['states'],saved['actions'])
        if not np.array_equal(cases,saved['contact_cases']): raise ValueError('Source labels changed')
        outcomes,outcome_counts=axis_outcomes(fractions,active)
        if job['pool']=='corner' and list(saved['corner_axis_outcomes'])!=outcomes:
            raise ValueError('Source corner outcome membership differs')
        cohort=str(saved['contact_first16_cohort']);_,members,counts=first16_cohorts(cases)
    return {**job,'path':str(path),'sha256':digest(path),'bytes':path.stat().st_size,'reused':reused,'storage':storage,
        'cohort':cohort,'sample_cohorts':members,'first16_case_counts':counts,'axis_outcomes_first16':outcomes,
        'axis_outcome_counts_first16':outcome_counts,'audit':audit}


def collect(root, *, train_poses=256, workers=4, seed=160_000_000,
            exclude_data_roots=(), reuse_train_roots=(), exclusion_radius=.25,replay_root=None):
    if (not isinstance(train_poses,int) or isinstance(train_poses,bool) or train_poses<1
        or not isinstance(workers,int) or isinstance(workers,bool) or workers<1
        or not isinstance(seed,int) or isinstance(seed,bool) or seed<0
        or not math.isfinite(exclusion_radius) or exclusion_radius<.25):
        raise ValueError('Corner counts/seed/workers must be valid; heldout exclusion may not be less than .25tiles')
    root=Path(root).resolve()
    excluded=list(dict.fromkeys(str(Path(p).resolve(strict=True)) for p in exclude_data_roots))
    reused=list(dict.fromkeys(str(Path(p).resolve(strict=True)) for p in reuse_train_roots))
    for path in reused:
        if path not in excluded: excluded.append(path)
    replay=str(Path(replay_root).resolve(strict=True)) if replay_root else None
    protected=[Path(p) for p in excluded+reused+([replay] if replay else [])]
    if any(root==p or root in p.parents or p in root.parents for p in protected):
        raise ValueError('Corner output must be separate from every immutable input corpus')
    config={'version':VERSION,'behavior_version':BEHAVIOR_VERSION,'profile':PROFILE,'seed':seed,'train_poses':train_poses,
        'exclude_data_roots':excluded,'excluded_plan_sha256':[digest(Path(p)/'plan.json') for p in excluded],
        'reuse_train_roots':reused,'replay_root':replay,'replay_manifest_sha256':digest(Path(replay)/'manifest.json') if replay else None,
        'exclusion_radius':float(exclusion_radius),'image_size':32,'transitions_per_clip':32,'branches':list(BRANCH_NAMES)}
    plan_path=root/'plan.json';manifest_path=root/'manifest.json'
    if plan_path.exists():
        plan=json.loads(plan_path.read_text(encoding='utf-8'))
        if plan['config']!=config: raise ValueError('Corner/composite plan is immutable; use a new root')
        if manifest_path.exists():
            completed=json.loads(manifest_path.read_text(encoding='utf-8'))
            if completed.get('status')=='complete':
                if completed.get('config')!=config or completed.get('fingerprint')!=plan['fingerprint']:
                    raise ValueError('Completed corner identity changed')
                listed=set()
                for record in completed['records']:
                    p=(root/record['relative_path']).resolve(strict=True);listed.add(p)
                    if root not in p.parents or digest(p)!=record['sha256']: raise ValueError('Completed corner bytes changed')
                if listed!={p.resolve() for p in root.rglob('*.npz')}: raise ValueError('Unlisted corner/composite NPZ')
                return {**completed,'verified_existing':True}
    else:
        plan=create_plan(config);root.mkdir(parents=True,exist_ok=True);_atomic_json(plan_path,plan)
    started=time.perf_counter();records=[]
    tasks=[(str(root),config,plan['fingerprint'],plan['poses'].get(j['pose_id']),j) for j in plan['jobs']]
    executor=ProcessPoolExecutor(max_workers=workers) if workers>1 else None
    try:
        for record in (executor.map(_job,tasks) if executor else map(_job,tasks)):
            records.append(record)
            if len(records)%8==0 or len(records)==len(tasks):
                elapsed=time.perf_counter()-started
                _atomic_json(root/'status.json',{'status':'running','completed':len(records),'total':len(tasks),
                    'elapsed_seconds':elapsed,'eta_seconds':elapsed/len(records)*(len(tasks)-len(records)),
                    'fingerprint':plan['fingerprint'],'last_file':record['relative_path']})
    finally:
        if executor: executor.shutdown(wait=True,cancel_futures=True)
    manifest={k:v for k,v in plan.items() if k not in ('poses','jobs')}
    pools={pool:[r['relative_path'] for r in records if r['pool']==pool] for pool in ('corner','singlewall')}
    members={key:sum(key in r['axis_outcomes_first16'] for r in records if r['pool']=='corner') for key in AXIS_OUTCOMES}
    if not all(members.values()): raise RuntimeError('Corner Source corpus lacks an actual per-axis outcome cohort')
    keys=set().union(*(r['audit'] for r in records))
    manifest.update(status='complete',completed=len(records),total=len(tasks),records=records,sampling_pools=pools,
        groups={'short/train':{'clips':len(records),'transitions':len(records)*32,'corner_clips':len(pools['corner']),
            'singlewall_exact_replay_clips':len(pools['singlewall'])},'short/val':{'clips':0,'transitions':0}},
        corner_axis_outcome_clip_memberships=members,
        corner_axis_outcome_step_counts={key:sum(r['axis_outcome_counts_first16'][key] for r in records if r['pool']=='corner')
            for key in list(AXIS_OUTCOMES)+['partial_or_unclassified_both_axes','single_or_inactive_axes']},
        audit={key:sum(r['audit'].get(key,0) for r in records) for key in sorted(keys)},
        elapsed_seconds=time.perf_counter()-started,label_source='offline continuous Source32; fullspeed corner branches; no runtime oracle',
        replay_source='TRAIN exact bytes; pool labels exist in manifest only; no original data changes')
    _atomic_json(manifest_path,manifest);_atomic_json(root/'status.json',{k:v for k,v in manifest.items() if k not in ('records','sampling_pools')})
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--train-poses',type=int,default=256)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seed',type=int,default=160_000_000)
    parser.add_argument('--exclude-data-root',type=Path,action='append',default=[])
    parser.add_argument('--reuse-train-root',type=Path,action='append',default=[])
    parser.add_argument('--exclusion-radius',type=float,default=.25)
    parser.add_argument('--replay-root',type=Path)
    args=parser.parse_args(argv)
    report=collect(args.root,train_poses=args.train_poses,workers=args.workers,seed=args.seed,
        exclude_data_roots=args.exclude_data_root,reuse_train_roots=args.reuse_train_root,
        exclusion_radius=args.exclusion_radius,replay_root=args.replay_root)
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','sampling_pools')},indent=2),flush=True)


if __name__=='__main__': main()
