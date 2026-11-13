"""Immutable Source32/128 pairs and frozen-neural-renderer SR input caches.

Source targets are rendered at their native resolution at the recorded physical
tick.  The downstream cache adds ``lr_npr``; it must never replace Source pixels
or introduce the target RGB into a renderer appearance feature.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .codec_battle_data import render_condition_key
from .engine import BEHAVIOR_VERSION, Game
from .prepare_training_data import _atomic_json
from .small_enemy_data import _observe, digest

SOURCE_PROTOCOL = 'semantic_sr_exact_source_pairs_v1'
CACHE_PROTOCOL = 'semantic_sr_frozen_npr_pairs_v1'
COHORTS = ('near', 'mid', 'far', 'guard')
SOURCE_ARRAY_KEYS = ('lr_source', 'hr_source', 'background', 'hr_background',
    'mask_lr', 'mask_hr', 'state', 'map', 'previous_rgb', 'previous_state', 'action', 'dynamic')
IMAGE_KEYS = {'lr_npr', 'lr_source', 'hr_source', 'background', 'hr_background', 'previous_rgb', 'codec_lr'}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_cohort(state, mask, navigation=False):
    """Native visible area defines visibility; coordinates only define distance."""
    state = np.asarray(state)
    area = int(np.asarray(mask).sum())
    distance = float(np.linalg.norm((state[8:10] - state[:2]) * 16))
    alive = bool(state[13] >= .5 and state[12] > 0)
    if state[4] <= 0:
        return 'guard', 'hp0', distance, area
    if not alive:
        return 'guard', 'corpse' if area else 'navigation' if navigation else 'empty_dead', distance, area
    if not area:
        return 'guard', 'occluded', distance, area
    cohort = 'near' if distance < 2 else 'mid' if distance < 5 else 'far'
    band = '1_8' if area <= 8 else '9_16' if area <= 16 else '17_64' if area <= 64 else '65_plus'
    return cohort, band, distance, area


def inventory_source(roots, split, *, exclude_keys=(), exclude_rgb=()):
    """Read Source arrays once; labels and split filters never alter pixels."""
    files, rows, keys, rgb_keys, seeds, contents = [], [], set(), set(), set(), set()
    excluded_keys, excluded_rgb = set(exclude_keys), set(exclude_rgb)
    exclusions = Counter()
    paths = sorted({p.resolve() for root in roots for p in Path(root).rglob('*.npz') if p.parent.name == split})
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            required = {'states', 'frames', 'map', 'split', 'object_masks', 'background_frames', 'behavior_version', 'seed'}
            if not required.issubset(z.files):
                continue
            if str(z['split']) != split or str(z['behavior_version']) != BEHAVIOR_VERSION:
                raise ValueError('Source split/behavior differs from the current engine')
            states, frames, masks, background, grid = (z[k] for k in ('states','frames','object_masks','background_frames','map'))
            dynamic = 'actions' in z
            if (states.shape != (len(frames), 20) or frames.shape[1:] != (32,32,3)
                    or frames.dtype != np.uint8 or background.shape != frames.shape
                    or masks.shape != (len(frames),32,32) or masks.dtype != np.bool_
                    or grid.shape != (16,16) or not np.isfinite(states).all()
                    or not np.array_equal(masks, np.any(frames != background, axis=-1))):
                raise ValueError('Need exact native32 Source pixels, mask, state20 and map16')
            if (dynamic and ('initial_state' not in z or len(z['actions'])+1 != len(states))
                    or not dynamic and 'snapshot_states' not in z):
                continue  # Older paired-SR exports are not independent source tapes.
            navigation = (not bool(z['enemies_enabled']) if 'enemies_enabled' in z else False)
            navigation |= bool(z['navigation_only']) if 'navigation_only' in z else False
            navigation |= str(z['stage']) == 'navigation' if 'stage' in z else False
            navigation |= 'navigation' in path.parts
            seed = int(z['seed'])
        sha = digest(path)
        info = {'path':str(path),'sha256':sha,'seed':seed,'dynamic':bool(dynamic)}
        source_index = len(files); files.append(info); seeds.add(seed); contents.add(sha)
        for frame in range(1 if dynamic else 0, len(frames)):
            state = states[frame]
            key = render_condition_key(state, grid, enemies_enabled=not navigation).hex()
            rgb = hashlib.sha256(frames[frame].tobytes()).hexdigest()
            keys.add(key); rgb_keys.add(rgb)
            if key in excluded_keys:
                exclusions['render_condition'] += 1; continue
            if rgb in excluded_rgb:
                exclusions['source_rgb'] += 1; continue
            cohort, stratum, distance, area = source_cohort(state, masks[frame], navigation)
            rows.append({'source':source_index,'frame':int(frame),'cohort':cohort,'stratum':stratum,
                'distance_tiles':distance,'native_area':area,'render_key':key,'rgb_sha256':rgb,
                'source_seed':seed,'source_path':str(path),'source_sha256':sha,'dynamic':bool(dynamic)})
    return {'files':files,'rows':rows,'scene_keys':keys,'rgb_keys':rgb_keys,
        'seeds':seeds,'contents':contents,'excluded':dict(exclusions)}


def select_source_rows(found, count, *, seed=173):
    if type(count) is not int or count <= 0 or count % 4:
        raise ValueError('Pair counts must be positive multiples of four')
    rng = np.random.default_rng(seed); buckets = {c:{} for c in COHORTS}
    for row in found['rows']:
        buckets[row['cohort']].setdefault(row['stratum'], []).append(row)
    for groups in buckets.values():
        for values in groups.values():
            rng.shuffle(values)
    result, used_rgb, used_keys = [], set(), set()
    for cohort in COHORTS:
        assigned = Counter()
        for _ in range(count//4):
            available = [k for k,v in buckets[cohort].items() if v]
            while available:
                stratum = min(available, key=lambda k:(assigned[k], k))
                candidates = buckets[cohort][stratum]
                while candidates:
                    row = candidates.pop()
                    if row['rgb_sha256'] in used_rgb or row['render_key'] in used_keys:
                        continue
                    used_rgb.add(row['rgb_sha256']); used_keys.add(row['render_key'])
                    result.append(dict(row)); assigned[stratum] += 1; break
                else:
                    available.remove(stratum); continue
                break
            else:
                raise ValueError('Insufficient unique real Source frames for '+cohort)
    rng.shuffle(result)
    return result


def replay_source_job(job):
    """Replay once per tape, observing only selected target and preceding ticks."""
    info = job['source']; path = Path(info['path'])
    if digest(path) != info['sha256']:
        raise ValueError('Immutable Source tape changed')
    with np.load(path, allow_pickle=False) as z:
        arrays = {k:z[k] for k in ('states','frames','map','object_masks','background_frames')}
        if info['dynamic']:
            arrays.update(actions=z['actions'],events=z['events'])
            initial = json.loads(str(z['initial_state']))
            if abs(float(z['dt'])-1/30) > 1e-7:
                raise ValueError('Source pairing requires 30Hz recorded ticks')
        else:
            snapshots = json.loads(str(z['snapshot_states']))
    game = Game(seed=info['seed'], endless=True)
    game.map = arrays['map'].copy(); game.map_height, game.map_width = game.map.shape
    wanted = {r['frame'] for r in job['rows']}
    if info['dynamic']:
        wanted |= {frame-1 for frame in wanted}
    observed = {}
    def observe(frame):
        if not np.array_equal(game.state_vector(), arrays['states'][frame]):
            raise ValueError('Source replay state mismatch at tick '+str(frame))
        values = _observe(game)
        if (not np.array_equal(values[0], arrays['frames'][frame])
                or not np.array_equal(values[2], arrays['background_frames'][frame])
                or not np.array_equal(values[4], arrays['object_masks'][frame])):
            raise ValueError('Source native32 replay mismatch at tick '+str(frame))
        observed[frame] = values
    if info['dynamic']:
        game.set_state(initial)
        for frame in range(max(wanted)+1):
            if frame in wanted:
                observe(frame)
            if frame < max(wanted):
                events = game.step(arrays['actions'][frame], dt=1/30)
                if not np.array_equal(Game.event_vector(events), arrays['events'][frame]):
                    raise ValueError('Source event replay mismatch at tick '+str(frame))
    else:
        for frame in sorted(wanted):
            game.set_state(snapshots[frame]); observe(frame)
    output = []
    for record in job['rows']:
        frame = record['frame']; low, high, bg, hrbg, mask, hrmask = observed[frame]
        values = {'lr_source':low,'hr_source':high,'background':bg,'hr_background':hrbg,
            'mask_lr':mask,'mask_hr':hrmask,'state':arrays['states'][frame], 'map':arrays['map'],
            'previous_rgb':observed[frame-1][0] if info['dynamic'] else np.zeros((32,32,3),np.uint8),
            'previous_state':arrays['states'][frame-1] if info['dynamic'] else np.zeros(20,np.float32),
            'action':arrays['actions'][frame-1] if info['dynamic'] else np.zeros(4,np.float32),
            'dynamic':np.asarray(info['dynamic'], dtype=np.bool_)}
        output.append((record, values))
    return output


def write_shard(root, split, index, samples, *, extra_arrays=None):
    """Write a stable shard and return records; content is never silently replaced."""
    if not samples:
        raise ValueError('Cannot write an empty SR shard')
    arrays = {key:np.stack([values[key] for _,values in samples]) for key in SOURCE_ARRAY_KEYS}
    if extra_arrays:
        arrays.update(extra_arrays)
    relative = f'shards/{split}/pairs_{index:05d}.npz'
    target = Path(root)/relative; target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        with np.load(target, allow_pickle=False) as old:
            if set(old.files) != set(arrays) or any(not np.array_equal(old[k],v) for k,v in arrays.items()):
                raise ValueError('Existing immutable SR shard differs')
    else:
        temporary = target.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, **arrays); temporary.replace(target)
    sha = digest(target)
    records = [{**row, 'split':split,'shard':relative,'offset':int(offset)}
               for offset,(row,_) in enumerate(samples)]
    return records, {'path':relative,'relative_path':relative,'sha256':sha,
        'rows':len(samples),'count':len(samples),'split':split}


def collect_source_pairs(output, source_roots, *, train_count=8192, val_count=1024,
                         workers=4, seed=173, shard_size=64):
    root = Path(output).resolve(); root.mkdir(parents=True, exist_ok=True)
    config = {'protocol':SOURCE_PROTOCOL,'source_roots':[str(Path(p).resolve()) for p in source_roots],
        'train_count':train_count,'val_count':val_count,'seed':seed,'shard_size':shard_size,
        'image_size':32,'output_size':128,'behavior_version':BEHAVIOR_VERSION,
        'cohort_weights':dict.fromkeys(COHORTS,.25),'distance_edges_tiles':[2,5]}
    manifest_path = root/'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if old['config'] != config:
            raise ValueError('Immutable Source pairing plan changed')
        if old['status'] == 'complete':
            if any(digest(root/s['path']) != s['sha256'] for s in old['shards']):
                raise ValueError('Completed immutable Source shard changed')
            return old
    started = time.perf_counter()
    _atomic_json(root/'status.json', {'status':'indexing','stage':'data','current':0,
        'total':train_count+val_count,'config':config})
    train = inventory_source(source_roots, 'train')
    val = inventory_source(source_roots, 'val', exclude_keys=train['scene_keys'], exclude_rgb=train['rgb_keys'])
    if train['seeds'] & val['seeds'] or train['contents'] & val['contents']:
        raise ValueError('TRAIN/VAL Source seeds or source-file content overlap')
    selected = {'train':select_source_rows(train,train_count,seed=seed),
                'val':select_source_rows(val,val_count,seed=seed+1)}
    identity = {'config':config,'parents':{'train':train['files'],'val':val['files']},
        'selected':selected,'validation_exclusions':val['excluded']}
    plan_sha = fingerprint(identity)
    records, shards, jobs = [], [], []
    for split, found in [('train',train),('val',val)]:
        grouped = {}
        for index,row in enumerate(selected[split]):
            row['sample_index'] = index
            grouped.setdefault(row['source'], []).append(row)
        jobs.extend({'source':found['files'][i], 'rows':rows, 'split':split} for i,rows in grouped.items())
    pending = {'train':[],'val':[]}; shard_index = {'train':0,'val':0}
    def publish(status):
        report = {'status':status,'stage':'data','current':len(records),'total':train_count+val_count,
            'protocol':SOURCE_PROTOCOL,'config':config,'fingerprint':plan_sha,
            'records':sorted(records,key=lambda r:(r['split'],r['sample_index'])),'shards':shards,
            'groups':{split:{'frames':sum(r['split']==split for r in records),
                'cohorts':dict(Counter(r['cohort'] for r in records if r['split']==split)),
                'strata':dict(Counter(r['cohort']+'/'+r['stratum'] for r in records if r['split']==split))}
                for split in ('train','val')},
            'source32_replay_bitexact':True,'true_source128':True,'interpolated_hr':False,
            'paired_same_pose_map_tick':True,'future_rgb_features':False,
            'train_val_seed_overlap':False,'train_val_source_content_overlap':False,
            'train_val_rgb32_overlap':False,'train_val_render_condition_overlap':False,
            'validation_exclusions':val['excluded'],'elapsed_seconds':time.perf_counter()-started}
        _atomic_json(manifest_path,report)
        _atomic_json(root/'status.json',{k:v for k,v in report.items() if k not in ('records','shards')})
        return report
    publish('collecting')
    def accept(job, results):
        split = job['split']; pending[split].extend(results)
        while len(pending[split]) >= shard_size:
            samples = pending[split][:shard_size]; del pending[split][:shard_size]
            new, shard = write_shard(root,split,shard_index[split],samples)
            records.extend(new); shards.append(shard); shard_index[split] += 1
        if len(records)%512 < shard_size:
            publish('collecting')
    try:
        if workers == 1:
            for job in jobs:
                accept(job,replay_source_job(job))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                # Keep shard layout deterministic across worker timings/restarts.
                for job, results in zip(jobs,pool.map(replay_source_job,jobs,chunksize=1)):
                    accept(job,results)
        for split in pending:
            if pending[split]:
                new, shard = write_shard(root,split,shard_index[split],pending[split])
                records.extend(new); shards.append(shard)
    except BaseException:
        publish('failed'); raise
    # Actual shard hashes join the selection fingerprint, so cached inputs pin
    # the bytes rather than merely an intended collection configuration.
    final = publish('complete')
    final['data_sha256'] = fingerprint({'selection':plan_sha,'shards':sorted(shards,key=lambda s:s['path'])})
    _atomic_json(manifest_path,final)
    return final


class SourcePairData:
    """CPU/torch reader; explicit reads and fixed panels never advance sampling."""
    requires_neural_input = False
    def __init__(self, root, split, *, seed=173, device='cpu', exclude_scene_keys=None):
        if split not in ('train','val'):
            raise ValueError('Explicit TRAIN/VAL split required')
        self.root = Path(root).resolve(); self.split = split; self.device = torch.device(device)
        manifest = json.loads((self.root/'manifest.json').read_text(encoding='utf-8'))
        if manifest['status'] != 'complete' or not manifest.get('true_source128') or manifest.get('interpolated_hr'):
            raise ValueError('Need complete true Source128 pairs')
        if self.requires_neural_input and manifest.get('protocol') != CACHE_PROTOCOL:
            raise ValueError('Semantic SR requires an actual frozen-neural-renderer input cache')
        if self.requires_neural_input:
            cache_identity = manifest.get('cache_identity')
            if (not isinstance(cache_identity,dict) or cache_identity.get('future_rgb_features') is not False
                    or cache_identity.get('paired_same_pose_map_tick') is not True):
                raise ValueError('Semantic SR cache must explicitly pin causal inputs and same-tick Source targets')
        self.manifest = manifest; self.rng = np.random.default_rng(seed)
        self.rows = [dict(r) for r in manifest['records'] if r['split'] == split]
        excluded = {k.hex() if isinstance(k,bytes) else k for k in (exclude_scene_keys or ())}
        if any(r['render_key'] in excluded for r in self.rows):
            raise ValueError('TRAIN render condition leaked into held-out paired SR data')
        self.scene_keys = {r['render_key'] for r in self.rows}
        self.buckets = {c:[i for i,r in enumerate(self.rows) if r['cohort']==c] for c in COHORTS}
        if any(not ids for ids in self.buckets.values()):
            raise ValueError('Need near/mid/far/guard support in each SR split')
        self.draws = 0; self.case_counts = dict.fromkeys(COHORTS,0); self._cache = OrderedDict()
        self.shards = {}
        for shard in manifest['shards']:
            path = self.root/shard['path']
            if not path.is_file():
                raise ValueError('Missing immutable paired SR shard')
            stat = path.stat()
            self.shards[shard['path']] = {**shard,'stat':(stat.st_size,stat.st_mtime_ns)}
        self.identity = {'protocol':manifest['protocol'],'fingerprint':manifest.get('data_sha256',manifest['fingerprint']),
            'split':split,'frames':len(self.rows),'category_counts':{k:len(v) for k,v in self.buckets.items()},
            'true_source128':True,'interpolated_hr':False,'paired_same_pose_map_tick':True,
            'future_rgb_features':False,'train_val_render_condition_overlap':False,
            'neural_input':bool(self.requires_neural_input),'cache_identity':manifest.get('cache_identity')}

    def __len__(self):
        return len(self.rows)

    def fixed_panel(self, count=None, *, seed=173):
        count = len(self.rows) if count is None else count
        if type(count) is not int or count <= 0 or count > len(self.rows):
            raise ValueError('Need distinct held-out Source pairs')
        rng = np.random.default_rng(seed)
        buckets = {k:rng.permutation(v).tolist() for k,v in self.buckets.items()}
        panel = []; counts = dict.fromkeys(COHORTS,0)
        while len(panel) < count:
            available = [k for k,v in buckets.items() if v]
            cohort = min(available,key=lambda k:(counts[k],k))
            panel.append(int(buckets[cohort].pop())); counts[cohort] += 1
        rng.shuffle(panel)
        return panel

    def panel_fingerprint(self, panel):
        return fingerprint({'identity':self.identity,'indices':list(map(int,panel)),
            'records':[self.rows[int(i)] for i in panel]})

    def _read(self, name):
        path = self.root/name; info = self.shards[name]; stat = path.stat()
        if (stat.st_size,stat.st_mtime_ns) != info['stat']:
            raise ValueError('Immutable paired SR shard changed')
        if name not in self._cache:
            if digest(path) != info['sha256']:
                raise ValueError('Immutable paired SR shard hash changed')
            with np.load(path,allow_pickle=False) as z:
                self._cache[name] = {k:z[k] for k in z.files}
            while len(self._cache) > 4:
                self._cache.popitem(last=False)
        self._cache.move_to_end(name)
        return self._cache[name]

    def batch(self, size, indices=None, rng=None):
        if indices is None:
            rng = self.rng if rng is None else rng; indices = []
            for _ in range(size):
                cohort = min(COHORTS,key=lambda k:(self.case_counts[k],k))
                indices.append(int(rng.choice(self.buckets[cohort])))
                self.case_counts[cohort] += 1; self.draws += 1
            rng.shuffle(indices)
        if len(indices) != size:
            raise ValueError('Explicit SR batch/index size mismatch')
        samples = []
        for index in indices:
            row = self.rows[int(index)]; shard = self._read(row['shard']); offset = row['offset']
            required = SOURCE_ARRAY_KEYS + (('lr_npr',) if self.requires_neural_input else ())
            if not set(required).issubset(shard):
                raise ValueError('Incomplete Source/neural paired SR shard')
            sample = {k:shard[k][offset] for k in required}
            if 'codec_lr' in shard:
                sample['codec_lr'] = shard['codec_lr'][offset]
            samples.append(sample)
        result = {}
        for key in samples[0]:
            value = torch.from_numpy(np.stack([s[key] for s in samples]))
            if key in IMAGE_KEYS:
                value = value.permute(0,3,1,2).float()/255
            elif key in ('mask_lr','mask_hr'):
                value = value[:,None].float()
            elif key == 'map':
                value = value.long()
            elif key == 'dynamic':
                value = value.bool()
            else:
                value = value.float()
            result[key] = value.to(self.device)
        result.update(cohort=[self.rows[int(i)]['cohort'] for i in indices],
            stratum=[self.rows[int(i)]['stratum'] for i in indices],
            distance_tiles=torch.tensor([self.rows[int(i)]['distance_tiles'] for i in indices],device=self.device),
            native_area=torch.tensor([self.rows[int(i)]['native_area'] for i in indices],device=self.device),
            row_index=torch.tensor(list(map(int,indices)),device=self.device))
        # Renderer cache generation uses the established causal renderer schema.
        result.update(target=result['lr_source'], source=result['lr_source'], state=result['state'],
            object_mask=result['mask_lr'],hr_object_mask=result['mask_hr'],lr=result['lr_source'],hr=result['hr_source'])
        return result

    def sampler_state(self):
        return {'rng':deepcopy(self.rng.bit_generator.state),'draws':int(self.draws),'case_counts':dict(self.case_counts)}

    def restore_sampler(self, state):
        if (set(state.get('case_counts',{})) != set(COHORTS)
                or any(type(v) is not int or v < 0 for v in state['case_counts'].values())
                or type(state.get('draws')) is not int or state['draws'] != sum(state['case_counts'].values())):
            raise ValueError('Paired SR sampler clock/quotas changed')
        self.rng.bit_generator.state = deepcopy(state['rng']); self.draws = state['draws']
        self.case_counts = dict(state['case_counts'])


class SemanticSRData(SourcePairData):
    requires_neural_input = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path,action='append',required=True)
    parser.add_argument('--train-count',type=int,default=8192)
    parser.add_argument('--val-count',type=int,default=1024)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seed',type=int,default=173)
    parser.add_argument('--shard-size',type=int,default=64)
    args = parser.parse_args()
    report = collect_source_pairs(args.output,args.source,train_count=args.train_count,val_count=args.val_count,
        workers=args.workers,seed=args.seed,shard_size=args.shard_size)
    print(json.dumps({'status':report['status'],'fingerprint':report['fingerprint'],'groups':report['groups']}),flush=True)


if __name__ == '__main__':
    main()
