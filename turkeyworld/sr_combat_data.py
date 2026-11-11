"""True Source32/128 pairs replayed at the identical recorded physical tick."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .engine import BEHAVIOR_VERSION, Game
from .prepare_training_data import _atomic_json
from .sr_data import SRPairData
from .staged_data import _observe


COHORTS = ('hit', 'miss', 'attack', 'death', 'respawn', 'terminal', 'blocked_melee')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_pairs(source_roots, split, count, seed=173):
    """Unique target ticks,25% navigation and balanced genuine event cohorts."""
    if split not in ('train', 'val') or count < 8 or count % 4:
        raise ValueError('Use explicit train/val and a pair count divisible by4')
    groups = {name: [] for name in COHORTS+('navigation',)}
    identities = []
    for source_root in source_roots:
        for path in sorted(Path(source_root).rglob('*.npz')):
            if path.parent.name != split: continue
            with np.load(path, allow_pickle=False) as z:
                if ('initial_state' not in z or 'actions' not in z or str(z['split']) != split
                        or str(z['behavior_version']) != BEHAVIOR_VERSION):
                    raise ValueError('SR pairs need exact current Source episodes and split provenance')
                states, events = z['states'], z['events']
                if states.shape != (len(events)+1, 20): raise ValueError('Source20 event alignment')
                stage, source_seed = str(z['stage']), int(z['seed'])
                respawn = ((states[:-1, 13] < .5) & (states[1:, 13] >= .5))
                negative = z['source_hard_negative_flags'][:, 0] if 'source_hard_negative_flags' in z else np.zeros(len(events), bool)
                masks = dict(hit=events[:, 1] > .5, miss=(events[:, 0] > .5)&(events[:, 1] < .5),
                    attack=events[:, 3] > .5, death=events[:, 2] > .5, respawn=respawn,
                    terminal=states[1:, 4] <= 0, blocked_melee=negative,
                    navigation=np.full(len(events), stage == 'navigation'))
                identity = dict(path=str(path.resolve()), sha256=digest(path), seed=source_seed)
                identities.append(identity)
                for name, mask in masks.items():
                    for index in np.flatnonzero(mask):
                        groups[name].append((len(identities)-1, int(index)+1))
    if any(not groups[name] for name in groups):
        raise ValueError('All actual combat/HP0/navigation cohorts are required: '+str({k:len(v)for k,v in groups.items()}))
    rng = np.random.default_rng(seed+(split == 'val'))
    used, selected = set(), []

    def take(name, limit):
        available = [pair for pair in groups[name] if pair not in used]
        rng.shuffle(available)
        for pair in available[:limit]:
            used.add(pair); selected.append(dict(source=pair[0], frame=pair[1], cohort=name))
        return min(limit, len(available))

    nav_count = count//4
    if take('navigation', nav_count) != nav_count: raise ValueError('Insufficient unique navigation targets')
    # Every rare true terminal observation is retained once when capacity permits.
    combat_count = count-nav_count
    terminal = min(len(groups['terminal']), combat_count//len(COHORTS) if split == 'train' else combat_count)
    take('terminal', terminal)
    assigned = Counter(row['cohort'] for row in selected)
    while len(selected) < count:
        names = sorted(COHORTS, key=lambda name: (assigned[name], name))
        for name in names:
            if take(name, 1):
                assigned[name] += 1
                break
        else: raise ValueError('Insufficient unique real combat targets')
    rng.shuffle(selected)
    return identities, selected


def replay_pairs(job):
    source = Path(job['source']['path'])
    if digest(source) != job['source']['sha256']: raise ValueError('Immutable Source changed')
    with np.load(source, allow_pickle=False) as z:
        states, frames, actions, events, grid = (z[key] for key in ('states','frames','actions','events','map'))
        initial = str(z['initial_state']); split = str(z['split']); source_seed = int(z['seed'])
    game = Game(seed=source_seed, endless=True)
    if not np.array_equal(game.map, grid): raise ValueError('Source map differs from recorded engine map')
    game.set_state(json.loads(initial))
    wanted = {frame for row in job['rows'] for frame in (row['frame']-1, row['frame'])}
    observed = {}
    for index in range(max(wanted)+1):
        if not np.array_equal(game.state_vector(), states[index]):
            raise ValueError('Source replay state mismatch at tick'+str(index))
        if index in wanted:
            lr = game.render(size=32, overlays=False)
            if not np.array_equal(lr, frames[index]): raise ValueError('Stored native32 RGB is not exact Source replay')
            hr, mask, _ = _observe(game, 128)
            observed[index] = (lr, hr, mask)
        if index < max(wanted):
            event = game.step(actions[index], dt=1/30)
            if not np.array_equal(Game.event_vector(event), events[index]): raise ValueError('Source event replay mismatch')
    records = []
    for row in job['rows']:
        frame = row['frame']; indices = [frame-1, frame]
        relative = f"rgb/{split}/pair_{source_seed:09d}_{frame:04d}.npz"
        path = Path(job['output'])/relative; path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, frames=np.stack([observed[i][0] for i in indices]),
                hr_frames=np.stack([observed[i][1] for i in indices]),
                hr_object_masks=np.stack([observed[i][2] for i in indices]),
                states=states[indices], actions=actions[frame-1:frame], map=grid,
                split=np.asarray(split), seed=np.asarray(source_seed), source_frame_indices=np.asarray(indices),
                source_path=np.asarray(str(source)), source_sha256=np.asarray(job['source']['sha256']),
                cohort=np.asarray(row['cohort']), hr_frame_source=np.asarray('reference_engine'),
                frame_source=np.asarray('reference_engine'), paired_source_rgb=np.asarray(True),
                behavior_version=np.asarray(BEHAVIOR_VERSION), state_dim=np.asarray(20), entity_slots=np.asarray(1),
                collection_protocol=np.asarray(job['fingerprint']), source_replay_bitexact=np.asarray(True))
            temporary.replace(path)
        else:
            with np.load(path) as saved:
                if str(saved['collection_protocol']) != job['fingerprint']: raise ValueError('SR pair cache identity differs')
        records.append(dict(relative_path=relative, split=split, source_seed=source_seed,
            source_path=str(source), source_frame=frame, cohort=row['cohort'], sha256=digest(path)))
    return records


def collect_pairs(output, source_roots, *, train_pairs=3072, val_pairs=384, workers=4, seed=173):
    root = Path(output).resolve(); root.mkdir(parents=True, exist_ok=True)
    config = dict(source_roots=[str(Path(p).resolve()) for p in source_roots], train_pairs=train_pairs,
        val_pairs=val_pairs, seed=seed, behavior_version=BEHAVIOR_VERSION, image_size=32, hr_size=128)
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    manifest = root/'manifest.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved['fingerprint'] != fingerprint: raise ValueError('Immutable SR pair plan differs')
        if saved['status'] == 'complete':
            if any(digest(root/r['relative_path']) != r['sha256'] for r in saved['records']):
                raise ValueError('Paired SR Source data changed')
            return saved
    jobs, source_seeds = [], {}
    for split, count in (('train',train_pairs),('val',val_pairs)):
        sources, selected = select_pairs(source_roots, split, count, seed)
        source_seeds[split] = {item['seed'] for item in sources}
        for index, source in enumerate(sources):
            rows = [row for row in selected if row['source'] == index]
            if rows: jobs.append(dict(source=source, rows=rows, output=str(root), fingerprint=fingerprint))
    if source_seeds['train'] & source_seeds['val']: raise ValueError('Source train/val seed overlap')
    start, records = time.perf_counter(), []
    def publish(status):
        report = dict(status=status, fingerprint=fingerprint, config=config, records=records,
            completed_pairs=len(records), total_pairs=train_pairs+val_pairs, groups={split:dict(
                pairs=sum(r['split']==split for r in records),
                cohorts=dict(Counter(r['cohort'] for r in records if r['split']==split))) for split in ('train','val')},
            source_lr_replay_bitexact=True, true_source_hr=True, interpolated_hr=False,
            source_training_validation_seed_overlap=False, elapsed_seconds=time.perf_counter()-start)
        _atomic_json(manifest, report); _atomic_json(root/'status.json',{k:v for k,v in report.items() if k!='records'})
        return report
    publish('collecting')
    try:
        if workers == 1:
            for job in jobs: records.extend(replay_pairs(job)); publish('collecting')
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for future in as_completed([executor.submit(replay_pairs,job) for job in jobs]):
                    records.extend(future.result())
                    if len(records)%32<8: publish('collecting')
    except BaseException:
        publish('failed'); raise
    records.sort(key=lambda row:row['relative_path'])
    return publish('complete')


class SRCombatPairData(SRPairData):
    """Use only target frame1; frame0 proves the adjacent Source-state pairing."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.cohorts=[]
        for episode in self.episodes:
            with np.load(episode['path']) as z: self.cohorts.append(str(z['cohort']))

    def fixed_panel(self,count=None,seed=373):
        count=len(self.episodes) if count is None else count
        if not 0<count<=len(self.episodes): raise ValueError('Need unique held-out SR target pairs')
        rng=np.random.default_rng(seed)
        terminal=[i for i,c in enumerate(self.cohorts) if c=='terminal']
        rest=[i for i in range(len(self.episodes)) if i not in terminal];rng.shuffle(rest)
        if len(terminal)>count:raise ValueError('Panel must include all terminal frames')
        return [(i,1) for i in (terminal+rest)[:count]]

    def batch(self,batch_size,rng=None,indices=None):
        if indices is None:
            rng=self.rng if rng is None else rng
            indices=np.column_stack([rng.integers(len(self.episodes),size=batch_size),np.ones(batch_size,int)])
        result=super().batch(batch_size,indices=indices)
        masks=[]
        for index,frame in indices:
            with np.load(self.episodes[int(index)]['path']) as z:masks.append(z['hr_object_masks'][int(frame)])
        import torch
        result['hr_object_mask']=torch.from_numpy(np.stack(masks))[:,None].to(self.device,dtype=torch.float32)
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path,action='append',required=True)
    parser.add_argument('--train-pairs',type=int,default=3072);parser.add_argument('--val-pairs',type=int,default=384)
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--seed',type=int,default=173)
    args=parser.parse_args();values=vars(args);values['source_roots']=values.pop('source')
    report=collect_pairs(**values);print(json.dumps({k:report[k]for k in ('status','completed_pairs','groups')}),flush=True)


if __name__=='__main__':main()
