"""Recorded one-NPC recovery windows and complete held-out spawn-event checks.

Source outcomes select TRAIN windows only. Neither sampling nor validation
corrects an autonomous state, observes future RGB, or chooses runtime spawns.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from .spawn import SPAWN_ANCHORS


ATTACK_BALANCED_PROTOCOL = 'source_attack_positive_negative_balanced_v2'
MICRO32_CONTINUATION_PROTOCOL = 'source_attack_balanced_micro32_continuation_v1'
MICRO16_CONTINUATION_PROTOCOL = 'source_attack_balanced_micro16_continuation_v1'


def upgrade_attack_balanced_sampler_state(previous):
    """Explicitly retain the complete old ledger; caller-owned RNG is untouched."""
    if previous.get('version') == 2:
        return deepcopy(previous)
    if (set(previous) != {'draws', 'negative_draws', 'anchor_counts'}
            or type(previous['draws']) is not int or type(previous['negative_draws']) is not int
            or previous['draws'] < 0 or previous['draws'] != 4*previous['negative_draws']
            or any(type(value) is not int or value < 0 for value in previous['anchor_counts'].values())
            or sum(previous['anchor_counts'].values()) != previous['negative_draws']):
        raise ValueError('Cannot upgrade an incomplete legacy recovery sampler ledger')
    retained = deepcopy(previous)
    return {**deepcopy(previous), 'version': 2, 'protocol': ATTACK_BALANCED_PROTOCOL,
        'v2_draws': 0, 'positive_draws': 0, 'retained_legacy_ledger': retained,
        'retained_legacy_ledger_sha256': hashlib.sha256(json.dumps(retained, sort_keys=True).encode()).hexdigest()}


def _attack_strata(saved, labels):
    """Recorded attacks/eligibility only; near-wall distance never labels attacks."""
    states = saved['states']; old, new = states[:-1, 8:18], states[1:, 8:18]
    dt = float(saved['dt'])
    eligible = ((old[:, 5] >= .5) & (old[:, 4] > 0) & (new[:, 5] >= .5) & (new[:, 4] > 0)
        & (states[:-1, 4] > 0) & (old[:, 6]-dt/1.2 <= 1e-6)
        & (((old[:, :2]-states[1:, :2])*16)**2).sum(-1).__le__(1.+1e-6))
    if (labels['attack_positive'] & ~eligible).any():
        raise ValueError('Recorded positive attack violates aligned Source melee eligibility')
    cells = np.argwhere(saved['map'] > 0)[:, ::-1]
    near = np.zeros(len(old), bool)
    if len(cells):
        for centers in (old[:, :2]*16, states[1:, :2]*16):
            gap = np.maximum(np.maximum(cells[None]-centers[:, None], centers[:, None]-(cells[None]+1)), 0.)
            near |= (gap*gap).sum(-1).min(-1) <= .75**2
    occluded = np.zeros(len(old), bool)
    if 'source_hard_negative_flags' in saved:
        flags = np.asarray(saved['source_hard_negative_flags'])
        if flags.shape != (len(old), 1) or (flags[:, 0].astype(bool) & ~labels['attack_negative']).any():
            raise ValueError('Recorded occluded negatives do not match genuine Source eligibility')
        occluded = flags[:, 0].astype(bool)
        if 'source_los_clear' in saved:
            clear = np.asarray(saved['source_los_clear'])
            if clear.shape != flags.shape or (occluded & clear[:, 0].astype(bool)).any():
                raise ValueError('Recorded Source occlusion flags conflict')
    return {'positive_near_wall': labels['attack_positive'] & near,
        'positive_open': labels['attack_positive'] & ~near,
        'negative_occluded': labels['attack_negative'] & occluded,
        'negative_other': labels['attack_negative'] & ~occluded}


def file_sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def prepare_recovery_corpus(original_root, negatives_root, destination):
    """Pin unmodified Source files under explicit main/auxiliary dataset roles.

    Long TRAIN tapes also supply short spawn windows. Their role aliases are
    disclosed and never counted as newly collected or independent Source data.
    """
    from .training_pipeline import atomic_json,now
    original_root,negatives_root,destination=map(lambda p:Path(p).resolve(),(original_root,negatives_root,destination))
    manifests=[]
    for root in (original_root,negatives_root):
        path=root/'manifest.json';saved=json.loads(path.read_text(encoding='utf-8'))
        if saved.get('status')!='complete':raise ValueError('Recovery Source corpus is incomplete')
        manifests.append((root,saved,file_sha(path)))
    identity={'version':1,'protocol':'one_enemy_recovery_source_roles_v1',
        'parents':[{'root':str(root),'manifest_sha256':sha} for root,_,sha in manifests]}
    manifest_path=destination/'manifest.json'
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text(encoding='utf-8'))
        if old.get('identity')!=identity:raise ValueError('Recovery corpus identity changed')
        if old.get('status')=='complete':
            for row in old['records']:
                if file_sha(destination/row['relative_path'])!=row['sha256']:raise ValueError('Pinned recovery file changed')
            return old
    destination.mkdir(parents=True,exist_ok=True)
    records=[];unique={};seeds={'train':set(),'val':set()};seen_roles=set()
    for root,source,source_sha in manifests:
        is_neg=root==negatives_root
        for row in source['records']:
            split=row['split'];stage='moving1' if is_neg else row['stage']
            roles=[stage]+(['moving1'] if stage=='full_game' and not is_neg else [])
            origin=(root/row['relative_path']).resolve()
            if root not in origin.parents or file_sha(origin)!=row['sha256']:
                raise ValueError('Source content/path changed before recovery binding')
            seeds[split].add(int(row['seed']))
            unique[(split,row['sha256'])]=int(row['transitions'])
            for role in roles:
                relative=f'{role}/rgb/{split}/{origin.name}'
                if relative in seen_roles:raise ValueError('Duplicate recovery role path')
                seen_roles.add(relative);target=destination/relative
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():
                    try:os.link(origin,target)
                    except OSError:shutil.copy2(origin,target)
                if file_sha(target)!=row['sha256']:raise ValueError('Recovery role does not match Source')
                records.append({**row,'stage':role,'relative_path':relative,'source_path':str(origin),
                    'source_manifest_sha256':source_sha,'original_stage':row.get('stage','moving1'),
                    'role_alias':role!=stage,'hard_negative_source':is_neg})
    if seeds['train']&seeds['val']:raise ValueError('Recovery TRAIN/VAL seeds overlap')
    train_sha={sha for split,sha in unique if split=='train'};val_sha={sha for split,sha in unique if split=='val'}
    if train_sha&val_sha:raise ValueError('Recovery TRAIN/VAL Source files overlap')
    groups={}
    for row in records:
        group=groups.setdefault(row['stage']+'/'+row['split'],{'files':0,'role_transitions':0})
        group['files']+=1;group['role_transitions']+=row['transitions']
    result={'format_version':1,'status':'complete','identity':identity,'updated_at':now(),
        'state_dim':20,'entity_slots':1,'image_size':32,'hz':30,'groups':groups,
        'unique_source_files':len(unique),
        'unique_source_transitions':{split:sum(n for (s,_),n in unique.items() if s==split) for split in seeds},
        'train_val_seed_overlap':False,'train_val_content_overlap':False,
        'long_tapes_aliased_for_auxiliary_spawn_windows':True,'source_physics_unchanged':True,
        'records':sorted(records,key=lambda row:row['relative_path'])}
    atomic_json(manifest_path,result);return result


def source_recovery_labels(states,attacks,dt):
    """Exact recorded-life/position labels matching combat loss eligibility."""
    states=np.asarray(states);attacks=np.asarray(attacks)
    if (states.ndim!=2 or states.shape[1]!=20 or len(states)<2
            or attacks.shape!=(len(states)-1,1) or not np.isfinite(states).all()
            or not np.isfinite(attacks).all() or not 0<float(dt)<=.25):
        raise ValueError('Recovery labels require finite aligned one-enemy Source')
    old,new=states[:-1,8:18],states[1:,8:18]
    old_alive=(old[:,5]>=.5)&(old[:,4]>0);new_alive=(new[:,5]>=.5)&(new[:,4]>0)
    revived=~old_alive&new_alive
    tiles=np.floor(new[:,:2]*16).astype(np.int64).clip(0,15)
    anchors=np.full(len(new),8,np.int64)
    for index,cell in enumerate(SPAWN_ANCHORS):anchors[(tiles==cell).all(-1)]=index
    eligible=(old_alive&new_alive&(states[:-1,4]>0)&(old[:,6]-float(dt)/1.2<=1e-6)
        &(((old[:,:2]-states[1:,:2])*16)**2).sum(-1).__le__(1.+1e-6))
    negative=eligible&~(attacks[:,0]>.5)
    phase=(np.rint(states[1:,-1].astype(np.float64)*300*30).astype(np.int64)//90)%8
    return {'spawn':revived,'anchor':anchors,'clock_phase':phase,
        'attack_negative':negative,'attack_positive':attacks[:,0]>.5}


def event_windows(event_indices,frames,context,horizon):
    maximum=frames-context-horizon
    if maximum<0:return np.empty(0,np.int64)
    windows=[]
    for event in event_indices:
        lo=max(0,int(event)-context-horizon+2);hi=min(maximum,int(event)-context+1)
        if hi>=lo:windows.extend(range(lo,hi+1))
    return np.unique(windows).astype(np.int64)


class RecoveryCombatSampler:
    """B16/B32 quarter spawn, quarter hard negatives, half legacy mix."""
    def __init__(self,data,*,context=1,horizon=16,cache_path=None,target_distance_balanced=False,
                 attack_positive_balanced=False):
        from .rssm_combat_protocol import CombatTrainingSampler
        if data.split!='train':raise ValueError('Recovery sampler accepts TRAIN only')
        if not isinstance(target_distance_balanced,bool):
            raise ValueError('Target distance balancing must be an explicit boolean')
        if not isinstance(attack_positive_balanced, bool) or attack_positive_balanced and not target_distance_balanced:
            raise ValueError('Attack-positive balancing requires explicit target-distance B32 sampling')
        self.attack_positive_balanced = attack_positive_balanced
        self.attack_groups = {name: [] for name in ('positive_near_wall', 'positive_open', 'negative_occluded', 'negative_other')}
        self.attack_masks = []; self.attack_event_counts = Counter()
        self.target_distance_balanced=target_distance_balanced
        self.target_groups={name:[] for name in ('hit_near','hit_mid','hit_far','fired_miss','wall_miss','other_miss')}
        self.target_masks=[];self.target_event_counts=Counter()
        self.base=CombatTrainingSampler(data,context=context,horizon=horizon,cache_path=cache_path)
        self.context,self.horizon=context,horizon;self.anchor_groups={i:[] for i in range(9)}
        negatives=[];self.event_counts=Counter()
        for episode,record in enumerate(self.base.base.records):
            with np.load(record['path'],allow_pickle=False) as saved:
                states=saved['states'];labels=source_recovery_labels(states,saved['attack_flags'],float(saved['dt']))
                if self.attack_positive_balanced:
                    masks = _attack_strata(saved, labels)
                    self.attack_masks.append(masks)
                    for name, mask in masks.items():
                        self.attack_event_counts[name] += int(mask.sum())
                        self.attack_groups[name].extend((episode, int(start)) for start in
                            event_windows(np.flatnonzero(mask), len(states), context, horizon))
                if self.target_distance_balanced:
                    events,hits,actions=saved['events'],saved['hit_ids'],saved['actions']
                    shot=events[:,0]>=.5;hit=shot&(hits>=0);miss=shot&(hits<0)
                    if (((hits>=0)!=(events[:,1]>=.5)).any() or ((hits>=0)&~shot).any()
                            or (shot&(actions[:,3]<.5)).any() or (shot&(states[:-1,4]<=0)).any()):
                        raise ValueError('Target sampling requires genuine aligned Source shots/hits')
                    distance=np.linalg.norm((states[:-1,8:10]-states[1:,:2])*16,axis=-1)
                    wall=record['masks']['wall_negative']&miss
                    masks={'hit_near':hit&(distance<2),'hit_mid':hit&(distance>=2)&(distance<5),
                        'hit_far':hit&(distance>=5),'fired_miss':miss,'wall_miss':wall,'other_miss':miss&~wall}
                    self.target_masks.append(masks)
                    for name,mask in masks.items():
                        self.target_event_counts[name]+=int(mask.sum())
                        self.target_groups[name].extend((episode,int(start)) for start in
                            event_windows(np.flatnonzero(mask),len(states),context,horizon))
            self.event_counts['attack_negative']+=int(labels['attack_negative'].sum())
            self.event_counts['attack_positive']+=int(labels['attack_positive'].sum())
            for anchor in self.anchor_groups:
                mask=labels['spawn']&(labels['anchor']==anchor)
                self.event_counts['spawn_anchor_'+str(anchor)]+=int(mask.sum())
                self.anchor_groups[anchor].extend((episode,int(start)) for start in
                    event_windows(np.flatnonzero(mask),len(states),context,horizon))
            negatives.extend((episode,int(start)) for start in
                event_windows(np.flatnonzero(labels['attack_negative']),len(states),context,horizon))
        self.anchor_groups={key:np.asarray(rows,np.int64).reshape(-1,2) for key,rows in self.anchor_groups.items() if rows}
        self.negatives=np.asarray(negatives,np.int64).reshape(-1,2)
        if set(range(8))-self.anchor_groups.keys():raise ValueError('Recovery training must cover all eight actual spawn anchors')
        if not len(self.negatives):raise ValueError('Recovery training needs real eligible Source negative attacks')
        if self.target_distance_balanced:
            self.target_groups={name:np.asarray(rows,np.int64).reshape(-1,2)for name,rows in self.target_groups.items()}
            missing=[name for name in ('hit_near','hit_mid','hit_far','fired_miss')if not len(self.target_groups[name])]
            if missing:raise ValueError('Target-balanced TRAIN pool is missing real future Source events: '+','.join(missing))
        self.anchor_counts={key:0 for key in self.anchor_groups};self.negative_draws=0;self.draws=0;self.last_sample=None
        self.v2_draws = self.positive_draws = 0
        self.micro16_sampling = None
        self.micro32_sampling = None
        self.micro16_continuation = None
        self.retained_legacy_ledger = {'draws': 0, 'negative_draws': 0, 'anchor_counts': self.anchor_counts.copy()}
        if self.attack_positive_balanced:
            self.attack_groups = {name: np.asarray(rows, np.int64).reshape(-1, 2) for name, rows in self.attack_groups.items()}
            if not any(len(self.attack_groups[name]) for name in ('positive_near_wall', 'positive_open')):
                raise ValueError('Attack-balanced TRAIN pool needs genuine future positive attacks')

    def sample(self,count,rng,*,allow_micro32_continuation=False,allow_micro16_continuation=False):
        if (not isinstance(count,int) or isinstance(count,bool) or count not in (16,32,64)
                or count == 64 and not self.attack_positive_balanced):
            raise ValueError('Recovery quotas require B16/B32, or explicit positive-balanced B64')
        if self.target_distance_balanced and count!=32 and not (self.attack_positive_balanced and count in(16,64)):
            raise ValueError('Target-distance recovery quotas require explicit B32 or positive-balanced B16/B64')
        if (type(allow_micro32_continuation) is not bool
                or allow_micro32_continuation and (count!=32 or not self.attack_positive_balanced)):
            raise ValueError('Balanced micro32 continuation requires an explicit positive-balanced B32 draw')
        if(type(allow_micro16_continuation)is not bool or allow_micro16_continuation and(
                count!=16 or not self.attack_positive_balanced or self.micro32_sampling is None
                or allow_micro32_continuation)):
            raise ValueError('Balanced micro16 continuation requires an explicit saved-micro32 to B16 transition')
        if self.micro16_continuation is not None and not allow_micro16_continuation:
            raise ValueError('Cannot silently disable the saved balanced micro16 continuation policy')
        if self.micro32_sampling is not None and count!=32 and not allow_micro16_continuation:
            raise ValueError('Cannot silently disable the saved balanced micro32 continuation policy')
        if allow_micro16_continuation and self.micro16_continuation is None:
            source=self.state_dict();self._validate_state(source)
            self.micro16_continuation=dict(kind=MICRO16_CONTINUATION_PROTOCOL,version=1,
                source_sampler_state=source,
                source_sampler_state_sha256=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest(),
                # Preserve the next B32 miss-stratum phase across two B16
                # draws; old16/old32 counters remain immutable history.
                rotation_base_batches=2*((self.v2_draws//32)%2),draws=0,batches=0)
        if allow_micro32_continuation and self.micro32_sampling is None:
            source=self.state_dict()
            self.micro32_sampling=dict(kind=MICRO32_CONTINUATION_PROTOCOL,version=1,
                source_sampler_state=source,
                source_sampler_state_sha256=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest(),
                draws=0,batches=0)
        micro16 = self.attack_positive_balanced and count == 16
        if self.micro16_sampling is not None and not micro16 and self.micro32_sampling is None:
            raise ValueError('Cannot silently disable the saved balanced micro16 sampling policy')
        if micro16 and self.micro16_sampling is None and self.micro16_continuation is None:
            self.micro16_sampling = dict(kind='source_attack_balanced_micro16_v1',version=1,
                source_v2_draws=self.v2_draws,draws=0,batches=0)
        micro16_index=((self.micro16_continuation['rotation_base_batches']+self.micro16_continuation['batches'])
            if self.micro16_continuation is not None else self.micro16_sampling['batches']if micro16 else None)
        factor = count//32 if self.attack_positive_balanced else 1
        quarter=count//4
        spawned=[];chosen=[]
        for _ in range(quarter):
            lowest=min(self.anchor_counts.values());tied=[key for key,n in self.anchor_counts.items() if n==lowest]
            key=int(rng.choice(tied));group=self.anchor_groups[key]
            spawned.append(group[int(rng.integers(len(group)))]);chosen.append(key);self.anchor_counts[key]+=1
        attack_selection=[];attack_subquotas={}
        if self.attack_positive_balanced:
            selected = {}
            for family in ('positive', 'negative'):
                candidates = [name for name in self.attack_groups if name.startswith(family+'_') and len(self.attack_groups[name])]
                if not candidates: raise ValueError('Attack-balanced TRAIN pool is missing '+family)
                quota = {name: quarter//len(candidates)+(i < quarter%len(candidates)) for i, name in enumerate(candidates)}
                if family == 'negative' and len(candidates) == 2:
                    quota = ({'negative_occluded':3,'negative_other':1} if micro16 else
                             {'negative_occluded': 6*factor, 'negative_other': 2*factor})
                pairs = []
                for name, size in quota.items():
                    attack_subquotas[name] = size
                    group = self.attack_groups[name]
                    for pair in group[rng.integers(len(group), size=size)]:
                        episode, start = map(int, pair); first = start+self.context-1
                        attack_selection.append({'requested': name, 'episode': episode, 'start': start,
                            'source_transition_indices': (np.flatnonzero(self.attack_masks[episode][name][first:first+self.horizon])+first).tolist()})
                        pairs.append(pair)
                selected[family] = np.asarray(pairs, np.int64)
            negative, positive = selected['negative'], selected['positive']
        else:
            negative=self.negatives[rng.integers(len(self.negatives),size=quarter)]
        target_selection=[];miss_quota={}
        if self.target_distance_balanced:
            per_hit = 1 if micro16 else 2*factor if self.attack_positive_balanced else 3
            groups=[('hit_near',per_hit),('hit_mid',per_hit),('hit_far',per_hit)]
            wall_present=bool(len(self.target_groups['wall_miss']));other_present=bool(len(self.target_groups['other_miss']))
            if micro16:
                # Four small draws retain the old effective64 target mixture.
                # Rotate miss strata only when a genuine miss is actually drawn.
                choose_wall=wall_present and(not other_present or(micro16_index//2)%2==0)
                miss_quota={'wall_miss':int(micro16_index%2==0 and choose_wall),
                            'other_miss':int(micro16_index%2==0 and not choose_wall)}
            elif self.attack_positive_balanced:
                # Both miss strata rotate across consecutive B32 draws when present.
                choose_wall = wall_present and (not other_present or (self.v2_draws//32)%2 == 0)
                miss_quota=({'wall_miss': int(choose_wall), 'other_miss': int(not choose_wall)} if factor == 1
                    else {'wall_miss': 1 if wall_present and other_present else 2*int(wall_present),
                          'other_miss': 1 if wall_present and other_present else 2*int(not wall_present)})
            elif wall_present and other_present:miss_quota={'wall_miss':2,'other_miss':1}
            elif wall_present:miss_quota={'wall_miss':3,'other_miss':0}
            else:miss_quota={'wall_miss':0,'other_miss':3}
            groups+=list(miss_quota.items());targeted=[]
            for name,size in groups:
                if not size:continue
                group=self.target_groups[name]
                for pair in group[rng.integers(len(group),size=size)]:
                    episode,start=map(int,pair);first=start+self.context-1
                    target_selection.append({'requested':name,'episode':episode,'start':start,
                        'source_transition_indices':(np.flatnonzero(self.target_masks[episode][name][first:first+self.horizon])+first).tolist()})
                    targeted.append(pair)
            legacy_count=(micro16_index%2 if micro16 else factor if self.attack_positive_balanced else 4)
            ordinary=np.asarray(targeted,np.int64)
            if legacy_count:
                ordinary=np.concatenate([ordinary,self.base.sample(legacy_count,rng)])
        else:
            ordinary=self.base.sample(count//2,rng)
        arrays = [np.asarray(spawned), negative]
        if self.attack_positive_balanced: arrays.append(positive)
        pairs=np.concatenate([*arrays,ordinary]);rng.shuffle(pairs)
        self.draws+=count;self.negative_draws+=quarter
        self.last_sample={'version':1,'quota':{'spawn':quarter,'attack_negative':quarter,'ordinary_combat':count//2},
            'selected_spawn_anchors':chosen,'anchor_draw_counts':dict(self.anchor_counts),
            'ordinary_sampler':None if micro16 and not legacy_count else self.base.last_sample,'source_labels_only':True}
        if self.target_distance_balanced:
            event_counts=Counter();window_counts=Counter()
            for episode,start in pairs:
                first=int(start)+self.context-1
                for name,mask in self.target_masks[int(episode)].items():
                    number=int(mask[first:first+self.horizon].sum())
                    event_counts[name]+=number;window_counts[name]+=int(number>0)
            self.last_sample.update(target_distance_balanced_sampling=True,
                target_subquotas=({'hit_near':1,'hit_mid':1,'hit_far':1,'fired_miss':1-micro16_index%2,'legacy':micro16_index%2} if micro16 else
                    {'hit_near':2*factor,'hit_mid':2*factor,'hit_far':2*factor,'fired_miss':factor,'legacy':factor} if self.attack_positive_balanced
                    else {'hit_near':3,'hit_mid':3,'hit_far':3,'fired_miss':3,'legacy':4}),
                fired_miss_subquotas=miss_quota,target_selected_windows=target_selection,
                target_future_event_counts=dict(event_counts),target_future_window_counts=dict(window_counts),
                target_counts_include_duplicate_draws=True,target_cohorts_are_nonexclusive=True)
        if self.attack_positive_balanced:
            self.v2_draws += count; self.positive_draws += quarter
            self.last_sample.update(version=2, protocol=ATTACK_BALANCED_PROTOCOL,
                quota={'spawn':quarter,'attack_negative':quarter,'attack_positive':quarter,'shooting_replay':quarter},
                attack_positive_balanced=True, attack_subquotas=attack_subquotas,
                attack_selected_windows=attack_selection,
                actual_attack_future_event_counts={name: int(sum(self.attack_masks[int(ep)][name][int(start)+self.context-1:int(start)+self.context-1+self.horizon].sum()
                    for ep,start in pairs)) for name in self.attack_groups},
                attack_counts_include_duplicate_draws=True)
            if self.micro16_continuation is not None:
                self.micro16_continuation['draws']+=16;self.micro16_continuation['batches']+=1
                self.last_sample.update(runtime_batch_size=16,micro16_continuation_policy=MICRO16_CONTINUATION_PROTOCOL,
                    micro16_continuation_batch_index=self.micro16_continuation['batches']-1,
                    micro16_rotation_index=micro16_index)
            elif micro16:
                self.micro16_sampling['draws']+=16;self.micro16_sampling['batches']+=1
                self.last_sample.update(runtime_batch_size=16,
                    micro16_sampling_policy='source_attack_balanced_micro16_v1',
                    micro16_batch_index=micro16_index)
            if self.micro32_sampling is not None and self.micro16_continuation is None:
                batch_index=self.micro32_sampling['batches']
                self.micro32_sampling['draws']+=32;self.micro32_sampling['batches']+=1
                self.last_sample.update(runtime_batch_size=32,
                    micro32_sampling_policy=MICRO32_CONTINUATION_PROTOCOL,
                    micro32_batch_index=batch_index)
        return pairs

    def state_dict(self):
        result = {'draws':self.draws,'negative_draws':self.negative_draws,'anchor_counts':self.anchor_counts.copy()}
        if self.attack_positive_balanced:
            result.update(version=2, protocol=ATTACK_BALANCED_PROTOCOL, v2_draws=self.v2_draws, positive_draws=self.positive_draws,
                retained_legacy_ledger=deepcopy(self.retained_legacy_ledger),
                retained_legacy_ledger_sha256=hashlib.sha256(json.dumps(self.retained_legacy_ledger,sort_keys=True).encode()).hexdigest())
            if self.micro16_sampling is not None:
                result['micro16_sampling']=deepcopy(self.micro16_sampling)
            if self.micro32_sampling is not None:
                result['micro32_sampling']=deepcopy(self.micro32_sampling)
            if self.micro16_continuation is not None:
                result['micro16_continuation']=deepcopy(self.micro16_continuation)
        return result

    def _validate_state(self,state):
        continuation=state.get('micro16_continuation')
        if continuation is not None:
            expected={'kind','version','source_sampler_state','source_sampler_state_sha256',
                'rotation_base_batches','draws','batches'}
            if(not self.attack_positive_balanced or not isinstance(continuation,dict)or set(continuation)!=expected
                    or continuation.get('kind')!=MICRO16_CONTINUATION_PROTOCOL or continuation.get('version')!=1
                    or any(type(continuation.get(key))is not int or continuation[key]<0 for key in('draws','batches','rotation_base_batches'))
                    or continuation['draws']!=16*continuation['batches']
                    or not isinstance(continuation.get('source_sampler_state'),dict)
                    or 'micro16_continuation'in continuation['source_sampler_state']):
                raise ValueError('Balanced micro16 continuation source/draw ledger changed')
            source=continuation['source_sampler_state'];source_counts=self._validate_state(source)
            if(not isinstance(source.get('micro32_sampling'),dict)
                    or hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()!=continuation['source_sampler_state_sha256']
                    or continuation['rotation_base_batches']!=2*((source['v2_draws']//32)%2)
                    or any(state.get(key)!=source.get(key)for key in('version','protocol','micro16_sampling','micro32_sampling',
                        'retained_legacy_ledger','retained_legacy_ledger_sha256'))
                    or any(type(state.get(key))is not int or state[key]!=source[key]+continuation['draws']//factor
                        for key,factor in(('draws',1),('v2_draws',1),('negative_draws',4),('positive_draws',4)))):
                raise ValueError('Balanced micro16 continuation changed its immutable saved-micro32 history')
            counts={int(key):value for key,value in state.get('anchor_counts',{}).items()}
            if(set(counts)!=set(self.anchor_counts)or any(type(value)is not int or value<source_counts[key]
                    for key,value in counts.items())or sum(counts.values())!=state['negative_draws']
                    or state['draws']!=4*state['negative_draws']):
                raise ValueError('Balanced micro16 continuation anchor/event counters changed')
            return counts
        if self.attack_positive_balanced:
            old = state.get('retained_legacy_ledger', {})
            micro16=state.get('micro16_sampling')
            micro32=state.get('micro32_sampling')
            source=None
            if micro32 is not None:
                if (not isinstance(micro32,dict)
                        or set(micro32)!={'kind','version','source_sampler_state','source_sampler_state_sha256','draws','batches'}
                        or micro32.get('kind')!=MICRO32_CONTINUATION_PROTOCOL
                        or micro32.get('version')!=1
                        or any(type(micro32.get(key))is not int or micro32[key]<0 for key in('draws','batches'))
                        or micro32['draws']%32 or micro32['batches']!=micro32['draws']//32
                        or not isinstance(micro32.get('source_sampler_state'),dict)
                        or 'micro32_sampling' in micro32['source_sampler_state']):
                    raise ValueError('Balanced micro32 continuation source/draw ledger changed')
                source=micro32['source_sampler_state']
                self._validate_state(source)
                if (hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()!=micro32['source_sampler_state_sha256']
                        or source.get('version')!=2 or source.get('protocol')!=ATTACK_BALANCED_PROTOCOL
                        or source.get('micro16_sampling')!=micro16
                        or source.get('retained_legacy_ledger')!=old
                        or source['v2_draws']+micro32['draws']!=state.get('v2_draws')
                        or source['draws']+micro32['draws']!=state.get('draws')
                        or source['negative_draws']+micro32['draws']//4!=state.get('negative_draws')):
                    raise ValueError('Balanced micro32 continuation changed its immutable source ledger')
            if micro16 is not None and (not isinstance(micro16,dict)
                    or set(micro16)!={'kind','version','source_v2_draws','draws','batches'}
                    or micro16.get('kind')!='source_attack_balanced_micro16_v1' or micro16.get('version')!=1
                    or any(type(micro16.get(key))is not int or micro16[key]<0 for key in('source_v2_draws','draws','batches'))
                    or micro16['source_v2_draws']%32 or micro16['draws']%16
                    or micro16['batches']!=micro16['draws']//16
                    or micro16['source_v2_draws']+micro16['draws']!=(source['v2_draws'] if source is not None else state.get('v2_draws'))):
                raise ValueError('Balanced micro16 source/draw/rotation ledger changed')
            if (state.get('version') != 2 or state.get('protocol') != ATTACK_BALANCED_PROTOCOL
                    or type(state.get('v2_draws')) is not int or state['v2_draws'] < 0
                    or micro16 is None and state['v2_draws']%32
                    or state.get('positive_draws') != state['v2_draws']//4
                    or old.get('draws',-1)+state['v2_draws'] != state['draws']
                    or old.get('negative_draws',-1)+state['v2_draws']//4 != state['negative_draws']
                    or hashlib.sha256(json.dumps(old,sort_keys=True).encode()).hexdigest() != state.get('retained_legacy_ledger_sha256')):
                raise ValueError('Attack-positive sampler requires explicit version2 ledger upgrade/consistent counters')
        elif state.get('version') == 2:
            raise ValueError('Cannot silently disable the saved positive-attack sampler')
        counts={int(k):v for k,v in state['anchor_counts'].items()}
        if (set(counts)!=set(self.anchor_counts) or any(not isinstance(n,int) or n<0 for n in counts.values())
                or sum(counts.values())!=state['negative_draws'] or state['draws']!=4*state['negative_draws']):
            raise ValueError('Recovery sampler ledger differs from its exact1/4+1/4+1/2 quotas')
        if self.attack_positive_balanced and source is not None:
            source_counts={int(k):v for k,v in source['anchor_counts'].items()}
            if any(counts[key]<source_counts[key] for key in counts):
                raise ValueError('Balanced micro32 continuation rolled back a Source anchor counter')
        return counts

    def restore_state(self,state):
        counts=self._validate_state(state)
        self.anchor_counts=counts;self.draws=state['draws'];self.negative_draws=state['negative_draws']
        if self.attack_positive_balanced:
            self.v2_draws, self.positive_draws = state['v2_draws'], state['positive_draws']
            self.retained_legacy_ledger = deepcopy(state['retained_legacy_ledger'])
            self.micro16_sampling=deepcopy(state.get('micro16_sampling'))
            self.micro32_sampling=deepcopy(state.get('micro32_sampling'))
            self.micro16_continuation=deepcopy(state.get('micro16_continuation'))

    def summary(self,*,batch_size=16):
        if (not isinstance(batch_size,int) or isinstance(batch_size,bool) or batch_size not in (16,32,64)
                or batch_size == 64 and not self.attack_positive_balanced):
            raise ValueError('Recovery summary requires B16/B32, or explicit positive-balanced B64')
        if self.target_distance_balanced and batch_size!=32 and not (self.attack_positive_balanced and batch_size in(16,64)):
            raise ValueError('Target-balanced summary requires B32 or positive-balanced B16/B64')
        factor = batch_size//32 if self.attack_positive_balanced else 1
        result={'version':1,'horizon':self.horizon,'context':self.context,
            'quota':{'spawn':batch_size//4,'attack_negative':batch_size//4,'ordinary':batch_size//2},
            'source_event_counts':dict(self.event_counts),'anchor_window_counts':{str(k):len(v) for k,v in self.anchor_groups.items()},
            'negative_windows':len(self.negatives),'missing_anchor_classes':[], 'source_future_conditioning':False}
        if self.target_distance_balanced:
            result.update(target_distance_balanced_sampling=True,
                target_subquotas={'hit_near':3,'hit_mid':3,'hit_far':3,'fired_miss':3,'legacy':4},
                target_source_event_counts=dict(self.target_event_counts),
                target_window_counts={name:len(rows)for name,rows in self.target_groups.items()},
                target_distance_tiles={'near':'<2','mid':'[2,5)','far':'>=5'},
                target_distance_geometry='Previous Source NPC position to next Source player firing position',
                target_window_definition='Event strictly inside future[start+context-1:start+context-1+horizon]',
                target_miss_policy='Two verified wall-occluded and one other fired miss when both exist; otherwise three available genuine fired misses')
        if self.attack_positive_balanced:
            if batch_size==16:
                result.update(version=2,protocol=ATTACK_BALANCED_PROTOCOL,attack_positive_balanced=True,
                    runtime_batch_size=16,micro16_sampling_policy='source_attack_balanced_micro16_v1',
                    quota={'spawn':4,'attack_negative':4,'attack_positive':4,'shooting_replay':4},
                    target_subquotas={'hit_near':1,'hit_mid':1,'hit_far':1,'alternating_miss_or_legacy':1},
                    four_micro_target_subquotas={'hit_near':4,'hit_mid':4,'hit_far':4,'fired_miss':2,'legacy':2},
                    attack_source_event_counts=dict(self.attack_event_counts),
                    attack_window_counts={name:len(rows)for name,rows in self.attack_groups.items()},
                    attack_positive_policy='Equal present near-wall/open positive windows;2each when both exist',
                    attack_negative_policy='Three Source-occluded and one other genuine negative when both exist',
                    target_miss_policy='Alternate one genuine miss and one legacy window; wall/other rotates on actual miss draws')
                return result
            result.update(version=2, protocol=ATTACK_BALANCED_PROTOCOL, attack_positive_balanced=True,
                quota={'spawn':8*factor,'attack_negative':8*factor,'attack_positive':8*factor,'shooting_replay':8*factor},
                target_subquotas={'hit_near':2*factor,'hit_mid':2*factor,'hit_far':2*factor,'fired_miss':factor,'legacy':factor},
                attack_source_event_counts=dict(self.attack_event_counts),
                attack_window_counts={name:len(rows) for name,rows in self.attack_groups.items()},
                attack_positive_policy='Equal present near-wall/open positive window quotas; near-wall means center-to-solid-tile distance <=.75tiles',
                attack_negative_policy=('Six recorded Source-occluded and two other eligible negative windows when both exist; otherwise eight available genuine negatives. Other is not a clear-LOS label.' if factor == 1 else
                    'Twelve recorded Source-occluded and four other eligible negative windows when both exist; otherwise sixteen available genuine negatives. Other is not a clear-LOS label.'),
                target_miss_policy=('One genuine miss per draw, alternating verified wall and other strata when both exist' if factor == 1 else
                    'Two genuine misses per draw: one verified wall and one other when both exist, otherwise two available genuine misses'))
        return result


def evaluate_spawn_events(model,data,output_dir,*,context=1,batch_size=16):
    """Check EVERY recorded revival in long VAL; no128/9-event subsampling."""
    import torch
    from .behavior_heads import _Metrics,_measure
    from .staged_validation import _isolated_validation
    from .training_pipeline import atomic_json
    if data.split not in ('val','test') or model.config.navigation_only:
        raise ValueError('Spawn checks need held-out data and the one-enemy combat profile')
    rows=[];source_ids=[]
    for episode,info in enumerate(data.episodes):
        with np.load(info['path'],allow_pickle=False) as saved:
            labels=source_recovery_labels(saved['states'],saved['attack_flags'],float(saved['dt']))
        source_ids.append({'episode':Path(info['path']).name,'sha256':file_sha(info['path'])})
        for tick in np.flatnonzero(labels['spawn']):
            if tick>=context-1:
                rows.append({'episode':episode,'start':int(tick)-context+1,'transition':int(tick),
                    'anchor':int(labels['anchor'][tick]),'clock_phase':int(labels['clock_phase'][tick])})
    if not rows:raise ValueError('Long VAL contains no actual Source revivals')
    device=next(model.parameters()).device;total=_Metrics();anchors={};phases={}
    with _isolated_validation(model,data,173):
        for first in range(0,len(rows),batch_size):
            chosen=rows[first:first+batch_size]
            pairs=np.asarray([[r['episode'],r['start']] for r in chosen],np.int64)
            batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in
                data.batch(len(chosen),context=context,horizon=1,indices=pairs).items()}
            fmap=model.encode_map(batch['map']);memory=model.initialize_from_image(batch['context_frames'],batch['states'][:,:context],fmap)
            result=model.transition(batch['states'][:,context-1],batch['actions'][:,0],fmap,memory)
            _measure(result,batch,context,total,model.config.simulation_hz,model.config.endless,game_training_v5=model.config.game_training_v5)
            for name,groups in [('anchor',anchors),('clock_phase',phases)]:
                for value in sorted({r[name] for r in chosen}):
                    mask=torch.tensor([r[name]==value for r in chosen],device=device)
                    sub={k:v[mask] if torch.is_tensor(v) and v.ndim and len(v)==len(chosen) else v for k,v in batch.items()}
                    predicted={k:v[mask] if torch.is_tensor(v) and v.ndim and len(v)==len(chosen) else v for k,v in result.items()}
                    meter=groups.setdefault(str(value),_Metrics())
                    _measure(predicted,sub,context,meter,model.config.simulation_hz,model.config.endless,game_training_v5=model.config.game_training_v5)
    specification={'context':context,'source_files':source_ids,'events':rows,'kind':'all_long_val_spawn_events_v1'}
    report={'kind':'all_long_val_spawn_events_v1','source_held_out':True,'context':context,'events':len(rows),
        'source_episode_count':len(source_ids),'overall':total.report(),
        'per_anchor':{key:meter.report() for key,meter in anchors.items()},
        'per_clock_phase':{key:meter.report() for key,meter in phases.items()},
        'missing_anchor_classes':[key for key in range(8) if str(key) not in anchors],
        'source_future_feedback':False,'future_rgb_inputs':0,'runtime_engine_calls':0,
        'panel_fingerprint':hashlib.sha256(json.dumps(specification,sort_keys=True).encode()).hexdigest(),
        'cohort_scope':'Every actual revival in recorded long VAL, overlapping old short-head reports are not pooled'}
    destination=Path(output_dir);destination.mkdir(parents=True,exist_ok=True)
    atomic_json(destination/'expanded_spawn_validation.json',report);return report
