"""Recorded one-NPC recovery windows and complete held-out spawn-event checks.

Source outcomes select TRAIN windows only. Neither sampling nor validation
corrects an autonomous state, observes future RGB, or chooses runtime spawns.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from .spawn import SPAWN_ANCHORS


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
    def __init__(self,data,*,context=1,horizon=16,cache_path=None,target_distance_balanced=False):
        from .rssm_combat_protocol import CombatTrainingSampler
        if data.split!='train':raise ValueError('Recovery sampler accepts TRAIN only')
        if not isinstance(target_distance_balanced,bool):
            raise ValueError('Target distance balancing must be an explicit boolean')
        self.target_distance_balanced=target_distance_balanced
        self.target_groups={name:[] for name in ('hit_near','hit_mid','hit_far','fired_miss','wall_miss','other_miss')}
        self.target_masks=[];self.target_event_counts=Counter()
        self.base=CombatTrainingSampler(data,context=context,horizon=horizon,cache_path=cache_path)
        self.context,self.horizon=context,horizon;self.anchor_groups={i:[] for i in range(9)}
        negatives=[];self.event_counts=Counter()
        for episode,record in enumerate(self.base.base.records):
            with np.load(record['path'],allow_pickle=False) as saved:
                states=saved['states'];labels=source_recovery_labels(states,saved['attack_flags'],float(saved['dt']))
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

    def sample(self,count,rng):
        if not isinstance(count,int) or isinstance(count,bool) or count not in (16,32):
            raise ValueError('Recovery quotas require B16 or B32')
        if self.target_distance_balanced and count!=32:
            raise ValueError('Target-distance recovery quotas require explicit B32')
        quarter=count//4
        spawned=[];chosen=[]
        for _ in range(quarter):
            lowest=min(self.anchor_counts.values());tied=[key for key,n in self.anchor_counts.items() if n==lowest]
            key=int(rng.choice(tied));group=self.anchor_groups[key]
            spawned.append(group[int(rng.integers(len(group)))]);chosen.append(key);self.anchor_counts[key]+=1
        negative=self.negatives[rng.integers(len(self.negatives),size=quarter)]
        target_selection=[];miss_quota={}
        if self.target_distance_balanced:
            groups=[('hit_near',3),('hit_mid',3),('hit_far',3)]
            wall_present=bool(len(self.target_groups['wall_miss']));other_present=bool(len(self.target_groups['other_miss']))
            if wall_present and other_present:miss_quota={'wall_miss':2,'other_miss':1}
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
            ordinary=np.concatenate([np.asarray(targeted,np.int64),self.base.sample(4,rng)])
        else:
            ordinary=self.base.sample(count//2,rng)
        pairs=np.concatenate([np.asarray(spawned),negative,ordinary]);rng.shuffle(pairs)
        self.draws+=count;self.negative_draws+=quarter
        self.last_sample={'version':1,'quota':{'spawn':quarter,'attack_negative':quarter,'ordinary_combat':count//2},
            'selected_spawn_anchors':chosen,'anchor_draw_counts':dict(self.anchor_counts),
            'ordinary_sampler':self.base.last_sample,'source_labels_only':True}
        if self.target_distance_balanced:
            event_counts=Counter();window_counts=Counter()
            for episode,start in pairs:
                first=int(start)+self.context-1
                for name,mask in self.target_masks[int(episode)].items():
                    number=int(mask[first:first+self.horizon].sum())
                    event_counts[name]+=number;window_counts[name]+=int(number>0)
            self.last_sample.update(target_distance_balanced_sampling=True,
                target_subquotas={'hit_near':3,'hit_mid':3,'hit_far':3,'fired_miss':3,'legacy':4},
                fired_miss_subquotas=miss_quota,target_selected_windows=target_selection,
                target_future_event_counts=dict(event_counts),target_future_window_counts=dict(window_counts),
                target_counts_include_duplicate_draws=True,target_cohorts_are_nonexclusive=True)
        return pairs

    def state_dict(self):
        return {'draws':self.draws,'negative_draws':self.negative_draws,'anchor_counts':self.anchor_counts.copy()}

    def restore_state(self,state):
        counts={int(k):v for k,v in state['anchor_counts'].items()}
        if (set(counts)!=set(self.anchor_counts) or any(not isinstance(n,int) or n<0 for n in counts.values())
                or sum(counts.values())!=state['negative_draws'] or state['draws']!=4*state['negative_draws']):
            raise ValueError('Recovery sampler ledger differs from its exact1/4+1/4+1/2 quotas')
        self.anchor_counts=counts;self.draws=state['draws'];self.negative_draws=state['negative_draws']

    def summary(self,*,batch_size=16):
        if not isinstance(batch_size,int) or isinstance(batch_size,bool) or batch_size not in (16,32):
            raise ValueError('Recovery summary requires B16 or B32')
        if self.target_distance_balanced and batch_size!=32:
            raise ValueError('Target-balanced summary requires B32')
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
