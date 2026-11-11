"""Immutable one-NPC Source frames and explicit empty navigation replay."""
from collections import Counter, OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .camera import project_sprite_bounds
from .distill_renderer import _digest
from .engine import BEHAVIOR_VERSION, STATE_DIM


ENEMY_WEIGHTS = {'large': .15, 'partial': .20, 'distant': .20, 'corpse': .15,
                 'medium': .05, 'empty_navigation': .125, 'empty_hidden': .125}
POSE_COLUMNS = [0,1,2,3,8,9]


def scene_key(state):
    return np.asarray(state,dtype=np.float32)[POSE_COLUMNS].tobytes()


class EnemyFrames:
    def __init__(self,data_root,empty_data_root,split,*,seed=173,exclude_scene_keys=None,weights=None,
                 source_directories=None):
        if split not in {'train','val'}:raise ValueError('Enemy renderer requires explicit train/val split')
        self.split,self.rng=split,np.random.default_rng(seed)
        self.weights=dict(ENEMY_WEIGHTS if weights is None else weights)
        if (set(self.weights)!=set(ENEMY_WEIGHTS) or any(not np.isfinite(v) or v<=0 for v in self.weights.values())
                or abs(sum(self.weights.values())-1)>1e-8):
            raise ValueError('Use positive normalized enemy/background sampling quotas')
        self.files,self.rows,self.scene_keys=[],[],set()
        self.buckets={name:[] for name in self.weights}
        self.draws,self.case_counts=0,dict.fromkeys(self.weights,0)
        self._cache=OrderedDict()
        excluded=set() if exclude_scene_keys is None else set(exclude_scene_keys)
        counts=Counter();excluded_keys=set()
        roots=[(Path(data_root)/'static_enemy/snapshots'/split,'static',False),
               (Path(data_root)/'moving1/rgb'/split,'moving',True),
               (Path(empty_data_root)/'short/rgb'/split,'empty_navigation',True)]
        if source_directories is not None:
            roots=[(Path(row['directory']),row['kind'],row['dynamic']) for row in source_directories]
            if not roots:raise ValueError('Explicit Source directories must be nonempty')
        for directory,kind,dynamic in roots:
            directory=directory.resolve()
            if not directory.is_dir():raise ValueError('Missing genuine Source split: '+str(directory))
            for path in sorted(directory.glob('*.npz')):
                with np.load(path,allow_pickle=False) as source:
                    states=source['states'].astype(np.float32)
                    frames,masks,background=source['frames'],source['object_masks'],source['background_frames']
                    grid=source['map']
                    if (str(source['split'])!=split or states.ndim!=2 or states.shape[-1]!=STATE_DIM
                            or frames.dtype!=np.uint8 or frames.shape!=(len(states),32,32,3)
                            or masks.dtype!=np.bool_ or masks.shape!=(len(states),32,32)
                            or background.dtype!=np.uint8 or background.shape!=frames.shape
                            or grid.shape!=(16,16) or str(source['behavior_version'])!=BEHAVIOR_VERSION
                            or not np.isfinite(states).all()):
                        raise ValueError('Expected genuine native32 state20 Source with exact masks/background')
                    if not np.array_equal(masks,np.any(frames!=background,axis=-1)):
                        raise ValueError('Source foreground masks/backgrounds are not bitexact')
                    if dynamic:
                        if 'actions' not in source or len(source['actions'])+1!=len(states):
                            raise ValueError('Dynamic Source episodes require actual adjacent controls')
                    elif 'actions' in source or bool(source['world_transition_compatible']):
                        raise ValueError('Independent snapshots must not masquerade as dynamics')
                    events=(source['events'].copy() if source_directories is not None and dynamic and 'events' in source
                            else np.zeros((max(0,len(states)-1),4),np.float32))
                    if events.shape!=(max(0,len(states)-1),4) or not np.isfinite(events).all():
                        raise ValueError('Source event labels require one finite four-event row per action')
                    if kind=='empty_navigation' and (np.any(states[:,12:14]) or masks.any()):
                        raise ValueError('Empty navigation replay must contain no visible/active NPC')
                    seed_value=int(source['seed'])
                file_index=len(self.files)
                self.files.append({'path':str(path),'seed':seed_value,'sha256':_digest(path),
                    'kind':kind,'dynamic':dynamic,'stat':(path.stat().st_size,path.stat().st_mtime_ns)})
                self.scene_keys.update(scene_key(state) for state in states)
                bounds=project_sprite_bounds(torch.from_numpy(states),image_size=32)['unclipped_boxes'][:,0].numpy()
                for frame in range(1 if dynamic else 0,len(states)):
                    key=scene_key(states[frame])
                    if key in excluded:
                        counts['excluded_'+kind]+=1;excluded_keys.add(key);continue
                    mask=masks[frame];fraction=float(mask.mean());state=states[frame]
                    if not mask.any() and kind!='empty_navigation' and not (state[13]>=.5 and state[12]>0):
                        counts['skipped_invisible_dead']+=1
                        continue
                    if not mask.any():category='empty_navigation' if kind=='empty_navigation' else 'empty_hidden'
                    elif state[13]<.5 or state[12]<=0:category='corpse'
                    elif np.any(bounds[frame,:2]<0) or np.any(bounds[frame,2:]>1):category='partial'
                    elif fraction>=.10:category='large'
                    elif np.linalg.norm((state[8:10]-state[:2])*16)>=4 or fraction<.01:category='distant'
                    else:category='medium'
                    index=len(self.rows)
                    self.rows.append({'file':file_index,'frame':frame,'category':category,
                        'mask_fraction':fraction,'hurt':bool(state[15]>0 and state[13]>=.5),
                        'feature_policy':'frozen_past_prior' if dynamic else 'snapshot_zero128'})
                    if source_directories is not None:
                        previous=states[frame-1] if dynamic else state
                        event=events[frame-1] if dynamic else np.zeros(4)
                        cases=[]
                        if event[1]>.5 or state[12]<previous[12]-1e-6:cases.append('hit')
                        if event[0]>.5 and event[1]<=.5:cases.append('miss')
                        if event[3]>.5 or state[4]<previous[4]-1e-6:cases.append('player_hurt')
                        if previous[13]>=.5 and state[13]<.5:cases.append('death')
                        if previous[13]<.5 and state[13]>=.5:cases.append('respawn')
                        self.rows[-1].update(source_kind=kind,combat_cases=cases,
                                             source_events=event.tolist())
                    self.buckets[category].append(index);counts[category]+=1
        if not self.rows or any(not values for values in self.buckets.values()):
            raise ValueError('Enemy panel needs all visible/death/empty categories: '+str(dict(counts)))
        self.excluded_scene_keys=len(excluded_keys)
        self.excluded_frames={k:int(v) for k,v in counts.items() if k.startswith('excluded_')}
        identities=[{k:v for k,v in item.items() if k!='stat'} for item in self.files]
        excluded_hash=hashlib.sha256(b''.join(sorted(excluded))).hexdigest()
        self.identity={'split':split,'files':identities,'frames':len(self.rows),
            'category_counts':{k:len(v) for k,v in self.buckets.items()},'weights':self.weights,
            'excluded_train_scene_keys':len(excluded),'excluded_pose_key_sha256':excluded_hash,
            'excluded_actual_scene_keys':self.excluded_scene_keys,'excluded_frames':self.excluded_frames,
            'empty_hidden_definition':'Exact Source mask0 with NPC alive and HP>0; never invisible corpses',
            'partial_definition':'Exact visible mask and clipped Source sprite rectangle; screen partial, not a hidden wall oracle'}
        self.identity['fingerprint']=hashlib.sha256(json.dumps(self.identity,sort_keys=True).encode()).hexdigest()

    def _select(self,size,rng,counts,draws,*,unique=False):
        result=[];used=set()
        for _ in range(size):
            category=max(self.weights,key=lambda name:(draws+1)*self.weights[name]-counts[name])
            available=self.buckets[category]
            if unique:available=[index for index in available if index not in used]
            if not available:raise ValueError('Insufficient disjoint fixed-panel capacity for '+category)
            index=int(rng.choice(available));result.append(index);used.add(index)
            counts[category]+=1;draws+=1
        rng.shuffle(result)
        return result,draws

    def fixed_panel(self,count,*,seed=173):
        counts=dict.fromkeys(self.weights,0)
        panel,_=self._select(count,np.random.default_rng(seed),counts,0,unique=True)
        return panel

    def panel_fingerprint(self,panel):
        records=[{**self.rows[i],'path':self.files[self.rows[i]['file']]['path']} for i in panel]
        return hashlib.sha256(json.dumps({'data':self.identity['fingerprint'],'records':records},sort_keys=True).encode()).hexdigest()

    def _read(self,index):
        item=self.files[index];path=Path(item['path'])
        if (path.stat().st_size,path.stat().st_mtime_ns)!=item['stat']:
            raise ValueError('Immutable enemy Source file changed')
        if index in self._cache:
            self._cache.move_to_end(index);return self._cache[index]
        with np.load(path,allow_pickle=False) as source:
            values={key:source[key] for key in ('states','frames','map','object_masks','background_frames')}
            if item['dynamic']:values['actions']=source['actions']
        self._cache[index]=values
        while len(self._cache)>8:self._cache.popitem(last=False)
        return values

    def batch(self,size,*,indices=None):
        if indices is None:
            indices,self.draws=self._select(size,self.rng,self.case_counts,self.draws)
        if len(indices)!=size:raise ValueError('Explicit enemy batch size/index mismatch')
        rows=[]
        for index in indices:
            item=self.rows[int(index)];info=self.files[item['file']];arrays=self._read(item['file'])
            frame=item['frame'];dynamic=info['dynamic']
            rows.append({'state':arrays['states'][frame],'target':arrays['frames'][frame],
                'map':arrays['map'],'object_mask':arrays['object_masks'][frame],
                'background':arrays['background_frames'][frame],
                'previous_state':arrays['states'][frame-1] if dynamic else np.zeros(STATE_DIM,np.float32),
                'previous_rgb':arrays['frames'][frame-1] if dynamic else np.zeros((32,32,3),np.uint8),
                'action':arrays['actions'][frame-1] if dynamic else np.zeros(4,np.float32),
                'dynamic':dynamic,'row_index':int(index)})
        result={}
        for key in rows[0]:
            value=torch.from_numpy(np.stack([row[key] for row in rows]))
            if key in ('target','background','previous_rgb'):value=value.permute(0,3,1,2).float()/255
            elif key=='object_mask':value=value[:,None].float()
            elif key=='map':value=value.long()
            elif key=='dynamic':value=value.bool()
            elif key!='row_index':value=value.float()
            result[key]=value
        return result

    def sampler_state(self):
        return {'rng':self.rng.bit_generator.state,'draws':self.draws,'case_counts':dict(self.case_counts)}

    def restore_sampler(self,state):
        if (state['draws']!=sum(state['case_counts'].values()) or set(state['case_counts'])!=set(self.weights)):
            raise ValueError('Enemy sampler successful-draw clocks changed')
        self.rng.bit_generator.state=state['rng'];self.draws=state['draws'];self.case_counts=dict(state['case_counts'])

