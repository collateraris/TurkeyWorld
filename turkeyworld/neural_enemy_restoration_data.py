"""Actual NPR32/Source128 pairs for RGB-only learned enemy restoration.

No hypothesis cache is loaded. Source state, masks and animation are labels
used by the sampler and losses; the inference input remains only ``lr_npr``.
"""
from copy import deepcopy
import math

import numpy as np
import torch

from .enemy_animation import ANIMATION_CLASSES,enemy_animation_index
from .sr_neural_pairs_data import SemanticSRData,fingerprint
from .small_enemy_data import digest


PROTOCOL='neural_enemy_restoration_data_v1'
SAMPLER_PROTOCOL='neural_enemy_restoration_visible_empty_animation_balanced_v1'
DISTANCES=('near','mid','far')


class NeuralEnemyRestorationData:
    def __init__(self,data_root,split,*,seed=173,device='cpu',exclude_scene_keys=None):
        self.base=SemanticSRData(data_root,split,seed=seed,device=device,exclude_scene_keys=exclude_scene_keys)
        self.root=self.base.root;self.device=self.base.device;self.split=split
        self.rows=self.base.rows;self.scene_keys=self.base.scene_keys
        self.rng=np.random.default_rng(seed);self.draws=0
        self.visible=[];self.distance=[];self.empty_kind=[];self.animation_labels=[]
        self.pools={'visible':{name:[]for name in DISTANCES},'empty':{}}
        self.animation_pools={name:{phase:[]for phase in ANIMATION_CLASSES}for name in DISTANCES}
        arrays={}
        for index,row in enumerate(self.rows):
            name=row['shard']
            if name not in arrays:
                shard=self.base._read(name);states=shard['state'];mask=shard['mask_hr']
                if(states.ndim!=2 or states.shape[1]!=20 or mask.shape!=(len(states),128,128)
                        or not np.isfinite(states).all()):
                    raise ValueError('Neural restoration requires finite state20 and actual native128 masks')
                arrays[name]=(states,mask.reshape(len(states),-1).any(1),
                    enemy_animation_index(torch.from_numpy(states).float()).numpy())
            states,flags,animation=arrays[name];offset=row['offset'];state=states[offset]
            distance=float(row['distance_tiles'])
            if not math.isfinite(distance)or distance<0:raise ValueError('Invalid Source enemy distance label')
            group='near'if distance<2 else'mid'if distance<5 else'far'
            visible=bool(flags[offset]);phase=int(animation[offset])
            empty=('hp0'if state[4]<=0 else'occluded'if state[13]>=.5 and state[12]>0
                else'navigation'if row.get('stratum')=='navigation'else'dead')
            self.visible.append(visible);self.distance.append(group)
            self.empty_kind.append(empty);self.animation_labels.append(phase)
            if visible:
                self.pools['visible'][group].append(index)
                self.animation_pools[group][ANIMATION_CLASSES[phase]].append(index)
            else:self.pools['empty'].setdefault(empty,[]).append(index)
        if any(not ids for ids in self.pools['visible'].values())or not self.pools['empty']:
            raise ValueError('Neural restoration requires actual HR-visible near/mid/far and empty images')
        self.case_counts={'visible':0,'empty':0}
        self.subcounts={pool:dict.fromkeys(groups,0)for pool,groups in self.pools.items()}
        self.animation_subcounts={distance:{phase:0 for phase,ids in phases.items()if ids}
            for distance,phases in self.animation_pools.items()}
        self.identity=dict(protocol=PROTOCOL,base=self.base.identity,split=split,
            base_manifest_sha256=digest(self.root/'manifest.json'),
            sampler_protocol=SAMPLER_PROTOCOL,inference_rgb_only=True,hypothesis_sidecar_required=False,
            labels_are_loss_only=True,actual_hr_visible_empty_weights={'visible':.5,'empty':.5},
            animation_classes=list(ANIMATION_CLASSES),
            animation_sampling='equal_present_animation_classes_within_each_actual_visible_distance',
            actual_hr_mask_pool_counts={pool:{group:len(ids)for group,ids in groups.items()}
                for pool,groups in self.pools.items()},
            animation_counts={distance:{phase:len(ids)for phase,ids in phases.items()}
                for distance,phases in self.animation_pools.items()})
        self.identity['fingerprint']=fingerprint(self.identity)

    def __len__(self):return len(self.rows)

    def fixed_panel(self,count=None,*,seed=173):return self.base.fixed_panel(count,seed=seed)

    def panel_fingerprint(self,panel):
        return fingerprint(dict(identity=self.identity,base_panel=self.base.panel_fingerprint(panel)))

    def batch(self,size,indices=None,rng=None):
        if type(size)is not int or size<1:raise ValueError('Positive neural restoration batch size required')
        if indices is None:
            rng=self.rng if rng is None else rng;indices=[]
            for _ in range(size):
                pool=min(self.case_counts,key=lambda name:(self.case_counts[name],name))
                group=min(self.subcounts[pool],key=lambda name:(self.subcounts[pool][name],name))
                if pool=='visible':
                    phase=min(self.animation_subcounts[group],key=lambda name:(self.animation_subcounts[group][name],name))
                    choices=self.animation_pools[group][phase];self.animation_subcounts[group][phase]+=1
                else:choices=self.pools[pool][group]
                indices.append(int(rng.choice(choices)))
                self.case_counts[pool]+=1;self.subcounts[pool][group]+=1;self.draws+=1
            rng.shuffle(indices)
        if len(indices)!=size or any(type(i)not in(int,np.int64,np.int32)or not 0<=int(i)<len(self)for i in indices):
            raise ValueError('Explicit neural restoration indices differ from this corpus')
        result=self.base.batch(size,indices=indices)
        result['visible_labels']=result['mask_hr'].detach().bool().flatten(1).any(1)[:,None]
        result['animation_label']=torch.tensor([self.animation_labels[int(i)]for i in indices],
            dtype=torch.long,device=self.device)
        result['distance_group']=[self.distance[int(i)]for i in indices]
        result['empty_kind']=[self.empty_kind[int(i)]for i in indices]
        return result

    def sampler_state(self):
        return dict(protocol=SAMPLER_PROTOCOL,identity_sha256=self.identity['fingerprint'],
            rng=deepcopy(self.rng.bit_generator.state),draws=self.draws,
            case_counts=deepcopy(self.case_counts),subcounts=deepcopy(self.subcounts),
            animation_subcounts=deepcopy(self.animation_subcounts))

    def restore_sampler(self,state):
        if(not isinstance(state,dict)or state.get('protocol')!=SAMPLER_PROTOCOL or state.get('identity_sha256')!=self.identity['fingerprint']
                or type(state.get('draws'))is not int or state['draws']<0
                or set(state.get('case_counts',{}))!=set(self.case_counts)
                or any(type(v)is not int or v<0 for v in state['case_counts'].values())
                or sum(state['case_counts'].values())!=state['draws']):
            raise ValueError('Neural restoration sampler identity or draw clock changed')
        if set(state.get('subcounts',{}))!=set(self.subcounts)or set(state.get('animation_subcounts',{}))!=set(self.animation_subcounts):
            raise ValueError('Neural restoration sampler pools changed')
        for pool,expected in self.subcounts.items():
            counts=state['subcounts'][pool]
            if(set(counts)!=set(expected)or any(type(v)is not int or v<0 for v in counts.values())
                    or sum(counts.values())!=state['case_counts'][pool]):
                raise ValueError('Neural restoration distance/empty strata counters changed')
        for group,expected in self.animation_subcounts.items():
            counts=state['animation_subcounts'][group]
            if(set(counts)!=set(expected)or any(type(v)is not int or v<0 for v in counts.values())
                    or sum(counts.values())!=state['subcounts']['visible'][group]):
                raise ValueError('Neural restoration animation counters changed')
        self.rng.bit_generator.state=deepcopy(state['rng']);self.draws=state['draws']
        self.case_counts=deepcopy(state['case_counts']);self.subcounts=deepcopy(state['subcounts'])
        self.animation_subcounts=deepcopy(state['animation_subcounts'])
