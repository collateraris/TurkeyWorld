"""Private motion replay PCGs with a fixed8+8 actor contract.

New policy corpora are immutable TRAIN-only additions. Half of each cohort
comes from the newest collection, half from original replay; no validation
file or independent combat/contact sampler is consumed.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .jepa_training_utils import SequenceData


class MotionReplay:
    def __init__(self, directory, sampler_state, *, stream, device='cpu', corpora=()):
        if stream not in ('navigation','enemy_motion'):
            raise ValueError('Motion replay requires a declared navigation/quiet cohort')
        self.stream=stream;self.device=device;self.rng=np.random.default_rng()
        self.protocol=sampler_state['protocol'];self.draws=sampler_state['draws']
        self.rng.bit_generator.state=deepcopy(sampler_state['rng'])
        directories=[Path(directory)]
        key='nav_train_directory'if stream=='navigation'else'enemy_train_directory'
        directories.extend(Path(receipt[key])for receipt in corpora)
        self.data=[]
        for data_index,path in enumerate(directories):
            if path.name!='train':raise ValueError('Motion replay refuses any split except TRAIN')
            files=sorted(path.glob('*.npz'))
            if data_index:
                receipt=corpora[data_index-1]
                declared=sorted(Path(row['path'])for row in receipt['episodes']
                    if Path(row['path']).resolve().parent==path.resolve())
                if [f.resolve()for f in files]!=[f.resolve()for f in declared]:
                    raise ValueError('Motion corpus directory changed after its immutable receipt')
            self.data.append(SequenceData(path,device=device,split='train',image_size=32,
                mode='stream',cache_bytes=128*1024**2,episode_paths=files))

    def state_dict(self):
        return dict(protocol=self.protocol,draws=self.draws,rng=deepcopy(self.rng.bit_generator.state))

    def restore_state(self,value):
        if value['protocol']!=self.protocol or type(value['draws'])is not int or value['draws']<0:
            raise ValueError('Motion replay identity or actor counter changed')
        self.rng.bit_generator.state=deepcopy(value['rng']);self.draws=value['draws']

    def batch(self,*,horizon):
        if horizon not in(16,48,80,112,144,180):raise ValueError('Unexpected six-second motion horizon')
        before=self.state_dict()
        try:
            pools=[]
            for data_index,data in enumerate(self.data):
                pools.append([(data_index,i)for i,episode in enumerate(data.episodes)
                    if len(episode['frames'])>=horizon+1])
            if any(not pool for pool in pools):raise ValueError('Motion replay has an incomplete policy corpus')
            if len(pools)>1:
                earlier=pools[0]
                selected=[earlier[int(i)]for i in self.rng.integers(len(earlier),size=4)]
                selected +=[pools[-1][int(i)]for i in self.rng.integers(len(pools[-1]),size=4)]
            else:
                selected=[pools[0][int(i)]for i in self.rng.integers(len(pools[0]),size=8)]
            batches=[];files=[]
            for data_index,episode_index in selected:
                data=self.data[data_index]
                batches.append(data.batch(1,context=1,horizon=horizon,
                    indices=np.asarray([[episode_index,0]],dtype=np.int64)))
                files.append(data.episodes[episode_index]['path'])
            batch={key:torch.cat([value[key]for value in batches],0)for key in batches[0]}
            states,actions=batch['states'],batch['actions']
            if bool(actions[...,3].ne(0).any()):raise ValueError('Motion replay cannot contain firing')
            if self.stream=='navigation':
                if bool(states[...,12:14].ne(0).any()):raise ValueError('No-enemy cohort contains an NPC')
            elif (not bool(((states[...,12]>0)&(states[...,13]>=.5)&(states[...,4]>0)).all())
                    or not torch.equal(states[...,4],states[:,:1,4].expand_as(states[...,4]))
                    or not torch.equal(states[...,12],states[:,:1,12].expand_as(states[...,12]))):
                raise ValueError('Quiet enemy replay contains death or changing HP')
            self.draws+=8
            return batch,dict(stream=self.stream,actors=8,context=1,horizon=horizon,
                episode_files=files,start_indices=[0]*8,private_pcg_draws=self.draws,
                latest_policy_actors=4 if len(pools)>1 else 0,
                previous_replay_actors=4 if len(pools)>1 else 8,Source_future_feedback=False)
        except BaseException:
            self.restore_state(before);raise
