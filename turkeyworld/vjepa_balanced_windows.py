"""Exact available-class window pools; no failed-quota rejection fallback."""
import numpy as np
import torch
from .vjepa_motion_aux import commands

def motion_window_index(episodes,horizon=180):
    pools={kind:[] for kind in ('moving','blocked','idle')}
    for ep,e in enumerate(episodes):
        actions=np.asarray(e['actions']);states=np.asarray(e['states']);count=len(actions)-horizon+1
        if count<=0:continue
        cmd=commands(torch.as_tensor(states[:-1]).float(),torch.as_tensor(actions).float()).numpy()
        requested=np.abs(cmd)>1e-5;delta=states[1:,:2]-states[:-1,:2]
        series=[requested.any(-1),requested.sum(-1),(requested&(np.abs(delta)<np.abs(cmd)*.25)).sum(-1)]
        sums=[]
        for values in series:
            prefix=np.r_[0,np.cumsum(values)];sums.append(prefix[32:32+count]-prefix[:count])
        active,axes,blocked=sums
        kinds=np.where(active<8,'idle',np.where(blocked>axes*.25,'blocked','moving'))
        for kind in pools:
            starts=np.flatnonzero(kinds==kind)
            if len(starts):pools[kind].append(np.column_stack((np.full(len(starts),ep),starts)))
    return {k:np.concatenate(v) for k,v in pools.items() if v}

def draw_available(pools,rng,rotation):
    available=tuple(pools)
    if not available:raise ValueError('No legal H180 TRAIN windows')
    kind=available[rotation%len(available)]
    episode,start=pools[kind][int(rng.integers(len(pools[kind])))]
    return int(episode),int(start),kind,available
