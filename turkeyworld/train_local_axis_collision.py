"""Isolated supervised learning of player collision decisions on TRAIN trajectories.

Analytic geometry is used here only to create offline labels; the inference module
has no dependency on this trainer or the reference game.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .engine import Game
from .learned_axis_collision import LocalAxisCollisionHead, with_local_axis_head
from .structured_state import player_command_delta
from .train_vjepa_sr_dynamics import NAV, ENEMY
from .training_pipeline import atomic_json


def walkable(maps, idx, xy, *, radius=Game.PLAYER_RADIUS):
    """Vectorized TRAIN teacher (circle vs occupied cells); validate against Game below."""
    cell = np.floor(xy).astype(int)
    valid = np.all((xy >= radius) & (xy < 16-radius),axis=1)
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            col=cell[:,0]+dx; row=cell[:,1]+dy
            opaque=maps[idx,np.clip(row,0,15),np.clip(col,0,15)]!=0
            ex=np.maximum.reduce((col-xy[:,0], xy[:,0]-col-1., np.zeros(len(xy))))
            ey=np.maximum.reduce((row-xy[:,1], xy[:,1]-row-1., np.zeros(len(xy))))
            valid &= ~(opaque & (ex*ex+ey*ey < radius**2))
    return valid


def teacher(maps,idx,state,action):
    cmd=player_command_delta(torch.from_numpy(state), torch.from_numpy(action)).numpy().astype(np.float64)
    xy=state[:,:2].astype(np.float64)*16
    # Exactly one 30 Hz Source step. Handle rounded command magnitudes above .08 too.
    d=cmd*16
    ns=np.maximum(1,np.ceil(np.abs(d).max(axis=1)/.08).astype(int))
    frac=np.zeros((len(state),2))
    for j in range(ns.max()):
        for axis in (0,1):
            to=xy.copy()
            to[:,axis]+=d[:,axis]/ns
            ok=(j<ns) & walkable(maps,idx,to)
            xy[ok,axis]=to[ok,axis]
            frac[ok,axis]+=1/ns[ok]
    return frac.astype(np.float32)


def read_train():
    states=[];actions=[];truth=[];indices=[];maps=[];mapids={};files=[]
    for folder in (NAV,ENEMY):
        if folder.name!='train':raise ValueError('TRAIN only')
        for file in sorted(folder.glob('episode*.npz')):
            with np.load(file,allow_pickle=False) as r:
                if r['split'].item()!='train' or abs(float(r['dt'])-1/30)>1e-6:raise ValueError(file)
                s=r['states'].astype(np.float32);a=r['actions'].astype(np.float32);g=r['map'].astype(np.uint8)
            if s.shape!=(len(a)+1,20):raise ValueError(file)
            key=g.tobytes()
            if key not in mapids:mapids[key]=len(maps);maps.append(g)
            cmd=player_command_delta(torch.from_numpy(s[:-1]),torch.from_numpy(a)).numpy()
            f=np.clip((s[1:,:2]-s[:-1,:2])/np.where(np.abs(cmd)>1e-6,cmd,1),0,1).astype(np.float32)
            states.append(s[:-1]);actions.append(a);truth.append(f);indices.append(np.full(len(a),mapids[key],np.int64));files.append(str(file))
    if not files:raise ValueError('Empty TRAIN dataset')
    return tuple(np.concatenate(v) for v in (states,actions,indices,truth)),np.stack(maps),files


def proposals(rng,real,maps,count):
    """Uniform subcell coverage + perturbed poses near genuine TRAIN trajectories.
    Include tangent, diagonal and reversing commands with random headings.
    """
    batches=[]
    for i in range(0,count,8192):
        n=min(count-i,8192); rows=rng.integers(len(real[0]),size=n)
        s,a,m=(v[rows].copy() for v in real[:3]);freeze=n//2
        # Choose only cells and maps seen in the TRAIN corpus.
        for mid in np.unique(m):
            mask=(m==mid)&(np.arange(n)>=freeze);free=np.argwhere(maps[mid]==0)
            cells=free[rng.integers(len(free),size=np.count_nonzero(mask))][:,::-1]
            s[mask,:2]=(cells+rng.random(cells.shape))/16
        s[:freeze,:2]+=rng.normal(0,.24,size=(freeze,2)).astype(np.float32)/16
        angle=rng.uniform(-np.pi,np.pi,n);s[:,2]=np.sin(angle);s[:,3]=np.cos(angle)
        a[:,:2]=rng.uniform(-1,1,(n,2));a[::3,:2]=rng.choice([-1.,0.,1.],size=(len(a[::3]),2))
        a[:,2]=rng.uniform(-1,1,n);a[:,3]=0
        ok=walkable(maps,m,s[:,:2].astype(np.float64)*16)
        s,a,m=s[ok],a[ok],m[ok];f=teacher(maps,m,s,a)
        batches.append((s,a,m,f))
    return tuple(np.concatenate(v) for v in zip(*batches))


def bins(data):
    s,a,m,f=data
    active=player_command_delta(torch.from_numpy(s),torch.from_numpy(a)).abs().numpy()>1e-5
    stop=(f<.25)&active
    return [np.flatnonzero(mask) for mask in (active.any(1)&~stop.any(1),stop.any(1)&((f>.75)&active).any(1),stop.any(1)&~((f>.75)&active).any(1))]


def sample(rng,data,pools,n):
    pools=[v for v in pools if len(v)]
    if not pools:raise ValueError('TRAIN bucket empty')
    ids=np.concatenate([rng.choice(v,n//len(pools)+(j<n%len(pools))) for j,v in enumerate(pools)])
    return tuple(v[ids] for v in data)


@torch.no_grad()
def own_train(rng,net,real,maps,grid_tensor,actors=384,ticks=32):
    """DAgger-style: supervise own visited poses. TRAIN targets never reset these trajectories."""
    ids=rng.integers(len(real[0]),size=actors)
    state,act,mid=(v[ids].copy() for v in real[:3]);acc=[]
    # Preserve TRAIN action distribution; keep command for a short burst then change.
    for t in range(ticks):
        if t and t%8==0:act=real[1][rng.integers(len(real[1]),size=actors)].copy()
        ss=torch.from_numpy(state);aa=torch.from_numpy(act)
        result=net(ss,aa,grid_tensor[mid])
        fraction=teacher(maps,mid,state,act)
        acc.append((state.copy(),act.copy(),mid.copy(),fraction))
        state[:,:2]+=result['raw_player_delta'].numpy()
        angle=np.arctan2(state[:,2],state[:,3])+act[:,2]*2.2/30
        state[:,2]=np.sin(angle);state[:,3]=np.cos(angle)
    return tuple(np.concatenate(v) for v in zip(*acc))


def verify_teacher(data,maps,rng):
    s,a,m,f=data
    ids=rng.choice(len(s),size=min(512,len(s)),replace=False)
    cmd=player_command_delta(torch.from_numpy(s[ids]),torch.from_numpy(a[ids])).numpy()*16
    worst=0
    for j,i in enumerate(ids):
        g=Game.__new__(Game);g.map=maps[m[i]];g.map_height,g.map_width=g.map.shape
        xy=s[i,:2].astype(np.float64)*16; p={'x':xy[0],'y':xy[1]}
        g._move(p,*cmd[j].astype(np.float64),Game.PLAYER_RADIUS)
        worst=max(worst,np.max(np.abs((np.array([p['x'],p['y']])-xy)-cmd[j]*f[i])))
    if worst>1e-6:raise ValueError(f'TRAIN teacher disagrees with Game: {worst}')
    return float(worst)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--steps',type=int,default=10000);p.add_argument('--batch',type=int,default=320);p.add_argument('--threads',type=int,default=2);p.add_argument('--seed',type=int,default=731)
    a=p.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(a.threads);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
    if (out/'candidate.pt').exists():raise FileExistsError('Use a fresh isolated output directory')
    real,maps,files=read_train();aug=proposals(rng,real,maps,400000);gt=torch.from_numpy(maps)
    p_real,p_aug=bins(real),bins(aug)
    receipt=dict(protocol='local_axis_train_only_v1',seed=a.seed,oracle_scope='TRAIN labels only',source='one-step Game._move X then Y',
                 source_train_files=files,unique_train_maps=len(maps),actual_transitions=len(real[0]),legal_augmented=len(aug[0]),
                 pools={k:list(map(len,v)) for k,v in [('actual',p_real),('augmented',p_aug)]},
                 teacher_max_error_tiles=verify_teacher(aug,maps,rng),lr=1e-5,batch=a.batch,planned_steps=a.steps,
                 trainable='local_axis_head only',training_uses_heldout=False,source_future_feedback=False)
    atomic_json(out/'train_plan.json',receipt)
    net=LocalAxisCollisionHead().train();opt=torch.optim.AdamW(net.parameters(),lr=1e-5,weight_decay=1e-4)
    own,p_own=None,None;seen=0;loss_sum=0
    for step in range(1,a.steps+1):
        if step>=501 and (step-501)%1000==0:
            own=own_train(rng,net,real,maps,gt);p_own=bins(own);seen+=len(own[0])
        n0=a.batch//4; n1=a.batch//4 if own is not None else 0
        samples=[sample(rng,real,p_real,n0),sample(rng,aug,p_aug,a.batch-n0-n1)]
        if n1:samples.append(sample(rng,own,p_own,n1))
        s,act,mid,f=(torch.from_numpy(np.concatenate(v)) for v in zip(*samples))
        res=net(s,act,gt[mid]);mask=res['player_command_delta'].abs()>1e-5
        logits=res['player_pass_logits'];loss=F.binary_cross_entropy_with_logits(logits[mask],f[mask]);opt.zero_grad();loss.backward();opt.step();loss_sum+=float(loss.detach())
        if step%500==0 or step==a.steps:
            with torch.no_grad():
                hard=res['player_pass_fraction'];pred=hard[mask]>.5;truth=f[mask]>.5
                stat=dict(step=step,avg_train_loss=loss_sum/(500 if step%500==0 else step%500),batch_false_pass=int((pred&~truth).sum()),batch_false_stop=int((~pred&truth).sum()),own_examples_generated=seen)
            loss_sum=0
            atomic_json(out/'train_status.json',stat)
            print(json.dumps(stat),flush=True)
    torch.save({'model':net.state_dict(),'optimizer':opt.state_dict(),'steps':a.steps,'plan':receipt},out/'head.pt')
    base=torch.load(a.base,map_location='cpu',mmap=True,weights_only=False)
    candidate=with_local_axis_head(base,net)
    candidate['local_axis_training']=dict(protocol=receipt['protocol'],head_steps=a.steps,lr=receipt['lr'],seed=a.seed,
                                          source_world_step=base['training_steps'],own_examples_generated=seen,optimizer=opt.state_dict())
    # Backbone, parent Adam, world clocks & RNG have never been passed to the training function.
    torch.save(candidate,out/'candidate.tmp.pt');(out/'candidate.tmp.pt').replace(out/'candidate.pt')
    atomic_json(out/'train_complete.json',dict(step=base['training_steps'],head_steps=a.steps,learned_parameters=sum(p.numel() for p in net.parameters()),
                   lr=receipt['lr'],own_examples_generated=seen,complete=True))
    print('Candidate complete',flush=True)


if __name__=='__main__':
    main()
