"""Train a small local NPC velocity head on quiet TRAIN only; paired one-step VAL.

All predictors start from the same recorded pose and NPC velocity for each
one-step forecast. Future states serve only as supervision, evaluation and
selection of diagnostic strata. This does not test closed-loop stability.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .audit_vjepa_npc_target import source_masks
from .models import load_world_checkpoint
from .rssm_combat_protocol import environment_profile
from .train_vjepa_sr_dynamics import ENEMY
from .training_pipeline import atomic_json, now
from .vjepa_policy_world import VJEPAImaginationAdapter
from .workspace_paths import ROOT


def features(state, grid):
    # Index the map using only current NPC coordinates; outside-map = solid.
    npc = state[:,8:10] * 16
    cell = npc.floor().long()
    d = torch.arange(-2,3,device=state.device)
    iy, ix = torch.meshgrid(d,d,indexing='ij')
    local = F.pad((grid != 0).float(), (3,3,3,3), value=1)
    patch = local[(cell[:,1,None] + 3 + iy.flatten()).clamp(0,21),
                  (cell[:,0,None] + 3 + ix.flatten()).clamp(0,21)]
    rel = (state[:,:2] - state[:,8:10])*16
    return torch.cat([patch, (npc-cell)-.5, rel/16, F.normalize(rel,dim=-1),
                      rel.norm(dim=-1,keepdim=True)/16, state[:,10:12]*2],1)


class DirectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(34,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,2))

    def forward(self,x):
        return self.net(x) + x[:,-2:]


def load(path, split):
    with np.load(path,allow_pickle=False) as d:
        if d['split'].item()!=split or d['enemy_motion_protocol'].item()!='source_single_enemy_motion_no_shooting_v1':
            raise ValueError('Not the requested quiet Source split')
        s,a,g = [torch.from_numpy(d[k].copy()) for k in ('states','actions','map')]
        if not bool((s[:,13] > .5).all()) or bool((a[:,3] > .5).any()) or not np.isclose(float(d['dt']),1/30):
            raise ValueError('Unexpected cohort')
        return s,a,g


def velocity_labels(before,after):
    return (after[:,8:10] - before[:,8:10]) * 480


def evaluate(pred,target):
    errors = (pred-target).norm(dim=-1)
    speed = target.norm(dim=-1)
    moving = speed > .05
    stopped = pred.norm(dim=-1) <= .05
    # A stopped prediction on a moving label counts as a 90° direction error.
    cos = F.cosine_similarity(pred,target,dim=-1).clamp(-1,1)
    angle = torch.where(stopped,torch.zeros_like(cos),cos).acos().rad2deg()
    return dict(vector_error_tiles_s=float(errors.mean()),moving_ticks=int(moving.sum()),
                angle_deg=float(angle[moving].mean()) if moving.any() else None,
                stopped_on_moving=int((stopped & moving).sum()))


def turn_onset_summary(table):
    rows = [r for r in table if r['group'] == 'occluded_recent_turn']
    carry, truth = (torch.tensor([r[k] for r in rows]) for k in ('carry','truth'))
    turn = ((carry.norm(dim=-1) > .05) & (truth.norm(dim=-1) > .05)
            & (F.cosine_similarity(carry,truth,dim=-1) < np.cos(np.pi/6)))
    return {k: dict(ticks=int(mask.sum()),
                    **{v:evaluate(torch.tensor([r[v] for r in rows])[mask],truth[mask])
                       for v in ('vjepa','head','carry')} if mask.any() else {})
            for k,mask in [('at_turn',turn),('remaining_recent_turn',~turn)]}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--world', required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu', help='V-JEPA evaluation only; local head always trains on CPU')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--batch',type=int,default=12)
    a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(a.threads);torch.manual_seed(2461)
    files=sorted(ENEMY.glob('episode*.npz'))
    if not files: raise ValueError('No train episodes')
    feat,vel,turns=[],[],[]
    for f in files:
        s,act,grid=load(f,'train')
        feat.append(features(s[:-1],grid))
        y=velocity_labels(s[:-1],s[1:]);vel.append(y)
        prev=s[:-1,10:12]*2
        turns.append((prev.norm(dim=-1)>.05)&(y.norm(dim=-1)>.05)&(F.cosine_similarity(y,prev,dim=-1)<np.cos(np.pi/6)))
    x,y,b=torch.cat(feat),torch.cat(vel),torch.cat(turns)
    ids=b.nonzero().flatten()
    if not ids.numel(): raise ValueError('No labeled training turns')
    net=DirectionHead()
    opt=torch.optim.AdamW(net.parameters(),lr=.002)
    for _ in range(600):
        sample=torch.cat([torch.randint(len(x),(192,)),ids[torch.randint(len(ids),(64,))]])
        pred=net(x[sample]); label=y[sample]
        loss=F.smooth_l1_loss(pred,label,beta=.1)
        opt.zero_grad();loss.backward();opt.step()
    net.eval();torch.save({'model':net.state_dict(),'protocol':'npc_local_velocity_v1','seed':2461},
                          a.out/'experimental_head.pt')
    panel=json.loads((ROOT/'training_runs/rssm_moe_nav15_enemy25_cap20_20261113/rssm_validation_latest.json').read_text())['enemy_motion']['panel']['windows']
    train_names={f.name for f in files}
    if any(w['episode'] in train_names for w in panel): raise ValueError('Episode leakage')
    rng=np.random.default_rng(2461);rows=[]
    for w in panel:
        s,act,grid=load(ENEMY.parent/'val'/w['episode'],'val')
        t=w['start'];m=source_masks(grid.numpy(),s[t].numpy(),s[t+1:t+181].numpy())
        x=features(s[t:t+180],grid);y=velocity_labels(s[t:t+180],s[t+1:t+181])
        # Evaluate ALL occluded post-turn ticks, plus at most 12 random comparison ticks per
        # window in each other stratum. This is purposefully not a population-wide mean.
        for name in ('occluded_recent_turn','occluded_no_recent_turn','visible'):
            idx=np.flatnonzero(m[name])
            if name!='occluded_recent_turn' and len(idx)>12:
                idx=np.sort(rng.choice(idx,12,replace=False))
            for j in idx:
                tt=int(t+j)
                rows.append(dict(group=name,episode=w['episode'],tick=tt,
                                 past=s[[max(0,tt+k) for k in (-3,-2,-1,0)]], action=act[tt],
                                 features=x[j], truth=y[j],grid=grid))
    world,saved=load_world_checkpoint(a.world,'cpu');step=saved['training_steps'];del saved
    if not world.vjepa_sr_feedback: raise ValueError('Expected SR feedback')
    adapter=VJEPAImaginationAdapter(world.to(a.device))
    with torch.no_grad(),environment_profile(world,navigation_only=False):
        for i in range(0,len(rows),a.batch):
            batch=rows[i:i+a.batch]
            past,actions,g=[torch.stack([r[k] for r in batch]).to(a.device) for k in ('past','action','grid')]
            cur=adapter.initialize(past,g,past.new_zeros(len(batch),4,3,32,32))
            # Last-step feedback updates only memory, not the one-step position we score.
            nxt=adapter.transition_prior(cur,actions)
            v=velocity_labels(cur.state,nxt.state).cpu()
            h=net(torch.stack([r['features'] for r in batch]))
            for j,r in enumerate(batch):r.update(vjepa=v[j],head=h[j],carry=r['past'][-1,10:12]*2)
            print(f'Evaluated {min(i+a.batch,len(rows))}/{len(rows)}',flush=True)
    result={}
    for name in ('occluded_recent_turn','occluded_no_recent_turn','visible'):
        group=[r for r in rows if r['group']==name]
        result[name]={'ticks':len(group),'episodes':len({r['episode'] for r in group}),
                      **{k:evaluate(torch.stack([r[k] for r in group]),torch.stack([r['truth'] for r in group]))
                         for k in ('vjepa','head','carry')}}
    table=[{k:(r[k].tolist() if torch.is_tensor(r[k]) else r[k]) for k in ('episode','tick','group','truth','vjepa','head','carry')} for r in rows]
    atomic_json(a.out/'report.json',dict(protocol='isolated_local_npc_velocity_v1', generated_at=now(),step=step,
                horizon=1,rollout_feedback='no future states; both use observed start',
                device=a.device, updates=600,seed=2461,head_parameters=sum(p.numel() for p in net.parameters()),
                training_episodes=len(files),training_ticks=sum(len(e) for e in vel),train_turn_ticks=len(ids),
                train_files=[f.name for f in files],panel=panel,validation_used_for_training_or_selection=False,
                summary=result,occluded_turn_detail=turn_onset_summary(table),
                speed_means_tiles_s={k:float(torch.tensor([e[k] for e in table if e['group']=='occluded_recent_turn']).norm(dim=-1).mean())
                                     for k in ('truth','vjepa','head','carry')},per_tick=table))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
