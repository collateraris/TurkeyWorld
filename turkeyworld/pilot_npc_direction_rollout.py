"""Conditional H180 control: reference player XY for all branches, no NPC teacher resets."""
import argparse
import json
from pathlib import Path

import torch

from .pilot_npc_direction import DirectionHead, features, load
from .structured_state import NPC_SPEED_BOUND
from .train_vjepa_sr_dynamics import ENEMY
from .training_pipeline import atomic_json, now
from .trajectory_objectives import circle_wall_penalty


@torch.no_grad()
def roll(initial, player_xy, grid, estimator):
    # Only Source input after initialization: player XY(t), available before
    # predicting NPC(t+1); never feed either true NPC trajectory or true velocity.
    s=initial.clone()[None]
    positions=[]
    for xy in player_xy:
        s[:,:2]=xy
        v = estimator(s,grid)
        v = v * (NPC_SPEED_BOUND/v.norm(dim=-1,keepdim=True).clamp_min(1e-8)).clamp_max(1)
        prev = s[:,8:10].clone()
        s[:,8:10]=(prev+v/480).clamp(0,1)
        s[:,10:12]=(s[:,8:10]-prev)*240
        positions.append(s[0,8:10].clone())
    return torch.stack(positions)


@torch.no_grad()
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    a=p.parse_args(argv);torch.set_num_threads(2)
    r=json.loads((a.pilot/'report.json').read_text())
    b=json.loads(a.baseline.read_text())
    if not b['complete'] or b['step']!=r['step']: raise ValueError('Expected matching, complete baseline')
    net=DirectionHead();net.load_state_dict(torch.load(a.pilot/'experimental_head.pt',map_location='cpu',weights_only=True)['model']);net.eval()
    baseline={(w['episode'],w['start']):w for w in b['per_window']}
    rows=[]
    for w in r['panel']:
        s,_,g=load(ENEMY.parent/'val'/w['episode'],'val')
        t=w['start'];prompt=s[t,:]; player=s[t:t+180,:2]
        target=s[t+1:t+181,8:10]
        row=dict(episode=w['episode'],start=t,vjepa_true_player_H180=baseline[w['episode'],t]['true_player_H180'])
        paths={'carry':roll(prompt,player,g,lambda state,_:state[:,10:12]*2),
               'head':roll(prompt,player,g,lambda state,grid:net(features(state,grid))),
               'source':target}
        for k,pred in paths.items():
            err=((pred-target)*16).norm(dim=-1)
            walls=circle_wall_penalty(pred[None]*16,g[None],.22)[0]>0
            row[k]=dict(H180_tiles=float(err[-1]),mean_error_tiles=float(err.mean()),overlap_ticks=int(walls.sum()))
        rows.append(row)
    summary={k:{m:sum(v[k][m] for v in rows)/len(rows) for m in rows[0][k]} for k in paths}
    summary['vjepa_true_player_H180']=b['summary']['H180']['true_player_H180']
    for k in ('carry','head'):
        summary[k]['better_than_vjepa_windows']=sum(v[k]['H180_tiles']<v['vjepa_true_player_H180'] for v in rows)
    atomic_json(a.pilot/'conditional_H180.json',dict(protocol='quiet_source_player_only_conditional_H180_v1',
                generated_at=now(),step=r['step'],baseline=str(a.baseline),windows=len(rows),
                source_after_context=['player XY only'], source_future_NPC_inputs=False,dt=1/30,
                radius=.22, speed_limit=NPC_SPEED_BOUND, native_wall_overlap_available=False,
                integration='accumulate own velocity, world boundary clamp, no wall or route oracle',
                summary=summary, per_window=rows))
    print(json.dumps(summary))


if __name__=='__main__':
    main()
