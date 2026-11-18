"""Held-out navigation stress test on new Source trajectories, independent of training.

The original supplies targets, three stationary context ticks and the initial condition.
The H180 prior is never reset; one-step contact probes start separately every 16 ticks.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .engine import Game
from .models import load_world_checkpoint
from .rssm_combat_protocol import environment_profile
from .structured_state import player_command_delta
from .training_pipeline import atomic_json, now
from .trajectory_objectives import circle_wall_penalty
from .vjepa_independent_panel import axis_errors, common_start_probes, PROBE_OFFSETS, summarize_probes
from .vjepa_policy_world import VJEPAImaginationAdapter

SEED = 170722148
COUNT = 12  # per scenario kind
RULE = 'nav_wall_slide_corner_departure_H180_v1'


def nav_contacts():
    rng = np.random.default_rng(SEED)
    game = Game(seed=SEED, endless=True, enemies_enabled=False)
    grid = game.map
    directions = [(1,0),(0,1),(-1,0),(0,-1)]
    specs = []
    for kind in ('strafe_wall','corner_exit'):
        options = []
        for y in range(1,15):
            for x in range(1,15):
                if grid[y,x]: continue
                for nx,ny in directions:
                    if not grid[y+ny,x+nx]: continue
                    for sign in (-1,1):
                        tx,ty = sign*ny,-sign*nx
                        if kind == 'strafe_wall':
                            if grid[y+ty,x+tx] or not grid[y+ty+ny,x+tx+nx]: continue
                        elif not grid[y+ty,x+tx]: continue
                        options.append((x,y,nx,ny,tx,ty))
        seen=set()
        for i in rng.permutation(len(options)):
            x,y,nx,ny,tx,ty=options[i]
            identity=(x,y) if kind=='strafe_wall' else (x,y,nx,ny)
            if identity in seen: continue
            seen.add(identity)
            pos=np.array([x+.5,y+.5])+rng.uniform(.18,.275)*np.array([nx,ny])
            pos+=rng.uniform(.14,.25)*np.array([tx,ty]) if kind=='corner_exit' else rng.uniform(-.23,.23)*np.array([tx,ty])
            # In the wall strafe case camera faces the wall. Strafe slides
            # sideways while the commanded forward axis keeps touching the wall.
            angle=math.atan2(ny,nx)+rng.uniform(-.11,.11)
            side=float(nx*ty-ny*tx)
            actions=np.zeros((180,4),np.float32)
            if kind=='strafe_wall':
                actions[:50,:2]=[.65,side]
                actions[50:98,:2]=[.45,-side]
                actions[98:126,:3]=[-.8,.35*side,-.12*side]
                actions[126:,:2]=[.85,.7*side]
            else:
                actions[:47,:2]=[.9,.85*side]
                actions[47:85,:2]=[-.8,.8*side]
                actions[85:138,:2]=[.5,-side]
                actions[138:,:3]=[-.65,-.65*side,.24*side]
            actions[:,:2]*=rng.uniform(.76,.99)
            specs.append((kind,pos,angle,actions))
            if len(seen)==COUNT: break
        if len(seen)!=COUNT:raise RuntimeError('Not enough novel starts')
    episodes=[]
    for i,(kind,xy,angle,actions) in enumerate(specs):
        g=game;g.reset(SEED+i+1)
        z=g.get_state();z['player'].update(x=float(xy[0]),y=float(xy[1]),angle=float(angle));g.set_state(z)
        states=[g.state_vector()]
        for a in [*np.zeros((3,4)),*actions]:
            g.step(a,1/30);states.append(g.state_vector())
        states=np.stack(states)
        payload=np.concatenate((states.flatten(),actions.flatten())).astype('<f4').tobytes()+grid.tobytes()
        episodes.append(dict(kind=kind,id=i,states=states,actions=actions,grid=grid.copy(),x=float(xy[0]),y=float(xy[1]),angle=float(angle),
                             trajectory_sha256=hashlib.sha256(payload).hexdigest()))
    return episodes


def manifest(episodes):
    keys=('id','kind','x','y','angle','trajectory_sha256')
    rows=[{k:e[k] for k in keys} for e in episodes]
    return dict(protocol=RULE,seed=SEED,fingerprint=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),episodes=rows)


def denominator(episodes):
    out={}
    for kind in ['strafe_wall','corner_exit']:
        rows=[e for e in episodes if e['kind']==kind]
        prev=torch.from_numpy(np.stack([e['states'][3:-1][list(PROBE_OFFSETS)] for e in rows])).flatten(0,1)
        following=torch.from_numpy(np.stack([e['states'][4:][list(PROBE_OFFSETS)] for e in rows])).flatten(0,1)
        ctrl=torch.from_numpy(np.stack([e['actions'][list(PROBE_OFFSETS)] for e in rows])).flatten(0,1)
        d=following[:,:2]-prev[:,:2]
        out[kind]=axis_errors(player_command_delta(prev,ctrl),d,d)
    return out


@torch.inference_mode()
def run(a):
    eps=nav_contacts(); spec=manifest(eps); coverage=denominator(eps)
    if any(v['free_axes']<40 or v['blocked_axes']<40 for v in coverage.values()):
        raise RuntimeError(f'Hard panel lost contact coverage: {coverage}')
    if a.manifest:
        if a.manifest.exists() and json.loads(a.manifest.read_text())!=spec: raise RuntimeError('Registered panel mismatch')
        atomic_json(a.manifest,spec)
    if a.world is None:
        print('Registered panel:',spec['fingerprint'],coverage);return
    if a.output is None: raise ValueError('Supply --output for checkpoint evaluation')
    world, saved=load_world_checkpoint(a.world,'cpu');step=saved['training_steps']; mode=world.config.player_motion_mode;del saved
    world=world.to(a.device);adapter=VJEPAImaginationAdapter(world)
    out=dict(protocol=RULE,panel_sha256=spec['fingerprint'],step=step,player_motion_mode=mode,generated_at=now(),
             closed_loop=180,context=4,source_future_feedback=False,generated_SR=True,per_class={},per_window=[])
    with environment_profile(world,navigation_only=True):
        for kind in ['strafe_wall','corner_exit']:
            rows=[e for e in eps if e['kind']==kind];probes=[];stats=[]
            for i in range(0,len(rows),a.batch):
                batch=rows[i:i+a.batch]
                def stack(k):return torch.from_numpy(np.stack([e[k] for e in batch])).to(a.device)
                states=stack('states');past=states[:,:4];target=states[:,4:];grid=stack('grid').long();actions=stack('actions')
                cur=adapter.initialize(past,grid,past.new_zeros(len(batch),4,3,32,32));pts=[]
                for tick in range(180):
                    cur=adapter.transition_prior(cur,actions[:,tick]);pts.append(cur.state[:,:2])
                pred=torch.stack(pts,1);errors=((pred-target[:,:,:2])*16).norm(dim=-1)
                overlap=(circle_wall_penalty(pred*16,grid,.20)>0).sum(1)
                gt_overlap=(circle_wall_penalty(target[:,:,:2]*16,grid,.20)>0).sum(1)
                probes.append(common_start_probes(adapter,past,grid,actions,target))
                for j,e in enumerate(batch):
                    stats.append(dict(id=e['id'],kind=kind,trajectory_sha256=e['trajectory_sha256'],H180=float(errors[j,-1]),
                                      mean_path_error=float(errors[j].mean()),H16=float(errors[j,15]),H112=float(errors[j,111]),
                                      wall_ticks=int(overlap[j]),reference_wall_ticks=int(gt_overlap[j])))
            out['per_window'].extend(stats); p=summarize_probes(probes)
            out['per_class'][kind]={**{k:float(np.mean([s[k] for s in stats])) for k in ('H180','mean_path_error','H16','H112','wall_ticks','reference_wall_ticks')},'probes':p}
            for k in ['free_axes','blocked_axes']:
                if p[k]!=coverage[kind][k]: raise RuntimeError('Common-start labels diverged')
            print(kind,out['per_class'][kind],flush=True)
    atomic_json(a.output,out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--world',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--manifest',type=Path);p.add_argument('--batch',type=int,default=4);p.add_argument('--device',default='cuda')
    a=p.parse_args();torch.set_num_threads(2);run(a)
