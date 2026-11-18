"""Deterministic Source-only TRAIN collection of sustained180-tick commands."""
import argparse,json,math,time,hashlib
from pathlib import Path
import numpy as np
from .collect_policy_motion import _initial_source,_cohort_valid,file_sha
from .engine import Game,BEHAVIOR_VERSION
from .motion_hard_collection import initial as wall_initial
from .training_pipeline import atomic_json,now
from .workspace_paths import identity_path

PROTOCOL='Source_targeted_motion180_idle_v2'
SCENARIOS=('forward','left','right','forward_left','forward_right','forward_reverse',
 'left_right','diagonal_switch','forward_turn_left','forward_turn_right',
 'wall_left','wall_right','corner_left','corner_right','orbit_left','orbit_right','idle','idle_then_forward','forward_then_idle','idle_turn')

def control(name,tick):
    second=tick>=180; t=tick%180
    values={'forward':(1,0,0),'left':(0,-1,0),'right':(0,1,0),
      'forward_left':(1,-1,0),'forward_right':(1,1,0),
      'forward_reverse':(1 if not second else -1,0,0),
      'left_right':(0,-1 if not second else 1,0),
      'diagonal_switch':(1,-1 if not second else 1,0),
      'forward_turn_left':(1,0,-.35 if second else 0),
      'forward_turn_right':(1,0,.35 if second else 0),
      'wall_left':(0,-1,0),'wall_right':(0,1,0),
      'corner_left':(1,-1,.25 if second else 0),
      'corner_right':(1,1,-.25 if second else 0),
      'orbit_left':(0,-1,-.35 if second else 0),
      'orbit_right':(0,1,.35 if second else 0),
      'idle':(0,0,0),'idle_then_forward':(1 if second else 0,0,0),
      'forward_then_idle':(0 if second else 1,0,0),'idle_turn':(0,0,.35 if second else 0)}
    a=list(values[name])
    # Hold the requested command for a complete H180, then vary it.
    if second and name in ('forward','left','right','forward_left','forward_right','wall_left','wall_right'):
        a[0]*=-1;a[1]*=-1
    return np.asarray([*a,0],np.float32)

def initialize(seed,stream,name,variant):
    g=_initial_source(seed,stream)
    if name.startswith(('wall','corner')):
        g=wall_initial(g,seed,'corner' if name.startswith('corner') else 'grazing',stream)
        snapshot=g.get_state()
        # Grazing initializer faces along the wall; strafe requires a wall-facing camera.
        if name.startswith('wall'):snapshot['player']['angle']+=math.pi/2
        # Vary gap to the nearest wall while retaining legal cell-center bounds.
        x,y=snapshot['player']['x'],snapshot['player']['y'];gap=(.0,.08,.16,.22)[variant%4]
        snapshot['player']['x']=math.floor(x)+.5+(x%1-.5)*(1-gap/.27)
        snapshot['player']['y']=math.floor(y)+.5+(y%1-.5)*(1-gap/.27)
        g.set_state(snapshot)
    if stream=='enemy_motion':
        snapshot=g.get_state();p=np.array([snapshot['player']['x'],snapshot['player']['y']])
        floor=np.argwhere(g.map==0)[:,::-1]+.5;distance=np.linalg.norm(floor-p,axis=1)
        desired=(10.,12.,14.,16.)[variant%4] if name.startswith('idle') else (4.,6.,8.,10.)[variant%4] if name.startswith('orbit') else (8.,10.,12.,14.)[variant%4]
        pool=np.flatnonzero(np.abs(distance-desired)<1.5)
        if not len(pool):pool=np.argsort(np.abs(distance-desired))[:8]
        rng=np.random.default_rng(seed+991);enemy=floor[int(rng.choice(pool))]
        snapshot['enemies'][0].update(x=float(enemy[0]),y=float(enemy[1]),vx=0.,vy=0.)
        if name.startswith('orbit'):snapshot['player']['angle']=math.atan2(enemy[1]-p[1],enemy[0]-p[0])
        g.set_state(snapshot)
    return g

def episode(seed,stream,name,variant):
    g=initialize(seed,stream,name,variant);initial=g.get_state();states=[g.state_vector().copy()];frames=[g.render(size=32,overlays=False)];actions=[];events=[];hit_ids=[];attack_flags=[]
    for tick in range(360):
        a=control(name,tick);event=g.step(a,dt=1/30);actions.append(a);states.append(g.state_vector().copy());frames.append(g.render(size=32,overlays=False));events.append(Game.event_vector(event));hit_ids.append(event.get('hit_id',-1));attack_flags.append([float(0 in event.get('attacker_ids',[]))])
    return dict(map=g.map.astype(np.int8),states=np.asarray(states,np.float32),frames=np.asarray(frames,np.uint8),actions=np.asarray(actions,np.float32),events=np.asarray(events,np.float32),hit_ids=np.asarray(hit_ids,np.int64),attack_flags=np.asarray(attack_flags,np.float32),behavior_version=np.asarray(BEHAVIOR_VERSION),initial_state=np.asarray(json.dumps(initial,sort_keys=True)),seed=np.asarray(seed),split=np.asarray('train'),motion_stream=np.asarray(stream),case=np.asarray(name),policy=np.asarray('scripted_targeted180'),collection_protocol=np.asarray(PROTOCOL),dt=np.asarray(1/30),hz=np.asarray(30))

def run(root,cycle):
    root=Path(root).absolute();root.mkdir(parents=True,exist_ok=True)
    receipt=root/'collection_receipt.json'
    if receipt.exists():return
    plan=dict(protocol=PROTOCOL,cycle_index=cycle,scenarios=SCENARIOS,variants=4,episodes_per_stream=4*len(SCENARIOS),horizon=360,hold_ticks=180,split='train',Source_state_repairs=0)
    if (root/'collection_plan.json').exists() and json.loads((root/'collection_plan.json').read_text())!=json.loads(json.dumps(plan)):raise ValueError('Targeted collection plan changed')
    atomic_json(root/'collection_plan.json',plan);rows=[];rejected=[];started=time.time()
    for role,stream in enumerate(('navigation','enemy_motion')):
      directory=root/stream/'rgb/train';directory.mkdir(parents=True,exist_ok=True)
      for index in range(4*len(SCENARIOS)):
        name=SCENARIOS[index%len(SCENARIOS)];variant=index//len(SCENARIOS);path=directory/f'episode_{index:04d}.npz'
        if not path.exists():
          for attempt in range(128):
            if (root/'stop.request').exists():raise InterruptedError('Targeted collection stopped')
            seed=1900000000+cycle*1000000+role*100000+index*128+attempt
            payload=episode(seed,stream,name,variant);valid,reason=_cohort_valid(payload['states'],payload['actions'],stream)
            if valid:
              delta=np.linalg.norm(np.diff(payload['states'][:,:2],axis=0)*16,axis=1)
              # Require real movement as well as sustained commands; retain blocked sections.
              valid=(int((delta>.005).sum())>=45) if name not in ('idle','idle_turn') else bool(np.all(delta<1e-5));reason='insufficient_actual_motion' if not valid else None
            if valid:
              temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**payload);temp.replace(path);break
            rejected.append(dict(stream=stream,index=index,attempt=attempt,reason=reason))
          else:raise RuntimeError('Targeted retries exhausted: '+stream+'/'+name)
        with np.load(path) as saved:
          valid,reason=_cohort_valid(saved['states'],saved['actions'],stream)
          if not valid or str(saved['case'])!=name:raise ValueError('Invalid existing targeted episode')
          delta=np.linalg.norm(np.diff(saved['states'][:,:2],axis=0)*16,axis=1)
          rows.append(dict(path=identity_path(path),sha256=file_sha(path),stream=stream,split='train',scenario=name,variant=variant,seed=int(saved['seed']),transitions=360,moving_ticks=int((delta>.005).sum()),initial_enemy_distance_tiles=float(np.linalg.norm((saved['states'][0,:2]-saved['states'][0,8:10])*16)) if role else None))
        atomic_json(root/'collection_progress.json',dict(status='collecting',accepted=len(rows),total_episodes=8*len(SCENARIOS),stream=stream,scenario=name,rejected=len(rejected),updated_at=now()))
        atomic_json(root/'rejected_attempts.json',rejected)
    atomic_json(receipt,dict(plan,episodes=rows,nav_train_directory=identity_path(root/'navigation/rgb/train'),enemy_train_directory=identity_path(root/'enemy_motion/rgb/train'),validation_unchanged=True,total_real_Source_transitions=8*len(SCENARIOS)*360,duration_seconds=time.time()-started,completed_at=now()))
    atomic_json(root/'collection_progress.json',dict(status='complete',accepted=8*len(SCENARIOS),total_episodes=8*len(SCENARIOS),updated_at=now()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--cycle-index',type=int,default=0);a=p.parse_args();run(a.output_dir,a.cycle_index)
if __name__=='__main__':main()
