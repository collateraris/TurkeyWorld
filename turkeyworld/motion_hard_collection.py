"""Legal Source wall starts and measured hard-case collection diagnostics."""
import math
import numpy as np
CATEGORIES=('uniform','corner','grazing','axis_stop','enemy_turn')

def initial(game,seed,category,stream):
    if category=='uniform':return game
    rng=np.random.default_rng(seed+97);choices=[]
    for y,x in np.argwhere(game.map==0):
        if not (0<x<15 and 0<y<15):continue
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            if game.map[y+dy,x+dx]==0:continue
            for side in (-1,1):
                if category=='corner' and game.map[y+side*dx,x+side*dy]==0:continue
                choices.append((x,y,dx,dy,side))
    x,y,dx,dy,side=choices[int(rng.integers(len(choices)))];px=x+.5+dx*.27;py=y+.5+dy*.27
    angle=math.atan2(dy,dx)+side*(math.pi/2-.07 if category=='grazing' else math.pi/4 if category in ('axis_stop','enemy_turn') else 0)
    state=game.get_state();state['player'].update(x=px,y=py,angle=angle)
    if stream=='enemy_motion':
        floor=np.argwhere(game.map==0)[:,::-1].astype(float)+.5
        far=floor[np.linalg.norm(floor-[px,py],axis=1)>=8.]
        blocked=[]
        for cell in far:
            line=np.linspace([px,py],cell,64).astype(int);blocked.append(bool(game.map[line[:,1],line[:,0]].any()))
        pool=far[np.asarray(blocked)] if any(blocked) else far
        ex,ey=pool[int(rng.integers(len(pool)))];state['enemies'][0].update(x=float(ex),y=float(ey),vx=0.,vy=0.)
    game.set_state(state);return game

def metrics(states,actions,grid):
    states=np.asarray(states);actions=np.asarray(actions);d=np.diff(states[:,:2],axis=0)*16
    stopped=np.abs(d)<1e-5;moving=np.linalg.norm(actions[:,:2],axis=1)>.1
    walls=np.argwhere(np.asarray(grid)!=0)[:,::-1]+.5
    def near(p):return np.abs(p[:,None]-walls).max(-1).min(-1)<.85
    close=near(states[1:,:2]*16);enemy=np.diff(states[:,8:10],axis=0)*16;speed=np.linalg.norm(enemy,axis=1)
    cos=(enemy[:-1]*enemy[1:]).sum(-1)/np.maximum(speed[:-1]*speed[1:],1e-12)
    return dict(corner=int((moving&stopped.all(-1)&close).sum()),
        grazing=int((moving&(np.linalg.norm(d,axis=1)>.001)&close).sum()),
        axis_stop=int((moving&(stopped.sum(-1)==1)&close).sum()),
        enemy_turn=int(((speed[:-1]>.001)&(speed[1:]>.001)&(cos<.98)&near(states[1:-1,8:10]*16)).sum()))
