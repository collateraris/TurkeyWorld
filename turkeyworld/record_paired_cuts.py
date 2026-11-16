"""Fixed 6-second Source/neural comparisons: eight mixed or six navigation cuts."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch

from .compare_runtime import PairedRuntime
from .engine import Game
from .neural_play import NeuralRuntime,_arguments
from .training_pipeline import atomic_json


def scenes(profile="mixed"):
    if profile=="navigation-six":
        positions=[((2.5,2.5),0.),((13.5,2.5),math.pi),((14.5,13.5),-math.pi/2),((1.5,2.5),math.pi/2),((8.5,1.5),math.pi/2),((8.5,14.5),-math.pi/2)]
        return [dict(index=i,total=6,chase=False,player=p,enemy=(5.5,2.5),angle=angle,seed=19000+i) for i,(p,angle) in enumerate(positions)]
    if profile!="mixed":raise ValueError("Unknown video profile")
    positions=[((2.5,2.5),(10.5,2.5),0.),((13.5,2.5),(5.5,2.5),math.pi),
        ((14.5,13.5),(14.5,5.5),-math.pi/2),((1.5,2.5),(1.5,10.5),math.pi/2)]
    return [dict(index=i+4*int(chase),chase=chase,player=p,enemy=e,angle=angle,seed=19000+i)
        for chase in (False,True) for i,(p,e,angle) in enumerate(positions)]


def scene_factory(scene):
    def create(*,seed,endless,enemies_enabled):
        if enemies_enabled!=scene['chase']:raise ValueError('Scenario profile changed')
        game=Game(seed=seed,endless=endless,enemies_enabled=enemies_enabled)
        snapshot=game.get_state();snapshot['player'].update(x=scene['player'][0],y=scene['player'][1],angle=scene['angle'])
        if enemies_enabled:snapshot['enemies'][0].update(x=scene['enemy'][0],y=scene['enemy'][1],vx=0.,vy=0.)
        if enemies_enabled and scene.get('walking_start'):
            delta=np.asarray(scene['player'])-np.asarray(scene['enemy']);velocity=.7*delta/np.linalg.norm(delta)
            snapshot['enemies'][0].update(vx=float(velocity[0]),vy=float(velocity[1]),animation_phase=.25)
        game.set_state(snapshot)
        return game
    return create


def action_at(scene,tick):
    if scene['chase']:
        # Back away gently, facing the pursuer; its initial eight-cell gap
        # prevents contact during the six-second quiet scene.
        return [-.15,0.,0.,0.]
    tapes=[[(.7,0,0,0),(0,0,.4,0),(.35,0,0,0),(0,.4,0,0),(0,0,-.4,0),(.5,0,0,0)],
           [(.6,0,0,0),(0,.3,0,0),(0,0,-.4,0),(.4,0,0,0),(0,0,.4,0),(-.3,0,0,0)],
           [(.65,0,0,0),(0,0,-.3,0),(.3,0,0,0),(0,0,.3,0),(.6,0,0,0),(0,-.2,0,0)],
           [(.7,0,0,0),(0,0,.3,0),(.4,0,0,0),(0,0,-.3,0),(.6,0,0,0),(-.25,0,0,0)]]
    return list(tapes[scene['index']%len(tapes)][min(5,tick//30)])


def compose(left,right,scene,tick,world_step,sr_step):
    image=Image.new('RGB',(1080,640),(13,19,24));draw=ImageDraw.Draw(image)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    label=('Преследование без боя' if scene['chase'] else 'Навигация без противника')
    draw.text((20,9),'Оригинал · Source512',font=font,fill=(210,240,230))
    draw.text((548,9),f'Нейроигра · новый SR{sr_step}',font=font,fill=(183,247,115))
    subtitle=(f'Старт {scene["player"][0]:g}, {scene["player"][1]:g} · {label} · одинаковые команды' if scene.get("curated_candidate") else f'{scene["index"]+1}/{scene.get("total",8)} · {label} · одинаковые команды в обоих окнах')
    draw.text((20,35),subtitle,font=small,fill=(190,203,210))
    image.paste(Image.fromarray(left),(20,60));image.paste(Image.fromarray(right),(548,60))
    draw.text((20,584),f'{(tick+1)/30:.2f} / 6.00 с симуляции · стрельба выключена · справа автономный rollout',font=small,fill=(240,210,160))
    draw.text((20,610),f'Офлайн-монтаж 30 кадров/с · не замер FPS · RSSM g{world_step} · RGB32 → SR128 → 512',font=small,fill=(165,183,192))
    return np.asarray(image)


def run(args):
    output=args.output_dir;output.mkdir(parents=True,exist_ok=True)
    cuts=scenes(getattr(args,'profile','mixed'));count=len(cuts)
    filename='navigation_six_starts.mp4' if getattr(args,'profile','mixed')=='navigation-six' else 'navigation_and_pursuit_8_cuts.mp4'
    argv=['-hide_banner','-loglevel','warning','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1080x640',
        '-r','30','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(output/filename)]
    rows=[];started=time.perf_counter();posters=[]
    with (output/'ffmpeg.stderr.log').open('wb') as log:
        process=subprocess.Popen([args.ffmpeg,*argv],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
        try:
            for scene in cuts:
                atomic_json(output/'status.json',dict(status='recording',clip=scene['index']+1,clips=count,frames=0))
                a=_arguments(['--world',str(args.world),'--renderer-checkpoint',str(args.renderer),
                    '--sr',str(args.sr),'--allow-unqualified-codec','--rf-euler-steps','3','--device','cuda',
                    '--headless','--display-size','512','--smooth','--warmup','0','--threads','2','--seed',str(scene['seed'])])
                a.initial_navigation_only=not scene['chase']
                factory=scene_factory(scene);neural=NeuralRuntime(a,initialization_factory=factory)
                pair=PairedRuntime(neural,source_size=512,horizon=180,source_factory=factory,shared_camera_entities='neural')
                initial=deepcopy(pair.source.get_state());hp=pair.source.player['hp'];npc_hp=pair.source.enemies[0]['hp']
                clip_began=time.perf_counter();visible=0;minimum_distance=float('inf')
                for tick in range(180):
                    action=action_at(scene,tick);assert action[3]==0
                    left,right=pair.step(action)
                    if pair.source.player['hp']!=hp or pair.source.enemies[0]['hp']!=npc_hp:
                        raise AssertionError('Quiet Source scene unexpectedly entered combat')
                    state=pair.source.state_vector();distance=float(np.linalg.norm((state[8:10]-state[:2])*16))
                    minimum_distance=min(minimum_distance,distance)
                    if scene['chase']:
                        delta=(state[8:10]-state[:2])*16;forward=delta[0]*state[3]+delta[1]*state[2];side=-delta[0]*state[2]+delta[1]*state[3]
                        visible+=int(forward>.12 and abs(side)<forward*.6745 and pair.source._line_clear(*(state[:2]*16),*(state[8:10]*16)))
                    frame=compose(left,right,scene,tick,neural.world_metadata['training_steps'],neural.sr_metadata['training_steps'])
                    process.stdin.write(frame.tobytes())
                    if tick==90:
                        Image.fromarray(frame).save(output/f'clip_{scene["index"]+1:02d}_poster.jpg',quality=95);posters.append(frame)
                    if (tick+1)%30==0:atomic_json(output/'status.json',dict(status='recording',clip=scene['index']+1,clips=count,frames=tick+1))
                stats=pair.status();assert stats['engine_corrections']==0 and stats['rssm_future_source_feedback'] is False
                rows.append(dict(scene=scene,initial_snapshot=initial,frames=180,duration_seconds=6,
                    start_seconds=scene['index']*6,Source_HP_unchanged=True,Source_enemy_HP_unchanged=True,
                    Source_visible_enemy_frames=visible,Source_min_enemy_distance=minimum_distance,
                    actual_recording_wall_seconds=time.perf_counter()-clip_began,pair=stats,
                    predicted_events=deepcopy(neural.event_counts)))
                atomic_json(output/'clips.json',rows)
                print(f'Clip {scene["index"]+1}/{scene.get("total",8)} complete; Source enemy visible {visible}/180',flush=True)
                del pair,neural;gc.collect();torch.cuda.empty_cache()
        finally:
            process.stdin.close()
        if process.wait()!=0:raise RuntimeError('Video encoding failed; see ffmpeg.stderr.log')
    columns=3 if count==6 else 4;tile_width=1080//columns;tile_height=640*tile_width//1080
    poster=Image.new('RGB',(1080,tile_height*((count+columns-1)//columns)))
    for i,frame in enumerate(posters):
        tile=Image.fromarray(frame).resize((tile_width,tile_height),Image.Resampling.LANCZOS)
        poster.paste(tile,((i%columns)*tile_width,(i//columns)*tile_height))
    poster.save(output/'contact_sheet.jpg',quality=95)
    atomic_json(output/'report.json',dict(clips=rows,total_frames=count*180,fps=30,duration_seconds=count*6,
        encoding='offline_simulation_time_not_real_time_performance',source_left=True,neural_right=True,
        world_checkpoint=str(args.world),renderer=str(args.renderer),sr=str(args.sr),
        world_step=rows[0]['pair']['neural_training_step'],Source_future_feedback=False,
        wall_seconds=time.perf_counter()-started))
    atomic_json(output/'status.json',dict(status='complete',clips=count,total_frames=count*180,duration_seconds=count*6))


def main():
    p=argparse.ArgumentParser();p.add_argument('--profile',choices=('mixed','navigation-six'),default='mixed');p.add_argument('--output-dir',type=Path,required=True)
    for name in ('world','renderer','sr','ffmpeg'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(2);torch.cuda.init()
    try:run(args)
    except BaseException as error:
        atomic_json(args.output_dir/'status.json',dict(status='failed',error=f'{type(error).__name__}: {error}'));raise


if __name__=='__main__':main()
