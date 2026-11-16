"""Curate six complete quiet-pursuit clips; keep all candidate metrics disclosed."""
import argparse,gc,json,math,subprocess,time
from pathlib import Path
from copy import deepcopy
import numpy as np
from PIL import Image
import torch
from .record_paired_cuts import scene_factory,compose
from .compare_runtime import PairedRuntime
from .neural_play import NeuralRuntime,_arguments
from .runtime_checks import circle_is_clear
from .training_pipeline import atomic_json

def candidates(close=False):
    rows=[]
    for y in (1.5,2.5,6.5,9.5,12.5,14.5):
        rows.extend([((5.5,y),(10.5,y),0.),((10.5,y),(5.5,y),math.pi)])
    for x in (1.5,14.5):rows.extend([((x,5.5),(x,10.5),math.pi/2),((x,10.5),(x,5.5),-math.pi/2)])
    if close:
        rows=[((9.5,y),(12.,y),0.) if angle==0 else ((6.5,y),(4.,y),math.pi) for (x,y),enemy,angle in rows[:12]]+[((x,9.5),(x,12.),math.pi/2) if angle>0 else ((x,6.5),(x,4.),-math.pi/2) for (x,y),enemy,angle in rows[12:]]
    return [dict(index=i,chase=True,curated_candidate=True,player=p,enemy=e,angle=a,seed=26000+i) for i,(p,e,a) in enumerate(rows)]

def action_at(tick,close=False):return [-.5 if close else -.25,0.,0.,0.]

def tracking_action(predicted,initial_angle=None):
    player=predicted[:2]*16;enemy=predicted[8:10]*16
    delta=enemy-player;angle=math.atan2(float(predicted[2]),float(predicted[3]))
    aim=math.atan2(float(delta[1]),float(delta[0]));error=math.atan2(math.sin(aim-angle),math.cos(aim-angle))
    if initial_angle is not None:
        from_initial=math.atan2(math.sin(aim-initial_angle),math.cos(aim-initial_angle))
        aim=initial_angle+float(np.clip(from_initial,-.2,.2))
        error=math.atan2(math.sin(aim-angle),math.cos(aim-angle))
    forward=float(np.clip((np.linalg.norm(delta)-3.)*.5-.4,-.45,-.3))
    return [forward,0.,float(np.clip(error*2.,-.5,.5)),0.]

def quality_score(metrics):
    return metrics['enemy_xy_mean']+.5*metrics['player_xy_mean']+20*(1-metrics['sprite_frames']/180)+2*metrics['wall_fault_frames']/180+.01*metrics['sprite_switches']

def run(args):
    root=args.output_dir;root.mkdir(parents=True,exist_ok=True)
    if args.finalize_only:return finalize(args,json.loads((root/'candidates.json').read_text()))
    results=[];began=time.perf_counter()
    for scene in candidates(args.near):
        if args.tracking:scene['walking_start']=True
        folder=root/f'candidate_{scene["index"]:02d}';folder.mkdir(exist_ok=True);video=folder/'clip.mp4'
        atomic_json(root/'status.json',dict(status='recording_candidates',candidate=scene['index']+1,total_candidates=16,frames=0))
        a=_arguments(['--world',str(args.world),'--renderer-checkpoint',str(args.renderer),'--sr',str(args.sr),'--allow-unqualified-codec','--rf-euler-steps','3','--device','cuda','--headless','--display-size','512','--smooth','--warmup','0','--threads','2','--seed',str(scene['seed'])]);a.initial_navigation_only=False
        factory=scene_factory(scene);neural=NeuralRuntime(a,initialization_factory=factory)
        pair=PairedRuntime(neural,source_size=512,horizon=180,source_factory=factory,shared_camera_entities='neural')
        raw={};original=neural.sr.forward_with_aux
        def forward(low,hypothesis=None):
            out=original(low,hypothesis);raw['sprite']=bool((out['sprite_alpha']>.01).any());return out
        neural.sr.forward_with_aux=forward
        encoder=subprocess.Popen([str(args.ffmpeg),'-hide_banner','-loglevel','warning','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1080x640','-r','30','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=(folder/'ffmpeg.log').open('wb'))
        source_hp=pair.source.player['hp'];npc_hp=pair.source.enemies[0]['hp'];metrics=dict(player_xy_mean=0.,enemy_xy_mean=0.,sprite_frames=0,sprite_switches=0,wall_fault_frames=0,source_visible_frames=0,source_min_distance=100.,Source_HP_unchanged=True);last_sprite=None
        try:
            for tick in range(180):
                control=tracking_action(neural.state[0].detach().float().cpu().numpy(),scene['angle']) if args.tracking else action_at(tick,args.near)
                left,right=pair.step(control);ref=pair.source.state_vector();pred=neural.state[0].detach().float().cpu().numpy()
                metrics['Source_HP_unchanged'] &= pair.source.player['hp']==source_hp and pair.source.enemies[0]['hp']==npc_hp
                metrics['player_xy_mean']+=float(np.linalg.norm((pred[:2]-ref[:2])*16))/180
                metrics['enemy_xy_mean']+=float(np.linalg.norm((pred[8:10]-ref[8:10])*16))/180
                metrics['sprite_frames']+=int(raw.get('sprite',False));metrics['sprite_switches']+=int(last_sprite is not None and last_sprite!=raw.get('sprite',False));last_sprite=raw.get('sprite',False)
                metrics['wall_fault_frames']+=int(not circle_is_clear(pair.source.map,pred[:2]*16,.2) or not circle_is_clear(pair.source.map,pred[8:10]*16,.25))
                delta=(ref[8:10]-ref[:2])*16;distance=float(np.linalg.norm(delta));metrics['source_min_distance']=min(metrics['source_min_distance'],distance)
                forward_dist=delta[0]*ref[3]+delta[1]*ref[2];side=-delta[0]*ref[2]+delta[1]*ref[3]
                metrics['source_visible_frames']+=int(forward_dist>.12 and abs(side)<forward_dist*.6745 and pair.source._line_clear(*(ref[:2]*16),*(ref[8:10]*16)))
                frame=compose(left,right,scene,tick,neural.world_metadata['training_steps'],neural.sr_metadata['training_steps']);encoder.stdin.write(frame.tobytes())
                if tick in (29,89,149):Image.fromarray(frame).save(folder/f'frame_{tick+1:03d}.jpg',quality=95)
                if (tick+1)%30==0:atomic_json(root/'status.json',dict(status='recording_candidates',candidate=scene['index']+1,total_candidates=16,frames=tick+1))
        finally:encoder.stdin.close()
        if encoder.wait()!=0:raise RuntimeError('Candidate video encoding failed')
        events=deepcopy(neural.event_counts);pair_report=pair.status()
        eligible=metrics['Source_HP_unchanged'] and metrics['source_visible_frames']==180 and metrics['source_min_distance']>1.5 and all(v==0 for v in events.values()) and metrics['sprite_frames']>=120 and pair_report['engine_corrections']==0 and not pair_report['rssm_future_source_feedback']
        result=dict(actions=deepcopy(pair.actions),control_policy="own_neural_enemy_tracking_safe_retreat_v2" if args.tracking else "fixed_retreat",scene=scene,metrics=metrics,score=quality_score(metrics),eligible=eligible,predicted_events=events,pair=pair_report,video=str(video),world_step=neural.world_metadata['training_steps'],sr_step=neural.sr_metadata['training_steps'])
        results.append(result);atomic_json(root/'candidates.json',results);print(f'Candidate {scene["index"]+1}/16 eligible={eligible} sprites={metrics["sprite_frames"]}/180 score={result["score"]:.3f}',flush=True)
        del pair,neural,original;gc.collect();torch.cuda.empty_cache()
    finalize(args,results,time.perf_counter()-began)

def finalize(args,results,elapsed=0.):
    root=args.output_dir
    for row in results:
        m=row['metrics'];row['score']=quality_score(m)
        row['eligible']=bool(m['Source_HP_unchanged'] and m['source_visible_frames']==180 and m['source_min_distance']>1.5 and m['sprite_frames']>=60 and all(v==0 for v in row['predicted_events'].values()) and row['pair']['engine_corrections']==0 and not row['pair']['rssm_future_source_feedback'])
    atomic_json(root/'candidates.json',results)
    eligible=sorted((r for r in results if r['eligible']),key=lambda r:r['score'])
    selected=[];used=set()
    for row in eligible:
        position=tuple(row['scene']['player'])
        if position in used:continue
        selected.append(row);used.add(position)
        if len(selected)==6:break
    if len(selected)!=6:raise RuntimeError('Fewer than six fully quiet candidates; inspect candidates.json')
    # Keep the complete six-second cuts. No frame replacements or Source corrections.
    playlist=root/'selected.txt';playlist.write_text(''.join("file '"+Path(row['video']).relative_to(root).as_posix()+"'\n" for row in selected))
    subprocess.run([str(args.ffmpeg),'-hide_banner','-loglevel','warning','-y','-f','concat','-safe','0','-i',str(playlist),'-c','copy','-movflags','+faststart',str(root/'quiet_enemy_six_starts.mp4')],check=True)
    sheet=Image.new('RGB',(1080,426))
    for i,row in enumerate(selected):
        poster=Path(row['video']).parent/'frame_090.jpg';sheet.paste(Image.open(poster).resize((360,213),Image.Resampling.LANCZOS),((i%3)*360,(i//3)*213))
    sheet.save(root/'contact_sheet.jpg',quality=95)
    atomic_json(root/'report.json',dict(selected=selected,candidates=16,selection='Six lowest declared scores among quiet candidates, unique starts; complete cuts',quality_score='enemy_xy_mean + .5*player_xy_mean + 20*(1-sprite_frames/180) + 2*wall_fault_frames/180 + .01*sprite_switches',total_frames=1080,duration_seconds=36,fps=30,source_left=True,neural_right=True,Source_future_feedback=False,offline_simulation_time=True,quality_qualified=False,wall_seconds=elapsed))
    atomic_json(root/'status.json',dict(status='complete',selected=6,candidates=16,duration_seconds=36,total_frames=1080))

def main():
    p=argparse.ArgumentParser();p.add_argument('--finalize-only',action='store_true');p.add_argument('--near',action='store_true');p.add_argument('--tracking',action='store_true');p.add_argument('--output-dir',type=Path,required=True)
    for key in ('world','sr','renderer','ffmpeg'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(2);torch.cuda.init()
    try:run(args)
    except BaseException as error:atomic_json(args.output_dir/'status.json',dict(status='failed',error=str(error)));raise
if __name__=='__main__':main()
