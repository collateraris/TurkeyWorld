"""Record the released six-start navigation and quiet pursuit demos with V-JEPA."""
import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .engine import EVENT_NAMES, Game
from .record_paired_cuts import scene_factory, scenes
from .record_quiet_chase import candidates, quality_score
from .runtime_checks import circle_is_clear
from .training_pipeline import atomic_json


FRAMES = 180
FPS = 30


def source(scene):
    return scene_factory(scene)(seed=scene['seed'], endless=True, enemies_enabled=scene['chase'])


def backup_scenes(chase, used, count=24):
    """Deterministically search other clear corridors with retreat room on the actual map."""
    g = Game(seed=35000, endless=True, enemies_enabled=chase)
    options = []
    for y in range(1, 15):
        for x in range(1, 15):
            p = np.array([x+.5, y+.5])
            for dx, dy in [(1,0),(0,1),(-1,0),(0,-1)]:
                if tuple(p) in used: continue
                d = np.array([dx,dy])
                e = p + 2.15*d
                if not all(circle_is_clear(g.map,p+t*d,.25) for t in np.linspace(-5.8 if chase else -1.0, 2.2 if chase else 4, 32)): continue
                options.append(dict(player=tuple(p), enemy=tuple(e), angle=math.atan2(dy,dx),
                                    chase=chase,walking_start=chase,curated_candidate=chase))
    rng=np.random.default_rng(36872)
    out=[]
    for i in rng.permutation(len(options)):
        s=options[i]
        if s['player'] in used: continue
        used.add(s['player']); s.update(index=32+len(out), seed=35000+len(out));out.append(s)
        if len(out)==count:break
    return out


def collect(adapter, batch, chase, root, *, action_tapes=None):
    from .video_vjepa_rollout import paired_rollout
    games=[source(s) for s in batch]
    states=np.stack([g.state_vector() for g in games])
    pred_prev=states.copy()
    b=len(games)
    candidates_data=[dict(sprite_frames=0, sprite_switches=0, player_xy_mean=0., enemy_xy_mean=0.,
                          wall_fault_frames=0, source_min_distance=100., source_visible_frames=0,
                          source_player_travel=0., predicted_player_travel=0., source_enemy_travel=0.,
                          predicted_enemy_travel=0., Source_HP_unchanged=True, predicted_HP_unchanged=True,
                          predicted_events=[0]*len(EVENT_NAMES), source_events=[0]*len(EVENT_NAMES)) for _ in batch]
    frames=np.empty((b,FRAMES,128,128,3),np.uint8)
    commands=np.empty((b,FRAMES,4),np.float32); truth=np.empty((b,FRAMES,20),np.float32)
    previous_sprite=np.zeros(b,dtype=bool); n=0
    for t,res in enumerate(paired_rollout(adapter,batch,chase=chase,action_tapes=action_tapes)):
        n=t+1
        action=np.asarray(res['action'],np.float32);pred=np.asarray(res['pred_state'],np.float32)
        sprite=np.asarray(res['sprite_visible'],bool).reshape(b)
        rgb=np.asarray(res['right'])
        if rgb.shape!=(b,128,128,3) or rgb.dtype!=np.uint8:raise ValueError('Neural render must be native SR128 RGB8')
        if not np.isfinite(action).all() or not np.isfinite(pred).all() or np.any(action[:,3]):raise ValueError('Nonfinite result or shooting')
        frames[:,t]=rgb;commands[:,t]=action
        if t%60==0:print(f'{"pursuit" if chase else "nav"} batch {batch[0]["index"]}..{batch[-1]["index"]}: tick {t}/180',flush=True)
        for j,(g,m) in enumerate(zip(games,candidates_data)):
            ev=g.step(action[j].copy(),dt=1/FPS); z=g.state_vector(); truth[j,t]=z
            for k,name in enumerate(EVENT_NAMES):
                m['source_events'][k]+=int(ev[name]);m['predicted_events'][k]+=int(res['events'][j,k])
            for who,after,prev in [('source',z,states[j]),('predicted',pred[j],pred_prev[j])]:
                m[who+'_player_travel']+=float(np.linalg.norm((after[:2]-prev[:2])*16))
                if chase:m[who+'_enemy_travel']+=float(np.linalg.norm((after[8:10]-prev[8:10])*16))
            m['Source_HP_unchanged'] &= bool(np.array_equal(z[[4,12]],states[j,[4,12]]))
            m['predicted_HP_unchanged'] &= bool(np.allclose(pred[j,[4,12]],pred_prev[j,[4,12]],atol=1e-5,rtol=0))
            m['player_xy_mean']+=float(np.linalg.norm((z[:2]-pred[j,:2])*16))/FRAMES
            if n == FRAMES:
                m['player_H180'] = float(np.linalg.norm((z[:2] - pred[j,:2]) * 16))
                if chase:
                    m['enemy_H180'] = float(np.linalg.norm((z[8:10] - pred[j,8:10]) * 16))
            bad_wall=not circle_is_clear(g.map,pred[j,:2]*16,Game.PLAYER_RADIUS)
            if chase:
                m['enemy_xy_mean']+=float(np.linalg.norm((z[8:10]-pred[j,8:10])*16))/FRAMES
                delta=(z[8:10]-z[:2])*16;dist=float(np.linalg.norm(delta))
                m['source_min_distance']=min(m['source_min_distance'],dist)
                forward=delta[0]*z[3]+delta[1]*z[2];side=-delta[0]*z[2]+delta[1]*z[3]
                m['source_visible_frames']+=int(forward>.12 and abs(side)<forward*.6745 and g._line_clear(*(z[:2]*16),*(z[8:10]*16)))
                bad_wall |= not circle_is_clear(g.map,pred[j,8:10]*16,Game.ENEMY_RADIUS)
            elif sprite[j]:raise ValueError('Sprite displayed in no-enemy navigation')
            m['wall_fault_frames']+=int(bad_wall);m['sprite_frames']+=int(sprite[j])
            m['sprite_switches']+=int(t>0 and sprite[j]!=previous_sprite[j]);states[j]=z
        pred_prev=pred.copy();previous_sprite=sprite.copy()
    if n!=FRAMES:raise ValueError('Incomplete neural cut')
    rows=[]
    for j,(s,m) in enumerate(zip(batch,candidates_data)):
        path=root/f'{"pursuit" if chase else "nav"}_{s["index"]:02}.npz'
        np.savez(path,right=frames[j],actions=commands[j],truth=truth[j])
        eligible=not any(m['predicted_events']+m['source_events']) and m['Source_HP_unchanged'] and m['predicted_HP_unchanged']
        if chase:
            eligible &= (m['source_visible_frames']==180 and m['source_min_distance']>1.5 and
                         m['sprite_frames']>=60 and min(m['source_enemy_travel'],m['predicted_enemy_travel'])>.25)
        else: eligible &= min(m['source_player_travel'],m['predicted_player_travel'])>.25
        rows.append(dict(scene=s,metrics=m,eligible=bool(eligible),score=quality_score(m) if chase else m['player_xy_mean'],
                         tape_sha256=hashlib.sha256(commands[j].tobytes()).hexdigest(),cache=str(path)))
    return rows


def header(scene, ix, label, meta):
    canvas=Image.new('RGB',(1080,640),'#111920');d=ImageDraw.Draw(canvas)
    path='/usr/share/fonts/truetype/dejavu/DejaVuSans'
    regular=ImageFont.truetype(path+'.ttf',16); title=ImageFont.truetype(path+'-Bold.ttf',18)
    d.text((20,12),'ОРИГИНАЛ',font=title,fill='#f2f4f6')
    d.text((548,12),'НЕЙРОИГРА',font=title,fill='#c4f893')
    d.text((20,36),'Source · 512 × 512',font=regular,fill='#adc2cf')
    d.text((548,36),f'V-JEPA 2-AC · шаг {meta["step"]} · Base32 → SR128',font=regular,fill='#adc2cf')
    d.text((20,581),f'{ix+1} / 6     {label}',font=title,fill='#f2f4f6')
    d.text((20,612),f'Старт ({scene["player"][0]:g}; {scene["player"][1]:g})   ·   одинаковые команды   ·   без стрельбы',font=regular,fill='#adc2cf')
    return canvas


def publish(rows,kind,root,meta,ffmpeg,ffprobe, *, replay=None):
    chase=kind=='pursuit'
    # Prefer full visibility and clean geometry when enough distinct starts qualify.
    bank=sorted(rows,key=lambda r:(r['metrics']['sprite_frames']!=FRAMES,
                                   r['metrics']['wall_fault_frames']!=0,r['score'])) if chase and replay is None else rows
    chosen=[];used=set()
    for r in bank:
        p=tuple(r['scene']['player'])
        if r['eligible'] and p not in used:used.add(p);chosen.append(r)
        if len(chosen)==6:break
    if len(chosen)!=6:raise RuntimeError(f'{kind}: only {len(chosen)} eligible distinct starts')
    movie=root/(kind+'.mp4')
    argv=[ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1080x640','-r','30','-i','pipe:0',
          '-an','-c:v','libx264','-threads','3','-preset','fast','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(movie)]
    sheet=Image.new('RGB',(1080,426))
    with subprocess.Popen(argv,stdin=subprocess.PIPE) as proc:
        try:
            for i,r in enumerate(chosen):
                s=r['scene'];g=source(s);template=header(s,i,'Преследование без боя' if chase else 'Навигация без противника',meta)
                with np.load(r['cache']) as data:
                    rgb=data['right'];actions=data['actions'];truth=data['truth']
                if hashlib.sha256(actions.tobytes()).hexdigest()!=r['tape_sha256']:raise RuntimeError('Action tape changed')
                for t in range(FRAMES):
                    g.step(actions[t].copy(),dt=1/FPS)
                    if not np.array_equal(g.state_vector(),truth[t]):raise RuntimeError('Source replay mismatch')
                    canvas=template.copy();canvas.paste(Image.fromarray(g.render(size=512,overlays=False)),(20,60))
                    canvas.paste(Image.fromarray(rgb[t]).resize((512,512),Image.Resampling.BILINEAR),(548,60))
                    if t==89:sheet.paste(canvas.resize((360,213),Image.Resampling.LANCZOS),(i%3*360,i//3*213))
                    proc.stdin.write(canvas.tobytes())
                print(f'{kind} encoded cut {i+1}/6',flush=True)
        finally:proc.stdin.close()
        if proc.wait():raise RuntimeError('ffmpeg encoding failed')
    sheet.save(root/(kind+'.jpg'),quality=95)
    subprocess.run([ffmpeg,'-v','error','-xerror','-i',str(movie),'-f','null','-'],check=True)
    probe=json.loads(subprocess.check_output([ffprobe,'-v','error','-select_streams','v:0','-show_entries',
        'stream=codec_name,width,height,nb_frames,avg_frame_rate,duration','-of','json',str(movie)]))['streams'][0]
    if not (int(probe['nb_frames'])==1080 and probe['width']==1080 and probe['height']==640 and probe['avg_frame_rate']=='30/1'
            and float(probe['duration'])==36 and probe['codec_name']=='h264'):raise ValueError(f'Invalid encoding: {probe}')
    compact=lambda row:{k:v for k,v in row.items() if k!='cache'}
    atomic_json(root/(kind+'.json'),dict(passed=True,profile=kind,source_left=True,neural_right=True,clips=6,seconds_per_clip=6,
        duration_seconds=36,total_frames=1080,fps=30,world_step=meta['step'],sr_step=meta['sr_step'],models=meta,autonomous_neural=True,
        Source_future_feedback=False,engine_corrections=0,Source_replay_verified_each_frame=True,ffmpeg_full_decode_passed=True,stream=probe,
        replay=replay,
        selection_priority=(['previously_published_order'] if replay else
                            ['full_sprite_visibility','zero_wall_fault_frames','quality_score','unique_starts'] if chase else ['scene_order','unique_starts']),
        quality_score=('enemy_xy_mean + .5*player_xy_mean + 20*(1-sprite_frames/180) + 2*wall_fault_frames/180 + .01*sprite_switches' if chase and replay is None else None),
        selection_candidates=len(rows),selected=[compact(r) for r in chosen],candidate_metrics=[compact(r) for r in rows],
        video_sha256=hashlib.sha256(movie.read_bytes()).hexdigest(),offline_simulation_time=True))


def replay_capture(adapter, archive, kind, cache, batch_size):
    """Use only archived commands as model input. Reference states verify the left panel."""
    manifest_path = archive / (kind + '.json')
    manifest = json.loads(manifest_path.read_text())
    rows = manifest['selected']
    if (manifest['profile'] != kind or not manifest['passed'] or len(rows) != 6
            or len({tuple(row['scene']['player']) for row in rows}) != 6):
        raise ValueError('Expected an archived six-start recording')
    result = []
    for index in range(0, len(rows), batch_size):
        group = rows[index:index + batch_size]
        scenes = [row['scene'] for row in group]
        tapes = []
        truths = []
        for row in group:
            filename = f"{kind}_{row['scene']['index']:02}.npz"
            with np.load(archive / 'capture_cache' / filename, allow_pickle=False) as data:
                actions = data['actions']
                if hashlib.sha256(actions.tobytes()).hexdigest() != row['tape_sha256']:
                    raise ValueError(f'Archived tape hash mismatch in {filename}')
                tapes.append(actions)
                truths.append(data['truth'])
        generated = collect(adapter, scenes, kind == 'pursuit', cache, action_tapes=np.stack(tapes))
        for previous, current, truth in zip(group, generated, truths):
            if current['tape_sha256'] != previous['tape_sha256']:
                raise ValueError('Replay altered the shared controls')
            with np.load(current['cache'], allow_pickle=False) as data:
                if not np.array_equal(truth, data['truth']):
                    raise ValueError('The Source reference trajectory has changed')
            current['source_states_sha256'] = hashlib.sha256(truth.tobytes()).hexdigest()
        result.extend(generated)
    provenance = dict(world_step=manifest['world_step'], video_sha256=manifest['video_sha256'],
                      manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                      action_tapes_verified=True, original_source_states_verified=True,
                      previous_selection_order_preserved=True)
    return result, provenance


def main():
    from .video_vjepa_rollout import load_video_world
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--world',required=True,type=Path);p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--ffmpeg',default=shutil.which('ffmpeg'));p.add_argument('--ffprobe',default=shutil.which('ffprobe'))
    p.add_argument('--profile',choices=['both','nav','pursuit'],default='both')
    p.add_argument('--replay-capture', type=Path, help='Replay the selected cuts and exact controls of an archived capture')
    a=p.parse_args();root=a.output_dir.resolve();root.mkdir(parents=True,exist_ok=True)
    if a.batch_size<1 or a.batch_size>8:p.error('--batch-size must be in [1,8]')
    if a.replay_capture and root == a.replay_capture.resolve():
        p.error('Write the retake to a new directory')
    cache=root/'capture_cache';cache.mkdir(exist_ok=True)
    import torch
    torch.set_num_threads(2)
    adapter,meta=load_video_world(a.world,a.device)
    for kind in ('nav','pursuit'):
        if a.profile not in ('both',kind):continue
        if a.replay_capture is not None:
            rows, provenance = replay_capture(adapter, a.replay_capture, kind, cache, a.batch_size)
            atomic_json(root/(kind+'_candidates.json'), rows)
            publish(rows, kind, root, meta, a.ffmpeg, a.ffprobe, replay=provenance)
            continue
        chase=kind=='pursuit';pool=[{**s,'walking_start':True} for s in candidates(close=True)] if chase else scenes('navigation-six')
        rows=[]
        for attempt in range(2):
            for k in range(0,len(pool),a.batch_size):
                rows+=collect(adapter,pool[k:k+a.batch_size],chase,cache)
                atomic_json(root/(kind+'_candidates.json'),rows)
            if len({tuple(r['scene']['player']) for r in rows if r['eligible']})>=6:break
            pool=backup_scenes(chase,{tuple(r['scene']['player']) for r in rows})
        publish(rows,kind,root,meta,a.ffmpeg,a.ffprobe)


if __name__=='__main__': main()
