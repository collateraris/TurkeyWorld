"""Exact visible-small-NPC Source32/128 pairs, with immutable guard replay.

Pixel area comes from the recorded Source-versus-no-NPC render difference.
Snapshots remain snapshots; dynamic pairs retain their genuine previous tick.
No target is resized and no foreground is pasted or synthesized.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .codec_battle_data import render_condition_key
from .engine import BEHAVIOR_VERSION, Game
from .prepare_training_data import _atomic_json

PROTOCOL = 'source_visible_small_enemy_pairs_v1'
WEIGHTS = {'tiny': .75, 'navigation': .0625, 'near': .0625, 'corpse': .0625, 'hp0': .0625}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as source:
        for block in iter(lambda:source.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _category(state,mask,navigation):
    area=int(mask.sum());alive=state[13]>=.5 and state[12]>0
    if state[4]<=0:return 'hp0'
    if alive and 1<=area<=16:return 'tiny'
    if navigation and not alive and not area:return 'navigation'
    if area and not alive:return 'corpse'
    if area and alive and np.linalg.norm((state[8:10]-state[:2])*16)<2:return 'near'
    return None


def inventory(roots,split,*,exclude_rgb=None,exclude_keys=None):
    rows=[];files=[];rgb_keys=set();scene_keys=set();seeds=set();contents=set();excluded=Counter()
    for path in sorted({p.resolve()for root in roots for p in Path(root).rglob('*.npz')if p.parent.name==split}):
        with np.load(path,allow_pickle=False)as z:
            if not {'states','frames','object_masks','background_frames','map','split'}.issubset(z.files):continue
            if str(z['split'])!=split or str(z['behavior_version'])!=BEHAVIOR_VERSION:
                raise ValueError('Small-enemy data requires explicit current Source split/behavior')
            states,frames,masks,bg,grid=(z[k]for k in ('states','frames','object_masks','background_frames','map'))
            dynamic='actions'in z
            if (states.shape!=(len(frames),20)or frames.shape[1:]!=(32,32,3)or frames.dtype!=np.uint8
                    or masks.shape!=(len(frames),32,32)or masks.dtype!=np.bool_
                    or bg.shape!=frames.shape or grid.shape!=(16,16)
                    or not np.array_equal(masks,np.any(frames!=bg,axis=-1))):
                raise ValueError('Small enemy masks/RGB/state must be exact native32 Source')
            if (dynamic and ('initial_state'not in z or len(z['actions'])+1!=len(states))
                    or not dynamic and 'snapshot_states'not in z):
                raise ValueError('Source replay needs genuine initial/actions or exact independent snapshots')
            stage=str(z['stage'])if'stage'in z else path.parents[2].name
            nav=(stage=='navigation'or bool(z['navigation_only'])if'navigation_only'in z else stage=='navigation')
            if 'enemies_enabled'in z:nav|=not bool(z['enemies_enabled'])
            seed=int(z['seed']);sha=digest(path);file_index=len(files)
            files.append({'path':str(path),'seed':seed,'sha256':sha,'dynamic':dynamic,'stage':stage})
            seeds.add(seed);contents.add(sha)
            for frame in range(1 if dynamic else 0,len(frames)):
                state=states[frame];rgb=hashlib.sha256(frames[frame].tobytes()).hexdigest()
                key=render_condition_key(state,grid,enemies_enabled=not nav).hex()
                rgb_keys.add(rgb);scene_keys.add(key)
                category=_category(state,masks[frame],nav)
                if category is None:continue
                if exclude_rgb is not None and rgb in exclude_rgb:excluded['rgb']+=1;continue
                if exclude_keys is not None and key in exclude_keys:excluded['render_condition']+=1;continue
                ys,xs=np.nonzero(masks[frame]);area=int(len(xs))
                bbox=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]if area else None
                rows.append({'source':file_index,'frame':frame,'cohort':category,'rgb_sha256':rgb,'render_key':key,
                    'mask_area':area,'mask_bbox':bbox,'screen_edge_partial':bool(area and(
                        xs.min()==0 or ys.min()==0 or xs.max()==31 or ys.max()==31)),
                    'distance_tiles':float(np.linalg.norm((state[8:10]-state[:2])*16)),
                    'player_hp0':bool(state[4]<=0),'npc_alive':bool(state[13]>=.5 and state[12]>0)})
    return {'files':files,'rows':rows,'rgb_keys':rgb_keys,'scene_keys':scene_keys,
        'seeds':seeds,'contents':contents,'excluded':dict(excluded)}


def select_rows(found,count,seed):
    if type(count)is not int or count<16 or count%4:raise ValueError('Pair count must be a positive multiple of4')
    rng=np.random.default_rng(seed);groups={name:[]for name in WEIGHTS};used=set();selected=[]
    for row in found['rows']:groups[row['cohort']].append(row)
    for rows in groups.values():rng.shuffle(rows)
    def take(name):
        while groups[name]:
            row=groups[name].pop()
            if row['rgb_sha256']in used:continue
            used.add(row['rgb_sha256']);selected.append(row);return True
        return False
    for _ in range(3*count//4):
        if not take('tiny'):raise ValueError('Insufficient distinct tiny1..16px Source frames')
    counts=Counter()
    for _ in range(count//4):
        available=[name for name in WEIGHTS if name!='tiny'and groups[name]]
        while available:
            name=min(available,key=lambda k:(counts[k],k))
            if take(name):counts[name]+=1;break
            available.remove(name)
        else:raise ValueError('Insufficient distinct Source guard replay')
    if any(not counts[k]for k in WEIGHTS if k!='tiny'):
        raise ValueError('Every navigation/near/corpse/HP0 replay guard must be represented')
    rng.shuffle(selected);return selected


def _observe(game):
    low=game.render(size=32,overlays=False);high=game.render(size=128,overlays=False)
    enemies=game.enemies;game.enemies=[]
    try:
        bg=game.render(size=32,overlays=False);hrbg=game.render(size=128,overlays=False)
    finally:game.enemies=enemies
    return low,high,bg,hrbg,np.any(low!=bg,axis=-1),np.any(high!=hrbg,axis=-1)


def replay_selected(job):
    info=job['source'];path=Path(info['path'])
    if digest(path)!=info['sha256']:raise ValueError('Immutable parent Source file changed')
    with np.load(path,allow_pickle=False)as z:
        arrays={name:z[name]for name in ('states','frames','map','object_masks','background_frames')}
        if info['dynamic']:
            arrays.update(actions=z['actions'],events=z['events']);initial=json.loads(str(z['initial_state']))
            dt=float(z['dt'])
            if abs(dt-1/30)>1e-7:raise ValueError('Small enemy Source replay requires30Hz')
        else:snapshots=json.loads(str(z['snapshot_states']))
    game=Game(seed=info['seed'],endless=True)
    game.map=arrays['map'].copy();game.map_height,game.map_width=game.map.shape
    wanted={row['frame']for row in job['rows']}
    if info['dynamic']:wanted|={frame-1 for frame in wanted}
    observed={}
    def observe(frame):
        if not np.array_equal(game.state_vector(),arrays['states'][frame]):raise ValueError('Source state replay mismatch')
        values=_observe(game)
        if (not np.array_equal(values[0],arrays['frames'][frame])
                or not np.array_equal(values[2],arrays['background_frames'][frame])
                or not np.array_equal(values[4],arrays['object_masks'][frame])):
            raise ValueError('Source native32/mask/background replay is not bitexact')
        observed[frame]=values
    if info['dynamic']:
        game.set_state(initial)
        for frame in range(max(wanted)+1):
            if frame in wanted:observe(frame)
            if frame<max(wanted):
                events=game.step(arrays['actions'][frame],dt=1/30)
                if not np.array_equal(Game.event_vector(events),arrays['events'][frame]):raise ValueError('Source event replay mismatch')
    else:
        for frame in sorted(wanted):game.set_state(snapshots[frame]);observe(frame)
    records=[]
    for row in job['rows']:
        frame=row['frame'];indices=[frame-1,frame]if info['dynamic']else[frame]
        relative=f"pairs/{job['split']}/sample_{row['sample_index']:06d}.npz";target=Path(job['output'])/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        fields={'frames':np.stack([observed[i][0]for i in indices]),'hr_frames':np.stack([observed[i][1]for i in indices]),
            'background_frames':np.stack([observed[i][2]for i in indices]),'hr_background_frames':np.stack([observed[i][3]for i in indices]),
            'object_masks':np.stack([observed[i][4]for i in indices]),'hr_object_masks':np.stack([observed[i][5]for i in indices]),
            'states':arrays['states'][indices],'actions':arrays['actions'][frame-1:frame]if info['dynamic']else np.zeros((0,4),np.float32),
            'map':arrays['map'],'split':np.asarray(job['split']),'seed':np.asarray(info['seed']),
            'source_path':np.asarray(info['path']),'source_sha256':np.asarray(info['sha256']),
            'source_frame_indices':np.asarray(indices),'cohort':np.asarray(row['cohort']),
            'render_key':np.asarray(row['render_key']),'frame_source':np.asarray('reference_engine'),
            'hr_frame_source':np.asarray('reference_engine'),'paired_source_rgb':np.asarray(True),
            'source_replay_bitexact':np.asarray(True),'behavior_version':np.asarray(BEHAVIOR_VERSION),
            'world_transition_compatible':np.asarray(info['dynamic']),'collection_protocol':np.asarray(job['fingerprint'])}
        if target.exists():
            with np.load(target,allow_pickle=False)as old:
                if any(not np.array_equal(old[k],v)for k,v in fields.items()):raise ValueError('Existing paired Source sample changed')
        else:
            temporary=target.with_suffix('.tmp.npz');np.savez_compressed(temporary,**fields);temporary.replace(target)
        records.append({**row,'relative_path':relative,'split':job['split'],'source_path':info['path'],
            'source_sha256':info['sha256'],'source_seed':info['seed'],'dynamic':info['dynamic'],'sha256':digest(target)})
    return records


def collect(output,source_roots,*,train_count=4096,val_count=512,workers=4,seed=173):
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True)
    config={'protocol':PROTOCOL,'source_roots':[str(Path(p).resolve())for p in source_roots],
        'train_count':train_count,'val_count':val_count,'seed':seed,'tiny_visible_pixels':[1,16],
        'tiny_fraction':.75,'image_size':32,'hr_image_size':128,'behavior_version':BEHAVIOR_VERSION}
    manifest=root/'manifest.json'
    if manifest.exists():
        old=json.loads(manifest.read_text(encoding='utf-8'))
        if old['config']!=config:raise ValueError('Immutable small-enemy collection configuration changed')
        if old['status']=='complete':
            if any(digest(root/r['relative_path'])!=r['sha256']for r in old['records']):raise ValueError('Small-enemy Source pair changed')
            return old
    train=inventory(source_roots,'train')
    val=inventory(source_roots,'val',exclude_rgb=train['rgb_keys'],exclude_keys=train['scene_keys'])
    if train['seeds']&val['seeds']or train['contents']&val['contents']:raise ValueError('Parent TRAIN/VAL seeds or source file content overlap')
    selected={'train':select_rows(train,train_count,seed),'val':select_rows(val,val_count,seed+1)}
    for rows in selected.values():
        for index,row in enumerate(rows):row['sample_index']=index
    identity={'config':config,'parents':{split:found['files']for split,found in [('train',train),('val',val)]},
        'selected':selected,'validation_exclusions':val['excluded']}
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    jobs=[]
    for split,found in [('train',train),('val',val)]:
        grouped={}
        for row in selected[split]:
            grouped.setdefault(row['source'],[]).append(row)
        jobs.extend({'source':found['files'][index],'rows':rows,'split':split,'output':str(root),
                     'fingerprint':fingerprint}for index,rows in grouped.items())
    records=[];started=time.perf_counter()
    def publish(status):
        summary={split:{'examples':sum(r['split']==split for r in records),
            'cohorts':dict(Counter(r['cohort']for r in records if r['split']==split)),
            'unique_source_episodes':len({r['source_sha256']for r in records if r['split']==split}),
            'visible_area_bands':dict(Counter('1'if r['mask_area']==1 else'2_4'if r['mask_area']<=4 else'5_16'
                for r in records if r['split']==split and r['cohort']=='tiny'))}for split in ('train','val')}
        report={'status':status,'config':config,'fingerprint':fingerprint,'identity':identity,'records':records,
            'completed':len(records),'total':train_count+val_count,'groups':summary,
            'source32_mask_background_replay_bitexact':True,'true_source128':True,'resized_hr':False,
            'train_val_seed_overlap':False,'train_val_source_content_overlap':False,'train_val_rgb32_content_overlap':False,
            'train_val_exact_render_condition_overlap':False,'elapsed_seconds':time.perf_counter()-started,
            'occlusion_scope':'Exact visible Source mask/bbox and screen-edge partial flag; no inferred hidden-wall visibility label'}
        _atomic_json(manifest,report);_atomic_json(root/'status.json',{k:v for k,v in report.items()if k not in ('identity','records')});return report
    publish('collecting')
    try:
        if workers==1:
            for job in jobs:records.extend(replay_selected(job));publish('collecting')
        else:
            with ProcessPoolExecutor(max_workers=workers)as pool:
                for future in as_completed([pool.submit(replay_selected,job)for job in jobs]):
                    records.extend(future.result())
                    if len(records)%64<16:publish('collecting')
    except BaseException:publish('failed');raise
    records.sort(key=lambda r:r['relative_path']);return publish('complete')


class SmallEnemyData:
    def __init__(self,root,split,*,seed=173,exclude_scene_keys=None,device='cpu'):
        if split not in ('train','val'):raise ValueError('Small-enemy data requires explicit split')
        self.root=Path(root).resolve();self.split=split;self.device=torch.device(device)
        manifest=json.loads((self.root/'manifest.json').read_text(encoding='utf-8'))
        if manifest['status']!='complete':raise ValueError('Small-enemy pairs are not complete')
        self.files=[];self.rows=[];self.scene_keys=set();self.rng=np.random.default_rng(seed)
        self.draws=0;self.case_counts=dict.fromkeys(WEIGHTS,0);self.weights=dict(WEIGHTS);self._cache=OrderedDict()
        excluded=set(exclude_scene_keys or ())
        for row in manifest['records']:
            if row['split']!=split:continue
            key=bytes.fromhex(row['render_key'])
            if key in excluded:continue
            path=self.root/row['relative_path'];info={'path':str(path),'seed':row['source_seed'],'sha256':row['sha256'],
                'dynamic':row['dynamic'],'kind':'small_enemy','source_path':row['source_path'],
                'stat':(path.stat().st_size,path.stat().st_mtime_ns)}
            self.rows.append({**row,'file':len(self.files),'frame':1 if row['dynamic']else 0,
                'category':row['cohort'],'mask_fraction':row['mask_area']/1024})
            self.files.append(info);self.scene_keys.add(key)
        self.buckets={name:[i for i,r in enumerate(self.rows)if r['category']==name]for name in WEIGHTS}
        if any(not v for v in self.buckets.values()):raise ValueError('Every tiny/replay guard must remain available')
        self.identity={'protocol':PROTOCOL,'fingerprint':manifest['fingerprint'],'split':split,
            'frames':len(self.rows),'category_counts':{k:len(v)for k,v in self.buckets.items()},
            'weights':self.weights,'source32_replay_bitexact':True,'true_source128':True}

    def fixed_panel(self,count,*,seed=173):
        if not 0<count<=len(self.rows):raise ValueError('Need enough unique held-out Source examples')
        return np.random.default_rng(seed).permutation(len(self.rows))[:count].tolist()

    def panel_fingerprint(self,panel):
        return hashlib.sha256(json.dumps({'identity':self.identity,'indices':list(map(int,panel)),
            'files':[self.files[int(i)]['sha256']for i in panel]},sort_keys=True).encode()).hexdigest()

    def batch(self,size,*,indices=None,rng=None):
        if indices is None:
            rng=self.rng if rng is None else rng;indices=[]
            for _ in range(size):
                category=max(self.weights,key=lambda k:(self.draws+1)*self.weights[k]-self.case_counts[k])
                indices.append(int(rng.choice(self.buckets[category])));self.case_counts[category]+=1;self.draws+=1
            rng.shuffle(indices)
        if len(indices)!=size:raise ValueError('Small-enemy batch indices differ from size')
        rows=[]
        for i in indices:
            i=int(i);info=self.files[i];path=Path(info['path'])
            if (path.stat().st_size,path.stat().st_mtime_ns)!=info['stat']:
                raise ValueError('Immutable paired Source example changed')
            if i not in self._cache:
                if digest(path)!=info['sha256']:raise ValueError('Immutable paired Source example changed')
                with np.load(path,allow_pickle=False)as z:self._cache[i]={k:z[k]for k in ('frames','hr_frames','background_frames',
                    'hr_background_frames','object_masks','hr_object_masks','states','actions','map')}
                while len(self._cache)>16:self._cache.popitem(last=False)
            z=self._cache[i];dynamic=info['dynamic']
            rows.append({'target':z['frames'][-1],'background':z['background_frames'][-1],
                'object_mask':z['object_masks'][-1],'hr':z['hr_frames'][-1],
                'hr_background':z['hr_background_frames'][-1],'hr_object_mask':z['hr_object_masks'][-1],
                'state':z['states'][-1],'map':z['map'],'previous_state':z['states'][0]if dynamic else np.zeros(20,np.float32),
                'previous_rgb':z['frames'][0]if dynamic else np.zeros((32,32,3),np.uint8),
                'action':z['actions'][0]if dynamic else np.zeros(4,np.float32),'dynamic':dynamic,'row_index':i})
        result={}
        for key in rows[0]:
            value=torch.from_numpy(np.stack([r[key]for r in rows]))
            if key in ('target','background','previous_rgb','hr','hr_background'):value=value.permute(0,3,1,2).float()/255
            elif key in ('object_mask','hr_object_mask'):value=value[:,None].float()
            elif key in ('map','row_index'):value=value.long()
            elif key=='dynamic':value=value.bool()
            else:value=value.float()
            result[key]=value.to(self.device)
        result['source']=result['lr']=result['target'];return result

    def sampler_state(self):return {'rng':self.rng.bit_generator.state,'draws':self.draws,'case_counts':dict(self.case_counts)}
    def restore_sampler(self,state):
        if state['draws']!=sum(state['case_counts'].values())or set(state['case_counts'])!=set(self.weights):
            raise ValueError('Small-enemy sampler ledger changed')
        self.rng.bit_generator.state=state['rng'];self.draws=state['draws'];self.case_counts=dict(state['case_counts'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,action='append',required=True)
    p.add_argument('--train-count',type=int,default=4096);p.add_argument('--val-count',type=int,default=512)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=173)
    args=p.parse_args();print(json.dumps(collect(args.output,args.source,train_count=args.train_count,
        val_count=args.val_count,workers=args.workers,seed=args.seed)['groups']),flush=True)


if __name__=='__main__':main()
