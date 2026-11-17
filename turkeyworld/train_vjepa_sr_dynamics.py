"""H180 motion continuation with pooled V-JEPA memory and frozen SR feedback."""
import argparse,json,time,copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .models import load_world_checkpoint
from .rssm_motion_only_data import MotionReplay
from .workspace_paths import ROOT,resolve_path
from .rssm_combat_protocol import environment_profile
from .rssm_coordinate_gradients import integrate_coordinate_gradients,COORDINATE_GRADIENT_PROTOCOL
from .trajectory_objectives import circle_wall_penalty
from .vjepa_sr_feedback import FrozenBaseSR,feedback_active,blend_memory
from .renderer_student import _hash_tensors
from .distill_renderer import _save
from .training_pipeline import atomic_json,now
from .power_guard import KeepAwake
from .vjepa_motion_aux import REVISION,window_kind,player_auxiliary

PROTOCOL='vjepa_pooled_SR_motion_H180_v1'
NAV=ROOT/'training_runs/staged_single_20261110/data_navigation_recovery_v1/long/rgb/train'
ENEMY=ROOT/'training_runs/rssm_enemy_motion_20261113/data/full_game/rgb/train'

class Replay:
    def __init__(self,stream,saved=None):
        self.rng=np.random.default_rng(173 if stream=='navigation' else 174)
        if saved:self.rng.bit_generator.state=saved
        paths=sorted((ROOT/'training_runs/dreamer_motion_cycle_20261114/cycles').glob('*/source/collection_receipt.json'))
        paths+=sorted((ROOT/'training_runs/vjepa_sr_cycle_20261117/cycles').glob('*/source/collection_receipt.json'))
        paths+=sorted((ROOT/'training_runs/vjepa_sr_cycle_20261117/cycles').glob('*/targeted/collection_receipt.json'))
        self.corpora=[json.loads(p.read_text()) for p in paths]
        state=dict(protocol=PROTOCOL,draws=0,rng=self.rng.bit_generator.state)
        self.data=MotionReplay(NAV if stream=='navigation' else ENEMY,state,stream=stream,corpora=self.corpora).data
    def draw(self):
        rows=[]
        self.balance_rotation=int(self.rng.integers(3))
        for actor in range(8):
            if actor<2 or len(self.data)==1:index=0
            elif actor<4:index=int(self.rng.integers(1,len(self.data)-1)) if len(self.data)>2 else 0
            else:index=len(self.data)-1
            data=self.data[index];pool=[i for i,e in enumerate(data.episodes) if len(e['actions'])>=180]
            desired=('moving','blocked','idle')[(actor+self.balance_rotation)%3]
            for attempt in range(24):
                episode=int(self.rng.choice(pool));e=data.episodes[episode];start=int(self.rng.integers(len(e['actions'])-180+1))
                end=start+32
                kind=window_kind(torch.as_tensor(e['states'][start:end]).float(),torch.as_tensor(e['actions'][start:end]).float(),torch.as_tensor(e['states'][start+1:end+1]).float())
                if kind==desired:break
            batch=data.batch(1,context=1,horizon=180,indices=np.asarray([[episode,start]]))
            # Source past only; the four-frame context is rendered through SR.
            initial=[max(0,start-3+i) for i in range(4)]
            batch['initial_states']=torch.as_tensor(np.asarray(e['states'][initial]))[None].float()
            batch['initial_frames']=torch.as_tensor(np.asarray(e['frames'][initial])).permute(0,3,1,2)[None].float()/255
            batch['offset']=start;batch['file']=e['path'];batch['corpus']=index
            batch['motion_kind']=kind;batch['requested_motion_kind']=desired;batch['sampling_attempts']=attempt+1
            rows.append(batch)
        return rows


def run(a):
    root=a.run_dir.resolve();path=root/'checkpoints/vjepa_world.pt';root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    model,saved=load_world_checkpoint(path,'cpu');model.to('cuda').eval()
    if not model.config.vjepa_pooled_memory:raise ValueError('SR-feedback trainer requires pooled V-JEPA memory')
    settings=saved['vjepa_sr_feedback'];renderer=FrozenBaseSR(settings,'cuda')
    old=saved.get('vjepa_sr_training',{});step=saved['training_steps'];updates=old.get('updates',0)
    names=saved['optimizer_parameter_names'];names=names if isinstance(names[0],list) else [names]
    parameters=dict(model.named_parameters());groups=[dict(params=[parameters[n] for n in group]) for group in names]
    optimizer=torch.optim.AdamW(groups,lr=1e-5);optimizer.load_state_dict(saved['optimizer'])
    for group in optimizer.param_groups:group['lr']=1e-5
    for n,p in parameters.items():
        train=n.startswith(('vjepa_backend.core.predictor.','vjepa_backend.readout_neck.','vjepa_backend.delta_head.',
            'vjepa_backend.map_projection.','vjepa_backend.spatial_visual_attention.','vjepa_backend.spatial_motion_'))
        if n.startswith('vjepa_backend.core.encoder.'):
            train=n.startswith('vjepa_backend.core.encoder.norm.') or (n.startswith('vjepa_backend.core.encoder.blocks.') and int(n.split('.')[4])>=16)
        p.requires_grad_(train)
    ema=copy.deepcopy(model.core.encoder).eval().requires_grad_(False)
    if saved.get('ema_target_encoder'):
        state={k[len('encoder.'):]:v for k,v in saved['ema_target_encoder'].items() if k.startswith('encoder.')}
        ema.load_state_dict(state,strict=True)
    replay={s:Replay(s,old.get('samplers',{}).get(s)) for s in ('navigation','enemy_motion')}
    frozen=lambda:{n:p for n,p in model.named_parameters() if not p.requires_grad}
    frozen_hash=_hash_tensors(frozen());render_hash=_hash_tensors(renderer.sr.state_dict());source_step=old.get('source_step',step)
    plan=dict(protocol=PROTOCOL,horizon=180,updates_per_round=100,MAIN=16,microbatch=a.microbatch,
        context='four_observed_past_states_rendered_Base_SR',full_memory_BPTT=True,GT_future_inputs=False,
        encoder_last_blocks_trainable=len(model.core.encoder.blocks)-16,capacity_growth=saved.get('vjepa_capacity_growth'),latent_label_stride=16,latent_weight=.1,lr=1e-5,
        parent_Adam_moments_preserved=True,renderer_frozen=True,feedback=settings,
        motion_revision=REVISION,player_velocity_weight=.25,player_contact_weight=.1,auxiliary_prefix=16,sampling_attempts=24)
    plan_path=root/'training_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:
        previous=json.loads(plan_path.read_text())
        if previous.get('motion_revision') not in (None,REVISION):raise ValueError('Unknown dynamics plan revision')
        atomic_json(root/('training_plan_before_'+REVISION+'_'+str(step)+'.json'),previous)
    atomic_json(plan_path,plan)
    status=dict(stage='world',model='V-JEPA pooled state dynamics + frozen SR feedback',run_id=root.name,
        status='running',current=step,total=None,loss_protocol=PROTOCOL,training_plan=plan,started_at=now())
    def publish(**kw):status.update(current=step,updated_at=now(),**kw);atomic_json(root/'status.json',status)
    def save():
        ledger=dict(protocol=PROTOCOL,source_step=source_step,updates=updates,
            samplers={k:copy.deepcopy(v.rng.bit_generator.state) for k,v in replay.items()},frozen_parameters_sha256=frozen_hash,
            replay_sampling_revision=REVISION)
        payload={**saved,'model':model.state_dict(),'config':vars(model.config),'optimizer':optimizer.state_dict(),
            'training_steps':step,'global_training_step':step,'step':step,'run_state':dict(global_training_step=step),
            'vjepa_sr_training':ledger,'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'rng_numpy_global':np.random.get_state()}
        _save(path,payload)
    if 'rng_torch' in saved:torch.set_rng_state(saved['rng_torch'].cpu())
    if saved.get('rng_cuda'):torch.cuda.set_rng_state_all([v.cpu() for v in saved['rng_cuda']])
    if saved.get('rng_numpy_global') is not None:np.random.set_state(saved['rng_numpy_global'])
    atomic_json(root/'resume_verification.json',dict(step=step,world_updates=updates,
        optimizer_states=len(optimizer.state),parent_Adam_moments_preserved=True,
        frozen_parameters_unchanged=not old.get('frozen_parameters_sha256') or old['frozen_parameters_sha256']==frozen_hash,
        sampler_states_restored=all(not old.get('samplers',{}).get(k) or v.rng.bit_generator.state==old['samplers'][k] for k,v in replay.items()),
        torch_rng_restored='rng_torch' in saved,replay_sampling_revision=REVISION))
    publish(world_updates=updates,round_target=a.until_updates,parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    # Sparse Source RGB labels are EMA-only; never supplied to the live prior.
    while updates<a.until_updates and not (root/'stop_training').exists():
        began=time.perf_counter();optimizer.zero_grad(set_to_none=True);totals=dict(player_loss=0.,enemy_loss=0.,latent_loss=0.);aux_totals=dict(player_velocity_loss=0.,player_contact_loss=0.);receipts=[]
        for stream,sampler in replay.items():
            rows=sampler.draw()
            for first in range(0,8,a.microbatch):
                selected=rows[first:first+a.microbatch];count=len(selected)
                batch={k:torch.cat([r[k] for r in selected],0).to('cuda') for k in ('states','frames','actions','target_states','map','initial_states')}
                state,grid=batch['states'][:,0],batch['map'];actions=batch['actions'];target=batch['target_states'];nav=stream=='navigation'
                with environment_profile(model,navigation_only=nav):
                    with torch.no_grad():
                        fmap=model.encode_map(grid).detach();past=batch['initial_states'].flatten(0,1)
                        rgb=renderer.observation(past,grid.repeat_interleave(4,0),navigation_only=nav).reshape(count,4,3,32,32)
                        labels={}
                        for tick in range(15,180,16):
                            frames=batch['frames'][:,max(0,tick-2):tick+2]
                            labels[tick]=ema(frames)[:,-1].mean(1,keepdim=True).detach()
                    # Train the online encoder on SR-domain context.
                    memory=checkpoint(lambda x:model.initialize_from_image(x,batch['initial_states'],fmap),rgb,use_reentrant=False)
                    points=[];latent=state.sum()*0
                    def transition(s,m,control):
                        value=model.transition(s,control,fmap,m)
                        return value['state'],value['memory'],value['latent_prediction'],value['respawn_flags']
                    for tick in range(180):
                        raw,memory,pred,respawn=checkpoint(transition,state.detach(),memory,actions[:,tick],use_reentrant=False)
                        state=integrate_coordinate_gradients(state,dict(state=raw,respawn_flags=respawn),COORDINATE_GRADIENT_PROTOCOL)['state']
                        active=feedback_active(state,settings)
                        if bool(active.any()):
                            observation=renderer.observation(state.detach(),grid,navigation_only=nav)
                            memory=checkpoint(lambda mem,img,mask:blend_memory(model,mem,img,settings,mask),memory,observation,active,use_reentrant=False)
                        if tick in labels:latent=latent+F.l1_loss(F.layer_norm(pred.float(),(pred.shape[-1],)),F.layer_norm(labels[tick].float(),(pred.shape[-1],)))/len(labels)
                        points.append(state)
                        if tick%30==29:publish(micro_progress=dict(stream=stream,actors_completed=first,tick=tick+1,horizon=180))
                    prediction=torch.stack(points,1)
                    player=F.smooth_l1_loss(prediction[...,:2]*16,target[...,:2]*16)+.25*circle_wall_penalty(prediction[...,:2]*16,grid,.20).mean()/(.20**2)
                    velocity,contact=player_auxiliary(prediction,target,batch['states'][:,0],actions)
                    player=player+.25*velocity+.1*contact
                    enemy=player*0
                    if not nav:
                        enemy=F.smooth_l1_loss(prediction[...,8:10]*16,target[...,8:10]*16)+.25*circle_wall_penalty(prediction[...,8:10]*16,grid,.22).mean()/(.22**2)
                        source_previous=torch.cat((batch['states'][:,:1,8:10],target[:,:-1,8:10]),1)
                        own_previous=torch.cat((batch['states'][:,:1,8:10],prediction[:,:-1,8:10]),1)
                        enemy=enemy+.25*F.smooth_l1_loss((prediction[...,8:10]-own_previous)*480,(target[...,8:10]-source_previous)*480)
                    loss=(.5*player+(enemy if not nav else 0)+.1*latent)*count/8
                    loss.backward()
                aux_totals['player_velocity_loss']+=float(velocity.detach())*.5*count/8
                aux_totals['player_contact_loss']+=float(contact.detach())*.5*count/8
                totals['player_loss']+=float(player.detach())*.5*count/8
                totals['enemy_loss']+=float(enemy.detach())*count/8
                totals['latent_loss']+=float(latent.detach())*.1*count/8
                receipts.extend(dict(stream=stream,file=r['file'],start=r['offset'],corpus=r['corpus'],motion_kind=r['motion_kind'],requested_motion_kind=r['requested_motion_kind'],sampling_attempts=r['sampling_attempts']) for r in selected)
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        optimizer.step();step+=1;updates+=1
        metrics=dict(loss_protocol=PROTOCOL,stage='world',step=step,world_updates=updates,train_loss=sum(totals.values()),**totals,**aux_totals,motion_revision=REVISION,gradient_norm=float(norm),
            update_seconds=time.perf_counter()-began,horizon=180,main_actors=16,feedback_every=8,Source_future_feedback=False,samples=receipts)
        with (root/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics,allow_nan=False)+'\n')
        if _hash_tensors(frozen())!=frozen_hash or _hash_tensors(renderer.sr.state_dict())!=render_hash:raise AssertionError('Frozen renderer/encoder/atlas changed')
        publish(loss=metrics['train_loss'],world_updates=updates,latest_metrics=metrics,micro_progress=None)
        save()
    save();publish(status='world_round_complete' if updates==a.until_updates else 'stopped',world_updates=updates)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--until-updates',type=int,required=True);p.add_argument('--microbatch',type=int,default=2);a=p.parse_args()
    try:
        with KeepAwake():run(a)
    except BaseException as error:
        atomic_json(a.run_dir/'failure.json',dict(error=repr(error),updated_at=now()));raise

if __name__=='__main__':main()
