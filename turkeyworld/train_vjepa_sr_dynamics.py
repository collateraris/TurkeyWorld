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
from .vjepa_motion_aux import REVISION,PREVIOUS_REVISION,segmented_player_auxiliary
from .vjepa_balanced_windows import motion_window_index,draw_available
from .vjepa_combat_training import (EVENT_HEADS, STREAM_ACTORS, THREE_MODE, combat_npc_motion,
                                    combat_objective, training_profile, dynamics_parameter_trainable)
from .structured_state import EVENT_NAMES

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
        paths+=sorted((ROOT/'training_runs/vjepa_sr_cycle_20261117/cycles').glob('*/targeted_v2/collection_receipt.json'))
        self.corpora=[json.loads(p.read_text()) for p in paths]
        state=dict(protocol=PROTOCOL,draws=0,rng=self.rng.bit_generator.state)
        self.data=MotionReplay(NAV if stream=='navigation' else ENEMY,state,stream=stream,corpora=self.corpora).data
        self.window_pools=[motion_window_index(data.episodes) for data in self.data]
    def draw(self, count=8):
        rows=[]
        self.balance_rotation=int(self.rng.integers(3))
        for actor in range(count):
            if actor < max(1, count//4) or len(self.data)==1:index=0
            elif actor < count//2:index=int(self.rng.integers(1,len(self.data)-1)) if len(self.data)>2 else 0
            else:index=len(self.data)-1
            data=self.data[index]
            episode,start,kind,available=draw_available(self.window_pools[index],self.rng,actor+self.balance_rotation)
            e=data.episodes[episode]
            batch=data.batch(1,context=1,horizon=180,indices=np.asarray([[episode,start]]))
            # Source past only; the four-frame context is rendered through SR.
            initial=[max(0,start-3+i) for i in range(4)]
            batch['initial_states']=torch.as_tensor(np.asarray(e['states'][initial]))[None].float()
            batch['initial_frames']=torch.as_tensor(np.asarray(e['frames'][initial])).permute(0,3,1,2)[None].float()/255
            batch['offset']=start;batch['file']=e['path'];batch['corpus']=index
            batch['motion_kind']=kind;batch['requested_motion_kind']=kind;batch['sampling_attempts']=1;batch['available_motion_kinds']=available
            rows.append(batch)
        return rows


def run(a):
    root=a.run_dir.resolve();path=root/'checkpoints/vjepa_world.pt';root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    model,saved=load_world_checkpoint(path,'cpu');model.to('cuda').eval()
    if not model.config.vjepa_pooled_memory:raise ValueError('SR-feedback trainer requires pooled V-JEPA memory')
    settings=saved['vjepa_sr_feedback'];renderer=FrozenBaseSR(settings,'cuda')
    old=saved.get('vjepa_sr_training',{});step=saved['training_steps'];updates=old.get('updates',0)
    clamp_gradient=a.player_clamp_gradient or old.get('player_clamp_gradient','hard')
    if clamp_gradient not in ('hard','ste'):raise ValueError(f'Unknown player clamp gradient {clamp_gradient!r}')
    # Instance-only override for the H180 optimizer. Inference and all NPC /
    # combat gradients continue to use the checkpoint's ordinary eval forward.
    model.vjepa_backend.train_player_clamp_ste=(clamp_gradient=='ste')
    names=saved['optimizer_parameter_names'];names=names if isinstance(names[0],list) else [names]
    parameters=dict(model.named_parameters());groups=[dict(params=[parameters[n] for n in group]) for group in names]
    optimizer=torch.optim.AdamW(groups,lr=1e-5);optimizer.load_state_dict(saved['optimizer'])
    for group in optimizer.param_groups:group['lr']=1e-5
    profile = training_profile(getattr(a, 'training_profile', None), old)
    stream_counts = STREAM_ACTORS[profile]
    three_mode = profile == THREE_MODE
    first_three_mode = three_mode and old.get('training_profile') != THREE_MODE
    if three_mode and not first_three_mode:
        if old.get('stream_actors') != stream_counts or set(old.get('samplers', {})) != set(stream_counts):
            raise ValueError('Incomplete saved three-mode sampling plan or RNG states')
    head_names = sorted(name for name in parameters if name.startswith(EVENT_HEADS))
    if three_mode and (not head_names or not set(head_names).issubset(set(sum(names, [])))):
        raise ValueError('Combat event heads must already belong to the saved optimizer groups')
    for name, param in parameters.items():
        param.requires_grad_(dynamics_parameter_trainable(name, three_mode))
    ema=copy.deepcopy(model.core.encoder).eval().requires_grad_(False)
    if saved.get('ema_target_encoder'):
        state={k[len('encoder.'):]:v for k,v in saved['ema_target_encoder'].items() if k.startswith('encoder.')}
        ema.load_state_dict(state,strict=True)
    replay={s:Replay(s,old.get('samplers',{}).get(s)) for s in ('navigation','enemy_motion')}
    if three_mode:
        from .vjepa_combat_replay import CombatReplay
        replay['combat'] = CombatReplay(root.parent, old.get('samplers', {}).get('combat'))
    frozen=lambda:{n:p for n,p in model.named_parameters() if not p.requires_grad}
    frozen_hash=_hash_tensors(frozen());render_hash=_hash_tensors(renderer.sr.state_dict());source_step=old.get('source_step',step)
    plan=dict(protocol=PROTOCOL,horizon=180,updates_per_round=100,MAIN=16,microbatch=a.microbatch,
        context='four_observed_past_states_rendered_Base_SR',full_memory_BPTT=True,GT_future_inputs=False,
        encoder_last_blocks_trainable=len(model.core.encoder.blocks)-16,capacity_growth=saved.get('vjepa_capacity_growth'),latent_label_stride=16,latent_weight=.1,lr=1e-5,
        parent_Adam_moments_preserved=True,renderer_frozen=True,feedback=settings,
        motion_revision=REVISION,player_velocity_weight=.25,player_contact_weight=.1,auxiliary_mode='blocks16',auxiliary_block_size=16,player_clamp_gradient=clamp_gradient,wall_coefficient=a.wall_weight,sampler='exact_available_window_classes')
    if three_mode:
        plan.update(training_profile=profile, stream_actors=dict(stream_counts), combat_head_names=head_names,
                    combat_loss='fired_shot_target_CE + melee_BCE + 0.25_soft_expected_damage',
                    npc_combat_motion='alive_same_generation_exclude_respawn_transitions',
                    stream_weights='per_actor_player1_enemy2_latent0.2_combat1')
    plan_path=root/'training_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:
        previous=json.loads(plan_path.read_text())
        if previous.get('motion_revision') not in (None,PREVIOUS_REVISION,REVISION):raise ValueError('Unknown dynamics plan revision')
        atomic_json(root/('training_plan_before_'+REVISION+'_'+str(step)+'.json'),previous)
    atomic_json(plan_path,plan)
    status=dict(stage='world',model='V-JEPA pooled state dynamics + frozen SR feedback',run_id=root.name,
        status='running',current=step,total=None,loss_protocol=PROTOCOL,training_plan=plan,started_at=now())
    def publish(**kw):status.update(current=step,updated_at=now(),**kw);atomic_json(root/'status.json',status)
    def save():
        ledger=dict(protocol=PROTOCOL,source_step=source_step,updates=updates,
            samplers={k:copy.deepcopy(v.rng.bit_generator.state) for k,v in replay.items()},frozen_parameters_sha256=frozen_hash,
            replay_sampling_revision=REVISION,auxiliary_mode='blocks16',player_clamp_gradient=clamp_gradient,wall_coefficient=a.wall_weight)
        if three_mode:
            ledger.update(training_profile=profile, stream_actors=dict(stream_counts),
                          combat_migration=combat_migration)
        payload={**saved,'model':model.state_dict(),'config':vars(model.config),'optimizer':optimizer.state_dict(),
            'training_steps':step,'global_training_step':step,'step':step,'run_state':dict(global_training_step=step),
            'vjepa_sr_training':ledger,'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'rng_numpy_global':np.random.get_state()}
        _save(path,payload)
    if 'rng_torch' in saved:torch.set_rng_state(saved['rng_torch'].cpu())
    if saved.get('rng_cuda'):torch.cuda.set_rng_state_all([v.cpu() for v in saved['rng_cuda']])
    if saved.get('rng_numpy_global') is not None:np.random.set_state(saved['rng_numpy_global'])
    from .learned_axis_collision import verify_frozen_resume
    if first_three_mode:
        # All weights, including the newly unfrozen heads, must still match
        # the previous fingerprint before accepting the reduced frozen set.
        parent_frozen = {**frozen(), **{n: parameters[n] for n in head_names}}
        parent_frozen_hash = _hash_tensors(parent_frozen)
        if not old.get('frozen_parameters_sha256'):
            raise ValueError('Three-mode continuation needs a verified frozen parent')
        frozen_ok = verify_frozen_resume(model, saved, parent_frozen, parent_frozen_hash)
        combat_migration = dict(source_step=step, parent_frozen_sha256=parent_frozen_hash,
                                frozen_sha256=frozen_hash, event_head_names=head_names,
                                initial_event_heads_sha256=_hash_tensors({n:parameters[n] for n in head_names}),
                                retained_adam_slots=len(optimizer.state))
    else:
        frozen_ok = verify_frozen_resume(model, saved, frozen(), frozen_hash)
        combat_migration = old.get('combat_migration')
        if three_mode and (not combat_migration or combat_migration['event_head_names']!=head_names
                           or combat_migration['frozen_sha256']!=frozen_hash):
            raise ValueError('Three-mode continuation does not match recorded trainable heads')
    restored_samplers = [name for name in replay if name in old.get('samplers', {})]
    new_samplers = [name for name in replay if name not in old.get('samplers', {})]
    if not frozen_ok:
        raise ValueError('Frozen parameters changed since the previous optimizer boundary')
    atomic_json(root/'resume_verification.json',dict(step=step,world_updates=updates,
        optimizer_states=len(optimizer.state),parent_Adam_moments_preserved=True,
        frozen_parameters_unchanged=frozen_ok, training_profile=profile,
        combat_migration=combat_migration, stream_actors=dict(stream_counts),
        restored_samplers=restored_samplers, newly_initialized_samplers=new_samplers,
        sampler_states_restored=all(replay[k].rng.bit_generator.state==old['samplers'][k] for k in restored_samplers),
        torch_rng_restored='rng_torch' in saved,replay_sampling_revision=REVISION,auxiliary_mode='blocks16',player_clamp_gradient=clamp_gradient,wall_coefficient=a.wall_weight))
    publish(world_updates=updates,round_target=a.until_updates,parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    # Sparse Source RGB labels are EMA-only; never supplied to the live prior.
    while updates<a.until_updates and not (root/'stop_training').exists():
        began=time.perf_counter();optimizer.zero_grad(set_to_none=True);totals=dict(player_loss=0.,enemy_loss=0.,latent_loss=0.);aux_totals=dict(player_velocity_loss=0.,player_contact_loss=0.);block_totals={};receipts=[]
        stream_metrics = {}
        for stream,sampler in replay.items():
            actors = stream_counts[stream]
            rows = sampler.draw(actors)
            if len(rows) != actors:
                raise ValueError(f'{stream}: expected {actors} training windows, got {len(rows)}')
            summary = stream_metrics[stream] = dict(actors=actors, player_loss=0., enemy_loss=0., latent_loss=0.)
            combat = stream == 'combat'
            if combat:
                summary.update(target_ce=0., attack_bce=0., damage_loss=0., hp_mae=0., combat_total=0.,
                               source_events={name:0 for name in EVENT_NAMES}, event_head_gradient_l2=None)
                totals['combat_loss'] = 0.
            for first in range(0,actors,a.microbatch):
                selected=rows[first:first+a.microbatch];count=len(selected)
                keys = ('states','frames','actions','target_states','map','initial_states')
                if combat:
                    keys += ('events','hit_ids','attack_flags')
                batch={k:torch.cat([r[k] for r in selected],0).to('cuda') for k in keys}
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
                    event_heads=[]
                    def transition(s,m,control):
                        value=model.transition(s,control,fmap,m)
                        outputs = (value['state'],value['memory'],value['latent_prediction'],value['respawn_flags'])
                        if combat:
                            outputs += (value['target_logits'],value['attack_logits'],value['event_probs'])
                        return outputs
                    for tick in range(180):
                        outputs=checkpoint(transition,state.detach(),memory,actions[:,tick],use_reentrant=False)
                        raw,memory,pred,respawn=outputs[:4]
                        if combat:
                            event_heads.append(outputs[4:])
                        state=integrate_coordinate_gradients(state,dict(state=raw,respawn_flags=respawn),COORDINATE_GRADIENT_PROTOCOL)['state']
                        active=feedback_active(state,settings)
                        if bool(active.any()):
                            observation=renderer.observation(state.detach(),grid,navigation_only=nav)
                            memory=checkpoint(lambda mem,img,mask:blend_memory(model,mem,img,settings,mask),memory,observation,active,use_reentrant=False)
                        if tick in labels:latent=latent+F.l1_loss(F.layer_norm(pred.float(),(pred.shape[-1],)),F.layer_norm(labels[tick].float(),(pred.shape[-1],)))/len(labels)
                        points.append(state)
                        if tick%30==29:publish(micro_progress=dict(stream=stream,actors_completed=first,tick=tick+1,horizon=180))
                    prediction=torch.stack(points,1)
                    player=F.smooth_l1_loss(prediction[...,:2]*16,target[...,:2]*16)+a.wall_weight*circle_wall_penalty(prediction[...,:2]*16,grid,.20).mean()/(.20**2)
                    velocity,contact,blocks=segmented_player_auxiliary(prediction,target,batch['states'][:,0],actions)
                    player=player+.25*velocity+.1*contact
                    enemy=player*0
                    if combat:
                        enemy = combat_npc_motion(prediction, target, batch['states'][:,0], grid, a.wall_weight)
                    elif not nav:
                        enemy=F.smooth_l1_loss(prediction[...,8:10]*16,target[...,8:10]*16)+a.wall_weight*circle_wall_penalty(prediction[...,8:10]*16,grid,.22).mean()/(.22**2)
                        source_previous=torch.cat((batch['states'][:,:1,8:10],target[:,:-1,8:10]),1)
                        own_previous=torch.cat((batch['states'][:,:1,8:10],prediction[:,:-1,8:10]),1)
                        enemy=enemy+.25*F.smooth_l1_loss((prediction[...,8:10]-own_previous)*480,(target[...,8:10]-source_previous)*480)
                    # Legacy per-actor scaling also keeps the original two-mode
                    # objective exactly when there are 8 NAV and 8 quiet actors.
                    loss=(.5*player+(enemy if not nav else 0)+.1*latent)*count/8
                    if combat:
                        logits = (torch.stack(t,dim=1) for t in zip(*event_heads))
                        terms = combat_objective(*logits, batch, prediction)
                        loss = loss + terms['combat_total']*(count/16)
                        totals['combat_loss'] += float(terms['combat_total'].detach())*(count/16)
                        for name,value in terms.items():
                            summary[name] += float(value.detach())*(count/actors)
                        for name,n in zip(EVENT_NAMES,batch['events'].sum((0,1)).int().tolist()):
                            summary['source_events'][name] += n
                    loss.backward()
                for name,value in (('player_loss',player), ('enemy_loss',enemy), ('latent_loss',latent)):
                    summary[name] += float(value.detach())*(count/actors)
                for begin,end,v,c in blocks:
                    values=block_totals.setdefault(f'{begin}-{end}',dict(velocity=0.,contact=0.))
                    values['velocity']+=float(v.detach())*.5*count/8;values['contact']+=float(c.detach())*.5*count/8
                aux_totals['player_velocity_loss']+=float(velocity.detach())*.5*count/8
                aux_totals['player_contact_loss']+=float(contact.detach())*.5*count/8
                totals['player_loss']+=float(player.detach())*.5*count/8
                totals['enemy_loss']+=float(enemy.detach())*count/8
                totals['latent_loss']+=float(latent.detach())*.1*count/8
                receipts.extend(dict(stream=stream,file=r['file'],start=r['offset'],corpus=r['corpus'],motion_kind=r['motion_kind'],requested_motion_kind=r['requested_motion_kind'],sampling_attempts=r['sampling_attempts'],available_motion_kinds=r['available_motion_kinds']) for r in selected)
        if three_mode:
            grads = [parameters[n].grad.detach().float().norm() for n in head_names if parameters[n].grad is not None]
            if not grads:
                raise RuntimeError('No combat head received a gradient')
            stream_metrics['combat']['event_head_gradient_l2'] = float(torch.stack(grads).norm())
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        optimizer.step();step+=1;updates+=1
        metrics=dict(loss_protocol=PROTOCOL,stage='world',step=step,world_updates=updates,train_loss=sum(totals.values()),**totals,**aux_totals,auxiliary_blocks=block_totals,player_clamp_gradient=clamp_gradient,wall_coefficient=a.wall_weight,motion_revision=REVISION,gradient_norm=float(norm),
            update_seconds=time.perf_counter()-began,horizon=180,main_actors=16,feedback_every=8,Source_future_feedback=False,samples=receipts)
        if three_mode:
            metrics.update(training_profile=profile, stream_actors=dict(stream_counts), streams=stream_metrics)
        with (root/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics,allow_nan=False)+'\n')
        if _hash_tensors(frozen())!=frozen_hash or _hash_tensors(renderer.sr.state_dict())!=render_hash:raise AssertionError('Frozen renderer/encoder/atlas changed')
        publish(loss=metrics['train_loss'],world_updates=updates,latest_metrics=metrics,micro_progress=None)
        save()
    save();publish(status='world_round_complete' if updates==a.until_updates else 'stopped',world_updates=updates)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--until-updates',type=int,required=True);p.add_argument('--microbatch',type=int,default=2);p.add_argument('--wall-weight',type=float,choices=[.25,.5],default=.25);p.add_argument('--player-clamp-gradient',choices=['hard','ste'],help='Player XY clamp backward only; default: inherit checkpoint (legacy: hard)');p.add_argument('--training-profile',choices=tuple(STREAM_ACTORS),default=None,
        help='World training streams; defaults to the profile saved with the optimizer');a=p.parse_args()
    try:
        with KeepAwake():run(a)
    except BaseException as error:
        atomic_json(a.run_dir/'failure.json',dict(error=repr(error),updated_at=now()));raise

if __name__=='__main__':main()
