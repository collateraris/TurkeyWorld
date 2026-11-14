"""Isolated Source/learned-world policy diagnostics on pinned held-out starts.

Only the explicitly named Source controller observes current real state. Free
learned rollouts and matched-action replays never receive future Source input.
Imagined head return is not a real-game reward or a gameplay qualification.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch

from .engine import Game,BEHAVIOR_VERSION
from .rssm_preview_assets import preserve_preview_rng
from .runtime_checks import circle_is_clear,segment_is_clear
from .sr_neural_pairs_data import fingerprint


TRAINING_PROTOCOL='dreamer_actor_critic_imagination_v1'
PROTOCOL='dreamer_policy_validation_v1'
PANEL_KIND='dreamer_policy_heldout_starts_v1'
ARMS=('random','scripted','learned_policy')
SOURCE_OBSERVER='actual_current_state20_with_continuous_prior_RSSM_memory; RGB_only_at_initialization; validation_only'


def file_sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def _pixels_sha(image):return hashlib.sha256(np.asarray(image,dtype=np.uint8).tobytes()).hexdigest()


def _source(record,source_factory=Game):
    snapshot=deepcopy(record['snapshot']);grid=np.asarray(record['map'],np.uint8)
    if grid.shape!=(16,16):raise ValueError('Policy Source start needs its actual16x16 map')
    game=source_factory(seed=int(snapshot['seed']),endless=bool(snapshot['endless']),
        enemies_enabled=bool(snapshot.get('enemies_enabled',True)))
    game.map=grid.copy();game.map_height,game.map_width=grid.shape;game.set_state(snapshot)
    state=game.state_vector().astype(np.float32);image=game.render(size=32,overlays=False)
    if(not np.array_equal(state,np.asarray(record['state'],np.float32))
            or _pixels_sha(image)!=record['initial_rgb_sha256']):
        raise ValueError('Held-out Source snapshot no longer reproduces its initial numerical state/RGB')
    return game,image


def build_policy_validation_panel(data_root,*,count=4,seed=173,include_terminal=False):
    if type(count)is not int or count<1 or type(seed)is not int:raise ValueError('Positive distinct policy starts and integer seed required')
    root=Path(data_root).resolve();manifest_path=root/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'));records=manifest.get('records',[])
    eligible=[]
    for record in records:
        relative=record.get('relative_path','').replace('\\','/')
        if '/rgb/val/'in '/'+relative and relative.endswith('.npz'):eligible.append(record)
    if len(eligible)<count:raise ValueError('Insufficient distinct pinned held-out VAL policy starts')
    if type(include_terminal)is not bool:raise ValueError('Policy terminal-panel opt-in must be explicit bool')
    rng=np.random.default_rng(seed)
    if not include_terminal:
        selected=[eligible[i]for i in sorted(rng.choice(len(eligible),count,replace=False).tolist())]
    else:
        if count<3:raise ValueError('Terminal-inclusive policy panel needs navigation, combat and terminal starts')
        navigation=[i for i,r in enumerate(eligible)if r['relative_path'].replace('\\','/').startswith('navigation/')]
        combat=[i for i,r in enumerate(eligible)if r['relative_path'].replace('\\','/').startswith('full_game/')]
        terminal=[]
        for i,record in enumerate(eligible):
            if not record['relative_path'].replace('\\','/').startswith('moving1/'):continue
            path=(root/record['relative_path']).resolve()
            if root not in path.parents or file_sha(path)!=record['sha256']:raise ValueError('Terminal policy Source record changed')
            with np.load(path,allow_pickle=False)as source:
                hp=source['states'][:,4]
                if bool(((hp[:-1]>0)&(hp[1:]<=0)).any()):terminal.append(i)
        if not navigation or not combat or not terminal:raise ValueError('Terminal policy panel needs genuine held-out nav/combat/death coverage')
        chosen=[int(rng.choice(values))for values in(navigation,combat,terminal)]
        remaining=[i for i in range(len(eligible))if i not in chosen]
        chosen.extend(rng.choice(remaining,count-len(chosen),replace=False).tolist())
        selected=[eligible[i]for i in sorted(chosen)]
    starts=[]
    for record in selected:
        path=(root/record['relative_path']).resolve()
        if root not in path.parents or file_sha(path)!=record['sha256']:raise ValueError('Policy held-out source path/hash changed')
        with np.load(path,allow_pickle=False)as source:
            if(str(source['split'])!='val' or 'initial_state'not in source
                    or source['states'].shape[-1]!=20 or abs(float(source['dt'])-1/30)>1e-7):
                raise ValueError('Policy validation requires genuine Source VAL initial snapshots at30Hz')
            snapshot=json.loads(str(source['initial_state']));grid=source['map'].copy();state=source['states'][0].copy()
            image=source['frames'][0].copy()
            hp=source['states'][:,4];terminal_in_episode=bool(((hp[:-1]>0)&(hp[1:]<=0)).any())
        start=dict(episode_path=str(path),episode_sha256=record['sha256'],split='val',start_frame=0,
            snapshot=snapshot,map=grid.tolist(),state=state.astype(float).tolist(),initial_rgb_sha256=_pixels_sha(image),
            environment='combat'if snapshot.get('enemies_enabled',True)else'navigation')
        if include_terminal:start['recorded_terminal_in_episode']=terminal_in_episode
        _source(start);starts.append(start)
    receipt=dict(kind=PANEL_KIND,version=1,split='val',seed=seed,frames=len(starts),starts=starts,
        data_root=str(root),data_identity_sha256=file_sha(manifest_path),behavior_version=BEHAVIOR_VERSION,
        initialization='exact_full_Source_snapshot_map_state0_RGB32',Source_initial_observation_only=True,
        training_uses_panel=False,source_held_out=True,max_horizon=450,simulation_hz=30)
    if include_terminal:receipt.update(include_terminal=True,
        terminal_definition='recorded_Source_player_HP_positive_to_nonpositive; policy_can_avoid_the_recorded_death',
        recorded_terminal_starts=sum(r['recorded_terminal_in_episode']for r in starts))
    receipt['fingerprint']=fingerprint(receipt);return receipt


def validate_panel(panel,*,verify_source_files=True):
    value=deepcopy(panel);expected=value.pop('fingerprint',None);starts=value.get('starts',[])
    if(value.get('kind')!=PANEL_KIND or value.get('version')!=1 or value.get('split')!='val'
            or value.get('behavior_version')!=BEHAVIOR_VERSION or value.get('frames')!=len(starts)or not starts
            or value.get('training_uses_panel')is not False or value.get('source_held_out')is not True
            or value.get('max_horizon')!=450 or value.get('simulation_hz')!=30
            or expected!=fingerprint(value)
            or len({(r.get('episode_sha256'),r.get('start_frame'))for r in starts})!=len(starts)
            or any(r.get('split')!='val'or r.get('start_frame')!=0 for r in starts)):
        raise ValueError('Policy VAL panel protocol, identity, unique starts or held-out labels changed')
    if verify_source_files:
        root=Path(value['data_root']).resolve()
        if file_sha(root/'manifest.json')!=value['data_identity_sha256']:raise ValueError('Policy VAL manifest changed')
        for start in starts:
            path=Path(start['episode_path']).resolve()
            if root not in path.parents or file_sha(path)!=start['episode_sha256']:raise ValueError('Policy held-out Source file changed')
    return panel


def _stats(values):
    a=np.asarray(values,float).reshape(-1)
    return dict(n=int(a.size),mean=float(a.mean())if a.size else None,p95=float(np.percentile(a,95))if a.size else None,
        maximum=float(a.max())if a.size else None)


def _action(policy,features):
    value=policy(features)
    if(not torch.is_tensor(value)or value.shape!=(1,4)or not bool(torch.isfinite(value).all())
            or bool((value[:,:3].abs()>1+1e-6).any())or bool(((value[:,3]<0)|(value[:,3]>1)).any())):
        raise ValueError('Policy adapter must return finite bounded action4[1,4], fire in[0,1]')
    return value.detach().float()


def _controller(name,learned,rng):
    if name=='learned_policy':return learned
    if name=='random':return lambda features:torch.tensor([[*rng.choice([-1.,0.,1.],3).tolist(),float(rng.integers(2))]],device=features.device)
    def scripted(features):
        s=features[0,:20].detach();delta=s[8:10]-s[:2];heading=torch.atan2(s[2],s[3]);angle=torch.atan2(delta[1],delta[0])-heading
        angle=torch.atan2(angle.sin(),angle.cos());alive=bool(s[12]>0 and s[13]>=.5);distance=float(delta.norm()*16)
        return torch.tensor([[float(alive and distance>1.5 and abs(float(angle))<.35),0.,
            max(-1.,min(1.,float(angle)*2)),float(alive and abs(float(angle))<.12)]],device=features.device)
    return scripted


def _initialize(adapter,game,image,device):
    state=torch.as_tensor(game.state_vector(),device=device,dtype=torch.float32)[None]
    grid=torch.as_tensor(game.map,device=device,dtype=torch.long)[None]
    rgb=torch.as_tensor(image.copy(),device=device,dtype=torch.float32).permute(2,0,1)[None]/255
    return adapter.initialize(state,grid,rgb)


def _step(adapter,world_state,action):
    result=adapter.step(world_state,action)
    if not isinstance(result,dict)or 'world_state'not in result:raise ValueError('Policy world adapter requires an explicit next WorldState')
    ws=result['world_state'];state=ws.state
    if state.shape!=(1,20)or not bool(torch.isfinite(state).all()):raise FloatingPointError('Nonfinite policy world state')
    reward=float(torch.as_tensor(result['reward']).reshape(-1)[0]);continuation=float(torch.as_tensor(result['continue_prob']).reshape(-1)[0])
    if not np.isfinite([reward,continuation]).all()or not 0<=continuation<=1:raise FloatingPointError('Invalid imagined reward/continue prediction')
    return ws,reward,continuation


def _geometry(previous,current,grid):
    a,b=np.asarray(previous,float),np.asarray(current,float)
    result=dict(player_wall_steps=int(not circle_is_clear(grid,b[:2]*16,.20)),
        player_wall_path_steps=int(not segment_is_clear(grid,a[:2]*16,b[:2]*16,.20)),npc_wall_steps=0,npc_wall_path_steps=0)
    if b[12]>0 and b[13]>=.5:result['npc_wall_steps']=int(not circle_is_clear(grid,b[8:10]*16,.25))
    if a[12]>0 and a[13]>=.5 and b[12]>0 and b[13]>=.5:
        result['npc_wall_path_steps']=int(not segment_is_clear(grid,a[8:10]*16,b[8:10]*16,.25))
    return result


def _rollout_source(adapter,controller,start,horizon,device,reward_settings,source_factory):
    from .dreamer_reward_continue import source_reward_continue
    game,image=_source(start,source_factory);ws=_initialize(adapter,game,image,device);states=[game.state_vector().copy()]
    actions=[];rewards=[];continuations=[];events=[];geometry=[];blocked=0
    for _ in range(horizon):
        if game.done:break
        action=_action(controller,adapter.features(ws));old=states[-1];event=game.step(action[0].cpu().numpy(),dt=1/30)
        current=game.state_vector().copy();ws,_,_=_step(adapter,ws,action)
        # Only this real-environment validation controller sees its current
        # observation. No real next state enters free learned/tape rollouts.
        actual=torch.as_tensor(current,device=device,dtype=torch.float32)[None];ws=replace(ws,state=actual)
        measured=source_reward_continue(torch.as_tensor(old,device=device)[None],actual,
            torch.as_tensor(Game.event_vector(event),device=device)[None],config=reward_settings)
        rewards.append(float(measured['reward'].reshape(-1)[0]));continuations.append(float(measured['continue_target'].reshape(-1)[0]))
        actions.append(action[0].cpu().numpy());states.append(current);events.append(Game.event_vector(event).tolist())
        geometry.append(_geometry(old,current,game.map));blocked+=int(np.linalg.norm(actions[-1][:2])>0 and np.linalg.norm(current[:2]-old[:2])*16<1e-6)
    return dict(states=states,actions=actions,rewards=rewards,continue_targets=continuations,events=events,
        geometry=geometry,blocked_movement_steps=blocked,controller_observation=SOURCE_OBSERVER)


def _rollout_learned(adapter,controller,start,horizon,device,source_factory,*,action_tape=None):
    game,image=_source(start,source_factory);ws=_initialize(adapter,game,image,device);states=[ws.state[0].cpu().numpy().copy()]
    rewards=[];continuations=[];geometry=[];actions=[]
    for tick in range(horizon if action_tape is None else len(action_tape)):
        if action_tape is None and states[-1][4]<=0:break
        action=_action(controller,adapter.features(ws))if action_tape is None else torch.as_tensor(action_tape[tick],device=device,dtype=torch.float32)[None]
        old=states[-1];ws,reward,continuation=_step(adapter,ws,action);state=ws.state[0].detach().cpu().numpy().copy()
        rewards.append(reward);continuations.append(continuation);states.append(state);actions.append(action[0].cpu().numpy())
        geometry.append(_geometry(old,state,game.map))
    return dict(states=states,actions=actions,rewards=rewards,continue_predictions=continuations,geometry=geometry,
        Source_future_feedback=False,Source_state_corrections=0,initial_Source_observation_only=True,
        action_tape_replay=action_tape is not None,replay_continues_after_predicted_death_for_diagnostics=action_tape is not None)


def _episode_summary(rollout,horizon,*,source):
    states=rollout['states'];count=len(rollout['actions']);first,last=states[0],states[-1]
    result=dict(transitions=count,simulated_seconds=count/30,return_sum=float(sum(rollout['rewards'])),
        return_kind='actual_Source_game_reward'if source else'predicted_frozen_reward_head_NOT_actual_game_reward',
        final_player_HP=float(last[4]*100),player_damage_HP=float(sum(max(0.,float(a[4]-b[4]))*100 for a,b in zip(states,states[1:]))),
        kill_delta=float(last[18]-first[18]),terminated_player_death=bool(last[4]<=0),
        time_limit_truncated=bool(count==horizon and last[4]>0),corpus_time_limit_is_terminal=False,
        **{key:sum(row[key]for row in rollout['geometry'])for key in('player_wall_steps','player_wall_path_steps','npc_wall_steps','npc_wall_path_steps')})
    if source:result['blocked_movement_steps']=rollout['blocked_movement_steps']
    else:result['continue_prediction']=_stats(rollout['continue_predictions'])
    return result


def _matched_drift(source,replay):
    a=np.asarray(source['states']);b=np.asarray(replay['states'])
    if a.shape!=b.shape:raise ValueError('Matched-action policy drift requires the exact same action tape length')
    player=np.linalg.norm((a[1:,:2]-b[1:,:2])*16,axis=-1);valid=a[1:,4]>0
    npc=np.linalg.norm((a[1:,8:10]-b[1:,8:10])*16,axis=-1);npc_alive=(a[1:,12]>0)&(a[1:,13]>=.5)
    return dict(transitions=len(player),same_actions=True,Source_future_feedback=False,Source_state_corrections=0,
        player_xy_tiles=_stats(player[valid]),player_final_xy_tiles=float(player[-1])if len(player)else None,
        npc_xy_tiles=_stats(npc[npc_alive]),npc_Source_alive_missing_predictions=int((npc_alive&((b[1:,12]<=0)|(b[1:,13]<.5))).sum()),
        final_player_HP_error=float(abs(a[-1,4]-b[-1,4])*100),final_kill_delta_error=float(abs(a[-1,18]-b[-1,18])),
        reward_head_MAE_on_same_actions=float(np.abs(np.asarray(source['rewards'])-np.asarray(replay['rewards'])).mean())if len(player)else None,
        continue_head_MAE_on_same_actions=float(np.abs(np.asarray(source['continue_targets'])-np.asarray(replay['continue_predictions'])).mean())if len(player)else None)


@torch.no_grad()
def evaluate_policy(policy_action_fn,adapter,panel,*,step,policy_identity,world_identity,reward_settings,
                    horizon=32,seed=173,source_factory=Game,verify_source_files=True):
    from .dreamer_reward_continue import REWARD_SETTINGS
    validate_panel(panel,verify_source_files=verify_source_files)
    pin=lambda value:isinstance(value,str)and re.fullmatch('[0-9a-f]{64}',value)is not None
    if(type(step)is not int or step<0 or type(horizon)is not int or not 1<=horizon<=450
            or not callable(policy_action_fn)or reward_settings!=REWARD_SETTINGS
            or not pin(policy_identity.get('checkpoint_sha256'))or policy_identity.get('step')!=step
            or not pin(policy_identity.get('plan_sha256'))or not pin(world_identity.get('checkpoint_sha256'))
            or type(world_identity.get('step'))is not int):
        raise ValueError('Policy validation requires exact policy/world/reward identities and a horizon no longer than450')
    world=getattr(adapter,'world',None)
    if world is None:raise ValueError('Policy validation adapter must expose its pinned frozen world')
    device=next(world.parameters()).device;modules=[]
    from .renderer_student import _hash_tensors
    measured_models=[obj for obj in(world,getattr(adapter,'heads',None),policy_action_fn)if isinstance(obj,torch.nn.Module)]
    before_hashes=[_hash_tensors(obj.state_dict())for obj in measured_models]
    for obj in(world,getattr(adapter,'heads',None),policy_action_fn):
        if isinstance(obj,torch.nn.Module):modules.extend((m,m.training)for m in obj.modules())
    latent_mode=getattr(adapter,'sample_latents',False);cases={name:[]for name in ARMS}
    try:
        for module,_ in modules:module.training=False
        if hasattr(adapter,'sample_latents'):adapter.sample_latents=False
        with preserve_preview_rng(device):
            for name in ARMS:
                for index,start in enumerate(panel['starts']):
                    source_controller=_controller(name,policy_action_fn,np.random.default_rng(seed+index))
                    source=_rollout_source(adapter,source_controller,start,horizon,device,reward_settings,source_factory)
                    learned_controller=_controller(name,policy_action_fn,np.random.default_rng(seed+index))
                    learned=_rollout_learned(adapter,learned_controller,start,horizon,device,source_factory)
                    replay=_rollout_learned(adapter,None,start,horizon,device,source_factory,action_tape=source['actions'])
                    cases[name].append(dict(episode_sha256=start['episode_sha256'],environment=start['environment'],
                        source=_episode_summary(source,horizon,source=True),learned=_episode_summary(learned,horizon,source=False),
                        matched_action_drift=_matched_drift(source,replay),
                        source_action_fingerprint=fingerprint([a.tolist()for a in source['actions']]),
                        learned_action_fingerprint=fingerprint([a.tolist()for a in learned['actions']])))
    finally:
        for module,training in modules:module.training=training
        if hasattr(adapter,'sample_latents'):adapter.sample_latents=latent_mode
    if any(_hash_tensors(obj.state_dict())!=old for obj,old in zip(measured_models,before_hashes)):
        raise AssertionError('Policy validation changed frozen world/head/policy weights')
    arms={}
    for name,values in cases.items():
        arms[name]=dict(cases=values,source_return=_stats([v['source']['return_sum']for v in values]),
            learned_predicted_head_return=_stats([v['learned']['return_sum']for v in values]),
            source_kills=_stats([v['source']['kill_delta']for v in values]),
            source_final_HP=_stats([v['source']['final_player_HP']for v in values]),
            source_wall_steps=sum(v['source']['player_wall_steps']for v in values),
            learned_wall_steps=sum(v['learned']['player_wall_steps']for v in values),
            matched_final_player_drift_tiles=_stats([v['matched_action_drift']['player_final_xy_tiles']for v in values
                if v['matched_action_drift']['player_final_xy_tiles']is not None]))
    report=dict(kind=PROTOCOL,version=1,training_protocol=TRAINING_PROTOCOL,step=step,horizon=horizon,
        max_game_horizon=450,simulation_hz=30,starts=len(panel['starts']),panel_fingerprint=panel['fingerprint'],
        data_identity_sha256=panel['data_identity_sha256'],policy_identity=deepcopy(policy_identity),
        world_identity=deepcopy(world_identity),reward_settings=deepcopy(reward_settings),arms=arms,
        Source_controller_observation=SOURCE_OBSERVER,Source_closedloop_validation_only=True,
        learned_world_Source_feedback=False,matched_action_Source_feedback=False,
        imagination_training_Source_feedback=False,training_modified=False,policy_controls_human_game=False,
        source_held_out=True,frozen_models_bitexact=True,
        interpretation='actual_Source_reward_and_predicted_head_reward_are_distinct; matched_actions_are_drift_diagnostics',
        gameplay_qualified=False,automatic_policy_deployment=False,quality_status='diagnostic_only_no_fixed_gameplay_pass_criterion')
    json.dumps(report,allow_nan=False);return report
