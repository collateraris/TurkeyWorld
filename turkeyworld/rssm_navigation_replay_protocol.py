"""Pinned no-enemy main-actor replay; existing combat auxiliary exposure stays intact."""
from copy import deepcopy
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from .rssm_combat_memory_protocol import digest_json,file_sha
from .rssm_moe_protocol import moe_training_horizon,MOE_HORIZONS

PROTOCOL='rssm_moe_three_way_main_replay_v1'
SETTINGS=dict(version=1,main_microbatch_size=64,gradient_accumulation_steps=4,effective_main_batch_size=256,
    combat_main_actors_per_microbatch=38,enemy_motion_main_actors_per_microbatch=16,navigation_main_actors_per_microbatch=10,
    combat_main_actors_per_update=152,enemy_motion_main_actors_per_update=64,navigation_main_actors_per_update=40,
    navigation_fraction=.15625,enemy_motion_fraction=.25,combat_fraction=.59375,
    combat_episode_rng_draws_per_microbatch=64,discarded_combat_rng_draws_per_microbatch=26,
    combat_main_player_weight=38/64,combat_main_router_weight=38/64,navigation_player_router_weight=10/64,
    combat_main_enemy_weight=38/54,enemy_motion_enemy_weight=16/54,enemy_motion_router_weight=16/64,
    enemy_normalization='fixed_actor_weighted_means_of_Source_continuing_alive_cohorts',combat_auxiliary_actors_per_microbatch=64,combat_late_actors_per_microbatch=8,
    combat_auxiliary_and_late_weights_unchanged=True,navigation_only_profile=True,
    navigation_losses=['player_xy','router_role','router_balance'],navigation_combat_source_queries=0,
    navigation_objective='rssm_moe_no_enemy_main_navigation_v1',enemy_motion_objective='rssm_moe_quiet_enemy_motion_v1',
    enemy_motion_losses={'xy':1.,'velocity':.1,'direction':.05},
    navigation_future_rgb_inputs=0,all_streams_same_horizon_context_latent_mode=True,
    separate_navigation_private_pcg=True,gate_reset=False,optimizer_reset=False)


def navigation_replay_plan(state,*,source_checkpoint,navigation_root,enemy_motion_root,seed=174173):
    moe_training_horizon(state)
    root=Path(navigation_root).resolve();gate=state['rssm_training']['moe_training']
    return dict(version=1,kind=PROTOCOL,settings=deepcopy(SETTINGS),seed=seed,
        source_checkpoint=str(Path(source_checkpoint).resolve()),source_checkpoint_sha256=file_sha(source_checkpoint),
        source_optimizer_step=state['global_training_step'],source_moe_updates=gate['updates'],
        source_gate_sha256=digest_json(gate),moe_training_plan_sha256=gate['sha256'],
        navigation_root=str(root),navigation_manifest_sha256=file_sha(root/'manifest.json'),
        navigation_train_directory=str(root/'long/rgb/train'),
        enemy_motion_root=str(Path(enemy_motion_root).resolve()),
        enemy_motion_manifest_sha256=file_sha(Path(enemy_motion_root)/'manifest.json'),
        enemy_motion_train_directory=str(Path(enemy_motion_root).resolve()/'full_game/rgb/train'))


def _replay_totals(record,step,gate=None):
    revision = None if gate is None else gate.get('batch_revision')
    if revision is None:
        updates=step-record['source_optimizer_step']
        return dict(updates=updates,navigation_main_actors=updates*40,
            enemy_motion_main_actors=updates*64,combat_main_actors=updates*152,
            discarded_combat_rng_draws=updates*104)
    from .rssm_moe_batch_revision import batch_settings_for_gate
    settings=batch_settings_for_gate(gate)
    baseline=revision.get('source_navigation_replay_revision')
    if (not isinstance(baseline,dict)
            or digest_json(baseline)!=revision.get('source_navigation_replay_revision_sha256')
            or any(record.get(key)!=baseline.get(key)for key in ('kind','sha256',
                'source_optimizer_step','source_moe_updates','source_checkpoint_sha256',
                'source_gate_sha256','navigation_manifest_sha256','enemy_motion_manifest_sha256'))):
        raise ValueError('Micro16 replay baseline differs from its original pinned navigation revision')
    _validate_record(baseline,revision['source_optimizer_step'],revision['source_gate'])
    updates=step-revision['source_optimizer_step']
    if (type(updates)is not int or updates<0 or revision.get('updates')!=updates
            or revision.get('last_optimizer_step')!=step):
        raise ValueError('Micro16 replay and batch optimizer clocks diverged')
    result={'updates':baseline['updates']+updates}
    for name,setting in (('navigation_main_actors','navigation_main_actors_per_update'),
        ('enemy_motion_main_actors','enemy_motion_main_actors_per_update'),
        ('combat_main_actors','combat_main_actors_per_update'),
        ('discarded_combat_rng_draws','discarded_combat_rng_draws_per_update')):
        result[name]=baseline[name]+updates*settings[setting]
    return result


def _validate_record(record,step,gate=None):
    updates=step-record['source_optimizer_step']
    expected=_replay_totals(record,step,gate)
    if (type(record.get('updates'))is not int or updates<0 or record['updates']!=updates
            or record['updates']!=expected['updates']
            or record.get('last_optimizer_step')!=step
            or any(record.get(key)!=value for key,value in expected.items())
            or record.get('sampler_state',{}).get('draws')!=expected['navigation_main_actors']
            or record.get('enemy_sampler_state',{}).get('draws')!=expected['enemy_motion_main_actors']):
        raise ValueError('Navigation replay complete optimizer/draw ledger changed')


def bind_navigation_replay(state,plan,*,path,sha256,loaded_checkpoint_sha256):
    raw=Path(path).read_bytes()
    if (hashlib.sha256(raw).hexdigest()!=sha256 or json.loads(raw.decode('utf-8'))!=plan
            or plan.get('kind')!=PROTOCOL or plan.get('version')!=1 or plan.get('settings')!=SETTINGS
            or type(plan.get('seed'))is not int or type(plan.get('source_optimizer_step'))is not int
            or file_sha(Path(plan['navigation_root'])/'manifest.json')!=plan['navigation_manifest_sha256']
            or Path(plan['navigation_train_directory']).resolve()!=Path(plan['navigation_root']).resolve()/'long/rgb/train'
            or file_sha(plan['source_checkpoint'])!=plan['source_checkpoint_sha256']
            or file_sha(Path(plan['enemy_motion_root'])/'manifest.json')!=plan['enemy_motion_manifest_sha256']
            or Path(plan['enemy_motion_train_directory']).resolve()!=Path(plan['enemy_motion_root']).resolve()/'full_game/rgb/train'):
        raise ValueError('Navigation replay immutable plan, data or source checkpoint changed')
    gate=state['rssm_training']['moe_training'];step=state['global_training_step'];start=plan['source_optimizer_step']
    if gate['sha256']!=plan['moe_training_plan_sha256']or gate['last_optimizer_step']!=step:
        raise ValueError('Navigation replay must preserve the existing complete MoE gate')
    record=state['rssm_training'].get('navigation_replay_revision')
    if record is None:
        if (step!=start or loaded_checkpoint_sha256!=plan['source_checkpoint_sha256']
                or gate['updates']!=plan['source_moe_updates']or digest_json(gate)!=plan['source_gate_sha256']):
            raise ValueError('Navigation replay activation needs the exact raw source gate before new budget binding')
        rng=np.random.default_rng(plan['seed'])
        record=dict(version=1,kind=PROTOCOL,plan=str(Path(path).resolve()),sha256=sha256,
            source_optimizer_step=start,source_moe_updates=plan['source_moe_updates'],
            source_gate_sha256=plan['source_gate_sha256'],source_checkpoint_sha256=plan['source_checkpoint_sha256'],
            last_optimizer_step=step,updates=0,navigation_main_actors=0,enemy_motion_main_actors=0,combat_main_actors=0,
            discarded_combat_rng_draws=0,navigation_manifest_sha256=plan['navigation_manifest_sha256'],
            enemy_motion_manifest_sha256=plan['enemy_motion_manifest_sha256'],
            enemy_sampler_state=dict(protocol=PROTOCOL,draws=0,rng=deepcopy(np.random.default_rng(plan['seed']+1).bit_generator.state)),
            sampler_state=dict(protocol=PROTOCOL,draws=0,rng=deepcopy(rng.bit_generator.state)))
        state['rssm_training']['navigation_replay_revision']=record
        state.setdefault('objective_history',[]).append(dict(step=step,kind=PROTOCOL,
            reason='user_requested_at_least_15_percent_actual_no_enemy_main_actors',
            source_loss=state.get('loss'),settings=deepcopy(SETTINGS)))
    elif (record.get('sha256')!=sha256 or record.get('source_optimizer_step')!=start
            or record.get('source_checkpoint_sha256')!=plan['source_checkpoint_sha256']
            or record.get('source_gate_sha256')!=plan['source_gate_sha256']
            or record.get('source_moe_updates')!=plan['source_moe_updates']
            or record.get('navigation_manifest_sha256')!=plan['navigation_manifest_sha256']
            or record.get('enemy_motion_manifest_sha256')!=plan['enemy_motion_manifest_sha256']):
        raise ValueError('Navigation replay saved revision identity changed')
    _validate_record(record,step,gate)
    if gate['updates']!=record['source_moe_updates']+record['updates']:
        raise ValueError('Navigation replay and original MoE optimizer clocks diverged')
    state.setdefault('training_protocol',{})['navigation_replay']=dict(protocol=PROTOCOL,plan_sha256=sha256,**deepcopy(SETTINGS))
    state['training_schedule']='simultaneous_combat_with_no_enemy_navigation_main_replay'
    return record


class NavigationReplaySampler:
    """Fresh or restored independent PCG, never the combat/private global RNG."""
    actors=10
    state_field='sampler_state'

    def __init__(self,data,record,*,batch_settings=None):
        if getattr(data,'split',None)!='train' or not data.episodes:
            raise ValueError('Navigation replay samples TRAIN episodes only')
        self.data=data;self.rng=np.random.default_rng()
        # Keep the old strict restoration rule by default. A pinned smaller
        # complete update may add five NAV/eight NPC draws to the old ledger.
        self.complete_draw_quantum=self.actors
        if batch_settings is not None:
            from math import gcd
            key=('navigation_main_actors_per_update' if self.state_field=='sampler_state'
                 else 'enemy_motion_main_actors_per_update')
            self.complete_draw_quantum=gcd(self.actors,batch_settings[key])
            # A new update can be complete even when its inherited total was
            # produced by five-NAV B32 updates. Keep every earlier draw/RNG;
            # the pinned recursive replay ledger validates the exact totals.
            history_key=('historical_navigation_actor_counts_per_update'if self.state_field=='sampler_state'
                         else'historical_enemy_motion_actor_counts_per_update')
            historical=batch_settings.get(history_key,[])
            if not isinstance(historical,list)or any(type(n)is not int or n<1 for n in historical):
                raise ValueError('Replay historical complete-update quanta must be positive integers')
            for count in historical:self.complete_draw_quantum=gcd(self.complete_draw_quantum,count)
        self.restore_state(record[self.state_field])

    def state_dict(self):
        return dict(protocol=PROTOCOL,draws=self.draws,rng=deepcopy(self.rng.bit_generator.state))

    def restore_state(self,state):
        self._restore_state(state,require_complete=True)

    def _restore_state(self,state,*,require_complete):
        if (state.get('protocol')!=PROTOCOL or type(state.get('draws'))is not int or state['draws']<0
                or require_complete and state['draws']%self.complete_draw_quantum):
            raise ValueError('Navigation sampler complete ten-actor draw ledger changed')
        private=np.random.default_rng()
        try:private.bit_generator.state=deepcopy(state['rng'])
        except (KeyError,TypeError,ValueError)as error:raise ValueError('Navigation sampler PCG changed')from error
        self.rng.bit_generator.state=deepcopy(private.bit_generator.state);self.draws=state['draws']

    def batch(self,*,context,horizon,actors=None):
        count=self.actors if actors is None else actors
        if (context not in(1,2)or horizon not in MOE_HORIZONS
                or type(count)is not int or not 1<=count<=self.actors):
            raise ValueError('Navigation replay must use the current MoE context/horizon')
        length=context+horizon
        eligible=[i for i,e in enumerate(self.data.episodes)if len(e['frames'])>=length]
        if not eligible:raise ValueError('No complete no-enemy TRAIN navigation tape fits this stage')
        before=self.state_dict()
        try:
            ids=self.rng.choice(eligible,size=count)
            pairs=np.asarray([[int(i),int(self.rng.integers(len(self.data.episodes[int(i)]['frames'])-length+1))]for i in ids],dtype=np.int64)
            batch=self.data.batch(count,context=context,horizon=horizon,indices=pairs)
            self.validate_batch(batch,actors=count)
            self.draws+=count
            return batch,dict(protocol=PROTOCOL,actors=count,pairs=pairs.tolist(),context=context,horizon=horizon,
                navigation_only=self.actors==10,private_pcg_draws=self.draws,source_future_feedback=False)
        except BaseException:
            self._restore_state(before,require_complete=False);raise


    def validate_batch(self,batch,*,actors=None):
        states=batch['states'];actions=batch['actions']
        if (len(states)!=(self.actors if actors is None else actors) or bool((states[...,12:14]!=0).any())
                or bool((actions[...,3]!=0).any())or not bool(torch.isfinite(states).all())):
            raise ValueError('Navigation replay requires genuine no-enemy/no-fire Source labels')


class EnemyMotionReplaySampler(NavigationReplaySampler):
    actors=16
    state_field='enemy_sampler_state'

    def validate_batch(self,batch,*,actors=None):
        states=batch['states'];actions=batch['actions']
        if (len(states)!=(self.actors if actors is None else actors) or not bool(((states[...,12]>0)&(states[...,13]>=.5)).all())
                or bool((actions[...,3]!=0).any())or not bool(torch.isfinite(states).all())):
            raise ValueError('Enemy-motion replay requires genuine live-NPC/no-fire Source labels')


def commit_navigation_replay(state,*,step,sampler_state,enemy_sampler_state,micro_rows):
    record=state['rssm_training']['navigation_replay_revision']
    gate=state['rssm_training']['moe_training']
    from .rssm_moe_batch_revision import batch_settings_for_gate
    settings=batch_settings_for_gate(gate)
    layouts=settings['micro_layouts']
    nav_count=settings['navigation_main_actors_per_update']
    enemy_count=settings['enemy_motion_main_actors_per_update']
    if (step!=state['global_training_step']or step!=record['last_optimizer_step']+1
            or gate['last_optimizer_step']!=step or gate['updates']!=record['source_moe_updates']+record['updates']+1
            or len(micro_rows)!=settings['gradient_accumulation_steps'] or any(r.get('main_navigation_actor_count')!=layout['navigation']
                or r.get('main_combat_actor_count')!=layout['combat'] or r.get('main_enemy_motion_actor_count')!=layout['enemy_motion']
                or r.get('main_discarded_combat_rng_count')!=settings['combat_rng_draws_per_microbatch']-layout['combat']
                for r,layout in zip(micro_rows,layouts))
            or sampler_state.get('protocol')!=PROTOCOL or sampler_state.get('draws')!=record['navigation_main_actors']+nav_count
            or enemy_sampler_state.get('protocol')!=PROTOCOL or enemy_sampler_state.get('draws')!=record['enemy_motion_main_actors']+enemy_count):
        raise ValueError('Navigation replay commit needs all completed pinned main-cohort microbatches')
    private=np.random.default_rng()
    try:
        private.bit_generator.state=deepcopy(sampler_state['rng'])
        private.bit_generator.state=deepcopy(enemy_sampler_state['rng'])
    except (KeyError,TypeError,ValueError)as error:raise ValueError('Navigation replay committed PCG changed')from error
    record.update(last_optimizer_step=step,updates=record['updates']+1,
        navigation_main_actors=record['navigation_main_actors']+nav_count,
        combat_main_actors=record['combat_main_actors']+settings['combat_main_actors_per_update'],
        enemy_motion_main_actors=record['enemy_motion_main_actors']+enemy_count,
        discarded_combat_rng_draws=record['discarded_combat_rng_draws']+settings['discarded_combat_rng_draws_per_update'],
        sampler_state=deepcopy(sampler_state),enemy_sampler_state=deepcopy(enemy_sampler_state))
    _validate_record(record,step,gate)
    return record
