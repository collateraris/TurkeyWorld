"""Six-second motion rounds: MAIN16=8 navigation+8 quiet enemy, no Combat loss.

Legacy plans, Adam groups and all three memories are retained. Only this new
revision owns stage advancement and the two motion sampler clocks. The cycle
orchestrator supplies immutable policy-collected TRAIN corpora between rounds.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .gradient_safety import clip_finite_grad_norm_
from .models import load_world_checkpoint
from .power_guard import KeepAwake
from .renderer_student import _hash_tensors
from .rssm_combat_protocol import environment_profile
from .rssm_motion_only_data import MotionReplay
from .rssm_motion_only_revision import (PROTOCOL,HORIZONS,bind_motion_only,current_horizon,
    commit_motion_only,begin_motion_round,publish_contract,runtime_player_settings,
    mark_motion_only_baseline,validate_motion_only,digest)
from .rssm_motion_only_training import motion_only_backward,configure_motion_only_trainable
from .rssm_migration import FROZEN_PREFIXES
from .staged_curriculum import StagedSequenceData
from .train_rssm_combat import RSSMCombatTraining
from .train_rssm_moe import restore_optimizer
from .training_pipeline import atomic_json,now

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'training_runs/rssm_moe_nav15_enemy25_cap20_20261113'


def file_sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


class MotionOnlyTraining(RSSMCombatTraining):
    def save(self,model,optimizer,metadata):
        super().save(model,optimizer,metadata)
        if self.state['global_training_step']==getattr(self,'resume_step',None)and not getattr(self,'initial_boundary_pinned',False):
            path=self.root/'motion_only_resume_boundaries'/f'g{self.resume_step}_{self.loaded_sha[:12]}.pt'
            path.parent.mkdir(exist_ok=True)
            if not path.exists():os.link(self.checkpoint,path)
            self.initial_boundary_pinned=True
            self.state['motion_only_initial_boundary_checkpoint']=str(path)
            self.publish()

    def publish(self):
        path=getattr(self.args,'cycle_status',None)
        if path is not None and path.is_file():
            self.state['dreamer_cycle']=json.loads(path.read_text(encoding='utf-8'))
        return super().publish()

    def frozen_tensors(self,model):
        return {name:value for name,value in model.state_dict().items()
            if name in self.frozen_names or name.startswith(FROZEN_PREFIXES)}

    def _data(self,path):
        return StagedSequenceData(path,self.device,mode='stream',cache_bytes=64*1024**2,
            image_size=32,episode_paths=sorted(Path(path).glob('*.npz')))

    def prepare(self):
        args=self.args
        torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        plan=json.loads(args.motion_plan.read_text(encoding='utf-8'))
        source=self.checkpoint if self.checkpoint.exists()else args.source_checkpoint
        loaded_sha=file_sha(source)
        self.loaded_sha=loaded_sha
        model,saved=load_world_checkpoint(source,'cpu')
        step=saved['training_steps']
        self.resume_step=step
        if saved['global_training_step']!=step or saved['run_state']['global_training_step']!=step:
            raise ValueError('Motion round requires a complete optimizer boundary')
        self.state=deepcopy(saved['run_state']);self.plan_sha=file_sha(args.motion_plan)
        self.plan=plan
        revision=bind_motion_only(self.state,plan,path=args.motion_plan,loaded_sha=loaded_sha)
        if args.round_corpus is not None:
            receipt=json.loads(args.round_corpus.read_text(encoding='utf-8'))
            if not any(row['sha256']==digest(receipt)for row in revision['corpus_revisions']):
                begin_motion_round(self.state,receipt)
        model.to(self.device).float()
        optimizer,self._optimizer_parameter_names,_=restore_optimizer(model,saved)
        self.freeze_policy=configure_motion_only_trainable(model)
        metadata={key:saved[key]for key in ('rssm_migration','source_teacher_checkpoint',
            'dual_rssm_migration','enemy_rssm_resize','rssm_fusion_migration','combat_rssm_migration',
            'rssm_five_migration','rssm_five_expanded_migration','rssm_spawn_conditioning_migration')if key in saved}
        self.model,self.optimizer,self.metadata=model,optimizer,metadata
        corpora=[row['receipt']for row in revision['corpus_revisions']]
        self.nav=MotionReplay(plan['navigation_directory'],revision['nav_sampler_state'],
            stream='navigation',device=self.device,corpora=corpora)
        self.enemy=MotionReplay(plan['enemy_motion_directory'],revision['enemy_sampler_state'],
            stream='enemy_motion',device=self.device,corpora=corpora)
        self.nav_val=self._data(args.navigation_data_root/'long/rgb/val')
        self.enemy_val=self._data(args.enemy_motion_data_root/'full_game/rgb/val')
        from .rssm_motion_only_validation import bind_motion_validation_plan
        previous_validation_plan=self.state.get('motion_only_validation_plan')
        self.validation_plan=bind_motion_validation_plan(args.navigation_data_root/'long/rgb/val',
            args.enemy_motion_data_root/'full_game/rgb/val',args.validation_plan,seed=args.seed)
        if previous_validation_plan is not None and any(previous_validation_plan[key]!=self.validation_plan[key]
                for key in('validation_plan_sha256','data_identity_sha256')):
            raise ValueError('Saved motion validation identity cannot be replaced on resume')
        self.state['motion_only_validation_plan']=self.validation_plan
        self.long_val=self.enemy_val;self.short_val=self.enemy_val
        # Keep the unchanged geometry/local objectives; the new main BPTT is H.
        parent=revision['source_rssm_training']
        self.player_settings=deepcopy(parent['player_navigation_revision']['settings'])
        self.motion_settings=deepcopy(parent['motion_correction_revision']['settings'])
        self.loss_protocol=PROTOCOL
        self.state.update(run_id=self.root.name,status='preparing',stage='rssm_motion_only_dynamics',
            model='Three RSSMs + Transformer MoE: navigation and quiet enemy motion',
            global_training_step=step,current=step,total=0,run_directory=str(self.root),
            quality_status='Six-second quality is measured on held-out Source trajectories',
            target_rollout_steps=180,prediction_every=10,training_environment='motion_only_navigation_quiet_enemy',
            renderer_training=dict(active=False,adversarial_active=False,codec_frozen=True,dynamics_active=True),
            keep_awake=dict(system_required=True,display_required=False),
            motion_only_freeze_policy=self.freeze_policy)
        self.state['training_protocol'].update(microbatch_size=16,gradient_accumulation_steps=1,
            effective_batch_size=16,combat_loss_enabled=False,auxiliary_fragments_per_update=0,
            combat_auxiliary_actors_per_microbatch=0,combat_late_actors_per_microbatch=0,
            validation_every=250,preview_every=10,observed_context_frames=1)
        self.state['rssm_training']['dynamics_trainable']=sum(p.numel()for p in model.parameters()if p.requires_grad)
        self.rng.bit_generator.state=deepcopy(saved['rng_state'])
        torch.set_rng_state(saved['torch_rng_state'].cpu())
        if self.device.type=='cuda':torch.cuda.set_rng_state_all([value.cpu()for value in saved['cuda_rng_state']])
        self.frozen_names=[name for name,p in model.named_parameters()if not p.requires_grad]
        self.frozen_sha=_hash_tensors(self.frozen_tensors(model))
        checks=dict(all_model_tensors_bitexact=_hash_tensors(model.state_dict())==_hash_tensors(saved['model']),
            named_Adam_order_preserved=self._optimizer_parameter_names==saved['optimizer_parameter_names'],
            Adam_group_count_preserved=len(optimizer.param_groups)==len(saved['optimizer']['param_groups']),
            Adam_state_count_preserved=len(optimizer.state)==len(saved['optimizer']['state']),
            global_Torch_RNG_bitexact=torch.equal(torch.get_rng_state(),saved['torch_rng_state'].cpu()),
            global_PCG_unchanged=self.rng.bit_generator.state==saved['rng_state'],
            no_architecture_migration=model.config.architecture=='dreamer_rssm')
        if not all(checks.values()):raise AssertionError('Motion-only activation changed the source boundary')
        atomic_json(self.root/'motion_only_resume_verification.json',dict(step=step,checks=checks,
            passed=True,checked_before_SGD=True,source_checkpoint_sha256=loaded_sha))
        self.publish()
        return saved

    def update(self,*,horizon=None,commit=True):
        revision=self.state['rssm_training']['motion_only_revision']
        selected=current_horizon(self.state)if horizon is None else horizon
        model=self.model;optimizer=self.optimizer
        # configure only once after canonical Adam restore; the mode does not
        # change requires_grad or the preserved optimizer groups.
        sample=revision['updates']%2==0
        model.core.train(sample);model.vjepa_backend.enemy_rssm.train(sample)
        model.vjepa_backend.combat_rssm.train(sample)
        configure_motion_only_trainable(model)
        nav,nav_receipt=self.nav.batch(horizon=selected)
        enemy,enemy_receipt=self.enemy.batch(horizon=selected)
        player=runtime_player_settings(self.player_settings,horizon=selected,plan_sha256=revision['sha256'])
        local_cycle=(4,8,16,32)
        local_horizon=min(local_cycle[revision['updates']%len(local_cycle)],selected)
        optimizer.zero_grad(set_to_none=True)
        start=time.perf_counter()
        last_progress=[0.]
        def progress(stream,processed,length):
            self.state['long_training_progress']=dict(step=self.state['global_training_step']+1,
                transitions_processed=processed,horizon=length,tbptt_chunk=selected,
                optimizer_updated=False,microbatch_size=16,effective_batch_size=16,
                completed_microbatches=0,microbatches_total=1,environment=stream)
            clock=time.perf_counter()
            if processed==length or clock-last_progress[0]>5:
                self.publish();last_progress[0]=clock
        metrics=motion_only_backward(model,nav,enemy,horizon=selected,context=1,observed_context=1,
            player_navigation_protocol=player,npc_navigation_protocol=self.motion_settings['npc_navigation'],
            player_local_protocol=self.motion_settings['player_local'],
            enemy_local_protocol=self.motion_settings['enemy_local'],
            player_local_horizon=local_horizon,enemy_local_horizon=local_horizon,
            progress_callback=progress)
        norm=clip_finite_grad_norm_(model.named_parameters(),1.,context='Motion-only MAIN16')
        optimizer.step()
        metrics.update(step=self.state['global_training_step']+1,gradient_norm=float(norm),
            objective_protocol=PROTOCOL,loss_protocol=PROTOCOL,motion_only_tbptt=selected,
            update_seconds=time.perf_counter()-start,motion_only_plan_sha256=revision['sha256'],
            navigation_sampling=nav_receipt,enemy_motion_sampling=enemy_receipt,
            optimizer_steps=1,latent_mode='sample'if sample else'argmax',
            main_actor_count=16,main_combat_episode_rng_draw_count=0)
        if commit:
            self.state['global_training_step']+=1
            commit_motion_only(self.state,step=self.state['global_training_step'],loss=metrics['train_loss'],
                nav_sampler_state=self.nav.state_dict(),enemy_sampler_state=self.enemy.state_dict(),metrics=metrics)
            gate=revision['gate']
            metrics.update(motion_only_round_index=gate['round_index'],motion_only_updates=revision['updates'],
                motion_only_stage_index=gate['stage_index'],motion_only_stage_updates=gate['stage_updates'],
                motion_only_round_updates=gate['updates'],motion_only_current_horizon=gate['current_horizon'])
            self.state.update(status='running',current=self.state['global_training_step'],
                loss=metrics['train_loss'],latest_metrics=metrics,loss_groups=metrics['loss_groups'])
            self.state['long_training_progress']['optimizer_updated']=True
            with(self.root/'metrics.jsonl').open('a',encoding='utf-8')as stream:
                stream.write(json.dumps(metrics,allow_nan=False)+'\n')
        return metrics

    def preview(self,model):
        # The image assets initialize in their unchanged full-game lineage.
        # Only the private inference clone gets the matching no-enemy profile.
        self.initialize_preview_assets(model)
        nav=self.state['rssm_training']['motion_only_revision']['updates']%20==10
        self.long_val=self.nav_val if nav else self.enemy_val
        with environment_profile(self.preview_model,navigation_only=nav):
            return super().preview(model)

    def validate(self,model):
        from .behavior_trajectory import evaluate_state_trajectories
        from .rssm_preview_assets import preserve_preview_rng
        from .rssm_motion_only_validation import verify_motion_validation_plan
        verify_motion_validation_plan(self.validation_plan)
        step=self.state['global_training_step'];directory=self.root/'validation'/f'motion_step_{step:06d}'
        self.state['validation_progress']=dict(status='running',step=step,horizon=180,
            scope='Held-out six-second navigation and quiet enemy trajectories')
        self.publish()
        with preserve_preview_rng(self.device,self.rng):
            with environment_profile(model,navigation_only=True):
                nav=evaluate_state_trajectories(model,self.nav_val,directory/'navigation',context=1,
                    horizon=180,count=8,batch_size=4,seed=self.args.seed,include_respawned_slots=False)
            with environment_profile(model,navigation_only=False):
                enemy=evaluate_state_trajectories(model,self.enemy_val,directory/'enemy_motion',context=1,
                    horizon=180,count=8,batch_size=4,seed=self.args.seed,include_respawned_slots=False)
        if(nav['fingerprint']!=self.validation_plan['navigation']['expected_fingerprint']
                or enemy['fingerprint']!=self.validation_plan['enemy_motion']['expected_fingerprint']):
            raise AssertionError('Motion validation evaluated a different held-out panel')
        report=dict(kind='rssm_motion_only_validation_v1',step=step,protocol=PROTOCOL,
            plan_sha256=self.plan_sha,target_horizon=180,navigation=nav,enemy_motion=enemy,
            validation_plan_sha256=self.validation_plan['validation_plan_sha256'],
            data_identity_sha256=self.validation_plan['data_identity_sha256'],
            source_held_out=True,source_future_feedback=False,renderer_evaluated=False,
            gameplay_qualified=False,budget_completion_is_quality_pass=False)
        atomic_json(directory/'motion_only_validation.json',report)
        atomic_json(self.root/'rssm_validation_latest.json',report)
        self.state.update(rssm_validation=report,validation_progress=dict(status='complete',step=step,horizon=180))
        self.publish()

    def run(self):
        self.prepare();model,optimizer,metadata=self.model,self.optimizer,self.metadata
        revision=self.state['rssm_training']['motion_only_revision']
        if self.args.probe_horizon is not None:
            before_frozen=self.frozen_sha
            metrics=self.update(horizon=self.args.probe_horizon,commit=False)
            after_frozen=_hash_tensors(self.frozen_tensors(model))
            atomic_json(self.root/'gpu_probe.json',dict(passed=before_frozen==after_frozen,
                discarded=True,nonresumable=True,actual_horizon=self.args.probe_horizon,
                actual_Adam_updates=1,frozen_images_and_pure_combat_readouts_bitexact=before_frozen==after_frozen,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3 if self.device.type=='cuda'else None,
                metrics=metrics))
            self.state.update(status='completed_discarded_probe',loss=metrics['train_loss'])
            self.publish();return
        if not revision['baseline_complete']:
            self.validate(model)
            mark_motion_only_baseline(self.state,step=self.state['global_training_step'])
        self.state['status']='running';self.save(model,optimizer,metadata)
        self.preview(model)
        began=revision['updates']
        while not revision['gate']['completed'] and(not self.args.updates or revision['updates']-began<self.args.updates):
            if(self.root/'stop_training').exists():
                self.state.update(status='stopped',message='Stopped at a complete motion optimizer boundary')
                self.save(model,optimizer,metadata);return
            metrics=self.update()
            self.save(model,optimizer,metadata)
            if revision['updates']%250==0 or revision['gate']['completed']:
                self.validate(model);self.save(model,optimizer,metadata)
            if revision['updates']%10==0:
                self.preview(model)
            if revision['updates']==1 or revision['updates']%30==0:
                frozen=_hash_tensors(self.frozen_tensors(model))
                if frozen!=self.frozen_sha:raise AssertionError('Motion-only training modified frozen image/combat task assets')
                self.state['motion_only_frozen_verification']=dict(step=self.state['global_training_step'],bitexact=True)
            self.publish()
        self.state.update(status='world_round_complete'if revision['gate']['completed']else'completed_update_budget',
            message='Motion round budget completed; held-out six-second quality is evaluated separately')
        self.save(model,optimizer,metadata)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=RUN)
    parser.add_argument('--source-checkpoint',type=Path,required=True)
    parser.add_argument('--motion-plan',type=Path,required=True)
    parser.add_argument('--round-corpus',type=Path)
    parser.add_argument('--cycle-status',type=Path)
    parser.add_argument('--validation-plan',type=Path,default=RUN/'motion_only_validation_plan.json')
    parser.add_argument('--updates',type=int,default=0)
    parser.add_argument('--probe-horizon',type=int,choices=HORIZONS)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--seed',type=int,default=173)
    parser.add_argument('--navigation-data-root',type=Path,default=ROOT/'training_runs/staged_single_20261110/data_navigation_recovery_v1')
    parser.add_argument('--enemy-motion-data-root',type=Path,default=ROOT/'training_runs/rssm_enemy_motion_20261113/data')
    parser.add_argument('--renderer-checkpoint',type=Path,default=ROOT/'training_runs/image_chain_20261114/handoffs/decoder/renderer.pt')
    parser.add_argument('--sr',type=Path,default=ROOT/'training_runs/sr_background_detail_20261114/frozen/sr_frozen_step_003000.pt')
    parser.add_argument('--preview-assets-plan',type=Path,default=ROOT/'training_runs/rssm25m_combat_20261111/preview_assets_batch128_g4398.json')
    parser.add_argument('--preview-assets-revision-plan',type=Path,default=RUN/'preview_assets_revision_npr32317_decoder4125_sr_detail3000_v5.json')
    args=parser.parse_args(argv)
    if args.updates<0:parser.error('updates must be nonnegative')
    args.batch=16
    if args.device=='cuda':torch.cuda.init()
    torch.set_num_threads(2)
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.resolve())
    args.run_dir.mkdir(parents=True,exist_ok=True)
    from .vqgan_pipeline import process_alive
    pid=args.run_dir/'training.pid'
    if pid.exists()and process_alive(int(pid.read_text())):parser.error('Dynamics worker is already active')
    pid.write_text(str(os.getpid()))
    trainer=MotionOnlyTraining(args)
    try:
        with KeepAwake():trainer.run()
    except BaseException as error:
        trainer.state.update(status='failed',error=f'{type(error).__name__}: {error}')
        trainer.publish();raise


if __name__=='__main__':main()
