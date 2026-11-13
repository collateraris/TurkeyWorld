"""Continue both RSSMs and sparse shared MoE on three tasks in every update.

The old phase ledgers are historical. Only the explicit MoE total-loss gate
advances; no legacy training loop, EMA normalizer or compulsory long update runs.
"""
from __future__ import annotations

import argparse
import copy
import gc
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
from .rssm_accumulation import accumulated_metrics, average_accumulated_gradients
from .rssm_migration import configure_rssm_training, FROZEN_PREFIXES
from .rssm_moe_protocol import (MOE_PROTOCOL, SETTINGS, bind_moe_training,
    moe_training_horizon, commit_moe_update)
from .rssm_moe_budget import curriculum_completion, stage_budget, loss_gate_settings
from .rssm_moe_batch_revision import (bind_moe_batch_revision, moe_batch_settings,
    microbatch_layout)
from .rssm_moe_training import (moe_dynamics_backward, summarize_moe_metrics,
    build_moe_combat_sampler, sample_moe_combat_batch)
from .staged_curriculum import StagedSequenceData
from .train_rssm_combat import RSSMCombatTraining
from .train_rssm_dynamics import verify_data, _digest
from .training_pipeline import atomic_json, now


def restore_optimizer(model, saved):
    """Restore named groups, including old mixed-age states and fresh MoE group."""
    names = saved['optimizer_parameter_names']
    policy = configure_rssm_training(model)
    parameters = dict(model.named_parameters())
    canonical = [n for n, p in model.named_parameters() if p.requires_grad]
    if names != canonical or not model.config.rssm_fusion_moe_enabled:
        raise ValueError('MoE continuation requires the complete canonical migrated optimizer')
    groups = [{'params': [parameters[names[i]] for i in group['params']]}
              for group in saved['optimizer']['param_groups']]
    optimizer = torch.optim.AdamW(groups, lr=1e-4, weight_decay=.01)
    optimizer.load_state_dict(saved['optimizer'])
    if any(group['lr'] != 1e-4 for group in optimizer.param_groups):
        raise ValueError('Retained Adam learning rate changed')
    return optimizer, names, policy


def merge_main_replay_metrics(combat,navigation,enemy,*,plan_sha256,
                              layout=None,auxiliary_batch_size=64,late_actor_count=8):
    """Count the actual actors; main, auxiliary and late streams are separate."""
    layout = layout or dict(combat=38, enemy_motion=16, navigation=10)
    c,e,n = (layout[key] for key in ('combat','enemy_motion','navigation'))
    batch_size = c+e+n
    row=copy.deepcopy(combat);row.update(navigation);row.update(enemy)
    row['moe_player_loss']=combat['moe_player_loss']+navigation['nav_player_loss']
    row['moe_enemy_loss']=combat['moe_enemy_loss']+enemy['enemy_motion_enemy_loss']
    for field in ('role','balance'):
        key='moe_router_'+field+'_loss'
        row[key]=combat[key]+navigation['nav_router_'+field+'_loss']+enemy['enemy_motion_router_'+field+'_loss']
    row['moe_router_loss']=row['moe_router_role_loss']+row['moe_router_balance_loss']
    row['train_loss']=sum(row[key]for key in ('moe_player_loss','moe_enemy_loss','moe_combat_loss','moe_router_loss'))
    row['train_loss_without_kl_floor']=row['train_loss']
    row.update(main_navigation_actor_count=n,main_enemy_motion_actor_count=e,main_combat_actor_count=c,
        main_actor_count=batch_size,main_discarded_combat_rng_count=batch_size-c,
        main_combat_episode_rng_draw_count=batch_size,
        main_navigation_fraction=n/batch_size,main_enemy_motion_fraction=e/batch_size,main_combat_fraction=c/batch_size,
        navigation_replay_plan_sha256=plan_sha256,combat_auxiliary_actor_count=auxiliary_batch_size,
        combat_late_actor_count=late_actor_count)
    row['loss_groups']=summarize_moe_metrics([row])
    return row


class RSSMMoETraining(RSSMCombatTraining):
    def _data(self, directory, cache_mb=32):
        return StagedSequenceData(directory, self.device, mode='stream', cache_bytes=cache_mb*1024**2)

    def finish_completed_curriculum(self, model, optimizer, metadata):
        """Save the terminal boundary before another forward/backward can start."""
        completion = curriculum_completion(self.moe)
        if completion is None:
            return False
        self.state.update(status=completion,
            message=('Final horizon update cap reached; training curriculum completed without a quality claim'
                     if completion == 'completed_budget' else
                     'Final horizon completed its qualifying total-loss streak; gameplay quality remains separate'),
            quality_status='Long gameplay quality is determined by independent Source validation')
        self.save(model, optimizer, metadata)
        return True

    def run(self):
        args = self.args
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if self.device.type == 'cuda':
            torch.cuda.init()
        checkpoint = self.checkpoint if self.checkpoint.exists() else args.source_checkpoint
        loaded_sha = _digest(checkpoint)
        model, saved = load_world_checkpoint(checkpoint, 'cpu')
        if saved.get('training_steps') != saved['run_state']['global_training_step']:
            raise ValueError('Partial optimizer boundary')
        self.state = copy.deepcopy(saved['run_state'])
        combat_memory_enabled = getattr(model.config, 'combat_rssm_enabled', False)
        self.state.update(run_id=self.root.name, status='preparing', current=saved['training_steps'],
            total=args.updates, run_directory=str(self.root),
            model=('Three RSSMs + shared Transformer MoE: Player / Enemy / Combat'
                   if combat_memory_enabled else 'Dual RSSM + shared Transformer MoE: Player / Enemy / Combat'))
        five_enabled = getattr(model.config, 'rssm_five_role_enabled', False)
        five_diagnostics = None
        if five_enabled:
            from .rssm_five_migration import FiveRoleDiagnostics, report_sha
            from .rssm_five_role_fusion import five_role_metadata
            five_report = saved['rssm_five_migration']
            if five_report.get('revision_sha256') != report_sha(five_report):
                raise ValueError('Five-role architecture migration pin changed')
            self.state['rssm_five_training'] = {**five_role_metadata(model.vjepa_backend.five_role_fusion),
                'enabled': True, 'migration_sha256': five_report['revision_sha256'],
                'source_step': five_report['source_step'], 'current_step': saved['training_steps']}
            self.state['model'] = 'Three RSSMs + pretrained shared trunk + five private Transformer role experts'
            pin = dict(protocol=self.state['rssm_five_training']['protocol'],
                migration_sha256=five_report['revision_sha256'], source_step=five_report['source_step'])
            if self.state.get('training_protocol', {}).get('rssm_five_role') != pin:
                raise ValueError('Five-role status/training protocol pin is missing or changed')
            five_diagnostics = FiveRoleDiagnostics()
            model.vjepa_backend.five_role_fusion.register_forward_hook(five_diagnostics)
        self.plan_sha = _digest(args.moe_plan)
        plan = json.loads(args.moe_plan.read_text(encoding='utf-8'))
        self.moe = bind_moe_training(self.state, plan, path=args.moe_plan,
            sha256=self.plan_sha, loaded_checkpoint_sha256=loaded_sha)
        self.batch_revision = None
        batch_path = getattr(args, 'batch_revision_plan', None)
        if batch_path is not None:
            self.batch_revision = bind_moe_batch_revision(self.state,
                json.loads(batch_path.read_text(encoding='utf-8')), path=batch_path,
                sha256=_digest(batch_path), loaded_checkpoint_sha256=loaded_sha)
        elif self.moe.get('batch_revision') is not None:
            raise ValueError('Saved MoE microbatch revision requires its immutable plan')
        batch_settings = moe_batch_settings(self.state)
        self.navigation_replay_revision=None
        replay_path=getattr(args,'navigation_replay_plan',None)
        if replay_path is not None:
            from .rssm_navigation_replay_protocol import bind_navigation_replay
            self.navigation_replay_plan=json.loads(replay_path.read_text(encoding='utf-8'))
            self.navigation_replay_revision=bind_navigation_replay(self.state,self.navigation_replay_plan,
                path=replay_path,sha256=_digest(replay_path),loaded_checkpoint_sha256=loaded_sha)
        elif self.state['rssm_training'].get('navigation_replay_revision')is not None:
            raise ValueError('Saved three-way main replay requires its immutable plan')
        budget_path=getattr(args,'stage_budget_plan',None)
        if budget_path is not None:
            from .rssm_moe_budget import bind_budget_policy
            self.stage_budget_policy=bind_budget_policy(self.state,
                json.loads(budget_path.read_text(encoding='utf-8')),path=budget_path,
                sha256=_digest(budget_path),loaded_checkpoint_sha256=loaded_sha)
        elif self.moe.get('advance_policy') is not None:
            raise ValueError('Saved stage-budget policy requires its immutable plan')
        else:self.stage_budget_policy=None
        foundation = Path(plan['source_checkpoint'])
        migration_path = foundation.with_suffix(foundation.suffix+'.fusion.json')
        if _digest(migration_path) != plan['architecture_migration_sha256']:
            raise ValueError('Pinned shared-MoE migration proof changed')
        self.combat_memory_revision = None
        if combat_memory_enabled:
            memory_plan_path = getattr(args, 'combat_memory_plan', None)
            if memory_plan_path is None:
                raise ValueError('Three-RSSM training requires its pinned combat-memory continuation plan')
            from .rssm_combat_memory_protocol import bind_combat_memory
            memory_plan = json.loads(memory_plan_path.read_text(encoding='utf-8'))
            self.combat_memory_revision = bind_combat_memory(self.state, memory_plan,
                path=memory_plan_path, sha256=_digest(memory_plan_path),
                loaded_checkpoint_sha256=loaded_sha)
            if self.combat_memory_revision['updates'] == 0:
                self.state.setdefault('combat_memory_baseline', {
                    'step': self.state['global_training_step'], 'loss': self.state.get('loss'),
                    'latest_metrics': copy.deepcopy(self.state.get('latest_metrics'))})
                self.state.update(loss=None, latest_metrics=None, loss_groups=None)
        elif getattr(args, 'combat_memory_plan', None) is not None:
            raise ValueError('Combat-memory plan cannot be applied to a two-RSSM checkpoint')
        self.respawn_timing_revision = None
        timing_path = getattr(args, 'respawn_timing_plan', None)
        if timing_path is not None:
            from .rssm_respawn_timing_protocol import bind_respawn_timing
            timing_plan = json.loads(timing_path.read_text(encoding='utf-8'))
            self.respawn_timing_revision = bind_respawn_timing(self.state, timing_plan,
                path=timing_path, sha256=_digest(timing_path), loaded_checkpoint_sha256=loaded_sha)
            if self.respawn_timing_revision['updates'] == 0:
                self.state.setdefault('respawn_timing_baseline', {
                    'step': self.state['global_training_step'], 'loss': self.state.get('loss'),
                    'latest_metrics': copy.deepcopy(self.state.get('latest_metrics'))})
                self.state.update(loss=None, latest_metrics=None, loss_groups=None)
        elif self.state['rssm_training'].get('respawn_timing_revision') is not None:
            raise ValueError('Saved respawn-clock revision requires its immutable timing plan')
        if self.moe['updates'] == 0:
            if self.state.get('rssm_validation') is not None:
                self.state.setdefault('validation_history', []).append(copy.deepcopy(self.state['rssm_validation']))
            self.state.pop('rssm_validation', None)
            self.state['rssm_training']['quality_streak'] = 0
        self.loss_protocol = MOE_PROTOCOL
        # Inherited save/validation deliberately see no active legacy revisions.
        self.loss_balancer = self.recovery_sampler = self.recovery_revision = None
        self.coordinate_curriculum = self.replay_revision = self.drift_only_revision = None
        self.loss_gated_curriculum = None
        self.feedback_mode = 'detached_with_integrated_coordinate_gradients'
        self.microbatch = batch_settings['microbatch_size']
        self.accumulation = batch_settings['gradient_accumulation_steps']
        effective_batch = batch_settings['effective_batch_size']
        auxiliary_batch_size = batch_settings['combat_auxiliary_actors_per_microbatch']
        late_actor_count = batch_settings['combat_late_actors_per_microbatch']
        model.to(self.device).float()
        optimizer, self._optimizer_parameter_names, policy = restore_optimizer(model, saved)
        metadata = {k: saved[k] for k in ('rssm_migration', 'source_teacher_checkpoint',
            'dual_rssm_migration', 'enemy_rssm_resize', 'rssm_fusion_migration', 'combat_rssm_migration',
            'rssm_five_migration') if k in saved}
        self._drift_optimizer, self._drift_metadata = optimizer, metadata
        roots = {'combat': args.combat_data_root, 'navigation': args.navigation_data_root,
                 'recovery': args.recovery_data_root}
        if plan['data_roots'] != {k: str(p) for k, p in roots.items()}:
            raise ValueError('CLI Source roots differ from the immutable MoE phase plan')
        fingerprints = {k: {'root': str(p), 'manifest_sha256': verify_data(p)} for k, p in roots.items()}
        if plan['source_identities'] != {k: v['manifest_sha256'] for k, v in fingerprints.items()}:
            raise ValueError('Pinned MoE Source manifest identity changed')
        if self.moe.get('data_fingerprints', fingerprints) != fingerprints:
            raise ValueError('MoE Source corpus changed after phase initialization')
        self.moe['data_fingerprints'] = fingerprints
        self.short = self._data(args.recovery_data_root/'moving1/rgb/train', 64)
        self.short_val = self._data(args.combat_data_root/'moving1/rgb/val')
        self.long_val = self._data(args.combat_data_root/'full_game/rgb/val')
        combat_long = self._data(args.combat_data_root/'full_game/rgb/train', 64)
        navigation_sampler=enemy_motion_sampler=None
        if self.navigation_replay_revision is not None:
            from .rssm_navigation_replay_protocol import NavigationReplaySampler,EnemyMotionReplaySampler
            if (self.navigation_replay_plan['navigation_root']!=str(args.navigation_data_root)
                    or getattr(args,'enemy_motion_data_root',None)is None
                    or self.navigation_replay_plan['enemy_motion_root']!=str(args.enemy_motion_data_root)
                    or verify_data(args.enemy_motion_data_root)!=self.navigation_replay_plan['enemy_motion_manifest_sha256']):
                raise ValueError('Three-way replay Source roots or manifest changed')
            self.navigation_long=self._data(args.navigation_data_root/'long/rgb/train',64)
            self.enemy_motion_long=self._data(args.enemy_motion_data_root/'full_game/rgb/train',64)
            navigation_sampler=NavigationReplaySampler(self.navigation_long,self.navigation_replay_revision,
                batch_settings=batch_settings)
            enemy_motion_sampler=EnemyMotionReplaySampler(self.enemy_motion_long,self.navigation_replay_revision,
                batch_settings=batch_settings)
        history = self.state['navigation_training_history']
        contact_root = Path(history['contact_revision']['dataset_directory'])
        self.contact_val = self._data(contact_root/'short/rgb/val')
        samplers = {}
        sampler_states = self.moe.setdefault('samplers', {})
        sampler_identities = self.moe.setdefault('sampler_identities', {})
        for horizon in (4, 8, 16):
            sampler = build_moe_combat_sampler(self.short, horizon=horizon,
                cache_path=self.root/f'combat_sampling_h{horizon}.json')
            identity = copy.deepcopy(sampler.moe_identity)
            if sampler_identities.get(str(horizon), identity) != identity:
                raise ValueError('Saved MoE event sampler horizon or Source identity changed')
            if str(horizon) in sampler_states and str(horizon) not in sampler_identities:
                raise ValueError('Saved MoE event sampler is missing its horizon identity')
            sampler_identities[str(horizon)] = identity
            if str(horizon) in sampler_states:
                sampler.restore_state(sampler_states[str(horizon)])
            samplers[horizon] = sampler
        # Restore the exact previous boundary after constructors and indexing.
        self.rng.bit_generator.state = saved['rng_state']
        torch.set_rng_state(saved['torch_rng_state'].cpu())
        if self.device.type == 'cuda':
            torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng_state']])
        from .rssm_navigation_protocol import verify_restored_boundary
        proof = verify_restored_boundary(model, optimizer, self.rng, saved)
        atomic_json(self.root/'resume_verification.json', proof)
        frozen = _hash_tensors({n: v for n, v in model.state_dict().items() if n.startswith(FROZEN_PREFIXES)})
        self.state.update(status='running', stage='rssm_combat_dynamics',
            quality_status='MoE multitask phase started; long gameplay quality not qualified',
            training_schedule=('simultaneous_combat_with_no_enemy_navigation_main_replay'
                if self.navigation_replay_revision is not None else 'simultaneous_player_enemy_combat_moe'),
            keep_awake={'system_required': True, 'display_required': False},
            prediction_every=10, target_rollout_steps=450, started_at=now())
        self.state['training_protocol'].update(SETTINGS)
        self.state['training_protocol'].update(microbatch_size=self.microbatch,
            gradient_accumulation_steps=self.accumulation, effective_batch_size=effective_batch,
            auxiliary_fragments_per_update=auxiliary_batch_size*self.accumulation,
            late_own_prefix_actors=late_actor_count)
        self.state['rssm_training'].update(phase='combat',
            phase_label='MoE: навигация игрока, движение врага и бой',
            dynamics_trainable=policy['dynamics_trainable'], validation_every=250,
            preview_every=10, renderer_frozen=True)
        del saved
        gc.collect()
        self.preview(model)
        self.save(model, optimizer, metadata)
        started_updates = self.moe['updates']
        while not args.updates or self.moe['updates'] - started_updates < args.updates:
            if self.finish_completed_curriculum(model, optimizer, metadata):
                return
            if (self.root/'stop_training').exists():
                self.state.update(status='stopped', message='Stopped at a complete optimizer boundary')
                self.save(model, optimizer, metadata)
                return
            horizon = moe_training_horizon(self.state)
            step = self.state['global_training_step'] + 1
            context = 1 + (self.moe['updates']//16) % 2
            mode = SETTINGS['latent_cycle'][self.moe['updates'] % len(SETTINGS['latent_cycle'])]
            configure_rssm_training(model)
            model.core.train(mode == 'sample')
            model.vjepa_backend.enemy_rssm.train(mode == 'sample')
            if combat_memory_enabled:
                model.vjepa_backend.combat_rssm.train(mode == 'sample')
            optimizer.zero_grad(set_to_none=True)
            if five_diagnostics is not None: five_diagnostics.begin()
            rows, samples = [], []
            update_started = time.perf_counter()
            for micro in range(self.accumulation):
                layout = microbatch_layout(self.state, micro)
                draws = layout['combat_rng_draws']
                pairs = np.column_stack((self.rng.integers(len(combat_long.episodes), size=draws), np.zeros(draws, np.int64)))
                combat_pairs=pairs if navigation_sampler is None else pairs[:layout['combat']]
                batch = combat_long.batch(len(combat_pairs), context=context, horizon=horizon, indices=combat_pairs)
                auxiliary_horizon = min(horizon, 16)
                sampler = samplers[auxiliary_horizon]
                auxiliary, receipt = sample_moe_combat_batch(sampler, self.short, self.rng,
                    horizon=horizon, batch_size=auxiliary_batch_size)
                samples.append(receipt)
                def progress(processed, length):
                    self.state['long_training_progress'] = dict(step=step,
                        transitions_processed=processed, horizon=length, tbptt_chunk=16,
                        optimizer_updated=False, microbatch_index=micro+1, microbatches_total=self.accumulation,
                        completed_microbatches=micro, microbatch_size=self.microbatch, effective_batch_size=effective_batch,
                        environment='combat', source_future_feedback=False)
                    self.publish()
                row = moe_dynamics_backward(model, batch, auxiliary,
                    context=context, observed_context=context, expected_horizon=horizon,
                    progress_callback=progress,
                    auxiliary_actor_limit=auxiliary_batch_size, late_own_prefix_actors=late_actor_count,
                    **({'main_player_weight':layout['main_player_weight'],
                        'main_router_weight':layout['main_router_weight'],
                        'main_enemy_weight':layout['main_enemy_weight']}
                       if navigation_sampler is not None else {}))
                if navigation_sampler is not None:
                    from .rssm_moe_navigation_objective import navigation_replay_backward
                    from .rssm_moe_enemy_motion import enemy_motion_backward
                    navigation_batch,navigation_receipt=navigation_sampler.batch(context=context,horizon=horizon,
                        actors=layout['navigation'])
                    enemy_batch,enemy_receipt=enemy_motion_sampler.batch(context=context,horizon=horizon,
                        actors=layout['enemy_motion'])
                    nav=navigation_replay_backward(model,navigation_batch,context=context,observed_context=context,
                        expected_horizon=horizon,weight=layout['navigation_player_router_weight'])
                    motion=enemy_motion_backward(model,enemy_batch,context=context,observed_context=context,
                        expected_horizon=horizon,enemy_weight=layout['enemy_motion_enemy_weight'],
                        router_weight=layout['enemy_motion_router_weight'])
                    row=merge_main_replay_metrics(row,nav,motion,
                        plan_sha256=self.navigation_replay_revision['sha256'], layout=layout,
                        auxiliary_batch_size=auxiliary_batch_size, late_actor_count=late_actor_count)
                    row.update(navigation_sampling=navigation_receipt,enemy_motion_sampling=enemy_receipt,
                        main_combat_pairs=combat_pairs.tolist(),discarded_combat_rng_pairs=pairs[layout['combat']:].tolist())
                    del navigation_batch,enemy_batch
                rows.append(row)
                del batch, auxiliary
            average_accumulated_gradients(model.parameters(), self.accumulation)
            metrics = accumulated_metrics(rows)
            metrics['loss_groups'] = summarize_moe_metrics(rows, step=step)
            metrics['train_loss'] = metrics['loss_groups']['total']
            for name in ('player', 'enemy', 'combat'):
                metrics[name+'_loss'] = metrics['loss_groups'][name]['weighted']
            metrics['router_role_loss'] = metrics['moe_router_role_loss']
            metrics['router_balance_loss'] = metrics['moe_router_balance_loss']
            if five_diagnostics is not None:
                metrics.update(rssm_five_role_protocol=self.state['rssm_five_training']['protocol'],
                    rssm_five_migration_sha256=five_report['revision_sha256'],
                    rssm_five_role_fusion=five_diagnostics.summary())
            if self.batch_revision is not None:
                metrics['moe_batch_revision_plan_sha256'] = self.batch_revision['sha256']
            json.dumps(metrics, allow_nan=False)
            architecture_updates = self.combat_memory_revision['updates'] if self.combat_memory_revision else self.moe['updates']
            diagnostic_updates = min(architecture_updates, self.respawn_timing_revision['updates']) if self.respawn_timing_revision else architecture_updates
            if five_enabled:
                diagnostic_updates = min(diagnostic_updates, step - five_report['source_step'] - 1)
            if diagnostic_updates < 3:
                gradient_groups = {}
                for label, prefix in {
                    'player_rssm': 'vjepa_backend.core.',
                    'enemy_rssm': 'vjepa_backend.enemy_rssm.',
                    'player_head': 'vjepa_backend.fusion_moe.player_head.',
                    'enemy_head': 'vjepa_backend.fusion_moe.enemy_head.',
                    'combat_head': 'vjepa_backend.fusion_moe.combat_head.',
                    'fusion_transformer': 'vjepa_backend.fusion_moe.blocks.',
                    'fusion_experts': 'vjepa_backend.fusion_moe.experts.',
                    **({'combat_rssm': 'vjepa_backend.combat_rssm.',
                        'combat_rssm_core': 'vjepa_backend.combat_rssm.core.',
                        'combat_rssm_task_head': 'vjepa_backend.combat_rssm.task_head.',
                        'combat_rssm_fusion_adapter': 'vjepa_backend.combat_rssm.fusion_adapter.'} if combat_memory_enabled else {}),
                    **({'five_role_experts': 'vjepa_backend.five_role_fusion.experts.',
                        'five_role_router': 'vjepa_backend.five_role_fusion.router.',
                        'five_player_heads': 'vjepa_backend.five_role_fusion.player_heads.',
                        'five_enemy_head': 'vjepa_backend.five_role_fusion.enemy_head.',
                        'five_combat_head': 'vjepa_backend.five_role_fusion.combat_head.'} if five_enabled else {}),
                }.items():
                    gradients = [p.grad.detach().float().square().sum()
                        for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
                    gradient_groups[label] = float(torch.stack(gradients).sum().sqrt()) if gradients else 0.
                metrics['first_update_gradient_norms'] = gradient_groups
            metrics['world_grad_norm'] = float(clip_finite_grad_norm_(model.named_parameters(), context=f'MoE update {step}'))
            optimizer.step()
            self.state['global_training_step'] = step
            if five_enabled: self.state['rssm_five_training']['current_step'] = step
            gate = commit_moe_update(self.state, step=step, horizon=horizon, train_loss=metrics['train_loss'],
                microbatches=self.accumulation,
                microbatch_size=self.microbatch, effective_batch_size=effective_batch,
                loss_gate_eligible=metrics['moe_direct_support_gate_eligible'])
            if self.combat_memory_revision is not None:
                from .rssm_combat_memory_protocol import commit_combat_memory
                commit_combat_memory(self.state, step=step)
                metrics.update(combat_rssm_updates=self.combat_memory_revision['updates'],
                    combat_memory_plan_sha256=self.combat_memory_revision['sha256'],
                    combat_rssm_enabled=True, combat_rssm_memory_size=model.vjepa_backend.combat_rssm.memory_size)
            if self.respawn_timing_revision is not None:
                from .rssm_respawn_timing_protocol import commit_respawn_timing
                timing = commit_respawn_timing(self.state, step=step)
                metrics.update(respawn_clock_protocol=timing['timer_protocol'],
                    respawn_timing_plan_sha256=timing['sha256'], respawn_timing_updates=timing['updates'])
            if navigation_sampler is not None:
                from .rssm_navigation_replay_protocol import commit_navigation_replay
                commit_navigation_replay(self.state,step=step,sampler_state=navigation_sampler.state_dict(),
                    enemy_sampler_state=enemy_motion_sampler.state_dict(),micro_rows=rows)
                metrics.update(navigation_replay_plan_sha256=self.navigation_replay_revision['sha256'],
                    navigation_replay_updates=self.navigation_replay_revision['updates'])
            sampler_states[str(auxiliary_horizon)] = sampler.state_dict()
            metrics.update(step=step, global_training_step=step, timestamp=now(), main_horizon=horizon,
                moe_training_plan_sha256=self.plan_sha,
                current_horizon=gate['current_horizon'], moe_training_updates=gate['updates'],
                moe_gate_streak=gate['consecutive_below_threshold'], moe_stage_advanced=gate['last_stage_advanced'],
                moe_gate_threshold=loss_gate_settings(gate)[0],
                moe_gate_required_consecutive_passes=loss_gate_settings(gate)[1],
                **({'stage_budget_plan_sha256':self.stage_budget_policy['sha256'],
                    'stage_budget':stage_budget(gate['stage_index'], self.stage_budget_policy),
                    'completed_stage_budget':stage_budget(gate['stage_index']-int(gate['last_stage_advanced']), self.stage_budget_policy),
                    'stage_advance_reason':gate.get('last_advance_reason'),
                    'stage_budget_policy_updates':self.stage_budget_policy['updates'],
                    'stage_budget_exhausted':gate.get('budget_exhausted', False),
                    'moe_curriculum_completed':gate.get('curriculum_completed', False)}if self.stage_budget_policy else{}),
                rssm_phase='combat', component_phase='combat', stage='rssm_combat_dynamics',
                microbatch_size=self.microbatch, gradient_accumulation_steps=self.accumulation,
                effective_batch_size=effective_batch, completed_microbatches=self.accumulation,
                training_environment=('mixed_main_combat_enemy_motion_navigation'
                    if self.navigation_replay_revision is not None else 'combat'),
                rssm_latent_training_mode=mode, image_objectives_active=False,
                dynamics_updated=True, combat_sampling=samples,
                optimizer_update_seconds=time.perf_counter()-update_started)
            self.state.update(current=step, loss=metrics['train_loss'], latest_metrics=metrics,
                loss_groups=metrics['loss_groups'], status='running',
                message='Simultaneous player navigation, enemy navigation and combat')
            self.state['long_training_progress'].update(optimizer_updated=True, completed_microbatches=self.accumulation)
            with (self.root/'metrics.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(metrics, ensure_ascii=False, allow_nan=False)+'\n')
            early_architecture = self.combat_memory_revision is not None and self.combat_memory_revision['updates'] <= 5
            early_timing = self.respawn_timing_revision is not None and self.respawn_timing_revision['updates'] <= 5
            early_budget = self.stage_budget_policy is not None and self.stage_budget_policy['updates'] <= 5
            early_replay=self.navigation_replay_revision is not None and self.navigation_replay_revision['updates']<=5
            # A later forward/backward failure must not lose completed Adam
            # updates between the ten-step image previews. Atomic replacement
            # keeps one full resumable boundary, not one file per update.
            self.save(model, optimizer, metadata)
            if gate['updates'] % 250 == 0:
                self.validate(model)
                self.save(model, optimizer, metadata)
            if step % 10 == 0 or gate['last_stage_advanced']:
                self.preview(model)
            if gate['updates'] == 1 or early_architecture or early_timing or gate['updates'] % 250 == 0:
                current_frozen = _hash_tensors({n: v for n, v in model.state_dict().items() if n.startswith(FROZEN_PREFIXES)})
                if current_frozen != frozen:
                    raise AssertionError('Frozen renderer/map/codec assets changed')
                self.state['frozen_verification'] = dict(step=step, all_assets_bitexact=True, fingerprint=frozen)
            self.publish()
            if self.finish_completed_curriculum(model, optimizer, metadata):
                return
        self.state.update(status='completed', message='Requested MoE update budget completed')
        self.save(model, optimizer, metadata)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'source-checkpoint', 'moe-plan', 'combat-data-root',
                 'navigation-data-root', 'recovery-data-root', 'renderer-checkpoint', 'sr',
                 'preview-assets-plan', 'preview-assets-revision-plan'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--combat-memory-plan', type=Path)
    parser.add_argument('--respawn-timing-plan', type=Path)
    parser.add_argument('--stage-budget-plan', type=Path)
    parser.add_argument('--batch-revision-plan', type=Path)
    parser.add_argument('--navigation-replay-plan',type=Path)
    parser.add_argument('--enemy-motion-data-root',type=Path)
    parser.add_argument('--seed', type=int, default=173)
    parser.add_argument('--updates', type=int, default=0)
    args = parser.parse_args(argv)
    if bool(args.navigation_replay_plan)!=bool(args.enemy_motion_data_root):
        parser.error('Three-way main replay requires both the pinned plan and enemy-motion data root')
    if args.updates < 0:
        parser.error('updates must be nonnegative; zero means continuous training')
    for key, value in vars(args).items():
        if isinstance(value, Path): setattr(args, key, value.resolve())
    if args.batch_revision_plan is None:
        args.batch, args.accumulation_steps = 64, 4
    else:
        from .rssm_moe_batch_revision import batch_settings_for_plan
        dimensions=batch_settings_for_plan(json.loads(args.batch_revision_plan.read_text(encoding='utf-8')))
        args.batch=dimensions['microbatch_size']
        args.accumulation_steps=dimensions['gradient_accumulation_steps']
    from .vqgan_pipeline import process_alive
    args.run_dir.mkdir(parents=True, exist_ok=True)
    pid = args.run_dir/'training.pid'
    if pid.exists() and process_alive(int(pid.read_text())):
        parser.error('MoE worker already active')
    atomic_json(pid, os.getpid())
    experiment = RSSMMoETraining(args)
    try:
        with KeepAwake(): experiment.run()
    except BaseException as error:
        experiment.state.update(status='failed', error=f'{type(error).__name__}: {error}')
        experiment.publish()
        raise


if __name__ == '__main__':
    main()
