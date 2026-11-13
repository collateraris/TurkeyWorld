"""Continue the saved RSSM/Adam on one-enemy Source combat plus navigation replay."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .distill_renderer import _save, _digest
from .gradient_safety import clip_finite_grad_norm_
from .models import load_world_checkpoint, STATE_SCHEMA
from .power_guard import KeepAwake
from .renderer_student import _hash_tensors
from .rssm_combat_protocol import (PROTOCOL, CombatTrainingSampler, environment_profile,
    prepare_combat_continuation, commit_combat_update)
from .rssm_migration import configure_rssm_training
from .rssm_objectives import rssm_dynamics_backward
from .staged_curriculum import StagedSequenceData
from .staged_validation import _isolated_validation
from .train_rssm_dynamics import RSSMTraining, verify_data, episode_names
from .training_pipeline import atomic_json, now


def combat_quality_gate(heads, trajectory):
    """Positive denominators are mandatory; missing cohorts never qualify."""
    failures, summary = [], {}
    cohorts = heads['cohorts']
    for cohort, metric, minimum, bound in (
        ('hit', 'hit_target_accuracy', True, .8), ('miss', 'miss_target_none_accuracy', True, .8),
        ('spawn', 'spawn_decoded_top1_tile_accuracy', True, .8),
        ('spawn', 'spawn_decoded_xy_error_tiles', False, .5)):
        measured = cohorts.get(cohort, {}).get('metrics', {}).get(metric, {})
        value = measured.get('mean'); summary[metric] = value
        if (measured.get('denominator', 0) < 8 or value is None
                or (value < bound if minimum else value > bound)): failures.append(metric)
    attack = cohorts.get('attack', {}).get('binary', {}).get('attack_accepted', {})
    for name in ('precision', 'recall'):
        summary['attack_'+name] = attack.get(name)
        if attack.get('source_positives', 0) < 8 or attack.get(name) is None or attack[name] < .8:
            failures.append('attack_'+name)
    death = cohorts.get('death', {}).get('binary', {}).get('npc_death_accepted_on_killable', {})
    if death.get('source_positives', 0) < 8 or death.get('recall') is None or death['recall'] < .8:
        failures.append('npc_death_recall')
    summary['npc_death_recall'] = death.get('recall')
    errors, npc_errors = [], []
    collision_steps = life_mismatches = 0
    for row in trajectory['offsets'].values():
        errors.append(row['player_xy_error_tiles']['mean'])
        if row['npc_xy_error_tiles']['mean'] is not None: npc_errors.append(row['npc_xy_error_tiles']['mean'])
        collision_steps += row['player_wall_collisions']+row['living_npc_wall_collisions']
        life_mismatches += row['npc_alive_false_negative']+row['npc_alive_false_positive']+row['player_alive_mismatches']
    summary.update(player_xy_mean=float(np.mean(errors)), npc_xy_mean=float(np.mean(npc_errors)) if npc_errors else None,
        wall_collision_steps=collision_steps, life_mismatches=life_mismatches)
    final = trajectory['offsets']['450']
    for who, maximum, p95_maximum in (('player', .1, .25), ('npc', .25, .5)):
        measured = final[who+'_xy_error_tiles']
        summary[who+'_final_xy_mean'] = measured['mean']
        summary[who+'_final_xy_p95'] = measured['p95']
        if measured['mean'] is None or measured['mean'] > maximum:
            failures.append(who+'_final_xy_mean')
        if measured['p95'] is None or measured['p95'] > p95_maximum:
            failures.append(who+'_final_xy_p95')
    if (not trajectory['all_states_finite'] or not trajectory['all_memory_finite_and_shape_valid']
            or trajectory['horizon'] != 450 or trajectory['count'] < 16): failures.append('finite_own450_coverage')
    if summary['player_xy_mean'] > .1: failures.append('player_xy_mean')
    if not npc_errors or summary['npc_xy_mean'] > .25: failures.append('npc_xy_mean')
    if collision_steps: failures.append('wall_collision_steps')
    if life_mismatches: failures.append('life_mismatches')
    return {'passed': not failures, 'failure_reasons': failures, 'summary': summary,
        'thresholds': {'head_positive_min': 8, 'hit_miss_attack_death_spawn_accuracy_min': .8,
            'spawn_xy_max_tiles': .5, 'player_xy_mean_max_tiles': .1,
            'npc_xy_mean_max_tiles': .25, 'wall_collision_steps_max': 0, 'life_mismatches_max': 0}}


class RSSMCombatTraining(RSSMTraining):
    def __init__(self, args):
        args.distill_updates = 200; args.teacher = args.source_checkpoint
        args.initial = args.source_checkpoint; args.follow_active_renderer = False
        super().__init__(args)
        self.started = time.perf_counter()

    def save(self, model, optimizer, metadata):
        if getattr(self, 'loss_balancer', None) is not None:
            self.state['rssm_training']['loss_balance_state'] = self.loss_balancer.state_dict()
        if getattr(self, 'recovery_sampler', None) is not None:
            self.state['rssm_training']['combat_recovery_revision']['sampler_state'] = self.recovery_sampler.state_dict()
        names = getattr(self, '_optimizer_parameter_names', None)
        if names is None:
            names = [name for name, value in model.named_parameters() if value.requires_grad]
        payload = {key: metadata[key] for key in ('rssm_migration', 'source_teacher_checkpoint','dual_rssm_migration',
            'enemy_rssm_resize', 'rssm_fusion_migration', 'combat_rssm_migration', 'rssm_five_migration',
            'rssm_five_expanded_migration') if key in metadata}
        payload.update(stage='world', state_schema=dict(STATE_SCHEMA), config=asdict(model.config),
            model=model.state_dict(), optimizer=optimizer.state_dict(), optimizer_parameter_names=names,
            training_steps=self.state['global_training_step'], global_training_step=self.state['global_training_step'],
            run_state=self.state, rssm_training=self.state['rssm_training'],
            rssm_combat_migration=self.state['combat_migration'], plan_sha256=self.plan_sha,
            rng_state=self.rng.bit_generator.state, torch_rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else [],
            quality_status=self.state['quality_status'])
        if model.config.navigation_only: raise AssertionError('Saved combat runtime must enable one-enemy respawn')
        _save(self.checkpoint, payload); self.state['checkpoint'] = str(self.checkpoint); self.publish()

    def validate(self, model):
        from .motion_recovery_validation import evaluate_motion_recovery, _short
        from .rssm_continuation import evaluate_heldout_idle, evaluate_one_frame450, complete_quality_gate
        from .behavior_heads import evaluate_heads
        from .behavior_trajectory import evaluate_state_trajectories
        step = self.state['global_training_step']; directory = self.root/'validation'/f'step_{step:06d}'
        self.state['validation_progress'] = {'status': 'running', 'step': step,
            'scope': 'Unchanged navigation gates plus held-out one-enemy heads and own450'}
        self.publish()
        with _isolated_validation(model, self.short_val, self.args.seed):
            with environment_profile(model, navigation_only=True):
                navigation = evaluate_motion_recovery(model, self.args.navigation_data_root/'short/rgb/val',
                    self.args.navigation_data_root/'long/rgb/val', directory/'navigation',
                    default_regression_data=self.args.navigation_data_root/'short/rgb/train',
                    context=1, count=128, long_count=16, batch_size=16, seed=self.args.seed)
                contact, _ = _short(model, self.contact_val, context=1, count=256, batch_size=16, seed=self.args.seed)
                navigation.update(contact_generalization=contact,
                    heldout_idle=evaluate_heldout_idle(model, self.contact_val, seed=self.args.seed),
                    one_frame450=evaluate_one_frame450(model, self.args.navigation_data_root/'long/rgb/val',
                        directory/'navigation/one_frame450', seed=self.args.seed))
                navigation['complete_quality_gate'] = complete_quality_gate(navigation)
                navigation['passed'] = navigation['complete_quality_gate']['passed']
            with environment_profile(model, navigation_only=False):
                heads = evaluate_heads(model, self.short_val, directory/'combat/heads', context=1,
                    batch_size=16, seed=self.args.seed)
                trajectory = evaluate_state_trajectories(model, self.long_val, directory/'combat/trajectory',
                    context=1, horizon=450, count=16, batch_size=4, seed=self.args.seed,
                    include_respawned_slots=True)
                expanded_spawn = melee = None
                if getattr(self, 'recovery_revision', None) is not None:
                    from .rssm_spawn_curriculum import evaluate_spawn_events
                    from .rssm_combat_recovery import evaluate_melee_negatives
                    expanded_spawn = evaluate_spawn_events(model, self.long_val, directory/'combat/spawn_all', context=1)
                    melee = evaluate_melee_negatives(model, self.negative_val, directory/'combat/melee', context=1)
        gate = combat_quality_gate(heads, trajectory)
        if expanded_spawn is not None:
            from .rssm_combat_recovery import recovery_quality_gate
            gate = recovery_quality_gate(gate, expanded_spawn, melee)
        report = {'step': step, 'phase': 'combat', 'loss_protocol': self.loss_protocol,
            'navigation': navigation, 'combat': {'heads': heads, 'trajectory': trajectory, **gate},
            'passed': bool(navigation['passed'] and gate['passed']), 'source_held_out': True,
            'source_future_feedback': False, 'renderer_evaluated': False, 'gameplay_qualified': False}
        if getattr(self, 'coordinate_curriculum', None) is not None:
            from .rssm_drift_selection import heldout_drift_score
            report['drift_selection'] = heldout_drift_score(trajectory,
                directory/'combat/trajectory/heldout_trajectory_states.npz')
            report['coordinate_curriculum_updates'] = self.coordinate_curriculum['updates']
            if self.coordinate_curriculum['updates'] not in self.coordinate_curriculum['completed_validation_updates']:
                self.coordinate_curriculum['completed_validation_updates'].append(self.coordinate_curriculum['updates'])
        if getattr(self, 'replay_revision', None) is not None:
            report['replay_curriculum_updates'] = self.replay_revision['updates']
            self.replay_revision['baseline_complete'] = True
        if getattr(self, 'drift_only_revision', None) is not None:
            report['objective_protocol'] = 'rssm_coordinate_drift_only_v1'
            self.drift_only_revision['baseline_complete'] = True
        if getattr(self, 'loss_gated_curriculum', None) is not None:
            report['loss_gated_curriculum_updates'] = self.loss_gated_curriculum['updates']
            self.loss_gated_curriculum['baseline_complete'] = True
        if expanded_spawn is not None:
            report['combat'].update(expanded_spawn=expanded_spawn, close_wall_melee=melee)
            self.recovery_revision['baseline_complete'] = True
        training = self.state['rssm_training']
        training['quality_streak'] = training['quality_streak']+1 if report['passed'] else 0
        training.update(last_validation_step=step, last_validation_passed=report['passed'],
            last_validation_report=str(directory/'rssm_combat_validation.json'))
        atomic_json(directory/'rssm_combat_validation.json', report)
        atomic_json(self.root/'rssm_validation_latest.json', report)
        self.state.update(rssm_validation=report, validation_progress={'status': 'complete', 'step': step})
        self.publish()
        if getattr(self, 'coordinate_curriculum', None) is not None:
            from .rssm_drift_selection import drift_candidate_improves, publish_best_drift
            selector_path = self.root/'best_drift.json'
            previous = json.loads(selector_path.read_text(encoding='utf-8')) if selector_path.exists() else None
            if drift_candidate_improves(report['drift_selection'], previous['score'] if previous else None):
                # Save the exact just-validated full model/Adam before publishing a selector.
                self.save(model, self._drift_optimizer, self._drift_metadata)
                self.state['best_drift'] = publish_best_drift(self.checkpoint, self.root, step=step,
                    score=report['drift_selection'], validation_report=directory/'rssm_combat_validation.json',
                    combined_quality_passed=report['passed'])
            elif previous:
                self.state['best_drift'] = previous
            self.publish()

    def run(self):
        args = self.args; torch.set_num_threads(4)
        if self.device.type == 'cuda': torch.cuda.init()
        args.run_dir.mkdir(parents=True, exist_ok=True)
        combat_sha = verify_data(args.combat_data_root)
        nav_sha = verify_data(args.navigation_data_root)
        original_sha = verify_data(args.navigation_original_root)
        initial = self.root/'checkpoints/combat_initial.pt'
        if not self.checkpoint.exists() and not initial.exists():
            proof = prepare_combat_continuation(args.source_checkpoint, initial, run_id=self.root.name)
            atomic_json(self.root/'combat_migration.json', proof)
        model, saved = load_world_checkpoint(self.checkpoint if self.checkpoint.exists() else initial, 'cpu')
        source_step = saved['run_state']['combat_migration']['source_step']
        plan = {'version': 1, 'loss_protocol': PROTOCOL, 'source_checkpoint': str(args.source_checkpoint),
            'source_sha256': saved['run_state']['combat_migration']['source_sha256'],
            'source_step': source_step, 'combat_manifest_sha256': combat_sha,
            'navigation_manifest_sha256': nav_sha, 'navigation_original_manifest_sha256': original_sha,
            'batch': 16, 'horizon': 450, 'tbptt_chunk': 16, 'auxiliary_horizon': 16,
            'validation_every': 250, 'preview_every': 10, 'update_cycle': ['combat']*3+['navigation'],
            'latent_cycle': ['argmax', 'argmax', 'sample', 'argmax'],
            'combat_loss': {'npc_motion_weight': 5., 'spawn_weight': 1., 'game_training_v5': True},
            'navigation_profile': 'explicit_no_enemies', 'runtime_navigation_only': False,
            'renderer_sha256': _digest(args.renderer_checkpoint), 'source_future_feedback': False}
        path = self.root/'combat_training_plan.json'
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != plan:
            raise ValueError('Combat dataset/continuation/loss/renderer plan changed')
        if not path.exists(): atomic_json(path, plan)
        self.plan_sha = _digest(path)
        if self.checkpoint.exists() and saved['plan_sha256'] != self.plan_sha:
            raise ValueError('Combat checkpoint plan identity changed')
        self.state = saved['run_state']; self.state['run_id'] = self.root.name
        self.state.update(total=None if not args.updates else source_step+args.updates,
            status='initializing', loss_protocol=PROTOCOL)
        revision_path = getattr(args, 'feedback_revision_plan', None)
        previous_revision = self.state['rssm_training'].get('feedback_revision')
        self.feedback_mode, self.loss_protocol = 'attached', PROTOCOL
        if revision_path is None and previous_revision:
            raise ValueError('Saved numeric-feedback revision requires its immutable plan')
        if revision_path is not None:
            revision = json.loads(revision_path.read_text(encoding='utf-8'))
            revision_sha = _digest(revision_path)
            if (revision.get('version') != 2 or revision.get('base_plan_sha256') != self.plan_sha
                    or revision.get('numeric_state_feedback') != 'detached'
                    or revision.get('active_environments') != ['combat']
                    or revision.get('tbptt_chunk') != 16
                    or revision.get('source_future_feedback') is not False):
                raise ValueError('Unsupported or changed numeric-feedback revision')
            if previous_revision:
                if (previous_revision['sha256'] != revision_sha
                        or previous_revision['last_optimizer_step'] != self.state['global_training_step']):
                    raise ValueError('Numeric-feedback revision identity/counters changed')
            else:
                if (revision['start_optimizer_step'] != self.state['global_training_step']
                        or _digest(Path(revision['source_checkpoint'])) != revision['source_checkpoint_sha256']):
                    raise ValueError('Numeric-feedback revision needs its exact saved Adam boundary')
                self.state.setdefault('objective_history', []).append({'loss_protocol': PROTOCOL,
                    'last_step': self.state['global_training_step'], 'reason': 'measured_typed_numeric_gradient_amplification'})
                previous_revision = {'sha256': revision_sha, 'plan': str(revision_path),
                    'start_optimizer_step': revision['start_optimizer_step'], 'updates': 0,
                    'last_optimizer_step': revision['start_optimizer_step'], 'numeric_state_feedback': 'detached'}
                self.state['rssm_training']['feedback_revision'] = previous_revision
            self.feedback_mode = 'detached'
            self.loss_protocol = 'rssm_one_enemy_combat_numeric_feedback_v2'
            self.state.update(loss_protocol=self.loss_protocol)
            self.state['training_protocol'].update(numeric_state_feedback='detached_on_combat_updates',
                latent_memory_gradient='full_TBPTT16', typed_forward_values_unchanged=True)
        recovery_path = getattr(args, 'combat_recovery_plan', None)
        self.recovery_revision = None; self.recovery_sampler = None; combat_branches = None
        if recovery_path is None and self.state['rssm_training'].get('combat_recovery_revision'):
            raise ValueError('Saved recovery revision requires its immutable plan')
        recovery_root = args.combat_data_root
        if recovery_path is not None:
            from .rssm_combat_recovery import bind_recovery_revision, RECOVERY_PROTOCOL
            recovery = json.loads(recovery_path.read_text(encoding='utf-8'))
            recovery_root = Path(recovery['data_root']).resolve()
            negative_root = Path(recovery['negative_root']).resolve()
            recovery_sha = verify_data(recovery_root); negative_sha = verify_data(negative_root)
            self.recovery_revision, combat_branches = bind_recovery_revision(self.state, recovery,
                sha256=_digest(recovery_path), path=recovery_path, base_plan_sha256=self.plan_sha,
                data_sha256=recovery_sha, negative_sha256=negative_sha,
                source_checkpoint_sha256=_digest(Path(recovery['source_checkpoint'])))
            if self.feedback_mode != 'detached': raise ValueError('Recovery requires detached numeric feedback')
            self.loss_protocol = RECOVERY_PROTOCOL
        self.batch_revision = None
        self.combat_loss_overrides = {}
        self.target_distance_balanced_sampling = False
        self.microbatch, self.accumulation = args.batch, args.accumulation_steps
        batch_path = getattr(args, 'batch_revision_plan', None)
        if batch_path is None:
            if (self.state['rssm_training'].get('batch_revision') or self.microbatch != 16 or self.accumulation != 1):
                raise ValueError('Changed effective batch requires its immutable revision plan')
        else:
            from .rssm_batch_revision import bind_batch_revision, BATCH_PROTOCOL
            if self.recovery_revision is None:raise ValueError('Batch revision requires the preserved recovery curriculum')
            batch_plan = json.loads(batch_path.read_text(encoding='utf-8'))
            self.batch_revision = bind_batch_revision(self.state, batch_plan, sha256=_digest(batch_path),path=batch_path,
                parent_recovery_sha=self.recovery_revision['sha256'],
                source_checkpoint_sha=_digest(Path(batch_plan['source_checkpoint'])),
                cli_microbatch=32 if getattr(args,'batch256_plan',None) is not None else self.microbatch,
                cli_accum=self.accumulation)
            preview_path = getattr(args, 'preview_assets_plan', None)
            if (batch_plan.get('preview_assets_sha256') is not None
                    and (preview_path is None or _digest(preview_path)!=batch_plan['preview_assets_sha256'])):
                raise ValueError('Preview assets must retain their explicitly pinned identity')
            self.loss_protocol = BATCH_PROTOCOL
            self.combat_loss_overrides=batch_plan.get('combat_loss_overrides',{})
            if self.combat_loss_overrides not in ({},{'target_supervision':'shot_distance_balanced'}):
                raise ValueError('Unknown explicitly bound combat loss overrides')
            if self.combat_loss_overrides:
                self.state['training_protocol']['combat_loss_overrides']=dict(self.combat_loss_overrides)
            self.target_distance_balanced_sampling=batch_plan.get('target_distance_balanced_sampling',False)
            if not isinstance(self.target_distance_balanced_sampling,bool):
                raise ValueError('Distance-balanced target sampling must be an explicit boolean')
            self.state['training_protocol']['target_distance_balanced_sampling']=self.target_distance_balanced_sampling
        self.loss_balancer = None
        self.loss_balance_revision = None
        balance_path = getattr(args, 'loss_balance_plan', None)
        if balance_path is None and self.state['rssm_training'].get('loss_balance_revision'):
            raise ValueError('Saved task normalization requires its immutable revision plan and EMA')
        if balance_path is not None:
            if self.batch_revision is None or not model.config.enemy_rssm_enabled:
                raise ValueError('Separate task normalization requires the dual RSSMs and preserved32x4 curriculum')
            from .rssm_loss_balance import bind_loss_balance_revision, LOSS_PROTOCOL
            balance_plan = json.loads(balance_path.read_text(encoding='utf-8'))
            self.loss_balance_revision, self.loss_balancer = bind_loss_balance_revision(self.state, balance_plan,
                path=balance_path, sha256=_digest(balance_path),
                source_checkpoint_sha256=_digest(Path(balance_plan['source_checkpoint'])),
                base_plan_sha256=self.plan_sha, batch_revision_sha256=self.batch_revision['sha256'],
                loaded_checkpoint_sha256=_digest(self.checkpoint if self.checkpoint.exists() else initial))
            self.loss_protocol = LOSS_PROTOCOL
            if self.loss_balance_revision['updates'] == 0:
                self.state.update(loss=None, latest_metrics=None, loss_groups=None)
        self.combat_only_revision = None
        combat_only_path = getattr(args, 'combat_only_plan', None)
        if combat_only_path is None and self.state['rssm_training'].get('combat_only_revision'):
            raise ValueError('Saved combat-only curriculum requires its immutable plan')
        if combat_only_path is not None:
            if self.loss_balancer is None:
                raise ValueError('Combat-only continuation must preserve the current dual-RSSM task normalization')
            from .rssm_training_schedule import bind_combat_only_revision
            schedule_plan = json.loads(combat_only_path.read_text(encoding='utf-8'))
            self.combat_only_revision = bind_combat_only_revision(self.state, schedule_plan,
                path=combat_only_path, sha256=_digest(combat_only_path),
                source_checkpoint_sha256=_digest(Path(schedule_plan['source_checkpoint'])),
                loaded_checkpoint_sha256=_digest(self.checkpoint if self.checkpoint.exists() else initial),
                base_plan_sha256=self.plan_sha, loss_balance_plan_sha256=self.loss_balance_revision['sha256'])
        self.combat_v6_revision = None
        self.attack_positive_balanced_sampling = False
        v6_path = getattr(args, 'combat_v6_plan', None)
        if v6_path is None and self.state['rssm_training'].get('combat_v6_revision'):
            raise ValueError('Saved late-event/component training requires its immutable v6 plan')
        if v6_path is not None:
            if self.combat_only_revision is None or self.loss_balancer is None:
                raise ValueError('Combat v6 preserves the current combat-only dual-RSSM continuation')
            from .rssm_combat_v6 import bind_combat_v6
            from .rssm_component_balance import LOSS_PROTOCOL
            v6_plan = json.loads(v6_path.read_text(encoding='utf-8'))
            calibration_path = Path(v6_plan['gradient_calibration'])
            calibration = json.loads(calibration_path.read_text(encoding='utf-8'))
            self.combat_v6_revision, self.loss_balancer, combat_branches = bind_combat_v6(
                self.state, v6_plan, calibration, path=v6_path, sha256=_digest(v6_path),
                calibration_sha256=_digest(calibration_path),
                source_checkpoint_sha256=_digest(Path(v6_plan['source_checkpoint'])),
                loaded_checkpoint_sha256=_digest(self.checkpoint if self.checkpoint.exists() else initial),
                base_plan_sha256=self.plan_sha)
            self.loss_protocol = LOSS_PROTOCOL
            self.attack_positive_balanced_sampling = True
        self.batch256_revision = None
        batch256_path = getattr(args,'batch256_plan',None)
        if batch256_path is None and self.state['rssm_training'].get('batch256_revision'):
            raise ValueError('Saved64x4 training requires its immutable batch256 plan')
        if batch256_path is not None:
            if self.combat_v6_revision is None:
                raise ValueError('Batch256 preserves the current aligned-event/component protocol')
            from .rssm_batch256 import bind_batch256_revision
            batch256_plan=json.loads(batch256_path.read_text(encoding='utf-8'))
            probe_path=Path(batch256_plan['batch64_probe'])
            probe=json.loads(probe_path.read_text(encoding='utf-8'))
            self.batch256_revision,combat_branches=bind_batch256_revision(self.state,batch256_plan,probe,
                path=batch256_path,sha256=_digest(batch256_path),probe_sha256=_digest(probe_path),
                source_checkpoint_sha256=_digest(Path(batch256_plan['source_checkpoint'])),
                loaded_checkpoint_sha256=_digest(self.checkpoint if self.checkpoint.exists()else initial),
                cli_microbatch=self.microbatch,cli_accumulation=self.accumulation)
        self.coordinate_curriculum = None
        self.coordinate_plan = None
        coordinate_path = getattr(args, 'coordinate_curriculum_plan', None)
        if coordinate_path is None and self.state['rssm_training'].get('coordinate_curriculum'):
            raise ValueError('Saved coordinate curriculum requires its immutable plan')
        if coordinate_path is not None:
            if self.batch256_revision is None or self.feedback_mode != 'detached':
                raise ValueError('Coordinate curriculum preserves the validated64x4 component continuation')
            from .rssm_coordinate_curriculum import bind_coordinate_curriculum
            self.coordinate_plan = json.loads(coordinate_path.read_text(encoding='utf-8'))
            self.coordinate_curriculum = bind_coordinate_curriculum(self.state, self.coordinate_plan,
                path=coordinate_path, sha256=_digest(coordinate_path),
                source_checkpoint_sha256=_digest(Path(self.coordinate_plan['source_checkpoint'])),
                loaded_checkpoint_sha256=_digest(self.checkpoint if self.checkpoint.exists() else initial),
                base_plan_sha256=self.plan_sha, cli_microbatch=self.microbatch,
                cli_accumulation=self.accumulation)
            self.combat_loss_overrides = {**self.combat_loss_overrides,
                'kinematic_protocol': self.coordinate_plan['kinematic_protocol']}
        self.replay_revision = None
        self.replay_plan = None
        replay_path = getattr(args, 'replay_curriculum_plan', None)
        if replay_path is None and (self.state['rssm_training'].get('replay_curriculum')
                or 'horizon_ema_banks' in self.state['rssm_training'].get('loss_balance_state', {})):
            raise ValueError('Saved long replay requires its immutable plan and separate horizon EMA banks')
        if replay_path is not None:
            if self.coordinate_curriculum is None or self.loss_balancer is None:
                raise ValueError('Long replay preserves the existing coordinate and component continuation')
            from .rssm_replay_curriculum import bind_replay_curriculum
            self.replay_plan = json.loads(replay_path.read_text(encoding='utf-8'))
            seed_path = Path(self.replay_plan['long_seed_checkpoint'])
            seed_checkpoint = torch.load(seed_path, map_location='cpu', weights_only=False, mmap=True)
            seed_balance = copy.deepcopy(seed_checkpoint['run_state']['rssm_training']['loss_balance_state'])
            del seed_checkpoint
            self.replay_revision, self.loss_balancer = bind_replay_curriculum(self.state, self.replay_plan,
                path=replay_path, sha256=_digest(replay_path),
                source_checkpoint_sha256=_digest(Path(self.replay_plan['source_checkpoint'])),
                loaded_checkpoint_sha256=_digest(self.checkpoint if self.checkpoint.exists() else initial),
                long_seed_checkpoint_sha256=_digest(seed_path), long_seed_loss_balance_state=seed_balance,
                base_plan_sha256=self.plan_sha, cli_microbatch=self.microbatch, cli_accumulation=self.accumulation)
        self.drift_only_revision = None
        drift_path = getattr(args, 'drift_only_plan', None)
        if drift_path is None and (self.state['rssm_training'].get('drift_only_revision')
                or self.state.get('objective_protocol') == 'rssm_coordinate_drift_only_v1'):
            raise ValueError('Saved coordinate-only training requires its immutable objective plan')
        if drift_path is not None:
            if self.replay_revision is None:
                raise ValueError('Coordinate-only training preserves the existing short/long replay curriculum')
            from .rssm_drift_only_protocol import bind_drift_only
            drift_plan = json.loads(drift_path.read_text(encoding='utf-8'))
            self.drift_only_revision = bind_drift_only(self.state, drift_plan,
                path=drift_path, sha256=_digest(drift_path),
                source_checkpoint_sha256=_digest(Path(drift_plan['source_checkpoint'])),
                loaded_checkpoint_sha256=_digest(self.checkpoint if self.checkpoint.exists() else initial),
                base_plan_sha256=self.plan_sha, optimizer_parameter_names=saved['optimizer_parameter_names'])
        self.loss_gated_curriculum = None
        gate_path = getattr(args, 'loss_gated_curriculum_plan', None)
        if gate_path is None and (self.state['rssm_training'].get('loss_gated_curriculum')
                or self.state.get('horizon_curriculum_protocol') == 'rssm_xy_loss_gated_horizon_v1'):
            raise ValueError('Saved loss-gated horizon training requires its immutable plan')
        if gate_path is not None:
            if self.drift_only_revision is None:
                raise ValueError('Loss-gated horizons preserve the existing pure-XY-only objective')
            from .rssm_loss_gated_curriculum import bind_loss_gated_curriculum
            gate_plan = json.loads(gate_path.read_text(encoding='utf-8'))
            self.loss_gated_curriculum = bind_loss_gated_curriculum(self.state,gate_plan,
                path=gate_path,sha256=_digest(gate_path),
                source_checkpoint_sha256=_digest(Path(gate_plan['source_checkpoint'])),
                loaded_checkpoint_sha256=_digest(self.checkpoint if self.checkpoint.exists() else initial),
                base_plan_sha256=self.plan_sha)
        model.rssm_migration = saved['rssm_migration']
        model.enemy_rssm_resize = saved.get('enemy_rssm_resize')
        model.rssm_fusion_migration = saved.get('rssm_fusion_migration')
        model.to(self.device).float(); policy = configure_rssm_training(model)
        names = [name for name, value in model.named_parameters() if value.requires_grad]
        if names != saved['optimizer_parameter_names']: raise ValueError('Combat Adam parameter names changed')
        self._optimizer_parameter_names = list(names)
        named_parameters=dict(model.named_parameters())
        groups=[{'params':[named_parameters[saved['optimizer_parameter_names'][index]]for index in group['params']]}
            for group in saved['optimizer']['param_groups']]
        optimizer = torch.optim.AdamW(groups, lr=1e-4, weight_decay=.01)
        optimizer.load_state_dict(saved['optimizer'])
        if any(group['lr'] != 1e-4 for group in optimizer.param_groups): raise ValueError('Existing Adam LR changed')
        frozen = _hash_tensors({name: value for name, value in model.state_dict().items()
            if name.startswith(('flow_renderer.', 'vjepa_backend.renderer.', 'vjepa_backend.feature_head.'))})
        self.short = StagedSequenceData(recovery_root/'moving1/rgb/train', self.device, mode='stream', cache_bytes=64*1024**2)
        self.short_val = StagedSequenceData(args.combat_data_root/'moving1/rgb/val', self.device, mode='stream', cache_bytes=32*1024**2)
        self.long_val = StagedSequenceData(args.combat_data_root/'full_game/rgb/val', self.device, mode='stream', cache_bytes=32*1024**2)
        combat_long = StagedSequenceData(args.combat_data_root/'full_game/rgb/train', self.device, mode='stream', cache_bytes=64*1024**2)
        nav_short = StagedSequenceData(args.combat_data_root/'navigation/rgb/train', self.device, mode='stream', cache_bytes=32*1024**2)
        nav_long = StagedSequenceData(args.navigation_original_root/'rgb/train', self.device, mode='stream', cache_bytes=64*1024**2)
        history = self.state['navigation_training_history']
        contact_root = Path(history['contact_revision']['dataset_directory'])
        self.contact_val = StagedSequenceData(contact_root/'short/rgb/val', self.device, mode='stream', cache_bytes=32*1024**2)
        if self.recovery_revision is not None:
            from .rssm_spawn_curriculum import RecoveryCombatSampler
            sampler = self.recovery_sampler = RecoveryCombatSampler(self.short, context=1, horizon=16,
                cache_path=self.root/'recovery_sampling.json',
                target_distance_balanced=self.target_distance_balanced_sampling,
                attack_positive_balanced=self.attack_positive_balanced_sampling)
            if self.recovery_revision['sampler_state'] is not None:
                sampler_state = self.recovery_revision['sampler_state']
                if self.attack_positive_balanced_sampling and sampler_state.get('version') != 2:
                    from .rssm_spawn_curriculum import upgrade_attack_balanced_sampler_state
                    sampler_state = upgrade_attack_balanced_sampler_state(sampler_state)
                sampler.restore_state(sampler_state)
            self.negative_val = StagedSequenceData(negative_root/'moving1/rgb/val', self.device,
                mode='stream', cache_bytes=32*1024**2)
            atomic_json(self.root/'recovery_sampling_summary.json', sampler.summary(batch_size=self.microbatch))
        else:
            sampler = CombatTrainingSampler(self.short, context=1, horizon=16, cache_path=self.root/'combat_sampling.json')
            atomic_json(self.root/'combat_sampling_summary.json', sampler.summary())
        settings = history['contact_revision']['loss_settings']
        branches = self.state['training_protocol'].get('contact_branch_protocol')
        self.state['training_protocol'].update(own_memory_contact_branches=branches is not None and self.combat_only_revision is None,
            contact_branch_active_environments=['navigation'] if branches and self.combat_only_revision is None else [])
        # Restore after all constructors/indexing. Validation/previews fork RNG.
        self.rng.bit_generator.state = saved['rng_state']
        torch.set_rng_state(saved['torch_rng_state'].cpu())
        if self.device.type == 'cuda': torch.cuda.set_rng_state_all([value.cpu() for value in saved['cuda_rng_state']])
        from .rssm_navigation_protocol import verify_restored_boundary
        restore_proof = verify_restored_boundary(model, optimizer, self.rng, saved)
        restore_proof.update(phase='combat', numeric_state_feedback=self.feedback_mode,
            no_rollback=True, preserved_global_step=saved['training_steps'], combat_recovery_active=self.recovery_revision is not None,
            microbatch_size=self.microbatch,gradient_accumulation_steps=self.accumulation,
            effective_batch_size=self.microbatch*self.accumulation)
        restore_tag=('groupbalance_v5' if self.loss_balancer is not None else
            'batch128_v4' if self.batch_revision else 'recovery_v3' if self.recovery_revision else self.feedback_mode)
        if self.combat_only_revision is not None:
            restore_tag = 'combatonly_v1'
            restore_proof.update(training_schedule='combat_only_v1', navigation_replay_enabled=False)
        if self.combat_v6_revision is not None:
            restore_tag = 'combat_components_v6'
            restore_proof.update(loss_protocol=self.loss_protocol, late_event_supervised_horizon=16,
                component_priorities=self.loss_balancer.state['components']['priorities'])
        if self.batch256_revision is not None:
            restore_tag='batch256_v7'
            restore_proof.update(batch_protocol='rssm_effective256_v7',recovery_branches_per_microbatch=16,
                direct_query_budget_per_microbatch=256)
        if self.coordinate_curriculum is not None:
            restore_tag = 'coordinate_curriculum_v1'
            restore_proof.update(coordinate_curriculum_plan_sha256=self.coordinate_curriculum['sha256'],
                coordinate_curriculum_updates=self.coordinate_curriculum['updates'],
                coordinate_gradient_protocol=self.coordinate_plan['coordinate_gradient_protocol'],
                memory_overshooting_protocol=self.coordinate_plan['overshooting_protocol'])
        if self.replay_revision is not None:
            restore_tag = 'coordinate_long_replay_v1'
            restore_proof.update(replay_curriculum_plan_sha256=self.replay_revision['sha256'],
                replay_curriculum_updates=self.replay_revision['updates'],
                separate_short_long_ema=True, long_ema_seed_step=5014)
        if self.drift_only_revision is not None:
            from .rssm_drift_only_protocol import configure_drift_only_training
            drift_policy = configure_drift_only_training(model, optimizer_parameter_names=self._optimizer_parameter_names)
            policy['dynamics_trainable'] = drift_policy['trainable_parameters']
            self.state['rssm_training']['drift_only_parameter_policy'] = drift_policy
            restore_tag = 'drift_only_v1'
            restore_proof.update(objective_protocol='rssm_coordinate_drift_only_v1',
                drift_only_plan_sha256=self.drift_only_revision['sha256'],
                full_optimizer_parameter_names_preserved=True, ema_values_frozen=True)
        if self.loss_gated_curriculum is not None:
            restore_tag = 'loss_gated_drift_v1'
            restore_proof.update(loss_gated_curriculum_plan_sha256=self.loss_gated_curriculum['sha256'],
                loss_gated_stage_index=self.loss_gated_curriculum['stage_index'],
                loss_gated_current_horizon=self.loss_gated_curriculum['current_horizon'],forced_long_replay=False)
        restore_name = f"resume_verification_g{saved['training_steps']}_{restore_tag}.json"
        atomic_json(self.root/restore_name, restore_proof)
        metadata = {key: saved[key] for key in ('rssm_migration', 'source_teacher_checkpoint','dual_rssm_migration',
            'enemy_rssm_resize', 'rssm_fusion_migration', 'combat_rssm_migration') if key in saved}
        self._drift_optimizer, self._drift_metadata = optimizer, metadata
        self.state.update(status='running', model_parameters=saved['run_state'].get('model_parameters'),
            frozen_assets_fingerprint=frozen, keep_awake={'system_required': True, 'display_required': False})
        self.state['rssm_training'].update(dynamics_trainable=policy['dynamics_trainable'])
        del saved; gc.collect()
        self.save(model, optimizer, metadata)
        if getattr(args, 'preview_assets_plan', None) is not None:
            preview_proof=self.initialize_preview_assets(model)
            atomic_json(self.root/'updated_preview_assets_verification.json',preview_proof)
            self.preview(model);self.save(model,optimizer,metadata)
        if (self.state['rssm_training']['last_validation_step'] is None
                or self.recovery_revision is not None and not self.recovery_revision['baseline_complete']
                or self.combat_v6_revision is not None and not self.combat_v6_revision['baseline_complete']
                or self.coordinate_curriculum is not None
                    and 0 not in self.coordinate_curriculum['completed_validation_updates']
                or self.replay_revision is not None and not self.replay_revision.get('baseline_complete', False)):
            self.validate(model)
            if self.combat_v6_revision is not None:self.combat_v6_revision['baseline_complete'] = True
            self.save(model, optimizer, metadata)
        elif self.drift_only_revision is not None and not self.drift_only_revision.get('baseline_complete', False):
            self.validate(model)
            if self.combat_v6_revision is not None:self.combat_v6_revision['baseline_complete'] = True
            self.save(model, optimizer, metadata)
        elif self.loss_gated_curriculum is not None and not self.loss_gated_curriculum.get('baseline_complete',False):
            self.validate(model);self.save(model,optimizer,metadata)
        while (not args.updates or self.state['rssm_training']['combat_completed'] < args.updates):
            training = self.state['rssm_training']; completed = training['combat_completed']
            if training['quality_streak'] >= 2 and self.loss_gated_curriculum is None: break
            if (self.root/'stop_training').exists():
                self.state['status'] = 'stopped'; self.save(model, optimizer, metadata); return
            step = self.state['global_training_step']+1
            from .rssm_training_schedule import next_training_environment
            environment = next_training_environment(self.state, plan['update_cycle'], completed)
            # Latent mode and observed context vary independently of the3:1 task cycle.
            mode = plan['latent_cycle'][(completed//4) % 4]
            context = 1+(completed//16) % 2
            main_horizon = 450
            if self.coordinate_curriculum is not None:
                from .rssm_coordinate_curriculum import coordinate_horizon
                main_horizon = coordinate_horizon(self.coordinate_curriculum['updates'])
            replay_choice = None
            if self.replay_revision is not None:
                from .rssm_replay_curriculum import replay_training_horizon
                replay_choice = replay_training_horizon(self.state)
                main_horizon = replay_choice['horizon']
                self.loss_balancer.select_update(horizon=main_horizon)
            configure_rssm_training(model)
            if self.drift_only_revision is not None:
                from .rssm_drift_only_protocol import configure_drift_only_training
                configure_drift_only_training(model, optimizer_parameter_names=self._optimizer_parameter_names)
            model.core.train(mode == 'sample')
            optimizer.zero_grad(set_to_none=True)
            pool = combat_long if environment == 'combat' else nav_long
            micro_rows=[];sample_rows=[];group_reports=[]
            for micro in range(self.accumulation):
                pairs = np.column_stack([self.rng.integers(len(pool.episodes),size=self.microbatch),
                    np.zeros(self.microbatch,np.int64)])
                batch = pool.batch(self.microbatch,context=context,horizon=main_horizon,indices=pairs)
                if environment == 'combat':
                    aux_pairs=sampler.sample(self.microbatch,self.rng)
                    auxiliary=(None if self.drift_only_revision is not None else
                        self.short.batch(self.microbatch,context=1,horizon=16,indices=aux_pairs))
                    sample_rows.append(copy.deepcopy(sampler.last_sample))
                    if self.drift_only_revision is not None:
                        sample_rows[-1].update(index_reservation_only=True, used_aux_fragments=0,
                            auxiliary_source_queries=0, training_cases_used=False)
                else:
                    auxiliary=nav_short.batch(self.microbatch,context=1,horizon=16,rng=self.rng)
                late_kwargs = {}
                if self.replay_revision is not None and (main_horizon == 450 or micro == 0):
                    late_kwargs['late_prefix_protocol'] = self.replay_plan['late_prefix_protocol']
                    if main_horizon < 450:
                        late_data = pool.batch(8, context=context, horizon=450, indices=pairs[:8])
                        late_kwargs['late_prefix_batch'] = {
                            'actions': late_data['actions'].detach(), 'map': late_data['map'].detach()}
                        del late_data
                def progress(processed,horizon):
                    self.state['long_training_progress']={'step':step,'transitions_processed':processed,
                        'horizon':horizon,'tbptt_chunk':16,'optimizer_updated':False,
                        'microbatch_index':micro+1,'microbatches_total':self.accumulation,
                        'completed_microbatches':micro,'microbatch_size':self.microbatch,
                        'effective_batch_size':self.microbatch*self.accumulation,
                        'environment':environment,'source_future_feedback':False}
                    if replay_choice is not None:
                        self.state['long_training_progress']['ema_bank'] = replay_choice['ema_bank']
                    if self.drift_only_revision is not None:
                        self.state['long_training_progress']['objective_protocol'] = 'rssm_coordinate_drift_only_v1'
                    self.publish()
                with environment_profile(model,navigation_only=environment=='navigation'):
                    task_losses = (self.loss_balancer.new_microbatch(environment)
                        if self.loss_balancer is not None and self.drift_only_revision is None else None)
                    if self.drift_only_revision is not None:
                        from .rssm_drift_only_objective import drift_only_backward
                        row=drift_only_backward(model,batch,context=context,observed_context=context,
                            expected_horizon=main_horizon,amp=nullcontext,progress_callback=progress,
                            coordinate_gradient_protocol=self.coordinate_plan['coordinate_gradient_protocol'],
                            combat_branch_protocol=combat_branches,**late_kwargs)
                    else:
                        row=rssm_dynamics_backward(model,batch,auxiliary_batch=auxiliary,context=context,
                        observed_context=context,auxiliary_context=1,auxiliary_observed_context=1,
                        expected_horizon=main_horizon,amp=nullcontext,progress_callback=progress,
                        contact_protocol=settings,**({'combat_protocol':{**plan['combat_loss'],**self.combat_loss_overrides},
                            'combat_branch_protocol':combat_branches}if environment=='combat'else
                            {'contact_branch_protocol':branches}if branches else{}),
                        numeric_state_feedback=self.feedback_mode if environment=='combat'else'attached',
                        **({'coordinate_gradient_protocol': self.coordinate_plan['coordinate_gradient_protocol'],
                            'memory_overshooting_protocol': self.coordinate_plan['overshooting_protocol']}
                           if self.coordinate_curriculum is not None else {}),
                        **late_kwargs,
                            **({'loss_groups': task_losses} if task_losses is not None else {}))
                if task_losses is not None or self.drift_only_revision is not None:group_reports.append(row['loss_groups'])
                micro_rows.append(row)
                del batch,auxiliary
            from .rssm_accumulation import accumulated_metrics,average_accumulated_gradients
            average_accumulated_gradients(model.parameters(),self.accumulation)
            metrics=accumulated_metrics(micro_rows)
            if self.loss_balancer is not None:
                if self.drift_only_revision is not None:
                    from .rssm_drift_only_protocol import summarize_drift_groups
                    metrics['loss_groups'] = summarize_drift_groups(group_reports, step=step)
                else:
                    metrics['loss_groups'] = self.loss_balancer.summarize(group_reports, environment=environment, step=step)
                metrics['train_loss'] = metrics['loss_groups']['total']
                metrics['train_loss_without_kl_floor'] = metrics['train_loss']-metrics['loss_groups']['kl_floor']
            # Reject invalid telemetry before an irreversible Adam/counter
            # commit, using the same encoder as the durable metrics writer.
            json.dumps(metrics, ensure_ascii=False, allow_nan=False)
            self.state['long_training_progress']['completed_microbatches']=self.accumulation
            metrics['world_grad_norm'] = float(clip_finite_grad_norm_(model.named_parameters(), context=f'RSSM combat update{step}'))
            optimizer.step();commit_combat_update(self.state,step=step,environment=environment,
                batch_size=self.microbatch*self.accumulation)
            if previous_revision is not None:
                previous_revision['updates'] += 1; previous_revision['last_optimizer_step'] = step
            if self.recovery_revision is not None:
                self.recovery_revision['updates'] += 1; self.recovery_revision['last_optimizer_step'] = step
                self.recovery_revision['combat_updates'] += int(environment == 'combat')
            if self.batch_revision is not None:
                self.batch_revision['updates']+=1;self.batch_revision['last_optimizer_step']=step
            if self.loss_balancer is not None:
                if self.drift_only_revision is not None:
                    from .rssm_drift_only_protocol import advance_frozen_balance_clock
                    advance_frozen_balance_clock(self.loss_balancer, step=step)
                else:
                    self.loss_balancer.commit(metrics['loss_groups'], step=step)
                self.loss_balance_revision['updates'] += 1
                self.loss_balance_revision['last_optimizer_step'] = step
                training['loss_balance_state'] = self.loss_balancer.state_dict()
                self.state['loss_groups'] = metrics['loss_groups']
            if self.combat_only_revision is not None:
                from .rssm_training_schedule import commit_combat_only_update
                commit_combat_only_update(self.state, step=step, environment=environment)
            if self.combat_v6_revision is not None:
                self.combat_v6_revision['updates'] += 1
                self.combat_v6_revision['last_optimizer_step'] = step
            if self.batch256_revision is not None:
                from .rssm_batch256 import commit_batch256_update
                commit_batch256_update(self.state,step=step,sampler_draws=self.recovery_sampler.draws)
            if self.coordinate_curriculum is not None:
                from .rssm_coordinate_curriculum import commit_coordinate_update
                commit_coordinate_update(self.state, step=step, horizon=main_horizon,
                    sampler_draws=self.recovery_sampler.draws)
                metrics.update(coordinate_curriculum_updates=self.coordinate_curriculum['updates'],
                    coordinate_curriculum_plan_sha256=self.coordinate_curriculum['sha256'],
                    coordinate_gradient_protocol=self.coordinate_plan['coordinate_gradient_protocol'],
                    memory_overshooting_protocol=self.coordinate_plan['overshooting_protocol'])
            if self.replay_revision is not None:
                from .rssm_replay_curriculum import commit_replay_update
                commit_replay_update(self.state, step=step, horizon=main_horizon,
                    forced_long_replay=replay_choice['forced_long_replay'])
                metrics.update(replay_curriculum_updates=self.replay_revision['updates'],
                    replay_curriculum_plan_sha256=self.replay_revision['sha256'],
                    long_replay_update=replay_choice['forced_long_replay'],
                    curriculum_scheduled_horizon=replay_choice['curriculum_horizon'],
                    horizon_ema_bank=replay_choice['ema_bank'])
            if self.drift_only_revision is not None:
                from .rssm_drift_only_protocol import commit_drift_only_update
                commit_drift_only_update(self.state,step=step,sampler_draws=self.recovery_sampler.draws)
                metrics.update(objective_protocol='rssm_coordinate_drift_only_v1',
                    drift_only_updates=self.drift_only_revision['updates'],
                    drift_only_plan_sha256=self.drift_only_revision['sha256'],
                    ema_values_frozen=True, ema_learning=False, index_reservation_only=True,
                    auxiliary_source_queries=0, used_aux_fragments=0,
                    full_optimizer_parameter_names_preserved=True)
            gate_advanced = False
            if self.loss_gated_curriculum is not None:
                from .rssm_loss_gated_curriculum import commit_loss_gated_update
                gate = commit_loss_gated_update(self.state,step=step,horizon=main_horizon,
                    train_loss=metrics['train_loss'],microbatches=len(micro_rows),
                    microbatch_size=self.microbatch,effective_batch_size=self.microbatch*self.accumulation)
                gate_advanced = gate['last_stage_advanced']
                metrics.update(loss_gated_curriculum_plan_sha256=gate['sha256'],
                    loss_gated_curriculum_updates=gate['updates'],loss_gated_stage_index=gate['stage_index'],
                    loss_gated_current_horizon=gate['current_horizon'],last_training_horizon=main_horizon,
                    loss_gate_threshold=.25,loss_gate_passed=gate['last_gate_passed'],
                    loss_gate_consecutive_below_threshold=gate['consecutive_below_threshold'],
                    loss_gate_required_consecutive_passes=10,loss_gate_goal_met=gate['goal_met'],
                    loss_gate_stage_advanced=gate_advanced,forced_long_replay_active=False)
            metrics.update(step=step, global_training_step=step, rssm_phase='combat', component_phase='combat',
                stage='rssm_combat_dynamics', loss_protocol=self.loss_protocol, training_environment=environment,
                contact_branch_active=environment == 'navigation' and branches is not None,
                combat_branch_active=environment == 'combat' and combat_branches is not None,
                environment_navigation_only=environment == 'navigation', rssm_latent_training_mode=mode,
                microbatch_size=self.microbatch,gradient_accumulation_steps=self.accumulation,
                effective_batch_size=self.microbatch*self.accumulation,completed_microbatches=len(micro_rows),
                dynamics_updated=True, image_objectives_active=False, timestamp=now(),
                main_horizon=main_horizon, main_numeric_context=context, main_observed_context=context,
                main_window_start=0)
            metrics['training_schedule'] = self.state.get('training_schedule', 'combat3_navigation1')
            metrics['batch_protocol']=self.state.get('batch_protocol','rssm_effective128_v4')
            if environment=='combat':
                metrics['combat_sampling']=(sample_rows[0]if self.accumulation==1 else
                    {'microbatches':sample_rows,'quota':{name:sum(row['quota'].get(name,0) for row in sample_rows)
                        for name in set().union(*(set(row['quota']) for row in sample_rows))},
                     'microbatch_quota':sample_rows[0]['quota'],'source_labels_only':True})
            self.state['long_training_progress']['optimizer_updated'] = True
            self.state.update(loss=metrics['train_loss'], latest_metrics=metrics, current=step, status='running',
                message=('Coordinate drift only; other training objectives frozen' if self.drift_only_revision is not None
                    else 'One-enemy Source combat' if environment == 'combat' else 'Explicit no-enemy navigation replay'))
            with (self.root/'metrics.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(metrics, ensure_ascii=False, allow_nan=False)+'\n')
            first_recovery = self.recovery_revision is not None and self.recovery_revision['updates'] == 1
            first_batch=self.batch_revision is not None and self.batch_revision['updates']==1
            first_balance=self.loss_balance_revision is not None and self.loss_balance_revision['updates']==1
            early_balance=self.loss_balance_revision is not None and self.loss_balance_revision['updates']<=4
            early_combat_only=self.combat_only_revision is not None and self.combat_only_revision['updates']<=4
            early_v6=self.combat_v6_revision is not None and self.combat_v6_revision['updates']<=4
            early_batch256=self.batch256_revision is not None and self.batch256_revision['updates']<=4
            early_coordinate=self.coordinate_curriculum is not None and self.coordinate_curriculum['updates']<=4
            early_replay=self.replay_revision is not None and self.replay_revision['updates']<=5
            early_drift=self.drift_only_revision is not None and self.drift_only_revision['updates']<=5
            early_gate=self.loss_gated_curriculum is not None and self.loss_gated_curriculum['updates']<=5
            if completed == 0 or first_recovery or first_batch or early_balance or early_combat_only or early_v6 or early_batch256 or early_coordinate or early_replay or early_drift or early_gate or gate_advanced or step % 10 == 0:
                self.save(model, optimizer, metadata)
            control = (self.combat_v6_revision is not None
                and self.combat_v6_revision['updates'] in (100,250)
                and self.combat_v6_revision['updates'] not in self.combat_v6_revision['completed_control_validations'])
            coordinate_control = (self.coordinate_curriculum is not None
                and self.loss_gated_curriculum is None
                and self.coordinate_curriculum['updates'] in self.coordinate_plan['validation_curriculum_updates']
                and self.coordinate_curriculum['updates'] not in self.coordinate_curriculum['completed_validation_updates'])
            if step % 250 == 0 or control or coordinate_control:
                self.validate(model)
                if control:self.combat_v6_revision['completed_control_validations'].append(self.combat_v6_revision['updates'])
                self.save(model, optimizer, metadata)
            if step % 10 == 0 or gate_advanced: self.preview(model)
            if completed == 0 or first_recovery or first_batch or first_balance or step % 250 == 0:
                current = _hash_tensors({name: value for name, value in model.state_dict().items()
                    if name.startswith(('flow_renderer.', 'vjepa_backend.renderer.', 'vjepa_backend.feature_head.'))})
                if current != frozen: raise AssertionError('Frozen renderer/map/codec/appearance changed')
                self.state['frozen_verification'] = {'step': step, 'all_assets_bitexact': True, 'fingerprint': frozen}
                self.publish()
        self.state.update(status='qualified' if self.state['rssm_training']['quality_streak'] >= 2 else 'completed',
            quality_status='Actual combined state/combat gate passed twice' if self.state['rssm_training']['quality_streak'] >= 2
                else 'Control budget completed; game quality not qualified')
        self.save(model, optimizer, metadata)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'source-checkpoint', 'combat-data-root', 'navigation-data-root',
                 'navigation-original-root', 'renderer-checkpoint', 'sr'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--updates', type=int, default=0, help='0 continues until two complete quality passes')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--accumulation-steps',type=int,default=1)
    parser.add_argument('--seed', type=int, default=173)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--feedback-revision-plan', type=Path)
    parser.add_argument('--combat-recovery-plan', type=Path)
    parser.add_argument('--batch-revision-plan',type=Path)
    parser.add_argument('--preview-assets-plan',type=Path)
    parser.add_argument('--preview-assets-revision-plan',type=Path)
    parser.add_argument('--loss-balance-plan',type=Path)
    parser.add_argument('--combat-only-plan',type=Path)
    parser.add_argument('--combat-v6-plan',type=Path)
    parser.add_argument('--batch256-plan',type=Path)
    parser.add_argument('--coordinate-curriculum-plan', type=Path)
    parser.add_argument('--replay-curriculum-plan', type=Path)
    parser.add_argument('--drift-only-plan', type=Path)
    parser.add_argument('--loss-gated-curriculum-plan', type=Path)
    args = parser.parse_args(argv)
    if ((args.batch,args.accumulation_steps)not in((16,1),(32,4),(64,4))or args.updates<0
            or (args.batch,args.accumulation_steps)in((32,4),(64,4))and args.batch_revision_plan is None
            or (args.batch,args.accumulation_steps)==(64,4)and(args.batch256_plan is None or args.combat_v6_plan is None)
            or args.batch256_plan is not None and(args.batch,args.accumulation_steps)!=(64,4)):
        parser.error('Use legacy16x1, revised32x4, or explicitly measured64x4; updates0 means no finite budget')
    for key, value in vars(args).items():
        if isinstance(value, Path): setattr(args, key, value.resolve())
    from .vqgan_pipeline import process_alive
    args.run_dir.mkdir(parents=True, exist_ok=True)
    pid = args.run_dir/'training.pid'
    if pid.exists() and process_alive(int(pid.read_text())): parser.error('Combat worker already active')
    atomic_json(pid, os.getpid())
    experiment = RSSMCombatTraining(args)
    try:
        with KeepAwake(): experiment.run()
    except BaseException as error:
        experiment.state.update(status='failed', error=f'{type(error).__name__}: {error}'); experiment.publish(); raise


if __name__ == '__main__': main()
