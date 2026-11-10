"""Resume compatible world weights in a single-enemy, quality-gated curriculum.

The source engine is an offline data producer only. Static-enemy snapshots
train Base/Flow/GAN without calling the dynamics network. The final phase
also trains uninterrupted own-state trajectories with truncated gradients.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .aligned_training import aligned_short_rollout_loss
from .engine import BEHAVIOR_VERSION, ENTITY_SLOTS, STATE_DIM
from .jepa_training_utils import EMATargetEncoder
from .models import load_world_checkpoint
from .power_guard import KeepAwake
from .staged_curriculum import (StagedCurriculum, StagedSequenceData, StagedTrainingPools,
    renderer_snapshot_loss)
from .training_pipeline import atomic_json, now
from .vjepa_training import VJEPAExperiment


class StagedWorldExperiment(VJEPAExperiment):
    PHASE = 'teacher175vq_world'

    def save(self, model, optimizer, phase, step, metrics, extra=None):
        fields = {'staged_curriculum': self.curriculum.sync(self.global_step),
            'behavior_version': BEHAVIOR_VERSION,
            'initialization': self.state.get('initialization'),
            'quality_status': 'Unqualified until the held-out 450-transition curriculum gate passes.'}
        fields.update(extra or {})
        result = super().save(model, optimizer, phase, step, metrics, fields)
        atomic_json(self.root/'staged_curriculum_state.json', self.curriculum.state)
        return result

    def publish_curriculum(self):
        self.state['staged_curriculum'] = self.curriculum.sync(self.global_step)
        self.state.setdefault('training_protocol', {}).update(persistent_object_slots=ENTITY_SLOTS,
            state_dim=STATE_DIM, validation_every=100, preview_every=10,
            stage_quality_gates=True, source_engine_feedback=False)
        atomic_json(self.root/'staged_curriculum_state.json', self.curriculum.state)
        atomic_json(self.root/'status.json', self.state)

    def evaluate_stage(self, model):
        from .staged_validation import evaluate_staged_model
        stage = self.curriculum.stage['id']
        info = self.curriculum.plan['datasets'][stage]
        root = self.root/'staged_validation'/stage/f'step_{self.global_step:06d}'
        self.state['staged_validation_progress'] = {'status': 'running', 'stage': stage,
            'step': self.global_step, 'every': 100, 'report_directory': str(root)}
        self.publish_curriculum()
        report = evaluate_staged_model(model, info['directory'], stage, self.device,
            context=self.args.context, horizon=29, count=32, batch_size=4,
            seed=self.args.seed, output_dir=root,
            thresholds=self.curriculum.plan.get('thresholds', {}).get(stage))
        report.setdefault('report_path', str(root/'report.json'))
        atomic_json(root/'report.json', report)
        self.curriculum.record_validation(report, self.global_step)
        self.state['staged_validation_progress']['status'] = 'complete'
        self.state['staged_curriculum'] = self.curriculum.state
        self.state['evaluation'].update(passed=self.curriculum.state['final_qualified'],
            status='passed' if self.curriculum.state['final_qualified'] else 'pending')
        atomic_json(self.root/'staged_validation_latest.json', report)
        self.metric({'stage': self.PHASE, 'step': self.global_step,
            'staged_stage': stage, 'staged_validation': report})
        self.publish_curriculum()
        model.train()
        return report

    def preview_stage(self, model):
        from .vjepa_validation import evaluate_autoregressive, save_autoregressive_preview
        from .staged_preview import static_preview
        stage = self.curriculum.stage['id']
        info = self.curriculum.plan['datasets'][stage]
        self.state['prediction_progress'] = {'status': 'running', 'step': self.global_step}
        self.publish_curriculum()
        if self.curriculum.stage['renderer_only']:
            contact = static_preview(model, Path(info['directory'])/'snapshots/val', self.global_step)
        else:
            data = self.validation_pool(stage)
            panel = data.fixed_panel(count=4, context=self.args.context, horizon=4, seed=self.args.seed)
            _, contact = evaluate_autoregressive(model, data, panel, context=self.args.context,
                horizon=4, batch_size=1, thresholds=self.quality_thresholds(), seed=self.args.seed,
                step=self.global_step, preview_every=10)
        contact['metadata'].update(staged_stage=stage, scope='stage_training_preview',
            loss_protocol=self.state['loss_protocol'], quality_gate_evaluated=False)
        self.state['prediction_preview'] = save_autoregressive_preview(contact, self.root/'preview.png',
            step=self.global_step, every=10, **self._prediction_sr_options(contact))
        self.state['prediction_progress'] = {'status': 'complete', 'step': self.global_step}
        self.publish_curriculum()
        model.train()

    def validation_pool(self, stage):
        if not hasattr(self, '_stage_validation_pools'):
            self._stage_validation_pools = {}
        if stage not in self._stage_validation_pools:
            info = self.curriculum.plan['datasets'][stage]
            self._stage_validation_pools[stage] = StagedSequenceData(Path(info['directory'])/'rgb/val',
                self.device, mode='stream', cache_bytes=64*1024**2,
                episode_paths=info.get('val_filenames'))
        return self._stage_validation_pools[stage]

    def image_refinement(self, model, endpoint):
        from .object_losses import balanced_image_l1, foreground_contrastive_loss
        base, state, target = (endpoint[name].detach() for name in ('base', 'state', 'target'))
        mask, background = endpoint['object_mask'], endpoint['background']
        with self.amp(cache_enabled=False):
            generated = model.flow_renderer.sample(base, state, checkpoint_velocity=True,
                map_condition=endpoint['map_condition'])
            rgb = .5*balanced_image_l1(generated, target, mask)
            foreground = .5*foreground_contrastive_loss(generated, target, background, mask)
        gan, metrics = self._adversarial_loss(generated, target, base, state,
            accumulation_steps=self.accumulation_steps(), microstep=0,
            counterfactual=background, counterfactual_mask=mask,
            counterfactual_weight=1., counterfactual_min_fraction=.005)
        metrics.update(sampled_rgb=float(rgb.detach()), foreground_sampled=float(foreground.detach()))
        return rgb+foreground+gan, metrics

    def long_backward(self, model, pools, batch_size):
        """One optimizer update, carried state/memory over up to450 transitions.

        Only graph history is truncated every16 transitions. Each chunk reads
        the same initialization map and the preceding own state and memory.
        No future RGB or state initializes a later chunk.
        """
        from .train import state_loss
        from .trajectory_objectives import trajectory_state_losses
        from .gradient_accumulation import MetricMean
        updates = self.curriculum.state['stage_updates']
        horizon = 64 if updates < 100 else 128 if updates < 200 else 450
        pool = pools.get('full_game')
        batch = pool.batch(batch_size, context=self.args.context, horizon=horizon, rng=self.rng)
        with self.amp(cache_enabled=False):
            fmap = model.encode_map(batch['map'])
            memory = model.initialize_from_image(batch['context_frames'], batch['states'][:, :self.args.context], fmap)
        current = batch['states'][:, self.args.context-1]
        sums = MetricMean()
        for first in range(0, horizon, self.args.chunk):
            last = min(horizon, first+self.args.chunk)
            if first:
                current, memory = current.detach(), memory.detach()
                with self.amp(cache_enabled=False):
                    fmap = model.encode_map(batch['map'])
            own_previous = current.detach()
            with self.amp(cache_enabled=False):
                states = []
                for offset in range(first, last):
                    result = model.transition(current, batch['actions'][:, offset], fmap, memory)
                    current, memory = result['state'], result['memory']
                    states.append(current)
                predicted = torch.stack(states, 1)
                target = batch['target_states'][:, first:last]
                state_value = 10*state_loss(predicted.flatten(0, 1), target.flatten(0, 1))
                trajectory, diagnostics = trajectory_state_losses(predicted,
                    batch['previous_states'][:, first], target, batch['map'],
                    offsets=range(1, last-first+1), previous_prediction=own_previous)
                value = state_value+sum(trajectory.values())
            if not bool(torch.isfinite(value)):
                raise FloatingPointError('Nonfinite long curriculum loss')
            (value*((last-first)/horizon)).backward()
            sums.add({'train_loss': float(value.detach()), 'state': float(state_value.detach()),
                **{key: float(v.detach()) for key, v in trajectory.items()}})
        return {**sums.mean(), 'long_rollout': True, 'long_horizon': horizon,
            'tbptt_chunk': self.args.chunk, 'long_gt_feedback': False,
            'long_memory_reset_count': 0, 'long_optimizer_updates': 1}

    def run(self):
        torch.set_num_threads(4)
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        phase = self.PHASE
        checkpoint = self.root/'checkpoints'/f'{phase}.pt'
        source = checkpoint if checkpoint.exists() else self.args.source
        model, metadata = load_world_checkpoint(source, 'cpu')
        if model.config.entity_slots != 1 or model.config.image_size != 32:
            raise ValueError('Staged teacher requires the single-enemy32px profile')
        model.to(self.device).train().requires_grad_(True)
        self.ema = EMATargetEncoder(model.core.encoder, self.args.ema).to(self.device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
            lr=self.args.lr, weight_decay=.01)
        completed = self.restore(model, optimizer, phase) if checkpoint.exists() else 0
        self.global_step = completed
        self.fragments_seen = getattr(self, 'fragments_seen', 0)
        previous = metadata.get('staged_curriculum') if checkpoint.exists() else None
        self.curriculum = StagedCurriculum(self.args.curriculum_manifest,
            start_step=self.global_step, previous=previous)
        pools = StagedTrainingPools(self.curriculum, None, device=self.device,
            seed=self.args.seed, cache_bytes=self.args.cache_mb*1024**2)
        self._setup_adversarial(phase)
        self.state.update(model='Single-enemy V-JEPA2-AC / VQGAN / CFM / GAN',
            teacher_size='175m_vqgan', image_size=32, global_training_step=completed,
            loss_protocol='single_enemy_staged_v1_aligned_v5',
            initialization={'mode': 'compatible_weights_from_eight_enemy_teacher',
                'source_checkpoint': str(self.args.source), 'world_weights_reused': True,
                'optimizer_reset_for_state_schema_migration': True, 'codec_reused_frozen': True},
            enemy_policy={'source_deterministic': True, 'version': BEHAVIOR_VERSION,
                'enemy_slots': ENTITY_SLOTS, 'state_dim': STATE_DIM,
                'source_spawn_rule': 'anchor=(game_tick//90)%8; scan next legal anchor; no hash',
                'neural_behavior': 'learned_deterministic_transition', 'neural_spawn': 'learned_argmax'},
            renderer_training={'active': True, 'adversarial_active': True, 'codec_frozen': True},
            plateau_policy={'enabled': False, 'reason': 'quality_gated_stages'},
            teacher_plateau={'enabled': False, 'reason': 'quality_gated_stages'},
            data_growth={'auto_expand': False, 'explicit_staged_corpus': True},
            prediction_every=10, target_rollout_steps=450,
            optimizer_budget={'start_optimizer_step': 0, 'stop_optimizer_step': self.args.updates,
                'limit_updates': self.args.updates, 'status': 'training'})
        self._publish_model_sizes(model)
        self.state['model_parameters']['target_teacher_total'] = 175000000
        atomic_json(self.root/'experiment_config.json', {key: str(v) if isinstance(v, Path) else v
            for key, v in vars(self.args).items()})
        self.publish_curriculum()
        metrics = {}
        if self.curriculum.validation_due(self.global_step):
            self.evaluate_stage(model)
        self.preview_stage(model)
        self.save(model, optimizer, phase, completed, metrics)
        started = time.perf_counter()
        for step in range(completed+1, self.args.updates+1):
            self.check_stop(model, optimizer, phase, step-1, metrics)
            self.curriculum.sync(self.global_step)
            optimizer.zero_grad(set_to_none=True)
            stage = self.curriculum.stage
            # Full-length own-state training alternates with short joint image,
            # event and motion updates, preserving both sources of supervision.
            long_update = stage['id'] == 'full_game' and step % 4 == 0
            self.state['renderer_training'].update(active=not long_update,
                adversarial_active=not long_update, dynamics_active=not stage['renderer_only'],
                reason='long_own_state_tbptt' if long_update else
                    'static_pose_base_flow_gan' if stage['renderer_only'] else 'joint_dynamics_flow_matching_gan')
            if long_update:
                metrics = self.long_backward(model, pools, self.args.batch)
            else:
                batch = pools.batch(self.args.batch, self.global_step, self.args.context, self.rng)
                loss_function = renderer_snapshot_loss if stage['renderer_only'] else aligned_short_rollout_loss
                with self.amp(cache_enabled=False):
                    loss, metrics, endpoint = loss_function(model, self.ema, batch,
                        context=self.args.context, render=True, object_region_boost=4., detach_flow_base=True)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite staged world/Flow objective')
                loss.backward()
                total = float(loss.detach())
                refinement, gan_metrics = self.image_refinement(model, endpoint)
                if not bool(torch.isfinite(refinement)):
                    raise FloatingPointError('Nonfinite staged GAN objective')
                refinement.backward()
                total += float(refinement.detach())
                metrics.update(gan_metrics)
                if not stage['renderer_only']:
                    from .combat_render_training import combat_image_objective
                    from .spawn_render_training import spawn_image_objective
                    for objective in (combat_image_objective, spawn_image_objective):
                        with self.amp(cache_enabled=False):
                            extra, values = objective(model, endpoint['combat_payload'])
                        if not bool(torch.isfinite(extra)):
                            raise FloatingPointError('Nonfinite staged combat/spawn image objective')
                        extra.backward()
                        total += float(extra.detach())
                        metrics.update(values)
                metrics['train_loss'] = total
                del loss, endpoint, refinement, batch
            flow = [p for p in model.flow_renderer.parameters() if p.requires_grad]
            flow_ids = {id(p) for p in flow}
            world = [p for p in model.parameters() if p.requires_grad and id(p) not in flow_ids]
            torch.nn.utils.clip_grad_norm_(world, 1., error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(flow, 1., error_if_nonfinite=True)
            optimizer.step()
            self.ema.update(model.core.encoder)
            self.global_step += 1
            self.fragments_seen += self.args.batch
            self.publish_accumulation(1, 'updated')
            self.curriculum.sync(self.global_step)
            metrics.update(microbatch_size=self.args.batch, gradient_accumulation_steps=1,
                effective_batch_size=self.args.batch, fragments_seen=self.fragments_seen,
                loss_protocol=self.state['loss_protocol'], staged_stage=stage['id'],
                staged_horizon=metrics.get('long_horizon', self.curriculum.horizon(self.global_step-1)),
                renderer_only=stage['renderer_only'], enemy_slots=1)
            if 'flow' in metrics:
                metrics['loss_without_flow'] = metrics['train_loss']-metrics['flow']
            for index in range(5):
                if metrics.get(f'flow_t{index}_count', 0):
                    metrics[f'flow_t{index}_velocity'] = metrics[f'flow_t{index}_sum']/metrics[f'flow_t{index}_count']
            self.metric({'stage': phase, 'step': step, 'global_training_step': step, **metrics})
            seconds = (time.perf_counter()-started)/max(1, step-completed)
            self.update(phase, step, self.args.updates, self.curriculum.stage['label'],
                global_training_step=step, loss=metrics['train_loss'],
                eta_seconds=round((self.args.updates-step)*seconds),
                eta_scope='Remaining update budget; stage quality completion time unknown')
            if step % 100 == 0:
                optimizer.zero_grad(set_to_none=True)
                self.evaluate_stage(model)
                self.save(model, optimizer, phase, step, metrics)
            if self.curriculum.state['final_qualified']:
                self.state['status'] = 'completed'
                self.save(model, optimizer, phase, step, metrics)
                break
            if step % 10 == 0:
                self.preview_stage(model)
            if step == completed+1 or step % self.args.save_every == 0:
                self.save(model, optimizer, phase, step, metrics)
            self.publish_curriculum()
        else:
            self.state['status'] = 'awaiting_user'
            self.state['optimizer_budget']['status'] = 'exhausted'
            self.save(model, optimizer, phase, self.global_step, metrics)
        self.state['renderer_training']['active'] = False
        self.update(phase, self.global_step, self.args.updates, self.curriculum.stage['label'],
            global_training_step=self.global_step)
        self.publish_curriculum()


def main(argv=None):
    from .train_500m import build_parser
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument('--curriculum-manifest', type=Path, required=True)
    parser.add_argument('--sr-checkpoint', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=5182)
    parser.set_defaults(batch=16, gradient_accumulation_steps=1, target_seconds=15,
        preview_every=10, save_every=10, teacher_until_quality=False,
        adversarial_weight=.005, adversarial_feature_weight=.01,
        adversarial_warmup=32, adversarial_ramp=128, image_size=32)
    args = parser.parse_args(argv)
    if args.gradient_accumulation_steps != 1 or args.batch < 1 or args.updates < 1:
        parser.error('This curriculum uses accumulation1 and a positive microbatch/update budget')
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    from .vqgan_pipeline import process_alive
    pid = args.run_dir/'training.pid'
    if (pid.exists() and int(pid.read_text()) != os.getpid()
            and process_alive(int(pid.read_text()))):
        parser.error('This staged teacher is already running')
    experiment = StagedWorldExperiment(args)
    atomic_json(pid, os.getpid())
    try:
        with KeepAwake() as guard:
            experiment.state['power_guard'] = {'active': guard.active, 'error': guard.error,
                'display_required': False, 'screen_lock_allowed': True}
            experiment.run()
    except Exception as error:
        experiment.state['status'] = 'failed'
        experiment.update('error', experiment.global_step, args.updates,
            'Staged training failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
