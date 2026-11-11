"""Distill the archived JEPA dynamics into RSSM, then train from Source data.

Run checkpoints, data identity, phase and RNG clocks are independent from the
archived teacher. Renderer, map, VQGAN, Base and appearance assets stay frozen.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .gradient_safety import clip_finite_grad_norm_
from .models import load_world_checkpoint, load_sr_checkpoint, NeuralWorldModel, WorldModelConfig, STATE_SCHEMA, parameter_counts
from .power_guard import KeepAwake
from .rssm_migration import configure_rssm_training
from .staged_curriculum import StagedSequenceData
from .training_pipeline import atomic_json, now, replace_file


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_data(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete':
        raise ValueError('RSSM navigation data is incomplete')
    for record in manifest['records']:
        source = (root/record['relative_path']).resolve()
        if root not in source.parents or _digest(source) != record['sha256']:
            raise ValueError('RSSM navigation data content changed or escaped its directory')
    return _digest(root/'manifest.json')


def episode_names(root, *, split, stage=None):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    prefix = (stage+'/' if stage else '')+'rgb/'+split+'/'
    names = sorted(Path(record['relative_path']).name for record in manifest['records']
        if record['relative_path'].replace('\\', '/').startswith(prefix))
    if not names:
        raise ValueError('No pinned dataset episodes for '+prefix)
    return names


class RSSMTraining:
    def __init__(self, args):
        self.args, self.root = args, args.run_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.checkpoint = self.root/'checkpoints/rssm_world.pt'
        self.rng = np.random.default_rng(args.seed)
        self.device = torch.device(args.device)
        self.state = {'run_id': self.root.name, 'status': 'preparing', 'stage': 'rssm_dynamics',
            'model': 'DreamerV3-style RSSM + frozen Base / Rectified Flow / VQGAN',
            'teacher_world_architecture': 'dreamer_rssm', 'image_size': 32,
            'global_training_step': 0, 'current': 0, 'total': args.updates,
            'prediction_every': 10, 'target_rollout_steps': 450,
            'loss_protocol': 'rssm_source_dominant_distillation_then_own450_v1',
            'started_at': now(), 'quality_status': 'Navigation quality not qualified',
            'renderer_training': {'active': False, 'adversarial_active': False,
                'codec_frozen': True, 'dynamics_active': True},
            'training_protocol': {'persistent_object_slots': 1, 'state_dim': 20,
                'source_engine_feedback': False, 'validation_every': 100, 'preview_every': 10,
                'microbatch_size': args.batch, 'gradient_accumulation_steps': 1,
                'effective_batch_size': args.batch, 'auxiliary_fragments_per_update': args.batch,
                'source_posterior_is_separate_auxiliary_stream': True, 'ema_target_encoder': False},
            'rssm_training': {'phase': 'distill', 'phase_label': 'Дистилляция динамики в RSSM',
                'distillation_updates': args.distill_updates, 'distillation_completed': 0,
                'source_completed': 0, 'source_updates': args.updates-args.distill_updates,
                'total_updates': args.updates,
                'teacher_step': 1633, 'teacher_checkpoint': str(args.teacher),
                'ema_target_encoder': False, 'renderer_frozen': True,
                'validation_every': 100, 'preview_every': 10, 'rollout_horizon': 16,
                'distill_teacher_weight': .1, 'source_supervision_during_distillation': True,
                'short_draws': 0, 'auxiliary_draws': 0, 'coverage_long_draws': 0,
                'original_long_draws': 0, 'gameplay_qualified': False}}
        self.preview_model = None

    def publish(self):
        self.state['updated_at'] = now()
        atomic_json(self.root/'status.json', self.state)

    def save(self, model, optimizer, metadata):
        path = self.checkpoint
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: value for key, value in metadata.items()
            if key in ('rssm_migration', 'source_teacher_checkpoint')}
        payload.update(stage='world', state_schema=dict(STATE_SCHEMA), config=asdict(model.config),
            model=model.state_dict(), optimizer=optimizer.state_dict(),
            optimizer_parameter_names=[name for name, p in model.named_parameters() if p.requires_grad],
            training_steps=self.state['global_training_step'],
            global_training_step=self.state['global_training_step'],
            rssm_training=self.state['rssm_training'], run_state=self.state,
            rng_state=self.rng.bit_generator.state, torch_rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else [],
            plan_sha256=self.plan_sha, quality_status=self.state['quality_status'])
        temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
        torch.save(payload, temporary)
        replace_file(temporary, path)
        self.state['checkpoint'] = str(path)
        self.publish()

    def preview(self, model):
        from .rssm_preview_assets import preserve_preview_rng
        with preserve_preview_rng(self.device, self.rng):
            return self._preview(model)

    def initialize_preview_assets(self, model):
        """Prepare pinned inference assets without replacing training modules."""
        from .rssm_preview_assets import (resolve_preview_assets_plan, preview_assets_cache_key,
            prepare_preview_assets, synchronize_preview_dynamics)
        path = getattr(self.args, 'preview_assets_plan', None)
        if path is None:
            if getattr(self.args, 'preview_assets_revision_plan', None) is not None or self.state.get('preview_assets_revision'):
                raise ValueError('Preview revision requires its original immutable preview-assets plan')
            return None
        saved_revision = self.state.get('preview_assets_revision')
        initial, plan, revision = resolve_preview_assets_plan(path,
            getattr(self.args, 'preview_assets_revision_plan', None), saved_revision)
        existing_plan = getattr(self, '_preview_assets_plan_sha', None)
        explicit_activation = (revision is not None and saved_revision is None
            and existing_plan == initial['plan_sha256'])
        if existing_plan is not None and existing_plan != plan['plan_sha256'] and not explicit_activation:
            raise ValueError('Immutable preview-assets plan changed during this worker')
        key = preview_assets_cache_key(plan, model.config)
        if self.preview_model is None or getattr(self, '_preview_assets_key', None) != key:
            client, sr, sr_metadata, proof = prepare_preview_assets(model, plan, self.device)
            proof['initialized_at_world_step'] = self.state['global_training_step']
            self.preview_model, self.sr, self.sr_metadata = client, sr, sr_metadata
            self._preview_assets_key = key
            self._preview_assets_plan_sha = plan['plan_sha256']
            self._preview_assets_proof = proof
            self._preview_renderer_checkpoint = plan['renderer']['checkpoint']
            self._preview_sr_checkpoint = plan['sr']['checkpoint']
        else:
            synchronize_preview_dynamics(self.preview_model, model)
        self.state['preview_assets'] = {**self._preview_assets_proof,
            'current_world_step': self.state['global_training_step']}
        history = self.state.setdefault('preview_assets_history', [])
        for asset_plan in (initial, plan):
            if not any(item['plan_sha256'] == asset_plan['plan_sha256'] for item in history):
                history.append({key: asset_plan[key] for key in ('plan_path', 'plan_sha256', 'renderer', 'sr')})
        if revision is not None:
            self.state['preview_assets_revision'] = revision
            self.state['preview_assets']['revision'] = revision
        return self.state['preview_assets']

    def _preview(self, model):
        from .renderer_student import load_renderer_student
        from .vjepa_validation import save_autoregressive_preview
        pinned = self.initialize_preview_assets(model)
        renderer_checkpoint = self._preview_renderer_checkpoint if pinned else self.args.renderer_checkpoint
        selection_path = self.root.parent/'active_renderer.json'
        if not pinned and getattr(self.args, 'follow_active_renderer', True) and selection_path.exists():
            selection = json.loads(selection_path.read_text(encoding='utf-8'))
            candidate = Path(selection.get('renderer_checkpoint', ''))
            if selection.get('selected_for_experimental_play') is True and candidate.is_absolute() and candidate.is_file():
                renderer_checkpoint = candidate
        if not pinned and (self.preview_model is None or getattr(self, '_preview_renderer_checkpoint', None) != str(renderer_checkpoint)):
            # Use a separate inference instance so replacing Flow/Base/codec
            # cannot change the optimizer model or its saved asset identities.
            self.preview_model = NeuralWorldModel(WorldModelConfig(**asdict(model.config)))
            self.preview_model.load_state_dict(model.state_dict(), strict=True)
            self.preview_model.rssm_migration = model.rssm_migration
            if hasattr(model, 'dual_rssm_migration'):
                self.preview_model.dual_rssm_migration = model.dual_rssm_migration
            if hasattr(model, 'enemy_rssm_resize'):
                self.preview_model.enemy_rssm_resize = model.enemy_rssm_resize
            load_renderer_student(self.preview_model, renderer_checkpoint,
                allow_unqualified_codec=True)
            self.preview_model.to(self.device).eval().requires_grad_(False)
            self.sr, self.sr_metadata = load_sr_checkpoint(self.args.sr, 'cpu')
            self.sr.to(self.device).eval().requires_grad_(False)
            self._preview_renderer_checkpoint = str(renderer_checkpoint)
            self._preview_sr_checkpoint = str(self.args.sr)
        client = self.preview_model
        if not pinned:
            current = model.state_dict()
            client_state = client.state_dict()
            with torch.no_grad():
                for name, value in current.items():
                    if (not name.startswith(('flow_renderer.', 'base_student.'))
                            and name in client_state and client_state[name].shape == value.shape):
                        client_state[name].copy_(value)
        batch = self.short_val.batch(1, context=1, horizon=4, indices=np.asarray([[0, 0]]))
        with torch.inference_mode():
            fmap = client.encode_map(batch['map'])
            state = batch['states'][:, 0].float()
            memory = client.initialize_from_image(batch['context_frames'], batch['states'][:, :1], fmap)
            images = []
            for offset in range(4):
                result = client.transition(state, batch['actions'][:, offset], fmap, memory)
                state, memory = result['state'], result['memory']
                images.append(client.render(state, result['features'], fmap)[0].float())
            generated = torch.stack(images)
        def rgb(value):
            return value.detach().float().clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        preview = {'context': rgb(batch['context_frames'][0]),
            'source': rgb(batch['target_frames'][0]), 'generated': rgb(generated),
            'generated_float': generated.cpu(), 'metadata': {
                'step': self.state['global_training_step'], 'every': 10, 'context': 1, 'horizon': 4,
                'kind': 'rssm_free_autoregressive_future', 'model_kind': 'RSSM',
                'component_phase': self.state['rssm_training']['phase'],
                'renderer_checkpoint': str(renderer_checkpoint),
                'sr_checkpoint': self._preview_sr_checkpoint,
                'preview_assets': pinned,
                'generated_at': now(), 'source_held_out': True, 'split': 'val',
                'future_source_feedback': False, 'generated_finite': bool(torch.isfinite(generated).all()),
                'caption': 'RSSM: четыре собственных перехода; замороженный нейрорендер ученика, один Euler шаг'+(' · updated NPR '+str(pinned['renderer_refinement_updates'])+' / SR '+str(pinned['sr_training_steps']) if pinned else ''),
                'row_labels': {'context': 'CONTEXT: one observed frame | RSSM',
                    'source': 'SOURCE: future measurement targets only',
                    'generated': 'RSSM MODEL: prior-only autonomous future | frozen student renderer'+(' | updated NPR '+str(pinned['renderer_refinement_updates'])+' SR '+str(pinned['sr_training_steps']) if pinned else '')}}}
        self.state['prediction_preview'] = save_autoregressive_preview(preview,
            self.root/'preview.png', step=self.state['global_training_step'], every=10,
            sr_model=self.sr, sr_metadata=self.sr_metadata)
        self.publish()

    def validate(self, model):
        from .motion_recovery_validation import evaluate_motion_recovery
        step = self.state['global_training_step']
        directory = self.root/'validation'/f'step_{step:06d}'
        self.state['validation_progress'] = {'status': 'running', 'step': step,
            'scope': '128 short action responses and 16 autonomous450 trajectories plus default-pose regression'}
        self.publish()
        report = evaluate_motion_recovery(model, self.args.data_root/'short/rgb/val',
            self.args.data_root/'long/rgb/val', directory, context=1, count=128,
            long_count=16, batch_size=16, seed=self.args.seed,
            default_regression_data=self.args.data_root/'short/rgb/train', prior_tape_path=self.args.prior_tape)
        report.update(step=step, architecture='dreamer_rssm',
            phase=self.state['rssm_training']['phase'], scope='State/memory navigation; image quality not qualified')
        revision = self.state['rssm_training'].get('contact_revision')
        if revision:
            from .motion_recovery_validation import _short
            if not hasattr(self, 'contact_val'):
                self.contact_val = StagedSequenceData(Path(revision['dataset_directory'])/'short/rgb/val',
                    self.device, mode='stream', cache_bytes=32*1024**2,
                    episode_paths=episode_names(revision['dataset_directory'], split='val', stage='short'))
            contact, arrays = _short(model, self.contact_val, context=1,
                count=revision.get('additional_validation_count', 256), batch_size=16, seed=self.args.seed)
            np.savez_compressed(directory/'contact_generalization_states.npz', **arrays)
            atomic_json(directory/'contact_generalization_validation.json', contact)
            report.update(contact_generalization=contact, contact_revision_id=revision['fingerprint'],
                contact_protocol_version=2, passed=bool(report['passed'] and contact['passed']))
            revision.update(last_validation_step=step,
                last_validation_report=str(directory/'motion_recovery_validation.json'))
        continuation = self.state['rssm_training'].get('continuation')
        if continuation:
            from .rssm_continuation import (evaluate_heldout_idle, evaluate_one_frame450,
                complete_quality_gate, record_continuation_validation)
            idle = evaluate_heldout_idle(model, self.contact_val, seed=self.args.seed)
            atomic_json(directory/'heldout_idle.json', idle)
            one = evaluate_one_frame450(model, self.args.data_root/'long/rgb/val',
                directory/'one_frame450', seed=self.args.seed)
            report.update(heldout_idle=idle, one_frame450=one,
                contact_coverage_with_idle_supplement=bool(idle['samples'] >= 32),
                validation_every=250, continuation_id=continuation['fingerprint'])
            gate = complete_quality_gate(report, continuation['quality_thresholds'])
            report.update(passed=gate['passed'], complete_quality_gate=gate,
                scope='Held-out navigation: short128/contact256 +32idle +16x450 with one/two observations; renderer/combat not evaluated')
            record_continuation_validation(continuation, report, step=step)
        navigation = self.state['rssm_training'].get('navigation_revision')
        if navigation:
            from .rssm_navigation_revision import consider_navigation_candidate
            assets = {'renderer_checkpoint': str(self.args.renderer_checkpoint),
                      'sr_checkpoint': str(self.args.sr), 'vqgan_frozen': True}
            active_renderer = self.root.parent/'active_renderer.json'
            if active_renderer.exists():
                assets['active_renderer'] = json.loads(active_renderer.read_text(encoding='utf-8'))
            candidate = {'stage': 'world', 'state_schema': dict(STATE_SCHEMA),
                'config': asdict(model.config), 'model': model.state_dict(),
                'training_steps': step, 'global_training_step': step,
                'rssm_migration': model.rssm_migration,
                'rssm_training': self.state['rssm_training'],
                'quality_status': self.state['quality_status']}
            selected = consider_navigation_candidate(self.root, candidate, report,
                expected_fingerprints=navigation['validation_fingerprints'], assets_metadata=assets)
            self.state['navigation_best'] = selected
            navigation.update(baseline_pending=False, last_validation_step=step,
                last_validation_passed=report['passed'])
            report.update(navigation_revision_id=navigation['fingerprint'],
                          navigation_best=selected)
        atomic_json(directory/'motion_recovery_validation.json', report)
        atomic_json(self.root/'rssm_validation_latest.json', report)
        self.state['rssm_validation'] = report
        self.state['validation_progress'].update(status='complete')
        self.state['rssm_training'].update(last_validation_step=step,
            last_validation_passed=report['passed'], last_validation_report=str(directory/'motion_recovery_validation.json'))
        self.publish()

    def run(self):
        from .rssm_objectives import rssm_dynamics_backward
        args = self.args
        torch.set_num_threads(4)
        if self.device.type == 'cuda':
            torch.cuda.init()
        torch.manual_seed(args.seed)
        manifest_sha = verify_data(args.data_root)
        original_sha = verify_data(args.original_data)
        continuation_arguments = {'continue_until_quality', 'continuation_contact_root', 'navigation_revision_root'}
        plan = {'version': 1, **{key: str(value.resolve()) if isinstance(value, Path) else value
            for key, value in vars(args).items() if key not in continuation_arguments}, 'dataset_manifest_sha256': manifest_sha,
            'original_manifest_sha256': original_sha, 'distillation_teacher_weight': .1,
            'own_state_horizon': 450, 'tbptt_chunk': 16, 'source_future_feedback': False,
            'source_teacher_validation': str(args.teacher.parent.parent/'rssm_migration_validation/teacher_g1633/motion_recovery_validation.json')}
        plan_path = self.root/'training_plan.json'
        if plan_path.exists() and json.loads(plan_path.read_text(encoding='utf-8')) != plan:
            raise ValueError('RSSM training plan changed on resume')
        if not plan_path.exists():
            atomic_json(plan_path, plan)
        self.plan_sha = _digest(plan_path)
        model, metadata = load_world_checkpoint(self.checkpoint if self.checkpoint.exists() else args.initial, 'cpu')
        model.to(self.device).float()
        policy = configure_rssm_training(model)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=.01)
        if self.checkpoint.exists():
            if metadata['plan_sha256'] != self.plan_sha:
                raise ValueError('RSSM checkpoint does not belong to this data/training plan')
            names = [name for name, p in model.named_parameters() if p.requires_grad]
            if names != metadata['optimizer_parameter_names']:
                raise ValueError('RSSM optimizer parameter identity changed')
            optimizer.load_state_dict(metadata['optimizer'])
            self.state = metadata['run_state']
            self.rng.bit_generator.state = metadata['rng_state']
            torch.set_rng_state(metadata['torch_rng_state'])
            if self.device.type == 'cuda':
                torch.cuda.set_rng_state_all(metadata['cuda_rng_state'])
            if getattr(args, 'navigation_revision_root', None) is not None:
                from .rssm_navigation_protocol import verify_restored_boundary
                audit = verify_restored_boundary(model, optimizer, self.rng, metadata)
                atomic_json(self.root/'navigation_revision'/f"resume_verification_g{metadata['training_steps']}.json", audit)
        else:
            self.state['rssm_training']['teacher_step'] = metadata['rssm_migration']['source_global_training_step']
        migration = metadata['rssm_migration']
        model.rssm_migration = migration
        from .rssm_contact_revision import bind_contact_revision
        self.contact_revision = bind_contact_revision(self.state, self.root/'contact_revision_plan.json',
            base_plan_sha=self.plan_sha, step=self.state['global_training_step'])
        from .rssm_continuation import (bind_continuation, commit_continuation,
            training_should_continue, validation_due)
        bound = bind_continuation(self.state, self.root/'continuation_plan.json',
            enabled=getattr(args, 'continue_until_quality', False),
            dataset_root=getattr(args, 'continuation_contact_root', None),
            base_plan_sha=self.plan_sha, contact_plan_path=self.root/'contact_revision_plan.json',
            checkpoint_path=self.checkpoint, step=self.state['global_training_step'])
        self.continuation = bound[0] if bound else None
        continuation_paths = bound[1] if bound else None
        from .rssm_navigation_protocol import bind_navigation_revision
        self.branch_protocol = None
        if getattr(args, 'navigation_revision_root', None) is not None:
            from .rssm_contact_branches import normalize_contact_branch_protocol
            self.branch_protocol = normalize_contact_branch_protocol({})
        navigation_bound = bind_navigation_revision(self.state, self.root/'navigation_revision_plan.json',
            dataset_root=getattr(args, 'navigation_revision_root', None), base_plan_sha=self.plan_sha,
            contact_plan_path=self.root/'contact_revision_plan.json',
            continuation_plan_path=self.root/'continuation_plan.json',
            checkpoint_path=self.checkpoint, step=self.state['global_training_step'],
            branch_protocol=self.branch_protocol)
        self.navigation_revision = navigation_bound[0] if navigation_bound else None
        navigation_paths = navigation_bound[1] if navigation_bound else None
        # Do not retain a second copy of all world tensors/Adam moments in CPU
        # metadata during the background run.
        metadata = {key: metadata[key] for key in ('rssm_migration', 'source_teacher_checkpoint') if key in metadata}
        gc.collect()
        counts = parameter_counts(model)
        self.state['model_parameters'] = {'teacher_total': counts['world_total'],
            'teacher_trainable': policy['dynamics_trainable'], 'teacher_frozen': policy['frozen'],
            'ema_target_total': 0, 'dynamics_including_map': counts['dynamics_including_map'],
            'target_dynamics_total': 25_000_000, 'renderer_total': counts['renderer']}
        self.state['rssm_training'].update(parameter_count=counts['dynamics_including_map'],
            deterministic_dim=model.config.rssm_deter_dim, stoch=model.config.rssm_stoch,
            classes=model.config.rssm_classes, memory_size=getattr(model.vjepa_backend,'memory_size',model.core.memory_size),
            dynamics_trainable=policy['dynamics_trainable'])
        self.state['keep_awake'] = {'system_required': True, 'display_required': False, 'screen_lock_allowed': True}
        self.short = StagedSequenceData(args.data_root/'short/rgb/train', self.device, mode='stream', cache_bytes=64*1024**2)
        self.short_val = StagedSequenceData(args.data_root/'short/rgb/val', self.device, mode='stream', cache_bytes=32*1024**2)
        coverage_long = StagedSequenceData(args.data_root/'long/rgb/train', self.device, mode='stream', cache_bytes=64*1024**2)
        original_long = StagedSequenceData(args.original_data/'rgb/train', self.device, mode='stream', cache_bytes=64*1024**2)
        if self.contact_revision:
            from .rssm_contact_revision import ContactAuxiliarySampler
            if self.navigation_revision:
                from .rssm_navigation_protocol import NavigationContactSampler
                self.contact_short = StagedSequenceData(Path(self.navigation_revision['dataset_directory'])/'short/rgb/train',
                    self.device, mode='stream', cache_bytes=64*1024**2, episode_paths=navigation_paths)
                self.contact_sampler = NavigationContactSampler(self.contact_short, self.short,
                    self.navigation_revision['dataset_directory'])
            elif self.continuation:
                self.contact_short = StagedSequenceData(Path(self.continuation['dataset_directory'])/'short/rgb/train',
                    self.device, mode='stream', cache_bytes=64*1024**2, episode_paths=continuation_paths)
                self.contact_sampler = ContactAuxiliarySampler(self.contact_short, self.short)
            else:
                self.contact_short = StagedSequenceData(Path(self.contact_revision['dataset_directory'])/'short/rgb/train',
                    self.device, mode='stream', cache_bytes=64*1024**2,
                    episode_paths=episode_names(self.contact_revision['dataset_directory'], split='train', stage='short'))
                self.contact_sampler = ContactAuxiliarySampler(self.contact_short, self.short)
            probe_path = self.root/'diagnostics/contact_revision/gradient_probe.json'
            if probe_path.exists():
                self.state['rssm_gradient_probe'] = json.loads(probe_path.read_text(encoding='utf-8'))
            self.state['training_protocol'].update(contact_protocol_version=2,
                auxiliary_new_fragments=12, auxiliary_old_replay_fragments=4,
                autonomous_source_feedback=False, contact_data_train_only_own_visits=True)
        if self.navigation_revision:
            self.state['training_protocol'].update(contact_data_train_only_own_visits=False,
                contact_data_profile='TRAIN Source corners, wall replay, legacy replay',
                own_contact_memory_source='current autonomous RSSM prefix')
        self.state['status'] = 'running'
        self.publish()
        self.save(model, optimizer, metadata)
        revision_baseline = bool((self.contact_revision and self.contact_revision.get('baseline_pending'))
            or (self.continuation and self.continuation.get('baseline_pending'))
            or (self.navigation_revision and self.navigation_revision.get('baseline_pending')))
        if 'rssm_validation' not in self.state or revision_baseline:
            self.validate(model)
            self.preview(model)
            if self.contact_revision and self.contact_revision.get('baseline_pending'):
                self.contact_revision.update(baseline_pending=False,
                    baseline_step=self.state['global_training_step'],
                    baseline_report=str(self.root/'validation'/f"step_{self.state['global_training_step']:06d}"/'motion_recovery_validation.json'))
            self.save(model, optimizer, metadata)
        teacher = None
        started = time.perf_counter()
        completed_at_start = self.state['global_training_step']
        while training_should_continue(self.state, args.updates):
            step = self.state['global_training_step']+1
            if (self.root/'stop_training').exists():
                self.state['status'] = 'stopped'
                self.save(model, optimizer, metadata)
                return
            phase = 'distill' if self.state['rssm_training']['distillation_completed'] < args.distill_updates else 'source'
            if phase == 'distill' and teacher is None:
                with torch.random.fork_rng(devices=[]):
                    teacher, teacher_metadata = load_world_checkpoint(args.teacher, 'cpu')
                teacher.to(self.device).float().eval().requires_grad_(False)
                del teacher_metadata
                gc.collect()
            if phase == 'source' and teacher is not None:
                del teacher
                teacher = None
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            configure_rssm_training(model)
            optimizer.zero_grad(set_to_none=True)
            protocol = self.state['rssm_training']
            latent_mode = 'sample'
            if self.contact_revision and self.contact_revision.get('latent_training_cycle'):
                cycle = self.contact_revision['latent_training_cycle']
                latent_mode = cycle[self.contact_revision['updates'] % len(cycle)]
                model.core.train(latent_mode == 'sample')
            navigation_setting = None
            if self.navigation_revision:
                from .rssm_navigation_revision import navigation_revision_setting
                navigation_setting = navigation_revision_setting(self.navigation_revision['revision_updates'])
                latent_mode = navigation_setting['latent_mode']
                model.core.train(latent_mode == 'sample')
            protocol.update(phase=phase, phase_label='Дистилляция динамики в RSSM' if phase == 'distill'
                else 'RSSM: обучение по Source, 450 собственных переходов',
                rollout_horizon=16 if phase == 'distill' else 450)
            source_index = protocol['source_completed']
            context = 1 if phase == 'distill' else 2
            observed = 1 if phase == 'distill' else (1, 1, 2, 2)[source_index % 4]
            source_name = 'short' if phase == 'distill' else 'coverage' if source_index % 2 == 0 else 'original'
            if navigation_setting:
                context = navigation_setting['context']
                observed = navigation_setting['observed_context']
                source_name = navigation_setting['long_pool']
            pool = self.short if phase == 'distill' else coverage_long if source_name == 'coverage' else original_long
            batch_options = {}
            if navigation_setting:
                batch_options['indices'] = np.column_stack([
                    self.rng.integers(len(pool.episodes), size=args.batch),
                    np.zeros(args.batch, dtype=np.int64)])
            batch = pool.batch(args.batch, context=context, horizon=16 if phase == 'distill' else 450,
                rng=self.rng, **batch_options)
            # Each auxiliary fragment begins at the legal shared initial pose.
            # In particular, the Source/default start action is actually seen.
            contact_selected = None
            if self.contact_revision:
                auxiliary, contact_selected = self.contact_sampler.sample(self.contact_revision, self.rng)
            else:
                indices = np.column_stack([self.rng.integers(len(self.short.episodes), size=args.batch),
                    np.zeros(args.batch, dtype=np.int64)])
                auxiliary = self.short.batch(args.batch, context=1, horizon=16, rng=self.rng, indices=indices)
            def progress(processed, horizon):
                self.state['long_training_progress'] = {'step': step, 'transitions_processed': processed,
                    'horizon': horizon, 'tbptt_chunk': 16, 'optimizer_updated': False,
                    'source_future_feedback': False, 'numerical_state_resets': 0}
                self.publish()
            metrics = rssm_dynamics_backward(model, batch, phase=phase, teacher=teacher,
                auxiliary_batch=auxiliary, context=context, observed_context=observed,
                auxiliary_context=1, auxiliary_observed_context=1, chunk_size=16,
                teacher_weight=.1, amp=nullcontext, progress_callback=progress,
                expected_horizon=16 if phase == 'distill' else 450,
                **({'contact_protocol':self.contact_revision['loss_settings']} if self.contact_revision else {}),
                **({'contact_branch_protocol': self.branch_protocol} if self.navigation_revision else {}))
            metrics['world_grad_norm'] = float(clip_finite_grad_norm_(
                [(name, p) for name, p in model.named_parameters() if p.requires_grad],
                1., context=f'RSSM optimizer step {step}'))
            optimizer.step()
            self.state['global_training_step'] = step
            self.state['current'] = step
            protocol['distillation_completed' if phase == 'distill' else 'source_completed'] += 1
            protocol['short_draws' if phase == 'distill' else source_name+'_long_draws'] += args.batch
            protocol['auxiliary_draws'] += args.batch
            if self.contact_revision:
                from .rssm_contact_revision import commit_contact_revision
                commit_contact_revision(self.contact_revision, step=step, selected=contact_selected)
                metrics.update(contact_revision_id=self.contact_revision['fingerprint'],
                    contact_revision_update=self.contact_revision['updates'],
                    contact_new_draws=self.contact_revision['new_auxiliary_draws'],
                    contact_old_replay_draws=self.contact_revision['old_auxiliary_draws'])
            if self.continuation:
                commit_continuation(self.continuation, step=step)
                metrics.update(continuation_id=self.continuation['fingerprint'],
                    continuation_update=self.continuation['updates'],
                    continuation_new_draws=self.continuation['new_auxiliary_draws'],
                    continuation_old_replay_draws=self.continuation['old_auxiliary_draws'])
            if self.navigation_revision:
                from .rssm_navigation_revision import commit_revision_update
                commit_revision_update(self.navigation_revision, step=step,
                    source_completed=protocol['source_completed'], long_batch_size=args.batch)
                metrics.update(navigation_revision_id=self.navigation_revision['fingerprint'],
                    navigation_revision_update=self.navigation_revision['revision_updates'],
                    main_numeric_context=context, main_observed_context=observed,
                    main_window_start=0, main_long_pool=source_name,
                    auxiliary_corner_fragments=8, auxiliary_singlewall_fragments=4,
                    auxiliary_legacy_fragments=4,
                    navigation_schedule_cycle=navigation_setting['cycle_index'])
            self.state['long_training_progress'].update(optimizer_updated=True, status='complete')
            metrics.update(step=step, global_training_step=step, stage='rssm_dynamics', timestamp=now(),
                component_phase=phase, microbatch_size=args.batch, gradient_accumulation_steps=1,
                effective_batch_size=args.batch, auxiliary_fragments=args.batch,
                source_future_feedback=False, ema_updated=False, dynamics_updated=True,
                image_objectives_active=False, loss_protocol=self.state['loss_protocol'],
                rssm_latent_training_mode=latent_mode)
            with (self.root/'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(metrics, ensure_ascii=False, allow_nan=False)+'\n')
            self.state.update(loss=metrics['train_loss'], latest_metrics=metrics,
                message=protocol['phase_label'],
                eta_seconds=None if self.continuation else int((time.perf_counter()-started)/(step-completed_at_start)*(args.updates-step)),
                eta_scope='Training until two complete quality passes; completion time unknown' if self.continuation else 'Remaining optimizer budget; quality completion time unknown')
            self.publish()
            handoff = phase == 'distill' and protocol['distillation_completed'] == args.distill_updates
            contact_probe_due = bool(self.contact_revision and self.contact_revision['updates'] in (4, 20))
            if validation_due(self.state, step) or (not self.continuation and (step == 4 or handoff or contact_probe_due)):
                optimizer.zero_grad(set_to_none=True)
                self.validate(model)
                self.save(model, optimizer, metadata)
            if step % 10 == 0 or step == 1:
                self.preview(model)
            if step % 10 == 0 or step == 1 or phase == 'source' and protocol['source_completed'] == 1 or self.contact_revision and self.contact_revision['updates'] == 1:
                self.save(model, optimizer, metadata)
            del batch, auxiliary
        if self.continuation:
            self.state.update(status='qualified', quality_status='Held-out navigation state/memory passed; renderer/combat not qualified',
                message='Two consecutive complete navigation quality checks passed')
            self.state['rssm_training']['gameplay_qualified'] = False
            self.save(model, optimizer, metadata)
            return
        self.state['status'] = 'awaiting_user'
        self.state['message'] = 'Бюджет RSSM завершён; качество определяется последней валидацией'
        self.save(model, optimizer, metadata)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'initial', 'teacher', 'data-root', 'original-data', 'renderer-checkpoint', 'sr', 'prior-tape'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--updates', type=int, default=2000)
    parser.add_argument('--distill-updates', type=int, default=200)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=173)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--continue-until-quality', action='store_true')
    parser.add_argument('--continuation-contact-root', type=Path)
    parser.add_argument('--navigation-revision-root', type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.distill_updates < args.updates or args.batch != 16 or args.lr <= 0:
        parser.error('Use batch16 and a positive distillation budget below the total update budget')
    if args.continue_until_quality != (args.continuation_contact_root is not None):
        parser.error('Use --continue-until-quality and --continuation-contact-root together')
    if args.continue_until_quality and args.lr != 1e-4:
        parser.error('Continuation preserves the existing learning rate1e-4')
    if args.navigation_revision_root is not None and not args.continue_until_quality:
        parser.error('--navigation-revision-root requires the restored --continue-until-quality run')
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    from .vqgan_pipeline import process_alive
    args.run_dir.mkdir(parents=True, exist_ok=True)
    pid_path = args.run_dir/'training.pid'
    if pid_path.exists():
        previous_pid = int(pid_path.read_text(encoding='ascii'))
        if previous_pid != os.getpid() and process_alive(previous_pid):
            parser.error('This RSSM training run is already active')
    atomic_json(pid_path, os.getpid())
    experiment = RSSMTraining(args)
    try:
        with KeepAwake() as awake:
            if args.device == 'cuda' and not awake.active:
                raise RuntimeError('Cannot keep background training awake: '+str(awake.error))
            experiment.run()
    except BaseException as error:
        experiment.state.update(status='failed', error=str(error), failed_at=now())
        experiment.publish()
        raise


if __name__ == '__main__':
    main()
