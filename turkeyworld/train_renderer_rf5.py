"""Continue the five-role renderer with expanded experts and five Euler steps."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .distill_renderer import _digest
from .gradient_safety import clip_finite_grad_norm_
from .models import load_world_checkpoint
from .power_guard import KeepAwake
from .renderer_combat_data import CombatRendererFrames
from .renderer_small_enemy_data import SmallEnemyBalancedFrames
from .renderer_far_wall_data import FarWallFrames
from .renderer_geometry_data import GeometryWallFrames
from .renderer_moe_five_objectives import FiveProfiledFrames
from .renderer_moe_five_phase import original_renderer_configuration
from .renderer_moe_objectives import moe_conditions
from .renderer_native_phase import _same, restore_renderer_native_rng
from .renderer_rf5_experts import upgrade_rf5_experts, rf5_expert_metadata
from .renderer_rf5_phase import (PROTOCOL, renderer_rf5_protocol, configure_rf5_training,
    build_rf5_optimizer, rf5_frozen_fingerprint, rf5_parameter_counts)
from .renderer_rf5_objectives import rf5_refinement_loss
from .renderer_rf5_validation import evaluate_rf5, save_rf5_preview
from .renderer_student import load_renderer_student, _hash_tensors
from .training_pipeline import atomic_json, now, replace_file


def durable_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    replace_file(temporary, path)


class RF5Training:
    def __init__(self, args):
        self.args, self.root = args, args.run_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        self.state = dict(run_id=self.root.name, status='initializing',
            stage='renderer_rf5_expanded_refinement', loss_protocol=PROTOCOL,
            current=0, global_training_step=0, total=args.steps, prediction_every=10,
            validation_every=100, image_size=32, gameplay_qualified=False,
            model='Five expanded RF Transformer experts; frozen Base / VQGAN / world')

    def publish(self, **values):
        self.state.update(values, updated_at=now(), elapsed_seconds=time.perf_counter() - self.started)
        atomic_json(self.root / 'status.json', self.state)

    def prepare(self):
        a = self.args
        if a.device == 'cuda':
            torch.cuda.init()
        torch.set_num_threads(a.threads)
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.publish(stage_label='Loading the complete stopped RF parent')
        self.parent = torch.load(a.parent_checkpoint, map_location='cpu', weights_only=False)
        self.parent_bundle = torch.load(a.renderer_checkpoint, map_location='cpu', weights_only=False)
        self.inherited = original_renderer_configuration(self.parent['configuration'])
        model, metadata = load_world_checkpoint(a.world, 'cpu')
        feature_step = metadata['training_steps']
        # Original teacher assets are needed to validate the nested parent;
        # the active working renderer already contains the compact codec.
        self.source_world = deepcopy(model).eval().requires_grad_(False)
        load_renderer_student(model, a.renderer_checkpoint, allow_unqualified_codec=True)
        if (not _same(model.base_student.state_dict(), self.parent['base'])
                or not _same(model.flow_renderer.state_dict(), self.parent['flow'])):
            raise ValueError('RF5 parent inference and full training snapshots differ')
        self.parent_identity = deepcopy(self.parent['identity'])
        if (self.parent_identity['feature_world_sha256'] != _digest(a.world)
                or self.parent_identity['feature_world_step'] != feature_step):
            raise ValueError('RF5 changed the frozen causal feature world')
        upgrade_rf5_experts(model.flow_renderer.output_moe)
        model.flow_renderer.steps = a.euler_steps
        self.model = model.float().to(a.device)
        configure_rf5_training(self.model)
        self.optimizer, adoption = build_rf5_optimizer(self.model, self.parent)
        self.frozen = rf5_frozen_fingerprint(self.model)
        self.publish(stage_label='Indexing the unchanged Source data and wall cohorts')
        raw = CombatRendererFrames(self.inherited['combat_data_root'],
            self.inherited['appearance_data_root'], 'train', seed=a.seed)
        val = CombatRendererFrames(self.inherited['combat_data_root'],
            self.inherited['appearance_data_root'], 'val', seed=a.seed + 1,
            exclude_scene_keys=raw.scene_keys)
        self.train = SmallEnemyBalancedFrames(raw)
        self.far = FarWallFrames(self.train, candidate_count=2048, seed=2718)
        self.geometry = GeometryWallFrames(self.train, candidate_count=8192, seed=31415)
        self.panel = list(self.parent_identity['panel'])
        self.geometry_panel = list(self.parent_identity['geometry_wall_panel'])
        identities = dict(train=raw.identity, val=val.identity,
            distance_sampling=self.train.identity['distance_sampling'],
            far_wall_aux_pool=self.far.identity, geometry_wall_aux_pool=self.geometry.identity,
            panel_fingerprint=val.panel_fingerprint(self.panel),
            geometry_wall_fingerprint=val.panel_fingerprint(self.geometry_panel))
        for key, value in identities.items():
            if not _same(self.parent_identity[key], value):
                raise ValueError('RF5 changed pinned Source data or panel: ' + key)
        self.profiled = dict(main=FiveProfiledFrames(self.train),
            far=FiveProfiledFrames(self.far, self.train.rows),
            geometry=FiveProfiledFrames(self.geometry, self.train.rows), val=FiveProfiledFrames(val))
        self.configuration = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
        self.identity = dict(parent_identity=self.parent_identity,
            parent_checkpoint_sha256=_digest(a.parent_checkpoint),
            parent_renderer_sha256=_digest(a.renderer_checkpoint),
            feature_world_sha256=_digest(a.world), feature_world_step=feature_step,
            panel_fingerprint=identities['panel_fingerprint'],
            geometry_panel_fingerprint=identities['geometry_wall_fingerprint'],
            protocol=renderer_rf5_protocol(a.euler_steps), expert_architecture=rf5_expert_metadata(self.model.flow_renderer.output_moe))
        self.phase = dict(protocol=PROTOCOL, configuration=self.configuration, identity=self.identity,
            adoption=adoption, frozen_fingerprint=self.frozen,
            parent_base_training_steps=self.parent['base_training_steps'],
            parent_renderer_training_steps=self.parent['renderer_training_steps'])
        phase_path = self.root / 'phase_plan.json'
        if phase_path.exists():
            if json.loads(phase_path.read_text(encoding='utf-8')) != self.phase:
                raise ValueError('The immutable RF5 phase plan changed')
        else:
            atomic_json(phase_path, self.phase)
        self.plan_sha = _digest(phase_path)
        self.checkpoint = self.root / 'checkpoints/renderer_last.pt'
        self.step = 0
        saved = self.parent
        if self.checkpoint.exists() and not a.probe:
            saved = torch.load(self.checkpoint, map_location='cpu', weights_only=False)
            if (saved['plan_sha256'] != self.plan_sha or saved['phase'] != self.phase
                    or saved['optimizer_parameter_names'] != adoption['optimizer_parameter_names']
                    or not 0 <= saved['step'] <= a.steps):
                raise ValueError('RF5 resume changed its phase, optimizer names or counter')
            self.model.flow_renderer.load_state_dict(saved['flow'], strict=True)
            self.optimizer.load_state_dict(saved['optimizer'])
            self.step = saved['step']
            if not _same(self.model.base_student.state_dict(), saved['base']):
                raise ValueError('RF5 resume changed frozen Base')
        for stream, key in ((self.train, 'sampler'), (self.far, 'far_wall_aux_sampler'),
                            (self.geometry, 'geometry_wall_aux_sampler')):
            stream.restore_sampler(saved[key])
            if not _same(stream.sampler_state(), saved[key]):
                raise AssertionError('RF5 Source sampler continuation was not exact')
        rng_proof = restore_renderer_native_rng(saved, device=a.device)
        counts = rf5_parameter_counts(self.model)
        self.training = dict(protocol=PROTOCOL, plan_sha256=self.plan_sha,
            feature_world_sha256=self.identity['feature_world_sha256'], feature_world_step=feature_step,
            local_step=self.step, local_budget=a.steps,
            parent_renderer_training_steps=self.parent['renderer_training_steps'],
            parent_base_training_steps=self.parent['base_training_steps'],
            solver='euler', solver_steps=a.euler_steps, experts=5, parameters_per_expert=503072,
            private_width=256, private_heads=8, private_hidden=416,
            base_frozen=True, codec_frozen=True, private_opacity_frozen=True,
            parameters=dict(base=counts['base'], rf=counts['rf_total_without_codec'],
                expanded_experts=2515360, added=counts['added_parameters']),
            main_batch=16, far_aux_batch=4, geometry_aux_batch=8,
            main_router_reduction='mean_over_actual_Euler_velocity_calls')
        self.publish(renderer_rf5_training=self.training, model_parameters=counts,
            validation_panel_fingerprint=self.identity['panel_fingerprint'],
            renderer_training_steps=self.parent['renderer_training_steps'] + self.step,
            base_training_steps=self.parent['base_training_steps'],
            renderer_training=dict(active=True, base_active=False, rf_active=True,
                codec_frozen=True, rssm_frozen=True, map_frozen=True,
                private_opacity_frozen=True, gan_active=False),
            current=self.step, global_training_step=self.step)
        atomic_json(self.root / 'initialization_verification.json', dict(passed=True,
            rng=rng_proof, adoption=adoption, all_three_Source_samplers_exact=True,
            base_clock_frozen=self.parent['base_training_steps'], solver_steps=a.euler_steps))

    def save(self):
        durable_save(self.checkpoint, dict(base=self.model.base_student.state_dict(),
            flow=self.model.flow_renderer.state_dict(), optimizer=self.optimizer.state_dict(),
            optimizer_parameter_names=self.phase['adoption']['optimizer_parameter_names'],
            step=self.step, renderer_training_steps=self.parent['renderer_training_steps'] + self.step,
            base_training_steps=self.parent['base_training_steps'], phase=self.phase,
            plan_sha256=self.plan_sha, frozen_fingerprint=self.frozen,
            sampler=self.train.sampler_state(), far_wall_aux_sampler=self.far.sampler_state(),
            geometry_wall_aux_sampler=self.geometry.sampler_state(),
            rng_torch=torch.get_rng_state(), rng_numpy=np.random.get_state(),
            rng_python=random.getstate(), rng_cuda=torch.cuda.get_rng_state_all()
                if self.args.device == 'cuda' else []))

    def update(self):
        configure_rf5_training(self.model)
        self.optimizer.zero_grad(set_to_none=True)
        records, total = {}, 0.
        started = time.perf_counter()
        for kind, size, weight in (('main', 16, 1.), ('far', 4, .25), ('geometry', 8, 1.)):
            batch = self.profiled[kind].batch(size)
            value, metrics = rf5_refinement_loss(self.model, batch, kind=kind,
                inherited=self.inherited, checkpoint_velocity=True)
            (weight * value).backward()
            total += weight * float(value.detach())
            records[kind] = metrics
            del value, batch
        private_gradients = {str(i): float(sum(p.grad.detach().float().abs().sum()
            for p in expert.private_transformer.parameters() if p.grad is not None))
            for i, expert in enumerate(self.model.flow_renderer.output_moe.experts)}
        norm = float(clip_finite_grad_norm_(self.model.named_parameters(), context='Expanded RF5'))
        self.optimizer.step()
        self.step += 1
        if rf5_frozen_fingerprint(self.model) != self.frozen:
            raise AssertionError('RF5 changed a frozen Base/world/map/codec/private-opacity tensor')
        if not np.isfinite(total):
            raise FloatingPointError('Nonfinite RF5 total training loss')
        return dict(step=self.step, loss_protocol=PROTOCOL, protocol=PROTOCOL,
            plan_sha256=self.plan_sha, feature_world_sha256=self.identity['feature_world_sha256'],
            solver_steps=self.args.euler_steps, train_loss=total, main_loss=records['main']['train_loss'],
            main=records['main'], far=records['far'], geometry=records['geometry'],
            gradient_norm=norm, private_transformer_gradient_abs=private_gradients,
            optimizer_update_seconds=time.perf_counter() - started,
            renderer_training_steps=self.parent['renderer_training_steps'] + self.step,
            base_training_steps=self.parent['base_training_steps'],
            **{key: records['main'][key] for key in ('renderer_rgb_loss', 'renderer_moe_role_loss',
                'renderer_moe_balance_loss', 'renderer_moe_role_weight', 'renderer_moe_balance_weight',
                'renderer_moe_heads')}, timestamp=now())

    def preview(self):
        from .renderer_geometry_phase import geometry_preview_arrays
        panel = self.panel
        ids = []
        for category in ('large', 'partial', 'distant', 'empty_navigation'):
            found = next((i for i in panel if self.profiled['val'].rows[i]['category'] == category), None)
            if found is not None and found not in ids:
                ids.append(found)
        ids += [i for i in panel if i not in ids][:4 - len(ids)]
        arrays = geometry_preview_arrays(self.model, self.profiled['val'], ids, conditions_fn=moe_conditions)
        arrays['categories'] = [self.profiled['val'].rows[i]['category'] for i in ids]
        save_rf5_preview(arrays, self.root / 'preview.png', self.step, solver_steps=self.args.euler_steps)
        meta = dict(step=self.step, every=10, kind=f'Source_RF{self.args.euler_steps}_GT_query',
            loss_protocol=PROTOCOL, plan_sha256=self.plan_sha,
            feature_world_sha256=self.identity['feature_world_sha256'],
            panel_fingerprint=self.identity['panel_fingerprint'], solver_steps=self.args.euler_steps,
            source_held_out=True, renderer_only=True, world_rollout=False,
            future_source_rgb_in_prior=False, future_source_pose_in_prior=False,
            native_resolution=[32, 32], row_count=4, generated_at=now(),
            caption=f'Source32 / frozen Base / expanded RF{self.args.euler_steps} / frozen codec; held-out GT-pose queries')
        atomic_json(self.root / 'preview.json', meta)
        self.publish(prediction_preview=meta, preview_url='/preview.png')

    def validate(self, baseline, parent_rf1):
        report, _ = evaluate_rf5(self.model, self.profiled['val'], self.panel,
            self.geometry_panel, step=self.step, batch_size=8, baseline=baseline,
            parent_rf1=parent_rf1, inherited=self.inherited)
        report.update(plan_sha256=self.plan_sha,
            feature_world_sha256=self.identity['feature_world_sha256'])
        atomic_json(self.root / 'validation' / f'step_{self.step:06d}.json', report)
        atomic_json(self.root / 'validation.json', report)
        with (self.root / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(step=self.step, loss_protocol=PROTOCOL,
                plan_sha256=self.plan_sha, feature_world_sha256=self.identity['feature_world_sha256'],
                solver_steps=self.args.euler_steps, val_loss=report['val_loss'],
                panel_fingerprint=self.identity['panel_fingerprint'],
                source_held_out=True, renderer_only=True, world_rollout=False,
                future_source_rgb_in_prior=False, future_source_pose_in_prior=False,
                validation_loss_excludes_far_geometry_auxiliary=True,
                validation_objective_matches_main_training=True), allow_nan=False) + '\n')
        from .renderer_rf5_bundle import export_rf5_renderer
        export_rf5_renderer(self.model, self.parent_bundle,
            self.root / 'checkpoints/renderer_for_play.pt',
            training_steps=self.parent['renderer_training_steps'] + self.step,
            local_updates=self.step, base_training_steps=self.parent['base_training_steps'],
            metrics=dict(source_validation=report, source_quality_passed=report['passed']),
            provenance=dict(parent_source_sha256=_digest(self.args.renderer_checkpoint), plan_sha256=self.plan_sha),
            source_model=self.source_world)
        self.publish(validation=report, renderer_rf5_validation=report)
        self.preview()
        return report

    def run(self):
        self.prepare()
        if self.args.probe:
            rows = [self.update() for _ in range(3)]
            if not all(row['main']['router_velocity_calls'] == self.args.euler_steps for row in rows):
                raise AssertionError('Expanded RF did not collect all Euler router graphs')
            if not all(value > 0 for value in rows[-1]['private_transformer_gradient_abs'].values()):
                raise AssertionError('An expanded RF expert received no deep gradient')
            self.save()
            proof = dict(passed=True, discarded_optimizer_updates=3, production_training_updates=0,
                frozen_assets_exact=True, rows=rows, parameters=rf5_parameter_counts(self.model),
                peak_cuda_GiB=torch.cuda.max_memory_allocated() / 2**30 if self.args.device == 'cuda' else 0.,
                average_seconds_per_update=sum(row['optimizer_update_seconds'] for row in rows) / 3)
            atomic_json(self.root / 'gpu_warmstart.json', proof)
            self.publish(status='completed', message='Three discarded RF5 preflight updates passed')
            print(json.dumps(dict(probe_passed=True, report=str(self.root / 'gpu_warmstart.json'),
                seconds=proof['average_seconds_per_update'], peak_GiB=proof['peak_cuda_GiB'])), flush=True)
            return
        reference = json.loads(self.args.parent_validation.read_text(encoding='utf-8'))
        baseline_path = self.root / 'validation_initial_rf5.json'
        baseline = json.loads(baseline_path.read_text(encoding='utf-8')) if baseline_path.exists() else None
        if self.step > 0 and baseline is None:
            raise ValueError('A resumed RF phase cannot invent a pre-training baseline from trained weights')
        if baseline is None:
            self.publish(stage_label='Fixed256 RF5 baseline before any optimizer update')
            baseline = self.validate(None, reference)
            atomic_json(baseline_path, baseline)
        self.save()
        self.preview()
        self.publish(status='running', stage_label='Expanded RF5; Source-supervised details and walls')
        while self.step < self.args.steps:
            if (self.root / 'stop_training').exists():
                self.save(); self.publish(status='stopped'); return
            row = self.update()
            with (self.root / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            self.training['local_step'] = self.step
            self.save()
            self.publish(current=self.step, global_training_step=self.step, loss=row['train_loss'],
                latest_metrics=row, renderer_rf5_training=self.training,
                renderer_training_steps=row['renderer_training_steps'], base_training_steps=row['base_training_steps'],
                frozen_verification=dict(step=self.step, all_frozen_parameters_buffers_bitexact=True))
            if self.step % 100 == 0:
                self.validate(baseline, reference)
            elif self.step % 10 == 0:
                self.preview()
        self.save()
        self.publish(status='completed', message='RF5 phase budget completed; quality is evaluated separately')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'world', 'parent-checkpoint', 'renderer-checkpoint', 'parent-validation'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=173)
    p.add_argument('--euler-steps', type=int, choices=(3,5), default=3)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--probe', action='store_true')
    a = p.parse_args(argv)
    if a.steps < 1 or a.threads < 1:
        p.error('steps and threads must be positive')
    for name, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, name, value.resolve())
    from .vqgan_pipeline import process_alive
    a.run_dir.mkdir(parents=True, exist_ok=True)
    pid = a.run_dir / 'training.pid'
    if pid.exists() and process_alive(int(pid.read_text())):
        p.error('RF5 worker is already running')
    pid.write_text(str(os.getpid()), encoding='ascii')
    experiment = RF5Training(a)
    try:
        with KeepAwake():
            experiment.run()
    except BaseException as error:
        experiment.publish(status='failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
