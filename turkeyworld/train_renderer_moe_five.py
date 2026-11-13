"""Continue the completed renderer with five hierarchical private Transformer roles."""
from copy import deepcopy
import argparse
import gc
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .distill_renderer import _digest, _save
from .gradient_safety import clip_finite_grad_norm_
from .models import load_world_checkpoint
from .power_guard import KeepAwake
from .renderer_combat_data import CombatRendererFrames
from .renderer_small_enemy_data import SmallEnemyBalancedFrames
from .renderer_far_wall_data import FarWallFrames
from .renderer_geometry_data import GeometryWallFrames
from .renderer_enemy_validation import evaluate_enemy_renderer, save_enemy_preview
from .renderer_geometry_objectives import geometry_wall_report, geometry_preservation_assessment
from .renderer_geometry_phase import geometry_preview_arrays
from .renderer_moe import enable_renderer_moe, configure_renderer_moe, migrate_renderer_moe_five
from .renderer_moe_objectives import moe_conditions
from .renderer_moe_five_objectives import FiveProfiledFrames, moe_five_refinement_loss
from .renderer_moe_five_phase import (renderer_moe_five_protocol, identity_digest,
    build_renderer_moe_five_optimizer, restore_renderer_moe_five_parent,
    original_renderer_configuration, renderer_five_parameter_groups, validate_renderer_moe_five_resume)
from .renderer_native_phase import _same, restore_renderer_native_rng
from .renderer_refinement import frozen_fingerprint, export_refined_bundle
from .renderer_student import load_renderer_student, _hash_tensors
from .staged_validation import _isolated_validation
from .training_pipeline import atomic_json, now


class RendererMoEFiveTraining:
    def __init__(self, args):
        self.args, self.root = args, args.run_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        self.state = dict(run_id=self.root.name, status='initializing', stage='renderer_moe_five_refinement',
            loss_protocol=renderer_moe_five_protocol()['name'], current=0, global_training_step=0,
            total=args.steps, prediction_every=args.preview_every, validation_every=args.validate_every,
            model='Frozen current RSSM features + Base/RF + five hierarchical private token Transformer experts',
            image_size=32, world_training=False, gameplay_qualified=False)

    def publish(self, **values):
        self.state.update(values, updated_at=now(), elapsed_seconds=time.perf_counter() - self.started)
        atomic_json(self.root / 'status.json', self.state)

    def run(self):
        args = self.args
        if args.device == 'cuda': torch.cuda.init()
        torch.set_num_threads(args.threads)
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        saved = torch.load(args.parent_checkpoint, map_location='cpu', weights_only=False)
        parent_configuration = deepcopy(saved['configuration'])
        inherited = original_renderer_configuration(parent_configuration)
        protocol = renderer_moe_five_protocol()
        self.publish(stage_label='Loading frozen current-world snapshot and completed renderer parent')
        source, metadata = load_world_checkpoint(args.world, 'cpu')
        source_step = int(metadata['training_steps']); del metadata
        source.eval().requires_grad_(False)
        if source.config.navigation_only:
            raise ValueError('Renderer five-role MoE feature snapshot must retain the current combat profile')
        working = deepcopy(source)
        load_renderer_student(working, args.renderer_checkpoint, allow_unqualified_codec=True)
        enable_renderer_moe(working)
        working.base_student.load_state_dict(saved['base'], strict=True)
        working.flow_renderer.load_state_dict(saved['flow'], strict=True)
        migrate_renderer_moe_five(working)
        working.float().to(args.device)
        configure_renderer_moe(working)
        optimizer = build_renderer_moe_five_optimizer(working, saved)
        frozen = frozen_fingerprint(working)
        parent_identity = saved['identity']
        self.publish(stage_label='Indexing unchanged real Source cohorts')
        raw_train = CombatRendererFrames(inherited['combat_data_root'], inherited['appearance_data_root'],
            'train', seed=args.seed)
        validation = CombatRendererFrames(inherited['combat_data_root'], inherited['appearance_data_root'],
            'val', seed=args.seed + 1, exclude_scene_keys=raw_train.scene_keys)
        if ({r['seed'] for r in raw_train.files} & {r['seed'] for r in validation.files}
                or {r['sha256'] for r in raw_train.files} & {r['sha256'] for r in validation.files}):
            raise ValueError('Renderer five-role MoE Source TRAIN/VAL identities overlap')
        train = SmallEnemyBalancedFrames(raw_train)
        far = FarWallFrames(train, candidate_count=2048, seed=2718)
        geometry = GeometryWallFrames(train, candidate_count=8192, seed=31415)
        panel = list(parent_identity['panel'])
        geometry_panel = list(parent_identity['geometry_wall_panel'])
        profiled = dict(main=FiveProfiledFrames(train), far=FiveProfiledFrames(far, train.rows),
                        geometry=FiveProfiledFrames(geometry, train.rows), val=FiveProfiledFrames(validation))
        identity = deepcopy(parent_identity)
        identity.update(parent_identity_sha256=identity_digest(parent_identity),
                        renderer_moe_protocol=protocol)
        if (identity['world_sha256'] != _digest(args.world)
                or identity['feature_world_sha256'] != _digest(args.world)
                or identity['feature_world_step'] != source_step
                or identity['renderer_sha256'] != _digest(args.renderer_checkpoint)
                or identity['codec_sha256'] != _hash_tensors(working.flow_renderer.codec.state_dict())):
            raise ValueError('Five-role continuation changed the pinned feature world/bootstrap/codec')
        if args.frozen_sr_checkpoint is not None:
            if identity.get('frozen_sr_asset', {}).get('sha256') != _digest(args.frozen_sr_checkpoint):
                raise ValueError('Five-role continuation changed its frozen SR asset')
        # Check the actual freshly indexed data, not only inherited JSON fields.
        for key, value in dict(train=raw_train.identity, val=validation.identity,
                distance_sampling=train.identity['distance_sampling'], far_wall_aux_pool=far.identity,
                geometry_wall_aux_pool=geometry.identity,
                panel_fingerprint=validation.panel_fingerprint(panel),
                geometry_wall_fingerprint=validation.panel_fingerprint(geometry_panel)).items():
            if not _same(parent_identity[key], value):
                raise ValueError('Renderer five-role MoE changed Source cohort/panel identity: ' + key)
        configuration = {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(args).items()}
        configuration['inherited_parent_configuration'] = parent_configuration
        checkpoint = self.root / 'checkpoints/renderer_enemy_last.pt'
        phase_path = self.root / 'renderer_moe_five_phase_plan.json'
        if checkpoint.exists():
            resume = torch.load(checkpoint, map_location='cpu', weights_only=False)
            phase = json.loads(phase_path.read_text(encoding='utf-8'))
            validate_renderer_moe_five_resume(resume, phase, identity=identity, configuration=configuration,
                frozen=frozen, plan_sha256=_digest(phase_path),
                parent_checkpoint_sha256=_digest(args.parent_checkpoint))
            working.base_student.load_state_dict(resume['base'], strict=True)
            working.flow_renderer.load_state_dict(resume['flow'], strict=True)
            optimizer.load_state_dict(resume['optimizer'])
            for stream, key in ((train, 'sampler'), (far, 'far_wall_aux_sampler'),
                                (geometry, 'geometry_wall_aux_sampler')):
                stream.restore_sampler(resume[key])
            restore_renderer_native_rng(resume, device=args.device)
            step = resume['step']
            del resume
        else:
            phase = restore_renderer_moe_five_parent(saved, working, optimizer, train, far, geometry,
                identity, configuration, frozen, parent_checkpoint_sha256=_digest(args.parent_checkpoint))
            phase.update(configuration=configuration, parent_checkpoint=str(args.parent_checkpoint),
                         status='initialized', parent_feature_world_changed_explicitly=False)
            if phase_path.exists() and json.loads(phase_path.read_text(encoding='utf-8')) != phase:
                raise ValueError('An incompatible immutable renderer five-role MoE phase already exists')
            if not phase_path.exists(): atomic_json(phase_path, phase)
            step = 0
        plan_sha = _digest(phase_path)
        parent_step = int(saved['step'])
        renderer_start, base_start = int(saved['renderer_training_steps']), int(saved['base_training_steps'])
        training = dict(protocol=protocol['name'], plan_sha256=plan_sha,
            parent_checkpoint_sha256=_digest(args.parent_checkpoint), parent_step=parent_step,
            feature_world_sha256=identity['feature_world_sha256'], feature_world_step=source_step,
            roles=protocol['roles'], routing=protocol['routing'], experts_per_head=5,
            role_schema=protocol['role_schema'], near_far_boundary_tiles=5.,
            near_far_comparison='near_strict_less_than', source_role_masks_loss_only=True,
            parent_renderer_training_steps=renderer_start, parent_base_training_steps=base_start,
            expert_transformer_blocks=1, expert_widths=protocol['expert_widths'],
            local_step=step, local_budget=args.steps, source_held_out_validation=True,
            frozen=dict(RSSMs=True, map=True, codec=True, private_opacity=True),
            parameters=dict(base=sum(p.numel() for p in working.base_student.parameters()),
                rf1=sum(p.numel() for p in working.flow_renderer.parameters() if p.requires_grad)
                    + sum(p.numel() for p in working.flow_renderer.geometry_opacity.parameters()),
                moe_added=phase['added_parameters'], moe_tensors=phase['added_parameter_tensors']))
        self.publish(renderer_moe_training=training, renderer_moe_phase=phase,
            validation_panel_fingerprint=identity['panel_fingerprint'], model_parameters=training['parameters'],
            renderer_training=dict(active=True, base_active=True, rf_active=True, codec_frozen=True,
                rssm_frozen=True, map_frozen=True, private_opacity_frozen=True, gan_active=False),
            current=step, global_training_step=step, renderer_training_steps=renderer_start + step,
            base_training_steps=base_start + step)
        del saved; gc.collect()

        def save():
            _save(checkpoint, dict(base=working.base_student.state_dict(), flow=working.flow_renderer.state_dict(),
                optimizer=optimizer.state_dict(), optimizer_parameter_names=renderer_five_parameter_groups(working)[1],
                step=step, parent_step=parent_step,
                renderer_training_steps=renderer_start + step, base_training_steps=base_start + step,
                identity=identity, configuration=configuration, frozen_fingerprint=frozen,
                renderer_moe_phase=phase, renderer_moe_plan_sha256=plan_sha,
                sampler=train.sampler_state(), far_wall_aux_sampler=far.sampler_state(),
                geometry_wall_aux_sampler=geometry.sampler_state(), rng_torch=torch.get_rng_state(),
                rng_numpy=np.random.get_state(), rng_python=random.getstate(),
                rng_cuda=torch.cuda.get_rng_state_all() if args.device == 'cuda' else []))

        baseline_path = self.root / 'validation_initial.json'
        baseline = json.loads(baseline_path.read_text(encoding='utf-8')) if baseline_path.exists() else None
        preview_ids = []
        for category in ('large', 'partial', 'distant', 'empty_navigation'):
            found = next((i for i in panel if validation.rows[i]['category'] == category), None)
            if found is not None and found not in preview_ids: preview_ids.append(found)
        preview_ids += [i for i in panel if i not in preview_ids][:4 - len(preview_ids)]

        def preview(at):
            # Dedicated four-case forward accepts all-empty panels too.
            arrays = geometry_preview_arrays(working, profiled['val'], preview_ids, conditions_fn=moe_conditions)
            arrays['categories'] = [validation.rows[i]['category'] for i in preview_ids]
            save_enemy_preview(arrays, self.root / 'preview.png', at)
            meta = dict(step=at, every=args.preview_every, kind='Source_enemy_GT_query',
                loss_protocol=protocol['name'], renderer_moe_plan_sha256=plan_sha,
                feature_world_sha256=identity['feature_world_sha256'], panel_fingerprint=identity['panel_fingerprint'],
                source_held_out=True, renderer_only=True, world_rollout=False,
                future_source_rgb_in_prior=False, future_source_pose_in_prior=False,
                gameplay_qualified=False, native_resolution=[32, 32], row_count=4,
                generated_at=now(), caption='Source32 / Base / RF1 / frozen codec; true held-out render queries')
            atomic_json(self.root / 'preview.json', meta)
            self.publish(prediction_preview=meta, preview_url='/preview.png')

        def validate(at):
            report, arrays = evaluate_enemy_renderer(working, profiled['val'], panel,
                step=at, batch_size=args.validation_batch, baseline=baseline,
                small_enemy_rebalance=True, far_wall_supervision=True, conditions_fn=moe_conditions)
            _, guard_arrays = evaluate_enemy_renderer(working, profiled['val'], geometry_panel,
                step=at, batch_size=args.validation_batch, small_enemy_rebalance=True, conditions_fn=moe_conditions)
            report.update(protocol=protocol['name'], plan_sha256=plan_sha,
                feature_world_sha256=identity['feature_world_sha256'], feature_world_step=source_step,
                parent_step=parent_step, parent_feature_world_changed_explicitly=False,
                dynamic_feature_policy=protocol['feature_policy'], automatic_active_promotion=False)
            report['geometry_wall_guard_validation'] = geometry_wall_report(guard_arrays)
            if baseline is not None:
                report['geometry_wall_preservation'] = geometry_preservation_assessment(
                    report['geometry_wall_guard_validation'], baseline['geometry_wall_guard_validation'])
            atomic_json(self.root / 'validation' / f'step_{at:06d}.json', report)
            atomic_json(self.root / 'validation.json', report)
            preview(at)
            self.publish(renderer_moe_validation=report, validation=report)
            return report

        if baseline is None:
            self.publish(stage_label='Initial held-out validation before any SGD')
            baseline = validate(step)
            atomic_json(baseline_path, baseline)
        else:
            preview(step)
            self.publish(renderer_moe_validation=json.loads((self.root / 'validation.json').read_text(encoding='utf-8')))
        save()
        self.publish(status='running', stage_label='Source-supervised Base/RF five-role Transformer experts')
        while step < args.steps:
            if (self.root / 'stop_training').exists():
                save(); self.publish(status='stopped'); return
            configure_renderer_moe(working)
            optimizer.zero_grad(set_to_none=True)
            loss = None; records = {}
            for kind, count, weight in (('main', args.batch, 1.), ('far', 4, .25), ('geometry', 8, 1.)):
                batch = profiled[kind].batch(count)
                value, metrics = moe_five_refinement_loss(working, batch, kind=kind, inherited=inherited)
                loss = weight * value if loss is None else loss + weight * value
                records[kind] = metrics
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite renderer five-role MoE objective')
            loss.backward()
            gradient = float(clip_finite_grad_norm_(working.named_parameters(), context='Renderer five-role MoE'))
            optimizer.step(); step += 1
            if frozen_fingerprint(working) != frozen:
                raise AssertionError('Renderer five-role MoE changed a frozen world/codec/private-opacity tensor')
            row = dict(step=step, stage='renderer_moe_five_refinement', loss_protocol=protocol['name'],
                renderer_moe_plan_sha256=plan_sha, feature_world_sha256=identity['feature_world_sha256'],
                train_loss=float(loss.detach()), gradient_norm=gradient, main=records['main'],
                far=records['far'], geometry=records['geometry'], **{key: records['main'][key] for key in
                    ('renderer_rgb_loss', 'renderer_moe_role_loss', 'renderer_moe_balance_loss',
                     'renderer_moe_role_weight', 'renderer_moe_balance_weight', 'renderer_moe_heads')})
            with (self.root / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            training['local_step'] = step
            self.publish(current=step, global_training_step=step, loss=row['train_loss'], latest_metrics=row,
                renderer_moe_training=training, renderer_training_steps=renderer_start + step,
                base_training_steps=base_start + step,
                frozen_verification=dict(renderer_moe_plan_sha256=plan_sha,
                    feature_world_sha256=identity['feature_world_sha256'], step=step,
                    checked_after_optimizer_step=step, all_frozen_parameters_buffers_bitexact=True,
                    codec_all_parameters_buffers_bitexact=True, private_opacity_bitexact=True))
            save()
            if step % args.validate_every == 0: validate(step)
            elif step % args.preview_every == 0: preview(step)
        save(); self.publish(status='completed', message='Requested experimental renderer five-role MoE budget completed')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'world', 'renderer-checkpoint', 'parent-checkpoint'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--frozen-sr-checkpoint', type=Path)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    for name, default in (('steps', 2000), ('batch', 16), ('validation-frames', 256),
                          ('validation-batch', 8), ('preview-every', 10), ('validate-every', 100),
                          ('threads', 2), ('seed', 173)):
        parser.add_argument('--' + name, type=int, default=default)
    args = parser.parse_args(argv)
    if (args.steps < 1 or args.batch != 16 or args.validation_frames != 256
            or args.validation_batch not in (4, 8) or args.preview_every != 10 or args.validate_every != 100):
        parser.error('Renderer five-role MoE uses B16, fixed256 validation, preview10 and validation100')
    for name, value in vars(args).items():
        if isinstance(value, Path): setattr(args, name, value.resolve())
    from .vqgan_pipeline import process_alive
    args.run_dir.mkdir(parents=True, exist_ok=True)
    pid = args.run_dir / 'training.pid'
    if pid.exists() and process_alive(int(pid.read_text())): parser.error('Renderer five-role MoE worker already active')
    import os
    pid.write_text(str(os.getpid()), encoding='ascii')
    experiment = RendererMoEFiveTraining(args)
    try:
        with KeepAwake(): experiment.run()
    except BaseException as error:
        experiment.publish(status='failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()


