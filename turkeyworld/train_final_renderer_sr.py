"""Continue the existing five-expert SR on a selected Base/RF3/decoder chain.

The cache contains actual frozen-renderer32 inputs and original Source128
targets. Only existing SR experts train; the renderer never enters the SGD
graph. A new data identity gets a fresh sampler, while all named Adam states
and the parent's augmentation/global RNG are retained.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gc
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .distill_renderer import _save
from .enemy_decoder_data import EnemyDecoderData
from .gradient_safety import clip_finite_grad_norm_
from .models import build_upscaler
from .power_guard import KeepAwake
from .renderer_student import _hash_tensors
from .renderer_native_phase import _same
from .sr_moe_objectives import sr_moe_loss, LOSS_WEIGHTS
from .sr_moe_phase import PROTOCOL, sr_moe_protocol
from .sr_moe_validation import evaluate_sr_moe, preview_sr_moe, save_sr_moe_preview
from .sr_semantic_migration import restore_semantic_optimizer, _sha
from .sr_transformer_moe import configure_sr_transformer_moe, sr_moe_metadata, validate_sr_moe_checkpoint
from .train_semantic_sr import _restore_rng, mix_semantic_inputs, mix_visual_first_hypotheses
from .training_pipeline import atomic_json, now


PHASE_PROTOCOL = 'sequential_renderer_sr_continuation_v1'
MIGRATIONS = ('semantic_migration', 'enemy_decoder_migration', 'visual_first_migration',
              'restoration_migration', 'atlas_migration', 'sr_moe_migration')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _immutable(path, value):
    path = Path(path)
    if path.exists():
        if _read(path) != value:
            raise ValueError('Immutable final SR input/plan changed: ' + str(path))
    else:
        atomic_json(path, value)


def _chain_identity(header, *, world_sha256, world_step, renderer_sha256):
    """A strict format8 loader still validates all nested tensors afterwards."""
    from .decoder_rf_chain_bundle import FORMAT_VERSION, KIND
    if (header.get('format_version') != FORMAT_VERSION or header.get('kind') != KIND
            or header.get('restorer_present') is not False or header.get('solver') != 'euler'
            or type(header.get('solver_steps')) is not int or header['solver_steps'] != 3
            or type(world_step) is not int or world_step != 5906):
        raise ValueError('Final SR needs the selected no-restorer RF3/decoder format8 and frozen world5906')
    for key in ('training_steps', 'base_training_steps', 'decoder_refinement_updates'):
        if type(header.get(key)) is not int or header[key] < 1:
            raise ValueError('Final renderer has invalid independent Base/RF/decoder clocks')
    return dict(world_sha256=world_sha256, world_step=world_step,
        renderer_sha256=renderer_sha256, renderer_step=header['training_steps'],
        renderer_format_version=FORMAT_VERSION, renderer_kind=KIND,
        base_training_steps=header['base_training_steps'],
        decoder_refinement_updates=header['decoder_refinement_updates'],
        rf_chain_local_updates=header.get('rf_chain_local_updates'),
        base_refinement_updates=header.get('base_refinement_updates'),
        decoder_local_updates=header.get('decoder_local_updates'),
        solver='euler', solver_steps=3, restorer_present=False,
        future_rgb_features=False, paired_same_pose_map_tick=True,
        Source_targets_are_labels_only=True, automatic_deployment=False)


def prepare_final_sr_cache(args):
    """Reuse or create full immutable Source-aligned paired/hypothesis caches."""
    from .cache_sr_moe_pairs import build_sr_moe_cache
    from .models import load_world_checkpoint
    from .renderer_student import codec_tensor_fingerprint, load_renderer_student
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device.index if device.index is not None else 0)
        torch.cuda.init()  # Before loading any CUDA-origin checkpoint on Windows.
    world_sha, renderer_sha = _sha(args.world), _sha(args.renderer)
    source_manifest_sha = _sha(Path(args.source_root) / 'manifest.json')
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / 'status.json', dict(stage='data', status='preparing', current=0,
        stage_label='Frozen final-renderer NPR32 / original Source128 pairs',
        child_status_path=str(Path(args.data_root).resolve() / 'status.json'), updated_at=now()))
    model, world = load_world_checkpoint(args.world, 'cpu')
    header = load_renderer_student(model, args.renderer, allow_unqualified_codec=True)
    identity = _chain_identity(header, world_sha256=world_sha,
        world_step=world.get('training_steps'), renderer_sha256=renderer_sha)
    if getattr(model.flow_renderer, 'conditional_restorer', None) is not None:
        raise ValueError('Final SR cache must use RF3 to Q to current D without a restorer')
    model.float().to(device).eval().requires_grad_(False)
    identity['codec_tensor_sha256'] = codec_tensor_fingerprint(model.flow_renderer.codec)
    result = build_sr_moe_cache(args.source_root, args.data_root, model,
        cache_identity=identity, world_sha256=world_sha, world_step=identity['world_step'],
        hypotheses_output=args.hypotheses_root, batch_size=args.cache_batch, seed=args.seed)
    if (_sha(args.world) != world_sha or _sha(args.renderer) != renderer_sha
            or _sha(Path(args.source_root) / 'manifest.json') != source_manifest_sha):
        raise AssertionError('Immutable renderer/world/Source changed during SR caching')
    paired = result['paired']
    if (paired.get('status') != 'complete' or paired.get('source_arrays_bitexact') is not True
            or paired.get('frozen_model_tensors_bitexact') is not True
            or paired.get('optimizer_steps') != 0):
        raise ValueError('Final renderer cache lacks complete frozen Source-alignment proof')
    receipt = dict(protocol=PHASE_PROTOCOL, cache_identity=paired['cache_identity'],
        paired_manifest_sha256=_sha(Path(args.data_root) / 'manifest.json'),
        hypotheses_manifest_sha256=_sha(Path(args.hypotheses_root) / 'manifest.json'),
        source_manifest_sha256=source_manifest_sha,
        source_arrays_bitexact=True, frozen_renderer_world_bitexact=True,
        optimizer_steps=0, automatic_deployment=False)
    _immutable(root / 'cache_receipt.json', receipt)
    del model, world
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return receipt


def _initial_model(saved, device):
    with torch.random.fork_rng(devices=[]):
        model = build_upscaler(saved['config'])
    model.load_state_dict(saved['model'], strict=True)
    model.sr_moe_migration = deepcopy(saved['sr_moe_migration'])
    validate_sr_moe_checkpoint(model, saved)
    model.to(device).float()
    configure_sr_transformer_moe(model)
    if saved.get('sr_moe_binding') != sr_moe_metadata(model):
        raise ValueError('Full SR weights and existing five-expert binding differ')
    return model


def run_training(args, cache_receipt):
    """Own only SR SGD. The cache step is independently replaceable in CPU tests."""
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device.index if device.index is not None else 0)
        torch.cuda.init()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_sha = _sha(args.parent_checkpoint)
    initial = torch.load(args.parent_checkpoint, map_location='cpu', weights_only=False)
    required = ('optimizer', 'optimizer_parameter_names', 'sr_moe_migration', 'sr_moe_binding',
                'sampler', 'rng_numpy', 'rng_torch', 'rng_python', 'rng_cuda')
    if (initial.get('stage') != 'sr' or initial.get('config', {}).get('sr_moe_enabled') is not True
            or any(key not in initial for key in required)
            or type(initial.get('training_steps')) is not int or initial['training_steps'] <= 0
            or type(initial.get('sr_moe_updates')) is not int or initial['sr_moe_updates'] <= 0):
        raise ValueError('Final SR needs a complete already-trained five-expert SR Adam/RNG checkpoint')
    train = EnemyDecoderData(args.data_root, 'train', hypotheses_root=args.hypotheses_root,
                            seed=args.seed, device=device)
    val = EnemyDecoderData(args.data_root, 'val', hypotheses_root=args.hypotheses_root,
                          seed=args.seed + 1, device=device)
    panel = val.fixed_panel(args.validation_frames, seed=args.seed)
    cache = train.identity['base']['cache_identity']
    if (cache != cache_receipt['cache_identity']
            or cache.get('world_step') != 5906 or cache.get('renderer_format_version') != 8
            or cache.get('restorer_present') is not False or cache.get('solver_steps') != 3
            or cache.get('future_rgb_features') is not False
            or cache.get('paired_same_pose_map_tick') is not True):
        raise ValueError('Final SR reader and selected frozen final-renderer cache identities differ')
    plan = dict(version=1, phase_protocol=PHASE_PROTOCOL, objective_protocol=PROTOCOL,
        parent_checkpoint=str(Path(args.parent_checkpoint).resolve()), parent_checkpoint_sha256=source_sha,
        source_training_steps=initial['training_steps'], source_sr_moe_updates=initial['sr_moe_updates'],
        cache=cache_receipt, train=train.identity, val=val.identity,
        panel=panel, panel_fingerprint=val.panel_fingerprint(panel),
        steps=args.steps, batch=8, seed=args.seed, preview_every=args.prediction_every,
        validation_every=args.validate_every, validation_frames=args.validation_frames,
        validation_batch=args.validation_batch, loss_weights=LOSS_WEIGHTS, settings=sr_moe_protocol(),
        only_existing_sr_moe_parameters_train=True, no_new_parameters_or_Adam_groups=True,
        fresh_private_sampler_draws=0, parent_sampler_retained_as_history=True,
        parent_augmentation_and_global_RNG_preserved=True, Source_targets_are_labels_only=True,
        source_future_feedback=False, automatic_deployment=False,
        reference_scope='full_trained_five_expert_SR_parent_on_identical_final_renderer_NPR_inputs',
        best_metric='models.current.summary.ssim_mean', budget_completion_is_quality=False,
        probe=bool(args.probe))
    plan_path = root / 'training_plan.json'
    _immutable(plan_path, plan)
    plan_sha = _sha(plan_path)
    last = root / 'checkpoints' / 'sr_last.pt'
    resuming = last.exists()
    saved = torch.load(last, map_location='cpu', weights_only=False) if resuming else initial
    step = saved.get('sequential_sr_updates', 0) if resuming else 0
    if (type(step) is not int or not 0 <= step <= args.steps or saved['config'] != initial['config']
            or resuming and (saved.get('protocol') != plan
                or saved['training_steps'] != initial['training_steps'] + step
                or saved['sr_moe_updates'] != initial['sr_moe_updates'] + step)):
        raise ValueError('Final SR resume changed its plan, architecture or completed-update clocks')
    model = _initial_model(saved, device)
    optimizer = restore_semantic_optimizer(model, saved)
    reference = _initial_model(initial, device).eval().requires_grad_(False)
    if resuming:
        train.restore_sampler(saved['sampler'])
    elif train.draws != 0:
        raise AssertionError('New final-renderer cache sampler must begin at draw zero')
    rng = np.random.default_rng(args.seed)
    _restore_rng(saved, rng, device)  # After every constructor, including the frozen reference.
    restore_checks = dict(model_bitexact=_same(saved['model'], model.state_dict()),
        all_named_Adam_states_options_bitexact=_same(saved['optimizer'], optimizer.state_dict()),
        augmentation_PCG_bitexact=_same(saved['rng_numpy'], rng.bit_generator.state),
        torch_CPU_RNG_bitexact=_same(saved['rng_torch'], torch.get_rng_state()),
        Python_RNG_bitexact=_same(saved['rng_python'], random.getstate()),
        private_sampler_correct=_same(saved['sampler'], train.sampler_state()) if resuming else train.draws == 0,
        CUDA_RNG_bitexact=(_same(saved['rng_cuda'], torch.cuda.get_rng_state_all())
            if device.type == 'cuda' and saved['rng_cuda'] else True))
    if not all(restore_checks.values()):
        raise AssertionError('Final SR complete optimizer/RNG restore failed: ' + str(restore_checks))
    atomic_json(root / 'resume_verification.json', dict(passed=True, checks=restore_checks,
        optimizer_steps=0, source_training_steps=initial['training_steps'], local_step=step,
        parent_checkpoint_sha256=source_sha, plan_sha256=plan_sha,
        parameter_tensors=len(list(model.parameters())), Adam_state_entries=len(optimizer.state),
        Adam_groups=len(optimizer.param_groups), added_Adam_groups=0,
        data_sampler_draws=train.draws, source_sampler_retained_as_history=True))
    frozen_names = [name for name in model.state_dict() if not name.startswith('sr_moe.')]
    fixed_hash = _hash_tensors({name: model.state_dict()[name] for name in frozen_names})
    best = deepcopy(saved.get('best')) if resuming else None
    started = time.perf_counter()
    status = dict(run_id=root.parent.name, stage='sr', status='initializing', current=step, total=args.steps,
        objective_protocol=PROTOCOL, loss_protocol=PROTOCOL, phase_protocol=PHASE_PROTOCOL,
        model='Five SR Transformer experts on final frozen Base/RF3/Q/decoder32',
        image_size=32, output_size=128, prediction_every=args.prediction_every,
        validation_every=args.validate_every, gameplay_qualified=False, automatic_deployment=False,
        model_parameters=dict(total=sum(p.numel() for p in model.parameters()),
                              trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)),
        sr_training=dict(input_size=32, output_size=128, scale=4, all_other_models_frozen=True))
    training = dict(protocol=PROTOCOL, phase_protocol=PHASE_PROTOCOL, plan_sha256=plan_sha,
        parent_training_steps=initial['training_steps'], source_parent_sha256=source_sha,
        local_step=step, local_budget=args.steps, world_sha256=cache['world_sha256'], world_step=cache['world_step'],
        renderer_sha256=cache['renderer_sha256'], renderer_training_steps=cache['renderer_step'],
        base_training_steps=cache.get('base_training_steps'), decoder_refinement_updates=cache.get('decoder_refinement_updates'),
        roles=sr_moe_protocol()['roles'], experts=5, private_Transformer_blocks_per_expert=1,
        width=64, heads=4, expert_hidden=128, grid=8, rgb_resolution=128, visibility_resolution=128,
        heat_resolution=32, rgb_amplitude=1., frozen_parent=True, Source_targets_are_labels_only=True,
        source_future_feedback=False, restorer_active=False, parameters=status['model_parameters'],
        validation_panel_fingerprint=plan['panel_fingerprint'], reference_training_steps=initial['training_steps'])

    def publish(**values):
        training['local_step'] = step
        status.update(values, current=step, sr_moe_training=training,
            cumulative_sr_training_steps=initial['training_steps'] + step,
            updated_at=now(), elapsed_seconds=time.perf_counter() - started)
        atomic_json(root / 'status.json', status)

    def payload():
        value = {key: deepcopy(initial[key]) for key in MIGRATIONS if key in initial}
        value.update(stage='sr', config=deepcopy(model.config), model=model.state_dict(),
            optimizer=optimizer.state_dict(), optimizer_parameter_names=deepcopy(initial['optimizer_parameter_names']),
            sr_moe_binding=sr_moe_metadata(model), training_steps=initial['training_steps'] + step,
            parent_training_steps=initial['parent_training_steps'], sr_moe_updates=initial['sr_moe_updates'] + step,
            step=step, sequential_sr_updates=step, continuation_source_training_steps=initial['training_steps'],
            completed_updates=step, phase_protocol=PHASE_PROTOCOL, protocol=plan, best=deepcopy(best),
            parent_atlas_updates=initial.get('parent_atlas_updates'), parent_sampler=deepcopy(initial.get('parent_sampler')),
            continuation_source_sampler=deepcopy(initial['sampler']), sampler=train.sampler_state(),
            rng_numpy=deepcopy(rng.bit_generator.state), rng_torch=torch.get_rng_state(), rng_python=random.getstate(),
            rng_cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
            automatic_deployment=False, gameplay_qualified=False, disposable_probe=bool(args.probe))
        return value

    def save():
        _save(last, payload())

    def attach(report, preview):
        pins = dict(plan_sha256=plan_sha, sr_moe_plan_sha256=plan_sha, phase_protocol=PHASE_PROTOCOL,
            parent_source_sha256=source_sha, reference_training_steps=initial['training_steps'],
            world_sha256=cache['world_sha256'], renderer_sha256=cache['renderer_sha256'],
            model_sha256=_hash_tensors(model.state_dict()), old_SR_frozen=True,
            restorer_active=False, source_future_feedback=False,
            reference_scope=plan['reference_scope'])
        report.update(pins)
        report['source_held_out'] = True
        preview['metadata'].update(pins, source_held_out=True,
            row_labels=['true Source128', 'final selected neural renderer32',
            'frozen full SR' + str(initial['training_steps']), 'adapted five-expert SR128'])

    def write_preview(preview):
        result = save_sr_moe_preview(preview, root / 'preview.png', every=args.prediction_every)
        result['caption'] = 'Same Source128/final NPR32; frozen full SR parent versus adapted SR experts'
        atomic_json(root / 'preview.json', result)
        return result

    def select_best(report, report_path):
        nonlocal best
        score = report['models']['current']['summary'].get('ssim_mean')
        if (type(score) not in (int, float) or not math.isfinite(score)
                or report.get('panel_fingerprint') != plan['panel_fingerprint']
                or report.get('plan_sha256') != plan_sha):
            raise ValueError('Best SR selection requires the actual pinned held-out current-NPR SSIM')
        if best is None or score > best['ssim_mean']:
            best = dict(step=step, ssim_mean=score, training_steps=initial['training_steps'] + step,
                report=str(Path(report_path).resolve()), report_sha256=_sha(report_path),
                panel_fingerprint=plan['panel_fingerprint'], data_identity_sha256=report['data_identity_sha256'],
                plan_sha256=plan_sha, parent_checkpoint_sha256=source_sha,
                world_sha256=cache['world_sha256'], renderer_sha256=cache['renderer_sha256'],
                automatic_deployment=False, quality_qualified=False)
            selected = root / 'checkpoints' / 'sr_best.pt'
            _save(selected, payload())
            export = {key: value for key, value in payload().items()
                      if key not in ('optimizer', 'optimizer_parameter_names', 'sampler', 'rng_numpy',
                                     'rng_torch', 'rng_python', 'rng_cuda')}
            _save(root / 'checkpoints' / 'sr_for_game.pt', export)
            atomic_json(root / 'best.json', {**best, 'checkpoint':str(selected.resolve()),
                'checkpoint_sha256':_sha(selected), 'model_sha256':_hash_tensors(model.state_dict()),
                'export':str((root / 'checkpoints' / 'sr_for_game.pt').resolve()),
                'export_sha256':_sha(root / 'checkpoints' / 'sr_for_game.pt')})

    baseline_path = root / 'validation_initial.json'
    if baseline_path.exists():
        baseline = _read(baseline_path)
    else:
        if step:
            raise ValueError('Final SR resume lost its initial held-out report')
        publish(stage_label='Initial final-NPR held-out validation before SR SGD')
        baseline, preview = evaluate_sr_moe(model, reference, val, panel, step=0, batch_size=args.validation_batch)
        attach(baseline, preview)
        atomic_json(baseline_path, baseline)
        zero_report = root / 'validation' / 'step_000000.json'
        atomic_json(zero_report, baseline)
        select_best(baseline, zero_report)
        status['prediction_preview'] = write_preview(preview)
    if (baseline.get('plan_sha256') != plan_sha or baseline.get('panel_fingerprint') != plan['panel_fingerprint']):
        raise ValueError('Final SR baseline belongs to another data/renderer/panel')
    training['data_identity_sha256'] = baseline['data_identity_sha256']
    if resuming and (_read(root / 'best.json').get('step') != best.get('step')):
        raise ValueError('Final SR full checkpoint and selected-best history differ')
    publish(status='running', sr_moe_validation=baseline)
    save()
    while step < args.steps:
        if (root / 'stop_training').exists():
            save()
            publish(status='stopped')
            return status
        update_started = time.perf_counter()
        configure_sr_transformer_moe(model)
        batch = train.batch(8)
        supplied_hint_frames = int((batch['hypothesis'][:, -1] > .5).sum())
        inputs, domains = mix_semantic_inputs(batch, rng)
        batch['hypothesis'], dropped, wrong = mix_visual_first_hypotheses(batch['hypothesis'], rng)
        optimizer.zero_grad(set_to_none=True)
        output = model.forward_with_aux(inputs, batch['hypothesis'].detach())
        loss, metrics = sr_moe_loss(output, batch, label_renderer=model.render_label_atlas)
        loss.backward()
        metrics['gradient_norm'] = float(clip_finite_grad_norm_(model.named_parameters(), context='Final renderer SR'))
        if step < 3:
            metrics['expert_gradient_norms'] = {str(i):sum(float(p.grad.detach().float().square().sum())
                for p in expert.parameters() if p.grad is not None)**.5 for i, expert in enumerate(model.sr_moe.experts)}
        optimizer.step()
        step += 1
        if _hash_tensors({name:model.state_dict()[name] for name in frozen_names}) != fixed_hash:
            raise AssertionError('Final SR SGD changed a frozen parent or embedded atlas tensor')
        metrics.update(step=step, stage='sr', phase_protocol=PHASE_PROTOCOL,
            objective_protocol=PROTOCOL, loss_protocol=PROTOCOL, sr_moe_plan_sha256=plan_sha,
            timestamp=now(), input_domain_counts=dict(Counter(domains)),
            source_row_indices=batch['row_index'].detach().cpu().tolist(),
            supplied_causal_hint_frames=supplied_hint_frames, unknown_static_hint_frames=8 - supplied_hint_frames,
            hypotheses_deliberately_dropped=dropped, mismatched_predicted_hypotheses=wrong,
            cumulative_sr_training_steps=initial['training_steps'] + step,
            optimizer_update_seconds=time.perf_counter() - update_started)
        with (root / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + '\n')
        publish(status='running', loss=float(loss.detach()), latest_metrics=metrics,
            frozen_verification=dict(step=step, passed=True, sha256=fixed_hash,
                all_old_SR_parameters_buffers_bitexact=True, sr_moe_plan_sha256=plan_sha))
        if step % args.validate_every == 0 or step == args.steps:
            report, preview = evaluate_sr_moe(model, reference, val, panel, step=step,
                batch_size=args.validation_batch, baseline_report=baseline)
            attach(report, preview)
            report_path = root / 'validation' / f'step_{step:06d}.json'
            atomic_json(report_path, report)
            atomic_json(root / 'validation.json', report)
            select_best(report, report_path)
            publish(sr_moe_validation=report, prediction_preview=write_preview(preview))
        elif step % args.prediction_every == 0:
            preview = preview_sr_moe(model, reference, val, panel, step=step,
                baseline_report=baseline, batch_size=args.validation_batch)
            attach({}, preview)
            publish(prediction_preview=write_preview(preview))
        save()  # Complete optimizer boundaries only; includes exact selection and sampler clocks.
    if _sha(args.parent_checkpoint) != source_sha:
        raise AssertionError('Immutable original full SR checkpoint changed')
    chosen = _read(root / 'best.json')
    if args.probe:
        frozen_ids = [index for group, names in zip(initial['optimizer']['param_groups'], initial['optimizer_parameter_names'])
                      for index, name in zip(group['params'], names) if not name.startswith('sr_moe.')]
        state = optimizer.state_dict()['state']
        checks = dict(restore_checks,
            three_discarded_optimizer_updates=step == 3,
            source_full_checkpoint_unchanged=_sha(args.parent_checkpoint) == source_sha,
            all_old_SR_parameters_buffers_bitexact=_hash_tensors({name:model.state_dict()[name] for name in frozen_names}) == fixed_hash,
            all_old_SR_Adam_states_bitexact=all(_same(initial['optimizer']['state'].get(index), state.get(index)) for index in frozen_ids),
            all_five_expert_gradients_positive_finite=all(math.isfinite(v) and v > 0
                for v in metrics.get('expert_gradient_norms', {}).values()) and len(metrics.get('expert_gradient_norms', {})) == 5,
            all_trainable_gradients_finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
            existing_SR_experts_changed=any(not _same(value, model.state_dict()[name])
                for name, value in initial['model'].items() if name.startswith('sr_moe.')),
            exactly_six_NPR_one_clean_one_codec=metrics['input_domain_counts'] == dict(npr=6, clean=1, codec=1),
            four_hints_dropped=metrics['hypotheses_deliberately_dropped'] == 4,
            no_restorer=cache['restorer_present'] is False,
            Source_labels_not_inference_inputs=plan['Source_targets_are_labels_only'] is True,
            only_existing_SR_experts_train=all(p.requires_grad == name.startswith('sr_moe.') for name,p in model.named_parameters()))
        atomic_json(root / 'gpu_warmstart.json', dict(passed=all(checks.values()), checks=checks,
            discarded_optimizer_updates=step, production_optimizer_updates=0,
            plan_sha256=plan_sha, parent_checkpoint_sha256=source_sha,
            frozen_SR_sha256=fixed_hash, last_metrics=metrics,
            actual_device=str(device), automatic_promotion=False))
        if not all(checks.values()):
            raise AssertionError('Disposable final SR probe failed: ' + str(checks))
    handoff = dict(stage='sr', protocol=PHASE_PROTOCOL, objective_protocol=PROTOCOL,
        completed_updates=step, complete_budget=args.steps, production_updates=0 if args.probe else step,
        disposable_probe=bool(args.probe), selected_full_checkpoint=chosen['checkpoint'],
        selected_full_checkpoint_sha256=chosen['checkpoint_sha256'], selected_bundle=chosen['export'],
        selected_bundle_sha256=chosen['export_sha256'], selected_local_step=chosen['step'],
        selected_training_steps=chosen['training_steps'], best_metric=chosen['ssim_mean'],
        panel_fingerprint=chosen['panel_fingerprint'], data_identity_sha256=chosen['data_identity_sha256'],
        parent_checkpoint_sha256=source_sha, renderer_sha256=cache['renderer_sha256'],
        world_sha256=cache['world_sha256'], automatic_promotion=False, gameplay_qualified=False,
        budget_completion_is_quality=False)
    atomic_json(root / 'handoff.json', handoff)
    publish(status='completed', best=chosen,
        message='Disposable SR probe completed' if args.probe else 'All3000 SR updates completed; selected export is not automatically deployed')
    return status


def run(args):
    return run_training(args, prepare_final_sr_cache(args))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'parent-checkpoint', 'world', 'renderer', 'source-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--hypotheses-root', type=Path)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--cache-batch', type=int, default=16)
    parser.add_argument('--seed', type=int, default=173)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--probe', action='store_true', help='Three disposable steps, never production handoff')
    args = parser.parse_args(argv)
    if args.steps != 3000 or not 1 <= args.cache_batch <= 32:
        parser.error('The sequential SR stage uses exactly3000 updates; cache batch must be1..32')
    if args.probe:
        args.steps = 3
    args.data_root = args.data_root or args.run_dir.parent / 'data' / 'paired'
    args.hypotheses_root = args.hypotheses_root or args.run_dir.parent / 'data' / 'hypotheses'
    args.validation_frames, args.validation_batch = 256, 8
    args.validate_every, args.prediction_every = 100, 15
    args.run_dir.mkdir(parents=True, exist_ok=True)
    from .vqgan_pipeline import process_alive
    pid = args.run_dir / 'training.pid'
    if pid.exists() and process_alive(int(pid.read_text())):
        parser.error('Final SR worker is already active')
    pid.write_text(str(os.getpid()), encoding='ascii')
    try:
        with KeepAwake():
            run(args)
    except BaseException as error:
        path = args.run_dir / 'status.json'
        old = _read(path) if path.exists() else {}
        atomic_json(path, {**old, 'status':'failed', 'error':f'{type(error).__name__}: {error}', 'updated_at':now()})
        raise


if __name__ == '__main__':
    main()
