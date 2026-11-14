"""Train a new bounded RGB residual over the exact preferred frozen SR row.

This is a new architecture with a fresh private sampler and Adam for its new
branch. Source5308 supplies frozen background weights, never its old RGBA head
or optimizer. Local optimizer steps start at0; no automatic game deployment.
Launch on Windows through diagnostic_tools/bootstrap_torch_cuda_20261114.py.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .background_detail_sr import (ARCHITECTURE, TRAINABLE_PREFIX, initialize_background_detail,
    frozen_background_state, background_detail_metadata)
from .background_detail_objectives import PROTOCOL, LOSS_WEIGHTS, OBJECTIVE_SETTINGS, background_detail_loss
from .background_detail_sr_validation import (evaluate_background_detail_sr,
    preview_background_detail_sr, save_background_detail_sr_preview, background_detail_candidate)
from .distill_renderer import _save
from .gradient_safety import clip_finite_grad_norm_
from .neural_enemy_restoration_data import NeuralEnemyRestorationData
from .neural_enemy_restoration_validation import build_detail_animation_panel
from .power_guard import KeepAwake
from .renderer_student import _hash_tensors
from .sr_neural_pairs_data import fingerprint
from .training_pipeline import atomic_json, now


ROOT = Path(__file__).resolve().parents[1]
SOURCE_STEP = 5308
SOURCE_SHA = '48c35544bc473d8ff1a8c5d30c7bec72038ce823cdfe917d0b4819c1ee620b69'
BACKGROUND_SHA = 'd22e704e6a919f362deb9b5d37bf86c03931c89e27745a9165cc6af97a6e39d3'
DEFAULT_SOURCE = ROOT / 'training_runs/sr_neural_enemy_detail_20261114/assets/sr_source_005308.pt'
DEFAULT_REFERENCE_PLAN = ROOT / 'training_runs/sr_neural_enemy_detail_20261114/training_plan.json'


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError('Background detail immutable plan changed')
    else:
        atomic_json(path, value)


def initialize_runtime(device):
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.init()
    torch.set_num_threads(2)
    return device


def zero_residual_probe(model):
    """Check native inference identity without any Source image or mask."""
    modes = {module: module.training for module in model.modules()}
    try:
        model.eval()
        with torch.inference_mode():
            low = torch.linspace(0, 1, 3*32*32).reshape(1, 3, 32, 32)
            result = model.forward_with_aux(low)
            if (not torch.equal(result['image'], result['background'])
                    or not torch.equal(result['residual'], torch.zeros_like(result['residual']))
                    or not bool(torch.isfinite(result['image']).all())):
                raise ValueError('Fresh background detail branch must exactly reproduce the preferred frozen row')
    finally:
        for module, training in modes.items():
            module.training = training


def prepare_experiment(args):
    if torch.device(args.device).type == 'cuda':
        torch.cuda.init()
    torch.set_num_threads(2)
    source_path = args.source_checkpoint.resolve()
    source_sha = file_sha(source_path)
    if source_sha != SOURCE_SHA:
        raise ValueError('Background detail source must be the pinned full5308 RGB restoration checkpoint')
    source = torch.load(source_path, map_location='cpu', weights_only=False, mmap=True)
    if source.get('training_steps') != SOURCE_STEP or file_sha(source_path) != source_sha:
        raise ValueError('Background detail source step or bytes changed while reading')
    model, migration = initialize_background_detail(source, seed=args.seed,
        detail_channels=args.detail_channels, detail_blocks=args.detail_blocks,
        residual_bound=args.residual_bound)
    if _hash_tensors(frozen_background_state(model)) != BACKGROUND_SHA:
        raise ValueError('Copied preferred background does not match the authorized frozen row')
    zero_residual_probe(model)
    train = NeuralEnemyRestorationData(args.data_root, 'train', seed=args.seed, device='cpu')
    val = NeuralEnemyRestorationData(args.data_root, 'val', seed=args.seed+1, device='cpu')
    if len(train) != 8192 or len(val) != 1024 or train.scene_keys & val.scene_keys:
        raise ValueError('Need exact disjoint8192TRAIN/1024VAL final-renderer pairs')
    source_plan = source.get('training_plan') or {}
    for split, data in (('train', train), ('val', val)):
        if (source_plan.get(split) or {}).get('base') != data.base.identity:
            raise ValueError('Source and background-detail paired corpus differ')
    cache = train.base.manifest.get('cache_identity', {})
    if (cache.get('future_rgb_features') is not False or cache.get('paired_same_pose_map_tick') is not True
            or cache.get('restorer_present') is not False or cache.get('solver_steps') != 3):
        raise ValueError('Background detail requires actual RF3/Q/decoder NPR32 paired cache')
    panel = val.fixed_panel(args.validation_frames, seed=args.seed)
    detail_panel = build_detail_animation_panel(val, seed=args.seed)
    reference_path = args.reference_plan.resolve()
    reference_sha = file_sha(reference_path)
    reference = json.loads(reference_path.read_text(encoding='utf-8'))
    if (args.validation_frames != 256 or panel != source_plan.get('panel')
            or panel != reference.get('panel') or detail_panel != reference.get('detail_animation_panel')
            or file_sha(reference_path) != reference_sha):
        raise ValueError('MAIN256/detail120 held-out receipts differ from the fixed prior experiment')
    plan = canonical(dict(kind=PROTOCOL, version=1, architecture=ARCHITECTURE,
        source_checkpoint=str(source_path), source_checkpoint_sha256=source_sha,
        source_training_steps=SOURCE_STEP, source_model_sha256=_hash_tensors(source['model']),
        preferred_background_sha256=BACKGROUND_SHA, config=deepcopy(model.config), migration=migration,
        data_root=str(args.data_root.resolve()), train=train.identity, val=val.identity,
        panel=panel, panel_fingerprint=val.panel_fingerprint(panel), detail_animation_panel=detail_panel,
        reference_panel_plan=str(reference_path), reference_panel_plan_sha256=reference_sha,
        batch=args.batch, steps=args.steps, seed=args.seed, validation_frames=len(panel),
        validation_batch=args.validation_batch, preview_every=args.prediction_every,
        validation_every=args.validate_every, loss_weights=deepcopy(LOSS_WEIGHTS),
        objective_settings=deepcopy(OBJECTIVE_SETTINGS), learning_rate=args.learning_rate,
        optimizer='fresh_Adam_new_detail_branch_only', optimizer_parameter_prefix=TRAINABLE_PREFIX,
        training_clock='new_local_optimizer_updates0_to_budget', inherited_optimizer_states=False,
        private_sampler='fresh_new_architecture_phase_seed_and_counter0',
        old_RGBA_head_used=False, old_RGBA_Adam_reused=False, frozen_background=True,
        initial_output_exact_preferred_background=True, inference_inputs='actual_NPR_RGB32_only',
        Source_targets_are_labels_only=True, atlas_in_new_model=False, GAN=False,
        automatic_deployment=False, gameplay_qualified=False, budget_completion_is_quality=False))
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    immutable_json(root / 'training_plan.json', plan)
    immutable_json(root / 'migration.json', canonical(migration))
    return plan, model


def restore_detail_optimizer(model, payload, *, step, learning_rate=1e-4):
    if type(step) is not int or step < 0:
        raise ValueError('Detail Adam requires a nonnegative completed local update')
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not names or any(not name.startswith(TRAINABLE_PREFIX) for name in names):
        raise ValueError('Only the new detail residual branch may train')
    optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate)
    if payload is None:
        if step != 0:
            raise ValueError('A fresh detail Adam must start at local0')
        return optimizer, names
    if payload.get('optimizer_parameter_names') != names:
        raise ValueError('Detail Adam parameter names/order changed')
    expected = [{name: value for name, value in group.items() if name != 'params'}
        for group in optimizer.param_groups]
    optimizer.load_state_dict(payload['optimizer'])
    actual = [{name: value for name, value in group.items() if name != 'params'}
        for group in optimizer.param_groups]
    if actual != expected or len(optimizer.state) != (len(names) if step else 0):
        raise ValueError('Detail Adam hyperparameters or complete state count changed')
    for parameter, state in optimizer.state.items():
        if (set(state) != {'step', 'exp_avg', 'exp_avg_sq'} or float(state['step']) != step
                or any(not torch.is_tensor(state[name]) or state[name].shape != parameter.shape
                    or not bool(torch.isfinite(state[name]).all()) for name in ('exp_avg', 'exp_avg_sq'))
                or bool((state['exp_avg_sq'] < 0).any())):
            raise ValueError('Detail Adam step/moment shape or finite values changed')
    return optimizer, names


def validate_sampler_exposure(train, *, step, batch):
    if (type(step) is not int or step < 0 or type(batch) is not int or batch < 1
            or type(train.draws) is not int or train.draws != step*batch):
        raise ValueError('Detail private sampler draws differ from completed local updates')


def restore_rng(payload, *, seed, device):
    if payload is None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
    else:
        random.setstate(payload['rng_python'])
        np.random.set_state(payload['rng_numpy'])
        torch.set_rng_state(payload['rng_torch'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all([value.cpu() for value in payload['rng_cuda']])


def run(args):
    device = initialize_runtime(args.device)
    plan, model = prepare_experiment(args)
    root = args.run_dir.resolve()
    plan_sha = file_sha(root / 'training_plan.json')
    if args.prepare_only:
        return dict(prepared=True, plan_sha256=plan_sha, parameters=background_detail_metadata(model))
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train = NeuralEnemyRestorationData(args.data_root, 'train', seed=args.seed, device=device)
    val = NeuralEnemyRestorationData(args.data_root, 'val', seed=args.seed+1, device=device)
    if canonical(train.identity) != plan['train'] or canonical(val.identity) != plan['val']:
        raise ValueError('Background detail paired corpus changed after preparation')
    last = root / 'checkpoints/sr_last.pt'
    saved = None
    step = 0
    best = visual_best = None
    if last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=False, mmap=True)
        if (saved.get('training_plan') != plan or saved.get('plan_sha256') != plan_sha
                or canonical(saved.get('config')) != canonical(model.config)
                or saved.get('disposable_probe') is not bool(args.probe)
                or type(saved.get('training_steps')) is not int
                or not 0 <= saved['training_steps'] <= args.steps):
            raise ValueError('Background detail resume identity/local clock changed')
        step = saved['training_steps']
        model.load_state_dict(saved['model'], strict=True)
        train.restore_sampler(saved['sampler'])
        best, visual_best = deepcopy(saved.get('best')), deepcopy(saved.get('visual_best'))
    validate_sampler_exposure(train, step=step, batch=args.batch)
    model.to(device).float().configure_training()
    if _hash_tensors(frozen_background_state(model)) != BACKGROUND_SHA:
        raise ValueError('Detail checkpoint changed the frozen preferred background')
    optimizer, names = restore_detail_optimizer(model, saved, step=step, learning_rate=args.learning_rate)
    restore_rng(saved, seed=args.seed, device=device)
    started = time.perf_counter()
    training = dict(protocol=PROTOCOL, plan_sha256=plan_sha, source_step=SOURCE_STEP,
        source_checkpoint_sha256=SOURCE_SHA, local_step=step, local_budget=args.steps,
        frozen_background=True, preferred_background_sha256=BACKGROUND_SHA,
        new_optimizer=True, optimizer='fresh_new_architecture_Adam', optimizer_parameter_prefix=TRAINABLE_PREFIX,
        old_Adam_reused=False, fresh_private_sampler=True, inference_rgb_only=True,
        input_size=32, output_size=128, parameters=background_detail_metadata(model),
        data_identity_sha256=fingerprint(val.identity), validation_panel_fingerprint=plan['panel_fingerprint'],
        detail_animation_panel_fingerprint=plan['detail_animation_panel']['fingerprint'],
        detail_animation_panel_frames=plan['detail_animation_panel']['frames'])
    status = dict(run_id=root.name, stage='sr', status='initializing', current=step, total=args.steps,
        model='Preferred frozen background + bounded learned RGB detail residual',
        objective_protocol=PROTOCOL, loss_protocol=PROTOCOL, started_at=now(),
        prediction_every=args.prediction_every, validation_every=args.validate_every,
        checkpoint=str(last), gameplay_qualified=False, automatic_deployment=False,
        disposable_probe=bool(args.probe), sr_training=dict(input_size=32, output_size=128, scale=4,
            all_other_models_frozen=True))

    def publish(**values):
        training['local_step'] = step
        status.update(values, current=step, background_detail_sr_training=deepcopy(training),
            updated_at=now(), elapsed_seconds=time.perf_counter()-started)
        atomic_json(root / 'status.json', status)

    def save():
        _save(last, dict(stage='sr', config=deepcopy(model.config), model=model.state_dict(),
            training_steps=step, background_detail_updates=step, source_training_steps=SOURCE_STEP,
            training_plan=plan, plan_sha256=plan_sha, migration=plan['migration'], loss_protocol=PROTOCOL,
            source_checkpoint_sha256=SOURCE_SHA, frozen_background_sha256=BACKGROUND_SHA,
            optimizer=optimizer.state_dict(), optimizer_parameter_names=names, sampler=train.sampler_state(),
            best=deepcopy(best), visual_best=deepcopy(visual_best), rng_python=random.getstate(),
            rng_numpy=np.random.get_state(), rng_torch=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
            standalone_RGB_only=True, old_RGBA_head_used=False, atlas_in_new_model=False,
            automatic_deployment=False, gameplay_qualified=False, disposable_probe=bool(args.probe)))

    def checkpoint_copy(destination):
        save()
        # Atomic independent full checkpoint, never an alias to the writable last.
        _save(destination, torch.load(last, map_location='cpu', weights_only=False, mmap=True))

    def attach(report, preview):
        pins = dict(plan_sha256=plan_sha, source_step=SOURCE_STEP, source_checkpoint_sha256=SOURCE_SHA,
            preferred_background_sha256=BACKGROUND_SHA, model_sha256=_hash_tensors(model.state_dict()))
        report.update(pins)
        preview['metadata'].update(pins)

    def validate(baseline_report=None):
        if file_sha(args.source_checkpoint) != SOURCE_SHA or file_sha(args.reference_plan) != plan['reference_panel_plan_sha256']:
            raise ValueError('Immutable source or held-out reference receipt changed')
        report, preview = evaluate_background_detail_sr(model, val, plan['panel'], step=step,
            batch_size=args.validation_batch, baseline_report=baseline_report,
            objective_fn=background_detail_loss, objective_weights=LOSS_WEIGHTS,
            detail_panel=plan['detail_animation_panel'])
        attach(report, preview)
        atomic_json(root / 'validation' / f'step_{step:06d}.json', report)
        atomic_json(root / 'validation.json', report)
        publish(background_detail_sr_validation=report,
            prediction_preview=save_background_detail_sr_preview(preview, root/'preview.png', every=args.prediction_every))
        return report

    def select_visual(report, baseline):
        nonlocal visual_best
        candidate = background_detail_candidate(report, baseline)
        publish(background_detail_visual_candidate=candidate)
        if candidate['eligible'] and (visual_best is None or candidate['score'] > visual_best['score']):
            visual_best = {**candidate, 'plan_sha256': plan_sha, 'model_sha256': report['model_sha256']}
            checkpoint_copy(root / 'checkpoints/sr_visual_best.pt')
            atomic_json(root / 'visual_best.json', visual_best)

    initial = root / 'checkpoints/sr_initial.pt'
    if not initial.exists():
        if step != 0:
            raise ValueError('Detail resume lost its exact local0 initial checkpoint')
        publish(stage_label='New detail branch local0: exact preferred background')
        checkpoint_copy(initial)
    baseline_path = root / 'validation_initial.json'
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
        if (baseline.get('plan_sha256') != plan_sha or baseline.get('panel_fingerprint') != plan['panel_fingerprint']
                or baseline.get('objective_protocol') != PROTOCOL):
            raise ValueError('Detail initial held-out baseline identity changed')
    else:
        if step != 0:
            raise ValueError('Detail resume lost its initial baseline')
        baseline = validate()
        atomic_json(baseline_path, baseline)
    if best is None:
        if step != 0:
            raise ValueError('Detail resume lost its objective-best ledger')
        best = dict(step=0, loss=baseline['validation_objective']['loss'], objective_protocol=PROTOCOL,
            panel_fingerprint=plan['panel_fingerprint'], initial_baseline=True, automatic_deployment=False)
        checkpoint_copy(root / 'checkpoints/sr_best.pt')
        atomic_json(root / 'best.json', best)
    if visual_best is None and step == 0:
        select_visual(baseline, baseline)
    publish(status='running', stage_label='Learning details over the preferred frozen background')
    save()
    invocation_start = step
    while step < args.steps:
        if (root / 'stop_training').exists():
            save()
            publish(status='stopped')
            return
        if args.max_updates and step-invocation_start >= args.max_updates:
            save()
            publish(status='stopped_probe', stage_label='Discarded exact-budget probe stopped at complete local update')
            return
        begin = time.perf_counter()
        model.configure_training()
        batch = train.batch(args.batch)
        optimizer.zero_grad(set_to_none=True)
        output = model.forward_with_aux(batch['lr_npr'].detach())
        loss, metrics = background_detail_loss(output, batch)
        loss.backward()
        metrics['grad_norm'] = float(clip_finite_grad_norm_(model.named_parameters(), context='Background detail residual'))
        optimizer.step()
        step += 1
        if _hash_tensors(frozen_background_state(model)) != BACKGROUND_SHA:
            raise AssertionError('Detail update changed the frozen preferred background')
        metrics.update(step=step, stage='sr', objective_protocol=PROTOCOL, loss_protocol=PROTOCOL,
            background_detail_sr_plan_sha256=plan_sha, timestamp=now(), source_step=SOURCE_STEP,
            source_row_indices=batch['row_index'].detach().cpu().tolist(),
            optimizer_update_seconds=time.perf_counter()-begin, frozen_background_bitexact=True,
            old_RGBA_used=False, atlas_used=False, hypothesis_used=False)
        save()
        if step == 1:
            checkpoint_copy(root / 'checkpoints/sr_first.pt')
        with (root / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(metrics, allow_nan=False)+'\n')
        publish(loss=float(loss.detach()), latest_metrics=metrics,
            frozen_verification=dict(passed=True, step=step, preferred_background_sha256=BACKGROUND_SHA))
        if step % args.validate_every == 0 or step == args.steps:
            report = validate(baseline)
            select_visual(report, baseline)
            score = report['validation_objective']['loss']
            if score < best['loss']:
                best = dict(step=step, loss=score, objective_protocol=PROTOCOL,
                    panel_fingerprint=plan['panel_fingerprint'], automatic_deployment=False)
                checkpoint_copy(root / 'checkpoints/sr_best.pt')
                atomic_json(root / 'best.json', best)
            save()
        elif step % args.prediction_every == 0:
            preview = preview_background_detail_sr(model, val, plan['panel'], step=step,
                baseline_report=baseline, batch_size=args.validation_batch,
                detail_panel=plan['detail_animation_panel'])
            attach({}, preview)
            publish(prediction_preview=save_background_detail_sr_preview(preview, root/'preview.png', every=args.prediction_every))
    save()
    publish(status='completed', stage_label='Detail budget completed; quality assessed against the preferred frozen row')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--source-checkpoint', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--reference-plan', type=Path, default=DEFAULT_REFERENCE_PLAN)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--seed', type=int, default=173)
    parser.add_argument('--validation-frames', type=int, default=256)
    parser.add_argument('--validation-batch', type=int, default=8)
    parser.add_argument('--validate-every', type=int, default=100)
    parser.add_argument('--prediction-every', type=int, default=15)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--detail-channels', type=int, default=48)
    parser.add_argument('--detail-blocks', type=int, default=5)
    parser.add_argument('--residual-bound', type=float, default=.25)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--max-updates', type=int, default=0)
    args = parser.parse_args(argv)
    if min(args.steps, args.batch, args.validation_frames, args.validation_batch,
            args.validate_every, args.prediction_every, args.learning_rate) <= 0 or args.max_updates < 0:
        parser.error('Positive budget/batch/reporting rates and nonnegative max-updates required')
    if args.max_updates and (not args.probe or 'DISCARDED' not in str(args.run_dir)):
        parser.error('A bounded invocation requires an explicitly DISCARDED probe directory')
    initialize_runtime(args.device)
    if args.prepare_only:
        print(json.dumps(run(args), allow_nan=False), flush=True)
        return
    args.run_dir.mkdir(parents=True, exist_ok=True)
    from .vqgan_pipeline import process_alive
    pid = args.run_dir / 'training.pid'
    if pid.exists() and process_alive(int(pid.read_text())):
        parser.error('Background detail worker is already active')
    pid.write_text(str(os.getpid()), encoding='ascii')
    try:
        with KeepAwake():
            run(args)
    except BaseException as error:
        path = args.run_dir / 'status.json'
        old = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        atomic_json(path, {**old, 'status': 'failed', 'error': f'{type(error).__name__}: {error}', 'updated_at': now()})
        raise


if __name__ == '__main__':
    main()
