"""Durable, opt-in Source continuation; original plans and optimizer stay intact."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .motion_recovery_validation import THRESHOLDS, _long_summary
from .navigation_experiment import experiment_fingerprint
from .training_pipeline import atomic_json


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verified_episode_paths(root, *, relative_directory='short/rgb/train'):
    """Verify all listed bytes and pin the loader to exactly manifest members."""
    root = Path(root).resolve()
    manifest_path = root/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    records = manifest.get('records', [])
    if manifest.get('status') != 'complete' or not records:
        raise ValueError('Continuation contact corpus must be complete')
    listed, selected = set(), []
    directory = (root/relative_directory).resolve()
    for record in records:
        path = (root/record['relative_path']).resolve()
        if root not in path.parents or path.suffix != '.npz' or path in listed:
            raise ValueError('Invalid or duplicate continuation corpus member')
        if file_sha256(path) != record['sha256']:
            raise ValueError('Continuation corpus member changed')
        listed.add(path)
        if path.parent == directory:
            selected.append(path)
    if listed != {path.resolve() for path in root.rglob('*.npz')}:
        raise ValueError('Continuation corpus contains unlisted or missing NPZ files')
    if not selected:
        raise ValueError('Continuation corpus has no explicit TRAIN episodes')
    return selected, file_sha256(manifest_path)


def bind_continuation(state, path, *, enabled, dataset_root, base_plan_sha,
                      contact_plan_path, checkpoint_path, step):
    training = state['rssm_training']
    saved = training.get('continuation')
    path = Path(path)
    if not enabled:
        if saved or path.exists():
            raise ValueError('Saved continuation requires --continue-until-quality')
        return None
    if dataset_root is None or not training.get('contact_revision'):
        raise ValueError('Continuation requires an explicit contact corpus and saved contact revision')
    paths, manifest_sha = verified_episode_paths(dataset_root)
    settings = {'version': 1, 'base_plan_sha256': base_plan_sha,
        'contact_plan_sha256': file_sha256(contact_plan_path),
        'dataset_directory': str(Path(dataset_root).resolve()),
        'dataset_manifest_sha256': manifest_sha, 'microbatch_size': 16,
        'gradient_accumulation_steps': 1, 'learning_rate': .0001,
        'validation_every': 250, 'preview_every': 10,
        'required_consecutive_passes': 2, 'quality_thresholds': dict(THRESHOLDS),
        'quality_scope': 'Held-out navigation state/memory, no-enemy450; renderer/combat not qualified',
        'new_auxiliary_examples': 12, 'old_auxiliary_examples': 4}
    if path.exists():
        plan = json.loads(path.read_text(encoding='utf-8'))
        if any(plan.get(key) != value for key, value in settings.items()):
            raise ValueError('Continuation plan identity/settings/data changed')
    else:
        if (saved or step != 2000 or training['phase'] != 'source'
                or training['distillation_completed'] != training['distillation_updates']):
            raise ValueError('Continuation must bind the saved Source optimizer boundary2000')
        plan = {**settings, 'start_optimizer_step': step,
            'source_completed_at_start': training['source_completed'],
            'source_checkpoint_at_binding': str(Path(checkpoint_path).resolve()),
            'source_checkpoint_sha256': file_sha256(checkpoint_path)}
        atomic_json(path, plan)
    fingerprint = experiment_fingerprint(plan)
    if saved:
        if (saved.get('fingerprint') != fingerprint or saved.get('last_optimizer_step') != step
                or saved.get('updates') != step-plan['start_optimizer_step']
                or saved.get('new_auxiliary_draws') != saved['updates']*12
                or saved.get('old_auxiliary_draws') != saved['updates']*4
                or training['source_completed'] != plan['source_completed_at_start']+saved['updates']):
            raise ValueError('Continuation checkpoint identity or successful-update clocks changed')
    else:
        if (step != plan['start_optimizer_step']
                or training['source_completed'] != plan['source_completed_at_start']
                or file_sha256(checkpoint_path) != plan['source_checkpoint_sha256']):
            raise ValueError('Continuation initial checkpoint does not match its immutable plan')
        saved = {**plan, 'fingerprint': fingerprint, 'status': 'training', 'updates': 0,
            'last_optimizer_step': step, 'new_auxiliary_draws': 0, 'old_auxiliary_draws': 0,
            'consecutive_passes': 0, 'last_validation_step': None, 'baseline_pending': True,
            'weights_preserved': True, 'optimizer_preserved': True, 'rng_preserved': True}
        training['continuation'] = saved
        state.setdefault('continuation_history', []).append({'fingerprint': fingerprint,
            'start_optimizer_step': step, 'dataset_directory': plan['dataset_directory']})
    training.update(total_updates=None, source_updates=None, validation_every=250, preview_every=10)
    state.update(training_until_quality=True, total=None, eta_seconds=None,
        eta_scope='Training until two complete quality passes; completion time unknown')
    state['training_protocol'].update(validation_every=250, preview_every=10)
    return saved, paths


def commit_continuation(continuation, *, step):
    if step != continuation['last_optimizer_step']+1:
        raise ValueError('Continuation commit requires one successful Adam update')
    continuation['last_optimizer_step'] = step
    continuation['updates'] += 1
    continuation['new_auxiliary_draws'] += 12
    continuation['old_auxiliary_draws'] += 4


def validation_due(state, step):
    continuation = state['rssm_training'].get('continuation')
    return step % (250 if continuation else 100) == 0


def training_should_continue(state, finite_updates):
    continuation = state['rssm_training'].get('continuation')
    if continuation:
        return continuation['status'] != 'qualified'
    return state['global_training_step'] < finite_updates


def _finite_limit(value, maximum):
    return isinstance(value, (int, float)) and math.isfinite(value) and value <= maximum


def complete_quality_gate(report, thresholds=None):
    """Require actual numerical and coverage evidence, never a naked passed flag."""
    limits, failures = thresholds or THRESHOLDS, []
    for name in ('short', 'contact_generalization'):
        short = report.get(name, {})
        metrics, counts, cohorts = short.get('metrics', {}), short.get('counts', {}), short.get('cohorts', {})
        coverage = (short.get('source_held_out') is True
            and counts.get('samples', 0) >= (128 if name == 'short' else 256)
            and all(counts.get(key, 0) > 0 for key in ('free_axes', 'blocked_axes', 'sliding_samples'))
            and all(cohorts.get(key, {}).get('samples', 0) > 0
                for key in ('positive_turn', 'negative_turn', 'moving_turn')))
        if name == 'short': coverage &= cohorts.get('idle', {}).get('samples', 0) > 0
        numerical = all((_finite_limit(metrics.get('velocity_error_tiles_s', {}).get('mean'), limits['short_velocity_error_mean_max']),
            _finite_limit(metrics.get('false_stop_rate'), limits['false_stop_rate_max']),
            _finite_limit(metrics.get('missed_block_rate'), limits['missed_block_rate_max'])))
        if not coverage or not numerical: failures.append(name)
    idle = report.get('heldout_idle', {})
    if not (idle.get('source_held_out') is True and idle.get('samples', 0) >= 32
            and idle.get('all_states_memory_finite') is True
            and _finite_limit(idle.get('velocity_error_tiles_s_max'), 1e-5)):
        failures.append('heldout_idle')
    for name in ('long450', 'one_frame450'):
        long = report.get(name, {})
        if not (long.get('all_states_finite') is True and long.get('all_memory_finite') is True
                and long.get('windows', 0) >= 16 and long.get('horizon') == 450
                and long.get('source_wall_steps') == 0 and long.get('npc_active_steps') == 0
                and long.get('model_wall_steps') == 0 and long.get('model_wall_path_steps') == 0
                and _finite_limit(long.get('final_player_xy_error_tiles', {}).get('mean'), limits['long_final_xy_mean_max'])
                and _finite_limit(long.get('final_player_xy_error_tiles', {}).get('p95'), limits['long_final_xy_p95_max'])):
            failures.append(name)
    default = report.get('default_spawn')
    if default is not None and default.get('passed') is not True:
        failures.append('default_spawn_train_regression')
    if report.get('source_future_feedback') is not False or report.get('future_rgb_encoded') is not False:
        failures.append('causality_metadata')
    return {'passed': not failures, 'failure_reasons': failures, 'thresholds': dict(limits)}


def record_continuation_validation(continuation, report, *, step):
    if step > continuation['last_optimizer_step']:
        raise ValueError('Validation cannot refer to an unsaved future optimizer step')
    previous = continuation.get('last_validation_step')
    if previous is not None and step <= previous:
        if step == previous: return False
        raise ValueError('Continuation validation clock moved backward')
    gate = complete_quality_gate(report, continuation['quality_thresholds'])
    continuation.update(last_validation_step=step, last_validation_passed=gate['passed'],
        last_failure_reasons=gate['failure_reasons'], baseline_pending=False,
        consecutive_passes=continuation['consecutive_passes']+1 if gate['passed'] else 0)
    if continuation['consecutive_passes'] >= continuation['required_consecutive_passes']:
        continuation.update(status='qualified', qualified_at_step=step)
    return True


def evaluate_heldout_idle(model, data, *, count=32, batch_size=16, seed=173):
    from .staged_validation import _isolated_validation
    candidates = []
    if data.split != 'val': raise ValueError('Idle qualification requires held-out val')
    for index, episode in enumerate(data.episodes):
        with np.load(episode['path'], allow_pickle=False) as source:
            idle = np.flatnonzero(~np.any(source['actions'] != 0, axis=-1))
        if len(idle): candidates.append((index, int(idle[0])))
    if len(candidates) < count: raise ValueError('Need32 distinct held-out idle episodes')
    chosen = sorted(np.random.default_rng(seed).choice(len(candidates), count, replace=False).tolist())
    panel = [candidates[index] for index in chosen]
    errors, finite, records = [], True, []
    with _isolated_validation(model, data, seed):
        for first in range(0, len(panel), batch_size):
            pairs = np.asarray(panel[first:first+batch_size], np.int64)
            batch = data.batch(len(pairs), context=1, horizon=1, indices=pairs)
            features = model.encode_map(batch['map']); current = batch['states'][:, 0]
            memory = model.initialize_from_image(batch['context_frames'], batch['states'][:, :1], features)
            result = model.transition(current, batch['actions'][:, 0], features, memory)
            finite &= bool(torch.isfinite(result['state']).all() and torch.isfinite(result['memory']).all())
            errors.extend(((result['state'][:, :2]-batch['target_states'][:, 0, :2])*480).norm(dim=-1).cpu().tolist())
            records.extend({'episode': Path(data.episodes[int(i)]['path']).name, 'start': int(start)} for i, start in pairs)
    return {'source_held_out': True, 'samples': len(errors), 'all_states_memory_finite': finite,
        'velocity_error_tiles_s_mean': float(np.mean(errors)), 'velocity_error_tiles_s_max': float(np.max(errors)),
        'passed': finite and max(errors) <= 1e-5, 'source_future_feedback': False,
        'future_rgb_encoded': False, 'context': 1, 'panel': records,
        'fingerprint': experiment_fingerprint(records)}


def evaluate_one_frame450(model, data, output, *, seed=173):
    from .behavior_trajectory import evaluate_state_trajectories
    from .staged_curriculum import StagedSequenceData
    if not hasattr(data, 'batch'):
        data = StagedSequenceData(data, str(next(model.parameters()).device),
            mode='stream', cache_bytes=64*1024**2)
    output = Path(output)
    report = evaluate_state_trajectories(model, data, output, context=1,
        horizon=450, count=16, batch_size=4, seed=seed)
    with np.load(output/'heldout_trajectory_states.npz', allow_pickle=False) as saved:
        summary = _long_summary(saved['predicted'], saved['reference'], saved['maps'])
    summary.update(context=1, source_held_out=True, fingerprint=report['fingerprint'],
        all_memory_finite=report['all_memory_finite_and_shape_valid'])
    summary['passed'] = bool(summary['all_states_finite'] and summary['all_memory_finite']
        and summary['final_player_xy_error_tiles']['mean'] <= THRESHOLDS['long_final_xy_mean_max']
        and summary['final_player_xy_error_tiles']['p95'] <= THRESHOLDS['long_final_xy_p95_max']
        and summary['model_wall_steps'] == summary['model_wall_path_steps'] == 0)
    atomic_json(output/'summary.json', summary)
    return summary
