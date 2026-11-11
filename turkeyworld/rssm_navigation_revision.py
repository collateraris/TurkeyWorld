"""Explicit joint navigation schedule and durable physical-quality selection.

This module never changes an old plan, training optimizer, renderer or deployed
game. New candidates are immutable inference checkpoints. Their selector is
published only after the checkpoint has been written, reopened and verified.
"""
from __future__ import annotations

from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import torch

from .training_pipeline import atomic_json, replace_file


VERSION = 1
COMBINATIONS = (('original', 1), ('coverage', 1), ('original', 2), ('coverage', 2))
LATENT_CYCLE = ('argmax', 'argmax', 'sample', 'argmax')
AUXILIARY_COUNTS = {'corner': 8, 'single_wall': 4, 'legacy': 4}


def _integer(value, name, *, minimum=0):
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def navigation_revision_setting(completed_revision_updates):
    """Return the NEXT update's complete independent16-update schedule setting."""
    done = _integer(completed_revision_updates, 'completed_revision_updates')
    index = done % 16
    pool, context = COMBINATIONS[index % 4]
    mode = LATENT_CYCLE[index//4]
    return {'revision_update': done+1, 'cycle_index': index,
            'latent_mode': mode, 'long_pool': pool, 'context': context,
            'numeric_context': context, 'observed_context': context,
            'window_start': 0, 'horizon': 450,
            'joint_key': f'{mode}|{pool}|context{context}'}


def new_revision_ledger(start_optimizer_step, source_completed, fingerprint=''):
    start = _integer(start_optimizer_step, 'start_optimizer_step')
    source = _integer(source_completed, 'source_completed')
    keys = [f'{mode}|{pool}|context{context}' for mode in ('argmax', 'sample')
            for pool, context in COMBINATIONS]
    return {'version': VERSION, 'fingerprint': str(fingerprint),
            'start_optimizer_step': start, 'last_optimizer_step': start,
            'source_completed_at_start': source, 'source_completed': source,
            'revision_updates': 0, 'auxiliary_draws': dict.fromkeys(AUXILIARY_COUNTS, 0),
            'joint_update_counts': dict.fromkeys(keys, 0), 'joint_long_draws': dict.fromkeys(keys, 0),
            'long_batch_size': None, 'status': 'training'}


def validate_revision_ledger(ledger, *, step=None, source_completed=None):
    start = _integer(ledger.get('start_optimizer_step'), 'start_optimizer_step')
    updates = _integer(ledger.get('revision_updates'), 'revision_updates')
    if ledger.get('version') != VERSION or ledger.get('last_optimizer_step') != start+updates:
        raise ValueError('Navigation revision optimizer clock mismatch')
    expected_source = _integer(ledger.get('source_completed_at_start'), 'source_completed_at_start')+updates
    if ledger.get('source_completed') != expected_source:
        raise ValueError('Navigation revision Source clock mismatch')
    if step is not None and step != start+updates:
        raise ValueError('Navigation revision is not at the saved optimizer boundary')
    if source_completed is not None and source_completed != expected_source:
        raise ValueError('Navigation revision saved Source count changed')
    if ledger.get('auxiliary_draws') != {key: count*updates for key, count in AUXILIARY_COUNTS.items()}:
        raise ValueError('Navigation revision8/4/4 auxiliary counters changed')
    counts = dict.fromkeys(new_revision_ledger(start, 0)['joint_update_counts'], 0)
    cycles, remainder = divmod(updates, 16)
    for index in range(16):
        counts[navigation_revision_setting(index)['joint_key']] += cycles+(index < remainder)
    if ledger.get('joint_update_counts') != counts:
        raise ValueError('Navigation revision joint schedule counts changed')
    batch = ledger.get('long_batch_size')
    if updates:
        _integer(batch, 'long_batch_size', minimum=1)
    elif batch is not None:
        raise ValueError('An empty revision cannot claim a committed long batch')
    if ledger.get('joint_long_draws') != {key: n*(batch or 0) for key, n in counts.items()}:
        raise ValueError('Navigation revision joint long-window draws changed')
    return ledger


def commit_revision_update(ledger, *, step, source_completed, long_batch_size=16):
    """Commit one successful Adam update, never a microbatch or validation step."""
    validate_revision_ledger(ledger)
    _integer(step, 'step')
    _integer(source_completed, 'source_completed')
    batch = _integer(long_batch_size, 'long_batch_size', minimum=1)
    if step == ledger['last_optimizer_step']:
        if source_completed != ledger['source_completed'] or ledger['long_batch_size'] not in (None, batch):
            raise ValueError('Duplicate revision commit disagrees with saved counters')
        return False
    if (step != ledger['last_optimizer_step']+1 or source_completed != ledger['source_completed']+1
            or ledger['long_batch_size'] not in (None, batch)):
        raise ValueError('Revision commit requires exactly one completed Source optimizer update')
    setting = navigation_revision_setting(ledger['revision_updates'])
    ledger.update(last_optimizer_step=step, source_completed=source_completed,
                  revision_updates=ledger['revision_updates']+1, long_batch_size=batch)
    for key, count in AUXILIARY_COUNTS.items():
        ledger['auxiliary_draws'][key] += count
    ledger['joint_update_counts'][setting['joint_key']] += 1
    ledger['joint_long_draws'][setting['joint_key']] += batch
    validate_revision_ledger(ledger, step=step, source_completed=source_completed)
    return True


def _nonnegative(value, name):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value < 0):
        raise ValueError('Invalid finite nonnegative physical metric: '+name)
    return float(value)


def navigation_validation_rank(report, *, expected_fingerprints):
    """Rank both actual450 contexts, independent of train loss and passed flags."""
    if (report.get('source_future_feedback') is not False or report.get('future_rgb_encoded') is not False
            or not isinstance(report.get('fingerprint'), str) or not report['fingerprint']):
        raise ValueError('Physical candidate requires an identified causal validation report')
    _integer(report.get('step'), 'validation step')
    walls, p95, means = [], [], []
    for name in ('long450', 'one_frame450'):
        value = report.get(name, {})
        fingerprint = expected_fingerprints.get(name)
        if not isinstance(fingerprint, str) or not fingerprint or value.get('fingerprint') != fingerprint:
            raise ValueError('Physical navigation validation panel fingerprint changed: '+name)
        if (value.get('horizon') != 450 or not isinstance(value.get('windows'), int)
                or isinstance(value.get('windows'), bool) or value['windows'] < 16
                or value.get('all_states_finite') is not True or value.get('all_memory_finite') is not True
                or value.get('source_wall_steps') != 0 or value.get('npc_active_steps') != 0
                or value.get('source_future_feedback') is not False
                or value.get('source_held_out', True) is not True):
            raise ValueError('Physical candidate needs complete finite held-out450 evidence: '+name)
        if 'context' in value and value['context'] != (1 if name == 'one_frame450' else 2):
            raise ValueError('Physical validation context mismatch: '+name)
        point = _integer(value.get('model_wall_steps'), name+' wall steps')
        swept = _integer(value.get('model_wall_path_steps'), name+' swept wall steps')
        if max(point, swept) > value['windows']*450:
            raise ValueError('Wall count exceeds the actual validation transitions')
        walls.append(point+swept)
        final = value.get('final_player_xy_error_tiles', {})
        if final.get('n') != value['windows']:
            raise ValueError('Physical final-XY denominator does not match the validation windows')
        means.append(_nonnegative(final.get('mean'), name+' finalXY mean'))
        p95.append(_nonnegative(final.get('p95'), name+' finalXY p95'))
    return tuple([sum(walls), max(p95), sum(means)/len(means)])


def _model_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.items()):
        if not torch.is_tensor(value):
            raise ValueError('Inference model state must contain tensors only')
        tensor = value.detach().cpu().contiguous()
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(torch.isfinite(tensor).all()):
            raise ValueError('Cannot select nonfinite model weights: '+name)
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _verify_candidate(path, identity):
    payload = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    try:
        if payload.get('navigation_candidate') != identity or _model_digest(payload['model']) != identity['model_sha256']:
            raise ValueError('Existing immutable navigation candidate does not match its weights/report')
    finally:
        del payload
        gc.collect()


def consider_navigation_candidate(root, checkpoint, validation, *, expected_fingerprints,
                                  assets_metadata=None, qualification=None, prune_replaced=True):
    """Atomically select best physical and best fully-qualified inference weights.

    The full Adam/RNG checkpoint is supplied read-only and remains separate.
    No pointer is used for automatic deployment or training-state restoration.
    Only superseded files named by our previous selector may be pruned, after
    its replacement is durable, and only inside our own candidates directory.
    """
    from .rssm_continuation import complete_quality_gate
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    candidate_dir = root/'candidates'
    selector_path = root/'navigation_best.json'
    rank = navigation_validation_rank(validation, expected_fingerprints=expected_fingerprints)
    step = _integer(checkpoint.get('training_steps'), 'checkpoint training_steps')
    if checkpoint.get('stage') != 'world' or step != validation['step']:
        raise ValueError('Candidate weights and actual validation must have the identical optimizer step')
    if checkpoint.get('global_training_step', step) != step:
        raise ValueError('Candidate global/training step mismatch')
    current = json.loads(selector_path.read_text(encoding='utf-8')) if selector_path.exists() else {
        'version': VERSION, 'expected_fingerprints': dict(expected_fingerprints),
        'best_physical': None, 'best_qualified': None, 'automatic_deployment': False}
    if current.get('version') != VERSION or current.get('expected_fingerprints') != expected_fingerprints:
        raise ValueError('Physical best selector belongs to different validation panels')
    for previous in (current['best_physical'], current['best_qualified']):
        if not previous:
            continue
        path = Path(previous['checkpoint']).resolve()
        if path.parent != candidate_dir.resolve() or not path.name.startswith('navigation_g') or path.suffix != '.pt':
            raise ValueError('Best selector references a file outside its owned candidates directory')
        if not path.is_file() or file_sha256(path) != previous['checkpoint_sha256']:
            raise ValueError('Previously selected immutable navigation checkpoint is missing or changed')
    gate_passed = complete_quality_gate(validation)['passed']
    proof = qualification or checkpoint.get('rssm_training', {}).get('continuation', {})
    passes, required = proof.get('consecutive_passes'), proof.get('required_consecutive_passes')
    qualified = bool(gate_passed and proof.get('status') == 'qualified'
        and isinstance(passes, int) and not isinstance(passes, bool)
        and isinstance(required, int) and not isinstance(required, bool) and required >= 2 and passes >= required
        and proof.get('last_validation_passed') is True and proof.get('last_validation_step') == step
        and isinstance(proof.get('fingerprint'), str) and proof['fingerprint']
        and proof['fingerprint'] == validation.get('continuation_id'))
    better_physical = not current['best_physical'] or rank < tuple(current['best_physical']['rank'])
    better_qualified = qualified and (not current['best_qualified'] or rank < tuple(current['best_qualified']['rank']))
    if not better_physical and not better_qualified:
        return current
    candidate_dir.mkdir(parents=True, exist_ok=True)
    model = {name: value.detach().cpu().clone() for name, value in checkpoint['model'].items()}
    model_hash = _model_digest(model)
    report_hash = _fingerprint(validation)
    identity = {'step': step, 'rank': list(rank), 'qualified': qualified,
                'complete_quality_gate_passed': gate_passed,
                'expected_fingerprints': dict(expected_fingerprints), 'validation_sha256': report_hash,
                'model_sha256': model_hash, 'automatic_deployment': False}
    filename = f'navigation_g{step:07d}_{model_hash[:12]}_{report_hash[:12]}.pt'
    candidate = candidate_dir/filename
    keep = ('stage', 'state_schema', 'config', 'training_steps', 'global_training_step',
            'rssm_migration', 'source_teacher_checkpoint', 'quality_status', 'plan_sha256')
    payload = {key: deepcopy(checkpoint[key]) for key in keep if key in checkpoint}
    if not isinstance(payload.get('config'), dict) or not payload['config']:
        raise ValueError('Minimal navigation candidate requires the actual model configuration')
    payload.update(model=model, navigation_candidate=identity, navigation_validation=deepcopy(validation),
                   navigation_assets=deepcopy(assets_metadata or {}))
    if not candidate.exists():
        temporary = candidate.with_name(candidate.name+f'.{os.getpid()}.{uuid.uuid4().hex}.pending')
        try:
            with temporary.open('xb') as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            replace_file(temporary, candidate)
        finally:
            if temporary.exists():
                temporary.unlink()
    _verify_candidate(candidate, identity)
    descriptor = {**identity, 'checkpoint': str(candidate), 'checkpoint_sha256': file_sha256(candidate)}
    updated = deepcopy(current)
    if better_physical:
        updated['best_physical'] = descriptor
    if better_qualified:
        updated['best_qualified'] = descriptor
    # The candidate exists and has been reopened successfully BEFORE the pointer moves.
    atomic_json(selector_path, updated)
    if prune_replaced:
        retained = {value['checkpoint'] for value in (updated['best_physical'], updated['best_qualified']) if value}
        for previous in (current['best_physical'], current['best_qualified']):
            if not previous or previous['checkpoint'] in retained:
                continue
            path = Path(previous['checkpoint']).resolve()
            if path.parent != candidate_dir.resolve() or not path.name.startswith('navigation_g') or path.suffix != '.pt':
                raise ValueError('Refusing candidate cleanup outside the owned candidates directory')
            if path.exists():
                path.unlink()
    return updated
