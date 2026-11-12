"""Physical drift ranking from a fixed held-out Source trajectory panel."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import os
import shutil

import numpy as np


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {'count': int(values.size), 'mean': float(values.mean()) if values.size else None,
            'p95': float(np.percentile(values, 95)) if values.size else None}


def source_masked_drift_score(predicted, reference, *, panel_fingerprint):
    """Source life masks retain missing predicted actors, including later lives.

    All 450 continuing-alive transitions contribute. Exact Source respawn
    transitions are excluded. Neither predicted HP nor predicted alive can
    remove a coordinate error from the ranking.
    """
    predicted, reference = (np.asarray(value, dtype=np.float64) for value in (predicted, reference))
    if (predicted.shape != reference.shape or predicted.ndim != 3 or predicted.shape[0] < 1
            or predicted.shape[1:] != (451, 20) or not panel_fingerprint
            or not np.isfinite(predicted).all() or not np.isfinite(reference).all()):
        raise ValueError('Best drift requires finite, matched held-out[B,451,20] arrays and panel identity')
    player_alive = reference[..., 4] > 0
    npc_alive = (reference[..., 12] > 0) & (reference[..., 13] >= .5)
    masks = {'player': player_alive[:, :-1] & player_alive[:, 1:],
             'npc': npc_alive[:, :-1] & npc_alive[:, 1:]}
    errors = {'player': np.linalg.norm((predicted[:, 1:, :2]-reference[:, 1:, :2])*16, axis=-1),
              'npc': np.linalg.norm((predicted[:, 1:, 8:10]-reference[:, 1:, 8:10])*16, axis=-1)}
    result = {'version': 1, 'panel_fingerprint': panel_fingerprint, 'horizon': 450,
        'count': predicted.shape[0], 'source_life_mask_only': True,
        'exact_source_spawn_frame_excluded': True, 'predicted_missing_actors_retained': True,
        'normalization_tiles': {'player': .1, 'npc': .25},
        'source_reference_sha256': hashlib.sha256(np.ascontiguousarray(reference).tobytes()).hexdigest()}
    for who in ('player', 'npc'):
        values = errors[who][masks[who]]
        if not values.size:
            raise ValueError('Best drift requires continuing-alive Source coverage for '+who)
        result[who] = _stats(values)
        result[who+'_final'] = _stats(errors[who][:, -1][masks[who][:, -1]])
    result['normalized_drift_mean'] = .5*(result['player']['mean']/.1+result['npc']['mean']/.25)
    result['normalized_drift_p95'] = .5*(result['player']['p95']/.1+result['npc']['p95']/.25)
    final = [result[who+'_final']['mean']/scale for who, scale in (('player', .1), ('npc', .25))
             if result[who+'_final']['mean'] is not None]
    result['normalized_final_mean'] = float(np.mean(final)) if final else None
    result['rank'] = [result['normalized_drift_mean'], result['normalized_drift_p95']]
    result['source_alive_player_model_dead_steps'] = int((masks['player'] & (predicted[:, 1:, 4] <= 0)).sum())
    pred_npc_alive = (predicted[:, 1:, 12] > 0) & (predicted[:, 1:, 13] >= .5)
    result['source_alive_npc_model_missing_steps'] = int((masks['npc'] & ~pred_npc_alive).sum())
    return result


def heldout_drift_score(report, arrays_path):
    """Require genuine full-horizon validation metadata before opening arrays."""
    if (report.get('source_held_out') is not True or report.get('source_future_feedback') is not False
            or report.get('future_rgb_encoded') is not False or report.get('horizon') != 450
            or report.get('all_states_finite') is not True
            or report.get('all_memory_finite_and_shape_valid') is not True):
        raise ValueError('Best drift only accepts finite autonomous held-out450 validation')
    with np.load(arrays_path, allow_pickle=False) as saved:
        score = source_masked_drift_score(saved['predicted'], saved['reference'],
                                         panel_fingerprint=report.get('fingerprint'))
        # Geometry is measured even when the MODEL calls a Source-alive actor dead.
        from .runtime_checks import circle_is_clear, PLAYER_RADIUS, NPC_RADIUS
        maps = saved['maps']; p = saved['predicted']; t = saved['reference']
        walls = {'player': 0, 'npc': 0}
        if maps.shape != (p.shape[0], 16, 16) or report.get('count') != p.shape[0]:
            raise ValueError('Best drift map/panel count does not match its trajectory report')
        for row in range(p.shape[0]):
            for offset in range(1, 451):
                if t[row, offset-1, 4] > 0 and t[row, offset, 4] > 0:
                    walls['player'] += int(not circle_is_clear(maps[row], p[row, offset, :2]*16, PLAYER_RADIUS))
                if (t[row, offset-1, 12] > 0 and t[row, offset-1, 13] >= .5
                        and t[row, offset, 12] > 0 and t[row, offset, 13] >= .5):
                    walls['npc'] += int(not circle_is_clear(maps[row], p[row, offset, 8:10]*16, NPC_RADIUS))
        score['source_alive_model_wall_steps'] = walls
    return score


def drift_candidate_improves(candidate, previous):
    if previous is None:
        return True
    for key in ('panel_fingerprint', 'source_reference_sha256', 'normalization_tiles', 'horizon', 'count'):
        if candidate.get(key) != previous.get(key):
            raise ValueError('Best drift selection cannot compare changed Source panels or mask conventions')
    return tuple(candidate['rank']) < tuple(previous['rank'])


def copy_drift_checkpoint(source, destination):
    """Publish the full saved model+Adam before the caller publishes its selector."""
    from .training_pipeline import replace_file
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name+f'.{os.getpid()}.tmp')
    try:
        with source.open('rb') as reader, temporary.open('xb') as writer:
            shutil.copyfileobj(reader, writer, length=8*1024**2)
            writer.flush(); os.fsync(writer.fileno())
        replace_file(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def publish_best_drift(saved_checkpoint, run_dir, *, step, score, validation_report, combined_quality_passed):
    """Commit candidate bytes, then alias, then selector; never deploy to a game."""
    from .training_pipeline import atomic_json, replace_file
    run_dir = Path(run_dir)
    selector = run_dir/'best_drift.json'
    previous = json.loads(selector.read_text(encoding='utf-8')) if selector.exists() else None
    if not drift_candidate_improves(score, previous['score'] if previous else None):
        return previous
    directory = run_dir/'checkpoints/best_drift_candidates'; directory.mkdir(parents=True, exist_ok=True)
    candidate = directory/f'world_g{step:06d}.pt'
    if candidate.exists():
        raise ValueError('An immutable drift candidate already exists at this optimizer step')
    copy_drift_checkpoint(saved_checkpoint, candidate)
    alias = run_dir/'checkpoints/best_drift.pt'
    temporary = alias.with_name(alias.name+f'.{os.getpid()}.tmp')
    try:
        try:
            os.link(candidate, temporary)
        except OSError:
            copy_drift_checkpoint(candidate, temporary)
        replace_file(temporary, alias)
    finally:
        if temporary.exists(): temporary.unlink()
    value = {'version': 1, 'step': step, 'checkpoint': str(candidate.resolve()),
        'checkpoint_alias': str(alias.resolve()), 'score': score,
        'validation_report': str(Path(validation_report).resolve()),
        'combined_quality_passed': bool(combined_quality_passed),
        'contains_full_model_and_Adam': True, 'automatic_runtime_deployment': False}
    atomic_json(selector, value)
    if previous:
        old = Path(previous['checkpoint']).resolve()
        if old.parent == directory.resolve() and old != candidate.resolve() and old.name.startswith('world_g'):
            old.unlink(missing_ok=True)
    return value
