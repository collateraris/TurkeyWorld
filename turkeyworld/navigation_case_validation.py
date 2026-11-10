"""Source-case navigation summaries from an already completed held-out rollout.

No model, RGB decoding or source-engine transitions are involved. NPC state and
collisions never enter these player-only metrics. Missing cohorts stay missing.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .engine import BEHAVIOR_VERSION
from .navigation_sampling import NAVIGATION_CASES
from .runtime_checks import circle_is_clear, segment_is_clear, PLAYER_RADIUS
from .structured_state import ENTITY_SLOTS, STATE_DIM, STATE_SCHEMA_VERSION


def _metric(values, kind='mean'):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return None
    return float(np.percentile(values, 95) if kind == 'p95' else
                 values.max() if kind == 'max' else values.mean())


def summarize_navigation_cases(trajectory_dir, data_dir):
    """Return all six disjoint Source ``movement_case`` cohorts as JSON data.

    ``data_dir`` is the navigation stage root or its ``rgb/val`` directory.
    Source-alive labels mask evaluation only. Predicted death cannot remove an
    error. The final-state distribution and pooled future-step distribution
    have separate names/counts; an absent/dead cohort has None error metrics.
    """
    trajectory = Path(trajectory_dir)
    directory = Path(data_dir).resolve()
    if directory.name != 'val':
        directory = directory/'rgb'/'val'
    if directory.name != 'val' or not directory.is_dir():
        raise ValueError('Navigation case validation requires an existing held-out val directory')
    report = json.loads((trajectory/'heldout_trajectory_validation.json').read_text(encoding='utf-8'))
    panel = report.get('panel', {})
    if (panel.get('split') != 'val' or report.get('source_held_out') is not True
            or report.get('source_future_feedback') is not False
            or report.get('future_rgb_encoded') is not False):
        raise ValueError('Navigation cases require held-out val trajectories without source-future feedback')
    fingerprint = hashlib.sha256(json.dumps(panel, sort_keys=True).encode()).hexdigest()
    if report.get('fingerprint') != fingerprint:
        raise ValueError('Trajectory panel fingerprint does not match its specification')
    with np.load(trajectory/'heldout_trajectory_states.npz', allow_pickle=False) as saved:
        predicted, source, maps, indices, dt, actions = [saved[key].copy() for key in
            ('predicted', 'reference', 'maps', 'indices', 'dt', 'actions')]
    if predicted.ndim != 3 or predicted.shape[-1] != STATE_DIM:
        raise ValueError('Trajectory arrays require the current state schema')
    batch, frames, _ = predicted.shape
    horizon = frames-1
    context = panel.get('context')
    windows = panel.get('windows', [])
    if (batch < 1 or horizon < 1 or source.shape != predicted.shape
            or maps.shape != (batch, 16, 16) or indices.shape != (batch, 2)
            or indices.dtype.kind not in 'iu' or dt.shape != (batch,)
            or actions.shape != (batch, horizon, 4) or len(windows) != batch
            or report.get('count') != batch or report.get('horizon') != horizon
            or panel.get('horizon') != horizon or report.get('context') != context
            or not isinstance(context, int) or isinstance(context, bool) or context < 1
            or not all(np.isfinite(value).all() for value in (predicted, source, maps, actions, dt))
            or not (dt > 0).all()):
        raise ValueError('Trajectory shapes, finite values or panel dimensions are inconsistent')
    files = sorted(directory.glob('*.npz'))
    if not files or any(path.resolve().parent != directory for path in files):
        raise ValueError('Validation source files must remain inside their val directory')
    identities, cohorts = [], {name: [] for name in NAVIGATION_CASES}
    for row, window in enumerate(windows):
        episode, start = map(int, indices[row])
        if (episode < 0 or episode >= len(files) or start < 0
                or window.get('episode_index') != episode or window.get('start') != start
                or window.get('episode') != files[episode].name):
            raise ValueError('Trajectory indices and episode/start identities do not match the validation source')
        path = files[episode]
        with np.load(path, allow_pickle=False) as saved:
            required = {'split', 'stage', 'movement_case', 'dataset_kind', 'world_transition_compatible',
                'behavior_version', 'state_dim', 'entity_slots', 'state_schema_version', 'states', 'actions', 'map', 'dt'}
            if not required.issubset(saved.files):
                raise ValueError(f'{path.name}: incomplete navigation validation provenance')
            if (str(saved['split']) != 'val' or str(saved['stage']) != 'navigation'
                    or str(saved['dataset_kind']) != 'world_transitions'
                    or not bool(saved['world_transition_compatible'])
                    or str(saved['behavior_version']) != BEHAVIOR_VERSION
                    or int(saved['state_dim']) != STATE_DIM or int(saved['entity_slots']) != ENTITY_SLOTS
                    or str(saved['state_schema_version']) != STATE_SCHEMA_VERSION):
                raise ValueError(f'{path.name}: incompatible stage/split/behavior/state provenance')
            case = str(saved['movement_case'])
            states, controls, tilemap, source_dt = saved['states'], saved['actions'], saved['map'], float(saved['dt'])
        first = start+context-1
        if (case not in cohorts or states.shape != (len(controls)+1, STATE_DIM)
                or controls.ndim != 2 or controls.shape[1:] != (4,) or first+horizon >= len(states)
                or not np.array_equal(states[first:first+horizon+1], source[row])
                or not np.array_equal(controls[first:first+horizon], actions[row])
                or not np.array_equal(tilemap, maps[row])
                or not np.isclose(source_dt, dt[row], rtol=0., atol=1e-7)
                or not np.array_equal(predicted[row, 0], source[row, 0])):
            raise ValueError(f'{path.name}: saved rollout does not match the exact Source window/context')
        stat = path.stat()
        identities.append({'episode': path.name, 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
        cohorts[case].append(row)
    identity = hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()
    if panel.get('episode_identity_sha256') != identity:
        raise ValueError('Validation source episode identity changed since the trajectory was measured')
    error = np.linalg.norm((predicted[:, 1:, :2]-source[:, 1:, :2])*16., axis=-1)
    valid = (source[:, :-1, 4] > 0) & (source[:, 1:, 4] > 0)
    heading = np.degrees(np.abs(np.arctan2(
        predicted[:, 1:, 2]*source[:, 1:, 3]-predicted[:, 1:, 3]*source[:, 1:, 2],
        predicted[:, 1:, 3]*source[:, 1:, 3]+predicted[:, 1:, 2]*source[:, 1:, 2])))
    heading[np.linalg.norm(predicted[:, 1:, 2:4], axis=-1) < 1e-6] = 180.
    overlaps, invalid_paths = np.zeros_like(valid), np.zeros_like(valid)
    for row in range(batch):
        for offset in np.flatnonzero(valid[row]):
            start_xy, end_xy = predicted[row, offset:offset+2, :2]*16.
            overlaps[row, offset] = not circle_is_clear(maps[row], end_xy, PLAYER_RADIUS)
            invalid_paths[row, offset] = not segment_is_clear(maps[row], start_xy, end_xy, PLAYER_RADIUS)
    result = {}
    for case, rows in cohorts.items():
        rows = np.asarray(rows, dtype=np.int64)
        endpoints = [error[row, -1] for row in rows if valid[row, -1]]
        pooled = error[rows][valid[rows]]
        case_heading = heading[rows][valid[rows]]
        per_window = []
        for row in rows:
            hits = np.flatnonzero(overlaps[row])
            per_window.append({'episode': windows[row]['episode'], 'start': int(indices[row, 1]),
                'valid_steps': int(valid[row].sum()),
                'endpoint_xy_tiles': float(error[row, -1]) if valid[row, -1] else None,
                'wall_collision_steps': int(overlaps[row].sum()),
                'first_player_wall_offset': int(hits[0]+1) if len(hits) else None,
                'first_player_wall_seconds': float((hits[0]+1)*dt[row]) if len(hits) else None})
        result[case] = {'count': int(len(rows)), 'windows': int(len(rows)), 'valid_endpoint_count': len(endpoints),
            'valid_steps': int(valid[rows].sum()), 'horizon': horizon,
            'mean_endpoint_xy_tiles': _metric(endpoints), 'p95_endpoint_xy_tiles': _metric(endpoints, 'p95'),
            'max_endpoint_xy_tiles': _metric(endpoints, 'max'), 'mean_all_steps_xy_tiles': _metric(pooled),
            'p95_all_steps_xy_tiles': _metric(pooled, 'p95'),
            'wall_collision_windows': int(overlaps[rows].any(-1).sum()),
            'wall_collision_steps': int(overlaps[rows].sum()),
            'wall_path_collision_windows': int(invalid_paths[rows].any(-1).sum()),
            'wall_path_collision_steps': int(invalid_paths[rows].sum()),
            'mean_heading_degrees': _metric(case_heading), 'p95_heading_degrees': _metric(case_heading, 'p95'),
            'source_held_out': True, 'player_only': True, 'source_alive_mask': True,
            'trajectory_fingerprint': fingerprint, 'per_window': per_window}
    return result
