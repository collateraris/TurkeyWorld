"""An explicit bounded sampling change, separate from the immutable data plan."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .navigation_sampling import NAVIGATION_CASES, NAVIGATION_FOCUS_WEIGHTS


def experiment_fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=True,
        separators=(',', ':')).encode()).hexdigest()


def sync_navigation_experiment(curriculum, step):
    state = curriculum.state.get('sampling_experiment')
    if not state:
        return None
    start, stop = state['start_optimizer_step'], state['stop_optimizer_step']
    effective_step = min(step, state.get('superseded_at_step', step))
    done = max(0, min(stop-start, effective_step-start))
    state.update(updates_completed=done, remaining_updates=stop-start-done)
    state['status'] = ('superseded' if 'superseded_at_step' in state else
        'not_started' if step < start else 'complete' if step >= stop
        else 'active' if curriculum.stage['id'] == state['stage'] else 'inactive_stage')
    return state


def bind_navigation_experiment(curriculum, path, step, microbatch):
    """Bind a new request exactly once; later resumes require identical identity."""
    path = Path(path)
    previous = curriculum.state.get('sampling_experiment')
    if not path.exists():
        if previous:
            raise ValueError('Saved navigation sampling experiment requires its immutable request file')
        return None
    config = json.loads(path.read_text(encoding='utf-8'))
    for key in ('start_optimizer_step', 'stop_optimizer_step', 'microbatch_size'):
        value = config.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError('Invalid navigation experiment integer: '+key)
    start, stop = config['start_optimizer_step'], config['stop_optimizer_step']
    if (config.get('version') != 1 or config.get('stage') != 'navigation'
            or not isinstance(config.get('id'), str) or not config['id']
            or stop <= start or config['microbatch_size'] != microbatch or microbatch < 1
            or config.get('weights') != NAVIGATION_FOCUS_WEIGHTS
            or config.get('base_plan_fingerprint') != curriculum.fingerprint
            or config.get('source_dataset_fingerprint') != curriculum.plan.get('data_fingerprint')
            or config.get('selection') != 'source_case_future_windows'
            or config.get('after_completion') != 'original_motion_curriculum'
            or not isinstance(config.get('source_fingerprint'), str) or not config['source_fingerprint']):
        raise ValueError('Navigation sampling request differs from its authorized weights, data or base plan')
    fingerprint = experiment_fingerprint(config)
    if previous:
        if previous.get('config_fingerprint') != fingerprint:
            raise ValueError('Navigation sampling request changed on resume')
        draws, counts = previous.get('draws'), previous.get('sample_counts', {})
        effective_step = min(step, previous.get('superseded_at_step', step))
        expected = max(0, min(stop-start, effective_step-start))*microbatch
        if (not isinstance(draws, int) or isinstance(draws, bool) or draws != expected
                or set(counts) != set(NAVIGATION_CASES)
                or any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in counts.values())
                or sum(counts.values()) != draws):
            raise ValueError('Navigation sampling checkpoint counters do not match completed optimizer updates')
    else:
        if step != start or curriculum.stage['id'] != 'navigation' or curriculum.horizon(step) != 29:
            raise ValueError('A new navigation sampling experiment must bind to its saved29-step navigation start')
        curriculum.state['sampling_experiment'] = {**config, 'config_fingerprint': fingerprint,
            'sample_counts': {case: 0 for case in NAVIGATION_CASES}, 'draws': 0,
            'baseline_step': None, 'baseline_report': None, 'final_validation_step': None}
    return sync_navigation_experiment(curriculum, step)


def navigation_focus_active(curriculum, step):
    state = sync_navigation_experiment(curriculum, step)
    return state is not None and state['status'] == 'active'


def supersede_navigation_focus(curriculum, step, replacement):
    state = curriculum.state.get('sampling_experiment')
    if state and 'superseded_at_step' not in state and state.get('status') == 'active':
        state.update(superseded_at_step=step, superseded_by=replacement)
        sync_navigation_experiment(curriculum, step)


def record_navigation_case_validation(curriculum, step, metrics):
    """Keep the original comparison and newest measured cohorts across resumes."""
    state = curriculum.state.get('sampling_experiment')
    if not state:
        return
    if step == state['start_optimizer_step']:
        state.setdefault('baseline_metrics_by_case', metrics)
    if step >= state.get('last_validation_step', -1):
        state.update(last_validation_step=step, last_validation_by_case=metrics)
