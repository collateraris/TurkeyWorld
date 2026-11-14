"""Training-only H450 TOTAL<1 for ten consecutive completed optimizer ticks.

This fresh revision clock does not reinterpret any historical stage streak.
It measures the actual committed TOTAL as supplied by the trainer; Source
support diagnostics, EMA, validation and component losses are not conditions.
"""
from copy import deepcopy
import math


PROTOCOL = 'rssm_h450_total_loss_goal_v1'
THRESHOLD = 1.
REQUIRED_CONSECUTIVE = 10


def h450_goal_spec():
    return dict(protocol=PROTOCOL, version=1, horizon=450, metric='train_loss',
        threshold=THRESHOLD, comparison='strict_less_than',
        required_consecutive_passes=REQUIRED_CONSECUTIVE,
        completed_optimizer_updates_only=True, activation_streak=0,
        historical_updates_reinterpreted=False, non_h450_streak=0,
        reset_on_total_at_or_above_threshold=True,
        source_support_eligibility_is_goal_requirement=False,
        EMA_is_goal_measurement=False, component_loss_is_goal_measurement=False,
        validation_loss_is_goal_measurement=False,
        final_stage_max_updates=None, budget_completion_is_goal_success=False,
        gameplay_quality_qualified=False)


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + ' must be an exact integer >= ' + str(minimum))
    return value


def _sha(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in '0123456789abcdef' for c in value)):
        raise ValueError('H450 goal requires its immutable plan SHA256')
    return value


def _total(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('H450 goal requires finite nonnegative committed TOTAL')
    return float(value)


def initial_h450_goal(*, source_optimizer_step, plan_sha256):
    """Start at streak0 regardless of source horizon, loss or old pass streak."""
    source = _integer(source_optimizer_step, 'source_optimizer_step')
    return dict(protocol=PROTOCOL, version=1, settings=h450_goal_spec(),
        source_optimizer_step=source, plan_sha256=_sha(plan_sha256),
        last_optimizer_step=source, updates=0, h450_updates=0, streak=0,
        goal_met=False, last_horizon=None, last_total_loss=None,
        last_gate_passed=False, qualifying_optimizer_steps=[], qualifying_totals=[],
        success_optimizer_step=None)


def validate_h450_goal(record, *, current_step=None):
    """Validate the bounded evidence of the complete qualifying streak."""
    expected_keys = set(initial_h450_goal(source_optimizer_step=0, plan_sha256='0'*64))
    if (not isinstance(record, dict) or set(record) != expected_keys
            or record.get('protocol') != PROTOCOL or type(record.get('version')) is not int
            or record['version'] != 1 or record.get('settings') != h450_goal_spec()):
        raise ValueError('Saved H450 goal schema or user criterion changed')
    _sha(record['plan_sha256'])
    source = _integer(record['source_optimizer_step'], 'source_optimizer_step')
    last = _integer(record['last_optimizer_step'], 'last_optimizer_step')
    updates = _integer(record['updates'], 'updates')
    final_updates = _integer(record['h450_updates'], 'h450_updates')
    streak = _integer(record['streak'], 'streak')
    if (last != source + updates or final_updates > updates or streak > final_updates
            or streak > REQUIRED_CONSECUTIVE
            or current_step is not None and _integer(current_step, 'current_step') != last
            or type(record['goal_met']) is not bool or record['goal_met'] != (streak == REQUIRED_CONSECUTIVE)
            or type(record['last_gate_passed']) is not bool):
        raise ValueError('Saved H450 goal optimizer clock, streak or result changed')
    steps, losses = record['qualifying_optimizer_steps'], record['qualifying_totals']
    if (not isinstance(steps, list) or not isinstance(losses, list)
            or len(steps) != streak or len(losses) != streak
            or any(type(step) is not int or not source < step <= last for step in steps)
            or any(b != a + 1 for a, b in zip(steps, steps[1:]))
            or any(_total(loss) >= THRESHOLD for loss in losses)):
        raise ValueError('Saved H450 goal lacks consecutive qualifying TOTAL evidence')
    if not updates:
        if (last != source or final_updates or streak or record['last_horizon'] is not None
                or record['last_total_loss'] is not None or record['last_gate_passed']
                or record['success_optimizer_step'] is not None):
            raise ValueError('Fresh H450 goal must not reuse historical measurements')
    else:
        horizon = _integer(record['last_horizon'], 'last_horizon', minimum=1)
        loss = _total(record['last_total_loss'])
        if horizon > 450:
            raise ValueError('H450 goal received a horizon outside the curriculum')
        passed = horizon == 450 and loss < THRESHOLD
        if (record['last_gate_passed'] is not passed or passed != bool(streak)
                or horizon == 450 and not final_updates
                or streak and (steps[-1] != last or _total(losses[-1]) != loss)
                or record['success_optimizer_step'] != (last if record['goal_met'] else None)):
            raise ValueError('Saved H450 goal last completed TOTAL decision changed')
    return record


def commit_h450_goal(record, *, step, horizon, total_loss, optimizer_completed=True):
    """Return a new clock after exactly one completed optimizer update.

    The input record is unchanged. Non-H450 updates advance the revision clock
    but never contribute to its final-horizon streak. There is no update cap.
    """
    validate_h450_goal(record)
    if record['goal_met']:
        raise ValueError('Completed H450 goal cannot accept another optimizer update')
    if optimizer_completed is not True or type(optimizer_completed) is not bool:
        raise ValueError('H450 goal counts only a completed optimizer update')
    step = _integer(step, 'step', minimum=1)
    horizon = _integer(horizon, 'horizon', minimum=1)
    loss = _total(total_loss)
    if step != record['last_optimizer_step'] + 1 or horizon > 450:
        raise ValueError('H450 goal requires the next consecutive optimizer clock and current horizon')
    result = deepcopy(record)
    result['updates'] += 1
    result['last_optimizer_step'] = step
    result['h450_updates'] += int(horizon == 450)
    result['last_horizon'] = horizon
    result['last_total_loss'] = loss
    passed = horizon == 450 and loss < THRESHOLD
    result['last_gate_passed'] = passed
    if passed:
        result['qualifying_optimizer_steps'].append(step)
        result['qualifying_totals'].append(loss)
        result['streak'] += 1
    else:
        result['qualifying_optimizer_steps'] = []
        result['qualifying_totals'] = []
        result['streak'] = 0
    result['goal_met'] = result['streak'] == REQUIRED_CONSECUTIVE
    result['success_optimizer_step'] = step if result['goal_met'] else None
    return validate_h450_goal(result, current_step=step)
