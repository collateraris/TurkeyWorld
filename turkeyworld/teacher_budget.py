"""Durable limits for a user-selected segment of teacher optimization."""
from __future__ import annotations

import json
from pathlib import Path

from .training_pipeline import atomic_json, now


def prepare_teacher_budget(path, *, phase, current_optimizer_step, current_phase_step,
                           limit_updates, subset_fingerprint):
    """Reuse a segment's original boundary across crashes and ordinary restarts."""
    path = Path(path)
    if min(current_optimizer_step, current_phase_step) < 0 or limit_updates < 1:
        raise ValueError('Teacher budget requires nonnegative counters and a positive limit')
    if not subset_fingerprint:
        raise ValueError('Teacher budget requires a fixed training subset fingerprint')
    if path.exists():
        budget = json.loads(path.read_text(encoding='utf-8'))
        expected = {'phase': phase, 'limit_updates': limit_updates,
                    'subset_fingerprint': subset_fingerprint}
        if any(budget.get(key) != value for key, value in expected.items()):
            raise ValueError('Existing teacher budget belongs to a different training segment')
        if (budget['stop_optimizer_step'] != budget['start_optimizer_step']+limit_updates
                or budget['stop_phase_step'] != budget['start_phase_step']+limit_updates):
            raise ValueError('Inconsistent saved teacher budget boundary')
        if not budget['start_optimizer_step'] <= current_optimizer_step <= budget['stop_optimizer_step']:
            raise ValueError('Checkpoint is outside the saved teacher budget')
        if current_phase_step-budget['start_phase_step'] != current_optimizer_step-budget['start_optimizer_step']:
            raise ValueError('Teacher phase and global counters disagree with the saved budget')
        if budget.get('status') == 'awaiting_user' and current_optimizer_step != budget['stop_optimizer_step']:
            raise ValueError('Completed teacher budget has no matching final checkpoint')
        return budget
    budget = {'format_version': 1, 'scope': 'teacher', 'phase': phase,
              'subset_fingerprint': subset_fingerprint, 'limit_updates': limit_updates,
              'start_optimizer_step': current_optimizer_step,
              'stop_optimizer_step': current_optimizer_step+limit_updates,
              'start_phase_step': current_phase_step,
              'stop_phase_step': current_phase_step+limit_updates,
              'status': 'running', 'extended_data_requires_user': True,
              'created_at': now()}
    atomic_json(path, budget)
    return budget


def teacher_budget_progress(budget, current_optimizer_step):
    current = int(current_optimizer_step)
    completed = current-budget['start_optimizer_step']
    if not 0 <= completed <= budget['limit_updates']:
        raise ValueError('Optimizer update exceeds the authorized teacher segment')
    return {**budget, 'current_optimizer_step': current, 'completed_updates': completed,
            'remaining_updates': budget['limit_updates']-completed}


class TeacherBudgetFinished(Exception):
    """Exit the teacher pipeline before automatic repair, handoff or student work."""
