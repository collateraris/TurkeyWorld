"""Serializable patience for distinct, fixed-protocol visual quality checks.

Numerical best and meaningful improvement are different: every lower score is
a new candidate for best weights, but patience resets only after improvement
relative to the last meaningful anchor. Small gains therefore accumulate.
"""
from __future__ import annotations

import math
from numbers import Integral


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _score(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("score must be finite and nonnegative") from error
    if isinstance(value, bool) or not math.isfinite(number) or number < 0:
        raise ValueError("score must be finite and nonnegative")
    return number


class PlateauTracker:
    """Track MAE+.5*(1-SSIM), where a smaller value is better."""

    def __init__(self, patience=6, min_relative_improvement=.005, *, phase='', fingerprint=''):
        self.patience = _integer(patience, 'patience')
        if self.patience < 1:
            raise ValueError("patience must be positive")
        self.min_relative_improvement = _score(min_relative_improvement)
        if self.min_relative_improvement > 1:
            raise ValueError("min_relative_improvement must be between zero and one")
        if not isinstance(phase, str) or not isinstance(fingerprint, str):
            raise ValueError("phase and fingerprint must be strings")
        self.phase, self.fingerprint = phase, fingerprint
        self.best_score = self.best_step = None
        self.anchor_score = self.anchor_step = None
        self.bad_checks = self.checks = 0
        self.last_step = None
        self.decision = 'continue'
        self.reset_reason = None

    def to_state(self):
        return {'version': 1, 'phase': self.phase, 'fingerprint': self.fingerprint,
                'patience': self.patience, 'min_relative_improvement': self.min_relative_improvement,
                'best_score': self.best_score, 'best_step': self.best_step,
                'anchor_score': self.anchor_score, 'anchor_step': self.anchor_step,
                'bad_checks': self.bad_checks, 'last_step': self.last_step, 'checks': self.checks,
                'decision': self.decision, 'reset_reason': self.reset_reason}

    def _result(self, *, improved, best_changed, repeated):
        return {**self.to_state(), 'improved': improved, 'best_changed': best_changed,
                'plateau': self.decision == 'plateau', 'repeated': repeated}

    def update_score(self, value, step):
        value, step = _score(value), _integer(step, 'step')
        if self.last_step is not None:
            if step < self.last_step:
                raise ValueError("Validation steps cannot move backward; restore with current_step to discard future state")
            if step == self.last_step:
                return self._result(improved=False, best_changed=False, repeated=True)
        best_changed = self.best_score is None or value < self.best_score
        improved = (self.anchor_score is None or
                    (value < self.anchor_score and
                     value <= self.anchor_score*(1-self.min_relative_improvement)))
        if best_changed:
            self.best_score, self.best_step = value, step
        if improved:
            self.anchor_score, self.anchor_step, self.bad_checks = value, step, 0
        else:
            self.bad_checks += 1
        self.checks += 1
        self.last_step = step
        self.decision = 'plateau' if self.bad_checks >= self.patience else 'continue'
        self.reset_reason = None
        return self._result(improved=improved, best_changed=best_changed, repeated=False)

    @classmethod
    def from_state(cls, state, *, patience=None, min_relative_improvement=None,
                   phase=None, fingerprint=None, current_step=None):
        """Restore exact patience, resetting for changed context or future state.

        Supply current_step when resuming model weights. A tracker whose last
        validation lies beyond those weights is discarded, rather than letting
        stale future checks force an immediate handoff.
        """
        if state is not None and not isinstance(state, dict):
            raise ValueError("Plateau state must be a dictionary or None")
        saved = state or {}
        tracker = cls(patience=saved.get('patience', 6) if patience is None else patience,
            min_relative_improvement=(saved.get('min_relative_improvement', .005)
                                      if min_relative_improvement is None else min_relative_improvement),
            phase=saved.get('phase', '') if phase is None else phase,
            fingerprint=saved.get('fingerprint', '') if fingerprint is None else fingerprint)
        if current_step is not None:
            current_step = _integer(current_step, 'current_step')
        if not saved:
            return tracker
        if saved.get('version', 1) != 1:
            raise ValueError("Unsupported plateau state version")
        previous_patience = _integer(saved.get('patience', 6), 'saved patience')
        previous_delta = _score(saved.get('min_relative_improvement', .005))
        if previous_patience < 1 or previous_delta > 1:
            raise ValueError("Saved plateau parameters are outside their valid ranges")
        last_step = saved.get('last_step')
        if last_step is not None:
            last_step = _integer(last_step, 'last_step')
        if (saved.get('phase', '') != tracker.phase or saved.get('fingerprint', '') != tracker.fingerprint
                or previous_patience != tracker.patience or previous_delta != tracker.min_relative_improvement):
            tracker.reset_reason = 'context_changed'
            return tracker
        if current_step is not None and last_step is not None and last_step > current_step:
            tracker.reset_reason = 'future_state'
            return tracker
        checks = _integer(saved.get('checks', 0), 'checks')
        bad_checks = _integer(saved.get('bad_checks', 0), 'bad_checks')
        decision = saved.get('decision', 'continue')
        if decision not in {'continue', 'plateau'}:
            raise ValueError("Unknown plateau decision")
        if checks == 0:
            if (bad_checks or last_step is not None or decision != 'continue'
                    or any(saved.get(key) is not None for key in
                           ('best_score', 'best_step', 'anchor_score', 'anchor_step'))):
                raise ValueError("An empty plateau tracker cannot contain validation history")
        else:
            if last_step is None or bad_checks >= checks:
                raise ValueError("Invalid plateau check counts/history")
            best_score, anchor_score = _score(saved.get('best_score')), _score(saved.get('anchor_score'))
            best_step = _integer(saved.get('best_step'), 'best_step')
            anchor_step = _integer(saved.get('anchor_step'), 'anchor_step')
            if best_score > anchor_score or not anchor_step <= best_step <= last_step:
                raise ValueError("Numerical best and meaningful anchor history disagree")
            expected = 'plateau' if bad_checks >= tracker.patience else 'continue'
            if decision != expected:
                raise ValueError("Plateau decision disagrees with patience")
            tracker.best_score, tracker.best_step = best_score, best_step
            tracker.anchor_score, tracker.anchor_step = anchor_score, anchor_step
        tracker.bad_checks, tracker.checks, tracker.last_step = bad_checks, checks, last_step
        tracker.decision = decision
        tracker.reset_reason = saved.get('reset_reason')
        return tracker
