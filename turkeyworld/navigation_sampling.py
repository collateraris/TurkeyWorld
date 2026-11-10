"""Checkpoint-external, disjoint navigation sampling on an immutable corpus.

Source labels select ordinary (episode, frame_start) pairs only. They never
enter model inputs. Weighted-fair case counters belong to the caller's saved
training state; this sampler has no evolving RNG or hidden draw counter.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .engine import BEHAVIOR_VERSION
from .structured_state import GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS, STATE_DIM, STATE_SCHEMA_VERSION


NAVIGATION_FOCUS_VERSION = 'navigation_focus_disjoint_v1'
NAVIGATION_CASES = ('slide', 'short_return', 'wall_stop', 'turn', 'straight', 'strafe')
NAVIGATION_FOCUS_WEIGHTS = dict(zip(NAVIGATION_CASES, (.30, .25, .20, .15, .05, .05)))


def _positive_integer(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


class NavigationFocusSampler:
    """Choose whole source windows with exact, resumable weighted-fair quotas.

    Case membership is the recorded ``movement_case``, so six groups cannot
    overlap. Return windows additionally contain a forward-to-backward command
    reversal in their future. For horizon1 only, a reversal on its predicted
    transition is eligible using the preceding action from observed context.
    """

    def __init__(self, data, context=2, horizon=29, weights=None):
        _positive_integer(context, 'context')
        _positive_integer(horizon, 'horizon')
        if getattr(data, 'split', None) != 'train':
            raise ValueError('Navigation focus requires train-only data')
        supplied = dict(NAVIGATION_FOCUS_WEIGHTS if weights is None else weights)
        if set(supplied) != set(NAVIGATION_CASES):
            raise ValueError('All six disjoint navigation case weights are required')
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) or v <= 0 for v in supplied.values()):
            raise ValueError('Navigation case weights must be finite and positive')
        if not math.isclose(sum(supplied.values()), 1., rel_tol=0., abs_tol=1e-10):
            raise ValueError('Navigation case weights must sum to one')
        fractions = [Fraction(str(supplied[name])) for name in NAVIGATION_CASES]
        denominator = math.lcm(*(value.denominator for value in fractions))
        self._numerators = [value.numerator*(denominator//value.denominator) for value in fractions]
        self._denominator = sum(self._numerators)
        self.weights = {name: float(value/sum(fractions)) for name, value in zip(NAVIGATION_CASES, fractions)}
        self.context, self.horizon = context, horizon
        paths = [Path(episode['path']).resolve() for episode in data.episodes]
        directory = Path(data.directory).resolve()
        if (directory.name != 'train' or not paths or len(paths) != len(set(paths))
                or any(path.parent != directory or path.suffix != '.npz' or not path.is_file() for path in paths)):
            raise ValueError('Navigation focus requires unique existing NPZ files inside one train directory')
        digest = hashlib.sha256(json.dumps({'version': NAVIGATION_FOCUS_VERSION,
            'context': context, 'horizon': horizon, 'weights': self.weights,
            'behavior_version': BEHAVIOR_VERSION, 'state_schema_version': STATE_SCHEMA_VERSION}, sort_keys=True).encode())
        groups = {name: [] for name in NAVIGATION_CASES}
        clips, dropped_return_windows = Counter(), 0
        self.paths = tuple(paths)
        for episode, path in enumerate(paths):
            with np.load(path, allow_pickle=False) as saved:
                required = {'states', 'actions', 'dt', 'split', 'stage', 'movement_case',
                    'dataset_kind', 'world_transition_compatible', 'behavior_version',
                    'state_dim', 'entity_slots', 'state_schema_version'}
                if not required.issubset(saved.files):
                    raise ValueError(f'{path.name}: incomplete navigation provenance')
                if (str(saved['split']) != 'train' or str(saved['stage']) != 'navigation'
                        or str(saved['dataset_kind']) != 'world_transitions'
                        or not bool(saved['world_transition_compatible'])):
                    raise ValueError(f'{path.name}: requires navigation train world transitions')
                if (str(saved['behavior_version']) != BEHAVIOR_VERSION
                        or int(saved['state_dim']) != STATE_DIM or int(saved['entity_slots']) != ENTITY_SLOTS
                        or str(saved['state_schema_version']) != STATE_SCHEMA_VERSION):
                    raise ValueError(f'{path.name}: incompatible source behavior/state schema')
                states, actions = saved['states'], saved['actions']
                dt, case = float(saved['dt']), str(saved['movement_case'])
            if (case not in groups or actions.ndim != 2 or actions.shape[1:] != (4,)
                    or states.shape != (len(actions)+1, STATE_DIM) or not len(actions)
                    or not np.isfinite(states).all() or not np.isfinite(actions).all()
                    or not math.isclose(dt, 1/30, rel_tol=0., abs_tol=1e-7)):
                raise ValueError(f'{path.name}: invalid finite navigation state/action sequence')
            npc = states[:, GLOBAL_DIM:-2].reshape(len(states), ENTITY_SLOTS, ENTITY_DIM)
            if not (states[:, 4] > 0).all() or ((npc[:, :, 4] > 0) | (npc[:, :, 5] >= .5)).any():
                raise ValueError(f'{path.name}: navigation requires a living player and no active NPC')
            clips[case] += 1
            starts = np.arange(max(0, len(states)-context-horizon+1), dtype=np.int64)
            if case == 'short_return' and len(starts):
                # Reversal index is the action that first commands backwards;
                # it must belong to this window's future target actions.
                flips = np.zeros(len(actions), dtype=bool)
                flips[1:] = (actions[:-1, 0] > .1) & (actions[1:, 0] < -.1)
                prefix = np.r_[0, np.cumsum(flips, dtype=np.int64)]
                first = starts+context-1
                # Longer windows contain BOTH commands in the predicted
                # future. For horizon1 the observed preceding action gives
                # the only meaningful command-change definition.
                earliest = first if horizon == 1 else first+1
                eligible = prefix[first+horizon]-prefix[earliest] > 0
                dropped_return_windows += int((~eligible).sum())
                starts = starts[eligible]
            groups[case].extend((episode, int(start)) for start in starts)
            digest.update(str(path).encode())
            digest.update(case.encode())
            for value in (states, actions):
                digest.update(value.dtype.str.encode())
                digest.update(np.ascontiguousarray(value).tobytes())
            digest.update(float(dt).hex().encode())
        self.groups = {name: np.asarray(values, dtype=np.int64).reshape(-1, 2) for name, values in groups.items()}
        missing = [name for name in NAVIGATION_CASES if not len(self.groups[name])]
        if missing:
            raise ValueError('No eligible future windows for required navigation cases: '+', '.join(missing))
        self.fingerprint = digest.hexdigest()
        self.clip_counts = dict(clips)
        self.dropped_return_windows = dropped_return_windows

    def sample(self, batch_size, rng, *, counts=None, draws=0):
        """Return indices, updated case counters, total draws, and batch counts.

        Save returned counters together with the caller RNG at optimizer-update
        boundaries. Restoring both reproduces subsequent batches exactly.
        """
        _positive_integer(batch_size, 'batch_size')
        if not isinstance(rng, np.random.Generator):
            raise ValueError('Sampling requires a caller-owned NumPy Generator')
        if not isinstance(draws, int) or isinstance(draws, bool) or draws < 0:
            raise ValueError('draws must be a nonnegative integer')
        if counts is None:
            if draws:
                raise ValueError('A resumed draw count requires all persisted case counters')
            counts = {name: 0 for name in NAVIGATION_CASES}
        if (set(counts) != set(NAVIGATION_CASES)
                or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in counts.values())
                or sum(counts.values()) != draws):
            raise ValueError('Case counters must be nonnegative integers summing to draws')
        updated = dict(counts)
        selected = []
        for offset in range(batch_size):
            t = draws+offset+1
            index = max(range(len(NAVIGATION_CASES)), key=lambda i:
                t*self._numerators[i]-updated[NAVIGATION_CASES[i]]*self._denominator)
            case = NAVIGATION_CASES[index]
            selected.append(case)
            updated[case] += 1
        selected = np.asarray(selected)[rng.permutation(batch_size)]
        result = np.empty((batch_size, 2), dtype=np.int64)
        sampled = Counter()
        for case in NAVIGATION_CASES:
            positions = np.flatnonzero(selected == case)
            if len(positions):
                choices = self.groups[case]
                result[positions] = choices[rng.integers(len(choices), size=len(positions))]
                sampled[case] = int(len(positions))
        return result, updated, draws+batch_size, dict(sampled)

    def summary(self):
        return {'version': NAVIGATION_FOCUS_VERSION, 'fingerprint': self.fingerprint,
            'context': self.context, 'horizon': self.horizon, 'weights': dict(self.weights),
            'clips_per_case': dict(self.clip_counts),
            'eligible_windows_per_case': {name: len(self.groups[name]) for name in NAVIGATION_CASES},
            'dropped_return_windows_without_future_reversal': self.dropped_return_windows,
            'disjoint_source_cases': True, 'labels_used_only_for_sampling': True,
            'source_future_feedback': False, 'behavior_version': BEHAVIOR_VERSION,
            'state_dim': STATE_DIM, 'entity_slots': ENTITY_SLOTS}
