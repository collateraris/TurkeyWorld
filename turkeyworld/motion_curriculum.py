"""Fixed-corpus Source-motion labels used only to select training windows.

Labels may inspect recorded transition outcomes; no label/GT state is passed
to a model by this module. Every returned pair is an ordinary SequenceData
(episode_index, frame_start). Longer phases never concatenate/reset episodes.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path

import numpy as np


CURRICULUM_VERSION = 'local_motion_curriculum_v2_future_cases'
MOTION_CASES = ('straight', 'stop', 'slide', 'turn')
# Offset since the checkpoint-backed migration start, not since process launch.
PHASES = (
    (0, 'straight_1', 1, {'straight': .6, 'stop': .2, 'slide': .2}),
    (100, 'walls_4', 4, {'straight': .2, 'stop': .4, 'slide': .4}),
    (300, 'mixed_8', 8, {'straight': .2, 'stop': .2, 'slide': .2, 'turn': .4}),
    (600, 'mixed_16', 16, {'straight': .15, 'stop': .15, 'slide': .2, 'turn': .5}),
    (1000, 'full_clip_29', 29, {'straight': .15, 'stop': .15, 'slide': .2, 'turn': .5}),
)
CASE_DEFINITIONS = {
    'straight': 'alive, commanded movement, abs(turn)<.1, displacement vector error<=.1*command norm',
    'stop': 'alive, commanded movement, actual displacement norm<.1*command norm',
    'slide': 'alive, both command axes>.1*command norm, one actual/command axis>=.7 and the other<=.2',
    'turn': 'alive, abs(turn control)>.2',
    'window': 'ANY predicted transition in [frame_start+context-1, frame_start+context-1+horizon); observed context is excluded',
}


def classify_player_motion(states, actions, dt=1/30):
    """Return transition masks from source labels, using post-turn game axes."""
    states, actions = np.asarray(states, dtype=np.float64), np.asarray(actions, dtype=np.float64)
    if (actions.ndim != 2 or actions.shape[1] != 4 or states.shape != (len(actions)+1, 90)
            or not len(actions) or not np.isfinite(states).all() or not np.isfinite(actions).all()
            or not math.isfinite(dt) or not 0 < dt <= .25):
        raise ValueError('Motion labels require finite states[T+1,90], actions[T,4] and valid dt')
    old, new = states[:-1], states[1:]
    alive = old[:, 4] > 0
    controls = np.clip(actions, [-1, -1, -1, 0], [1, 1, 1, 1])
    sine, cosine = old[:, 2].copy(), old[:, 3].copy()
    norm = np.hypot(sine, cosine)
    valid = norm > 1e-6
    sine = np.divide(sine, norm, out=np.zeros_like(sine), where=valid)
    cosine = np.divide(cosine, norm, out=np.ones_like(cosine), where=valid)
    angle = controls[:, 2]*2.2*dt
    new_sine = sine*np.cos(angle)+cosine*np.sin(angle)
    new_cosine = cosine*np.cos(angle)-sine*np.sin(angle)
    movement = controls[:, :2]/np.maximum(1., np.linalg.norm(controls[:, :2], axis=1))[:, None]
    forward, side = movement.T
    command = np.stack([new_cosine*forward-new_sine*side,
                        new_sine*forward+new_cosine*side], -1)*(2.4*dt)
    actual = (new[:, :2]-old[:, :2])*16.
    command_norm, actual_norm = np.linalg.norm(command, axis=1), np.linalg.norm(actual, axis=1)
    moving = alive & (command_norm > 1e-5)
    denominator = np.maximum(command_norm, 1e-12)
    axis_commanded = np.abs(command) > np.maximum(.1*command_norm[:, None], 1e-12)
    axis_ratio = np.abs(actual)/np.maximum(np.abs(command), 1e-12)
    slide = moving & axis_commanded.all(1) & (((axis_ratio[:, 0] >= .7) & (axis_ratio[:, 1] <= .2))
                                           | ((axis_ratio[:, 1] >= .7) & (axis_ratio[:, 0] <= .2)))
    return {'straight': moving & (np.abs(controls[:, 2]) < .1)
                & (np.linalg.norm(actual-command, axis=1) <= .1*denominator+1e-10),
        'stop': moving & (actual_norm < .1*command_norm), 'slide': slide,
        'turn': alive & (np.abs(controls[:, 2]) > .2), 'moving': moving,
        'alive': alive, 'idle': alive & ~moving}


class MotionCurriculum:
    """Deterministic fixed episode index; randomness belongs to the caller RNG.

    At least25% of every plan is uniform over complete windows. Missing cases
    redistribute their conditional mass over remaining cases; all-unavailable
    falls back to100% uniform. Source labels never enter returned model inputs.
    """
    def __init__(self, data, context, start_step, max_horizon=29,
                 combat_sampling=None, combat_fraction=.40):
        if (any(not isinstance(value, int) or isinstance(value, bool) for value in (context, start_step, max_horizon))
                or context < 1 or start_step < 0 or max_horizon < 1
                or getattr(data, 'split', None) != 'train'):
            raise ValueError('Motion curriculum requires train data, positive context/horizon and a nonnegative start step')
        self.context, self.start_step = context, start_step
        if (not isinstance(combat_fraction, (int, float)) or isinstance(combat_fraction, bool)
                or not math.isfinite(combat_fraction) or not 0 <= combat_fraction <= .75):
            raise ValueError('Combat mixture mass must be finite and between zero and .75')
        self.combat_sampling = combat_sampling if combat_fraction else None
        self.combat_fraction = float(combat_fraction)
        self.last_sample = None
        self._sampled_phases = {}
        self.requested_max_horizon = max_horizon
        self.records, self._windows = [], {}
        paths = [Path(episode['path']).resolve() for episode in data.episodes]
        if not paths or len(paths) != len(set(paths)):
            raise ValueError('Motion curriculum requires a nonempty fixed unique training episode list')
        directory = Path(data.directory).resolve() if hasattr(data, 'directory') else paths[0].parent
        if directory.name != 'train' or any(path.parent != directory or path.suffix != '.npz' for path in paths):
            raise ValueError('Motion curriculum episodes must stay inside one explicit train directory')
        if (self.combat_sampling is not None
                and (self.combat_sampling.context != context
                     or [record['path'] for record in self.combat_sampling.records] != paths)):
            raise ValueError('Combat and motion sampling must share context and identical ordered source episodes')
        digest = hashlib.sha256(CURRICULUM_VERSION.encode())
        self.counts, self.episode_counts = Counter(), Counter()
        for path in paths:
            with np.load(path, allow_pickle=False) as saved:
                if not {'states', 'actions', 'dt', 'split'}.issubset(saved.files) or str(saved['split']) != 'train':
                    raise ValueError(f'{path.name}: motion curriculum refuses non-training or incomplete source provenance')
                states, actions, dt = saved['states'], saved['actions'], float(saved['dt'])
            masks = classify_player_motion(states, actions, dt)
            for name, mask in masks.items():
                self.counts[name] += int(mask.sum())
                self.episode_counts[name] += int(mask.any())
            self.counts['transitions'] += len(actions)
            digest.update(str(path).encode())
            for values in (states, actions):
                digest.update(values.dtype.str.encode())
                digest.update(np.ascontiguousarray(values).tobytes())
            digest.update(float(dt).hex().encode())
            self.records.append({'path': path, 'frames': len(states), 'masks': masks})
        self.fingerprint = digest.hexdigest()
        self.max_horizon = min(max_horizon, max(record['frames']-context for record in self.records))
        if self.max_horizon < 1:
            raise ValueError('No source episode contains a complete context and future transition')

    def _index(self, horizon):
        if horizon in self._windows:
            return self._windows[horizon]
        groups = {name: [] for name in ('uniform', *MOTION_CASES)}
        for episode, record in enumerate(self.records):
            last_start = record['frames']-self.context-horizon
            if last_start < 0:
                continue
            starts = np.arange(last_start+1, dtype=np.int64)
            pairs = np.stack([np.full(len(starts), episode, dtype=np.int64), starts], -1)
            groups['uniform'].append(pairs)
            first_target = starts+self.context-1
            for name in MOTION_CASES:
                # Prefix counts inspect labels only inside the future target
                # window. A case later in a long prediction remains eligible;
                # an event confined to observed context never qualifies it.
                counts = np.r_[0, np.cumsum(record['masks'][name], dtype=np.int64)]
                contains = counts[first_target+horizon]-counts[first_target] > 0
                groups[name].append(pairs[contains])
        result = {name: np.concatenate(values, axis=0) if values else np.empty((0, 2), dtype=np.int64)
                  for name, values in groups.items()}
        if not len(result['uniform']):
            raise ValueError('No complete source windows for the requested curriculum horizon')
        self._windows[horizon] = result
        return result

    def plan(self, global_step):
        if (not isinstance(global_step, int) or isinstance(global_step, bool) or global_step < self.start_step):
            raise ValueError('Curriculum step must be an integer at or after its persisted migration start')
        offset = global_step-self.start_step
        _, phase, requested, conditional = next(value for value in reversed(PHASES) if offset >= value[0])
        horizon = min(requested, self.max_horizon)
        windows = self._index(horizon)
        available = {name: len(pairs) for name, pairs in windows.items()}
        remaining = {name: weight for name, weight in conditional.items() if available[name]}
        total = sum(remaining.values())
        weights = {'uniform': .25 if total else 1.}
        weights.update({name: .75*weight/total for name, weight in remaining.items()})
        combat = self.combat_sampling.plan(horizon) if self.combat_sampling is not None else None
        mixture = None
        if combat is not None and combat['enabled']:
            motion_mass = .75-self.combat_fraction if total else 0.
            combat_mass = self.combat_fraction if total else .75
            weights = {'uniform': .25}
            weights.update({name: motion_mass*weight/total for name, weight in remaining.items() if motion_mass})
            weights.update({'combat:'+name: combat_mass*weight
                            for name, weight in combat['case_weights'].items() if combat_mass})
            mixture = {'uniform': .25, 'motion': motion_mass, 'combat': combat_mass}
        result = {'version': CURRICULUM_VERSION, 'phase': phase, 'horizon': horizon,
            'context': self.context,
            'case_weights': weights, 'available_windows': available,
            'missing_cases': [name for name in conditional if not available[name]],
            'offset_updates': offset, 'start_step': self.start_step,
            'fingerprint': self.fingerprint, 'labels_used_only_for_sampling': True,
            'maximum_source_prediction_horizon': self.max_horizon}
        if combat is not None:
            result.update(combat_sampling=combat,
                mixture_weights=mixture or {'uniform': .25 if total else 1., 'motion': .75 if total else 0., 'combat': 0.},
                sampled_effectiveness=self._sampled_phases.get(phase, {}))
        return result

    def sample_indices(self, batch_size, step, rng):
        if (not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1
                or not isinstance(rng, np.random.Generator)):
            raise ValueError('Sampling requires a positive batch size and caller-owned NumPy Generator')
        plan = self.plan(step)
        groups = self._index(plan['horizon'])
        if self.combat_sampling is not None:
            groups = {**groups, **{'combat:'+name: values
                for name, values in self.combat_sampling._index(plan['horizon']).items()}}
        names = list(plan['case_weights'])
        selected = rng.choice(len(names), size=batch_size, p=[plan['case_weights'][name] for name in names])
        result = np.empty((batch_size, 2), dtype=np.int64)
        for index, name in enumerate(names):
            positions = np.flatnonzero(selected == index)
            if len(positions):
                choices = groups[name]
                result[positions] = choices[rng.integers(len(choices), size=len(positions))]
        if self.combat_sampling is not None:
            self.last_sample = {'phase': plan['phase'], 'horizon': plan['horizon'],
                'selected_case_counts': dict(Counter(names[int(index)] for index in selected)),
                'combat': self.combat_sampling.sample_report(result, plan['horizon'])}
            phase_counts = self._sampled_phases.setdefault(plan['phase'], {
                'sampled_windows': 0, 'selected_case_counts': {}, 'combat_windows_with_case': {},
                'combat_case_transitions_with_multiplicity': {}})
            phase_counts['sampled_windows'] += batch_size
            for target, values in (
                ('selected_case_counts', self.last_sample['selected_case_counts']),
                ('combat_windows_with_case', self.last_sample['combat']['windows_with_case']),
                ('combat_case_transitions_with_multiplicity', self.last_sample['combat']['case_transitions_with_multiplicity'])):
                phase_counts[target] = dict(Counter(phase_counts[target])+Counter(values))
        return result

    def summary(self):
        horizons = sorted({min(phase[2], self.max_horizon) for phase in PHASES})
        result = {'version': CURRICULUM_VERSION, 'fingerprint': self.fingerprint,
            'episodes': len(self.records), 'transitions': self.counts['transitions'],
            'counts': dict(self.counts), 'episodes_with_case': dict(self.episode_counts),
            'context': self.context, 'start_step': self.start_step,
            'max_horizon': self.max_horizon, 'requested_max_horizon': self.requested_max_horizon,
            'windows_by_horizon': {str(h): {name: len(values) for name, values in self._index(h).items()}
                                   for h in horizons},
            'case_definitions': CASE_DEFINITIONS, 'case_counts_are_nonexclusive': True,
            'labels_used_only_for_sampling': True, 'runtime_engine_feedback': False,
            'long_rollout_qualification_claimed': False}
        if self.combat_sampling is not None:
            result['combat_sampling'] = self.combat_sampling.summary()
            result['requested_mixture_weights'] = {'uniform': .25, 'motion': .75-self.combat_fraction,
                                                   'combat': self.combat_fraction}
        return result
