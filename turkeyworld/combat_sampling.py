"""Fixed Source combat labels used only to choose ordinary training windows.

No image array, model output, engine step or renderer is consulted here. Labels
may inspect recorded outcomes; returned values are only (episode, frame_start)
indices in the caller's existing ordered training split.
"""
from __future__ import annotations

from .engine import GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS, STATE_DIM

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from .geometry_targets import _wall_depths


COMBAT_SAMPLING_VERSION = 'combat_and_spawn_sampling_v2'
COMBAT_CASES = ('hit', 'miss', 'attack', 'death', 'wall_negative', 'spawn')
CASE_DEFINITIONS = {
    'hit': 'recorded hit_ids>=0',
    'miss': 'recorded shot event and hit_ids<0; not every miss is a wall negative',
    'attack': 'any recorded NPC attack_flag',
    'death': 'any persistent NPC slot changes alive to dead',
    'spawn': 'any persistent NPC slot changes dead to alive in recorded Source states',
    'wall_negative': 'recorded shot miss; pre-NPC-motion live enemy intersects the firing ray; a static map wall is hit before its circle entry',
    'window': 'ANY case inside [start+context-1,start+context-1+horizon); observed context excluded',
}


def _positive_integer(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def _validated_labels(states, actions, events, hit_ids, attack_flags, tilemap, dt):
    arrays = [np.asarray(value) for value in (states, actions, events, hit_ids, attack_flags, tilemap)]
    states, actions, events, hit_ids, attack_flags, tilemap = arrays
    count = len(actions) if actions.ndim == 2 else 0
    if (not count or actions.shape != (count, 4) or states.shape != (count+1, STATE_DIM)
            or events.shape != (count, 4) or hit_ids.shape != (count,)
            or attack_flags.shape != (count, ENTITY_SLOTS) or tilemap.shape != (16, 16)
            or any(not np.issubdtype(value.dtype, np.number) and value.dtype != bool
                   or np.issubdtype(value.dtype, np.complexfloating) for value in arrays)
            or any(not np.isfinite(value).all() for value in arrays)
            or not np.issubdtype(hit_ids.dtype, np.integer)
            or ((hit_ids < -1) | (hit_ids >= ENTITY_SLOTS)).any()
            or ((events < 0) | (events > 1)).any()
            or ((attack_flags < 0) | (attack_flags > 1)).any()
            or not math.isfinite(dt) or not 0 < dt <= .25):
        raise ValueError('Combat labels require finite aligned states/actions/events/hit_ids/attack_flags/map and valid dt')
    return states, actions, events, hit_ids, attack_flags, tilemap


def _wall_blocked_shots(states, misses, tilemap):
    """Conservative offline Source firing-ray geometry, not a runtime oracle.

    Source moves/turns the player before shooting and moves NPCs afterwards.
    Consequently the recorded next player pose and previous NPC positions are
    the shot-time geometry. Same-step respawns are excluded conservatively.
    The .25 tile enemy radius matches the fixed Source geometry, without an
    import of Game or any engine call.
    """
    result = np.zeros(len(misses), dtype=bool)
    height, width = tilemap.shape
    for index in np.flatnonzero(misses):
        old, new = states[index], states[index+1]
        if old[4] <= 0:
            continue
        x, y = new[:2]*np.asarray([width, height])
        sine, cosine = new[2:4]
        length = math.hypot(sine, cosine)
        if (length < 1e-8 or not 0 <= x < width or not 0 <= y < height
                or tilemap[math.floor(y), math.floor(x)] != 0):
            continue
        sine, cosine = sine/length, cosine/length
        entities = old[GLOBAL_DIM:-2].reshape(ENTITY_SLOTS, ENTITY_DIM)
        positions = entities[:, :2]*np.asarray([width, height])
        delta = positions-np.asarray([x, y])
        along = delta[:, 0]*cosine+delta[:, 1]*sine
        perpendicular = np.abs(delta[:, 0]*sine-delta[:, 1]*cosine)
        inside = ((positions >= 0) & (positions < [width, height])).all(1)
        free = np.zeros(ENTITY_SLOTS, dtype=bool)
        free[inside] = tilemap[positions[inside, 1].astype(int), positions[inside, 0].astype(int)] == 0
        candidate = ((entities[:, 5] > .5) & (entities[:, 4] > 0) & free
                     & (along > 0) & (perpendicular < .25))
        if not candidate.any():
            continue
        entry = along[candidate]-np.sqrt(np.maximum(0., .25**2-perpendicular[candidate]**2))
        wall = float(_wall_depths(tilemap, x, y, np.asarray([cosine]), np.asarray([sine]), 64.)[0])
        # An unclosed map exit is not evidence of an actual static wall tile.
        wall_x, wall_y = x+(wall+1e-7)*cosine, y+(wall+1e-7)*sine
        static_wall = (0 <= wall_x < width and 0 <= wall_y < height
                       and tilemap[math.floor(wall_y), math.floor(wall_x)] != 0)
        result[index] = static_wall and bool(((entry >= 0) & (wall < entry-1e-6)).any())
    return result


def classify_combat(states, actions, events, hit_ids, attack_flags, tilemap, dt=1/30):
    """Return nonexclusive transition masks, retaining all eight Source slots."""
    states, actions, events, hit_ids, attack_flags, tilemap = _validated_labels(
        states, actions, events, hit_ids, attack_flags, tilemap, dt)
    shot, hit = events[:, 0] > .5, hit_ids >= 0
    miss = shot & ~hit
    alive = states[:, GLOBAL_DIM:-2].reshape(-1, ENTITY_SLOTS, ENTITY_DIM)[:, :, 5] > .5
    return {'hit': hit, 'miss': miss, 'attack': (attack_flags > .5).any(1),
            'death': (alive[:-1] & ~alive[1:]).any(1),
            'spawn': (~alive[:-1] & alive[1:]).any(1),
            'wall_negative': _wall_blocked_shots(states, miss, tilemap), 'shot': shot}


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, separators=(',', ':'), allow_nan=False)
        for attempt in range(41):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 40:
                    raise
                time.sleep(.05)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class CombatSampling:
    """Balanced conditional sampling over a fixed, training-only source list.

    Each nonempty combat case has equal probability. Missing cases redistribute
    over available cases; an entirely empty combat corpus falls back to uniform.
    A caller mixing with motion must reserve its own overall uniform component.
    """
    def __init__(self, data, context, cache_path=None):
        _positive_integer(context, 'context')
        if getattr(data, 'split', None) != 'train':
            raise ValueError('Combat sampling requires the explicit training split')
        self.context, self.records, self._windows = context, [], {}
        paths = [Path(episode['path']).resolve() for episode in data.episodes]
        directory = Path(data.directory).resolve() if hasattr(data, 'directory') else paths[0].parent if paths else None
        if (not paths or len(paths) != len(set(paths)) or directory.name != 'train'
                or any(path.parent != directory or path.suffix != '.npz' for path in paths)):
            raise ValueError('Combat sampling requires a nonempty unique episode list inside one train directory')
        cache = Path(cache_path).resolve() if cache_path is not None else None
        if cache is not None and (cache.suffix != '.json' or cache.parent == directory):
            raise ValueError('Combat counter cache must be a JSON sidecar outside the source train directory')
        cached = None
        if cache is not None and cache.is_file():
            try:
                cached = json.loads(cache.read_text(encoding='utf-8'))
            except (ValueError, OSError):
                pass
        digest = hashlib.sha256(COMBAT_SAMPLING_VERSION.encode()+str(context).encode())
        self.source_ids, pending = [], []
        required = {'states', 'actions', 'events', 'hit_ids', 'attack_flags', 'map', 'dt', 'split'}
        for path in paths:
            with np.load(path, allow_pickle=False) as saved:
                if not required.issubset(saved.files) or str(saved['split']) != 'train':
                    raise ValueError(f'{path.name}: missing combat labels or non-training provenance')
                values = {name: saved[name] for name in required-{'split', 'dt'}}
                dt = float(saved['dt'])
                _validated_labels(values['states'], values['actions'], values['events'], values['hit_ids'],
                                  values['attack_flags'], values['map'], dt)
                identity = {'path': str(path), 'filename': path.name, 'split': 'train', 'dt': dt,
                    'transitions': len(values['actions'])}
                for field in ('seed', 'policy', 'behavior_version'):
                    if field in saved.files:
                        identity[field] = saved[field].item()
            source_digest = hashlib.sha256()
            for name in sorted(values):
                value = np.ascontiguousarray(values[name])
                source_digest.update(name.encode()+value.dtype.str.encode()+str(value.shape).encode()+value.tobytes())
            identity['labels_sha256'] = source_digest.hexdigest()
            self.source_ids.append(identity)
            digest.update(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode())
            pending.append((path, values, dt))
        self.fingerprint = digest.hexdigest()
        self.cache_reused = False
        if isinstance(cached, dict) and cached.get('fingerprint') == self.fingerprint:
            try:
                encoded = cached['records']
                if cached['version'] != COMBAT_SAMPLING_VERSION or len(encoded) != len(pending):
                    raise ValueError('Wrong cache shape')
                for (path, values, _), record in zip(pending, encoded):
                    count = len(values['actions'])
                    masks = {}
                    for name in (*COMBAT_CASES, 'shot'):
                        packed = np.asarray(record[name], dtype=np.uint8)
                        if packed.shape != ((count+7)//8,):
                            raise ValueError('Wrong cached mask shape')
                        masks[name] = np.unpackbits(packed, count=count).astype(bool)
                    self.records.append({'path': path, 'frames': count+1, 'masks': masks})
                self.cache_reused = True
            except (KeyError, TypeError, ValueError):
                self.records.clear()
        if not self.cache_reused:
            for path, values, dt in pending:
                masks = classify_combat(values['states'], values['actions'], values['events'], values['hit_ids'],
                                        values['attack_flags'], values['map'], dt)
                self.records.append({'path': path, 'frames': len(values['states']), 'masks': masks})
            if cache is not None:
                _atomic_json(cache, {'version': COMBAT_SAMPLING_VERSION, 'fingerprint': self.fingerprint,
                    'context': context, 'source_ids': self.source_ids,
                    'records': [{name: np.packbits(mask).tolist() for name, mask in record['masks'].items()}
                                for record in self.records]})
        self.max_horizon = max(record['frames']-context for record in self.records)
        if self.max_horizon < 1:
            raise ValueError('No complete combat source context and future window')
        self.counts, self.episode_counts = Counter(), Counter()
        for record in self.records:
            self.counts['transitions'] += record['frames']-1
            for name, mask in record['masks'].items():
                self.counts[name] += int(mask.sum())
                self.episode_counts[name] += int(mask.any())
        self.last_sample = None

    def _index(self, horizon):
        _positive_integer(horizon, 'horizon')
        if horizon in self._windows:
            return self._windows[horizon]
        groups = {name: [] for name in ('uniform', *COMBAT_CASES)}
        for episode, record in enumerate(self.records):
            last_start = record['frames']-self.context-horizon
            if last_start < 0:
                continue
            starts = np.arange(last_start+1, dtype=np.int64)
            pairs = np.stack([np.full(len(starts), episode, dtype=np.int64), starts], -1)
            groups['uniform'].append(pairs)
            first = starts+self.context-1
            for name in COMBAT_CASES:
                counts = np.r_[0, np.cumsum(record['masks'][name], dtype=np.int64)]
                groups[name].append(pairs[counts[first+horizon]-counts[first] > 0])
        result = {name: np.concatenate(values) if values else np.empty((0, 2), dtype=np.int64)
                  for name, values in groups.items()}
        if not len(result['uniform']):
            raise ValueError('No complete source windows for the requested combat horizon')
        self._windows[horizon] = result
        return result

    def plan(self, horizon):
        groups = self._index(horizon)
        active = [name for name in COMBAT_CASES if len(groups[name])]
        return {'version': COMBAT_SAMPLING_VERSION, 'fingerprint': self.fingerprint,
            'context': self.context, 'horizon': horizon, 'enabled': bool(active),
            'case_weights': {name: 1/len(active) for name in active} if active else {'uniform': 1.},
            'available_windows': {name: len(values) for name, values in groups.items()},
            'missing_cases': [name for name in COMBAT_CASES if name not in active],
            'labels_used_only_for_sampling': True, 'runtime_engine_feedback': False}

    def sample_indices(self, batch_size, horizon, rng):
        _positive_integer(batch_size, 'batch_size')
        if not isinstance(rng, np.random.Generator):
            raise ValueError('Combat sampling requires a caller-owned NumPy Generator')
        plan, groups = self.plan(horizon), self._index(horizon)
        names = list(plan['case_weights'])
        selected = rng.choice(len(names), size=batch_size, p=list(plan['case_weights'].values()))
        result = np.empty((batch_size, 2), dtype=np.int64)
        for index, name in enumerate(names):
            positions = np.flatnonzero(selected == index)
            if len(positions):
                values = groups[name]
                result[positions] = values[rng.integers(len(values), size=len(positions))]
        self.last_sample = {**self.sample_report(result, horizon),
                            'selected_case_counts': dict(Counter(names[int(i)] for i in selected))}
        return result

    def sample_report(self, indices, horizon):
        self._index(horizon)
        pairs = np.asarray(indices)
        if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.issubdtype(pairs.dtype, np.integer):
            raise ValueError('Combat sample report requires integer episode/start pairs')
        windows, transitions = Counter(), Counter()
        for episode, start in pairs:
            if (not 0 <= episode < len(self.records) or start < 0
                    or start+self.context+horizon > self.records[int(episode)]['frames']):
                raise ValueError('Combat sample window crosses a source episode boundary')
            record = self.records[int(episode)]
            first = int(start)+self.context-1
            for name in COMBAT_CASES:
                count = int(record['masks'][name][first:first+horizon].sum())
                windows[name] += int(count > 0)
                transitions[name] += count
        return {'sampled_windows': len(pairs), 'horizon': horizon,
            'windows_with_case': dict(windows), 'case_transitions_with_multiplicity': dict(transitions),
            'case_counts_are_nonexclusive': True, 'labels_used_only_for_sampling': True}

    def summary(self):
        return {'version': COMBAT_SAMPLING_VERSION, 'fingerprint': self.fingerprint,
            'context': self.context, 'episodes': len(self.records), 'max_horizon': self.max_horizon,
            'transitions': self.counts['transitions'], 'counts': dict(self.counts),
            'episodes_with_case': dict(self.episode_counts), 'source_ids': self.source_ids,
            'case_definitions': CASE_DEFINITIONS, 'case_counts_are_nonexclusive': True,
            'labels_used_only_for_sampling': True, 'runtime_engine_feedback': False,
            'source_corpus_expanded': False}
