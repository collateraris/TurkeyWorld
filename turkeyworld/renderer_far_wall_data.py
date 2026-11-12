"""Independent bounded TRAIN Source pool for far-wall auxiliary updates.

The main renderer sampler's RNG, counters and mixture remain untouched. Pool
membership uses only exact Source map/pose/occluder labels, never model errors.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json

import numpy as np
import torch

from .renderer_wall_depth import source_wall_depth_labels


FAR_WALL_DATA_PROTOCOL = 'source_train_far_wall_aux_pool_v1'
STRATA = ('depth_ge_12', 'thin_face_ge_5', 'other_depth_ge_8')
AREA_KEYS = ('tiny_1_16', 'small_17_64', 'large_65_plus', 'corpse', 'empty')
SUPPORT_KEYS = ('depth_ge_8', 'depth_ge_12', 'thin_face_ge_5', 'thin_height_le_2')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _counter(values, keys):
    return {key: int(values.get(key, 0)) for key in keys}


class FarWallFrames:
    """Fresh private auxiliary sampler over a fixed bounded TRAIN candidate set.

    The strata are disjoint, in priority order: exposed walls at depth >=12,
    remaining one/two-column far faces, then remaining exposed depth >=8 walls.
    Equal stratum quotas retain the weakest native-pixel signal. Each explicit
    parent read is checked against the parent's complete sampler state.
    """
    protocol = FAR_WALL_DATA_PROTOCOL
    split = 'train'

    def __init__(self, parent, *, candidate_count=2048, seed=2718, label_batch_size=32):
        if getattr(parent, 'split', None) != 'train':
            raise ValueError('Far-wall auxiliary pool requires the TRAIN split')
        if (type(candidate_count) is not int or not 3 <= candidate_count <= 4096
                or type(label_batch_size) is not int or not 1 <= label_batch_size <= 64
                or type(seed) is not int or seed < 0 or not len(parent.rows)):
            raise ValueError('Far-wall pool requires positive bounded counts and an integer seed')
        source_identity = getattr(parent, 'identity', {}).get('fingerprint')
        if not isinstance(source_identity, str) or not source_identity:
            raise ValueError('Far-wall pool requires an immutable parent Source identity')
        self.parent = parent
        self.seed = seed
        self.rng = np.random.default_rng(seed + 1)
        pool_rng = np.random.default_rng(seed)
        count = min(candidate_count, len(parent.rows))
        self.candidate_indices = sorted(int(v) for v in pool_rng.choice(len(parent.rows), count, replace=False))
        self.pool_buckets = {key: [] for key in STRATA}
        self.facts = {}
        candidate_support = Counter()
        candidate_area = Counter()
        directions = Counter()
        candidate_wall_pixels = Counter()
        for first in range(0, count, label_batch_size):
            ids = self.candidate_indices[first:first + label_batch_size]
            batch = self._explicit_batch(ids)
            inputs = [batch[key] for key in ('map', 'state', 'object_mask')]
            if any(not torch.is_tensor(value) or value.device.type != 'cpu' for value in inputs):
                raise ValueError('Far-wall candidate geometry labels must be built on CPU only')
            labels = source_wall_depth_labels(*inputs)
            aggregates = labels['aggregates']
            area = (batch['object_mask'][:, 0] > .5).flatten(1).sum(-1)
            states = batch['state']
            for offset, index in enumerate(ids):
                d8 = int(aggregates['depth_ge_8']['wall'][offset].sum())
                d12 = int(aggregates['depth_ge_12']['wall'][offset].sum())
                thin = int(aggregates['depth_ge_5']['thin_face'][offset].sum())
                thin_height = int(labels['thin_height'][offset].sum())
                native_area = int(area[offset])
                alive = bool(states[offset, 13] >= .5 and states[offset, 12] > 0)
                area_key = ('empty' if native_area == 0 else 'corpse' if not alive
                            else 'tiny_1_16' if native_area <= 16
                            else 'small_17_64' if native_area <= 64 else 'large_65_plus')
                support = {'depth_ge_8': d8 > 0, 'depth_ge_12': d12 > 0,
                           'thin_face_ge_5': thin > 0, 'thin_height_le_2': thin_height > 0}
                candidate_support.update(key for key, present in support.items() if present)
                candidate_wall_pixels.update({'depth_ge_8': d8, 'depth_ge_12': d12,
                                               'thin_face_ge_5': thin, 'thin_height_le_2': thin_height})
                candidate_area[area_key] += 1
                angle = float(torch.atan2(states[offset, 2], states[offset, 3]))
                directions[str(int((angle % (2 * np.pi)) // (np.pi / 2)))] += 1
                stratum = 'depth_ge_12' if d12 else 'thin_face_ge_5' if thin else 'other_depth_ge_8' if d8 else None
                if stratum is None:
                    continue
                self.pool_buckets[stratum].append(index)
                row = parent.rows[index]
                self.facts[index] = {'stratum': stratum, 'area': native_area, 'area_key': area_key,
                    'source_category': str(row['category']), 'support': support,
                    'depth_ge_8_wall_pixels': d8, 'depth_ge_12_wall_pixels': d12,
                    'thin_face_ge_5_pixels': thin, 'thin_height_le_2_pixels': thin_height}
        missing = [key for key, ids in self.pool_buckets.items() if not ids]
        if missing:
            raise ValueError('Bounded TRAIN far-wall candidate pool misses ' + ', '.join(missing)
                             + '; enlarge the fixed TRAIN candidate count explicitly')
        def row_identity(index):
            row = parent.rows[index]
            info = parent.files[int(row['file'])]
            return {'index': int(index), 'file': str(info['path']), 'frame': int(row['frame']),
                    'source_file_sha256': info.get('sha256'), 'facts': self.facts[index]}
        self.identity = {
            'protocol': self.protocol, 'split': 'train', 'parent_source_fingerprint': source_identity,
            'candidate_limit': candidate_count, 'candidate_count': count, 'seed': seed,
            'sampler_seed': seed + 1, 'label_batch_size': label_batch_size,
            'candidate_indices_sha256': _digest(self.candidate_indices),
            'candidate_indices': list(self.candidate_indices),
            'pool_counts': {key: len(ids) for key, ids in self.pool_buckets.items()},
            'pool_hashes': {key: _digest([row_identity(i) for i in ids]) for key, ids in self.pool_buckets.items()},
            'pool_indices': copy.deepcopy(self.pool_buckets),
            'candidate_support_frames': _counter(candidate_support, SUPPORT_KEYS),
            'candidate_support_pixels': _counter(candidate_wall_pixels, SUPPORT_KEYS),
            'candidate_native_area_counts': _counter(candidate_area, AREA_KEYS),
            'candidate_camera_quadrants': _counter(directions, ('0', '1', '2', '3')),
            'pool_native_area_counts': {key: _counter(Counter(self.facts[i]['area_key'] for i in ids), AREA_KEYS)
                                        for key, ids in self.pool_buckets.items()},
            'stratum_priority': list(STRATA), 'stratum_weights': {key: 1 / 3 for key in STRATA},
            'source_geometry_and_masks_are_labels_only': True,
            'model_error_or_generated_image_mining': False, 'validation_used': False,
            'gpu_used_for_pool': False, 'parent_sampler_unchanged': True,
        }
        self.identity['fingerprint'] = _digest(self.identity)
        self.draws = 0
        self.stratum_counts = dict.fromkeys(STRATA, 0)
        self.actual_area_counts = dict.fromkeys(AREA_KEYS, 0)
        self.source_category_counts = dict.fromkeys(sorted({v['source_category'] for v in self.facts.values()}), 0)
        self.actual_support_counts = dict.fromkeys(SUPPORT_KEYS, 0)

    def _explicit_batch(self, indices):
        before = copy.deepcopy(self.parent.sampler_state())
        batch = self.parent.batch(len(indices), indices=indices)
        if self.parent.sampler_state() != before:
            raise ValueError('Explicit far-wall Source read changed the main parent sampler')
        return batch

    def batch(self, size=4, *, indices=None):
        if type(size) is not int or size < 1:
            raise ValueError('Far-wall auxiliary batch size must be a positive integer')
        if indices is not None:
            if (len(indices) != size or any(isinstance(i, (bool, np.bool_)) or int(i) != i
                                           or int(i) not in self.facts for i in indices)):
                raise ValueError('Explicit far-wall reads require retained TRAIN pool indices')
            return self._explicit_batch([int(i) for i in indices])
        before = self.sampler_state()
        selected = []
        for _ in range(size):
            stratum = min(STRATA, key=lambda key: self.stratum_counts[key])
            index = int(self.rng.choice(self.pool_buckets[stratum]))
            selected.append(index)
            fact = self.facts[index]
            self.stratum_counts[stratum] += 1
            self.actual_area_counts[fact['area_key']] += 1
            self.source_category_counts[fact['source_category']] += 1
            for key, present in fact['support'].items():
                self.actual_support_counts[key] += int(present)
            self.draws += 1
        self.rng.shuffle(selected)
        try:
            return self._explicit_batch(selected)
        except Exception:
            self.restore_sampler(before)
            raise

    def sampler_state(self):
        return {'protocol': self.protocol, 'fingerprint': self.identity['fingerprint'],
                'draws': self.draws, 'rng': copy.deepcopy(self.rng.bit_generator.state),
                'stratum_counts': dict(self.stratum_counts), 'actual_area_counts': dict(self.actual_area_counts),
                'source_category_counts': dict(self.source_category_counts),
                'actual_support_counts': dict(self.actual_support_counts)}

    def restore_sampler(self, state):
        counters = ('stratum_counts', 'actual_area_counts', 'source_category_counts')
        if (state.get('protocol') != self.protocol or state.get('fingerprint') != self.identity['fingerprint']
                or type(state.get('draws')) is not int or state['draws'] < 0
                or any(set(state.get(key, {})) != set(getattr(self, key))
                       or any(type(v) is not int or v < 0 for v in state[key].values())
                       or sum(state[key].values()) != state['draws'] for key in counters)
                or any(state['stratum_counts'][key] != state['draws'] // len(STRATA)
                       + int(i < state['draws'] % len(STRATA)) for i, key in enumerate(STRATA))
                or set(state.get('actual_support_counts', {})) != set(SUPPORT_KEYS)
                or any(type(v) is not int or not 0 <= v <= state['draws'] for v in state['actual_support_counts'].values())):
            raise ValueError('Far-wall auxiliary pool identity or successful-draw clocks changed')
        private = np.random.default_rng()
        try:
            private.bit_generator.state = copy.deepcopy(state['rng'])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('Far-wall private RNG state changed') from error
        self.rng.bit_generator.state = copy.deepcopy(private.bit_generator.state)
        self.draws = state['draws']
        for key in (*counters, 'actual_support_counts'):
            setattr(self, key, dict(state[key]))
