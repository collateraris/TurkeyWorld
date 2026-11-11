"""Opt-in distance sampling; Source labels never become renderer inputs."""
from collections import OrderedDict
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .renderer_enemy_data import EnemyFrames
from .distill_renderer import _digest


DISTANCE_WEIGHTS = {'near': .25, 'mid': .25, 'far': .25, 'replay': .25}


def distance_band(state, foreground_pixels):
    if not foreground_pixels or state[13] < .5 or state[12] <= 0:
        return None
    distance = float(np.linalg.norm((state[8:10]-state[:2])*16))
    return 'near' if distance < 2 else 'mid' if distance < 5 else 'far'


class DistanceBalancedFrames:
    """75% alive/visible distance quotas +25% the unchanged parent mixture.

    Parent and phase clocks are separate. A new phase explicitly adopts the
    completed parent's sampler; resuming this protocol restores both clocks.
    """
    def __init__(self, parent):
        self.parent = parent
        self.distance_buckets = {key: [] for key in ('near', 'mid', 'far')}
        self.row_bands = {}
        by_file = {}
        for i, row in enumerate(parent.rows):
            by_file.setdefault(row['file'], []).append((i, row))
        for file_index, rows in by_file.items():
            with np.load(parent.files[file_index]['path'], allow_pickle=False) as source:
                states = source['states']
            for index, row in rows:
                band = distance_band(states[row['frame']], row['mask_fraction'] > 0)
                self.row_bands[index] = band or ('corpse' if row['category']=='corpse' else 'empty')
                if band is not None:
                    self.distance_buckets[band].append(index)
        if any(not values for values in self.distance_buckets.values()):
            raise ValueError('Distance training requires visible live Source examples in all three bands')
        self.phase_draws = 0
        self.distance_counts = dict.fromkeys(DISTANCE_WEIGHTS, 0)
        self.parent_draws = parent.draws
        self.case_counts = dict(parent.case_counts)
        self.actual_distance_counts = dict.fromkeys(('near','mid','far','corpse','empty'),0)
        self.identity = {**copy.deepcopy(parent.identity), 'distance_sampling': {
            'protocol': 'visible_source_distance_v1', 'weights': DISTANCE_WEIGHTS,
            'near_max': 2., 'far_min': 5.,
            'band_frames': {key: len(value) for key, value in self.distance_buckets.items()}}}
        self.identity['fingerprint'] = hashlib.sha256(json.dumps(
            {k:v for k,v in self.identity.items() if k != 'fingerprint'}, sort_keys=True).encode()).hexdigest()

    def __getattr__(self, name):
        return getattr(self.parent, name)

    @property
    def draws(self):
        return self.parent_draws + self.phase_draws

    def adopt_parent_sampler(self, state):
        if self.phase_draws:
            raise ValueError('Cannot adopt a parent sampler after distance draws')
        self.parent.restore_sampler(state)
        self.parent_draws = self.parent.draws
        self.case_counts = dict(self.parent.case_counts)

    def batch(self, size, *, indices=None):
        if indices is not None:
            return self.parent.batch(size, indices=indices)
        selected = []
        for _ in range(size):
            band = max(DISTANCE_WEIGHTS, key=lambda key:
                       (self.phase_draws+1)*DISTANCE_WEIGHTS[key]-self.distance_counts[key])
            if band == 'replay':
                ids, self.parent.draws = self.parent._mixture_select(1, self.parent.rng,
                    self.parent.mix_counts, self.parent.cohort_counts, self.parent.draws)
                index = ids[0]
                self.parent.case_counts[self.rows[index]['category']] += 1
            else:
                index = int(self.parent.rng.choice(self.distance_buckets[band]))
            self.case_counts[self.rows[index]['category']] += 1
            self.actual_distance_counts[self.row_bands[index]] += 1
            selected.append(index)
            self.distance_counts[band] += 1
            self.phase_draws += 1
        self.parent.rng.shuffle(selected)
        return self.parent.batch(size, indices=selected)

    def sampler_state(self):
        return {'protocol': 'visible_source_distance_v1', 'parent_draws': self.parent_draws,
                'phase_draws': self.phase_draws, 'distance_counts': dict(self.distance_counts),
                'actual_distance_counts':dict(self.actual_distance_counts),
                'case_counts': dict(self.case_counts), 'parent': self.parent.sampler_state(),
                'fingerprint': self.identity['fingerprint']}

    def restore_sampler(self, state):
        if (state.get('protocol') != 'visible_source_distance_v1'
                or state.get('fingerprint') != self.identity['fingerprint']
                or set(state.get('distance_counts', {})) != set(DISTANCE_WEIGHTS)
                or sum(state['distance_counts'].values()) != state['phase_draws']
                or state['parent']['draws'] != state['parent_draws']+state['distance_counts']['replay']
                or set(state.get('case_counts', {})) != set(self.parent.case_counts)
                or sum(state['case_counts'].values()) != state['parent_draws']+state['phase_draws']
                or set(state.get('actual_distance_counts',{}))!=set(self.actual_distance_counts)
                or sum(state['actual_distance_counts'].values())!=state['phase_draws']):
            raise ValueError('Distance sampler protocol/clocks changed')
        self.parent.restore_sampler(state['parent'])
        self.case_counts = dict(state['case_counts'])
        self.actual_distance_counts = dict(state['actual_distance_counts'])
        self.parent_draws, self.phase_draws = state['parent_draws'], state['phase_draws']
        self.distance_counts = dict(state['distance_counts'])


class SourceDistancePanel:
    """Immutable independent static VAL images, with an explicit saved panel."""
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.records = json.loads(self.path.read_text(encoding='utf-8'))
        self.rows = [{'category': row['band']} for row in self.records]
        self.sources={path:{'sha256':_digest(path),'stat':(Path(path).stat().st_size,Path(path).stat().st_mtime_ns)}
            for path in sorted({row['path']for row in self.records})}
        self.identity = {'fingerprint': hashlib.sha256(json.dumps({'records':self.records,
            'sources':{path:value['sha256']for path,value in self.sources.items()}},sort_keys=True).encode()).hexdigest()}
        self.split = 'val'
        self._cache = OrderedDict()

    def panel_fingerprint(self, panel):
        return hashlib.sha256(json.dumps({'source':self.identity['fingerprint'],
            'records':[self.records[i]for i in panel]},sort_keys=True).encode()).hexdigest()

    def batch(self, size, *, indices):
        if size != len(indices):
            raise ValueError('Distance panel size mismatch')
        rows = []
        for index in indices:
            record = self.records[index]
            path = record['path']
            if (Path(path).stat().st_size,Path(path).stat().st_mtime_ns)!=self.sources[path]['stat']:
                raise ValueError('Immutable distance Source file changed')
            if path not in self._cache:
                with np.load(path, allow_pickle=False) as source:
                    if str(source['split']) != 'val' or 'actions' in source:
                        raise ValueError('Distance panel requires independent Source VAL snapshots')
                    self._cache[path] = {key: source[key] for key in
                        ('states','frames','map','object_masks','background_frames')}
                while len(self._cache) > 8:
                    self._cache.popitem(last=False)
            value = self._cache[path]; frame = record['frame']
            state = value['states'][frame]
            if distance_band(state, value['object_masks'][frame].sum()) != record['band']:
                raise ValueError('Source distance panel changed')
            if (np.asarray(state,dtype=np.float32)[[0,1,2,3,8,9]].tobytes().hex()!=record['scene_key']
                    or int(value['object_masks'][frame].sum())!=record['foreground_pixels']
                    or not np.array_equal(value['object_masks'][frame],np.any(
                        value['frames'][frame]!=value['background_frames'][frame],axis=-1))):
                raise ValueError('Distance Source pose/foreground identity changed')
            rows.append({'state': state, 'target': value['frames'][frame], 'map': value['map'],
                'object_mask': value['object_masks'][frame], 'background': value['background_frames'][frame],
                'previous_state': np.zeros(20,np.float32), 'previous_rgb': np.zeros((32,32,3),np.uint8),
                'action': np.zeros(4,np.float32), 'dynamic': False, 'row_index': index})
        result = {}
        for key in rows[0]:
            value = torch.from_numpy(np.stack([row[key] for row in rows]))
            if key in ('target','background','previous_rgb'):
                value = value.permute(0,3,1,2).float()/255
            elif key == 'object_mask':
                value = value[:,None].float()
            result[key] = value
        return result
