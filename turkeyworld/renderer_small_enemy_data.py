"""Area/distance quotas over genuine immutable TRAIN Source frames only."""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np

from .renderer_distance_data import DistanceBalancedFrames
from .renderer_small_enemy_protocol import SMALL_ENEMY_PROTOCOL, small_enemy_protocol


class SmallEnemyBalancedFrames(DistanceBalancedFrames):
    """Explicit new phase; original replay selection and its RNG stay intact."""
    protocol = SMALL_ENEMY_PROTOCOL['name']

    def __init__(self, parent):
        if parent.split != 'train':
            raise ValueError('Native-area training quotas require the TRAIN split')
        super().__init__(parent)
        self.quota_buckets = {'tiny_1_16': [], 'small_17_64': [],
                              'far': list(self.distance_buckets['far'])}
        self.row_areas = []
        for index, row in enumerate(parent.rows):
            pixels = float(row['mask_fraction'])*1024
            area = int(round(pixels))
            if not np.isfinite(pixels) or abs(pixels-area) > 1e-5 or not 0 <= area <= 1024:
                raise ValueError('Area quotas require exact native32 Source mask counts')
            self.row_areas.append(area)
            if self.row_bands[index] in ('near','mid','far'):
                if 1 <= area <= 16: self.quota_buckets['tiny_1_16'].append(index)
                elif 17 <= area <= 64: self.quota_buckets['small_17_64'].append(index)
        if any(not ids for ids in self.quota_buckets.values()):
            raise ValueError('Native-area training requires live visible tiny, small and far Source pools')
        self.distance_counts = dict.fromkeys(SMALL_ENEMY_PROTOCOL['source_quotas'], 0)
        self.actual_area_counts = dict.fromkeys(('tiny_1_16','small_17_64','large_65_plus','corpse','empty'),0)
        self.parent_sampler_draws_at_start = parent.draws
        self.identity['distance_sampling'] = {**small_enemy_protocol(),
            'quota_frames': {key:len(ids) for key,ids in self.quota_buckets.items()},
            'parent_replay': 'Unchanged original combat/appearance/navigation mixture',
            'hard_definition': 'Exact native visible area or Source distance; no validation/model-error mining'}
        self.identity['fingerprint'] = hashlib.sha256(json.dumps(
            {k:v for k,v in self.identity.items() if k != 'fingerprint'},sort_keys=True).encode()).hexdigest()

    def adopt_parent_sampler(self, state):
        raise ValueError('Native-area v5 begins an explicit fresh optimizer/sampler phase')

    def batch(self, size, *, indices=None):
        if indices is not None:
            return self.parent.batch(size, indices=indices)
        if type(size) is not int or size < 1:
            raise ValueError('Native-area batch size must be a positive integer')
        selected = []
        weights = SMALL_ENEMY_PROTOCOL['source_quotas']
        for _ in range(size):
            quota = max(weights,key=lambda key:(self.phase_draws+1)*weights[key]-self.distance_counts[key])
            if quota == 'replay':
                ids, self.parent.draws = self.parent._mixture_select(1,self.parent.rng,
                    self.parent.mix_counts,self.parent.cohort_counts,self.parent.draws)
                index = int(ids[0])
                self.parent.case_counts[self.rows[index]['category']] += 1
            else:
                index = int(self.parent.rng.choice(self.quota_buckets[quota]))
            self.case_counts[self.rows[index]['category']] += 1
            self.actual_distance_counts[self.row_bands[index]] += 1
            area = self.row_areas[index]
            area_key = ('corpse' if self.row_bands[index] == 'corpse' else 'empty' if area == 0 else 'tiny_1_16' if area <= 16
                        else 'small_17_64' if area <= 64 else 'large_65_plus')
            self.actual_area_counts[area_key] += 1
            selected.append(index); self.distance_counts[quota] += 1; self.phase_draws += 1
        self.parent.rng.shuffle(selected)
        return self.parent.batch(size,indices=selected)

    def sampler_state(self):
        return {'protocol':self.protocol,'fingerprint':self.identity['fingerprint'],
            'parent_draws':self.parent_draws,'phase_draws':self.phase_draws,
            'parent_sampler_draws_at_start':self.parent_sampler_draws_at_start,
            'distance_counts':dict(self.distance_counts),'actual_distance_counts':dict(self.actual_distance_counts),
            'actual_area_counts':dict(self.actual_area_counts),'case_counts':dict(self.case_counts),
            'parent':self.parent.sampler_state()}

    def restore_sampler(self, state):
        counters = {'distance_counts':self.distance_counts,'actual_distance_counts':self.actual_distance_counts,
                    'actual_area_counts':self.actual_area_counts}
        if (state.get('protocol') != self.protocol or state.get('fingerprint') != self.identity['fingerprint']
                or type(state.get('phase_draws')) is not int or state['phase_draws'] < 0
                or type(state.get('parent_draws')) is not int or state['parent_draws'] < 0
                or type(state.get('parent_sampler_draws_at_start')) is not int
                or state['parent_sampler_draws_at_start'] > state['parent_draws']
                or any(set(state.get(key,{})) != set(expected)
                       or any(type(v) is not int or v < 0 for v in state[key].values())
                       or sum(state[key].values()) != state['phase_draws'] for key,expected in counters.items())
                or state['parent']['draws'] != state['parent_sampler_draws_at_start']+state['distance_counts']['replay']
                or set(state.get('case_counts',{})) != set(self.case_counts)
                or sum(state['case_counts'].values()) != state['parent_draws']+state['phase_draws']):
            raise ValueError('Native-area sampler identity or successful-draw clocks changed')
        self.parent.restore_sampler(copy.deepcopy(state['parent']))
        for key in counters: setattr(self,key,dict(state[key]))
        self.case_counts = dict(state['case_counts'])
        self.parent_draws, self.phase_draws = state['parent_draws'], state['phase_draws']
        self.parent_sampler_draws_at_start = state['parent_sampler_draws_at_start']
