"""Source-only combat/appearance/detailed-navigation frame mixture.

Labels select examples, never model inputs. Dynamic rendering keeps the
existing strictly previous-image/state/action appearance path.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .renderer_enemy_data import EnemyFrames


class CombatRendererFrames(EnemyFrames):
    MIX = {'combat': .5, 'appearance': .25, 'navigation': .25}
    COMBAT_CASES = ('hit', 'miss', 'player_hurt', 'death', 'respawn')
    APPEARANCE_CASES = ('large', 'partial', 'distant', 'corpse', 'medium', 'empty_hidden')

    def __init__(self, data_root, appearance_root, split, *, seed=173, exclude_scene_keys=None):
        root, appearance = Path(data_root), Path(appearance_root)
        sources = [
            {'directory': appearance/'static_enemy/snapshots'/split, 'kind': 'static', 'dynamic': False},
            {'directory': root/'moving1/rgb'/split, 'kind': 'moving', 'dynamic': True},
            {'directory': root/'full_game/rgb'/split, 'kind': 'combat_full', 'dynamic': True},
            {'directory': root/'navigation/rgb'/split, 'kind': 'empty_navigation', 'dynamic': True}]
        super().__init__(data_root, appearance_root, split, seed=seed,
                         exclude_scene_keys=exclude_scene_keys, source_directories=sources)
        self.mix_buckets = {'combat': {case: [] for case in self.COMBAT_CASES},
                            'appearance': {case: [] for case in self.APPEARANCE_CASES},
                            'navigation': {}}
        # Navigation policy/case labels are provenance only; all are genuine
        # Source controls and exact Source pixels. Unknown labels stay explicit.
        for index, row in enumerate(self.rows):
            if row['source_kind'] == 'empty_navigation':
                info = self.files[row['file']]
                if 'navigation_case' not in info:
                    with np.load(info['path'], allow_pickle=False) as episode:
                        key = next((key for key in ('nav_case', 'case', 'policy') if key in episode), None)
                        value = episode[key] if key else None
                        info['navigation_case'] = str(value.item()) if value is not None and value.ndim == 0 else 'unlabelled_source_navigation'
                row['navigation_case'] = info['navigation_case']
                self.mix_buckets['navigation'].setdefault(row['navigation_case'], []).append(index)
            else:
                self.mix_buckets['appearance'][row['category']].append(index)
                for case in row['combat_cases']:
                    self.mix_buckets['combat'][case].append(index)
        if any(not any(groups.values()) for groups in self.mix_buckets.values()):
            raise ValueError('Combat renderer needs nonempty real combat, appearance and navigation pools')
        self.mix_counts = dict.fromkeys(self.MIX, 0)
        self.cohort_counts = {pool: dict.fromkeys(cases, 0) for pool, cases in self.mix_buckets.items()}
        self.identity.update(protocol='combat_appearance_detailed_navigation_v1',
            mix_weights=self.MIX, cohort_frames={pool: {case: len(ids) for case, ids in cases.items()}
                for pool, cases in self.mix_buckets.items()},
            missing_combat_cohorts=[case for case, ids in self.mix_buckets['combat'].items() if not ids],
            combat_definition='Recorded Source events and adjacent HP/alive deltas; no inferred wall or model labels')
        self.identity['fingerprint'] = hashlib.sha256(json.dumps(
            {key: value for key, value in self.identity.items() if key != 'fingerprint'}, sort_keys=True).encode()).hexdigest()

    def _mixture_select(self, size, rng, pools, cases, draws, *, unique=False):
        indices, used = [], set()
        for _ in range(size):
            available = {pool: {case: ([i for i in ids if i not in used] if unique else ids)
                               for case, ids in groups.items()} for pool, groups in self.mix_buckets.items()}
            candidates = [pool for pool in self.MIX if any(available[pool].values())]
            if not candidates:
                raise ValueError('Insufficient distinct Source frames for a fixed renderer panel')
            pool = max(candidates, key=lambda p: (draws+1)*self.MIX[p]-pools[p])
            active = [case for case, ids in available[pool].items() if ids]
            case = min(active, key=lambda c: cases[pool][c])
            index = int(rng.choice(available[pool][case]))
            indices.append(index); used.add(index)
            pools[pool] += 1; cases[pool][case] += 1; draws += 1
        rng.shuffle(indices)
        return indices, draws

    def fixed_panel(self, count, *, seed=173):
        pools = dict.fromkeys(self.MIX, 0)
        cases = {pool: dict.fromkeys(groups, 0) for pool, groups in self.mix_buckets.items()}
        return self._mixture_select(count, np.random.default_rng(seed), pools, cases, 0, unique=True)[0]

    def batch(self, size, *, indices=None):
        if indices is None:
            indices, self.draws = self._mixture_select(size, self.rng, self.mix_counts,
                self.cohort_counts, self.draws)
            for index in indices:self.case_counts[self.rows[index]['category']] += 1
        return super().batch(size, indices=indices)

    def sampler_state(self):
        return {**super().sampler_state(), 'mix_counts': dict(self.mix_counts),
                'cohort_counts': {pool: dict(cases) for pool, cases in self.cohort_counts.items()}}

    def restore_sampler(self, state):
        if (set(state.get('mix_counts', {})) != set(self.MIX)
                or sum(state['mix_counts'].values()) != state['draws']
                or set(state.get('cohort_counts', {})) != set(self.mix_buckets)
                or any(set(state['cohort_counts'][pool]) != set(groups)
                       or sum(state['cohort_counts'][pool].values()) != state['mix_counts'][pool]
                       for pool, groups in self.mix_buckets.items())):
            raise ValueError('Combat/appearance/navigation sampler clocks changed')
        super().restore_sampler(state)
        self.mix_counts = dict(state['mix_counts'])
        self.cohort_counts = {pool: dict(cases) for pool, cases in state['cohort_counts'].items()}

    def cohort_rows(self, indices):
        return {'combat': {case: [i for i, index in enumerate(indices) if case in self.rows[index]['combat_cases']]
                           for case in self.COMBAT_CASES},
                'navigation': {case: [i for i, index in enumerate(indices)
                    if self.rows[index].get('navigation_case') == case]
                    for case in self.mix_buckets['navigation']}}
