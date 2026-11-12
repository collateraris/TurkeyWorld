"""B64 scales true Source quotas; saved B32 ledgers/RNG remain compatible."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from test_rssm_attack_positive_sampler import source_files, sampler
from test_rssm_target_sampler import data_for
from turkeyworld.engine import Game
from turkeyworld.rssm_spawn_curriculum import RecoveryCombatSampler, source_recovery_labels


class SamplerBatch64Tests(unittest.TestCase):
    def test_actual_source64_all_quotas_future_labels_and_distance_strata(self):
        with TemporaryDirectory() as folder:
            paths = source_files(Path(folder)/'train')
            hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
            with patch.object(Game, 'step', side_effect=AssertionError('No Source calls in sampling')), \
                    patch.object(Game, 'render', side_effect=AssertionError('No image calls in sampling')):
                selected = sampler(paths, context=2, horizon=16)
                pairs = selected.sample(64, np.random.default_rng(173))
            report = selected.last_sample
            self.assertEqual(pairs.shape, (64, 2))
            quota = {'spawn': 16, 'attack_negative': 16, 'attack_positive': 16, 'shooting_replay': 16}
            self.assertEqual(report['quota'], quota)
            self.assertEqual(selected.summary(batch_size=64)['quota'], quota)
            self.assertEqual(report['target_subquotas'], {'hit_near': 4, 'hit_mid': 4, 'hit_far': 4, 'fired_miss': 2, 'legacy': 2})
            self.assertEqual(report['attack_subquotas'], {'positive_near_wall': 8, 'positive_open': 8,
                'negative_occluded': 12, 'negative_other': 4})
            self.assertEqual(report['fired_miss_subquotas'], {'wall_miss': 1, 'other_miss': 1})
            self.assertEqual(report['ordinary_sampler']['sampled_windows'], 2)
            self.assertEqual(Counter(row['requested'] for row in report['attack_selected_windows']), report['attack_subquotas'])
            for row in report['attack_selected_windows']:
                self.assertIn([row['episode'], row['start']], pairs.tolist())
                with np.load(paths[row['episode']]) as source:
                    label = source_recovery_labels(source['states'], source['attack_flags'], float(source['dt']))
                    ticks = row['source_transition_indices']; first = row['start']+1
                    self.assertTrue(ticks)
                    self.assertTrue(all(first <= tick < first+16 for tick in ticks))
                    mask = label['attack_positive'] if row['requested'].startswith('positive') else label['attack_negative']
                    self.assertTrue(mask[ticks].all())
                    if row['requested'] == 'negative_occluded':
                        self.assertTrue(source['source_hard_negative_flags'][ticks, 0].all())
            self.assertEqual(Counter(row['requested'] for row in report['target_selected_windows']),
                {'hit_near': 4, 'hit_mid': 4, 'hit_far': 4, 'wall_miss': 1, 'other_miss': 1})
            for row in report['target_selected_windows']:
                with np.load(paths[row['episode']]) as source:
                    ticks = row['source_transition_indices']; first = row['start']+1
                    self.assertTrue(ticks)
                    self.assertTrue(all(first <= tick < first+16 for tick in ticks))
                    self.assertTrue((source['events'][ticks, 0] > .5).all())
                    if row['requested'].startswith('hit_'):
                        self.assertTrue((source['hit_ids'][ticks] == 0).all())
                        distance = np.linalg.norm((source['states'][ticks, 8:10]-source['states'][np.asarray(ticks)+1, :2])*16, axis=-1)
                        self.assertTrue((distance < 2).all() if row['requested'] == 'hit_near' else
                            ((distance >= 2) & (distance < 5)).all() if row['requested'] == 'hit_mid' else (distance >= 5).all())
                    else: self.assertTrue((source['hit_ids'][ticks] < 0).all())
            self.assertEqual(hashes, [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths])

    def test_four64_draws_resume_existing_v2_without_reset_and_preserve_rng_exact(self):
        with TemporaryDirectory() as folder:
            paths = source_files(Path(folder)/'train'); selected = sampler(paths)
            rng = np.random.default_rng(173); selected.sample(32, rng)
            parent = deepcopy(selected.state_dict()); rng_at_boundary = deepcopy(rng.bit_generator.state)
            restored = sampler(paths); restored.restore_state(parent)
            resumed_rng = np.random.default_rng(1); resumed_rng.bit_generator.state = rng_at_boundary
            for _ in range(4):
                expected = selected.sample(64, rng)
                np.testing.assert_array_equal(expected, restored.sample(64, resumed_rng))
                self.assertEqual(selected.last_sample, restored.last_sample)
            state = restored.state_dict()
            self.assertEqual(state, selected.state_dict())
            self.assertEqual(resumed_rng.bit_generator.state, rng.bit_generator.state)
            self.assertEqual(state['draws']-parent['draws'], 256)
            self.assertEqual(state['negative_draws']-parent['negative_draws'], 64)
            self.assertEqual(state['positive_draws']-parent['positive_draws'], 64)
            self.assertEqual(state['v2_draws']-parent['v2_draws'], 256)
            self.assertEqual(sum(state['anchor_counts'].values())-sum(parent['anchor_counts'].values()), 64)
            self.assertEqual(state['retained_legacy_ledger'], parent['retained_legacy_ledger'])
            self.assertEqual(state['retained_legacy_ledger_sha256'], parent['retained_legacy_ledger_sha256'])
            self.assertLessEqual(max(state['anchor_counts'].values())-min(state['anchor_counts'].values()), 1)

    def test_b32_full_trace_is_bitexact_to_pre_b64_implementation(self):
        # Captured immediately before the B64-only extension. The trace covers
        # four draws, all selected indices/labels/counters and the caller's RNG.
        with TemporaryDirectory() as folder:
            selected = sampler(source_files(Path(folder)/'train'))
            rng = np.random.default_rng(173); trace = []
            for _ in range(4):
                pairs = selected.sample(32, rng)
                trace.append({'pairs': pairs.tolist(), 'sample': selected.last_sample,
                    'state': selected.state_dict(), 'rng': rng.bit_generator.state})
            self.assertEqual(hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest(),
                '5d1ca8820aaee8ad2ac995462fba0d2beb2b9df67394187206a0e4ca7f25189c')

    def test_b64_requires_v2_opt_in_before_any_rng_or_counter_mutation(self):
        with TemporaryDirectory() as folder:
            paths = source_files(Path(folder)/'train')
            for target in (False, True):
                selected = RecoveryCombatSampler(data_for(paths), target_distance_balanced=target)
                rng = np.random.default_rng(173); before, before_rng = selected.state_dict(), deepcopy(rng.bit_generator.state)
                with self.assertRaises(ValueError): selected.sample(64, rng)
                with self.assertRaises(ValueError): selected.summary(batch_size=64)
                self.assertEqual(selected.state_dict(), before)
                self.assertEqual(rng.bit_generator.state, before_rng)


if __name__ == '__main__': unittest.main()
