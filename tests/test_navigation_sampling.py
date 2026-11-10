"""Exact bounded case quotas, future return windows and checkpoint resume."""
from collections import Counter
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from turkeyworld.engine import BEHAVIOR_VERSION
from turkeyworld.navigation_sampling import (NavigationFocusSampler, NAVIGATION_CASES,
                                             NAVIGATION_FOCUS_WEIGHTS)
from turkeyworld.structured_state import ENTITY_SLOTS, STATE_DIM, STATE_SCHEMA_VERSION


class NavigationSamplingTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)/'navigation'/'rgb'/'train'
        self.directory.mkdir(parents=True)
        self.paths, self.cases = [], []
        for case in NAVIGATION_CASES:
            for duplicate in range(2):
                path = self.directory/f'episode_{len(self.paths):06d}.npz'
                self.write(path, case)
                self.paths.append(path)
                self.cases.append(case)
        self.data = self.pool()

    def pool(self, paths=None, **overrides):
        values = dict(split='train', directory=self.directory,
                      episodes=[{'path': p} for p in self.paths if paths is None or p in paths])
        values.update(overrides)
        return SimpleNamespace(**values)

    def write(self, path, case, *, flip=16, changes=None):
        actions = np.zeros((32, 4), np.float32)
        if case == 'turn':
            actions[:, 2] = 1
        elif case == 'strafe':
            actions[:, 1] = 1
        else:
            actions[:, 0] = 1
        if case == 'slide':
            actions[:, 1] = 1
        if case == 'short_return' and flip is not None:
            actions[flip:, 0] = -1
        states = np.zeros((33, STATE_DIM), np.float32)
        states[:, :2], states[:, 3:6] = .5, 1
        values = dict(states=states, actions=actions, dt=np.asarray(1/30),
            split=np.asarray('train'), stage=np.asarray('navigation'), movement_case=np.asarray(case),
            dataset_kind=np.asarray('world_transitions'), world_transition_compatible=np.asarray(True),
            behavior_version=np.asarray(BEHAVIOR_VERSION), state_dim=np.asarray(STATE_DIM),
            entity_slots=np.asarray(ENTITY_SLOTS), state_schema_version=np.asarray(STATE_SCHEMA_VERSION))
        values.update(changes or {})
        np.savez_compressed(path, **values)

    def test_300_updates_of_batch16_have_exact_requested_disjoint_proportions(self):
        selector = NavigationFocusSampler(self.data)
        rng = np.random.default_rng(173)
        counts, draws = None, 0
        for _ in range(300):
            pairs, counts, draws, batch_counts = selector.sample(16, rng, counts=counts, draws=draws)
            self.assertEqual(pairs.shape, (16, 2))
            self.assertEqual(pairs.dtype, np.int64)
            self.assertEqual(Counter(self.cases[int(row)] for row in pairs[:, 0]), batch_counts)
            self.assertTrue(bool(((pairs[:, 1] >= 0) & (pairs[:, 1]+2+29 <= 33)).all()))
        self.assertEqual(draws, 4800)
        self.assertEqual(counts, dict(zip(NAVIGATION_CASES, (1440, 1200, 960, 720, 240, 240))))
        summary = selector.summary()
        self.assertTrue(summary['disjoint_source_cases'])
        self.assertTrue(summary['labels_used_only_for_sampling'])
        self.assertFalse(summary['source_future_feedback'])
        self.assertEqual(summary['eligible_windows_per_case'], dict.fromkeys(NAVIGATION_CASES, 6))

    def test_resume_restores_caller_rng_and_case_counters_without_hidden_state(self):
        first = NavigationFocusSampler(self.data)
        rng = np.random.default_rng(89)
        _, counts, draws, _ = first.sample(16, rng)
        saved_counts, saved_rng = copy.deepcopy(counts), copy.deepcopy(rng.bit_generator.state)
        expected = [first.sample(16, rng, counts=counts, draws=draws)]
        for _ in range(8):
            previous = expected[-1]
            expected.append(first.sample(16, rng, counts=previous[1], draws=previous[2]))
        restored = NavigationFocusSampler(self.data)
        self.assertEqual(first.fingerprint, restored.fingerprint)
        restored_rng = np.random.default_rng()
        restored_rng.bit_generator.state = saved_rng
        for target in expected:
            actual = restored.sample(16, restored_rng, counts=saved_counts, draws=draws)
            np.testing.assert_array_equal(actual[0], target[0])
            self.assertEqual(actual[1:], target[1:])
            saved_counts, draws = actual[1:3]
        self.assertEqual(counts, dict(zip(NAVIGATION_CASES, (5, 4, 3, 2, 1, 1))))

    def test_selection_changes_neither_source_arrays_global_rng_nor_input_counter(self):
        selector = NavigationFocusSampler(self.data)
        global_before = np.random.get_state()
        original = {p: p.read_bytes() for p in self.paths}
        counts = dict.fromkeys(NAVIGATION_CASES, 0)
        _, changed, _, _ = selector.sample(16, np.random.default_rng(33), counts=counts)
        self.assertEqual(counts, dict.fromkeys(NAVIGATION_CASES, 0))
        self.assertIsNot(changed, counts)
        global_after = np.random.get_state()
        self.assertEqual(global_before[0], global_after[0])
        np.testing.assert_array_equal(global_before[1], global_after[1])
        self.assertEqual(global_before[2:], global_after[2:])
        self.assertTrue(all(p.read_bytes() == content for p, content in original.items()))

    def test_return_windows_contain_both_commands_in_future_and_h1_has_boundary_definition(self):
        for path, case in zip(self.paths, self.cases):
            if case == 'short_return':
                self.write(path, case, flip=5)
        selector = NavigationFocusSampler(self.data, context=2, horizon=4)
        for row, start in selector.groups['short_return']:
            self.assertEqual(self.cases[row], 'short_return')
            self.assertIn(start, (1, 2, 3))
            with np.load(self.paths[row]) as saved:
                future = saved['actions'][start+1:start+5, 0]
            self.assertTrue(bool(((future[:-1] > .1) & (future[1:] < -.1)).any()))
        one = NavigationFocusSampler(self.data, context=2, horizon=1)
        self.assertEqual(one.groups['short_return'][:, 1].tolist(), [4, 4])

    def test_return_metadata_without_actual_reversal_is_not_an_eligible_bucket(self):
        for path, case in zip(self.paths, self.cases):
            if case == 'short_return':
                self.write(path, case, flip=None)
        with self.assertRaisesRegex(ValueError, 'short_return'):
            NavigationFocusSampler(self.data)
        # A return before the observed context is also insufficient.
        for path, case in zip(self.paths, self.cases):
            if case == 'short_return':
                self.write(path, case, flip=1)
        with self.assertRaisesRegex(ValueError, 'short_return'):
            NavigationFocusSampler(self.data)

    def test_train_validation_path_duplicates_and_missing_case_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'train-only'):
            NavigationFocusSampler(self.pool(split='val'))
        with self.assertRaisesRegex(ValueError, 'unique existing'):
            NavigationFocusSampler(self.pool(episodes=[{'path': self.paths[0]}]*2))
        outside = self.directory.parent/'val'; outside.mkdir()
        val_path = outside/'episode.npz'; self.write(val_path, 'slide')
        with self.assertRaisesRegex(ValueError, 'inside one train'):
            NavigationFocusSampler(self.pool(episodes=self.data.episodes+[{'path': val_path}]))
        missing = [p for p, case in zip(self.paths, self.cases) if case != 'slide']
        with self.assertRaisesRegex(ValueError, 'slide'):
            NavigationFocusSampler(self.pool(paths=missing))

    def test_source_provenance_shape_and_dead_player_are_rejected(self):
        invalid = [({'split': np.asarray('val')}, 'navigation train'),
            ({'stage': np.asarray('moving1')}, 'navigation train'),
            ({'dataset_kind': np.asarray('renderer_snapshots')}, 'navigation train'),
            ({'behavior_version': np.asarray('previous_behavior')}, 'behavior/state schema'),
            ({'entity_slots': np.asarray(8)}, 'behavior/state schema'),
            ({'states': np.zeros((33, 90), np.float32)}, 'state/action'),
            ({'actions': np.zeros((32, 3), np.float32)}, 'state/action'),
            ({'states': np.full((33, STATE_DIM), np.nan)}, 'state/action'),
            ({'states': np.zeros((33, STATE_DIM), np.float32)}, 'living player')]
        for changes, error in invalid:
            with self.subTest(changes=list(changes)):
                self.write(self.paths[0], self.cases[0], changes=changes)
                with self.assertRaisesRegex(ValueError, error):
                    NavigationFocusSampler(self.data)
                self.write(self.paths[0], self.cases[0])

    def test_weights_and_saved_counter_integrity(self):
        for bad in ({'slide': 1.}, {**NAVIGATION_FOCUS_WEIGHTS, 'turn': 0.},
                    {**NAVIGATION_FOCUS_WEIGHTS, 'turn': float('nan')},
                    {**NAVIGATION_FOCUS_WEIGHTS, 'turn': .16}):
            with self.subTest(weights=bad), self.assertRaises(ValueError):
                NavigationFocusSampler(self.data, weights=bad)
        selector = NavigationFocusSampler(self.data)
        zero = dict.fromkeys(NAVIGATION_CASES, 0)
        for counts, draws in ((None, 1), (zero, 1), ({'slide': 0}, 0),
                              ({**zero, 'turn': -1}, 0), ({**zero, 'turn': True}, 1)):
            with self.subTest(counts=counts, draws=draws), self.assertRaises(ValueError):
                selector.sample(16, np.random.default_rng(1), counts=counts, draws=draws)


if __name__ == '__main__':
    unittest.main()
