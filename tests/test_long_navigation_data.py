"""Long no-enemy navigation is an explicit, replayable Source environment."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from turkeyworld.engine import Game, STATE_DIM
from turkeyworld.long_navigation_data import CASES, ENVIRONMENT_PROFILE, collect_long_navigation, record_episode


class LongNavigationDataTests(unittest.TestCase):
    def test_all_six_cases_use_one_initialization_and_normal_continuous_player_steps(self):
        for index, case in enumerate(CASES):
            with self.subTest(case=case):
                payload = record_episode(90_000_000+index, case, transitions=48)
                replay = Game(endless=True)
                replay.set_state(json.loads(str(payload['initial_state'])))
                self.assertFalse(replay.enemies_enabled)
                for offset in range(49):
                    np.testing.assert_array_equal(replay.state_vector(), payload['states'][offset])
                    np.testing.assert_array_equal(replay.render(32, False), payload['frames'][offset])
                    if offset < 48:
                        event = replay.step(payload['actions'][offset], 1/30)
                        np.testing.assert_array_equal(Game.event_vector(event), payload['events'][offset])
                self.assertEqual(payload['states'].shape, (49, STATE_DIM))
                self.assertFalse(payload['object_masks'].any())
                self.assertFalse(payload['respawn_flags'].any())
                self.assertFalse(payload['states'][:, 12:14].any())
                np.testing.assert_array_equal(payload['frames'], payload['background_frames'])
                self.assertTrue(np.all(payload['states'][:, 4] == 1))

    def test_full_fifteen_seconds_include_more_than_respawn_delay_without_actor_or_clock_repairs(self):
        payload = record_episode(90_000_005, 'short_return')
        self.assertEqual(payload['frames'].shape, (452, 32, 32, 3))
        self.assertEqual(payload['actions'].shape, (451, 4))
        self.assertEqual(payload['attack_flags'].shape, (451, 1))
        self.assertFalse(payload['events'].any())
        self.assertFalse(payload['states'][:, 12:14].any())
        self.assertAlmostEqual(float((payload['states'][-1, -1]-payload['states'][1, -1])*300), 15., places=5)
        np.testing.assert_allclose(np.diff(payload['states'][:, -1])*300, 1/30, atol=2e-6, rtol=0)
        self.assertGreaterEqual(float(payload['states'][-1, 16]*.6), 2.999)
        self.assertTrue((payload['actions'][:, 0] > 0).any())
        self.assertTrue((payload['actions'][:, 0] < 0).any())

    def test_resume_immutable_config_split_and_loader_schema(self):
        from turkeyworld.staged_curriculum import StagedSequenceData
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'dataset'
            report = collect_long_navigation(root, 2, 2, transitions=12, workers=1)
            self.assertEqual(report['completed_files'], 4)
            paths = sorted(root.rglob('*.npz'))
            bytes_before = {path: path.read_bytes() for path in paths}
            mtimes = {path: path.stat().st_mtime_ns for path in paths}
            with patch('turkeyworld.long_navigation_data.record_episode', side_effect=AssertionError('Must reuse')):
                again = collect_long_navigation(root, 2, 2, transitions=12, workers=1)
            self.assertEqual(again['status'], 'complete')
            for path in paths:
                self.assertEqual(path.read_bytes(), bytes_before[path]); self.assertEqual(path.stat().st_mtime_ns, mtimes[path])
            with self.assertRaisesRegex(ValueError, 'Immutable'):
                collect_long_navigation(root, 3, 2, transitions=12, workers=1)
            seeds = []
            for path in paths:
                with np.load(path, allow_pickle=False) as saved:
                    self.assertEqual(str(saved['environment_profile']), ENVIRONMENT_PROFILE)
                    self.assertEqual(str(saved['stage']), 'navigation')
                    self.assertEqual(int(saved['resets_inside_trajectory']), 0)
                    seeds.append(int(saved['seed']))
            self.assertEqual(len(set(seeds)), 4)
            data = StagedSequenceData(root/'rgb/val', 'cpu', mode='stream')
            with patch.object(Game, 'step', side_effect=AssertionError('No training oracle')), \
                    patch.object(Game, 'render', side_effect=AssertionError('No training oracle')):
                batch = data.batch(2, context=2, horizon=8, indices=[(0, 0), (1, 1)])
            self.assertEqual(tuple(batch['states'].shape), (2, 10, STATE_DIM))
            self.assertEqual(tuple(batch['target_object_masks'].shape), (2, 8, 1, 32, 32))
            self.assertEqual(tuple(batch['target_background_frames'].shape), (2, 8, 3, 32, 32))

    def test_seed_repeatability_and_nonempty_unknown_destination_rejection(self):
        first = record_episode(90_000_032, 'slide', transitions=12)
        second = record_episode(90_000_032, 'slide', transitions=12)
        for field in ('states', 'frames', 'actions', 'events'):
            np.testing.assert_array_equal(first[field], second[field])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root/'foreign.txt').write_text('preserve')
            with self.assertRaisesRegex(ValueError, 'nonempty'):
                collect_long_navigation(root, 1, 0, workers=1)
            self.assertEqual((root/'foreign.txt').read_text(), 'preserve')


if __name__ == '__main__': unittest.main()
