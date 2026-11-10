"""Fixed-label sampling curriculum never changes the model-input contract."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from turkeyworld.motion_curriculum import MotionCurriculum, classify_player_motion


class MotionCurriculumTests(unittest.TestCase):
    @staticmethod
    def episode(kind, count=32):
        states = np.zeros((count+1, STATE_DIM), dtype=np.float32)
        states[:, :2], states[:, 3:6] = [.3, .4], 1.
        actions = np.zeros((count, 4), dtype=np.float32)
        if kind != 'idle':actions[:, 0] = 1.
        if kind == 'slide':actions[:, 1] = 1.
        if kind == 'turn':actions[:, 2] = 1.
        for index in range(count):
            states[index+1] = states[index]
            angle = np.arctan2(states[index, 2], states[index, 3])+actions[index, 2]*2.2/30
            states[index+1, 2:4] = np.sin(angle), np.cos(angle)
            movement = actions[index, :2]/max(1., np.linalg.norm(actions[index, :2]))
            dx = (np.cos(angle)*movement[0]-np.sin(angle)*movement[1])*.08/16
            dy = (np.sin(angle)*movement[0]+np.cos(angle)*movement[1])*.08/16
            if kind != 'stop':states[index+1, 0] += dx
            if kind not in ('stop', 'slide'):states[index+1, 1] += dy
        return states, actions

    @classmethod
    def data(cls, root, kinds=('straight', 'stop', 'slide', 'turn', 'idle'), count=32):
        directory = root/'train';directory.mkdir()
        episodes = []
        for index, kind in enumerate(kinds):
            path = directory/f'episode_{index:03d}.npz'
            states, actions = cls.episode(kind, count)
            np.savez(path, states=states, actions=actions, dt=np.asarray(1/30), split=np.asarray('train'),
                frames=np.zeros((count+1, 8, 8, 3), dtype=np.uint8), events=np.zeros((count, 4), np.float32),
                hit_ids=np.full(count, -1, dtype=np.int64), attack_flags=np.zeros((count, ENTITY_SLOTS), np.float32),
                map=np.zeros((16, 16), np.int64))
            episodes.append({'path': str(path)})
        return SimpleNamespace(directory=directory, split='train', episodes=episodes)

    def test_actual_source_geometry_classification_and_terminal_masks(self):
        for kind in ('straight', 'stop', 'slide', 'turn', 'idle'):
            with self.subTest(kind=kind):
                states, actions = self.episode(kind)
                masks = classify_player_motion(states, actions)
                self.assertEqual(int(masks[kind].sum()), 32)
                self.assertEqual(int(masks['alive'].sum()), 32)
                dead = states.copy();dead[:, 4] = 0
                result = classify_player_motion(dead, actions)
                self.assertTrue(all(not mask.any() for mask in result.values()))

    def test_plan_boundaries_are_relative_to_persisted_start_and_uniform_is_retained(self):
        with TemporaryDirectory() as directory:
            curriculum = MotionCurriculum(self.data(Path(directory)), context=4, start_step=50)
            for offset, horizon in ((0, 1), (99, 1), (100, 4), (299, 4), (300, 8),
                                    (599, 8), (600, 16), (999, 16), (1000, 29), (5000, 29)):
                plan = curriculum.plan(50+offset)
                self.assertEqual(plan['horizon'], horizon)
                self.assertEqual(plan['context'], 4)
                self.assertGreaterEqual(plan['case_weights']['uniform'], .25)
                self.assertAlmostEqual(sum(plan['case_weights'].values()), 1.)
                self.assertTrue(plan['labels_used_only_for_sampling'])
            first = curriculum.plan(50)['case_weights']
            self.assertAlmostEqual(first['straight'], .45)
            self.assertAlmostEqual(first['stop'], .15)
            self.assertAlmostEqual(first['slide'], .15)
            with self.assertRaises(ValueError):curriculum.plan(49)

    def test_no_rgb_decoding_and_episode_order_stays_fixed_when_directory_grows(self):
        with TemporaryDirectory() as directory:
            data = self.data(Path(directory))
            data.episodes.reverse()
            with np.load(data.episodes[0]['path'], allow_pickle=False) as saved:archive = type(saved)
            original = archive.__getitem__

            def only_small_fields(saved, name):
                self.assertIn(name, {'states', 'actions', 'dt', 'split'})
                return original(saved, name)

            with patch.object(archive, '__getitem__', only_small_fields):
                curriculum = MotionCurriculum(data, 4, 51)
            self.assertEqual([r['path'].name for r in curriculum.records],
                             [Path(e['path']).name for e in data.episodes])
            before = curriculum.summary()
            (data.directory/'append_unindexed.npz').write_bytes(b'Not in the fixed caller episode list')
            self.assertEqual(curriculum.summary(), before)
            self.assertFalse(before['long_rollout_qualification_claimed'])
            self.assertEqual(before['transitions'], 160)

    def test_any_future_transition_qualifies_case_but_observed_context_does_not(self):
        with TemporaryDirectory() as directory:
            root = Path(directory);data = self.data(root, ('straight',))
            path = Path(data.episodes[0]['path'])
            states, actions = self.episode('straight')
            # A stop occurs only at transition4. Withcontext4/frame_start0,
            # it lies inside horizon29, although its first transition3 is free.
            states[5:, 0] -= .08/16
            np.savez(path, states=states, actions=actions, dt=1/30, split='train')
            curriculum = MotionCurriculum(data, 4, 0)
            self.assertEqual(curriculum.summary()['counts']['stop'], 1)
            self.assertEqual(curriculum.plan(1000)['available_windows']['stop'], 1)
            pairs = curriculum._index(1)['stop']
            np.testing.assert_array_equal(pairs, np.asarray([[0, 1]]))
            np.testing.assert_array_equal(curriculum._index(29)['stop'], np.asarray([[0, 0]]))
            # A stop attransition0 belongs exclusively to observed context.
            states, actions = self.episode('straight');states[1:, 0] -= .08/16
            np.savez(path, states=states, actions=actions, dt=1/30, split='train')
            observed_only = MotionCurriculum(data, 4, 0)
            self.assertEqual(observed_only.summary()['counts']['stop'], 1)
            self.assertEqual(observed_only.plan(1000)['available_windows']['stop'], 0)
            # The right boundary is exclusive: horizon4 atstart0 predicts
            # transitions3,4,5,6, so a stop at7 must not qualify start0.
            states, actions = self.episode('straight');states[8:, 0] -= .08/16
            np.savez(path, states=states, actions=actions, dt=1/30, split='train')
            boundary = MotionCurriculum(data, 4, 0)
            self.assertFalse(any(start == 0 for episode, start in boundary._index(4)['stop']))
            self.assertTrue(any(start == 1 for episode, start in boundary._index(4)['stop']))

    def test_every_conditioned_window_contains_its_case_inside_complete_future_targets(self):
        with TemporaryDirectory() as directory:
            curriculum = MotionCurriculum(self.data(Path(directory)), 4, 0)
            for horizon in (1, 4, 8, 16, 29):
                for name, pairs in curriculum._index(horizon).items():
                    for episode, start in pairs:
                        self.assertGreaterEqual(start, 0)
                        record = curriculum.records[int(episode)]
                        self.assertLessEqual(start+4+horizon, record['frames'])
                        if name != 'uniform':
                            first = start+4-1
                            self.assertTrue(record['masks'][name][first:first+horizon].any())

    def test_sampling_resume_rng_and_complete_sequence_data_windows(self):
        with TemporaryDirectory() as directory:
            data = self.data(Path(directory))
            curriculum = MotionCurriculum(data, 4, 0)
            first, resumed = np.random.default_rng(4), np.random.default_rng(999)
            global_state = np.random.get_state()
            curriculum.sample_indices(13, 99, first)
            resumed.bit_generator.state = copy.deepcopy(first.bit_generator.state)
            for step in (100, 300, 600, 1000):
                a = curriculum.sample_indices(40, step, first)
                b = curriculum.sample_indices(40, step, resumed)
                np.testing.assert_array_equal(a, b)
                self.assertEqual(a.dtype, np.int64)
                self.assertEqual(a.shape, (40, 2))
                horizon = curriculum.plan(step)['horizon']
                self.assertTrue(np.all(a[:, 1] >= 0))
                self.assertTrue(np.all(a[:, 1]+4+horizon <= 33))
            after = np.random.get_state()
            np.testing.assert_array_equal(after[1], global_state[1])
            self.assertEqual(after[2:], global_state[2:])
            from turkeyworld.jepa_training_utils import SequenceData
            source = SequenceData(data.directory, device='cpu', mode='stream', episode_paths=[Path(e['path']).name for e in data.episodes])
            pairs = curriculum.sample_indices(2, 1000, first)
            batch = source.batch(2, context=4, horizon=29, indices=pairs)
            self.assertEqual(tuple(batch['frames'].shape), (2, 33, 3, 8, 8))
            self.assertEqual(tuple(batch['actions'].shape), (2, 29, 4))
            self.assertEqual(tuple(batch['context_frames'].shape), (2, 4, 3, 8, 8))

    def test_missing_cases_redistribute_and_short_source_never_pretends_long_training(self):
        for kind in ('straight', 'idle'):
            with self.subTest(kind=kind), TemporaryDirectory() as directory:
                curriculum = MotionCurriculum(self.data(Path(directory), (kind,), count=5), 4, 0)
                self.assertEqual(curriculum.max_horizon, 2)
                plan = curriculum.plan(1000)
                self.assertEqual(plan['horizon'], 2)
                self.assertIn('stop', plan['missing_cases'])
                self.assertEqual(plan['case_weights'], {'uniform': 1.} if kind == 'idle' else {'uniform': .25, 'straight': .75})
                pairs = curriculum.sample_indices(10, 1000, np.random.default_rng(0))
                self.assertTrue(np.all(pairs[:, 1]+4+2 <= 6))

    def test_wrong_split_nonfinite_labels_and_duplicate_members_are_rejected(self):
        for problem in ('val', 'duplicate', 'nonfinite'):
            with self.subTest(problem=problem), TemporaryDirectory() as directory:
                data = self.data(Path(directory), ('straight',))
                if problem == 'val':data.split = 'val'
                elif problem == 'duplicate':data.episodes.append(data.episodes[0])
                else:
                    states, actions = self.episode('straight');states[0, 0] = np.nan
                    np.savez(data.episodes[0]['path'], states=states, actions=actions, dt=1/30, split='train')
                with self.assertRaises(ValueError):MotionCurriculum(data, 4, 0)


if __name__ == '__main__':
    unittest.main()
