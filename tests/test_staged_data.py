"""Staged one-enemy data stays exact source physics, never frozen AI labels."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from turkeyworld.engine import BEHAVIOR_VERSION, STATE_DIM, Game
from turkeyworld.staged_data import collect_staged


class StagedDataTests(unittest.TestCase):
    def files(self, root, stage, split='train'):
        kind = 'snapshots' if stage == 'static_enemy' else 'rgb'
        return sorted((root/stage/kind/split).glob('*.npz'))

    def replay(self, path):
        with np.load(path, allow_pickle=False) as saved:
            game = Game(endless=True)
            game.set_state(json.loads(str(saved['initial_state'])))
            for index in range(len(saved['states'])):
                np.testing.assert_array_equal(game.state_vector(), saved['states'][index])
                np.testing.assert_array_equal(game.render(32, False), saved['frames'][index])
                if index < len(saved['actions']):
                    event = game.step(saved['actions'][index], float(saved['dt']))
                    np.testing.assert_array_equal(Game.event_vector(event), saved['events'][index])
                    self.assertEqual(event.get('hit_id', -1), saved['hit_ids'][index])

    def test_native_schema_separate_static_claims_and_disjoint_split_seeds(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = collect_staged(root, train_clips=1, val_clips=1, frames=4, workers=1)
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['world_transition_examples'], 16)
            self.assertEqual(report['renderer_snapshot_examples'], 8)
            seeds = []
            for stage in ('navigation', 'static_enemy', 'moving1'):
                for split in ('train', 'val'):
                    path = self.files(root, stage, split)[0]
                    with np.load(path, allow_pickle=False) as saved:
                        static = stage == 'static_enemy'
                        observations = 4 if static else 5
                        self.assertEqual(str(saved['behavior_version']), BEHAVIOR_VERSION)
                        self.assertEqual(saved['states'].shape, (observations, STATE_DIM))
                        self.assertEqual(saved['frames'].shape, (observations, 32, 32, 3))
                        self.assertEqual(saved['object_masks'].shape, (observations, 32, 32))
                        self.assertEqual(saved['object_masks'].dtype, np.bool_)
                        self.assertEqual(saved['background_frames'].shape, saved['frames'].shape)
                        self.assertEqual(bool(saved['world_transition_compatible']), not static)
                        self.assertEqual(int(saved['entity_slots']), 1)
                        self.assertEqual(str(saved['split']), split)
                        self.assertNotIn('hr_frames', saved.files)
                        if static:
                            self.assertNotIn('actions', saved.files)
                            self.assertNotIn('events', saved.files)
                        else:
                            self.assertEqual(saved['attack_flags'].shape, (4, 1))
                            self.assertEqual(saved['respawn_flags'].shape, (4, 1))
                        seeds.append(int(saved['seed']))
                    if not static:
                        self.replay(path)
            self.assertEqual(len(set(seeds)), len(seeds))

    def test_navigation_disabled_enemy_stays_dead_invisible_and_below_respawn_time(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            collect_staged(root, train_clips=1, val_clips=0, stages=('navigation',), workers=1)
            path = self.files(root, 'navigation')[0]
            with np.load(path, allow_pickle=False) as saved:
                npc = saved['states'][:, 8:-2].reshape(-1, 1, 10)
                self.assertFalse(bool(npc[:, :, 5].any()))
                self.assertFalse(bool(npc[:, :, 4].any()))
                self.assertFalse(bool(saved['respawn_flags'].any()))
                self.assertFalse(bool(saved['object_masks'].any()))
                self.assertLess(float(npc[-1, 0, 8]*.6), 3.)
                self.assertAlmostEqual(float(npc[0, 0, 8]*.6), .6, places=6)
                np.testing.assert_array_equal(saved['frames'], saved['background_frames'])
            self.replay(path)

    def test_all_six_navigation_cases_have_no_corpse_pixels_under_source_renderer(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            collect_staged(root, train_clips=6, val_clips=6, stages=('navigation',), workers=1)
            for split in ('train', 'val'):
                for path in self.files(root, 'navigation', split):
                    with np.load(path, allow_pickle=False) as saved:
                        self.assertEqual(int(saved['staged_data_version']), 2)
                        self.assertFalse(bool(saved['object_masks'].any()), str(saved['movement_case']))
                        np.testing.assert_array_equal(saved['frames'], saved['background_frames'])
                        np.testing.assert_array_equal(saved['states'][0, 8:10], saved['states'][0, :2])
                    self.replay(path)

    def test_moving_cases_include_normal_motion_attacks_kills_and_genuine_respawn(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            collect_staged(root, train_clips=10, val_clips=0, stages=('moving1',), workers=1)
            cases = {}
            for path in self.files(root, 'moving1'):
                with np.load(path, allow_pickle=False) as saved:
                    case = str(saved['movement_case'])
                    cases[case] = {key: saved[key].copy() for key in ('states', 'actions', 'events', 'attack_flags', 'respawn_flags')}
                    self.assertFalse(bool(saved['ai_freeze']))
                    self.assertFalse(bool(saved['engine_semantics_modified']))
            self.assertEqual(len(cases), 10)
            self.assertGreater(float(cases['straight']['states'][:, 10:12].max()), 0.)
            self.assertTrue(bool(cases['combat']['events'][:, 1].any()))
            self.assertTrue(bool(cases['combat']['events'][:, 2].any()))
            self.assertTrue(bool(cases['attack']['attack_flags'].any()))
            attack_indices = np.flatnonzero(cases['attack']['attack_flags'].any(-1))
            self.assertGreaterEqual(int(attack_indices[0]), 1)
            self.assertTrue(bool(cases['attack']['attack_flags'][1:].any()))
            self.assertTrue(bool(cases['respawn']['respawn_flags'].any()))
            blocked = np.linalg.norm(np.diff(cases['wall_stop']['states'][:, :2], axis=0), axis=-1)
            self.assertTrue(bool((blocked < 1e-7).any()))
            returning = cases['short_return']['states']
            self.assertGreater(float(np.linalg.norm(returning[16, :2]-returning[0, :2])), .05)
            np.testing.assert_allclose(returning[-1, :2], returning[0, :2], atol=1e-6)

    def test_static_snapshots_replay_exactly_without_any_game_step(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(Game, 'step', side_effect=AssertionError('Static snapshots are not transitions')):
                collect_staged(root, train_clips=1, val_clips=0, stages=('static_enemy',), frames=8, workers=1)
            with np.load(self.files(root, 'static_enemy')[0], allow_pickle=False) as saved:
                snapshots = json.loads(str(saved['snapshot_states']))
                self.assertEqual(len(snapshots), 8)
                self.assertNotIn('actions', saved.files)
                self.assertTrue(bool(saved['object_masks'].any()))
                game = Game(endless=True)
                for index, state in enumerate(snapshots):
                    game.set_state(state)
                    np.testing.assert_array_equal(game.state_vector(), saved['states'][index])
                    np.testing.assert_array_equal(game.render(32, False), saved['frames'][index])
                    self.assertEqual(len(game.enemies), 1)
                    enemies = game.enemies
                    game.enemies = []
                    background = game.render(32, False)
                    game.enemies = enemies
                    np.testing.assert_array_equal(background, saved['background_frames'][index])
                    np.testing.assert_array_equal(np.any(saved['frames'][index] != background, axis=-1), saved['object_masks'][index])

    def test_resume_preserves_npz_bytes_mtime_and_does_not_instantiate_engine(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = dict(train_clips=1, val_clips=0, stages=('moving1',), frames=4, workers=1)
            collect_staged(root, **args)
            path = self.files(root, 'moving1')[0]
            original, mtime = path.read_bytes(), path.stat().st_mtime_ns
            with patch.object(Game, '__init__', side_effect=AssertionError('Existing files must be reused')):
                report = collect_staged(root, **args)
            self.assertEqual(report['reused_files'], 1)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(path.stat().st_mtime_ns, mtime)
            with self.assertRaisesRegex(ValueError, 'configuration differs'):
                collect_staged(root, **{**args, 'frames': 5})

    def test_equal_seeds_produce_equal_arrays_in_new_roots(self):
        with tempfile.TemporaryDirectory() as folder:
            roots = [Path(folder)/name for name in ('first', 'second')]
            for root in roots:
                collect_staged(root, train_clips=1, val_clips=0, stages=('moving1',), frames=4, workers=1)
            with np.load(self.files(roots[0], 'moving1')[0], allow_pickle=False) as first, \
                    np.load(self.files(roots[1], 'moving1')[0], allow_pickle=False) as second:
                self.assertEqual(first.files, second.files)
                for key in first.files:
                    np.testing.assert_array_equal(first[key], second[key], err_msg=key)

    def test_full_game_records_genuine_context_plus_450_future_steps(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = collect_staged(root, stages=('full_game',), full_train_clips=1, full_val_clips=1,
                workers=1, full_frames=451, full_attempts=1)
            self.assertGreaterEqual(sum(group['eligible_450'] for group in result['groups'].values()), 1)
            initial_vectors = []
            for split in ('train', 'val'):
                path = self.files(root, 'full_game', split)[0]
                with np.load(path, allow_pickle=False) as saved:
                    self.assertEqual(len(saved['actions']), 451)
                    self.assertEqual(saved['states'].shape, (452, 20))
                    self.assertEqual(saved['initial_time_offset'], 0)
                    self.assertEqual(len(json.loads(str(saved['initial_state']))['enemies']), 1)
                    self.assertTrue(np.all(np.diff(saved['states'][:, -1]) > 0))
                    initial_vectors.append(saved['states'][0].copy())
                self.replay(path)
            self.assertFalse(np.array_equal(initial_vectors[0], initial_vectors[1]))

    def test_parallel_collection_is_bounded_and_reproducible(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = collect_staged(root, stages=('navigation',), train_clips=2, val_clips=0,
                                    frames=2, workers=2)
            self.assertEqual(report['completed_files'], 2)
            self.assertEqual(report['world_transition_examples'], 4)
            self.assertFalse(list(root.rglob('*.tmp')))
            for path in self.files(root, 'navigation'):
                self.replay(path)


if __name__ == '__main__':
    unittest.main()
