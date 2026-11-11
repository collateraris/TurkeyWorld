import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from turkeyworld.engine import Game
from turkeyworld.navigation_recovery_data import (ACTIONS, collect, record_episode,
    source_axis_labels, valid_own_poses)


class NavigationRecoveryDataTests(unittest.TestCase):
    def test_real_counterfactual_branches_source_replay_and_standard_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = collect(root, short_train=16, short_val=1, long_train=1, long_val=0, workers=1)
            self.assertEqual(report['completed'], 18)
            self.assertEqual(report['train_val_initial_pose_overlap'], 0)
            files = sorted((root/'short/rgb/train').glob('*.npz'))
            initial, first_actions = [], []
            for path in files:
                with np.load(path, allow_pickle=False) as data:
                    self.assertEqual(data['states'].shape, (33, 20))
                    self.assertEqual(data['frames'].shape, (33, 32, 32, 3))
                    self.assertEqual(data['attack_flags'].shape, (32, 1))
                    self.assertFalse(data['object_masks'].any())
                    self.assertTrue(np.array_equal(data['background_frames'], data['frames']))
                    self.assertFalse(data['states'][:, 12:14].any())
                    initial.append(data['states'][0].copy())
                    first_actions.append(data['actions'][0].copy())
            self.assertTrue(all(np.array_equal(initial[0], value) for value in initial))
            self.assertTrue(np.allclose(initial[0][:4], [2.5/16, 2.5/16, 0, 1]))
            self.assertGreater(len(np.unique(first_actions, axis=0)), 12)
            with np.load(files[9], allow_pickle=False) as data:
                game = Game(seed=0, endless=True, enemies_enabled=False)
                game.set_state(json.loads(str(data['initial_state'])))
                for index, controls in enumerate(data['actions']):
                    game.step(controls, dt=1/30)
                    self.assertTrue(np.array_equal(game.state_vector(), data['states'][index+1]))
                self.assertTrue(np.array_equal(game.render(size=32, overlays=False), data['frames'][-1]))
            self.assertGreater(report['audit']['moving_turn_steps'], 0)
            self.assertGreater(report['audit']['positive_turn_steps'], 0)
            self.assertGreater(report['audit']['negative_turn_steps'], 0)
            self.assertGreater(report['audit']['idle_steps'], 0)
            from turkeyworld.staged_curriculum import StagedSequenceData
            data = StagedSequenceData(root/'short/rgb/train', 'cpu', mode='stream')
            batch = data.batch(2, context=1, horizon=16, indices=np.asarray([[0, 0], [9, 0]]))
            self.assertEqual(tuple(batch['states'].shape), (2, 17, 20))
            self.assertEqual(tuple(batch['actions'].shape), (2, 16, 4))
            self.assertEqual(tuple(batch['target_object_masks'].shape), (2, 16, 1, 32, 32))
            long = next((root/'long/rgb/train').glob('*.npz'))
            with np.load(long, allow_pickle=False) as saved:
                self.assertEqual(saved['states'].shape, (452, 20))
                self.assertEqual(saved['actions'].shape, (451, 4))
                self.assertEqual(int(saved['resets_inside_trajectory']), 0)

    def test_resumption_reuses_files_without_any_source_step_or_render_and_rejects_changed_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collect(root, short_train=2, short_val=1, long_train=0, long_val=0)
            mtimes = {path: path.stat().st_mtime_ns for path in root.rglob('*.npz')}
            with (patch.object(Game, 'step', side_effect=AssertionError('No source during resume')),
                  patch.object(Game, 'render', side_effect=AssertionError('No source render during resume'))):
                result = collect(root, short_train=2, short_val=1, long_train=0, long_val=0)
            self.assertTrue(all(record['reused'] for record in result['records']))
            self.assertEqual(mtimes, {path: path.stat().st_mtime_ns for path in mtimes})
            with self.assertRaises(ValueError):
                collect(root, short_train=3, short_val=1, long_train=0, long_val=0)

    def test_own_initialization_rejects_wall_and_clipped_state_without_repairs(self):
        game = Game(seed=0, endless=True, enemies_enabled=False)
        good = game.state_vector()
        wall, clipped = good.copy(), good.copy()
        wall[:2] = 0
        clipped[4] = 1.5
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'own.npz'
            np.savez(path, predicted=np.stack([good, wall, clipped]))
            original = path.read_bytes()
            with patch.object(Game, 'step', side_effect=AssertionError('No fake trajectory')):
                poses, audit = valid_own_poses(path)
            self.assertEqual((audit['accepted'], audit['rejected']), (1, 2))
            self.assertTrue(np.array_equal(poses[0]['state'], good))
            self.assertEqual(original, path.read_bytes())

    def test_axis_audit_matches_postturn_source_motion_and_keeps_inputs_unchanged(self):
        game = Game(seed=0, endless=True, enemies_enabled=False)
        pose = {'state': game.state_vector().tolist()}
        data, _ = record_episode({'seed': 1, 'branch': 9, 'stage': 'short', 'index': 0, 'frames': 8}, pose)
        actions = data['actions'].astype(float)
        previous = actions.copy()
        command, fraction, active, audit = source_axis_labels(data['states'], actions)
        self.assertTrue(np.array_equal(actions, previous))
        actual = np.diff(data['states'][:, :2], axis=0)*16
        self.assertTrue(np.allclose((command*fraction)[active], actual[active], atol=1e-6))
        self.assertGreater(audit['moving_turn_steps'], 0)


if __name__ == '__main__':
    unittest.main()
