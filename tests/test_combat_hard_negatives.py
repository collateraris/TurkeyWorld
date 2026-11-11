import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from turkeyworld.collect_combat_hard_negatives import build_pose_plan, collect_dataset
from turkeyworld.engine import Game
from turkeyworld.staged_curriculum import StagedSequenceData


class CombatHardNegativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_default_plan_is_legal_occluded_and_geographically_split(self):
        plan = build_pose_plan()
        self.assertEqual(plan, build_pose_plan())
        train = [row for row in plan if row['split'] == 'train']
        val = [row for row in plan if row['split'] == 'val']
        self.assertEqual((len(train), len(val)), (256, 64))
        self.assertFalse({row['seed'] for row in train}&{row['seed'] for row in val})
        self.assertFalse({tuple(row['corner'][:2]) for row in train}&
                         {tuple(row['corner'][:2]) for row in val})
        a, b = (np.asarray([row['pose'] for row in rows]) for rows in (train, val))
        self.assertGreaterEqual(np.linalg.norm(a[:, None, :2]-b[None, :, :2], axis=-1).min(), .05)
        self.assertGreaterEqual(np.linalg.norm(a[:, None, 2:]-b[None, :, 2:], axis=-1).min(), .05)
        game = Game(endless=True)
        for row in plan:
            player, npc = np.asarray(row['pose']).reshape(2, 2)
            self.assertTrue(game._walkable(*player, game.PLAYER_RADIUS))
            self.assertTrue(game._walkable(*npc, game.ENEMY_RADIUS))
            self.assertLessEqual(np.linalg.norm(npc-player), 1.)
            self.assertFalse(game._line_clear(*npc, *player))

    def test_exact_source_replay_has_initial_negative_then_natural_ai_positives(self):
        with TemporaryDirectory() as folder:
            report = collect_dataset(folder, train_clips=6, val_clips=3, workers=1)
            self.assertEqual(report['status'], 'complete')
            self.assertGreater(sum(row['later_positive'] for row in report['records']), 0)
            for split in ('train', 'val'):
                row = next(row for row in report['records'] if row['split'] == split)
                with np.load(Path(folder)/row['relative_path']) as source:
                    game = Game(seed=int(source['seed']), endless=True)
                    game.set_state(json.loads(str(source['initial_state'])))
                    self.assertTrue(source['source_melee_eligible'][0, 0])
                    self.assertFalse(source['source_los_clear'][0, 0])
                    self.assertFalse(source['attack_flags'][0, 0])
                    self.assertTrue(source['source_hard_negative_flags'][0, 0])
                    np.testing.assert_array_equal(game.state_vector(), source['states'][0])
                    for index, action in enumerate(source['actions']):
                        old = game.state_vector()
                        clear = game._line_clear(game.enemies[0]['x'], game.enemies[0]['y'],
                                                 game.player['x'], game.player['y'])
                        event = game.step(action, dt=1/30)
                        np.testing.assert_array_equal(game.state_vector(), source['states'][index+1])
                        np.testing.assert_array_equal(Game.event_vector(event), source['events'][index])
                        np.testing.assert_array_equal(game.render(size=32, overlays=False), source['frames'][index+1])
                        self.assertEqual(clear, bool(source['source_los_clear'][index, 0]))
                        if source['source_hard_negative_flags'][index, 0]:
                            self.assertFalse(clear)
                            self.assertEqual(event['attacker_ids'], [])
                    self.assertEqual(source['frames'].shape, (33, 32, 32, 3))
                    np.testing.assert_array_equal(source['object_masks'],
                                                  np.any(source['frames'] != source['background_frames'], -1))
            data = StagedSequenceData(Path(folder)/'moving1/rgb/train', 'cpu', mode='stream')
            batch = data.batch(2, context=1, horizon=16, indices=np.array([[0, 0], [1, 0]]))
            self.assertEqual(tuple(batch['states'].shape), (2, 17, 20))
            self.assertEqual(tuple(batch['attack_flags'].shape), (2, 16, 1))

    def test_complete_resume_preserves_manifest_bytes_and_refuses_reuse_mismatch(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            collect_dataset(root, train_clips=2, val_clips=2, workers=1)
            manifest = (root/'manifest.json').read_bytes()
            files = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in root.rglob('*.npz')}
            collect_dataset(root, train_clips=2, val_clips=2, workers=1)
            self.assertEqual((root/'manifest.json').read_bytes(), manifest)
            for path, (mtime, content) in files.items():
                self.assertEqual(path.stat().st_mtime_ns, mtime)
                self.assertEqual(path.read_bytes(), content)
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                collect_dataset(root, train_clips=3, val_clips=2, workers=1)
            self.assertEqual((root/'manifest.json').read_bytes(), manifest)


if __name__ == '__main__':
    unittest.main()
