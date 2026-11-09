"""Native LR and genuine HR are observed from the same source state."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from turkeyworld.data_expansion import prepare_expansion
from turkeyworld.engine import Game


class PairedExpansionTests(unittest.TestCase):
    def test_native32_hr128_replay_matches_both_resolutions_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'data'
            options = dict(rgb_train=9, rgb_val=1, long_train=0, long_val=0,
                clip_frames=2, long_frames=30, size=32, hr_size=128,
                seed=50_000_000, workers=1, rgb_time_span=120.)
            report = prepare_expansion(root, **options)
            self.assertEqual(report['groups']['rgb/train']['transitions'], 18)
            self.assertEqual(report['groups']['rgb/val']['transitions'], 2)
            for path in root.glob('rgb/*/*.npz'):
                with np.load(path, allow_pickle=False) as saved:
                    self.assertEqual(saved['frames'].shape, (3, 32, 32, 3))
                    self.assertEqual(saved['hr_frames'].shape, (3, 128, 128, 3))
                    self.assertEqual(str(saved['hr_frame_source']), 'reference_engine')
                    self.assertTrue(bool(saved['paired_source_rgb']))
                    game = Game(seed=int(saved['seed']), endless=True)
                    game.set_state(json.loads(str(saved['initial_state'])))
                    for index in range(3):
                        self.assertTrue(np.array_equal(game.render(size=32, overlays=False), saved['frames'][index]))
                        self.assertTrue(np.array_equal(game.render(size=128, overlays=False), saved['hr_frames'][index]))
                        if index < 2:
                            game.step(saved['actions'][index], dt=float(saved['dt']))
            identities = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in root.glob('rgb/*/*.npz')}
            again = prepare_expansion(root, **options)
            self.assertEqual(again['groups']['rgb/train']['transitions'], 18)
            self.assertEqual(identities, {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
                                         for p in root.glob('rgb/*/*.npz')})
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                prepare_expansion(root, **{**options, 'hr_size': 0})

    def test_rejects_non_x4_hr_before_creating_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'invalid'
            with self.assertRaisesRegex(ValueError, 'exactly four'):
                prepare_expansion(root, size=32, hr_size=256)
            self.assertFalse(root.exists())


if __name__ == '__main__':
    unittest.main()
