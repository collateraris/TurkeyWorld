import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld import vjepa_kd_data as kd


class DistillationWindowsTest(unittest.TestCase):
    def test_whole_episode_holdout_and_action_state_alignment(self):
        with tempfile.TemporaryDirectory() as root:
            pools = {mode: [] for mode in kd.MODES}
            for i, mode in enumerate(kd.MODES):
                for j in range(3):
                    seed = 100 * i + j
                    path = Path(root) / f'{seed}.npz'
                    states = np.zeros((41, 20), np.float32)
                    states[:, 0] = np.arange(41)
                    states[:, 1] = seed
                    actions = np.zeros((40, 4), np.float32)
                    actions[:, 0] = np.arange(40)
                    respawns = np.zeros((40, 1))
                    respawns[20] = 1
                    np.savez(path, seed=seed, split='train', navigation_only=i==0, states=states,
                             actions=actions, respawn_flags=respawns, events=np.zeros((40, 4)),
                             map=np.zeros((16, 16), np.int8))
                    pools[mode].append((path, seed, sha256(path.read_bytes()).hexdigest()))
            with patch.object(kd, '_candidates', return_value=pools), patch.object(kd, 'TRAIN_EPISODES', 2), patch.object(kd, 'HELDOUT_EPISODES', 1):
                train, heldout, manifest = kd.load_kd_windows(Path(root), seed=169)
            self.assertFalse({r['sha256'] for r in manifest['train']} &
                             {r['sha256'] for r in manifest['heldout']})
            for batch, split in ((train, 'train'), (heldout, 'heldout')):
                torch.testing.assert_close(batch['actions'][:, :, 0], batch['states'][:, :, 0])
                for episode_index, row in enumerate(manifest[split]):
                    start = episode_index * kd.SAMPLES
                    samples = batch['states'][start:start + kd.SAMPLES]
                    self.assertTrue(torch.all(samples[:, :, 1] == row['seed']))
                    self.assertEqual(samples[:, 0, 0].tolist(), row['starts'])
                    if row['mode'] == 'combat':
                        self.assertTrue(any(t <= 20 < t + kd.WINDOW for t in row['starts']))


if __name__ == '__main__':
    unittest.main()
