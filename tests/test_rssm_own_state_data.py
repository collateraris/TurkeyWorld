from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from turkeyworld.rssm_own_state_data import collect_own_training_states


class OwnCollectionTests(unittest.TestCase):
    def test_own_rollout_uses_only_train_context_and_future_source_cannot_correct_it(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__(); self.weight = nn.Parameter(torch.ones(()))
                self.config = SimpleNamespace(architecture='dreamer_rssm', navigation_only=True)
                self.transitions = 0; self.observations = 0
            def encode_map(self, grid): return grid.float()
            def initialize_from_image(self, rgb, state, fmap):
                self.observations += 1
                assert rgb.shape[1] == state.shape[1] == 1
                return torch.zeros(len(state), 3)
            def transition(self, state, action, fmap, memory):
                self.transitions += 1
                return {'state': state+state.new_tensor([0.00001]+[0.]*19), 'memory': memory+1}
        class Data:
            future = 0.
            def __init__(self, directory, *args, **kwargs):
                assert Path(directory).parts[-3:] == ('long', 'rgb', 'train')
                self.episodes = [{'path': 'train/episode.npz'}]
            def batch(self, count, *, context, horizon, indices):
                assert context == 1 and horizon == 450
                states = torch.zeros(count, 451, 20)
                states[:, :, 4] = 1
                states[:, 1:] += self.future
                return {'states': states, 'actions': torch.zeros(count, 450, 4),
                    'context_frames': torch.zeros(count, 1, 3, 32, 32),
                    'map': torch.zeros(count, 16, 16, dtype=torch.long)}
        with TemporaryDirectory() as tmp, patch('turkeyworld.rssm_own_state_data.StagedSequenceData', Data):
            results = []
            for index, future in enumerate((0., .5)):
                Data.future = future
                model = Model()
                output = Path(tmp)/f'own{index}.npz'
                with patch('turkeyworld.rssm_own_state_data.load_world_checkpoint', return_value=(model, {'training_steps':339})):
                    metadata = collect_own_training_states(Path(tmp)/'world.pt', Path(tmp), output,
                        count=1, batch_size=1, device='cpu')
                with np.load(output) as saved:
                    results.append(saved['predicted'].copy())
                    self.assertEqual(str(saved['split']), 'train')
                self.assertEqual(model.transitions, 450)
                self.assertEqual(model.observations, 1)
                self.assertEqual(metadata['engine_corrections'], 0)
                self.assertFalse(metadata['future_source_feedback'])
            np.testing.assert_array_equal(*results)


if __name__ == '__main__':
    unittest.main()
