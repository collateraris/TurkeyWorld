"""Own-model state collection remains separate from oracle-labelled branches."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Optional PyTorch dependency is not installed')
class ModelVisitedCollectionTests(unittest.TestCase):
    def model(self, invalid=False):
        from turkeyworld.models import WorldModelConfig

        class FrozenModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(()))
                self.config = WorldModelConfig(kind='teacher', architecture='vjepa2_ac',
                    image_size=128, endless=True)
                self.initializations, self.inputs, self.outputs = [], [], []

            def encode_map(self, grid):
                return grid.float()[:, None]

            def initialize_from_image(self, rgb, state, fmap):
                self.initializations.append((rgb.clone(), state.clone()))
                return state.new_zeros(state.shape[0], 1)

            def transition(self, state, action, fmap, memory):
                self.inputs.append(state.clone())
                result = state.clone()
                result[:, 0] = -1. if invalid else result[:, 0]+.0002
                result[:, -1] += 1/(30*300)
                self.outputs.append(result.clone())
                return {'state': result, 'memory': memory+1}

        return FrozenModel()

    def test_neural_rollout_uses_only_own_previous_state_and_one_initial_image(self):
        from turkeyworld.collect_model_states import collect_model_visited_clips
        model = self.model()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'train'
            with patch('turkeyworld.collect_model_states.load_world_checkpoint', return_value=(model, {})):
                manifest = collect_model_visited_clips(Path(directory)/'teacher.pt', output,
                    count=1, steps=6, max_model_steps=8, device='cpu')
            self.assertTrue(manifest['complete'])
            self.assertEqual(manifest['completed_clips'], 1)
            self.assertEqual(manifest['source_image_initializations'], 1)
            self.assertEqual(len(model.initializations), 1)
            self.assertEqual(len(model.inputs), 8)
            for previous_output, next_input in zip(model.outputs[:-1], model.inputs[1:]):
                torch.testing.assert_close(previous_output, next_input, rtol=0, atol=0)
            self.assertFalse(model.training)
            self.assertFalse(model.weight.requires_grad)
            self.assertIsNone(model.weight.grad)
            with np.load(manifest['records'][0]['path'], allow_pickle=False) as data:
                self.assertEqual(str(data['collection_kind']), 'model_visited_state_oracle_branch')
                self.assertEqual(int(data['visited_step']), 8)
                np.testing.assert_allclose(data['model_visited_state'], model.outputs[-1][0].numpy(), rtol=0, atol=0)
                np.testing.assert_allclose(data['states'][0], data['model_visited_state'], rtol=0, atol=2e-6)
                self.assertEqual(data['states'].shape, (7, STATE_DIM))
                self.assertEqual(data['frames'].shape, (7, 128, 128, 3))
                self.assertEqual(data['depth'].shape, (7, 32))
                self.assertEqual(data['visibility'].shape, (7, ENTITY_SLOTS))
                self.assertFalse(bool(data['future_oracle_feedback']))
                self.assertEqual(int(data['model_state_repairs']), 0)

    def test_saved_oracle_branch_replays_states_events_and_render_exactly(self):
        from turkeyworld.collect_model_states import collect_model_visited_clips
        from turkeyworld.engine import Game, EVENT_NAMES
        model = self.model()
        with tempfile.TemporaryDirectory() as directory:
            with patch('turkeyworld.collect_model_states.load_world_checkpoint', return_value=(model, {})):
                manifest = collect_model_visited_clips(Path(directory)/'teacher.pt', Path(directory)/'train',
                    count=1, steps=10, max_model_steps=4, device='cpu')
            with np.load(manifest['records'][0]['path'], allow_pickle=False) as data:
                game = Game(seed=int(data['seed']), endless=True)
                game.set_state(json.loads(str(data['initial_state'])))
                np.testing.assert_array_equal(game.render(128, overlays=False), data['frames'][0])
                np.testing.assert_array_equal(game.state_vector(), data['states'][0])
                for index, action in enumerate(data['actions']):
                    before = np.array([enemy['alive'] for enemy in game.enemies])
                    events = game.step(action, dt=float(data['dt']))
                    expected = [float(bool(events.get(name, False))) for name in EVENT_NAMES]
                    np.testing.assert_array_equal(expected, data['events'][index])
                    self.assertEqual(events.get('hit_id', -1), int(data['hit_ids'][index]))
                    np.testing.assert_array_equal(game.state_vector(), data['states'][index+1])
                    np.testing.assert_array_equal(game.render(128, overlays=False), data['frames'][index+1])
                    respawn = ~before & np.array([enemy['alive'] for enemy in game.enemies])
                    np.testing.assert_array_equal(respawn, data['respawn_flags'][index])

    def test_invalid_model_states_are_skipped_without_reference_fallback_or_repairs(self):
        from turkeyworld.collect_model_states import collect_model_visited_clips
        model = self.model(invalid=True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'train'
            with patch('turkeyworld.collect_model_states.load_world_checkpoint', return_value=(model, {})):
                manifest = collect_model_visited_clips(Path(directory)/'teacher.pt', output,
                    count=2, steps=4, max_model_steps=4, max_attempts=2, device='cpu')
            self.assertFalse(manifest['complete'])
            self.assertEqual(manifest['completed_clips'], 0)
            self.assertEqual(manifest['invalid_candidates'], 2)
            self.assertEqual(manifest['attempts_without_valid_state'], 2)
            self.assertEqual(len(list(output.glob('visited_*.npz'))), 0)
            self.assertEqual(manifest['model_state_repairs'], 0)
            self.assertFalse(manifest['future_oracle_feedback'])

    def test_existing_clips_resume_without_more_neural_transitions(self):
        from turkeyworld.collect_model_states import collect_model_visited_clips
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, output = Path(directory)/'teacher.pt', Path(directory)/'train'
            first, second = self.model(), self.model()
            with patch('turkeyworld.collect_model_states.load_world_checkpoint', return_value=(first, {})):
                original = collect_model_visited_clips(checkpoint, output, count=1,
                    steps=4, max_model_steps=4, device='cpu')
            with patch('turkeyworld.collect_model_states.load_world_checkpoint', return_value=(second, {})):
                resumed = collect_model_visited_clips(checkpoint, output, count=1,
                    steps=4, max_model_steps=4, device='cpu')
            self.assertTrue(resumed['complete'])
            self.assertTrue(resumed['records'][0]['resumed'])
            self.assertEqual(original['records'][0]['path'], resumed['records'][0]['path'])
            self.assertEqual(len(second.inputs), 0)
            self.assertEqual(len(second.initializations), 0)

    def test_validation_destination_and_numeric_repairs_are_rejected(self):
        from turkeyworld.collect_model_states import collect_model_visited_clips, _restore_unmodified_candidate
        from turkeyworld.engine import Game
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                collect_model_visited_clips('unused.pt', Path(directory)/'val', device='cpu')
        game = Game(endless=True)
        state = game.state_vector()
        state[4] = 1.1
        with self.assertRaisesRegex(ValueError, 'repair'):
            _restore_unmodified_candidate(game, state)


if __name__ == '__main__':
    unittest.main()
