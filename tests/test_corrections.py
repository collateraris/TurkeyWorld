"""Model-visited training queries must never repair or splice the model rollout."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from turkeyworld.collect_corrections import collect_corrections, query_branch
from turkeyworld.engine import Game


class RecordingModel(torch.nn.Module):
    """Fast injectable transition, intentionally using noncanonical ammo."""
    def __init__(self, nonfinite=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(simulation_hz=30, transition_mode='event_v1')
        self.inputs, self.memories = [], []
        self.nonfinite = nonfinite

    def encode_map(self, grid):
        return torch.zeros(len(grid), 1, 16, 16, device=grid.device)

    def initial_memory(self, batch, device, dtype):
        return torch.zeros(batch, 1, device=device, dtype=dtype)

    def transition(self, state, action, map_features, memory):
        self.inputs.append(state.detach().cpu().clone())
        self.memories.append(memory.detach().cpu().clone())
        result = state.clone()
        result[:, 0] += .0001
        result[:, 5] = .5003  # Oracle canonicalization must not feed back here.
        result[:, -1] += 1 / 9000
        if self.nonfinite:
            result[:, 0] = float('nan')
        return {'state': result, 'memory': memory+1}


class CorrectionCollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_invalid_model_state_is_rejected_without_input_mutation(self):
        game = Game()
        state = game.state_vector()
        state[:2] = np.array([4.5, 4.5]) / 16  # An actual occupied tile.
        before = state.copy()
        payload, audit = query_branch(state, game.map, np.zeros((3, 4), dtype=np.float32))
        self.assertIsNone(payload)
        self.assertEqual(audit['reason'], 'invalid_state')
        np.testing.assert_array_equal(state, before)
        bad = before.copy()
        bad[0] = np.nan
        payload, audit = query_branch(bad, game.map, np.zeros((3, 4), dtype=np.float32))
        self.assertIsNone(payload)
        self.assertEqual(audit['reason'], 'nonfinite')
        self.assertTrue(np.isnan(bad[0]))

    def test_canonical_start_labels_and_full_oracle_replay_align(self):
        game = Game(seed=8)
        state = game.state_vector()
        state[5] = .5003  # Noninteger ammo is rounded by the oracle adapter.
        state[-2] = .37  # Adapter derives kills from persistent life flags.
        original = state.copy()
        controls = np.zeros((12, 4), dtype=np.float32)
        controls[0, 3] = controls[8, 3] = 1
        payload, audit = query_branch(state, game.map, controls, seed=8,
                                      size=32, rgb=True, geometry=True)
        self.assertTrue(audit['accepted'])
        self.assertGreater(audit['canonicalization_max_abs_diff'], 0)
        np.testing.assert_array_equal(state, original)
        np.testing.assert_array_equal(payload['student_state'], original)
        self.assertAlmostEqual(float(payload['states'][0, 5]), 50 / 99, places=6)
        self.assertEqual(float(payload['states'][0, -2]), 0)
        self.assertEqual(payload['states'].shape, (13, STATE_DIM))
        self.assertEqual(payload['frames'].shape, (13, 32, 32, 3))
        self.assertEqual(payload['depth'].shape, (13, 32))
        self.assertEqual(payload['visibility'].shape, (13, ENTITY_SLOTS))
        replay = Game(seed=8)
        replay.set_state(json.loads(str(payload['initial_state'])))
        np.testing.assert_array_equal(replay.state_vector(), payload['states'][0])
        for index, action in enumerate(payload['actions']):
            event = replay.step(action, dt=float(payload['dt']))
            np.testing.assert_array_equal(replay.state_vector(), payload['states'][index+1])
            np.testing.assert_array_equal(Game.event_vector(event), payload['events'][index])
            self.assertEqual(event['hit_id'], int(payload['hit_ids'][index]))
            attackers = np.zeros(ENTITY_SLOTS, dtype=np.float32)
            attackers[event['attacker_ids']] = 1
            np.testing.assert_array_equal(attackers, payload['attack_flags'][index])

    def test_independent_branches_never_splice_or_correct_model_state_or_memory(self):
        model = RecordingModel()
        with TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            report = collect_corrections(model, temporary, steps=5, every=2,
                                          horizon=2, scenario='explore', max_branches=3)
            self.assertEqual(report['valid_branches'], 3)
            self.assertEqual(report['neural_steps'], 5)
            self.assertFalse(report['oracle_feedback_to_model_state_or_memory'])
            self.assertGreater(report['canonicalization_max_abs_diff'], 0)
            paths = [Path(path) for path in report['files']]
            self.assertEqual(len(set(paths)), 3)
            for index, path in enumerate(paths):
                with np.load(path, allow_pickle=False) as branch:
                    length = len(branch['actions'])
                    self.assertEqual(len(branch['states']), length+1)
                    self.assertTrue(bool(branch['independent_branch']))
                    self.assertFalse(bool(branch['model_trajectory_corrected']))
                    self.assertEqual(int(branch['student_step']), index*2)
                    self.assertNotIn('frames', branch.files)
                    # The initial model state is independently recorded per
                    # branch; canonicalization changes only the source query.
                    np.testing.assert_array_equal(branch['student_state'], model.inputs[index*2][0].numpy())
            for index, memory in enumerate(model.memories):
                self.assertEqual(float(memory[0, 0]), index)
            for state in model.inputs[1:]:
                self.assertAlmostEqual(float(state[0, 5]), .5003, places=6)
            first_bytes = paths[0].read_bytes()
            with self.assertRaises(FileExistsError):
                collect_corrections(RecordingModel(), temporary, steps=1)
            self.assertEqual(paths[0].read_bytes(), first_bytes)

    def test_nonfinite_final_model_prediction_is_reported_without_querying_it(self):
        model = RecordingModel(nonfinite=True)
        with TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            report = collect_corrections(model, temporary, steps=1, every=1,
                                          horizon=1, max_branches=1)
        self.assertEqual(report['neural_steps'], 1)
        self.assertEqual(report['valid_branches'], 1)
        self.assertEqual(report['rejected_nonfinite'], 1)
        self.assertEqual(report['neural_episodes_completed'], 0)

    def test_terminal_branch_is_not_written_as_empty_training_sequence(self):
        game = Game()
        state = game.state_vector()
        state[4] = 0
        payload, audit = query_branch(state, game.map, np.zeros((3, 4), dtype=np.float32))
        self.assertIsNone(payload)
        self.assertEqual(audit['reason'], 'terminal')


if __name__ == '__main__':
    unittest.main()
