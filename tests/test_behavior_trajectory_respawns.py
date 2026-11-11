import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from turkeyworld.behavior_trajectory import evaluate_state_trajectories
from turkeyworld.engine import STATE_DIM


class _Data:
    split = 'val'

    def __init__(self, directory, source):
        path = Path(directory)/'episode.npz'
        if not path.exists():
            np.savez(path, source=source)
        self.source = torch.from_numpy(source)[None]
        self.episodes = [dict(path=path, frames=np.zeros((len(source), 2, 2, 3), np.uint8),
            states=source, actions=np.zeros((len(source)-1, 4), np.float32),
            map=np.pad(np.zeros((14, 14), np.int8), 1, constant_values=1))]

    def batch(self, batch_size, *, context, horizon, indices):
        assert batch_size == 1 and context == 1 and len(indices) == 1
        return dict(states=self.source.clone(), actions=torch.zeros(1, horizon, 4),
            context_frames=torch.zeros(1, 1, 3, 2, 2),
            map=torch.from_numpy(self.episodes[0]['map'])[None], dt=torch.tensor([1/30]))


class _Model(nn.Module):
    def __init__(self, predicted):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.register_buffer('predictions', torch.from_numpy(predicted))
        self.config = SimpleNamespace(simulation_hz=30)
        self.calls = 0

    def encode_map(self, grid):
        return grid

    def initialize_from_image(self, rgb, state, fmap):
        assert rgb.shape[1] == state.shape[1] == 1
        return torch.zeros(len(state), 1)

    def transition(self, state, action, fmap, memory):
        index = int(memory[0, 0])
        self.calls += 1
        return dict(state=self.predictions[index+1][None].clone(), memory=memory+1)


class RespawnTrajectoryTests(unittest.TestCase):
    def _paths(self, initially_alive=True):
        source = np.zeros((5, STATE_DIM), np.float32)
        source[:, :2] = 2.5/16
        source[:, 3:6] = 1
        source[:, 8] = 5.5/16
        source[:, 9] = 2.5/16
        alive = [1, 0, 1, 1, 1] if initially_alive else [0, 1, 1, 1, 1]
        source[:, 12] = np.asarray(alive)*.6
        source[:, 13] = alive
        spawn = 2 if initially_alive else 1
        source[spawn:, 8] = 9.5/16
        predicted = source.copy()
        predicted[spawn:, 8] += 1/16
        # Missing predicted actors remain in the primary Source-alive metric.
        predicted[spawn+1, 12:14] = 0
        return source, predicted, spawn

    def _evaluate(self, directory, source, predicted, **kwargs):
        model = _Model(predicted)
        model.train()
        report = evaluate_state_trajectories(model, _Data(directory, source),
            Path(directory)/'report', context=1, horizon=4, count=1, batch_size=1, **kwargs)
        self.assertTrue(model.training)
        self.assertEqual(model.calls, 4)
        self.assertTrue(report['all_states_finite'])
        self.assertFalse(report['source_future_feedback'])
        return report

    def test_default_mask_and_fingerprint_preserve_legacy_protocol(self):
        source, predicted, _ = self._paths()
        with TemporaryDirectory() as directory:
            default = self._evaluate(directory, source, predicted)
            explicit = self._evaluate(directory, source, predicted, include_respawned_slots=False)
            self.assertEqual(default['fingerprint'], explicit['fingerprint'])
            self.assertEqual(default['panel']['version'], 'heldout_free_state_v1')
            self.assertEqual(default['panel']['source_content_sha256'], explicit['panel']['source_content_sha256'])
            self.assertFalse(default['include_respawned_slots'])
            self.assertEqual(default['npc_geometry_protocol_version'], 1)
            for offset in ('2', '3', '4'):
                self.assertEqual(default['offsets'][offset]['npc_xy_error_tiles']['n'], 0)
                self.assertEqual(default['offsets'][offset]['source_respawned_slots_excluded'], 1)

    def test_new_life_geometry_counts_error_and_false_missing_but_not_teleport(self):
        source, predicted, spawn = self._paths()
        with TemporaryDirectory() as directory:
            report = self._evaluate(directory, source, predicted, include_respawned_slots=True)
            self.assertEqual(report['panel']['version'], 'heldout_free_state_v2_respawn')
            self.assertEqual(report['npc_geometry_protocol_version'], 2)
            self.assertEqual(report['offsets'][str(spawn)]['npc_xy_error_tiles']['n'], 0)
            after = report['offsets'][str(spawn+1)]
            self.assertEqual(after['npc_xy_error_tiles']['n'], 1)
            self.assertEqual(after['npc_xy_error_tiles']['mean'], 1.)
            self.assertEqual(after['npc_common_alive_xy_error_tiles']['n'], 0)
            self.assertEqual(after['npc_alive_false_negative'], 1)
            self.assertEqual(after['npc_displacement_error_tiles']['mean'], 0.)
            self.assertEqual(after['source_respawned_slots_excluded'], 0)
            saved = json.loads((Path(directory)/'report/heldout_trajectory_validation.json').read_text())
            self.assertTrue(saved['include_respawned_slots'])

    def test_slot_dead_at_context_is_measured_after_first_respawn(self):
        source, predicted, spawn = self._paths(initially_alive=False)
        with TemporaryDirectory() as directory:
            report = self._evaluate(directory, source, predicted, include_respawned_slots=True)
            self.assertEqual(report['offsets'][str(spawn)]['npc_velocity_error_tiles_s']['n'], 0)
            self.assertEqual(report['offsets'][str(spawn+1)]['npc_xy_error_tiles']['mean'], 1.)


if __name__ == '__main__':
    unittest.main()
