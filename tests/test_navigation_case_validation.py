"""Per-case player navigation metrics reuse saved trajectories without a model."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from turkeyworld.engine import BEHAVIOR_VERSION, Game
from turkeyworld.navigation_case_validation import summarize_navigation_cases
from turkeyworld.navigation_sampling import NAVIGATION_CASES
from turkeyworld.structured_state import STATE_DIM, STATE_SCHEMA_VERSION


class NavigationCaseValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root/'navigation'/'rgb'/'val'
        self.directory.mkdir(parents=True)
        self.trajectory = self.root/'trajectory'; self.trajectory.mkdir()
        self.paths, self.windows = [], []
        self.source = np.zeros((12, 5, STATE_DIM), np.float32)
        self.source[:, :, :2], self.source[:, :, 3:6] = .5, 1.
        self.predicted = self.source.copy()
        self.maps = np.zeros((12, 16, 16), np.int64)
        self.maps[:, 5, 5] = 1
        self.actions = np.zeros((12, 4, 4), np.float32)
        self.indices = np.asarray([[row, 0] for row in range(12)], dtype=np.int64)
        self.dt = np.full(12, 1/30, np.float32)
        for row in range(12):
            path = self.directory/f'episode_{row:05d}.npz'
            states = np.concatenate([self.source[row, :1], self.source[row]], axis=0)
            np.savez_compressed(path, states=states, actions=np.zeros((5, 4), np.float32),
                map=self.maps[row], dt=self.dt[row], split='val', stage='navigation',
                movement_case=NAVIGATION_CASES[row//2], dataset_kind='world_transitions',
                world_transition_compatible=True, behavior_version=BEHAVIOR_VERSION,
                state_dim=STATE_DIM, entity_slots=1, state_schema_version=STATE_SCHEMA_VERSION)
            self.paths.append(path)
            self.windows.append({'episode': path.name, 'episode_index': row, 'start': 0})
        self.write_artifacts()

    def write_artifacts(self):
        identities = [{'episode': p.name, 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                      for p in self.paths]
        self.panel = {'split': 'val', 'context': 2, 'horizon': 4, 'windows': self.windows,
            'episode_identity_sha256': hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()}
        self.report = {'panel': self.panel, 'fingerprint': hashlib.sha256(
            json.dumps(self.panel, sort_keys=True).encode()).hexdigest(), 'count': 12, 'horizon': 4,
            'context': 2, 'source_held_out': True, 'source_future_feedback': False, 'future_rgb_encoded': False}
        self.save_report()
        self.save_arrays()

    def save_report(self):
        (self.trajectory/'heldout_trajectory_validation.json').write_text(json.dumps(self.report), encoding='utf-8')

    def save_arrays(self):
        np.savez_compressed(self.trajectory/'heldout_trajectory_states.npz', predicted=self.predicted,
            reference=self.source, actions=self.actions, maps=self.maps, indices=self.indices, dt=self.dt)

    def summarize(self):
        return summarize_navigation_cases(self.trajectory, self.root/'navigation')

    def rewrite_source(self, row, **changes):
        path = self.paths[row]
        with np.load(path, allow_pickle=False) as saved:
            values = {name: saved[name] for name in saved.files}
        values.update(changes)
        np.savez_compressed(path, **values)

    def test_endpoint_and_all_step_statistics_have_distinct_correct_denominators(self):
        self.predicted[0, 1:, 0] += np.arange(1, 5)*.1/16
        self.predicted[1, 1:, 0] += np.arange(1, 5)*.2/16
        self.save_arrays()
        result = self.summarize()
        slide = result['slide']
        self.assertEqual(set(result), set(NAVIGATION_CASES))
        self.assertEqual((slide['windows'], slide['count'], slide['valid_endpoint_count'], slide['valid_steps']), (2, 2, 2, 8))
        self.assertAlmostEqual(slide['mean_endpoint_xy_tiles'], .6, places=5)
        self.assertAlmostEqual(slide['p95_endpoint_xy_tiles'], .78, places=5)
        self.assertAlmostEqual(slide['max_endpoint_xy_tiles'], .8, places=5)
        self.assertAlmostEqual(slide['mean_all_steps_xy_tiles'], .375, places=5)
        self.assertEqual(result['short_return']['mean_endpoint_xy_tiles'], 0.)
        json.dumps(result, allow_nan=False)

    def test_player_walls_are_counted_and_npc_walls_do_not_affect_them(self):
        self.predicted[:, 1:, 8:10] = 5.5/16
        self.predicted[:, 1:, 12:14] = [.6, 1.]
        self.predicted[0, 1, :2] = 5.5/16
        self.save_arrays()
        result = self.summarize()
        self.assertEqual(result['slide']['wall_collision_windows'], 1)
        self.assertEqual(result['slide']['wall_collision_steps'], 1)
        self.assertEqual(result['slide']['per_window'][0]['first_player_wall_offset'], 1)
        self.assertAlmostEqual(result['slide']['per_window'][0]['first_player_wall_seconds'], 1/30)
        self.assertGreaterEqual(result['slide']['wall_path_collision_steps'], 1)
        for case in NAVIGATION_CASES[1:]:
            self.assertEqual(result[case]['wall_collision_steps'], 0)

    def test_source_alive_masks_only_and_predicted_death_cannot_hide_error(self):
        self.source[0, 1:, 4] = 0
        source_states = np.concatenate([self.source[0, :1], self.source[0]], axis=0)
        self.rewrite_source(0, states=source_states)
        self.predicted[0, 1:, :2] = 5.5/16
        self.predicted[1, 1:, 0] += 1/16
        self.predicted[1, 1:, 4] = 0
        self.write_artifacts()
        slide = self.summarize()['slide']
        self.assertEqual((slide['valid_steps'], slide['valid_endpoint_count']), (4, 1))
        self.assertEqual(slide['mean_endpoint_xy_tiles'], 1.)
        self.assertEqual(slide['wall_collision_steps'], 0)

    def test_no_engine_rgb_or_model_execution_and_no_source_changes(self):
        before = {p: p.read_bytes() for p in self.paths}
        with patch.object(Game, '__init__', side_effect=AssertionError('No engine')), \
                patch.object(Game, 'render', side_effect=AssertionError('No RGB')), \
                patch.object(Game, 'step', side_effect=AssertionError('No Source rollout')):
            result = self.summarize()
        self.assertEqual(result['short_return']['windows'], 2)
        self.assertTrue(all(p.read_bytes() == content for p, content in before.items()))

    def test_train_split_and_changed_panel_or_indices_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'held-out val'):
            summarize_navigation_cases(self.trajectory, self.directory.parent/'train')
        self.report['panel']['split'] = 'train'; self.save_report()
        with self.assertRaisesRegex(ValueError, 'held-out val'):
            self.summarize()
        self.write_artifacts()
        self.indices[0, 0] = 1; self.save_arrays()
        with self.assertRaisesRegex(ValueError, 'episode/start identities'):
            self.summarize()

    def test_source_stage_provenance_and_reference_tampering_fail_closed(self):
        self.rewrite_source(0, stage='moving1')
        with self.assertRaisesRegex(ValueError, 'provenance'):
            self.summarize()
        self.rewrite_source(0, stage='navigation')
        self.write_artifacts()
        self.source[0, 2, 0] += .01; self.save_arrays()
        with self.assertRaisesRegex(ValueError, 'exact Source window'):
            self.summarize()

    def test_nonfinite_predictions_and_source_replacement_are_not_silently_dropped(self):
        self.predicted[0, 1, 0] = np.nan; self.save_arrays()
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.summarize()
        self.predicted = self.source.copy(); self.save_arrays()
        # Same arrays, but a changed file identity still invalidates the report.
        self.rewrite_source(0, unrelated_metadata='modified')
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            self.summarize()


if __name__ == '__main__':
    unittest.main()
