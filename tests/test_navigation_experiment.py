"""A temporary sampling plan must not reset weights, stages or sampler clocks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from turkeyworld.navigation_experiment import (bind_navigation_experiment, navigation_focus_active,
    record_navigation_case_validation, supersede_navigation_focus)
from turkeyworld.navigation_sampling import NAVIGATION_CASES, NAVIGATION_FOCUS_WEIGHTS
from turkeyworld.staged_curriculum import STAGES, StagedCurriculum, StagedTrainingPools


class NavigationExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan_path, self.request = self.root/'plan.json', self.root/'focus.json'
        plan = {'version': 1, 'target_steps': 450, 'validation_every': 100,
            'replay_fraction': .25, 'required_passes': 2, 'data_fingerprint': 'frozen-data',
            'datasets': {stage['id']: {'directory': str(self.root/stage['id'])} for stage in STAGES}}
        self.plan_path.write_text(json.dumps(plan))
        self.curriculum = StagedCurriculum(self.plan_path, start_step=0)
        self.config = {'version': 1, 'id': 'focus-test', 'stage': 'navigation',
            'start_optimizer_step': 625, 'stop_optimizer_step': 925, 'microbatch_size': 16,
            'weights': NAVIGATION_FOCUS_WEIGHTS, 'source_fingerprint': 'frozen-case-windows',
            'source_dataset_fingerprint': 'frozen-data', 'base_plan_fingerprint': self.curriculum.fingerprint,
            'selection': 'source_case_future_windows', 'after_completion': 'original_motion_curriculum'}
        self.request.write_text(json.dumps(self.config))

    def bind(self):
        return bind_navigation_experiment(self.curriculum, self.request, 625, 16)

    def test_new_request_binds_once_without_changing_base_plan_or_training_clock(self):
        original = self.curriculum.fingerprint
        state = self.bind()
        self.assertEqual(state['updates_completed'], 0)
        self.assertEqual(state['remaining_updates'], 300)
        self.assertEqual(state['draws'], 0)
        self.assertEqual(self.curriculum.fingerprint, original)
        self.assertEqual(self.curriculum.state['stage_start_step'], 0)
        resumed = StagedCurriculum(self.plan_path, start_step=625, previous=copy.deepcopy(self.curriculum.state))
        bind_navigation_experiment(resumed, self.request, 625, 16)
        self.assertEqual(resumed.state['sampling_experiment']['start_optimizer_step'], 625)
        with self.assertRaisesRegex(ValueError, 'start'):
            bind_navigation_experiment(StagedCurriculum(self.plan_path, start_step=0), self.request, 626, 16)

    def test_changed_request_missing_file_and_forged_checkpoint_counts_are_rejected(self):
        state = self.bind()
        self.config['stop_optimizer_step'] = 926
        self.request.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, 'changed'):
            bind_navigation_experiment(self.curriculum, self.request, 625, 16)
        self.request.unlink()
        with self.assertRaisesRegex(ValueError, 'requires'):
            bind_navigation_experiment(self.curriculum, self.request, 625, 16)
        self.config['stop_optimizer_step'] = 925
        self.request.write_text(json.dumps(self.config))
        state['draws'] = 16  # The saved optimizer clock is still625.
        with self.assertRaisesRegex(ValueError, 'counters'):
            bind_navigation_experiment(self.curriculum, self.request, 625, 16)

    def test_case_counts_are_committed_only_after_one_real_update(self):
        state = self.bind()
        pools = StagedTrainingPools(self.curriculum, None)
        class SourcePool:
            def batch(self, count, **options):
                return {'selected': options['indices']}
        class Selector:
            fingerprint = 'frozen-case-windows'
            def sample(self, count, rng, *, counts, draws):
                updated = dict(counts)
                delta = dict(zip(NAVIGATION_CASES, (5, 4, 3, 2, 1, 1)))
                for name, n in delta.items(): updated[name] += n
                return np.zeros((count, 2), np.int64), updated, draws+count, delta
        pools.pools['navigation'] = SourcePool()
        pools.focus_sampler = Selector()
        pools.batch(16, 625, 2, np.random.default_rng(173))
        self.assertEqual(state['draws'], 0)
        self.assertEqual(sum(state['sample_counts'].values()), 0)
        with self.assertRaisesRegex(RuntimeError, 'committed'):
            pools.batch(16, 625, 2, np.random.default_rng(173))
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            pools.commit_sampling(627)
        self.assertFalse(pools.commit_sampling(626))
        self.assertEqual(state['draws'], 16)
        self.assertEqual(state['updates_completed'], 1)
        bind_navigation_experiment(self.curriculum, self.request, 626, 16)

    def test_quality_passes_do_not_end_the300_update_navigation_segment_early(self):
        state = self.bind()
        for step in (625, 700, 800, 900):
            advanced = self.curriculum.record_validation({'passed': True}, step)
            self.assertFalse(advanced)
            self.assertEqual(self.curriculum.stage['id'], 'navigation')
            self.assertEqual(self.curriculum.state['gate']['status'], 'passed')
            self.assertEqual(self.curriculum.state['gate']['promotion_hold'], 'navigation_sampling_experiment')
        state['draws'] = 4800
        state['sample_counts'] = {case: int(4800*weight) for case, weight in NAVIGATION_FOCUS_WEIGHTS.items()}
        self.assertTrue(self.curriculum.record_validation({'passed': True}, 925))
        self.assertEqual(self.curriculum.stage['id'], 'static_enemy')
        self.assertEqual(state['status'], 'complete')
        self.assertEqual(state['remaining_updates'], 0)

    def test_half_open_boundary_and_fallback_never_resample_the_focus_after300(self):
        state = self.bind()
        self.assertTrue(navigation_focus_active(self.curriculum, 625))
        self.assertTrue(navigation_focus_active(self.curriculum, 924))
        self.assertFalse(navigation_focus_active(self.curriculum, 925))
        self.assertEqual(state['status'], 'complete')
        pools = StagedTrainingPools(self.curriculum, None)
        expected = {'original': np.zeros((16, 2))}
        with patch.object(pools, '_dynamic_batch', return_value=expected) as ordinary,\
                patch.object(pools, '_focused_batch', side_effect=AssertionError('Expired focus')):
            self.assertIs(pools.batch(16, 925, 2, np.random.default_rng(173)), expected)
        self.assertEqual(ordinary.call_args.args[:3], ('navigation', 16, 29))

    def test_new_experiment_uses_current_data_identity_and_batch(self):
        self.config['source_dataset_fingerprint'] = 'other-data'
        self.request.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, 'data'):
            self.bind()
        self.config['source_dataset_fingerprint'] = 'frozen-data'
        self.request.write_text(json.dumps(self.config))
        with self.assertRaises(ValueError):
            bind_navigation_experiment(self.curriculum, self.request, 625, 27)

    def test_saved_baseline_recovery_cannot_replace_newer_validation_or_sampling_counts(self):
        state = self.bind()
        baseline = {'slide': {'windows': 5, 'mean_endpoint_xy_tiles': .2}}
        latest = {'slide': {'windows': 5, 'mean_endpoint_xy_tiles': .1}}
        record_navigation_case_validation(self.curriculum, 700, latest)
        record_navigation_case_validation(self.curriculum, 625, baseline)
        self.assertEqual(state['baseline_metrics_by_case'], baseline)
        self.assertEqual(state['last_validation_by_case'], latest)
        self.assertEqual(state['last_validation_step'], 700)
        self.assertEqual(state['draws'], 0)
        record_navigation_case_validation(self.curriculum, 625, latest)
        self.assertEqual(state['baseline_metrics_by_case'], baseline)

    def test_explicit_replacement_stops_old_focus_clock_without_forging_completion(self):
        state = self.bind()
        state['draws'] = 2432
        state['sample_counts']['slide'] = 2432
        supersede_navigation_focus(self.curriculum, 777, 'component-plan')
        self.assertEqual(state['status'], 'superseded')
        self.assertEqual(state['updates_completed'], 152)
        self.assertEqual(state['remaining_updates'], 148)
        self.assertFalse(navigation_focus_active(self.curriculum, 1500))
        bind_navigation_experiment(self.curriculum, self.request, 1500, 16)
        self.assertEqual(state['updates_completed'], 152)
        self.assertEqual(state['draws'], 2432)


if __name__ == '__main__':
    unittest.main()
