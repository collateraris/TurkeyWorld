"""An immutable450-step opt-in preserves clocks and survives exact resumes."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

import test_component_transition as transition_fixture
from turkeyworld.component_curriculum import ComponentCurriculum
from turkeyworld.long_rollout_protocol import bind_long_rollout
from turkeyworld.navigation_experiment import experiment_fingerprint
from turkeyworld.navigation_sampling import NAVIGATION_CASES


class LongRolloutProtocolTests(unittest.TestCase):
    def setUp(self):
        fixture = transition_fixture.ComponentTransitionTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.components = fixture.controller
        self.components.apply_transition_request(fixture.request, fixture.step)
        fixture.fixture.advance(self.components, 76)
        self.step = 1520
        self.components.phase.update(gate={'status': 'failed', 'step': 1500,
            'quality_passed': False, 'metrics': {'trajectory_horizon': 29}},
            pass_streak=1, best_score=.25, checks_without_improvement=3,
            validation_by_case={'short_return': {'mean_endpoint_xy_tiles': .9}})
        self.components.dynamics_sampling().update(draws=1216,
            sample_counts=dict(zip(NAVIGATION_CASES, (365, 304, 243, 182, 61, 61))))
        self.components.sync()
        self.root = fixture.archive.parent/'long_data'
        self.records = []
        for split in ('train', 'val'):
            path = self.root/'rgb'/split/'episode.npz'
            path.parent.mkdir(parents=True)
            path.write_bytes(('fake NPZ bytes: '+split).encode())
            self.records.append({'relative_path': path.relative_to(self.root).as_posix(),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        self.manifest = {'status': 'complete', 'config': {'fingerprint': 'fixed-data-config',
            'environment_profile': 'navigation_no_enemies_v1', 'enemies_enabled': False,
            'train_clips': 1, 'val_clips': 1}, 'records': self.records}
        self.manifest_path = self.root/'manifest.json'
        self.manifest_path.write_text(json.dumps(self.manifest), encoding='utf-8')
        self.plan = {'version': 1, 'horizon': 450, 'simulation_hz': 30, 'tbptt_chunk': 16,
            'microbatch_size': 16, 'component_plan_fingerprint': self.components.state['fingerprint'],
            'environment_profile': 'navigation_no_enemies_v1', 'joint_start_horizon': 450,
            'start_optimizer_step': self.step, 'dataset_directory': str(self.root),
            'dataset_config_fingerprint': 'fixed-data-config',
            'dataset_content_fingerprint': experiment_fingerprint(sorted(
                (item['relative_path'], item['sha256']) for item in self.records))}
        self.path = fixture.archive.parent/'long_protocol.json'
        self.write_plan()

    def write_plan(self):
        self.path.write_text(json.dumps(self.plan), encoding='utf-8')

    def test_new_binding_resets_only_validation_baselines_and_preserves_original_plan_and_clocks(self):
        original_plan = self.fixture.fixture.path.read_bytes()
        original_state = copy.deepcopy(self.components.state)
        original_sampling = copy.deepcopy(self.components.dynamics_sampling())
        phase_updates = self.components.phase['updates']
        state = bind_long_rollout(self.components, self.path, self.step)
        self.assertEqual(state['horizon'], 450)
        self.assertEqual(state['short_update_count'], 76)
        self.assertEqual(state['long_update_count'], 0)
        self.assertEqual(state['committed_long_update_count'], 0)
        self.assertEqual(state['optimizer_updates_per_rollout'], 1)
        self.assertEqual(state['numerical_state_resets'], 0)
        self.assertFalse(state['source_future_feedback'])
        self.assertEqual(self.components.state['last_optimizer_step'], self.step)
        self.assertEqual(self.components.phase['updates'], phase_updates)
        self.assertEqual(self.components.dynamics_sampling(), original_sampling)
        self.assertEqual(self.components.state['history'], original_state['history'])
        self.assertEqual(self.components.state['applied_transition_requests'], original_state['applied_transition_requests'])
        self.assertEqual(self.components.phase['validation_protocol_history'][-1], original_state['phases'][1]['gate'])
        self.assertEqual(self.components.phase['gate'], {'status': 'pending', 'horizon': 450})
        self.assertEqual(self.components.phase['pass_streak'], 0)
        self.assertIsNone(self.components.phase['best_score'])
        self.assertEqual(self.components.phase['checks_without_improvement'], 0)
        self.assertEqual(self.components.phase['validation_by_case'], {})
        self.assertEqual(self.fixture.fixture.path.read_bytes(), original_plan)

    def test_checkpoint_resume_preserves_long_progress_and_does_not_reset_gate_again(self):
        state = bind_long_rollout(self.components, self.path, self.step)
        state.update(long_update_count=1, committed_long_update_count=1,
                     sampling={'draws': 16, 'sample_counts': {'slide': 5}})
        self.fixture.fixture.advance(self.components, 1)
        self.components.phase.update(gate={'status': 'failed', 'step': self.step+1, 'horizon': 450},
                                     checks_without_improvement=1)
        self.components.sync()
        saved = json.loads(json.dumps(self.components.state))
        resumed = ComponentCurriculum(self.fixture.fixture.path, step=self.step+1,
            base_plan_fingerprint=self.fixture.fixture.base_fingerprint, previous=saved)
        before = copy.deepcopy(resumed.state)
        result = bind_long_rollout(resumed, self.path, self.step+1)
        self.assertEqual(resumed.state, before)
        self.assertEqual(result['sampling']['draws'], 16)
        self.assertEqual(result['long_update_count'], 1)
        self.assertEqual(len(resumed.phase['validation_protocol_history']), 1)

    def test_new_binding_requires_the_current_completed_dynamics_boundary(self):
        before = copy.deepcopy(self.components.state)
        with self.assertRaises(ValueError):
            bind_long_rollout(self.components, self.path, self.step+1)
        self.assertEqual(self.components.state, before)
        # Matching the request argument is insufficient when the controller
        # itself is still at a different optimizer boundary.
        self.components.state['last_optimizer_step'] = self.step-1
        with self.assertRaises(ValueError):
            bind_long_rollout(self.components, self.path, self.step)

    def test_renderer_phase_and_mismatched_component_plan_are_rejected(self):
        renderer = self.fixture.fixture.curriculum()
        self.plan.update(start_optimizer_step=777)
        self.write_plan()
        with self.assertRaisesRegex(ValueError, 'dynamics boundary'):
            bind_long_rollout(renderer, self.path, 777)
        self.plan.update(start_optimizer_step=self.step, component_plan_fingerprint='other-plan')
        self.write_plan()
        with self.assertRaisesRegex(ValueError, 'protocol'):
            bind_long_rollout(self.components, self.path, self.step)

    def test_changed_source_bytes_or_config_fail_closed_without_mutating_controller(self):
        before = copy.deepcopy(self.components.state)
        source = self.root/self.records[0]['relative_path']
        source.write_bytes(b'changed source bytes')
        with self.assertRaisesRegex(ValueError, 'content changed'):
            bind_long_rollout(self.components, self.path, self.step)
        self.assertEqual(self.components.state, before)
        source.write_bytes(b'fake NPZ bytes: train')
        self.manifest['config']['fingerprint'] = 'changed-data-config'
        self.manifest_path.write_text(json.dumps(self.manifest), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'different profile'):
            bind_long_rollout(self.components, self.path, self.step)
        self.assertEqual(self.components.state, before)

    def test_changed_request_on_resume_and_missing_required_request_are_rejected(self):
        bind_long_rollout(self.components, self.path, self.step)
        before = copy.deepcopy(self.components.state)
        self.plan['reason'] = 'changed immutable request'
        self.write_plan()
        with self.assertRaisesRegex(ValueError, 'changed on resume'):
            bind_long_rollout(self.components, self.path, self.step)
        self.assertEqual(self.components.state, before)
        self.path.unlink()
        with self.assertRaisesRegex(ValueError, 'immutable protocol file'):
            bind_long_rollout(self.components, self.path, self.step)
        self.assertEqual(self.components.state, before)

    def test_absent_opt_in_is_a_noop_and_source_escape_is_rejected(self):
        self.path.unlink()
        before = copy.deepcopy(self.components.state)
        self.assertIsNone(bind_long_rollout(self.components, self.path, self.step))
        self.assertEqual(self.components.state, before)
        self.write_plan()
        outside = self.root.parent/'escaped.npz'; outside.write_bytes(b'outside')
        self.manifest['records'][0] = {'relative_path': '../escaped.npz',
            'sha256': hashlib.sha256(outside.read_bytes()).hexdigest()}
        self.manifest_path.write_text(json.dumps(self.manifest), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'escapes'):
            bind_long_rollout(self.components, self.path, self.step)


if __name__ == '__main__':
    unittest.main()
