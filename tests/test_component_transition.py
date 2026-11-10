"""Explicit user handoffs preserve failed gates, saved boundaries and resume."""
import copy
import hashlib
import json
import unittest

import test_component_curriculum as component_fixture
from turkeyworld.component_curriculum import ComponentCurriculum
from turkeyworld.navigation_sampling import NAVIGATION_CASES, NAVIGATION_FOCUS_WEIGHTS


class ComponentTransitionTests(unittest.TestCase):
    def setUp(self):
        fixture = component_fixture.ComponentCurriculumTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.controller = fixture.curriculum()
        fixture.advance(self.controller, 667)
        self.step = 1444
        self.controller.record_validation(fixture.renderer_report(good=False), self.step)
        self.archive = fixture.path.parent/'renderer_full_1444.pt'
        self.archive.write_bytes(b'immutable full renderer checkpoint including D/Adam/EMA/RNG')
        self.request = {'version': 1, 'id': 'user_renderer_to_dynamics_1444',
            'component_plan_fingerprint': self.controller.state['fingerprint'],
            'from_phase': 'renderer', 'to_phase': 'dynamics', 'optimizer_step': self.step,
            'reason': 'user_requested', 'saved_checkpoint': str(self.archive.resolve()),
            'saved_checkpoint_sha256': hashlib.sha256(self.archive.read_bytes()).hexdigest()}

    def restore(self, state, step):
        return ComponentCurriculum(self.fixture.path, step=step,
            base_plan_fingerprint=self.fixture.base_fingerprint, previous=state)

    def test_user_transition_keeps_failed_quality_and_exact_original_update_clock(self):
        old = copy.deepcopy(self.controller.state)
        gate = copy.deepcopy(self.controller.phase['gate'])
        archive = self.archive.read_bytes()
        self.assertTrue(self.controller.apply_transition_request(self.request, self.step))
        state = self.controller.state
        self.assertEqual(self.controller.name, 'dynamics')
        self.assertEqual(state['last_optimizer_step'], self.step)
        self.assertEqual(state['start_optimizer_step'], 777)
        self.assertEqual((state['phase_start_step'], state['phase_updates']), (self.step, 0))
        self.assertFalse(state['gameplay_qualified'])
        completed = state['history'][0]
        self.assertEqual(completed['completion_reason'], 'user_requested')
        self.assertEqual(completed['completed_step'], self.step)
        self.assertEqual(completed['updates'], 667)
        self.assertEqual(completed['gate'], gate)
        self.assertEqual(completed['gate']['status'], 'failed')
        self.assertFalse(completed['transition_quality_passed'])
        self.assertEqual(completed['saved_checkpoint'], str(self.archive.resolve()))
        self.assertEqual(completed['saved_checkpoint_sha256'], self.request['saved_checkpoint_sha256'])
        self.assertEqual(self.controller.phase['gate']['status'], 'pending')
        self.assertEqual(self.controller.phase['losses'], {})
        self.assertEqual(self.archive.read_bytes(), archive)
        self.assertEqual(self.controller.plan, self.fixture.plan)
        # Journal snapshots are independent of the live completed phase record.
        self.controller.state['phases'][0]['losses']['changed_later'] = 1.
        self.assertNotIn('changed_later', completed['losses'])
        self.assertEqual(old['phase_updates'], 667)

    def test_new_dynamics_sampling_starts_at_zero_without_touching_original_focus(self):
        checkpoint = {'training_steps': self.step, 'global_training_step': self.step,
            'optimizer': {'state': {'core': {'step': 777, 'exp_avg': [1., 2.]},
                                    'flow': {'step': self.step, 'exp_avg': [3., 4.]}}},
            'ema_target_encoder': {'steps': 777, 'weights': [5., 6.]},
            'adversarial_training': {'updates': self.step, 'optimizer_step': self.step},
            'rng_numpy': {'state': 12345}, 'rng_torch': [3, 2, 1],
            'training_counters': {'optimizer_updates': self.step, 'fragments_seen': self.step*16},
            'staged_curriculum': {'stage_excluded_updates': 667,
                'sampling_experiment': {'updates_completed': 152, 'draws': 2432, 'status': 'superseded'}},
            'component_training': copy.deepcopy(self.controller.state)}
        outside = copy.deepcopy({k: v for k, v in checkpoint.items() if k != 'component_training'})
        restored = self.restore(checkpoint['component_training'], self.step)
        restored.apply_transition_request(self.request, self.step)
        checkpoint['component_training'] = copy.deepcopy(restored.state)
        self.assertEqual({k: v for k, v in checkpoint.items() if k != 'component_training'}, outside)
        sampling = restored.dynamics_sampling()
        self.assertEqual(sampling['draws'], 0)
        self.assertEqual(sampling['sample_counts'], dict.fromkeys(NAVIGATION_CASES, 0))
        self.assertEqual(sampling['weights'], NAVIGATION_FOCUS_WEIGHTS)
        self.assertEqual(sampling['source_fingerprint'], self.fixture.plan['navigation_source_fingerprint'])
        journal = checkpoint['component_training']['applied_transition_requests']
        self.assertEqual(len(journal), 1)
        self.assertEqual((journal[0]['id'], journal[0]['step']), (self.request['id'], self.step))

    def test_consumed_request_is_idempotent_across_checkpoint_resume_and_later_updates(self):
        self.controller.apply_transition_request(self.request, self.step)
        saved = json.loads(json.dumps(self.controller.state))
        resumed = self.restore(saved, self.step)
        before = copy.deepcopy(resumed.state)
        self.assertFalse(resumed.apply_transition_request(self.request, self.step))
        self.assertEqual(resumed.state, before)
        self.fixture.advance(resumed, 3)
        resumed.dynamics_sampling().update(draws=48,
            sample_counts=dict(zip(NAVIGATION_CASES, (14, 12, 10, 7, 3, 2))))
        later = self.restore(json.loads(json.dumps(resumed.state)), self.step+3)
        before = copy.deepcopy(later.state)
        self.assertFalse(later.apply_transition_request(self.request, self.step+3))
        self.assertEqual(later.state, before)
        self.assertEqual(later.state['phase_updates'], 3)
        self.assertEqual(later.dynamics_sampling()['draws'], 48)

    def test_same_consumed_id_with_changed_request_identity_is_rejected(self):
        self.controller.apply_transition_request(self.request, self.step)
        before = copy.deepcopy(self.controller.state)
        for changes in ({'saved_checkpoint_sha256': 'a'*64}, {'optimizer_step': self.step+1},
                        {'to_phase': 'joint'}, {'saved_checkpoint': str(self.archive)+'.other'}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'request changed'):
                self.controller.apply_transition_request({**self.request, **changes}, self.step)
            self.assertEqual(self.controller.state, before)

    def test_wrong_boundary_phase_skip_and_archive_identity_format_fail_before_state_changes(self):
        changes = ({'optimizer_step': self.step-1}, {'from_phase': 'dynamics'}, {'to_phase': 'joint'},
            {'component_plan_fingerprint': 'changed-plan'}, {'reason': 'quality'},
            {'saved_checkpoint': ''}, {'saved_checkpoint_sha256': 'short'},
            {'saved_checkpoint_sha256': 'x'*64}, {'id': ''}, {'version': 2})
        before = copy.deepcopy(self.controller.state)
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.controller.apply_transition_request({**self.request, **change}, self.step)
            self.assertEqual(self.controller.state, before)
        with self.assertRaises(ValueError):
            self.controller.apply_transition_request(self.request, self.step+1)
        self.assertEqual(self.controller.state, before)


if __name__ == '__main__':
    unittest.main()
