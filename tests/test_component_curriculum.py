"""CPU checks for real optimizer boundaries and truthful component handoffs."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from turkeyworld.component_curriculum import ComponentCurriculum
from turkeyworld.navigation_sampling import NAVIGATION_CASES, NAVIGATION_FOCUS_WEIGHTS


class ComponentCurriculumTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)/'component_plan.json'
        self.base_fingerprint = 'immutable-scene-plan-sha256'
        self.plan = {
            'version': 1, 'base_plan_fingerprint': self.base_fingerprint,
            'start_optimizer_step': 777, 'validation_every': 100,
            'preview_every': 10, 'microbatch_size': 16,
            'scene_at_start': 'navigation', 'required_passes': 2,
            'plateau_patience': 5, 'plateau_relative_delta': .01,
            'navigation_weights': dict(NAVIGATION_FOCUS_WEIGHTS),
            'navigation_source_fingerprint': 'held-out-source-files-sha256',
            'phases': [
                {'id': 'renderer', 'minimum_updates': 200, 'budget_updates': 1000},
                {'id': 'dynamics', 'minimum_updates': 300, 'budget_updates': 1000},
                {'id': 'joint', 'minimum_updates': 0, 'budget_updates': 5182},
            ],
        }
        self.write_plan()

    def write_plan(self):
        self.path.write_text(json.dumps(self.plan, sort_keys=True), encoding='utf-8')

    def curriculum(self, *, step=777, previous=None):
        return ComponentCurriculum(self.path, step=step,
            base_plan_fingerprint=self.base_fingerprint, previous=previous)

    @staticmethod
    def advance(curriculum, updates):
        for _ in range(updates):
            curriculum.record_update(curriculum.state['last_optimizer_step']+1,
                {'total': 1.25, 'zero': 0, 'component_phase': curriculum.name})

    @staticmethod
    def renderer_report(*, good=True, finite=True):
        return {'passed': False, 'report_path': 'held_out_renderer/report.json',
            'metrics': {'rgb_finite': finite, 'rgb_ssim': .95 if good else .50,
                        'rgb_mae': .01 if good else .20}}

    @staticmethod
    def motion_report(*, good=True):
        return {'passed': False, 'report_path': 'held_out_motion/report.json',
            'metrics': {'all_states_finite': True, 'all_memory_finite': True,
                'player_xy_mean': .05, 'player_xy_p95': .12,
                'wall_windows': 0 if good else 1, 'trajectory_windows': 32,
                # Rendering is deliberately bad: a motion-only gate ignores it.
                'rgb_finite': False, 'rgb_ssim': .01, 'rgb_mae': 1.}}

    def enter_dynamics_by_quality(self, curriculum):
        self.advance(curriculum, 200)
        self.assertFalse(curriculum.record_validation(self.renderer_report(),
            curriculum.state['last_optimizer_step']))
        self.advance(curriculum, 100)
        self.assertTrue(curriculum.record_validation(self.renderer_report(),
            curriculum.state['last_optimizer_step']))
        self.assertEqual(curriculum.name, 'dynamics')

    def enter_joint_by_budget(self):
        self.plan['phases'][0]['budget_updates'] = 200
        self.plan['phases'][1]['budget_updates'] = 300
        self.write_plan()
        curriculum = self.curriculum()
        self.advance(curriculum, 200)
        curriculum.record_validation(self.renderer_report(finite=False),
            curriculum.state['last_optimizer_step'])
        self.advance(curriculum, 300)
        curriculum.record_validation(self.motion_report(good=False),
            curriculum.state['last_optimizer_step'])
        self.assertEqual(curriculum.name, 'joint')
        return curriculum

    def test_exact_resume_boundary_real_update_counts_and_deep_copy(self):
        for step in (776, 778):
            with self.subTest(step=step), self.assertRaises(ValueError):
                self.curriculum(step=step)
        curriculum = self.curriculum()
        self.assertEqual(curriculum.state['phase_updates'], 0)
        curriculum.record_update(778, {'total': 0., 'flag': True, 'nan': float('nan'),
            'infinity': float('inf'), 'phase': 'renderer'})
        self.assertEqual(curriculum.phase['losses'], {'total': 0.})
        snapshot = copy.deepcopy(curriculum.state)
        for step in (777, 778, 780):
            with self.subTest(step=step), self.assertRaises(ValueError):
                curriculum.record_update(step, {'total': 5})
            self.assertEqual(curriculum.state, snapshot)
        saved = json.loads(json.dumps(snapshot))
        resumed = self.curriculum(step=778, previous=saved)
        self.assertEqual(resumed.state, snapshot)
        saved['phases'][0]['updates'] = 999
        self.assertEqual(resumed.state['phase_updates'], 1)
        resumed.record_update(779, {'total': .3})
        self.assertEqual(resumed.state['phase_updates'], 2)
        self.assertEqual(resumed.state['loss_step'], 779)
        for step in (777, 779):
            with self.subTest(resume_step=step), self.assertRaises(ValueError):
                self.curriculum(step=step, previous=snapshot)

    def test_plan_identity_and_bounds_are_immutable_and_runtime_does_not_write_plan(self):
        curriculum = self.curriculum()
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.advance(curriculum, 1)
        curriculum.record_validation(self.renderer_report(), 778)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), before)
        original = copy.deepcopy(self.plan)
        changes = [
            ('validation_every', 101), ('preview_every', 20), ('microbatch_size', 8),
            ('scene_at_start', 'moving1'), ('base_plan_fingerprint', 'another-source'),
            ('required_passes', 1), ('plateau_patience', 4),
            ('plateau_relative_delta', .02), ('navigation_source_fingerprint', 'different-files'),
        ]
        for key, value in changes:
            with self.subTest(field=key):
                self.plan = copy.deepcopy(original)
                self.plan[key] = value
                self.write_plan()
                with self.assertRaises(ValueError):
                    self.curriculum(step=778, previous=curriculum.state)
        for low, high in ((0, 1000), (True, 1000), (200, 199), (200, True)):
            with self.subTest(minimum=low, budget=high):
                self.plan = copy.deepcopy(original)
                self.plan['phases'][0].update(minimum_updates=low, budget_updates=high)
                self.write_plan()
                with self.assertRaises(ValueError):
                    self.curriculum()

    def test_renderer_requires_minimum_and_two_distinct_passing_checks(self):
        curriculum = self.curriculum()
        report = self.renderer_report()
        self.assertFalse(curriculum.record_validation(report, 777))
        self.assertEqual(curriculum.phase['pass_streak'], 0)
        self.advance(curriculum, 199)
        self.assertFalse(curriculum.record_validation(report, 976))
        self.assertEqual(curriculum.phase['pass_streak'], 0)
        self.advance(curriculum, 1)
        self.assertFalse(curriculum.record_validation(report, 977))
        self.assertEqual(curriculum.phase['pass_streak'], 1)
        self.assertTrue(curriculum.phase['gate']['minimum_updates_met'])
        with self.assertRaises(ValueError):
            curriculum.record_validation(report, 977)
        resumed = self.curriculum(step=977, previous=copy.deepcopy(curriculum.state))
        self.assertEqual(resumed.phase['pass_streak'], 1)
        self.advance(resumed, 100)
        self.assertTrue(resumed.record_validation(report, 1077))
        self.assertEqual(resumed.name, 'dynamics')
        finished = resumed.state['history'][0]
        self.assertEqual(finished['updates'], 300)
        self.assertEqual(finished['completion_reason'], 'quality')
        self.assertEqual(finished['pass_streak'], 2)
        self.assertFalse(resumed.state['gameplay_qualified'])
        self.assertEqual(resumed.state['phase_updates'], 0)
        self.assertEqual(resumed.state['phase_start_step'], 1077)
        # A new phase may evaluate its own baseline at the handoff boundary.
        resumed.record_validation(self.motion_report(good=False), 1077)

    def test_motion_gate_uses_own_state_xy_and_walls_not_rgb(self):
        curriculum = self.curriculum()
        self.enter_dynamics_by_quality(curriculum)
        self.advance(curriculum, 300)
        bad = self.motion_report(good=False)
        bad['passed'] = True
        bad['metrics'].update(rgb_finite=True, rgb_ssim=1., rgb_mae=0.)
        self.assertFalse(curriculum.record_validation(bad, 1377))
        self.assertEqual(curriculum.phase['gate']['status'], 'failed')
        self.assertEqual(curriculum.phase['pass_streak'], 0)
        self.advance(curriculum, 100)
        self.assertFalse(curriculum.record_validation(self.motion_report(), 1477))
        self.assertEqual(curriculum.phase['gate']['status'], 'passed')
        self.assertEqual(curriculum.phase['pass_streak'], 1)
        self.advance(curriculum, 100)
        self.assertTrue(curriculum.record_validation(self.motion_report(), 1577))
        self.assertEqual(curriculum.name, 'joint')
        self.assertEqual(curriculum.state['history'][-1]['completion_reason'], 'quality')
        self.assertFalse(curriculum.state['gameplay_qualified'])

    def test_five_unimproved_checks_handoff_is_plateau_not_quality(self):
        curriculum = self.curriculum()
        report = self.renderer_report(good=False)
        curriculum.record_validation(report, 777)
        for index in range(1, 6):
            self.advance(curriculum, 100)
            transitioned = curriculum.record_validation(report, 777+100*index)
            self.assertEqual(transitioned, index == 5)
            if index < 5:
                self.assertEqual(curriculum.name, 'renderer')
                self.assertEqual(curriculum.phase['checks_without_improvement'], index)
        self.assertEqual(curriculum.name, 'dynamics')
        old = curriculum.state['history'][0]
        self.assertEqual(old['completion_reason'], 'plateau')
        self.assertEqual(old['gate']['status'], 'failed')
        self.assertFalse(old['gate']['quality_passed'])
        self.assertFalse(curriculum.state['gameplay_qualified'])

    def test_budget_handoff_does_not_turn_nonfinite_or_colliding_reports_into_passes(self):
        curriculum = self.enter_joint_by_budget()
        self.assertEqual([phase['completion_reason'] for phase in curriculum.state['history']],
                         ['budget', 'budget'])
        self.assertEqual([phase['updates'] for phase in curriculum.state['history']], [200, 300])
        self.assertTrue(all(phase['gate']['status'] == 'failed' for phase in curriculum.state['history']))
        self.assertFalse(curriculum.state['gameplay_qualified'])
        self.assertEqual(curriculum.state['last_optimizer_step'], 1277)
        self.assertEqual(curriculum.state['phase_start_step'], 1277)

    def test_joint_scene_pass_cannot_qualify_game_without_explicit_final_flag(self):
        curriculum = self.enter_joint_by_budget()
        report = {'passed': True, 'scope': 'navigation', 'metrics': {'player_xy_mean': 0.},
                  'thresholds': {'player_xy_mean_max': .10}}
        self.assertFalse(curriculum.record_validation(report, 1277))
        self.assertTrue(curriculum.phase['gate']['scene_quality_passed'])
        self.assertEqual(curriculum.phase['gate']['status'], 'pending')
        self.assertEqual(curriculum.state['status'], 'training')
        self.assertFalse(curriculum.state['gameplay_qualified'])
        self.advance(curriculum, 1)
        final = {**report, 'scope': 'full_game', 'metrics': {'checked_steps': 450}}
        self.assertFalse(curriculum.record_validation(final, 1278, gameplay_qualified=True))
        self.assertTrue(curriculum.state['gameplay_qualified'])
        self.assertEqual(curriculum.state['status'], 'completed')
        self.assertEqual(curriculum.phase['gate']['status'], 'passed')
        self.assertEqual(curriculum.phase['completion_reason'], 'quality')

    def test_dynamics_focus_starts_at_zero_and_resumes_its_own_counts(self):
        curriculum = self.curriculum()
        with self.assertRaises(ValueError):
            curriculum.dynamics_sampling()
        curriculum.state['superseded_navigation_sampling'] = {'draws': 2432}
        self.enter_dynamics_by_quality(curriculum)
        sampling = curriculum.dynamics_sampling()
        self.assertEqual(sampling['weights'], NAVIGATION_FOCUS_WEIGHTS)
        self.assertEqual(sampling['sample_counts'], {case: 0 for case in NAVIGATION_CASES})
        self.assertEqual(sampling['draws'], 0)
        self.assertEqual(sampling['source_fingerprint'], self.plan['navigation_source_fingerprint'])
        sampling['sample_counts'][NAVIGATION_CASES[0]] = 16
        sampling['draws'] = 16
        self.advance(curriculum, 1)
        resumed = self.curriculum(step=1078, previous=copy.deepcopy(curriculum.state))
        self.assertEqual(resumed.dynamics_sampling()['draws'], 16)
        self.assertEqual(resumed.dynamics_sampling()['sample_counts'][NAVIGATION_CASES[0]], 16)
        self.assertEqual(resumed.state['superseded_navigation_sampling']['draws'], 2432)
        self.assertEqual(resumed.state['phase_updates'], 1)

    def test_validation_cadence_and_budget_boundary_are_not_elapsed_time(self):
        self.plan['phases'][0]['budget_updates'] = 200
        self.write_plan()
        curriculum = self.curriculum()
        self.assertTrue(curriculum.validation_due(777))
        curriculum.record_validation(self.renderer_report(good=False), 777)
        self.assertFalse(curriculum.validation_due(799))
        self.advance(curriculum, 23)
        self.assertTrue(curriculum.validation_due(800))
        curriculum.record_validation(self.renderer_report(good=False), 800)
        self.assertFalse(curriculum.validation_due(899))
        self.advance(curriculum, 100)
        curriculum.record_validation(self.renderer_report(good=False), 900)
        self.advance(curriculum, 77)
        self.assertTrue(curriculum.validation_due(977))
        self.assertEqual(curriculum.phase['updates'], 200)
        self.assertEqual(curriculum.name, 'renderer')
        self.assertTrue(curriculum.record_validation(self.renderer_report(good=False), 977))
        self.assertEqual(curriculum.state['history'][0]['completion_reason'], 'budget')

    def test_validation_rejects_future_or_stale_optimizer_boundary_without_mutating_state(self):
        curriculum = self.curriculum()
        self.advance(curriculum, 200)
        before = copy.deepcopy(curriculum.state)
        for step in (976, 978, 10000):
            with self.subTest(step=step):
                with self.assertRaises(ValueError):
                    curriculum.record_validation(self.renderer_report(), step)
                self.assertEqual(curriculum.state, before)


if __name__ == '__main__':
    unittest.main()
