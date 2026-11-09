"""Patience counts distinct checks and preserves numerical bests independently."""
import json
import unittest

from turkeyworld.plateau import PlateauTracker


class PlateauTrackerTests(unittest.TestCase):
    def test_six_bad_distinct_checks_trigger_plateau_after_initial_baseline(self):
        tracker = PlateauTracker()
        first = tracker.update_score(.2, 0)
        self.assertTrue(first['improved'])
        self.assertTrue(first['best_changed'])
        for index in range(1, 7):
            result = tracker.update_score(.21, index*500)
            self.assertEqual(result['bad_checks'], index)
            self.assertEqual(result['plateau'], index == 6)
        self.assertEqual(result['checks'], 7)
        self.assertEqual(result['best_step'], 0)

    def test_subthreshold_numerical_bests_accumulate_since_last_meaningful_anchor(self):
        tracker = PlateauTracker()
        tracker.update_score(1., 0)
        for step, value in [(500, .998), (1000, .996)]:
            result = tracker.update_score(value, step)
            self.assertTrue(result['best_changed'])
            self.assertFalse(result['improved'])
            self.assertEqual(result['anchor_score'], 1.)
        result = tracker.update_score(.994, 1500)
        self.assertTrue(result['improved'])
        self.assertEqual(result['bad_checks'], 0)
        self.assertEqual(result['anchor_score'], .994)
        self.assertEqual(result['best_step'], 1500)

    def test_exact_relative_threshold_is_meaningful_and_equal_score_is_not(self):
        tracker = PlateauTracker()
        tracker.update_score(1., 0)
        result = tracker.update_score(.995, 500)
        self.assertTrue(result['improved'])
        result = tracker.update_score(.995, 1000)
        self.assertFalse(result['improved'])
        self.assertFalse(result['best_changed'])
        self.assertEqual(result['bad_checks'], 1)

    def test_duplicate_step_does_not_change_patience_or_best_weights(self):
        tracker = PlateauTracker()
        tracker.update_score(.2, 500)
        tracker.update_score(.201, 1000)
        before = tracker.to_state()
        result = tracker.update_score(.1, 1000)
        self.assertTrue(result['repeated'])
        self.assertFalse(result['best_changed'])
        self.assertEqual(tracker.to_state(), before)
        with self.assertRaises(ValueError):
            tracker.update_score(.1, 500)

    def test_json_resume_matches_uninterrupted_updates_and_plateau_decision(self):
        first = PlateauTracker(phase='teacher_quality_75m', fingerprint='panel-A')
        for step, value in [(0, .2), (500, .2001), (1000, .1998)]:
            first.update_score(value, step)
        state = json.loads(json.dumps(first.to_state(), allow_nan=False))
        second = PlateauTracker.from_state(state, phase='teacher_quality_75m', fingerprint='panel-A', current_step=1000)
        self.assertEqual(second.to_state(), first.to_state())
        duplicate = second.update_score(.1998, 1000)
        self.assertTrue(duplicate['repeated'])
        for step, value in [(1500, .1997), (2000, .1996), (2500, .1995), (3000, .1994)]:
            self.assertEqual(first.update_score(value, step), second.update_score(value, step))
        self.assertTrue(second.update_score(.1994, 3000)['plateau'])

    def test_changed_panel_phase_or_patience_resets_context(self):
        tracker = PlateauTracker(phase='teacher-A', fingerprint='panel-A')
        tracker.update_score(.3, 100)
        tracker.update_score(.31, 600)
        for options in ({'phase': 'teacher-B'}, {'fingerprint': 'panel-B'},
                        {'patience': 7}, {'min_relative_improvement': .01}):
            with self.subTest(options=options):
                fresh = PlateauTracker.from_state(tracker.to_state(), **options)
                self.assertEqual(fresh.checks, 0)
                self.assertEqual(fresh.bad_checks, 0)
                self.assertEqual(fresh.reset_reason, 'context_changed')

    def test_future_history_is_discarded_for_an_earlier_restored_checkpoint(self):
        tracker = PlateauTracker()
        tracker.update_score(.3, 500)
        tracker.update_score(.31, 1000)
        fresh = PlateauTracker.from_state(tracker.to_state(), current_step=750)
        self.assertEqual(fresh.reset_reason, 'future_state')
        self.assertEqual(fresh.checks, 0)
        self.assertTrue(fresh.update_score(.32, 750)['improved'])

    def test_nonfinite_negative_scores_and_invalid_configuration_are_rejected(self):
        tracker = PlateauTracker()
        for value in (float('nan'), float('inf'), -float('inf'), -.1, None, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                tracker.update_score(value, 0)
        for options in ({'patience': 0}, {'patience': True}, {'patience': 1.5},
                        {'min_relative_improvement': float('nan')},
                        {'min_relative_improvement': -1}, {'min_relative_improvement': 1.1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                PlateauTracker(**options)

    def test_malformed_saved_history_is_rejected(self):
        tracker = PlateauTracker()
        tracker.update_score(.3, 0)
        saved = tracker.to_state()
        for updates in ({'best_score': float('nan')}, {'bad_checks': 1},
                        {'best_step': 100}, {'decision': 'plateau'}, {'version': 9}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                PlateauTracker.from_state({**saved, **updates})

    def test_perfect_zero_score_is_finite_and_still_obeys_patience(self):
        tracker = PlateauTracker(patience=2)
        tracker.update_score(0., 0)
        self.assertFalse(tracker.update_score(0., 500)['plateau'])
        self.assertTrue(tracker.update_score(0., 1000)['plateau'])
        json.dumps(tracker.to_state(), allow_nan=False)


if __name__ == '__main__':
    unittest.main()
