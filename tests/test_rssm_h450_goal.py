"""H450 exact TOTAL threshold, ten completed ticks, fresh revision and resume."""
from copy import deepcopy
import json
import math
import unittest

from turkeyworld.rssm_h450_goal import (
    h450_goal_spec, initial_h450_goal, commit_h450_goal, validate_h450_goal)


def initial():
    return initial_h450_goal(source_optimizer_step=6130, plan_sha256='a'*64)


def update(goal, loss, horizon=450):
    return commit_h450_goal(goal, step=goal['last_optimizer_step']+1,
                            horizon=horizon, total_loss=loss)


class H450GoalTests(unittest.TestCase):
    def test_exactly_ten_completed_H450_updates_no_single_pass_and_no_after_success(self):
        goal = initial()
        for count in range(1, 11):
            previous = deepcopy(goal)
            goal = update(goal, math.nextafter(1., 0.))
            self.assertEqual(previous['updates'], count-1)
            self.assertEqual(goal['streak'], count)
            self.assertEqual(goal['goal_met'], count == 10)
            self.assertEqual(goal['qualifying_optimizer_steps'], list(range(6131, 6131+count)))
            validate_h450_goal(goal, current_step=6130+count)
        self.assertEqual(goal['success_optimizer_step'], 6140)
        with self.assertRaisesRegex(ValueError, 'Completed'):
            update(goal, .1)

    def test_equal_one_resets_and_nine_good_then_reset_requires_ten_new_good(self):
        goal = initial()
        for _ in range(9): goal = update(goal, .999)
        goal = update(goal, 1.)
        self.assertEqual(goal['streak'], 0)
        self.assertEqual(goal['qualifying_totals'], [])
        self.assertFalse(goal['goal_met'])
        for _ in range(9): goal = update(goal, .9)
        self.assertFalse(goal['goal_met'])
        goal = update(goal, .9)
        self.assertTrue(goal['goal_met'])
        self.assertEqual(goal['updates'], 20)

    def test_nonH450_total_never_counts_and_clears_a_partial_final_streak(self):
        goal = initial()
        for horizon in (4, 240, 288, 368, 432):
            for _ in range(10): goal = update(goal, 0., horizon)
        self.assertEqual(goal['streak'], 0)
        self.assertEqual(goal['h450_updates'], 0)
        self.assertFalse(goal['goal_met'])
        goal = update(goal, .2)
        self.assertEqual(goal['streak'], 1)
        goal = update(goal, .1, 432)
        self.assertEqual(goal['streak'], 0)

    def test_H450_is_unbounded_beyond100_and2000_then_only_actual_TOTAL_qualifies(self):
        goal = initial()
        for _ in range(2100): goal = update(goal, 1.001)
        self.assertFalse(goal['goal_met'])
        self.assertEqual(goal['h450_updates'], 2100)
        for _ in range(10): goal = update(goal, .99)
        self.assertTrue(goal['goal_met'])
        spec = h450_goal_spec()
        self.assertIsNone(spec['final_stage_max_updates'])
        self.assertFalse(spec['budget_completion_is_goal_success'])
        self.assertFalse(spec['source_support_eligibility_is_goal_requirement'])
        self.assertFalse(spec['EMA_is_goal_measurement'])
        self.assertFalse(spec['validation_loss_is_goal_measurement'])

    def test_fresh_activation_does_not_retrocount_source_streak_or_loss(self):
        old = dict(horizon=450, updates=100, streak=9, train_loss=.1)
        goal = initial()
        self.assertEqual(goal['streak'], 0)
        self.assertEqual(goal['h450_updates'], 0)
        self.assertFalse(goal['goal_met'])
        self.assertIsNone(goal['last_total_loss'])
        self.assertEqual(old['streak'], 9)
        spec = h450_goal_spec(); spec['required_consecutive_passes'] = 1
        self.assertEqual(h450_goal_spec()['required_consecutive_passes'], 10)

    def test_nonfinite_bool_negative_or_uncommitted_partial_updates_rejected_without_mutation(self):
        goal = initial()
        for value in (float('nan'), float('inf'), -1., True, '0.9', None):
            before = deepcopy(goal)
            with self.subTest(loss=value), self.assertRaises(ValueError):
                update(goal, value)
            self.assertEqual(goal, before)
        for kwargs in [dict(step=6131, horizon=450, total_loss=.9, optimizer_completed=False),
                       dict(step=6131, horizon=450, total_loss=.9, optimizer_completed=1),
                       dict(step=6132, horizon=450, total_loss=.9),
                       dict(step=6131, horizon=True, total_loss=.9),
                       dict(step=6131, horizon=451, total_loss=.9)]:
            with self.assertRaises(ValueError): commit_h450_goal(goal, **kwargs)

    def test_resume_roundtrip_retains_streak_exact_and_tampered_evidence_rejects(self):
        goal = initial()
        for value in (.9, .8, .7, .6): goal = update(goal, value)
        restored = json.loads(json.dumps(goal, allow_nan=False))
        self.assertEqual(validate_h450_goal(restored, current_step=6134), goal)
        for _ in range(6): restored = update(restored, .9)
        self.assertTrue(restored['goal_met'])
        changes = [lambda x: x.__setitem__('streak', 10),
                   lambda x: x.__setitem__('goal_met', True),
                   lambda x: x.__setitem__('last_total_loss', 1.),
                   lambda x: x['qualifying_totals'].__setitem__(0, 1.),
                   lambda x: x['qualifying_optimizer_steps'].__setitem__(0, 6000),
                   lambda x: x['qualifying_optimizer_steps'].__setitem__(1, 6134),
                   lambda x: x.__setitem__('last_optimizer_step', 6135),
                   lambda x: x['settings'].__setitem__('required_consecutive_passes', 1),
                   lambda x: x.__setitem__('plan_sha256', 'bad')]
        for index, change in enumerate(changes):
            bad = deepcopy(goal); change(bad)
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_h450_goal(bad)


if __name__ == '__main__':
    unittest.main()
