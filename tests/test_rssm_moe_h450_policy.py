from copy import deepcopy
import json
import unittest

from test_rssm_moe_horizon100 import MoEHorizon100Tests
from turkeyworld.rssm_combat_memory_protocol import file_sha
from turkeyworld.rssm_moe_budget import (h450_loss_goal_plan,bind_budget_policy,
    curriculum_completion,stage_budget,loss_gate_settings)
from turkeyworld.rssm_moe_protocol import moe_training_horizon


class H450PolicyTests(unittest.TestCase):
    def setUp(self):
        self.f=MoEHorizon100Tests('test_ordinary_stage_twentieth_budget_and_strict_threshold_still_work')
        self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.f.activate()

    @property
    def gate(self):return self.f.gate

    def activate(self):return self.f.activate(h450_loss_goal_plan,'v6')

    def to_final(self):
        while self.gate['current_horizon']!=450:self.f.update(.75)

    def test_full_path_retains_intermediate_budgets_and_H450_has_no_cap_then_ten_strict_TOTALs(self):
        self.activate()
        self.to_final()
        counts=self.gate['horizon_update_counts']
        self.assertEqual((counts['240'],counts['368']), (100,100))
        self.assertEqual(counts['288'],20)
        self.assertEqual(loss_gate_settings(self.gate),(1.,10))
        self.assertFalse(self.gate['last_gate_passed']) # .75 at H448 used .5, not the next1.0.
        self.assertEqual(self.gate['final_horizon_goal']['streak'],0)
        self.assertIsNone(stage_budget(30,self.gate['advance_policy']))
        for _ in range(105):self.f.update(1.)
        self.assertIsNone(curriculum_completion(self.gate))
        self.assertEqual(self.gate['stage_updates'],105)
        for _ in range(9):self.f.update(.99,eligible=False)
        self.assertEqual(self.gate['consecutive_below_threshold'],9)
        self.assertEqual(self.gate['final_horizon_goal']['streak'],9)
        self.f.update(1.)
        self.assertEqual(self.gate['final_horizon_goal']['streak'],0)
        for _ in range(9):self.f.update(.8)
        self.assertIsNone(curriculum_completion(self.gate))
        self.f.update(.999999)
        self.assertEqual(curriculum_completion(self.gate),'completed_loss_gate')
        self.assertTrue(self.gate['goal_met'])
        self.assertFalse(self.gate['budget_exhausted'])
        before=deepcopy(self.gate)
        with self.assertRaises(ValueError):self.f.update(.1)
        self.assertEqual(before,self.gate)

    def test_activation_preserves_current_H_counts_history_streak_and_fresh_final_goal(self):
        for _ in range(4):self.f.update(.4)
        before=deepcopy(self.gate)
        plan,path=self.activate()
        for key in ('stage_index','stage_updates','current_horizon','updates','advance_history',
                    'horizon_update_counts','consecutive_below_threshold','main_examples','main_transitions'):
            self.assertEqual(self.gate[key],before[key])
        self.assertEqual(self.gate['advance_policy']['source_gate'],before)
        self.assertEqual(self.gate['final_horizon_goal']['streak'],0)
        self.f.update(.4)
        self.assertEqual(self.gate['current_horizon'],8)
        restored=json.loads(json.dumps(self.f.state))
        bind_budget_policy(restored,plan,path=path,sha256=file_sha(path),loaded_checkpoint_sha256='f'*64)
        self.assertEqual(restored,self.f.state)

    def test_already_H450_activation_does_not_reuse_old_streak_or_95_completed_updates(self):
        self.to_final()
        for _ in range(95):self.f.update(.1)
        self.assertEqual(self.gate['consecutive_below_threshold'],5)
        self.activate()
        self.assertEqual((self.gate['stage_updates'],self.gate['consecutive_below_threshold']),(95,0))
        self.assertEqual(self.gate['final_horizon_goal']['h450_updates'],0)
        for _ in range(9):self.f.update(.1)
        self.assertIsNone(curriculum_completion(self.gate))
        self.f.update(.1)
        self.assertEqual(curriculum_completion(self.gate),'completed_loss_gate')

    def test_resume_with_eight_H450_passes_retains_exact_streak_and_goal_tamper_fails_closed(self):
        plan,path=self.activate(); self.to_final()
        for _ in range(8):self.f.update(.75)
        restored=json.loads(json.dumps(self.f.state))
        bind_budget_policy(restored,plan,path=path,sha256=file_sha(path),loaded_checkpoint_sha256='e'*64)
        self.assertEqual(restored,self.f.state)
        for mutate in (lambda g:g['advance_policy'].update(final_stage_max_updates=100),
                       lambda g:g['final_horizon_goal'].update(streak=10,goal_met=True),
                       lambda g:g['final_horizon_goal']['qualifying_totals'].__setitem__(0,1.),
                       lambda g:g.update(consecutive_below_threshold=7),
                       lambda g:g['advance_policy']['mandatory_horizon_updates'].update({'450':100})):
            bad=deepcopy(self.f.state); mutate(bad['rssm_training']['moe_training'])
            with self.assertRaises(ValueError):moe_training_horizon(bad)
        self.f.update(.75);self.assertIsNone(curriculum_completion(self.gate))
        self.f.update(.75);self.assertTrue(self.gate['goal_met'])


if __name__=='__main__':unittest.main()
