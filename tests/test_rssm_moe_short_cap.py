from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from turkeyworld.rssm_combat_memory_protocol import file_sha
from turkeyworld.rssm_moe_budget import (short_cap_plan, flat_cap_plan,
    bind_budget_policy, curriculum_completion, SHORT_CAP_PROTOCOL)
from turkeyworld.rssm_moe_protocol import (MOE_HORIZONS, moe_training_plan,
    bind_moe_training, commit_moe_update, moe_training_horizon)


class MoEShortCapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root/'source.pt'
        self.source.write_bytes(b'full-boundary')
        self.state = {'global_training_step':5352, 'rssm_training':{}}
        plan = moe_training_plan(self.state, source_checkpoint=self.source,
            source_checkpoint_sha256=file_sha(self.source), architecture_migration_sha256='a'*64)
        path = self.root/'moe.json'; path.write_text(json.dumps(plan), encoding='utf-8')
        bind_moe_training(self.state, plan, path=path, sha256=file_sha(path),
                          loaded_checkpoint_sha256=file_sha(self.source))

    @property
    def gate(self):
        return self.state['rssm_training']['moe_training']

    def activate(self, legacy=False):
        source = self.root/f'boundary_{self.state["global_training_step"]}.pt'
        source.write_bytes(str(self.state['global_training_step']).encode())
        plan = (flat_cap_plan(self.state, source_checkpoint=source) if legacy else
                short_cap_plan(self.state, source_checkpoint=source))
        path = self.root/('cap50.json' if legacy else 'cap25.json')
        path.write_text(json.dumps(plan), encoding='utf-8')
        bind_budget_policy(self.state, plan, path=path, sha256=file_sha(path),
                           loaded_checkpoint_sha256=file_sha(source))
        return plan, path

    def update(self, loss=1., *, eligible=True):
        horizon = moe_training_horizon(self.state)
        self.state['global_training_step'] += 1
        return commit_moe_update(self.state, step=self.state['global_training_step'],
                                  horizon=horizon, train_loss=loss, loss_gate_eligible=eligible)

    def test_missing_direct_labels_reset_loss_streak_but_preserve_twenty_update_budget(self):
        plan = short_cap_plan(self.state, source_checkpoint=self.source, cap=20)
        path = self.root/'cap20.json'; path.write_text(json.dumps(plan), encoding='utf-8')
        bind_budget_policy(self.state, plan, path=path, sha256=file_sha(path),
                           loaded_checkpoint_sha256=file_sha(self.source))
        for _ in range(4): self.update(.01)
        self.update(.01, eligible=False)
        self.assertEqual((self.gate['current_horizon'], self.gate['consecutive_below_threshold']), (4,0))
        self.assertFalse(self.gate['last_gate_passed'])
        self.assertEqual(self.gate['last_train_loss'], .01)
        self.assertEqual(moe_training_horizon(json.loads(json.dumps(self.state))), 4)
        for _ in range(15): self.update(.01, eligible=False)
        self.assertEqual((self.gate['current_horizon'], self.gate['stage_updates']), (8,0))
        entry = self.gate['advance_history'][-1]
        self.assertEqual(entry['advance_reason'], 'stage_budget')
        self.assertEqual(entry['qualifying_streak'], 0)
        self.assertFalse(entry['loss_gate_eligible'])
        self.assertFalse(self.gate['goal_met'])
        self.assertEqual(moe_training_horizon(json.loads(json.dumps(self.state))), 8)
        for _ in range(5): self.update(.01)
        self.assertEqual(self.gate['current_horizon'], 16)
        self.assertTrue(self.gate['last_loss_gate_eligible'])

    def test_five_consecutive_losses_and_exact_half_resets(self):
        self.activate()
        for _ in range(4): self.update(.499999)
        self.assertEqual((self.gate['current_horizon'], self.gate['consecutive_below_threshold']), (4,4))
        self.update(.5)
        self.assertEqual(self.gate['consecutive_below_threshold'], 0)
        for _ in range(4): self.update(.49)
        self.assertEqual(self.gate['current_horizon'], 4)
        self.update(.49)
        self.assertEqual((self.gate['current_horizon'], self.gate['stage_updates']), (8,0))
        self.assertEqual(self.gate['last_advance_reason'], 'loss_streak')
        self.assertEqual(self.gate['advance_history'][-1]['qualifying_streak'], 5)

    def test_exact_twenty_fifth_commit_advances(self):
        self.activate()
        for _ in range(24): self.update(.5)
        self.assertEqual((self.gate['current_horizon'], self.gate['stage_updates']), (4,24))
        self.update(.5)
        self.assertEqual((self.gate['current_horizon'], self.gate['stage_updates']), (8,0))
        self.assertEqual(self.gate['last_advance_reason'], 'stage_budget')
        self.assertFalse(self.gate['goal_met'])

    def test_loss_streak_wins_when_both_conditions_arrive_together(self):
        self.activate()
        for _ in range(20): self.update()
        for _ in range(5): self.update(.4)
        self.assertEqual(self.gate['last_advance_reason'], 'loss_streak')

    def test_v3_parent_history_and_overdue_counts_preserved_once(self):
        self.activate(legacy=True)
        for _ in range(10): self.update(.2)  # Historical ten-pass success.
        for _ in range(44): self.update(1.)
        before = deepcopy(self.state)
        plan, path = self.activate()
        self.assertEqual(self.state['global_training_step'], before['global_training_step'])
        self.assertEqual((self.gate['current_horizon'],self.gate['stage_updates']), (16,0))
        self.assertEqual(self.gate['horizon_update_counts']['8'], 44)
        self.assertEqual(self.gate['updates'], 54)
        self.assertEqual(self.gate['advance_policy']['source_gate'], before['rssm_training']['moe_training'])
        self.assertEqual(self.gate['advance_history'][0], before['rssm_training']['moe_training']['advance_history'][0])
        self.assertTrue(self.gate['advance_history'][-1]['activation_boundary'])
        bound = deepcopy(self.state)
        bind_budget_policy(self.state,plan,path=path,sha256=file_sha(path),loaded_checkpoint_sha256='b'*64)
        self.assertEqual(self.state,bound)
        for _ in range(5): self.update(.4)
        self.assertEqual(self.gate['current_horizon'],32)
        self.assertEqual(self.gate['advance_history'][-1]['completed_updates_at_previous_stage'],5)

    def test_new_policy_does_not_reinterpret_old_streak_and_resumes_exactly(self):
        self.activate(legacy=True)
        for _ in range(4): self.update(.2)
        plan,path = self.activate()
        self.assertEqual((self.gate['stage_updates'],self.gate['consecutive_below_threshold']),(4,0))
        for _ in range(4): self.update(.4)
        resumed = json.loads(json.dumps(self.state))
        bind_budget_policy(resumed,plan,path=path,sha256=file_sha(path),loaded_checkpoint_sha256='b'*64)
        self.assertEqual(resumed,self.state)
        self.update(.4)
        expected = deepcopy(self.state)
        self.state = resumed; self.update(.4)
        self.assertEqual(self.state,expected)

    def test_final_budget_and_loss_stop_without_an_extra_commit(self):
        for low in (False,True):
            with self.subTest(low=low):
                if low: self.setUp()
                self.activate()
                for _ in range(30*(5 if low else 25)):
                    self.update(.4 if low else .5)
                self.assertEqual(self.gate['current_horizon'],450)
                for _ in range((5 if low else 25)-1): self.update(.4 if low else .5)
                self.assertIsNone(curriculum_completion(self.gate))
                self.update(.4 if low else .5)
                self.assertEqual(curriculum_completion(self.gate),'completed_loss_gate' if low else 'completed_budget')
                self.assertEqual(self.gate['goal_met'],low)
                before=deepcopy(self.gate)
                with self.assertRaises(ValueError):
                    commit_moe_update(self.state,step=self.state['global_training_step']+1,horizon=450,train_loss=.4)
                self.assertEqual(self.gate,before)

    def test_threshold_parent_and_history_tampering_rejected(self):
        self.activate(legacy=True)
        for _ in range(44): self.update()
        self.activate(); before=deepcopy(self.state)
        for mutate in (
            lambda g:g['advance_policy'].update(threshold=.25),
            lambda g:g['advance_policy'].update(required_consecutive_passes=10),
            lambda g:g['advance_policy'].update(previous_policy_sha256='f'*64),
            lambda g:g['advance_policy']['source_gate'].update(last_train_loss=123.),
            lambda g:g['advance_history'][-1].update(stage_budget=50)):
            self.state=deepcopy(before);mutate(self.gate)
            with self.assertRaises(ValueError):moe_training_horizon(self.state)


if __name__=='__main__':unittest.main()
