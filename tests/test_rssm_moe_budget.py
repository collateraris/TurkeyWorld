from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from turkeyworld.rssm_combat_memory_protocol import file_sha
from turkeyworld.rssm_moe_budget import budget_plan,bind_budget_policy,stage_budget
from turkeyworld.rssm_moe_protocol import (moe_training_plan,bind_moe_training,
    commit_moe_update,moe_training_horizon,MOE_HORIZONS)


class MoEBudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=self.root/'source.pt';self.source.write_bytes(b'complete-checkpoint')
        self.state={'global_training_step':5352,'rssm_training':{}}
        p=self.root/'moe.json';plan=moe_training_plan(self.state,source_checkpoint=self.source,
            source_checkpoint_sha256=file_sha(self.source),architecture_migration_sha256='a'*64)
        p.write_text(json.dumps(plan),encoding='utf-8');bind_moe_training(self.state,plan,path=p,
            sha256=file_sha(p),loaded_checkpoint_sha256=file_sha(self.source))

    def update(self,loss=1.):
        horizon=moe_training_horizon(self.state);step=self.state['global_training_step']+1
        self.state['global_training_step']=step
        return commit_moe_update(self.state,step=step,horizon=horizon,train_loss=loss)

    def activate(self):
        self.plan=budget_plan(self.state,source_checkpoint=self.source);self.path=self.root/'budget.json'
        self.path.write_text(json.dumps(self.plan),encoding='utf-8')
        return self.bind()

    def bind(self,loaded=None):
        return bind_budget_policy(self.state,self.plan,path=self.path,sha256=file_sha(self.path),
            loaded_checkpoint_sha256=loaded or file_sha(self.source))

    def test_fiftieth_update_advances_and_sixtyfifth_advances_next_stage(self):
        self.activate()
        for _ in range(49):self.update(.25)
        self.assertEqual(moe_training_horizon(self.state),4)
        gate=self.update(.25)
        self.assertEqual((gate['current_horizon'],gate['stage_updates'],gate['last_advance_reason']),(8,0,'stage_budget'))
        for _ in range(64):self.update(2.)
        self.assertEqual(moe_training_horizon(self.state),8)
        self.assertEqual(self.update(2.)['current_horizon'],16)

    def test_tenth_consecutive_success_advances_early_and_equality_resets(self):
        self.activate()
        for _ in range(8):self.update(.2)
        self.update(.25)
        self.assertEqual(self.state['rssm_training']['moe_training']['consecutive_below_threshold'],0)
        for _ in range(9):self.update(.2)
        self.assertEqual(moe_training_horizon(self.state),4)
        gate=self.update(.2)
        self.assertEqual((gate['current_horizon'],gate['last_advance_reason']),(8,'loss_streak'))

    def test_already_completed_updates_count_and_activation_advances_only_once(self):
        for _ in range(83):self.update(1.)
        before=deepcopy(self.state);self.activate();gate=self.state['rssm_training']['moe_training']
        self.assertEqual(self.state['global_training_step'],before['global_training_step'])
        self.assertEqual((gate['current_horizon'],gate['stage_updates'],gate['updates']),(8,0,83))
        self.assertEqual(gate['horizon_update_counts']['4'],83)
        self.assertEqual(gate['advance_policy']['updates'],0)
        self.bind(loaded='b'*64)
        self.assertEqual(gate['current_horizon'],8)
        self.assertEqual(gate['advance_policy']['source_gate'],before['rssm_training']['moe_training'])

    def test_existing_streak_preserved_and_resumes_exactly(self):
        for _ in range(9):self.update(.2)
        self.activate();self.bind(loaded='b'*64)
        self.assertEqual(self.state['rssm_training']['moe_training']['consecutive_below_threshold'],9)
        self.assertEqual(self.update(.2)['current_horizon'],8)
        self.bind(loaded='c'*64)

    def test_all31_zero_based_budgets_final_continues_without_quality_claim(self):
        self.activate()
        for index,horizon in enumerate(MOE_HORIZONS):
            self.assertEqual(moe_training_horizon(self.state),horizon)
            for _ in range(stage_budget(index)):self.update(1.)
        gate=self.update(1.)
        self.assertEqual((gate['current_horizon'],gate['stage_index'],gate['goal_met']),(450,30,False))
        self.assertEqual([x['stage_budget']for x in gate['advance_history']],[50+15*i for i in range(30)])

    def test_source_or_policy_tampering_rejected(self):
        self.activate();self.state['rssm_training']['moe_training']['advance_policy']['budget_base']=49
        with self.assertRaises(ValueError):moe_training_horizon(self.state)
        self.state['rssm_training']['moe_training']['advance_policy']['budget_base']=50
        self.source.write_bytes(b'changed')
        with self.assertRaises(ValueError):self.bind()


if __name__=='__main__':unittest.main()
