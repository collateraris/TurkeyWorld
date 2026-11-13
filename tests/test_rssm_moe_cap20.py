from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from turkeyworld.rssm_combat_memory_protocol import file_sha
from turkeyworld.rssm_moe_budget import short_cap_plan,bind_budget_policy,curriculum_completion
from turkeyworld.rssm_moe_protocol import moe_training_plan,bind_moe_training,moe_training_horizon,commit_moe_update


class Cap20Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=self.root/'initial.pt';self.source.write_bytes(b'complete-boundary')
        self.state={'global_training_step':5352,'rssm_training':{}}
        plan=moe_training_plan(self.state,source_checkpoint=self.source,
            source_checkpoint_sha256=file_sha(self.source),architecture_migration_sha256='a'*64)
        path=self.root/'moe.json';path.write_text(json.dumps(plan),encoding='utf-8')
        bind_moe_training(self.state,plan,path=path,sha256=file_sha(path),loaded_checkpoint_sha256=file_sha(self.source))

    @property
    def gate(self):return self.state['rssm_training']['moe_training']

    def activate(self,cap):
        source=self.root/f'boundary_{cap}_{self.state["global_training_step"]}.pt'
        source.write_text(str(self.state['global_training_step']))
        plan=short_cap_plan(self.state,source_checkpoint=source,cap=cap)
        path=self.root/f'cap{cap}.json';path.write_text(json.dumps(plan),encoding='utf-8')
        bind_budget_policy(self.state,plan,path=path,sha256=file_sha(path),loaded_checkpoint_sha256=file_sha(source))
        return plan,path

    def update(self,loss=.5):
        horizon=moe_training_horizon(self.state);self.state['global_training_step']+=1
        commit_moe_update(self.state,step=self.state['global_training_step'],horizon=horizon,train_loss=loss)

    def test_exact_twentieth_update_advances_and_final450_stops_at20(self):
        self.activate(20)
        for _ in range(19):self.update()
        self.assertEqual((self.gate['current_horizon'],self.gate['stage_updates']),(4,19))
        self.update();self.assertEqual((self.gate['current_horizon'],self.gate['stage_updates']),(8,0))
        self.assertEqual(self.gate['last_advance_reason'],'stage_budget')
        for _ in range(29*20):self.update()
        self.assertEqual(self.gate['current_horizon'],450)
        for _ in range(19):self.update()
        self.assertIsNone(curriculum_completion(self.gate));self.update()
        self.assertEqual(curriculum_completion(self.gate),'completed_budget')
        with self.assertRaises(ValueError):self.update()

    def test_twenty_five_parent_history_and_overdue_stage_advance_once(self):
        old_plan,old_path=self.activate(25)
        for _ in range(24):self.update()
        before=deepcopy(self.state);old_bytes=old_path.read_bytes()
        plan,path=self.activate(20)
        self.assertEqual((self.gate['current_horizon'],self.gate['stage_updates']),(8,0))
        self.assertEqual(self.gate['horizon_update_counts']['4'],24)
        self.assertEqual(self.gate['updates'],24)
        self.assertEqual(self.gate['advance_policy']['source_gate'],before['rssm_training']['moe_training'])
        self.assertEqual(old_path.read_bytes(),old_bytes);self.assertEqual(old_plan['budget_base'],25)
        now=deepcopy(self.state)
        bind_budget_policy(self.state,plan,path=path,sha256=file_sha(path),loaded_checkpoint_sha256='b'*64)
        self.assertEqual(self.state,now)

    def test_five_low_losses_still_advances_early_and_policy_resume_is_exact(self):
        self.activate(25)
        for _ in range(4):self.update(.4)
        plan,path=self.activate(20)
        self.assertEqual((self.gate['stage_updates'],self.gate['consecutive_below_threshold']),(4,0))
        for _ in range(4):self.update(.4)
        restored=json.loads(json.dumps(self.state))
        bind_budget_policy(restored,plan,path=path,sha256=file_sha(path),loaded_checkpoint_sha256='b'*64)
        self.update(.4);expected=deepcopy(self.state);self.state=restored;self.update(.4)
        self.assertEqual(self.state,expected);self.assertEqual(self.gate['current_horizon'],8)
        self.assertEqual(self.gate['last_advance_reason'],'loss_streak')

    def test_unsupported_cap_and_tampered_parent_rejected(self):
        for value in (True,20.,'20',0,16,21):
            with self.assertRaises(ValueError):short_cap_plan(self.state,source_checkpoint=self.source,cap=value)
        self.activate(25);self.activate(20)
        self.gate['advance_policy']['source_gate']['last_train_loss']=123
        with self.assertRaises(ValueError):moe_training_horizon(self.state)


if __name__=='__main__':unittest.main()
