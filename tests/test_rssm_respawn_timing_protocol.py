from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from turkeyworld.rssm_combat_memory_protocol import file_sha
from turkeyworld.rssm_moe_protocol import (moe_training_plan, bind_moe_training,
    commit_moe_update, moe_training_horizon)
from turkeyworld.rssm_respawn_timing_protocol import (respawn_timing_plan,
    bind_respawn_timing, commit_respawn_timing)


class RespawnTimingProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.source = root/'source.pt'
        self.source.write_bytes(b'unchanged-complete-checkpoint')
        self.state = {'global_training_step': 5352, 'rssm_training': {}}
        plan = moe_training_plan(self.state, source_checkpoint=self.source,
            source_checkpoint_sha256=file_sha(self.source), architecture_migration_sha256='a'*64)
        p = root/'moe.json';p.write_text(json.dumps(plan), encoding='utf-8')
        bind_moe_training(self.state, plan, path=p, sha256=file_sha(p), loaded_checkpoint_sha256=file_sha(self.source))
        for _ in range(29):
            step = self.state['global_training_step']+1
            horizon = moe_training_horizon(self.state)
            self.state['global_training_step'] = step
            commit_moe_update(self.state, step=step, horizon=horizon, train_loss=.2)
        self.plan = respawn_timing_plan(self.state, source_checkpoint=self.source)
        self.path = root/'timing.json';self.path.write_text(json.dumps(self.plan), encoding='utf-8')

    def bind(self, loaded=None):
        return bind_respawn_timing(self.state, self.plan, path=self.path,
            sha256=file_sha(self.path), loaded_checkpoint_sha256=loaded or file_sha(self.source))

    def test_preserves_H16_nine_successes_and_next_update_can_advance_once(self):
        before = deepcopy(self.state['rssm_training']['moe_training'])
        self.assertEqual((before['current_horizon'], before['consecutive_below_threshold']), (16,9))
        revision = self.bind()
        self.assertEqual(before,self.state['rssm_training']['moe_training'])
        step = self.state['global_training_step']+1
        self.state['global_training_step'] = step
        commit_moe_update(self.state, step=step, horizon=16, train_loss=.2)
        commit_respawn_timing(self.state, step=step)
        self.assertEqual(moe_training_horizon(self.state),32)
        self.assertEqual(self.bind(loaded='b'*64)['updates'],1)
        self.assertEqual(revision['source_gate'],before)
        with self.assertRaises(ValueError):commit_respawn_timing(self.state,step=step)

    def test_rejects_wrong_source_gate_code_and_changed_source_file(self):
        with self.assertRaises(ValueError):self.bind(loaded='b'*64)
        with patch('turkeyworld.rssm_respawn_timing_protocol.code_fingerprints',return_value={}):
            with self.assertRaises(ValueError):self.bind()
        self.state['rssm_training']['moe_training']['consecutive_below_threshold']=8
        with self.assertRaises(ValueError):self.bind()
        self.state['rssm_training']['moe_training']['consecutive_below_threshold']=9
        self.bind();self.source.write_bytes(b'changed')
        with self.assertRaises(ValueError):self.bind()


if __name__=='__main__':unittest.main()
