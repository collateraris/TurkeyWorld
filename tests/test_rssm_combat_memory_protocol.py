from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from turkeyworld.rssm_moe_protocol import (moe_training_plan, bind_moe_training,
    commit_moe_update, moe_training_horizon)
from turkeyworld.rssm_combat_memory_protocol import (combat_memory_plan,
    bind_combat_memory, commit_combat_memory, file_sha)


class CombatMemoryProtocolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.state = {'global_training_step': 5352, 'rssm_training': {}}
        p = self.root/'moe.json'
        plan = moe_training_plan(self.state, source_checkpoint='source.pt',
            source_checkpoint_sha256='a'*64, architecture_migration_sha256='b'*64)
        p.write_text(json.dumps(plan), encoding='utf-8')
        bind_moe_training(self.state, plan, path=p, sha256=file_sha(p), loaded_checkpoint_sha256='a'*64)
        for _ in range(29):
            step = self.state['global_training_step']+1
            horizon = moe_training_horizon(self.state)
            self.state['global_training_step'] = step
            commit_moe_update(self.state, step=step, horizon=horizon, train_loss=.2)
        self.initial = self.root/'initial.pt'
        self.proof = self.root/'migration.json'
        self.source = self.root/'source.pt'
        for p, content in ((self.initial, b'initial'), (self.proof, b'proof'), (self.source, b'source')):
            p.write_bytes(content)
        self.plan = combat_memory_plan(self.state, initial_checkpoint=self.initial,
            migration_proof=self.proof, source_checkpoint=self.source,
            added_parameters=20_007_020, old_memory_size=3632, new_memory_size=5232)
        self.path = self.root/'combat.json'
        self.path.write_text(json.dumps(self.plan), encoding='utf-8')

    def bind(self, loaded_sha=None):
        return bind_combat_memory(self.state, self.plan, path=self.path,
            sha256=file_sha(self.path), loaded_checkpoint_sha256=loaded_sha or file_sha(self.initial))

    def test_preserves_existing_H16_nine_successes_and_tenth_advances_normally(self):
        before = deepcopy(self.state['rssm_training']['moe_training'])
        self.assertEqual((before['current_horizon'], before['consecutive_below_threshold']), (16, 9))
        revision = self.bind()
        self.assertEqual(before, self.state['rssm_training']['moe_training'])
        self.assertEqual(revision['updates'], 0)
        step = self.state['global_training_step']+1
        self.state['global_training_step'] = step
        gate = commit_moe_update(self.state, step=step, horizon=16, train_loss=.2)
        commit_combat_memory(self.state, step=step)
        self.assertEqual((gate['current_horizon'], gate['consecutive_below_threshold']), (32, 0))
        self.assertEqual(self.bind(loaded_sha='c'*64)['updates'], 1)
        with self.assertRaises(ValueError):
            commit_combat_memory(self.state, step=step)

    def test_wrong_initial_checkpoint_or_mutated_gate_cannot_activate(self):
        with self.assertRaises(ValueError):
            self.bind(loaded_sha='c'*64)
        self.state['rssm_training']['moe_training']['consecutive_below_threshold'] = 8
        with self.assertRaises(ValueError):
            self.bind()

    def test_changed_migration_or_initial_file_rejected_even_on_resume(self):
        self.bind()
        self.proof.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            self.bind()
        self.proof.write_bytes(b'proof')
        self.initial.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            self.bind()

    def test_rejects_wrong_measured_size_and_duplicate_or_skipped_commits(self):
        with self.assertRaises(ValueError):
            combat_memory_plan(self.state, initial_checkpoint=self.initial,
                migration_proof=self.proof, source_checkpoint=self.source,
                added_parameters=15_000_000, old_memory_size=3632, new_memory_size=5232)
        self.bind()
        with self.assertRaises(ValueError):
            commit_combat_memory(self.state, step=self.state['global_training_step'])
        self.state['global_training_step'] += 2
        with self.assertRaises(ValueError):
            commit_combat_memory(self.state, step=self.state['global_training_step'])


if __name__ == '__main__':
    unittest.main()
