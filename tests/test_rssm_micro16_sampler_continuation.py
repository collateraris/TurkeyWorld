"""Explicit AUX32→AUX16 continues real counters and immutable policy history."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from turkeyworld.engine import Game
from turkeyworld.rssm_moe_training import sample_moe_combat_batch
from turkeyworld.rssm_spawn_curriculum import MICRO16_CONTINUATION_PROTOCOL
from test_rssm_attack_positive_sampler import source_files,sampler


class Micro16ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder=TemporaryDirectory();cls.paths=source_files(Path(cls.folder.name)/'train')

    @classmethod
    def tearDownClass(cls):cls.folder.cleanup()

    def prepared(self):
        chosen=sampler(self.paths);rng=np.random.default_rng(173)
        chosen.sample(64,rng)
        for _ in range(3):chosen.sample(16,rng)
        for _ in range(2):chosen.sample(32,rng,allow_micro32_continuation=True)
        return chosen,rng

    def test_explicit_transition_keeps_old_ledgers_counters_and_actual_balanced16_rows(self):
        chosen,rng=self.prepared();source=deepcopy(chosen.state_dict());clock=deepcopy(rng.bit_generator.state)
        with self.assertRaisesRegex(ValueError,'silently disable'):chosen.sample(16,rng)
        self.assertEqual(source,chosen.state_dict());self.assertEqual(clock,rng.bit_generator.state)
        rows=[]
        with(patch.object(Game,'step',side_effect=AssertionError('No Source step during sampling')),
                patch.object(Game,'render',side_effect=AssertionError('No Source future RGB'))):
            for index in range(4):
                pairs=chosen.sample(16,rng,allow_micro16_continuation=True);rows.append(deepcopy(chosen.last_sample))
                self.assertEqual(pairs.shape,(16,2));self.assertEqual(chosen.last_sample['runtime_batch_size'],16)
                self.assertEqual(chosen.last_sample['quota'],dict(spawn=4,attack_negative=4,attack_positive=4,shooting_replay=4))
                self.assertEqual(chosen.last_sample['micro16_continuation_policy'],MICRO16_CONTINUATION_PROTOCOL)
                self.assertEqual(chosen.last_sample['micro16_continuation_batch_index'],index)
                self.assertNotIn('micro32_sampling_policy',chosen.last_sample)
        saved=chosen.state_dict();ledger=saved['micro16_continuation']
        self.assertEqual(ledger['source_sampler_state'],source)
        self.assertEqual(ledger['source_sampler_state_sha256'],hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest())
        self.assertEqual((ledger['draws'],ledger['batches']),(64,4))
        for key in ('micro16_sampling','micro32_sampling','retained_legacy_ledger'):
            self.assertEqual(saved[key],source[key])
        for key,delta in dict(draws=64,v2_draws=64,negative_draws=16,positive_draws=16).items():
            self.assertEqual(saved[key],source[key]+delta)
        self.assertEqual(sum(saved['anchor_counts'].values())-sum(source['anchor_counts'].values()),16)
        totals=Counter()
        for row in rows:totals.update(row['target_subquotas'])
        self.assertEqual(totals,dict(hit_near=4,hit_mid=4,hit_far=4,fired_miss=2,legacy=2))
        self.assertEqual(sum(row['fired_miss_subquotas']['wall_miss']for row in rows),1)
        self.assertEqual(sum(row['fired_miss_subquotas']['other_miss']for row in rows),1)
        restored=sampler(self.paths);restored.restore_state(saved)

    def test_json_resume_reproduces_next_rows_receipt_rng_and_source_history(self):
        chosen,rng=self.prepared()
        for _ in range(3):chosen.sample(16,rng,allow_micro16_continuation=True)
        saved=json.loads(json.dumps(chosen.state_dict()));clock=deepcopy(rng.bit_generator.state)
        expected=chosen.sample(16,rng,allow_micro16_continuation=True)
        restored=sampler(self.paths);restored.restore_state(saved)
        other=np.random.default_rng();other.bit_generator.state=clock
        np.testing.assert_array_equal(expected,restored.sample(16,other,allow_micro16_continuation=True))
        self.assertEqual(chosen.last_sample,restored.last_sample)
        self.assertEqual(json.loads(json.dumps(chosen.state_dict())),json.loads(json.dumps(restored.state_dict())))
        self.assertEqual(rng.bit_generator.state,other.bit_generator.state)

    def test_no_silent_policy_switch_and_corrupt_restore_is_transactional(self):
        chosen,rng=self.prepared();chosen.sample(16,rng,allow_micro16_continuation=True)
        valid=deepcopy(chosen.state_dict());clock=deepcopy(rng.bit_generator.state)
        for count,flags in ((16,{}),(32,{}),(32,{'allow_micro32_continuation':True}),
                (64,{}),(32,{'allow_micro16_continuation':True})):
            with self.subTest(count=count,flags=flags),self.assertRaises(ValueError):chosen.sample(count,rng,**flags)
            self.assertEqual(valid,chosen.state_dict());self.assertEqual(clock,rng.bit_generator.state)
        corrupt=[]
        for key in ('draws','batches','rotation_base_batches','source_sampler_state_sha256'):
            bad=deepcopy(valid);bad['micro16_continuation'][key]='wrong'if key.endswith('sha256')else bad['micro16_continuation'][key]+1
            corrupt.append(bad)
        bad=deepcopy(valid);bad['micro32_sampling']['draws']+=32;corrupt.append(bad)
        bad=deepcopy(valid);bad['v2_draws']+=16;corrupt.append(bad)
        for bad in corrupt:
            with self.assertRaises(ValueError):chosen.restore_state(bad)
            self.assertEqual(valid,chosen.state_dict())

    def test_wrapper_requires_explicit_new_policy_and_uses_actual16_source_fragments(self):
        chosen,rng=self.prepared();chosen.moe_identity={'fingerprint':'a'*64};calls=[]
        def batch(count,**kwargs):calls.append((count,kwargs));return {'pairs':kwargs['indices']}
        data=SimpleNamespace(split='train',batch=batch)
        with self.assertRaisesRegex(ValueError,'silently disable'):
            sample_moe_combat_batch(chosen,data,rng,horizon=450,batch_size=16)
        result,receipt=sample_moe_combat_batch(chosen,data,rng,horizon=450,batch_size=16,allow_micro16_continuation=True)
        self.assertEqual(len(calls),1);self.assertEqual(calls[0][0],16)
        self.assertEqual(calls[0][1]['context'],1);self.assertEqual(calls[0][1]['horizon'],16)
        np.testing.assert_array_equal(result['pairs'],np.asarray(receipt['pairs']))
        self.assertEqual(receipt['used_aux_fragments'],16);self.assertEqual(receipt['runtime_batch_size'],16)
        self.assertEqual(receipt['micro16_continuation_policy'],MICRO16_CONTINUATION_PROTOCOL)
        json.dumps(receipt,allow_nan=False)


if __name__=='__main__':unittest.main()
