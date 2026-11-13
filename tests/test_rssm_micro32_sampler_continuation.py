"""AUX32 is an explicit continuation of the saved AUX16 Source/RNG ledger."""
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
from turkeyworld.rssm_spawn_curriculum import MICRO32_CONTINUATION_PROTOCOL
from test_rssm_attack_positive_sampler import source_files, sampler


class Micro32SamplerContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder=TemporaryDirectory()
        cls.paths=source_files(Path(cls.folder.name)/'train')

    @classmethod
    def tearDownClass(cls):cls.folder.cleanup()

    def prepared(self):
        selected=sampler(self.paths);rng=np.random.default_rng(173)
        selected.sample(64,rng)
        # Deliberately leave an odd number of B16 draws: the absolute draw
        # clock is not divisible by32 and cannot be reset or rounded.
        for _ in range(3):selected.sample(16,rng)
        return selected,rng

    def test_explicit_transition_retains_source_cases_quota_rotation_and_all_counters(self):
        selected,rng=self.prepared();source=deepcopy(selected.state_dict())
        hashes=[hashlib.sha256(path.read_bytes()).hexdigest()for path in self.paths]
        reference=deepcopy(selected);reference.micro16_sampling=None
        other=np.random.default_rng();other.bit_generator.state=deepcopy(rng.bit_generator.state)
        with patch.object(Game,'step',side_effect=AssertionError('No Source simulation in sampling')),\
             patch.object(Game,'render',side_effect=AssertionError('No Source RGB in sampling')):
            expected=reference.sample(32,other)
            actual=selected.sample(32,rng,allow_micro32_continuation=True)
        np.testing.assert_array_equal(actual,expected)
        self.assertEqual(rng.bit_generator.state,other.bit_generator.state)
        row=selected.last_sample
        self.assertEqual(row['quota'],dict(spawn=8,attack_negative=8,attack_positive=8,shooting_replay=8))
        self.assertEqual(row['attack_subquotas'],dict(positive_near_wall=4,positive_open=4,negative_occluded=6,negative_other=2))
        self.assertEqual(row['target_subquotas'],dict(hit_near=2,hit_mid=2,hit_far=2,fired_miss=1,legacy=1))
        self.assertEqual(row['fired_miss_subquotas'],reference.last_sample['fired_miss_subquotas'])
        saved=selected.state_dict();segment=saved['micro32_sampling']
        self.assertEqual(segment['kind'],MICRO32_CONTINUATION_PROTOCOL)
        self.assertEqual(segment['source_sampler_state'],source)
        self.assertEqual(segment['draws'],32);self.assertEqual(segment['batches'],1)
        self.assertEqual(saved['micro16_sampling'],source['micro16_sampling'])
        for key,delta in dict(draws=32,v2_draws=32,negative_draws=8,positive_draws=8).items():
            self.assertEqual(saved[key],source[key]+delta)
        self.assertEqual(saved['retained_legacy_ledger'],source['retained_legacy_ledger'])
        self.assertEqual(saved['anchor_counts'],reference.state_dict()['anchor_counts'])
        self.assertEqual(sum(saved['anchor_counts'].values())-sum(source['anchor_counts'].values()),8)
        self.assertLessEqual(max(saved['anchor_counts'].values())-min(saved['anchor_counts'].values()),1)
        self.assertEqual(hashes,[hashlib.sha256(path.read_bytes()).hexdigest()for path in self.paths])

    def test_json_resume_keeps_next_rows_receipt_PCG_and_historical_micro16_exact(self):
        selected,rng=self.prepared()
        for _ in range(2):selected.sample(32,rng,allow_micro32_continuation=True)
        state=json.loads(json.dumps(selected.state_dict()));clock=deepcopy(rng.bit_generator.state)
        expected=selected.sample(32,rng,allow_micro32_continuation=True)
        restored=sampler(self.paths);restored.restore_state(state)
        other=np.random.default_rng();other.bit_generator.state=clock
        np.testing.assert_array_equal(expected,restored.sample(32,other,allow_micro32_continuation=True))
        self.assertEqual(selected.last_sample,restored.last_sample)
        self.assertEqual(json.loads(json.dumps(selected.state_dict())),json.loads(json.dumps(restored.state_dict())))
        self.assertEqual(rng.bit_generator.state,other.bit_generator.state)
        self.assertEqual(restored.state_dict()['micro32_sampling']['draws'],96)

    def test_switch_requires_explicit_optin_and_tampered_lineage_restore_is_transactional(self):
        selected,rng=self.prepared();before=deepcopy(selected.state_dict());clock=deepcopy(rng.bit_generator.state)
        with self.assertRaisesRegex(ValueError,'silently disable'):selected.sample(32,rng)
        self.assertEqual(before,selected.state_dict());self.assertEqual(clock,rng.bit_generator.state)
        selected.sample(32,rng,allow_micro32_continuation=True);valid=deepcopy(selected.state_dict())
        for count in(16,64):
            with self.assertRaisesRegex(ValueError,'silently disable'):selected.sample(count,rng)
            self.assertEqual(valid,selected.state_dict())
        bad_states=[]
        for key in('draws','batches','source_sampler_state_sha256'):
            broken=deepcopy(valid);record=broken['micro32_sampling']
            record[key]='wrong' if key.endswith('sha256')else record[key]+1;bad_states.append(broken)
        broken=deepcopy(valid);broken['micro16_sampling']['draws']+=16;bad_states.append(broken)
        broken=deepcopy(valid);source=broken['micro32_sampling']['source_sampler_state']
        source['micro16_sampling']['batches']+=1
        broken['micro32_sampling']['source_sampler_state_sha256']=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()
        bad_states.append(broken)
        for broken in bad_states:
            with self.assertRaises(ValueError):selected.restore_state(broken)
            self.assertEqual(valid,selected.state_dict())

    def test_wrapper32_activates_continuation_and_uses_actual32_Source_fragments(self):
        selected,rng=self.prepared();selected.moe_identity={'fingerprint':'a'*64,'summary':selected.summary(batch_size=64)}
        identity=deepcopy(selected.moe_identity);calls=[]
        def source_batch(size,**kwargs):
            calls.append((size,deepcopy(kwargs)))
            return {'actual_Source_pairs':kwargs['indices']}
        data=SimpleNamespace(split='train',batch=source_batch)
        source=deepcopy(selected.state_dict())
        result,receipt=sample_moe_combat_batch(selected,data,rng,horizon=144,batch_size=32)
        self.assertEqual(calls[0][0],32);self.assertEqual(calls[0][1]['context'],1)
        self.assertEqual(calls[0][1]['horizon'],16)
        np.testing.assert_array_equal(result['actual_Source_pairs'],np.asarray(receipt['pairs']))
        self.assertEqual(receipt['used_aux_fragments'],32)
        self.assertEqual(receipt['runtime_batch_size'],32)
        self.assertEqual(receipt['micro32_sampling_policy'],MICRO32_CONTINUATION_PROTOCOL)
        self.assertTrue(receipt['training_cases_used']);self.assertFalse(receipt['index_reservation_only'])
        self.assertEqual(selected.state_dict()['micro32_sampling']['source_sampler_state'],source)
        self.assertEqual(selected.moe_identity,identity)
        json.dumps(receipt,allow_nan=False)


if __name__=='__main__':unittest.main()
