from copy import deepcopy
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from turkeyworld.dreamer_reward_continue import DreamerRewardContinue
from turkeyworld.dreamer_reward_validation import evaluate_reward_heads


def episode(path,seed,*,death=True):
    count=18;state=np.zeros((count+1,20),np.float32);state[:,4]=.08
    state[:,3]=1;state[:,5]=1;state[:,12:14]=[.6,1.]
    events=np.zeros((count,4),np.float32)
    if death:state[-1,4]=0;events[-1,3]=1
    np.savez_compressed(path,states=state,frames=np.zeros((count+1,32,32,3),np.uint8),
        actions=np.zeros((count,4),np.float32),events=events,hit_ids=np.full(count,-1,np.int64),
        attack_flags=events[:,3:4],map=np.zeros((16,16),np.int8),dt=np.asarray(1/30),seed=np.asarray(seed),split=np.asarray('val'))


class FakeAdapter:
    def __init__(self):
        self.world=nn.Linear(1,1)
        self.heads=DreamerRewardContinue(config=dict(feature_dim=20,hidden=8,depth=1,low=-math.log1p(2),high=math.log1p(2)))
        self.observed=[]
    def observe_batch(self,batch):
        self.observed.append(batch['start_indices'].tolist())
        torch.rand(1) # Prove that even stochastic posterior validation preserves TRAIN RNG.
        return torch.cat((batch['states'][:,:1],batch['target_states']),1)


class RewardValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.directory=Path(self.tmp.name)/'val';self.directory.mkdir()

    def test_real_death_and_same_file_low_HP_controls_report_support_overlap_and_no_gradients(self):
        episode(self.directory/'a.npz',1);episode(self.directory/'b.npz',2)
        adapter=FakeAdapter();before=deepcopy(adapter.heads.state_dict());rng=torch.get_rng_state().clone()
        report=evaluate_reward_heads(adapter,self.directory,horizon=16,batch_size=4)
        self.assertEqual(adapter.observed,[[2,0,2,0]])
        self.assertEqual(report['windows'],4);self.assertEqual(report['unique_Source_episodes'],2)
        self.assertEqual(report['window_transition_measurements'],64)
        self.assertEqual(report['unique_source_transitions'],36);self.assertEqual(report['duplicate_window_measurements'],28)
        self.assertEqual(report['metrics']['death_transitions'],2)
        self.assertEqual(report['by_role']['matched_low_HP_alive']['death_transitions'],0)
        self.assertEqual(report['metrics']['death_recall'],0.) # Initial continue.5 is not a successful death classifier.
        self.assertEqual(report['metrics']['reward_target_clipped_count'],0)
        self.assertFalse(report['population_prevalence_estimate']);self.assertFalse(report['gameplay_qualified'])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertTrue(adapter.heads.training)
        self.assertTrue(all(torch.equal(value,adapter.heads.state_dict()[name])for name,value in before.items()))
        self.assertTrue(all(p.grad is None for p in adapter.heads.parameters()))

    def test_missing_terminal_or_wrong_split_never_infers_perfect_recall(self):
        episode(self.directory/'alive.npz',1,death=False)
        with self.assertRaisesRegex(ValueError,'No real held-out deaths'):
            evaluate_reward_heads(FakeAdapter(),self.directory)
        with self.assertRaisesRegex(ValueError,'explicit VAL'):
            evaluate_reward_heads(FakeAdapter(),self.directory.parent)

    def test_file_changed_during_measurement_fails_identity_check_and_restores_rng(self):
        path=self.directory/'a.npz';episode(path,1);adapter=FakeAdapter();original=adapter.observe_batch
        def observe(batch):
            result=original(batch)
            with path.open('ab')as stream:stream.write(b'changed')
            return result
        adapter.observe_batch=observe;rng=torch.get_rng_state().clone()
        with self.assertRaisesRegex(ValueError,'changed during calibration'):
            evaluate_reward_heads(adapter,self.directory)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))


if __name__=='__main__':unittest.main()
