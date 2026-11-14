from copy import deepcopy
import unittest

import torch
from torch.nn import functional as F

from turkeyworld.dreamer_reward_continue import(DreamerRewardContinue,RewardContinueConfig,
    source_reward_continue,reward_continue_loss,reward_settings)


def transition(hp=1.,next_hp=1.,kills=0.,next_kills=0.,events=None):
    before=torch.zeros(1,20);before[:,4]=hp;before[:,18]=kills;after=before.clone();after[:,4]=next_hp;after[:,18]=next_kills
    return before,after,torch.tensor([events or [0.,0.,0.,0.]])


class RewardContinueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_recorded_kill_hit_HP_and_new_death_give_declared_reward_no_invented_nav_reward(self):
        cases=[(transition(),0.,1.),(transition(events=[1,1,0,0]),.1,1.),
            (transition(kills=3,next_kills=4,events=[1,1,1,0]),1.1,1.),
            (transition(next_hp=.92,events=[0,0,0,1]),-.08,1.),
            (transition(hp=.04,next_hp=0,events=[0,0,0,1]),-1.04,0.)]
        for (before,after,events),reward,continues in cases:
            labels=source_reward_continue(before,after,events)
            self.assertAlmostEqual(float(labels['reward']),reward,places=5)
            self.assertEqual(float(labels['continue_target']),continues)
        before,after,events=transition();after[:,:2]=.8
        self.assertEqual(float(source_reward_continue(before,after,events)['reward']),0.)

    def test_time_limit_remains_bootstrap_and_postdeath_padding_is_not_training_mass(self):
        before,after,events=transition();after[:,19]=15/300
        labels=source_reward_continue(before,after,events,truncated=torch.tensor([True]))
        self.assertEqual(float(labels['continue_target']),1);self.assertFalse(bool(labels['terminated'][0]))
        labels=source_reward_continue(*transition(hp=0,next_hp=0))
        self.assertFalse(bool(labels['valid'][0]));self.assertEqual(float(labels['reward']),0)

    def test_missing_or_contradictory_events_reset_and_healing_fail_closed(self):
        for data in(transition(next_kills=1),transition(kills=1,next_kills=0),transition(hp=.5,next_hp=1),
                    transition(events=[0,.5,0,0])):
            with self.assertRaises(ValueError):source_reward_continue(*data)
        a,b,e=transition()
        with self.assertRaises(ValueError):source_reward_continue(a,b,None)

    def test_heads_fit_only_sidecar_parameters_without_gradients_into_world_features_or_labels(self):
        heads=DreamerRewardContinue(12,hidden=16);features=torch.randn(4,12,requires_grad=True)
        before=torch.zeros(4,20);before[:,4]=1;after=before.clone();after[1,4]=0
        events=torch.zeros(4,4);events[1,3]=1
        labels=source_reward_continue(before,after,events);labels['reward'].requires_grad_()
        optimizer=torch.optim.Adam(heads.parameters(),lr=.001)
        for _ in range(2):
            optimizer.zero_grad();out=heads(features);loss,metrics=reward_continue_loss(out,labels);loss.backward();optimizer.step()
        self.assertIsNone(features.grad);self.assertIsNone(labels['reward'].grad)
        self.assertGreater(float(heads.trunk[0].weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all())for p in heads.parameters()))
        self.assertEqual(metrics['death_transitions'],1);self.assertEqual(metrics['valid_transitions'],4)
        self.assertEqual(metrics['reward_target_clipped_count'],0)
        self.assertEqual(metrics['reward_target_target_count'],4)
        prediction=heads.predict(features)
        self.assertEqual(prediction['reward'].shape,(4,));self.assertTrue(bool(((prediction['continue_prob']>=0)&(prediction['continue_prob']<=1)).all()))

    def test_importance_weights_use_calibrated_weighted_mean_not_implicit_class_balance(self):
        heads=DreamerRewardContinue(8,hidden=8);out=heads(torch.zeros(2,8))
        out['continue_logit']=torch.tensor([2.,-1.],requires_grad=True)
        before=torch.zeros(2,20);before[:,4]=1;after=before.clone();after[1,4]=0
        labels=source_reward_continue(before,after,torch.zeros(2,4));weights=torch.tensor([1.,9.])
        _,metrics=reward_continue_loss(out,labels,weights)
        expected=(F.binary_cross_entropy_with_logits(out['continue_logit'],labels['continue_target'],reduction='none')*weights).sum()/weights.sum()
        self.assertAlmostEqual(metrics['continue_loss'],float(expected.detach()),places=6)
        self.assertFalse(metrics['implicit_class_balancing']);self.assertEqual(metrics['death_recall'],1.)
        self.assertAlmostEqual(metrics['effective_weighted_samples'],100/82,places=6)

    def test_config_roundtrip_source_weights_detached_and_no_death_metric_is_not_perfect_recall(self):
        heads=DreamerRewardContinue(config=dict(feature_dim=9,hidden=8,depth=1,bins=31))
        restored=DreamerRewardContinue(config=heads.metadata()['config']);restored.load_state_dict(heads.state_dict())
        features=torch.randn(1,3,9);out=heads(features);before=torch.zeros(1,3,20);before[...,4]=1
        labels=source_reward_continue(before,before.clone(),torch.zeros(1,3,4))
        _,metrics=reward_continue_loss(out,labels)
        self.assertIsNone(metrics['death_recall']);self.assertEqual(metrics['death_transitions'],0)
        self.assertTrue(torch.equal(heads.predict(features)['reward'],restored.predict(features)['reward']))
        altered=reward_settings();altered['kill']=2.
        self.assertEqual(reward_settings(altered)['kill'],2.)
        with self.assertRaises(ValueError):reward_continue_loss(out,labels,torch.full((1,3),-1.))


if __name__=='__main__':unittest.main()
