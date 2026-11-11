"""Opt-in branch task weights preserve the legacy objective and its gradients."""
import copy
import unittest
from unittest.mock import patch

import torch

from turkeyworld.rssm_loss_balance import GroupLossAccumulator
from turkeyworld.rssm_contact_branches import contact_branches_backward,select_contact_branches
from turkeyworld.rssm_combat_branches import combat_branches_backward,select_combat_branches
from turkeyworld.rssm_combat_protocol import environment_profile
from test_rssm_contact_branches import BranchWorld,fixture as contact_fixture
from test_rssm_combat_branches import fixture as combat_fixture


def gradients(model):
    return {name:torch.zeros_like(parameter)if parameter.grad is None else parameter.grad.detach().clone()
            for name,parameter in model.named_parameters()if parameter.requires_grad}


class OnlyGroup:
    def __init__(self,name):self.name=name;self.parts=None
    def combine(self,parts,**kwargs):
        self.parts=parts
        return parts[self.name]


class BranchGroupLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_contact_all_player_unit_weights_exact_and_priority_scales_actual_backward(self):
        original=BranchWorld();candidate=contact_fixture(original)
        chosen,selection=select_contact_branches([candidate],contact_branch_protocol={})
        models=[copy.deepcopy(original)for _ in range(3)]
        collectors=[None,GroupLossAccumulator(),GroupLossAccumulator(priorities={'player':3.})]
        results=[]
        with patch('turkeyworld.rssm_contact_branches.select_contact_branches',return_value=(chosen,selection)):
            for model,collector in zip(models,collectors):
                results.append(contact_branches_backward(model,[candidate],contact_branch_protocol={},loss_groups=collector))
        a,b,c=map(gradients,models)
        for name in a:
            torch.testing.assert_close(a[name],b[name],rtol=0,atol=0)
            torch.testing.assert_close(c[name],3*a[name],rtol=2e-6,atol=2e-6)
        self.assertEqual(results[0]['loss'],results[1]['loss'])
        self.assertEqual(results[0]['loss'],results[2]['loss'])
        self.assertAlmostEqual(results[2]['balanced_loss'],3*results[0]['loss'],places=5)
        report=collectors[1].report()
        self.assertAlmostEqual(report['player']['raw'],results[0]['loss'],places=6)
        self.assertEqual(report['enemy']['raw'],0);self.assertEqual(report['interaction']['raw'],0)
        self.assertEqual(collectors[1].calls,1)

    def test_combat_unit_preserves_all_old_protocol_factors_and_custom_weights_match_group_gradients(self):
        original,candidate=combat_fixture()
        protocol={'weight':.35,'xy_weight':.4,'state_weight':1.7,'wall_weight':.2,'anchor_combat_weight':.9}
        chosen,selection=select_combat_branches([candidate],combat_branch_protocol=protocol)
        self.assertTrue(chosen)
        models=[copy.deepcopy(original)for _ in range(6)]
        custom=GroupLossAccumulator(priorities={'player':2.,'enemy':3.,'interaction':4.})
        collectors=[None,GroupLossAccumulator(),OnlyGroup('player'),OnlyGroup('enemy'),OnlyGroup('interaction'),custom]
        results=[]
        with patch('turkeyworld.rssm_combat_branches.select_combat_branches',return_value=(chosen,selection)):
            for model,collector in zip(models,collectors):
                with environment_profile(model,navigation_only=False):
                    results.append(combat_branches_backward(model,[candidate],combat_branch_protocol=protocol,loss_groups=collector))
        raw,unit,player,enemy,interaction,weighted=map(gradients,models)
        for name in raw:
            torch.testing.assert_close(raw[name],unit[name],rtol=3e-5,atol=2e-7,msg=name)
            torch.testing.assert_close(weighted[name],2*player[name]+3*enemy[name]+4*interaction[name],
                rtol=3e-5,atol=4e-7,msg=name)
        self.assertGreater(sum(float(v.abs().sum())for v in player.values()),0)
        self.assertGreater(sum(float(v.abs().sum())for v in enemy.values()),0)
        self.assertGreater(sum(float(v.abs().sum())for v in interaction.values()),0)
        for result in results[1:]:self.assertAlmostEqual(result['loss'],results[0]['loss'],places=6)
        self.assertAlmostEqual(results[1]['balanced_loss'],results[0]['loss'],places=5)
        self.assertAlmostEqual(collectors[1].report()['total'],results[0]['loss'],places=5)
        self.assertEqual(collectors[1].calls,1)
        for group in ('player','enemy','interaction'):self.assertGreater(collectors[1].report()[group]['raw'],0)


if __name__=='__main__':unittest.main()
