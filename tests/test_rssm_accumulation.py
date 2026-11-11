"""Real gradient/Adam checks; pooled equivalence here uses separable MSE only.

Actual RSSM batch-statistic/class-balanced terms remain microbatch32 objectives.
"""
import copy
import unittest

import torch
from torch import nn

from turkeyworld.gradient_safety import clip_finite_grad_norm_
from turkeyworld.rssm_accumulation import accumulated_metrics, average_accumulated_gradients


class RSSMAccumulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_four_complete32_backwards_mean_and_one_adam_match_separable_full128(self):
        with torch.random.fork_rng():
            torch.manual_seed(173)
            first=nn.Sequential(nn.Linear(5,9),nn.Tanh(),nn.Linear(9,2)).double()
            second=copy.deepcopy(first)
            inputs=torch.randn(128,5,dtype=torch.float64);targets=torch.randn(128,2,dtype=torch.float64)
        optimizers=[torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01) for model in [first,second]]
        # Start from nonempty Adam moments to test continuation, not just step1.
        for model,optimizer in zip([first,second],optimizers):
            optimizer.zero_grad(set_to_none=True)
            (model(inputs[:8])-targets[:8]).square().mean().backward();optimizer.step()
        optimizers[0].zero_grad(set_to_none=True)
        grad_pointers=[];losses=[]
        for index in range(4):
            part=slice(index*32,(index+1)*32)
            value=(first(inputs[part])-targets[part]).square().mean();value.backward()
            losses.append(float(value.detach()));grad_pointers.append([p.grad.data_ptr() for p in first.parameters()])
        self.assertTrue(all(row==grad_pointers[0] for row in grad_pointers))
        average_accumulated_gradients(first.parameters(),4)
        optimizers[1].zero_grad(set_to_none=True)
        full_loss=(second(inputs)-targets).square().mean();full_loss.backward()
        self.assertAlmostEqual(sum(losses)/4,float(full_loss.detach()),places=13)
        for a,b in zip(first.parameters(),second.parameters()):torch.testing.assert_close(a.grad,b.grad,rtol=1e-12,atol=1e-12)
        norms=[clip_finite_grad_norm_(model.named_parameters(),max_norm=.2) for model in [first,second]]
        self.assertAlmostEqual(float(norms[0]),float(norms[1]),places=12)
        for optimizer in optimizers:optimizer.step()
        for a,b in zip(first.parameters(),second.parameters()):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
        for a,b in zip(optimizers[0].state.values(),optimizers[1].state.values()):
            self.assertEqual(float(a['step']),2)
            for key in a:torch.testing.assert_close(a[key],b[key],rtol=1e-12,atol=1e-12)

    def test_gradient_average_does_not_clear_detach_reallocate_or_touch_missing_gradients(self):
        parameter=nn.Parameter(torch.tensor(2.,dtype=torch.float64));unused=nn.Parameter(torch.tensor(7.))
        accumulated=[]
        for coefficient in [1.,2.,3.,4.]:
            (coefficient*parameter.square()).backward();accumulated.append(float(parameter.grad))
        self.assertEqual(accumulated,[4.,12.,24.,40.]);before=parameter.grad
        average_accumulated_gradients([parameter,unused],4)
        self.assertIs(parameter.grad,before);self.assertEqual(float(parameter.grad),10.);self.assertIsNone(unused.grad)
        average_accumulated_gradients([parameter],1);self.assertEqual(float(parameter.grad),10.)
        for invalid in [0,-1,True,4.,None]:
            with self.assertRaises(ValueError):average_accumulated_gradients([parameter],invalid)
        self.assertEqual(float(parameter.grad),10.)

    def test_losses_mean_named_counts_sum_and_actor_ids_remain_microbatch_local(self):
        rows=[{'train_loss':float(i+1),'rssm_aux_loss':.1*(i+1),'rssm_aux_combat_source_hit_count':i,
            'rssm_combat_branch_branches':i,'rssm_combat_branch_backward_calls':int(i>0),
            'rssm_combat_branch_forward_transitions':16*i,'rssm_combat_branch_source_probes':16,
            'rssm_combat_branch_selected_cohorts':{'spawn':i,'attack':1},
            'rssm_combat_branch_selected_anchors':[{'actor':0,'step':i+1,'cohort':'spawn'}],
            'all_finite':True,'source_future_feedback':False,'mode':'argmax'} for i in range(4)]
        untouched=copy.deepcopy(rows);result=accumulated_metrics(rows)
        self.assertEqual(rows,untouched);self.assertEqual(result['train_loss'],2.5)
        self.assertAlmostEqual(result['rssm_aux_loss'],.25)
        self.assertEqual(result['rssm_aux_combat_source_hit_count'],6)
        self.assertEqual(result['rssm_combat_branch_branches'],6)
        self.assertEqual(result['rssm_combat_branch_backward_calls'],3)
        self.assertEqual(result['rssm_combat_branch_forward_transitions'],96)
        self.assertEqual(result['rssm_combat_branch_source_probes'],64)
        self.assertEqual(result['rssm_combat_branch_selected_cohorts'],{'spawn':6,'attack':4})
        self.assertEqual([v['microbatch_index'] for v in result['rssm_combat_branch_selected_anchors']],[1,2,3,4])
        self.assertEqual([v['actor'] for v in result['rssm_combat_branch_selected_anchors']],[0]*4)
        self.assertFalse(result['source_future_feedback']);self.assertTrue(result['all_finite'])
        self.assertEqual(result['microbatch_metrics'],rows)
        result['microbatch_metrics'][0]['train_loss']=99;self.assertEqual(rows[0]['train_loss'],1.)
        self.assertIn('train_loss',result['metric_aggregation']['averaged_keys'])
        self.assertIn('rssm_combat_branch_branches',result['metric_aggregation']['summed_keys'])

    def test_optional_branch_counts_sum_even_when_first_or_last_microbatch_has_no_branch(self):
        rows=[{'train_loss':1.,'rssm_combat_branch_branches':0},
            {'train_loss':2.,'rssm_combat_branch_branches':2,'rssm_combat_branch_source_hit_count':3,
             'rssm_combat_branch_anchor_source_counts':{'hit':2,'spawn':1},'optional_branch_xy_loss':.7},
            {'train_loss':3.,'rssm_combat_branch_branches':1,'rssm_combat_branch_source_hit_count':2,
             'rssm_combat_branch_anchor_source_counts':{'spawn':2}},
            {'train_loss':4.,'rssm_combat_branch_branches':0}]
        result=accumulated_metrics(rows)
        self.assertEqual(result['rssm_combat_branch_branches'],3)
        self.assertEqual(result['rssm_combat_branch_source_hit_count'],5)
        self.assertEqual(result['rssm_combat_branch_anchor_source_counts'],{'hit':2,'spawn':3})
        self.assertEqual(result['train_loss'],2.5)
        self.assertNotIn('optional_branch_xy_loss',result['metric_aggregation']['averaged_keys'])
        self.assertIn('optional_branch_xy_loss',result['metric_aggregation']['variable_keys_not_averaged'])
        self.assertEqual(result['microbatch_metrics'][1]['optional_branch_xy_loss'],.7)

    def test_nonfinite_objectives_or_count_dictionaries_fail_before_an_optimizer_can_commit(self):
        for bad in [float('nan'),float('inf'),-float('inf')]:
            for key,value in [('train_loss',bad),('source_hit_count',bad),('selected_cohorts',{'spawn':bad})]:
                with self.subTest(key=key,bad=bad):
                    with self.assertRaises(FloatingPointError):
                        accumulated_metrics([{key:value}]+[{key:{'spawn':1} if isinstance(value,dict) else 1.}]*3)
        with self.assertRaises(ValueError):accumulated_metrics([])


if __name__=='__main__':unittest.main()
