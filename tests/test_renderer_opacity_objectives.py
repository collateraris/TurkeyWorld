import json
import math
import unittest
import copy
from types import SimpleNamespace

import numpy as np

import torch
from torch import nn

from turkeyworld.renderer_opacity_objectives import opacity_depth_loss, opacity_depth_report


def expected_depth(alpha):
    survival=1-alpha
    inclusive=survival.cumprod(-1)
    transmission=torch.cat((torch.ones_like(inclusive[...,:1]),inclusive[...,:-1]),-1)
    centers=(torch.arange(64,dtype=alpha.dtype,device=alpha.device)+.5)*.375
    return (transmission*alpha*centers).sum(-1)+inclusive[...,-1]*24


class OpacityDepthTests(unittest.TestCase):
    def test_exact_first_hit_has_near_zero_loss_and_json_metrics(self):
        centers=(torch.arange(64)+.5)*.375
        indices=torch.tensor([0,2,8,16,24,36,50,63]*4)
        alpha=torch.zeros(1,32,64)
        alpha.scatter_(-1,indices[None,:,None],1.)
        labels={'depth':centers[indices][None,None,None]}
        loss,metrics=opacity_depth_loss({'opacity':alpha,'depth':expected_depth(alpha)},labels)
        self.assertLess(float(loss),1e-4)
        self.assertEqual(metrics['present_bands'],5)
        self.assertEqual(metrics['hit_bin_min'],0)
        self.assertEqual(metrics['hit_bin_max'],63)
        json.dumps(metrics,allow_nan=False)

    def test_free_prefix_moves_down_hit_moves_up_and_source_has_no_grad(self):
        alpha=torch.full((1,32,64),.2,requires_grad=True)
        target=torch.full((1,1,1,32),13.78,requires_grad=True)
        loss,metrics=opacity_depth_loss({'opacity':alpha,'depth':expected_depth(alpha)},{'depth':target})
        loss.backward()
        self.assertEqual(metrics['hit_bin_min'],36)
        self.assertTrue(bool((alpha.grad[:,:,0]>0).all()))
        self.assertTrue(bool((alpha.grad[:,:,36]<0).all()))
        self.assertIsNone(target.grad)
        self.assertTrue(bool(torch.isfinite(alpha.grad).all()))

    def test_close_and_deep_labels_have_correct_bins_and_no_empty_prefix_nan(self):
        target=torch.tensor([.172,13.78]*16).reshape(1,1,1,32)
        alpha=torch.full((1,32,64),.3,requires_grad=True)
        loss,metrics=opacity_depth_loss({'opacity':alpha,'depth':expected_depth(alpha)},{'depth':target})
        loss.backward()
        self.assertEqual(metrics['hit_bin_min'],0)
        self.assertEqual(metrics['hit_bin_max'],36)
        self.assertEqual(metrics['no_prefix_rays'],16)
        self.assertEqual(metrics['present_bands'],2)
        self.assertEqual(metrics['bands']['near_0_2']['prefix_samples'],0)
        self.assertTrue(math.isfinite(metrics['loss']))
        self.assertEqual(metrics['bands']['mid_2_5']['images'],0)

    def test_bands_have_equal_weight_and_images_have_equal_weight(self):
        targets=torch.full((2,1,1,32),1.)
        targets[0,0,0,0]=7.
        targets[1]=7.
        alpha=torch.full((2,32,64),.2,requires_grad=True)
        _,metrics=opacity_depth_loss({'opacity':alpha,'depth':expected_depth(alpha)},{'depth':targets})
        means=[row['loss']for row in metrics['bands'].values()if row['present']]
        self.assertAlmostEqual(metrics['loss'],sum(means)/len(means),places=6)
        self.assertEqual(metrics['bands']['near_0_2']['images'],1)
        self.assertEqual(metrics['bands']['far_5_8']['images'],2)

    def test_private_network_receives_finite_gradients_only(self):
        network=nn.Sequential(nn.Linear(5,32),nn.SiLU(),nn.Linear(32,1))
        source=torch.randn(2,32,64,5,requires_grad=True)
        alpha=network(source.detach()).squeeze(-1).sigmoid()
        target=torch.full((2,1,1,32),6.,requires_grad=True)
        loss,_=opacity_depth_loss({'opacity':alpha,'depth':expected_depth(alpha)},{'depth':target})
        loss.backward()
        self.assertIsNone(source.grad)
        self.assertIsNone(target.grad)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()for p in network.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum())for p in network.parameters()),0.)

    def test_termination_likelihood_sums_every_free_sample(self):
        alpha=torch.full((1,32,64),.2)
        target=torch.full((1,1,1,32),3.1875)  # center of bin8
        _,metrics=opacity_depth_loss({'opacity':alpha,'depth':expected_depth(alpha)},{'depth':target})
        self.assertAlmostEqual(metrics['hit_nll'],-math.log(.2),places=5)
        self.assertAlmostEqual(metrics['prefix_free_nll'],-8*math.log(.8),places=5)
        self.assertAlmostEqual(metrics['first_surface_nll'],-math.log(.2)-8*math.log(.8),places=5)
        # Altering samples AFTER the first surface creates no opacity labels.
        alpha[:,:,9:]=.99
        _,changed=opacity_depth_loss({'opacity':alpha,'depth':expected_depth(alpha)},{'depth':target})
        self.assertEqual(changed['first_surface_nll'],metrics['first_surface_nll'])

    def test_invalid_depth_probability_and_shapes_fail_closed(self):
        alpha=torch.full((1,32,64),.2)
        for bad in (0.,float('nan'),25.):
            with self.assertRaises(ValueError):
                opacity_depth_loss({'opacity':alpha,'depth':expected_depth(alpha)},
                    {'depth':torch.full((1,1,1,32),bad)})

    def test_report_whole_panel_tail_rng_modes_sampler_and_no_grad(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__();self.logit=nn.Parameter(torch.tensor(-1.))
                self.child=nn.Linear(1,1);self.child.eval()
                self.config=SimpleNamespace(flow_geometry_opacity_enabled=True)
            def encode_map(self,grid):return grid.float()
            def render_geometry(self,state,fmap):
                alpha=self.logit.sigmoid().expand(len(state),32,64)
                torch.rand(1)  # Explicit global-RNG preservation regression.
                return {'depth':expected_depth(alpha),'opacity':alpha}
        class Data:
            split='val';identity={'fingerprint':'fixed_source_test'}
            def __init__(self):self.rng=np.random.default_rng(31);self.draws=4
            def panel_fingerprint(self,panel):return 'fixed_'+str(list(panel))
            def sampler_state(self):return {'rng':copy.deepcopy(self.rng.bit_generator.state),'draws':self.draws}
            def restore_sampler(self,state):self.rng.bit_generator.state=state['rng'];self.draws=state['draws']
            def batch(self,n,*,indices):
                from turkeyworld.engine import Game
                game=Game(seed=0,endless=True,enemies_enabled=False)
                states=torch.tensor(np.stack([game.state_vector()for _ in indices]))
                return {'state':states,'map':torch.tensor(np.stack([game.map for _ in indices])),
                    'object_mask':torch.zeros(n,1,32,32)}
        model=Model();data=Data();sampler=data.sampler_state();rng=torch.get_rng_state().clone()
        modes=[module.training for module in model.modules()]
        report=opacity_depth_report(model,data,[0,1,2,3,4],batch_size=2)
        self.assertEqual(report['frames'],5)
        self.assertEqual(report['rays'],160)
        self.assertEqual(report['images'],5)
        self.assertEqual(data.sampler_state(),sampler)
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        self.assertEqual([module.training for module in model.modules()],modes)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(report['private_opacity_enabled'])
        self.assertFalse(report['renderer_RGB_evaluated'])
        same=opacity_depth_report(model,data,[0,1,2,3,4],batch_size=4)
        self.assertEqual(report['loss'],same['loss'])
        self.assertEqual(report['bands'],same['bands'])
        data.split='train'
        with self.assertRaises(ValueError):opacity_depth_report(model,data,[0])


if __name__=='__main__':unittest.main()
