import unittest
from unittest.mock import patch

import torch

from test_renderer_enemy_objectives import tiny_model, batch
from turkeyworld.object_losses import balanced_image_l1
from turkeyworld.renderer_enemy_objectives import balanced_base_regression_loss, enemy_refinement_loss


class RendererBasePreservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_region_and_example_regressions_cannot_cancel_and_labels_are_detached(self):
        target=torch.zeros(2,3,2,2,requires_grad=True)
        base=torch.full_like(target,.2,requires_grad=True)
        values=torch.zeros_like(target);values[0,:,0,0]=.4;values[1]=.4;values[1,:,0,0]=0
        prediction=values.requires_grad_(True)
        mask=torch.zeros(2,1,2,2,requires_grad=True)
        with torch.no_grad():mask[:,:,0,0]=1
        guard=balanced_base_regression_loss(prediction,target,base,mask)
        self.assertAlmostEqual(float(guard.detach()),.1,places=6)
        guard.backward()
        self.assertGreater(float(prediction.grad.abs().sum()),0)
        self.assertIsNone(base.grad);self.assertIsNone(target.grad);self.assertIsNone(mask.grad)
        improved=torch.full_like(target,.1,requires_grad=True)
        zero=balanced_base_regression_loss(improved,target,base,mask)
        self.assertEqual(float(zero.detach()),0);zero.backward()
        self.assertEqual(float(improved.grad.abs().sum()),0)
        tied=base.detach().clone().requires_grad_(True)
        balanced_base_regression_loss(tied,target,base,mask).backward()
        self.assertEqual(float(tied.grad.abs().sum()),0)
        background=balanced_base_regression_loss(torch.full_like(base,.3),target,base,torch.zeros_like(mask))
        self.assertAlmostEqual(float(background.detach()),.1,places=6)

    def test_old_defaults_are_exact_and_common_objective_does_not_reward_a_weight_change(self):
        model=tiny_model();source=batch((False,True))
        with torch.no_grad():model.flow_renderer.output.weight.fill_(.002)
        default,_=enemy_refinement_loss(model,source,latent_weight=.1)
        explicit,a=enemy_refinement_loss(model,source,latent_weight=.1,detach_base=True,base_regression_weight=0.)
        corrected,b=enemy_refinement_loss(model,source,latent_weight=.03,detach_base=False,base_regression_weight=1.)
        self.assertTrue(torch.equal(default,explicit))
        self.assertEqual(a['common_rgb_objective'],b['common_rgb_objective'])
        self.assertEqual(a['common_old_latent01_objective'],b['common_old_latent01_objective'])
        self.assertAlmostEqual(b['train_loss'],b['common_rgb_objective']+.03*b['source_latent_mse']
            +b['rf1_base_regression_loss'],places=6)
        self.assertFalse(b['flow_base_detached']);self.assertTrue(b['base_regression_reference_detached'])
        self.assertTrue(torch.isfinite(corrected))

    def test_actual_rf_rgb_gradient_reaches_base_only_with_explicit_coupling_and_codec_stays_fixed(self):
        model=tiny_model();source=batch((False,True))
        with torch.no_grad():model.flow_renderer.output.weight.fill_(.004)
        codec={k:v.clone() for k,v in model.flow_renderer.codec.state_dict().items()}
        for detach in [True,False]:
            model.zero_grad(set_to_none=True);captured=[]
            def capture(*args,**kwargs):
                value=balanced_image_l1(*args,**kwargs);captured.append(value);return value
            with patch('turkeyworld.renderer_enemy_objectives.balanced_image_l1',side_effect=capture):
                enemy_refinement_loss(model,source,detach_base=detach)
            self.assertEqual(len(captured),2)
            captured[1].backward()
            norm=sum(float(p.grad.abs().sum()) for p in model.base_student.parameters() if p.grad is not None)
            self.assertEqual(norm,0) if detach else self.assertGreater(norm,0)
            self.assertTrue(all(p.grad is None for p in model.flow_renderer.codec.parameters()))
            self.assertTrue(all(torch.equal(v,model.flow_renderer.codec.state_dict()[k]) for k,v in codec.items()))

    def test_invalid_knobs_rejected_and_base_only_never_runs_a_guard_or_codec(self):
        model=tiny_model();source=batch((False,))
        for kwargs in [{'detach_base':1},{'base_regression_weight':-1},{'base_regression_weight':float('nan')}]:
            with self.assertRaises(ValueError):enemy_refinement_loss(model,source,**kwargs)
        with patch.object(model.flow_renderer.codec,'encode',side_effect=AssertionError('No codec')):
            value,metrics=enemy_refinement_loss(model,source,base_only=True,detach_base=False,
                latent_weight=.03,base_regression_weight=1.)
        self.assertTrue(torch.isfinite(value));self.assertFalse(metrics['flow_base_gradient_active'])
        self.assertNotIn('rf1_base_regression_loss',metrics)


if __name__=='__main__':unittest.main()
