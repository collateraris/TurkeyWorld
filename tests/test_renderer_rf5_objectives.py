"""Real tiny VQ/Transformer RF5 graphs, including checkpoint recomputation."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import torch

from test_renderer_moe_five import trained_three
from test_renderer_enemy_objectives import batch
from turkeyworld.renderer_moe import migrate_renderer_moe_five
from turkeyworld.renderer_moe_objectives import moe_conditions
from turkeyworld.renderer_rf5_objectives import rf5_forward,rf5_refinement_loss
from turkeyworld.renderer_student import _hash_tensors


def model(steps=5):
    result=trained_three();migrate_renderer_moe_five(result)
    result.eval().requires_grad_(False)
    result.flow_renderer.train().requires_grad_(True)
    result.flow_renderer.codec.eval().requires_grad_(False)
    result.flow_renderer.geometry_opacity.eval().requires_grad_(False)
    # The new production architecture/phase owns its explicit RF5 config.
    # This existing real tiny fixture exercises the same five-step integrator.
    result.flow_renderer.steps=steps
    return result


def source():
    result=batch((False,False));result['navigation_only']=torch.zeros(2,dtype=torch.bool)
    return result


class RendererRF5ObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_all_five_actual_router_calls_and_terminal_VQ_decode_receive_image_gradient(self):
        network=model();conditions=moe_conditions(network,source())
        before=_hash_tensors(network.state_dict())
        with patch('turkeyworld.engine.Game.render',side_effect=AssertionError('No Source inference')):
            base,rgb,endpoint,routers=rf5_forward(network,conditions,checkpoint_velocity=False)
        self.assertEqual(len(routers),5)
        self.assertFalse(base.requires_grad)
        self.assertTrue(rgb.requires_grad);self.assertTrue(endpoint.requires_grad)
        endpoint.retain_grad()
        for aux in routers:
            self.assertEqual(aux['router_logits'].shape[-1],5)
            self.assertTrue(aux['router_logits'].requires_grad)
        rgb.square().mean().backward()
        self.assertGreater(float(endpoint.grad.abs().sum()),0)
        self.assertGreater(float(network.flow_renderer.image_input.weight.grad.abs().sum()),0)
        for expert in network.flow_renderer.output_moe.experts:
            self.assertGreater(sum(float(p.grad.abs().sum())for p in expert.parameters()if p.grad is not None),0)
        for name,p in network.named_parameters():
            if not name.startswith('flow_renderer.')or name.startswith(('flow_renderer.codec.','flow_renderer.geometry_opacity.')):
                self.assertIsNone(p.grad,name)
        self.assertEqual(before,_hash_tensors(network.state_dict()))
        self.assertEqual(len(network.flow_renderer.output_moe._forward_hooks),0)

    def test_checkpoint_and_ordinary_five_step_losses_and_gradients_agree_without_duplicate_router_labels(self):
        first=model();second=deepcopy(first);data=source()
        loss_a,metrics_a=rf5_refinement_loss(first,data,checkpoint_velocity=False)
        loss_a.backward()
        with patch.object(second.flow_renderer,'velocity',wraps=second.flow_renderer.velocity)as calls:
            loss_b,metrics_b=rf5_refinement_loss(second,data,checkpoint_velocity=True)
            self.assertEqual(calls.call_count,5)
            self.assertEqual(len(second.flow_renderer.output_moe._forward_hooks),0)
            loss_b.backward()
            self.assertGreater(calls.call_count,5)
        self.assertTrue(torch.allclose(loss_a,loss_b,atol=1e-7,rtol=1e-6))
        self.assertEqual(metrics_a['rf5_router_steps'],metrics_b['rf5_router_steps'])
        self.assertEqual(len(metrics_b['rf5_router_steps']),5)
        self.assertAlmostEqual(metrics_b['renderer_moe_role_loss'],
            sum(row['role_ce']for row in metrics_b['rf5_router_steps'])/5,places=6)
        for name,p in first.named_parameters():
            other=dict(second.named_parameters())[name]
            if p.grad is None:self.assertIsNone(other.grad,name)
            else:self.assertTrue(torch.allclose(p.grad,other.grad,atol=2e-6,rtol=2e-4),name)

    def test_target_RGB_masks_and_background_never_condition_rf5_forward(self):
        network=model();data=source();conditions=moe_conditions(network,data)
        with torch.no_grad():a=rf5_forward(network,conditions,checkpoint_velocity=False)
        changed=dict(conditions)
        for key in ('target','background','object_mask'):changed[key]=torch.full_like(changed[key],float('nan'))
        with torch.no_grad():b=rf5_forward(network,changed,checkpoint_velocity=False)
        for x,y in zip(a[:3],b[:3]):self.assertTrue(torch.equal(x,y))

    def test_three_actual_Euler_calls_checkpoint_gradients_and_diagnostics_use_three_not_five(self):
        first=model(3);second=deepcopy(first);data=source()
        loss_a,metrics_a=rf5_refinement_loss(first,data,checkpoint_velocity=False)
        loss_a.backward()
        with patch.object(second.flow_renderer,'velocity',wraps=second.flow_renderer.velocity)as calls:
            loss_b,metrics=rf5_refinement_loss(second,data,checkpoint_velocity=True)
            self.assertEqual(calls.call_count,3)
            loss_b.backward();self.assertGreater(calls.call_count,3)
        self.assertEqual(metrics['solver_steps'],3);self.assertEqual(metrics['router_velocity_calls'],3)
        self.assertEqual(set(metrics['renderer_moe_heads']),{'rf3'})
        self.assertEqual([x['euler_time']for x in metrics['rf_router_steps']],[0,1/3,2/3])
        self.assertEqual(len(metrics['rf3_router_steps']),3);self.assertNotIn('rf5_router_steps',metrics)
        self.assertNotIn('rf5_source_l1',metrics);self.assertIn('rf3_source_l1',metrics)
        self.assertAlmostEqual(metrics['renderer_moe_role_loss'],sum(x['role_ce']for x in metrics['rf_router_steps'])/3,places=6)
        self.assertEqual(metrics_a['rf_router_steps'],metrics['rf_router_steps'])
        self.assertTrue(torch.allclose(loss_a,loss_b,atol=1e-7,rtol=1e-6))
        for name,p in first.named_parameters():
            other=dict(second.named_parameters())[name]
            if p.grad is None:self.assertIsNone(other.grad,name)
            else:self.assertTrue(torch.allclose(p.grad,other.grad,atol=2e-6,rtol=2e-4),name)
        third=model(3);conditions=moe_conditions(third,data)
        _,image,endpoint,routers=rf5_forward(third,conditions,checkpoint_velocity=False)
        endpoint.retain_grad();image.square().mean().backward()
        self.assertEqual(len(routers),3);self.assertGreater(float(endpoint.grad.abs().sum()),0)
        self.assertGreater(float(third.flow_renderer.image_input.weight.grad.abs().sum()),0)
        for expert in third.flow_renderer.output_moe.experts:
            self.assertGreater(sum(float(p.grad.abs().sum())for p in expert.parameters()if p.grad is not None),0)

    def test_main_far_geometry_are_finite_RF_only_and_Source_labels_have_no_gradient(self):
        for kind in ('main','far','geometry'):
            with self.subTest(kind=kind):
                network=model();data=source()
                for key in ('state','target','background','object_mask'):
                    data[key]=data[key].detach().requires_grad_()
                with patch.object(network.flow_renderer.codec,'encode',wraps=network.flow_renderer.codec.encode)as encoded:
                    loss,metrics=rf5_refinement_loss(network,data,kind=kind,checkpoint_velocity=False)
                    # Base encode always conditions RF; Source encode is an
                    # additional frozen label ONLY for the main latent term.
                    self.assertEqual(encoded.call_count,2 if kind=='main' else 1)
                loss.backward()
                self.assertTrue(torch.isfinite(loss));self.assertEqual(metrics['solver_steps'],5)
                self.assertFalse(metrics['base_loss_active']);self.assertFalse(metrics['base_router_loss_active'])
                self.assertFalse(metrics['GAN_active']);self.assertFalse(metrics['source_future_feedback'])
                self.assertEqual(set(metrics['renderer_moe_heads']),{'rf5'})
                self.assertAlmostEqual(float(loss.detach()),metrics['renderer_rgb_loss']+
                    .01*metrics['renderer_moe_role_loss']+.001*metrics['renderer_moe_balance_loss'],places=6)
                for key in ('state','target','background','object_mask'):self.assertIsNone(data[key].grad,key)
                for name,p in network.named_parameters():
                    if name.startswith(('base_student.','flow_renderer.codec.','flow_renderer.geometry_opacity.')):
                        self.assertIsNone(p.grad,name)
                json.dumps(metrics,allow_nan=False)

    def test_frozen_policy_solver_and_invalid_settings_fail_before_forward(self):
        for mutate in (
            lambda n:setattr(n.flow_renderer,'steps',1),
            lambda n:n.base_student.train(),
            lambda n:next(n.base_student.parameters()).requires_grad_(True),
            lambda n:n.flow_renderer.codec.quantize.train(),
            lambda n:next(n.flow_renderer.geometry_opacity.parameters()).requires_grad_(True),
        ):
            network=model();mutate(network)
            with self.assertRaisesRegex(ValueError,'frozen eval'):
                rf5_refinement_loss(network,source())
        with self.assertRaises(ValueError):rf5_refinement_loss(model(),source(),checkpoint_velocity=1)
        with self.assertRaises(ValueError):rf5_refinement_loss(model(),source(),kind='unknown')
        with self.assertRaises(ValueError):rf5_refinement_loss(model(),source(),inherited={'ssim_weight':float('nan')})


if __name__=='__main__':unittest.main()
