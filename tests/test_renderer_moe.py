"""Real compact renderer token experts: exact migration and causal gradients."""
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from test_renderer_geometry_opacity import working_model
from test_renderer_enemy_objectives import batch
from test_renderer_student import tiny_config
from turkeyworld.base_student import BaseStudentConfig, CompactBaseHead
from turkeyworld.latent_flow_renderer import LatentFlowRenderer
from turkeyworld.models import NeuralWorldModel
from turkeyworld.renderer_enemy_objectives import enemy_conditions
from turkeyworld.renderer_geometry_opacity import enable_geometry_opacity
from turkeyworld.renderer_moe import (RendererTokenMoE, enable_renderer_moe,
    configure_renderer_moe, output_moe_metadata)
from turkeyworld.renderer_refinement import refinement_forward
from turkeyworld.renderer_student import _hash_tensors


class RendererMoETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_actual_private_transformer_parameter_budgets_and_independent_experts(self):
        base=CompactBaseHead(BaseStudentConfig(output_moe_enabled=True))
        self.assertEqual(sum(p.numel()for p in base.parameters()),999939)
        self.assertEqual(sum(p.numel()for p in base.output_moe.parameters()),3095)
        flow=LatentFlowRenderer(kind='student',image_size=32,mode='rectified_flow',steps=1,
            width=256,depth=5,heads=8,mlp_hidden=896,spatial_conditioning=True,
            block_conditioning=True,patch_size=2,output_moe_enabled=True,
            codec_config=dict(image_size=32,base_channels=4,latent_channels=8,
                embedding_dim=16,n_embed=8,data_init=False))
        self.assertEqual(sum(p.numel()for p in flow.output_moe.parameters()),36067)
        # Original RF budget excludes its separately frozen codec; private MLP adds6241.
        actual=sum(p.numel()for n,p in flow.named_parameters()if not n.startswith('codec.'))+6241
        self.assertEqual(actual,7034500)
        for module in(base.output_moe,flow.output_moe):
            self.assertEqual(len(module.experts),3)
            self.assertTrue(all(isinstance(e,nn.TransformerEncoderLayer)for e in module.experts))
            parameters=[{p.data_ptr()for p in e.parameters()}for e in module.experts]
            self.assertTrue(all(parameters[i].isdisjoint(parameters[j])for i in range(3)for j in range(i)))
            self.assertTrue(all(e.dropout.p==0 and e.norm_first for e in module.experts))

    def test_append_preserves_rng_modes_all_old_tensors_and_exact_Base_RF_outputs(self):
        model=working_model();enable_geometry_opacity(model)
        source=batch((False,False));conditions=enemy_conditions(model,source)
        with torch.inference_mode():old=refinement_forward(model,conditions)
        before={n:t.clone()for n,t in model.state_dict().items()}
        buffers={n:t.clone()for n,t in model.named_buffers()}
        modes={n:m.training for n,m in model.named_modules()}
        flags={n:p.requires_grad for n,p in model.named_parameters()}
        rng=torch.get_rng_state().clone()
        enable_renderer_moe(model)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(t,model.state_dict()[n])for n,t in before.items()))
        self.assertTrue(all(torch.equal(t,dict(model.named_buffers())[n])for n,t in buffers.items()))
        self.assertTrue(all(dict(model.named_modules())[n].training==v for n,v in modes.items()))
        self.assertTrue(all(dict(model.named_parameters())[n].requires_grad==v for n,v in flags.items()))
        with torch.inference_mode():actual=refinement_forward(model,conditions)
        for a,b in zip(old,actual):self.assertTrue(torch.equal(a,b))
        self.assertTrue(model.config.flow_output_moe_enabled)
        self.assertTrue(model.base_student.config.output_moe_enabled)

    def test_actual_forward_router_role_gradients_then_all_experts_receive_image_gradients(self):
        model=working_model();enable_geometry_opacity(model)
        # Current world may be COMBAT; no navigation-only configure helper is allowed.
        model.config=replace(model.config,navigation_only=False)
        configure_renderer_moe(model)
        source=batch((False,False));conditions=enemy_conditions(model,source)
        with patch('turkeyworld.engine.Game.render',side_effect=AssertionError('No runtime Source')):
            base,rgb,_=refinement_forward(model,conditions)
        loss=base.square().mean()+rgb.square().mean()
        for head in(model.base_student,model.flow_renderer):
            aux=head.output_moe.pop_diagnostics()
            self.assertIsNone(head.output_moe.pop_diagnostics())
            self.assertTrue(aux['router_logits'].requires_grad)
            self.assertTrue(aux['expert_outputs'].requires_grad)
            self.assertTrue(torch.all(aux['gates']>0))
            self.assertTrue(torch.allclose(aux['gates'].sum(-1),torch.ones_like(aux['gates'][...,0])))
            targets=torch.arange(aux['router_logits'].shape[1]).remainder(3)[None].expand(len(base),-1)
            loss=loss+F.cross_entropy(aux['router_logits'].flatten(0,1),targets.flatten())
        loss.backward()
        for head in(model.base_student,model.flow_renderer):
            self.assertGreater(float(head.output_moe.router.weight.grad.abs().sum()),0.)
            self.assertGreater(float(head.output_moe.output.weight.grad.abs().sum()),0.)
            # ZERO output keeps each newly appended deep expert closed for the first step.
            self.assertTrue(all(p.grad is None or not bool(p.grad.any())
                for expert in head.output_moe.experts for p in expert.parameters()))
            with torch.no_grad():head.output_moe.output.weight.fill_(.01)
        model.zero_grad(set_to_none=True)
        conditions=enemy_conditions(model,source)
        base,rgb,_=refinement_forward(model,conditions)
        (base.square().mean()+rgb.square().mean()).backward()
        for head in(model.base_student,model.flow_renderer):
            for expert in head.output_moe.experts:
                gradients=[p.grad for p in expert.parameters()if p.grad is not None]
                self.assertTrue(all(torch.isfinite(g).all()for g in gradients))
                self.assertGreater(sum(float(g.abs().sum())for g in gradients),0.)
        for name,p in model.named_parameters():
            if not name.startswith(('base_student.','flow_renderer.'))or name.startswith(
                    ('flow_renderer.codec.','flow_renderer.geometry_opacity.')):
                self.assertFalse(p.requires_grad,name);self.assertIsNone(p.grad,name)
        self.assertTrue(all(not p.requires_grad for p in model.flow_renderer.geometry_opacity.parameters()))
        self.assertFalse(model.flow_renderer.codec.training)

    def test_own_state_prior_is_soft_only_and_cache_is_not_serialized(self):
        module=RendererTokenMoE(12,6,8,4)
        tokens=torch.randn(2,16,12)
        state=batch((False,False))['state']
        with patch('turkeyworld.engine.Game.render',side_effect=AssertionError('No Source')):
            residual=module(tokens,state)
        self.assertTrue(torch.equal(residual,torch.zeros_like(residual)))
        aux=module.pop_diagnostics()
        self.assertGreater(float(aux['gates'].detach().std()),0.)
        self.assertTrue((aux['gates']>0).all())
        self.assertTrue(all('last_aux'not in n for n in module.state_dict()))
        self.assertTrue(output_moe_metadata(type('Head',(),{'output_moe':module})())['experts_are_private_transformers'])

    def test_explicit_fullworld_flag_strict_reconstruction_keeps_legacy_initialization_and_rng(self):
        config=tiny_config(flow_kind='student',renderer_mode='rectified_flow',flow_steps=1)
        rng=torch.get_rng_state().clone();old=NeuralWorldModel(config)
        after=torch.get_rng_state().clone()
        torch.set_rng_state(rng);new=NeuralWorldModel(replace(config,flow_output_moe_enabled=True))
        self.assertTrue(torch.equal(after,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v,new.state_dict()[n])for n,v in old.state_dict().items()))
        clone=NeuralWorldModel(new.config);clone.load_state_dict(new.state_dict(),strict=True)
        self.assertEqual(_hash_tensors(new.state_dict()),_hash_tensors(clone.state_dict()))
        with self.assertRaises(RuntimeError):old.load_state_dict(new.state_dict(),strict=True)


if __name__=='__main__':unittest.main()
