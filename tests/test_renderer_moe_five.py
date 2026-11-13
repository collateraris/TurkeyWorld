"""Five-role split migration retains trained three-expert outputs and names."""
from copy import deepcopy
from dataclasses import asdict,replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from test_renderer_geometry_opacity import working_model
from test_renderer_enemy_objectives import batch
from test_renderer_moe_bundle import unloaded_base
from turkeyworld.base_student import BaseStudentConfig,CompactBaseHead
from turkeyworld.latent_flow_renderer import LatentFlowRenderer
from turkeyworld.renderer_enemy_objectives import enemy_conditions
from turkeyworld.renderer_geometry_opacity import enable_geometry_opacity
from turkeyworld.renderer_moe import (THREE_SCHEMA,FIVE_SCHEMA,FIVE_KIND,FIVE_ROLES,
    enable_renderer_moe,configure_renderer_moe,migrate_renderer_moe_five,renderer_moe_metadata)
from turkeyworld.renderer_refinement import refinement_forward,export_refined_bundle
from turkeyworld.renderer_student import load_renderer_student,_hash_tensors


def trained_three():
    model=working_model();enable_geometry_opacity(model);enable_renderer_moe(model)
    # A trained shared output/router is essential: zero-head equality is trivial.
    with torch.no_grad():
        for head in(model.base_student,model.flow_renderer):
            head.output_moe.output.weight.normal_(0,.025)
            head.output_moe.output.bias.normal_(0,.01)
            head.output_moe.router.weight.normal_(0,.04)
            head.output_moe.router.bias.copy_(torch.tensor([.1,-.2,.3]))
    return model.eval().requires_grad_(False)


class RendererMoEFiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_trained_split_is_bitexact_with_all_old_names_tensors_rng_modes_and_gradient_flags_preserved(self):
        model=trained_three();source=batch((False,False));conditions=enemy_conditions(model,source)
        with torch.inference_mode():before=refinement_forward(model,conditions)
        tensors={n:v.clone()for n,v in model.state_dict().items()}
        pointers={n:p.data_ptr()for n,p in model.named_parameters()}
        flags={n:p.requires_grad for n,p in model.named_parameters()}
        modes={n:m.training for n,m in model.named_modules()}
        rng=torch.get_rng_state().clone()
        metadata=migrate_renderer_moe_five(model)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(metadata['kind'],FIVE_KIND)
        self.assertEqual(metadata['version'],2)
        self.assertTrue(all(torch.equal(v,model.state_dict()[n])for n,v in tensors.items()))
        self.assertTrue(all(dict(model.named_parameters())[n].data_ptr()==v for n,v in pointers.items()))
        self.assertTrue(all(dict(model.named_parameters())[n].requires_grad==v for n,v in flags.items()))
        self.assertTrue(all(dict(model.named_modules())[n].training==v for n,v in modes.items()))
        with torch.inference_mode():after=refinement_forward(model,conditions)
        for a,b in zip(before,after):self.assertTrue(torch.equal(a,b))
        for head in(model.base_student,model.flow_renderer):
            module=head.output_moe
            self.assertEqual(len(module.experts),5)
            for old,new in((0,3),(1,4)):
                self.assertEqual(_hash_tensors(module.experts[old].state_dict()),_hash_tensors(module.experts[new].state_dict()))
                self.assertTrue(all(a.data_ptr()!=b.data_ptr()for a,b in zip(module.experts[old].parameters(),module.experts[new].parameters())))
            aux=module.pop_diagnostics()
            self.assertEqual(aux['role_names'],FIVE_ROLES)
            self.assertTrue(torch.allclose(aux['gates'].sum(-1),torch.ones_like(aux['gates'][...,0])))
            self.assertTrue(torch.equal(aux['gates'][...,0]+aux['gates'][...,1],aux['parent_gates'][...,0]))
            self.assertTrue(torch.equal(aux['gates'][...,2]+aux['gates'][...,3],aux['parent_gates'][...,1]))

    def test_all_five_private_blocks_and_both_routers_receive_gradients_without_Source_or_frozen_world_gradients(self):
        model=trained_three();migrate_renderer_moe_five(model);configure_renderer_moe(model)
        source=batch((False,False));source['state'].requires_grad_(True)
        with patch('turkeyworld.engine.Game.render',side_effect=AssertionError('No runtime Source')):
            conditions=enemy_conditions(model,source)
            base,rgb,_=refinement_forward(model,conditions)
        loss=base.square().mean()+rgb.square().mean()
        for head in(model.base_student,model.flow_renderer):
            aux=head.output_moe.pop_diagnostics()
            self.assertEqual(aux['router_logits'].shape[-1],5)
            self.assertTrue(torch.allclose(aux['router_logits'].softmax(-1),aux['gates'],atol=1e-7))
            target=torch.arange(aux['router_logits'].shape[1]).remainder(5)[None].expand(2,-1)
            loss=loss+F.cross_entropy(aux['router_logits'].flatten(0,1),target.flatten())
        loss.backward()
        for head in(model.base_student,model.flow_renderer):
            for router in(head.output_moe.router,head.output_moe.split_router):
                self.assertGreater(float(router.weight.grad.abs().sum()),0.)
            for expert in head.output_moe.experts:
                values=[p.grad for p in expert.parameters()if p.grad is not None]
                self.assertTrue(all(torch.isfinite(v).all()for v in values))
                self.assertGreater(sum(float(v.abs().sum())for v in values),0.)
        self.assertIsNone(source['state'].grad)
        self.assertTrue(all(p.grad is None for n,p in model.named_parameters()
            if not n.startswith(('base_student.','flow_renderer.'))or n.startswith(('flow_renderer.codec.','flow_renderer.geometry_opacity.'))))

    def test_explicit_five_config_counts_and_legacy_three_ceiling_are_separate(self):
        base=CompactBaseHead(BaseStudentConfig(output_moe_enabled=True,output_moe_role_schema=FIVE_SCHEMA,output_moe_experts=5))
        self.assertEqual(sum(p.numel()for p in base.parameters()),1001157)
        self.assertEqual(sum(p.numel()for p in base.output_moe.parameters()),4313)
        flow=LatentFlowRenderer(kind='student',image_size=32,mode='rectified_flow',steps=1,
            width=256,depth=5,heads=8,mlp_hidden=896,spatial_conditioning=True,
            block_conditioning=True,patch_size=2,output_moe_enabled=True,
            output_moe_role_schema=FIVE_SCHEMA,output_moe_experts=5,
            codec_config=dict(image_size=32,base_channels=4,latent_channels=8,
                embedding_dim=16,n_embed=8,data_init=False))
        self.assertEqual(sum(p.numel()for n,p in flow.named_parameters()if not n.startswith('codec.'))+6241,7051654)
        with self.assertRaises(ValueError):BaseStudentConfig(output_moe_role_schema=FIVE_SCHEMA,output_moe_experts=5)
        with self.assertRaises(ValueError):BaseStudentConfig(output_moe_enabled=True,output_moe_role_schema=FIVE_SCHEMA,output_moe_experts=3)
        self.assertEqual(BaseStudentConfig().output_moe_role_schema,THREE_SCHEMA)
        self.assertEqual(sum(p.numel()for p in CompactBaseHead(BaseStudentConfig(output_moe_enabled=True)).parameters()),999939)

    def test_strict_five_export_reload_and_transactional_schema_split_weight_and_binding_rejections(self):
        source=working_model();target=unloaded_base(deepcopy(source))
        working=trained_three();migrate_renderer_moe_five(working)
        # Match the source world used for the trained3 helper, not another initialization.
        source.load_state_dict({n:v for n,v in working.state_dict().items()if n in source.state_dict()},strict=True)
        target=unloaded_base(deepcopy(source))
        with tempfile.TemporaryDirectory()as directory:
            directory=Path(directory)
            export_refined_bundle(source,working,directory,updates=1,report={'passed':False},
                initial_renderer_steps=10,initial_base_steps=10,provenance={})
            path=directory/'renderer_student.pt';saved=torch.load(path,weights_only=False)
            self.assertEqual(saved['renderer_output_moe']['kind'],FIVE_KIND)
            self.assertEqual(saved['renderer_output_moe']['base']['roles'],list(FIVE_ROLES))
            rng=torch.get_rng_state().clone()
            load_renderer_student(target,path)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(_hash_tensors(working.base_student.state_dict()),_hash_tensors(target.base_student.state_dict()))
            self.assertEqual(_hash_tensors(working.flow_renderer.state_dict()),_hash_tensors(target.flow_renderer.state_dict()))
            self.assertEqual(target.config.flow_output_moe_experts,5)
            for mode in('missing_header','three_schema','wrong_order','split_weight','missing_clone','wrong_Base_schema'):
                broken=deepcopy(saved)
                if mode=='missing_header':broken.pop('renderer_output_moe')
                elif mode=='three_schema':broken['config'].update(flow_output_moe_role_schema=THREE_SCHEMA,flow_output_moe_experts=3)
                elif mode=='wrong_order':broken['renderer_output_moe']['flow']['role_expert_indices']=[0,1,2,3,4]
                elif mode=='split_weight':broken['model']['output_moe.split_router.weight'].add_(.1)
                elif mode=='missing_clone':broken['model'].pop('output_moe.experts.4.linear1.weight')
                else:broken['base_student_bundle']['config'].update(output_moe_role_schema=THREE_SCHEMA,output_moe_experts=3)
                bad=directory/(mode+'.pt');torch.save(broken,bad)
                fresh=unloaded_base(deepcopy(source));before=_hash_tensors(fresh.state_dict());old_flow=fresh.flow_renderer
                with self.subTest(mode=mode),self.assertRaises((ValueError,RuntimeError)):load_renderer_student(fresh,bad)
                self.assertIs(fresh.flow_renderer,old_flow)
                self.assertEqual(before,_hash_tensors(fresh.state_dict()))


if __name__=='__main__':unittest.main()
