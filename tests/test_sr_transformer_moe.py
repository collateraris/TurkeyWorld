"""RGB-only five-expert atlas SR: real forward, gradients and strict reload."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

import test_atlas_enemy_sr as atlas_fixture
from test_enemy_decoder_visual_first import force_visual,hypothesis
from turkeyworld.models import build_upscaler,load_sr_checkpoint
from turkeyworld.renderer_student import _hash_tensors
from turkeyworld.sr_transformer_moe import (ROLES,CONFIG,enable_sr_transformer_moe,
    configure_sr_transformer_moe,sr_moe_metadata,validate_sr_moe_checkpoint)


def parent_hash(model):
    return _hash_tensors({n:v for n,v in model.state_dict().items()if not n.startswith('sr_moe.')})


class SRTransformerMoETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_append_preserves_all_old_prefix_weights_buffers_modes_rng_and_every_old_output_exactly(self):
        old=atlas_fixture.small();force_visual(old)
        old.eval();rgb=torch.rand(2,3,32,32);hint=hypothesis()
        upgraded=deepcopy(old);before={n:v.clone()for n,v in upgraded.state_dict().items()}
        names=list(dict(upgraded.named_parameters()));pointers={n:p.data_ptr()for n,p in upgraded.named_parameters()}
        modes={n:m.training for n,m in upgraded.named_modules()};rng=torch.get_rng_state().clone()
        proof=enable_sr_transformer_moe(upgraded)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(list(dict(upgraded.named_parameters()))[:len(names)],names)
        self.assertTrue(all(dict(upgraded.named_parameters())[n].data_ptr()==v for n,v in pointers.items()))
        self.assertTrue(all(dict(upgraded.named_modules())[n].training==v for n,v in modes.items()))
        self.assertTrue(all(torch.equal(v,upgraded.state_dict()[n])for n,v in before.items()))
        self.assertEqual(proof['source_model_sha256'],parent_hash(upgraded))
        for training in(False,True):
            old.train(training);upgraded.train(training)
            for value in(None,hint):
                with torch.no_grad():a=old.forward_with_aux(rgb,value);b=upgraded.forward_with_aux(rgb,value)
                for key in a:
                    if torch.is_tensor(a[key]):self.assertTrue(torch.equal(a[key],b[key]),(training,key))
                self.assertEqual(b['sr_moe']['role_names'],ROLES)
                self.assertTrue(torch.allclose(b['sr_moe']['gates'].sum(-1),torch.ones(2,64)))
        self.assertEqual(_hash_tensors({'a':old.atlas_frames_rgba}),_hash_tensors({'a':upgraded.atlas_frames_rgba}))

    def test_only_new_parameters_learn_and_all_five_blocks_open_after_two_CPU_Adam_steps(self):
        model=atlas_fixture.small();force_visual(model);enable_sr_transformer_moe(model)
        configure_sr_transformer_moe(model);frozen=parent_hash(model)
        optimizer=torch.optim.Adam([p for p in model.parameters()if p.requires_grad],lr=.002)
        rgb=torch.rand(2,3,32,32);hint=hypothesis();expert_grads=[]
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            with patch('turkeyworld.engine.Game.render',side_effect=AssertionError('No Source')),\
                 patch.object(model.enemy_decoder_rgba,'forward',side_effect=AssertionError('No learned paint')):
                result=model.forward_with_aux(rgb,hint)
            aux=model.sr_moe.pop_diagnostics();self.assertIs(aux,result['sr_moe'])
            self.assertIsNone(model.sr_moe.pop_diagnostics())
            target=torch.arange(64)[None].expand(2,-1).remainder(5)
            loss=F.cross_entropy(aux['router_logits'].flatten(0,1),target.flatten())
            for key in('presence_logits','visual_confirmation_logits','enemy_confirmation_logits'):
                loss=loss+F.binary_cross_entropy_with_logits(result[key],torch.ones_like(result[key]))
            loss=loss+F.cross_entropy(result['visual_animation_logits'],torch.tensor([1,4]))
            loss=loss+F.binary_cross_entropy_with_logits(result['atlas_player_dead_logits'],torch.tensor([[0.],[1.]]))
            visible=torch.zeros_like(result['enemy_visible_logits_hr']);visible[...,20:90,25:95]=1
            loss=loss+F.binary_cross_entropy_with_logits(result['enemy_visible_logits_hr'],visible)
            loss=loss+F.mse_loss(result['visual_bbox'],torch.tensor([[.1,.15,.5,.7],[.4,.2,.8,.8]]))
            loss=loss+F.mse_loss(result['image'],torch.rand_like(result['image']))
            loss.backward()
            self.assertGreater(float(model.sr_moe.detection_head.weight.grad.abs().sum()),0.)
            self.assertGreater(float(model.sr_moe.geometry_head.weight.grad.abs().sum()),0.)
            self.assertGreater(float(model.sr_moe.visibility_head.weight.grad.abs().sum()),0.)
            if step:
                for expert in model.sr_moe.experts:
                    gradients=[p.grad for p in expert.parameters()if p.grad is not None]
                    self.assertTrue(all(torch.isfinite(g).all()for g in gradients))
                    self.assertGreater(sum(float(g.abs().sum())for g in gradients),0.)
            self.assertTrue(all(p.grad is None and not p.requires_grad for n,p in model.named_parameters()if not n.startswith('sr_moe.')))
            optimizer.step();self.assertEqual(parent_hash(model),frozen)

    def test_config_and_strict_saved_migration_binding_reject_tampering_and_keep_embedded_atlas_self_contained(self):
        model=atlas_fixture.small().eval();report=enable_sr_transformer_moe(model)
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);payload=dict(stage='sr',config=model.config,model=model.state_dict(),
                training_steps=106105,sr_moe_migration=report,sr_moe_binding=sr_moe_metadata(model))
            path=root/'sr.pt';torch.save(payload,path)
            with patch('turkeyworld.atlas_enemy_sr.Image.open',side_effect=AssertionError('No atlasfile')):
                restored,_=load_sr_checkpoint(path,'cpu')
                self.assertEqual(_hash_tensors(model.state_dict()),_hash_tensors(restored.state_dict()))
                rgb=torch.rand(2,3,32,32)
                with torch.inference_mode():self.assertTrue(torch.equal(model(rgb,None),restored(rgb,None)))
            for mode in('missing_binding','missing_migration','expert_weight','parent_weight','role_schema','atlas'):
                broken=deepcopy(payload)
                if mode=='missing_binding':broken.pop('sr_moe_binding')
                elif mode=='missing_migration':broken.pop('sr_moe_migration')
                elif mode=='expert_weight':broken['model']['sr_moe.experts.3.linear1.weight'].add_(.01)
                elif mode=='parent_weight':broken['model']['atlas_player_dead_head.bias'].add_(.01)
                elif mode=='role_schema':broken['config']['sr_moe_role_schema']='other'
                else:broken['model']['atlas_frames_rgba'][0,0,0,0]^=1
                bad=root/(mode+'.pt');torch.save(broken,bad)
                with self.subTest(mode=mode),self.assertRaises((ValueError,RuntimeError)):load_sr_checkpoint(bad,'cpu')

    def test_old_config_loader_remains_unchanged_and_adapter_has_five_distinct_small_blocks(self):
        old=atlas_fixture.small();self.assertNotIn('sr_moe_enabled',old.config)
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'old.pt';torch.save(dict(stage='sr',config=old.config,model=old.state_dict()),path)
            loaded,_=load_sr_checkpoint(path);self.assertFalse(hasattr(loaded,'sr_moe'))
            self.assertEqual(_hash_tensors(old.state_dict()),_hash_tensors(loaded.state_dict()))
        enable_sr_transformer_moe(old)
        count=sum(p.numel()for p in old.sr_moe.parameters());self.assertLess(count,500000)
        self.assertEqual(len(old.sr_moe.experts),5)
        ptrs=[{p.data_ptr()for p in e.parameters()}for e in old.sr_moe.experts]
        self.assertTrue(all(ptrs[i].isdisjoint(ptrs[j])for i in range(5)for j in range(i)))
        self.assertEqual(old.sr_moe.input(torch.rand(1,3,32,32)).shape,(1,64,8,8))
        for key,value in dict(sr_moe_width=32,sr_moe_experts=3,sr_moe_role_schema='other',sr_moe_grid=8.).items():
            config={**old.config,key:value}
            with self.subTest(key=key),self.assertRaises(ValueError):build_upscaler(config)

    def test_native128_subpixel_heads_can_generate_alternating_single_pixel_RGB_and_visibility(self):
        model=atlas_fixture.small();enable_sr_transformer_moe(model);module=model.sr_moe.eval()
        alternating=torch.tensor([1.,-1.]*8)[None].expand(16,-1)
        with torch.no_grad():
            module.rgb_heads[0].bias.copy_(alternating[None].expand(3,-1,-1).reshape(-1))
            module.visibility_head.bias.copy_(alternating.reshape(-1))
        with patch('turkeyworld.sr_transformer_moe.F.interpolate',side_effect=AssertionError('No fixed HR interpolation')):
            output=module(torch.rand(2,3,32,32))
        rgb=output['base_rgb_residual'];visibility=output['visibility_residual']
        self.assertEqual(rgb.shape,(2,3,128,128));self.assertEqual(visibility.shape,(2,1,128,128))
        self.assertEqual(output['heat_residual'].shape,(2,1,32,32))
        self.assertTrue((rgb[...,::2]>0).all());self.assertTrue((rgb[...,1::2]<0).all())
        self.assertGreater(float(rgb.detach().abs().max()),.1)
        self.assertTrue((visibility[...,::2]>0).all());self.assertTrue((visibility[...,1::2]<0).all())
        self.assertTrue(torch.equal(rgb[...,0],rgb[...,126]))


if __name__=='__main__':unittest.main()
