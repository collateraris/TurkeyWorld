"""Function-preserving optional post-codec pixels, strict bundles and real gradients."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from turkeyworld.base_student import (BaseStudentConfig,CompactBaseHead,build_base_student,
    enable_base_pixel_residual,export_base_student,load_base_student,prepare_base_student)
from turkeyworld.models import NeuralWorldModel,WorldModelConfig,apply_pixel_residual
from turkeyworld.renderer_refinement import configure_refinement,refinement_forward,frozen_fingerprint
from turkeyworld.renderer_student import build_renderer_student,export_renderer_student,load_renderer_student
from turkeyworld.spatial_entities import entity_spatial_features
from turkeyworld.combat_objectives import combat_count_features
from turkeyworld.models import render_state_view


def teacher():
    config=WorldModelConfig(kind='teacher',architecture='vjepa2_ac',image_size=32,endless=True,
        navigation_only=True,spatial_entities=True,spatial_combat=True,renderer_mode='flow_matching',
        renderer_codec='vqgan',renderer_codec_config=dict(image_size=32,base_channels=4,
            latent_channels=8,embedding_dim=4,n_embed=8,data_init=False),flow_steps=1,
        flow_width=16,flow_depth=1,flow_heads=2,vjepa_width=16,vjepa_encoder_depth=1,
        vjepa_predictor_depth=1,vjepa_heads=2,vjepa_patch_size=8,vjepa_context_frames=2)
    model=NeuralWorldModel(config).eval().requires_grad_(False)
    model.flow_renderer.codec_loaded.fill_(True)
    return model


def conditions(model,count=2):
    state=torch.zeros(count,20);state[:,:2]=torch.tensor([.2,.3]);state[:,3:6]=1
    state[:,8:10]=torch.tensor([.5,.3]);state[:,12:14]=torch.tensor([.6,1.])
    grid=torch.zeros(count,16,16,dtype=torch.long)
    with torch.no_grad():
        maps=model.encode_map(grid)
        geometry=model.vjepa_backend.geometry(state,maps)
    return {'state':state,'features':torch.randn(count,128),'map':grid,'geometry':geometry},maps


def legacy_forward(head,state,features,geometry):
    """Frozen pre-extension execution order; used only as the compatibility oracle."""
    dtype=head.scene_projection.weight.dtype;pose=render_state_view(state).to(dtype)
    condition=head.condition(torch.cat([pose,features.to(dtype)],-1));gate=geometry['render_gate'].to(dtype)
    objects=geometry['object_identity_tokens']+gate[...,None]*geometry['object_appearance_tokens']
    scene=torch.cat([geometry['ray_tokens'],objects],1).to(dtype)
    participants=torch.cat([entity_spatial_features(pose),combat_count_features(state)],-1).to(dtype)
    context=torch.cat([head.scene_projection(scene),head.participants(participants),condition[:,None]],1)
    weights=torch.cat([torch.ones_like(geometry['ray_tokens'][...,0]),gate,
        torch.ones(len(state),3,device=state.device,dtype=dtype)],-1).to(dtype)
    queries=head.screen_position(head.screen_coordinates.to(dtype))+condition[:,None]
    for block in head.blocks:queries=block(queries,context,key_weights=weights)
    pixels=head.rgb(head.norm(queries)).reshape(-1,16,16,3,2,2)
    return pixels.permute(0,3,1,4,2,5).reshape(-1,3,32,32).sigmoid()


class BasePixelResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_default_false_keys_rng_and_legacy_forward_remain_bitexact(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(173);left=CompactBaseHead();rng=torch.random.get_rng_state().clone()
            torch.manual_seed(173);right=CompactBaseHead(BaseStudentConfig(pixel_residual=False))
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in left.parameters()),996844)
        self.assertEqual(left.state_dict().keys(),right.state_dict().keys())
        self.assertFalse(any(k.startswith('pixel_residual.') for k in left.state_dict()))
        self.assertTrue(all(torch.equal(v,right.state_dict()[k]) for k,v in left.state_dict().items()))
        source=teacher();query,_=conditions(source)
        with torch.no_grad():
            expected=legacy_forward(left,query['state'],query['features'],query['geometry'])
            actual=left(query['state'],query['features'],query['geometry'])
        self.assertTrue(torch.equal(expected,actual))
        with self.assertRaises(ValueError):BaseStudentConfig(pixel_residual=1)

    def test_upgrade_preserves_all_source_weights_rng_and_zero_final_rgb_exactly(self):
        source=teacher();head=build_base_student(source).eval();before={k:v.clone() for k,v in head.state_dict().items()}
        query,_=conditions(source);rng=torch.random.get_rng_state().clone()
        with patch.object(torch.cuda,'manual_seed_all',side_effect=AssertionError('No CUDA seeding')):
            upgraded=enable_base_pixel_residual(head)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertIsNot(head,upgraded);self.assertFalse(head.config.pixel_residual)
        self.assertEqual(sum(p.numel() for p in upgraded.parameters()),998584)
        self.assertEqual(sum(p.numel() for p in upgraded.pixel_residual.parameters()),1740)
        self.assertTrue(all(torch.equal(v,head.state_dict()[k]) and torch.equal(v,upgraded.state_dict()[k]) for k,v in before.items()))
        with torch.no_grad(),patch.object(upgraded.blocks[0],'forward',wraps=upgraded.blocks[0].forward) as block:
            base,residual=upgraded.forward_with_residual(query['state'],query['features'],query['geometry'])
            original=head(query['state'],query['features'],query['geometry'])
        self.assertEqual(block.call_count,1)
        self.assertTrue(torch.equal(base,original))
        self.assertTrue(torch.equal(residual,torch.zeros_like(residual)))
        decoded=torch.rand_like(base)
        self.assertTrue(torch.equal(apply_pixel_residual(decoded,residual),decoded))
        self.assertIs(apply_pixel_residual(decoded,None),decoded)

    def test_runtime_aux_and_training_apply_same_bounded_postdecode_residual(self):
        model=teacher();model.flow_renderer=build_renderer_student(model,steps=1,width=16,depth=1,heads=2,mlp_hidden=32)
        model.base_student=enable_base_pixel_residual(build_base_student(model))
        model.eval();query,maps=conditions(model)
        with torch.no_grad():model.base_student.pixel_residual.bias.fill_(.2)
        with (torch.no_grad(),patch('turkeyworld.engine.Game.step',side_effect=AssertionError('No engine')),
              patch('turkeyworld.engine.Game.render',side_effect=AssertionError('No Source RGB'))):
            base,rgb,endpoint,residual=refinement_forward(model,query,return_residual=True)
            runtime=model.render(query['state'],query['features'],maps)
            aux=model.render_with_aux(query['state'],query['features'],maps)
            self.assertTrue(torch.equal(runtime,rgb))
            self.assertTrue(torch.equal(aux['rgb'],rgb))
            self.assertTrue(torch.equal(aux['base_rgb'],base))
            self.assertTrue(torch.equal(base,model.render_base(query['state'],query['features'],maps)))
            self.assertTrue(torch.allclose(residual,torch.full_like(residual,.5*torch.tanh(torch.tensor(.2)))))
            decoded=model.flow_renderer.codec.decode(model.flow_renderer.codec.quantize(
                model.flow_renderer.denormalize_latents(endpoint))['quantized'])
            self.assertTrue(torch.equal(rgb,(decoded+residual).clamp(0,1)))

    def test_postcodec_loss_backpropagates_residual_and_flow_without_changing_any_codec_state(self):
        model=teacher();model.flow_renderer=build_renderer_student(model,steps=1,width=16,depth=1,heads=2,mlp_hidden=32)
        model.base_student=enable_base_pixel_residual(build_base_student(model))
        configure_refinement(model);query,_=conditions(model)
        frozen=frozen_fingerprint(model);codec={k:v.clone() for k,v in model.flow_renderer.codec.state_dict().items()}
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
        base,rgb,_,residual=refinement_forward(model,query,return_residual=True)
        self.assertTrue(torch.equal(residual,torch.zeros_like(residual)))
        target=torch.full_like(rgb,.9)
        ((rgb-target).square().mean()+(base-target).square().mean()).backward()
        self.assertGreater(float(model.base_student.pixel_residual.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.base_student.pixel_residual.bias.grad.abs().sum()),0)
        self.assertGreater(float(model.base_student.rgb.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.flow_renderer.output.weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in model.flow_renderer.codec.parameters()))
        optimizer.step()
        self.assertEqual(frozen,frozen_fingerprint(model))
        self.assertTrue(all(torch.equal(v,model.flow_renderer.codec.state_dict()[k]) for k,v in codec.items()))
        self.assertFalse(model.flow_renderer.codec.training)
        self.assertFalse(model.flow_renderer.codec.quantize.training)

    def test_old_missing_flag_loads_and_new_bundle_requires_explicit_matching_provenance(self):
        source=teacher()
        with TemporaryDirectory() as temporary:
            path=Path(temporary)/'base.pt';old=build_base_student(source)
            export_base_student(source,old,path,training_steps=100)
            payload=torch.load(path,weights_only=False);payload['config'].pop('pixel_residual')
            restored,_,_=prepare_base_student(source,payload)
            self.assertFalse(restored.config.pixel_residual)
            self.assertTrue(all(torch.equal(v,restored.state_dict()[k]) for k,v in old.state_dict().items()))
            new=enable_base_pixel_residual(old)
            with torch.no_grad():new.pixel_residual.bias.fill_(.1)
            export_base_student(source,new,path,training_steps=101,provenance={'new_pixel_residual':'zero_init_postcodec_v1'})
            payload=torch.load(path,weights_only=False)
            self.assertTrue(payload['config']['pixel_residual'])
            self.assertEqual(payload['head_parameters'],998584)
            self.assertEqual(payload['pixel_residual']['position'],'after_frozen_codec_decode')
            restored,_,_=prepare_base_student(source,payload)
            self.assertTrue(all(torch.equal(v,restored.state_dict()[k]) for k,v in new.state_dict().items()))
            altered=deepcopy(payload);altered['pixel_residual']['amplitude']=1.
            before={k:v.clone() for k,v in source.state_dict().items()}
            with self.assertRaisesRegex(ValueError,'provenance'):load_base_student(source,altered)
            self.assertTrue(all(torch.equal(v,source.state_dict()[k]) for k,v in before.items()))

    def test_standalone_renderer_bundle_roundtrip_attaches_new_base_without_changing_shared_source(self):
        source=teacher();reference=deepcopy(source);query,maps=conditions(source)
        with TemporaryDirectory() as temporary:
            root=Path(temporary);head=enable_base_pixel_residual(build_base_student(source))
            with torch.no_grad():head.pixel_residual.bias.fill_(.15)
            export_base_student(source,head,root/'base.pt',training_steps=10)
            load_base_student(source,root/'base.pt')
            flow=build_renderer_student(source,steps=1,width=16,depth=1,heads=2,mlp_hidden=32)
            export_renderer_student(source,flow,root/'rf.pt',training_steps=11)
            shared_before={k:v.clone() for k,v in reference.state_dict().items() if not k.startswith('flow_renderer.')}
            load_renderer_student(reference,root/'rf.pt')
            self.assertTrue(reference.base_student.config.pixel_residual)
            self.assertEqual(reference.flow_renderer.steps,1)
            self.assertTrue(all(torch.equal(v,reference.state_dict()[k]) for k,v in shared_before.items()))
            reference.eval()
            with torch.no_grad():
                maps=reference.encode_map(query['map'])
                base,residual=reference.base_student.forward_with_residual(query['state'],query['features'],query['geometry'])
                expected=apply_pixel_residual(reference.flow_renderer.sample(base,query['state']),residual)
                actual=reference.render(query['state'],query['features'],maps)
            self.assertTrue(torch.equal(actual,expected))


if __name__=='__main__':unittest.main()
