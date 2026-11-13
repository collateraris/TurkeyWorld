"""Real camera/Base/RF checks for the renderer-owned opacity calibration path."""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from test_renderer_enemy_objectives import tiny_model, batch
from test_renderer_student import tiny_config
from turkeyworld.base_student import load_base_student
from turkeyworld.geometry_targets import build_geometry_targets
from turkeyworld.models import NeuralWorldModel, WorldModelConfig, load_world_checkpoint
from turkeyworld.renderer_enemy_objectives import enemy_conditions
from turkeyworld.renderer_geometry_opacity import (enable_geometry_opacity, configure_geometry_opacity,
    geometry_opacity_metadata, shared_camera_geometry)
from turkeyworld.renderer_refinement import refinement_forward, export_refined_bundle
from turkeyworld.renderer_student import _hash_tensors, load_renderer_student


def working_model():
    model = tiny_model()
    # tiny_model manually attaches its student; production's strict loader
    # performs this same renderer-config assignment.
    model.config = WorldModelConfig(**model.flow_renderer._renderer_student_config)
    return model.eval()


class RendererGeometryOpacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_copy_activation_is_bitexact_preserves_old_state_and_rng_and_never_changes_prior_geometry(self):
        model=working_model();source=batch((False,False))
        state=source['state'];features=torch.zeros(2,128);fmap=model.encode_map(source['map'])
        with torch.no_grad():
            old_geometry=model.geometry(state,fmap)
            old_base=model.render_base(state,features,fmap)
            old_rgb=model.render(state,features,fmap)
        old={k:v.detach().clone()for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
        metadata=enable_geometry_opacity(model)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(metadata['parameter_count'],6241)
        self.assertEqual(len(list(model.flow_renderer.geometry_opacity.parameters())),4)
        self.assertTrue(model.config.flow_geometry_opacity_enabled)
        self.assertTrue(all(torch.equal(v,model.state_dict()[k])for k,v in old.items()))
        with torch.no_grad():
            self.assertTrue(torch.equal(old_base,model.render_base(state,features,fmap)))
            self.assertTrue(torch.equal(old_rgb,model.render(state,features,fmap)))
            for k,v in old_geometry.items():
                if torch.is_tensor(v):self.assertTrue(torch.equal(v,model.render_geometry(state,fmap)[k]),k)
            model.flow_renderer.geometry_opacity[-1].bias.add_(2.)
            self.assertFalse(torch.equal(old_geometry['depth'],model.render_geometry(state,fmap)['depth']))
            self.assertTrue(torch.equal(old_geometry['depth'],model.geometry(state,fmap)['depth']))
        self.assertEqual(_hash_tensors(shared_camera_geometry(model).opacity_network.state_dict()),metadata['source_opacity_sha256'])

    def test_source_depth_and_base_wall_gradients_reach_only_private_opacity_not_Source_or_frozen_world(self):
        model=working_model();source=batch((False,False))
        for key in('state','target','previous_state','previous_rgb'):source[key].requires_grad_(True)
        configure_geometry_opacity(model,only_private=True)
        fixed={k:v.detach().clone()for k,v in model.state_dict().items()if not k.startswith('flow_renderer.geometry_opacity.')}
        targets=torch.stack([torch.tensor(build_geometry_targets(grid.numpy(),state.detach().numpy())['depth'])
                             for grid,state in zip(source['map'],source['state'])]).float()
        with patch('turkeyworld.engine.Game.render',side_effect=AssertionError('No runtime Source rendering')):
            conditions=enemy_conditions(model,source)
            self.assertTrue(conditions['geometry']['depth'].requires_grad)
            base,_,_=refinement_forward(model,conditions)
            private_parameters=tuple(model.flow_renderer.geometry_opacity.parameters())
            image_gradients=torch.autograd.grad(F.l1_loss(base,conditions['target']),private_parameters,
                                               retain_graph=True,allow_unused=True)
            self.assertGreater(sum(float(g.abs().sum())for g in image_gradients if g is not None),0.)
            loss=F.l1_loss(conditions['geometry']['depth'],targets)+F.l1_loss(base,conditions['target'])
            loss.backward()
        private=[p for n,p in model.named_parameters()if n.startswith('flow_renderer.geometry_opacity.')]
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()for p in private))
        self.assertGreater(sum(float(p.grad.abs().sum())for p in private),0.)
        self.assertTrue(all(p.grad is None for n,p in model.named_parameters()if not n.startswith('flow_renderer.geometry_opacity.')))
        self.assertTrue(all(source[k].grad is None for k in('state','target','previous_state','previous_rgb')))
        self.assertTrue(all(torch.equal(v,model.state_dict()[k])for k,v in fixed.items()))

    def test_full_world_flag_constructs_private_tensors_before_strict_load_without_changing_legacy_default(self):
        config=tiny_config(flow_kind='student',renderer_mode='rectified_flow',flow_steps=1,
                           flow_geometry_opacity_enabled=True)
        model=NeuralWorldModel(config).eval()
        with torch.no_grad():model.flow_renderer.geometry_opacity[-1].bias.add_(.4)
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'world.pt'
            torch.save(dict(stage='world',config=asdict(config),model=model.state_dict()),path)
            clone,_=load_world_checkpoint(path,'cpu')
            self.assertEqual(_hash_tensors(model.state_dict()),_hash_tensors(clone.state_dict()))
            no_private=NeuralWorldModel(replace(config,flow_geometry_opacity_enabled=False))
            self.assertFalse(hasattr(no_private.flow_renderer,'geometry_opacity'))
            with self.assertRaises(RuntimeError):no_private.load_state_dict(model.state_dict(),strict=True)

    def test_private_bundle_and_Base_binding_strict_roundtrip_and_transactional_tamper_rejection(self):
        source=working_model();working=deepcopy(source)
        enable_geometry_opacity(working)
        with torch.no_grad():working.flow_renderer.geometry_opacity[-1].bias.add_(.5)
        with tempfile.TemporaryDirectory()as directory:
            directory=Path(directory)
            export_refined_bundle(source,working,directory,updates=3,
                report={'passed':False},initial_renderer_steps=10,initial_base_steps=10,provenance={})
            path=directory/'renderer_student.pt';saved=torch.load(path,weights_only=False)
            binding=saved['geometry_opacity_override']
            self.assertEqual(binding,saved['base_student_bundle']['geometry_opacity_binding'])
            target=working_model();target.load_state_dict({k:v for k,v in source.state_dict().items()},strict=True)
            old_flow=target.flow_renderer;old_hash=_hash_tensors(target.state_dict())
            with self.assertRaisesRegex(ValueError,'matching private'):
                load_base_student(target,directory/'base_student.pt')
            with patch('turkeyworld.engine.Game.render',side_effect=AssertionError('No Source inference')):
                load_renderer_student(target,path)
            self.assertIsNot(target.flow_renderer,old_flow)
            self.assertEqual(geometry_opacity_metadata(target.flow_renderer,target),binding)
            self.assertEqual(_hash_tensors(working.flow_renderer.state_dict()),_hash_tensors(target.flow_renderer.state_dict()))
            query=batch((False,));fmap=target.encode_map(query['map'])
            with torch.inference_mode():actual=target.render_base(query['state'],torch.zeros(1,128),fmap)
            self.assertTrue(torch.isfinite(actual).all())
            for mode in('missing_metadata','missing_weight','changed_weight','wrong_base_binding','wrong_source'):
                broken=deepcopy(saved)
                if mode=='missing_metadata':broken.pop('geometry_opacity_override')
                elif mode=='missing_weight':broken['model'].pop('geometry_opacity.0.weight')
                elif mode=='changed_weight':broken['model']['geometry_opacity.0.weight'].add_(.01)
                elif mode=='wrong_base_binding':broken['base_student_bundle']['geometry_opacity_binding']['state_sha256']='0'*64
                else:broken['geometry_opacity_override']['source_opacity_sha256']='f'*64
                bad=directory/(mode+'.pt');torch.save(broken,bad)
                fresh=deepcopy(source);old=_hash_tensors(fresh.state_dict());flow=fresh.flow_renderer
                with self.subTest(mode=mode),self.assertRaises((ValueError,RuntimeError)):
                    load_renderer_student(fresh,bad)
                self.assertIs(flow,fresh.flow_renderer);self.assertEqual(old,_hash_tensors(fresh.state_dict()))


if __name__=='__main__':unittest.main()
