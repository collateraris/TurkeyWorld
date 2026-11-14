from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

import test_base_geometry_bundle as support
from test_renderer_enemy_objectives import batch
from turkeyworld.renderer_student import load_renderer_student,_hash_tensors
from turkeyworld.renderer_rf_chain_bundle import export_rf_chain_renderer,rf_chain_state


class RFChainBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.support=support.BaseGeometryBundleTests;cls.support.setUpClass()
        cls.source=deepcopy(cls.support.source);cls.model=deepcopy(cls.source)
        load_renderer_student(cls.model,cls.support.path)
        with torch.no_grad():cls.model.flow_renderer.output.bias.add_(.03)
        cls.folder=tempfile.TemporaryDirectory();cls.path=Path(cls.folder.name)/'rf7.pt'
        export_rf_chain_renderer(cls.model,cls.support.path,cls.path,local_updates=3,source_model=cls.source)
        cls.payload=torch.load(cls.path,map_location='cpu',weights_only=False)

    @classmethod
    def tearDownClass(cls):cls.folder.cleanup();cls.support.tearDownClass()

    def test_nested7_6_3_strict_roundtrip_same_Base_and_codec_actual_RF3_RGB(self):
        result=deepcopy(self.source);clock=torch.get_rng_state().clone()
        meta=load_renderer_student(result,self.path)
        self.assertTrue(torch.equal(clock,torch.get_rng_state()))
        self.assertEqual(meta['format_version'],7);self.assertEqual(meta['local_updates'],3)
        self.assertEqual(meta['refinement_updates'],self.support.payload['refinement_updates']+3)
        self.assertEqual(meta['base_training_steps'],self.support.payload['base_training_steps'])
        self.assertEqual(_hash_tensors(result.base_student.state_dict()),_hash_tensors(self.model.base_student.state_dict()))
        self.assertEqual(_hash_tensors(result.flow_renderer.state_dict()),_hash_tensors(self.model.flow_renderer.state_dict()))
        self.assertEqual(_hash_tensors(rf_chain_state(self.payload)),self.payload['model_sha256'])
        data=batch((False,));features=torch.zeros(1,128)
        for net in(result,self.model):net.eval().requires_grad_(False)
        with torch.inference_mode():
            a=result.render(data['state'],features,result.encode_map(data['map']))
            b=self.model.render(data['state'],features,self.model.encode_map(data['map']))
        self.assertTrue(torch.equal(a,b));self.assertFalse(meta['automatic_promotion'])

    def test_override_and_parent_tamper_rejected_transactionally(self):
        target=deepcopy(self.source);before=_hash_tensors(target.state_dict())
        mutations=(lambda p:p.update(base_student_identity='a'*64),lambda p:p.update(codec_sha256='a'*64),
            lambda p:p.update(solver_steps=5),lambda p:p.update(parent_bundle_sha256='a'*64),
            lambda p:p['rf_override'].update({'codec.decoder.conv_out.bias':torch.ones(3)}),
            lambda p:p.update(refinement_updates=3))
        for mutate in mutations:
            payload=deepcopy(self.payload);mutate(payload);path=Path(self.folder.name)/'bad.pt';torch.save(payload,path)
            with self.assertRaises((ValueError,RuntimeError)):load_renderer_student(target,path)
            self.assertEqual(_hash_tensors(target.state_dict()),before)

    def test_frozen_Base_private_D_changes_rejected_before_export(self):
        for mutate in (lambda m:m.base_student.rgb.bias.add_(.01),
                       lambda m:m.flow_renderer.geometry_opacity[0].bias.add_(.01),
                       lambda m:m.flow_renderer.codec.decoder.conv_out.bias.add_(.01)):
            net=deepcopy(self.model)
            with torch.no_grad():mutate(net)
            with self.assertRaises(ValueError):export_rf_chain_renderer(net,self.support.path,
                Path(self.folder.name)/'bad.pt',local_updates=4,source_model=self.source)

    def test_RF_overlay_does_not_trigger_legacy_missing_normalization_defaults(self):
        from turkeyworld.latent_flow_renderer import LatentFlowRenderer
        original=LatentFlowRenderer._load_from_state_dict
        visits=[]
        def guarded(module,state,prefix,*args,**kwargs):
            # Legacy full-checkpoint compatibility creates zero/unit values
            # when BOTH fields are absent. An RF-only overlay must never take
            # that branch, even though unit-stat toy fixtures hide its effect.
            self.assertIn(prefix+'latent_mean',state)
            self.assertIn(prefix+'latent_std',state)
            visits.append((state[prefix+'latent_mean'].clone(),state[prefix+'latent_std'].clone()))
            return original(module,state,prefix,*args,**kwargs)
        with patch.object(LatentFlowRenderer,'_load_from_state_dict',new=guarded):
            target=deepcopy(self.source);load_renderer_student(target,self.path)
        self.assertGreater(len(visits),0)
        self.assertTrue(torch.equal(target.flow_renderer.latent_mean,self.model.flow_renderer.latent_mean))
        self.assertTrue(torch.equal(target.flow_renderer.latent_std,self.model.flow_renderer.latent_std))


if __name__=='__main__':unittest.main()
