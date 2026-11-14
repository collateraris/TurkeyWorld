from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch

import test_decoder_rf_adaptation_bundle as support
from test_renderer_enemy_objectives import batch
from turkeyworld.base_expanded_experts import upgrade_base_experts
from turkeyworld.base_geometry_bundle import export_base_geometry,PROTOCOL
from turkeyworld.decoder_rf_adaptation_bundle import export_decoder_adaptation
from turkeyworld.renderer_student import _hash_tensors,load_renderer_student


class BaseGeometryBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.support=support.DecoderRFAdaptationBundleTests;cls.support.setUpClass()
        cls.source=deepcopy(cls.support.source);cls.model=deepcopy(cls.support.adapted)
        upgrade_base_experts(cls.model.base_student)
        cls.folder=tempfile.TemporaryDirectory();cls.root=Path(cls.folder.name);cls.path=cls.root/'base.pt'
        export_base_geometry(cls.model,cls.support.path,cls.path,local_updates=0,source_model=cls.source)
        cls.payload=torch.load(cls.path,map_location='cpu',weights_only=False)

    @classmethod
    def tearDownClass(cls):cls.folder.cleanup();cls.support.tearDownClass()

    def test_zero_activation_and_trained_full_bundle_rebind_without_changing_RF_or_codec(self):
        target=deepcopy(self.source);rng=torch.get_rng_state().clone()
        meta=load_renderer_student(target,self.path)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(meta['format_version'],6);self.assertEqual(meta['objective_protocol'],PROTOCOL)
        self.assertEqual(meta['training_steps'],self.support.payload['training_steps'])
        self.assertEqual(meta['base_training_steps'],29209)
        self.assertEqual(sum(p.numel()for p in target.base_student.parameters()),2498447)
        self.assertEqual(_hash_tensors(target.flow_renderer.state_dict()),self.support.payload['adapted_flow_sha256'])
        query=batch((False,));features=torch.zeros(1,128)
        for net in(target,self.support.adapted):net.eval().requires_grad_(False)
        with torch.inference_mode():
            a=target.render_base(query['state'],features,target.encode_map(query['map']))
            b=self.support.adapted.render_base(query['state'],features,self.support.adapted.encode_map(query['map']))
        self.assertTrue(torch.equal(a,b))
        working=deepcopy(self.model)
        with torch.no_grad():working.base_student.rgb.bias.add_(.2)
        path=self.root/'trained.pt'
        export_base_geometry(working,self.support.path,path,local_updates=3,source_model=self.source)
        loaded=deepcopy(self.source);meta=load_renderer_student(loaded,path)
        self.assertEqual(meta['base_training_steps'],29212);self.assertEqual(meta['decoder_refinement_updates'],20)
        self.assertEqual(meta['renderer_training_steps'],self.support.payload['training_steps'])
        self.assertEqual(_hash_tensors(loaded.base_student.state_dict()),_hash_tensors(working.base_student.state_dict()))
        self.assertEqual(loaded.flow_renderer._renderer_base_identity,meta['base_student_identity'])
        self.assertNotEqual(meta['base_student_identity'],meta['parent_base_student_identity'])
        self.assertFalse(meta['RF_base_condition_binding']['RF_adapted_to_new_Base'])
        self.assertFalse(meta['gameplay_qualified'])

    def test_tampered_Base_RF_binding_or_unflagged_architecture_is_transactional(self):
        changes={
            'RFclock':lambda p:p.update(training_steps=p['training_steps']+1),
            'Dclock':lambda p:p.update(decoder_refinement_updates=21),
            'Baseclock':lambda p:p.update(base_training_steps=29210),
            'oldflag':lambda p:p['base_student_bundle']['config'].update(output_moe_expanded_enabled=False),
            'binding':lambda p:p['RF_base_condition_binding'].update(RF_adapted_to_new_Base=True),
            'parent':lambda p:p.update(parent_bundle_sha256='a'*64),
            'frozenflow':lambda p:p.update(frozen_flow_sha256='a'*64),
            'basemissing':lambda p:p['base_student_bundle']['model'].pop('rgb.bias'),
            'moeheader':lambda p:p['renderer_output_moe']['base'].update(kind='renderer_token_output_moe_five_v2'),
        }
        target=deepcopy(self.source);before=_hash_tensors(target.state_dict());ids=[id(p)for p in target.parameters()]
        for key,mutate in changes.items():
            broken=deepcopy(self.payload);mutate(broken);path=self.root/'bad.pt';torch.save(broken,path)
            rng=torch.get_rng_state().clone()
            with self.subTest(key=key),self.assertRaises((ValueError,RuntimeError,TypeError)):
                load_renderer_student(target,path)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(_hash_tensors(target.state_dict()),before)
            self.assertEqual([id(p)for p in target.parameters()],ids)

    def test_frozen_RF_codec_or_geometry_cannot_be_hidden_and_old_format3_stays_strict(self):
        for change in[lambda m:m.flow_renderer.output.bias.add_(.01),
                      lambda m:m.flow_renderer.codec.decoder.conv_out.bias.add_(.01),
                      lambda m:m.flow_renderer.fixed_noise.add_(.01)]:
            net=deepcopy(self.model)
            with torch.no_grad():change(net)
            with self.assertRaises(ValueError):
                export_base_geometry(net,self.support.path,self.root/'bad-export.pt',local_updates=1,source_model=self.source)
        with self.assertRaises(ValueError):
            export_decoder_adaptation(self.model,self.support.parent,self.root/'wrong-format3.pt',
                decoder_updates=20,source_model=self.source)


if __name__=='__main__':unittest.main()
