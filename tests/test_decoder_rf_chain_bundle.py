from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch

import test_base_geometry_bundle as support
from test_renderer_enemy_objectives import batch
from turkeyworld.decoder_rf_chain_bundle import export_decoder_rf_chain,load_decoder_rf_chain_payload
from turkeyworld.decoder_rf_adaptation_objectives import generate_decoder_training_codes,decoder_rf_adaptation_loss
from turkeyworld.renderer_rf_chain_bundle import export_rf_chain_renderer,load_rf_chain_payload
from turkeyworld.renderer_student import _hash_tensors,load_renderer_student


def query():
    value=batch((False,));value['navigation_only']=torch.zeros(1,dtype=torch.bool)
    value['dynamic']=torch.zeros(1,dtype=torch.bool)
    return value


class DecoderRFChainBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.support=support.BaseGeometryBundleTests;cls.support.setUpClass()
        cls.folder=tempfile.TemporaryDirectory();cls.root=Path(cls.folder.name)
        cls.source=deepcopy(cls.support.source);cls.model=deepcopy(cls.source)
        load_renderer_student(cls.model,cls.support.path)
        with torch.no_grad():cls.model.flow_renderer.output.bias.add_(.03)
        cls.parent=cls.root/'rf7.pt'
        export_rf_chain_renderer(cls.model,cls.support.path,cls.parent,local_updates=3,source_model=cls.source)
        cls.parent_payload=torch.load(cls.parent,map_location='cpu',weights_only=False)
        cls.adapted=deepcopy(cls.model)
        with torch.no_grad():
            cls.adapted.flow_renderer.codec.decoder.conv_out.bias.add_(.11)
            cls.adapted.flow_renderer.codec.post_quant_conv.weight.add_(.01)
        cls.path=cls.root/'decoder8.pt'
        export_decoder_rf_chain(cls.adapted,cls.parent,cls.path,local_updates=7,source_model=cls.source)
        cls.payload=torch.load(cls.path,map_location='cpu',weights_only=False)

    @classmethod
    def tearDownClass(cls):cls.folder.cleanup();cls.support.tearDownClass()

    def test_actual_format8_roundtrip_keeps_expanded_Base_RF_EQ_and_independent_clocks(self):
        target=deepcopy(self.source);rng=torch.get_rng_state().clone()
        meta=load_decoder_rf_chain_payload(target,self.payload,checkpoint=self.path)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(meta['format_version'],8)
        self.assertEqual(meta['training_steps'],self.parent_payload['training_steps'])
        self.assertEqual(meta['base_training_steps'],self.parent_payload['base_training_steps'])
        self.assertEqual(meta['decoder_parent_updates'],20);self.assertEqual(meta['decoder_refinement_updates'],27)
        self.assertEqual(meta['decoder_local_updates'],7);self.assertEqual(meta['rf_chain_local_updates'],3)
        self.assertFalse(meta['restorer_present']);self.assertFalse(meta['gameplay_qualified'])
        self.assertEqual(_hash_tensors(target.base_student.state_dict()),_hash_tensors(self.model.base_student.state_dict()))
        self.assertEqual(_hash_tensors(target.flow_renderer.state_dict()),_hash_tensors(self.adapted.flow_renderer.state_dict()))
        target.eval().requires_grad_(False);reference=deepcopy(self.model).eval().requires_grad_(False)
        data=query();old=generate_decoder_training_codes(reference,data);new=generate_decoder_training_codes(target,data)
        self.assertTrue(torch.equal(old[0],new[0]));self.assertTrue(torch.equal(old[1],new[1]))
        with torch.no_grad():
            self.assertFalse(torch.equal(reference.flow_renderer.codec.decode(old[0]),target.flow_renderer.codec.decode(new[0])))
        # The public dispatch is owned by the root, but when available it must return the same exact model.
        loaded=deepcopy(self.source);public=load_renderer_student(loaded,self.path)
        self.assertEqual(public['format_version'],8)
        self.assertEqual(_hash_tensors(loaded.flow_renderer.state_dict()),meta['adapted_flow_sha256'])

    def test_changed_overlay_parent_counter_or_frozen_hash_is_transactionally_rejected(self):
        changes=[lambda p:p.update(decoder_refinement_updates=28),lambda p:p.update(base_training_steps=29210),
            lambda p:p.update(training_steps=p['training_steps']+1),lambda p:p.update(solver_steps=5),
            lambda p:p.update(parent_bundle_sha256='a'*64),lambda p:p.update(frozen_flow_sha256='b'*64),
            lambda p:p.update(encoder_quantizer_sha256='c'*64),lambda p:p.update(adapted_codec_sha256='d'*64),
            lambda p:p.update(restorer_present=True),lambda p:p['decoder_override'].pop(next(iter(p['decoder_override']))),
            lambda p:p['decoder_override'][next(iter(p['decoder_override']))].add_(.1)]
        target=deepcopy(self.source);before=_hash_tensors(target.state_dict());ids=[id(p)for p in target.parameters()]
        for mutate in changes:
            broken=deepcopy(self.payload);mutate(broken);rng=torch.get_rng_state().clone()
            with self.assertRaises((ValueError,RuntimeError,TypeError)):
                load_decoder_rf_chain_payload(target,broken,checkpoint=self.path)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(_hash_tensors(target.state_dict()),before)
            self.assertEqual([id(p)for p in target.parameters()],ids)

    def test_exact_new_Base_RF3_paired_objective_only_trains_D_and_export_rejects_RF_or_EQ_change(self):
        model=deepcopy(self.model);data=query();before=deepcopy(model.state_dict())
        value,metrics=decoder_rf_adaptation_loss(model,data);value.backward()
        self.assertTrue(torch.isfinite(value));self.assertEqual(metrics['branch_weights'],dict(rf_quantized=.5,source_quantized=.5))
        gradients=[n for n,p in model.named_parameters()if p.grad is not None]
        self.assertTrue(gradients);self.assertTrue(all(n.startswith(('flow_renderer.codec.decoder.','flow_renderer.codec.post_quant_conv.'))for n in gradients))
        for key,value in model.state_dict().items():self.assertTrue(torch.equal(value,before[key]),key)
        for mutate in(lambda m:m.flow_renderer.output.bias.add_(.1),
                       lambda m:m.flow_renderer.codec.encoder.conv_out.bias.add_(.1),
                       lambda m:m.base_student.rgb.bias.add_(.1)):
            changed=deepcopy(self.adapted)
            with torch.no_grad():mutate(changed)
            with self.assertRaises(ValueError):export_decoder_rf_chain(changed,self.parent,self.root/'bad.pt',local_updates=7,source_model=self.source)


if __name__=='__main__':unittest.main()
