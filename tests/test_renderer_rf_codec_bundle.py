from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch

from test_decoder_rf_adaptation_bundle import DecoderRFAdaptationBundleTests as Support
from test_renderer_enemy_objectives import batch
from turkeyworld.renderer_rf_codec_bundle import export_rf_codec_renderer, PROTOCOL
from turkeyworld.renderer_rf_codec_phase import rf_codec_trainable_names
from turkeyworld.renderer_student import _hash_tensors, load_renderer_student


class RFCodecBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        Support.setUpClass()
        cls.source, cls.model = deepcopy(Support.source), deepcopy(Support.adapted)
        cls.folder = tempfile.TemporaryDirectory()
        cls.path = Path(cls.folder.name)/'rf-codec.pt'
        with torch.no_grad():cls.model.flow_renderer.output.bias.add_(.1)
        export_rf_codec_renderer(cls.model, Support.path, cls.path,
                                 local_updates=3, source_model=cls.source)
        cls.payload = torch.load(cls.path,map_location='cpu',weights_only=False)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup();Support.tearDownClass()

    def test_actual_format5_roundtrip_has_only_RF_override_and_keeps_current_D_and_independent_clocks(self):
        target, rng = deepcopy(self.source), torch.get_rng_state().clone()
        codec_sha = _hash_tensors(self.model.flow_renderer.codec.state_dict())
        meta = load_renderer_student(target, self.path)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(meta['format_version'],5);self.assertEqual(meta['objective_protocol'],PROTOCOL)
        self.assertEqual(meta['training_steps'],Support.payload['training_steps']+3)
        self.assertEqual(meta['decoder_refinement_updates'],20)
        self.assertEqual(meta['base_training_steps'],29209)
        self.assertEqual(meta['rf_parameter_names'],rf_codec_trainable_names(target))
        self.assertEqual(_hash_tensors(target.flow_renderer.state_dict()),_hash_tensors(self.model.flow_renderer.state_dict()))
        self.assertEqual(_hash_tensors(target.flow_renderer.codec.state_dict()),codec_sha)
        self.assertFalse(hasattr(target.flow_renderer,'conditional_restorer'))
        self.assertFalse(meta['gameplay_qualified'])
        query = batch((False,));features = torch.zeros(1,128)
        for net in (target,self.model):net.eval().requires_grad_(False)
        with torch.inference_mode():
            a=target.render(query['state'],features,target.encode_map(query['map']))
            b=self.model.render(query['state'],features,self.model.encode_map(query['map']))
        self.assertTrue(torch.equal(a,b))

    def test_tamper_cannot_expand_RF_allowlist_or_mutate_live_teacher(self):
        mutations={
            'codec_override':lambda p:p['rf_override'].update({'codec.decoder.conv_out.bias':torch.zeros(3)}),
            'missing_RF':lambda p:p['rf_override'].pop(next(iter(p['rf_override']))),
            'restorer':lambda p:p.update(restorer_present=True),
            'Dclock':lambda p:p.update(decoder_refinement_updates=1726),
            'RFclock':lambda p:p.update(renderer_training_steps=1),
            'solver':lambda p:p.update(solver_steps=5),
            'protocol':lambda p:p.update(objective_protocol='renderer_rf_expanded_experts_v1'),
            'parent':lambda p:p.update(parent_bundle_sha256='a'*64),
            'Base':lambda p:p.update(base_student_identity='a'*64),
            'nonfinite':lambda p:p['rf_override'][next(iter(p['rf_override']))].fill_(float('nan')),
        }
        target=deepcopy(self.source);before=_hash_tensors(target.state_dict());ids=[id(x)for x in target.parameters()]
        for label,mutate in mutations.items():
            broken=deepcopy(self.payload);mutate(broken)
            broken['rf_override_sha256']=_hash_tensors(broken['rf_override'])
            path=Path(self.folder.name)/'bad.pt';torch.save(broken,path);rng=torch.get_rng_state().clone()
            with self.subTest(label=label),self.assertRaises((ValueError,RuntimeError,TypeError)):
                load_renderer_student(target,path)
            self.assertEqual(_hash_tensors(target.state_dict()),before)
            self.assertEqual([id(x)for x in target.parameters()],ids)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_export_rejects_frozen_assets_buffers_and_restorer_even_if_RF_is_valid(self):
        for change in [lambda m:m.flow_renderer.codec.decoder.conv_out.bias.add_(.01),
                       lambda m:m.flow_renderer.geometry_opacity[0].weight.add_(.01),
                       lambda m:m.flow_renderer.fixed_noise.add_(.01),
                       lambda m:m.base_student.output_moe.output.bias.add_(.01)]:
            net=deepcopy(self.model)
            with torch.no_grad():change(net)
            with self.assertRaises(ValueError):
                export_rf_codec_renderer(net,Support.path,Path(self.folder.name)/'bad-export.pt',
                    local_updates=1,source_model=self.source)
        net=deepcopy(self.model);net.flow_renderer.conditional_restorer=torch.nn.Identity()
        with self.assertRaisesRegex(ValueError,'restorer'):
            export_rf_codec_renderer(net,Support.path,Path(self.folder.name)/'bad-export.pt',
                local_updates=1,source_model=self.source)


if __name__=='__main__':unittest.main()
