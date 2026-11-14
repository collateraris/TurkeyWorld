from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_decoder_rf_adaptation_bundle as support
from test_decoder_rf_adaptation_validation import HeldOut
from turkeyworld.base_expanded_experts import upgrade_base_experts
from turkeyworld.base_geometry_phase import configure_base_geometry_training
from turkeyworld.base_geometry_objectives import base_geometry_refinement_loss,base_geometry_forward
from turkeyworld.base_geometry_validation import evaluate_base_geometry,save_base_geometry_preview,PROTOCOL,KIND
from turkeyworld.renderer_moe_objectives import moe_conditions
from turkeyworld.renderer_student import _hash_tensors
from turkeyworld.staged_validation import _isolated_validation


class BaseGeometryValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.support=support.DecoderRFAdaptationBundleTests;cls.support.setUpClass()
        cls.reference=deepcopy(cls.support.adapted).eval().requires_grad_(False)
        cls.model=deepcopy(cls.reference);upgrade_base_experts(cls.model.base_student)
        configure_base_geometry_training(cls.model);cls.panel=[0,1,2,3,4]

    @classmethod
    def tearDownClass(cls):cls.support.tearDownClass()

    def test_zero_Base_function_and_five_measured_branches_never_call_RF_or_Source(self):
        net,ref,data=deepcopy(self.model),deepcopy(self.reference),HeldOut()
        with patch.object(net.flow_renderer,'latent_sample',side_effect=AssertionError('No RF')),\
             patch.object(ref.flow_renderer,'latent_sample',side_effect=AssertionError('No reference RF')),\
             patch('turkeyworld.engine.Game.render',side_effect=AssertionError('No Source RGB inference')),\
             patch('turkeyworld.engine.Game.step',side_effect=AssertionError('No Source transition')):
            report,arrays=evaluate_base_geometry(net,ref,data,self.panel,[0,2,3],step=0,batch_size=2)
        self.assertEqual(report['kind'],KIND);self.assertEqual(report['protocol'],PROTOCOL)
        self.assertEqual(set(report['models']),{'base_old','base_new','base_codec_old','base_codec_new','source_codec'})
        # The frozen reference and trainable current Base select different CPU
        # MHA paths by requires_grad flags even under inference_mode. The measured
        # RGB floor is <3e-7. With matched flags, the zero append is bit exact.
        torch.testing.assert_close(arrays['base_old'],arrays['base_new'],rtol=0,atol=5e-7)
        # Tiny changes before a hard nearest-code lookup need not preserve a
        # quantizer cell. Both actual codec branches remain separately measured.
        self.assertEqual(report['models']['base_codec_old']['summary']['frames'],5)
        self.assertEqual(report['models']['base_codec_new']['summary']['frames'],5)
        self.assertAlmostEqual(report['models']['base_old']['summary']['ssim_mean'],
                               report['models']['base_new']['summary']['ssim_mean'],places=6)
        matched=deepcopy(ref);matched.base_student.requires_grad_(True)
        with _isolated_validation(net,data,173),_isolated_validation(matched,data,173):
            conditions=moe_conditions(net,data.batch(2,indices=[0,1]))
            a,_=base_geometry_forward(net,conditions);b,_=base_geometry_forward(matched,conditions)
            codec=net.flow_renderer.codec
            qa=codec.decode(codec.encode(a)['quantized']);qb=codec.decode(codec.encode(b)['quantized'])
        self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.equal(qa,qb))
        self.assertEqual(report['RF_forward_calls'],0);self.assertEqual(report['geometry']['frames'],3)
        self.assertEqual(report['best_metric'],'models.base_new.summary.ssim_mean')
        self.assertTrue(report['full_codec_tensors_unchanged']);self.assertTrue(report['RF_tensors_unchanged'])
        self.assertFalse(report['RF_adapted_to_new_Base']);self.assertFalse(report['budget_completion_is_quality'])
        self.assertFalse(report['Source_Q_image_branch']);self.assertFalse(report['restorer_active'])
        self.assertFalse(report['passed']);json.dumps(report,allow_nan=False)

    def test_MAIN_is_actual_image_count_weighted_and_model_flags_RNG_sampler_labels_are_untouched(self):
        net,ref,data=deepcopy(self.model),deepcopy(self.reference),HeldOut()
        hashes=[_hash_tensors(n.state_dict())for n in(net,ref)]
        modes=[{k:m.training for k,m in n.named_modules()}for n in(net,ref)]
        flags=[{k:p.requires_grad for k,p in n.named_parameters()}for n in(net,ref)]
        sampler=data.sampler_state();rng=torch.get_rng_state().clone();numpy=deepcopy(np.random.get_state());python=random.getstate()
        seen=[];actual=data.batch
        def batch(size,*,indices):
            torch.rand(1);np.random.random();random.random();data.rng.random();data.draws+=size
            return actual(size,indices=indices)
        def objective(n,b,**kw):
            loss,metrics=base_geometry_refinement_loss(n,b,**kw);seen.append((len(b['state']),float(loss),metrics));return loss,metrics
        for key in('state','target','background','object_mask'):data.values[key].requires_grad_(True)
        with patch.object(data,'batch',side_effect=batch):
            report,_=evaluate_base_geometry(net,ref,data,self.panel,step=2,batch_size=2,objective_fn=objective)
        self.assertEqual([n for n,_,_ in seen],[2,2,1])
        self.assertAlmostEqual(report['val_loss'],sum(n*l for n,l,_ in seen)/5,places=7)
        self.assertAlmostEqual(report['validation_objective']['components']['global_ssim_loss'],
                               sum(n*m['global_ssim_loss']for n,_,m in seen)/5,places=7)
        self.assertEqual(report['validation_objective']['frames'],5)
        self.assertTrue(report['validation_objective_matches_main_training'])
        self.assertEqual(hashes,[_hash_tensors(n.state_dict())for n in(net,ref)])
        self.assertEqual(modes,[{k:m.training for k,m in n.named_modules()}for n in(net,ref)])
        self.assertEqual(flags,[{k:p.requires_grad for k,p in n.named_parameters()}for n in(net,ref)])
        self.assertEqual(sampler,data.sampler_state());self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(np.array_equal(numpy[1],np.random.get_state()[1]));self.assertEqual(python,random.getstate())
        self.assertTrue(all(p.grad is None for n in(net,ref)for p in n.parameters()))
        self.assertTrue(all(data.values[k].grad is None for k in('state','target','background','object_mask')))

    def test_stale_parent_or_panels_reject_and_preview_keeps_raw_vs_codec_rows_distinct(self):
        from PIL import Image
        net,ref,data=deepcopy(self.model),deepcopy(self.reference),HeldOut()
        report,arrays=evaluate_base_geometry(net,ref,data,self.panel,step=0,batch_size=2)
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'preview.png';meta=save_base_geometry_preview(arrays,path,0)
            with Image.open(path)as image:self.assertEqual(image.size,(768,1085))
        self.assertEqual(meta['kind'],'base_geometry_GT_query');self.assertEqual(meta['RF_forward_calls'],0)
        self.assertEqual(meta['rows'],['source','base_old','base_new','base_codec_old','base_codec_new'])
        for key in('panel_fingerprint','data_identity_sha256','codec_sha256','frozen_flow_sha256'):
            bad=deepcopy(report);bad[key]='foreign'
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'identical'):
                evaluate_base_geometry(net,ref,data,self.panel,step=1,baseline=bad)
        with torch.no_grad():net.flow_renderer.codec.decoder.conv_out.bias.add_(.01)
        with self.assertRaisesRegex(ValueError,'byte-identical'):
            evaluate_base_geometry(net,ref,data,self.panel,step=1)
        data.split='train'
        with self.assertRaisesRegex(ValueError,'held-out'):
            evaluate_base_geometry(self.model,self.reference,data,self.panel,step=0)


if __name__=='__main__':unittest.main()
