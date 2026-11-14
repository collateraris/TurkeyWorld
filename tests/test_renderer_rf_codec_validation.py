from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_decoder_rf_adaptation_validation import HeldOut
from test_renderer_rf5_objectives import model
from turkeyworld.renderer_rf_codec_objectives import rf_codec_refinement_loss
from turkeyworld.renderer_rf_codec_validation import evaluate_rf_codec,save_rf_codec_preview,PROTOCOL,KIND
from turkeyworld.renderer_student import _hash_tensors


class RFCodecValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model=model(3).eval().requires_grad_(False)
        cls.model.flow_renderer.train()
        for name,parameter in cls.model.flow_renderer.named_parameters():
            parameter.requires_grad_(not name.startswith(('codec.','geometry_opacity.')))
        cls.model.flow_renderer.codec.eval();cls.model.flow_renderer.geometry_opacity.eval()
        cls.reference=deepcopy(cls.model).eval().requires_grad_(False)
        cls.panel=[0,1,2,3,4]

    def test_actual_current_decoder_same_RF_comparison_geometry_and_no_restorer_Source_feedback(self):
        net,ref,data=deepcopy(self.model),deepcopy(self.reference),HeldOut()
        with patch('turkeyworld.engine.Game.render',side_effect=AssertionError('No Source render')),\
             patch('turkeyworld.engine.Game.step',side_effect=AssertionError('No Source transition')):
            report,arrays=evaluate_rf_codec(net,ref,data,self.panel,[0,2,3],step=0,batch_size=2)
        self.assertEqual(report['kind'],KIND);self.assertEqual(report['protocol'],PROTOCOL)
        self.assertEqual(set(report['models']),{'base','rf_old','rf_new','source_codec'})
        self.assertTrue(torch.equal(arrays['rf_old'],arrays['rf_new']))
        self.assertEqual(report['models']['rf_old'],report['models']['rf_new'])
        self.assertTrue(report['full_codec_tensors_unchanged']);self.assertFalse(report['restorer_active'])
        self.assertEqual(report['geometry']['frames'],3)
        self.assertEqual(report['validation_objective']['frames'],5)
        self.assertTrue(report['validation_objective_matches_main_training'])
        self.assertTrue(report['validation_objective']['main_only'])
        for key in ['world_rollout','autonomous_rollout','sr_quality_evaluated','Source_Q_image_branch',
                    'gameplay_qualified','future_source_rgb_in_prior','future_source_pose_in_prior']:
            self.assertFalse(report[key],key)
        self.assertFalse(report['rf_codec_ssim_goal']['goal_met'])
        json.dumps(report,allow_nan=False)

    def test_actual_MAIN_unequal_tail_preserves_weights_modes_RNG_private_sampler_and_label_grads(self):
        net,ref,data=deepcopy(self.model),deepcopy(self.reference),HeldOut()
        hashes=[_hash_tensors(n.state_dict())for n in (net,ref)]
        modes=[{k:m.training for k,m in n.named_modules()}for n in (net,ref)]
        sampler=data.sampler_state();rng=torch.get_rng_state().clone();numpy=deepcopy(np.random.get_state());python=random.getstate()
        seen=[];actual=data.batch
        def batch(size,*,indices):
            torch.rand(1);np.random.random();random.random();data.rng.random();data.draws+=size
            return actual(size,indices=indices)
        def objective(n,b,**kwargs):
            loss,metrics=rf_codec_refinement_loss(n,b,**kwargs)
            seen.append((len(b['state']),float(loss),metrics));return loss,metrics
        for key in ['target','background','object_mask','state']:data.values[key].requires_grad_(True)
        with patch.object(data,'batch',side_effect=batch):
            report,_=evaluate_rf_codec(net,ref,data,self.panel,step=1,batch_size=2,objective_fn=objective)
        self.assertEqual([n for n,_,_ in seen],[2,2,1])
        self.assertAlmostEqual(report['val_loss'],sum(n*l for n,l,_ in seen)/5,places=7)
        self.assertAlmostEqual(report['validation_objective']['components']['global_ssim_loss'],
            sum(n*m['global_ssim_loss']for n,_,m in seen)/5,places=7)
        self.assertEqual(hashes,[_hash_tensors(n.state_dict())for n in (net,ref)])
        self.assertEqual(modes,[{k:m.training for k,m in n.named_modules()}for n in (net,ref)])
        self.assertEqual(sampler,data.sampler_state());self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(np.array_equal(numpy[1],np.random.get_state()[1]));self.assertEqual(python,random.getstate())
        self.assertTrue(all(p.grad is None for n in (net,ref)for p in n.parameters()))
        self.assertTrue(all(data.values[k].grad is None for k in ['target','background','object_mask','state']))

    def test_frozen_codec_or_foreign_baseline_is_rejected_and_preview_is_native_five_rows(self):
        from PIL import Image
        net,ref,data=deepcopy(self.model),deepcopy(self.reference),HeldOut()
        report,arrays=evaluate_rf_codec(net,ref,data,self.panel,step=0,batch_size=2)
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'preview.png';meta=save_rf_codec_preview(arrays,path,0)
            with Image.open(path)as im:self.assertEqual(im.size,(768,1085))
        self.assertEqual(meta['rows'],['source','base','rf_old','rf_new','source_codec'])
        self.assertFalse(meta['restorer_active']);self.assertEqual(meta['kind'],'renderer_rf_codec_GT_query')
        for key in ['panel_fingerprint','data_identity_sha256','codec_sha256','solver_steps','protocol']:
            bad=deepcopy(report);bad[key]='foreign'
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'identical'):
                evaluate_rf_codec(net,ref,data,self.panel,step=1,baseline=bad)
        with torch.no_grad():net.flow_renderer.codec.decoder.conv_out.bias.add_(.01)
        with self.assertRaisesRegex(ValueError,'same frozen'):
            evaluate_rf_codec(net,ref,data,self.panel,step=1)
        data.split='train'
        with self.assertRaisesRegex(ValueError,'held-out'):
            evaluate_rf_codec(self.model,self.reference,data,self.panel,step=0)


if __name__=='__main__':unittest.main()
