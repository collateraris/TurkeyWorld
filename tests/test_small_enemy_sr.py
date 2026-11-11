"""Independent CPU checks for the ordered pipeline's true32-to128 SR stage."""
import copy
import random
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from turkeyworld.compact_codec import CompactCodec, compact_codec_config
from turkeyworld.models import EDSRUpscaler
from turkeyworld.train_small_enemy_sr import evaluate_small_sr
from turkeyworld.vqgan import VQGANConfig


class PairedHeldout:
    """All required cohorts; HR is a distinct same-tick target, not LR resize."""
    def __init__(self):
        self.rng = np.random.default_rng(103)
        self.calls = []
        self.lr = torch.linspace(.1, .8, 6*3*32*32).reshape(6, 3, 32, 32)
        checker = ((torch.arange(128)[:, None]+torch.arange(128)[None, :]) % 2)*.04
        self.hr = (F.interpolate(self.lr, scale_factor=4, mode='nearest')+checker).requires_grad_()
        self.mask = torch.zeros(6, 32, 32, requires_grad=False)
        self.mask[0, 8, 8] = 1
        self.mask[1, 8:12, 8:12] = 1
        self.mask[2, 8:12, 8:13] = 1
        self.mask[3, 8:12, 8:12] = 1
        self.mask[4, 8, 8] = 1
        self.hr_mask = self.mask.repeat_interleave(4, -1).repeat_interleave(4, -2).requires_grad_()
        self.state = torch.zeros(6, 20)
        self.state[:, 4] = 1
        self.state[:3, 12:14] = 1
        self.state[4, 12:14] = 1
        self.state[4, 4] = 0
        self.state[:, 8] = 1/16
        self.state.requires_grad_()

    def panel_fingerprint(self, panel):
        return 'heldout_true_source32_128_'+','.join(map(str, panel))

    def batch(self, count, *, indices):
        assert count == len(indices)
        self.calls.append(list(indices))
        self.rng.random(); np.random.random(); random.random(); torch.rand(1)
        ids = torch.as_tensor(indices)
        return {'lr': self.lr[ids], 'hr': self.hr[ids],
                'object_mask': self.mask[ids], 'hr_object_mask': self.hr_mask[ids],
                'state': self.state[ids]}


class TinySR(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(.01))
        self.branch = nn.Identity()

    def forward(self, image):
        return (F.interpolate(image, scale_factor=4, mode='bilinear', align_corners=False)+self.offset).clamp(0, 1)


class TinyFrozenCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(.02), requires_grad=False)
        self.register_buffer('unchanged', torch.arange(3))
        self.inputs = []

    def forward(self, image):
        self.inputs.append(image.detach().clone())
        return {'reconstruction': (image+self.offset).clamp(0, 1)}


class SmallEnemySRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_fixed_native_cohorts_true_hr_target_both_domains_rng_modes_and_preview(self):
        data = PairedHeldout(); model = TinySR().train(); model.branch.eval()
        codec = TinyFrozenCodec().train()
        modes = [m.training for m in model.modules()]
        codec_modes = [m.training for m in codec.modules()]
        codec_before = {k:v.clone() for k,v in codec.state_dict().items()}
        private = copy.deepcopy(data.rng.bit_generator.state)
        python = random.getstate(); numpy = np.random.get_state(); tensor = torch.get_rng_state().clone()
        report, preview = evaluate_small_sr(model, codec, data, list(range(6)), step=120, batch_size=2)
        self.assertEqual(data.calls, [[0,1],[2,3],[4,5]]*2)
        self.assertEqual(private, data.rng.bit_generator.state)
        self.assertEqual(python, random.getstate())
        self.assertTrue(all(np.array_equal(a,b) for a,b in zip(numpy,np.random.get_state())))
        self.assertTrue(torch.equal(tensor, torch.get_rng_state()))
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertEqual(codec_modes, [m.training for m in codec.modules()])
        for key, value in codec_before.items(): self.assertTrue(torch.equal(value, codec.state_dict()[key]), key)
        for domain in ('clean', 'codec'):
            rows = report['small_enemy_cohorts'][domain]
            self.assertEqual({k:v['frames'] for k,v in rows.items()},
                             {'overall':6,'tiny':2,'near':1,'corpse':1,'hp0':1,'navigation':1})
        expected = (model(data.lr).detach()-data.hr.detach()).abs().mean().item()
        self.assertAlmostEqual(report['clean']['summary']['mae_mean'], expected, places=6)
        self.assertNotEqual(report['clean']['summary']['mae_mean'], report['codec']['summary']['mae_mean'])
        self.assertEqual(preview['source_lr'].shape, (4,32,32,3))
        for key in ('sr','source_hr','difference'):
            self.assertEqual(preview[key].shape, (4,128,128,3)); self.assertEqual(preview[key].dtype,np.uint8)
        self.assertTrue(preview['metadata']['source_input_allowed_for_sr'])
        self.assertFalse(preview['metadata']['world_rollout'])
        self.assertEqual(preview['metadata']['panel_fingerprint'], report['panel_fingerprint'])
        self.assertFalse(report['small_enemy_stage_gate']['passed'])
        self.assertFalse(report['gameplay_qualified']); self.assertFalse(report['automatic_deployment'])
        self.assertIsNone(data.hr.grad); self.assertIsNone(data.state.grad); self.assertIsNone(data.hr_mask.grad)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_relative_gate_recomputed_missing_or_nonfinite_denominators_never_pass(self):
        data=PairedHeldout(); model=TinySR(); codec=TinyFrozenCodec().eval(); panel=list(range(6))
        baseline,_=evaluate_small_sr(model,codec,data,panel,step=0)
        for domain in ('clean','codec'):
            baseline['small_enemy_cohorts'][domain]['tiny']['foreground_mae'] += .01
        report,_=evaluate_small_sr(model,codec,data,panel,step=100,baseline=baseline)
        self.assertTrue(report['small_enemy_stage_gate']['passed'])
        for kind in ('missing','nan','panel','count'):
            bad=copy.deepcopy(baseline)
            if kind=='missing':del bad['small_enemy_cohorts']['clean']['hp0']
            elif kind=='nan':bad['small_enemy_cohorts']['codec']['near']['mae_mean']=float('nan')
            elif kind=='panel':bad['panel_fingerprint']='different'
            else:bad['small_enemy_cohorts']['clean']['tiny']['frames']=3
            with self.subTest(kind=kind):
                current,_=evaluate_small_sr(model,codec,data,panel,step=100,baseline=bad)
                self.assertFalse(current['small_enemy_stage_gate']['passed'])

    def test_actual_edsr_shape_backward_keeps_every_compact_codec_tensor_frozen(self):
        model=EDSRUpscaler().train()
        codec=CompactCodec(compact_codec_config(VQGANConfig(image_size=32,n_embed=1488,data_init=False)))
        codec.eval().requires_grad_(False)
        self.assertEqual(sum(p.numel() for p in model.parameters()),1517571)
        self.assertEqual(sum(p.numel() for p in codec.parameters()),997539)
        before={k:v.clone() for k,v in codec.state_dict().items()}
        batch=PairedHeldout().batch(1,indices=[0])
        with torch.no_grad():inputs=codec(batch['lr'])['reconstruction']
        prediction=model(inputs)
        self.assertEqual(prediction.shape,(1,3,128,128)); self.assertTrue(torch.isfinite(prediction).all())
        (prediction-batch['hr'].detach()).abs().mean().backward()
        self.assertGreater(float(model.output.weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in codec.parameters()))
        self.assertFalse(codec.training)
        for key,value in before.items():self.assertTrue(torch.equal(value,codec.state_dict()[key]),key)


if __name__ == '__main__':unittest.main()
