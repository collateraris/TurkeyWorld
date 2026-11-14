from copy import deepcopy
import io
import unittest
from unittest.mock import patch

import torch

from turkeyworld.models import build_upscaler
from turkeyworld.neural_enemy_restoration_sr import NeuralEnemyRestorationSR
from turkeyworld.background_detail_sr import (BackgroundDetailSR,BackgroundDetailResidual,
    initialize_background_detail,frozen_background_state,background_detail_metadata,TRAINABLE_PREFIX)
from turkeyworld.background_detail_objectives import background_detail_loss


def source_payload():
    old=NeuralEnemyRestorationSR(channels=8,blocks=1,semantic_channels=8,semantic_blocks=1,
        restoration_channels=8)
    return old,dict(config=deepcopy(old.config),model={n:v.detach().clone()for n,v in old.state_dict().items()},
        training_steps=5308,standalone_RGB_only=True,atlas_in_new_model=False)


class BackgroundDetailModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_zero_initialized_image_is_exact_preferred_base_without_assets_or_old_replacement(self):
        old,parent=source_payload();before={n:v.clone()for n,v in parent['model'].items()};low=torch.rand(2,3,32,32)
        with patch('builtins.open',side_effect=AssertionError('No sprite/atlas/source assets allowed')):
            model,report=initialize_background_detail(parent,detail_channels=8,detail_blocks=1)
            for training in(True,False):
                model.train(training);old.train(training)
                expected=old.forward_with_aux(low)['background'].clamp(0,1)
                out=model.forward_with_aux(low,hypothesis=torch.full((2,21),float('nan')))
                self.assertTrue(torch.equal(out['image'],expected));self.assertTrue(torch.equal(out['background'],expected))
                self.assertEqual(int(torch.count_nonzero(out['residual'])),0)
                self.assertFalse(out['background'].requires_grad)
                self.assertNotIn('alpha',out);self.assertNotIn('enemy_rgb',out)
        self.assertTrue(report['frozen_background_bitexact']);self.assertFalse(report['old_Adam_reused'])
        self.assertTrue(all(torch.equal(v,parent['model'][n])for n,v in before.items()))
        self.assertTrue(all(n.startswith('enemy_restoration.')for n in report['parent_omitted_tensor_names']))
        self.assertFalse(any('atlas'in n.lower()or n.startswith('enemy_restoration.')for n in model.state_dict()))

    def test_only_new_residual_trains_and_copied_background_stays_bitexact(self):
        _,parent=source_payload();model,_=initialize_background_detail(parent,detail_channels=8,detail_blocks=1)
        low=torch.rand(2,3,32,32);frozen={n:v.clone()for n,v in frozen_background_state(model).items()}
        optimizer=torch.optim.Adam([p for p in model.parameters()if p.requires_grad],lr=.001)
        before=model(low).detach()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True);out=model.forward_with_aux(low)
            target=(out['background']+.1).clamp(0,1)
            loss,_=background_detail_loss(out,dict(hr_source=target,mask_hr=torch.zeros(2,1,128,128)))
            loss.backward();optimizer.step()
        self.assertGreater(float(model.detail_residual.output.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.detail_residual.stem[0].weight.grad.abs().sum()),0)
        self.assertFalse(torch.equal(model(low),before))
        self.assertTrue(all(p.grad is None for n,p in model.named_parameters()if not n.startswith(TRAINABLE_PREFIX)))
        self.assertTrue(all(torch.equal(model.state_dict()[n],v)for n,v in frozen.items()))
        self.assertTrue(all(not child.training for name,child in model.named_children()if name!='detail_residual'))
        self.assertTrue(all(n.startswith(TRAINABLE_PREFIX)for n,p in model.named_parameters()if p.requires_grad))
        self.assertEqual(len(optimizer.state),len(list(model.detail_residual.parameters())))

    def test_bound_does_not_force_highpass_and_initial_display_clipping_is_exact(self):
        model=BackgroundDetailSR(channels=8,blocks=1,semantic_channels=8,semantic_blocks=1,detail_channels=8,detail_blocks=1)
        low=torch.rand(1,3,32,32)
        with patch.object(model.sr_moe,'forward',return_value=dict(base_rgb_residual=torch.ones(1,3,128,128))):
            out=model.forward_with_aux(low)
            self.assertTrue(bool((out['background_raw']>1).any()))
            self.assertTrue(torch.equal(out['image'],out['background_raw'].clamp(0,1)))
        with torch.no_grad():model.detail_residual.output.bias.fill_(100.)
        out=model.forward_with_aux(low)
        self.assertLessEqual(float(out['residual'].detach().abs().max()),.25)
        self.assertGreater(float(out['residual'].detach().mean()),.24)
        self.assertTrue(bool(((out['image']>=0)&(out['image']<=1)).all()))

    def test_initializer_preserves_CPU_rng_and_source_while_seeding_only_new_head(self):
        _,parent=source_payload();rng=torch.get_rng_state().clone()
        a,_=initialize_background_detail(parent,seed=178,detail_channels=8,detail_blocks=1)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        b,_=initialize_background_detail(parent,seed=178,detail_channels=8,detail_blocks=1)
        c,_=initialize_background_detail(parent,seed=179,detail_channels=8,detail_blocks=1)
        self.assertTrue(all(torch.equal(x,y)for x,y in zip(a.state_dict().values(),b.state_dict().values())))
        self.assertFalse(torch.equal(a.detail_residual.stem[0].weight,c.detail_residual.stem[0].weight))

    def test_standalone_factory_reload_has_only_RGB_input_and_no_file_dependency(self):
        _,parent=source_payload();model,_=initialize_background_detail(parent,detail_channels=8,detail_blocks=1)
        model.eval();low=torch.rand(1,3,32,32);expected=model(low)
        stream=io.BytesIO();torch.save(dict(config=model.config,model=model.state_dict()),stream);stream.seek(0)
        payload=torch.load(stream,weights_only=False)
        with patch('builtins.open',side_effect=AssertionError('Standalone detail inference requires no files')):
            restored=build_upscaler(payload['config']);restored.load_state_dict(payload['model'],strict=True);restored.eval()
            self.assertTrue(torch.equal(restored(low,hypothesis=torch.randn(1,21)),expected))
        self.assertFalse(model.config.get('sr_moe_enabled',False))
        metadata=background_detail_metadata(model)
        self.assertFalse(metadata['alpha_used']);self.assertFalse(metadata['atlas_asset_required'])

    def test_small_branch_count_and_invalid_sources_are_rejected(self):
        branch=BackgroundDetailResidual();count=sum(p.numel()for p in branch.parameters())
        self.assertEqual(count,526131);self.assertEqual(len(list(branch.parameters())),42)
        for mutate in(lambda p:p['model'].pop('semantic_input.weight'),
                      lambda p:p['config'].update(architecture='atlas_enemy_edsr'),
                      lambda p:p['model'].update(atlas_pixels=torch.zeros(1))):
            _,parent=source_payload();mutate(parent)
            with self.assertRaises(ValueError):initialize_background_detail(parent,detail_channels=8,detail_blocks=1)
        model=BackgroundDetailSR(channels=8,blocks=1,semantic_channels=8,semantic_blocks=1,detail_channels=8,detail_blocks=1)
        for value in(torch.zeros(1,3,64,64),torch.full((1,3,32,32),float('nan')),torch.full((1,3,32,32),2.)):
            with self.assertRaises(ValueError):model(value)


if __name__=='__main__':unittest.main()
