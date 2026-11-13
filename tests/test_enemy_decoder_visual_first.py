"""RGB-first routes, trainability and full named-Adam append invariants."""
from copy import deepcopy
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from turkeyworld.enemy_decoder_sr import EnemyDecoderEDSRUpscaler
from turkeyworld.enemy_decoder_migration import (prepare_visual_first_continuation,
    restore_enemy_decoder_optimizer)
from turkeyworld.models import build_upscaler


def small_model(visual_first=True):
    return build_upscaler(dict(architecture='enemy_decoder_edsr', channels=4, blocks=1,
        semantic_channels=4,semantic_blocks=1,enemy_decoder_channels=8,
        enemy_decoder_stable=True,enemy_decoder_visual_first=visual_first))


def hypothesis(count=2):
    state=torch.zeros(count,21)
    state[:,0:2]=.25;state[:,3]=1;state[:,4]=1
    state[:,8]=.25+3/16;state[:,9]=.25
    state[:,12:14]=torch.tensor([.6,1.]);state[:,-1]=1
    return state


def force_visual(model, *, presence=10., confirmation=10.):
    with torch.no_grad():
        model.enemy_decoder_global_presence.weight.zero_()
        model.enemy_decoder_global_presence.bias.fill_(presence)
        model.enemy_decoder_visual_confirmation[-1].weight.zero_()
        model.enemy_decoder_visual_confirmation[-1].bias.fill_(confirmation)
        model.enemy_decoder_confirmation[-1].weight.zero_()
        model.enemy_decoder_confirmation[-1].bias.fill_(10.)
        model.enemy_decoder_box[-1].weight.zero_();model.enemy_decoder_box[-1].bias.zero_()
        model.enemy_decoder_size[-1].weight.zero_();model.enemy_decoder_size[-1].bias.fill_(-1.)
        model.enemy_decoder_visible[-1].weight.zero_();model.enemy_decoder_visible[-1].bias.fill_(5.)
        model.enemy_decoder_rgba[-1].weight.zero_();model.enemy_decoder_rgba[-1].bias.fill_(5.)
        model.enemy_decoder_gain.weight.fill_(1.)
        model.enemy_decoder_visual_gain.weight.fill_(1.)
        model.enemy_decoder_animation.weight.zero_()
        model.enemy_decoder_animation.bias.zero_();model.enemy_decoder_animation.bias[4]=10.


class VisualFirstEnemyDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_confident_RGB_draws_without_hint_and_is_exact_under_corrupt_or_behind_hint(self):
        torch.manual_seed(173)
        model=small_model();force_visual(model)
        low=torch.full((2,3,32,32),.25)
        correct=hypothesis();wrong=correct.clone();wrong[:,9]+=.5
        behind=correct.clone();behind[:,8]=.1
        for training in (False, True):
            model.train(training)
            expected=model.forward_with_aux(low,None)
            parent=super(EnemyDecoderEDSRUpscaler,model).forward_with_aux(low)['image']
            self.assertTrue(expected['visual_route'].all())
            self.assertFalse(expected['guided_route'].any())
            self.assertGreater(float(expected['visual_composite_alpha_hr'].detach().sum()),0.)
            self.assertFalse(torch.equal(expected['image'],parent))
            for hint in (correct,wrong,behind):
                output=model.forward_with_aux(low,hint)
                for key in ('image','visual_bbox','visual_rgba','visual_alpha_hr',
                            'visual_confirmation_logits','visual_animation_logits'):
                    torch.testing.assert_close(output[key],expected[key],rtol=0,atol=0)
                self.assertEqual(output['visual_animation_source'],'RGB_only')
                self.assertEqual(output['visual_animation_index'].tolist(),[4,4])

    def test_absent_visual_evidence_cannot_be_overridden_by_a_valid_confident_hint(self):
        model=small_model().eval();force_visual(model,presence=-10.,confirmation=-10.)
        low=torch.full((2,3,32,32),.25)
        parent=super(EnemyDecoderEDSRUpscaler,model).forward_with_aux(low)['image']
        for hint in (None,hypothesis()):
            output=model.forward_with_aux(low,hint)
            torch.testing.assert_close(output['image'],parent,rtol=0,atol=0)
            self.assertFalse(output['visual_route'].any())
            self.assertFalse(output['guided_route'].any())
        # Uncertain visual localization may use a valid RGB-confirmed hint.
        force_visual(model,presence=10.,confirmation=0.)
        output=model.forward_with_aux(low,hypothesis())
        self.assertFalse(output['visual_route'].any())
        self.assertTrue(output['guided_route'].all())
        model.enemy_decoder_confirmation[-1].bias.data.fill_(-10.)
        self.assertFalse(model.forward_with_aux(low,hypothesis())['guided_route'].any())

    def test_first_below_threshold_training_step_has_nonzero_visual_gain_gradient(self):
        model=small_model().train();force_visual(model,presence=0.,confirmation=-2.)
        with torch.no_grad(): model.enemy_decoder_visual_gain.weight.zero_()
        image=torch.full((2,3,32,32),.25)
        hint=hypothesis().requires_grad_()
        output=model.forward_with_aux(image,hint)
        self.assertFalse(output['visual_route'].any())
        self.assertTrue((output['visual_route_weight']>0).all())
        loss=(output['image']-1.).square().mean()
        loss.backward()
        gradient=model.enemy_decoder_visual_gain.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertLess(float(gradient),0.)
        self.assertIsNone(hint.grad)
        self.assertTrue(all(p.grad is None for n,p in model.named_parameters()
                            if not n.startswith('enemy_decoder_')))
        optimizer=torch.optim.SGD([model.enemy_decoder_visual_gain.weight],lr=.01)
        optimizer.step()
        self.assertGreater(float(model.enemy_decoder_visual_gain.weight.detach()),0.)

    def test_visual_adapter_and_recognition_gradients_open_while_parent_stays_frozen(self):
        model=small_model().train();low=torch.rand(2,3,32,32)
        optimizer=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=1e-3)
        parent={n:p.detach().clone()for n,p in model.named_parameters()if not n.startswith('enemy_decoder_')}
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            output=model.forward_with_aux(low,None)
            objective=(output['visual_rgba']-.8).square().mean()
            objective+=F.binary_cross_entropy_with_logits(output['visual_confirmation_logits'],torch.ones(2,1))
            objective+=.01*output['visual_animation_logits'].square().mean()
            objective.backward();optimizer.step()
        for prefix in ('enemy_decoder_visual_condition.0.', 'enemy_decoder_visual_condition.2.',
                       'enemy_decoder_visual_confirmation.', 'enemy_decoder_seed.', 'enemy_decoder_rgba.'):
            gradients=[p.grad.abs().sum()for n,p in model.named_parameters()if n.startswith(prefix)and p.grad is not None]
            self.assertGreater(float(torch.stack(gradients).sum()),0.,prefix)
        for name,value in parent.items():
            torch.testing.assert_close(dict(model.named_parameters())[name],value,rtol=0,atol=0)

    def test_old_default_strict_load_and_explicit_close_bbox_range(self):
        old=small_model(False).eval();copy=build_upscaler(old.config).eval()
        copy.load_state_dict(old.state_dict(),strict=True)
        self.assertNotIn('enemy_decoder_visual_first',old.config)
        self.assertFalse(any('visual_condition.'in name for name,_ in old.named_parameters()))
        low=torch.rand(1,3,32,32)
        torch.testing.assert_close(old(low,hypothesis(1)),copy(low,hypothesis(1)),rtol=0,atol=0)
        model=small_model().eval()
        with torch.no_grad():
            model.enemy_decoder_size[-1].weight.zero_();model.enemy_decoder_size[-1].bias.fill_(10.)
        output=model.forward_with_aux(low,None)
        self.assertGreater(float((output['visual_proposal_bbox'][:,2:]-output['visual_proposal_bbox'][:,:2]).detach().min()),.9)
        for bad in (True,float('nan'),1.,.5):
            with self.assertRaises(ValueError):
                build_upscaler({**old.config,'enemy_decoder_visual_first':True,
                                'enemy_decoder_visual_threshold':bad})

    def test_full_named_mixed_age_Adam_RNG_sampler_append_and_old_classifier_copy(self):
        old=small_model(False)
        names=list(dict(old.named_parameters()));params=dict(old.named_parameters())
        groups=[names[:2],names[2:-47],names[-47:]]
        optimizer=torch.optim.AdamW([{'params':[params[n]for n in group],'lr':rate}
            for group,rate in zip(groups,(5e-5,4e-5,3e-5))],weight_decay=.01)
        for p in old.parameters(): p.grad=torch.full_like(p,.01)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        for i,p in enumerate(old.parameters()): optimizer.state[p]['step'].fill_(1+i%3)
        saved=dict(stage='sr',config=old.config,model=deepcopy(old.state_dict()),
            optimizer=deepcopy(optimizer.state_dict()),optimizer_parameter_names=groups,
            semantic_migration={'kind':'semantic_fixture'},enemy_decoder_migration={'kind':'stable_fixture'},
            training_steps=51452,parent_training_steps=47600,semantic_updates=3852,
            decoder_updates=3852,completed_updates=3852,
            sampler={'draws':30816,'private_rng':np.random.default_rng(173).bit_generator.state},
            protocol={'kind':'old_stable'},best={'step':3700},
            rng_torch=torch.get_rng_state(),rng_numpy=np.random.default_rng(174).bit_generator.state,
            rng_python=random.getstate(),rng_cuda=[])
        with tempfile.TemporaryDirectory()as directory:
            source=Path(directory)/'source.pt';target=Path(directory)/'visual.pt';torch.save(saved,source)
            source_bytes=source.read_bytes();global_rng=torch.get_rng_state().clone()
            report=prepare_visual_first_continuation(source,target)
            new=torch.load(target,weights_only=False)
            self.assertEqual(source.read_bytes(),source_bytes)
            torch.testing.assert_close(torch.get_rng_state(),global_rng,rtol=0,atol=0)
            self.assertEqual(new['optimizer_parameter_names'][:-1],groups)
            self.assertEqual(new['optimizer']['param_groups'][:-1],saved['optimizer']['param_groups'])
            for index,row in saved['optimizer']['state'].items():
                for key,value in row.items(): torch.testing.assert_close(value,new['optimizer']['state'][index][key],rtol=0,atol=0)
            for name,value in saved['model'].items(): torch.testing.assert_close(value,new['model'][name],rtol=0,atol=0)
            self.assertEqual(new['sampler'],saved['sampler'])
            self.assertEqual(new['rng_numpy'],saved['rng_numpy']);self.assertEqual(new['rng_python'],saved['rng_python'])
            torch.testing.assert_close(new['rng_torch'],saved['rng_torch'],rtol=0,atol=0)
            self.assertEqual(new['rng_cuda'],saved['rng_cuda'])
            self.assertEqual((new['visual_first_updates'],new['decoder_updates'],new['semantic_updates']),(0,0,0))
            self.assertEqual(new['parent_decoder_updates'],3852)
            self.assertEqual(new['parent_training_steps'],51452)
            self.assertEqual(new['training_steps'],51452)
            self.assertEqual(new['parent_protocol'],saved['protocol'])
            self.assertNotIn('protocol',new)
            model=build_upscaler(new['config']);model.load_state_dict(new['model'],strict=True)
            opt=restore_enemy_decoder_optimizer(model,new)
            self.assertTrue(all(p not in opt.state for p in opt.param_groups[-1]['params']))
            self.assertEqual(len(opt.state),len(optimizer.state))
            for name,value in model.enemy_decoder_confirmation.state_dict().items():
                torch.testing.assert_close(value,model.enemy_decoder_visual_confirmation.state_dict()[name],rtol=0,atol=0)
            self.assertEqual(float(model.enemy_decoder_visual_gain.weight.detach()),0.)
            self.assertEqual(float(model.enemy_decoder_visual_condition[2].weight.detach().abs().sum()),0.)
            self.assertTrue(report['old_Adam_and_RNG_preserved'])
            self.assertTrue(report['sampler_preserved'])
            self.assertEqual(len(report['added_parameter_names']),9)
            self.assertTrue(all(not p.requires_grad for n,p in model.named_parameters()if not n.startswith('enemy_decoder_')))


if __name__=='__main__': unittest.main()
