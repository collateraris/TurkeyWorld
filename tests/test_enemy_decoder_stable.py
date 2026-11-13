"""Frozen RGB ancestry survives real SGD while independent new heads learn."""
import copy
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

import test_enemy_decoder_sr as support
from turkeyworld.models import build_upscaler
from turkeyworld.enemy_decoder_sr import EnemyDecoderEDSRUpscaler
from turkeyworld.enemy_decoder_objectives import enemy_decoder_loss
from turkeyworld.enemy_decoder_migration import prepare_enemy_decoder_continuation,restore_enemy_decoder_optimizer
from turkeyworld.gradient_safety import clip_finite_grad_norm_
from turkeyworld.enemy_animation import enemy_animation_index


class StableEnemyDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_multiple_SGD_preserves_all_parent_weights_and_Adam_and_trains_new_recognition(self):
        parent,_=support.model_pair();low,h,batch=support.labels()
        names=list(dict(parent.named_parameters()));params=dict(parent.named_parameters())
        opt=torch.optim.AdamW([{'params':[params[n]for n in names[:2]],'lr':5e-5},
                              {'params':[params[n]for n in names[2:]],'lr':4e-5}],weight_decay=.01)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            out=parent.forward_with_aux(low)
            loss=(out['image']-.7).square().mean()+.01*out['presence_logits'].square().mean()+.01*out['mask_logits_lr'].square().mean()
            loss.backward();opt.step()
        old_model=copy.deepcopy(parent.state_dict());old_optimizer=copy.deepcopy(opt.state_dict())
        saved=dict(stage='sr',config=parent.config,model=old_model,optimizer=old_optimizer,
            optimizer_parameter_names=[names[:2],names[2:]],semantic_migration={'kind':'parent_fixture'},
            training_steps=40102,semantic_updates=2,completed_updates=2,
            sampler={'draws':16},rng_numpy=np.random.default_rng(173).bit_generator.state,
            rng_torch=torch.get_rng_state(),rng_python=random.getstate(),rng_cuda=[])
        with tempfile.TemporaryDirectory()as temporary:
            source=Path(temporary)/'parent.pt';target=Path(temporary)/'stable.pt';torch.save(saved,source)
            report=prepare_enemy_decoder_continuation(source,target,stable=True)
            loaded=torch.load(target,weights_only=False)
            model=build_upscaler(loaded['config']);model.load_state_dict(loaded['model']);optimizer=restore_enemy_decoder_optimizer(model,loaded)
            self.assertEqual(report['added_parameter_count'],
                sum(p.numel()for name,p in model.named_parameters()if name.startswith('enemy_decoder_')))
            self.assertTrue(model.config['enemy_decoder_stable'])
            self.assertTrue(all(not p.requires_grad for n,p in model.named_parameters()if not n.startswith('enemy_decoder_')))
            self.assertTrue(all(p.requires_grad for n,p in model.named_parameters()if n.startswith('enemy_decoder_')))
            before_new={n:p.detach().clone()for n,p in model.named_parameters()if n.startswith('enemy_decoder_')}
            before_parent_rgb=parent(low).detach()
            for _ in range(3):
                model.train();optimizer.zero_grad(set_to_none=True)
                self.assertTrue(all(not p.requires_grad for n,p in model.named_parameters()if not n.startswith('enemy_decoder_')))
                output=model.forward_with_aux(low,h)
                torch.testing.assert_close(output['mask_logits_lr'],torch.nn.functional.avg_pool2d(output['enemy_visible_logits_hr'],4),rtol=0,atol=0)
                loss,_=enemy_decoder_loss(output,batch);loss.backward()
                self.assertTrue(all(p.grad is None for n,p in model.named_parameters()if not n.startswith('enemy_decoder_')))
                self.assertGreater(float(model.enemy_decoder_global_presence.weight.grad.abs().sum()),0)
                self.assertGreater(float(model.enemy_decoder_visible[-1].weight.grad.abs().sum()),0)
                clip_finite_grad_norm_(model.named_parameters());optimizer.step()
                inherited=super(EnemyDecoderEDSRUpscaler,model).forward_with_aux(low)['image']
                torch.testing.assert_close(inherited,before_parent_rgb,rtol=0,atol=0)
            after=model.state_dict()
            for name,value in old_model.items():torch.testing.assert_close(value,after[name],rtol=0,atol=0)
            state=optimizer.state_dict()
            self.assertEqual(state['param_groups'][:2],old_optimizer['param_groups'])
            for index,row in old_optimizer['state'].items():
                for key,value in row.items():torch.testing.assert_close(value,state['state'][index][key],rtol=0,atol=0)
            self.assertTrue(any(not torch.equal(value,after[name])for name,value in before_new.items()))
            self.assertFalse(torch.equal(before_new['enemy_decoder_global_presence.weight'],after['enemy_decoder_global_presence.weight']))
            self.assertFalse(torch.equal(before_new['enemy_decoder_rgba.1.weight'],after['enemy_decoder_rgba.1.weight']))

    def test_default_profile_remains_loadable_without_added_head_or_frozen_parent(self):
        _,old=support.model_pair();payload=copy.deepcopy(old.state_dict())
        self.assertNotIn('enemy_decoder_stable',old.config)
        self.assertTrue(all(p.requires_grad for p in old.parameters()))
        loaded=build_upscaler(old.config);loaded.load_state_dict(payload,strict=True)
        self.assertFalse(hasattr(loaded,'enemy_decoder_global_presence'))

    def test_predicted_animation_onehot_changes_RGBA_without_changing_geometry_or_RGB_confirmation(self):
        model=build_upscaler({'architecture':'enemy_decoder_edsr','channels':4,'blocks':1,
            'semantic_channels':4,'semantic_blocks':1,'enemy_decoder_channels':8,'enemy_decoder_stable':True}).eval()
        h=support.hypothesis(4)
        h[1,15]=.7
        h[2,12:14]=0;h[2,16]=.1
        h[3,12:14]=0;h[3,16]=.9
        with torch.no_grad():
            model.enemy_decoder_condition[0].weight[:,:24].zero_()
            model.enemy_decoder_condition[0].weight[:,24:].zero_()
            for index in range(7):model.enemy_decoder_condition[0].weight[index,24+index]=index+1.
        image=torch.full((4,3,32,32),.3)
        output=model.forward_with_aux(image,h)
        self.assertEqual(output['enemy_animation_index'].tolist(),[0,4,5,6])
        torch.testing.assert_close(output['enemy_animation_index'],enemy_animation_index(h[:,:20]))
        self.assertEqual(model.enemy_decoder_condition[0].in_features,31)
        self.assertEqual(output['enemy_animation_logits'].shape,(4,7))
        for index in range(1,4):
            torch.testing.assert_close(output['hypothesis_bbox'][0],output['hypothesis_bbox'][index],rtol=0,atol=0)
            torch.testing.assert_close(output['enemy_confirmation_logits'][0],output['enemy_confirmation_logits'][index],rtol=0,atol=0)
            torch.testing.assert_close(output['enemy_animation_logits'][0],output['enemy_animation_logits'][index],rtol=0,atol=0)
            self.assertFalse(torch.equal(output['enemy_rgba'][0],output['enemy_rgba'][index]))

    def test_animation_conflict_stops_canonical_target_but_trains_RGB_animation_classification(self):
        _,legacy=support.model_pair()
        model=build_upscaler({**legacy.config,'enemy_decoder_stable':True})
        low,h,batch=support.labels()
        batch['animation_label']=torch.full((4,),4,dtype=torch.long)
        output=model.forward_with_aux(low,h)
        loss,metrics=enemy_decoder_loss(output,batch);loss.backward()
        self.assertEqual(metrics['matching_hypotheses'],1)
        self.assertEqual(metrics['canonical_matching_hypotheses'],0)
        self.assertEqual(metrics['animation_phase_conflict_count'],1)
        self.assertEqual(metrics['canonical_rgb'],0.)
        self.assertEqual(metrics['canonical_alpha'],0.)
        self.assertGreater(metrics['animation_ce'],0.)
        self.assertGreater(float(model.enemy_decoder_animation.weight.grad.abs().sum()),0)
        del batch['animation_label']
        _,absent=enemy_decoder_loss(model.forward_with_aux(low,h),batch)
        self.assertEqual(absent['animation_ce'],0.)
        self.assertEqual(absent['canonical_matching_hypotheses'],1)


if __name__=='__main__':unittest.main()
