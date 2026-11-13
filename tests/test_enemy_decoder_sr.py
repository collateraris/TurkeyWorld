import copy
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from turkeyworld.models import build_upscaler
from turkeyworld.enemy_decoder_sr import project_enemy_hypothesis
from turkeyworld.enemy_decoder_objectives import enemy_decoder_loss,balanced_binary_loss,LOSS_WEIGHTS
from turkeyworld.enemy_decoder_migration import prepare_enemy_decoder_continuation,restore_enemy_decoder_optimizer


def hypothesis(count=4):
    values=torch.zeros(count,21);values[:,0:2]=.25;values[:,3]=1;values[:,4]=1
    values[:,8]=.25+3/16;values[:,9]=.25;values[:,12:14]=torch.tensor([.6,1.]);values[:,-1]=1
    return values


def model_pair():
    parent=build_upscaler({'architecture':'semantic_edsr','channels':4,'blocks':1,
        'semantic_channels':4,'semantic_blocks':1})
    child=build_upscaler({**parent.config,'architecture':'enemy_decoder_edsr','enemy_decoder_channels':8})
    state=child.state_dict();state.update(parent.state_dict());child.load_state_dict(state)
    return parent,child


def labels():
    low=torch.full((4,3,32,32),.25);low[[0,1,3],0,14:20,14:18]=.9
    mask=torch.zeros(4,1,128,128);mask[[0,1,3],0,56:80,56:72]=1
    high=torch.full((4,3,128,128),.25);high[:,0:1]+=mask*.65
    low_mask=torch.nn.functional.max_pool2d(mask,4)
    boxes=torch.zeros(4,4);boxes[[0,1,3]]=torch.tensor([56/128,56/128,72/128,80/128])
    h=hypothesis();h[1,9]+=.9/16;h[3,-1]=0
    return low,h,dict(hr_source=high,hr_background=torch.full_like(high,.25),mask_hr=mask,
        mask_lr=low_mask,bbox_hr=boxes,visible_labels=mask.flatten(1).any(-1)[:,None])


class EnemyDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_projection_heading_front_behind_invalid_and_strict_hypothesis(self):
        h=hypothesis(2);h[1,2:4]=torch.tensor([1.,0.]);h[1,8:10]=torch.tensor([.25,.25+3/16])
        _,box,valid=project_enemy_hypothesis(h,batch=2,device='cpu')
        self.assertTrue(valid.all());torch.testing.assert_close(box[0],box[1],rtol=0,atol=1e-6)
        self.assertAlmostEqual(float((box[0,0]+box[0,2])/2),.5)
        h[0,8]=.1
        self.assertFalse(project_enemy_hypothesis(h,batch=2,device='cpu')[2][0])
        for bad in (torch.zeros(2,20),torch.full((2,21),float('nan')),h.clone()):
            if bad.shape==(2,21)and torch.isfinite(bad).all():bad[:,-1]=.2
            with self.assertRaises(ValueError):project_enemy_hypothesis(bad,batch=2,device='cpu')

    def test_zero_gain_parent_function_exact_and_unknown_geometry_rgb_fallback(self):
        parent,child=model_pair();low=torch.rand(2,3,32,32);h=hypothesis(2)
        self.assertEqual(list(dict(child.named_parameters()))[:len(list(parent.parameters()))],list(dict(parent.named_parameters())))
        for training in (True,False):
            parent.train(training);child.train(training)
            torch.testing.assert_close(child(low,h),parent(low),rtol=0,atol=0)
        with torch.no_grad():child.enemy_decoder_gain.weight.fill_(1.)
        for unknown in (None,torch.zeros_like(h)):
            torch.testing.assert_close(child(low,unknown),parent(low),rtol=0,atol=0)
        h[:,8]=.1
        torch.testing.assert_close(child(low,h),parent(low),rtol=0,atol=0)
        h[:,8]=1e30
        output=child.forward_with_aux(low,h)
        self.assertTrue(torch.isfinite(output['image']).all());self.assertFalse(output['enemy_applicable'].any())

    def test_confirmation_reads_rgb_roi_not_state_embedding_and_inference_can_reject(self):
        parent,child=model_pair();low=torch.full((2,3,32,32),.25);h=hypothesis(2)
        h[1,12]=.1;h[1,14:18]=torch.tensor([.9,.8,4.,.7]);h[1,19]=2
        child.eval()
        output=child.forward_with_aux(low,h)
        torch.testing.assert_close(output['enemy_confirmation_logits'][0],output['enemy_confirmation_logits'][1],rtol=0,atol=0)
        with torch.no_grad():
            child.enemy_decoder_gain.weight.fill_(1.)
            child.enemy_decoder_confirmation[-1].weight.zero_();child.enemy_decoder_confirmation[-1].bias.fill_(-10)
        rejected=child.forward_with_aux(low,h)
        self.assertEqual(float(rejected['enemy_alpha_hr'].detach().sum()),0.)
        parent.eval();torch.testing.assert_close(rejected['image'],parent(low),rtol=0,atol=0)
        child.train();soft=child.forward_with_aux(low,h)
        self.assertGreater(float(soft['enemy_alpha_hr'].detach().sum()),0.)
        soft['enemy_confirmation_logits'].retain_grad();soft['image'].sum().backward()
        self.assertGreater(float(soft['enemy_confirmation_logits'].grad.abs().sum()),0.)

    def test_wrong_predicted_box_is_confirmation_negative_not_global_presence_negative(self):
        _,model=model_pair();low,h,batch=labels();h.requires_grad_()
        for value in batch.values():
            if value.is_floating_point():value.requires_grad_()
        output=model.forward_with_aux(low,h)
        loss,metrics=enemy_decoder_loss(output,batch);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertEqual(metrics['applicable_hypotheses'],3)
        self.assertEqual(metrics['matching_hypotheses'],1)
        self.assertEqual(metrics['confirmation_positive_count'],1)
        self.assertEqual(metrics['confirmation_negative_count'],2)
        self.assertEqual(metrics['global_presence_positive_count'],3)
        self.assertIsNone(h.grad)
        self.assertTrue(all(value.grad is None for value in batch.values()))
        for prefix in ('enemy_decoder_visual.','enemy_decoder_condition.','enemy_decoder_seed.',
                       'enemy_decoder_rgba.','enemy_decoder_visible.','enemy_decoder_box.',
                       'enemy_decoder_confirmation.','enemy_decoder_heatmap.'):
            grad=sum(float(p.grad.abs().sum())for name,p in model.named_parameters()if name.startswith(prefix)and p.grad is not None)
            self.assertGreater(grad,0,prefix)
        self.assertAlmostEqual(metrics['train_loss'],sum(metrics[name+'_weighted']for name in LOSS_WEIGHTS),places=5)

    def test_balanced_presence_and_confirmation_keep_class_mass_under_rare_negatives(self):
        logits=torch.tensor([1.,-1.,-1.,-1.],requires_grad=True);labels=torch.tensor([1.,0.,0.,0.])
        first=balanced_binary_loss(logits,labels)
        second=balanced_binary_loss(torch.cat((logits[:1].repeat(5),logits[1:])),torch.tensor([1.]*5+[0.]*3))
        torch.testing.assert_close(first,second,rtol=0,atol=1e-7)
        first.backward();self.assertTrue(torch.isfinite(logits.grad).all())

    def test_fp32_predicted_hypothesis_with_half_rgb_and_half_weights(self):
        parent,model=model_pair();parent.half().eval();model.half().eval()
        image=torch.full((1,3,32,32),.3,dtype=torch.float16)
        with torch.inference_mode():
            output=model.forward_with_aux(image,hypothesis(1))
            torch.testing.assert_close(output['image'],parent(image),rtol=0,atol=0)
        self.assertEqual(output['image'].dtype,torch.float16)
        self.assertEqual(output['enemy_bbox'].dtype,torch.float32)
        self.assertTrue(torch.isfinite(output['image']).all())

    def test_full_semantic_parent_adam_rng_and_history_append_exact(self):
        parent,_=model_pair();names=list(dict(parent.named_parameters()));parameters=dict(parent.named_parameters())
        optimizer=torch.optim.AdamW([{'params':[parameters[n]for n in names[:2]],'lr':5e-5},
                                    {'params':[parameters[n]for n in names[2:]],'lr':4e-5}],weight_decay=.01)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True);sum(p.square().sum()for p in parent.parameters()).backward();optimizer.step()
        optimizer.state[parameters[names[0]]]['step'].fill_(1)
        saved=dict(stage='sr',config=parent.config,model=copy.deepcopy(parent.state_dict()),optimizer=copy.deepcopy(optimizer.state_dict()),
            optimizer_parameter_names=[names[:2],names[2:]],semantic_migration={'kind':'parent_proof'},
            training_steps=46400,parent_training_steps=40100,semantic_updates=6300,completed_updates=6300,
            sampler={'draws':50400},protocol={'immutable':'old'},best={'step':6200},
            rng_torch=torch.get_rng_state(),rng_numpy=np.random.default_rng(173).bit_generator.state,
            rng_python=random.getstate(),rng_cuda=[])
        with tempfile.TemporaryDirectory() as temporary:
            source=Path(temporary)/'source.pt';target=Path(temporary)/'initial.pt';torch.save(saved,source)
            before=source.read_bytes();rng=torch.get_rng_state().clone()
            report=prepare_enemy_decoder_continuation(source,target)
            self.assertEqual(source.read_bytes(),before);torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
            new=torch.load(target,weights_only=False)
            for name,value in saved['model'].items():torch.testing.assert_close(value,new['model'][name],rtol=0,atol=0)
            self.assertEqual(new['optimizer']['param_groups'][:2],saved['optimizer']['param_groups'])
            for index,state in saved['optimizer']['state'].items():
                for key,value in state.items():torch.testing.assert_close(value,new['optimizer']['state'][index][key],rtol=0,atol=0)
            self.assertEqual(new['rng_numpy'],saved['rng_numpy']);self.assertEqual(new['rng_python'],saved['rng_python'])
            torch.testing.assert_close(new['rng_torch'],saved['rng_torch'],rtol=0,atol=0)
            self.assertEqual(new['semantic_migration'],saved['semantic_migration'])
            self.assertEqual(new['parent_sampler'],saved['sampler']);self.assertNotIn('sampler',new)
            self.assertEqual(new['semantic_updates'],0);self.assertEqual(new['decoder_updates'],0)
            self.assertEqual(new['parent_semantic_updates'],6300);self.assertEqual(new['parent_training_steps'],46400)
            model=build_upscaler(new['config']);model.load_state_dict(new['model'])
            opt=restore_enemy_decoder_optimizer(model,new)
            self.assertEqual(len(opt.state),len(optimizer.state));self.assertEqual(len(opt.param_groups),3)
            self.assertTrue(all(p not in opt.state for p in opt.param_groups[-1]['params']))
            self.assertEqual(report['added_parameter_count'],290014)


if __name__=='__main__':unittest.main()
