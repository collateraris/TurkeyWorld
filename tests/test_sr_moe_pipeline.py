from copy import deepcopy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from test_atlas_enemy_sr import small
from test_enemy_decoder_validation import GuidedPairs
from turkeyworld.models import build_upscaler, load_sr_checkpoint
from turkeyworld.atlas_enemy_sr import AtlasEnemyEDSRUpscaler
from turkeyworld.renderer_native_phase import _same
from turkeyworld.renderer_student import _hash_tensors
from turkeyworld.sr_moe_phase import prepare_sr_moe_continuation, PROTOCOL
from turkeyworld.sr_moe_objectives import sr_moe_loss, LOSS_WEIGHTS, source_sr_moe_labels
from turkeyworld.sr_moe_validation import evaluate_sr_moe, preview_sr_moe, save_sr_moe_preview
from turkeyworld.sr_semantic_migration import restore_semantic_optimizer
from turkeyworld.sr_transformer_moe import enable_sr_transformer_moe, configure_sr_transformer_moe, sr_moe_metadata


class Pairs(GuidedPairs):
    identity={'kind':'refreshed_current_NPR_fixture','world_step':5947,'renderer_step':29609}
    def __init__(self):
        super().__init__()
        self.state[:,:2]=.3;self.state[:,3]=1;self.state[:,8]+=.3;self.state[:,9]=.3
    def batch(self,count,*,indices):
        b=super().batch(count,indices=indices)
        grid=torch.zeros(count,16,16,dtype=torch.long)
        grid[:,0]=grid[:,-1]=1;grid[:,:,0]=grid[:,:,-1]=1
        b['map']=grid
        return b


def networks():
    parent=small();model=deepcopy(parent);enable_sr_transformer_moe(model);configure_sr_transformer_moe(model)
    return model,parent.eval().requires_grad_(False)


class SRMoEPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_full_Atlas_weights_160_Adam_RNG_preserved_then_old_frozen_after_three_actual_SGD(self):
        parent=AtlasEnemyEDSRUpscaler();parent.requires_grad_(True)
        parameters=list(parent.parameters());names=list(dict(parent.named_parameters()));split=len(names)//2
        optimizer=torch.optim.AdamW([{'params':parameters[:split],'lr':1e-4},
                                     {'params':parameters[split:],'lr':5e-5}])
        sum(p.square().mean()+p.mean()*.001 for p in parameters).backward();optimizer.step()
        saved=dict(stage='sr',config=deepcopy(parent.config),model=deepcopy(parent.state_dict()),
            optimizer=deepcopy(optimizer.state_dict()),optimizer_parameter_names=[names[:split],names[split:]],
            training_steps=106105,atlas_updates=5175,sampler={'old_identity':'old_cache','draws':999},
            rng_numpy=np.random.default_rng(19).bit_generator.state,rng_torch=torch.get_rng_state(),
            rng_python=random.getstate(),rng_cuda=[])
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory);source=directory/'parent.pt';selected=directory/'export.pt';initial=directory/'initial.pt'
            torch.save(saved,source);torch.save({k:saved[k] for k in ('stage','config','model','training_steps')},selected)
            original=source.read_bytes();proof=prepare_sr_moe_continuation(source,initial,selected_checkpoint=selected)
            payload=torch.load(initial,weights_only=False)
            model,_=load_sr_checkpoint(initial,'cpu');configure_sr_transformer_moe(model)
            opt=restore_semantic_optimizer(model,payload)
            self.assertEqual(proof['old_Adam_states'],160);self.assertEqual(proof['fresh_Adam_state_entries'],0)
            self.assertEqual(payload['sr_moe_updates'],0);self.assertEqual(payload['training_steps'],106105)
            self.assertNotIn('sampler',payload);self.assertEqual(payload['parent_sampler'],saved['sampler'])
            self.assertTrue(_same(saved['optimizer']['state'],payload['optimizer']['state']))
            self.assertEqual(saved['optimizer']['param_groups'],payload['optimizer']['param_groups'][:-1])
            for key in ('rng_numpy','rng_torch','rng_python','rng_cuda'):self.assertTrue(_same(saved[key],payload[key]))
            batch=Pairs().batch(2,indices=[0,4]);before=deepcopy(model.state_dict());moments=deepcopy(opt.state_dict())
            for _ in range(3):
                configure_sr_transformer_moe(model);opt.zero_grad(set_to_none=True)
                output=model.forward_with_aux(batch['lr_npr'].detach(),batch['hypothesis'].detach())
                loss,_=sr_moe_loss(output,batch,label_renderer=model.render_label_atlas);loss.backward();opt.step()
            for name,value in before.items():
                if not name.startswith('sr_moe.'):self.assertTrue(torch.equal(value,model.state_dict()[name]),name)
            self.assertTrue(_same(moments['state'],{k:v for k,v in opt.state_dict()['state'].items() if k in moments['state']}))
            self.assertTrue(all(p.grad is None and not p.requires_grad for n,p in model.named_parameters() if not n.startswith('sr_moe.')))
            self.assertTrue(any(not torch.equal(v,model.state_dict()[n]) for n,v in before.items() if n.startswith('sr_moe.')))
            self.assertEqual(source.read_bytes(),original)

    def test_actual_loss_two_steps_all_five_blocks_and_RGB_geometry_recognition_heads_reachable(self):
        model,_=networks();batch=Pairs().batch(2,indices=[0,4]);opt=torch.optim.AdamW(model.sr_moe.parameters(),lr=.001)
        for key in ('hr_source','mask_hr','state'):
            batch[key]=batch[key].detach().requires_grad_()
        for _ in range(2):
            opt.zero_grad(set_to_none=True);output=model.forward_with_aux(batch['lr_npr'].detach(),batch['hypothesis'].detach())
            loss,metrics=sr_moe_loss(output,batch,label_renderer=model.render_label_atlas);loss.backward()
            self.assertTrue(torch.isfinite(loss));self.assertAlmostEqual(float(loss.detach()),sum(metrics[k+'_weighted'] for k in LOSS_WEIGHTS),places=5)
            self.assertGreater(float(model.sr_moe.detection_head.weight.grad.abs().sum()),0)
            self.assertGreater(float(model.sr_moe.geometry_head.weight.grad.abs().sum()),0)
            self.assertGreater(float(model.sr_moe.visibility_head.weight.grad.abs().sum()),0)
            opt.step()
        for expert in model.sr_moe.experts:
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in expert.parameters() if p.grad is not None),0)
        self.assertIsNone(batch['hr_source'].grad);self.assertIsNone(batch['mask_hr'].grad)
        self.assertIsNone(batch['state'].grad)

    def test_guided_confirmation_positive_wrong_hint_and_empty_negative_train_detection_row2(self):
        model,_=networks();batch=Pairs().batch(2,indices=[0,4])
        output=model.forward_with_aux(batch['lr_npr'].detach(),batch['hypothesis'].detach())
        loss,metrics=sr_moe_loss(output,batch,label_renderer=model.render_label_atlas);loss.backward()
        self.assertEqual(metrics['guided_confirmation_positive_frames'],1)
        self.assertEqual(metrics['guided_confirmation_negative_frames'],1)
        self.assertGreater(float(model.sr_moe.detection_head.weight.grad[2].abs().sum()),0.)
        wrong=Pairs().batch(2,indices=[0,1]);wrong['hypothesis'][1,9]+=.08
        output=model.forward_with_aux(wrong['lr_npr'].detach(),wrong['hypothesis'].detach())
        _,metrics=sr_moe_loss(output,wrong,label_renderer=model.render_label_atlas)
        self.assertGreaterEqual(metrics['guided_confirmation_negative_frames'],1)

    def test_five_role_partition_retains_one_HR_pixel_and_kill_counter_has_no_player_tint(self):
        batch=Pairs().batch(2,indices=[0,1]);batch['mask_hr']=torch.zeros(2,1,128,128,requires_grad=True)
        with torch.no_grad():batch['mask_hr'][:,:,3,3]=1
        labels=source_sr_moe_labels(batch,device='cpu')
        self.assertTrue(torch.allclose(labels['roles'].sum(1),torch.ones(2,32,32)))
        self.assertGreater(float(labels['roles'][:,0].sum()),0)
        self.assertTrue(torch.equal(labels['roles'][:,0],labels['roles'][:,1]))
        changed=deepcopy({k:v.detach() if torch.is_tensor(v) else v for k,v in batch.items()});changed['state'][:,18]=12
        self.assertTrue(torch.equal(labels['roles'],source_sr_moe_labels(changed,device='cpu')['roles']))
        self.assertFalse(labels['roles'].requires_grad)

    def test_heldout_current_NPR_only_validation_and_light_four_case_preview_preserve_every_state(self):
        model,parent=networks();data=Pairs();sampler=data.sampler_state();rng=torch.get_rng_state().clone()
        modes=[[m.training for m in net.modules()] for net in (model,parent)]
        hashes=[_hash_tensors(net.state_dict()) for net in (model,parent)]
        report,preview=evaluate_sr_moe(model,parent,data,list(range(8)),step=0,batch_size=3)
        self.assertEqual(report['kind'],'sr_moe_five_validation_v1')
        self.assertEqual(set(report['models']),{'current','frozen_atlas_parent'})
        objective=report['validation_objective'];self.assertEqual(objective['protocol'],PROTOCOL)
        self.assertEqual(set(objective['components']),set(LOSS_WEIGHTS))
        self.assertAlmostEqual(objective['loss'],sum(x['weighted'] for x in objective['components'].values()),places=5)
        self.assertIn('near_wall_mae',report['models']['current']['summary'])
        self.assertFalse(report['quality_goal']['passed']);self.assertEqual(report['quality_goal']['input_kind'],'actual_NPR_RGB32')
        self.assertEqual(sampler,data.sampler_state());self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(modes,[[m.training for m in net.modules()] for net in (model,parent)])
        self.assertEqual(hashes,[_hash_tensors(net.state_dict()) for net in (model,parent)])
        self.assertTrue(all(p.grad is None for net in (model,parent) for p in net.parameters()))
        with patch.object(model,'forward_with_aux',wraps=model.forward_with_aux) as calls:
            light=preview_sr_moe(model,parent,data,list(range(8)),step=15,baseline_report=report)
        self.assertEqual(sum(len(call.args[0]) for call in calls.call_args_list),4)
        self.assertTrue(light['metadata']['preview_only']);self.assertEqual(light['metadata']['reference_validation_step'],0)
        with tempfile.TemporaryDirectory() as directory:
            meta=save_sr_moe_preview(light,Path(directory)/'preview.png')
            self.assertEqual(meta['kind'],'sr_moe_five_preview_v1');self.assertEqual(meta['every'],15)
            with Image.open(meta['path']) as image:self.assertEqual(image.size,(1024,1120))


if __name__=='__main__':unittest.main()
