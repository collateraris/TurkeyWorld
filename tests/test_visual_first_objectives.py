import json
import unittest

import torch
from torch.nn import functional as F

from turkeyworld.visual_first_objectives import visual_first_loss,LOSS_WEIGHTS,PROTOCOL
from turkeyworld.models import build_upscaler


def fixture(*,empty=False):
    count=4
    mask=torch.zeros(count,1,128,128)
    if not empty:mask[:3,0,64:80,64:80]=1
    boxes=torch.zeros(count,4)
    if not empty:boxes[:3]=torch.tensor([.5,.5,.625,.625])
    target=torch.full((count,3,128,128),.1);target[:,:1]+=mask*.7
    batch={'hr_source':target.requires_grad_(),'hr_background':torch.full_like(target,.1).requires_grad_(),
        'mask_hr':mask.requires_grad_(),'mask_lr':F.max_pool2d(mask.detach(),4).requires_grad_(),
        'bbox_hr':boxes.requires_grad_(),'visible_labels':mask.detach().flatten(1).any(1)[:,None],
        'animation_label':torch.tensor([0,4,5,6]),'hypothesis':torch.zeros(count,21,requires_grad=True),
        'stratum':['1_8','9_16','corpse','occluded']}
    def leaf(shape,value=0.):return torch.full(shape,value,requires_grad=True)
    output={'image':leaf((count,3,128,128),.3),'mask_logits_lr':leaf((count,1,32,32)),
        'presence_logits':leaf((count,1)),'enemy_applicable':torch.zeros(count,1,dtype=torch.bool),
        'hypothesis_bbox':torch.zeros(count,4),'enemy_bbox':leaf((count,4),.3),
        'enemy_visual_bbox':leaf((count,4),.3),'enemy_visible_logits_hr':leaf((count,1,128,128)),
        'enemy_rgba':leaf((count,4,32,32),.25),'enemy_confirmation_logits':leaf((count,1)),
        'enemy_animation_index':torch.zeros(count,dtype=torch.long),'enemy_animation_logits':leaf((count,7)),
        'visual_bbox':torch.tensor([[.3,.3,.8,.8]]*count,requires_grad=True),
        'visual_rgba':leaf((count,4,32,32),.25),'visual_confirmation_logits':leaf((count,1)),
        'visual_visible_logits_hr':leaf((count,1,128,128)),'visual_animation_logits':leaf((count,7)),
        'visual_route':torch.tensor([[True],[True],[True],[False]]),'guided_route':torch.zeros(count,1,dtype=torch.bool)}
    output['visual_proposal_bbox']=torch.tensor([[.48,.48,.65,.65],[.05,.05,.15,.15],
        [.48,.48,.65,.65],[.48,.48,.65,.65]],requires_grad=True)
    return output,batch


class VisualFirstObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_no_hint_or_static_visible_rows_train_all_visual_heads_without_Source_gradients(self):
        output,batch=fixture()
        loss,metrics=visual_first_loss(output,batch);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics['vision_canonical_frames'],3)
        self.assertEqual(metrics['vision_localization_frames'],3)
        self.assertEqual(metrics['vision_animation_frames'],3)
        self.assertEqual(metrics['vision_missing_hint_visible_frames'],3)
        self.assertEqual(metrics['matching_hypotheses'],0)
        self.assertGreater(metrics['vision_canonical_rgb'],0)
        for name in ('visual_bbox','visual_rgba','visual_confirmation_logits',
                     'visual_visible_logits_hr','visual_animation_logits'):
            self.assertIsNotNone(output[name].grad,name)
            self.assertTrue(torch.isfinite(output[name].grad).all(),name)
            self.assertGreater(float(output[name].grad.abs().sum()),0,name)
        self.assertTrue((output['visual_rgba'].grad[:3].abs().flatten(1).sum(1)>0).all())
        self.assertIsNone(output['visual_proposal_bbox'].grad)
        for name in ('hr_source','hr_background','mask_hr','mask_lr','bbox_hr','hypothesis'):
            self.assertIsNone(batch[name].grad,name)
        self.assertEqual(metrics['loss_protocol'],PROTOCOL)
        self.assertAlmostEqual(metrics['train_loss'],sum(metrics[k+'_weighted']for k in LOSS_WEIGHTS),places=5)
        json.dumps(metrics,allow_nan=False)

    def test_wrong_hint_and_conflicting_hint_phase_do_not_disable_RGB_color_shape_or_global_presence(self):
        output,batch=fixture();_,first=visual_first_loss(output,batch)
        changed=dict(output)
        changed['enemy_applicable']=torch.ones(4,1,dtype=torch.bool)
        changed['hypothesis_bbox']=torch.tensor([[.01,.01,.1,.1]]*4)
        changed['enemy_animation_index']=torch.tensor([6,6,6,6])
        wrong_batch={**batch,'hypothesis':torch.cat((torch.zeros(4,20),torch.ones(4,1)),1)}
        _,second=visual_first_loss(changed,wrong_batch)
        for key in LOSS_WEIGHTS:
            if key.startswith('vision_'):
                self.assertAlmostEqual(first[key],second[key],places=7)
        self.assertEqual(second['global_presence_positive_count'],3)
        self.assertEqual(second['vision_canonical_frames'],3)
        self.assertFalse(second['vision_losses_require_RSSM'])
        self.assertFalse(second['vision_canonical_phase_gate'])

    def test_local_confirmation_uses_original_visual_proposal_and_preserves_visible_global_positive(self):
        output,batch=fixture()
        # Correcting the refined box cannot turn an unrelated original proposal
        # into positive local-confirmation supervision.
        output['visual_bbox']=batch['bbox_hr'].detach().clone().requires_grad_()
        loss,metrics=visual_first_loss(output,batch);loss.backward()
        self.assertEqual(metrics['vision_confirmation_positive_count'],2)
        self.assertEqual(metrics['vision_confirmation_negative_count'],2)
        self.assertEqual(metrics['vision_visible_bad_proposal_count'],1)
        self.assertEqual(metrics['global_presence_positive_count'],3)
        self.assertEqual(metrics['vision_hidden_empty_frames'],1)
        self.assertLess(float(output['visual_confirmation_logits'].grad[0]),0)
        self.assertGreater(float(output['visual_confirmation_logits'].grad[1]),0)
        self.assertLess(float(output['visual_confirmation_logits'].grad[2]),0)
        self.assertGreater(float(output['visual_confirmation_logits'].grad[3]),0)

    def test_all_empty_batch_is_finite_and_has_no_fake_canonical_or_animation_positives(self):
        output,batch=fixture(empty=True)
        loss,metrics=visual_first_loss(output,batch);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for key in ('vision_bbox_l1','vision_bbox_iou','vision_canonical_rgb',
                    'vision_canonical_alpha','vision_animation_ce'):
            self.assertEqual(metrics[key],0.)
        self.assertEqual(metrics['vision_empty_frames'],4)
        self.assertEqual(metrics['vision_confirmation_positive_count'],0)
        self.assertEqual(metrics['vision_confirmation_negative_count'],4)
        self.assertTrue(torch.isfinite(output['visual_confirmation_logits'].grad).all())
        self.assertTrue((output['visual_confirmation_logits'].grad>0).all())
        json.dumps(metrics,allow_nan=False)

    def test_real_visual_first_model_learns_shared_decoder_with_no_RSSM_argument(self):
        model=build_upscaler({'architecture':'enemy_decoder_edsr','channels':4,'blocks':1,
            'semantic_channels':4,'semantic_blocks':1,'enemy_decoder_channels':4,
            'enemy_decoder_stable':True,'enemy_decoder_visual_first':True}).train()
        _,batch=fixture()
        low=torch.full((4,3,32,32),.2);low[:3,0,16:20,16:20]=.8
        output=model.forward_with_aux(low,None)
        loss,metrics=visual_first_loss(output,batch);loss.backward()
        self.assertEqual(metrics['matching_hypotheses'],0)
        self.assertEqual(metrics['vision_canonical_frames'],3)
        self.assertEqual(metrics['vision_missing_hint_visible_frames'],3)
        for prefix in ('enemy_decoder_visual_condition.','enemy_decoder_seed.','enemy_decoder_rgba.'):
            gradient=sum(float(p.grad.abs().sum())for name,p in model.named_parameters()
                if name.startswith(prefix)and p.grad is not None)
            self.assertGreater(gradient,0,prefix)
        self.assertTrue(all(p.grad is None for name,p in model.named_parameters()
            if not name.startswith('enemy_decoder_')))
        self.assertIsNone(batch['hypothesis'].grad)


if __name__=='__main__':unittest.main()
