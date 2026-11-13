"""Full-sprite geometry, separate visibility and non-opaque native alpha labels."""
from copy import deepcopy
import math
import unittest

import torch

from turkeyworld.atlas_enemy_objectives import (atlas_enemy_loss,source_full_sprite_geometry,
                                               LOSS_WEIGHTS,_soft_dice)
from turkeyworld.enemy_animation import enemy_animation_index


def states(count=2):
    state=torch.zeros(count,20)
    state[:,:2]=.5;state[:,3]=1;state[:,4]=1
    state[:,8]=.5+3/16;state[:,9]=.5;state[:,12:14]=torch.tensor([.6,1.])
    return state


class LabelAtlas:
    """Fixed synthetic translucent asset, not a Source image or NN forward."""
    def __init__(self):self.calls=[]
    def __call__(self,boxes,phase_index,dead_index=None):
        self.calls.append((boxes.detach().clone(),phase_index.detach().clone(),torch.is_grad_enabled()))
        axis=torch.arange(128,device=boxes.device,dtype=torch.float32)/128
        y,x=torch.meshgrid(axis,axis,indexing='ij')
        support=((x[None]>=boxes[:,0,None,None])&(x[None]<boxes[:,2,None,None])
                 &(y[None]>=boxes[:,1,None,None])&(y[None]<boxes[:,3,None,None]))[:,None]
        rgba=boxes.new_full((len(boxes),4,128,128),.7)
        rgba[:,3:4]=support.float()*.25
        return rgba


def fixture(count=2,*,partial=False,visible=True):
    state=states(count);geometry=source_full_sprite_geometry(state);renderer=LabelAtlas()
    intrinsic=renderer(geometry['full_bbox'],geometry['phase'])
    mask=(intrinsic[:,3:4]>0).float()
    if partial:mask[:,:,:,64:]=0
    if not visible:mask.zero_()
    alpha=intrinsic[:,3:4]*mask
    background=torch.full((count,3,128,128),.2)
    target=background+alpha*(intrinsic[:,:3]-background)
    box=geometry['full_bbox'].clone().requires_grad_()
    logits=(mask*12-6).requires_grad_()
    phase=torch.full((count,7),-10.)
    phase[torch.arange(count),geometry['phase']]=10.;phase.requires_grad_()
    output=dict(image=target.clone().requires_grad_(),atlas_full_bbox=box,visual_full_bbox=box,
        atlas_visible_logits_hr=logits,atlas_effective_alpha_hr=alpha.clone().requires_grad_(),
        atlas_phase_logits=phase,atlas_player_dead_logits=torch.zeros(count,1,requires_grad=True),
        presence_logits=torch.zeros(count,1,requires_grad=True),
        visual_confirmation_logits=torch.zeros(count,1,requires_grad=True))
    batch=dict(state=state,hr_source=target,hr_background=background,mask_hr=mask,
        bbox_hr=torch.zeros(count,4),animation_label=enemy_animation_index(state))
    renderer.calls.clear()
    return output,batch,renderer,geometry


class AtlasEnemyObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_full_bounds_match_source_integer_camera_formula_and_death_does_not_flatten_geometry(self):
        state=states(4)
        state[1,8:10]=torch.tensor([.5+1.3/16,.5+1./16])
        state[2,12:14]=0;state[2,16]=.8
        state[3,8]=.5-.1/16
        result=source_full_sprite_geometry(state)
        projection=128/(2*math.tan(math.radians(68)/2))
        for i in range(3):
            dx=float((state[i,8]-state[i,0])*16);dy=float((state[i,9]-state[i,1])*16)
            height=int(min(640,projection*.93/dx));cx=int(64+projection*dy/dx);bottom=int(64+projection*.5/dx)
            expected=torch.tensor([cx-height//2,bottom-height,cx-height//2+height,bottom])/128
            torch.testing.assert_close(result['full_bbox'][i],expected,rtol=0,atol=0)
        self.assertGreater(float(result['full_bbox'][1,2]),1.)
        torch.testing.assert_close(result['full_bbox'][0],result['full_bbox'][2],rtol=0,atol=0)
        self.assertFalse(result['geometry_valid'][3])
        self.assertEqual(result['phase'].tolist(),[0,0,6,0])

    def test_partial_occlusion_never_stretches_tight_visible_crop_and_preserves_soft_intrinsic_alpha(self):
        output,batch,renderer,geometry=fixture(partial=True)
        loss,metrics=atlas_enemy_loss(output,batch,label_renderer=renderer)
        self.assertEqual(metrics['bbox_full_l1'],0.)
        self.assertEqual(metrics['bbox_full_iou'],0.)
        self.assertEqual(metrics['sprite_height_log_l1'],0.)
        self.assertEqual(metrics['alpha_l1'],0.)
        self.assertEqual(metrics['alpha_dice'],0.)
        self.assertEqual(metrics['full_geometry_frames'],2)
        callback_box,phase,grad_enabled=renderer.calls[-1]
        torch.testing.assert_close(callback_box,geometry['full_bbox'],rtol=0,atol=0)
        self.assertFalse(grad_enabled)
        changed=dict(batch,bbox_hr=torch.ones_like(batch['bbox_hr'])*.99)
        other,_=atlas_enemy_loss(output,changed,label_renderer=renderer)
        torch.testing.assert_close(loss,other,rtol=0,atol=0)
        opaque=dict(output,atlas_effective_alpha_hr=batch['mask_hr'].clone().requires_grad_())
        _,opaque_metrics=atlas_enemy_loss(opaque,batch,label_renderer=renderer)
        self.assertGreater(opaque_metrics['alpha_l1'],0.)
        self.assertGreater(opaque_metrics['alpha_dice'],0.)

    def test_soft_alpha_dice_has_zero_error_and_gradient_at_identical_translucent_labels(self):
        target=torch.tensor([[[[0.,.01,.25,1.]]]])
        predicted=target.clone().requires_grad_();error=_soft_dice(predicted,target)
        self.assertEqual(float(error.detach()),0.)
        error.sum().backward();torch.testing.assert_close(predicted.grad,torch.zeros_like(predicted),rtol=0,atol=0)

    def test_hard_ST_hidden_mask_has_nonzero_finite_gradient_and_no_source_or_label_gradients(self):
        output,batch,renderer,geometry=fixture(partial=True)
        for value in batch.values():
            if torch.is_tensor(value)and value.is_floating_point():value.requires_grad_()
        logits=torch.full_like(batch['mask_hr'],-2.,requires_grad=True)
        probability=logits.sigmoid();hard=(logits>=0).float()
        gate=hard.detach()+(probability-probability.detach())
        intrinsic=renderer(geometry['full_bbox'],geometry['phase'])
        alpha=intrinsic[:,3:4]*gate
        output.update(atlas_visible_logits_hr=logits,atlas_effective_alpha_hr=alpha,
            image=batch['hr_background'].detach()+alpha*(intrinsic[:,:3]-batch['hr_background'].detach()))
        output['visual_full_bbox']=output['atlas_full_bbox']=torch.nn.Parameter(geometry['full_bbox']+.02)
        loss,metrics=atlas_enemy_loss(output,batch,label_renderer=renderer);loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()),0.)
        self.assertLess(float(logits.grad[batch['mask_hr'].bool()].mean()),0.)
        self.assertGreater(float(output['atlas_full_bbox'].grad.abs().sum()),0.)
        self.assertGreater(float(output['atlas_phase_logits'].grad.abs().sum()),0.)
        self.assertGreater(float(output['atlas_player_dead_logits'].grad.abs().sum()),0.)
        self.assertTrue(all(value.grad is None for value in batch.values()if torch.is_tensor(value)))
        self.assertAlmostEqual(metrics['train_loss'],sum(metrics[k+'_weighted']for k in LOSS_WEIGHTS),places=5)
        self.assertFalse(metrics['GAN_active']);self.assertFalse(metrics['learned_RGBA_canonical_loss_active'])
        self.assertNotIn('canonical_rgb',metrics)

    def test_fully_occluded_source_never_supervises_hidden_position_or_phase_but_keeps_visibility_negative(self):
        output,batch,renderer,_=fixture(visible=False)
        output['visual_full_bbox']=output['atlas_full_bbox']=torch.nn.Parameter(torch.tensor([[.1,.1,.9,.9]]).repeat(2,1))
        loss,metrics=atlas_enemy_loss(output,batch,label_renderer=renderer);loss.backward()
        self.assertEqual(metrics['full_geometry_frames'],0)
        self.assertEqual(metrics['phase_ce'],0.)
        self.assertEqual(metrics['bbox_full_l1'],0.)
        torch.testing.assert_close(output['atlas_full_bbox'].grad,torch.zeros_like(output['atlas_full_bbox']),rtol=0,atol=0)
        self.assertGreater(float(output['atlas_visible_logits_hr'].grad.abs().sum()),0.)

    def test_HP0_is_a_separate_source_label_and_invalid_softalpha_or_phase_labels_fail(self):
        output,batch,renderer,_=fixture();batch['state'][0,4]=0
        loss,metrics=atlas_enemy_loss(output,batch,label_renderer=renderer);loss.backward()
        self.assertEqual(metrics['source_player_dead_frames'],1)
        self.assertLess(float(output['atlas_player_dead_logits'].grad[0]),0.)
        self.assertGreater(float(output['atlas_player_dead_logits'].grad[1]),0.)
        with self.assertRaisesRegex(ValueError,'pure fixed-asset'):atlas_enemy_loss(output,batch)
        wrong=dict(batch,animation_label=torch.tensor([7,0]))
        with self.assertRaisesRegex(ValueError,'animation labels'):atlas_enemy_loss(output,wrong,label_renderer=renderer)
        with self.assertRaisesRegex(ValueError,'invalid RGB/alpha'):
            atlas_enemy_loss(output,batch,label_renderer=lambda *a,**k:torch.full((2,4,128,128),float('nan')))

    def test_actual_fixed_atlas_model_backward_trains_heads_but_not_paint_asset_or_source_labels(self):
        from turkeyworld.atlas_enemy_sr import AtlasEnemyEDSRUpscaler,UNUSED_PAINT_PREFIXES
        model=AtlasEnemyEDSRUpscaler(channels=4,blocks=1,semantic_channels=4,
                                    semantic_blocks=1,enemy_decoder_channels=8).train()
        source=states();source[0,4]=0;source[0,15]=.7
        geometry=source_full_sprite_geometry(source)
        rgba=model.render_label_atlas(geometry['full_bbox'],geometry['phase'],dead_index=geometry['player_dead'])
        mask=(rgba[:,3:4]>0).float();mask[:,:,:,64:]=0
        background=torch.full((2,3,128,128),.2)
        target=background+rgba[:,3:4]*mask*(rgba[:,:3]-background)
        batch=dict(state=source.requires_grad_(),mask_hr=mask.requires_grad_(),
            hr_source=target.requires_grad_(),hr_background=background.requires_grad_(),
            animation_label=geometry['phase'])
        # A deliberately different synthetic predicted hint is not the Source
        # pose used by the loss. Geometry still has to be learned from RGB.
        predicted=states();predicted[:,8]+=.5/16
        hint=torch.cat((predicted,torch.ones(2,1)),-1).requires_grad_()
        output=model.forward_with_aux(torch.rand(2,3,32,32),hint)
        loss,metrics=atlas_enemy_loss(output,batch,label_renderer=model.render_label_atlas)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for prefix in ('enemy_decoder_visual.','enemy_decoder_size.', 'enemy_decoder_box.',
                       'enemy_decoder_visible.','enemy_decoder_animation.','atlas_player_dead_head.'):
            gradients=[p.grad.abs().sum()for n,p in model.named_parameters()if n.startswith(prefix)and p.grad is not None]
            self.assertTrue(gradients,prefix)
            self.assertGreater(float(torch.stack(gradients).sum()),0.,prefix)
        self.assertTrue(all(p.grad is None for n,p in model.named_parameters()
                            if n.startswith(UNUSED_PAINT_PREFIXES)))
        self.assertFalse(model.atlas_frames_rgba.requires_grad)
        self.assertIsNone(hint.grad)
        self.assertTrue(all(value.grad is None for value in batch.values()if torch.is_tensor(value)))
        self.assertEqual(metrics['full_geometry_frames'],2)


if __name__=='__main__':unittest.main()
