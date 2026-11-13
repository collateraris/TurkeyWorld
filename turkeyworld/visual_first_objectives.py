"""Train RGB object restoration independently of optional RSSM proposals.

Source boxes, masks, colors and animation phases are detached labels only.
The guided legacy objective remains explicit; missing or incorrect hints never
disable the independent visual localization, shape, color or animation losses.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .enemy_decoder_objectives import (enemy_decoder_loss,LOSS_WEIGHTS as GUIDED_WEIGHTS,
    _mean,_mask_terms,balanced_binary_loss)
from .enemy_decoder_sr import box_iou,sample_box

PROTOCOL = 'visual_first_enemy_decoder_sr_v2'
VISION_WEIGHTS = {'vision_bbox_l1':.1,'vision_bbox_iou':.1,
    'vision_mask_bce':.1,'vision_mask_dice':.05,'vision_canonical_rgb':.1,
    'vision_canonical_alpha':.05,'vision_confirmation_bce':.1,'vision_animation_ce':.05}
LOSS_WEIGHTS = {**GUIDED_WEIGHTS,**VISION_WEIGHTS}


def _proposal_overlap(proposal,mask):
    proposal = proposal.detach().float()
    axis = (torch.arange(128,device=mask.device,dtype=torch.float32)+.5)/128
    yy,xx = torch.meshgrid(axis,axis,indexing='ij')
    inside = ((xx[None]>=proposal[:,0,None,None])&(xx[None]<=proposal[:,2,None,None])
        &(yy[None]>=proposal[:,1,None,None])&(yy[None]<=proposal[:,3,None,None]))
    return (mask[:,0]*inside).flatten(1).sum(-1)/mask.flatten(1).sum(-1).clamp_min(1)


def _class_balanced_mask(logits,mask,visible):
    """Normalize pixels within images and actual visible/empty image classes."""
    positive = _mask_terms(logits,mask,visible)
    negative = _mask_terms(logits,mask,~visible)
    present = visible.any().float()+(~visible).any().float()
    return tuple((a+b)/present.clamp_min(1) for a,b in zip(positive,negative))


def _local_confirmation(logits,positive,visible,strata):
    """Equal positive/negative mass; preserve hidden/other empty hard negatives."""
    positive = positive.detach().bool(); visible = visible.detach().bool()
    hidden = torch.tensor([name=='occluded' for name in strata],device=visible.device,dtype=torch.bool)
    groups = (visible&~positive,(~visible)&hidden,(~visible)&~hidden)
    errors = F.binary_cross_entropy_with_logits(logits.float().reshape(-1),positive.float(),reduction='none')
    positive_loss = _mean(errors,positive)
    present_groups = sum(group.any().float() for group in groups)
    negative_loss = sum(_mean(errors,group) for group in groups)/present_groups.clamp_min(1)
    present_classes = positive.any().float()+(~positive).any().float()
    return (positive_loss+negative_loss)/present_classes.clamp_min(1)


def visual_first_loss(output,batch):
    """All RGB supervision remains available with None, static or wrong hints."""
    guided,metrics = enemy_decoder_loss(output,batch)
    image = output['image'];count = len(image);device = image.device
    mask = batch['mask_hr'].detach().to(device=device,dtype=torch.float32)
    boxes = batch['bbox_hr'].detach().to(device=device,dtype=torch.float32)
    visible = batch['visible_labels'].detach().to(device=device).bool().reshape(-1)
    if (mask.shape!=(count,1,128,128) or boxes.shape!=(count,4) or visible.shape!=(count,)
            or not torch.equal(visible,mask.bool().flatten(1).any(1))
            or not bool(torch.isfinite(boxes).all()) or bool(((boxes<0)|(boxes>1)).any())
            or bool((visible&((boxes[:,2:]<=boxes[:,:2]).any(1))).any())):
        raise ValueError('Visual Source labels require aligned actual-HR visibility and finite tight xyxy boxes')
    labels = batch['animation_label'].detach().to(device=device).long().reshape(-1)
    if labels.shape!=(count,) or bool(((labels<0)|(labels>6)).any()):
        raise ValueError('Visual Source animation labels must be classes0..6')
    proposal = output['visual_proposal_bbox']
    predicted = output['visual_bbox'].float()
    rgba = output['visual_rgba'].float()
    logits = output['visual_visible_logits_hr'].float()
    confirmation = output['visual_confirmation_logits'].float()
    animation = output['visual_animation_logits'].float()
    if (proposal.shape!=(count,4) or predicted.shape!=(count,4) or rgba.ndim!=4
            or rgba.shape[:2]!=(count,4) or logits.shape!=(count,1,128,128)
            or confirmation.shape!=(count,1) or animation.shape!=(count,7)
            or not all(bool(torch.isfinite(value).all()) for value in (proposal,predicted,rgba,logits,confirmation,animation))
            or bool(((rgba<0)|(rgba>1)).any())):
        raise ValueError('Visual-first model outputs must include finite independent bbox/RGBA/mask/confirmation/animation heads')
    center = ((predicted[:,:2]+predicted[:,2:]-boxes[:,:2]-boxes[:,2:])/2).abs().mean(-1)
    bbox = _mean((predicted-boxes).abs().mean(-1)+.5*center,visible)
    iou = _mean(1-box_iou(predicted,boxes),visible)
    mask_bce,mask_dice = _class_balanced_mask(logits,mask,visible)
    source_crop = sample_box(batch['hr_source'].detach().float(),boxes,rgba.shape[-1])
    alpha_crop = sample_box(mask,boxes,rgba.shape[-1]).clamp(0,1)
    colors = ((rgba[:,:3]-source_crop).abs()*alpha_crop).flatten(1).sum(-1)
    colors = colors/(3*alpha_crop.flatten(1).sum(-1)).clamp_min(1)
    alpha = F.binary_cross_entropy(rgba[:,3:4].clamp(1e-6,1-1e-6),alpha_crop,reduction='none').flatten(1).mean(-1)
    overlap = _proposal_overlap(proposal,mask)
    confirmed = visible&(overlap>=.1)
    strata = batch.get('stratum',['unknown']*count)
    if len(strata)!=count:
        raise ValueError('Source hard-negative strata must align with visual examples')
    terms = {'vision_bbox_l1':bbox,'vision_bbox_iou':iou,'vision_mask_bce':mask_bce,
        'vision_mask_dice':mask_dice,'vision_canonical_rgb':_mean(colors,visible),
        'vision_canonical_alpha':_mean(alpha,visible),
        'vision_confirmation_bce':_local_confirmation(confirmation,confirmed,visible,strata),
        'vision_animation_ce':_mean(F.cross_entropy(animation,labels,reduction='none'),visible)}
    total = guided+sum(VISION_WEIGHTS[name]*value for name,value in terms.items())
    if not bool(torch.isfinite(total)):
        raise FloatingPointError('Nonfinite visual-first enemy decoder objective')
    for name,value in terms.items():
        metrics[name] = float(value.detach());metrics[name+'_weighted'] = VISION_WEIGHTS[name]*metrics[name]
    hints = batch.get('hypothesis')
    available = (hints[:,-1].detach()>0.5 if hints is not None else torch.zeros(count,device=device,dtype=torch.bool))
    visual_route = output.get('visual_route');guided_route = output.get('guided_route')
    metrics.update(train_loss=float(total.detach()),loss_protocol=PROTOCOL,objective_protocol=PROTOCOL,
        vision_localization_frames=int(visible.sum()),vision_canonical_frames=int(visible.sum()),
        vision_animation_frames=int(visible.sum()),vision_empty_frames=int((~visible).sum()),
        vision_missing_hint_visible_frames=int((visible&~available).sum()),
        vision_confirmation_positive_count=int(confirmed.sum()),
        vision_confirmation_negative_count=int((~confirmed).sum()),
        vision_visible_bad_proposal_count=int((visible&~confirmed).sum()),
        vision_hidden_empty_frames=int(((~visible)&torch.tensor([bool(name=='occluded')for name in strata],
            device=device,dtype=torch.bool)).sum()),
        visual_route_count=int((visual_route.detach()>.5).sum()) if visual_route is not None else 0,
        guided_route_count=int((guided_route.detach()>.5).sum()) if guided_route is not None else 0,
        vision_losses_require_RSSM=False,vision_canonical_phase_gate=False,
        visual_confirmation_uses_original_visual_proposal=True,
        source_labels_detached=True,inference_rgb_only=False,inference_primary_rgb=True,
        rssm_hypothesis_optional=True,inference_inputs=['NPR_RGB32','optional_predicted_RSSM_state20_and_validbit'])
    return total,metrics
