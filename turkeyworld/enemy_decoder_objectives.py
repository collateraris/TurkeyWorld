"""Source-label-only losses for a predicted-RSSM-proposal learned RGBA decoder."""
import torch
from torch.nn import functional as F

from .enemy_decoder_sr import box_iou,sample_box
from .semantic_sr_objectives import semantic_sr_loss,LOSS_WEIGHTS as BASE_WEIGHTS

PROTOCOL='rssm_guided_enemy_decoder_sr_v1'
LOSS_WEIGHTS={**BASE_WEIGHTS,'bbox_l1':.1,'bbox_iou':.1,
    'visible_mask_bce':.1,'visible_mask_dice':.05,'canonical_rgb':.1,
    'canonical_alpha':.05,'confirmation_bce':.1,'animation_ce':.05}


def _mean(value,selected):
    selected=selected.detach().reshape(-1).float()
    return (value.reshape(-1)*selected).sum()/selected.sum().clamp_min(1)


def balanced_binary_loss(logits,labels,selected=None):
    labels=labels.detach().float().reshape(-1)
    values=logits.float().reshape(-1)
    chosen=torch.ones_like(labels,dtype=torch.bool)if selected is None else selected.detach().bool().reshape(-1)
    positive=chosen&(labels>=.5);negative=chosen&~(labels>=.5)
    error=F.binary_cross_entropy_with_logits(values,labels,reduction='none')
    return (_mean(error,positive)+_mean(error,negative))/((positive.any()).float()+(negative.any()).float()).clamp_min(1)


def _mask_terms(logits,mask,selected):
    mask=mask.detach().float();error=F.binary_cross_entropy_with_logits(logits.float(),mask,reduction='none').flatten(1)
    target=mask.flatten(1);pos=target.sum(-1);neg=(1-target).sum(-1)
    bce=((error*target).sum(-1)/pos.clamp_min(1)+(error*(1-target)).sum(-1)/neg.clamp_min(1))
    bce=bce/((pos>0).float()+(neg>0).float()).clamp_min(1)
    probability=logits.float().sigmoid().flatten(1)
    dice=1-(2*(probability*target).sum(-1)+1e-6)/(probability.sum(-1)+pos+1e-6)
    return _mean(bce,selected),_mean(dice,selected)


def enemy_decoder_loss(output,batch):
    """True HR image/mask targets never enter a network's forward arguments.

    Proposal match/confirmation labels are computed from the ORIGINAL proposal,
    not a refined predicted box. An unrelated hypothesis therefore cannot turn
    a real visible NPC into a global negative or game its own supervision mask.
    """
    base,metrics=semantic_sr_loss(output,batch)
    low_mask=batch['mask_lr'].detach().float()
    old_presence_label=(low_mask.flatten(1).sum(-1)>0).float()[:,None]
    global_presence=batch['visible_labels'].detach().float().reshape(-1,1)
    ordinary_presence=F.binary_cross_entropy_with_logits(output['presence_logits'].float(),old_presence_label)
    presence=balanced_binary_loss(output['presence_logits'],global_presence)
    base=base+BASE_WEIGHTS['presence_bce']*(presence-ordinary_presence)
    metrics['presence_bce']=float(presence.detach())
    metrics['presence_bce_weighted']=BASE_WEIGHTS['presence_bce']*metrics['presence_bce']
    boxes=batch['bbox_hr'].detach().to(output['image'].device,dtype=torch.float32)
    visible=batch['visible_labels'].detach().bool().reshape(-1)
    mask=batch['mask_hr'].detach().float()
    if boxes.shape!=(len(mask),4)or visible.shape!=(len(mask),):
        raise ValueError('Source visible bbox labels must be[B,4] and[B,1]')
    if not bool(torch.isfinite(boxes).all())or bool(((boxes<0)|(boxes>1)).any()):
        raise ValueError('Source HR boxes must be finite normalized coordinates')
    applicable=output['enemy_applicable'].detach().bool().reshape(-1)
    proposal=output['hypothesis_bbox'].detach().float()
    # Pixel centers inside the original predicted proposal; no target geometry
    # is used in a model query, warp, correction or inference gate.
    axis=(torch.arange(128,device=mask.device,dtype=torch.float32)+.5)/128
    yy,xx=torch.meshgrid(axis,axis,indexing='ij')
    inside=((xx[None]>=proposal[:,0,None,None])&(xx[None]<=proposal[:,2,None,None])
           &(yy[None]>=proposal[:,1,None,None])&(yy[None]<=proposal[:,3,None,None]))
    source_area=mask.flatten(1).sum(-1)
    overlap=(mask[:,0]*inside).flatten(1).sum(-1)/source_area.clamp_min(1)
    confirmation_label=visible&(overlap>=.1)
    matched=applicable&visible&(overlap>=.5)
    canonical_matched=matched
    animation_ce=output['image'].sum()*0
    phase_conflicts=0
    if 'animation_label'in batch and 'enemy_animation_logits'in output:
        animation_label=batch['animation_label'].detach().long().reshape(-1)
        if animation_label.shape!=(len(mask),)or bool(((animation_label<0)|(animation_label>6)).any()):
            raise ValueError('Source animation labels must be class0..6 per frame')
        predicted_animation=output['enemy_animation_index'].detach().long().reshape(-1)
        phase_agrees=predicted_animation==animation_label
        canonical_matched=matched&phase_agrees
        phase_conflicts=int((matched&~phase_agrees).sum())
        animation_ce=_mean(F.cross_entropy(output['enemy_animation_logits'].float(),
            animation_label,reduction='none'),matched)
    visible_cases=applicable&visible
    predicted_box=output['enemy_bbox'].float()
    visual_box=output['enemy_visual_bbox'].float()
    def box_regression(predicted):
        # Corner L1/IoU alone has no center gradient while a large prediction
        # fully contains the target. Explicit center L1 keeps tiny localization
        # learnable in that common initial configuration.
        center_error=((predicted[:,:2]+predicted[:,2:]-boxes[:,:2]-boxes[:,2:])/2).abs().mean(-1)
        return (predicted-boxes).abs().mean(-1)+.5*center_error
    bbox_l1=.5*(_mean(box_regression(predicted_box),matched)
                +_mean(box_regression(visual_box),visible_cases))
    bbox_iou=.5*(_mean(1-box_iou(predicted_box,boxes),matched)
                 +_mean(1-box_iou(visual_box,boxes),visible_cases))
    mask_bce,mask_dice=_mask_terms(output['enemy_visible_logits_hr'],mask,applicable)
    rgba=output['enemy_rgba'].float()
    source_crop=sample_box(batch['hr_source'].detach().float(),boxes,rgba.shape[-1])
    alpha_crop=sample_box(mask,boxes,rgba.shape[-1]).clamp(0,1)
    canonical_error=((rgba[:,:3]-source_crop).abs()*alpha_crop).flatten(1).sum(-1)
    canonical_error=canonical_error/(3*alpha_crop.flatten(1).sum(-1)).clamp_min(1)
    alpha_error=F.binary_cross_entropy(rgba[:,3:4].clamp(1e-6,1-1e-6),alpha_crop,reduction='none').flatten(1).mean(-1)
    terms={'bbox_l1':bbox_l1,'bbox_iou':bbox_iou,'visible_mask_bce':mask_bce,
        'visible_mask_dice':mask_dice,'canonical_rgb':_mean(canonical_error,canonical_matched),
        'canonical_alpha':_mean(alpha_error,canonical_matched),
        'confirmation_bce':balanced_binary_loss(output['enemy_confirmation_logits'],confirmation_label,applicable),
        'animation_ce':animation_ce}
    total=base+sum(LOSS_WEIGHTS[name]*value for name,value in terms.items())
    if not bool(torch.isfinite(total)):
        raise FloatingPointError('Nonfinite predicted-hypothesis enemy decoder objective')
    for name,value in terms.items():
        metrics[name]=float(value.detach());metrics[name+'_weighted']=LOSS_WEIGHTS[name]*metrics[name]
    metrics.update(train_loss=float(total.detach()),loss_protocol=PROTOCOL,
        applicable_hypotheses=int(applicable.sum()),matching_hypotheses=int(matched.sum()),
        canonical_matching_hypotheses=int(canonical_matched.sum()),animation_phase_conflict_count=phase_conflicts,
        confirmation_positive_count=int((confirmation_label&applicable).sum()),
        confirmation_negative_count=int((~confirmation_label&applicable).sum()),
        global_presence_positive_count=int(global_presence.sum()),
        global_presence_negative_count=int((global_presence==0).sum()),
        hypothesis_match_from_original_proposal=True,source_labels_detached=True,
        presence_normalization='present_positive_negative_class_means_1_to_1',
        global_presence_target='actual_Source_HR_visible_mask_including_subpixel_LR',
        inference_inputs=['NPR_RGB32','predicted_RSSM_state20_and_validbit'])
    return total,metrics
