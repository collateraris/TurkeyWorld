"""Source-only labels for NN geometry/visibility/phase plus fixed-atlas SR.

Geometry uses the FULL projected sprite rectangle, never the tight visible
crop. Occlusion, transparency and offscreen clipping remain separate. The label
renderer is a pure fixed-asset rasterizer called under no_grad, not a neural
forward, wall query, gameplay transition or inference-time Source correction.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .camera import project_sprite_bounds
from .enemy_animation import enemy_animation_index
from .enemy_decoder_objectives import balanced_binary_loss
from .enemy_decoder_sr import box_iou
from .object_losses import (balanced_image_l1,balanced_edge_l1,decoded_ssim_loss,
                            foreground_contrastive_loss)

PROTOCOL='atlas_enemy_hybrid_sr_v1'
LOSS_WEIGHTS=dict(pixel=1.,ssim=.2,edge=.1,boundary_edge=.1,foreground_contrast=.1,
    bbox_full_l1=.2,bbox_full_iou=.1,sprite_height_log_l1=.1,
    visibility_bce=.2,visibility_dice=.1,alpha_l1=.1,alpha_dice=.05,
    presence_bce=.05,confirmation_bce=.1,phase_ce=.1,player_dead_bce=.05)


def _mean(values,selected):
    selected=selected.detach().reshape(-1).float()
    return (values.reshape(-1)*selected).sum()/selected.sum().clamp_min(1)


def _class_mean(values,visible):
    return (_mean(values,visible)+_mean(values,~visible))/(
        visible.any().float()+(~visible).any().float()).clamp_min(1)


@torch.no_grad()
def source_full_sprite_geometry(state):
    """Detached 128px Source camera labels; square live AND corpse bounds."""
    if (not torch.is_tensor(state)or state.ndim!=2 or state.shape[-1]!=20
            or not state.is_floating_point()or not bool(torch.isfinite(state).all())):
        raise ValueError('Atlas geometry labels require finite Source state[B,20]')
    projection=project_sprite_bounds(state.detach().float(),image_size=128,map_size=16)
    box=projection['unclipped_boxes'][:,0]
    return dict(full_bbox=box,geometry_valid=projection['sprite_frustum'][:,0].bool(),
        full_sprite_height=box[:,3]-box[:,1],phase=enemy_animation_index(state.detach()),
        player_dead=state.detach()[:,4]<=0,
        coordinate_convention='Source128_full_unclipped_xyxy_square_integer_raster',
        original_state_precision='recorded_float32_pose_near_integer_threshold_may_differ_by_one_pixel')


def _balanced_soft_pixel(error,target):
    target=target.detach().float().flatten(1);error=error.flatten(1)
    positive,negative=target.sum(-1),(1-target).sum(-1)
    return (((error*target).sum(-1)/positive.clamp_min(1e-8)
             +(error*(1-target)).sum(-1)/negative.clamp_min(1e-8))
            /((positive>0).float()+(negative>0).float()).clamp_min(1))


def _soft_dice(probability,target):
    p,t=probability.float().flatten(1),target.detach().float().flatten(1)
    # The squared denominator has zero error for IDENTICAL soft PNG alpha.
    return 1-(2*(p*t).sum(-1)+1e-6)/(p.square().sum(-1)+t.square().sum(-1)+1e-6)


def _boundary_error(image,target,mask):
    numerator=image.new_zeros(len(image));denominator=image.new_zeros(len(image))
    for axis in(-1,-2):
        a,b=[slice(None)]*4,[slice(None)]*4
        a[axis],b[axis]=slice(1,None),slice(None,-1);a,b=tuple(a),tuple(b)
        boundary=(mask[a]-mask[b]).abs().detach()
        error=((image[a]-image[b])-(target[a]-target[b])).abs()
        numerator+=(error*boundary).flatten(1).sum(-1)
        denominator+=3*boundary.flatten(1).sum(-1)
    return numerator/denominator.clamp_min(1),denominator>0


def _proposal_confirmed(box,mask):
    box=box.detach().float()
    axis=(torch.arange(128,device=box.device,dtype=torch.float32)+.5)/128
    y,x=torch.meshgrid(axis,axis,indexing='ij')
    inside=((x[None]>=box[:,0,None,None])&(x[None]<box[:,2,None,None])
            &(y[None]>=box[:,1,None,None])&(y[None]<box[:,3,None,None]))
    mass=mask.flatten(1).sum(-1)
    overlap=(mask[:,0]*inside).flatten(1).sum(-1)/mass.clamp_min(1)
    return (mass>0)&(overlap>=.1)


def atlas_enemy_loss(output,batch,*,label_renderer=None):
    """Only NN heads/final composite learn; atlas colors and Source labels do not.

    ``label_renderer`` is model.render_label_atlas(full_box, phase_index,
    dead_index=None), a pure fixed-asset operation. Source arguments are passed
    ONLY to this detached label builder, never to the ordinary SR forward.
    """
    image=output['image'];device=image.device;count=len(image)
    target=batch['hr_source'].detach().to(device=device,dtype=torch.float32)
    background=batch['hr_background'].detach().to(device=device,dtype=torch.float32)
    mask=batch['mask_hr'].detach().to(device=device,dtype=torch.float32)
    state=batch['state'].detach().to(device=device,dtype=torch.float32)
    if (image.shape!=(count,3,128,128)or target.shape!=image.shape or background.shape!=image.shape
            or mask.shape!=(count,1,128,128)or state.shape!=(count,20)
            or not all(bool(torch.isfinite(v).all())for v in(image,target,background,mask,state))
            or bool(((mask<0)|(mask>1)).any())):
        raise ValueError('Atlas objective requires aligned finite Source128/state20/visible-mask labels')
    labels=source_full_sprite_geometry(state)
    visible=mask.flatten(1).any(-1)
    geometry_selected=visible&labels['geometry_valid']
    if not callable(label_renderer):
        raise ValueError('Pass the pure fixed-asset render_label_atlas callback for soft-alpha labels')
    with torch.no_grad():
        source_atlas=label_renderer(labels['full_bbox'],labels['phase'],dead_index=labels['player_dead'])
        if (source_atlas.shape!=(count,4,128,128)or not bool(torch.isfinite(source_atlas).all())
                or bool(((source_atlas<0)|(source_atlas>1)).any())):
            raise ValueError('Fixed Source atlas label renderer returned invalid RGB/alpha')
        alpha_target=source_atlas[:,3:4].detach()*mask
    full_box=output['atlas_full_bbox'].float()
    visual_box=output.get('visual_full_bbox',full_box).float()
    logits=output['atlas_visible_logits_hr'].float()
    effective_alpha=output['atlas_effective_alpha_hr'].float()
    phase_logits=output['atlas_phase_logits'].float()
    dead_logits=output['atlas_player_dead_logits'].float()
    if (full_box.shape!=(count,4)or visual_box.shape!=(count,4)
            or logits.shape!=mask.shape or effective_alpha.shape!=mask.shape
            or phase_logits.shape!=(count,7)or dead_logits.shape!=(count,1)
            or not all(bool(torch.isfinite(v).all())for v in
                (full_box,visual_box,logits,effective_alpha,phase_logits,dead_logits))
            or bool(((effective_alpha<0)|(effective_alpha>1)).any())):
        raise ValueError('Atlas objective needs finite full-box/visibility/soft-alpha/7-phase/HP0 heads')
    if 'animation_label'in batch and not torch.equal(batch['animation_label'].detach().to(device).long().reshape(-1),labels['phase']):
        raise ValueError('Source animation labels disagree with the Source state phase selector')
    boxes=[(visual_box,geometry_selected)]
    guided=output.get('guided_full_bbox')
    guided_selected=torch.zeros_like(visible)
    if guided is not None:
        guided=guided.float()
        if guided.shape!=(count,4)or not bool(torch.isfinite(guided).all()):
            raise ValueError('Invalid full guided atlas box')
        applicable=output.get('guided_applicable',output.get('enemy_applicable'))
        if applicable is not None:
            original=output.get('guided_hypothesis_full_bbox',guided).detach().float()
            guided_selected=geometry_selected&applicable.detach().bool().reshape(-1)&(box_iou(original,labels['full_bbox'])>=.1)
            if bool(guided_selected.any()):boxes.append((guided,guided_selected))
    bbox_l1=image.sum()*0;bbox_iou=bbox_l1;height=bbox_l1
    for predicted,selected in boxes:
        center=((predicted[:,:2]+predicted[:,2:]-labels['full_bbox'][:,:2]-labels['full_bbox'][:,2:])/2).abs().mean(-1)
        bbox_l1+=_mean((predicted-labels['full_bbox']).abs().mean(-1)+.5*center,selected)
        bbox_iou+=_mean(1-box_iou(predicted,labels['full_bbox']),selected)
        predicted_height=(predicted[:,3]-predicted[:,1]).clamp_min(1/1024)
        height+=_mean(F.smooth_l1_loss(predicted_height.log(),
            labels['full_sprite_height'].clamp_min(1/1024).log(),reduction='none'),selected)
    bce=F.binary_cross_entropy_with_logits(logits,mask,reduction='none')
    visibility_bce=_class_mean(_balanced_soft_pixel(bce,mask),visible)
    visibility_dice=_class_mean(_soft_dice(logits.sigmoid(),mask),visible)
    alpha_l1=_class_mean(_balanced_soft_pixel((effective_alpha-alpha_target).abs(),alpha_target),visible)
    alpha_dice=_class_mean(_soft_dice(effective_alpha,alpha_target),visible)
    boundary,has_boundary=_boundary_error(image.float(),target,mask)
    confirmation_logits=output.get('atlas_confirmation_logits',output.get('visual_confirmation_logits'))
    if confirmation_logits is None:raise ValueError('Atlas SR requires an RGB-only ROI confirmation head')
    presence=output['presence_logits']
    terms=dict(pixel=balanced_image_l1(image,target,mask),ssim=decoded_ssim_loss(image,target,mask),
        edge=balanced_edge_l1(image,target,mask),boundary_edge=_mean(boundary,has_boundary),
        foreground_contrast=foreground_contrastive_loss(image,target,background,mask),
        bbox_full_l1=bbox_l1/len(boxes),bbox_full_iou=bbox_iou/len(boxes),
        sprite_height_log_l1=height/len(boxes),visibility_bce=visibility_bce,visibility_dice=visibility_dice,
        alpha_l1=alpha_l1,alpha_dice=alpha_dice,
        presence_bce=balanced_binary_loss(presence,visible),
        confirmation_bce=balanced_binary_loss(confirmation_logits,_proposal_confirmed(visual_box,mask)),
        phase_ce=_mean(F.cross_entropy(phase_logits,labels['phase'],reduction='none'),visible),
        player_dead_bce=balanced_binary_loss(dead_logits,labels['player_dead']))
    total=sum(LOSS_WEIGHTS[name]*value for name,value in terms.items())
    if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite fixed-atlas enemy SR objective')
    metrics={name:float(value.detach())for name,value in terms.items()}
    metrics.update({name+'_weighted':LOSS_WEIGHTS[name]*metrics[name]for name in terms})
    metrics.update(train_loss=float(total.detach()),loss_protocol=PROTOCOL,objective_protocol=PROTOCOL,
        source_visible_frames=int(visible.sum()),source_empty_frames=int((~visible).sum()),
        full_geometry_frames=int(geometry_selected.sum()),guided_geometry_frames=int(guided_selected.sum()),
        source_visible_invalid_geometry=int((visible&~labels['geometry_valid']).sum()),
        source_player_dead_frames=int(labels['player_dead'].sum()),source_labels_detached=True,
        bbox_target='Source128_full_projected_unclipped_not_tight_visible',
        alpha_target='fixed_native_Source_atlas_alpha_times_actual_Source_visible_mask',
        phase_classes=7,unused_eighth_asset_frame_not_a_phase=True,
        learned_RGBA_canonical_loss_active=False,GAN_active=False,
        Source_targets_in_NN_forward=False,source_future_inputs=False)
    return total,metrics
