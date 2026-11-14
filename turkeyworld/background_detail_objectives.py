"""Source-only detail losses without alpha, masks at inference, or replacement.

The complete image is supervised. The Source NPC mask only balances measured
foreground/background errors; neither region can turn off the full-image loss.
"""
from copy import deepcopy

import torch
from torch.nn import functional as F

from .neural_enemy_restoration_objectives import _ssim_error_map

PROTOCOL='background_detail_sr_v1'
LOSS_WEIGHTS=dict(image_l1=1.,image_ssim=.2,image_edge=.1,
    foreground_l1=1.,foreground_edge=.25,foreground_ssim=.25,
    background_l1=.25,background_edge=.1,high_frequency=.2,
    low_frequency_preservation=.5,residual_l1=.01)
OBJECTIVE_SETTINGS=dict(protocol=PROTOCOL,version=1,weights=deepcopy(LOSS_WEIGHTS),
    inference_inputs='RGB32_only',Source_labels_loss_only=True,alpha_used=False,
    predicted_mask_gates_loss=False,full_image_supervision=True,
    foreground_background_reduction='equal_present_image_means_normalized_by_Source_region_area',
    background_edge_scope='Source_background_outside_NPC_not_wall_only',
    high_frequency_kernel=5,high_frequency_pool='stride1_count_include_pad_false',
    low_frequency_scales=[4,8],low_frequency_target='same_frozen_display_background',
    low_frequency_is_soft_regularizer=True,strict_highpass_inference=False,
    residual_regularization='mean_absolute_bounded_proposed_residual',GAN=False,
    fixed_weights_without_EMA=True)


def _region(error,mask):
    mass=mask.flatten(1).sum(-1)*error.shape[1]
    value=(error*mask).flatten(1).sum(-1)/mass.clamp_min(1e-12)
    return value,mass>0


def _present(values,present):
    return values[present].mean()if bool(present.any())else values.sum()*0


def high_frequency_image(image):
    return image-F.avg_pool2d(image,5,stride=1,padding=2,count_include_pad=False)


def low_frequency_image_change(image,background):
    """Per-image soft preservation of 32/16px coarse color and geometry."""
    return torch.stack([(F.avg_pool2d(image,k)-F.avg_pool2d(background,k)).abs().flatten(1).mean(-1)
        for k in(4,8)]).mean(0)


def _edges(image,target,mask=None,*,background=False):
    values=[];supports=[]
    for axis in(-1,-2):
        error=(image.diff(dim=axis)-target.diff(dim=axis)).abs()
        if mask is None:
            value=error.flatten(1).mean(-1);present=torch.ones(len(image),dtype=torch.bool,device=image.device)
        else:
            first,second=[slice(None)]*4,[slice(None)]*4
            first[axis]=slice(1,None);second[axis]=slice(None,-1)
            a,b=mask[tuple(first)],mask[tuple(second)]
            support=torch.minimum(a,b)if background else torch.maximum(a,b)
            value,present=_region(error,support)
        values.append(value);supports.append(present)
    supports=torch.stack(supports)
    return (torch.stack(values)*supports).sum(0)/supports.sum(0).clamp_min(1),supports.any(0)


def background_detail_loss(output,batch):
    tensors={}
    for key in('image','background','residual'):
        value=output.get(key)
        if(not torch.is_tensor(value)or not value.is_floating_point()or value.ndim!=4
                or value.shape[1:]!=(3,128,128)or not len(value)or not bool(torch.isfinite(value).all())):
            raise ValueError('Detail loss requires finite native128 image/background/residual')
        tensors[key]=value.float()
    image,base,residual=(tensors[key]for key in('image','background','residual'))
    if(any(value.shape!=image.shape or value.device!=image.device for value in tensors.values())
            or output['background'].requires_grad or bool(((image<0)|(image>1)).any())
            or bool(((base<0)|(base>1)).any())):
        raise ValueError('Detail loss requires aligned bounded RGB and a frozen display background')
    source=batch.get('hr_source');mask=batch.get('mask_hr')
    if not torch.is_tensor(source)or not torch.is_tensor(mask):raise ValueError('Detail loss requires Source RGB and visible-NPC mask labels')
    source=source.detach().to(image);mask=mask.detach().to(image)
    if(source.shape!=image.shape or mask.shape!=(len(image),1,128,128)
            or not bool(torch.isfinite(source).all())or not bool(torch.isfinite(mask).all())
            or bool(((source<0)|(source>1)).any())or bool(((mask<0)|(mask>1)).any())):
        raise ValueError('Detail Source labels must be finite aligned128 images and masks in[0,1]')
    error=(image-source).abs();background=1-mask
    foreground_l1,visible=_region(error,mask);background_l1,has_background=_region(error,background)
    fg_edge,fg_edge_support=_edges(image,source,mask)
    bg_edge,bg_edge_support=_edges(image,source,background,background=True)
    edge,_=_edges(image,source)
    ssim_error=_ssim_error_map(image,source)
    fg_ssim,fg_ssim_support=_region(ssim_error,F.avg_pool2d(mask,11,stride=1))
    high_frequency=(high_frequency_image(image)-high_frequency_image(source)).abs().mean()
    low_frequency=low_frequency_image_change(image,base.detach())
    terms=dict(image_l1=error.mean(),image_ssim=ssim_error.mean(),image_edge=edge.mean(),
        foreground_l1=_present(foreground_l1,visible),foreground_edge=_present(fg_edge,fg_edge_support),
        foreground_ssim=_present(fg_ssim,fg_ssim_support),background_l1=_present(background_l1,has_background),
        background_edge=_present(bg_edge,bg_edge_support),high_frequency=high_frequency,
        low_frequency_preservation=low_frequency.mean(),residual_l1=residual.abs().mean())
    loss=sum(LOSS_WEIGHTS[name]*value for name,value in terms.items())
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite background detail objective')
    metrics={name:float(value.detach())for name,value in terms.items()}
    metrics.update({name+'_weighted':LOSS_WEIGHTS[name]*metrics[name]for name in terms})
    metrics.update(train_loss=float(loss.detach()),objective_protocol=PROTOCOL,loss_protocol=PROTOCOL,
        visible_images=int(visible.sum()),empty_images=int((~visible).sum()),
        background_images=int(has_background.sum()),foreground_edge_images=int(fg_edge_support.sum()),
        foreground_ssim_images=int(fg_ssim_support.sum()),Source_labels_detached=True,
        inference_rgb_only=True,alpha_used=False,predicted_mask_gates_loss=False,
        full_image_supervision=True,low_frequency_is_soft_regularizer=True,
        source_background_reconstruction_gradient=True,background_edge_scope='outside_NPC_not_walls_only',
        GAN_active=False,atlas_pixels_used=False)
    return loss,metrics
