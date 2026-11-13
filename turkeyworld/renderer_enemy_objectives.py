"""One-NPC Source image supervision with strictly causal frozen appearance.

Independent static snapshots have no preceding observations and therefore use
zero appearance features. Dynamic rows observe only their recorded previous
image/state, then run a frozen prior before the next Source pose/image is read.
"""
from __future__ import annotations

import math

import torch

from .models import STATE_DIM
from .object_losses import (balanced_edge_l1, balanced_image_l1, balanced_mse, decoded_ssim_loss,
                            foreground_contrastive_loss, balanced_region_losses)
from .renderer_refinement import refinement_forward


def enemy_conditions(model, batch):
    """Build GT-pose render queries; masks/background/next RGB remain loss-only."""
    device = next(model.parameters()).device
    state = batch['state'].detach().to(device=device, dtype=torch.float32)
    grid = batch['map'].detach().to(device=device, dtype=torch.long)
    dynamic = batch['dynamic'].detach().to(device)
    count = len(state)
    if (state.shape != (count, STATE_DIM) or grid.shape != (count, 16, 16)
            or dynamic.shape != (count,) or dynamic.dtype != torch.bool or count < 1
            or not bool(torch.isfinite(state).all())):
        raise ValueError('Enemy conditions require finite[B,state20],map[B,16,16],dynamic[B]bool')
    features = state.new_zeros(count, 128)
    indices = torch.nonzero(dynamic, as_tuple=False).flatten()
    memory = None
    with torch.no_grad():
        if len(indices):
            # Read only the actual previous observations for dynamic rows.
            previous = batch['previous_state'].detach().to(device=device, dtype=torch.float32)[indices]
            image = batch['previous_rgb'].detach().to(device=device, dtype=torch.float32)[indices]
            controls = batch['action'].detach().to(device=device, dtype=torch.float32)[indices]
            if (previous.shape != (len(indices), STATE_DIM) or image.shape != (len(indices), 3, 32, 32)
                    or controls.shape != (len(indices), 4)
                    or not all(bool(torch.isfinite(value).all()) for value in (previous, image, controls))
                    or bool(((image < 0) | (image > 1)).any())):
                raise ValueError('Dynamic enemy rows require genuine finite previous state/RGB/action')
            dynamic_features = model.encode_map(grid[indices])
            initial_memory = model.initialize_from_image(image, previous, dynamic_features)
            prior = model.transition(previous, controls, dynamic_features, initial_memory)
            value, memory = prior['features'].detach().float(), prior['memory'].detach()
            if value.shape != (len(indices), 128) or not bool(torch.isfinite(value).all() and torch.isfinite(memory).all()):
                raise FloatingPointError('Frozen causal prior returned invalid enemy appearance/memory')
            features[indices] = value
        # The prior subset changes map/spawn caches to its batch shape. Restore
        # all rows before GT-pose geometry; no prior features are recomputed.
        map_features = model.encode_map(grid)
        private_opacity = getattr(model.flow_renderer, 'geometry_opacity', None)
        if private_opacity is None:
            geometry = {name: value.detach() for name, value in
                        model.vjepa_backend.geometry(state, map_features).items() if torch.is_tensor(value)}
    if private_opacity is not None:
        # Map, dynamics and labels remain detached; only the renderer-owned MLP
        # builds a graph through otherwise frozen camera/position projections.
        geometry = {name:value for name,value in model.render_geometry(state.detach(),map_features.detach()).items()
                    if torch.is_tensor(value)}
    target = batch['target'].detach().to(device=device, dtype=torch.float32)
    background = batch['background'].detach().to(device=device, dtype=torch.float32)
    mask = batch['object_mask'].detach().to(device=device, dtype=torch.float32)
    if (target.shape != (count, 3, 32, 32) or background.shape != target.shape
            or mask.shape != (count, 1, 32, 32)
            or not all(bool(torch.isfinite(value).all()) for value in (target, background, mask))
            or any(bool(((value < 0) | (value > 1)).any()) for value in (target, background, mask))):
        raise ValueError('Enemy targets/backgrounds/masks must be matching finite native32 labels in[0,1]')
    return {'state': state, 'features': features.detach(), 'geometry': geometry, 'map': grid,
            'target': target, 'background': background, 'object_mask': mask,
            'dynamic': dynamic, 'dynamic_indices': indices, 'dynamic_prior_memory': memory}


def balanced_base_regression_loss(prediction, target, base, mask=None):
    """Penalize RF being worse than detached Base in each present image region.

    Foreground/background and individual images cannot compensate one another.
    The moving Base reference and Source labels receive no gradient; an RF to
    Base gradient may still exist through the explicitly coupled forward path.
    This training penalty is not an inference guarantee or held-out release.
    """
    if (not torch.is_tensor(base) or base.shape != prediction.shape or base.device != prediction.device
            or not base.is_floating_point() or not bool(torch.isfinite(base).all())):
        raise ValueError('Base regression reference must match finite prediction RGB')
    current = balanced_region_losses(prediction, target, mask)
    reference = balanced_region_losses(base.detach(), target, mask)
    foreground = (current['foreground']-reference['foreground'].detach()).relu()
    background = (current['background']-reference['background'].detach()).relu()
    fg, bg = current['foreground_present'].to(foreground.dtype), current['background_present'].to(background.dtype)
    loss = ((foreground*fg+background*bg)/(fg+bg).clamp_min(1)).mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite Base-preservation loss')
    return loss


def source_object_crop_ssim(prediction, target, mask, *, padding=2):
    """Native-pixel local structure around actual Source foreground.

    A tiny opponent keeps its native pixels: no synthetic enlargement or
    interpolated target is used. Each nonempty image receives equal weight;
    empty images contribute no term. The crop is a training label only.
    """
    if (prediction.ndim != 4 or prediction.shape[1] != 3 or target.shape != prediction.shape
            or mask.shape != (len(prediction),1,*prediction.shape[-2:])
            or not isinstance(padding,int) or isinstance(padding,bool) or padding < 0
            or not all(bool(torch.isfinite(v).all()) for v in (prediction,target,mask))):
        raise ValueError('Object crops require matching finite RGB and Source occupancy masks')
    target, mask = target.detach(), mask.detach()
    terms=[]
    height,width=prediction.shape[-2:]
    for index in range(len(prediction)):
        pixels=torch.nonzero(mask[index,0]>.5,as_tuple=False)
        if not len(pixels):
            continue
        low,high=pixels.amin(0),pixels.amax(0)+1
        y0,x0=max(0,int(low[0])-padding),max(0,int(low[1])-padding)
        y1,x1=min(height,int(high[0])+padding),min(width,int(high[1])+padding)
        terms.append(decoded_ssim_loss(prediction[index:index+1,:,y0:y1,x0:x1],
            target[index:index+1,:,y0:y1,x0:x1],mask[index:index+1,:,y0:y1,x0:x1]))
    return torch.stack(terms).mean() if terms else prediction.sum()*0


def enemy_refinement_loss(model, batch, *, ssim_weight=.2, edge_weight=.1, contrast_weight=.1,
                          base_only=False, latent_weight=0., detach_base=True,
                          base_regression_weight=0., object_crop_weight=0., small_enemy_rebalance=False,
                          far_wall_supervision=False,conditions_fn=None):
    """Supervise compact Base/RF1 with equal present foreground/background mass."""
    for name, value in (('ssim_weight', ssim_weight), ('edge_weight', edge_weight),
                        ('contrast_weight', contrast_weight), ('latent_weight', latent_weight),
                        ('base_regression_weight', base_regression_weight),('object_crop_weight', object_crop_weight)):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(name+' must be finite and nonnegative')
    if not isinstance(base_only, bool):
        raise ValueError('base_only must be an explicit boolean')
    if not isinstance(detach_base, bool):
        raise ValueError('detach_base must be an explicit boolean')
    if not isinstance(small_enemy_rebalance, bool):
        raise ValueError('small_enemy_rebalance must be an explicit boolean')
    if not isinstance(far_wall_supervision,bool)or(far_wall_supervision and not small_enemy_rebalance):
        raise ValueError('Far-wall v6 requires an explicit boolean and the native-area v5 objective')
    if small_enemy_rebalance:
        from .renderer_small_enemy_protocol import separate_region_loss, native_crop_loss, small_enemy_protocol
    flow = model.flow_renderer
    codec = flow.codec
    if (flow.kind != 'student' or flow.steps != 1 or codec.training or codec.quantize.training
            or any(parameter.requires_grad for parameter in codec.parameters())):
        raise ValueError('Enemy refinement requires RF1 and the entire codec/quantizer frozen in eval')
    conditions = (conditions_fn or enemy_conditions)(model, batch)
    residual_enabled = bool(getattr(model.base_student.config, 'pixel_residual', False))
    residual = None
    if base_only:
        # Direct Base curriculum: no RF sample, quantization or codec decoding.
        base = model.base_student(conditions['state'], conditions['features'], conditions['geometry'])
        predictions = (('base', base),)
    else:
        if residual_enabled:
            base, rgb, endpoint, residual = refinement_forward(model, conditions, detach_base=detach_base, return_residual=True)
        else:
            base, rgb, endpoint = refinement_forward(model, conditions, detach_base=detach_base)
        predictions = (('base', base), ('rf1', rgb))
    if not all(bool(torch.isfinite(value).all()) for _, value in predictions):
        raise FloatingPointError('Nonfinite enemy Base/RF1 output')
    target, mask, background = (conditions[name] for name in ('target', 'object_mask', 'background'))
    terms = {}
    separate_regions = {}
    crop_terms = {}
    wall_terms = {}
    far_terms = {}
    if small_enemy_rebalance:
        from .renderer_wall_supervision import source_wall_masks,wall_loss_terms
        wall_labels = source_wall_masks(conditions['map'],conditions['state'],mask)
    if far_wall_supervision:
        from .renderer_wall_depth import source_wall_depth_labels
        from .renderer_far_wall_protocol import far_wall_loss,far_wall_protocol
        far_labels=source_wall_depth_labels(conditions['map'],conditions['state'],mask)
    for name, prediction in predictions:
        if small_enemy_rebalance:
            separate_regions[name] = separate_region_loss(prediction, target, mask)
            terms[name+'_source_l1'] = separate_regions[name]['loss']
            crop_terms[name] = native_crop_loss(prediction, target, mask)
            wall_terms[name] = wall_loss_terms(prediction,target,wall_labels)
            if far_wall_supervision:far_terms[name]=far_wall_loss(prediction,target,far_labels)
        else:
            terms[name+'_source_l1'] = balanced_image_l1(prediction, target, mask)
        terms[name+'_source_ssim_loss'] = decoded_ssim_loss(prediction, target, mask)
        terms[name+'_source_edge'] = balanced_edge_l1(prediction, target, mask)
        terms[name+'_foreground_contrast'] = foreground_contrastive_loss(prediction, target, background, mask)
    total = sum(terms[name+'_source_l1']+ssim_weight*terms[name+'_source_ssim_loss']
                +edge_weight*terms[name+'_source_edge']+contrast_weight*terms[name+'_foreground_contrast']
                for name, _ in predictions)
    common_rgb = total
    if small_enemy_rebalance:
        total = total+sum(value[0] for value in crop_terms.values())+sum(value['loss'] for value in wall_terms.values())
    if far_wall_supervision:total=total+sum(value[0]for value in far_terms.values())
    if object_crop_weight > 0:
        for name,prediction in predictions:
            crop=source_object_crop_ssim(prediction,target,mask)
            terms[name+'_object_crop_ssim_loss']=crop
            total=total+object_crop_weight*crop
    latent = None
    if latent_weight > 0 and not base_only:
        # Source RGB is encoded ONLY into a frozen loss target. It is never a
        # Flow/Base condition or an observation supplied to the world prior.
        with torch.no_grad():
            target_z = flow.normalize_latents(codec.encode(target)['quantized']).detach()
        latent = (separate_region_loss(endpoint,target_z,mask,squared=True)['loss']
                  if small_enemy_rebalance else balanced_mse(endpoint, target_z, mask))
        total = total+latent_weight*latent
    regression = None
    if base_regression_weight > 0 and not base_only:
        regression = balanced_base_regression_loss(rgb, target, base, mask)
        total = total+base_regression_weight*regression
    if not bool(torch.isfinite(total)):
        raise FloatingPointError('Nonfinite enemy renderer objective')
    metrics = {name: float(value.detach()) for name, value in terms.items()}
    metrics.update(train_loss=float(total.detach()),
        static_examples=int((~conditions['dynamic']).sum()), dynamic_examples=int(conditions['dynamic'].sum()),
        visible_enemy_examples=int(mask.flatten(1).any(-1).sum()), foreground_pixel_fraction=float(mask.mean()),
        static_appearance_features_zero=True, dynamic_appearance_from_previous_rgb_only=True,
        future_source_rgb_in_prior=False, future_source_state_in_prior=False,
        source_pose_aligned=True, flow_base_detached=detach_base, codec_frozen=True,
        flow_base_gradient_active=not detach_base and not base_only,
        base_only=base_only, flow_forward_active=not base_only, solver_steps=0 if base_only else 1,
        pixel_residual_active=residual_enabled and not base_only,
        source_latent_active=latent is not None,
        source_latent_weight=float(latent_weight) if latent is not None else 0.,
        common_rgb_objective=float(common_rgb.detach()),
        common_old_latent01_objective=float((common_rgb+.1*latent).detach()) if latent is not None else None,
        common_objective_scope='Same RGB terms; canonical latent weight0.1 excludes the new preservation penalty',
        region_reduction='equal_present_foreground_background_per_image')
    if small_enemy_rebalance:
        metrics.update(small_enemy_protocol=small_enemy_protocol(),
            region_reduction='L1_and_latent: .5*mean_visible_foreground+.5*mean_present_background',
            native_crop_active=True, source_wall_supervision_active=True, source_wall_masks_are_loss_only=True,
            source_pixels_resized=False,
            common_objective_scope='v5 independent regional L1; legacy SSIM/edge/contrast, crop reported separately')
        metrics.pop('common_old_latent01_objective',None)
        for name in separate_regions:
            for region in ('foreground','background'):
                metrics[name+'_'+region+'_l1'] = float(separate_regions[name][region].detach())
                metrics[name+'_'+region+'_images'] = separate_regions[name][region+'_images']
            _, crop = crop_terms[name]
            metrics.update({name+'_'+key:float(value.detach()) if torch.is_tensor(value) else int(value)
                            for key,value in crop.items()})
            metrics.update({name+'_wall_'+key:float(value.detach()) if torch.is_tensor(value) else int(value)
                            for key,value in wall_terms[name].items()})
    if object_crop_weight > 0:
        metrics.update(object_crop_weight=float(object_crop_weight),
            object_crop_scope='Native Source foreground bbox+2pixels; loss only, no resize or inference inputs')
    if far_wall_supervision:
        metrics.update(far_wall_protocol=far_wall_protocol(),far_wall_supervision_active=True)
        for name,(_,values)in far_terms.items():
            metrics.update({name+'_far_wall_'+key:value for key,value in values.items()})
    if regression is not None:
        metrics.update(rf1_base_regression_loss=float(regression.detach()),
            base_regression_weight=float(base_regression_weight),
            weighted_rf1_base_regression_loss=float(regression.detach())*base_regression_weight,
            base_regression_reference_detached=True)
    if latent is not None:
        metrics.update(source_latent_mse=float(latent.detach()),
                       weighted_source_latent_mse=float(latent.detach())*latent_weight,
                       source_latent_target_scope='frozen compact E+Q(Source); normalized endpoint loss only')
    if residual is not None:
        metrics.update(pixel_residual_abs_mean=float(residual.detach().abs().mean()),
                       pixel_residual_max_abs=float(residual.detach().abs().max()))
    return total, metrics


def far_wall_refinement_loss(model,batch,*,detach_base=True,conditions_fn=None):
    """Auxiliary GT-pose renderer query with ONLY the explicit v6 far terms."""
    from .renderer_far_wall_protocol import far_wall_loss,far_wall_protocol
    from .renderer_wall_depth import source_wall_depth_labels
    flow=model.flow_renderer;codec=flow.codec
    if (not isinstance(detach_base,bool)or flow.kind!='student'or flow.steps!=1
            or codec.training or codec.quantize.training
            or any(parameter.requires_grad for parameter in codec.parameters())):
        raise ValueError('Far-only auxiliary requires RF1 and the entire frozen codec in eval')
    conditions=(conditions_fn or enemy_conditions)(model,batch)
    base,rgb,_=refinement_forward(model,conditions,detach_base=detach_base)
    labels=source_wall_depth_labels(conditions['map'],conditions['state'],conditions['object_mask'])
    total=base.sum()*0;metrics={'far_wall_protocol':far_wall_protocol(),
        'far_wall_aux_examples':len(base),'far_wall_aux_objective_scope':'v6 far-depth terms only',
        'far_wall_aux_future_source_feedback':False}
    for name,prediction in (('base',base),('rf1',rgb)):
        value,terms=far_wall_loss(prediction,conditions['target'],labels)
        total=total+value
        metrics.update({'far_wall_aux_'+name+'_'+key:term for key,term in terms.items()})
    if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite far-only renderer auxiliary')
    metrics['far_wall_aux_loss']=float(total.detach())
    return total,metrics
