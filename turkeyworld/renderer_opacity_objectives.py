"""Source-label-only first-surface calibration for a private opacity network."""
from __future__ import annotations

import math
from copy import deepcopy

import torch
from torch.nn import functional as F

from .renderer_wall_depth import DEPTH_BANDS


PROTOCOL = 'private_learned_opacity_first_surface_v1'
MAX_DEPTH = 24.
RAY_COUNT = 32
SAMPLES = 64
LOG_DEPTH_WEIGHT = .1
HEIGHT_WEIGHT = .05


def opacity_depth_loss(geometry, labels):
    """Return balanced first-hit/depth/height loss and plain JSON diagnostics.

    Exact Source depth creates detached training labels, never neural runtime
    conditions. Each ray contributes the first-termination negative log
    likelihood: hit NLL plus the SUM over every free-prefix sample. Each present Source depth band
    then has equal weight after per-image, per-present-ray means. Samples after
    the first surface have no opacity label and cannot dominate its prefix.
    """
    opacity, depth = geometry.get('opacity'), geometry.get('depth')
    target = labels.get('depth')
    if (not torch.is_tensor(opacity) or opacity.ndim != 3 or opacity.shape[1:] != (RAY_COUNT, SAMPLES)
            or not torch.is_tensor(depth) or depth.shape != opacity.shape[:2]
            or not torch.is_tensor(target) or target.shape != (len(opacity), 1, 1, RAY_COUNT)
            or len(opacity) < 1):
        raise ValueError('Opacity calibration requires opacity[B,32,64], depth[B,32], Source depth[B,1,1,32]')
    if (not bool(torch.isfinite(opacity).all() and torch.isfinite(depth).all() and torch.isfinite(target).all())
            or bool(((opacity < 0) | (opacity > 1)).any()) or bool((depth <= 0).any())
            or bool((target <= 0).any()) or bool((target > MAX_DEPTH).any())):
        raise ValueError('Opacity calibration requires finite probabilities and positive depths within the sampled map')
    alpha = opacity.float()
    predicted = depth.float()
    target = target.detach().to(device=predicted.device, dtype=torch.float32)[:, 0, 0]
    centers = (torch.arange(SAMPLES, device=predicted.device, dtype=torch.float32)+.5)*(MAX_DEPTH/SAMPLES)
    # argmin is explicit at half-bin ties; Source values never become inputs.
    hit = (centers[None, None]-target[..., None]).abs().argmin(-1)
    prefix = torch.arange(SAMPLES, device=predicted.device)[None, None] < hit[..., None]
    prefix_count = prefix.sum(-1)
    safe_alpha = alpha.clamp(1e-6, 1-1e-6)
    hit_nll = -safe_alpha.gather(-1, hit[..., None]).squeeze(-1).log()
    prefix_nll = (-torch.log1p(-safe_alpha)*prefix).sum(-1)
    first_surface = hit_nll+prefix_nll
    log_depth = F.smooth_l1_loss(predicted.clamp_min(.02).log(), target.clamp_min(.02).log(), reduction='none')
    focal = 32/(2*math.tan(math.radians(68)/2))
    target_height = (focal/target).clamp(max=32)
    predicted_height = (focal/predicted.clamp_min(.02)).clamp(max=32)
    height = F.smooth_l1_loss(predicted_height, target_height, reduction='none')
    zero = alpha.sum()*0

    def reduce(value, mask):
        counts = mask.sum(-1)
        present = counts > 0
        per_image = (value*mask).sum(-1)/counts.clamp_min(1)
        return (per_image[present].mean() if bool(present.any()) else zero), int(present.sum())

    bands, losses = {}, []
    term_values = {name:[] for name in ('first_surface_nll', 'hit_nll', 'prefix_free_nll',
        'log_depth_smooth_l1', 'projected_height_smooth_l1')}
    for name, low, high in DEPTH_BANDS:
        mask = (target >= low) & (target < high)
        if not bool(mask.any()):
            bands[name] = {'images':0, 'rays':0, 'prefix_samples':0, 'no_prefix_rays':0, 'present':False}
            continue
        values = {}
        for field, error in (('first_surface_nll',first_surface),('hit_nll',hit_nll),
                ('prefix_free_nll',prefix_nll),('log_depth_smooth_l1',log_depth),
                ('projected_height_smooth_l1',height)):
            values[field], images = reduce(error,mask)
            term_values[field].append(values[field])
        loss = values['first_surface_nll']+LOG_DEPTH_WEIGHT*values['log_depth_smooth_l1']+HEIGHT_WEIGHT*values['projected_height_smooth_l1']
        losses.append(loss)
        mae,_ = reduce((predicted-target).abs(),mask)
        height_mae,_ = reduce((predicted_height-target_height).abs(),mask)
        bands[name] = {'images':images,'rays':int(mask.sum()),
            'prefix_samples':int(prefix_count[mask].sum()),'no_prefix_rays':int((mask&(prefix_count==0)).sum()),'present':True,
            **{field:float(value.detach()) for field,value in values.items()},
            'loss':float(loss.detach()),'depth_mae_tiles':float(mae.detach()),
            'projected_height_mae_pixels':float(height_mae.detach())}
    loss = torch.stack(losses).mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite private opacity calibration objective')
    metrics = {'protocol':PROTOCOL,'loss':float(loss.detach()),
        **{name:float(torch.stack(values).mean().detach()) for name,values in term_values.items()},
        'log_depth_weight':LOG_DEPTH_WEIGHT,'projected_height_weight':HEIGHT_WEIGHT,
        'depth_mae_tiles':float((predicted-target).abs().mean().detach()),
        'projected_height_mae_pixels':float((predicted_height-target_height).abs().mean().detach()),
        'images':int(len(opacity)),'rays':int(target.numel()),'present_bands':int(len(losses)),
        'hit_samples':int(target.numel()),'free_prefix_samples':int(prefix.sum()),
        'no_prefix_rays':int((prefix_count==0).sum()),
        'hit_bin_min':int(hit.min()),'hit_bin_max':int(hit.max()),'bands':bands,
        'reduction':'mean_present_bands(mean_present_images(mean_rays(hit_nll+sum_free_prefix_nll)))',
        'after_surface_samples_supervised':False,'source_labels_detached':True,
        'Source_RGB_or_object_mask_used_as_neural_input':False,'source_future_feedback':False,
        'runtime_exact_raycast':False,'projected_height_clipped_at_native_size':32,
        'quality_qualified':False}
    return loss,metrics


def opacity_depth_report(model, data, panel, *, batch_size=8):
    """Read-only fixed held-out geometry report with whole-panel denominators.

    Only state/map condition the learned geometry. Exact ray intersections are
    detached offline labels. Source RGB, current images and posterior encoders
    are not queried. The function never changes sampler clocks, modes or RNG.
    """
    from .renderer_wall_depth import source_wall_depth_labels
    from .renderer_student import _hash_tensors
    from .rssm_preview_assets import preserve_preview_rng
    from .staged_validation import _isolated_validation
    if (getattr(data,'split',None) != 'val' or type(batch_size) is not int
            or batch_size<1 or not len(panel)):
        raise ValueError('Opacity depth report requires a nonempty held-out panel and positive batch size')
    indices=[int(index) for index in panel]
    if len(indices)!=len(set(indices)) or min(indices)<0:
        raise ValueError('Opacity validation panel indices must be distinct and nonnegative')
    fingerprint=data.panel_fingerprint(indices)
    sampler=deepcopy(data.sampler_state()) if hasattr(data,'sampler_state') else None
    device=next(model.parameters()).device
    before=_hash_tensors(model.state_dict())
    parts={key:[] for key in ('depth','opacity','target_depth')}
    try:
        with preserve_preview_rng(device), _isolated_validation(model,data,173):
            for first in range(0,len(indices),batch_size):
                selected=indices[first:first+batch_size]
                batch=data.batch(len(selected),indices=selected)
                state=batch['state'].detach().to(device=device,dtype=torch.float32)
                grid=batch['map'].detach().to(device=device)
                fmap=model.encode_map(grid)
                geometry=model.render_geometry(state,fmap)
                labels=source_wall_depth_labels(batch['map'],batch['state'],batch['object_mask'])
                parts['depth'].append(geometry['depth'].detach().float().cpu())
                parts['opacity'].append(geometry['opacity'].detach().float().cpu())
                parts['target_depth'].append(labels['depth'].detach().float().cpu())
            # One aggregation over every image, not an average of batch means;
            # an unequal final batch cannot change a band denominator.
            joined={key:torch.cat(values) for key,values in parts.items()}
            _,report=opacity_depth_loss({'depth':joined['depth'],'opacity':joined['opacity']},
                {'depth':joined['target_depth']})
    finally:
        if sampler is not None:
            data.restore_sampler(sampler)
    if (_hash_tensors(model.state_dict())!=before
            or data.panel_fingerprint(indices)!=fingerprint):
        raise AssertionError('Read-only opacity validation changed model weights or its Source panel')
    report.update(panel_fingerprint=fingerprint,frames=len(indices),source_held_out=True,
        split=data.split,data_identity_fingerprint=getattr(data,'identity',{}).get('fingerprint'),
        source_sampler_unchanged=True,model_tensors_unchanged=True,
        top_boundary_mae_pixels=.5*report['projected_height_mae_pixels'],
        bottom_boundary_mae_pixels=.5*report['projected_height_mae_pixels'],
        private_opacity_enabled=bool(getattr(getattr(model,'config',None),'flow_geometry_opacity_enabled',False)),
        renderer_RGB_evaluated=False,source_RGB_encoded=False,
        scope='Held-out true-pose learned-depth conditioning only; projected boundaries are not final RGB masks')
    for band in report['bands'].values():
        if band['present']:
            band['top_boundary_mae_pixels']=.5*band['projected_height_mae_pixels']
            band['bottom_boundary_mae_pixels']=.5*band['projected_height_mae_pixels']
    return report
