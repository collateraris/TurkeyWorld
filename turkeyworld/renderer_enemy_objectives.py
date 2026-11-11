"""One-NPC Source image supervision with strictly causal frozen appearance.

Independent static snapshots have no preceding observations and therefore use
zero appearance features. Dynamic rows observe only their recorded previous
image/state, then run a frozen prior before the next Source pose/image is read.
"""
from __future__ import annotations

import math

import torch

from .models import STATE_DIM
from .object_losses import (balanced_edge_l1, balanced_image_l1, decoded_ssim_loss,
                            foreground_contrastive_loss)
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
        geometry = {name: value.detach() for name, value in
                    model.vjepa_backend.geometry(state, map_features).items() if torch.is_tensor(value)}
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


def enemy_refinement_loss(model, batch, *, ssim_weight=.2, edge_weight=.1, contrast_weight=.1):
    """Supervise compact Base/RF1 with equal present foreground/background mass."""
    for name, value in (('ssim_weight', ssim_weight), ('edge_weight', edge_weight), ('contrast_weight', contrast_weight)):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(name+' must be finite and nonnegative')
    flow = model.flow_renderer
    codec = flow.codec
    if (flow.kind != 'student' or flow.steps != 1 or codec.training or codec.quantize.training
            or any(parameter.requires_grad for parameter in codec.parameters())):
        raise ValueError('Enemy refinement requires RF1 and the entire codec/quantizer frozen in eval')
    conditions = enemy_conditions(model, batch)
    base, rgb, _ = refinement_forward(model, conditions, detach_base=True)
    if not bool(torch.isfinite(base).all() and torch.isfinite(rgb).all()):
        raise FloatingPointError('Nonfinite enemy Base/RF1 output')
    target, mask, background = (conditions[name] for name in ('target', 'object_mask', 'background'))
    terms = {}
    for name, prediction in (('base', base), ('rf1', rgb)):
        terms[name+'_source_l1'] = balanced_image_l1(prediction, target, mask)
        terms[name+'_source_ssim_loss'] = decoded_ssim_loss(prediction, target, mask)
        terms[name+'_source_edge'] = balanced_edge_l1(prediction, target, mask)
        terms[name+'_foreground_contrast'] = foreground_contrastive_loss(prediction, target, background, mask)
    total = sum(terms[name+'_source_l1']+ssim_weight*terms[name+'_source_ssim_loss']
                +edge_weight*terms[name+'_source_edge']+contrast_weight*terms[name+'_foreground_contrast']
                for name in ('base', 'rf1'))
    if not bool(torch.isfinite(total)):
        raise FloatingPointError('Nonfinite enemy renderer objective')
    metrics = {name: float(value.detach()) for name, value in terms.items()}
    metrics.update(train_loss=float(total.detach()),
        static_examples=int((~conditions['dynamic']).sum()), dynamic_examples=int(conditions['dynamic'].sum()),
        visible_enemy_examples=int(mask.flatten(1).any(-1).sum()), foreground_pixel_fraction=float(mask.mean()),
        static_appearance_features_zero=True, dynamic_appearance_from_previous_rgb_only=True,
        future_source_rgb_in_prior=False, future_source_state_in_prior=False,
        source_pose_aligned=True, flow_base_detached=True, codec_frozen=True, solver_steps=1,
        region_reduction='equal_present_foreground_background_per_image')
    return total, metrics
