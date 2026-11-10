"""Explicit renderer, dynamics and joint objectives for the staged world.

Freezing preserves optimizer membership/moments. Shared map representations are
frozen in both isolated phases, so dynamics updates cannot change the renderer's
fixed-pose condition. Only joint training may update that shared encoder.
"""
from __future__ import annotations

import torch

from .object_losses import (balanced_image_l1, balanced_edge_l1,
                            foreground_contrastive_loss, object_region_weights)
from .structured_state import STATE_DIM


COMPONENT_PHASES = ('renderer', 'dynamics', 'joint')


def configure_component_phase(model, phase):
    """Set parameter/mode ownership, clearing stale gradients at safe boundaries.

    Call after any enclosing ``model.train()`` reset and before starting the
    next objective. This never replaces parameters or clears Adam moments.
    The complete optimizer must be constructed/restored before this function.
    """
    if phase not in COMPONENT_PHASES:
        raise ValueError('Component phase must be renderer, dynamics or joint')
    backend, flow = getattr(model, 'vjepa_backend', None), getattr(model, 'flow_renderer', None)
    renderer = getattr(backend, 'renderer', None)
    map_encoder = getattr(renderer, 'map_encoder', None)
    if backend is None or flow is None or renderer is None or map_encoder is None:
        raise ValueError('Component phases require the V-JEPA world, Base renderer and Flow renderer')
    parameters = list(model.parameters())
    ids = lambda module: {id(p) for p in module.parameters()} if module is not None else set()
    codec = getattr(flow, 'codec', None)
    codec_ids, map_ids = ids(codec), ids(map_encoder)
    base_ids, flow_ids = ids(renderer)-map_ids, ids(flow)-codec_ids
    appearance = getattr(backend, 'feature_head', None)
    appearance_ids = ids(appearance)
    dynamic_ids = {id(p) for p in parameters}-base_ids-flow_ids-codec_ids-map_ids-appearance_ids
    selected = (base_ids | flow_ids if phase == 'renderer' else dynamic_ids if phase == 'dynamics'
                else {id(p) for p in parameters}-codec_ids)
    model.train()
    for parameter in parameters:
        parameter.grad = None
        parameter.requires_grad_(id(parameter) in selected)
    if phase == 'renderer':
        backend.eval()
        renderer.train()
        flow.train()
    elif phase == 'dynamics':
        renderer.eval()
        flow.eval()
        if appearance is not None:
            appearance.eval()
    if phase != 'joint':
        map_encoder.eval()
    if codec is not None:
        codec.eval().requires_grad_(False)
    count = lambda subset: sum(p.numel() for p in parameters if id(p) in subset and p.requires_grad)
    return {'phase': phase, 'component_phase': phase,
        'renderer_enabled': phase != 'dynamics', 'dynamics_enabled': phase != 'renderer',
        'map': 'trainable' if phase == 'joint' else 'frozen',
        'shared_map_policy': 'frozen_until_joint', 'codec_frozen': True,
        'trainable_parameters': sum(p.numel() for p in parameters if p.requires_grad),
        'frozen_parameters': sum(p.numel() for p in parameters if not p.requires_grad),
        'base_trainable_parameters': count(base_ids), 'flow_trainable_parameters': count(flow_ids),
        'dynamics_trainable_parameters': count(dynamic_ids), 'map_trainable_parameters': count(map_ids),
        'appearance_trainable_parameters': count(appearance_ids)}


def _recorded_image_query(batch):
    if 'target_states' in batch and 'target_frames' in batch:
        states, frames = batch['target_states'], batch['target_frames']
        masks, backgrounds = batch.get('target_object_masks'), batch.get('target_background_frames')
        if (states.ndim != 3 or frames.ndim != 5 or states.shape[:2] != frames.shape[:2]
                or states.shape[1] < 1):
            raise ValueError('Renderer sequence supervision requires matching target pose/frame windows')
        state, target = states[:, -1], frames[:, -1]
        mask = masks[:, -1] if masks is not None else None
        background = backgrounds[:, -1] if backgrounds is not None else None
    else:
        state, target = batch.get('states'), batch.get('frames')
        mask, background = batch.get('object_masks'), batch.get('background_frames')
    if (state is None or target is None or state.ndim != 2 or state.shape[-1] != STATE_DIM
            or target.ndim != 4 or target.shape[:2] != (len(state), 3)
            or mask is None or background is None or background.shape != target.shape
            or mask.shape[0] != len(state) or mask.shape[-2:] != target.shape[-2:]):
        raise ValueError('Renderer supervision requires exact matching GT poses, RGB, object masks and backgrounds')
    return tuple(value.detach() for value in (state, target, mask, background))


def render_only_loss(model, ema, batch, *, context=2, render=True,
                     object_region_boost=4., detach_flow_base=True, **unused):
    """GT-pose Base+Flow supervision, without transitions, encoder or EMA calls.

    A navigation sequence supplies its last matching future pose/frame pair;
    independent snapshot batches use their direct recorded pair. The returned
    endpoint supports the trainer's separate sampled-RGB/GAN refinement.
    """
    if not render:
        raise ValueError('The renderer component objective requires rendering')
    state, target, mask, background = _recorded_image_query(batch)
    grid = batch['map'].detach()
    with torch.no_grad():
        map_features = model.encode_map(grid).detach()
    features = state.new_zeros(len(state), 128)
    base = model.render_base(state, features, map_features)
    parts = {
        'base_image': balanced_image_l1(base, target, mask)+.2*balanced_edge_l1(base, target, mask),
        'foreground_base': .25*foreground_contrastive_loss(base, target, background, mask),
    }
    flow, flow_metrics = model.flow_renderer.loss(target, base.detach(), state,
        object_mask=mask, map_condition=grid, detach_base=True)
    parts['flow'] = flow
    metrics = {name: float(value.detach()) for name, value in parts.items()}
    metrics.update({name: float(value) for name, value in flow_metrics.items()})
    metrics.update(component_phase='renderer', renderer_only=True, dynamics_updated=False,
        rgb_supervised_steps=1, supervised_pose_aligned=True, ar_future_gt_feedback=False,
        map_features_detached=True, flow_base_detached=True)
    endpoint = {'base': base, 'state': state, 'target': target, 'features': features,
        'object_mask': mask, 'background': background, 'map_condition': grid,
        'pixel_weights': object_region_weights(mask, object_region_boost), 'pose_aligned': True}
    return sum(parts.values()), metrics, endpoint


def dynamics_only_loss(model, ema, batch, *, context=2, **options):
    """Free AR/latent/motion/event losses with no Base or Flow evaluation."""
    from .aligned_training import aligned_short_rollout_loss
    options = dict(options)
    options.update(render=False, image_objectives=False)
    loss, metrics, endpoint = aligned_short_rollout_loss(model, ema, batch, context=context, **options)
    if endpoint is not None:
        raise RuntimeError('Dynamics-only supervision must not produce an image endpoint')
    metrics.update(component_phase='dynamics', renderer_only=False, dynamics_updated=True)
    return loss, metrics, None
