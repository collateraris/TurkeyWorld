"""Post-backward combat-head image supervision on detached Source contexts.

The main autoregressive trajectory must already have finished. Source events
select an auxiliary image query; Source pose fixes its camera/object geometry.
Neither these labels nor this recomputed image state feed the main trajectory.
Only the combat heads are recomputed from the captured, detached context. Base
and the complete Flow sampler render the accepted combat fields differentiably.
"""
from __future__ import annotations

import math

import torch

from .object_losses import object_region_weights, weighted_image_l1
from .structured_state import structured_transition


def _payload_tensors(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get('contexts'), dict):
        raise ValueError('Combat image payload requires captured contexts')
    contexts = payload['contexts']
    names = ('features', 'target_states', 'actions', 'events', 'target_frames', 'object_masks',
             'backgrounds', 'map_features')
    if any(name not in payload for name in names) or any(name not in contexts for name in ('state', 'pooled', 'spatial')):
        raise ValueError('Combat image payload is missing aligned tensors')
    values = {name: payload[name] for name in names}
    values.update({'context_'+name: contexts[name] for name in ('state', 'pooled', 'spatial')})
    if any(not torch.is_tensor(value) or not value.is_floating_point() for value in values.values()):
        raise ValueError('Combat image payload values must be floating point tensors')
    previous = values['context_state']
    if previous.ndim != 3 or previous.shape[-1] != 90 or min(previous.shape[:2]) < 1:
        raise ValueError('Captured combat states require shape[B,H,90]')
    batch, horizon = previous.shape[:2]
    pooled = values['context_pooled']
    images = values['target_frames']
    if (pooled.ndim != 3 or pooled.shape[:2] != (batch, horizon) or pooled.shape[-1] < 1
            or values['context_spatial'].shape != (batch, horizon, 9, pooled.shape[-1])
            or values['features'].shape != (batch, horizon, 128)
            or values['target_states'].shape != previous.shape
            or values['actions'].shape != (batch, horizon, 4)
            or values['events'].shape != (batch, horizon, 4)
            or images.ndim != 5 or images.shape[:3] != (batch, horizon, 3)
            or min(images.shape[-2:]) < 1 or images.shape[-2] != images.shape[-1]
            or values['backgrounds'].shape != images.shape
            or values['map_features'].ndim != 4 or values['map_features'].shape[0] != batch
            or values['map_features'].shape[1] < 1 or values['map_features'].shape[-2:] != (16, 16)):
        raise ValueError('Combat image payload tensors have inconsistent batch/horizon/geometry shapes')
    mask = values['object_masks']
    if mask.shape == (batch, horizon, 1, *images.shape[-2:]):
        mask = mask[:, :, 0]
    if mask.shape != (batch, horizon, *images.shape[-2:]):
        raise ValueError('Combat image masks must match the native target image grid')
    values['object_masks'] = mask
    device = previous.device
    if (any(value.device != device for value in values.values())
            or any(not bool(torch.isfinite(value.detach()).all()) for value in values.values())
            or bool(((mask.detach() < 0) | (mask.detach() > 1)).any())
            or bool(((values['events'].detach() < 0) | (values['events'].detach() > 1)).any())):
        raise ValueError('Combat image tensors must be finite, colocated, with valid masks/events')
    # Detach even if the caller already did so. This is safe after the original
    # context graph has been freed by the primary backward pass.
    return {name: value.detach() for name, value in values.items()}


def _select_offsets(values):
    previous, targets = values['context_state'], values['target_states']
    batch, horizon = previous.shape[:2]
    old = previous[:, :, 8:88].reshape(batch, horizon, 8, 10)
    target = targets[:, :, 8:88].reshape_as(old)
    old_alive = (old[..., 5] >= .5) & (old[..., 4] > 0)
    target_alive = (target[..., 5] >= .5) & (target[..., 4] > 0)
    valid = ~(~old_alive & target_alive).any(-1)
    events = values['events'] > .5
    # Lexicographic priorities: a kill wins over every combination of lower
    # cases; equal-priority offsets choose the most recent query. No event is
    # fed into the head or accepted transition.
    priority = torch.zeros((batch, horizon), dtype=torch.long, device=previous.device)
    for event, rank in ((0, 1), (3, 2), (1, 3), (2, 4)):
        priority = torch.where(events[..., event], torch.full_like(priority, rank), priority)
    offset = torch.arange(horizon, device=previous.device)[None].expand(batch, -1)
    score = (priority*(horizon+1)+offset).masked_fill(~valid, -1)
    best = score.argmax(-1)
    accepted = valid.any(-1)
    return torch.where(accepted, best, torch.full_like(best, -1)), accepted, valid


def _combat_zero(backend):
    parameters = []
    for name in ('hit_target_head', 'attack_head', 'spatial_combat_head'):
        module = getattr(backend, name, None)
        if module is not None:
            parameters.extend(module.parameters())
    if not parameters:
        raise ValueError('Combat image supervision requires trainable combat heads')
    # Keep the all-skipped objective connected to combat parameters only; it
    # must not manufacture gradients on a visual encoder or captured state.
    return sum((parameter.sum()*0 for parameter in parameters), parameters[0].new_zeros(()))


def combat_image_objective(model, payload):
    """Return (scalar loss, JSON metrics) for one auxiliary query per sample.

    The loss is .25*mean-one NPC-boost4 RGB L1 plus .25*foreground contrast
    against the offline no-NPC image. Respawn offsets are ineligible. The full
    configured Flow sampler is used, with activation checkpointing requested;
    the codec remains frozen according to the owning model's existing policy.
    This helper performs no backward/optimizer operation itself.
    """
    from .combat_objectives import combat_image_state
    from .object_losses import foreground_contrastive_loss

    values = _payload_tensors(payload)
    backend = model.vjepa_backend
    offsets, accepted, valid_offsets = _select_offsets(values)
    selected_offsets = offsets.detach().cpu().tolist()
    batch = len(offsets)
    metrics = {'combat_image_selected_offsets': selected_offsets,
        'combat_image_selected_samples': int(accepted.sum().item()),
        'combat_image_skipped_samples': int((~accepted).sum().item()),
        'combat_image_skipped_respawn_offsets': int((~valid_offsets).sum().item()),
        'combat_image_source_labels_selection_only': True,
        'combat_image_future_feedback': False}
    if not bool(accepted.any()):
        zero = _combat_zero(backend)
        metrics.update(combat_image_loss=0., combat_image_rgb_l1=0., combat_image_foreground_contrast=0.,
                       combat_image_npc_mae=0., combat_image_npc_pixel_fraction=0.,
                       combat_image_selected_event_counts={name: 0 for name in ('shot', 'hit', 'kill', 'player_hit')})
        return zero, metrics
    rows = torch.arange(batch, device=offsets.device)[accepted]
    steps = offsets[accepted]
    selected = {name: value[rows, steps] for name, value in values.items() if name != 'map_features'}
    previous = selected['context_state']
    target_pose = selected['target_states']
    action = selected['actions']
    hz = float(getattr(model.config, 'simulation_hz', 30))
    if not math.isfinite(hz) or hz <= 0:
        raise ValueError('Combat image simulation frequency must be finite and positive')
    dt, endless = 1/hz, bool(getattr(model.config, 'endless', True))
    target_logits, attack_logits = backend.combat_logits(previous, selected['context_pooled'], selected['context_spatial'])
    result = structured_transition(previous, target_pose, action, target_logits, attack_logits,
                                   dt=dt, training=True, endless=endless)
    image_state = combat_image_state(result, previous, target_pose, action=action, dt=dt, endless=endless)
    base = model.render_base(image_state, selected['features'], values['map_features'][rows])
    flow_options = {}
    if getattr(model.flow_renderer, 'uses_wall_conditioning', False):
        if 'map_grid' not in payload:
            raise ValueError('Wall-conditioned combat images require the explicit source initialization map')
        flow_options['map_condition'] = payload['map_grid'].detach()[rows]
    prediction = model.flow_renderer.sample(base, image_state, checkpoint_velocity=True, **flow_options)
    target, mask = selected['target_frames'], selected['object_masks']
    weights = object_region_weights(mask, boost=4.)
    if getattr(model.config, 'game_training_v5', False):
        from .object_losses import balanced_image_l1
        rgb = balanced_image_l1(prediction, target, mask)
    else:
        rgb = weighted_image_l1(prediction, target, weights)
    contrast = foreground_contrastive_loss(prediction, target, selected['backgrounds'], mask, margin_fraction=.25)
    total = .25*rgb+.25*contrast
    with torch.no_grad():
        npc_mae = weighted_image_l1(prediction.detach(), target, mask)
        metrics.update(combat_image_loss=float(total.detach()), combat_image_rgb_l1=float(rgb.detach()),
            combat_image_foreground_contrast=float(contrast.detach()), combat_image_npc_mae=float(npc_mae),
            combat_image_npc_pixel_fraction=float(mask.mean()),
            combat_image_selected_event_counts={name: int((selected['events'][:, index] > .5).sum())
                for index, name in enumerate(('shot', 'hit', 'kill', 'player_hit'))})
    return total, metrics
