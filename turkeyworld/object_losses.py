"""Object-aware training losses; target masks/state never condition the model.

Exact visible-NPC masks are preferred. Analytic sprite rectangles are a fallback
for probes: they include corpses and clipped sprites but cannot resolve wall,
sprite-alpha, weapon or inter-object occlusion. Every weight is detached.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def _mask_tensor(mask):
    if not isinstance(mask, torch.Tensor):
        raise TypeError('Object mask must be a tensor')
    value = mask.detach().float()
    if value.ndim == 2:
        value = value[None, None]
    elif value.ndim == 3:
        value = value[:, None]
    if value.ndim != 4 or value.shape[1] != 1 or min(value.shape) < 1:
        raise ValueError('Object mask must have shape [B,1,H,W] or [B,H,W]')
    if not bool(torch.isfinite(value).all()) or bool(((value < 0) | (value > 1)).any()):
        raise ValueError('Object mask must be finite and in [0,1]')
    return value


def object_region_weights(mask, boost=4.0):
    """Detached [B,1,H,W] mean-one weights with foreground/background ratio boost.

    Binary masks give foreground multiplier4 by default, not1+4. Soft masks
    (e.g. area-downsampled pixel masks) interpolate between the two weights.
    Empty/full foreground and boost1 produce exact all-one weights.
    """
    if not isinstance(boost, (int, float)) or not math.isfinite(boost) or boost < 1:
        raise ValueError('Object boost must be finite and at least1')
    value = _mask_tensor(mask)
    weights = 1+(float(boost)-1)*value
    return weights/weights.mean(dim=(-2, -1), keepdim=True)


def npc_region_weights(state, height, width, boost=4.0):
    """Mean-one weights over approximate full projected sprite rectangles.

    Height/width describe a resized source-frame grid. Source128px integer
    sprite bounds are preserved in normalized coordinates, including clipped
    screen-edge sprites whose centers lie beyond the camera's center frustum.
    Fractional pixel coverage softens rectangle boundaries. Overlaps use max,
    so eight overlapping objects never multiply the foreground boost.
    """
    if not isinstance(height, int) or not isinstance(width, int) or min(height, width) < 1:
        raise ValueError('Region image dimensions must be positive integers')
    from .camera import project_sprite_bounds
    source = state.detach().float()
    if not bool(torch.isfinite(source).all()):
        raise ValueError('Object state must be finite')
    projection = project_sprite_bounds(source, image_size=128)
    boxes = projection['boxes']
    left, top, right, bottom = boxes.unbind(-1)
    xx = torch.arange(width, device=source.device, dtype=torch.float32)
    yy = torch.arange(height, device=source.device, dtype=torch.float32)
    horizontal = (torch.minimum(xx[None, None]+1, right[..., None]*width)
                  - torch.maximum(xx[None, None], left[..., None]*width)).clamp(0, 1)
    vertical = (torch.minimum(yy[None, None]+1, bottom[..., None]*height)
                - torch.maximum(yy[None, None], top[..., None]*height)).clamp(0, 1)
    coverage = vertical[..., :, None]*horizontal[..., None, :]
    coverage = coverage*projection['sprite_frustum'][..., None, None].to(coverage.dtype)
    union = coverage.amax(dim=1, keepdim=True)
    return object_region_weights(union, boost=boost)


def _loss_inputs(prediction, target, weights):
    if prediction.ndim != 4 or prediction.shape != target.shape or min(prediction.shape) < 1:
        raise ValueError('Prediction and target must have matching nonempty [B,C,H,W] shapes')
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError('Prediction and target must be floating point tensors')
    if prediction.device != target.device:
        raise ValueError('Prediction and target must share a device')
    dtype = torch.promote_types(prediction.dtype, target.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    prediction, target = prediction.to(dtype), target.detach().to(dtype)
    if weights is None:
        return prediction, target, None
    value = weights.detach().to(device=prediction.device, dtype=dtype)
    if value.ndim == 2:
        value = value[None, None]
    elif value.ndim == 3:
        value = value[:, None]
    if (value.ndim != 4 or value.shape[0] not in (1, prediction.shape[0])
            or value.shape[1] not in (1, prediction.shape[1]) or min(value.shape) < 1):
        raise ValueError('Weights must broadcast as [B,1,H,W] or [B,C,H,W]')
    if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
        raise ValueError('Loss weights must be finite and nonnegative')
    if value.shape[-2:] != prediction.shape[-2:]:
        # This is the RGB-mask -> continuous-latent weighting path. Area
        # interpolation retains fractional object coverage, not nearest labels.
        value = F.interpolate(value, size=prediction.shape[-2:], mode='area')
    return prediction, target, value.expand_as(prediction)


def weighted_image_l1(prediction, target, weights=None):
    """RGB/feature L1 normalized by expanded weight mass (including channels).

    Only prediction receives gradients. Exact all-one weights use the identical
    legacy mean reduction; an all-zero external weight mask has zero loss.
    """
    prediction, target, weights = _loss_inputs(prediction, target, weights)
    if weights is None or bool((weights == 1).all()):
        return F.l1_loss(prediction, target)
    denominator = weights.sum().clamp_min(torch.finfo(weights.dtype).tiny)
    return ((prediction-target).abs()*weights).sum()/denominator


def weighted_mse(prediction, target, weights=None):
    """Continuous-latent/RGB MSE with detached area-resized spatial weights."""
    prediction, target, weights = _loss_inputs(prediction, target, weights)
    if weights is None or bool((weights == 1).all()):
        return F.mse_loss(prediction, target)
    denominator = weights.sum().clamp_min(torch.finfo(weights.dtype).tiny)
    return ((prediction-target).square()*weights).sum()/denominator


def weighted_image_loss(prediction, target, weights=None):
    """Preserve the source RGB objective: L1 + .1*(horizontal+vertical L1).

    Edge weights are the average of the two adjacent pixel weights. Differences
    are formed in the prediction's original dtype, matching the legacy image
    objective under autocast. A singleton axis contributes zero edge loss.
    """
    _, _, weights = _loss_inputs(prediction, target, weights)
    target = target.detach()
    base = weighted_image_l1(prediction, target, weights)
    horizontal = vertical = base.new_zeros(())
    if prediction.shape[-1] > 1:
        edge_weights = None if weights is None else .5*(weights[..., :, 1:]+weights[..., :, :-1])
        horizontal = weighted_image_l1(prediction[..., :, 1:]-prediction[..., :, :-1],
            target[..., :, 1:]-target[..., :, :-1], edge_weights)
    if prediction.shape[-2] > 1:
        edge_weights = None if weights is None else .5*(weights[..., 1:, :]+weights[..., :-1, :])
        vertical = weighted_image_l1(prediction[..., 1:, :]-prediction[..., :-1, :],
            target[..., 1:, :]-target[..., :-1, :], edge_weights)
    return base+.1*(horizontal+vertical)


@torch.no_grad()
def object_image_diagnostics(prediction, target, mask):
    """Detached foreground/background L1 and foreground fraction for logging."""
    foreground = _mask_tensor(mask)
    foreground_error = weighted_image_l1(prediction.detach(), target.detach(), foreground)
    background_error = weighted_image_l1(prediction.detach(), target.detach(), 1-foreground)
    return {'foreground_l1': foreground_error, 'background_l1': background_error,
            'foreground_fraction': foreground.mean()}


@torch.no_grad()
def region_error_metrics(prediction, target, mask):
    """Object/background MAE diagnostics with explicit empty-region flags."""
    foreground = _mask_tensor(mask)
    metrics = object_image_diagnostics(prediction, target, foreground)
    return {'npc_rgb_mae': metrics['foreground_l1'],
            'background_rgb_mae': metrics['background_l1'],
            'npc_pixel_fraction': metrics['foreground_fraction'],
            'npc_present': (foreground.sum() > 0).detach(),
            'background_present': ((1-foreground).sum() > 0).detach()}
