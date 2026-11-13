"""Coordinate-consistent, gain-independent supervision of learned enemy RGBA.

Only Source labels are sampled here. Neither crops nor masks enter inference.
The displayed restoration curriculum and the learned legacy gains cannot turn
off these losses. RGB is supervised only on actual visible enemy pixels; alpha
also learns empty, occluded and incorrectly localized proposals.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .enemy_decoder_sr import sample_box, warp_box
from .enemy_decoder_objectives import _mean
from .visual_first_objectives import visual_first_loss, LOSS_WEIGHTS as PREVIOUS_WEIGHTS
from .object_losses import decoded_ssim_loss, balanced_edge_l1

PROTOCOL = 'decoder_restoration_sr_v3'
REPLACED_CANONICAL = ('canonical_rgb', 'canonical_alpha',
                      'vision_canonical_rgb', 'vision_canonical_alpha')
RESTORATION_WEIGHTS = {
    'restoration_rgb': .5, 'restoration_alpha_bce': .2,
    'restoration_alpha_dice': .1, 'restoration_rgb_ssim': .1,
    'restoration_rgb_edge': .1, 'restoration_alpha_edge': .1,
    'restoration_hr_mask_bce': .2, 'restoration_hr_mask_dice': .1,
    'restoration_raw_composite': .5, 'restoration_raw_composite_edge': .1,
}
LOSS_WEIGHTS = {**{k: v for k, v in PREVIOUS_WEIGHTS.items() if k not in REPLACED_CANONICAL},
                **RESTORATION_WEIGHTS}


def restoration_crop_targets(batch, boxes, size=32):
    """Detached Source RGB/visible alpha in the *actual warp box* coordinates.

    Occlusion and offscreen clipping remain in their original positions. A
    tight visible Source bbox must never be stretched across a larger predicted
    box. Outside the visible alpha, RGB has no canonical-color supervision.
    """
    box = boxes.detach().float()
    source, mask = (batch[key].detach().float() for key in ('hr_source', 'mask_hr'))
    if (source.shape != (len(box), 3, 128, 128) or mask.shape != (len(box), 1, 128, 128)
            or box.shape != (len(source), 4) or not bool(torch.isfinite(box).all())
            or bool(((box < 0) | (box > 1)).any())):
        raise ValueError('Restoration labels require Source128 RGB/mask and normalized actual xyxy boxes')
    # Area labels retain thin distant silhouettes that a single canonical pixel
    # center could miss. Premultiplied pooling keeps wall/sky colors out of the
    # RGB target of a partly covered foreground pixel. Both use the same box.
    resolution = max(128, size)
    rgb_hi = sample_box(source * mask, box, resolution)
    alpha_hi = sample_box(mask, box, resolution).clamp(0, 1)
    alpha = F.adaptive_avg_pool2d(alpha_hi, (size, size))
    premultiplied = F.adaptive_avg_pool2d(rgb_hi, (size, size))
    color = torch.where(alpha > 0, premultiplied / alpha.clamp_min(1e-8),
                        torch.zeros_like(premultiplied))
    return color, alpha


def _class_mean(values, foreground, selected):
    """Equal present positive/negative image mass after per-area normalization."""
    foreground, selected = foreground.detach().bool(), selected.detach().bool()
    positive, negative = selected & foreground, selected & ~foreground
    return (_mean(values, positive) + _mean(values, negative)) / (
        positive.any().float() + negative.any().float()).clamp_min(1)


def _region_per_image(error, mask, support=None):
    """Average present FG/BG region means, including color channel mass."""
    if support is None:
        support = torch.ones_like(mask)
    foreground = (mask * support).expand_as(error)
    background = ((1 - mask) * support).expand_as(error)
    fg_mass, bg_mass = foreground.flatten(1).sum(-1), background.flatten(1).sum(-1)
    fg = (error * foreground).flatten(1).sum(-1) / fg_mass.clamp_min(1e-8)
    bg = (error * background).flatten(1).sum(-1) / bg_mass.clamp_min(1e-8)
    return (fg + bg) / ((fg_mass > 0).float() + (bg_mass > 0).float()).clamp_min(1)


def _alpha_terms(probability, target, selected, *, support=None):
    probability = probability.float().clamp(1e-6, 1-1e-6)
    target = target.detach().float()
    support = torch.ones_like(target) if support is None else support.detach()
    error = F.binary_cross_entropy(probability, target, reduction='none')
    bce = _region_per_image(error, target, support)
    present = (target * support).flatten(1).sum(-1) > 0
    pred, truth = probability * support, target * support
    dice = 1 - (2 * (pred * target).flatten(1).sum(-1) + 1e-6) / (
        pred.flatten(1).sum(-1) + truth.flatten(1).sum(-1) + 1e-6)
    return _class_mean(bce, present, selected), _class_mean(dice, present, selected)


def _edge_per_image(prediction, target, mask=None, support=None):
    terms = []
    for dimension in (-1, -2):
        first, last = [slice(None)] * 4, [slice(None)] * 4
        first[dimension], last[dimension] = slice(1, None), slice(None, -1)
        first, last = tuple(first), tuple(last)
        error = ((prediction[first] - prediction[last]) - (target[first] - target[last])).abs()
        edge_mask = (torch.ones_like(error[:, :1]) if mask is None else
                     .5 * (mask[first] + mask[last]))
        edge_support = None if support is None else torch.maximum(support[first], support[last])
        terms.append(_region_per_image(error, edge_mask, edge_support))
    return sum(terms) / len(terms)


def _routes(output, batch):
    if output.get('restoration') is not True:
        raise ValueError('Restoration objective requires its explicit model profile')
    count = len(output['image'])
    for name, rgba_key, box_key, chosen in (
        ('visual', 'visual_rgba', 'visual_bbox', torch.ones(count, device=output['image'].device, dtype=torch.bool)),
        ('guided', 'enemy_rgba', 'enemy_bbox', output['enemy_applicable'].detach().reshape(-1).bool()),
    ):
        rgba, box = output[rgba_key].float(), output[box_key].detach().float()
        if (rgba.shape != (count, 4, 32, 32) or box.shape != (count, 4)
                or not bool(torch.isfinite(rgba).all()) or bool(((rgba < 0) | (rgba > 1)).any())):
            raise ValueError('Raw restoration routes require finite RGBA32 and actual bbox[B,4]')
        valid = ((box[:, 2:] - box[:, :2]) > 1e-5).all(-1)
        yield name, rgba, box, chosen & valid


def _route_terms(rgba, box, selected, batch):
    source_crop, alpha_crop = restoration_crop_targets(batch, box, rgba.shape[-1])
    present = alpha_crop.flatten(1).sum(-1) > 0
    positive = selected & present
    mass = alpha_crop.flatten(1).sum(-1)
    rgb = ((rgba[:, :3] - source_crop).abs() * alpha_crop).flatten(1).sum(-1) / (3 * mass).clamp_min(1e-8)
    alpha_bce, alpha_dice = _alpha_terms(rgba[:, 3:4], alpha_crop, selected)
    # Masked color targets cannot reward painting sky/walls into transparent
    # pixels. The target alpha is detached, so RGB detail cannot hide in alpha.
    zero = rgba.sum() * 0
    if bool(positive.any()):
        first = rgba[positive, :3] * alpha_crop[positive]
        second = source_crop[positive] * alpha_crop[positive]
        ssim = decoded_ssim_loss(first, second, alpha_crop[positive])
        edge = balanced_edge_l1(first, second, alpha_crop[positive])
    else:
        ssim = edge = zero
    alpha_edge = _class_mean(_edge_per_image(rgba[:, 3:4], alpha_crop), present, selected)
    # Identical detached boxes are used for label crops and raw-layer warps.
    # Localization has its own bbox losses and cannot shrink a box to game a
    # differentiable label sampling window.
    warped = warp_box(rgba, box)
    support = warp_box(torch.ones_like(rgba[:, 3:4]), box).detach().clamp(0, 1)
    mask = batch['mask_hr'].detach().float()
    hr_bce, hr_dice = _alpha_terms(warped[:, 3:4], mask, selected, support=support)
    background = batch['hr_background'].detach().float()
    source = batch['hr_source'].detach().float()
    raw_composite = background + warped[:, 3:4] * (warped[:, :3] - background)
    per_image = _region_per_image((raw_composite - source).abs(), mask, support)
    per_edge = _edge_per_image(raw_composite, source, mask, support)
    return {
        'restoration_rgb': _mean(rgb, positive),
        'restoration_alpha_bce': alpha_bce, 'restoration_alpha_dice': alpha_dice,
        'restoration_rgb_ssim': ssim, 'restoration_rgb_edge': edge,
        'restoration_alpha_edge': alpha_edge,
        'restoration_hr_mask_bce': hr_bce, 'restoration_hr_mask_dice': hr_dice,
        'restoration_raw_composite': _class_mean(per_image, present, selected),
        'restoration_raw_composite_edge': _class_mean(per_edge, present, selected),
    }, {'frames': int(selected.sum()), 'positive_frames': int(positive.sum()),
        'negative_frames': int((selected & ~present).sum())}


def decoder_restoration_loss(output, batch):
    """Supervise RGB/alpha before gains, display curriculum and route gates.

    Reuse the unchanged v2 detection/image objective. Its four obsolete tight
    canonical terms are constant-only in this call and removed exactly from
    the total, then replaced by actual-warp-coordinate objectives. They receive
    no gradients; legacy profiles and their saved objective remain unchanged.
    """
    legacy = dict(output, enemy_rgba=output['enemy_rgba'].detach(),
                  visual_rgba=output['visual_rgba'].detach())
    total, metrics = visual_first_loss(legacy, batch)
    old = sum(metrics[name + '_weighted'] for name in REPLACED_CANONICAL)
    total = total - total.new_tensor(old)
    for name in REPLACED_CANONICAL:
        del metrics[name]
        del metrics[name + '_weighted']
    terms, counts, active = {}, {}, 0
    for name, rgba, box, selected in _routes(output, batch):
        values, receipt = _route_terms(rgba, box, selected, batch)
        counts[name] = receipt
        if bool(selected.any()):
            active += 1
            for key, value in values.items():
                terms[key] = terms.get(key, total * 0) + value
    for name, weight in RESTORATION_WEIGHTS.items():
        value = terms.get(name, total * 0) / max(active, 1)
        total = total + weight * value
        metrics[name] = float(value.detach())
        metrics[name + '_weighted'] = weight * metrics[name]
    if not bool(torch.isfinite(total)):
        raise FloatingPointError('Nonfinite coordinate-consistent decoder restoration objective')
    metrics.update(train_loss=float(total.detach()), loss_protocol=PROTOCOL, objective_protocol=PROTOCOL,
        restoration_routes=counts, restoration_crop_coordinates='actual_predicted_bbox_detached',
        restoration_rgb_target='actual_Source_visible_foreground_only',
        restoration_raw_losses_before_gain=True, restoration_legacy_canonical_terms_removed=True,
        restoration_blend_scale=float(output['restoration_blend_scale']),
        restoration_vision_requires_hint=False, source_labels_detached=True,
        source_future_states_used_as_input=False)
    return total, metrics


@torch.no_grad()
def restoration_raw_metrics(output, batch):
    """Full-HR raw-layer diagnostics, with no learned/display gain or mask gate.

    Choose the hard guided route where active, otherwise the visual decoder,
    including uncertain/no-hint rows. Missed foreground remains in the metric
    denominator. Source background is a loss-only counterfactual canvas, never
    an inference input. These are raw-decoder metrics, not final image quality.
    """
    routes = {name: (rgba, box) for name, rgba, box, _ in _routes(output, batch)}
    guided = output['guided_route'].detach().reshape(-1).bool()
    rgba = torch.where(guided[:, None, None, None], routes['guided'][0], routes['visual'][0])
    box = torch.where(guided[:, None], routes['guided'][1], routes['visual'][1])
    warped = warp_box(rgba, box)
    source, background, mask = (batch[key].detach().float() for key in ('hr_source', 'hr_background', 'mask_hr'))
    composite = background + warped[:, 3:4] * (warped[:, :3] - background)
    mass = mask.flatten(1).sum(-1)
    visible = mass > 0
    fg_error = ((composite - source).abs() * mask).flatten(1).sum(-1) / (3 * mass).clamp_min(1e-8)
    edge_errors = []
    for dimension in (-1, -2):
        a, b = [slice(None)] * 4, [slice(None)] * 4
        a[dimension], b[dimension] = slice(1, None), slice(None, -1)
        a, b = tuple(a), tuple(b)
        edge_mask = torch.maximum(mask[a], mask[b])
        error = ((composite[a] - composite[b]) - (source[a] - source[b])).abs()
        edge_errors.append((error * edge_mask).flatten(1).sum(-1)
                           / (3 * edge_mask.flatten(1).sum(-1)).clamp_min(1e-8))
    bg = 1 - mask
    bg_error = ((composite - background).abs() * bg).flatten(1).sum(-1) / (
        3 * bg.flatten(1).sum(-1)).clamp_min(1e-8)
    empty_error = (composite - background).abs().flatten(1).mean(-1)
    support = warp_box(torch.ones_like(rgba[:, 3:4]), box)
    misses = visible & ((support * mask).flatten(1).sum(-1) == 0)
    return {'frames': len(mask), 'visible_frames': int(visible.sum()), 'empty_frames': int((~visible).sum()),
        'raw_foreground_mae': float(_mean(fg_error, visible)),
        'raw_foreground_edge_mae': float(_mean(sum(edge_errors) / len(edge_errors), visible)),
        'raw_empty_mae': float(_mean(empty_error, ~visible)),
        'raw_background_mae': float(bg_error.mean()),
        'raw_visual_frames': int((~guided).sum()), 'raw_guided_frames': int(guided.sum()),
        'raw_box_miss_visible_frames': int(misses.sum()),
        'metric_counts': {'raw_foreground_mae': int(visible.sum()),
            'raw_foreground_edge_mae': int(visible.sum()), 'raw_empty_mae': int((~visible).sum()),
            'raw_background_mae': len(mask)},
        'scope': 'hard_guided_else_visual_including_uncertain_full_Source_visible_denominator',
        'before_gain_and_route_or_visible_mask_gates': True,
        'Source_background_is_labels_only': True, 'source_future_as_input': False}
