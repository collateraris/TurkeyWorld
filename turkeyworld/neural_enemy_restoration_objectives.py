"""Source-only RGB/visibility losses for a learned full-frame enemy layer.

The decoder receives only NPR RGB32. Source RGB, visible masks and clean
backgrounds are detached labels here. Direct appearance supervision never
uses the predicted opacity, so hiding a poor RGB layer cannot lower its loss.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .object_losses import balanced_region_losses


PROTOCOL = 'neural_enemy_restoration_sr_v1'
FOREGROUND_PROTOCOL = 'neural_enemy_restoration_foreground_v2'
DETAIL_PROTOCOL = 'neural_enemy_restoration_detail_v3'
LOSS_WEIGHTS = dict(image_l1=1., image_ssim=.2, image_edge=.1,
    enemy_rgb_l1=1., enemy_rgb_edge=.2, enemy_rgb_ssim=.2,
    alpha_bce=.5, alpha_dice=.25, empty_alpha=.25,
    background_preservation=.25, empty_image_change=.25)
FOREGROUND_LOSS_WEIGHTS = dict(foreground_image_l1=1., foreground_image_ssim=.2,
    foreground_image_edge=.1, enemy_rgb_l1=1., enemy_rgb_edge=.2, enemy_rgb_ssim=.2,
    alpha_bce=.5, alpha_dice=.25, hard_negative_alpha_bce=.25, empty_alpha=.25,
    background_preservation=.25, hard_background_preservation=.25, empty_image_change=.25)
FOREGROUND_TOPK_PIXELS = 32
DETAIL_POSITIVE_TOPK_PIXELS = 32
DETAIL_RGB_TOPK_PIXELS = 32
DETAIL_CROP_PADDING = 2
DETAIL_CROP_SCALES = ('native', 32, 64)
DETAIL_LOSS_WEIGHTS = dict(FOREGROUND_LOSS_WEIGHTS)
for _name in ('alpha_bce', 'alpha_dice', 'hard_negative_alpha_bce', 'empty_alpha'):
    DETAIL_LOSS_WEIGHTS[_name] *= .5
for _name in ('enemy_rgb_l1', 'enemy_rgb_edge', 'enemy_rgb_ssim'):
    DETAIL_LOSS_WEIGHTS[_name] *= 4.
DETAIL_LOSS_WEIGHTS.update(positive_alpha_consistency=.025, hard_positive_alpha_bce=.025,
    hard_foreground_rgb_l1=1., rgb_multiscale_crop_l1=.4,
    rgb_multiscale_crop_edge=.4, rgb_multiscale_crop_ssim=.4)
DETAIL_SETTINGS = dict(protocol=DETAIL_PROTOCOL, version=1,
    gradient_reference_step=5308,
    gradient_reference_checkpoint_sha256='48c35544bc473d8ff1a8c5d30c7bec72038ce823cdfe917d0b4819c1ee620b69',
    fixed_group_scales=dict(alpha_mask=.5, direct_RGB=4., composite_foreground_RGB=1., preservation=1.),
    base_additional_weights=dict(positive_alpha_consistency=.05, hard_positive_alpha_bce=.05,
        hard_foreground_rgb_l1=.25, rgb_multiscale_crop_l1=.1,
        rgb_multiscale_crop_edge=.1, rgb_multiscale_crop_ssim=.1),
    positive_topk_pixels=DETAIL_POSITIVE_TOPK_PIXELS, rgb_topk_pixels=DETAIL_RGB_TOPK_PIXELS,
    negative_topk_pixels=FOREGROUND_TOPK_PIXELS, crop_padding=DETAIL_CROP_PADDING,
    crop_scales=list(DETAIL_CROP_SCALES), crop_resize='bilinear_same_transform_preserving_aspect_ratio',
    crop_reduction='equal_present_images_then_equal_scales_then_Source_foreground_area',
    edge_ssim_context='same_frozen_display_background_both_sides',
    source_background_reconstruction_gradient=False, predicted_alpha_gates_appearance_loss=False,
    fixed_weights_without_EMA=True, architecture_change=False, optimizer_reset=False,
    Source_labels_loss_only=True, GAN=False, effective_weights=dict(DETAIL_LOSS_WEIGHTS))


def loss_weights(protocol=PROTOCOL):
    """Return an independent copy of the exact selected objective weights."""
    if type(protocol) is not str or protocol not in (PROTOCOL, FOREGROUND_PROTOCOL, DETAIL_PROTOCOL):
        raise ValueError('Unknown neural enemy restoration loss protocol')
    return dict({PROTOCOL: LOSS_WEIGHTS, FOREGROUND_PROTOCOL: FOREGROUND_LOSS_WEIGHTS,
                 DETAIL_PROTOCOL: DETAIL_LOSS_WEIGHTS}[protocol])


def _inputs(output, batch):
    tensors = {}
    for key in ('image', 'background', 'enemy_rgb', 'alpha_logits', 'alpha'):
        value = output.get(key)
        if not torch.is_tensor(value) or not value.is_floating_point():
            raise ValueError('Neural enemy SR needs floating image/background/RGB/opacity outputs')
        tensors[key] = value.float()
    image = tensors['image']; size = (len(image), 3, 128, 128)
    if (image.ndim != 4 or image.shape != size or not len(image)
            or tensors['background'].shape != size or tensors['enemy_rgb'].shape != size
            or tensors['alpha_logits'].shape != (len(image), 1, 128, 128)
            or tensors['alpha'].shape != (len(image), 1, 128, 128)
            or any(value.device != image.device for value in tensors.values())
            or not all(bool(torch.isfinite(value).all()) for value in tensors.values())):
        raise ValueError('Neural enemy SR outputs must be finite aligned native128 RGB/alpha')
    if output['background'].requires_grad:
        raise ValueError('Neural enemy SR background must be frozen and detached')
    # The frozen parent's pre-composite RGB includes its signed Transformer
    # residual and can overshoot[0,1]. Preserve it numerically for composition;
    # only displayed RGB, learned enemy RGB and opacity must be bounded.
    for key in ('image', 'enemy_rgb', 'alpha'):
        if bool(((tensors[key] < 0) | (tensors[key] > 1)).any()):
            raise ValueError('Neural enemy SR RGB and alpha must lie in[0,1]')
    labels = {}
    for key in ('hr_source', 'hr_background', 'mask_hr'):
        value = batch.get(key)
        if not torch.is_tensor(value):
            raise ValueError('Neural enemy SR requires detached Source128/background/visible-mask labels')
        labels[key] = value.detach().to(device=image.device, dtype=torch.float32)
    if (labels['hr_source'].shape != size or labels['hr_background'].shape != size
            or labels['mask_hr'].shape != (len(image), 1, 128, 128)
            or not all(bool(torch.isfinite(value).all()) for value in labels.values())
            or any(bool(((value < 0) | (value > 1)).any()) for value in labels.values())):
        raise ValueError('Source128 RGB/background/visible-mask labels must match native128 in[0,1]')
    return tensors, labels


def _region(error, mask):
    """Per-image area/channel mean, without a batch-wide foreground denominator."""
    mass = mask.flatten(1).sum(-1) * error.shape[1]
    values = (error * mask).flatten(1).sum(-1) / mass.clamp_min(1e-12)
    return values, mass > 0


def _present_mean(values, present):
    return values[present].mean() if bool(present.any()) else values.sum() * 0


def _case_mean(values, visible, present=None):
    """Give visible and empty cases equal mass when both occur in this batch."""
    eligible = torch.ones_like(visible) if present is None else present
    groups = [values[mask].mean() for mask in (visible & eligible, ~visible & eligible) if bool(mask.any())]
    return torch.stack(groups).mean() if groups else values.sum() * 0


def _ssim_error_map(first, second):
    """The existing Gaussian11/sigma1.5 RGB SSIM formula, unreduced in space."""
    with torch.autocast(first.device.type, enabled=False):
        first, second = first.float(), second.detach().float()
        axis = torch.arange(11, device=first.device, dtype=torch.float32) - 5
        gaussian = (-axis.square() / (2 * 1.5 ** 2)).exp()
        gaussian = gaussian / gaussian.sum()
        window = (gaussian[:, None] * gaussian[None]).expand(3, 1, 11, 11).contiguous()
        a, b = F.conv2d(first, window, groups=3), F.conv2d(second, window, groups=3)
        variance_a = (F.conv2d(first.square(), window, groups=3) - a.square()).clamp_min(0)
        variance_b = (F.conv2d(second.square(), window, groups=3) - b.square()).clamp_min(0)
        covariance = F.conv2d(first * second, window, groups=3) - a * b
        similarity = ((2 * a * b + .01 ** 2) * (2 * covariance + .03 ** 2)
            / ((a.square() + b.square() + .01 ** 2) * (variance_a + variance_b + .03 ** 2)))
        return 1 - similarity.clamp(-1, 1)


def _edge_terms(prediction, target, mask, *, foreground_only):
    values, supports = [], []
    for axis in (-1, -2):
        first, second = [slice(None)] * 4, [slice(None)] * 4
        first[axis], second[axis] = slice(1, None), slice(None, -1)
        support = torch.maximum(mask[tuple(first)], mask[tuple(second)])
        error = (prediction.diff(dim=axis) - target.diff(dim=axis)).abs()
        if foreground_only:
            value, present = _region(error, support)
        else:
            value = balanced_region_losses(error, torch.zeros_like(error), support)['per_image']
            present = torch.ones_like(value, dtype=torch.bool)
        values.append(value); supports.append(present)
    supports = torch.stack(supports)
    per_image = (torch.stack(values) * supports).sum(0) / supports.sum(0).clamp_min(1)
    return per_image, supports.any(0)


def _legacy_neural_enemy_restoration_loss(output, batch):
    """Return a genuine RGB/opacity training loss, never an atlas objective.

    Every present image has equal foreground mass regardless of opponent size.
    Source foreground/background regions and visible/empty cases are reduced
    separately. Internal raw and weighted components are both reported.
    """
    prediction, labels = _inputs(output, batch)
    image, rgb, alpha, logits = (prediction[key] for key in ('image', 'enemy_rgb', 'alpha', 'alpha_logits'))
    # Alpha0 displays the clipped frozen pre-composite image. Comparing a
    # bounded final image with an unbounded parent would add an unattainable
    # constant preservation error at every overshooting background pixel.
    background = prediction['background'].detach().clamp(0, 1)
    target, source_background, mask = (labels[key] for key in ('hr_source', 'hr_background', 'mask_hr'))
    mass = mask.flatten(1).sum(-1); visible = mass > 0
    # Label-side context only. It preserves native wall occlusion and enemy
    # contours without allowing predicted alpha to hide appearance errors.
    appearance = rgb * mask + source_background * (1 - mask)
    rgb_l1, present = _region((rgb - target).abs(), mask)
    rgb_edge, edge_present = _edge_terms(appearance, target, mask, foreground_only=True)
    occupancy = F.avg_pool2d(mask, 11, stride=1)
    rgb_ssim, ssim_present = _region(_ssim_error_map(appearance, target), occupancy)
    image_ssim_map = _ssim_error_map(image, target)
    image_ssim = balanced_region_losses(image_ssim_map, torch.zeros_like(image_ssim_map), occupancy)['per_image']
    image_edge, _ = _edge_terms(image, target, mask, foreground_only=False)
    image_l1 = balanced_region_losses(image, target, mask)['per_image']
    bce = F.binary_cross_entropy_with_logits(logits, mask, reduction='none')
    positive, has_positive = _region(bce, mask)
    negative, has_negative = _region(bce, 1 - mask)
    alpha_bce = ((positive * has_positive + negative * has_negative)
                 / (has_positive.to(positive.dtype) + has_negative).clamp_min(1))
    flat_alpha, flat_mask = alpha.flatten(1), mask.flatten(1)
    dice = 1 - (2 * (flat_alpha * flat_mask).sum(-1) + 1e-6) / (flat_alpha.sum(-1) + mass + 1e-6)
    preserve, has_background = _region((image - background).abs(), 1 - mask)
    terms = dict(image_l1=_case_mean(image_l1, visible), image_ssim=_case_mean(image_ssim, visible),
        image_edge=_case_mean(image_edge, visible), enemy_rgb_l1=_present_mean(rgb_l1, present),
        enemy_rgb_edge=_present_mean(rgb_edge, edge_present), enemy_rgb_ssim=_present_mean(rgb_ssim, ssim_present),
        alpha_bce=_case_mean(alpha_bce, visible), alpha_dice=_present_mean(dice, visible),
        empty_alpha=_present_mean(flat_alpha.mean(-1), ~visible),
        background_preservation=_case_mean(preserve, visible, has_background),
        empty_image_change=_present_mean((image - background).abs().flatten(1).mean(-1), ~visible))
    loss = sum(LOSS_WEIGHTS[name] * value for name, value in terms.items())
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite neural enemy restoration SR objective')
    metrics = {name: float(value.detach()) for name, value in terms.items()}
    metrics.update({name + '_weighted': LOSS_WEIGHTS[name] * metrics[name] for name in terms})
    metrics.update(train_loss=float(loss.detach()), loss_protocol=PROTOCOL, objective_protocol=PROTOCOL,
        visible_images=int(visible.sum()), empty_images=int((~visible).sum()),
        foreground_pixels=float(mass.sum()), foreground_edge_images=int(edge_present.sum()),
        foreground_ssim_images=int(ssim_present.sum()), background_images=int(has_background.sum()),
        source_labels_detached=True, inference_rgb_only=True, predicted_alpha_gates_appearance_loss=False,
        full_frame_pixel_aligned_decoder=True, GAN_active=False, atlas_pixels_used=False)
    return loss, metrics


def _topk_background(error, background_support):
    """Equal per-image means of the worst32 actual Source-background pixels.

    K is capped by each image's available negative pixels. Selection uses only
    the loss error; it does not create an inference mask or consume RNG.
    """
    values, counts, available = [], [], []
    for index in range(len(error)):
        candidates = error[index, 0][background_support[index, 0]]
        count = min(FOREGROUND_TOPK_PIXELS, candidates.numel())
        values.append(candidates.topk(count, sorted=False).values.mean() if count else error[index].sum() * 0)
        counts.append(count); available.append(candidates.numel())
    values = torch.stack(values)
    present = torch.as_tensor([count > 0 for count in counts], device=error.device)
    return values, present, counts, available


def _foreground_neural_enemy_restoration_loss(output, batch):
    """Train the enemy layer on foreground and preserve the frozen background.

    Source background errors are diagnostics only. For BOTH sides of the raw
    RGB and composite edge/SSIM comparisons, non-foreground pixels are the same
    detached display-clipped frozen background. Thus Source wall colors cannot
    reward this enemy-only head for painting an opaque correction patch.
    """
    prediction, labels = _inputs(output, batch)
    image, rgb, alpha, logits = (prediction[key] for key in ('image', 'enemy_rgb', 'alpha', 'alpha_logits'))
    background = prediction['background'].detach().clamp(0, 1)
    target, mask = labels['hr_source'], labels['mask_hr']
    mass = mask.flatten(1).sum(-1); visible = mass > 0
    reference = target * mask + background * (1 - mask)
    appearance = rgb * mask + background * (1 - mask)
    composite = image * mask + background * (1 - mask)
    rgb_l1, present = _region((rgb - target).abs(), mask)
    image_l1, _ = _region((image - target).abs(), mask)
    rgb_edge, edge_present = _edge_terms(appearance, reference, mask, foreground_only=True)
    image_edge, _ = _edge_terms(composite, reference, mask, foreground_only=True)
    occupancy = F.avg_pool2d(mask, 11, stride=1)
    rgb_ssim, ssim_present = _region(_ssim_error_map(appearance, reference), occupancy)
    image_ssim, _ = _region(_ssim_error_map(composite, reference), occupancy)
    bce = F.binary_cross_entropy_with_logits(logits, mask, reduction='none')
    positive, has_positive = _region(bce, mask)
    negative, has_negative = _region(bce, 1 - mask)
    alpha_bce = ((positive * has_positive + negative * has_negative)
                 / (has_positive.to(positive.dtype) + has_negative).clamp_min(1))
    flat_alpha, flat_mask = alpha.flatten(1), mask.flatten(1)
    dice = 1 - (2 * (flat_alpha * flat_mask).sum(-1) + 1e-6) / (flat_alpha.sum(-1) + mass + 1e-6)
    background_support = mask == 0
    hard_alpha, hard_present, alpha_k, available = _topk_background(F.softplus(logits), background_support)
    change = (image - background).abs()
    preserve, has_background = _region(change, 1 - mask)
    hard_preserve, _, preserve_k, _ = _topk_background(change.mean(1, keepdim=True), background_support)
    terms = dict(foreground_image_l1=_present_mean(image_l1, present),
        foreground_image_ssim=_present_mean(image_ssim, ssim_present),
        foreground_image_edge=_present_mean(image_edge, edge_present),
        enemy_rgb_l1=_present_mean(rgb_l1, present), enemy_rgb_edge=_present_mean(rgb_edge, edge_present),
        enemy_rgb_ssim=_present_mean(rgb_ssim, ssim_present), alpha_bce=_case_mean(alpha_bce, visible),
        alpha_dice=_present_mean(dice, visible), hard_negative_alpha_bce=_case_mean(hard_alpha, visible, hard_present),
        empty_alpha=_present_mean(flat_alpha.mean(-1), ~visible),
        background_preservation=_case_mean(preserve, visible, has_background),
        hard_background_preservation=_case_mean(hard_preserve, visible, hard_present),
        empty_image_change=_present_mean(change.flatten(1).mean(-1), ~visible))
    loss = sum(FOREGROUND_LOSS_WEIGHTS[name] * value for name, value in terms.items())
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite foreground neural enemy restoration objective')
    # Full Source-image readings remain honest diagnostics; they contribute no
    # gradients or scalar terms to the enemy-only foreground objective.
    with torch.no_grad():
        source_image_l1 = (image.detach() - target).abs().mean()
        source_image_ssim = _ssim_error_map(image.detach(), target).mean()
    metrics = {name: float(value.detach()) for name, value in terms.items()}
    metrics.update({name + '_weighted': FOREGROUND_LOSS_WEIGHTS[name] * metrics[name] for name in terms})
    metrics.update(train_loss=float(loss.detach()), loss_protocol=FOREGROUND_PROTOCOL,
        objective_protocol=FOREGROUND_PROTOCOL, visible_images=int(visible.sum()), empty_images=int((~visible).sum()),
        foreground_pixels=float(mass.sum()), foreground_edge_images=int(edge_present.sum()),
        foreground_ssim_images=int(ssim_present.sum()), background_images=int(has_background.sum()),
        hard_negative_topk_pixels=FOREGROUND_TOPK_PIXELS, hard_negative_topk_per_image=alpha_k,
        hard_background_topk_per_image=preserve_k, hard_negative_available_pixels_per_image=available,
        hard_negative_visible_images=int((hard_present & visible).sum()),
        hard_negative_empty_images=int((hard_present & ~visible).sum()),
        hard_negative_support='Source_visible_mask_exactly_zero',
        source_image_l1_diagnostic=float(source_image_l1), source_image_ssim_diagnostic=float(source_image_ssim),
        source_background_reconstruction_gradient=False, edge_ssim_background_context='same_frozen_display_background_both_sides',
        source_labels_detached=True, inference_rgb_only=True, predicted_alpha_gates_appearance_loss=False,
        full_frame_pixel_aligned_decoder=True, GAN_active=False, atlas_pixels_used=False)
    return loss, metrics


def _topk_foreground(error, support, limit):
    values, counts = [], []
    for index in range(len(error)):
        candidates = error[index, 0][support[index, 0]]
        count = min(limit, candidates.numel())
        values.append(candidates.topk(count, sorted=False).values.mean() if count else error[index].sum() * 0)
        counts.append(count)
    return torch.stack(values), torch.as_tensor([count > 0 for count in counts], device=error.device), counts


def _detail_ssim_error_map(first, second):
    """Gaussian RGB SSIM on a native crop, including a one-pixel opponent."""
    with torch.autocast(first.device.type, enabled=False):
        first, second = first.float(), second.detach().float()
        size = min(11, *first.shape[-2:]); size -= 1 - size % 2
        axis = torch.arange(size, device=first.device, dtype=torch.float32) - (size - 1) / 2
        gaussian = (-axis.square() / (2 * 1.5 ** 2)).exp(); gaussian = gaussian / gaussian.sum()
        window = (gaussian[:, None] * gaussian[None]).expand(3, 1, size, size).contiguous()
        a, b = F.conv2d(first, window, groups=3), F.conv2d(second, window, groups=3)
        va = (F.conv2d(first.square(), window, groups=3) - a.square()).clamp_min(0)
        vb = (F.conv2d(second.square(), window, groups=3) - b.square()).clamp_min(0)
        covariance = F.conv2d(first * second, window, groups=3) - a * b
        similarity = ((2 * a * b + .01 ** 2) * (2 * covariance + .03 ** 2)
            / ((a.square() + b.square() + .01 ** 2) * (va + vb + .03 ** 2)))
        return 1 - similarity.clamp(-1, 1), size


def _rgb_detail_crops(appearance, reference, mask):
    """Loss-only Source boxes; native and aspect-preserving32/64 crop scales.

    Crop transforms are applied identically to the raw RGB prediction, target
    foreground and frozen context. The network never receives these boxes.
    Each nonempty image, scale and Source foreground area is normalized apart.
    """
    results = dict(l1=[], edge=[], ssim=[]); shapes = []
    height, width = appearance.shape[-2:]
    for index in range(len(appearance)):
        pixels = torch.nonzero(mask[index, 0] > 0, as_tuple=False)
        if not len(pixels): shapes.append([]); continue
        low, high = pixels.amin(0), pixels.amax(0) + 1
        y0, x0 = max(0, int(low[0]) - DETAIL_CROP_PADDING), max(0, int(low[1]) - DETAIL_CROP_PADDING)
        y1, x1 = min(height, int(high[0]) + DETAIL_CROP_PADDING), min(width, int(high[1]) + DETAIL_CROP_PADDING)
        first = appearance[index:index+1, :, y0:y1, x0:x1]
        second = reference[index:index+1, :, y0:y1, x0:x1]
        source_mask = mask[index:index+1, :, y0:y1, x0:x1]
        local = dict(l1=[], edge=[], ssim=[]); image_shapes = []
        for scale in DETAIL_CROP_SCALES:
            if scale == 'native':
                a, b, occupancy = first, second, source_mask
            else:
                longest = max(first.shape[-2:])
                target_size = tuple(max(1, round(value * scale / longest)) for value in first.shape[-2:])
                a = F.interpolate(first, target_size, mode='bilinear', align_corners=False)
                b = F.interpolate(second, target_size, mode='bilinear', align_corners=False)
                occupancy = (F.interpolate(source_mask, target_size, mode='area') if scale <= longest else
                             F.interpolate(source_mask, target_size, mode='bilinear', align_corners=False))
            image_shapes.append(list(a.shape[-2:]))
            l1, present = _region((a - b).abs(), occupancy)
            if not bool(present.any()): continue
            edge, edge_present = _edge_terms(a, b, occupancy, foreground_only=True)
            error, window = _detail_ssim_error_map(a, b)
            ssim, ssim_present = _region(error, F.avg_pool2d(occupancy, window, stride=1))
            local['l1'].append(_present_mean(l1, present))
            local['edge'].append(_present_mean(edge, edge_present))
            local['ssim'].append(_present_mean(ssim, ssim_present))
        shapes.append(image_shapes)
        for name in results:
            if local[name]: results[name].append(torch.stack(local[name]).mean())
    zero = appearance.sum() * 0
    return {name: torch.stack(values).mean() if values else zero for name, values in results.items()}, shapes


def _detail_neural_enemy_restoration_loss(output, batch):
    prediction, labels = _inputs(output, batch)
    image, rgb, alpha, logits = (prediction[key] for key in ('image', 'enemy_rgb', 'alpha', 'alpha_logits'))
    background = prediction['background'].detach().clamp(0, 1)
    target, mask = labels['hr_source'], labels['mask_hr']
    mass = mask.flatten(1).sum(-1); visible = mass > 0
    reference = target * mask + background * (1 - mask)
    appearance = rgb * mask + background * (1 - mask)
    composite = image * mask + background * (1 - mask)
    rgb_l1, present = _region((rgb - target).abs(), mask)
    image_l1, _ = _region((image - target).abs(), mask)
    rgb_edge, edge_present = _edge_terms(appearance, reference, mask, foreground_only=True)
    image_edge, _ = _edge_terms(composite, reference, mask, foreground_only=True)
    occupancy = F.avg_pool2d(mask, 11, stride=1)
    rgb_ssim, ssim_present = _region(_ssim_error_map(appearance, reference), occupancy)
    image_ssim, _ = _region(_ssim_error_map(composite, reference), occupancy)
    bce = F.binary_cross_entropy_with_logits(logits, mask, reduction='none')
    positive, has_positive = _region(bce, mask)
    negative, has_negative = _region(bce, 1 - mask)
    alpha_bce = ((positive * has_positive + negative * has_negative)
                 / (has_positive.to(positive.dtype) + has_negative).clamp_min(1))
    flat_alpha, flat_mask = alpha.flatten(1), mask.flatten(1)
    dice = 1 - (2 * (flat_alpha * flat_mask).sum(-1) + 1e-6) / (flat_alpha.sum(-1) + mass + 1e-6)
    hard_alpha, hard_present, negative_k, available = _topk_background(F.softplus(logits), mask == 0)
    change = (image - background).abs(); preserve, has_background = _region(change, 1 - mask)
    hard_preserve, _, preserve_k, _ = _topk_background(change.mean(1, keepdim=True), mask == 0)
    positive_consistency, _ = _region(1 - alpha, mask)
    hard_positive, positive_present, positive_k = _topk_foreground(
        F.softplus(-logits), mask > 0, DETAIL_POSITIVE_TOPK_PIXELS)
    hard_rgb, rgb_present, rgb_k = _topk_foreground(
        (rgb - target).abs().mean(1, keepdim=True), mask > 0, DETAIL_RGB_TOPK_PIXELS)
    crops, crop_shapes = _rgb_detail_crops(appearance, reference, mask)
    terms = dict(foreground_image_l1=_present_mean(image_l1, present),
        foreground_image_ssim=_present_mean(image_ssim, ssim_present),
        foreground_image_edge=_present_mean(image_edge, edge_present),
        enemy_rgb_l1=_present_mean(rgb_l1, present), enemy_rgb_edge=_present_mean(rgb_edge, edge_present),
        enemy_rgb_ssim=_present_mean(rgb_ssim, ssim_present), alpha_bce=_case_mean(alpha_bce, visible),
        alpha_dice=_present_mean(dice, visible), hard_negative_alpha_bce=_case_mean(hard_alpha, visible, hard_present),
        empty_alpha=_present_mean(flat_alpha.mean(-1), ~visible),
        background_preservation=_case_mean(preserve, visible, has_background),
        hard_background_preservation=_case_mean(hard_preserve, visible, hard_present),
        empty_image_change=_present_mean(change.flatten(1).mean(-1), ~visible),
        positive_alpha_consistency=_present_mean(positive_consistency, present),
        hard_positive_alpha_bce=_present_mean(hard_positive, positive_present),
        hard_foreground_rgb_l1=_present_mean(hard_rgb, rgb_present),
        rgb_multiscale_crop_l1=crops['l1'], rgb_multiscale_crop_edge=crops['edge'],
        rgb_multiscale_crop_ssim=crops['ssim'])
    loss = sum(DETAIL_LOSS_WEIGHTS[name] * value for name, value in terms.items())
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite detail neural enemy restoration objective')
    with torch.no_grad():
        source_image_l1 = (image.detach() - target).abs().mean()
        source_image_ssim = _ssim_error_map(image.detach(), target).mean()
    metrics = {name: float(value.detach()) for name, value in terms.items()}
    metrics.update({name + '_weighted': DETAIL_LOSS_WEIGHTS[name] * metrics[name] for name in terms})
    metrics.update(train_loss=float(loss.detach()), loss_protocol=DETAIL_PROTOCOL, objective_protocol=DETAIL_PROTOCOL,
        visible_images=int(visible.sum()), empty_images=int((~visible).sum()), foreground_pixels=float(mass.sum()),
        foreground_edge_images=int(edge_present.sum()), foreground_ssim_images=int(ssim_present.sum()),
        background_images=int(has_background.sum()), hard_negative_topk_pixels=FOREGROUND_TOPK_PIXELS,
        hard_negative_topk_per_image=negative_k, hard_background_topk_per_image=preserve_k,
        hard_negative_available_pixels_per_image=available,
        hard_positive_topk_pixels=DETAIL_POSITIVE_TOPK_PIXELS, hard_positive_topk_per_image=positive_k,
        hard_foreground_rgb_topk_pixels=DETAIL_RGB_TOPK_PIXELS, hard_foreground_rgb_topk_per_image=rgb_k,
        detail_crop_scales=list(DETAIL_CROP_SCALES), detail_crop_padding=DETAIL_CROP_PADDING,
        detail_crop_shapes_per_image=crop_shapes,
        fixed_gradient_group_scales=dict(DETAIL_SETTINGS['fixed_group_scales']),
        source_image_l1_diagnostic=float(source_image_l1), source_image_ssim_diagnostic=float(source_image_ssim),
        source_background_reconstruction_gradient=False,
        edge_ssim_background_context='same_frozen_display_background_both_sides',
        source_labels_detached=True, inference_rgb_only=True, predicted_alpha_gates_appearance_loss=False,
        full_frame_pixel_aligned_decoder=True, GAN_active=False, atlas_pixels_used=False)
    return loss, metrics


def neural_enemy_restoration_loss(output, batch, *, protocol=PROTOCOL):
    """Select a pinned objective; the default reproduces the original V1 loss."""
    loss_weights(protocol)  # Fail before doing model-output work on an unknown revision.
    if protocol == PROTOCOL:
        return _legacy_neural_enemy_restoration_loss(output, batch)
    if protocol == FOREGROUND_PROTOCOL:
        return _foreground_neural_enemy_restoration_loss(output, batch)
    return _detail_neural_enemy_restoration_loss(output, batch)
