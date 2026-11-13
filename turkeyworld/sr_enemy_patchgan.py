"""Conditional enemy-ROI PatchGAN over raw, gain-independent learned RGBA.

Real and fake share exactly the same detached RGB32 input crop and canvas.
Source RGB/visible alpha are discriminator labels, never generator conditions.
The existing reconstruction, alpha, geometry and edge objectives remain owned
by the SR trainer; this module supplies a separately optimized adversary only.
"""
from __future__ import annotations

from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from .enemy_decoder_sr import sample_box
from .decoder_restoration_objectives import restoration_crop_targets

PROTOCOL = 'conditional_raw_enemy_roi_patchgan_v1'


class EnemyROIPatchDiscriminator(nn.Module):
    """206,889-parameter spectral-normalized conditional RGB+alpha PatchGAN."""
    def __init__(self, channels=(24, 48, 96, 128), roi_size=64):
        super().__init__()
        if (not isinstance(channels, (tuple, list)) or len(channels) != 4
                or any(type(width) is not int or width < 4 for width in channels)
                or type(roi_size) is not int or roi_size not in (32, 64)):
            raise ValueError('Enemy PatchGAN needs four explicit channel widths and ROI32 or64')
        self.config = dict(architecture=PROTOCOL, channels=list(channels), roi_size=roi_size,
            condition_channels=3, candidate_channels=4, spectral_normalization=True,
            adversarial_loss='hinge', separate_optimizer=True)
        incoming, blocks = 7, []
        for index, width in enumerate(channels):
            blocks.append(nn.Sequential(spectral_norm(nn.Conv2d(incoming, width,
                4 if index < 3 else 3, stride=2 if index < 3 else 1, padding=1)),
                nn.LeakyReLU(.2, inplace=False)))
            incoming = width
        self.blocks = nn.ModuleList(blocks)
        self.logits = spectral_norm(nn.Conv2d(incoming, 1, 3, padding=1))

    def forward(self, condition, candidate, *, return_features=False):
        size = self.config['roi_size']
        if (condition.shape != (len(candidate), 3, size, size)
                or candidate.shape != (len(condition), 4, size, size)
                or condition.device != candidate.device or not condition.is_floating_point()
                or not candidate.is_floating_point()
                or not bool(torch.isfinite(condition).all()) or not bool(torch.isfinite(candidate).all())):
            raise ValueError('Enemy PatchGAN expects aligned finite RGB condition and RGB+alpha candidate ROIs')
        value = torch.cat((condition, candidate), 1)
        features = []
        for block in self.blocks:
            value = block(value)
            features.append(value)
        logits = self.logits(value)
        return (logits, features) if return_features else logits


def make_enemy_patch_optimizer(discriminator, *, lr=1e-4, betas=(0., .99)):
    """An explicitly separate Adam; no SR parameter can enter this optimizer."""
    if (not isinstance(discriminator, EnemyROIPatchDiscriminator)
            or type(lr) not in (int, float) or not math.isfinite(lr) or lr <= 0):
        raise ValueError('Enemy PatchGAN requires its own finite positive learning rate')
    return torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas, weight_decay=0.)


def build_enemy_patch_pairs(output, batch, low_rgb=None, *, roi_size=64):
    """Query both valid raw routes at their own *unchanged predicted* boxes.

    Candidate RGB is composited over the shared input crop, before learned gain,
    gradual blend, route confidence or visible-mask gates. The optional low_rgb
    must be the actual SR input after domain mixing; default is cached NPR32.
    No Source box, pose, future RGB or mask is ever used to generate a fake.
    """
    low = batch['lr_npr'] if low_rgb is None else low_rgb
    if (type(roi_size) is not int or roi_size not in (32, 64)
            or low.ndim != 4 or low.shape[1:] != (3, 32, 32) or len(low) < 1
            or not low.is_floating_point() or not bool(torch.isfinite(low).all())
            or bool(((low < 0) | (low > 1)).any())):
        raise ValueError('Enemy PatchGAN conditions require actual finite SR input RGB32 in[0,1]')
    low = low.detach().float()
    count, device = len(low), low.device
    source_mask = batch['mask_hr'].detach().to(device=device, dtype=torch.float32)
    source = batch['hr_source'].detach().to(device=device, dtype=torch.float32)
    state = batch['state'].detach().to(device=device, dtype=torch.float32)
    if (source_mask.shape != (count, 1, 128, 128) or source.shape != (count, 3, 128, 128)
            or state.shape != (count, 20)
            or not all(bool(torch.isfinite(value).all()) for value in (source_mask, source, state))
            or bool(((source_mask < 0) | (source_mask > 1)).any())):
        raise ValueError('PatchGAN labels require aligned Source128 RGB/visible alpha and state20')
    visible = source_mask.flatten(1).sum(-1) > 0
    distance = torch.linalg.vector_norm((state[:, 8:10] - state[:, :2]) * 16, dim=-1)
    bands = torch.where(distance < 2, 0, torch.where(distance < 5, 1, 2)).long()
    fields = dict(condition=[], real=[], fake=[], boxes=[], source_index=[], roi_visible=[])
    route_names, route_counts, invalid_counts = [], {}, {}
    for route, rgba_key, box_key, active in (
        ('visual', 'visual_rgba', 'visual_bbox', torch.ones(count, device=device, dtype=torch.bool)),
        ('guided', 'enemy_rgba', 'enemy_bbox', output['enemy_applicable'].detach().reshape(-1).bool()),
    ):
        rgba, boxes = output[rgba_key].float(), output[box_key].detach().float()
        if (rgba.shape != (count, 4, 32, 32) or boxes.shape != (count, 4)
                or rgba.device != device or boxes.device != device or active.shape != (count,)
                or not bool(torch.isfinite(rgba).all()) or not bool(torch.isfinite(boxes).all())
                or bool(((rgba < 0) | (rgba > 1)).any())
                or bool(((boxes < 0) | (boxes > 1)).any())):
            raise ValueError('PatchGAN raw routes need finite RGBA32 and actual normalized bbox[B,4]')
        valid = ((boxes[:, 2:] - boxes[:, :2]) > 1e-5).all(-1)
        selected = active & valid
        ids = selected.nonzero(as_tuple=False).flatten()
        route_counts[route] = int(len(ids)); invalid_counts[route] = int((active & ~valid).sum())
        # Query first, then select: this also keeps an empty differentiable fake
        # tensor tied to the generator when every proposed box is invalid.
        condition = sample_box(low, boxes, roi_size).detach()
        real_rgb, real_alpha = restoration_crop_targets(
            {'hr_source': source, 'mask_hr': source_mask}, boxes, roi_size)
        real = torch.cat((condition * (1-real_alpha) + real_rgb * real_alpha, real_alpha), 1).detach()
        fake_alpha = F.interpolate(rgba[:, 3:4], size=roi_size, mode='bilinear', align_corners=False)
        fake_premultiplied = F.interpolate(rgba[:, :3] * rgba[:, 3:4], size=roi_size,
                                           mode='bilinear', align_corners=False)
        fake = torch.cat((condition * (1-fake_alpha) + fake_premultiplied, fake_alpha), 1)
        for name, value in (('condition', condition), ('real', real), ('fake', fake), ('boxes', boxes)):
            fields[name].append(value[ids])
        fields['source_index'].append(ids)
        fields['roi_visible'].append(real_alpha[ids].flatten(1).sum(-1) > 0)
        route_names.extend([route] * len(ids))
    pairs = {name: torch.cat(values, 0) for name, values in fields.items()}
    pairs.update(source_count=count, source_visible=visible, distance_band=bands, routes=route_names)
    selected = torch.bincount(pairs['source_index'], minlength=count) > 0
    labels = visible[pairs['source_index']]
    by_distance = {name:int((selected & visible & (bands == index)).sum())
                   for index, name in enumerate(('near', 'mid', 'far'))}
    native = batch.get('mask_lr')
    native_area = (native.detach().float().flatten(1).sum(-1) if native is not None else
                   torch.full((count,), -1., device=device))
    pairs['receipt'] = dict(protocol=PROTOCOL, source_frames=count,
        valid_source_frames=int(selected.sum()), pairs=len(route_names), route_pairs=route_counts,
        invalid_route_boxes=invalid_counts, source_visible_frames=int((selected & visible).sum()),
        source_empty_frames=int((selected & ~visible).sum()), visible_distance_frames=by_distance,
        roi_visible_pairs=int(pairs['roi_visible'].sum()),
        missed_visible_box_pairs=int((labels & ~pairs['roi_visible']).sum()),
        tiny_native_1_4_visible_frames=int((selected & visible & (native_area > 0) & (native_area <= 4)).sum()),
        small_native_5_16_visible_frames=int((selected & visible & (native_area > 4) & (native_area <= 16)).sum()),
        subpixel_native_visible_frames=int((selected & visible & (native_area == 0)).sum()),
        roi_size=roi_size, crop_coordinates='actual_predicted_bbox_detached_no_GT_box_rescue',
        condition='same_actual_SR_input_RGB32_crop', real_and_fake_share_NPR_canvas=True,
        Source_RGB_and_alpha_are_discriminator_labels_only=True,
        fake_is_before_gain_route_and_visible_mask_gates=True,
        interpolation='same_bbox_bilinear_align_corners_false_premultiplied_RGBA',
        normalization='patch_mean_then_routes_per_Source_frame_then_visible_distances_then_visible_empty_classes')
    return pairs


def reduce_enemy_patch_values(values, pairs):
    """Do not double a frame's influence merely because it has a valid hint."""
    ids, count = pairs['source_index'], pairs['source_count']
    if values.shape != (len(ids),):
        raise ValueError('Enemy patch reduction requires one patch-mean value per valid route')
    total = values.new_zeros(count).index_add(0, ids, values)
    number = torch.bincount(ids, minlength=count)
    mean, active = total / number.clamp_min(1), number > 0
    visible, bands = pairs['source_visible'].detach().bool(), pairs['distance_band'].detach()
    positive = []
    for band in range(3):
        selected = active & visible & (bands == band)
        if bool(selected.any()): positive.append(mean[selected].mean())
    empty = active & ~visible
    groups = ([torch.stack(positive).mean()] if positive else [])
    if bool(empty.any()): groups.append(mean[empty].mean())
    return torch.stack(groups).mean() if groups else values.sum() * 0


@contextmanager
def _frozen_discriminator(discriminator):
    """No D gradients or spectral-norm buffer updates during generator work."""
    flags = [(parameter, parameter.requires_grad) for parameter in discriminator.parameters()]
    modes = [(module, module.training) for module in discriminator.modules()]
    try:
        discriminator.eval()
        for parameter, _ in flags: parameter.requires_grad_(False)
        yield
    finally:
        for parameter, flag in flags: parameter.requires_grad_(flag)
        for module, mode in modes: module.training = mode


def enemy_patch_discriminator_loss(discriminator, pairs):
    """Hinge D objective; fake/condition/Source labels are all detached."""
    if len(pairs['source_index']) == 0:
        return next(discriminator.parameters()).sum() * 0, dict(
            discriminator_loss=0., enemy_patch_d_loss=0., enemy_patch_skipped_no_valid_boxes=True, **pairs['receipt'])
    condition, real, fake = (pairs[name].detach() for name in ('condition', 'real', 'fake'))
    real_score, fake_score = discriminator(condition, real), discriminator(condition, fake)
    real_hinge = F.relu(1-real_score).flatten(1).mean(-1)
    fake_hinge = F.relu(1+fake_score).flatten(1).mean(-1)
    loss = reduce_enemy_patch_values(real_hinge, pairs) + reduce_enemy_patch_values(fake_hinge, pairs)
    if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite enemy PatchGAN discriminator loss')
    return loss, dict(discriminator_loss=float(loss.detach()), enemy_patch_d_loss=float(loss.detach()),
        enemy_patch_d_real_score=float(reduce_enemy_patch_values(real_score.flatten(1).mean(-1), pairs).detach()),
        enemy_patch_d_fake_score=float(reduce_enemy_patch_values(fake_score.flatten(1).mean(-1), pairs).detach()),
        enemy_patch_skipped_no_valid_boxes=False, **pairs['receipt'])


def enemy_patch_generator_loss(discriminator, pairs, *, feature_weight=.1):
    """Raw-layer adversarial/feature gradients; no learned gain can disable them.

    The caller applies its explicit small GAN schedule/gradient cap. D has its
    own optimizer and is not stepped or mutated by this function.
    """
    if type(feature_weight) not in (int, float) or not math.isfinite(feature_weight) or feature_weight < 0:
        raise ValueError('Enemy PatchGAN feature matching weight must be finite and nonnegative')
    if len(pairs['source_index']) == 0:
        return pairs['fake'].sum() * 0, dict(enemy_patch_g_total=0., enemy_patch_g_hinge=0.,
            adv_loss=0., feature_matching_loss=0., enemy_patch_feature_matching=0.,
            enemy_patch_skipped_no_valid_boxes=True, **pairs['receipt'])
    with _frozen_discriminator(discriminator):
        fake_score, fake_features = discriminator(pairs['condition'].detach(), pairs['fake'], return_features=True)
        hinge = -reduce_enemy_patch_values(fake_score.flatten(1).mean(-1), pairs)
        matching = hinge * 0
        if feature_weight:
            with torch.no_grad():
                _, real_features = discriminator(pairs['condition'].detach(), pairs['real'].detach(), return_features=True)
            per_route = torch.stack([(fake-real).abs().flatten(1).mean(-1)
                                     for fake, real in zip(fake_features, real_features)]).mean(0)
            matching = reduce_enemy_patch_values(per_route, pairs)
        loss = hinge + feature_weight * matching
    if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite enemy PatchGAN generator loss')
    return loss, dict(adv_loss=float(hinge.detach()), feature_matching_loss=float(matching.detach()),
        enemy_patch_g_total=float(loss.detach()), enemy_patch_g_hinge=float(hinge.detach()),
        enemy_patch_feature_matching=float(matching.detach()), enemy_patch_feature_weight=float(feature_weight),
        enemy_patch_generator_before_gain=True, enemy_patch_D_frozen_for_generator=True,
        enemy_patch_skipped_no_valid_boxes=False, **pairs['receipt'])


# Short public names used by the standalone SR adversarial trainer. The more
# descriptive names above remain available to diagnostics and focused tests.
EnemyPatchDiscriminator = EnemyROIPatchDiscriminator
discriminator_loss = enemy_patch_discriminator_loss
generator_loss = enemy_patch_generator_loss
