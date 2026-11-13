"""All-depth wall geometry supervision from detached Source labels only.

No rasterized wall or Source RGB is fed into the renderer. Both compact Base
and the one-step RF renderer receive their usual state/map/frozen features.
"""
from copy import deepcopy
import math

import torch

from .renderer_enemy_objectives import enemy_conditions
from .renderer_refinement import refinement_forward
from .renderer_wall_depth import (DEPTH_BANDS, DEPTH_WALL_PROTOCOL,
                                  source_wall_depth_labels, depth_banded_wall_terms)


GEOMETRY_PROTOCOL = {
    'name': 'one_enemy_all_depth_wall_geometry_renderer_v7', 'version': 1,
    'parent_protocol': 'one_enemy_native_area_far_geometry_renderer_v6',
    'source_labels': deepcopy(DEPTH_WALL_PROTOCOL),
    'loss_bands': [name for name, _, _ in DEPTH_BANDS],
    'weights_per_head': {'wall_l1': .25, 'boundary_l1': .3, 'wall_edge_l1': .2,
        'corner_l1': .2, 'thin_face_l1': .2, 'thin_height_l1': .1,
        'cross_boundary_contrast': .2},
    'contrast_margin_fraction': .25, 'contrast_denominator_floor': .1,
    'reduction': 'Per-image region means, mean present images, then mean present depth bands per term',
    'contrast_scope': 'Source wall silhouettes and plane corners, correct versus swapped adjacent RGB targets',
    'absent_regions': 'Excluded from denominator; zero contribution and explicit support counts',
    'old_v6_objective_and_samplers_unchanged': True,
    'new_parameters': 0, 'codec_and_current_feature_world_frozen': True,
    'source_rgb_or_labels_condition_renderer': False,
    'auxiliary': {'batch_size': 8, 'weight': 1., 'candidate_count': 8192, 'seed': 31415},
    'validation': {'candidate_count': 4096, 'frames': 128, 'seed': 31416},
}


def geometry_wall_protocol():
    return deepcopy(GEOMETRY_PROTOCOL)


def _support_key(field):
    return field.removesuffix('_l1') + '_images'


def _cross_boundary_contrast(prediction, target, masks):
    """Require the two sides of a Source edge to remain distinguishable.

    At identical target RGB the term is exactly zero. A flattened boundary
    receives a gradient toward the correct side on BOTH pixels. NPC and gun
    pixels are excluded by the same loss-only masks used in wall supervision.
    """
    target = target.detach().to(prediction)
    region = torch.maximum(masks['boundary'], masks['corner']).detach().to(prediction)
    values, presents = [], []
    pixels = 0
    for axis, key in ((-1, 'horizontal'), (-2, 'vertical')):
        left, right = [slice(None)] * 4, [slice(None)] * 4
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        pa, pb, ta, tb = prediction[left], prediction[right], target[left], target[right]
        support = masks[key].detach().to(prediction) * torch.maximum(region[left], region[right])
        correct = (pa-ta).abs().mean(1, keepdim=True) + (pb-tb).abs().mean(1, keepdim=True)
        swapped = (pa-tb).abs().mean(1, keepdim=True) + (pb-ta).abs().mean(1, keepdim=True)
        separation = 2*(ta-tb).abs().mean(1, keepdim=True)
        margin = GEOMETRY_PROTOCOL['contrast_margin_fraction'] * separation
        error = (correct-swapped+margin).relu() / separation.clamp_min(
            GEOMETRY_PROTOCOL['contrast_denominator_floor'])
        mass = support.flatten(1).sum(-1)
        values.append((error*support).flatten(1).sum(-1) / mass.clamp_min(1))
        presents.append(mass > 0)
        pixels += int(support.sum())
    values, presents = torch.stack(values), torch.stack(presents)
    per_image = (values*presents).sum(0) / presents.sum(0).clamp_min(1)
    active = presents.any(0)
    loss = per_image[active].mean() if bool(active.any()) else prediction.sum()*0
    return loss, int(active.sum()), pixels


def geometry_wall_terms(prediction, target, labels):
    target = target.detach().to(prediction)
    terms = depth_banded_wall_terms(prediction, target, labels)['bands']
    for band in GEOMETRY_PROTOCOL['loss_bands']:
        value, count, pixels = _cross_boundary_contrast(prediction, target, labels['bands'][band])
        terms[band].update(cross_boundary_contrast=value,
            cross_boundary_contrast_images=count, cross_boundary_contrast_pixels=pixels)
    return terms


def geometry_wall_loss(prediction, target, labels):
    bands = geometry_wall_terms(prediction, target, labels)
    result = {}; total = prediction.sum()*0
    for field, weight in GEOMETRY_PROTOCOL['weights_per_head'].items():
        support = _support_key(field)
        selected = [bands[band][field] for band in GEOMETRY_PROTOCOL['loss_bands']
                    if bands[band][support] > 0]
        value = torch.stack(selected).mean() if selected else prediction.sum()*0
        total = total + weight*value
        result[field] = float(value.detach())
        result[field+'_present_bands'] = len(selected)
    result['weighted_loss'] = float(total.detach())
    result['band_support'] = {band: {key: int(value) for key, value in terms.items()
        if key.endswith(('_images', '_pixels'))} for band, terms in bands.items()}
    return total, result


def geometry_refinement_loss(model, batch, *, detach_base=True, opacity_depth_weight=0.):
    """Extra all-depth geometry batch; no changes to the v6 main objective."""
    if type(opacity_depth_weight) not in (int,float) or not math.isfinite(opacity_depth_weight) or opacity_depth_weight<0:
        raise ValueError('Private opacity depth weight must be finite and nonnegative')
    flow = model.flow_renderer; codec = flow.codec
    if opacity_depth_weight and getattr(flow,'geometry_opacity',None) is None:
        raise ValueError('Private opacity supervision requires its explicit renderer-only network')
    if (not isinstance(detach_base, bool) or flow.kind != 'student' or flow.steps != 1
            or codec.training or codec.quantize.training
            or any(p.requires_grad for p in codec.parameters())):
        raise ValueError('Geometry refinement requires RF1 and the frozen codec in eval')
    conditions = enemy_conditions(model, batch)
    base, rgb, _ = refinement_forward(model, conditions, detach_base=detach_base)
    labels = source_wall_depth_labels(conditions['map'], conditions['state'], conditions['object_mask'])
    total = base.sum()*0
    metrics = {'geometry_aux_examples': len(base), 'geometry_wall_protocol': geometry_wall_protocol(),
        'geometry_aux_source_labels_only': True, 'geometry_aux_future_source_feedback': False,
        'geometry_aux_objective_scope': 'all-depth wall RGB, boundaries, corners, thin faces and paired contrast'}
    for head, prediction in (('base', base), ('rf1', rgb)):
        value, terms = geometry_wall_loss(prediction, conditions['target'], labels)
        total = total+value
        metrics.update({'geometry_aux_'+head+'_'+key: value for key, value in terms.items()})
    if opacity_depth_weight:
        from .renderer_opacity_objectives import opacity_depth_loss
        depth_loss,depth_metrics=opacity_depth_loss(conditions['geometry'],labels)
        total=total+opacity_depth_weight*depth_loss
        metrics.update({'opacity_depth_'+key:value for key,value in depth_metrics.items()})
        metrics.update(opacity_depth_weight=float(opacity_depth_weight),
            opacity_depth_weighted_loss=opacity_depth_weight*float(depth_loss.detach()),
            opacity_depth_source_labels_only=True)
    if not bool(torch.isfinite(total)):
        raise FloatingPointError('Nonfinite all-depth geometry renderer objective')
    metrics['geometry_aux_loss'] = float(total.detach())
    return total, metrics


def geometry_wall_report(arrays):
    labels = source_wall_depth_labels(arrays['map'], arrays['states'], arrays['mask'])
    result = {'protocol': geometry_wall_protocol(), 'gt_source_poses': True,
        'source_geometry_is_loss_and_validation_labels_only': True,
        'codec_reference_is_a_proven_lower_bound': False}
    for head, key in (('base', 'base'), ('rf1', 'rgb'), ('codec_reference', 'codec_reference')):
        bands = geometry_wall_terms(arrays[key], arrays['source'], labels)
        result[head] = {'bands': {band: {field: float(value.detach()) if torch.is_tensor(value) else value
            for field, value in row.items()} for band, row in bands.items()}}
    return result


def geometry_preservation_assessment(report, baseline, *, tolerance=.002):
    """Same-panel guards; absence never becomes a quality claim."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Geometry guard requires a finite nonnegative tolerance')
    failures = []; measured = []; unmeasured = []
    if report.get('protocol') != geometry_wall_protocol() or baseline.get('protocol') != geometry_wall_protocol():
        failures.append('geometry_protocol_missing_or_changed')
    for head in ('base', 'rf1'):
        for band in GEOMETRY_PROTOCOL['loss_bands']:
            current = report.get(head, {}).get('bands', {}).get(band, {})
            old = baseline.get(head, {}).get('bands', {}).get(band, {})
            for field in GEOMETRY_PROTOCOL['weights_per_head']:
                label = head+'_'+band+'_'+field
                support = _support_key(field)
                count, previous = current.get(support), old.get(support)
                if type(count) is not int or type(previous) is not int or count < 0 or count != previous:
                    failures.append(label+'_support_missing_or_changed'); continue
                if not count:
                    unmeasured.append(label); continue
                a, b = current.get(field), old.get(field)
                if any(type(v) not in (int, float) or not math.isfinite(v) for v in (a, b)):
                    failures.append(label+'_nonfinite'); continue
                measured.append(label)
                if a > b+tolerance: failures.append(label+'_regressed')
    if not measured: failures.append('no_measured_geometry_regions')
    return {'passed': not failures and bool(measured), 'failure_reasons': failures,
        'measured_regions': measured, 'unmeasured_regions': unmeasured,
        'absolute_loss_tolerance': tolerance, 'scope': 'Same Source depth bands and support; image-reference geometry only'}


def geometry_report_score(report):
    score = 0.
    for head in ('base', 'rf1'):
        for field, weight in GEOMETRY_PROTOCOL['weights_per_head'].items():
            values = [report[head]['bands'][band][field] for band in GEOMETRY_PROTOCOL['loss_bands']
                if report[head]['bands'][band][_support_key(field)] > 0]
            if values: score += weight*sum(values)/len(values)
    return score
