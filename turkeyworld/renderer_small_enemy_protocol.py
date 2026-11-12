"""Opt-in native-pixel NPC objectives; Source labels never condition rendering."""
from __future__ import annotations

import copy
import math

import torch

from .codec_enemy_objectives import native_object_crop_terms
from .object_losses import balanced_region_losses


SMALL_ENEMY_PROTOCOL = {
    'name': 'one_enemy_native_area_balanced_renderer_v5',
    'version': 1,
    'native_size': 32,
    'source_quotas': {'tiny_1_16': .25, 'small_17_64': .25, 'far': .25, 'replay': .25},
    'far_min_tiles': 5.,
    'foreground_l1_weight': .5,
    'background_l1_weight': .5,
    'native_crop_l1_weight': .25,
    'native_crop_edge_weight': .1,
    'native_crop_padding': 2,
    'wall_boundary_l1_weight': .2,
    'wall_edge_weight': .1,
    'quantized_base_condition_unchanged': True,
    'frozen_source_quantized_latent_target_unchanged': True,
    'codec_frozen': True,
    'source_masks_are_loss_and_sampling_labels_only': True,
}


def small_enemy_protocol():
    """Return an independent copy suitable for immutable experiment receipts."""
    return copy.deepcopy(SMALL_ENEMY_PROTOCOL)


def separate_region_loss(prediction, target, mask, *, squared=False):
    """Mean visible foreground and mean present background with fixed weights.

    An empty image is excluded from the foreground denominator, rather than
    adding a zero foreground term. Native object area never sets image weight.
    Missing regions contribute zero without transferring weight to the other
    region. Targets/masks remain detached, including area-reduced latent masks.
    """
    regions = balanced_region_losses(prediction, target, mask, squared=squared)
    zero = prediction.sum()*0
    result = {}
    for name in ('foreground', 'background'):
        present = regions[name+'_present']
        result[name] = regions[name][present].mean() if bool(present.any()) else zero
        result[name+'_images'] = int(present.sum())
    result['loss'] = (.5*result['foreground']+.5*result['background'])
    return result


def native_crop_loss(prediction, target, mask):
    l1, edge, count = native_object_crop_terms(prediction, target, mask,
        padding=SMALL_ENEMY_PROTOCOL['native_crop_padding'])
    return (SMALL_ENEMY_PROTOCOL['native_crop_l1_weight']*l1
            +SMALL_ENEMY_PROTOCOL['native_crop_edge_weight']*edge), {
        'native_crop_l1': l1, 'native_crop_edge': edge, 'native_crop_images': count}


def native_area_report(arrays):
    """Source-native cohorts for Base, RF1 and the frozen codec reference.

    The codec reconstruction is an empirical comparison, not a proven lower
    bound. All predictions use the same exact Source masks/poses and crops.
    """
    source, mask, states = (arrays[name] for name in ('source', 'mask', 'states'))
    if (source.ndim != 4 or source.shape[1:] != (3,32,32)
            or mask.shape != (len(source),1,32,32) or states.shape != (len(source),20)
            or not all(bool(torch.isfinite(v).all()) for v in (source,mask,states))):
        raise ValueError('Native NPC validation requires finite Source32, masks and state20')
    area = (mask[:,0] > .5).flatten(1).sum(-1)
    alive = (states[:,13] >= .5) & (states[:,12] > 0)
    distance = ((states[:,8:10]-states[:,:2])*16).norm(dim=-1)
    cohorts = {'tiny_1_16': alive & (area > 0) & (area <= 16),
               'small_17_64': alive & (area >= 17) & (area <= 64),
               'large_65_plus': alive & (area >= 65),
               'far': alive & (area > 0) & (distance >= 5),
               'empty': area == 0}
    report = {'protocol': SMALL_ENEMY_PROTOCOL['name'], 'native_size': 32,
        'source_area_resized': False, 'codec_reference_is_a_proven_lower_bound': False,
        'codec_reference_scope': 'Frozen compact E+Q+D(Source); empirical reconstruction reference',
        'source_foreground_pixels': [int(v) for v in area]}
    for name, key in (('base','base'), ('rf1','rgb'), ('codec_reference','codec_reference')):
        prediction = arrays[key]
        if prediction.shape != source.shape or not bool(torch.isfinite(prediction).all()):
            raise ValueError('Native NPC validation predictions must match finite Source RGB')
        report[name] = {}
        for cohort, selected in cohorts.items():
            count = int(selected.sum())
            if not count:
                report[name][cohort] = {'frames': 0}
                continue
            p, t, m = prediction[selected], source[selected], mask[selected]
            regions = separate_region_loss(p, t, m)
            _, crop = native_crop_loss(p, t, m)
            report[name][cohort] = {'frames': count,
                'foreground_frames': regions['foreground_images'],
                'background_frames': regions['background_images'],
                'foreground_mae': float(regions['foreground']) if regions['foreground_images'] else None,
                'background_mae': float(regions['background']) if regions['background_images'] else None,
                'native_crop_l1': float(crop['native_crop_l1']) if crop['native_crop_images'] else None,
                'native_crop_edge': float(crop['native_crop_edge']) if crop['native_crop_images'] else None,
                'native_crop_images': int(crop['native_crop_images']),
                'full_image_mae': float((p-t).abs().mean())}
    return report


def native_area_assessment(report, baseline, *, improvement=.0005, background_tolerance=.002,
                           cohorts=('tiny_1_16','small_17_64')):
    """Require both heads to improve tiny/small NPCs without hiding BG regressions."""
    failures = []
    for head in ('base', 'rf1'):
        for cohort in cohorts:
            current = report.get(head,{}).get(cohort,{})
            old = baseline.get(head,{}).get(cohort,{})
            if not current.get('frames') or current.get('frames') != old.get('frames'):
                failures.append(head+'_'+cohort+'_missing_or_changed'); continue
            for field in ('foreground_mae','background_mae','native_crop_l1','native_crop_edge'):
                a, b = current.get(field), old.get(field)
                if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) for v in (a,b)):
                    failures.append(head+'_'+cohort+'_'+field+'_nonfinite')
                elif field == 'foreground_mae' and a >= b-improvement:
                    failures.append(head+'_'+cohort+'_foreground_not_improved')
                elif field == 'background_mae' and a > b+background_tolerance:
                    failures.append(head+'_'+cohort+'_background_regressed')
        current,old=report.get(head,{}).get('empty',{}),baseline.get(head,{}).get('empty',{})
        if current.get('frames') or old.get('frames'):
            a,b=current.get('full_image_mae'),old.get('full_image_mae')
            if not current.get('frames') or current.get('frames')!=old.get('frames'):
                failures.append(head+'_empty_missing_or_changed')
            elif any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) for v in (a,b)):
                failures.append(head+'_empty_nonfinite')
            elif a>b+background_tolerance:
                failures.append(head+'_empty_background_regressed')
    return {'passed': not failures, 'failure_reasons': failures,
        'minimum_foreground_improvement': improvement, 'background_mae_tolerance': background_tolerance,
        'scope': 'Same held-out native Source tiny/small cohorts; no full-game qualification'}


def native_preview_positions(mask, states):
    """Pick distinct tiny/small/far/empty columns from the unchanged VAL panel."""
    area = (mask.detach()[:,0] > .5).flatten(1).sum(-1)
    source = states.detach()
    alive = (source[:,13] >= .5) & (source[:,12] > 0)
    distance = ((source[:,8:10]-source[:,:2])*16).norm(dim=-1)
    cohorts = (alive & (area > 0) & (area <= 16), alive & (area >= 17) & (area <= 64),
               alive & (area > 0) & (distance >= 5), area == 0)
    selected = []
    for subset in cohorts:
        available = [int(i) for i in torch.nonzero(subset,as_tuple=False).flatten() if int(i) not in selected]
        if available: selected.append(available[0])
    selected += [i for i in range(len(source)) if i not in selected][:max(0,4-len(selected))]
    return selected


def save_small_enemy_preview(arrays, path, step):
    from .renderer_enemy_validation import save_enemy_preview
    selected = native_preview_positions(arrays['mask'],arrays['states'])
    view = {key:value[selected] for key,value in arrays.items() if torch.is_tensor(value)}
    areas = (arrays['mask'][:,0] > .5).flatten(1).sum(-1)
    view['categories'] = [('empty / walls' if int(areas[i]) == 0 else str(int(areas[i]))+' native NPC pixels')
                          for i in selected]
    save_enemy_preview(view,path,step)
