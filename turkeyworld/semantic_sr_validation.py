"""Held-out NPR32 -> Source128 evaluation with explicit RGB/guided profiles."""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from .renderer_student import _hash_tensors
from .rssm_preview_assets import preserve_preview_rng
from .semantic_sr_objectives import semantic_sr_loss, LOSS_WEIGHTS
from .training_pipeline import atomic_json, now, replace_file
from .visual_quality import image_quality_metrics, strict_render_precision


KIND = 'semantic_npr_super_resolution_v1'
BASELINES = ('sr38100', 'sr40100')
DISTANCES = ('near', 'mid', 'far')
AREA_BANDS = ('area_1_4', 'area_5_16', 'area_17_64', 'area_gt64')
ANIMATION_NAMES = ('idle', 'walk_a', 'walk_b', 'attack', 'hurt', 'death_start', 'death_late')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _pixels(value):
    return value.detach().float().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()


def _png(pixels):
    stream = io.BytesIO(); Image.fromarray(pixels).save(stream, format='PNG')
    return base64.b64encode(stream.getvalue()).decode('ascii')


def _unpng(value):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(value))).convert('RGB')).copy()


def _crop(mask):
    """Source-defined HR crop with two-pixel context and at least11x11 for SSIM."""
    yy, xx = torch.where(mask > .5)
    if not len(xx): return None
    height, width = mask.shape
    def axis(lo, hi, limit):
        lo, hi = max(0, int(lo)-2), min(limit, int(hi)+3)
        extra = max(0, 11-(hi-lo)); lo = max(0, lo-extra//2); hi = min(limit, hi+(extra-extra//2))
        lo = max(0, min(lo, hi-11)); hi = min(limit, max(hi, lo+11))
        return lo, hi
    left, right = axis(xx.min(), xx.max(), width); top, bottom = axis(yy.min(), yy.max(), height)
    return left, top, right, bottom


def _rows(prediction, target, mask, background, mask_lr, aux=None):
    """Per-frame reductions retain present-only foreground denominators."""
    quality = image_quality_metrics(prediction, target)
    pred, truth, masks = prediction.float(), target.detach().float(), mask.detach().float()
    difference = (pred-truth).abs()
    area = masks.flatten(1).sum(-1); background_area = (1-masks).flatten(1).sum(-1)
    foreground = (difference*masks).flatten(1).sum(-1)/(3*area).clamp_min(1)
    back = (difference*(1-masks)).flatten(1).sum(-1)/(3*background_area).clamp_min(1)
    absent = ((pred-background.detach()).abs()*masks).flatten(1).sum(-1)/(3*area).clamp_min(1)
    dx = ((pred[..., 1:]-pred[..., :-1])-(truth[..., 1:]-truth[..., :-1])).abs()
    dy = ((pred[..., 1:, :]-pred[..., :-1, :])-(truth[..., 1:, :]-truth[..., :-1, :])).abs()
    mx, my = torch.maximum(masks[..., 1:], masks[..., :-1]), torch.maximum(masks[..., 1:, :], masks[..., :-1, :])
    bx, by = (masks[..., 1:]-masks[..., :-1]).abs(), (masks[..., 1:, :]-masks[..., :-1, :]).abs()
    def edges(x, y):
        support = x.flatten(1).sum(-1)+y.flatten(1).sum(-1)
        value = ((dx*x).flatten(1).sum(-1)+(dy*y).flatten(1).sum(-1))/(3*support).clamp_min(1)
        return value, support
    edge, edge_support = edges(mx, my); boundary, boundary_support = edges(bx, by)
    segmentation = None
    if aux is not None:
        logits, presence = aux.get('mask_logits_lr'), aux.get('presence_logits')
        if (not torch.is_tensor(logits) or logits.shape != mask_lr.shape
                or not torch.is_tensor(presence) or presence.numel() != len(pred)
                or not bool(torch.isfinite(logits).all() and torch.isfinite(presence).all())):
            raise ValueError('Semantic SR auxiliaries require finite LR mask and per-frame presence logits')
        actual, expected = logits.sigmoid() >= .5, mask_lr.detach() >= .5
        intersection = (actual & expected).flatten(1).sum(-1).float()
        union = (actual | expected).flatten(1).sum(-1).float()
        total = actual.flatten(1).sum(-1)+expected.flatten(1).sum(-1)
        segmentation = dict(iou=torch.where(union > 0, intersection/union.clamp_min(1), torch.ones_like(union)),
            f1=torch.where(total > 0, 2*intersection/total.clamp_min(1), torch.ones_like(union)),
            presence=presence.reshape(-1).sigmoid() >= .5,
            target_presence=expected.flatten(1).any(-1),
            false_mask_fraction=actual.flatten(1).float().mean(-1))
    result = []
    for index in range(len(pred)):
        crop = _crop(masks[index, 0])
        roi_quality = None
        if crop is not None:
            left, top, right, bottom = crop
            roi_quality = image_quality_metrics(pred[index:index+1, :, top:bottom, left:right],
                truth[index:index+1, :, top:bottom, left:right])
        row = {key: float(value[index]) for key, value in quality.items()}
        row.update(foreground_mae=float(foreground[index]) if area[index] > 0 else None,
            background_mae=float(back[index]) if background_area[index] > 0 else None,
            foreground_edge_mae=float(edge[index]) if edge_support[index] > 0 else None,
            silhouette_edge_mae=float(boundary[index]) if boundary_support[index] > 0 else None,
            closer_to_true_NPC_than_background=bool(absent[index] > foreground[index]) if area[index] > 0 else None,
            roi_crop_ssim=None if roi_quality is None else float(roi_quality['ssim'][0]),
            roi_crop_mae=None if roi_quality is None else float(roi_quality['mae'][0]))
        if segmentation is not None:
            row.update(aux_mask_iou=float(segmentation['iou'][index]),
                aux_mask_pixel_f1=float(segmentation['f1'][index]),
                aux_presence=bool(segmentation['presence'][index]),
                aux_target_presence=bool(segmentation['target_presence'][index]),
                aux_false_mask_fraction=float(segmentation['false_mask_fraction'][index]))
        result.append(row)
    return result


def _summary(rows, selected):
    chosen = [row for row, keep in zip(rows, selected) if keep]
    if not chosen: return {'frames': 0}
    result = {'frames': len(chosen)}
    for key in ('psnr', 'ssim', 'mae', 'foreground_mae', 'background_mae', 'foreground_edge_mae',
                'silhouette_edge_mae', 'roi_crop_ssim', 'roi_crop_mae', 'aux_mask_iou',
                'aux_mask_pixel_f1', 'closer_to_true_NPC_than_background'):
        values = [row[key] for row in chosen if row.get(key) is not None]
        name = key+'_mean' if key in ('psnr', 'ssim', 'mae', 'roi_crop_ssim', 'roi_crop_mae',
                                      'aux_mask_iou', 'aux_mask_pixel_f1') else key
        result[name] = float(np.mean(values)) if values else None
        if key in ('foreground_mae', 'background_mae', 'roi_crop_ssim'):
            result[{'foreground_mae': 'foreground_frames', 'background_mae': 'background_frames',
                    'roi_crop_ssim': 'roi_crop_frames'}[key]] = len(values)
    aux = [row for row in chosen if 'aux_presence' in row]
    if aux:
        tp = sum(r['aux_presence'] and r['aux_target_presence'] for r in aux)
        fp = sum(r['aux_presence'] and not r['aux_target_presence'] for r in aux)
        fn = sum(not r['aux_presence'] and r['aux_target_presence'] for r in aux)
        negatives = sum(not r['aux_target_presence'] for r in aux)
        result.update(aux_presence_tp=int(tp), aux_presence_fp=int(fp), aux_presence_fn=int(fn),
            aux_presence_precision=tp/(tp+fp) if tp+fp else None,
            aux_presence_recall=tp/(tp+fn) if tp+fn else None,
            aux_presence_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            aux_presence_negative_frames=int(negatives),
            aux_presence_false_positive_rate=fp/negatives if negatives else None)
    return result


def _cohorts(metadata):
    result = {key: [] for key in ('overall', *DISTANCES, *AREA_BANDS,
        'visible_enemy', 'empty', 'occluded', 'navigation', 'corpse', 'hp0', 'subpixel_lr')}
    animation = any('animation_label' in row for row in metadata)
    if animation:
        result.update({'animation_'+name: [] for name in ANIMATION_NAMES})
    for row in metadata:
        alive, player = row['alive'], row['player_alive']; area = row['native_area']
        visible = alive and player and area > 0
        distance = row['distance_tiles']
        values = dict(overall=True, visible_enemy=visible,
            near=visible and distance < 2, mid=visible and 2 <= distance < 5, far=visible and distance >= 5,
            area_1_4=visible and 1 <= area <= 4, area_5_16=visible and 5 <= area <= 16,
            area_17_64=visible and 17 <= area <= 64, area_gt64=visible and area > 64,
            empty=not row['hr_visible'] and area == 0,
            occluded=alive and player and not row['hr_visible'] and area == 0,
            navigation=not alive and player and not row['hr_visible'] and area == 0,
            corpse=not alive and row['hr_visible'], hp0=not player,
            subpixel_lr=alive and player and area == 0 and row['hr_visible'])
        if animation:
            values.update({'animation_'+name: row.get('animation_label') == index and row['hr_visible']
                           for index, name in enumerate(ANIMATION_NAMES)})
        for key in result: result[key].append(bool(values[key]))
    return result


def _assess(report):
    current = report['domains']['npr']['models']['current']
    failures = []
    for baseline in BASELINES:
        old = report['domains']['npr']['models'].get(baseline)
        if old is None:
            failures.append(baseline+'_reference_missing'); continue
        for band in DISTANCES:
            a, b = current['cohorts'][band], old['cohorts'][band]
            if (not a.get('foreground_frames') or a.get('foreground_frames') != b.get('foreground_frames')
                    or a['foreground_mae'] > b['foreground_mae']-.0005):
                failures.append(baseline+'_'+band+'_foreground_not_improved')
        for cohort in ('overall', 'empty', 'navigation', 'occluded', 'corpse'):
            a, b = current['cohorts'][cohort], old['cohorts'][cohort]
            if not a.get('frames') or a['frames'] != b.get('frames'):
                failures.append(baseline+'_'+cohort+'_guard_missing'); continue
            if a['mae_mean'] > b['mae_mean']+.002 or a['ssim_mean'] < b['ssim_mean']-.01:
                failures.append(baseline+'_'+cohort+'_RGB_regression')
    for domain in ('clean', 'codec'):
        if domain not in report['domains']: continue
        models = report['domains'][domain]['models']
        old = models.get('sr40100')
        if old is None:
            failures.append(domain+'_reference_missing'); continue
        current_guard = models['current']['summary']; old_guard = old['summary']
        if (current_guard['mae_mean'] > old_guard['mae_mean']+.002
                or current_guard['ssim_mean'] < old_guard['ssim_mean']-.01
                or current_guard['background_mae'] is None or old_guard['background_mae'] is None
                or current_guard['background_mae'] > old_guard['background_mae']+.002):
            failures.append(domain+'_full_or_background_regression')
    empty = current['cohorts']['empty']
    fpr = empty.get('aux_presence_false_positive_rate')
    if fpr is None or fpr > .05: failures.append('aux_presence_false_positive_rate')
    relative = dict(passed=not failures, failure_reasons=failures,
        foreground_mae_improvement=.0005, guard_mae_tolerance=.002, guard_ssim_tolerance=.01,
        aux_presence_false_positive_rate_max=.05, scope='Relative fixed-panel RGB and auxiliary-head progress only')
    def strict(domain):
        summary = report['domains'][domain]['models']['current']['summary']
        return dict(passed=bool(summary['ssim_mean'] > .91 and summary['mae_mean'] < 2.5/255),
            ssim_mean=summary['ssim_mean'], mae_mean=summary['mae_mean'], ssim_operator='>',
            ssim_threshold=.91, mae_operator='<', mae_threshold=2.5/255)
    clean, npr = strict('clean'), strict('npr')
    return relative, clean, dict(passed=relative['passed'] and clean['passed'] and npr['passed'],
        npr_strict=npr, clean_strict=clean, relative_progress_passed=relative['passed'],
        gameplay_qualified=False, automatic_deployment=False)


def _guided_decoder(model):
    return getattr(model, 'config', {}).get('architecture') == 'enemy_decoder_edsr'


def _visual_first_decoder(model):
    return bool(getattr(model, 'config', {}).get('enemy_decoder_visual_first'))


def _restoration_decoder(model):
    return bool(getattr(model,'config',{}).get('enemy_decoder_restoration'))


def _restoration_metadata(model):
    scale=float(model.config['restoration_blend_scale'])
    phase=getattr(model,'restoration_phase_metadata',{}).get('phase')
    return dict(phase=phase or ('standalone'if scale==0 else 'restoration'if scale==1 else 'blending'),
                blend_scale=scale)


def _accumulate_restoration_raw(accumulator,metrics):
    """Conditional raw means use their actual cohort counts, including tail batches."""
    from .decoder_restoration_protocol import METRICS
    for key in ('frames','visible_frames','empty_frames','raw_visual_frames',
                'raw_guided_frames','raw_box_miss_visible_frames'):
        count=metrics[key]
        if type(count)is not int or count<0:raise ValueError('Invalid raw restoration cohort count')
        accumulator[key]=accumulator.get(key,0)+count
    totals=accumulator.setdefault('_totals',{name:0. for name in METRICS})
    counts=accumulator.setdefault('metric_counts',{name:0 for name in METRICS})
    for name in METRICS:
        count=metrics['metric_counts'][name];value=metrics[name]
        if type(count)is not int or count<0 or not math.isfinite(value):
            raise ValueError('Nonfinite raw restoration metrics or cohort denominator')
        totals[name]+=value*count;counts[name]+=count
    for key in ('scope','before_gain_and_route_or_visible_mask_gates',
                'Source_background_is_labels_only','source_future_as_input'):
        if key in accumulator and accumulator[key]!=metrics[key]:
            raise ValueError('Raw restoration diagnostic scope changed between batches')
        accumulator[key]=metrics[key]


def _finish_restoration_raw(accumulator):
    result={key:copy.deepcopy(value)for key,value in accumulator.items()if key!='_totals'}
    result.update({name:value/accumulator['metric_counts'][name]
        if accumulator['metric_counts'][name] else None for name,value in accumulator['_totals'].items()})
    result['aggregation']='actual_conditional_image_count_weighted_means_all_fixed_NPR_rows'
    return result


def _current_forward(model, low, batch):
    if _visual_first_decoder(model):
        hint = batch.get('hypothesis')
        return model.forward_with_aux(low, None if hint is None else hint.detach())
    if _guided_decoder(model):
        if 'hypothesis' not in batch:
            raise ValueError('Enemy decoder validation needs pinned causal RSSM hypotheses')
        return model.forward_with_aux(low, batch['hypothesis'].detach())
    return model.forward_with_aux(low)


def _validation_profile(model):
    if _restoration_decoder(model):
        from .decoder_restoration_objectives import decoder_restoration_loss,LOSS_WEIGHTS as weights
        return ('decoder_restoration_super_resolution_v3','decoder_restoration_sr_v3',
                decoder_restoration_loss,weights,'actual_NPR_RGB32_with_optional_causal_RSSM_hint')
    if _visual_first_decoder(model):
        from .visual_first_objectives import visual_first_loss, LOSS_WEIGHTS as weights
        return ('visual_first_enemy_decoder_super_resolution_v2', 'visual_first_enemy_decoder_sr_v2',
                visual_first_loss, weights, 'actual_NPR_RGB32_with_optional_causal_RSSM_hint')
    if _guided_decoder(model):
        from .enemy_decoder_objectives import enemy_decoder_loss, LOSS_WEIGHTS as weights
        return ('enemy_decoder_super_resolution_v1', 'rssm_guided_enemy_decoder_sr_v1',
                enemy_decoder_loss, weights, 'actual_NPR_RGB32_and_causal_RSSM_prior')
    return KIND, 'rgb_only_semantic_npr_sr_v1', semantic_sr_loss, LOSS_WEIGHTS, 'only_actual_NPR_RGB32'


@torch.no_grad()
def evaluate_semantic_sr(model, data, panel, *, step=0, batch_size=8, baselines=None,
                         codec=None, baseline_report=None):
    """Evaluate fixed held-out actual NPR inputs; never send any GT labels to SR.

    Initial baselines are frozen explicit SR38100/SR40100 modules. Later calls
    can reuse their statistics and encoded preview pixels from that initial
    report, only under the exact same data identity and fixed panel.
    """
    kind, objective_protocol, objective_fn, objective_weights, inference_input = _validation_profile(model)
    panel = list(panel)
    if (getattr(data, 'split', None) != 'val' or not panel or batch_size < 1
            or len(set(map(int, panel))) != len(panel)):
        raise ValueError('Semantic SR requires unique explicit held-out VAL rows')
    identity = copy.deepcopy(data.identity); identity_sha = _digest(identity)
    fingerprint = data.panel_fingerprint(panel)
    cache = baseline_report
    if cache is not None and (cache.get('kind') != kind or cache.get('panel_fingerprint') != fingerprint
            or cache.get('data_identity_sha256') != identity_sha):
        raise ValueError('Cached SR baselines belong to a different Source/NPR/panel identity')
    baselines = {} if baselines is None else dict(baselines)
    if set(baselines)-set(BASELINES): raise ValueError('Unknown semantic SR baseline name')
    if not baselines and cache is None: raise ValueError('Initial evaluation requires the two frozen SR references')
    if cache is None and set(baselines) != set(BASELINES):
        raise ValueError('Initial evaluation must measure both38100 and40100 on the same actual NPR')
    networks = {'current': model, **baselines}
    modules = [module for net in (*networks.values(), codec) if net is not None for module in net.modules()]
    modes = [(module, module.training) for module in modules]
    weights = {name: _hash_tensors(net.state_dict()) for name, net in networks.items()}
    if any(p.dtype != torch.float32 for net in networks.values() for p in net.parameters()):
        raise ValueError('Semantic SR validation requires FP32 current and reference weights')
    device = next(model.parameters()).device
    private = copy.deepcopy(data.sampler_state()) if hasattr(data, 'sampler_state') else None
    domain_rows, metadata, cases = {}, [], {}
    objective_sum, objective_frames, objective_batches = 0., 0, 0
    objective_components = {name: {'raw': 0., 'weighted': 0., 'weight': weight}
                            for name, weight in objective_weights.items()}
    decoder_counts = dict(hypothesis_frames=0, eligible_frames=0, tp=0, fp=0, fn=0, tn=0)
    decoder_iou = []
    restoration_raw={}
    visual_diagnostics = None
    if _visual_first_decoder(model):
        from .visual_first_validation import VisualFirstDiagnostics, wrong_prediction_hint
        visual_diagnostics = VisualFirstDiagnostics()
    animation_counts = dict(predicted=[0]*7, aligned_visible=0, eligible_visible=0,
                            visual_correct=0, visual_frames=0)
    codec_present = codec is not None
    try:
        for net in (*networks.values(), codec):
            if net is not None: net.eval()
        with preserve_preview_rng(device, getattr(data, 'rng', None)), strict_render_precision(), \
                torch.inference_mode(), torch.autocast(device.type, enabled=False):
            for first in range(0, len(panel), batch_size):
                ids = panel[first:first+batch_size]
                batch = data.batch(len(ids), indices=ids)
                target, mask, low_mask = batch['hr_source'].detach(), batch['mask_hr'].detach(), batch['mask_lr'].detach()
                if target.shape != (len(ids), 3, 128, 128) or mask.shape != (len(ids), 1, 128, 128) or low_mask.shape != (len(ids), 1, 32, 32):
                    raise ValueError('Expected exact Source128 and independent visible masks32/128')
                if (not all(bool(torch.isfinite(value).all()) for value in (target, mask, low_mask))
                        or bool(((mask < 0) | (mask > 1)).any())
                        or bool(((low_mask < 0) | (low_mask > 1)).any())):
                    raise FloatingPointError('Source HR and visible mask labels must be finite and masks in[0,1]')
                inputs = {'npr': batch['lr_npr'].detach(), 'clean': batch['lr_source'].detach()}
                if 'codec_lr' in batch:
                    inputs['codec'] = batch['codec_lr'].detach(); codec_present = True
                elif codec is not None:
                    inputs['codec'] = codec(inputs['clean'])['reconstruction'].detach()
                states = batch['state'].detach().cpu()
                native = low_mask.flatten(1).sum(-1).cpu()
                batch_meta = [dict(row_index=int(ids[i]), alive=bool(states[i, 13] >= .5 and states[i, 12] > 0),
                    player_alive=bool(states[i, 4] > 0), native_area=int(native[i]),
                    hr_visible=bool(mask[i].any()),
                    distance_tiles=float(torch.linalg.vector_norm((states[i, 8:10]-states[i, :2])*16)))
                    for i in range(len(ids))]
                if 'animation_label' in batch:
                    for row, label in zip(batch_meta, batch['animation_label'].detach().cpu().reshape(-1).tolist()):
                        row['animation_label'] = int(label)
                metadata.extend(batch_meta)
                cohorts = _cohorts(batch_meta)
                picked = {}
                for role in (*DISTANCES, 'empty'):
                    if role not in cases:
                        selected = next((i for i, value in enumerate(cohorts[role]) if value), None)
                        if selected is not None:
                            picked[role] = selected
                            cases[role] = dict(metadata=batch_meta[selected], crop=_crop(mask[selected, 0]),
                                input=_pixels(inputs['npr'][selected]), target=_pixels(target[selected]), predictions={})
                for domain, low in inputs.items():
                    if low.shape != (len(ids), 3, 32, 32) or not bool(torch.isfinite(low).all()):
                        raise ValueError('SR inference input must be finite native RGB32 only')
                    domain_rows.setdefault(domain, {name: [] for name in networks})
                    for name, net in networks.items():
                        auxiliary = _current_forward(net, low, batch) if name == 'current' else None
                        prediction = auxiliary['image'] if auxiliary is not None else net(low)
                        if prediction.shape != target.shape or not bool(torch.isfinite(prediction).all()) or bool(((prediction < 0) | (prediction > 1)).any()):
                            raise FloatingPointError('Invalid SR RGB128 output')
                        if domain == 'npr' and name == 'current':
                            # The exact training formula, evaluated only on held-out
                            # actual NPR. Reuse this RGB-only forward; labels are
                            # consumed by the loss and never by the SR network.
                            objective, objective_metrics = objective_fn(auxiliary, batch)
                            count = len(ids)
                            objective_sum += float(objective) * count
                            objective_frames += count
                            objective_batches += 1
                            for component, values in objective_components.items():
                                values['raw'] += objective_metrics[component] * count
                                values['weighted'] += objective_metrics[component+'_weighted'] * count
                            if _guided_decoder(model):
                                available = batch['hypothesis'][:, -1] > .5
                                eligible = auxiliary['enemy_applicable'].reshape(-1).bool()
                                present = mask.flatten(1).any(-1)
                                confirmed = auxiliary['enemy_confirmation_logits'].reshape(-1) >= 0
                                decoder_counts['hypothesis_frames'] += int(available.sum())
                                decoder_counts['eligible_frames'] += int(eligible.sum())
                                for key, selected in (('tp', confirmed & present), ('fp', confirmed & ~present),
                                                      ('fn', ~confirmed & present), ('tn', ~confirmed & ~present)):
                                    decoder_counts[key] += int((eligible & selected).sum())
                                alpha = auxiliary['enemy_alpha_hr'] > .5
                                truth = mask > .5
                                intersections = (alpha & truth).flatten(1).sum(-1).float()
                                unions = (alpha | truth).flatten(1).sum(-1).float().clamp_min(1)
                                decoder_iou.extend((intersections/unions)[eligible & present].cpu().tolist())
                                if 'enemy_animation_index' in auxiliary and 'animation_label' in batch:
                                    predicted_phase = auxiliary['enemy_animation_index'].reshape(-1)
                                    labels = batch['animation_label'].reshape(-1)
                                    for phase in range(7):
                                        animation_counts['predicted'][phase] += int((available & (predicted_phase == phase)).sum())
                                    selected_animation = eligible & present
                                    animation_counts['eligible_visible'] += int(selected_animation.sum())
                                    animation_counts['aligned_visible'] += int((selected_animation & (predicted_phase == labels)).sum())
                                    if 'enemy_animation_logits' in auxiliary:
                                        visual_phase = auxiliary['enemy_animation_logits'].argmax(-1)
                                        animation_counts['visual_frames'] += int(selected_animation.sum())
                                        animation_counts['visual_correct'] += int((selected_animation & (visual_phase == labels)).sum())
                        domain_rows[domain][name].extend(_rows(prediction, target, mask,
                            batch['hr_background'], low_mask, auxiliary))
                        no_hint_output = None
                        if visual_diagnostics is not None and domain == 'npr' and name == 'current':
                            no_hint_output = model.forward_with_aux(low, None)
                            wrong_output = model.forward_with_aux(low, wrong_prediction_hint(batch.get('hypothesis')))
                            variants = {'with_hint': auxiliary, 'no_hint': no_hint_output, 'wrong_hint': wrong_output}
                            for mode, value in variants.items():
                                image = value['image']
                                if (image.shape != target.shape or not bool(torch.isfinite(image).all())
                                        or bool(((image < 0) | (image > 1)).any())):
                                    raise FloatingPointError('Invalid RGB in visual-first hint comparison')
                                image_rows = (domain_rows[domain][name][-len(ids):] if mode == 'with_hint' else
                                    _rows(image, target, mask, batch['hr_background'], low_mask, value))
                                visual_diagnostics.add(mode, value, batch, image_rows)
                            visual_diagnostics.compare(auxiliary, {'no_hint': no_hint_output, 'wrong_hint': wrong_output})
                            if _restoration_decoder(model):
                                from .decoder_restoration_objectives import restoration_raw_metrics
                                for mode,value in variants.items():
                                    _accumulate_restoration_raw(restoration_raw.setdefault(mode,{}),
                                                               restoration_raw_metrics(value,batch))
                        if domain == 'npr':
                            for role, index in picked.items():
                                cases[role]['predictions'][name] = _pixels(prediction[index])
                                if no_hint_output is not None:
                                    cases[role]['predictions']['no_hint'] = _pixels(no_hint_output['image'][index])
    finally:
        for module, training in modes: module.training = training
        if private is not None: data.restore_sampler(private)
    if any(_hash_tensors(net.state_dict()) != weights[name] for name, net in networks.items()):
        raise AssertionError('Validation changed current/reference model weights or buffers')
    selected = _cohorts(metadata); domains = {}
    for domain, models in domain_rows.items():
        reports = {}
        if cache is not None:
            reports.update({name: copy.deepcopy(values) for name, values in
                cache.get('domains', {}).get(domain, {}).get('models', {}).items() if name in BASELINES})
        for name, rows in models.items():
            cohorts = {key: _summary(rows, mask) for key, mask in selected.items()}
            reports[name] = dict(summary=cohorts['overall'], cohorts=cohorts, input_kind=domain+'_rgb',
                frames=len(rows), source_held_out=True, world_rollout=False)
        domains[domain] = dict(models=reports, primary=domain == 'npr')
    baseline_pixels = copy.deepcopy(cache.get('baseline_preview', {})) if cache is not None else {}
    for role, case in cases.items():
        saved = baseline_pixels.setdefault(role, dict(row_index=case['metadata']['row_index'], images={}))
        if saved['row_index'] != case['metadata']['row_index']:
            raise ValueError('Cached reference preview case changed despite the fixed panel')
        for name in BASELINES:
            if name in case['predictions']: saved['images'][name] = _png(case['predictions'][name])
            elif name in saved['images']: case['predictions'][name] = _unpng(saved['images'][name])
    baseline_hash = {**({} if cache is None else cache['baseline_model_sha256']),
        **{name: value for name, value in weights.items() if name in BASELINES}}
    report = dict(kind=kind, step=int(step), frames=len(panel), input_kind='npr_rgb',
        source_held_out=True, world_rollout=False, source_input_allowed_for_sr=False,
        inference_input=inference_input, source_HR_target='exact_same_pose_map_and_time_Source128',
        source_masks_and_state_are_labels_only=True, data_identity=identity, data_identity_sha256=identity_sha,
        panel_fingerprint=fingerprint, domains=domains, baseline_model_sha256=baseline_hash,
        baseline_preview=baseline_pixels, native_area_bands=list(AREA_BANDS),
        distance_bands=dict(near='<2', mid='[2,5)', far='>=5'), codec_guard_present=codec_present,
        segmentation_scope='Auxiliary LR mask/presence heads; not an independent RGB recognition detector',
        ROI_SSIM_scope='Source-defined HR bounding crop +2px, min11px; whole-frame SSIM is separate',
        gameplay_qualified=False, release_passed=False, automatic_deployment=False)
    if objective_frames != len(panel):
        raise AssertionError('Validation objective did not cover every actual NPR row exactly once')
    report['validation_objective'] = dict(
        protocol=objective_protocol, input_kind='npr_rgb',
        loss=objective_sum/objective_frames, frames=objective_frames, batch_size=batch_size,
        batch_count=objective_batches, aggregation='image_count_weighted_mean_of_actual_batch_objectives',
        components={name: {'raw': value['raw']/objective_frames,
                           'weighted': value['weighted']/objective_frames, 'weight': value['weight']}
                    for name, value in objective_components.items()},
        source_held_out=True,
        conditional_reductions='Present-foreground/crop reductions are computed within each actual batch, as in training',
        training_input_domain_difference=dict(validation='100% actual frozen NPR RGB32',
            training='at least75% actual NPR plus clean/cached-codec guard inputs',
            same_loss_formula=True, same_input_mixture=False))
    report['relative_progress'], report['clean_source_quality'], report['training_goal'] = _assess(report)
    base_quality_goal = report['training_goal']['passed']
    if _guided_decoder(model):
        counts = decoder_counts
        report['enemy_decoder'] = dict(**counts,
            confirmation_false_positive_rate=counts['fp']/max(1, counts['fp']+counts['tn']),
            confirmation_recall=counts['tp']/max(1, counts['tp']+counts['fn']),
            visible_mask_iou_mean=float(np.mean(decoder_iou)) if decoder_iou else None,
            visible_mask_frames=len(decoder_iou), proposal_scope='Causal one-step RSSM prior from actual preceding context',
            source_future_state_allowed=False, visible_source_masks_are_labels_only=True,
            renders_native_atlas=False, inference_inputs=['NPR_RGB32', 'predicted_RSSM_state20', 'hypothesis_available'])
        if any('animation_label' in row for row in metadata):
            report['enemy_decoder']['animations'] = dict(
                source_visible_counts={name: domains['npr']['models']['current']['cohorts']['animation_'+name]['frames']
                                       for name in ANIMATION_NAMES},
                predicted_counts=dict(zip(ANIMATION_NAMES, animation_counts['predicted'])),
                aligned_visible=animation_counts['aligned_visible'], eligible_visible=animation_counts['eligible_visible'],
                visual_accuracy=animation_counts['visual_correct']/max(1, animation_counts['visual_frames']),
                visual_frames=animation_counts['visual_frames'],
                labels_are_Source_only=True, condition_is_predicted_RSSM_only=True)
        report['rssm_hypotheses_are_predictions'] = True
        report['source_future_state_allowed_for_sr'] = False
        report['training_goal']['decoder_confirmation_passed'] = bool(
            counts['fp']+counts['tn'] > 0 and counts['tp']+counts['fn'] > 0
            and report['enemy_decoder']['confirmation_false_positive_rate'] <= .05
            and report['enemy_decoder']['confirmation_recall'] >= .8)
        report['training_goal']['passed'] = bool(report['training_goal']['passed']
            and report['training_goal']['decoder_confirmation_passed'])
        if visual_diagnostics is not None:
            comparison = visual_diagnostics.report()
            report['enemy_decoder']['visual_first'] = comparison
            mode_gates = {mode: bool(values['positive_frames'] and values['negative_frames']
                and values['confirmation_false_positive_rate'] <= .05 and values['confirmation_recall'] >= .8)
                for mode, values in comparison['domains'].items()}
            report['training_goal']['visual_first_mode_gates'] = mode_gates
            report['training_goal']['decoder_confirmation_passed'] = mode_gates['with_hint']
            invariant = comparison['high_visual_output_invariance']
            report['training_goal']['passed'] = bool(base_quality_goal
                and all(mode_gates.values()) and invariant['passed'] is True)
    current = domains['npr']['models']['current']
    if _restoration_decoder(model):
        report['restoration_raw_modes']={name:_finish_restoration_raw(value)
                                        for name,value in restoration_raw.items()}
        report['restoration_raw']=copy.deepcopy(report['restoration_raw_modes']['with_hint'])
        report['enemy_decoder']['restoration']={**_restoration_metadata(model),**report['restoration_raw'],
            'raw_metrics_are_final_image_quality':False,'inference_uses_saved_blend_scale':True}
    fg = [current['cohorts'][band].get('foreground_mae') for band in DISTANCES]
    edge = current['cohorts']['visible_enemy'].get('foreground_edge_mae')
    report['selection_score'] = (float(np.mean(fg))+.25*current['summary']['mae_mean']+.1*edge
        if all(value is not None for value in fg) and edge is not None else None)
    preview = dict(cases=cases, metadata=dict(kind=kind, step=int(step), input_kind='npr_rgb',
        source_held_out=True, world_rollout=False, source_input_allowed_for_sr=False,
        image_size=32, output_size=128, panel_fingerprint=fingerprint,
        data_identity_sha256=identity_sha, inference_input=inference_input,
        row_count=5, row_labels=['NPR32 input', 'frozen SR38100', 'new semantic SR128',
            'true paired Source128', 'absolute error x4']))
    if visual_diagnostics is not None:
        preview['metadata'].update(row_count=6, row_labels=['NPR32 input', 'frozen SR38100',
            'NEW with_hint SR128', 'NEW no_hint SR128 (RSSM OFF)', 'true paired Source128', 'absolute error x4'],
            comparison_modes=['with_hint', 'no_hint'])
    if _restoration_decoder(model):
        preview['metadata'].update(restoration_phase=_restoration_metadata(model)['phase'],
                                   restoration_blend_scale=model.config['restoration_blend_scale'])
    json.dumps(report, allow_nan=False)
    return report, preview


@torch.no_grad()
def preview_semantic_sr(model, data, panel, *, step, baseline_report, batch_size=8):
    """Light preview of the same four fixed cases, without recomputing VAL metrics."""
    kind, _, _, _, inference_input = _validation_profile(model)
    if (getattr(data, 'split', None) != 'val' or baseline_report.get('kind') != kind
            or baseline_report.get('panel_fingerprint') != data.panel_fingerprint(list(panel))
            or baseline_report.get('data_identity_sha256') != _digest(data.identity)):
        raise ValueError('Light SR preview requires the exact initial held-out NPR/Source panel')
    roles = (*DISTANCES, 'empty'); cached = baseline_report.get('baseline_preview', {})
    if any(role not in cached for role in roles):
        raise ValueError('Fixed SR preview is missing near/mid/far/empty Source cases')
    ids = [int(cached[role]['row_index']) for role in roles]
    device = next(model.parameters()).device
    modes = [(module, module.training) for module in model.modules()]
    private = copy.deepcopy(data.sampler_state()) if hasattr(data, 'sampler_state') else None
    cases = {}
    try:
        model.eval()
        with preserve_preview_rng(device, getattr(data, 'rng', None)), strict_render_precision(), \
                torch.inference_mode(), torch.autocast(device.type, enabled=False):
            for first in range(0, len(ids), batch_size):
                chosen = ids[first:first+batch_size]
                batch = data.batch(len(chosen), indices=chosen)
                low = batch['lr_npr'].detach()
                prediction = _current_forward(model, low, batch)['image'].float()
                no_hint = model.forward_with_aux(low, None)['image'].float() if _visual_first_decoder(model) else None
                if (prediction.shape != (len(chosen), 3, 128, 128)
                        or not bool(torch.isfinite(prediction).all())
                        or bool(((prediction < 0) | (prediction > 1)).any())):
                    raise FloatingPointError('Invalid light SR RGB128 output')
                for offset, index in enumerate(chosen):
                    role = roles[first+offset]
                    state = batch['state'][offset].detach()
                    metadata = dict(row_index=index, native_area=int(batch['mask_lr'][offset].sum()),
                        distance_tiles=float(torch.linalg.vector_norm((state[8:10]-state[:2])*16)))
                    references = {name: _unpng(cached[role]['images'][name]) for name in BASELINES}
                    cases[role] = dict(metadata=metadata, crop=_crop(batch['mask_hr'][offset, 0]),
                        input=_pixels(low[offset]), target=_pixels(batch['hr_source'][offset]),
                        predictions={**references, 'current': _pixels(prediction[offset])})
                    if no_hint is not None:
                        cases[role]['predictions']['no_hint'] = _pixels(no_hint[offset])
    finally:
        for module, training in modes: module.training = training
        if private is not None: data.restore_sampler(private)
    result = dict(cases=cases, metadata=dict(kind=kind, step=int(step), input_kind='npr_rgb',
        source_held_out=True, world_rollout=False, source_input_allowed_for_sr=False,
        image_size=32, output_size=128, panel_fingerprint=baseline_report['panel_fingerprint'],
        data_identity_sha256=baseline_report['data_identity_sha256'],
        inference_input=inference_input, preview_only=True,
        reference_validation_step=baseline_report['step'], row_count=5,
        row_labels=['NPR32 input', 'frozen SR38100', 'new semantic SR128',
            'true paired Source128', 'absolute error x4']))
    if _visual_first_decoder(model):
        result['metadata'].update(row_count=6, row_labels=['NPR32 input', 'frozen SR38100',
            'NEW with_hint SR128', 'NEW no_hint SR128 (RSSM OFF)', 'true paired Source128', 'absolute error x4'],
            comparison_modes=['with_hint', 'no_hint'])
    if _restoration_decoder(model):
        result['metadata'].update(restoration_phase=_restoration_metadata(model)['phase'],
                                  restoration_blend_scale=model.config['restoration_blend_scale'])
    return result


def save_semantic_sr_preview(preview, path, *, every=100):
    """Fixed near/mid/far/empty full frames and same-box HR ROI zooms."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    roles = (*DISTANCES, 'empty'); cell, row_height = 256, 280
    rows = int(preview['metadata']['row_count'])
    visual_first = preview['metadata']['kind'] in ('visual_first_enemy_decoder_super_resolution_v2',
                                                  'decoder_restoration_super_resolution_v3')
    if rows != (6 if visual_first else 5):
        raise ValueError('SR preview row count differs from its explicit inference profile')
    width, height = cell*4, row_height*rows+32+3*row_height
    canvas = Image.new('RGB', (width, height), (14, 22, 31)); draw = ImageDraw.Draw(canvas)
    labels = preview['metadata']['row_labels']
    for row, label in enumerate(labels):
        top = row*row_height; draw.text((5, top+3), label, fill=(235, 239, 245))
        for column, role in enumerate(roles):
            case = preview['cases'].get(role)
            if case is None: continue
            truth = case['target']; prediction = case['predictions']['current']
            values = [case['input'], case['predictions'].get('sr38100'), prediction]
            if visual_first:
                values.append(case['predictions']['no_hint'])
            values.extend([truth, np.clip(np.abs(prediction.astype(np.int16)-truth.astype(np.int16))*4,
                                         0, 255).astype(np.uint8)])
            pixels = values[row]
            if pixels is None: continue
            canvas.paste(Image.fromarray(pixels).resize((cell, cell), Image.Resampling.NEAREST), (column*cell, top+24))
            draw.text((column*cell+4, top+13), role+' | '+str(case['metadata']['native_area'])+' native pixels', fill=(220, 225, 235))
    zoom_top = row_height*rows+32
    for row, role in enumerate(DISTANCES):
        case = preview['cases'].get(role)
        if case is None or case['crop'] is None: continue
        top = zoom_top+row*row_height; left, y0, right, bottom = case['crop']
        draw.text((4, top+3), role.upper()+' ROI | frozen38100 / candidate40100 / NEW / true Source128', fill=(235, 239, 245))
        values = [case['predictions'].get('sr38100'), case['predictions'].get('sr40100'),
            case['predictions']['current'], case['target']]
        for column, pixels in enumerate(values):
            if pixels is not None:
                canvas.paste(Image.fromarray(pixels[y0:bottom, left:right]).resize((cell, cell), Image.Resampling.NEAREST),
                    (column*cell, top+24))
    temporary = path.with_name(path.stem+'.tmp.png'); canvas.save(temporary, format='PNG'); replace_file(temporary, path)
    metadata = {**preview['metadata'], 'path': str(path), 'generated_at': now(), 'every': every,
        'native_lr_size': [32, 32], 'native_hr_size': [128, 128], 'preview_image_size': [width, height],
        'ROI_zoom_rows': list(DISTANCES), 'ROI_boxes_Source_labels_only': True,
        'caption': 'Actual NPR32 / frozen SR38100 / new SR128 / true same-tick Source128 / error; near-mid-far ROI comparisons'}
    if visual_first:
        metadata['caption'] = ('Actual NPR32 / frozen SR38100 / NEW with_hint / NEW no_hint (RSSM OFF) / '
                               'true same-tick Source128 / error; near-mid-far ROI comparisons')
    atomic_json(path.with_suffix('.json'), metadata)
    return metadata
