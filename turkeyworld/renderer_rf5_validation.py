"""Source-held-out RF5 evaluation, preserving the fixed RF1 comparison panels."""
from __future__ import annotations

from copy import deepcopy
import math

import torch

from .renderer_enemy_validation import evaluate_enemy_renderer, save_enemy_preview
from .renderer_geometry_objectives import geometry_wall_report, geometry_preservation_assessment
from .renderer_moe_objectives import moe_conditions
from .staged_validation import _isolated_validation


PROTOCOL = 'renderer_rf_expanded_experts_v1'


def _rename(value, steps=3):
    if isinstance(value, dict):
        return {key.replace('rf1', 'rf'+str(steps)): _rename(item, steps) for key, item in value.items()}
    if isinstance(value, list):
        return [_rename(item, steps) for item in value]
    if isinstance(value, str):
        return value.replace('RF1', 'RF'+str(steps))
    return value


def _same_support_comparison(current, initial):
    failures = []
    if (current.get('panel_fingerprint') != initial.get('panel_fingerprint')
            or current.get('geometry_panel_fingerprint') != initial.get('geometry_panel_fingerprint')
            or current.get('solver_steps') != initial.get('solver_steps')):
        raise ValueError('Expanded RF comparison requires the exact same panels and solver')
    key = 'rf'+str(current['solver_steps'])
    for name in ('overall', 'visible_enemy'):
        a, b = current[key][name], initial[key][name]
        if a['frames'] != b['frames']:
            failures.append(name + '_support_changed')
    now, old = current[key]['overall'], initial[key]['overall']
    if now['mae_mean'] > old['mae_mean'] + .002:
        failures.append('overall_mae_regressed')
    if now['ssim_mean'] < old['ssim_mean'] - .002:
        failures.append('overall_ssim_regressed')
    a = current[key]['visible_enemy'].get('foreground_mae')
    b = initial[key]['visible_enemy'].get('foreground_mae')
    if a is None or b is None or a >= b - .0005:
        failures.append('npc_foreground_not_improved')
    for category in ('empty_navigation', 'empty_hidden'):
        a = current[key]['categories'].get(category, {})
        b = initial[key]['categories'].get(category, {})
        if not a.get('frames') or a.get('frames') != b.get('frames'):
            failures.append(category + '_support_missing_or_changed')
        elif a['mae_mean'] > b['mae_mean'] + .002 or a['ssim_mean'] < b['ssim_mean'] - .01:
            failures.append(category + '_regressed')
    return failures


def evaluate_rf5(model, data, panel, geometry_panel, *, step, batch_size=8,
                 baseline=None, parent_rf1=None, inherited=None):
    """Compare five-step outputs with RF5@0; keep RF1 as a separate reference.

    Legacy image reports already use the module's real solver. Their old RF1
    field labels are converted only after computing the established metrics.
    The chart objective is evaluated separately and never replaced by RGB MAE.
    """
    from .renderer_rf5_objectives import rf5_refinement_loss
    steps = model.flow_renderer.steps
    if type(steps) is not int or steps not in (3,5):
        raise ValueError('Expanded RF validation requires its actual three/five-step sampler')
    if parent_rf1 is not None and parent_rf1.get('panel_fingerprint') != data.panel_fingerprint(panel):
        raise ValueError('RF parent reference uses a different held-out panel')
    report, arrays = evaluate_enemy_renderer(model, data, panel, step=step,
        batch_size=batch_size, small_enemy_rebalance=True, far_wall_supervision=True,
        conditions_fn=moe_conditions)
    _, geometry = evaluate_enemy_renderer(model, data, geometry_panel, step=step,
        batch_size=batch_size, small_enemy_rebalance=True, conditions_fn=moe_conditions)
    guard = geometry_wall_report(geometry)
    weighted, count = 0., 0
    with _isolated_validation(model, data, 173):
        for first in range(0, len(panel), batch_size):
            ids = panel[first:first + batch_size]
            batch = data.batch(len(ids), indices=ids)
            loss, _ = rf5_refinement_loss(model, batch, kind='main', inherited=inherited,
                                           checkpoint_velocity=False)
            value = float(loss.detach())
            if not math.isfinite(value):
                raise FloatingPointError('Nonfinite held-out RF5 objective')
            weighted += len(ids) * value
            count += len(ids)
    # Keep the unrenamed geometry schema for the established same-panel guard.
    preservation = (geometry_preservation_assessment(guard, baseline['_geometry_guard_raw'])
                    if baseline is not None else None)
    report = _rename(report, steps)
    report.update(protocol=PROTOCOL, solver='euler', solver_steps=steps,
        base_frozen=True, base_loss_in_train=False, codec_frozen=True,
        geometry_wall_guard_validation=_rename(guard, steps), _geometry_guard_raw=guard,
        geometry_panel_fingerprint=data.panel_fingerprint(geometry_panel),
        geometry_wall_preservation=preservation, val_loss=weighted / count,
        validation_objective='RF_MAIN_RGB_regions_crop_walls_latent_plus_mean_actual_steps_router',
        validation_objective_matches_main_training=True,
        validation_loss_excludes_far_geometry_auxiliary=True,
        automatic_active_promotion=False, gameplay_qualified=False)
    quality = report['rf'+str(steps)]['overall']
    fg = report['rf'+str(steps)]['visible_enemy'].get('foreground_mae')
    report['passed'] = bool(quality['ssim_mean'] >= .9 and quality['mae_mean'] <= .03
                            and fg is not None and fg <= .05)
    report['thresholds'] = dict(source_ssim_mean=.9, source_mae_mean=.03,
                                npc_foreground_mae=.05, applies_to=f'RF{steps} only; Base is frozen')
    if baseline is not None:
        failures = _same_support_comparison(report, baseline)
        if preservation is None or not preservation['passed']:
            failures.append('same_panel_wall_preservation_failed')
        if report['base'] != baseline['base']:
            failures.append('frozen_Base_metrics_changed')
        report['experimental_selection'] = dict(validated_improvement=not failures,
            failure_reasons=failures, reference=f'expanded_RF{steps}_before_any_optimizer_update',
            frozen_Base_requires_preservation_not_improvement=True)
    if parent_rf1 is not None:
        report['parent_rf1_reference'] = dict(
            training_steps=29609, solver_steps=1, summary=deepcopy(parent_rf1['summary']),
            foreground_mae=parent_rf1['rf1']['visible_enemy'].get('foreground_mae'),
            panel_fingerprint=parent_rf1['panel_fingerprint'],
            note='Different solver; this difference alone is not an SGD improvement')
    report['summary'] = report['rf'+str(steps)]['overall']
    return report, arrays


def save_rf5_preview(arrays, path, step, *, solver_steps=3):
    """Reuse the four-row image layout with truthful RF5 and frozen Base labels."""
    from PIL import Image, ImageDraw
    import numpy as np
    from pathlib import Path

    categories = arrays['categories']
    chosen = []
    for name in ('large', 'partial', 'distant', 'empty_navigation'):
        found = next((i for i, value in enumerate(categories) if value == name and i not in chosen), None)
        if found is not None:
            chosen.append(found)
    chosen += [i for i in range(len(categories)) if i not in chosen][:4 - len(chosen)]
    size = 192
    labels = [('source', 'SOURCE32'), ('base', 'FROZEN BASE'),
              ('rgb', f'RF{solver_steps} | {solver_steps} Euler steps'), ('codec_reference', 'FROZEN CODEC REFERENCE')]
    image = Image.new('RGB', (size * len(chosen), (size + 25) * 4), (12, 17, 25))
    draw = ImageDraw.Draw(image)
    for row, (key, title) in enumerate(labels):
        for column, index in enumerate(chosen):
            frame = arrays[key][index].detach().float().clamp(0, 1)
            pixels = frame.mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
            image.paste(Image.fromarray(np.asarray(pixels)).resize((size, size), Image.Resampling.NEAREST),
                        (column * size, row * (size + 25) + 25))
            draw.text((column * size + 3, row * (size + 25) + 3),
                      f'{title} g{step} | {categories[index]}', fill='white')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.png')
    image.save(temporary)
    temporary.replace(path)
