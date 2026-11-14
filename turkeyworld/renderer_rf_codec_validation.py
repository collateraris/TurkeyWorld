"""Native32 held-out RF3/Q/current-codec comparisons, with no restorer."""
from contextlib import contextmanager
from copy import deepcopy
import math

import torch

from .decoder_rf_adaptation_validation import (
    _identity, _plain, _preservation, _geometry_preservation)
from .renderer_enemy_validation import renderer_metrics
from .renderer_geometry_objectives import geometry_wall_terms, geometry_wall_protocol
from .renderer_moe_objectives import moe_conditions
from .renderer_rf_codec_bundle import PROTOCOL, _names
from .renderer_rf_codec_objectives import (
    rf_codec_forward, rf_codec_refinement_loss, renderer_rf_codec_protocol)
from .renderer_student import _hash_tensors
from .renderer_wall_depth import source_wall_depth_labels
from .renderer_wall_supervision import source_wall_masks, wall_loss_terms
from .staged_validation import _isolated_validation

KIND = 'renderer_rf_codec_validation_v1'
PREVIEW_KIND = 'renderer_rf_codec_GT_query'
BRANCHES = ('base', 'rf_old', 'rf_new', 'source_codec')


@contextmanager
def _readonly(model, reference, data):
    if getattr(data, 'split', None) not in ('val', 'test'):
        raise ValueError('RF/codec validation requires explicit held-out Source data')
    for net in (model, reference):
        if net.flow_renderer.steps != 3:
            raise ValueError('RF/codec comparison needs actual three-step samplers')
        _names(net.flow_renderer)
    allowed = {'flow_renderer.'+n for n in _names(model.flow_renderer)}
    if (any(p.requires_grad and n not in allowed for n,p in model.named_parameters())
            or any(p.requires_grad for p in reference.parameters())):
        raise ValueError('Only current RF may train; the complete reference and codec must be frozen')
    codecs = [_hash_tensors(n.flow_renderer.codec.state_dict()) for n in (model, reference)]
    bases = [_hash_tensors(n.base_student.state_dict()) for n in (model, reference)]
    if len(set(codecs)) != 1 or len(set(bases)) != 1:
        raise ValueError('RF comparison must share the exact same frozen current D/E/Q and Base')
    nets = (model, reference)
    hashes = [(_hash_tensors(n.flow_renderer.state_dict()), _hash_tensors(n.base_student.state_dict())) for n in nets]
    sampler = deepcopy(data.sampler_state()) if callable(getattr(data, 'sampler_state', None)) else None
    try:
        with _isolated_validation(model, data, 173), _isolated_validation(reference, data, 173):
            yield codecs[0]
    finally:
        if sampler is not None:
            data.restore_sampler(sampler)
        if hashes != [(_hash_tensors(n.flow_renderer.state_dict()), _hash_tensors(n.base_student.state_dict())) for n in nets]:
            raise RuntimeError('Read-only RF validation changed renderer weights or buffers')


def _collect(model, reference, data, panel, batch_size, objective_fn=None):
    parts = {key:[] for key in (*BRANCHES, 'source', 'mask', 'background', 'states', 'map', 'dynamic')}
    total, frames, sums = 0., 0, {}
    for first in range(0, len(panel), batch_size):
        ids = panel[first:first+batch_size]
        batch = data.batch(len(ids), indices=ids)
        current = moe_conditions(model, batch)
        old = moe_conditions(reference, batch)
        base, rgb, _, _ = rf_codec_forward(model, current, checkpoint_velocity=False)
        old_base, old_rgb, _, _ = rf_codec_forward(reference, old, checkpoint_velocity=False)
        if not torch.equal(base, old_base):
            raise ValueError('Frozen Base/causal feature outputs differ across the RF comparison')
        source_q = model.flow_renderer.codec.encode(current['target'].detach())['quantized']
        decoded = model.flow_renderer.codec.decode(source_q)
        values = dict(base=base, rf_old=old_rgb, rf_new=rgb, source_codec=decoded,
            source=current['target'], mask=current['object_mask'], background=current['background'],
            states=current['state'], map=current['map'], dynamic=current['dynamic'])
        if any(not bool(torch.isfinite(v).all()) for v in values.values()):
            raise FloatingPointError('Nonfinite native32 RF/codec comparison')
        for name, value in values.items():
            parts[name].append(value.detach().cpu())
        if objective_fn is not None:
            loss, metrics = objective_fn(model, batch, kind='main', checkpoint_velocity=False)
            value = float(loss.detach())
            if not math.isfinite(value):
                raise FloatingPointError('Nonfinite actual held-out MAIN objective')
            total += len(ids)*value
            frames += len(ids)
            def visit(item, path=()):
                if isinstance(item, dict):
                    for key, value in item.items():visit(value, (*path, key))
                elif type(item) in (int, float):
                    if not math.isfinite(item):raise FloatingPointError('Nonfinite MAIN component')
                    support = path[-1].endswith(('_images', '_pixels'))
                    sums[path] = sums.get(path, 0.)+item*(1 if support else len(ids))
            visit(metrics)
    arrays = {key:torch.cat(value) for key,value in parts.items()}
    arrays['categories'] = [str(data.rows[int(i)]['category']) for i in panel]
    arrays['row_indices'] = list(panel)
    components = {}
    for path, value in sums.items():
        row = components
        for key in path[:-1]:row = row.setdefault(key, {})
        row[path[-1]] = int(value) if path[-1].endswith(('_images', '_pixels')) else value/frames
    return arrays, dict(loss=total/frames if frames else None, frames=frames, components=components)


def _measure(arrays):
    labels = source_wall_masks(arrays['map'], arrays['states'], arrays['mask'])
    result = {}
    for name in BRANCHES:
        raw = renderer_metrics(arrays[name], arrays['source'], arrays['mask'], arrays['background'],
                               arrays['categories'], arrays['dynamic'])
        wall = _plain(wall_loss_terms(arrays[name], arrays['source'], labels))
        summary = dict(raw['overall'], wall_boundary_l1=wall['boundary_l1'] if wall['boundary_images'] else None,
                       wall_boundary_frames=wall['boundary_images'])
        result[name] = dict(summary=summary, overall=deepcopy(summary), cohorts=raw['categories'],
            visible_enemy=raw['visible_enemy'], wall=wall,
            static_zero_features=raw['static_zero_features'],
            dynamic_past_prior_features=raw['dynamic_past_prior_features'])
    return result


def evaluate_rf_codec(model, reference_model, data, panel, geometry_panel=None, *, step,
                      batch_size=8, baseline=None, objective_fn=None):
    panel = [int(i) for i in panel]
    geometry_panel = None if geometry_panel is None else [int(i) for i in geometry_panel]
    if (type(step) is not int or step < 0 or type(batch_size) is not int or batch_size < 1
            or not panel or len(set(panel)) != len(panel)
            or geometry_panel is not None and (not geometry_panel or len(set(geometry_panel)) != len(geometry_panel))):
        raise ValueError('RF/codec validation needs unique nonempty panels and a positive batch size')
    identity, identity_sha = _identity(data)
    panel_sha = data.panel_fingerprint(panel)
    geometry_sha = data.panel_fingerprint(geometry_panel) if geometry_panel is not None else None
    with _readonly(model, reference_model, data) as codec_sha:
        if baseline is not None and any(baseline.get(k) != v for k,v in dict(kind=KIND,
            protocol=PROTOCOL, panel_fingerprint=panel_sha, geometry_panel_fingerprint=geometry_sha,
            data_identity_sha256=identity_sha, codec_sha256=codec_sha, solver_steps=3).items()):
            raise ValueError('RF/codec baseline must use identical panels, data and frozen current decoder')
        arrays, objective = _collect(model, reference_model, data, panel, batch_size,
                                     objective_fn or rf_codec_refinement_loss)
        models = _measure(arrays)
        geometry = None
        if geometry_panel is not None:
            geo, _ = _collect(model, reference_model, data, geometry_panel, batch_size)
            labels = source_wall_depth_labels(geo['map'], geo['states'], geo['mask'])
            measured = {name:dict(bands=_plain(geometry_wall_terms(geo[name], geo['source'], labels)))
                        for name in BRANCHES}
            geometry = dict(frames=len(geometry_panel), panel_fingerprint=geometry_sha,
                protocol=geometry_wall_protocol(), models=measured,
                rf_preservation=_geometry_preservation(measured['rf_new'], measured['rf_old']))
    if baseline is not None and any(models[name] != baseline['models'][name]
                                   for name in ('base', 'rf_old', 'source_codec')):
        raise ValueError('Frozen Base/RF/Source-codec outputs differ from the pinned initial comparison')
    rf = _preservation(models['rf_new']['summary'], models['rf_old']['summary'])
    if geometry is not None and not geometry['rf_preservation']['passed']:
        rf['passed'] = False
        rf['failure_reasons'].append('same_panel_RF_geometry_regressed')
    summary = models['rf_new']['summary']
    goal_met = len(panel) == 256 and summary['ssim_mean'] > .85
    objective.update(protocol=PROTOCOL, input_kind='RF3_terminal_Q_frozen_current_decoder',
        aggregation='batch_image_weighted', matches_training_formula=True, source_held_out=True,
        main_only=True, excludes_far_geometry_auxiliary=True,
        Source_Q_image_branch=False, weights=renderer_rf_codec_protocol()['weights'])
    report = dict(kind=KIND, protocol=PROTOCOL, step=step, frames=len(panel), native_resolution=32,
        solver='euler', solver_steps=3, models=models, summary=summary, geometry=geometry,
        panel_fingerprint=panel_sha, geometry_panel_fingerprint=geometry_sha,
        data_identity=identity, data_identity_sha256=identity_sha, codec_sha256=codec_sha,
        full_codec_tensors_unchanged=True, Base_tensors_unchanged=True,
        reference_RF_tensors_unchanged=True, restorer_active=False,
        source_sampler_unchanged=True, source_held_out=True, renderer_only=True,
        Source_RGB_into_RF_conditioning=False, future_source_rgb_in_prior=False,
        future_source_pose_in_prior=False, world_rollout=False, autonomous_rollout=False,
        Source_Q_image_branch=False, sr_quality_evaluated=False,
        gameplay_qualified=False, automatic_active_promotion=False,
        scope='Native32 GT-pose RF3→Q→same frozen current D; Source-E/Q→D diagnostic only',
        source_preservation=dict(passed=True, frozen_codec=True, frozen_source_codec_outputs=True),
        rf_preservation=rf, val_loss=objective['loss'], validation_objective=objective,
        validation_objective_matches_main_training=True,
        passed=goal_met, quality_passed=False,
        rf_codec_ssim_goal=dict(target=.85, comparison='strict_greater_than', frames=256,
            measured_frames=len(panel), metric='models.rf_new.summary.ssim_mean',
            measured_ssim=summary['ssim_mean'], goal_met=goal_met,
            geometry_preservation_is_goal_requirement=False, gameplay_quality_qualified=False))
    if baseline is not None:report['baseline_step'] = baseline['step']
    arrays.update(panel_fingerprint=panel_sha, step=step)
    return report, arrays


def save_rf_codec_preview(arrays, path, step):
    from pathlib import Path
    from PIL import Image, ImageDraw
    selected = []
    for category in ('large', 'partial', 'distant', 'empty_navigation'):
        index = next((i for i,c in enumerate(arrays['categories']) if c == category and i not in selected), None)
        if index is not None:selected.append(index)
    selected += [i for i in range(len(arrays['categories'])) if i not in selected][:4-len(selected)]
    rows = [('source','SOURCE32'), ('base','FROZEN BASE'), ('rf_old','FROZEN RF3 | D1725'),
            ('rf_new','TRAINED RF3 | SAME D1725'), ('source_codec','SOURCE E/Q | SAME D1725')]
    image = Image.new('RGB', (192*len(selected), 217*len(rows)), (12,17,25))
    draw = ImageDraw.Draw(image)
    for row,(key,label) in enumerate(rows):
        for column,index in enumerate(selected):
            value = arrays[key][index].detach().float().clamp(0,1)
            pixels = value.mul(255).round().byte().permute(1,2,0).numpy()
            image.paste(Image.fromarray(pixels).resize((192,192), Image.Resampling.NEAREST),
                        (column*192,row*217+25))
            draw.text((column*192+3,row*217+3), f'{label} g{step} | {arrays["categories"][index]}',fill='white')
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+'.tmp.png');image.save(temporary);temporary.replace(path)
    return dict(kind=PREVIEW_KIND, protocol=PROTOCOL, step=step,
        panel_fingerprint=arrays['panel_fingerprint'], row_count=5,
        rows=[key for key,_ in rows], row_labels=[title for _,title in rows],
        row_indices=[arrays['row_indices'][i] for i in selected],
        columns=[arrays['categories'][i] for i in selected], native_resolution=32,
        restorer_active=False, solver_steps=3, source_held_out=True, renderer_only=True,
        world_rollout=False, future_source_rgb_in_prior=False, future_source_pose_in_prior=False,
        sr_quality_evaluated=False, gameplay_qualified=False, enlargement='nearest_neighbor_display_only')
