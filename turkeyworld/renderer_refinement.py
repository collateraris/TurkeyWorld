"""Source-supervised compact Base/RF1 refinement with frozen causal dynamics."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import copy

import numpy as np
import torch
from torch.nn import functional as F

from .base_student import export_base_student
from .object_losses import balanced_edge_l1, decoded_ssim_loss
from .renderer_student import _hash_tensors, export_renderer_student
from .staged_validation import _isolated_validation
from .visual_quality import image_quality_metrics


def configure_refinement(model):
    if (not hasattr(model, 'base_student') or model.flow_renderer.kind != 'student'
            or model.flow_renderer.steps != 1 or not model.config.navigation_only):
        raise ValueError('Source refinement requires the compact Base, RF1 and navigation-only profile')
    model.eval().requires_grad_(False)
    for name, parameter in model.named_parameters():
        if name.startswith('base_student.') or name.startswith('flow_renderer.') and not name.startswith('flow_renderer.codec.'):
            parameter.requires_grad_(True)
    model.base_student.train()
    model.flow_renderer.train()  # LatentFlowRenderer keeps the entire codec eval/frozen.
    return {'base_trainable': sum(p.numel() for p in model.base_student.parameters() if p.requires_grad),
        'flow_trainable': sum(p.numel() for p in model.flow_renderer.parameters() if p.requires_grad),
        'codec_parameters': sum(p.numel() for p in model.flow_renderer.codec.parameters()),
        'dynamics_frozen': True, 'map_geometry_appearance_frozen': True, 'codec_frozen': True,
        'solver': 'euler', 'solver_steps': 1}


def frozen_fingerprint(model):
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return _hash_tensors({name: value for name, value in model.state_dict().items() if name not in trainable})


def causal_conditions(model, batch, *, context=1):
    """Compute prior features before reading any target RGB or target pose."""
    device = next(model.parameters()).device
    current = batch['states'][:, context-1].to(device).float()
    controls = batch['actions'][:, 0].to(device).float()
    grid = batch['map'].to(device)
    with torch.no_grad():
        map_features = model.encode_map(grid)
        # Restrict RGB/state observations to the genuinely recorded past.
        observed = batch['frames'][:, :context].to(device).float()
        memory = model.initialize_from_image(observed, batch['states'][:, :context].to(device).float(), map_features)
        prior = model.transition(current, controls, map_features, memory)
        features = prior['features'].detach()
        state = batch['target_states'][:, 0].to(device).float().detach()
        geometry = {key: value.detach() for key, value in model.vjepa_backend.geometry(state, map_features).items()
                    if torch.is_tensor(value)}
    # The future image is a loss label, never an observed posterior input.
    target = batch['target_frames'][:, 0].to(device).float().detach()
    return {'state': state, 'features': features, 'geometry': geometry, 'map': grid,
            'target': target, 'prior_memory': prior['memory'].detach()}


def refinement_forward(model, conditions, *, detach_base=True, return_residual=False):
    from .models import apply_pixel_residual
    if getattr(model.base_student.config, 'pixel_residual', False):
        base, residual = model.base_student.forward_with_residual(
            conditions['state'], conditions['features'], conditions['geometry'])
    else:
        base = model.base_student(conditions['state'], conditions['features'], conditions['geometry'])
        residual = None
    flow = model.flow_renderer
    endpoint = flow.latent_sample(base.detach() if detach_base else base,
        conditions['state'], noise=flow.fixed_noise, map_condition=conditions['map'])
    quantized = flow.codec.quantize(flow.denormalize_latents(endpoint))['quantized']
    rgb = apply_pixel_residual(flow.codec.decode(quantized), residual)
    return (base, rgb, endpoint, residual) if return_residual else (base, rgb, endpoint)


def refinement_loss(model, batch, *, context=1, ssim_weight=.2, edge_weight=.1,
                    latent_weight=0., detach_base=True):
    conditions = causal_conditions(model, batch, context=context)
    base, rgb, endpoint = refinement_forward(model, conditions, detach_base=detach_base)
    target = conditions['target']
    base_l1, rgb_l1 = F.l1_loss(base, target), F.l1_loss(rgb, target)
    base_ssim, rgb_ssim = decoded_ssim_loss(base, target), decoded_ssim_loss(rgb, target)
    base_edge, rgb_edge = balanced_edge_l1(base, target), balanced_edge_l1(rgb, target)
    latent = endpoint.sum()*0
    if latent_weight:
        with torch.no_grad():
            target_latent = model.flow_renderer.normalize_latents(model.flow_renderer.codec.encode(target)['quantized'])
        latent = F.mse_loss(endpoint, target_latent)
    total = base_l1+rgb_l1+ssim_weight*(base_ssim+rgb_ssim)+edge_weight*(base_edge+rgb_edge)+latent_weight*latent
    metrics = {'train_loss': float(total.detach()), 'base_source_l1': float(base_l1.detach()),
        'rf1_source_l1': float(rgb_l1.detach()), 'base_source_ssim_loss': float(base_ssim.detach()),
        'rf1_source_ssim_loss': float(rgb_ssim.detach()), 'base_source_edge': float(base_edge.detach()),
        'rf1_source_edge': float(rgb_edge.detach()), 'source_latent_mse': float(latent.detach()),
        'flow_base_detached': detach_base, 'solver_steps': 1,
        'future_source_rgb_in_prior': False, 'source_pose_aligned': True}
    return total, metrics


def evaluate_refinement(model, data, panel, *, step=0, context=1, batch_size=8,
                        baseline=None, sr=None, target_ssim=.90, target_mae=.03):
    base_images, rgb_images, sources, sr_images = [], [], [], []
    with _isolated_validation(model, data, 173):
        for first in range(0, len(panel), batch_size):
            batch = data.batch(len(panel[first:first+batch_size]), indices=panel[first:first+batch_size], context=context)
            conditions = causal_conditions(model, batch, context=context)
            base, rgb, _ = refinement_forward(model, conditions)
            if not all(bool(torch.isfinite(value).all()) for value in (base, rgb, conditions['target'])):
                raise FloatingPointError('Nonfinite Source renderer validation')
            base_images.append(base.float().cpu()); rgb_images.append(rgb.float().cpu()); sources.append(conditions['target'].cpu())
            if sr is not None and first == 0:
                with torch.no_grad():
                    enhanced = sr(rgb[:4])
                if enhanced.shape[-2:] != (128, 128) or not bool(torch.isfinite(enhanced).all()):
                    raise ValueError('Frozen SR must produce finite native128 output')
                sr_images.append(enhanced.float().cpu())
    arrays = {'base': torch.cat(base_images), 'rgb': torch.cat(rgb_images), 'source': torch.cat(sources)}
    def quality(prediction, target):
        measured = image_quality_metrics(prediction, target)
        return {name+'_mean': float(value.mean()) for name, value in measured.items() if name in {'psnr', 'ssim', 'mae'}}
    summaries = {'base': quality(arrays['base'], arrays['source']), 'rf1': quality(arrays['rgb'], arrays['source'])}
    report = {'step': step, 'frames': len(panel), 'source_held_out': True,
        'scope': 'Native32 Source RGB at GT next poses with frozen causal-prior appearance features',
        'base': summaries['base'], 'rf1': summaries['rf1'], 'summary': summaries['rf1'],
        'passed': all(value['ssim_mean'] >= target_ssim and value['mae_mean'] <= target_mae for value in summaries.values()),
        'thresholds': {'ssim_mean': target_ssim, 'mae_mean': target_mae, 'applies_to': 'both Base and RF1 versus Source'},
        'prior_future_source_rgb': False, 'future_source_state_in_prior': False,
        'gameplay_qualified': False, 'sr_quality_evaluated': False,
        'panel_fingerprint': data.panel_fingerprint(panel), 'solver_steps': 1}
    if baseline is not None:
        if not torch.equal(baseline['source'], arrays['source']):
            raise ValueError('Baseline/finetuned Source validation samples changed')
        report['old_rf1_comparison'] = quality(arrays['rgb'], baseline['rgb'])
        report['old_rf1_source_quality'] = quality(baseline['rgb'], baseline['source'])
    if sr_images:
        arrays['sr'] = torch.cat(sr_images)
    return report, arrays


def export_refined_bundle(source, working, directory, *, updates, report, initial_renderer_steps,
                          initial_base_steps, provenance):
    """Rebind the newly validated Base while retaining original codec truth."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    base_path, renderer_path = directory/'base_student.pt', directory/'renderer_student.pt'
    base_metrics = {'source_validation': report, 'passed': report['passed'], 'refinement_updates': updates,
                    'criterion': 'Source RGB; previous imitation gate is not reused'}
    export_base_student(source, working.base_student, base_path,
        training_steps=initial_base_steps+updates, metrics=base_metrics,
        provenance={**provenance, 'refinement_updates': updates, 'initial_training_steps': initial_base_steps})
    base_bundle = torch.load(base_path, map_location='cpu', weights_only=False)
    source.base_student = working.base_student
    source.base_student_bundle = base_bundle
    source.base_student_metadata = {key: value for key, value in base_bundle.items() if key != 'model'}
    working.flow_renderer._renderer_base_bundle = copy.deepcopy(base_bundle)
    working.flow_renderer._renderer_base_identity = _hash_tensors(working.base_student.state_dict())
    metrics = {'source_validation': report, 'source_quality_passed': report['passed'],
               'refinement_updates': updates, 'initial_training_steps': initial_renderer_steps,
               'criterion': 'Source RGB; old RF1 imitation is reported separately'}
    return export_renderer_student(source, working.flow_renderer, renderer_path,
        training_steps=initial_renderer_steps+updates, metrics=metrics)
