"""Opt-in, class-balanced physical contacts; no inference corrections."""
from contextlib import nullcontext
import copy
import math
import random

import numpy as np
import torch
from torch.nn import functional as F

from .motion_recovery_training import pass_fraction_targets, balanced_pass_weights
from .trajectory_objectives import circle_wall_penalty


CONTACT_PROTOCOL_V2 = {'version': 2, 'pass_bce_weight': 1., 'physical_motion_weight': 7.,
    'endpoint_margin_weight': 2., 'contact_wall_weight': 2., 'autonomous_wall_weight': 2.,
    'autonomous_xy_weight': 1.,
    'wall_margin_tiles': .02,
    'wall_normalizer_tiles': .20, 'source_stop_fraction': .05, 'source_free_fraction': .95,
    'physical_error_fraction': .20, 'segment_samples': 5}


def normalize_contact_protocol(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value)-set(CONTACT_PROTOCOL_V2):
        raise ValueError('Contact protocol must contain only known explicit v2 settings')
    result = {**CONTACT_PROTOCOL_V2, **value}
    if (result['version'] != 2 or isinstance(result['version'], bool)
            or not isinstance(result['segment_samples'], int) or isinstance(result['segment_samples'], bool)
            or result['segment_samples'] < 2):
        raise ValueError('Contact protocol requires version2 and at least2 segment samples')
    for key, item in result.items():
        if key in ('version', 'segment_samples'): continue
        if not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(item) or item < 0:
            raise ValueError('Contact settings must be finite and nonnegative: '+key)
    if (not 0 <= result['source_stop_fraction'] < result['source_free_fraction'] <= 1
            or not 0 < result['physical_error_fraction'] < 1 or result['wall_normalizer_tiles'] <= 0):
        raise ValueError('Invalid contact thresholds or wall normalization')
    return result


def prepare_contact_targets(previous, target, action, *, contact_protocol, dt=1/30):
    """Global BxH detached labels and equal disjoint Source-cohort mass."""
    config = normalize_contact_protocol(contact_protocol)
    if config is None: raise ValueError('Contact targets require an opt-in v2 protocol')
    if previous.ndim == 2:
        previous, target, action = previous[:, None], target[:, None], action[:, None]
    if previous.ndim != 3 or target.shape != previous.shape or action.shape != (*previous.shape[:2], 4):
        raise ValueError('Contact labels require matching[B,H,state] Source pairs/controls')
    previous, target, action = previous.detach().float(), target.detach().float(), action.detach().float()
    batch, horizon = previous.shape[:2]
    fraction, active, command = pass_fraction_targets(previous.flatten(0, 1), target.flatten(0, 1),
                                                      action.flatten(0, 1), dt=dt)
    fraction, active, command = (value.reshape(batch, horizon, 2) for value in (fraction, active, command))
    free = active & (fraction >= config['source_free_fraction'])
    blocked = active & (fraction <= config['source_stop_fraction'])
    partial = active & ~free & ~blocked
    slide = free.any(-1) & blocked.any(-1) & ~partial.any(-1)
    groups = {'free': free.any(-1) & ~blocked.any(-1) & ~partial.any(-1),
              'blocked': blocked.any(-1) & ~free.any(-1) & ~partial.any(-1),
              'slide': slide, 'partial': partial.any(-1)}
    present = sum(bool(mask.any()) for mask in groups.values())
    sample_weight = torch.zeros_like(fraction[..., 0])
    for mask in groups.values():
        sample_weight += mask.to(fraction.dtype)/mask.sum().clamp_min(1)/max(present, 1)
    axis_weight = sample_weight[..., None]*active/active.sum(-1, keepdim=True).clamp_min(1)
    return {'fraction': fraction, 'active': active, 'command': command, 'free': free, 'blocked': blocked,
        'slide': slide, 'sample_weight': sample_weight.detach(), 'axis_weight': axis_weight.detach(),
        'pass_weight': balanced_pass_weights(fraction, active),
        'margin_weight': balanced_pass_weights(fraction, free | blocked),
        'groups': {name: int(mask.sum()) for name, mask in groups.items()}, 'dt': dt}


def _swept(old, new, samples):
    """Normalized XY -> map tiles, endpoint and both legal X-then-Y legs."""
    old, new = old.float()*16, new.float()*16
    elbow = torch.stack([new[..., 0], old[..., 1]], -1)
    fractions = torch.linspace(0, 1, samples, device=new.device, dtype=new.dtype)[1:-1]
    shape = [1]*(new.ndim-1)+[len(fractions), 1]
    fractions = fractions.reshape(shape)
    first = old[..., None, :]+fractions*(elbow-old)[..., None, :]
    second = elbow[..., None, :]+fractions*(new-elbow)[..., None, :]
    return torch.cat([first, elbow[..., None, :], second, new[..., None, :]], -2)


def swept_player_wall_loss(old, predicted, reference_old, reference, grid, sample_weight, *, contact_protocol):
    """Extra circle penetration relative to the genuine legal Source path.

    A small safety margin never penalizes an unavoidable Source near-contact:
    its detached reference margin penalty is subtracted before ReLU. Radius
    normalization limits deep-wall gradient scales; Source weights prevent a
    predicted life flag or abundant free samples from hiding collisions.
    """
    config = normalize_contact_protocol(contact_protocol)
    radius = .20+config['wall_margin_tiles']
    points = _swept(old, predicted, config['segment_samples'])
    source = _swept(reference_old.detach(), reference.detach(), config['segment_samples'])
    penalty = circle_wall_penalty(points, grid, radius)
    with torch.no_grad():
        baseline = circle_wall_penalty(source, grid, radius)
    extra = F.relu(penalty-baseline)/config['wall_normalizer_tiles']**2
    return (extra.mean(-1)*sample_weight.detach()).sum()


def differentiable_contact_terms(prior, previous, target, action, grid, *, contact_protocol,
                                 supervision=None, index=0):
    """Unweighted scalar BCE, physical delta, pass-margin and swept geometry."""
    config = normalize_contact_protocol(contact_protocol)
    labels = (prepare_contact_targets(previous, target, action, contact_protocol=config)
              if supervision is None else supervision)
    logits = prior['player_pass_logits'].float()
    probability = logits.sigmoid()
    expected = target.detach().float()[:, :2]-previous.detach().float()[:, :2]
    displacement = prior['state'].float()[:, :2]-previous.detach().float()[:, :2]
    fraction, active = labels['fraction'][:, index], labels['active'][:, index]
    scale = 16/(2.4*labels['dt'])
    physical = F.smooth_l1_loss(displacement*scale, expected*scale, reduction='none')
    margin = (F.relu(config['source_free_fraction']-probability).square()*labels['free'][:, index]
              +F.relu(probability-config['source_stop_fraction']).square()*labels['blocked'][:, index])
    return {'pass_bce': (F.binary_cross_entropy_with_logits(logits, fraction, reduction='none')
                         *labels['pass_weight'][:, index]).sum(),
        'physical_motion': (physical*labels['axis_weight'][:, index]).sum(),
        'endpoint_margin': (margin*labels['margin_weight'][:, index]).sum(),
        'contact_wall': swept_player_wall_loss(previous[:, None, :2], prior['state'][:, None, :2],
            previous.detach()[:, None, :2], target.detach()[:, None, :2], grid,
            labels['sample_weight'][:, index:index+1], contact_protocol=config)}


def contact_gradient_probe(model, batch, *, context=1, observed_context=1, contact_protocol,
                           parameter_scope='all', seed=173):
    """Measure real component gradient norms without Adam/.grad/RNG changes."""
    config = normalize_contact_protocol(contact_protocol)
    if parameter_scope not in ('all', 'player_head'):
        raise ValueError('Contact gradient scope must be all or player_head')
    states, actions, targets, frames, grid = (batch[name].detach() for name in
        ('states', 'actions', 'target_states', 'frames', 'map'))
    horizon = actions.shape[1]
    if not 1 <= horizon <= 16 or not 1 <= observed_context <= context:
        raise ValueError('Contact probe requires1..16 Source transitions and genuine observed context')
    previous = states[:, context-1:context-1+horizon].float()
    supervision = prepare_contact_targets(previous, targets, actions, contact_protocol=config)
    modes = [(module, module.training) for module in model.modules()]
    caches = [(module, name, hasattr(module, name), (getattr(module, name).detach().clone()
        if torch.is_tensor(getattr(module, name, None)) else getattr(module, name, None)))
        for module in model.modules() for name in ('_map_condition', '_spawn_walkable')]
    numpy_state, python_state = np.random.get_state(), random.getstate()
    device = next(model.parameters()).device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    parameters = [parameter for name, parameter in model.named_parameters() if parameter.requires_grad
        and (parameter_scope == 'all' or 'player_pass_head.' in name)]
    if not parameters: raise ValueError('No trainable parameters in the requested contact probe scope')
    result = {}
    try:
        with torch.random.fork_rng(devices=devices), torch.enable_grad():
            torch.random.default_generator.manual_seed(seed)
            if device.type == 'cuda': torch.cuda.manual_seed(seed)
            with torch.no_grad(): features = model.encode_map(grid).detach()
            memory = model.initialize_from_image(frames[:, context-observed_context:context],
                states[:, context-observed_context:context], features)
            totals = {}
            for index in range(horizon):
                if index and index % 4 == 0: memory = memory.detach()
                observed = model.vjepa_backend.observe_step(previous[:, index], actions[:, index], features,
                    memory, frames[:, context+index])
                memory = observed['memory']
                parts = differentiable_contact_terms(observed['prior_result'], previous[:, index], targets[:, index],
                    actions[:, index], grid, contact_protocol=config, supervision=supervision, index=index)
                for name, value in parts.items(): totals[name] = totals.get(name, 0)+value
            for name, value in totals.items():
                gradients = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
                norms = [torch.linalg.vector_norm(gradient.detach(), dtype=torch.float64)
                         for gradient in gradients if gradient is not None]
                norm = torch.linalg.vector_norm(torch.stack(norms)) if norms else torch.tensor(0., dtype=torch.float64)
                if not bool(torch.isfinite(norm)): raise FloatingPointError('Nonfinite contact probe gradient: '+name)
                weight = config[name+'_weight']
                result[name] = {'loss': float(value.detach()), 'weight': weight,
                    'unweighted_norm': float(norm), 'weighted_norm': float(norm)*weight}
    finally:
        for module, mode in modes: module.training = mode
        for module, name, existed, value in caches:
            if existed: setattr(module, name, value)
            elif hasattr(module, name): delattr(module, name)
        np.random.set_state(numpy_state); random.setstate(python_state)
    return {'components': result, 'cohorts': supervision['groups'], 'parameter_scope': parameter_scope,
        'contact_protocol': config, 'parameter_gradients_mutated': False, 'optimizer_steps': 0,
        'rng_restored': True, 'source_windows': len(states), 'horizon': horizon}
