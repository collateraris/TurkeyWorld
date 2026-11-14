"""Strict loss-only continuation of an RGB enemy restoration experiment.

The new run has its own objective ledger and held-out baseline. Model, Adam,
private sampler and process RNG continue from a complete source checkpoint.
"""
from copy import deepcopy

import torch

from .neural_enemy_restoration_sr import frozen_background_state
from .renderer_student import _hash_tensors


KIND = 'neural_enemy_restoration_loss_continuation_v1'


def continuation_source(source, base_plan, model):
    """Validate the existing architecture/exposure without changing any state."""
    old = source.get('training_plan')
    step = source.get('training_steps')
    if (not isinstance(old, dict) or type(step) is not int or step < 1
            or source.get('config') != model.config
            or old.get('config') != model.config
            or source.get('standalone_RGB_only') is not True
            or source.get('atlas_in_new_model') is not False
            or source.get('disposable_probe') is not False
            or not isinstance(source.get('model'), dict)
            or not isinstance(source.get('optimizer'), dict)):
        raise ValueError('Continuation needs a complete production RGB-only restoration checkpoint')
    for key in ('train', 'val', 'panel', 'panel_fingerprint', 'batch', 'seed',
                'parent_checkpoint_sha256'):
        if old.get(key) != base_plan.get(key):
            raise ValueError('Restoration continuation changed source ' + key)
    if (source.get('plan_sha256') is None
            or source.get('neural_enemy_restoration_updates') != step
            or (source.get('sampler') or {}).get('draws') != step * base_plan['batch']
            or not all(key in source for key in ('rng_python', 'rng_numpy', 'rng_torch', 'rng_cuda'))):
        raise ValueError('Restoration continuation lost its complete source clocks or RNG')
    expected_names = [name for name, p in model.named_parameters() if p.requires_grad]
    if source.get('optimizer_parameter_names') != expected_names:
        raise ValueError('Restoration continuation changed trainable parameter names/order')
    # Strict tensor checks also reject truncated model dictionaries, NaNs and
    # frozen-background substitutions before allocating a new run's ledger.
    expected = model.state_dict()
    actual = source['model']
    if set(actual) != set(expected):
        raise ValueError('Restoration continuation model tensor names changed')
    for name, value in expected.items():
        other = actual[name]
        if (not torch.is_tensor(other) or value.shape != other.shape or value.dtype != other.dtype
                or not bool(torch.isfinite(other).all())):
            raise ValueError('Restoration continuation incompatible tensor: ' + name)
    background = {name: actual[name] for name in frozen_background_state(model)}
    if _hash_tensors(background) != base_plan['migration']['copied_background_sha256']:
        raise ValueError('Restoration continuation changed its frozen background')
    return dict(kind=KIND, training_steps=step, plan_sha256=source['plan_sha256'],
        model_sha256=_hash_tensors(actual), sampler_draws=source['sampler']['draws'],
        optimizer_parameter_names=deepcopy(expected_names),
        optimizer_state_count=len(source['optimizer']['state']),
        optimizer_reset=False, weights_reset=False, sampler_reset=False, RNG_reset=False,
        previous_best=deepcopy(source.get('best')), previous_best_used_for_new_objective=False)


def validate_continuation_checkpoint(source, plan):
    """Reject a resume source replaced after immutable plan preparation."""
    pin = plan['resume_source']
    if (source.get('training_steps') != pin['training_steps']
            or source.get('plan_sha256') != pin['plan_sha256']
            or source.get('config') != plan['config']
            or _hash_tensors(source.get('model', {})) != pin['model_sha256']):
        raise ValueError('Restoration continuation checkpoint no longer matches its pinned source')

