"""Strict, explicit v5-to-v6 renderer continuation with the complete old Adam.

No checkpoint is read here. The caller supplies the pinned completed parent
and persists the returned receipt. All RNG streams and the existing Source
sampler's private PCG clock continue from that completed boundary.
"""
from __future__ import annotations

import copy
import random
import re

import numpy as np
import torch


CONFIGURATION_CHANGES = frozenset({'run_dir', 'steps', 'native_phase_parent', 'far_wall_supervision'})
PARENT_PROTOCOL = 'one_enemy_native_area_balanced_renderer_v5'
QUOTAS = frozenset({'tiny_1_16', 'small_17_64', 'far', 'replay'})
IDENTITY_ADDITIONS = frozenset({'far_wall_protocol', 'far_wall_aux_pool'})


def _same(left, right):
    if torch.is_tensor(left):
        return (torch.is_tensor(right) and left.dtype == right.dtype and left.shape == right.shape
                and torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_same(v, right[k]) for k, v in left.items())
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def _sampler_view(data):
    """Clone mutable sampler state without duplicating immutable Source arrays."""
    # copy.copy probes __setstate__ before populating __dict__; forwarding
    # wrappers implement __getattr__ through parent and recurse at that point.
    result = object.__new__(type(data))
    result.__dict__ = dict(data.__dict__)
    for key, value in data.__dict__.items():
        if key == 'parent':
            setattr(result, key, _sampler_view(value))
        elif key == 'rng' or key.endswith('_counts'):
            setattr(result, key, copy.deepcopy(value))
    return result


def _validate_rng(saved):
    for key in ('rng_torch', 'rng_numpy', 'rng_python', 'rng_cuda'):
        if key not in saved:
            raise ValueError('Native renderer parent lacks complete RNG state: '+key)
    value = saved['rng_torch']
    if not torch.is_tensor(value) or value.dtype != torch.uint8 or value.ndim != 1 or value.numel() < 1:
        raise ValueError('Invalid parent CPU Torch RNG state')
    value = saved['rng_numpy']
    if not isinstance(value, tuple) or len(value) != 5 or not isinstance(value[1], np.ndarray):
        raise ValueError('Invalid parent NumPy RNG state')
    value = saved['rng_python']
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError('Invalid parent Python RNG state')
    value = saved['rng_cuda']
    if (not isinstance(value, (list, tuple)) or any(not torch.is_tensor(v)
            or v.dtype != torch.uint8 or v.ndim != 1 or not v.numel() for v in value)):
        raise ValueError('Invalid parent CUDA RNG state')
    try:
        torch.Generator(device='cpu').set_state(saved['rng_torch'].detach().cpu())
        np.random.RandomState(0).set_state(copy.deepcopy(saved['rng_numpy']))
        random.Random(0).setstate(copy.deepcopy(saved['rng_python']))
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError('Malformed parent global RNG state') from error


def restore_renderer_native_rng(saved, *, device):
    """Restore all global streams, converting mapped CUDA ByteTensors to CPU."""
    _validate_rng(saved)
    cuda = torch.device(device).type == 'cuda'
    if cuda and not saved['rng_cuda']:
        raise ValueError('CUDA continuation requires the saved parent CUDA RNG stream')
    torch.set_rng_state(saved['rng_torch'].detach().cpu())
    np.random.set_state(copy.deepcopy(saved['rng_numpy']))
    random.setstate(copy.deepcopy(saved['rng_python']))
    if cuda:
        torch.cuda.set_rng_state_all([value.detach().cpu() for value in saved['rng_cuda']])
    return {'cpu_torch_numpy_python_rng_restored': True,
            'cuda_rng_restored': cuda, 'cuda_rng_restore_required': bool(saved['rng_cuda']) and not cuda,
            'saved_cuda_streams': len(saved['rng_cuda'])}


def restore_renderer_native_parent(saved, model, optimizer, data, identity, configuration, frozen, *,
                                   parent_checkpoint_sha256, far_wall_protocol):
    """Load the full completed v5 parent after whitelisted v6 preflight checks.

    This restores weights, all Adam states, private sampler PCG, and the global
    Torch/NumPy/Python/CUDA RNG streams. Validation/probing after adoption must
    preserve those streams before the first actual update.
    """
    from .renderer_far_wall_protocol import far_wall_protocol as canonical_protocol
    if not isinstance(parent_checkpoint_sha256, str) or not re.fullmatch('[0-9a-f]{64}', parent_checkpoint_sha256):
        raise ValueError('Native continuation requires a pinned parent checkpoint SHA256')
    if not _same(far_wall_protocol, canonical_protocol()):
        raise ValueError('Far-wall continuation protocol differs from the canonical immutable settings')
    required = {'base', 'flow', 'optimizer', 'step', 'sampler', 'configuration', 'identity', 'frozen_fingerprint'}
    if not isinstance(saved, dict) or not required <= saved.keys():
        raise ValueError('Native continuation requires the full parent, including Adam/sampler/settings')
    _validate_rng(saved)
    device = next(model.parameters()).device
    if device.type == 'cuda' and not saved['rng_cuda']:
        raise ValueError('CUDA continuation requires the saved parent CUDA RNG stream')
    old = saved['configuration']
    step = saved['step']
    if (type(step) is not int or step < 1 or type(old.get('steps')) is not int or step != old['steps']
            or type(configuration.get('steps')) is not int or configuration['steps'] <= step):
        raise ValueError('The new native phase must extend a completed positive parent optimizer boundary')
    if (old.get('small_enemy_rebalance') is not True or configuration.get('small_enemy_rebalance') is not True
            or old.get('far_wall_supervision', False) is not False
            or configuration.get('far_wall_supervision') is not True
            or not isinstance(configuration.get('native_phase_parent'), str)
            or not configuration['native_phase_parent']
            or old.get('run_dir') == configuration.get('run_dir')):
        raise ValueError('Only an explicit new native-area v5-to-far-wall v6 phase is permitted')
    before = {key: value for key, value in old.items() if key not in CONFIGURATION_CHANGES}
    after = {key: value for key, value in configuration.items() if key not in CONFIGURATION_CHANGES}
    if not _same(before, after):
        changed = sorted(k for k in before.keys() | after.keys() if not _same(before.get(k), after.get(k)))
        raise ValueError('Native continuation changed inherited settings: '+','.join(changed))
    if (IDENTITY_ADDITIONS & saved['identity'].keys() or 'far_wall_protocol' not in identity
            or not _same(identity['far_wall_protocol'], far_wall_protocol)
            or not _same(saved['identity'], {key: value for key, value in identity.items() if key not in IDENTITY_ADDITIONS})):
        raise ValueError('Native parent Source/frozen-feature/codec/validation identity changed')
    if 'far_wall_aux_pool' in identity:
        pool = identity['far_wall_aux_pool']
        source_fingerprints = {saved['identity'].get('train', {}).get('fingerprint'),
                               saved['sampler'].get('fingerprint')}
        if (not isinstance(pool, dict) or pool.get('protocol') != 'source_train_far_wall_aux_pool_v1'
                or pool.get('split') != 'train' or pool.get('parent_source_fingerprint') not in source_fingerprints
                or not isinstance(pool.get('fingerprint'), str)
                or not re.fullmatch('[0-9a-f]{64}', pool['fingerprint'])):
            raise ValueError('Far-wall auxiliary identity must reference the unchanged TRAIN Source pool')
    if saved['frozen_fingerprint'] != frozen:
        raise ValueError('Native parent frozen world/codec identity changed')

    # Check layout, finite parent tensors, and the immutable codec/norm/noise
    # BEFORE allowing any parent Base/RF weight to enter the target.
    states = [model.base_student.state_dict(), model.flow_renderer.state_dict()]
    expected_groups = [list(model.base_student.parameters()),
                       [p for n, p in model.flow_renderer.named_parameters()
                        if not n.startswith('codec.') and p.requires_grad]]
    noncodec_names = {n for n, p in model.flow_renderer.named_parameters()
                      if not n.startswith('codec.') and p.requires_grad}
    for label, target in zip(('base', 'flow'), states):
        parent = saved[label]
        if (not isinstance(parent, dict) or parent.keys() != target.keys()
                or any(not torch.is_tensor(v) or v.shape != target[k].shape
                       or v.dtype != target[k].dtype or not bool(torch.isfinite(v).all())
                       for k, v in parent.items())):
            raise ValueError('Native parent '+label+' tensor architecture/dtype/finiteness changed')
    frozen_flow = {key: value for key, value in saved['flow'].items() if key not in noncodec_names}
    if not _same(frozen_flow, {key: states[1][key] for key in frozen_flow}):
        raise ValueError('Native parent codec/all buffers/latent normalization/fixed noise changed')
    groups = saved['optimizer'].get('param_groups', [])
    if (len(groups) != 2 or len(optimizer.param_groups) != 2
            or any([id(p) for p in current['params']] != [id(p) for p in wanted]
                   for current, wanted in zip(optimizer.param_groups, expected_groups))
            or any(len(group['params']) != len(wanted) for group, wanted in zip(groups, expected_groups))
            or any(not _same({k: v for k, v in old_group.items() if k != 'params'},
                             {k: v for k, v in current.items() if k != 'params'})
                   for old_group, current in zip(groups, optimizer.param_groups))):
        raise ValueError('Native parent ordered Base/non-codec RF Adam groups or options changed')
    ids = [i for group in groups for i in group['params']]
    optimizer_states = saved['optimizer'].get('state', {})
    if len(ids) != len(set(ids)) or set(ids) != set(optimizer_states):
        raise ValueError('Native continuation needs every parent Adam state exactly once')
    for group, parameters in zip(groups, expected_groups):
        for index, parameter in zip(group['params'], parameters):
            state = optimizer_states[index]
            if (not isinstance(state, dict) or not all(k in state for k in ('step', 'exp_avg', 'exp_avg_sq'))
                    or float(state['step']) != step or any(not torch.is_tensor(state[k])
                        or state[k].shape != parameter.shape or state[k].dtype != parameter.dtype
                        or not bool(torch.isfinite(state[k]).all()) for k in ('exp_avg', 'exp_avg_sq'))):
                raise ValueError('Native parent Adam moments, shape or completed-step clock changed')
    sampler = saved['sampler']
    if (sampler.get('protocol') != PARENT_PROTOCOL or sampler.get('parent_draws') != 0
            or sampler.get('parent_sampler_draws_at_start') != 0
            or sampler.get('phase_draws') != step*old['batch']
            or set(sampler.get('distance_counts', {})) != QUOTAS):
        raise ValueError('Native parent sampler protocol or historical draw clock changed')
    private = _sampler_view(data)
    private.restore_sampler(copy.deepcopy(sampler))
    if not _same(private.sampler_state(), sampler):
        raise ValueError('Native sampler cannot restore the exact saved PCG/counters')

    rng = torch.get_rng_state().clone()
    model.base_student.load_state_dict(saved['base'], strict=True)
    model.flow_renderer.load_state_dict(saved['flow'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    data.restore_sampler(copy.deepcopy(sampler))
    if (not _same(saved['base'], model.base_student.state_dict())
            or not _same(saved['flow'], model.flow_renderer.state_dict())
            or not _same(saved['optimizer'], optimizer.state_dict())
            or not _same(sampler, data.sampler_state())):
        raise AssertionError('Native continuation changed parent weights, Adam, or sampler during adoption')
    if not torch.equal(rng, torch.get_rng_state()):
        raise AssertionError('Native parent adoption consumed CPU Torch RNG')
    restored_rng = restore_renderer_native_rng(saved, device=device)
    return {
        'version': 1, 'kind': 'native_area_v5_to_far_geometry_v6_full_Adam_continuation',
        'start_step': step, 'parent_optimizer_step': step, 'parent_checkpoint_sha256': parent_checkpoint_sha256,
        'identity': copy.deepcopy(identity), 'far_wall_protocol': copy.deepcopy(far_wall_protocol),
        'allowed_configuration_changes': sorted(CONFIGURATION_CHANGES),
        'all_base_flow_tensors_bitexact': True, 'codec_normalization_noise_bitexact': True,
        'adam_all_moments_options_bitexact': True, 'adam_state_count': len(optimizer_states),
        'parent_sampler_and_private_rng_bitexact': True, 'sampler_draws_preserved': sampler['phase_draws'],
        'parent_draws': sampler['parent_draws'], 'parent_sampler_draws_at_start': 0,
        'parent_mix_counts': dict.fromkeys(sampler['parent'].get('mix_counts', {}), 0),
        'parent_sampler_counts_at_boundary': copy.deepcopy(sampler),
        'parent_best': copy.deepcopy(saved.get('best')), 'parent_streak': saved.get('streak', 0),
        'parent_enemy_streak': saved.get('enemy_streak', 0),
        'source_identity_preserved': True,
        'identity_additions': sorted(IDENTITY_ADDITIONS & identity.keys()),
        'new_far_wall_aux_sampler_is_independent': 'far_wall_aux_pool' in identity,
        'global_rng_restored': not restored_rng['cuda_rng_restore_required'],
        'global_rng_restoration_required': restored_rng['cuda_rng_restore_required'],
        **restored_rng,
        'source_2000_validation_reference_retained': True, 'selected_6600_reference_retained': True,
        'new_phase_progress_does_not_reset_optimizer_or_sampler': True,
        'automatic_active_promotion': False, 'gameplay_qualified': False,
    }
