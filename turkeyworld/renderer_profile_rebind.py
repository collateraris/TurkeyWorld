"""Explicit, tensor-identical navigation-to-combat renderer profile binding.

This creates a new bundle in memory. It never changes a live model, checkpoint,
active-run pointer or the renderer's quality findings.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
import tempfile

import torch

from .renderer_student import (_hash_tensors, load_renderer_student,
                               renderer_condition_identity)


def _view(model):
    result = copy.copy(model)
    result._modules = dict(model._modules)
    result.config = copy.deepcopy(model.config)
    return result


def prepare_renderer_profile_rebind(source_world, combat_world, checkpoint, *,
                                    allow_unqualified_codec=False):
    """Return ``(new_payload, proof)`` after strict old and new load validation.

    The only allowed world configuration difference is navigation_only=True
    to False; every world tensor must be byte-identical. The original bundle
    is first validated against its actual navigation world, including RSSM
    architecture lineage, Base/map/appearance and codec/Q/norm/noise guards.
    Callers explicitly persist/use the returned artifact after reviewing it.
    """
    before, after = asdict(source_world.config), asdict(combat_world.config)
    changed = {key for key in before if before[key] != after[key]}
    if (changed != {'navigation_only'} or before['navigation_only'] is not True
            or after['navigation_only'] is not False
            or before['architecture'] != 'dreamer_rssm'):
        raise ValueError('Only the explicit RSSM navigation-to-combat profile change is allowed')
    source_state, target_state = source_world.state_dict(), combat_world.state_dict()
    if (source_state.keys() != target_state.keys()
            or any(a.dtype != target_state[key].dtype or a.shape != target_state[key].shape
                   or not torch.equal(a.detach().cpu(), target_state[key].detach().cpu())
                   for key, a in source_state.items())):
        raise ValueError('Profile rebinding requires every source/combat world tensor bitexact')
    source_identity = renderer_condition_identity(source_world)
    if renderer_condition_identity(combat_world) != source_identity:
        raise ValueError('Renderer assets changed during the profile switch')
    payload = (torch.load(Path(checkpoint), map_location='cpu', weights_only=False)
               if isinstance(checkpoint, (str, Path)) else checkpoint)
    if not isinstance(payload, dict):
        raise ValueError('Expected a renderer bundle mapping')
    with tempfile.TemporaryDirectory(prefix='turkeyworld_profile_') as directory:
        temporary = Path(directory)/'renderer.pt'
        torch.save(payload, temporary)
        validated_old = _view(source_world)
        load_renderer_student(validated_old, temporary,
                              allow_unqualified_codec=allow_unqualified_codec)
        result = copy.deepcopy(payload)
        proof = {'version': 1, 'kind': 'navigation_to_one_enemy_combat',
            'source_navigation_only': True, 'target_navigation_only': False,
            'allowed_config_changes': ['navigation_only'],
            'world_tensors_bitexact': True, 'world_tensor_sha256': _hash_tensors(source_state),
            'renderer_conditioning_identity': source_identity,
            'original_source_teacher_config': copy.deepcopy(payload['source_teacher_config']),
            'original_conditioning_identity': copy.deepcopy(payload['conditioning_identity']),
            'renderer_weights_bitexact': True, 'base_weights_bitexact': True,
            'codec_normalization_noise_unchanged': True,
            'quality_findings_unchanged': True, 'automatic_active_promotion': False}
        result['source_teacher_config'] = copy.deepcopy(after)
        result['config'].update({key: value for key, value in after.items()
            if key.startswith(('rssm_', 'vjepa_', 'enemy_rssm_'))
            or key in {'architecture', 'player_motion_mode', 'navigation_only'}})
        result['renderer_profile_rebind'] = proof
        base = result.get('base_student_bundle')
        if base is not None:
            base['source_teacher_config'] = copy.deepcopy(after)
            base['provenance'] = {**base.get('provenance', {}),
                                  'renderer_profile_rebind': copy.deepcopy(proof)}
        torch.save(result, temporary)
        validated_new = _view(combat_world)
        load_renderer_student(validated_new, temporary,
                              allow_unqualified_codec=allow_unqualified_codec)
        if (_hash_tensors(validated_new.flow_renderer.state_dict())
                != _hash_tensors(validated_old.flow_renderer.state_dict())
                or (base is not None and _hash_tensors(validated_new.base_student.state_dict())
                    != _hash_tensors(validated_old.base_student.state_dict()))):
            raise AssertionError('Rebinding changed learned renderer tensors')
    return result, proof
