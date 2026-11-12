"""Prepare an isolated current-feature renderer phase without touching live assets.

The original combat world and renderer are validated before changing the single
navigation profile flag. This is an explicit data/training profile, not a change
to the saved dynamics, the active game, or any renderer quality findings.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile

import torch

from .models import load_world_checkpoint
from .renderer_student import _hash_tensors, load_renderer_student, renderer_condition_identity


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _private_view(model, *, navigation_only=None):
    result = copy.copy(model)
    result._modules = dict(model._modules)
    result.config = copy.deepcopy(model.config)
    if navigation_only is not None:
        result.config = replace(result.config, navigation_only=navigation_only)
    backend = copy.copy(model.vjepa_backend)
    backend._modules = dict(model.vjepa_backend._modules)
    if hasattr(backend, 'config'):
        backend.config = result.config
    result.vjepa_backend = backend
    return result


def prepare_renderer_feature_profile(source_world, checkpoint, *, allow_unqualified_codec=False):
    """Return a raw NAV world view, rebound bundle, and proof; mutate no inputs.

    Only combat(False)->NAV(True) is allowed. All world tensors remain exactly
    the current snapshot's tensors; all renderer/Base/codec tensors remain the
    selected renderer's tensors. Dynamic examples will obtain causal features
    from the current frozen world rather than the historical training world.
    """
    before = asdict(source_world.config)
    if before['architecture'] != 'dreamer_rssm' or before['navigation_only'] is not False:
        raise ValueError('Current-feature preparation needs the raw combat RSSM world')
    if hasattr(source_world, 'base_student'):
        raise ValueError('Expected the raw world before attaching any compact renderer')
    payload = (torch.load(Path(checkpoint), map_location='cpu', weights_only=False)
               if isinstance(checkpoint, (str, Path)) else checkpoint)
    if not isinstance(payload, dict):
        raise ValueError('Expected a trained renderer bundle')
    original_hash = _hash_tensors(source_world.state_dict())
    source_identity = renderer_condition_identity(source_world)
    nav_world = _private_view(source_world, navigation_only=True)
    after = asdict(nav_world.config)
    if {key for key in before if before[key] != after[key]} != {'navigation_only'}:
        raise AssertionError('Unexpected configuration change')
    if _hash_tensors(nav_world.state_dict()) != original_hash:
        raise AssertionError('Profile preparation changed world tensors')
    with torch.random.fork_rng(devices=[]), tempfile.TemporaryDirectory(prefix='renderer_features_') as tmp:
        path = Path(tmp)/'renderer.pt'
        torch.save(payload, path)
        old = _private_view(source_world)
        # This strict check must precede the rebind; malformed lineage or
        # conditioning hashes cannot be laundered by rewriting metadata.
        load_renderer_student(old, path, allow_unqualified_codec=allow_unqualified_codec)
        result = copy.deepcopy(payload)
        proof = {
            'version': 1, 'kind': 'current_frozen_rssm_combat_to_navigation_renderer_features',
            'source_navigation_only': False, 'target_navigation_only': True,
            'allowed_config_changes': ['navigation_only'],
            'world_tensor_sha256': original_hash, 'world_tensors_bitexact': True,
            'renderer_conditioning_identity': source_identity,
            'original_source_teacher_config': copy.deepcopy(payload['source_teacher_config']),
            'original_conditioning_identity': copy.deepcopy(payload['conditioning_identity']),
            'parent_profile_rebind': copy.deepcopy(payload.get('renderer_profile_rebind')),
            'renderer_weights_bitexact': True, 'base_weights_bitexact': True,
            'codec_normalization_noise_unchanged': True, 'quality_findings_unchanged': True,
            'frozen_dynamics_feature_source_changed_explicitly': True,
            'feature_policy': 'current_frozen_prior_previous_Source_RGB_state_action_only',
            'future_source_rgb_in_prior': False, 'future_source_pose_in_prior': False,
            'source_pose_rendering_preserves_active_npc': True,
            'fresh_renderer_optimizer_phase': True, 'old_adam_not_claimed_preserved': True,
            'automatic_active_promotion': False, 'live_world_or_game_changed': False,
        }
        result['source_teacher_config'] = copy.deepcopy(after)
        result['config'].update({key: value for key, value in after.items()
            if key.startswith(('rssm_', 'vjepa_', 'enemy_rssm_'))
            or key in {'architecture', 'player_motion_mode', 'navigation_only'}})
        result['renderer_profile_rebind'] = copy.deepcopy(proof)
        base = result.get('base_student_bundle')
        if base is not None:
            base['source_teacher_config'] = copy.deepcopy(after)
            base['provenance'] = {**base.get('provenance', {}),
                                  'current_feature_profile_rebind': copy.deepcopy(proof)}
        torch.save(result, path)
        new = _private_view(nav_world)
        load_renderer_student(new, path, allow_unqualified_codec=allow_unqualified_codec)
        for name in ('flow_renderer', 'base_student'):
            if hasattr(old, name) != hasattr(new, name):
                raise AssertionError('Profile preparation changed renderer modules')
            if hasattr(old, name) and _hash_tensors(getattr(old, name).state_dict()) != _hash_tensors(getattr(new, name).state_dict()):
                raise AssertionError('Profile preparation changed learned renderer tensors')
        proof.update(base_tensor_sha256=_hash_tensors(new.base_student.state_dict()) if base is not None else None,
                     flow_tensor_sha256=_hash_tensors(new.flow_renderer.state_dict()),
                     codec_tensor_sha256=_hash_tensors(new.flow_renderer.codec.state_dict()),
                     strict_original_reload_verified=True, strict_navigation_reload_verified=True)
    if asdict(source_world.config) != before or _hash_tensors(source_world.state_dict()) != original_hash:
        raise AssertionError('Current raw world was mutated')
    return nav_world, result, proof


def prepare_current_renderer_inputs(source_checkpoint, renderer_checkpoint, output_directory, *,
                                    source_checkpoint_sha256, renderer_checkpoint_sha256,
                                    expected_codec_sha256, allow_unqualified_codec=False):
    """Write new immutable phase inputs and JSON provenance after strict checks."""
    source_path, renderer_path = Path(source_checkpoint).resolve(), Path(renderer_checkpoint).resolve()
    if _digest(source_path) != source_checkpoint_sha256 or _digest(renderer_path) != renderer_checkpoint_sha256:
        raise ValueError('Pinned source world or selected renderer file SHA256 changed')
    root = Path(output_directory).resolve()
    targets = [root/'world_navigation.pt', root/'renderer_parent_navigation.pt', root/'feature_inputs_verification.json']
    if any(path.exists() for path in targets):
        raise FileExistsError('Current-feature inputs already exist; do not overwrite immutable phase inputs')
    with torch.random.fork_rng(devices=[]):
        source, saved = load_world_checkpoint(source_path, 'cpu')
        nav, payload, proof = prepare_renderer_feature_profile(source, renderer_path,
            allow_unqualified_codec=allow_unqualified_codec)
        if proof['codec_tensor_sha256'] != expected_codec_sha256:
            raise ValueError('Selected renderer codec differs from the fixed codec pin')
        step = saved.get('training_steps')
        if not isinstance(step, int) or isinstance(step, bool) or step < 1:
            raise ValueError('Current world has no positive saved optimizer step')
        # This is a frozen rendering source, not a resumable dynamics checkpoint.
        output = {key: copy.deepcopy(saved[key]) for key in
                  ('stage', 'state_schema', 'rssm_migration', 'dual_rssm_migration', 'enemy_rssm_resize')
                  if key in saved}
        output.update(config=asdict(nav.config), model=nav.state_dict(), training_steps=step,
                      renderer_feature_profile_rebind=copy.deepcopy(proof))
        root.mkdir(parents=True, exist_ok=True)
        torch.save(output, targets[0])
        torch.save(payload, targets[1])
        # Reload the exact files the GPU trainer will consume.
        reloaded, restored = load_world_checkpoint(targets[0], 'cpu')
        if (_hash_tensors(reloaded.state_dict()) != proof['world_tensor_sha256']
                or restored['training_steps'] != step or not reloaded.config.navigation_only
                or not reloaded.vjepa_backend.config.navigation_only):
            raise AssertionError('Saved raw NAV world is not the tensor-identical current snapshot')
        load_renderer_student(reloaded, targets[1], allow_unqualified_codec=allow_unqualified_codec)
        if (_hash_tensors(reloaded.flow_renderer.state_dict()) != proof['flow_tensor_sha256']
                or _hash_tensors(reloaded.base_student.state_dict()) != proof['base_tensor_sha256']):
            raise AssertionError('Saved NAV renderer does not match the selected renderer')
    proof.update(status='verified', source_saved_step=step,
                 source_checkpoint=str(source_path), source_checkpoint_sha256=source_checkpoint_sha256,
                 selected_renderer=str(renderer_path), selected_renderer_file_sha256=renderer_checkpoint_sha256,
                 world_navigation={'path': str(targets[0]), 'sha256': _digest(targets[0])},
                 renderer_navigation={'path': str(targets[1]), 'sha256': _digest(targets[1])},
                 all_source_world_tensors_bitexact=True, strict_saved_inputs_reload_verified=True,
                 no_source_optimizer_restoration_claim=True, source_training_untouched=True)
    targets[2].write_text(json.dumps(proof, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-checkpoint', 'renderer-checkpoint', 'output-directory'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('source-checkpoint-sha256', 'renderer-checkpoint-sha256', 'expected-codec-sha256'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--allow-unqualified-codec', action='store_true')
    parser.add_argument('--initialize-cuda', action='store_true', help='Initialize context before CPU checkpoint load; no GPU inference')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.initialize_cuda:
        torch.cuda.init()
    values = vars(args); values.pop('initialize_cuda')
    print(json.dumps(prepare_current_renderer_inputs(**values), ensure_ascii=True))


if __name__ == '__main__':
    main()
