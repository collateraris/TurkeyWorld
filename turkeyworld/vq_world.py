"""Qualified shared-codec profiles and honest pixel-to-latent world migration.

World dynamics are retained exactly. Image projections change representation,
so their shape-incompatible tensors are initialized anew; learned compatible
DiT/state/time weights transfer and spatial positions are interpolated. No claim
of preserving the original RGB function is made across this domain change.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from .models import NeuralWorldModel, WorldModelConfig, load_world_checkpoint
from .vqgan import VQGAN, VQGANConfig


STUDENT_VQ_PARAMETERS = 22_475_096
TEACHER_VQ_PARAMETERS = 504_857_752


def student_vq_config(codec_config, **overrides):
    """22.48M complete student, including a frozen 1488-entry shared VQGAN."""
    codec = asdict(codec_config) if isinstance(codec_config, VQGANConfig) else dict(codec_config)
    codec = asdict(VQGANConfig(**codec))
    values = dict(kind='student', architecture='vjepa2_ac', image_size=128, endless=True,
        renderer_mode='rectified_flow', flow_steps=4, renderer_codec='vqgan',
        renderer_codec_config=codec, vjepa_width=320, vjepa_encoder_depth=4,
        vjepa_predictor_depth=5, vjepa_heads=5, vjepa_context_frames=4,
        vjepa_readout_mlp_ratio=5, flow_width=192, flow_depth=4, flow_heads=4,
        flow_mlp_hidden=736)
    values.update(overrides)
    if values['kind'] != 'student' or values['architecture'] != 'vjepa2_ac' or values['renderer_codec'] != 'vqgan':
        raise ValueError('The VQ student profile requires a V-JEPA student and shared VQGAN')
    return WorldModelConfig(**values)


def codec_fingerprint(codec):
    """Same digest as train_vqgan, without importing the experiment controller."""
    digest = hashlib.sha256(json.dumps(codec.get_config(), sort_keys=True).encode())
    for name, value in sorted(codec.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _quality_qualified(saved):
    gate = saved.get('quality_gate', {})
    required, consecutive = gate.get('required_passes', 3), gate.get('consecutive_passes', 0)
    return (saved.get('stage') == 'frozen_vqgan_codec' and gate.get('passed') is True
        and gate.get('frozen') is True and isinstance(required, int) and required >= 3
        and isinstance(consecutive, int) and consecutive >= required)


def _user_accepted(saved):
    acceptance, gate = saved.get('quality_acceptance', {}), saved.get('quality_gate', {})
    checks = gate.get('checks', {})
    integrity = ('all_frames_finite', 'all_frames_in_range', 'quantization_finite',
                 'valid_code_indices', 'codebook_broad_usage')
    return (saved.get('stage') == 'frozen_vqgan_codec' and gate.get('frozen') is True
        and acceptance.get('mode') == 'user_requested' and acceptance.get('accepted') is True
        and bool(acceptance.get('reason')) and bool(acceptance.get('accepted_at'))
        and isinstance(gate.get('passed'), bool)
        and acceptance.get('quality_gate_passed') is gate.get('passed')
        and all(checks.get(key) is True for key in integrity))


def _load_codec(path, require_qualified, accept_user_requested=False):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(saved, dict) or 'codec_config' not in saved or 'model' not in saved:
        raise ValueError('A trained codec checkpoint with its configuration and weights is required')
    if require_qualified:
        if not (_quality_qualified(saved) or (accept_user_requested and _user_accepted(saved))):
            raise ValueError('Codec must pass and freeze after at least three consecutive quality checks')
    config = VQGANConfig(**saved['codec_config'])
    codec = VQGAN(config).eval().requires_grad_(False)
    codec.load_state_dict(saved['model'], strict=True)
    if not bool(codec.quantize.initialized):
        raise ValueError('The codec codebook has not been initialized from training')
    actual = codec_fingerprint(codec)
    if not saved.get('codec_fingerprint') or saved['codec_fingerprint'] != actual:
        raise ValueError('Codec weight/configuration fingerprint does not match its frozen checkpoint')
    normalization = saved.get('latent_normalization')
    if not isinstance(normalization, dict) or 'mean' not in normalization or 'std' not in normalization:
        raise ValueError('Frozen codec must include measured per-channel latent normalization')
    mean, std = (torch.as_tensor(normalization[key], dtype=torch.float32) for key in ('mean', 'std'))
    if (mean.shape != (config.embedding_dim,) or std.shape != (config.embedding_dim,)
            or not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()) or not bool((std > 0).all())):
        raise ValueError('Frozen codec latent normalization is invalid')
    if require_qualified and (config.image_size, config.embedding_dim, config.n_embed) != (128, 16, 1488):
        raise ValueError('Production world profiles require the selected 128px, 16D, 1488-entry codec')
    return saved, codec, actual


def validate_frozen_codec(checkpoint, *, require_qualified=True, accept_user_requested=False):
    """Validate the compact codec alone, without loading a world checkpoint."""
    saved, codec, fingerprint = _load_codec(checkpoint, require_qualified, accept_user_requested)
    return {'checkpoint': str(Path(checkpoint).resolve()), 'codec_config': codec.get_config(),
        'codec_fingerprint': fingerprint, 'quality_gate': saved.get('quality_gate'),
        'latent_normalization': saved['latent_normalization'],
        'generator_parameters': sum(parameter.numel() for parameter in codec.parameters()),
        'training_steps': int(saved.get('training_steps', 0)), 'qualified': _quality_qualified(saved),
        'quality_acceptance': saved.get('quality_acceptance'),
        'accepted_by_user': bool(accept_user_requested and _user_accepted(saved))}


def _pretrain_parameters(model):
    return [(name, value) for name, value in model.named_parameters()
            if not name.startswith(('flow_renderer.', 'vjepa_backend.renderer.'))]


def _resume_compatibility(source, original, migrated):
    if source.get('pipeline_phase') != 'teacher500_world':
        return False, 'Source phase is not teacher500_world'
    optimizer = source.get('optimizer')
    if not isinstance(optimizer, dict) or not isinstance(optimizer.get('param_groups'), list):
        return False, 'Source does not contain an optimizer checkpoint'
    old, new = _pretrain_parameters(original), _pretrain_parameters(migrated)
    if [(name, value.shape) for name, value in old] != [(name, value.shape) for name, value in new]:
        return False, 'Ordered trainable world-core parameter names/shapes changed'
    groups = optimizer['param_groups']
    if len(groups) != 1 or len(groups[0].get('params', [])) != len(old):
        return False, 'Optimizer parameter groups do not match the frozen-renderer pretraining core'
    identifiers = groups[0]['params']
    if len(set(identifiers)) != len(identifiers):
        return False, 'Optimizer parameter IDs are duplicated'
    if groups[0].get('param_names', [name for name, _ in old]) != [name for name, _ in old]:
        return False, 'Optimizer explicit parameter order differs from the world core'
    states = optimizer.get('state', {})
    if not isinstance(states, dict) or set(states)-set(identifiers):
        return False, 'Optimizer state references unknown parameters'
    for identifier, (_, parameter) in zip(identifiers, old):
        entry = states.get(identifier, {})
        if not isinstance(entry, dict):
            return False, 'Optimizer moment entry is not a state dictionary'
        for key, value in entry.items():
            if torch.is_tensor(value) and key != 'step' and value.shape != parameter.shape:
                return False, 'Optimizer moment shape differs from its world-core parameter'
    ema = source.get('ema_target_encoder')
    if not isinstance(ema, dict):
        return False, 'Source EMA target encoder is missing'
    if (not torch.is_tensor(ema.get('_ema_steps')) or not isinstance(ema.get('_extra_state'), dict)
            or ema['_extra_state'].get('format_version') != 1):
        return False, 'Source EMA bookkeeping/configuration is missing'
    for name, value in original.core.encoder.state_dict().items():
        target = ema.get('encoder.'+name)
        if not torch.is_tensor(target) or target.shape != value.shape:
            return False, 'Source EMA target encoder is incompatible'
    if not {'rng_numpy', 'rng_torch', 'rng_cuda'}.issubset(source):
        return False, 'Source optimizer resume RNG state is incomplete'
    if (not isinstance(source['rng_numpy'], dict) or not torch.is_tensor(source['rng_torch'])
            or source['rng_torch'].dtype != torch.uint8 or source['rng_torch'].ndim != 1
            or not isinstance(source['rng_cuda'], list)):
        return False, 'Source RNG state has incompatible types'
    return True, 'Unchanged ordered world core; compatible optimizer moments, EMA and RNG'


def _save_new(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    try:
        torch.save(payload, temporary)
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite migration output: {path}')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def migrate_teacher_to_vq(source_world_checkpoint500, frozen_codec_checkpoint, output_initial,
                           resume_output=None, *, require_qualified=True, accept_user_requested=False):
    """Retain the world core and warm-start compatible latent-renderer tensors.

    An incompatible resume request produces the initial artifact only, with an
    explicit reset reason. It never writes an optimizer-less file masquerading
    as a resumable training checkpoint. Existing output files are not replaced.
    """
    source_path, codec_path, output = (Path(value).resolve() for value in
                                      (source_world_checkpoint500, frozen_codec_checkpoint, output_initial))
    resume = None if resume_output is None else Path(resume_output).resolve()
    targets = [output]+([] if resume is None else [resume])
    if len(set(targets)) != len(targets) or any(path in (source_path, codec_path) or path.exists() for path in targets):
        raise FileExistsError('VQ migration requires distinct new output paths, separate from both sources')
    saved_codec, codec, fingerprint = _load_codec(codec_path, require_qualified, accept_user_requested)
    original, source = load_world_checkpoint(source_path, 'cpu')
    if (original.config.kind != 'teacher' or original.config.architecture != 'vjepa2_ac'
            or original.config.renderer_codec != 'pixel' or original.config.renderer_mode != 'flow_matching'):
        raise ValueError('VQ teacher migration requires a pixel-flow V-JEPA teacher')
    source_count = sum(parameter.numel() for parameter in original.parameters())
    if require_qualified and abs(source_count-500_000_000) > 2_500_000:
        raise ValueError(f'Expected the compact 500M teacher source, got {source_count}')
    config = asdict(original.config)
    config.update(renderer_codec='vqgan', renderer_codec_config=codec.get_config())
    model = NeuralWorldModel(WorldModelConfig(**config)).eval()
    old, new = original.state_dict(), model.state_dict()
    retained, interpolated = [], []
    with torch.no_grad():
        for name, value in old.items():
            if name not in new:
                raise ValueError(f'Migration would silently discard a source tensor: {name}')
            destination = new[name]
            if value.shape == destination.shape:
                destination.copy_(value)
                retained.append(name)
            elif name.startswith('flow_renderer.') and name.endswith('.position'):
                before, after = math.isqrt(value.shape[1]), math.isqrt(destination.shape[1])
                if before**2 != value.shape[1] or after**2 != destination.shape[1] or value.shape[-1] != destination.shape[-1]:
                    raise ValueError(f'Cannot interpolate this renderer position grid: {name}')
                spatial = value.transpose(1, 2).reshape(1, value.shape[-1], before, before)
                spatial = F.interpolate(spatial.float(), size=(after, after), mode='bilinear', align_corners=False)
                destination.copy_(spatial.flatten(2).transpose(1, 2).to(destination.dtype))
                interpolated.append(name)
            elif not (name.startswith('flow_renderer.') and any(
                    part in name for part in ('.image_input.', '.output.'))):
                raise ValueError(f'Unexpected shape change outside image projections: {name}')
    model.flow_renderer.load_codec(saved_codec)
    for name in retained:
        if not torch.equal(old[name], model.state_dict()[name]):
            raise RuntimeError(f'Transferred tensor changed unexpectedly: {name}')
    core_names = [name for name in old if not name.startswith('flow_renderer.')]
    if any(not torch.equal(old[name], model.state_dict()[name]) for name in core_names):
        raise RuntimeError('World core/map/base renderer did not transfer exactly')
    valid_resume, resume_reason = _resume_compatibility(source, original, model)
    use_resume = resume is not None and valid_resume
    count = sum(parameter.numel() for parameter in model.parameters())
    if require_qualified and count != TEACHER_VQ_PARAMETERS:
        raise ValueError(f'Teacher/codec configuration does not match the selected complete profile: {count}')
    fresh = sorted(set(new)-set(retained)-set(interpolated))
    projected = [name for name in old if name not in retained and name not in interpolated]
    report = {'source_checkpoint': str(source_path), 'codec_checkpoint': str(codec_path),
        'codec_fingerprint': fingerprint, 'codec_quality_verified': _quality_qualified(saved_codec),
        'codec_quality_accepted_by_user': bool(accept_user_requested and _user_accepted(saved_codec)),
        'codec_quality_acceptance': saved_codec.get('quality_acceptance'),
        'source_generator_parameters': source_count, 'generator_parameters': count,
        'codec_parameters': sum(parameter.numel() for parameter in codec.parameters()),
        'retained_tensor_count': len(retained), 'retained_core_tensor_count': len(core_names),
        'retained_core_all_values_exact': True, 'retained_same_shape_all_values_exact': True,
        'interpolated_renderer_keys': interpolated, 'reinitialized_source_renderer_keys': projected,
        'new_renderer_keys': fresh, 'renderer_function_preserved': False,
        'codec_frozen': True, 'codec_normalization_loaded': True,
        'optimizer_preserved': use_resume, 'ema_preserved': use_resume, 'rng_preserved': use_resume,
        'resume_compatible': valid_resume, 'resume_reason': resume_reason,
        'initial_checkpoint': str(output), 'resume_checkpoint': str(resume) if use_resume else None,
        'source_phase': source.get('pipeline_phase'), 'source_training_steps': int(source.get('training_steps', 0)),
        'source_global_training_step': int(source.get('global_training_step', source.get('training_steps', 0))),
        'effective_widths': {'latent_width': model.core.width, 'encoder_depth': len(model.core.encoder.blocks),
            'predictor_depth': len(model.core.predictor.blocks), 'flow_width': model.flow_renderer.width,
            'flow_depth': len(model.flow_renderer.blocks), 'codec_embedding_dim': codec.config.embedding_dim,
            'codec_entries': codec.config.n_embed},
        'created_at': datetime.now(timezone.utc).isoformat()}
    payload = {'format_version': 1, 'stage': 'world', 'config': asdict(model.config), 'model': model.state_dict(),
        'training_steps': 0, 'global_training_step': report['source_global_training_step'],
        'pipeline_phase': 'teacher500vq_initial', 'vq_migration': report,
        'codec_fingerprint': fingerprint, 'codec_quality_gate': saved_codec.get('quality_gate'),
        'source_training_steps': report['source_training_steps'], 'latent_renderer_training_steps': 0,
        'optimizer_reset_required': not use_resume, 'quality_status': 'Retained dynamics; latent renderer unverified.',
        'created_at': report['created_at']}
    if use_resume:
        # Deliberately omit source phase-complete, quality/selection, D, and
        # long-rollout state; all describe the previous renderer/phase context.
        resumed = {**payload, 'pipeline_phase': 'teacher500vq_world',
            'training_steps': report['source_training_steps'], 'phase_complete': False,
            **{name: source[name] for name in ('optimizer', 'ema_target_encoder', 'rng_numpy', 'rng_torch', 'rng_cuda')}}
        _save_new(resume, resumed)
    _save_new(output, payload)
    return report
