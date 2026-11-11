"""Initialize an RSSM student while preserving the exact rendering assets."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path

import torch

from .models import NeuralWorldModel, WorldModelConfig, STATE_SCHEMA, load_world_checkpoint, parameter_counts
from .training_pipeline import atomic_json, replace_file


FROZEN_PREFIXES = ('flow_renderer.', 'vjepa_backend.renderer.', 'vjepa_backend.feature_head.')


def configure_rssm_training(model):
    if model.config.architecture != 'dreamer_rssm':
        raise ValueError('RSSM training requires an RSSM world checkpoint')
    model.train()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith(FROZEN_PREFIXES))
    if hasattr(model, 'flow_renderer'):
        model.flow_renderer.eval().requires_grad_(False)
    model.vjepa_backend.renderer.eval().requires_grad_(False)
    model.vjepa_backend.feature_head.eval().requires_grad_(False)
    return {
        'dynamics_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'frozen': sum(p.numel() for p in model.parameters() if not p.requires_grad),
        'renderer': 'frozen', 'map': 'frozen', 'appearance_projection': 'frozen',
        'rssm': 'trainable', 'ema_target_encoder': 'absent',
    }


def build_rssm_student(teacher, *, seed=173, config_overrides=None):
    if teacher.config.architecture != 'vjepa2_ac':
        raise ValueError('The distillation source must be the existing JEPA teacher')
    config = asdict(teacher.config)
    config.update(architecture='dreamer_rssm', player_motion_mode='axis_fraction')
    config.update(config_overrides or {})
    config = WorldModelConfig(**config)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        student = NeuralWorldModel(config)
    previous, target = teacher.state_dict(), student.state_dict()
    copied = []
    for name, tensor in previous.items():
        if name in target and target[name].shape == tensor.shape and target[name].dtype == tensor.dtype:
            target[name] = tensor.detach().clone()
            copied.append(name)
    for name in previous:
        if name.startswith(FROZEN_PREFIXES) and name not in copied:
            raise ValueError('RSSM migration cannot preserve renderer asset: '+name)
    student.load_state_dict(target, strict=True)
    for name in copied:
        if not torch.equal(student.state_dict()[name], previous[name]):
            raise AssertionError('RSSM warm-start changed a copied tensor: '+name)
    policy = configure_rssm_training(student)
    counts = parameter_counts(student)
    from .renderer_student import renderer_condition_identity
    identity = (renderer_condition_identity(teacher)
        if teacher.config.kind == 'teacher' and teacher.config.renderer_codec == 'vqgan' else None)
    report = {
        'format_version': 1, 'source_architecture': teacher.config.architecture,
        'architecture': student.config.architecture, 'seed': seed,
        'copied_tensors': copied, 'copied_tensor_count': len(copied),
        'new_tensor_count': len(target)-len(copied),
        'frozen_render_assets_bitexact': True, 'previous_optimizer_reused': False,
        'source_teacher_config': asdict(teacher.config),
        'source_renderer_conditioning_identity': identity,
        'optimizer_reason': 'New RSSM parameters and latent state have different identities; original optimizer archived',
        'ema_target_encoder': 'absent', 'memory': {
            'deterministic': config.rssm_deter_dim,
            'categoricals': config.rssm_stoch, 'classes': config.rssm_classes,
            'packed_size': config.rssm_deter_dim+config.rssm_stoch*config.rssm_classes,
            'runtime_sampling': 'deterministic_argmax', 'training_sampling': 'straight_through_categorical'},
        'model_parameters': counts, 'parameter_policy': policy,
        'untrained_pass_fraction': .5,
        'initialization_quality': 'Untrained RSSM; copied heads and known kinematics do not qualify learned navigation',
    }
    student.rssm_migration = report
    return student, report


def initialize_rssm_checkpoint(source, destination, *, seed=173, config_overrides=None):
    source, destination = Path(source).resolve(strict=True), Path(destination).resolve()
    if source == destination or destination.exists():
        raise ValueError('RSSM must start in a new checkpoint, preserving the original teacher')
    teacher, metadata = load_world_checkpoint(source, 'cpu')
    student, report = build_rssm_student(teacher, seed=seed, config_overrides=config_overrides)
    report.update(source_checkpoint=str(source),
        source_global_training_step=int(metadata.get('global_training_step', metadata.get('training_steps', 0))),
        source_checkpoint_bytes=source.stat().st_size)
    payload = {'config': asdict(student.config), 'model': student.state_dict(),
        'training_steps': 0, 'global_training_step': 0,
        'stage': 'world', 'state_schema': dict(STATE_SCHEMA),
        'training_phase': 'rssm_distillation', 'rssm_migration': report,
        'source_teacher_checkpoint': str(source),
        'quality_status': 'RSSM initialized; no training or gameplay qualification yet'}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix+'.tmp')
    try:
        torch.save(payload, temporary)
        replace_file(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    atomic_json(destination.with_suffix(destination.suffix+'.migration.json'), report)
    return report
