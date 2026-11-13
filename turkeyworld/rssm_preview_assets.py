"""Pinned, inference-only renderer/SR assets for a separate RSSM preview clone."""
from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
import torch


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def preserve_preview_rng(device, private_rng=None):
    """Preview construction/inference cannot consume any training RNG stream."""
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    python_state, numpy_state = random.getstate(), np.random.get_state()
    private_state = copy.deepcopy(private_rng.bit_generator.state) if private_rng is not None else None
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if private_rng is not None:
            private_rng.bit_generator.state = private_state


def read_preview_assets_plan(path):
    """Verify immutable file identities before considering the preview cache."""
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding='utf-8'))
    if type(raw.get('version')) is not int or raw.get('version') != 1 or raw.get('kind') != 'frozen_preview_assets_v1':
        raise ValueError('Unknown preview-assets plan protocol')
    if not isinstance(raw.get('allow_unqualified_codec', False), bool):
        raise ValueError('Preview codec opt-in must be an explicit boolean')
    result = copy.deepcopy(raw)
    for name in ('renderer', 'sr'):
        asset = raw.get(name)
        if not isinstance(asset, dict):
            raise ValueError('Preview plan requires renderer and SR assets')
        source = Path(asset.get('checkpoint', ''))
        expected = asset.get('file_sha256')
        if (not source.is_absolute() or not source.is_file() or not isinstance(expected, str)
                or re.fullmatch(r'[0-9a-f]{64}', expected) is None or file_sha256(source) != expected):
            raise ValueError('Pinned preview '+name+' file SHA/path changed')
        result[name]['checkpoint'] = str(source.resolve())
    result['plan_path'] = str(path)
    result['plan_sha256'] = file_sha256(path)
    return result


def preview_assets_cache_key(plan, config):
    return hashlib.sha256(json.dumps({'plan_sha256': plan['plan_sha256'],
        'renderer_sha256': plan['renderer']['file_sha256'], 'sr_sha256': plan['sr']['file_sha256'],
        'world_config': asdict(config)}, sort_keys=True).encode()).hexdigest()


def resolve_preview_assets_plan(initial_path, revision_path=None, saved_revision=None):
    """Resolve an explicit preview-only revision while retaining the v4 pin.

    Saved revisions cannot be dropped or silently replaced on a later resume.
    Both the initial plan and the selected plan/files remain independently pinned.
    """
    if saved_revision is not None and not isinstance(saved_revision, dict):
        raise ValueError('Malformed saved preview-assets revision')
    initial = read_preview_assets_plan(initial_path)
    if revision_path is None:
        if saved_revision is not None:
            raise ValueError('Resume requires the saved explicit preview-assets revision plan')
        return initial, initial, None
    def resolve_revision(path, ancestors=()):
        path=Path(path).resolve()
        if path in ancestors or len(ancestors)>=32:
            raise ValueError('Preview-assets revision chain has a cycle or excessive depth')
        if not path.is_file():raise ValueError('Previous preview-assets revision SHA/path changed')
        data=path.read_bytes();raw=json.loads(data.decode('utf-8'))
        if (not isinstance(raw,dict)or type(raw.get('version'))is not int
                or (raw.get('version'),raw.get('kind'))not in
                    ((1,'preview_assets_revision_v1'),(2,'preview_assets_revision_v2'))
                or raw.get('initial_plan_sha256')!=initial['plan_sha256']):
            raise ValueError('Preview-assets revision changed its initial immutable plan')
        parent=None
        if raw['version']==2:
            previous=Path(raw.get('previous_revision_plan',''))
            previous_sha=raw.get('previous_revision_sha256')
            if (not previous.is_absolute()or not previous.is_file()or not isinstance(previous_sha,str)
                    or re.fullmatch(r'[0-9a-f]{64}',previous_sha)is None):
                raise ValueError('Previous preview-assets revision SHA/path changed')
            _,parent,_=resolve_revision(previous,ancestors+(path,))
            if parent['revision_sha256']!=previous_sha:
                raise ValueError('Previous preview-assets revision SHA/path changed')
        selected_path=Path(raw.get('selected_plan',''));expected=raw.get('selected_plan_sha256')
        if (not selected_path.is_absolute()or not selected_path.is_file()or not isinstance(expected,str)
                or re.fullmatch(r'[0-9a-f]{64}',expected)is None or file_sha256(selected_path)!=expected):
            raise ValueError('Selected preview-assets plan SHA/path changed')
        selected=read_preview_assets_plan(selected_path)
        record={'kind':raw['kind'],'revision_plan_path':str(path),
            'revision_sha256':hashlib.sha256(data).hexdigest(),'initial_plan_path':initial['plan_path'],
            'initial_plan_sha256':initial['plan_sha256'],'selected_plan_path':selected['plan_path'],
            'selected_plan_sha256':selected['plan_sha256'],'preview_only':True,
            'training_assets_changed':False,'game_assets_changed':False}
        if parent is not None:
            record.update(previous_revision_plan_path=parent['revision_plan_path'],
                previous_revision_sha256=parent['revision_sha256'],
                previous_selected_plan_sha256=parent['selected_plan_sha256'])
        return selected,record,parent

    selected,record,parent=resolve_revision(revision_path)
    if parent is None:
        # Preserve the original v1 transition/resume contract exactly.
        if saved_revision is not None and any(saved_revision.get(key)!=value for key,value in record.items()):
            raise ValueError('Saved preview-assets revision identity changed on resume')
    elif saved_revision!=record and saved_revision!=parent:
        raise ValueError('Preview-assets v2 requires the exact saved revision or its immediate parent')
    return initial, selected, record


def synchronize_preview_dynamics(client, model):
    """Update current world weights while keeping approved image assets intact."""
    from .renderer_student import renderer_condition_identity
    source_identity, target_identity = renderer_condition_identity(model), renderer_condition_identity(client)
    for key in ('base_map_sha256', 'appearance_sha256', 'latent_normalization_sha256', 'fixed_noise_sha256'):
        if source_identity[key] != target_identity[key]:
            raise ValueError('Current RSSM renderer conditioning differs from the pinned preview assets')
    current, target = model.state_dict(), client.state_dict()
    keys = [name for name in current if not name.startswith(('flow_renderer.', 'base_student.'))]
    if any(name not in target or current[name].shape != target[name].shape for name in keys):
        raise ValueError('Preview world architecture differs from current training dynamics')
    with torch.no_grad():
        for name in keys:
            target[name].copy_(current[name])
    return len(keys)


def prepare_preview_assets(model, plan, device, *, clone_factory=None):
    """Transactionally build a frozen inference clone; never attach to training."""
    from .models import NeuralWorldModel, WorldModelConfig, load_sr_checkpoint
    from .renderer_student import _hash_tensors, load_renderer_student
    device = torch.device(device)
    factory = clone_factory or (lambda config: NeuralWorldModel(WorldModelConfig(**asdict(config))))
    source_hash = _hash_tensors(model.state_dict())
    parameter_ids = tuple(id(parameter) for parameter in model.parameters())
    with preserve_preview_rng(device):
        client = factory(model.config)
        client.load_state_dict(model.state_dict(), strict=True)
        for name in ('rssm_migration', 'dual_rssm_migration', 'enemy_rssm_resize','rssm_fusion_migration',
                     'combat_rssm_migration'):
            if hasattr(model, name):
                setattr(client, name, copy.deepcopy(getattr(model, name)))
        renderer = load_renderer_student(client, plan['renderer']['checkpoint'],
            allow_unqualified_codec=plan.get('allow_unqualified_codec', False))
        if client.flow_renderer.kind != 'student' or client.flow_renderer.steps != 1:
            raise ValueError('Pinned updated preview requires the actual one-Euler student')
        actual_updates = renderer.get('metrics', {}).get('refinement_updates')
        requested_updates = plan['renderer'].get('refinement_updates')
        if requested_updates is not None and requested_updates != actual_updates:
            raise ValueError('Preview renderer update label differs from its saved metadata')
        sr, sr_metadata = load_sr_checkpoint(plan['sr']['checkpoint'], 'cpu')
        trained = sr_metadata.get('training_steps')
        if not isinstance(trained, (int, float)) or isinstance(trained, bool) or not np.isfinite(trained) or trained <= 0:
            raise ValueError('Pinned preview SR must actually be trained')
        selected = plan['sr'].get('selected_local_update')
        if selected is not None and selected != sr_metadata.get('refinement_updates'):
            raise ValueError('Preview SR selected-step label differs from its saved metadata')
        for name, expected in (('input_size', model.config.image_size), ('output_size', model.config.image_size*4)):
            if name in sr_metadata and sr_metadata[name] != expected:
                raise ValueError('Preview SR resolution differs from current native RGB')
        client.to(device).eval().requires_grad_(False)
        sr.to(device).eval().requires_grad_(False)
        copied = synchronize_preview_dynamics(client, model)
    if (_hash_tensors(model.state_dict()) != source_hash
            or tuple(id(parameter) for parameter in model.parameters()) != parameter_ids):
        raise AssertionError('Preview initialization mutated training model tensors/parameters')
    proof = {'kind': 'frozen_preview_assets_v1', 'plan_path': plan['plan_path'],
        'plan_sha256': plan['plan_sha256'], 'renderer': copy.deepcopy(plan['renderer']),
        'sr': copy.deepcopy(plan['sr']), 'renderer_training_steps': renderer['training_steps'],
        'renderer_refinement_updates': actual_updates, 'renderer_solver_steps': 1,
        'base_training_steps': getattr(client, 'base_student_bundle', {}).get('training_steps'),
        'sr_training_steps': trained, 'sr_refinement_updates': sr_metadata.get('refinement_updates'),
        'sr_source_quality_passed': sr_metadata.get('source_quality_passed'),
        'codec_quality_qualified': renderer.get('codec_quality_qualified'),
        'renderer_source_quality_passed': renderer.get('metrics', {}).get('source_quality_passed'),
        'renderer_release_passed': renderer.get('release_passed',
            renderer.get('metrics', {}).get('source_validation', {}).get('release_passed')), 'gameplay_qualified': False,
        'all_preview_parameters_frozen': True, 'training_model_tensors_bitexact': True,
        'training_parameter_ids_preserved': True, 'training_optimizer_untouched': True,
        'current_world_tensors_copied': copied, 'training_model_tensor_sha256': source_hash,
        'verification_scope': 'Initialization before/after; current dynamics copied on every preview',
        'source_future_feedback': False, 'automatic_active_promotion': False}
    return client, sr, sr_metadata, proof
