"""Validated RF renderer bundles for an unchanged teacher world and Base.

Only locally created/trusted torch checkpoints should be loaded. Bundles bind
to exact Base/map/appearance/codec conditions, while allowing the same teacher
dynamics weights to continue learning independently.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from .latent_flow_renderer import LatentFlowRenderer
from .models import STATE_SCHEMA, WorldModelConfig
from .vqgan import VQGAN, VQGANConfig


FORMAT_VERSION = 1
STAGE = 'renderer_student'
EXPERIMENTAL_CODEC_REASON = 'compressed renderer preview requested; validation preserved'


def _valid_solver_steps(value):
    return isinstance(value, int) and not isinstance(value, bool) and value in (1, 4)


def _hash_tensors(values):
    digest = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)],
            separators=(',', ':')).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def codec_tensor_fingerprint(codec):
    """Canonical tensor identity shared with compact-codec distillation."""
    return _hash_tensors(codec.state_dict())


def _world_config(config):
    return asdict(WorldModelConfig(**(asdict(config) if isinstance(config, WorldModelConfig) else config)))


def _backbone_config(config, *, codec_replacement=False):
    values = _world_config(config)
    if values['architecture'] in {'vjepa2_ac', 'dreamer_rssm'}:
        values['architecture'] = 'spatial_explicit_world'
        values = {key: value for key, value in values.items()
                  if not key.startswith(('vjepa_', 'rssm_'))}
    return {key: value for key, value in values.items()
            if not key.startswith(('flow_', 'enemy_rssm_')) and key not in {'renderer_mode', 'player_motion_mode'}
            and not (codec_replacement and key == 'renderer_codec_config')}


def _require_teacher(teacher):
    renderer = getattr(teacher, 'flow_renderer', None)
    if (teacher.config.kind != 'teacher' or teacher.config.architecture not in {'vjepa2_ac', 'dreamer_rssm'}
            or not isinstance(renderer, LatentFlowRenderer)
            or not bool(renderer.codec_loaded)):
        raise ValueError('Renderer student requires a teacher V-JEPA world with a loaded VQGAN codec')
    return renderer


def renderer_condition_identity(teacher):
    """Hash actual conditioning weights; learned dynamics are not frozen here."""
    flow = _require_teacher(teacher)
    state = teacher.state_dict()
    base = {key: value for key, value in state.items() if key.startswith('vjepa_backend.renderer.')}
    appearance = {key: value for key, value in state.items() if key.startswith('vjepa_backend.feature_head.')}
    if not base or not appearance:
        raise ValueError('Teacher Base/map and appearance weights must be identifiable')
    return {
        'base_map_sha256': _hash_tensors(base),
        'appearance_sha256': _hash_tensors(appearance),
        'codec_sha256': _hash_tensors(flow.codec.state_dict()),
        'latent_normalization_sha256': _hash_tensors({'mean': flow.latent_mean, 'std': flow.latent_std}),
        'fixed_noise_sha256': _hash_tensors({'fixed_noise': flow.fixed_noise}),
    }


def validate_renderer_source_lineage(teacher, source_config):
    """A copied renderer may cross dynamics architectures only via migration.

    Tensor identities are still checked independently by every bundle loader.
    This explicit lineage check prevents a scratch RSSM from claiming a prior
    teacher's trained image assets through configuration normalization alone.
    """
    source = _world_config(source_config)
    validate_dual_renderer_lineage(teacher, source)
    if source['architecture'] == teacher.config.architecture:
        return
    report = getattr(teacher, 'rssm_migration', None)
    if (source['architecture'] != 'vjepa2_ac' or teacher.config.architecture != 'dreamer_rssm'
            or not isinstance(report, dict) or report.get('format_version') != 1
            or report.get('source_architecture') != 'vjepa2_ac'
            or report.get('architecture') != 'dreamer_rssm'
            or report.get('frozen_render_assets_bitexact') is not True
            or not isinstance(report.get('source_teacher_config'), dict)
            or _world_config(report['source_teacher_config']) != source):
        raise ValueError('Cross-architecture renderer reuse requires explicit RSSM migration lineage')
    identity = report.get('source_renderer_conditioning_identity')
    current = renderer_condition_identity(teacher)
    if (not isinstance(identity, dict) or any(identity.get(key) != current[key]
            for key in ('base_map_sha256', 'appearance_sha256'))):
        raise ValueError('RSSM migration renderer lineage does not match the copied frozen assets')


def _enemy_config(config):
    return {key: value for key, value in config.items() if key.startswith('enemy_rssm_')}


def _validated_enemy_resize_source(teacher):
    """Validate an explicit enemy-only resize; image identity stays strict."""
    report = getattr(teacher, 'enemy_rssm_resize', None)
    if (not isinstance(report, dict) or report.get('format_version') != 1
            or report.get('kind') != 'resize_independent_enemy_rssm'
            or report.get('player_and_shared_model_tensors_bitexact') is not True
            or report.get('player_and_shared_adam_states_bitexact') is not True
            or report.get('frozen_render_assets_bitexact') is not True
            or not isinstance(report.get('source_config'), dict)
            or not isinstance(report.get('target_config'), dict)):
        raise ValueError('Enemy renderer reuse requires explicit verified enemy-RSSM resize lineage')
    previous, resized = _world_config(report['source_config']), _world_config(report['target_config'])
    target = _world_config(teacher.config)
    changes = {key for key in previous if previous[key] != resized[key]}
    allowed = report.get('allowed_config_changes')
    for name in ('source', 'target'):
        expected = hashlib.sha256(json.dumps(report[name+'_config'], sort_keys=True).encode()).hexdigest()
        if report.get(name+'_config_sha256') != expected:
            raise ValueError('Enemy-RSSM resize configuration fingerprint differs')
    if (not previous['enemy_rssm_enabled'] or not resized['enemy_rssm_enabled']
            or not changes or not changes <= {'enemy_rssm_deter_dim', 'enemy_rssm_hidden_dim'}
            or not isinstance(allowed, list) or len(allowed) != len(changes) or set(allowed) != changes
            or _enemy_config(resized) != _enemy_config(target)
            or any(resized[key] != target[key] for key in resized
                   if not key.startswith('flow_') and key not in {'renderer_mode', 'renderer_codec_config'})
            or _backbone_config(previous, codec_replacement=True) != _backbone_config(resized, codec_replacement=True)
            or _backbone_config(resized, codec_replacement=True) != _backbone_config(target, codec_replacement=True)
            or report.get('source_renderer_conditioning_identity') != renderer_condition_identity(teacher)):
        raise ValueError('Enemy-RSSM resize lineage does not match the unchanged player/image assets/configuration')
    return previous


def validate_dual_renderer_lineage(teacher, source_config):
    """Only verified append/resize chains may reuse a prior image bundle."""
    source, target = _world_config(source_config), _world_config(teacher.config)
    old, new = _enemy_config(source), _enemy_config(target)
    if old == new or not source['enemy_rssm_enabled'] and not target['enemy_rssm_enabled']:
        return
    if source['enemy_rssm_enabled'] and target['enemy_rssm_enabled']:
        previous = _validated_enemy_resize_source(teacher)
        if (old != _enemy_config(previous)
                or _backbone_config(source, codec_replacement=True) != _backbone_config(previous, codec_replacement=True)):
            raise ValueError('Renderer source does not belong to the verified enemy-RSSM resize')
        return
    report = getattr(teacher, 'dual_rssm_migration', None)
    if (source['enemy_rssm_enabled'] or not target['enemy_rssm_enabled']
            or not isinstance(report, dict) or report.get('format_version') != 1
            or report.get('kind') != 'append_independent_enemy_rssm'
            or report.get('source_enemy_rssm_enabled') is not False
            or report.get('enemy_rssm_enabled') is not True
            or report.get('old_model_tensors_bitexact') is not True
            or report.get('frozen_render_assets_bitexact') is not True
            or not isinstance(report.get('source_config'), dict)
            or not isinstance(report.get('target_config'), dict)):
        raise ValueError('Legacy renderer reuse requires explicit verified dual-RSSM migration lineage')
    previous, migrated = _world_config(report['source_config']), _world_config(report['target_config'])
    if _enemy_config(migrated) != new:
        resize_previous = _validated_enemy_resize_source(teacher)
        if (_enemy_config(migrated) != _enemy_config(resize_previous)
                or _backbone_config(migrated, codec_replacement=True) != _backbone_config(resize_previous, codec_replacement=True)):
            raise ValueError('Dual-RSSM append and enemy resize do not form the same verified lineage')
    if (previous['enemy_rssm_enabled'] or not migrated['enemy_rssm_enabled']
            or any(previous[key] != migrated[key] for key in previous if not key.startswith('enemy_rssm_'))
            or _backbone_config(previous, codec_replacement=True) != _backbone_config(source, codec_replacement=True)
            or _backbone_config(migrated, codec_replacement=True) != _backbone_config(target, codec_replacement=True)
            or report.get('source_renderer_conditioning_identity') != renderer_condition_identity(teacher)):
        raise ValueError('Dual-RSSM renderer lineage does not match the unchanged image assets/configuration')


def _validate_compact_codec(teacher, codec, provenance, *, allow_unqualified_codec=False):
    if not isinstance(allow_unqualified_codec, bool):
        raise ValueError('allow_unqualified_codec must be an explicit boolean')
    source = _require_teacher(teacher)
    steps = provenance.get('training_steps')
    gate = provenance.get('quality_gate')
    passed = gate.get('passed') if isinstance(gate, dict) else None
    if (provenance.get('stage') != 'distilled_vqgan_codec'
            or not isinstance(steps, int) or isinstance(steps, bool) or steps < 1
            or provenance.get('frozen') is not True or provenance.get('fixed_codebook') is not True
            or not isinstance(passed, bool)
            or provenance.get('teacher_codec_fingerprint') != codec_tensor_fingerprint(source.codec)):
        raise ValueError('Compact codec needs a qualified frozen distillation from this teacher codec')
    if not passed:
        acceptance = provenance.get('acceptance', {})
        if (not allow_unqualified_codec or not isinstance(acceptance, dict)
                or acceptance.get('scope') != 'experimental_preview'
                or acceptance.get('quality_gate_passed') is not False
                or acceptance.get('allow_unqualified_codec') is not True
                or acceptance.get('reason') != EXPERIMENTAL_CODEC_REASON):
            raise ValueError('Unqualified codec requires explicit experimental-preview acceptance and allow_unqualified_codec=True')
    config, original = codec.config, source.codec.config
    if (config.image_size != original.image_size or config.image_size != 32
            or config.latent_size != 8 or config.downsample_factor != 4
            or config.embedding_dim != 16 or config.n_embed != 1488
            or (config.latent_size, config.embedding_dim, config.n_embed) !=
               (original.latent_size, original.embedding_dim, original.n_embed)
            or sum(parameter.numel() for parameter in codec.parameters()) > 1_000_000
            or codec.get_config() != asdict(VQGANConfig(**provenance.get('codec_config', {})))):
        raise ValueError('Compact codec must fit one million total parameters and preserve the teacher latent/codebook geometry')
    if _hash_tensors(codec.quantize.state_dict()) != _hash_tensors(source.codec.quantize.state_dict()):
        raise ValueError('Compact codec changed the teacher codebook or quantizer state')
    normalization = provenance.get('latent_normalization', {})
    for key, value in (('mean', source.latent_mean), ('std', source.latent_std)):
        try:
            actual = torch.as_tensor(normalization[key], dtype=value.dtype).reshape(-1)
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise ValueError('Compact codec lacks exact teacher latent normalization') from error
        if not torch.equal(actual.cpu(), value.detach().cpu().reshape(-1)):
            raise ValueError('Compact codec latent normalization differs from the teacher')
    if not all(bool(torch.isfinite(value).all()) for value in codec.state_dict().values()):
        raise ValueError('Compact codec weights must be finite')


def _load_compact_codec(teacher, path, *, allow_unqualified_codec=False):
    payload = torch.load(Path(path), map_location='cpu', weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get('model'), dict):
        raise ValueError('Expected a distilled compact-codec checkpoint')
    with torch.random.fork_rng(devices=[]):
        codec = VQGAN(payload['codec_config']).eval().requires_grad_(False)
    codec.load_state_dict(payload['model'], strict=True)
    provenance = {key: copy.deepcopy(value) for key, value in payload.items() if key != 'model'}
    gate = provenance.get('quality_gate')
    if allow_unqualified_codec is True and isinstance(gate, dict) and gate.get('passed') is False:
        provenance['acceptance'] = {'scope': 'experimental_preview', 'quality_gate_passed': False,
            'reason': EXPERIMENTAL_CODEC_REASON, 'allow_unqualified_codec': True}
    _validate_compact_codec(teacher, codec, provenance, allow_unqualified_codec=allow_unqualified_codec)
    provenance['checkpoint'] = str(Path(path).resolve())
    provenance['codec_fingerprint'] = codec_tensor_fingerprint(codec)
    provenance['parameters'] = sum(parameter.numel() for parameter in codec.parameters())
    return codec, provenance


def renderer_student_config(teacher, *, kind='rectified', steps=4, width=256,
                            depth=5, heads=8, mlp_hidden=896):
    _require_teacher(teacher)
    if kind not in {'rectified', 'student'} or not _valid_solver_steps(steps):
        raise ValueError('This distilled renderer requires an explicit integer of one or four Euler steps')
    return replace(teacher.config, flow_kind='student', renderer_mode='rectified_flow',
        flow_steps=steps, flow_width=width, flow_depth=depth, flow_heads=heads,
        flow_mlp_hidden=mlp_hidden, flow_adapter_width=0, flow_adapter_depth=0,
        flow_adapter_heads=0)


def _construct(config, source, *, share_codec, codec=None):
    student = LatentFlowRenderer(kind=config.flow_kind or config.kind,
        image_size=config.image_size, mode=config.renderer_mode, steps=config.flow_steps,
        width=config.flow_width, depth=config.flow_depth, heads=config.flow_heads,
        mlp_ratio=config.flow_mlp_ratio, mlp_hidden=config.flow_mlp_hidden,
        adapter_width=0, adapter_depth=0, adapter_heads=0,
        codec_config=config.renderer_codec_config, spatial_conditioning=config.spatial_combat,
        patch_size=config.flow_patch_size, block_conditioning=config.flow_block_conditioning,
        improved_loss=config.game_training_v5)
    parameter = source.image_input.weight
    student.to(device=parameter.device, dtype=parameter.dtype)
    # Attach after casting the velocity net: a mixed-precision source codec
    # and its normalization must remain byte-identical, including their dtype.
    actual_codec = source.codec if codec is None else codec
    if actual_codec is not source.codec:
        codec_parameter = next(source.codec.parameters())
        actual_codec.to(device=codec_parameter.device, dtype=codec_parameter.dtype)
    student.codec = actual_codec if share_codec else copy.deepcopy(actual_codec)
    for name in ('codec_loaded', 'latent_mean', 'latent_std', 'fixed_noise'):
        setattr(student, name, getattr(source, name).detach().clone())
    student._renderer_student_config = asdict(config)
    return student


def build_renderer_student(teacher, *, kind='rectified', steps=4, seed=173,
                           width=256, depth=5, heads=8, mlp_hidden=896,
                           codec_checkpoint=None, allow_unqualified_codec=False):
    """New velocity network, exact shared frozen codec, same spatial conditions."""
    source = _require_teacher(teacher)
    config = renderer_student_config(teacher, kind=kind, steps=steps, width=width,
        depth=depth, heads=heads, mlp_hidden=mlp_hidden)
    if not isinstance(allow_unqualified_codec, bool):
        raise ValueError('allow_unqualified_codec must be an explicit boolean')
    codec, provenance = (_load_compact_codec(teacher, codec_checkpoint,
        allow_unqualified_codec=allow_unqualified_codec) if codec_checkpoint is not None else (None, None))
    if codec is not None:
        config = replace(config, renderer_codec_config=codec.get_config())
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        student = _construct(config, source, share_codec=True, codec=codec)
    if provenance is not None:
        student._renderer_codec_distillation = provenance
        student._renderer_allow_unqualified_codec = allow_unqualified_codec
    if hasattr(teacher, 'base_student'):
        from .base_student import prepare_base_student
        bundle = getattr(teacher, 'base_student_bundle', None)
        if not isinstance(bundle, dict):
            raise ValueError('Compact Base conditioning needs its validated distillation bundle')
        head, _, bundle = prepare_base_student(teacher, bundle)
        if _hash_tensors(head.state_dict()) != _hash_tensors(teacher.base_student.state_dict()):
            raise ValueError('Active compact Base differs from its approved distillation bundle')
        student._renderer_base_bundle = copy.deepcopy(bundle)
        student._renderer_base_identity = _hash_tensors(head.state_dict())
    return student


def _student_config(student, teacher):
    raw = getattr(student, '_renderer_student_config', None)
    if not isinstance(raw, dict):
        raise ValueError('Build the renderer student with build_renderer_student before export')
    config = WorldModelConfig(**raw)
    validate_dual_renderer_lineage(teacher, asdict(config))
    replacement = isinstance(getattr(student, '_renderer_codec_distillation', None), dict)
    if (config.kind != 'teacher' or config.flow_kind != 'student'
            or config.renderer_mode != 'rectified_flow' or not _valid_solver_steps(config.flow_steps)
            or student.kind != 'student' or student.mode != 'rectified_flow'
            or not _valid_solver_steps(student.steps) or student.steps != config.flow_steps
            or not bool(student.codec_loaded)
            or hasattr(student, 'residual_adapter')
            or _backbone_config(config, codec_replacement=replacement) !=
               _backbone_config(teacher.config, codec_replacement=replacement)):
        raise ValueError('Student renderer configuration changed its teacher backbone or one/four-step RF protocol')
    for name in ('flow_patch_size', 'flow_block_conditioning', 'spatial_combat'):
        if getattr(config, name) != getattr(teacher.config, name):
            raise ValueError('Student spatial/patch conditioning differs from the source teacher')
    return config


def _shared_identity(student):
    return {'codec_sha256': _hash_tensors(student.codec.state_dict()),
        'latent_normalization_sha256': _hash_tensors({'mean': student.latent_mean, 'std': student.latent_std}),
        'fixed_noise_sha256': _hash_tensors({'fixed_noise': student.fixed_noise})}


def export_renderer_student(teacher, student, path, *, training_steps, metrics=None):
    """Write a standalone trained renderer, never a replacement world checkpoint."""
    if not isinstance(training_steps, int) or isinstance(training_steps, bool) or training_steps < 1:
        raise ValueError('A playable renderer bundle requires positive completed training_steps')
    config = _student_config(student, teacher)
    identity = renderer_condition_identity(teacher)
    base_bundle = getattr(student, '_renderer_base_bundle', None)
    base_identity = getattr(student, '_renderer_base_identity', None)
    if base_bundle is not None:
        from .base_student import prepare_base_student
        head, _, _ = prepare_base_student(teacher, base_bundle)
        if (base_identity != _hash_tensors(head.state_dict()) or not hasattr(teacher, 'base_student')
                or base_identity != _hash_tensors(teacher.base_student.state_dict())):
            raise ValueError('Compact Base RF conditioning changed after its renderer was built')
    elif hasattr(teacher, 'base_student'):
        raise ValueError('Renderer RF bundle did not train with the current compact Base')
    provenance = getattr(student, '_renderer_codec_distillation', None)
    if provenance is not None:
        _validate_compact_codec(teacher, student.codec, provenance,
            allow_unqualified_codec=getattr(student, '_renderer_allow_unqualified_codec', False))
    shared = _shared_identity(student)
    if any(identity[name] != value for name, value in shared.items()
            if name != 'codec_sha256' or provenance is None):
        raise ValueError('Student codec, latent normalization or fixed noise changed')
    state = {name: value.detach().cpu().clone() for name, value in student.state_dict().items()}
    if not all(bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError('Renderer bundle has non-finite weights or normalization')
    payload = {'format_version': FORMAT_VERSION, 'stage': STAGE,
        'state_schema': copy.deepcopy(STATE_SCHEMA), 'source_teacher_config': asdict(teacher.config),
        'conditioning_identity': identity, 'config': asdict(config), 'model': state,
        'student_codec_identity': shared, 'codec_distillation': copy.deepcopy(provenance),
        'base_student_bundle': copy.deepcopy(base_bundle), 'base_student_identity': base_identity,
        'codec_quality_qualified': provenance['quality_gate']['passed'] if provenance is not None else None,
        'experimental_preview': bool(provenance is not None and provenance['quality_gate']['passed'] is False),
        'gameplay_qualified': False,
        'training_steps': training_steps, 'metrics': metrics or {},
        'solver': 'euler', 'solver_steps': student.steps,
        'quality_status': 'Distilled renderer; gameplay accuracy is not qualified'}
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name+f'.{os.getpid()}.tmp')
    try:
        torch.save(payload, temporary)
        for attempt in range(21):
            try:
                os.replace(temporary, output)
                break
            except PermissionError:
                if attempt == 20:
                    raise
                time.sleep(.1)
    finally:
        temporary.unlink(missing_ok=True)
    return {key: value for key, value in payload.items() if key not in {'model', 'base_student_bundle'}}


def load_renderer_student(teacher, path, *, require_trained=True, allow_unqualified_codec=False):
    """Validate all tensors/dependencies before replacing only teacher.flow_renderer."""
    payload = torch.load(Path(path), map_location='cpu', weights_only=False)
    if not isinstance(allow_unqualified_codec, bool):
        raise ValueError('allow_unqualified_codec must be an explicit boolean')
    if (payload.get('format_version') != FORMAT_VERSION or payload.get('stage') != STAGE
            or payload.get('state_schema') != STATE_SCHEMA):
        raise ValueError('Expected a compatible standalone renderer-student bundle')
    steps = payload.get('training_steps')
    if require_trained and (not isinstance(steps, int) or isinstance(steps, bool) or steps < 1):
        raise ValueError('Renderer student has no positive completed training steps')
    if payload.get('solver') != 'euler' or not _valid_solver_steps(payload.get('solver_steps')):
        raise ValueError('Renderer bundle must use an integer of one or four Euler steps')
    validate_renderer_source_lineage(teacher, payload['source_teacher_config'])
    if (_backbone_config(payload['source_teacher_config']) != _backbone_config(teacher.config)
            or payload.get('conditioning_identity') != renderer_condition_identity(teacher)):
        raise ValueError('Renderer bundle source Base/map, appearance, codec or configuration mismatch')
    # Renderer bundles may predate a dynamics-only head upgrade. Keep that
    # live mode rather than reverting it to the bundle's historical setting.
    renderer_config = dict(payload['config'])
    live_config = asdict(teacher.config)
    renderer_config.update({key: value for key, value in live_config.items()
        if key.startswith(('rssm_', 'vjepa_', 'enemy_rssm_')) or key in {'architecture', 'player_motion_mode'}})
    config = WorldModelConfig(**renderer_config)
    if (not _valid_solver_steps(config.flow_steps)
            or payload['solver_steps'] != config.flow_steps):
        raise ValueError('Renderer bundle solver metadata differs from its configured sampler')
    provenance = payload.get('codec_distillation')
    base_bundle = payload.get('base_student_bundle')
    pending_base = None
    if base_bundle is not None:
        from .base_student import prepare_base_student
        head, base_metadata, base_bundle = prepare_base_student(teacher, base_bundle)
        if payload.get('base_student_identity') != _hash_tensors(head.state_dict()):
            raise ValueError('Renderer bundle compact Base fingerprint differs from its approved weights')
        if hasattr(teacher, 'base_student'):
            if _hash_tensors(teacher.base_student.state_dict()) != payload['base_student_identity']:
                raise ValueError('Already loaded compact Base differs from this renderer RF conditioning')
            head = teacher.base_student
        pending_base = (head, base_metadata, base_bundle)
    elif hasattr(teacher, 'base_student') or payload.get('base_student_identity') is not None:
        raise ValueError('Renderer bundle has no matching approved compact Base conditioning')
    with torch.random.fork_rng(devices=[]):
        # Build a private codec first: strict-load failures cannot touch teacher.
        codec = VQGAN(config.renderer_codec_config).eval().requires_grad_(False) if provenance is not None else None
        student = _construct(config, _require_teacher(teacher), share_codec=False, codec=codec)
    if provenance is not None:
        if not isinstance(provenance, dict):
            raise ValueError('Compact codec replacement needs explicit distillation provenance')
        student._renderer_codec_distillation = provenance
        student._renderer_allow_unqualified_codec = allow_unqualified_codec
    _student_config(student, teacher)
    state = payload.get('model')
    if (not isinstance(state, dict) or not all(isinstance(value, torch.Tensor)
            and bool(torch.isfinite(value).all()) for value in state.values())):
        raise ValueError('Renderer bundle must contain finite tensor weights')
    student.load_state_dict(state, strict=True)
    student._check_codec()
    if provenance is not None:
        _validate_compact_codec(teacher, student.codec, provenance,
            allow_unqualified_codec=allow_unqualified_codec)
        passed = provenance['quality_gate']['passed']
        if ((not passed and (payload.get('codec_quality_qualified') is not False
                             or payload.get('experimental_preview') is not True))
                or (passed and (payload.get('codec_quality_qualified', True) is not True
                                or payload.get('experimental_preview', False) is not False))):
            raise ValueError('Renderer bundle codec qualification metadata differs from its original validation')
        if payload.get('gameplay_qualified', False) is not False:
            raise ValueError('Renderer component acceptance cannot qualify gameplay')
    shared = _shared_identity(student)
    if (payload.get('student_codec_identity', shared) != shared
            or any(payload['conditioning_identity'][name] != value for name, value in shared.items()
                   if name != 'codec_sha256' or provenance is None)):
        raise ValueError('Renderer bundle codec, normalization or noise tensors differ from source identity')
    if provenance is None:
        student.codec = teacher.flow_renderer.codec
    student.eval()
    if pending_base is not None:
        teacher.base_student, teacher.base_student_metadata, teacher.base_student_bundle = pending_base
        student._renderer_base_bundle = copy.deepcopy(base_bundle)
        student._renderer_base_identity = payload['base_student_identity']
    teacher.flow_renderer = student
    teacher.config = config
    metadata = {key: value for key, value in payload.items() if key not in {'model', 'base_student_bundle'}}
    if pending_base is not None:
        metadata['base_student'] = base_metadata
    metadata['checkpoint'] = str(Path(path).resolve())
    teacher.renderer_student_metadata = metadata
    return metadata


def initialize_renderer_student(teacher, student, checkpoint, *, allow_unqualified_codec=False):
    """Initialize compatible RF weights without changing the teacher or counters.

    The source can retain its solver or initialize a new one-step student from
    a four-step bundle. No optimizer state or training-step counter is copied.
    """
    target_config = _student_config(student, teacher)
    # Loading on this isolated module view changes its registry/config only.
    # Source teacher tensors stay shared read-only; validation builds all new
    # renderer assets privately before committing them to the isolated view.
    view = copy.copy(teacher)
    view._modules = dict(teacher._modules)
    view.config = copy.deepcopy(teacher.config)
    metadata = load_renderer_student(view, checkpoint,
        allow_unqualified_codec=allow_unqualified_codec)
    source_config = asdict(view.config)
    target_values = asdict(target_config)
    source_steps, target_steps = source_config.pop('flow_steps'), target_values.pop('flow_steps')
    if (source_config != target_values or (source_steps != target_steps
            and (source_steps, target_steps) != (4, 1))):
        raise ValueError('Renderer initialization permits only identical conditions and four-to-one solver conversion')
    if (_shared_identity(view.flow_renderer) != _shared_identity(student)
            or metadata.get('base_student_identity') != getattr(student, '_renderer_base_identity', None)):
        raise ValueError('Renderer initialization codec, normalization, noise or Base conditions differ')
    source_state, target_state = view.flow_renderer.state_dict(), student.state_dict()
    if (source_state.keys() != target_state.keys() or any(source_state[name].shape != value.shape
            or source_state[name].dtype != value.dtype
            for name, value in target_state.items())):
        raise ValueError('Renderer initialization tensor shapes differ')
    # Shared codec tensors were already proven identical. Do not even bump
    # their tensor versions by copying them onto the teacher's shared module.
    with torch.no_grad():
        for name, value in target_state.items():
            if not name.startswith('codec.'):
                value.copy_(source_state[name])
    return {**metadata, 'initialization_source_solver_steps': source_steps,
        'target_solver_steps': target_steps, 'optimizer_state_copied': False,
        'training_counters_copied': False}
