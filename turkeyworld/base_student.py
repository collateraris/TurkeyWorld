"""Compact Base RGB head over an unchanged teacher map and CameraGeometry.

The parameter budget covers every parameter in the new head. Map encoding and
camera geometry remain shared world components, reported separately. No engine,
source image, future observation or numerical state correction is used here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from .combat_objectives import combat_count_features
from .models import STATE_SCHEMA, WorldModelConfig, render_state_view
from .renderer_student import _hash_tensors
from .spatial_entities import entity_spatial_features
from .structured_state import STATE_DIM
from .transformer_model import TokenReadBlock


@dataclass
class BaseStudentConfig:
    width: int = 144
    depth: int = 3
    heads: int = 6
    geometry_width: int = 192
    image_size: int = 32

    def __post_init__(self):
        if (self.image_size != 32 or self.geometry_width != 192
                or min(self.width, self.depth, self.heads) < 1 or self.width % self.heads):
            raise ValueError('Compact Base requires native32, shared192D geometry and valid attention dimensions')


class CompactBaseHead(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or BaseStudentConfig()
        if isinstance(self.config, dict):
            self.config = BaseStudentConfig(**self.config)
        width = self.config.width
        self.scene_projection = nn.Linear(self.config.geometry_width, width)
        self.condition = nn.Sequential(nn.Linear(STATE_DIM+128, 128), nn.GELU(), nn.Linear(128, width))
        self.screen_position = nn.Sequential(nn.Linear(18, width), nn.GELU(), nn.Linear(width, width))
        self.participants = nn.Sequential(nn.Linear(36, width), nn.GELU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([TokenReadBlock(width, self.config.heads) for _ in range(self.config.depth)])
        self.norm = nn.LayerNorm(width)
        self.rgb = nn.Linear(width, 12)
        y, x = torch.meshgrid((torch.arange(16)+.5)/8-1, (torch.arange(16)+.5)/8-1, indexing='ij')
        xy = torch.stack([x, y], -1).reshape(1, 256, 2)
        angles = xy[..., None]*(torch.pi*(2**torch.arange(4)))
        self.register_buffer('screen_coordinates', torch.cat([xy, angles.sin().flatten(-2),
                                                             angles.cos().flatten(-2)], -1), persistent=False)
        if sum(value.numel() for value in self.parameters()) > 1_000_000:
            raise ValueError('Compact Base head exceeds one million real parameters')

    def forward(self, state, features, geometry):
        if state.ndim != 2 or state.shape[-1] != STATE_DIM or features.shape != (len(state), 128):
            raise ValueError('Base head expects state[B,20] and actual auxiliary features[B,128]')
        dtype = self.scene_projection.weight.dtype
        pose = render_state_view(state).to(dtype)
        condition = self.condition(torch.cat([pose, features.to(dtype)], -1))
        gate = geometry['render_gate'].to(dtype)
        objects = geometry['object_identity_tokens']+gate[..., None]*geometry['object_appearance_tokens']
        scene = torch.cat([geometry['ray_tokens'], objects], 1).to(dtype)
        participants = torch.cat([entity_spatial_features(pose), combat_count_features(state)], -1).to(dtype)
        context = torch.cat([self.scene_projection(scene), self.participants(participants), condition[:, None]], 1)
        weights = torch.cat([torch.ones_like(geometry['ray_tokens'][..., 0]), gate,
                             torch.ones(len(state), 3, device=state.device, dtype=dtype)], -1).to(dtype)
        queries = self.screen_position(self.screen_coordinates.to(dtype))+condition[:, None]
        for block in self.blocks:
            queries = block(queries, context, key_weights=weights)
        pixels = self.rgb(self.norm(queries)).reshape(-1, 16, 16, 3, 2, 2)
        return pixels.permute(0, 3, 1, 4, 2, 5).reshape(-1, 3, 32, 32).sigmoid()


def build_base_student(teacher=None, *, seed=173, config=None):
    if teacher is not None:
        _require_teacher(teacher)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return CompactBaseHead(config)


def _require_teacher(teacher):
    if (teacher.config.kind != 'teacher' or teacher.config.architecture not in {'vjepa2_ac', 'dreamer_rssm'}
            or teacher.config.image_size != 32 or not teacher.config.spatial_combat):
        raise ValueError('Compact Base requires the native32 spatial teacher architecture')


def _source_config(config):
    values = asdict(WorldModelConfig(**(asdict(config) if isinstance(config, WorldModelConfig) else config)))
    if values['architecture'] in {'vjepa2_ac', 'dreamer_rssm'}:
        values['architecture'] = 'spatial_explicit_world'
        values = {key: value for key, value in values.items()
                  if not key.startswith(('vjepa_', 'rssm_'))}
    return {key: value for key, value in values.items() if not key.startswith('flow_')
            and key not in {'renderer_mode', 'renderer_codec', 'renderer_codec_config', 'player_motion_mode'}}


def base_condition_identity(teacher):
    _require_teacher(teacher)
    state = teacher.state_dict()
    base = {key: value for key, value in state.items() if key.startswith('vjepa_backend.renderer.')}
    appearance = {key: value for key, value in state.items() if key.startswith('vjepa_backend.feature_head.')}
    if not base or not appearance:
        raise ValueError('Original Base/map/geometry and appearance weights are required')
    return {'base_map_geometry_sha256': _hash_tensors(base), 'appearance_sha256': _hash_tensors(appearance)}


def export_base_student(teacher, head, path, *, training_steps, metrics=None, provenance=None):
    from .training_pipeline import replace_file
    if not isinstance(training_steps, int) or isinstance(training_steps, bool) or training_steps < 1:
        raise ValueError('A Base bundle needs positive completed training steps')
    state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    if not all(bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError('Base weights must be finite')
    payload = {'format_version': 1, 'stage': 'base_student', 'state_schema': dict(STATE_SCHEMA),
        'config': asdict(head.config), 'source_teacher_config': asdict(teacher.config),
        'conditioning_identity': base_condition_identity(teacher), 'model': state,
        'model_sha256': _hash_tensors(state), 'training_steps': training_steps, 'metrics': metrics or {},
        'provenance': provenance or {}, 'head_parameters': sum(value.numel() for value in head.parameters()),
        'shared_components': 'Original teacher map encoder and CameraGeometry; not part of compact head',
        'quality_status': 'Distilled Base head; gameplay accuracy is not qualified'}
    if payload['head_parameters'] > 1_000_000:
        raise ValueError('Base head exceeds the requested parameter budget')
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix+'.tmp')
    torch.save(payload, temporary)
    replace_file(temporary, output)
    return {key: value for key, value in payload.items() if key != 'model'}


def prepare_base_student(teacher, path_or_payload):
    """Validate/build privately, returning(head, metadata, fullbundle); no mutation."""
    _require_teacher(teacher)
    payload = (torch.load(Path(path_or_payload), map_location='cpu', weights_only=False)
               if isinstance(path_or_payload, (str, Path)) else path_or_payload)
    if (not isinstance(payload, dict) or payload.get('format_version') != 1
            or payload.get('stage') != 'base_student' or payload.get('state_schema') != STATE_SCHEMA):
        raise ValueError('Expected a compatible compact Base bundle')
    steps = payload.get('training_steps')
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError('Base bundle has no positive completed training steps')
    from .renderer_student import validate_renderer_source_lineage
    validate_renderer_source_lineage(teacher, payload['source_teacher_config'])
    if (_source_config(payload['source_teacher_config']) != _source_config(teacher.config)
            or payload.get('conditioning_identity') != base_condition_identity(teacher)):
        raise ValueError('Base bundle source map/geometry/appearance or backbone configuration mismatch')
    state = payload.get('model')
    if (not isinstance(state, dict) or not all(isinstance(value, torch.Tensor)
            and bool(torch.isfinite(value).all()) for value in state.values())
            or payload.get('model_sha256') != _hash_tensors(state)):
        raise ValueError('Base bundle weights are invalid or their fingerprint differs')
    head = build_base_student(teacher, config=payload['config'])
    head.load_state_dict(state, strict=True)
    count = sum(value.numel() for value in head.parameters())
    if payload.get('head_parameters') != count or count > 1_000_000:
        raise ValueError('Base bundle parameter count differs from its compact configuration')
    parameter = next(teacher.parameters())
    head.to(device=parameter.device, dtype=parameter.dtype).eval().requires_grad_(False)
    metadata = {key: value for key, value in payload.items() if key != 'model'}
    if isinstance(path_or_payload, (str, Path)):
        metadata['checkpoint'] = str(Path(path_or_payload).resolve())
    return head, metadata, payload


def load_base_student(teacher, path_or_payload):
    head, metadata, payload = prepare_base_student(teacher, path_or_payload)
    teacher.base_student = head
    teacher.base_student_metadata = metadata
    teacher.base_student_bundle = payload
    return metadata
