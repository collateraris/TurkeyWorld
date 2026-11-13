"""Renderer-owned learned opacity; shared dynamics geometry stays untouched."""
from copy import deepcopy
from dataclasses import replace
import re

import torch
from torch import nn

KIND = 'private_renderer_geometry_opacity_v1'
PARAMETER_COUNT = 6241


def shared_camera_geometry(model):
    backend = getattr(model, 'vjepa_backend', None)
    camera = getattr(getattr(backend, 'renderer', None), 'geometry_reader', None)
    opacity = getattr(camera, 'opacity_network', None)
    if (camera is None or not isinstance(opacity, nn.Sequential)
            or not isinstance(opacity[0], nn.Linear) or opacity[0].in_features != 193
            or opacity[0].out_features != 32 or not isinstance(opacity[-1], nn.Linear)
            or opacity[-1].in_features != 32 or opacity[-1].out_features != 1
            or sum(p.numel() for p in opacity.parameters()) != PARAMETER_COUNT):
        raise ValueError('Private renderer opacity requires the existing192D CameraGeometry/6241-parameter MLP')
    return camera


def attach_geometry_opacity(student, original, *, initial=None):
    """Append four independently owned tensors without consuming any RNG."""
    if hasattr(student, 'geometry_opacity'):
        raise ValueError('Private renderer opacity is already attached')
    if sum(p.numel() for p in original.parameters()) != PARAMETER_COUNT:
        raise ValueError('Private opacity source parameter layout changed')
    with torch.random.fork_rng(devices=[]):
        opacity = deepcopy(original if initial is None else initial)
    if opacity is original or any(a is b for a, b in zip(opacity.parameters(), original.parameters())):
        raise AssertionError('Private opacity must never alias shared geometry parameters')
    student.geometry_opacity = opacity
    if hasattr(student, '_renderer_student_config'):
        student._renderer_student_config = {**student._renderer_student_config,
                                           'flow_geometry_opacity_enabled': True}
    return opacity


def enable_geometry_opacity(model):
    """Opt in one renderer instance; shared map/camera parameters are not changed."""
    camera = shared_camera_geometry(model)
    flow = getattr(model, 'flow_renderer', None)
    if flow is None or flow.kind != 'student' or flow.steps != 1:
        raise ValueError('Private opacity requires the actual compact one-step student renderer')
    config = replace(model.config, flow_geometry_opacity_enabled=True)
    if not hasattr(flow, 'geometry_opacity'):
        attach_geometry_opacity(flow, camera.opacity_network)
    model.config = config
    if hasattr(flow, '_renderer_student_config'):
        flow._renderer_student_config['flow_geometry_opacity_enabled'] = True
    return geometry_opacity_metadata(flow, model)


def configure_geometry_opacity(model, *, only_private=False):
    """Optionally isolate calibration gradients to the four new opacity tensors."""
    if type(only_private) is not bool:
        raise ValueError('only_private must be an explicit boolean')
    camera = shared_camera_geometry(model)
    if not only_private and any(p.requires_grad for p in camera.parameters()):
        raise ValueError('Original shared geometry must remain frozen during private calibration')
    enable_geometry_opacity(model)
    if only_private:
        model.eval().requires_grad_(False)
    if any(p.requires_grad for p in camera.parameters()):
        raise ValueError('Original shared geometry must remain frozen during private calibration')
    model.flow_renderer.geometry_opacity.requires_grad_(True)
    return {**geometry_opacity_metadata(model.flow_renderer, model),
            'only_private_trainable': only_private,
            'private_parameter_names': ['flow_renderer.geometry_opacity.'+n
                                        for n, _ in model.flow_renderer.geometry_opacity.named_parameters()]}


def geometry_opacity_metadata(student, teacher):
    from .renderer_student import _hash_tensors
    opacity = getattr(student, 'geometry_opacity', None)
    if opacity is None:
        return None
    original = shared_camera_geometry(teacher).opacity_network
    if (sum(p.numel() for p in opacity.parameters()) != PARAMETER_COUNT
            or list(opacity.state_dict()) != list(original.state_dict())
            or any(a.shape != original.state_dict()[n].shape for n, a in opacity.state_dict().items())):
        raise ValueError('Private opacity state layout differs from the shared source')
    return dict(kind=KIND, version=1, parameter_count=PARAMETER_COUNT, input_width=193, hidden_width=32,
        source_opacity_sha256=_hash_tensors(original.state_dict()),
        state_sha256=_hash_tensors(opacity.state_dict()),
        runtime_source_queries=False, runtime_ray_intersection_oracle=False)


def validate_geometry_opacity_source(header, teacher):
    from .renderer_student import _hash_tensors
    original = shared_camera_geometry(teacher).opacity_network
    if (not isinstance(header, dict) or header.get('kind') != KIND or header.get('version') != 1
            or header.get('parameter_count') != PARAMETER_COUNT or header.get('input_width') != 193
            or header.get('hidden_width') != 32 or header.get('runtime_source_queries') is not False
            or header.get('runtime_ray_intersection_oracle') is not False
            or header.get('source_opacity_sha256') != _hash_tensors(original.state_dict())
            or not isinstance(header.get('state_sha256'), str)
            or re.fullmatch('[0-9a-f]{64}', header['state_sha256']) is None):
        raise ValueError('Private opacity metadata/source identity is missing or changed')


def validate_geometry_opacity_bundle(student, teacher, header, base_bundle):
    actual = geometry_opacity_metadata(student, teacher)
    if actual is None:
        if header is not None or (base_bundle or {}).get('geometry_opacity_binding') is not None:
            raise ValueError('Compact Base requires its missing private geometry opacity')
        return
    validate_geometry_opacity_source(header, teacher)
    if actual != header or not isinstance(base_bundle, dict) or base_bundle.get('geometry_opacity_binding') != actual:
        raise ValueError('Private opacity weights and compact Base binding differ')
