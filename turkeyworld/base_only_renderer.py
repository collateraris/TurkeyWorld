"""Direct compact Base renderer retaining its calibrated geometry, without RF/VQGAN."""
from torch import nn


def enable_base_only(model):
    if not hasattr(model,'base_student'):
        raise ValueError('Base-only rendering requires a strictly loaded compact Base bundle')
    original=model.flow_renderer
    geometry=nn.Module()
    geometry.kind='base_only';geometry.steps=0
    if hasattr(original,'geometry_opacity'):geometry.geometry_opacity=original.geometry_opacity
    removed=sum(p.numel() for p in original.parameters())-sum(p.numel() for p in geometry.parameters())
    model.flow_renderer=geometry
    # Keep the validated training configuration as provenance; execution is
    # explicitly selected by this runtime flag, like the existing RF override.
    model.base_only_enabled=True
    return dict(protocol='base_only_renderer_v1',flow_matching=False,codec=False,solver_steps=0,
        removed_parameters=removed,base_training_steps=model.base_student_metadata.get('training_steps'),
        private_geometry_opacity_retained=hasattr(geometry,'geometry_opacity'))
