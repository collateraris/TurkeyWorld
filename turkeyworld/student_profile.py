"""Explicit22.48M student profile; legacy checkpoint defaults stay unchanged."""
from __future__ import annotations

from .models import WorldModelConfig


STUDENT_22M_PARAMETERS = 22_475_013
STUDENT_22M_PROFILE = {
    'kind': 'student', 'architecture': 'vjepa2_ac', 'image_size': 128,
    'endless': True, 'renderer_mode': 'rectified_flow', 'flow_steps': 4,
    'vjepa_width': 320, 'vjepa_encoder_depth': 8, 'vjepa_predictor_depth': 5,
    'vjepa_heads': 5, 'vjepa_context_frames': 4, 'vjepa_patch_size': 16,
    'vjepa_readout_mlp_ratio': 4, 'flow_width': 192, 'flow_depth': 4, 'flow_heads': 4,
}


def student_config_22m(image_size=128, endless=True, **overrides):
    """Return an explicit configuration, also serialized into trained weights.

    The parameter target is for128px game frames. Overrides allow deliberate
    small test profiles; they do not change any global model defaults.
    """
    if image_size != 128:
        raise ValueError('The22.48M game profile is calibrated for128px frames')
    values = {**STUDENT_22M_PROFILE, 'image_size': image_size, 'endless': endless, **overrides}
    if values['kind'] != 'student' or values['architecture'] != 'vjepa2_ac' or values['renderer_mode'] != 'rectified_flow':
        raise ValueError('The student profile requires a V-JEPA student with Rectified Flow')
    return WorldModelConfig(**values)
