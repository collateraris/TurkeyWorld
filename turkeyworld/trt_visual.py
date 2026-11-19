"""TensorRT boundaries for the learned RGB32 → RGB128 display renderer."""
import torch
from torch import nn

from .rrdb_material_sr import RRDBMaterialSR


class DisplayBackground(nn.Module):
    """Only the learned RRDB trunk and material readouts are exported."""
    def __init__(self, sr):
        super().__init__()
        for name in ('stem', 'trunk', 'up', 'material_head', 'seam_head', 'color_head', 'texture_head'):
            setattr(self, name, getattr(sr, name))

    forward = RRDBMaterialSR.background_layers


@torch.inference_mode()
def enable_visual_trt(runtime, factory):
    sr = runtime.sr
    if not isinstance(sr, RRDBMaterialSR):
        return []
    if sr.training:
        raise ValueError('Display TensorRT requires an upscaler in evaluation mode')
    sample = sr.stem.weight.new_zeros(1, 3, 32, 32)
    engine = factory('display_rrdb', DisplayBackground(sr).eval(), (sample,),
                     ('features', 'material_logits', 'seam_logits', 'color', 'texture'))
    sr.background_layers = engine.__call__
    for renderer in {getattr(runtime, 'display_base', None), runtime.feedback_renderer}:
        if renderer is not None:
            renderer.cache_game_map(runtime.feedback_grid)
    return ['display_rrdb']
