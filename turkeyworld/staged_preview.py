"""Truthfully labelled held-out independent-pose renderer previews."""
from datetime import datetime, timezone

import numpy as np
import torch

from .staged_curriculum import SnapshotData
from .visual_quality import strict_render_precision


def static_preview(model, directory, step):
    data = SnapshotData(directory, next(model.parameters()).device)
    indices = [(index % data.files, 0) for index in range(4)]
    batch = data.batch(4, indices=indices)
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with strict_render_precision(), torch.inference_mode():
            fmap = model.encode_map(batch['map'])
            state = batch['states']
            base = model.render_base(state, state.new_zeros(len(state), 128), fmap)
            image = model.flow_renderer.sample(base, state, map_condition=batch['map'])
    finally:
        for module, mode in modes:
            module.training = mode
    def rgb(value):
        return value.detach().float().clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    return {'context': rgb(base), 'source': rgb(batch['frames']), 'generated': rgb(image),
        'generated_float': image.detach().float().cpu(),
        'metadata': {'step': int(step), 'every': 10, 'context': 0, 'horizon': 0,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'source_held_out': True, 'split': 'val', 'kind': 'static_scene_query',
            'generated_finite': bool(torch.isfinite(image).all()),
            'future_source_feedback': False, 'renderer_only': True,
            'caption': 'Held-out independent source poses; renderer only; no dynamics prediction',
            'row_labels': {'context': 'BASE: held-out independent pose queries',
                'source': 'SOURCE: held-out static object targets',
                'generated': 'MODEL: static scene renderer; dynamics not evaluated'}}}
