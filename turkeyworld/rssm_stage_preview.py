"""Current-stage free previews with four labelled samples of the full rollout."""
from __future__ import annotations

import numpy as np
import torch


STAGE_HORIZONS = (16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 450)


def stage_preview_spec(state):
    """Read the active gate stage, never an inherited replay/next-H field."""
    training = state.get('rssm_training', {})
    gate = training.get('loss_gated_curriculum')
    if gate is None:
        return {'gated': False, 'horizon': 4, 'context': 1, 'display_offsets': [1, 2, 3, 4]}
    horizon = gate.get('current_horizon')
    if type(horizon) is not int or horizon not in STAGE_HORIZONS:
        raise ValueError('Current-stage preview requires the exact active loss-gated horizon')
    completed = training.get('combat_completed', 0)
    if type(completed) is not int or completed < 0:
        raise ValueError('Current-stage preview requires a valid committed combat clock')
    return {'gated': True, 'horizon': horizon, 'context': 1+(completed//16)%2,
        'display_offsets': np.rint(np.linspace(1, horizon, 4)).astype(int).tolist(),
        'stage_index': gate.get('stage_index'), 'plan_sha256': gate.get('sha256')}


def stage_preview_batch(data, spec):
    """Choose the first complete held-out episode for the actual full H."""
    required = spec['context']+spec['horizon']
    eligible = next((index for index, episode in enumerate(data.episodes)
                     if len(episode['frames']) >= required), None)
    if eligible is None:
        raise ValueError('No complete held-out episode fits the current preview horizon/context')
    return data.batch(1, context=spec['context'], horizon=spec['horizon'],
                      indices=np.asarray([[eligible, 0]], dtype=np.int64))


def generate_stage_preview(client, batch, spec):
    """Generate all H own transitions; render only the four displayed offsets."""
    horizon, context = spec['horizon'], spec['context']
    offsets = spec['display_offsets']
    if (batch['actions'].shape[:2] != (1, horizon)
            or batch['states'].shape[1] < context+horizon
            or batch['context_frames'].shape[:2] != (1, context)):
        raise ValueError('Preview batch must match its exact active horizon and observed context')
    with torch.inference_mode():
        fmap = client.encode_map(batch['map'])
        state = batch['states'][:, context-1].float()
        memory = client.initialize_from_image(batch['context_frames'], batch['states'][:, :context], fmap)
        memory_shape = tuple(memory.shape)
        images = []
        for offset in range(1, horizon+1):
            result = client.transition(state, batch['actions'][:, offset-1], fmap, memory)
            state, memory = result['state'], result['memory']
            if (state.shape != (1, 20) or tuple(memory.shape) != memory_shape
                    or not bool(torch.isfinite(state).all() and torch.isfinite(memory).all())):
                raise FloatingPointError('Nonfinite current-stage preview state/h-z; no reset or repair')
            if offset in offsets:
                images.append(client.render(state, result['features'], fmap)[0].float())
        selected = torch.tensor([offset-1 for offset in offsets], device=batch['target_frames'].device)
        return torch.stack(images), batch['target_frames'][0].index_select(0, selected)
