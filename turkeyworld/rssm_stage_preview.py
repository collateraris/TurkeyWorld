"""Current-stage free previews with four labelled samples of the full rollout."""
from __future__ import annotations

import numpy as np
import torch


STAGE_HORIZONS = (16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 450)
MOE_STAGE_HORIZONS = (4,8,16,*range(32,449,16),450)


def stage_preview_spec(state):
    """Read the active gate stage, never an inherited replay/next-H field."""
    training = state.get('rssm_training', {})
    moe=training.get('moe_training')
    if moe is not None:
        if not isinstance(moe,dict)or type(moe.get('current_horizon'))is not int or moe['current_horizon']not in MOE_STAGE_HORIZONS:
            raise ValueError('Current-stage MoE preview requires the exact active 31-stage horizon')
        completed=moe.get('updates')
        if type(completed)is not int or completed<0:
            raise ValueError('Current-stage MoE preview requires a committed MoE update clock')
        horizon=moe['current_horizon']
        return {'gated':True,'moe':True,'horizon':horizon,'context':1+(completed//16)%2,
            'display_offsets':np.rint(np.linspace(1,horizon,4)).astype(int).tolist(),
            'stage_index':moe.get('stage_index'),'plan_sha256':moe.get('sha256')}
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


def generate_stage_preview(client, batch, spec, *, collect_states=False):
    """Generate H own transitions and optionally retain their diagnostic states.

    The trace includes the initial state and uses this same rollout, without
    additional model calls or any future Source state as an input.
    """
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
        states = [state[0].detach().clone()] if collect_states else None
        for offset in range(1, horizon+1):
            result = client.transition(state, batch['actions'][:, offset-1], fmap, memory)
            state, memory = result['state'], result['memory']
            if (state.shape != (1, 20) or tuple(memory.shape) != memory_shape
                    or not bool(torch.isfinite(state).all() and torch.isfinite(memory).all())):
                raise FloatingPointError('Nonfinite current-stage preview state/h-z; no reset or repair')
            if states is not None:
                states.append(state[0].detach().clone())
            if offset in offsets:
                images.append(client.render(state, result['features'], fmap)[0].float())
        selected = torch.tensor([offset-1 for offset in offsets], device=batch['target_frames'].device)
        result = (torch.stack(images), batch['target_frames'][0].index_select(0, selected))
        return (*result, torch.stack(states)) if states is not None else result
