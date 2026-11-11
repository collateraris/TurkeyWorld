"""Record current RSSM visits on training tapes for offline Source labeling.

Only the initial recorded RGB/state/map initialize each trajectory. Future
Source arrays are measurements and never replace predicted state or memory.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from .models import load_world_checkpoint
from .staged_curriculum import StagedSequenceData


def collect_own_training_states(world, data_root, output, *, count=32, batch_size=8, seed=173, device='cuda'):
    torch.set_num_threads(4)
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.init()
    model, metadata = load_world_checkpoint(world, 'cpu')
    step = int(metadata['training_steps'])
    del metadata
    gc.collect()
    if model.config.architecture != 'dreamer_rssm' or not model.config.navigation_only:
        raise ValueError('Own contact labels require the navigation RSSM profile')
    model.to(device).float().eval().requires_grad_(False)
    data = StagedSequenceData(Path(data_root)/'long/rgb/train', device, mode='stream', cache_bytes=64*1024**2)
    if count > len(data.episodes):
        raise ValueError('Own visits require distinct training episodes')
    rng = np.random.default_rng(seed)
    selected = sorted(rng.choice(len(data.episodes), count, replace=False).tolist())
    predicted, reference, actions, maps = [], [], [], []
    started = time.perf_counter()
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []), torch.inference_mode():
        torch.manual_seed(seed)
        for first in range(0, count, batch_size):
            chosen = selected[first:first+batch_size]
            pairs = np.column_stack([chosen, np.zeros(len(chosen), dtype=np.int64)])
            batch = data.batch(len(chosen), context=1, horizon=450, indices=pairs)
            fmap = model.encode_map(batch['map'])
            current = batch['states'][:, 0].float()
            memory = model.initialize_from_image(batch['context_frames'], batch['states'][:, :1], fmap)
            path = [current.cpu().numpy().copy()]
            for offset in range(450):
                result = model.transition(current, batch['actions'][:, offset], fmap, memory)
                current, memory = result['state'], result['memory']
                if not bool(torch.isfinite(current).all()) or not bool(torch.isfinite(memory).all()):
                    raise FloatingPointError('Invalid own RSSM visits; no state repairs allowed')
                path.append(current.cpu().numpy().copy())
            predicted.append(np.stack(path, 1))
            reference.append(batch['states'][:, :451].cpu().numpy())
            actions.append(batch['actions'].cpu().numpy())
            maps.append(batch['map'].cpu().numpy())
    panel = {'split': 'train', 'source_checkpoint': str(Path(world).resolve()),
        'source_optimizer_step': step, 'source_episode_paths': [data.episodes[i]['path'] for i in selected],
        'context': 1, 'horizon': 450, 'seed': seed, 'future_source_feedback': False,
        'engine_corrections': 0, 'state_resets': 0, 'memory_resets': 0,
        'collection_seconds': time.perf_counter()-started}
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError('Own-state collections are immutable; choose a new output')
    np.savez_compressed(output, predicted=np.concatenate(predicted), reference=np.concatenate(reference),
        actions=np.concatenate(actions), maps=np.concatenate(maps), split=np.asarray('train'),
        panel=np.asarray(json.dumps(panel)))
    output.with_suffix('.json').write_text(json.dumps(panel, indent=2)+'\n', encoding='utf-8')
    return panel


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('world', 'data-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=173)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args(argv)
    if min(args.count, args.batch_size) < 1:
        parser.error('Positive episode and batch counts required')
    print(json.dumps(collect_own_training_states(args.world, args.data_root, args.output,
        count=args.count, batch_size=args.batch_size, seed=args.seed, device=args.device), indent=2), flush=True)


if __name__ == '__main__':
    main()
