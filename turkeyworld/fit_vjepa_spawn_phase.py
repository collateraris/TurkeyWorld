"""Fit a separate V-JEPA spawn candidate on labelled TRAIN respawns."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .anchor_spawn import anchor_target_labels, hierarchical_spawn_logits
from .distill_renderer import _save
from .spawn import BEHAVIOR_VERSION
from .spawn_phase_head import SpawnPhaseHead
from .small_enemy_data import digest
from .training_pipeline import atomic_json


def training_events(folders):
    states, maps, targets = [], [], []
    seeds = set()
    for folder in folders:
        paths = sorted(folder.glob('episode_*.npz'))
        if not paths:
            raise ValueError(f'Empty training folder {folder}')
        for path in paths:
            with np.load(path, allow_pickle=False) as episode:
                seed = int(episode['seed'])
                if seed in seeds:
                    raise ValueError(f'Duplicate training seed {seed}')
                seeds.add(seed)
                if (episode['split'].item() != 'train' or episode['stage'].item() != 'full_game'
                        or episode['behavior_version'].item() != BEHAVIOR_VERSION
                        or abs(float(episode['dt'])-1/30) > 1e-7):
                    raise ValueError(f'Unexpected training episode {path}')
                state = episode['states']
                ticks = np.flatnonzero(episode['respawn_flags'][:, 0])
                if not np.array_equal(ticks, np.flatnonzero((state[:-1, 13] < .5) & (state[1:, 13] >= .5))):
                    raise ValueError(f'Respawn annotation mismatch {path}')
                if len(ticks):
                    states.extend(state[ticks])
                    maps.extend([episode['map']] * len(ticks))
                    targets.extend(state[ticks + 1, 8:10])
    return tuple(torch.from_numpy(np.stack(a)) for a in (states, maps, targets)), sorted(seeds)


def fit(parent, folders, output, coverage=None):
    if output.exists():
        raise FileExistsError(output)
    (states, maps, targets), seeds = training_events(folders)
    observed_events = len(states)
    if coverage is not None:
        with np.load(coverage, allow_pickle=False) as extra:
            if (extra['split'].item() != 'train' or extra['protocol'].item() != 'respawn_camera_coverage_v1'
                    or extra['behavior_version'].item() != BEHAVIOR_VERSION):
                raise ValueError('Unexpected supplementary respawn training data')
            additional = torch.from_numpy(extra['states'])
            labels_xy = torch.from_numpy(extra['targets'])
            grid = torch.from_numpy(extra['grid'])
            if (not len(additional) or additional.shape != (len(additional), 20)
                    or labels_xy.shape != (len(additional), 2) or grid.shape != (16, 16)):
                raise ValueError('Incomplete supplementary respawn training data')
            states = torch.cat([states, additional])
            maps = torch.cat([maps, grid.expand(len(additional), -1, -1)])
            targets = torch.cat([targets, labels_xy])
    n = len(states)
    head = SpawnPhaseHead()
    labels = anchor_target_labels(targets).long()
    legal = hierarchical_spawn_logits(torch.zeros(n, 1, 256), torch.zeros(n, 1, 9),
                                      states, maps == 0, 1/30)['spawn_anchor_legal'][:, 0]
    if not legal[torch.arange(n), labels].all():
        raise ValueError('Reference training event has an illegal target')
    optimizer = torch.optim.LBFGS(head.parameters(), lr=.8, max_iter=250, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        logits = head(states, dt=1/30)[:, 0].masked_fill(~legal, -1e4)
        loss = F.cross_entropy(logits, labels) + 5e-6 * head.scores.square().mean()
        loss.backward()
        return loss
    optimizer.step(closure)
    with torch.no_grad():
        logits = head(states, dt=1/30)[:, 0].masked_fill(~legal, -1e4)
        correct = (logits.argmax(-1) == labels).sum().item()
    payload = torch.load(parent, mmap=True, weights_only=False, map_location='cpu')
    cfg = payload['config']
    if cfg.get('architecture') != 'vjepa2_ac' or not cfg.get('game_training_v5') or cfg.get('vjepa_spawn_phase_head'):
        raise ValueError('Expected an unmodified hierarchical V-JEPA checkpoint')
    meta = dict(parent=str(parent.resolve()), parent_step=payload['training_steps'],
                train_folders=[str(p) for p in folders], train_episodes=len(seeds),
                train_events=n, observed_episode_events=observed_events, coverage_events=n-observed_events,
                coverage_sha256=digest(coverage) if coverage else None,
                train_correct=correct, seen_labels=labels.bincount(minlength=9).tolist(),
                train_seeds=seeds, parameters=head.scores.numel(),
                inference_inputs='own next clock phase; own player and initial grid for native legal mask',
                fitted_by='train-only legal-category cross-entropy with L2; 250 maximum LBFGS iterations',
                parent_parameter_updates=0, fallback='original conditional legal non-anchor score ranking')
    payload['config'] = {**cfg, 'vjepa_spawn_phase_head': True}
    payload['model']['vjepa_backend.spawn_phase_head.scores'] = head.scores.detach()
    payload['vjepa_spawn_phase_training'] = meta
    _save(output, payload)
    atomic_json(output.parent / 'fit_spawn.json', meta)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', required=True, type=Path)
    p.add_argument('--train-folder', action='append', required=True, type=Path)
    p.add_argument('--coverage', type=Path, help='Additional one-tick TRAIN scenes from collect_respawn_coverage.')
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    torch.set_num_threads(2)
    fit(a.parent, a.train_folder, a.output, a.coverage)
