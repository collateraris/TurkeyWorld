"""Immutable real replay for policy reward/continuation fitting, never imagination.

Rare reward/death windows are sampled explicitly with inverse-probability
weights. This changes coverage without silently changing the reward prior.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .jepa_training_utils import SequenceData


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def window_index(states, events, horizon):
    """Index complete, alive starts; dead padding is not an episode restart."""
    n = len(events)-horizon+1
    if n <= 0:
        return np.empty(0, np.int64), np.empty(0, bool), np.empty(0, bool)
    starts = np.arange(n, dtype=np.int64)
    valid = states[:n, 4] > 0
    starts = starts[valid]
    changed = ((events[:, 1:3] > .5).any(-1)
        | (states[:-1, 4] > states[1:, 4]))
    death = (states[:-1, 4] > 0) & (states[1:, 4] <= 0)
    def occurs(flags):
        prefix = np.concatenate([[0], np.cumsum(flags, dtype=np.int64)])
        return (prefix[starts+horizon]-prefix[starts]) > 0
    return starts, occurs(changed), occurs(death)


def mixture_probabilities(size, event, death):
    """Per-window proposal probability, including overlaps of rare strata."""
    event, death = np.asarray(event, bool), np.asarray(death, bool)
    mass = np.array([.5, .3 if event.any() else 0., .2 if death.any() else 0.])
    mass /= mass.sum()
    p = np.full(size, mass[0]/size)
    if event.any(): p[event] += mass[1]/event.sum()
    if death.any(): p[death] += mass[2]/death.sum()
    return mass, p


class PolicyReplay:
    def __init__(self, directories, *, device='cpu', horizon=16, seed=173,
                 identities=None):
        self.horizon, self.rng = horizon, np.random.default_rng(seed)
        self.streams, self.identities = [], []
        for stream, directory in enumerate(directories):
            directory = Path(directory).resolve()
            paths = sorted(directory.glob('*.npz'))
            if not paths or directory.name not in ('train', 'val'):
                raise ValueError('Policy replay requires an explicit nonempty TRAIN/VAL split')
            records, panels, event_rows, death_rows = [], [], [], []
            for i, path in enumerate(paths):
                records.append(dict(path=str(path), sha256=file_sha(path)))
                with np.load(path, allow_pickle=False) as saved:
                    if str(saved['split']) != directory.name:
                        raise ValueError('Policy replay refuses mixed splits')
                    starts, event, death = window_index(saved['states'], saved['events'], horizon)
                panels.extend((i, int(start)) for start in starts)
                event_rows.extend(event.tolist()); death_rows.extend(death.tolist())
            identity = dict(directory=str(directory), split=directory.name,
                horizon=horizon, episodes=records, windows=len(panels),
                reward_windows=sum(event_rows), terminal_windows=sum(death_rows))
            if identities is not None and identity != identities[stream]:
                raise ValueError('Immutable policy replay identity changed')
            self.identities.append(identity)
            data = SequenceData(directory, device=device, image_size=32,
                split=directory.name, mode='stream', cache_bytes=128*1024**2,
                episode_paths=paths, refresh_interval=0)
            panel = np.asarray(panels, dtype=np.int64)
            if not len(panel): raise ValueError('No complete alive replay windows')
            mass, probabilities = mixture_probabilities(len(panel), event_rows, death_rows)
            self.streams.append(dict(data=data, panel=panel, event=np.flatnonzero(event_rows),
                death=np.flatnonzero(death_rows), mass=mass, probabilities=probabilities))
        self.identity_sha256 = digest(self.identities)

    def batch(self, per_stream=6, *, balanced=True):
        batches, weights = [], []
        for stream in self.streams:
            n = len(stream['panel'])
            if balanced:
                stratum = self.rng.choice(3, size=per_stream, p=stream['mass'])
                rows = []
                for kind in stratum:
                    pool = stream['event'] if kind == 1 else stream['death'] if kind == 2 else None
                    rows.append(int(self.rng.choice(pool)) if pool is not None else int(self.rng.integers(n)))
                rows = np.asarray(rows)
                weight = (1/n)/stream['probabilities'][rows]
            else:
                rows = self.rng.integers(n, size=per_stream)
                weight = np.ones(per_stream)
            batches.append(stream['data'].batch(per_stream, context=1,
                horizon=self.horizon, indices=stream['panel'][rows]))
            weights.extend(weight.tolist())
        # SequenceData's scalar dt is a per-row tensor; every field is batched.
        batch = {key: torch.cat([item[key] for item in batches], dim=0) for key in batches[0]}
        batch['sampling_weights'] = torch.tensor(weights, device=batch['states'].device,
            dtype=torch.float32)[:, None].expand(-1, self.horizon)
        return batch

    def state_dict(self):
        return dict(identity_sha256=self.identity_sha256, rng=self.rng.bit_generator.state)

    def load_state_dict(self, state):
        if state['identity_sha256'] != self.identity_sha256:
            raise ValueError('Policy replay RNG belongs to different data')
        self.rng.bit_generator.state = state['rng']
