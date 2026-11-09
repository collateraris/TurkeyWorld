"""Bounded streaming of genuinely paired source RGB for learned x4 SR.

The collector records ``frames`` and ``hr_frames`` at the same engine state.
This reader never creates HR targets by resizing LR and never calls the engine.
Only NPZ headers and small provenance fields are read during indexing.
"""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch


SOURCE_LABELS = {'engine', 'reference_engine', 'source_engine', 'source', 'ground_truth'}


def _header(path, name):
    with zipfile.ZipFile(path) as archive, archive.open(name+'.npy') as stream:
        version = np.lib.format.read_magic(stream)
        read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, _, dtype = read(stream)
    return tuple(shape), np.dtype(dtype)


def _identity(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


class SRPairData:
    def __init__(self, directory, device='cpu', *, split=None, image_size=32,
                 scale=4, cache_bytes=256*1024**2, seed=373, refresh_interval=30.):
        self.directory = Path(directory).resolve()
        self.split = split or self.directory.name
        if self.split not in {'train', 'val'}:
            raise ValueError('SR requires an explicit train or val split')
        if image_size not in (32, 64, 128) or scale != 4 or cache_bytes < 1 or refresh_interval < 0:
            raise ValueError('SR requires native32/64/128 RGB, x4 scale and positive cache budget')
        self.device, self.image_size, self.scale = torch.device(device), image_size, scale
        self.cache_limit_bytes = int(cache_bytes)
        self.rng = np.random.default_rng(seed)
        self.refresh_interval = float(refresh_interval)
        self.episodes, self._known, self._cache = [], set(), OrderedDict()
        self.raw_bytes, self._last_refresh = 0, 0.
        self.refresh(force=True)
        if not self.episodes:
            raise ValueError(f'No paired source SR clips found: {self.directory}')

    def _metadata(self, path):
        with np.load(path, allow_pickle=False) as saved:
            if not {'frames', 'hr_frames', 'states', 'actions', 'split',
                    'hr_frame_source', 'paired_source_rgb'}.issubset(saved.files):
                raise ValueError(f'{path.name}: genuine paired source hr_frames/provenance are required')
            if str(saved['split']) != self.split:
                raise ValueError(f'{path.name}: refusing to mix SR train and val splits')
            if (str(saved['hr_frame_source']).lower() not in SOURCE_LABELS
                    or not bool(np.asarray(saved['paired_source_rgb']).item())):
                raise ValueError(f'{path.name}: HR must be paired source-engine RGB, never upsampled LR')
            for key in ('frame_source', 'rgb_source'):
                if key in saved.files and str(saved[key]).lower() not in SOURCE_LABELS:
                    raise ValueError(f'{path.name}: LR must be source RGB')
            for key in ('model_generated_rgb', 'synthetic_rgb', 'frames_are_model_predictions'):
                if key in saved.files and bool(np.asarray(saved[key]).item()):
                    raise ValueError(f'{path.name}: generated model RGB is not a source SR target')
            behavior = str(saved['behavior_version']) if 'behavior_version' in saved.files else None
        lr, lr_dtype = _header(path, 'frames')
        hr, hr_dtype = _header(path, 'hr_frames')
        states, _ = _header(path, 'states')
        actions, _ = _header(path, 'actions')
        size, output = self.image_size, self.image_size*self.scale
        if (len(lr) != 4 or lr[1:] != (size, size, 3) or lr_dtype != np.uint8
                or hr != (lr[0], output, output, 3) or hr_dtype != np.uint8
                or lr[0] < 2 or states != (lr[0], 90) or actions != (lr[0]-1, 4)):
            raise ValueError(f'{path.name}: native LR/HR/state frames must be aligned {size}/{output} pairs')
        decoded = int(np.prod(lr)+np.prod(hr))
        if decoded > self.cache_limit_bytes:
            raise ValueError('One decoded LR/HR episode exceeds SR cache_bytes')
        return {'path': path, 'frames': lr[0], 'lr_shape': lr, 'hr_shape': hr,
                'identity': _identity(path), 'decoded_bytes': decoded, 'behavior_version': behavior}

    def refresh(self, force=False):
        if not force and (self.split != 'train' or time.monotonic()-self._last_refresh < self.refresh_interval):
            return 0
        added = 0
        for path in sorted(self.directory.glob('*.npz')):
            if path not in self._known:
                self.episodes.append(self._metadata(path))
                self._known.add(path)
                added += 1
        self._cumulative = np.cumsum([episode['frames'] for episode in self.episodes], dtype=np.int64)
        self._last_refresh = time.monotonic()
        return added

    def __len__(self):
        return int(self._cumulative[-1]) if len(self._cumulative) else 0

    @property
    def stats(self):
        return {'directory': str(self.directory), 'split': self.split, 'episodes': len(self.episodes),
                'frames': len(self), 'transitions': len(self)-len(self.episodes),
                'image_size': self.image_size, 'hr_image_size': self.image_size*self.scale,
                'scale': self.scale, 'paired_source_rgb': True, 'hr_field': 'hr_frames',
                'cache_bytes': self.raw_bytes, 'cache_limit_bytes': self.cache_limit_bytes,
                'cached_episodes': len(self._cache), 'augmentations': []}

    def _load(self, index):
        episode = self.episodes[index]
        if _identity(episode['path']) != episode['identity']:
            raise ValueError('An immutable paired SR source episode changed')
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        while self._cache and self.raw_bytes+episode['decoded_bytes'] > self.cache_limit_bytes:
            _, removed = self._cache.popitem(last=False)
            self.raw_bytes -= sum(array.nbytes for array in removed)
        with np.load(episode['path'], allow_pickle=False) as saved:
            pair = (np.asarray(saved['frames']), np.asarray(saved['hr_frames']))
        if pair[0].shape != episode['lr_shape'] or pair[1].shape != episode['hr_shape']:
            raise ValueError('An immutable paired SR source changed shape')
        self._cache[index] = pair
        self.raw_bytes += sum(array.nbytes for array in pair)
        return pair

    def fixed_panel(self, count=100, seed=373):
        if self.split != 'val' or not 0 < count <= len(self):
            raise ValueError('A fixed SR panel requires enough unique held-out val frames')
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(self.episodes))
        choices = {int(index): list(rng.permutation(self.episodes[index]['frames'])) for index in order}
        panel = []
        while len(panel) < count:
            for index in order:
                if choices[int(index)]:
                    panel.append((int(index), int(choices[int(index)].pop())))
                    if len(panel) == count:
                        break
        return panel

    def panel_fingerprint(self, panel):
        records = [{'episode': self.episodes[int(index)]['path'].name, 'frame': int(frame),
                    'identity': self.episodes[int(index)]['identity'],
                    'behavior_version': self.episodes[int(index)]['behavior_version']}
                   for index, frame in panel]
        return hashlib.sha256(json.dumps({'records': records, 'split': self.split,
            'image_size': self.image_size, 'scale': self.scale}, sort_keys=True).encode()).hexdigest()

    def batch(self, batch_size, rng=None, indices=None):
        if batch_size < 1:
            raise ValueError('SR batch size must be positive')
        if indices is None:
            self.refresh()
            rng = self.rng if rng is None else rng
            sampled = rng.integers(len(self), size=batch_size)
            episodes = np.searchsorted(self._cumulative, sampled, side='right')
            starts = sampled-np.r_[0, self._cumulative[:-1]][episodes]
            indices = np.stack([episodes, starts], axis=1)
        else:
            indices = np.asarray(indices)
            if (indices.ndim != 2 or indices.shape[1] != 2 or not len(indices)
                    or not np.issubdtype(indices.dtype, np.integer)):
                raise ValueError('SR indices must contain integer(episode,frame) pairs')
        lr, hr = [], []
        for index, frame in indices:
            if not 0 <= index < len(self.episodes) or not 0 <= frame < self.episodes[index]['frames']:
                raise ValueError('SR frame index lies outside its source episode')
            low, high = self._load(int(index))
            lr.append(low[int(frame)])
            hr.append(high[int(frame)])
        def tensor(values):
            return torch.from_numpy(np.stack(values)).permute(0, 3, 1, 2).to(self.device, dtype=torch.float32)/255
        return {'lr': tensor(lr), 'hr': tensor(hr), 'indices': torch.as_tensor(indices, dtype=torch.long, device=self.device)}
