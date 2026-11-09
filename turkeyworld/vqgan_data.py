"""Ground-truth adjacent RGB pairs for from-scratch VQGAN training.

Only the canonical source ``frames`` field is sampled. A pair never crosses
an episode boundary. Future states/actions are not codec inputs; states/maps
and the event between the two frames are returned as optional supervision.
No geometric or pixel augmentation is applied, including horizontal flips.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import torch

from .jepa_training_utils import SequenceData


_SOURCE_LABELS = {"source", "reference", "reference_engine", "game_engine", "engine", "engine_oracle",
                  "ground_truth", "gt", "oracle", "source_engine", "clean_rgb"}


class VQGANData:
    """Bounded-memory source pairs, with optional policy/combat balancing.

    ``balance=True`` samples policies uniformly (or with ``policy_weights``)
    and reserves ``event_fraction`` for real recorded event transitions.
    Event labels affect sampling only. Images always come from source frames.
    Source provenance is checked before a file can yield a sampled pair.
    Frame/state tensors are float32 [0,1] RGB and the original90D state.

    ``initial_file_limit`` bootstraps a bounded indexed subset. With a positive
    ``growth_files``, each refresh raises that limit before discovering clips.
    Zero limits retain full-corpus discovery. Statistics distinguish the
    indexed/sampled subset from the number of source files available on disk.
    """
    def __init__(self, directory, device="cpu", *, split=None, image_size=128,
                 cache_bytes=512*1024**2, refresh_interval=30., balance=False,
                 event_fraction=.35, policy_weights=None, seed=273,
                 initial_file_limit=0, growth_files=0):
        if not 0 <= event_fraction <= 1 or refresh_interval < 0:
            raise ValueError("Event fraction must be in[0,1] and refresh interval nonnegative")
        if (not isinstance(initial_file_limit, (int, np.integer)) or initial_file_limit < 0
                or not isinstance(growth_files, (int, np.integer)) or growth_files < 0):
            raise ValueError("Initial file limit and growth files must be nonnegative integers")
        self.initial_file_limit, self.growth_files = int(initial_file_limit), int(growth_files)
        self.directory = Path(directory).resolve()
        self.data = SequenceData(self.directory, device=device, split=split, image_size=image_size,
                                 mode="stream", cache_bytes=cache_bytes, refresh_interval=refresh_interval,
                                 seed=seed, limit=self.initial_file_limit)
        self.split, self.device = self.data.split, self.data.device
        if self.split not in {"train", "val"}:
            raise ValueError("VQGAN requires an explicit train or val source split")
        self.balance = bool(balance and self.split == "train")
        self.event_fraction, self.policy_weights = float(event_fraction), dict(policy_weights or {})
        if any(not np.isfinite(value) or value < 0 for value in self.policy_weights.values()):
            raise ValueError("Policy sampling weights must be finite and nonnegative")
        self.rng = np.random.default_rng(seed)
        self.refresh_interval = float(refresh_interval)
        self._last_refresh = 0.
        self._catalog, self._checked_sources = {}, set()
        self._pair_counts = self._pair_cumulative = np.empty(0, np.int64)
        self._positive_counts = self._positive_cumulative = np.empty(0, np.int64)
        self._policy_groups = {}
        self._rebuild_counts()
        if self.balance:
            self._complete_catalog()
        self._available_source_files = sum(1 for _ in self.directory.glob("*.npz"))
        self._last_refresh = time.monotonic()

    @property
    def episodes(self):
        return self.data.episodes

    def __len__(self):
        return int(self._pair_counts.sum())

    @property
    def files(self):
        return self.data.files

    @property
    def count(self):
        return len(self)

    @property
    def frame_count(self):
        return self.data.frame_count

    def _rebuild_counts(self):
        self._pair_counts = np.asarray([len(episode["frames"])-1 for episode in self.data.episodes], np.int64)
        self._pair_cumulative = self._pair_counts.cumsum()

    def _read_source(self, index, events=False):
        path = self.data.episodes[index]["path"]
        with np.load(path, allow_pickle=False) as saved:
            for name in ("frame_source", "rgb_source", "frames_source"):
                if name in saved.files:
                    label = str(saved[name]).strip().lower()
                    if label not in _SOURCE_LABELS:
                        raise ValueError(f"{Path(path).name}: codec requires clean source RGB, declared {name}={label}")
            for name in ("model_generated_rgb", "synthetic_rgb", "frames_are_model_predictions"):
                if name in saved.files and bool(np.asarray(saved[name]).item()):
                    raise ValueError(f"{Path(path).name}: generated/model RGB is not a codec source target")
            if events:
                policy = str(saved["policy"]) if "policy" in saved.files else "unspecified"
                labels = np.asarray(saved["events"], dtype=np.float32)
                positive = np.flatnonzero((labels > .5).any(-1)).astype(np.int32)
                record = {"policy": policy, "positive_starts": positive}
            else:
                record = None
        self._checked_sources.add(index)
        return record

    def _complete_catalog(self):
        if len(self._catalog) == self.data.files and len(self._positive_counts) == self.data.files:
            return
        # Source files are immutable and stream discovery only appends indices.
        # Read provenance/policy/events only for the newly indexed suffix.
        for index in range(len(self._catalog), self.data.files):
            self._catalog[index] = self._read_source(index, events=True)
        self._positive_counts = np.asarray([len(self._catalog[index]["positive_starts"]) for index in range(self.data.files)], np.int64)
        self._positive_cumulative = self._positive_counts.cumsum()
        groups = defaultdict(list)
        for index, record in self._catalog.items():
            groups[record["policy"]].append(index)
        self._policy_groups = {name: np.asarray(indices, np.int64) for name, indices in groups.items()}

    def refresh(self, force=False):
        """Append newly completed clips; existing episode/panel indices stay fixed."""
        if not force and time.monotonic()-self._last_refresh < self.refresh_interval:
            return 0
        # SequenceData.batch may perform its own timed refresh. Force this
        # explicit discovery after increasing the bootstrap limit so its timer
        # cannot defer the newly admitted files for another full interval.
        from .streaming_sequences import refresh_stream
        if self.data._stream_limit and self.growth_files:
            self.data._stream_limit += self.growth_files
        added = refresh_stream(self.data, force=True)
        if added or not len(self._pair_counts):
            self._rebuild_counts()
            if self.balance:
                self._complete_catalog()
        self._available_source_files = sum(1 for _ in self.directory.glob("*.npz"))
        self._last_refresh = time.monotonic()
        return added

    @property
    def stats(self):
        return {"source_files": self.data.files, "episodes": self.data.files,
                "indexed_source_files": self.data.files,
                "available_source_files": self._available_source_files,
                "unindexed_source_files": max(0, self._available_source_files-self.data.files),
                "source_index_complete": self.data.files >= self._available_source_files,
                "indexing_scope": "all_available_source_files" if self.data.files >= self._available_source_files else "indexed_subset",
                "initial_file_limit": self.initial_file_limit, "current_file_limit": self.data._stream_limit,
                "growth_files": self.growth_files,
                "rgb_frames": self.data.frame_count, "adjacent_pairs": len(self), "split": self.split,
                "directory": str(self.directory), "source_rgb_field": "frames", "augmentations": [],
                "cache_bytes": self.data.raw_bytes, "cache_limit_bytes": self.data.cache_limit_bytes,
                "logical_decoded_bytes": self.data.total_raw_bytes,
                "source_checked_files": len(self._checked_sources), "balanced_sampling": self.balance,
                "event_positive_pairs": int(self._positive_counts.sum()) if self.balance else None,
                "policy_pair_counts": {name: int(self._pair_counts[indices].sum()) for name, indices in self._policy_groups.items()}}

    def fixed_panel(self, count=128, seed=273):
        """Unique reproducible validation pairs, suitable for persisted indices."""
        if self.split != "val":
            raise ValueError("Fixed codec evaluation panels require the explicit val split")
        if count < 1 or count > len(self):
            raise ValueError("Not enough unique adjacent pairs for the requested validation panel")
        panel = self.data.fixed_panel(count, context=1, horizon=1, seed=seed)
        for index, _ in panel:
            if index not in self._checked_sources:
                self._read_source(index)
        return panel

    def _uniform_indices(self, batch_size, rng):
        global_index = rng.integers(len(self), size=batch_size)
        episodes = np.searchsorted(self._pair_cumulative, global_index, side="right")
        starts = global_index-np.r_[0, self._pair_cumulative[:-1]][episodes]
        return np.stack([episodes, starts], axis=1)

    def _balanced_indices(self, batch_size, rng):
        self._complete_catalog()
        result = np.empty((batch_size, 2), np.int64)
        names = list(self._policy_groups)
        weights = np.asarray([self.policy_weights.get(name, 1.) for name in names], np.float64)
        if weights.sum() <= 0:
            raise ValueError("At least one source policy must have a positive sampling weight")
        weights /= weights.sum()
        total_positive = int(self._positive_counts.sum())
        for index in range(batch_size):
            if total_positive and rng.random() < self.event_fraction:
                selected = int(rng.integers(total_positive))
                episode = int(np.searchsorted(self._positive_cumulative, selected, side="right"))
                before = int(self._positive_cumulative[episode-1]) if episode else 0
                start = int(self._catalog[episode]["positive_starts"][selected-before])
            else:
                name = names[int(rng.choice(len(names), p=weights))]
                candidates = self._policy_groups[name]
                counts = self._pair_counts[candidates]
                episode = int(rng.choice(candidates, p=counts/counts.sum()))
                start = int(rng.integers(self._pair_counts[episode]))
            result[index] = episode, start
        return result

    def batch(self, batch_size, rng=None, indices=None):
        """Return GT pairs[B,2,3,H,W], original states/maps and intervening events."""
        if batch_size < 1:
            raise ValueError("Codec batch size must be positive")
        if indices is None:
            self.refresh()
            rng = self.rng if rng is None else rng
            indices = self._balanced_indices(batch_size, rng) if self.balance else self._uniform_indices(batch_size, rng)
        else:
            indices = np.asarray(indices)
            if (indices.ndim != 2 or indices.shape[1] != 2 or not len(indices)
                    or not np.issubdtype(indices.dtype, np.integer)):
                raise ValueError("Codec indices must contain integer(episode,start) pairs")
        for index, start in indices:
            if not 0 <= index < self.data.files or not 0 <= start < self._pair_counts[index]:
                raise ValueError("Codec pair crosses an episode boundary")
            if int(index) not in self._checked_sources:
                self._read_source(int(index))
        batch = self.data.batch(len(indices), context=1, horizon=1, indices=indices)
        return {"frames": batch["frames"], "states": batch["states"], "map": batch["map"],
                "events": batch["events"], "indices": torch.stack([batch["episode_indices"], batch["start_indices"]], -1)}


CodecPairData = VQGANData
VQGANPairData = VQGANData
