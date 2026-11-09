"""Bounded LRU storage used by SequenceData(mode='stream')."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import math
from pathlib import Path
import time
import zipfile

import numpy as np
import torch


REQUIRED = {"frames", "states", "actions", "events", "hit_ids", "attack_flags", "map"}


def resolve_episode_paths(directory, episode_paths):
    """Validate an ordered fixed selection; relative paths start at directory."""
    if episode_paths is None:
        return None
    if isinstance(episode_paths, (str, Path)):
        raise ValueError('episode_paths must be a nonempty sequence of NPZ paths')
    directory = Path(directory).resolve()
    paths, seen = [], set()
    for value in episode_paths:
        path = Path(value)
        path = (path if path.is_absolute() else directory/path).resolve()
        if path.parent != directory:
            raise ValueError('Selected episode must belong directly to the explicit split directory')
        if path.suffix.lower() != '.npz':
            raise ValueError('Selected episode must be an .npz file')
        if path in seen:
            raise ValueError('Selected episode paths must not contain duplicates')
        if not path.is_file():
            raise ValueError(f'Selected episode does not exist: {path}')
        seen.add(path)
        paths.append(path)
    if not paths:
        raise ValueError('episode_paths must be a nonempty sequence of NPZ paths')
    return tuple(paths)


def _frame_header(path):
    with zipfile.ZipFile(path) as archive, archive.open("frames.npy") as stream:
        version = np.lib.format.read_magic(stream)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, _, dtype = reader(stream)
    return tuple(shape), np.dtype(dtype)


def _small_fields(saved, path, expected_split):
    if not REQUIRED.issubset(saved.files):
        raise ValueError(f"{path.name}: missing fields {sorted(REQUIRED-set(saved.files))}")
    declared = str(saved["split"]) if "split" in saved.files else None
    if declared is not None and expected_split is not None and declared != expected_split:
        raise ValueError(f"{path.name}: refusing to mix dataset splits {declared}/{expected_split}")
    episode = {key: np.asarray(saved[key]) for key in REQUIRED if key != "frames"}
    count = len(episode["actions"])
    if (count < 1 or episode["states"].shape != (count+1, 90) or episode["actions"].shape != (count, 4)
            or episode["events"].shape != (count, 4) or episode["hit_ids"].shape != (count,)
            or episode["attack_flags"].shape != (count, 8) or episode["map"].shape != (16, 16)):
        raise ValueError(f"{path.name}: incompatible state/action/event schema")
    dt = float(saved["dt"]) if "dt" in saved.files else 1/30
    if not math.isfinite(dt) or not 0 < dt <= .25:
        raise ValueError(f"{path.name}: unsupported episode timestep")
    for key in ("states", "actions", "events", "attack_flags"):
        episode[key] = episode[key].astype(np.float32, copy=False)
        if not np.isfinite(episode[key]).all():
            raise ValueError(f"{path.name}: nonfinite {key}")
    episode["hit_ids"] = episode["hit_ids"].astype(np.int64, copy=False)
    episode["map"] = episode["map"].astype(np.int64, copy=False)
    if "respawn_flags" in saved.files:
        episode["respawn_flags"] = np.asarray(saved["respawn_flags"], dtype=np.float32)
    else:
        alive = episode["states"][:, 8:88].reshape(-1, 8, 10)[:, :, 5] >= .5
        episode["respawn_flags"] = (~alive[:-1]&alive[1:]).astype(np.float32)
    if episode["respawn_flags"].shape != (count, 8):
        raise ValueError(f"{path.name}: expected respawn_flags [T,8]")
    episode.update(path=str(path), dt=dt)
    return episode, declared, count


def _metadata(path, expected_split):
    with np.load(path, allow_pickle=False) as saved:
        fields, split, count = _small_fields(saved, path, expected_split)
    shape, dtype = _frame_header(path)
    if dtype != np.uint8 or len(shape) != 4 or shape[0] != count+1 or shape[-1] != 3:
        raise ValueError(f"{path.name}: expected uint8 frames [T+1,H,W,3]")
    size = int(np.prod(shape))*dtype.itemsize + sum(value.nbytes for value in fields.values() if isinstance(value, np.ndarray))
    return {"path": path, "split": split, "count": count, "shape": shape, "dtype": dtype,
            "dt": fields["dt"], "decoded_bytes": size}


def _load(owner, metadata):
    key = str(metadata["path"])
    if key in owner._stream_cache:
        owner._stream_cache.move_to_end(key)
        return owner._stream_cache[key]
    # Evict BEFORE decoding the new RGB payload, keeping cache plus the next
    # decoded episode within its explicit budget (apart from caller-held slices).
    while owner._stream_cache and owner.raw_bytes+metadata["decoded_bytes"] > owner.cache_limit_bytes:
        _, removed = owner._stream_cache.popitem(last=False)
        owner.raw_bytes -= sum(value.nbytes for value in removed.values() if isinstance(value, np.ndarray))
    if metadata["decoded_bytes"] > owner.cache_limit_bytes:
        raise ValueError("One decoded episode exceeds stream cache_bytes; use shorter clips or a larger explicit cache")
    with np.load(metadata["path"], allow_pickle=False) as saved:
        episode, _, _ = _small_fields(saved, metadata["path"], owner.split)
        episode["frames"] = np.asarray(saved["frames"], dtype=np.uint8)
    if tuple(episode["frames"].shape) != metadata["shape"]:
        raise ValueError("An immutable streamed episode changed shape")
    size = sum(value.nbytes for value in episode.values() if isinstance(value, np.ndarray))
    owner._stream_cache[key] = episode
    owner.raw_bytes += size
    owner.cached_bytes = owner.raw_bytes
    return episode


class LazyFrames:
    def __init__(self, owner, metadata):
        self.owner, self.metadata = owner, metadata
        self.shape, self.dtype = metadata["shape"], metadata["dtype"]
        self.ndim = 4
        self.size = int(np.prod(self.shape))
        self.nbytes = self.size*self.dtype.itemsize

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        return _load(self.owner, self.metadata)["frames"][index]

    def __array__(self, dtype=None, copy=None):
        value = np.asarray(_load(self.owner, self.metadata)["frames"], dtype=dtype)
        return value.copy() if copy else value


class LazyEpisode(Mapping):
    def __init__(self, owner, metadata):
        self.owner, self.metadata = owner, metadata
        self.frames = LazyFrames(owner, metadata)

    def __getitem__(self, key):
        if key == "frames":
            return self.frames
        if key == "path":
            return str(self.metadata["path"])
        if key == "dt":
            return self.metadata["dt"]
        return _load(self.owner, self.metadata)[key]

    def __len__(self):
        return len(REQUIRED)+3  # required arrays, respawn_flags, dt and path

    def __iter__(self):
        return iter(REQUIRED|{"respawn_flags", "dt", "path"})


def initialize_stream(owner, directory, device="cpu", limit=0, max_bytes=6*1024**3,
                      split=None, image_size=None, seed=0, cache_bytes=512*1024**2,
                      refresh_interval=30., episode_paths=None, object_masks_directory=None,
                      counterfactual_backgrounds=False):
    if limit < 0 or cache_bytes < 1 or max_bytes < 1 or refresh_interval < 0 or image_size is not None and image_size < 1:
        raise ValueError("Invalid streamed dataset limit/cache/image size")
    owner.directory, owner.device = Path(directory), torch.device(device)
    owner._fixed_episode_paths = resolve_episode_paths(directory, episode_paths)
    owner.split = split or (owner.directory.name if owner.directory.name in {"train", "val", "test"} else None)
    owner.mode, owner.image_size, owner.rng = "stream", image_size, np.random.default_rng(seed)
    owner.episodes, owner.episode_ranges = [], []
    owner.count = owner.frame_count = owner.raw_bytes = owner.cached_bytes = owner.total_raw_bytes = 0
    owner.cache_limit_bytes, owner._stream_limit = int(cache_bytes), limit
    owner.refresh_interval, owner._last_refresh = float(refresh_interval), 0.
    owner._stream_cache, owner._stream_known = OrderedDict(), set()
    refresh_stream(owner, force=True)
    if not owner.episodes:
        raise ValueError(f"No RGB episodes in explicit split directory {directory}")
    if hasattr(owner, "_initialize_object_masks"):
        owner._initialize_object_masks(object_masks_directory, counterfactual_backgrounds)
    elif object_masks_directory is not None or counterfactual_backgrounds:
        raise ValueError("Object mask loading requires a SequenceData owner")


def refresh_stream(owner, force=False):
    """Append allowed completed NPZ files; fixed selections never discover others."""
    if not force and time.monotonic()-owner._last_refresh < owner.refresh_interval:
        return 0
    paths = (owner._fixed_episode_paths if owner._fixed_episode_paths is not None
             else sorted(owner.directory.glob("*.npz")))
    if owner._stream_limit:
        paths = paths[:owner._stream_limit]
    added = 0
    for path in paths:
        key = str(path)
        if key in owner._stream_known:
            continue
        metadata = _metadata(path, owner.split)
        if metadata["decoded_bytes"] > owner.cache_limit_bytes:
            raise ValueError("One decoded episode exceeds stream cache_bytes; increase the explicit per-episode budget")
        if metadata["split"] is not None:
            owner.split = metadata["split"]
        owner.episodes.append(LazyEpisode(owner, metadata))
        owner.episode_ranges.append({"path": str(path), "start": owner.count, "stop": owner.count+metadata["count"]})
        owner.count += metadata["count"]
        owner.frame_count += metadata["count"]+1
        owner.total_raw_bytes += metadata["decoded_bytes"]
        owner._stream_known.add(key)
        added += 1
    owner.files = len(owner.episodes)
    owner._last_refresh = time.monotonic()
    return added
