"""EMA targets, from-scratch JEPA losses and bounded episode-aligned RGB batches.

The variance/covariance auxiliary is a VICReg-style adaptation for training
without a pretrained visual encoder; it is not claimed to be the official
V-JEPA2 pretraining objective. Future source frames are targets, separated from
context frames and from controls that describe the predicted transitions.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class EMATargetEncoder(nn.Module):
    """Frozen/eval encoder initialized as an exact copy of the online encoder.

    Parameters use EMA; buffers are copied exactly (including integer counters
    and fixed positional tables). state_dict contains encoder.*, _ema_steps
    and _extra_state with the momentum configuration. Target parameters are
    training-only and should be counted separately from the deployed model.
    """

    def __init__(self, online_encoder, momentum=.996):
        super().__init__()
        if not math.isfinite(momentum) or not 0 <= momentum <= 1:
            raise ValueError("EMA momentum must be finite and in [0,1]")
        self.momentum = float(momentum)
        self.encoder = copy.deepcopy(online_encoder).requires_grad_(False).eval()
        self.register_buffer("_ema_steps", torch.zeros((), dtype=torch.int64))
        self.eval()

    @property
    def steps(self):
        return int(self._ema_steps.item())

    @property
    def target(self):
        return self.encoder

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.encoder.parameters())

    def train(self, mode=True):
        # A surrounding trainer's .train() must not enable target dropout/BN.
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, *values, **options):
        return self.encoder(*values, **options)

    @torch.no_grad()
    def update(self, online_encoder, momentum=None):
        rate = self.momentum if momentum is None else float(momentum)
        if not math.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError("EMA momentum must be finite and in [0,1]")
        source_parameters, target_parameters = dict(online_encoder.named_parameters()), dict(self.encoder.named_parameters())
        source_buffers, target_buffers = dict(online_encoder.named_buffers()), dict(self.encoder.named_buffers())
        for source, target in ((source_parameters, target_parameters), (source_buffers, target_buffers)):
            if source.keys() != target.keys() or any(source[name].shape != target[name].shape for name in source):
                raise ValueError("Online/EMA encoder structure does not match")
        for name, target in target_parameters.items():
            source = source_parameters[name].detach().to(device=target.device, dtype=target.dtype)
            target.mul_(rate).add_(source, alpha=1 - rate)
        for name, target in target_buffers.items():
            target.copy_(source_buffers[name].detach().to(device=target.device, dtype=target.dtype))
        self._ema_steps.add_(1)
        self.encoder.eval()

    def get_extra_state(self):
        return {"format_version": 1, "momentum": self.momentum}

    def set_extra_state(self, state):
        if state != {"format_version": 1, "momentum": self.momentum}:
            raise ValueError("EMA checkpoint momentum/configuration mismatch")


def latent_prediction_loss(prediction, target):
    """L1 on per-token LayerNorm latents [B,T,N,D], with an absorbing stop-grad target."""
    if prediction.ndim != 4 or prediction.shape != target.shape or not all(prediction.shape):
        raise ValueError("Expected matching nonempty latent tensors [B,T,N,D]")
    predicted = F.layer_norm(prediction.float(), (prediction.shape[-1],), eps=1e-6)
    expected = F.layer_norm(target.detach().float(), (target.shape[-1],), eps=1e-6)
    loss = F.l1_loss(predicted, expected)
    predicted_flat, expected_flat = predicted.reshape(-1, predicted.shape[-1]), expected.reshape(-1, expected.shape[-1])
    metrics = {"jepa_l1": loss.detach(),
               "latent_cosine": F.cosine_similarity(predicted_flat, expected_flat, dim=-1, eps=1e-6).mean().detach(),
               "prediction_std": predicted_flat.std(dim=0, unbiased=False).mean().detach(),
               "target_std": expected_flat.std(dim=0, unbiased=False).mean().detach()}
    return loss, metrics


def variance_covariance_loss(features, variance_weight=1., covariance_weight=.04, eps=1e-4):
    """Population variance floor and off-diagonal covariance over B*T*N tokens.

    Statistics and losses are FP32. Population normalization (unbiased=False)
    keeps the one-sample case finite; epsilon protects sqrt at zero variance.
    A positive penalty on collapsed features is an auxiliary training signal,
    not a claim that every possible collapsed stationary point is impossible.
    """
    if features.ndim < 2 or not all(features.shape):
        raise ValueError("Features must be nonempty with a final embedding dimension")
    if (not all(math.isfinite(value) for value in (variance_weight, covariance_weight, eps))
            or min(variance_weight, covariance_weight) < 0 or eps <= 0):
        raise ValueError("Regularizer weights must be nonnegative and epsilon positive")
    values = features.float().reshape(-1, features.shape[-1])
    centered = values - values.mean(dim=0, keepdim=True)
    standard_deviation = (centered.square().mean(dim=0) + eps).sqrt()
    variance = F.relu(1 - standard_deviation).mean()
    # Disable a surrounding autocast so covariance is accumulated in FP32 too.
    with torch.autocast(features.device.type, enabled=False):
        covariance_matrix = centered.transpose(0, 1) @ centered / values.shape[0]
        mask = ~torch.eye(values.shape[1], dtype=torch.bool, device=values.device)
        covariance = covariance_matrix.masked_select(mask).square().sum() / values.shape[1]
        loss = variance_weight * variance + covariance_weight * covariance
    return loss, {"variance": variance.detach(), "covariance": covariance.detach(),
                  "feature_std": standard_deviation.mean().detach(), "feature_samples": values.shape[0]}


class SequenceData:
    """One episode-aligned NPZ split, with memory or bounded streaming storage.

    Reading only *.npz in the supplied directory avoids recursively mixing
    train/val or states-only files. The default 6GiB decoded limit covers the
    current 1024-clip training split. Images stay uint8 on CPU; only sampled
    sequences are converted/moved to the requested device.

    mode='stream' preflights headers/labels without decoding the RGB corpus.
    Lazy episode access uses an LRU capped by cache_bytes (default512MiB).
    raw_bytes/cached_bytes describe the resident cache; total_raw_bytes is the
    logical decoded corpus size. refresh() appends newly completed clips, and
    ordinary training batches also discover them every refresh_interval.
    """

    def __init__(self, directory, device="cpu", limit=0, max_bytes=6 * 1024**3,
                 split=None, image_size=None, seed=0, mode="memory", cache_bytes=512*1024**2,
                 refresh_interval=30.):
        if mode not in {"memory", "stream"}:
            raise ValueError("SequenceData mode must be memory or stream")
        if mode == "stream":
            from .streaming_sequences import initialize_stream
            initialize_stream(self, directory, device, limit, max_bytes, split, image_size,
                              seed, cache_bytes, refresh_interval)
            return
        self.mode = "memory"
        self.directory, self.device = Path(directory), torch.device(device)
        if limit < 0 or max_bytes < 1 or image_size is not None and image_size < 1:
            raise ValueError("File limit must be nonnegative; memory/image sizes must be positive")
        paths = sorted(self.directory.glob("*.npz"))
        if limit:
            paths = paths[:limit]
        if not paths:
            raise ValueError(f"No RGB episodes in explicit split directory {directory}")
        self.split = split or (self.directory.name if self.directory.name in {"train", "val", "test"} else None)
        self.image_size, self.rng = image_size, np.random.default_rng(seed)
        self.episodes, self.episode_ranges = [], []
        self.count = self.frame_count = self.raw_bytes = 0
        observed_split = self.split
        required = {"frames", "states", "actions", "events", "hit_ids", "attack_flags", "map"}
        for path in paths:
            with np.load(path, allow_pickle=False) as saved:
                if not required.issubset(saved.files):
                    raise ValueError(f"{path.name}: missing fields {sorted(required - set(saved.files))}")
                declared = str(saved["split"]) if "split" in saved.files else None
                if declared is not None:
                    if observed_split is not None and declared != observed_split:
                        raise ValueError(f"{path.name}: refusing to mix dataset splits {declared}/{observed_split}")
                    observed_split = declared
                episode = {name: np.asarray(saved[name]) for name in required}
                episode["dt"] = float(saved["dt"]) if "dt" in saved.files else 1 / 30
                episode["path"] = str(path)
                count = len(episode["actions"])
                if (count < 1 or episode["states"].shape != (count + 1, 90)
                        or episode["actions"].shape != (count, 4)
                        or episode["events"].shape != (count, 4)
                        or episode["hit_ids"].shape != (count,)
                        or episode["attack_flags"].shape != (count, 8)
                        or episode["map"].shape != (16, 16)):
                    raise ValueError(f"{path.name}: incompatible state/action/event schema")
                frames = episode["frames"]
                if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[0] != count + 1 or frames.shape[-1] != 3:
                    raise ValueError(f"{path.name}: expected uint8 frames [T+1,H,W,3]")
                if not math.isfinite(episode["dt"]) or not 0 < episode["dt"] <= .25:
                    raise ValueError(f"{path.name}: unsupported episode timestep")
                for name in ("states", "actions", "events", "attack_flags"):
                    episode[name] = episode[name].astype(np.float32, copy=False)
                    if not np.isfinite(episode[name]).all():
                        raise ValueError(f"{path.name}: nonfinite {name}")
                episode["hit_ids"] = episode["hit_ids"].astype(np.int64, copy=False)
                episode["map"] = episode["map"].astype(np.int64, copy=False)
                if "respawn_flags" in saved.files:
                    episode["respawn_flags"] = np.asarray(saved["respawn_flags"], dtype=np.float32)
                else:
                    alive = episode["states"][:, 8:88].reshape(-1, 8, 10)[:, :, 5] >= .5
                    episode["respawn_flags"] = (~alive[:-1] & alive[1:]).astype(np.float32)
                if episode["respawn_flags"].shape != (count, 8):
                    raise ValueError(f"{path.name}: expected respawn_flags [T,8]")
                decoded_bytes = sum(value.nbytes for value in episode.values() if isinstance(value, np.ndarray))
                if self.raw_bytes + decoded_bytes > max_bytes:
                    raise ValueError("Decoded RGB split exceeds max_bytes; use a file limit or a larger explicit memory budget")
                self.raw_bytes += decoded_bytes
                self.episode_ranges.append({"path": str(path), "start": self.count, "stop": self.count + count})
                self.count += count
                self.frame_count += count + 1
                self.episodes.append(episode)
        self.split, self.files = observed_split, len(self.episodes)

    def __len__(self):
        return self.count

    def refresh(self):
        """Discover completed new clips in stream mode without loading RGB."""
        if self.mode == "stream":
            from .streaming_sequences import refresh_stream
            return refresh_stream(self, force=True)
        return 0

    def clear_cache(self):
        if self.mode == "stream":
            self._stream_cache.clear()
            self.raw_bytes = self.cached_bytes = 0

    def window_count(self, context=2, horizon=4):
        if min(context, horizon) < 1:
            raise ValueError("Context and prediction horizon must be positive")
        length = context + horizon
        return sum(max(0, len(episode["frames"]) - length + 1) for episode in self.episodes)

    def fixed_panel(self, count=64, context=2, horizon=4, seed=73):
        """Deterministic file-balanced windows selected without model predictions.

        Episode quotas are balanced up to each file's capacity. Separate bins
        of possible starts prevent duplicate windows while covering different
        portions of each selected episode. No training/validation resplit is
        performed here: the panel belongs entirely to this explicit split.
        """
        if min(count, context, horizon) < 1:
            raise ValueError("Panel count, context and horizon must be positive")
        length = context + horizon
        capacities = np.array([max(0, len(episode["frames"]) - length + 1)
                               for episode in self.episodes], dtype=np.int64)
        count = min(count, int(capacities.sum()))
        if not count:
            raise ValueError("No complete context/prediction sequence fits inside an episode")
        quotas = np.zeros(self.files, dtype=np.int64)
        remaining = count
        random = np.random.default_rng(seed)
        while remaining:
            available = np.flatnonzero(quotas < capacities)
            take = min(remaining, len(available))
            if take < len(available):
                # A fixed file stride aliases periodic collection policies
                # (e.g.512files/128windows can select only2of8 policies).
                available = random.choice(available, size=take, replace=False)
            quotas[available[:take]] += 1
            remaining -= take
        panel = []
        for episode_index, quota in enumerate(quotas):
            if not quota:
                continue
            for starts in np.array_split(np.arange(capacities[episode_index]), quota):
                panel.append((episode_index, int(random.choice(starts))))
        return panel

    def batch(self, batch_size, context=2, horizon=4, rng=None, indices=None):
        """Return context+horizon frames, plus horizon aligned future actions.

        For context2/horizon4, frames are s0..s5 and actions are a1..a4:
        each action maps the preceding context/prediction frame to its target.
        states/events/hits/attacks are supervision only; the model's predictor
        should receive context_frames and actions, never target_frames/states.
        """
        if self.mode == "stream" and indices is None:
            from .streaming_sequences import refresh_stream
            refresh_stream(self)
        if min(batch_size, context, horizon) < 1:
            raise ValueError("Batch, context and horizon must be positive")
        length = context + horizon
        random = self.rng if rng is None else rng
        if indices is None:
            eligible = [index for index, episode in enumerate(self.episodes) if len(episode["frames"]) >= length]
            if not eligible:
                raise ValueError("No complete context/prediction sequence fits inside an episode")
            chosen = np.asarray(random.choice(eligible, size=batch_size), dtype=np.int64)
            starts = [int(random.integers(len(self.episodes[int(index)]["frames"]) - length + 1))
                      for index in chosen]
        else:
            panel = np.asarray(indices)
            if (panel.ndim != 2 or panel.shape[1] != 2 or not len(panel)
                    or not np.issubdtype(panel.dtype, np.integer)):
                raise ValueError("Fixed panel must contain integer (episode_index,start) pairs")
            chosen, starts = panel[:, 0].astype(np.int64), panel[:, 1].astype(np.int64).tolist()
            batch_size = len(chosen)
            for episode_index, start in zip(chosen, starts):
                if (not 0 <= episode_index < self.files or start < 0
                        or start + length > len(self.episodes[int(episode_index)]["frames"])):
                    raise ValueError("Fixed panel window crosses an episode boundary")
        selected = []
        for episode_index, start in zip(chosen, starts):
            episode = self.episodes[int(episode_index)]
            first_action = start + context - 1
            selected.append({"frames": episode["frames"][start:start + length],
                             "states": episode["states"][start:start + length],
                             "actions": episode["actions"][first_action:first_action + horizon],
                             "events": episode["events"][first_action:first_action + horizon],
                             "hit_ids": episode["hit_ids"][first_action:first_action + horizon],
                             "attack_flags": episode["attack_flags"][first_action:first_action + horizon],
                             "respawn_flags": episode["respawn_flags"][first_action:first_action + horizon],
                             "all_actions": episode["actions"][start:start + length - 1],
                             "map": episode["map"], "dt": episode["dt"]})
        result = {}
        for name in selected[0]:
            array = np.asarray([item[name] for item in selected])
            tensor = torch.from_numpy(array).to(self.device)
            result[name] = tensor.long() if name in {"map", "hit_ids"} else tensor.float()
        images = result["frames"].permute(0, 1, 4, 2, 3).contiguous().div_(255)
        if self.image_size is not None and images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(images.flatten(0, 1), (self.image_size, self.image_size),
                                   mode="bilinear", align_corners=False, antialias=True).reshape(
                                       batch_size, length, 3, self.image_size, self.image_size)
        result.update(frames=images, context_frames=images[:, :context], target_frames=images[:, context:],
                      previous_states=result["states"][:, context - 1:-1],
                      target_states=result["states"][:, context:],
                      episode_indices=torch.as_tensor(chosen, device=self.device),
                      start_indices=torch.as_tensor(starts, device=self.device))
        result["hits"], result["attacks"] = result["hit_ids"], result["attack_flags"]
        return result


MemorySequenceDataset = SequenceData
