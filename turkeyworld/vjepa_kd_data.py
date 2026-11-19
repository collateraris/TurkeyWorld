"""Episode-disjoint, deterministic student training windows from completed Source collections."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch


WINDOW = 6
SAMPLES = 12
TRAIN_EPISODES = 40
HELDOUT_EPISODES = 8
MODES = ('navigation', 'enemy_motion', 'combat')


def _candidates(cycles):
    pools = defaultdict(list)
    for collector in ('source', 'combat_v1'):
        receipts = sorted(cycles.glob(f'cycle_*/{collector}/collection_receipt.json'))[-4:]
        for receipt in receipts:
            metadata = json.loads(receipt.read_text())
            if metadata['split'] != 'train':
                raise ValueError(f'Unexpected Source split in {receipt}')
            for row in metadata['episodes']:
                mode = 'combat' if collector == 'combat_v1' else row['stream']
                if row['split'] != 'train' or mode not in MODES:
                    raise ValueError(f'Invalid Source episode in {receipt}')
                filename = (row.get('relative_path') or row['path']).replace('\\', '/').split('/')[-1]
                folder = 'full_game' if mode == 'combat' else mode
                path = receipt.parent / folder / 'rgb' / 'train' / filename
                pools[mode].append((path, int(row['seed']), row['sha256']))
    return pools


def _starts(rng, events, respawns, combat):
    last = len(events) - WINDOW
    if last + 1 < SAMPLES:
        raise ValueError('Source episode is too short to draw distinct windows')
    picked = set()
    if combat:
        # Place up to two genuine events of each type in positions 2..4 of a six-tick window.
        for flags in (respawns, events[:, 2], events[:, 3], events[:, 1], events[:, 0]):
            ticks = np.flatnonzero(flags)
            for tick in rng.permutation(ticks)[:2]:
                picked.add(int(np.clip(tick - rng.integers(2, 5), 0, last)))
    for interval in np.array_split(np.arange(last + 1), SAMPLES - len(picked)):
        options = [int(v) for v in interval if v not in picked]
        if options:
            picked.add(int(rng.choice(options)))
    if len(picked) < SAMPLES:
        available = np.setdiff1d(np.arange(last + 1), list(picked))
        picked.update(rng.choice(available, SAMPLES-len(picked), replace=False).tolist())
    return sorted(picked)


def load_kd_windows(cycles: Path, seed: int):
    """Return train/heldout CPU tensors and an auditable split of whole Source episodes.

    At offset t, actions[t] advances states[t] to states[t+1]. Six consecutive
    states and six outgoing controls are returned. Windows never cross episodes.
    The student holdout comes from original training corpora and may have been
    seen by its teacher.
    """
    cycles = Path(cycles).resolve()
    pools = _candidates(cycles)
    rng = np.random.default_rng(seed)
    batches = {split: defaultdict(list) for split in ('train', 'heldout')}
    manifest = dict(seed=seed, cycles=str(cycles), window=WINDOW, samples_per_episode=SAMPLES,
                    source_split='train', split_unit='whole episode',
                    holdout_scope='student only; original teacher may have trained on these episodes',
                    train=[], heldout=[])
    seen_seeds, seen_hashes = set(), set()
    for mode in MODES:
        chosen = 0
        corpus = pools[mode]
        for i in rng.permutation(len(corpus)):
            path, episode_seed, digest = corpus[i]
            if episode_seed in seen_seeds or digest in seen_hashes:
                continue
            if sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f'Source collection digest differs for {path}')
            with np.load(path, allow_pickle=False) as data:
                if str(data['split']) != 'train' or int(data['seed']) != episode_seed:
                    raise ValueError(f'Episode identity differs for {path}')
                if bool(data['navigation_only']) != (mode == 'navigation'):
                    raise ValueError(f'Unexpected navigation flag in {path}')
                states = data['states']
                actions = data['actions']
                grid = data['map']
                events = data['events']
                respawns = data['respawn_flags'].reshape(-1)
                if (states.shape != (len(actions)+1, 20) or actions.shape[1:] != (4,)
                        or events.shape != actions.shape or len(respawns) != len(actions)
                        or grid.shape != (16, 16)):
                    raise ValueError(f'Misaligned Source transitions in {path}')
                starts = _starts(rng, events, respawns, mode == 'combat')
                ticks = np.asarray(starts)[:, None] + np.arange(WINDOW)
                split = 'train' if chosen < TRAIN_EPISODES else 'heldout'
                batch = batches[split]
                batch['states'].append(states[ticks])
                batch['actions'].append(actions[ticks])
                batch['grid'].append(np.repeat(grid[None], len(starts), axis=0))
                batch['nav'].append(np.full(len(starts), mode == 'navigation', dtype=bool))
                manifest[split].append(dict(mode=mode, path=str(path), seed=episode_seed,
                                           sha256=digest, starts=starts))
            seen_seeds.add(episode_seed)
            seen_hashes.add(digest)
            chosen += 1
            if chosen == TRAIN_EPISODES + HELDOUT_EPISODES:
                break
        if chosen != TRAIN_EPISODES + HELDOUT_EPISODES:
            raise ValueError(f'Insufficient distinct Source episodes for {mode}: {chosen}')
    dtypes = dict(states=torch.float32, actions=torch.float32, grid=torch.long, nav=torch.bool)
    for split, columns in batches.items():
        batches[split] = {key: torch.from_numpy(np.concatenate(values)).to(dtypes[key])
                          for key, values in columns.items()}
    return batches['train'], batches['heldout'], manifest
