"""Build matched Source32 and Source128 stills by replaying exact Source episodes."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .engine import BEHAVIOR_VERSION, Game
from .collect_combat_training import COMBAT_CASES, NAV_CASES, collect_job
from .staged_data import _dynamic, _observe

GROUPS = ("nav", "quiet", "combat", "flash", "hurt", "death", "melee")
ORIGINAL = Path("training_runs/combat_source_20261111/data")
QUIET = Path("training_runs/rssm_enemy_motion_20261113/data/full_game/rgb")
FRESH = Path(
    "training_runs/vjepa_sr_cycle_20261117/cycles/cycle_000008/combat_v1/full_game/rgb/train"
)


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def categories(states, visible, stream):
    """Select real visual poses, with visibility required for enemy-specific groups."""
    n = len(states)
    if stream == "nav":
        return {"nav": np.arange(n)}
    if stream == "quiet":
        return {"quiet": np.flatnonzero(visible >= 3)}
    s = states
    alive = s[:, 13] >= 0.5
    visible_enemy = visible >= 4
    hurt = alive & (s[:, 15] > 0)
    melee = alive & ~hurt & (s[:, 14] * Game.ATTACK_INTERVAL > 1.0)
    flash = s[:, 7] * Game.FLASH_DURATION > 0.045
    return dict(
        combat=np.flatnonzero(alive & ~hurt & ~melee & ~flash),
        flash=np.flatnonzero(flash),
        hurt=np.flatnonzero(hurt & visible_enemy),
        death=np.flatnonzero(~alive & visible_enemy),
        melee=np.flatnonzero(melee & visible_enemy),
    )


def episode_pool(folder, split, limit):
    """Partition the held-out Source episodes, within each behavior, before frame selection."""
    paths = sorted(folder.glob("episode*.npz"))
    if split == "train":
        # A seeded sample from the original, long archives plus every fresh episode.
        rng = np.random.default_rng(31837)
        return [paths[i] for i in sorted(rng.permutation(len(paths))[:limit])]
    by_case = defaultdict(list)
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            case = next(
                (z[k].item() for k in ("case", "movement_case") if k in z), "quiet"
            )
        by_case[case].append(path)
    offset = 0 if split == "val" else 1
    return sorted(path for items in by_case.values() for path in items[offset::2])


def choose(split, count):
    original_split = "train" if split == "train" else "val"
    folders = [
        ("combat", ORIGINAL / "full_game/rgb" / original_split, 96),
        ("nav", ORIGINAL / "navigation/rgb" / original_split, 48),
        ("quiet", QUIET / original_split, 48),
    ]
    if split == "train":
        folders.append(("combat", FRESH, 64))
    rng = np.random.default_rng({"train": 8723, "val": 8724, "test": 8725}[split])
    pools = defaultdict(list)
    for stream, folder, limit in folders:
        candidates = episode_pool(folder, split, limit)
        if not candidates:
            raise ValueError(f"No Source data at {folder}")
        for p in candidates:
            with np.load(p, allow_pickle=False) as z:
                if (
                    z["split"].item() != original_split
                    or z["behavior_version"].item() != BEHAVIOR_VERSION
                ):
                    raise ValueError(f"Unexpected episode metadata for {p}")
                states = z["states"]
                visible = z["object_masks"].reshape(len(states), -1).sum(1)
            for group, frames in categories(states, visible, stream).items():
                pools[group].extend((p, int(i)) for i in frames)
    # Rare states first; assign each episode/tick to at most one primary cohort.
    # Preserve full split size by allocating leftover capacity to common events.
    assigned = defaultdict(list)
    used = set()
    for group in sorted(GROUPS, key=lambda g: len(pools[g])):
        rng.shuffle(pools[group])
        quota = count // len(GROUPS)
        for key in pools[group]:
            if key not in used:
                assigned[group].append(key)
                used.add(key)
                if len(assigned[group]) == quota:
                    break
        if not assigned[group]:
            raise ValueError(f"No cohort {group} in {split}")
    for group in ("flash", "hurt", "death", "combat", "quiet", "nav"):
        for key in pools[group]:
            if len(used) == count:
                break
            if key not in used:
                assigned[group].append(key)
                used.add(key)
    if len(used) != count:
        raise ValueError("Too few distinct annotated frames")
    by_path = defaultdict(dict)
    for group, keys in assigned.items():
        for path, tick in keys:
            by_path[path][tick] = group
    return [(str(p), ticks) for p, ticks in sorted(by_path.items())]


def weapon_labels(game, size):
    """Renderer-aligned gun support, plus flash-frame changed pixels around its muzzle."""
    scale = int(size * 0.67)
    yy, xx = np.meshgrid(np.arange(scale), np.arange(scale), indexing="ij")
    sprites = game._guns
    sy = yy * sprites[0].shape[0] // scale
    sx = xx * sprites[0].shape[1] // scale
    x0, y0 = size // 2 - scale // 2, size - scale
    result = []
    for phase, sprite in enumerate(sprites):
        sample = sprite[sy, sx]
        weapon = np.zeros((size, size), bool)
        flash = np.zeros_like(weapon)
        weapon[y0 : y0 + scale, x0 : x0 + scale] = sample[..., 3] > 0
        if phase == 1:
            changed = (sample != sprites[0][sy, sx]).any(-1)
            flash[y0 : y0 + scale, x0 : x0 + scale] = (
                changed & (sample[..., 3] > 0) & (yy < 0.56 * scale)
            )
        result.append((weapon, flash))
    return result


def replay(job):
    name, selected = job
    path = Path(name)
    with np.load(path, allow_pickle=False) as z:
        states, actions, frames = (z[k] for k in ("states", "actions", "frames"))
        seed = int(z["seed"])
        initial = json.loads(z["initial_state"].item())
        recorded_events = z["events"]
        grid = z["map"].astype(np.int8)
    game = Game(
        seed=seed, endless=True, enemies_enabled=initial.get("enemies_enabled", True)
    )
    game.set_state(initial)
    if not np.array_equal(grid, game.map) or len(states) != len(actions) + 1:
        raise ValueError(f"Invalid Source episode {name}")
    masks = {size: weapon_labels(game, size) for size in (32, 128)}
    answers = []
    native_mismatches = 0
    cumulative_rgb_mae = 0.0
    for tick in range(max(selected) + 1):
        if tick:
            events = game.step(actions[tick - 1], dt=1 / 30)
            if not np.array_equal(Game.event_vector(events), recorded_events[tick - 1]):
                raise AssertionError(f"Event replay mismatch {path.name}/{tick}")
        if not np.array_equal(game.state_vector(), states[tick]):
            raise AssertionError(f"State replay mismatch {path.name}/{tick}")
        if tick not in selected:
            continue
        rgb32, enemy32, _ = _observe(game, 32)
        if not np.array_equal(rgb32, frames[tick]):
            native_mismatches += 1
            cumulative_rgb_mae += np.abs(rgb32.astype(np.int16) - frames[tick]).mean()
        rgb128, enemy128, _ = _observe(game, 128)
        timer = game.player["shot_flash"]
        phase = 1 if timer > 0.045 else 2 if timer > 0 else 0
        row = dict(
            states=states[tick],
            maps=grid,
            rgb32=rgb32,
            rgb128=rgb128,
            enemy32=enemy32,
            enemy128=enemy128,
            seed=seed,
            frame=tick,
            group=selected[tick],
            state_json=json.dumps(game.get_state(), separators=(",", ":")),
            navigation=not game.enemies_enabled,
        )
        for size in (32, 128):
            row["weapon" + str(size)], row["flash" + str(size)] = masks[size][phase]
        answers.append(row)
    return (
        dict(
            path=name,
            seed=seed,
            sha256=digest(path),
            ticks=sorted(selected),
            archive_native_rgb32_mismatches=native_mismatches,
            archive_rgb32_mae_0_255=cumulative_rgb_mae / len(selected),
        ),
        answers,
    )


def generate(output, counts, workers):
    output.mkdir(parents=True, exist_ok=True)
    provenance = {}
    seen = set()
    for split, count in counts.items():
        destination = output / f"{split}.npz"
        if destination.exists():
            raise FileExistsError(destination)
        jobs = choose(split, count)
        print(split, "selected", len(jobs), "episodes", flush=True)
        records = []
        columns = defaultdict(list)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for record, rows in pool.map(replay, jobs, chunksize=1):
                records.append(record)
                for row in rows:
                    for key, value in row.items():
                        columns[key].append(value)
        seeds = {r["seed"] for r in records}
        if len(seeds) != len(records) or seen & seeds:
            raise ValueError("Episode seed collision across dataset splits")
        seen.update(seeds)
        temp = destination.with_suffix(".tmp.npz")
        np.savez(temp, **dict(columns))
        os.replace(temp, destination)
        provenance[split] = dict(
            frames=count,
            groups=dict(Counter(columns["group"])),
            episodes=records,
            sha256=digest(destination),
        )
        print(split, "complete", provenance[split]["groups"], flush=True)
    manifest = dict(
        protocol="combat_visual_pairs_v1",
        splits=provenance,
        state_and_events_replay_bitexact=True,
        native_rgb32_rerendered_with_current_source=True,
        source128_from_exact_replay=True,
        disjoint_episode_seeds=True,
        flash_mask="changed gun pixels above 56% of weapon footprint on a bright muzzle-flash frame",
        enemy_mask="difference of full render and render without NPC, with gun occlusion",
        weapon_mask="visible support of final gun atlas layer",
        heldout_partition="VAL and TEST are disjoint halves of original validation episodes within each behavior",
    )
    temp = output / "manifest.tmp.json"
    temp.write_text(json.dumps(manifest, indent=2))
    temp.replace(output / "manifest.json")


# The second panel was frozen after the first panel had been evaluated.
# Keep the two seed ranges and selection seeds immutable for reproduction.
BLIND_PANELS = {
    "blind": (2026111800000, 202611181, 48, ("train", "val", "test")),
    "final_blind": (2026111810000, 202611182, 64, ("train", "val", "test", "blind")),
}


def collect_blind_episode(job):
    directory, split, stream, index, seed = job
    path = directory / f"episode_{seed}.npz"
    if stream == "quiet":
        case = ("straight", "turn", "strafe", "short_return")[index % 4]
        payload = _dynamic(
            dict(
                seed=seed,
                stage="moving1",
                index=index,
                case=case,
                frames=32,
                rgb_time_span=60.0,
                size=32,
            )
        )
        payload.update(
            split=np.asarray(split),
            case=np.asarray(case),
            behavior_version=np.asarray(BEHAVIOR_VERSION),
        )
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **payload)
        tmp.replace(path)
    else:
        cases = NAV_CASES if stream == "nav" else COMBAT_CASES
        collect_job(
            dict(
                seed=seed,
                index=index,
                stage="navigation" if stream == "nav" else "full_game",
                case=cases[index % len(cases)],
                frames=40 if stream == "nav" else 200,
                split=split,
                protocol="fresh_blind_combat_visual_1",
                path=path,
                relative_path=path.name,
            )
        )
    return str(path), stream


def frame_key(image):
    return hashlib.blake2b(image, digest_size=16).digest()


def sample_keys(data):
    return {
        "seeds": set(data["seed"]),
        "states": {s.tobytes() for s in data["states"]},
        "rgb32": {frame_key(x) for x in data["rgb32"]},
        "rgb128": {frame_key(x) for x in data["rgb128"]},
    }


def generate_blind(output, split, workers):
    base, selection_seed, quota, reference_splits = BLIND_PANELS[split]
    target = output / f"{split}.npz"
    if target.exists():
        raise FileExistsError(target)
    prior = {}
    for name in reference_splits:
        with np.load(output / f"{name}.npz", allow_pickle=False) as z:
            prior[name] = sample_keys(z)
    prior32 = set.union(*(keys["rgb32"] for keys in prior.values()))
    source = output / f"{split}_source"
    source.mkdir(parents=True, exist_ok=True)
    jobs = [
        (source, split, stream, index, base + offset * 1000 + index)
        for offset, (stream, count) in enumerate(
            (("combat", 36), ("nav", 16), ("quiet", 24))
        )
        for index in range(count)
    ]
    pools = defaultdict(list)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for name, stream in pool.map(collect_blind_episode, jobs):
            with np.load(name, allow_pickle=False) as z:
                visible = z["object_masks"].sum((1, 2))
                event_ticks = np.flatnonzero(z["events"].any(axis=1))
                groups = categories(z["states"], visible, stream)
                # A step's event leads to frame t+1, so frame t is still quiet.
                if stream == "quiet" and len(event_ticks):
                    groups["quiet"] = groups["quiet"][groups["quiet"] <= event_ticks[0]]
                # Exclude identical native images before selecting the final panel.
                blocked = set()
                if split == "final_blind":
                    blocked = set(
                        np.flatnonzero([frame_key(x) in prior32 for x in z["frames"]])
                    )
                for group, ticks in groups.items():
                    pools[group].extend(
                        (name, int(t)) for t in ticks if t not in blocked
                    )
    rng = np.random.default_rng(selection_seed)
    chosen = defaultdict(dict)
    for group in sorted(GROUPS, key=lambda g: len(pools[g])):
        rng.shuffle(pools[group])
        n = 0
        for name, tick in pools[group]:
            if tick in chosen[name]:
                continue
            chosen[name][tick] = group
            n += 1
            if n == quota:
                break
        if n != quota:
            raise ValueError(f"Insufficient fresh {group} frames ({n}/{quota})")
    columns = defaultdict(list)
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for record, rows in pool.map(replay, sorted(chosen.items())):
            if record["archive_native_rgb32_mismatches"]:
                raise ValueError(f"Native Source32 replay mismatch in {record['path']}")
            records.append(record)
            for row in rows:
                for k, value in row.items():
                    columns[k].append(value)
    keys = sample_keys(columns)
    overlap = {
        name: {k: len(v & ref[k]) for k, v in keys.items()}
        for name, ref in prior.items()
    }
    if any(n for checks in overlap.values() for n in checks.values()):
        raise ValueError(f"Blind panel overlaps previous data: {overlap}")
    temporary = target.with_suffix(".tmp.npz")
    np.savez(temporary, **columns)
    temporary.replace(target)
    manifest = dict(
        protocol="fresh_source_combat_visual_blind_v1",
        groups=dict(Counter(columns["group"])),
        frames=len(columns["group"]),
        episodes=records,
        sha256=digest(target),
        overlap_with_prior_splits=overlap,
        seed_base=base,
        selection_seed=selection_seed,
        frames_per_cohort=quota,
        source_episodes_generated=len(jobs),
        source_episodes_sampled=len(records),
    )
    path = output / f"{split}_manifest.json"
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(path)
    print(split, manifest["groups"], overlap, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--split",
        choices=("standard", *BLIND_PANELS),
        default="standard",
        help="Build main splits, or a fresh blind panel after its preceding splits exist.",
    )
    args = p.parse_args()
    if args.split == "standard":
        generate(args.out, {"train": 4096, "val": 384, "test": 384}, args.workers)
    else:
        generate_blind(args.out, args.split, args.workers)
