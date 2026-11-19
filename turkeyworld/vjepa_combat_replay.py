"""Source combat collection and TRAIN-only replay for the three-stream V-JEPA cycle."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

from .collect_combat_training import COMBAT_CASES, collect_job, digest
from .engine import BEHAVIOR_VERSION, EVENT_NAMES
from .jepa_training_utils import SequenceData
from .training_pipeline import atomic_json, now
from .workspace_paths import ROOT

PROTOCOL = "vjepa_source_combat360_v1"
ORIGINAL = ROOT / "training_runs/combat_source_20261111/data"
HORIZON = 180
EPISODES = 64
STEPS = 360


def collection_plan(cycle_index):
    if cycle_index < 0:
        raise ValueError("Negative combat round")
    return dict(
        protocol=PROTOCOL,
        cycle_index=cycle_index,
        split="train",
        stream="combat",
        behavior_version=BEHAVIOR_VERSION,
        episodes_per_stream=EPISODES,
        max_episode_steps=STEPS,
        hz=30,
        entity_slots=1,
        cases=list(COMBAT_CASES),
        seed_base=3000000000 + cycle_index * EPISODES,
        source_state_repairs=0,
    )


def collection_jobs(root, plan):
    for i in range(EPISODES):
        seed = plan["seed_base"] + i
        relative = f"full_game/rgb/train/episode_{seed}.npz"
        yield dict(
            index=i // len(COMBAT_CASES),
            seed=seed,
            case=COMBAT_CASES[i % len(COMBAT_CASES)],
            protocol=PROTOCOL,
            stage="full_game",
            split="train",
            frames=STEPS,
            relative_path=relative,
            path=str(root / relative),
        )


def collect(root, cycle_index, workers=4):
    """Each worker publishes a separate atomic episode. Resume consumes completed jobs."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if workers < 1:
        raise ValueError("At least one collector worker is required")
    with FileLock(str(root / "collector.lock")):
        plan = collection_plan(cycle_index)
        receipt = root / "collection_receipt.json"
        identity = root / "collection_plan.json"
        if identity.is_file() and json.loads(identity.read_text()) != plan:
            raise ValueError("Existing combat collection plan differs")
        if receipt.is_file():
            saved = json.loads(receipt.read_text())
            if saved["plan"] != plan:
                raise ValueError("Existing combat collection receipt differs")
            return saved
        atomic_json(identity, plan)
        jobs = iter(collection_jobs(root, plan))
        rows = []

        def progress(status):
            atomic_json(
                root / "collection_progress.json",
                dict(
                    status=status,
                    accepted=len(rows),
                    total_episodes=EPISODES,
                    cycle_index=cycle_index,
                    updated_at=now(),
                ),
            )

        progress("collecting")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            current = set()
            while len(rows) < EPISODES:
                if (root / "stop.request").exists():
                    progress("stopped")
                    raise InterruptedError("Stopped combat collection between episodes")
                for _ in range(workers - len(current)):
                    job = next(jobs, None)
                    if job is None:
                        break
                    current.add(pool.submit(collect_job, job))
                done, current = wait(current, return_when=FIRST_COMPLETED)
                for future in done:
                    row = future.result()
                    if row["transitions"] < HORIZON:
                        raise ValueError("Source combat ended before one H180 window")
                    rows.append(row)
                progress("collecting")
        rows.sort(key=lambda row: row["seed"])
        counts = {k: sum(r["events"][k] for r in rows) for k in (*EVENT_NAMES, "respawn")}
        if not all(counts.values()) or counts["shot"] <= counts["hit"]:
            raise ValueError(f"Incomplete TRAIN combat event coverage: {counts}")
        output = dict(
            protocol=PROTOCOL,
            plan=plan,
            episodes=rows,
            split="train",
            events=counts,
            total_real_Source_transitions=sum(r["transitions"] for r in rows),
            completed_at=now(),
        )
        atomic_json(receipt, output)
        progress("complete")
        return output


COMBAT_WINDOWS = ("uniform", "shot_miss", "hit", "player_hit", "kill", "respawn")


def window_groups(events, respawn):
    """An H180 window at start s uses events[s:s+180], including its last tick."""
    n = len(events) - HORIZON + 1
    if n < 1:
        raise ValueError("Incomplete combat H180 window")
    shot, hit, kill, player_hit = events.astype(bool).T
    flags = (shot & ~hit, hit, player_hit, kill, respawn[:, 0].astype(bool))
    groups = {"uniform": np.arange(n, dtype=np.int32)}
    for name, flag in zip(COMBAT_WINDOWS[1:], flags, strict=True):
        prefix = np.concatenate(([0], np.cumsum(flag, dtype=np.int64)))
        groups[name] = np.flatnonzero(prefix[HORIZON:] > prefix[:-HORIZON]).astype(np.int32)
    return groups


def corpus_rows(directory, records, protocol, seen_seeds):
    train = (directory / "full_game/rgb/train").resolve()
    files, indexes = [], {name: [] for name in COMBAT_WINDOWS}
    for i, row in enumerate(records):
        path = (directory / row["relative_path"]).resolve()
        if (
            row["stage"] != "full_game"
            or row["split"] != "train"
            or path.parent != train
            or path.name != f"episode_{row['seed']:09d}.npz"
            or row["seed"] in seen_seeds
        ):
            raise ValueError("Invalid combat TRAIN identity or duplicate seed")
        if digest(path) != row["sha256"]:
            raise ValueError(f"Changed Source combat TRAIN episode: {path}")
        seen_seeds.add(row["seed"])
        with np.load(path, allow_pickle=False) as ep:
            events, attacks = ep["events"], ep["attack_flags"]
            respawn, states, hits = ep["respawn_flags"], ep["states"], ep["hit_ids"]
            if (
                str(ep["split"]) != "train"
                or str(ep["stage"]) != "full_game"
                or int(ep["seed"]) != row["seed"]
                or ep["behavior_version"].item() != BEHAVIOR_VERSION
                or ep["collection_protocol"].item() != protocol
                or abs(float(ep["dt"]) - 1 / 30) > 1e-7
                or not ep["enemies_enabled"].item()
                or len(events) != row["transitions"]
                or not np.array_equal(events[:, 1] > 0.5, hits >= 0)
                or not np.isin(hits, [-1, 0]).all()
                or not np.array_equal(events[:, 3], attacks[:, 0])
                or not np.array_equal(
                    respawn[:, 0] > 0.5, (states[:-1, 13] < 0.5) & (states[1:, 13] > 0.5)
                )
            ):
                raise ValueError(f"Misaligned combat labels or schema: {path}")
            for name, starts in window_groups(events, respawn).items():
                if len(starts):
                    indexes[name].append(np.column_stack((np.full(len(starts), i, np.int32), starts)))
        files.append(path)
    if not files or set(files) != set(train.glob("episode*.npz")):
        raise ValueError(f"Combat Source TRAIN files disagree with the receipt: {train}")
    data = SequenceData(
        train,
        device="cpu",
        split="train",
        image_size=32,
        mode="stream",
        cache_bytes=64 * 1024**2,
        episode_paths=files,
    )
    return data, {name: np.concatenate(parts) for name, parts in indexes.items() if parts}


class CombatReplay:
    """Draw from original/history/newest Source combat TRAIN; reserve half for the newest."""

    def __init__(self, cycle_root, saved=None):
        self.rng = np.random.default_rng(175)
        if saved is not None:
            self.rng.bit_generator.state = saved
        manifest = json.loads((ORIGINAL / "manifest.json").read_text())
        if manifest["train_val_seed_overlap"] or manifest["status"] != "complete":
            raise ValueError("Original combat train/validation provenance is incomplete")
        original = [
            row for row in manifest["records"] if row["stage"] == "full_game" and row["split"] == "train"
        ]
        available = [(ORIGINAL, original, manifest["fingerprint"])]
        for path in sorted(
            (Path(cycle_root) / "cycles").glob("cycle_*/combat_v1/collection_receipt.json")
        ):
            receipt = json.loads(path.read_text())
            number = int(path.parent.parent.name.removeprefix("cycle_"))
            plan = collection_plan(number)
            expected_seeds = set(range(plan["seed_base"], plan["seed_base"] + EPISODES))
            if (
                receipt.get("protocol") != PROTOCOL
                or receipt.get("plan") != plan
                or len(receipt["episodes"]) != EPISODES
                or {r["seed"] for r in receipt["episodes"]} != expected_seeds
            ):
                raise ValueError(f"Invalid combat round receipt: {path}")
            available.append((path.parent, receipt["episodes"], PROTOCOL))
        if len(available) < 2:
            raise ValueError("Missing completed fresh combat TRAIN collection for three-stream mode")
        self.data, self.pools, self.records = [], [], []
        seen = set()
        for folder, rows, protocol in available:
            rows = sorted(rows, key=lambda r: r["relative_path"])
            data, groups = corpus_rows(folder, rows, protocol, seen)
            self.data.append(data)
            self.pools.append(groups)
            self.records.append(rows)

    def draw(self, count):
        if count < 1:
            raise ValueError("Empty combat draw")
        older_count = count // 2
        original_count = (older_count + 1) // 2 if len(self.data) > 2 else older_count
        rotation = int(self.rng.integers(len(COMBAT_WINDOWS)))
        rows = []
        for i in range(count):
            if i < original_count:
                corpus, role = 0, "original"
            elif i < older_count:
                corpus, role = int(self.rng.integers(1, len(self.data) - 1)), "history"
            else:
                corpus, role = len(self.data) - 1, "newest"
            pool = self.pools[corpus]
            kinds = list(pool)
            kind = kinds[(rotation + i) % len(kinds)]
            options = pool[kind]
            episode, start = map(int, options[self.rng.integers(len(options))])
            data = self.data[corpus]
            item = data.episodes[episode]
            batch = data.batch(1, context=1, horizon=HORIZON, indices=np.array([[episode, start]]))
            history = np.maximum(0, np.arange(start - 3, start + 1))
            batch["initial_states"] = torch.as_tensor(item["states"][history]).unsqueeze(0).float()
            batch["initial_frames"] = (
                torch.as_tensor(item["frames"][history]).permute(0, 3, 1, 2).unsqueeze(0).float() / 255
            )
            batch.update(
                offset=start,
                file=str(item["path"]),
                corpus=corpus,
                corpus_role=role,
                motion_kind=kind,
                requested_motion_kind=kind,
                sampling_attempts=1,
                available_motion_kinds=kinds,
                episode_sha256=self.records[corpus][episode]["sha256"],
            )
            rows.append(batch)
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cycle-index", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    receipt = collect(args.output_dir, args.cycle_index, args.workers)
    print(
        json.dumps(
            dict(
                episodes=len(receipt["episodes"]),
                events=receipt["events"],
                transitions=receipt["total_real_Source_transitions"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
