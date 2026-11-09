"""Prepare an immutable replay mixture without activating any training run.

The manifest uses the ordinary fixed-subset schema. Additional replay provenance
records both source memberships and whole-file hashes, without decoding RGB.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .data_expansion import TRAIN_POLICIES
from .prepare_training_data import current_behavior_version
from .training_subset import (_digest, _episode_metadata, _identity,
                              _training_directory, load_training_subset)


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _joint_policy_quotas(old_count, new_count, old_capacities, new_capacities, seed):
    """Balance each origin and the combined set, respecting actual capacities."""
    def possibilities(count, capacities, origin):
        base, remainder = divmod(count, len(TRAIN_POLICIES))
        if any(capacities.get(policy, 0) < base for policy in TRAIN_POLICIES):
            raise ValueError(f"Insufficient {origin} policy capacity for balanced replay")
        eligible = [p for p in TRAIN_POLICIES if capacities.get(p, 0) > base]
        if len(eligible) < remainder:
            raise ValueError(f"Insufficient {origin} policy capacity for balanced replay extras")
        choices = list(combinations(eligible, remainder))
        choices.sort(key=lambda extra: _digest([seed, origin, list(extra)]))
        return base, choices

    old_base, old_choices = possibilities(old_count, old_capacities, "old")
    new_base, new_choices = possibilities(new_count, new_capacities, "new")
    for old_extra in old_choices:
        old = {p: old_base + int(p in old_extra) for p in TRAIN_POLICIES}
        for new_extra in new_choices:
            new = {p: new_base + int(p in new_extra) for p in TRAIN_POLICIES}
            total = [old[p] + new[p] for p in TRAIN_POLICIES]
            if max(total) - min(total) <= 1:
                return old, new
    raise ValueError("Policy capacities cannot balance both replay origins and their combined set")


def _validation_identity(directory, behavior):
    directory = Path(directory).resolve()
    if not directory.is_dir() or directory.name != "val":
        raise ValueError("Current validation must be an existing explicit val directory")
    records = []
    seeds = set()
    for path in sorted(directory.glob("*.npz")):
        if path.resolve().parent != directory:
            raise ValueError("Validation episode escapes its directory")
        with np.load(path, allow_pickle=False) as saved:
            if (not {"seed", "split", "behavior_version"}.issubset(saved.files)
                    or str(saved["split"]) != "val"
                    or str(saved["behavior_version"]) != behavior):
                raise ValueError(f"{path.name}: invalid current validation provenance")
            seed = int(saved["seed"])
        if seed in seeds:
            raise ValueError("Duplicate current validation seeds")
        seeds.add(seed)
        records.append({"filename": path.name, "seed": seed})
    if not records:
        raise ValueError("Current validation directory is empty")
    return {"directory": str(directory), "records": records, "fingerprint": _digest(records)}, seeds


def _unique_seeds(records, origin):
    seeds = [record["seed"] for record in records]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Duplicate {origin} source seeds")
    return set(seeds)


def _select(records, quotas, seed, origin):
    selected = []
    for policy in TRAIN_POLICIES:
        group = [r for r in records if r["policy"] == policy]
        group.sort(key=lambda r: (_digest([seed, origin, policy, r["filename"]]), r["filename"]))
        selected.extend(group[:quotas[policy]])
    return sorted(selected, key=lambda record: record["filename"])


def _verify_replay(manifest, request, source_manifest, validation):
    try:
        provenance = manifest["replay_provenance"]
        if _digest(provenance) != manifest["replay_fingerprint"]:
            raise ValueError("Replay provenance fingerprint mismatch")
        if provenance["request"] != request:
            raise ValueError("Existing immutable replay has a different request")
        if provenance["current_subset_fingerprint"] != source_manifest["fingerprint"]:
            raise ValueError("Current source subset fingerprint changed")
        if provenance["validation"] != validation:
            raise ValueError("Current validation seed identity changed")
        members = provenance["old_members"] + provenance["new_members"]
        if (sorted(member["filename"] for member in members) != manifest["filenames"]
                or len(members) != len(manifest["filenames"])):
            raise ValueError("Replay provenance membership mismatch")
        for origin in ("old", "new"):
            directory = Path(request[f"{origin}_train_directory"])
            group = provenance[f"{origin}_members"]
            if (sum(member["identity"]["transitions"] for member in group)
                    != request[f"{origin}_transitions"]):
                raise ValueError("Replay origin transition count changed")
            for member in group:
                name = member["filename"]
                source = directory / name
                if (Path(name).name != name or source.resolve().parent != directory
                        or _episode_metadata(source, request["behavior_version"]) != member["identity"]
                        or _file_hash(source) != member["sha256"]
                        or _file_hash(Path(manifest["source_directory"]) / name) != member["sha256"]):
                    raise ValueError(f"Replay source or destination content changed: {name}")
    except (KeyError, TypeError, OSError) as error:
        raise ValueError("Malformed or missing replay provenance/content") from error
    return manifest


def build_replay_subset(current_manifest, old_train_dir, new_train_dir,
                        destination_train_dir, output_manifest, *,
                        old_transitions=16_000, new_transitions=48_000, seed=173,
                        validation_directory=None):
    """Prepare old/new clips as hardlinks (copy fallback); never start training.

    Old membership is restricted to the current fixed manifest. Every fresh
    candidate must be a complete 32-transition clip, even if not selected.
    Existing immutable output is validated and reused without rewriting it.
    """
    old_dir, new_dir = _training_directory(old_train_dir), _training_directory(new_train_dir)
    destination = Path(destination_train_dir).resolve()
    output = Path(output_manifest).resolve()
    behavior = current_behavior_version()
    if (destination.name != "train" or len({old_dir, new_dir, destination}) != 3
            or destination.exists() and not destination.is_dir()):
        raise ValueError("Replay destination must be a separate explicit train directory")
    if (any(not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in (old_transitions, new_transitions))
            or not isinstance(seed, int) or isinstance(seed, bool)):
        raise ValueError("Replay transitions must be positive integers and seed must be an integer")
    if old_transitions % 32 or new_transitions % 32:
        raise ValueError("Replay transitions must be exact multiples of 32")
    current = load_training_subset(current_manifest, old_dir, behavior)
    if current["clip_transitions"] != 32:
        raise ValueError("Replay requires 32-transition source clips")
    validation, validation_seeds = _validation_identity(
        validation_directory if validation_directory is not None else old_dir.parent / "val", behavior)
    old_records = current["records"]
    old_seeds = _unique_seeds(old_records, "old")
    if old_seeds & validation_seeds:
        raise ValueError("Old source seeds overlap current validation")
    request = {"old_train_directory": str(old_dir), "new_train_directory": str(new_dir),
               "destination_train_directory": str(destination), "current_manifest": str(Path(current_manifest).resolve()),
               "old_transitions": old_transitions, "new_transitions": new_transitions,
               "seed": seed, "behavior_version": behavior, "clip_transitions": 32}
    if output.exists():
        existing = load_training_subset(output, destination, behavior)
        return _verify_replay(existing, request, current, validation)

    new_records = []
    for path in sorted(new_dir.glob("*.npz")):
        if path.resolve().parent != new_dir:
            raise ValueError("New episode escapes its training directory")
        record = _episode_metadata(path, behavior)
        if record["transitions"] != 32:
            raise ValueError(f"{path.name}: fresh replay candidate is not a complete 32-transition clip")
        new_records.append(record)
    new_seeds = _unique_seeds(new_records, "new")
    if new_seeds & (old_seeds | validation_seeds):
        raise ValueError("New source seeds overlap current subset or validation")
    capacities = [Counter(r["policy"] for r in records) for records in (old_records, new_records)]
    old_quotas, new_quotas = _joint_policy_quotas(old_transitions // 32, new_transitions // 32,
                                                *capacities, seed)
    selected = {"old": _select(old_records, old_quotas, seed, "old"),
                "new": _select(new_records, new_quotas, seed, "new")}
    records = sorted(selected["old"] + selected["new"], key=lambda record: record["filename"])
    names = [record["filename"] for record in records]
    if len(names) != len(set(names)):
        raise ValueError("Old/new replay filenames collide")
    if destination.exists() and any(path.name not in names for path in destination.glob("*.npz")):
        raise ValueError("Replay destination contains unrelated episodes")
    provenance = {"format_version": 1, "prepared_only": True,
        "request": request, "current_subset_fingerprint": current["fingerprint"],
        "validation": validation, "old_fraction": old_transitions / (old_transitions + new_transitions),
        "old_policy_counts": old_quotas, "new_policy_counts": new_quotas,
        "identity_scope": "Complete selected NPZ bytes (SHA256), source metadata and membership; RGB is not decoded."}
    for origin, directory in (("old", old_dir), ("new", new_dir)):
        provenance[f"{origin}_members"] = [{"filename": record["filename"], "identity": record,
            "sha256": _file_hash(directory / record["filename"])} for record in selected[origin]]

    # Complete and validate the plan before making any destination entries.
    destination.mkdir(parents=True, exist_ok=True)
    for origin, directory in (("old", old_dir), ("new", new_dir)):
        for member in provenance[f"{origin}_members"]:
            source, target = directory / member["filename"], destination / member["filename"]
            if target.exists():
                if (target.resolve().parent != destination or not target.is_file()
                        or _file_hash(target) != member["sha256"]):
                    raise ValueError(f"Existing replay destination episode changed: {target.name}")
            else:
                try:
                    os.link(source, target)
                except OSError:
                    # Exclusive creation also works on filesystems without links.
                    try:
                        with source.open("rb") as incoming, target.open("xb") as outgoing:
                            shutil.copyfileobj(incoming, outgoing)
                        shutil.copystat(source, target)
                    except FileExistsError:
                        if _file_hash(target) != member["sha256"]:
                            raise ValueError(f"Concurrent replay destination episode changed: {target.name}")
                if _file_hash(target) != member["sha256"]:
                    raise ValueError(f"Replay source changed while preparing: {target.name}")
    counts = Counter(record["policy"] for record in records)
    manifest = {"format_version": 1, "selection_version": 1, "source_directory": str(destination),
        "split": "train", "behavior_version": behavior, "seed": seed, "clip_transitions": 32,
        "requested_transitions": old_transitions + new_transitions,
        "total_transitions": sum(r["transitions"] for r in records), "count_files": len(records),
        "policy_counts": {p: counts[p] for p in TRAIN_POLICIES}, "filenames": names, "records": records,
        "immutable": True, "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_policy_source": "balanced_fixed_subset_replay",
        "replay_provenance": provenance, "replay_fingerprint": _digest(provenance)}
    manifest["fingerprint"] = _digest(_identity(manifest))
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Never publish a manifest whose members changed during preparation.
        load_training_subset(temporary, destination, behavior)
        try:
            os.link(temporary, output)
        except FileExistsError:
            existing = load_training_subset(output, destination, behavior)
            return _verify_replay(existing, request, current, validation)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return load_training_subset(output, destination, behavior)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-manifest", type=Path, required=True)
    parser.add_argument("--old-train-dir", type=Path, required=True)
    parser.add_argument("--new-train-dir", type=Path, required=True)
    parser.add_argument("--destination-train-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--old-transitions", type=int, default=16_000)
    parser.add_argument("--new-transitions", type=int, default=48_000)
    parser.add_argument("--seed", type=int, default=173)
    parser.add_argument("--validation-directory", type=Path)
    args = parser.parse_args(argv)
    manifest = build_replay_subset(**vars(args))
    print(json.dumps({key: manifest[key] for key in ("count_files", "total_transitions", "policy_counts",
                                                     "fingerprint", "replay_fingerprint")}), flush=True)
    return manifest


if __name__ == "__main__":
    main()
