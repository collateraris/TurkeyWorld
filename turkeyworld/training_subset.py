"""Create an immutable, policy-balanced training-only transition subset.

Only selected episodes' small metadata/actions arrays are decoded. RGB arrays
are never read. Growing the source corpus does not grow an existing manifest.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from .data_expansion import TRAIN_POLICIES
from .prepare_training_data import current_behavior_version


def _training_directory(directory):
    directory = Path(directory).resolve()
    if not directory.is_dir() or directory.name != "train":
        raise ValueError("Subset source must be an existing explicit train directory; val/test are forbidden")
    return directory


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode("utf-8")).hexdigest()


def _episode_metadata(path, expected_behavior):
    """Validate actual provenance without opening frames.npy or states.npy."""
    with np.load(path, allow_pickle=False) as saved:
        required = {"behavior_version", "split", "policy", "actions", "seed", "frames"}
        if not required.issubset(saved.files):
            raise ValueError(f"{path.name}: missing subset provenance fields {sorted(required-set(saved.files))}")
        if str(saved["behavior_version"]) != expected_behavior:
            raise ValueError(f"{path.name}: wrong behavior version")
        if str(saved["split"]) != "train":
            raise ValueError(f"{path.name}: refusing non-training split")
        policy = str(saved["policy"])
        if policy not in TRAIN_POLICIES:
            raise ValueError(f"{path.name}: unknown training policy {policy}")
        actions = saved["actions"]
        if (actions.ndim != 2 or actions.shape[1] != 4 or len(actions) < 1
                or not np.isfinite(actions).all()):
            raise ValueError(f"{path.name}: invalid actions shape or values")
        return {"filename": path.name, "policy": policy, "transitions": len(actions),
                "seed": int(saved["seed"]), "size_bytes": path.stat().st_size,
                "actions_sha256": hashlib.sha256(actions.dtype.str.encode("ascii")
                    + np.ascontiguousarray(actions).tobytes()).hexdigest()}


def _candidate_policies(directory, expected_behavior):
    """Use the fresh expansion seed rule, otherwise inspect small NPZ fields."""
    paths = sorted(directory.glob("*.npz"))
    manifest_path = directory.parent.parent / "manifest.json"
    if manifest_path.is_file():
        corpus = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = corpus.get("config", {})
        if (corpus.get("behavior_version") != expected_behavior
                or config.get("behavior_version") != expected_behavior):
            raise ValueError("Source corpus manifest has the wrong behavior version")
        if (corpus.get("expansion_version") == 1 and config.get("source") is None
                and corpus.get("train_policies") == list(TRAIN_POLICIES)):
            # Fresh expansion jobs use base_seed+index and policy[index %9].
            # Every selected file is independently checked against this hint.
            base = int(config["seed"])
            planned = int(config["targets"]["rgb_train"])
            groups = {policy: [] for policy in TRAIN_POLICIES}
            for path in paths:
                match = re.fullmatch(r"episode_(\d+)\.npz", path.name)
                if match and 0 <= int(match[1])-base < planned:
                    groups[TRAIN_POLICIES[(int(match[1])-base) % len(TRAIN_POLICIES)]].append(path.name)
            return groups, int(config["clip_frames"]), "fresh_expansion_seed_rule"
    groups = {policy: [] for policy in TRAIN_POLICIES}
    lengths = set()
    for path in paths:
        metadata = _episode_metadata(path, expected_behavior)
        groups[metadata["policy"]].append(path.name)
        lengths.add(metadata["transitions"])
    if len(lengths) != 1:
        raise ValueError("Fixed subset requires nonempty, uniformly sized transition clips")
    return groups, lengths.pop(), "validated_episode_metadata"


def _policy_quotas(count, seed, capacities=None):
    quotient, remainder = divmod(count, len(TRAIN_POLICIES))
    eligible = list(TRAIN_POLICIES)
    if capacities is not None:
        if any(capacities.get(policy, 0) < quotient for policy in TRAIN_POLICIES):
            raise ValueError("Insufficient policy capacity for balanced base quotas")
        eligible = [policy for policy in TRAIN_POLICIES if capacities.get(policy, 0) >= quotient+1]
        if len(eligible) < remainder:
            raise ValueError("Insufficient policy capacity for balanced extra quotas")
    extra = set(sorted(eligible, key=lambda policy:
        hashlib.sha256(f"{seed}/quota/{policy}".encode("utf-8")).digest())[:remainder])
    return {policy: quotient+int(policy in extra) for policy in TRAIN_POLICIES}


def _select_names(groups, count, seed, inspect=None):
    quotas = _policy_quotas(count, seed, {policy: len(groups[policy]) for policy in TRAIN_POLICIES})
    selected = []
    for policy in TRAIN_POLICIES:
        quota = quotas[policy]
        candidates = groups[policy]
        if len(candidates) < quota:
            raise ValueError(f"Insufficient {policy} clips: need {quota}, found {len(candidates)}")
        ranked = sorted(candidates, key=lambda name: (
            hashlib.sha256(f"{seed}/{policy}/{name}".encode("utf-8")).digest(), name))
        accepted = 0
        for name in ranked:
            if accepted == quota:
                break
            if inspect is None or inspect(name, policy):
                selected.append((name, policy))
                accepted += 1
        if accepted < quota:
            raise ValueError(f"Insufficient full-length {policy} clips: need {quota}, found {accepted}")
    return sorted(selected)


def _identity(manifest):
    return {key: manifest[key] for key in ("format_version", "selection_version", "source_directory",
        "split", "behavior_version", "seed", "clip_transitions", "requested_transitions",
        "total_transitions", "count_files", "policy_counts", "filenames", "records")}


def load_training_subset(path, directory, expected_behavior=None):
    """Validate and load fixed filenames; no discovery, append, or file writes."""
    directory = _training_directory(directory)
    behavior = current_behavior_version() if expected_behavior is None else str(expected_behavior)
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        if (manifest["format_version"] != 1 or manifest["selection_version"] != 1
                or manifest["split"] != "train" or manifest["behavior_version"] != behavior
                or manifest["source_directory"] != str(directory)):
            raise ValueError("Subset source directory, split, version or behavior mismatch")
        if _digest(_identity(manifest)) != manifest["fingerprint"]:
            raise ValueError("Subset manifest fingerprint mismatch")
        names, records = manifest["filenames"], manifest["records"]
        if (not names or names != sorted(set(names)) or len(names) != manifest["count_files"]
                or len(records) != len(names) or [record["filename"] for record in records] != names
                or any(Path(name).name != name or not name.endswith(".npz") for name in names)):
            raise ValueError("Subset filenames are invalid, duplicated or inconsistent")
        policies = Counter()
        total = 0
        for name, recorded in zip(names, records):
            target = directory / name
            if target.resolve().parent != directory:
                raise ValueError("Subset episode escapes the training directory")
            if not target.is_file():
                raise ValueError(f"Selected subset episode is missing: {name}")
            actual = _episode_metadata(target, behavior)
            if actual != recorded or actual["transitions"] != manifest["clip_transitions"]:
                raise ValueError(f"Subset episode changed: {name}")
            policies[actual["policy"]] += 1
            total += actual["transitions"]
        expected_counts = {policy: policies[policy] for policy in TRAIN_POLICIES}
        if (total != manifest["total_transitions"] or total != manifest["requested_transitions"]
                or expected_counts != manifest["policy_counts"]
                or max(expected_counts.values())-min(expected_counts.values()) > 1):
            raise ValueError("Subset transition total or policy quotas are inconsistent")
    except (KeyError, TypeError) as error:
        raise ValueError("Malformed training subset manifest") from error
    return manifest


def create_training_subset(source, output, transitions=128_000, seed=173, expected_behavior=None):
    """Create exactly the requested transition count, or reuse an identical plan."""
    directory = _training_directory(source)
    output = Path(output)
    behavior = current_behavior_version() if expected_behavior is None else str(expected_behavior)
    if not isinstance(transitions, int) or transitions < 1 or not isinstance(seed, int):
        raise ValueError("Transition count must be a positive integer and seed must be an integer")
    if output.exists():
        manifest = load_training_subset(output, directory, behavior)
        if manifest["requested_transitions"] != transitions or manifest["seed"] != seed:
            raise ValueError("Existing immutable subset has different requested transitions or seed")
        return manifest
    groups, clip_size, inference = _candidate_policies(directory, behavior)
    if clip_size < 1 or transitions % clip_size:
        raise ValueError("Requested transitions must be an exact multiple of the clip length")
    checked = {}

    def inspect(name, policy):
        path = directory / name
        if path.resolve().parent != directory:
            raise ValueError("Subset episode escapes the training directory")
        record = _episode_metadata(path, behavior)
        if (record["policy"] != policy or inference == "fresh_expansion_seed_rule"
                and record["seed"] != int(re.fullmatch(r"episode_(\d+)\.npz", name)[1])):
            raise ValueError(f"Selected episode does not match collector policy/seed: {name}")
        if record["transitions"] != clip_size:
            return False
        checked[name] = record
        return True

    selected = _select_names(groups, transitions // clip_size, seed, inspect)
    records = [checked[name] for name, _ in selected]
    counts = Counter(record["policy"] for record in records)
    manifest = {"format_version": 1, "selection_version": 1, "source_directory": str(directory),
        "split": "train", "behavior_version": behavior, "seed": seed,
        "clip_transitions": clip_size, "requested_transitions": transitions,
        "total_transitions": sum(record["transitions"] for record in records),
        "count_files": len(records), "policy_counts": {policy: counts[policy] for policy in TRAIN_POLICIES},
        "filenames": [record["filename"] for record in records], "records": records,
        "candidate_policy_source": inference, "immutable": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "identity_scope": "Selected names, provenance, sizes and action hashes; RGB is never decoded or content-hashed."}
    if manifest["total_transitions"] != transitions:
        raise ValueError("Selected subset does not contain the exact requested transition count")
    manifest["fingerprint"] = _digest(_identity(manifest))
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=output.name+".", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # Atomic no-clobber publication: an existing immutable plan wins.
            os.link(temporary, output)
        except FileExistsError:
            existing = load_training_subset(output, directory, behavior)
            if existing["seed"] != seed or existing["requested_transitions"] != transitions:
                raise ValueError("Concurrent immutable subset has different configuration")
            return existing
    finally:
        Path(temporary).unlink(missing_ok=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transitions", type=int, default=128_000)
    parser.add_argument("--seed", type=int, default=173)
    parser.add_argument("--behavior-version", help="Require this source behavior version; defaults to the current engine")
    args = parser.parse_args(argv)
    manifest = create_training_subset(args.source, args.output, args.transitions, args.seed, args.behavior_version)
    print(json.dumps({key: manifest[key] for key in ("count_files", "total_transitions", "policy_counts", "fingerprint")},
                     ensure_ascii=False), flush=True)
    return manifest


if __name__ == "__main__":
    main()
