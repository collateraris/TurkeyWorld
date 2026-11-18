"""Compare paired H180 quiet pursuit reports before installing a checkpoint."""
import argparse
import json
from pathlib import Path

from .training_pipeline import atomic_json


def qualify(directory, baseline, candidate, *, holdout=None):
    metrics = {}
    checks = {}
    step = None
    panels = [(name, name, directory) for name in ("fixed", "independent", "hard_quiet")]
    fingerprints = set()
    if holdout is not None:
        panels.append(("fresh", "hard_quiet", holdout))
    for panel, dataset, folder in panels:
        reports = [
            json.loads((folder / f"{name}_{dataset}.json").read_text()) for name in (baseline, candidate)
        ]
        before, after = reports
        if before["step"] != after["step"] or step not in (None, before["step"]):
            raise ValueError("Mismatched parent checkpoint steps")
        step = before["step"]
        if dataset == "hard_quiet":
            for report in reports:
                if (
                    report["protocol"] != "quiet_pursuit_hard_H180_v1"
                    or report["frames"] != 180
                    or report["source_future_inputs"]
                    or report["reference_combat_events"]
                    or report["windows"] != len(report["per_window"])
                    or report["windows"] != 24
                ):
                    raise ValueError("Invalid hard pursuit validation protocol")
            if before["panel_sha256"] != after["panel_sha256"] or [
                w["trajectory_sha256"] for w in before["per_window"]
            ] != [w["trajectory_sha256"] for w in after["per_window"]]:
                raise ValueError("Unpaired hard episodes")
            if before["panel_sha256"] in fingerprints:
                raise ValueError("The fresh panel reuses a previous panel fingerprint")
            fingerprints.add(before["panel_sha256"])
            values = [r["summary"]["all"] for r in reports]
            keys = ("player_H180", "npc_H180", "player_wall_ticks", "npc_wall_ticks_engine_radius")
        else:
            if before["panel_fingerprint"] != after["panel_fingerprint"] or any(
                r["panel"] != panel
                or r["horizon"] != 180
                or r["Source_future_feedback"]
                or r["feedback_mode"] != "on"
                or not r["source_held_out"]
                for r in reports
            ):
                raise ValueError("Unpaired standard H180 protocols")
            checks[panel + "_navigation_identical"] = before["navigation"] == after["navigation"]
            values = [r["enemy_motion"] for r in reports]
            if values[0]["windows"] != values[1]["windows"] or any(r["count"] != 8 for r in values):
                raise ValueError("Unpaired pursuit windows")
            keys = ("player_XY_H180_mean_tiles", "enemy_XY_H180_mean_tiles", "player_wall_overlap_ticks_mean")
        paired = {key: [v[key] for v in values] for key in keys}
        metrics[panel] = paired
        for key in keys[:2]:
            old, new = paired[key]
            checks[f"{panel}_{key}"] = new < old
        for key in keys[2:]:
            old_wall, new_wall = paired[key]
            checks[f"{panel}_{key}"] = new_wall <= old_wall
    return dict(step=step, passed=all(checks.values()), checks=checks, metrics=metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--baseline", default="base")
    parser.add_argument("--candidate", default="candidate")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--holdout", type=Path, help="Separate directory for a second 24-window hard panel")
    args = parser.parse_args()
    result = qualify(args.directory, args.baseline, args.candidate, holdout=args.holdout)
    if args.output:
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
