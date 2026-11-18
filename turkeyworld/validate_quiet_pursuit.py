"""Locked, Source-only selection of 24 hard no-combat pursuits, followed by free H180 V-JEPA forecasts.

Tape decisions use a random seed and initial relative heading only, never Source future, model outputs
or benchmark metrics. Source future verifies non-combat and membership in each requested stratum,
then supplies labels exclusively. All scored states after the four initial observations are predicted.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .audit_vjepa_npc_target import source_masks
from .engine import Game, EVENT_NAMES
from .models import load_world_checkpoint
from .quiet_pair_profile import MODE as QUIET_MODE, quiet_mask
from .rssm_combat_protocol import environment_profile
from .structured_state import MAP_EXTENT, PLAYER_SPEED
from .training_pipeline import atomic_json, now
from .trajectory_objectives import circle_wall_penalty
from .vjepa_policy_world import VJEPAImaginationAdapter

PROTOCOL = "quiet_pursuit_hard_H180_v1"
SEED = 459927813
KINDS = ("occluded_turn", "strafe_facing", "distance_change")
N = 8


def visibility(g, s):
    """End-of-tick visible through map and inside horizontal camera frustum (not SR sprite detection)."""
    d = (s[:, 8:10] - s[:, :2]) * 16
    fw = d[:, 0] * s[:, 3] + d[:, 1] * s[:, 2]
    side = d[:, 1] * s[:, 3] - d[:, 0] * s[:, 2]
    return (
        (fw > 0.12)
        & (np.abs(side) < fw * math.tan(math.radians(34)))
        & np.array([g._line_clear(*(z[:2] * 16), *(z[8:10] * 16)) for z in s])
    )


def tapes(rng, kind):
    a = np.zeros((180, 4), dtype=np.float32)
    sign = rng.choice([-1.0, 1.0])
    scale = rng.uniform(0.75, 1)
    if kind == "occluded_turn":
        a[:60, :3] = [-0.15, sign * 0.75, 0.15 * sign]
        a[60:120, :3] = [0.5, 0.45 * sign, -0.22 * sign]
        a[120:, :3] = [-0.5, -0.4 * sign, 0]
    elif kind == "strafe_facing":
        a[:72, :3] = [-0.27, sign, -0.3 * sign]
        a[72:144, :3] = [-0.29, -sign, 0.29 * sign]
        a[144:, :3] = [-0.55, 0.45 * sign, -0.15 * sign]
    else:
        a[:46, :3] = [0.8, sign * 0.2, 0]
        a[46:142, :3] = [-1, sign * 0.4, -0.12 * sign]
        a[142:, :3] = [0.6, -0.3 * sign, 0.2 * sign]
    a[:, :3] *= scale
    return a


def generate(seed=SEED):
    rng = np.random.default_rng(seed)
    g = Game(seed=seed, endless=True, enemies_enabled=True)
    cells = np.argwhere(g.map == 0)[:, ::-1]
    out = []
    for kind in KINDS:
        count = 0
        for attempt in range(10000):
            starts = cells[rng.integers(len(cells), size=2)] + rng.uniform(0.24, 0.76, (2, 2))
            p, e = starts
            delta = e - p
            dist = np.linalg.norm(delta)
            if dist < 3.3 or dist > 10.5:
                continue
            if not g._walkable(*p, Game.PLAYER_RADIUS) or not g._walkable(*e, Game.ENEMY_RADIUS):
                continue
            clear = g._line_clear(*p, *e)
            if kind == "occluded_turn" and clear or kind == "strafe_facing" and not clear:
                continue
            g.reset(seed + len(out) + 1)
            st = g.get_state()
            st["player"].update(
                x=p[0], y=p[1], angle=math.atan2(delta[1], delta[0]) + rng.uniform(-0.12, 0.12)
            )
            st["enemies"][0].update(x=e[0], y=e[1])
            g.set_state(st)
            a = tapes(rng, kind)
            z = [g.state_vector()]
            for cmd in np.vstack([np.zeros((3, 4)), a]):
                ev = g.step(cmd, 1 / 30)
                if any(ev[k] for k in EVENT_NAMES):
                    break
                z.append(g.state_vector())
            if len(z) != 184:
                continue
            z = np.stack(z)
            t = z[4:]
            assert not a[:, 3].any() and np.all(z[:, [4, 12, 13]] == z[0, [4, 12, 13]])
            masks = source_masks(g.map, z[3], t)
            vis = visibility(g, t)
            gap = np.linalg.norm(t[:, :2] - t[:, 8:10], axis=1) * 16
            if kind == "occluded_turn" and (
                masks["occluded"].sum() < 60 or masks["occluded_recent_turn"].sum() < 6
            ):
                continue
            if kind == "strafe_facing" and (
                vis.sum() < 60 or np.linalg.norm(np.diff(z[:, 0:2], axis=0), axis=1).sum() * 16 < 2
            ):
                continue
            if kind == "distance_change" and (np.ptp(gap) < 2 or np.min(gap) > 3.1):
                continue
            arr_hash = hashlib.sha256(z.tobytes() + a.tobytes() + g.map.tobytes()).hexdigest()
            out.append(
                dict(
                    id=len(out),
                    kind=kind,
                    states=z,
                    actions=a,
                    grid=g.map.copy(),
                    visible_frames=int(vis.sum()),
                    occluded_frames=int(masks["occluded"].sum()),
                    occluded_turn_frames=int(masks["occluded_recent_turn"].sum()),
                    gap_min=float(gap.min()),
                    gap_max=float(gap.max()),
                    trajectory_sha256=arr_hash,
                )
            )
            count += 1
            if count == N:
                break
        if count != N:
            raise ValueError(f"Could not fill independent class {kind}")
    return out


def prepare(cache, seed=SEED):
    path = cache / "quiet_panel.npz"
    specpath = cache / "quiet_panel_manifest.json"
    cache.mkdir(parents=True, exist_ok=True)
    if path.exists() and specpath.exists():
        manifest = json.loads(specpath.read_text())
        loaded = []
        if manifest["protocol"] != PROTOCOL or manifest["seed"] != seed:
            raise ValueError("Wrong panel in cache")
        with np.load(path) as d:
            for j, row in enumerate(manifest["windows"]):
                e = {**row, **{k: d[k][j] for k in ("states", "actions", "grid")}}
                if (
                    hashlib.sha256(
                        b"".join(e[k].tobytes() for k in ("states", "actions", "grid"))
                    ).hexdigest()
                    != row["trajectory_sha256"]
                ):
                    raise ValueError("Cached source states or actions changed")
                loaded.append(e)
        if len(loaded) != N * len(KINDS) or manifest["panel_sha256"] != panel_fingerprint(loaded):
            raise ValueError("Incomplete or changed panel")
        return loaded, manifest
    eps = generate(seed)
    windows = [{k: v for k, v in row.items() if k not in ("states", "actions", "grid")} for row in eps]
    manifest = dict(protocol=PROTOCOL, seed=seed, windows=windows, panel_sha256=panel_fingerprint(eps))
    np.savez_compressed(path, **{k: np.stack([e[k] for e in eps]) for k in ("states", "actions", "grid")})
    atomic_json(specpath, manifest)
    return eps, manifest


def panel_fingerprint(eps):
    return hashlib.sha256(
        json.dumps([(e["kind"], e["id"], e["trajectory_sha256"]) for e in eps]).encode()
    ).hexdigest()


@torch.inference_mode()
def run(world, eps, fingerprint, out, device, batch):
    model, saved = load_world_checkpoint(world, "cpu")
    step = saved["training_steps"]
    del saved
    adapter = VJEPAImaginationAdapter(model.to(device))
    g = Game(endless=True, enemies_enabled=True)
    rows = []
    summary = {}
    with environment_profile(model, navigation_only=False):
        for kind in KINDS:
            subset = [e for e in eps if e["kind"] == kind]
            for i in range(0, len(subset), batch):
                group = subset[i : i + batch]

                def arr(k):
                    return torch.from_numpy(np.stack([e[k] for e in group])).to(device)

                s = arr("states")
                a = arr("actions")
                gmap = arr("grid").long()
                cur = adapter.initialize(s[:, :4], gmap, s.new_zeros(len(group), 4, 3, 32, 32))
                pt = []
                fallback_ticks = gmap.new_zeros(len(group))
                near_fallback_ticks = gmap.new_zeros(len(group))
                for tick in range(180):
                    if model.config.player_motion_mode == QUIET_MODE:
                        dt = 1 / model.config.simulation_hz
                        fallback = ~quiet_mask(cur.state, a[:, tick], dt)
                        # For this no-fire panel, distinguish the melee boundary from HP/timer gates.
                        distance = (cur.state[:, 8:10] - cur.state[:, :2]).norm(dim=-1) * MAP_EXTENT
                        fallback_ticks += fallback.long()
                        near_fallback_ticks += (fallback & (distance <= 1 + PLAYER_SPEED * dt)).long()
                    cur = adapter.transition_prior(cur, a[:, tick])
                    pt.append(cur.state)
                pred = torch.stack(pt, dim=1)
                truth = s[:, 4:]
                dis = {}
                if model.config.player_motion_mode == QUIET_MODE:
                    dis["quiet_fallback_ticks"] = fallback_ticks.tolist()
                    dis["fallback_within_melee_margin_ticks"] = near_fallback_ticks.tolist()
                for tag, sl, rad in [("player", slice(0, 2), 0.20), ("npc", slice(8, 10), 0.22)]:
                    err = ((pred[..., sl] - truth[..., sl]) * 16).norm(dim=-1)
                    for h in (16, 48, 112, 180):
                        dis[f"{tag}_H{h}"] = err[:, h - 1].tolist()
                    dis[f"{tag}_mean_path"] = err.mean(1).tolist()
                    dis[f"{tag}_wall_ticks"] = (
                        (circle_wall_penalty(pred[..., sl] * 16, gmap, rad) > 0).sum(1).tolist()
                    )
                dis["npc_wall_ticks_engine_radius"] = (
                    (circle_wall_penalty(pred[..., 8:10] * 16, gmap, Game.ENEMY_RADIUS) > 0).sum(1).tolist()
                )
                for j, e in enumerate(group):
                    z = pred[j].cpu().numpy()
                    sv = visibility(g, e["states"][4:])
                    pv = visibility(g, z)
                    rows.append(
                        dict(
                            kind=kind,
                            id=e["id"],
                            trajectory_sha256=e["trajectory_sha256"],
                            visibility_disagreement_ticks=int(np.count_nonzero(sv != pv)),
                            predicted_hp_change_ticks=int(
                                np.any(np.abs(z[:, [4, 12]] - e["states"][3, [4, 12]]) > 1e-6, axis=1).sum()
                            ),
                            **{k: v[j] for k, v in dis.items()},
                        )
                    )
            rr = [r for r in rows if r["kind"] == kind]
            summary[kind] = {
                k: float(np.mean([r[k] for r in rr]))
                for k in rr[0]
                if k not in ("kind", "id", "trajectory_sha256")
            }
            print(kind, summary[kind], flush=True)
    summary["all"] = {k: float(np.mean([r[k] for r in rows])) for k in summary[KINDS[0]]}
    atomic_json(
        out,
        dict(
            protocol=PROTOCOL,
            step=step,
            panel_sha256=fingerprint,
            now=now(),
            mode=model.config.player_motion_mode,
            frames=180,
            source_future_inputs=False,
            reference_combat_events=0,
            windows=len(rows),
            visibility="geometry-only NPC inside camera FOV and line of sight, not sprite detection",
            summary=summary,
            per_window=rows,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--world", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--batch", default=4, type=int)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    torch.set_num_threads(2)
    eps, spec = prepare(a.panel_dir, a.seed)
    print("Panel", spec["panel_sha256"], flush=True)
    if a.world:
        if not a.output:
            p.error("--output required")
        run(a.world, eps, spec["panel_sha256"], a.output, a.device, a.batch)
