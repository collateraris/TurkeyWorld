"""Offline teacher fit for the quiet pursuit head. Only quiet TRAIN episodes/maps."""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .engine import Game
from .learned_quiet_pursuit import QuietPursuitHead
from .pilot_npc_direction import load
from .train_vjepa_sr_dynamics import ENEMY
from .training_pipeline import atomic_json


def vec_walkable(xy, walls, radius):
    cell = np.floor(xy).astype(int)
    valid = np.ones(len(cell), bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            c = cell + [dx, dy]
            dist = np.maximum(abs(xy - c - 0.5) - 0.5, 0.0)
            wall = walls[c[:, 1].clip(0, 15), c[:, 0].clip(0, 15)] | ((c < 0) | (c >= 16)).any(1)
            valid &= ~(wall & ((dist**2).sum(1) < radius**2))
    return valid


def visible(n, p, grid):
    # Offline labels: check continuous segment vs the interiors of TRAIN wall tiles.
    labels = np.ones(len(n), bool)
    delta = p - n
    for y, x in np.argwhere(grid != 0):
        with np.errstate(divide="ignore", invalid="ignore"):
            lo = (np.array([x, y]) - n) / delta
            hi = (np.array([x + 1, y + 1]) - n) / delta
        a = np.maximum(np.minimum(lo, hi).max(1), 0.0)
        b = np.minimum(np.maximum(lo, hi).min(1), 1.0)
        labels &= a >= b
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=12000)
    a = ap.parse_args()
    out = a.out
    out.mkdir(exist_ok=True, parents=True)
    if (out / "pursuit_head.pt").exists():
        raise ValueError("Head already exists")
    torch.set_num_threads(2)
    torch.manual_seed(2601)
    rng = np.random.default_rng(2601)
    head = QuietPursuitHead()
    g = Game.__new__(Game)
    files = sorted(ENEMY.glob("episode*.npz"))
    data = []
    grid0 = None
    for path in files:
        s, act, grid = load(path, "train")
        grid = grid.numpy()
        if grid0 is not None and not np.array_equal(grid0, grid):
            raise ValueError("This training receipt expects exactly one TRAIN map")
        grid0 = grid
        data.append(
            (
                s[:-1, 8:10].numpy() * 16,
                s[1:, :2].numpy() * 16,
                ((s[1:, 8:10] - s[:-1, 8:10]) * 480).norm(dim=-1).numpy(),
            )
        )
    g.map = grid0
    g.map_height, g.map_width = g.map.shape
    g._flow_cell = None
    grid = torch.tensor(g.map)
    walls = g.map != 0
    free = np.argwhere(~walls)[:, ::-1]
    n0, p0, speed = (np.concatenate([r[k] for r in data]).astype(np.float32) for k in range(3))
    # The typical unconstrained speed is fitted to real transitions, not the engine constant.
    observed_speed = float(np.median(speed[speed > 0.01]))
    nraw = free[rng.integers(len(free), size=100000)] + rng.random((100000, 2))
    accepted = vec_walkable(nraw, walls, Game.ENEMY_RADIUS)
    legal = nraw[accepted].astype(np.float32)
    n = np.concatenate([n0, legal[rng.integers(len(legal), size=70000)]])
    p = np.concatenate([p0, legal[rng.integers(len(legal), size=70000)]])
    # Exclude degenerate endpoint singularities of the vectorized TRAIN teacher.
    ok = abs(p - n).min(1) > 1e-6
    n = n[ok]
    p = p[ok]
    los = visible(n, p, g.map)
    ids = rng.choice(len(n), 400, replace=False)
    reference = np.array([g._line_clear(*n[i].astype(float), *p[i].astype(float)) for i in ids])
    assert np.array_equal(los[ids], reference)
    with torch.no_grad():
        feats = torch.cat(
            [
                head.visibility_features(
                    torch.tensor(n[i : i + 384]), torch.tensor(p[i : i + 384]), grid[None]
                )
                for i in range(0, len(n), 384)
            ]
        )
    los = torch.from_numpy(los.astype(np.float32))
    # Learn contact from raw free/blocked proposals near the TRAIN geometry.
    # The same geometric features and nine learned weights are evaluated by the head.
    # The labels are never used at inference.
    valid = vec_walkable(nraw, walls, Game.ENEMY_RADIUS)
    for i in rng.choice(len(nraw), 128, replace=False):
        assert valid[i] == g._walkable(*nraw[i], Game.ENEMY_RADIUS)
    xy = torch.tensor(nraw, dtype=torch.float32)
    free_ids = np.flatnonzero(valid)
    blocked_ids = np.flatnonzero(~valid)
    cv = torch.tensor(valid, dtype=torch.float32)
    # TRAIN-only global targets and tie preferences from reference routes.
    dist = []
    tgt = []
    largest = 0
    dr = head.dirs.numpy()
    for gx, gy in free:
        g.player = {"x": gx + 0.5, "y": gy + 0.5}
        field = g._navigation()
        largest = max(largest, int(field[field < 10000].max()))
        for nx, ny in free:
            moves = dr + [nx, ny]
            ds = field[moves[:, 1], moves[:, 0]].copy()
            ds[walls[moves[:, 1], moves[:, 0]]] = 10000
            label = min(range(4), key=lambda i: (ds[i], moves[i, 0], moves[i, 1]))
            tied = ds == ds.min()
            if tied.sum() > 1:
                dist.append(tied)
                tgt.append(label)
    tied = torch.tensor(np.stack(dist))
    label = torch.tensor(tgt)
    # Learn the move/stop boundary using synthetic TRAIN queries with reference NPC dynamics.
    # Use true Game.step and discard the first (previously initialized) velocity.
    g2 = Game(seed=991, endless=True, enemies_enabled=True)
    assert np.array_equal(g2.map, g.map)
    base = g2.get_state()
    stop_x = []
    stop_y = []
    for length in np.linspace(0.10, 2.0, 1601):
        # Empty TRAIN corridor to disambiguate stopping from a blocked axis.
        base["player"].update(x=2.5 + length, y=1.5)
        base["enemies"][0].update(x=2.5, y=1.5)
        g2.set_state(base)
        g2.step([0, 0, 0, 0])
        e = g2.enemies[0]
        stop_x.append(length)
        stop_y.append(int(np.hypot(e["vx"], e["vy"]) > 0.001))
    sx = torch.tensor(stop_x, dtype=torch.float32)[:, None]
    sy = torch.tensor(stop_y, dtype=torch.float32)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-5, weight_decay=1e-4)
    costs = grid.new_tensor([1, largest + 1], dtype=torch.float32)
    occ = torch.tensor([[0.0], [1.0]])
    for step in range(1, a.steps + 1):
        ids = rng.integers(len(n), size=384)
        vl = F.binary_cross_entropy_with_logits(head.visibility(feats[ids]).flatten() * 256, los[ids])
        ids = rng.integers(len(sx), size=256)
        sl = F.binary_cross_entropy_with_logits(head.stop_model(sx[ids]).flatten() * 256, sy[ids])
        clidx = np.r_[rng.choice(free_ids, 192), rng.choice(blocked_ids, 192)]
        cl = F.binary_cross_entropy_with_logits(
            head.contact._pass_logit(xy[clidx], grid.expand(384, -1, -1)), cv[clidx]
        )
        ec = F.smooth_l1_loss(F.softplus(head.edge(occ).flatten() * 512.0) + 0.01, costs)
        ids = rng.integers(len(label), size=256)
        rank = F.cross_entropy((head.rank * 128.0).expand(256, -1).masked_fill(~tied[ids], -1000), label[ids])
        sp = F.smooth_l1_loss(F.softplus(head.raw_speed * 128.0), torch.tensor(observed_speed), beta=0.01)
        opt.zero_grad()
        (cl + vl + sl + ec + rank + sp).backward()
        opt.step()
        if step % 1000 == 0:
            print(
                f"{step}/{a.steps} contact {cl.item():.3f} LOS {vl.item():.3f} stop {sl.item():.3f} route {ec.item():.3f} tie {rank.item():.3f}",
                flush=True,
            )
    head.eval()
    with torch.no_grad():
        v = head.visibility(feats).flatten() >= 0
        stop = head.stop_model(sx).flatten() >= 0
        rt = head.rank.float().tolist()
        map_costs = (F.softplus(head.edge(occ) * 512) + 0.01).flatten().tolist()
    receipt = dict(
        protocol="quiet_learned_pursuit_v1",
        steps=a.steps,
        lr=1e-5,
        seed=2601,
        training_uses_heldout=False,
        source_future_feedback=False,
        train_files=[p.name for p in files],
        train_transitions=len(speed),
        train_map_sha256=hashlib.sha256(walls.tobytes()).hexdigest(),
        distinct_train_maps=1,
        train_synthetic_visibility_queries=len(n) - len(n0),
        train_visibility_accuracy=float((v == los.bool()).float().mean()),
        contact_train_queries=len(xy),
        train_move_gate_accuracy=float((stop == sy.bool()).float().mean()),
        train_route_ties=len(tied),
        learned_traversal_costs=map_costs,
        learned_speed=float(F.softplus(head.raw_speed * 128).detach()),
        speed_target_from_train=observed_speed,
        learned_route_rank=rt,
        parameter_count=sum(p.numel() for p in head.parameters()),
        features="learned tile traversal cost, 32 global min-plus layers, learned rank, learned gate on static-map samples, distance stop, speed and learned XY contact",
        validation_used_for_tuning=False,
        train_origin=str(ENEMY),
    )
    torch.save(
        {"model": head.state_dict(), "optimizer": opt.state_dict(), "train_receipt": receipt},
        out / "pursuit_head.tmp.pt",
    )
    (out / "pursuit_head.tmp.pt").replace(out / "pursuit_head.pt")
    atomic_json(out / "pursuit_training.json", receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    main()
