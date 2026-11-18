"""Fit a local wall contact classifier to free/blocked probes on TRAIN maps."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .learned_axis_collision import LocalAxisCollisionHead, head_digest
from .train_local_axis_collision import read_train, walkable
from .training_pipeline import atomic_json


def bracket(maps, ids, start, end, iterations=24, *, teacher=walkable):
    """Locate a class boundary along each free-to-blocked segment by bisection."""
    free, blocked = start.copy(), end.copy()
    if not teacher(maps, ids, free).all() or teacher(maps, ids, blocked).any():
        raise ValueError("Expected a free start and a blocked endpoint for each segment")
    for _ in range(iterations):
        midpoint = (free + blocked) * 0.5
        allowed = teacher(maps, ids, midpoint)
        free[allowed] = midpoint[allowed]
        blocked[~allowed] = midpoint[~allowed]
    return (free + blocked) * 0.5


def features(xy, maps, ids):
    """Two squared tile-edge distances used by LocalAxisCollisionHead."""
    offsets = np.array([[x, y] for y in (-1, 0, 1) for x in (-1, 0, 1)])
    cells = np.floor(xy).astype(np.int32)[:, None, :] + offsets
    edge_distance = np.maximum(np.abs(xy[:, None, :] - cells - 0.5) - 0.5, 0)
    x, y = cells.transpose(2, 0, 1)
    wall = maps[ids[:, None], y.clip(0, 15), x.clip(0, 15)] != 0
    wall |= (x < 0) | (x >= 16) | (y < 0) | (y >= 16)
    if not wall.any(axis=1).all():
        raise ValueError("Boundary probe has no occupied neighbor")
    nearest = np.argmin(np.where(wall, (edge_distance**2).sum(-1), np.inf), axis=1)
    distances = edge_distance[np.arange(len(xy)), nearest]
    return (4 * np.sort(distances, axis=1)[:, ::-1]) ** 2


def sample_segments(maps, map_ids, rng, count):
    ids = rng.choice(map_ids, count)
    start = np.zeros((count, 2))
    end = start.copy()
    for index, grid in enumerate(maps):
        mask = ids == index
        n = mask.sum()
        occupied = np.argwhere(grid != 0)[:, ::-1]
        free = np.argwhere(grid == 0)[:, ::-1]
        start[mask] = free[rng.integers(len(free), size=n)] + rng.uniform(0.25, 0.75, (n, 2))
        end[mask] = occupied[rng.integers(len(occupied), size=n)] + rng.uniform(0.05, 0.95, (n, 2))
    return ids, start, end


def fit_contact(maps, map_ids, seed, steps, *, teacher=walkable, count=48000):
    rng = np.random.default_rng(seed)
    ids, start, end = sample_segments(maps, map_ids, rng, count)
    distances = features(bracket(maps, ids, start, end, teacher=teacher), maps, ids)
    face = distances[:, 1] < 1e-8
    if min(face.sum(), (~face).sum()) < 100:
        raise ValueError("Missing TRAIN face or corner coverage")
    weights = np.where(face, 0.5 / face.sum(), 0.5 / (~face).sum())
    # A common positive scale has no effect on the pass decision. Fix one
    # coefficient and fit the other coefficient plus the intercept.
    scale = 0.05
    design = np.column_stack((distances[:, 1], np.ones(len(distances))))
    gram = torch.tensor(design.T @ (design * weights[:, None]), dtype=torch.float64)
    linear = torch.tensor(design.T @ (weights * scale * distances[:, 0]), dtype=torch.float64)
    parameters = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    optimizer = torch.optim.Adam([parameters], lr=1e-5)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = parameters @ gram @ parameters + 2 * parameters @ linear
        loss.backward()
        optimizer.step()

    corner_weight, bias = parameters.detach().tolist()
    head = LocalAxisCollisionHead()
    with torch.no_grad():
        head.score.weight.zero_()
        head.score.weight[0, 4] = scale
        head.score.weight[0, 5] = corner_weight
        head.score.bias.fill_(bias)
    residual = scale * distances[:, 0] + corner_weight * distances[:, 1] + bias
    metrics = dict(
        brackets=count,
        corner_brackets=int((~face).sum()),
        boundary_rms=float(np.sqrt(np.mean(residual**2))),
    )
    return head, metrics, optimizer.state_dict()


def train(directory, seed=208, steps=18000):
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "quiet_player_head.pt"
    if destination.exists():
        raise FileExistsError(destination)
    torch.set_num_threads(1)
    (_, _, map_ids, _), maps, files = read_train()
    head, metrics, optimizer = fit_contact(maps, map_ids, seed, steps)
    receipt = dict(
        protocol="quiet_player_bracketed_contact_train_v1",
        training_uses_heldout=False,
        source_future_feedback=False,
        training_scope="TRAIN map label probes only; source collision code absent at inference",
        source_files=files,
        maps=len(maps),
        **metrics,
        seed=seed,
        steps=steps,
        lr=1e-5,
        fit="positive leading squared offset fixes logit scale; second offset coefficient and intercept fitted with Adam",
        head_sha256=head_digest(head),
    )
    with destination.open("xb") as output:
        torch.save(dict(model=head.state_dict(), train_receipt=receipt, optimizer=optimizer), output)
    atomic_json(directory / "quiet_player_training.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = train(args.out)
    print({k: v for k, v in result.items() if k != "source_files"})
