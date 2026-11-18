"""Refit the pursuit contact decision on randomized TRAIN wall boundaries."""
import argparse
from functools import partial
import hashlib
from pathlib import Path

import numpy as np
import torch

from .engine import Game
from .learned_axis_collision import head_digest
from .learned_quiet_pursuit import QuietPursuitHead
from .train_local_axis_collision import walkable
from .train_quiet_player_contact import bracket, features, fit_contact, sample_segments
from .train_vjepa_sr_dynamics import ENEMY
from .training_pipeline import atomic_json


def check_train_corners(maps, teacher, original, refined):
    # Independent seed and labels. Probe both sides of a fresh boundary and
    # exercise sequential X/Y proposals after an initially allowed pose.
    rng = np.random.default_rng(61007)
    ids, start, end = sample_segments(maps, [0], rng, 10000)
    center = bracket(maps, ids, start, end, teacher=teacher)
    keep = features(center, maps, ids)[:, 1] > 1e-8
    center, start, end = center[keep], start[keep], end[keep]
    direction = start - end
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    jitter = direction * rng.uniform(0.0002, 0.035, (len(center), 1))
    xy = np.concatenate([center + jitter, center - jitter]).astype(np.float32)
    map_tensor = torch.from_numpy(maps[0])
    game = Game.__new__(Game)
    game.map = maps[0]
    game.map_height, game.map_width = game.map.shape

    def labels(positions):
        return np.array([game._walkable(float(x), float(y), Game.ENEMY_RADIUS) for x, y in positions])

    def predicts(head, positions):
        coords = torch.from_numpy(positions).float()
        with torch.no_grad():
            return (head._pass_logit(coords, map_tensor.expand(len(coords), -1, -1)) >= 0).numpy()

    truth = labels(xy)
    if not np.array_equal(truth, teacher(maps, np.zeros(len(xy), np.int64), xy.astype(np.float64))):
        raise ValueError("Reference and vectorized NPC check differ")
    position = xy[truth]
    angle = rng.uniform(0, 2 * np.pi, len(position))
    delta = rng.uniform(0.01, 0.025, (len(position), 1)) * np.stack([np.cos(angle), np.sin(angle)], axis=1)
    after_x = position.copy()
    after_x[:, 0] += delta[:, 0]
    allow_x = labels(after_x)
    after_y = np.where(allow_x[:, None], after_x, position)
    after_y[:, 1] += delta[:, 1]
    allow_y = labels(after_y)
    escapes = (~allow_x) & allow_y
    if escapes.sum() == 0:
        raise ValueError("TRAIN probe missed single-axis releases")
    stats = {}
    for name, head in (("original", original), ("refined", refined)):
        pred = predicts(head, xy)
        pred_x = predicts(head, after_x)
        own_y = np.where(pred_x[:, None], after_x, position)
        own_y[:, 1] += delta[:, 1]
        pred_y = predicts(head, own_y)
        stats[name] = dict(
            false_stops=int((truth & ~pred).sum()),
            false_passes=int((~truth & pred).sum()),
            wrong_axis_steps=int(((pred_x != allow_x) | (pred_y != allow_y)).sum()),
            wrong_on_x_blocked_y_free=int(((pred_x | ~pred_y) & escapes).sum()),
        )
    return dict(
        seed=61007,
        corner_probes=len(xy),
        allowed=int(truth.sum()),
        blocked=int((~truth).sum()),
        axis_steps=len(position),
        x_blocked_y_free=int(escapes.sum()),
        **stats
    )


def refine(parent, out):
    out.mkdir(exist_ok=True, parents=True)
    destination = out / "pursuit_head.pt"
    if destination.exists() or destination.resolve() == parent.resolve():
        raise FileExistsError(destination)
    torch.set_num_threads(1)
    saved = torch.load(parent, weights_only=True, map_location="cpu")
    before = saved["train_receipt"]
    if before.get("training_uses_heldout") is not False or before.get("distinct_train_maps") != 1:
        raise ValueError("Expected a single-map TRAIN pursuit head")
    files = sorted(ENEMY.glob("episode*.npz"))
    if not files or [p.name for p in files] != before["train_files"]:
        raise ValueError("Parent TRAIN corpus changed")
    for path in files:
        with np.load(path, allow_pickle=False) as ep:
            grid = ep["map"]
            if (
                ep["split"].item() != "train"
                or ep["enemy_motion_protocol"].item() != "source_single_enemy_motion_no_shooting_v1"
                or hashlib.sha256((grid != 0).tobytes()).hexdigest() != before["train_map_sha256"]
            ):
                raise ValueError("Expected quiet TRAIN map of the parent")
    maps = grid[None]
    original = QuietPursuitHead()
    original.load_state_dict(saved["model"], strict=True)
    teacher = partial(walkable, radius=Game.ENEMY_RADIUS)
    seed, steps = 61006, 18000
    contact, metrics, optimizer = fit_contact(maps, [0], seed, steps, teacher=teacher)
    checks = check_train_corners(maps, teacher, original.contact, contact)
    old_hash = head_digest(original)
    old_contact = head_digest(original.contact)
    original.contact.load_state_dict(contact.state_dict())
    receipt = {
        **before,
        "parent_head_sha256": old_hash,
        "head_sha256": head_digest(original),
        "contact_refinement": dict(
            protocol="npc_bracketed_contact_train_v1",
            seed=seed,
            steps=steps,
            lr=1e-5,
            teacher="TRAIN Game NPC walkability",
            original_contact_sha256=old_contact,
            refined_contact_sha256=head_digest(contact),
            random_boundaries_only=True,
            contact_fit="fixed positive leading square scale; fitted second square and bias",
            training_uses_heldout=False,
            fresh_train_corner_checks=checks,
            **metrics
        ),
    }
    with destination.open("xb") as output:
        torch.save(
            dict(
                model=original.state_dict(),
                optimizer=optimizer,
                optimizer_scope="contact_boundary_fit_only",
                train_receipt=receipt,
            ),
            output,
        )
    atomic_json(out / "pursuit_training.json", receipt)
    return receipt["contact_refinement"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(refine(args.parent, args.out), flush=True)
