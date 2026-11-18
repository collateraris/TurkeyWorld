"""Replay archived six pursuit cuts using the exact original controls, not a new tracking policy."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .models import load_world_checkpoint
from .engine import Game
from .record_vjepa_demos import source
from .rssm_combat_protocol import environment_profile
from .training_pipeline import atomic_json
from .trajectory_objectives import circle_wall_penalty
from .vjepa_policy_world import VJEPAImaginationAdapter


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--world", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    torch.set_num_threads(2)
    published = json.loads((a.capture / "pursuit.json").read_text())
    rows = json.loads((a.capture / "pursuit_candidates.json").read_text())
    selection = {r["scene"]["index"]: r["tape_sha256"] for r in published["selected"]}
    world, saved = load_world_checkpoint(a.world, "cpu")
    step = saved["training_steps"]
    del saved
    adapter = VJEPAImaginationAdapter(world.to(a.device))
    scored = []
    with environment_profile(world, navigation_only=False):
        for ix in range(0, len(rows), 6):
            group = rows[ix : ix + 6]
            if all(r["scene"]["index"] not in selection for r in group):
                continue
            games = [source(r["scene"]) for r in group]
            initial = torch.tensor(np.stack([g.state_vector() for g in games]), device=a.device)
            grid = torch.tensor(np.stack([g.map for g in games]), device=a.device)
            tapes = []
            labels = []
            for row in group:
                with np.load(a.capture / "capture_cache" / Path(row["cache"]).name) as d:
                    tape = d["actions"]
                    truth = d["truth"]
                    h = hashlib.sha256(tape.tobytes()).hexdigest()
                    if (
                        tape[:, 3].any()
                        or h != row["tape_sha256"]
                        or selection.get(row["scene"]["index"], h) != h
                    ):
                        raise ValueError("Changed action tape")
                    if not np.all(truth[:, 4] == truth[0, 4]) or not np.all(truth[:, 12] == truth[0, 12]):
                        raise ValueError("Nonquiet Source")
                    tapes.append(tape)
                    labels.append(truth)
            ctrl = torch.tensor(np.stack(tapes), device=a.device)
            target = torch.tensor(np.stack(labels), device=a.device)
            cur = adapter.initialize(initial, grid, initial.new_zeros(len(group), 3, 32, 32))
            states = []
            for t in range(180):
                cur = adapter.transition_prior(cur, ctrl[:, t])
                states.append(cur.state)
            pred = torch.stack(states, 1)
            st = {}
            for who, sl, rad in [("player", slice(0, 2), 0.20), ("npc", slice(8, 10), 0.22)]:
                error = ((pred[:, :, sl] - target[:, :, sl]) * 16).norm(dim=-1)
                st[who + "_H180"] = error[:, -1].tolist()
                st[who + "_mean_path"] = error.mean(1).tolist()
                st[who + "_wall_ticks"] = (
                    (circle_wall_penalty(pred[:, :, sl] * 16, grid, rad) > 0).sum(dim=1).tolist()
                )
            st["npc_wall_ticks_engine_radius"] = (
                (circle_wall_penalty(pred[..., 8:10] * 16, grid, Game.ENEMY_RADIUS) > 0).sum(1).tolist()
            )
            st["predicted_hp_change_ticks"] = (
                (pred[..., [4, 12]] - initial[:, None, [4, 12]]).abs().gt(1e-6).any(-1).sum(1).tolist()
            )
            for j, row in enumerate(group):
                if row["scene"]["index"] in selection:
                    scored.append(
                        dict(
                            index=row["scene"]["index"],
                            tape_sha256=row["tape_sha256"],
                            **{k: v[j] for k, v in st.items()}
                        )
                    )
    if len(scored) != 6:
        raise ValueError("Expected six distinct archived cuts")
    summary = {
        k: float(np.mean([s[k] for s in scored])) for k in scored[0] if k not in ("tape_sha256", "index")
    }
    atomic_json(
        a.output,
        dict(
            protocol="archived_quiet_tapes_v1",
            scope="six curated cuts; not independent VAL",
            step=step,
            H=180,
            reference_after_context=False,
            summary=summary,
            per_window=scored,
        ),
    )
    print(step, summary, flush=True)


if __name__ == "__main__":
    main()
