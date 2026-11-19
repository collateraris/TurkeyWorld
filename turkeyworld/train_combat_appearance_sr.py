"""Fit the combat-only SR branch on Base predictions from aligned TRAIN states."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .base_only_renderer import enable_base_only
from .distill_renderer import _save
from .enemy_animation import enemy_animation_index
from .models import load_sr_checkpoint, load_world_checkpoint
from .renderer_student import load_renderer_student
from .rssm_combat_protocol import environment_profile
from .small_enemy_data import digest
from .training_pipeline import atomic_json
from .sprite_scene import base_scene_state, set_sprite_scene
from .combat_appearance_sr import CombatAppearanceSR


def table(path):
    with np.load(path, allow_pickle=False) as f:
        return {
            k: f[k]
            for k in (
                "states",
                "maps",
                "rgb128",
                "enemy128",
                "weapon128",
                "flash128",
                "navigation",
                "group",
            )
        }


def command_indices(data, rng):
    state = torch.from_numpy(data["states"])
    phase = enemy_animation_index(state).numpy()
    parts = []
    for kind in (3, 4, 5, 6):
        ids = np.flatnonzero((~data["navigation"].astype(bool)) & (phase == kind))
        if len(ids):
            parts.extend(rng.choice(ids, min(len(ids), 384), replace=False))
    flash = np.flatnonzero((~data["navigation"].astype(bool)) & (phase < 3) & (data["states"][:, 7] > 0))
    if len(flash):
        parts.extend(rng.choice(flash, min(len(flash), 192), replace=False))
    return np.asarray(sorted(parts))


@torch.inference_mode()
def render_cache(data, ids, model, sr, device, batch_size, val=False):
    n = len(ids)
    cache = {k: [] for k in ("low", "prior", "final")}
    for start in range(0, n, batch_size):
        selected = ids[start : start + batch_size]
        state = torch.from_numpy(data["states"][selected]).float().to(device)
        maps = torch.from_numpy(data["maps"][selected]).long().to(device)
        nav = torch.as_tensor(data["navigation"][selected], dtype=torch.bool, device=device)
        low = torch.empty(len(nav), 3, 32, 32, device=device)
        prior = torch.empty(len(nav), 3, 128, 128, device=device)
        final = torch.empty_like(prior)
        for mode in (False, True):
            g = nav == mode
            if not g.any():
                continue
            with environment_profile(model, navigation_only=mode):
                fmap = model.encode_map(maps[g])
                background = base_scene_state(state[g], sr, navigation_only=mode)
                raw = model.render(background, state.new_zeros(int(g.sum()), 128), fmap)
            set_sprite_scene(sr, state[g], maps[g], navigation_only=mode)
            out = sr.forward_with_aux(raw)
            low[g], prior[g], final[g] = raw, out["scene_before_weapon"], out["image"]
        # Half precision cache; inference retains the parent precision. No rounding images to uint8.
        cache["low"].append(low.half().cpu())
        cache["prior"].append(prior.half().cpu())
        if val:
            cache["final"].append(final.half().cpu())
        if start % 256 == 0:
            print("cache", start, n, flush=True)
    return {k: torch.cat(v).float().to(device) for k, v in cache.items() if v}


def gather(t, idx, px):
    return t.permute(0, 2, 3, 1).reshape(len(t), -1, t.shape[1])[idx[:, None], px]


@torch.inference_mode()
def evaluate(data, cache, head, sr, device):
    point = {
        k: [0.0, 0.0, 0.0]
        for k in (
            "all",
            "enemy",
            "combat_enemy",
            "weapon",
            "flash",
            "nav",
            "quiet",
            "hurt",
            "death",
            "melee",
        )
    }
    before = cache["final"]
    for i in range(0, len(before), 4):
        sl = slice(i, i + 4)
        l = cache["low"][sl]
        p = cache["prior"][sl]
        s = torch.from_numpy(data["states"][sl]).float().to(device)
        nav = torch.from_numpy(data["navigation"][sl]).bool().to(device)
        own = before[sl].clone()
        if (~nav).any():
            k = ~nav
            base_logits = sr.weapon_memory.phase_head(l[k, :, 20:])
            w = sr.weapon_memory.compose(
                torch.where(s[k, 7:8] > 0, head.phase_logits(s[k]), base_logits)
            )
            rgb = head(l[k], p[k], s[k])
            # Entirely quiet rows use the exact parent image.
            active = (enemy_animation_index(s[k]) >= 3) | (s[k, 7] > 0)
            composed = rgb * (1 - w["alpha"]) + w["rgb"] * w["alpha"]
            own[k] = torch.where(active[:, None, None, None], composed, before[sl][k])
        truth = torch.from_numpy(data["rgb128"][sl]).permute(0, 3, 1, 2).to(device).float() / 255
        errors = [(v - truth).abs().mean(1) for v in (before[sl], own)]
        em = torch.from_numpy(data["enemy128"][sl]).bool().to(device)
        ps = enemy_animation_index(s)
        masks = {
            "all": torch.ones_like(em),
            "enemy": em,
            "combat_enemy": em & (ps >= 3)[:, None, None],
            "weapon": torch.from_numpy(data["weapon128"][sl]).bool().to(device),
            "flash": torch.from_numpy(data["flash128"][sl]).bool().to(device),
        }
        groups = data["group"][sl]
        for key in ("nav", "quiet", "hurt", "death", "melee"):
            mask = torch.as_tensor(groups == key, device=device)[:, None, None]
            masks[key] = mask & (em if key in ("hurt", "death", "melee") else torch.ones_like(em))
        for key, mask in masks.items():
            point[key][0] += mask.sum().item()
            for j, error in enumerate(errors):
                point[key][j + 1] += (error * mask).sum().item()
    return {
        k: {"pixels": int(n), "prior": a / max(n, 1), "candidate": b / max(n, 1)}
        for k, (n, a, b) in point.items()
    }


def fit(args):
    device = torch.device("cuda")
    torch.manual_seed(71083)
    rng = np.random.default_rng(71083)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / "sr.pt").exists():
        raise FileExistsError(out / "sr.pt")
    t0 = time.monotonic()
    model, meta = load_world_checkpoint(args.world, "cpu")
    del meta
    load_renderer_student(model, args.renderer, allow_unqualified_codec=True)
    enable_base_only(model)
    model.to(device).eval().requires_grad_(False)
    sr, checkpoint = load_sr_checkpoint(args.parent, "cpu")
    if sr.config["architecture"] != "rrdb_material_sprite_sr" or sr.combat_appearance is not None:
        raise ValueError("Need the original patrol/material RRDB parent")
    sr.to(device).eval().requires_grad_(False)
    train = table(args.data / "train.npz")
    val = table(args.data / "val.npz")
    ids = command_indices(train, rng)
    cache = render_cache(train, ids, model, sr, device, args.batch)
    valid_cache = render_cache(
        val, np.arange(len(val["states"])), model, sr, device, args.batch, val=True
    )
    train = {k: v[ids] for k, v in train.items()}
    del model
    head = CombatAppearanceSR().to(device)
    inherited = 0
    if args.initial_head:
        saved = torch.load(args.initial_head, map_location="cpu", weights_only=True)
        head.load_state_dict(saved["head"])
        inherited = int(saved["updates"])
    opt = torch.optim.AdamW(
        [
            {"params": head.decoder.parameters(), "lr": args.lr},
            {"params": head.weapon_phase.parameters(), "lr": 0.004},
        ],
        weight_decay=1e-5,
    )
    lr_up = F.interpolate(cache["low"], size=(128, 128), mode="bilinear", align_corners=False)
    target = torch.from_numpy(train["rgb128"]).permute(0, 3, 1, 2).to(device)
    state = torch.from_numpy(train["states"]).float().to(device)
    masks = train["enemy128"].astype(bool) & ~train["weapon128"].astype(bool)
    nonweapon = ~train["weapon128"].astype(bool)
    enemy = [np.flatnonzero(m.reshape(-1)) for m in masks]
    background = [np.flatnonzero(m.reshape(-1)) for m in nonweapon]
    phase = enemy_animation_index(state).cpu().numpy()
    cat = [
        np.flatnonzero((phase == j) & (masks.reshape(len(state), -1).sum(1) > 8)) for j in (3, 4, 5, 6)
    ]
    if any(len(c) == 0 for c in cat):
        raise ValueError("Missing active combat phase in TRAIN")
    flash = torch.nonzero(state[:, 7] > 0).flatten()
    if not len(flash):
        raise ValueError("No muzzle flashes in TRAIN")
    atomic_json(
        out / "plan.json",
        dict(
            parent=str(args.parent),
            parent_sha256=digest(args.parent),
            renderer=str(args.renderer),
            renderer_sha256=digest(args.renderer),
            train_sha256=digest(args.data / "train.npz"),
            val_sha256=digest(args.data / "val.npz"),
            optimizer="AdamW",
            lr=args.lr,
            steps=args.steps,
            distinct_train_rows=len(ids),
            decoder_trainable_parameters=sum(p.numel() for p in head.parameters()),
            combat_atlas=False,
            self_predicted_state_at_inference=True,
            train_only_updates=True,
        ),
    )
    for step in range(1, args.steps + 1):
        chosen = np.concatenate([rng.choice(pool, 4) for pool in cat])
        # Foreground and surrounding scene are both supervised. Adjacent samples
        # share a loss on their color difference to preserve thin contours.
        px = np.stack(
            [np.concatenate((rng.choice(enemy[i], 48), rng.choice(background[i], 48))) for i in chosen]
        )
        neighbor = px + np.where(px % 128 == 127, -1, 1)
        px = torch.as_tensor(np.concatenate((px, neighbor), axis=1), device=device)
        batch = torch.as_tensor(chosen, device=device)
        src = gather(cache["prior"], batch, px)
        rgb = gather(lr_up, batch, px)
        pred, logits, _ = head.decode(rgb, src, state[batch], px % 128, px // 128)
        y = gather(target, batch, px).float() / 255
        labels = torch.as_tensor(masks[chosen].reshape(len(chosen), -1), device=device).gather(1, px)
        use = torch.as_tensor(nonweapon[chosen].reshape(len(chosen), -1), device=device).gather(1, px)
        pixel = ((pred - y).abs().mean(-1) * use).sum() / use.sum().clamp_min(1)
        a, b = pred.chunk(2, dim=1)
        c, d = y.chunk(2, dim=1)
        wa, wb = use.chunk(2, dim=1)
        pair = wa & wb
        edge = (((b - a) - (d - c)).abs().mean(-1) * pair).sum() / pair.sum().clamp_min(1)
        auxiliary = F.binary_cross_entropy_with_logits(logits.float(), labels.float())
        fi = flash[torch.randint(len(flash), (160,), device=device)]
        phase_target = torch.where(state[fi, 7] > 0.45, 1, 2)
        gun = F.cross_entropy(head.phase_logits(state[fi]), phase_target)
        loss = pixel + 0.6 * edge + 0.015 * auxiliary + 0.4 * gun
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0, error_if_nonfinite=True)
        opt.step()
        if step % 200 == 0 or step == args.steps:
            print(
                "update", step, "MAE", float(pixel.detach()), "flash_ce", float(gun.detach()), flush=True
            )
    head.eval()
    metrics = evaluate(val, valid_cache, head, sr, device)
    torch.save({"head": head.cpu().state_dict(), "updates": inherited + args.steps}, out / "head.pt")
    parent_step = int(checkpoint["training_steps"])
    weights = dict(checkpoint["model"])
    for n, t in head.state_dict().items():
        weights["combat_appearance." + n] = t.detach().cpu()
    artifact = {
        **checkpoint,
        "model": weights,
        "config": {**sr.config, "combat_appearance": True},
        "training_steps": parent_step + inherited + args.steps,
        "combat_sr_training": {
            "own_updates": inherited + args.steps,
            "parent_step": parent_step,
            "renderer_sha256": digest(args.renderer),
            "combat_atlas": False,
            "optimizer": opt.state_dict(),
            "val": metrics,
            "initial_head_sha256": digest(args.initial_head) if args.initial_head else None,
        },
    }
    # Store a portable CPU optimizer with the new branch.
    artifact["combat_sr_training"]["optimizer"]["state"] = {
        i: {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in slot.items()}
        for i, slot in artifact["combat_sr_training"]["optimizer"]["state"].items()
    }
    _save(out / "sr.pt", artifact)
    check, _ = load_sr_checkpoint(out / "sr.pt", "cpu")
    assert check.config["combat_appearance"] and all(
        torch.equal(check.state_dict()[n], t) for n, t in checkpoint["model"].items()
    )
    atomic_json(
        out / "metrics.json",
        dict(
            step=artifact["training_steps"],
            elapsed=time.monotonic() - t0,
            val=metrics,
            old_weights_exact=True,
        ),
    )
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--initial-head", type=Path)
    fit(parser.parse_args())
