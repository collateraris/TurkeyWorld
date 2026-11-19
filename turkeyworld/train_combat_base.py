"""Fit a Base-only combat RGB readout on TRAIN and track performance on independent episodes."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .base_only_renderer import enable_base_only
from .combat_base_renderer import CombatBaseReadout, combat_bundle
from .models import load_world_checkpoint
from .renderer_student import load_renderer_student
from .rssm_combat_protocol import environment_profile


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--world', type=Path, default=Path('training_runs/sr_base_only_20261116/assets/world.pt'))
    parser.add_argument('--renderer', type=Path, default=Path('training_runs/sr_base_only_20261116/assets/renderer.pt'))
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=.0005)
    parser.add_argument('--seed', type=int, default=4815)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def load_base(args, path):
    model, saved = load_world_checkpoint(args.world, 'cpu')
    del saved
    load_renderer_student(model, path, allow_unqualified_codec=True)
    enable_base_only(model)
    return model.eval().requires_grad_(False).to(args.device)


@torch.no_grad()
def collect(split, model, args):
    with np.load(args.data / (split + '.npz')) as data:
        group = data['group']
        state = torch.from_numpy(data['states']).to(args.device)
        maps = torch.from_numpy(data['maps']).to(args.device).long()
        target = torch.from_numpy(data['rgb32']).to(args.device).permute(0, 3, 1, 2).float().div_(255)
        masks = torch.stack([torch.from_numpy(data[name+'32']) for name in ('enemy', 'weapon', 'flash')], 1).to(args.device)
        nav = data['navigation'].astype(bool).reshape(-1)
    head = model.base_student
    q = state.new_empty(len(state), 256, head.config.width)
    logits = state.new_empty(len(state), 256, 12)
    for navigation in [False, True]:
        rows = np.where(nav == navigation)[0]
        with environment_profile(model, navigation_only=navigation):
            for batch in np.array_split(rows, max(1, (len(rows) + 23) // 24)):
                if len(batch) == 0:
                    continue
                i = torch.as_tensor(batch, device=state.device)
                geo = model.render_geometry(state[i], model.encode_map(maps[i]))
                q[i] = head._queries(state[i], state.new_zeros(len(i), 128), geo)
                logits[i] = head._rgb_logits(q[i], state[i])
    print('cache', split, len(group), flush=True)
    return dict(q=q, logits=logits, state=state, maps=maps, target=target, masks=masks, group=group,
                nav=torch.from_numpy(nav).to(args.device))


def draw(d, i, head, adapter):
    logits = d['logits'][i]
    if adapter is not None:
        combat = adapter(d['q'][i], d['state'][i], head.screen_coordinates, logits)
        logits = torch.where(d['nav'][i, None, None], logits, combat)
    return head._pixels(logits).sigmoid()


def mask_mean(error, mask):
    return (error * mask).sum() / mask.sum().clamp_min(1)


def loss(pred, target, masks):
    error = (pred - target).abs().mean(1)
    # Active regions often cover only a few RGB32 pixels. Supervise them regardless of image area.
    enemy, gun, flash = masks.unbind(1)
    grad = (F.l1_loss(pred.diff(dim=2), target.diff(dim=2)) + F.l1_loss(pred.diff(dim=3), target.diff(dim=3))) / 2
    return (.6 * error.mean() + .45 * mask_mean(error, enemy) + .20 * mask_mean(error, gun)
            + .22 * mask_mean(error, flash) + .12 * grad)


@torch.no_grad()
def evaluate(d, head, adapter, subset=None):
    scores = {}
    for name in np.unique(d['group']):
        rows = np.where(d['group'] == name)[0]
        if subset:
            rows = rows[:subset]
        all_err = []
        for batch in np.array_split(rows, max(1, (len(rows) + 31) // 32)):
            i = torch.as_tensor(batch, device=d['state'].device)
            all_err.append((draw(d, i, head, adapter) - d['target'][i]).abs().mean(1))
        error = torch.cat(all_err).contiguous()
        masks = d['masks'][rows]
        entry = {'frames': len(rows), 'frame_mae': float(error.mean())}
        for region, mask in zip(('enemy','weapon','flash'), masks.unbind(1)):
            pixels = int(mask.sum())
            entry[region+'_pixels'] = pixels
            entry[region+'_mae'] = float(mask_mean(error, mask)) if pixels else None
        scores[name] = entry
    return scores


def selection_metric(result):
    per_group = []
    for g in ('combat', 'melee', 'flash', 'hurt', 'death'):
        m = result[g]
        parts = [v for v in [m['frame_mae'], m['enemy_mae'], m['weapon_mae'], m['flash_mae']] if v is not None]
        per_group.append(sum(parts) / len(parts))
    return sum(per_group) / len(per_group)


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def main():
    args = get_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(3)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = load_base(args, args.renderer)
    print('Loaded immutable production Base', model.base_student_metadata.get('training_steps'), flush=True)
    head = model.base_student
    train = collect('train', model, args)
    val = collect('val', model, args)
    original = evaluate(val, head, None)
    adapter = CombatBaseReadout(head.config.width).to(args.device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=.002)
    combat_groups = ('flash', 'hurt', 'death', 'melee', 'combat')
    pools = [np.where(train['group'] == group)[0] for group in combat_groups]
    best = selection_metric(original)
    best_weights = {}
    best_step = 0
    history = []
    for step in range(1, args.steps + 1):
        sampled = [rng.choice(pool, size=args.batch // len(pools) + (j < args.batch % len(pools)), replace=True)
                   for j, pool in enumerate(pools)]
        i = torch.as_tensor(np.concatenate(sampled), device=args.device)
        optimizer.zero_grad(set_to_none=True)
        objective = loss(draw(train, i, head, adapter), train['target'][i], train['masks'][i])
        objective.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.)
        optimizer.step()
        if step % 100 == 0 or step == args.steps:
            current = evaluate(val, head, adapter)
            score = selection_metric(current)
            if score < best:
                best, best_step = score, step
                best_weights = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
            history.append(dict(step=step, validation_score=score, batch_loss=float(objective.detach())))
            print('updates', step, 'val_score', round(score, 6), 'best', best_step, flush=True)
    if not best_weights:
        raise RuntimeError('No held-out improvement over parent readout')
    adapter.load_state_dict(best_weights)
    final = evaluate(val, head, adapter)
    report = dict(parent_steps=model.base_student_metadata['training_steps'], updates_run=args.steps,
                  selected_step=best_step, adapter_parameters=sum(w.numel() for w in adapter.parameters()),
                  train_frames=len(train['group']), optimizer_pool_frames=sum(len(p) for p in pools),
                  optimizer_groups={g: len(p) for g, p in zip(combat_groups, pools)},
                  val_frames=len(val['group']), lr=args.lr,
                  batch=args.batch, history=history, baseline=original, trained=final,
                  train_sanity_baseline=evaluate(train, head, None, 64), train_sanity_final=evaluate(train, head, adapter, 64))
    parent = torch.load(args.renderer, map_location='cpu', weights_only=False)
    path = args.out / 'renderer.pt'
    tmp = path.with_suffix('.pt.tmp')
    torch.save(combat_bundle(parent, adapter, steps=best_step), tmp)
    del parent, train
    # Fresh production loader with frozen Base, zero features and the same environment profiles.
    fresh = load_base(args, tmp)
    checks = {}
    for group in np.unique(val['group']):
        i = torch.as_tensor(np.where(val['group'] == group)[0][:8], device=val['state'].device)
        with torch.no_grad(), environment_profile(fresh, navigation_only=bool(val['nav'][i[0]])):
            prediction = fresh.render(val['state'][i], val['state'].new_zeros(len(i), 128), fresh.encode_map(val['maps'][i]))
            expected = draw(val, i, head, adapter)
            checks[group] = float((expected - prediction).abs().max())
    if max(checks.values()) > 3e-4 or not fresh.base_only_enabled or hasattr(fresh.flow_renderer, 'codec'):
        raise RuntimeError(f'Deployed Base-only roundtrip check failed {checks}')
    report['runtime_max_abs_cache_vs_fresh'] = checks
    atomic_json(args.out / 'validation.json', report)
    os.replace(tmp, path)
    print('saved', path, 'selected', best_step, flush=True)


if __name__ == '__main__':
    main()
