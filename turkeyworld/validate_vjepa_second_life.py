"""Isolate second-life dynamics on held-out windows starting just after the first kill."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .models import load_sr_checkpoint, load_world_checkpoint
from .rssm_combat_protocol import environment_profile
from .small_enemy_data import digest
from .training_pipeline import atomic_json
from .validate_vjepa_combat import CONTEXT, VAL
from .vjepa_policy_world import VJEPAImaginationAdapter


def read_panel():
    """Use the first 12 eligible episodes in seed order, independently of predictions."""
    rows = []
    for path in sorted(VAL.glob('episode_*.npz')):
        with np.load(path, allow_pickle=False) as data:
            kills = np.flatnonzero(data['events'][:, 2])
            if not len(kills):
                continue
            start = int(kills[0]) + 1
            horizon = 135
            if start < CONTEXT - 1 or start + horizon > len(data['actions']):
                continue
            if data['split'].item() != 'val' or not data['enemies_enabled'].item():
                raise ValueError(f'Unexpected data in {path}')
            states = data['states'][start - CONTEXT + 1:start + horizon + 1]
            if states[CONTEXT - 1, 13] >= .5:
                raise ValueError('Expected a confirmed first death')
            flags = data['respawn_flags'][start:start + horizon, 0]
            rows.append(dict(seed=int(data['seed']), case=str(data['case']), context_end=start,
                             states=states, actions=data['actions'][start:start + horizon],
                             grid=data['map'], respawns=flags))
            if len(rows) == 12:
                return rows
    raise ValueError('Too few held-out first-kill windows')


@torch.inference_mode()
def sprite_visible(states, grid, sprite):
    # Both arms use the same display checkpoint's native sprite and wall clipping.
    # Measure geometric visibility before the gun is overlaid.
    output = []
    for part in states.split(16):
        canvas = part.new_zeros(len(part), 3, 128, 128)
        scene = sprite(canvas, part, grid.expand(len(part), -1, -1))
        output.extend(scene['sprite_alpha'].flatten(1).amax(dim=1).gt(0).cpu().tolist())
    return np.asarray(output, bool)


def first_tick(mask):
    ticks = np.flatnonzero(mask)
    return int(ticks[0] + 1) if len(ticks) else None


@torch.inference_mode()
def evaluate(args):
    rows = read_panel()
    world, info = load_world_checkpoint(args.world, 'cpu')
    step = info['training_steps']
    del info
    sr, meta = load_sr_checkpoint(args.display_sr, 'cpu')
    if sr.config.get('source_sprite_compositor') is not True:
        raise ValueError('Supply the gameplay SR with embedded source sprites')
    sprite = sr.sprite_renderer.to(args.device).eval()
    del meta, sr
    world.to(args.device).eval()
    adapter = VJEPAImaginationAdapter(world)
    measurements = []
    with environment_profile(world, navigation_only=False):
        for start in range(0, len(rows), args.batch_size):
            group = rows[start:start + args.batch_size]
            reference = torch.as_tensor(np.stack([row['states'] for row in group]), device=args.device).float()
            controls = torch.as_tensor(np.stack([row['actions'] for row in group]), device=args.device).float()
            grid = torch.as_tensor(np.stack([row['grid'] for row in group]), device=args.device).long()
            current = adapter.initialize(reference[:, :CONTEXT], grid, reference.new_zeros(len(group), CONTEXT, 3, 32, 32))
            predicted, flags = [], []
            for command in controls.unbind(1):
                current, info = adapter.transition_with_info(current, command)
                predicted.append(current.state)
                flags.append(info['respawn_flags'][:, 0])
            predicted = torch.stack(predicted, dim=1)
            flags = torch.stack(flags, dim=1).cpu().numpy().astype(bool)
            for i, row in enumerate(group):
                source_tick, predicted_tick = first_tick(row['respawns']), first_tick(flags[i])
                truth = reference[i, CONTEXT:]
                own = predicted[i]
                truth_vis = sprite_visible(truth, grid[i], sprite)
                own_vis = sprite_visible(own, grid[i], sprite)
                truth_new = row['respawns'].cumsum() == 1
                own_new = flags[i].cumsum() == 1
                truth_vis &= truth_new & (truth[:, 13].cpu().numpy() >= .5)
                own_vis &= own_new & (own[:, 13].cpu().numpy() >= .5)
                record = {key: row[key] for key in ('seed', 'case', 'context_end')}
                record.update(reference_respawn_tick=source_tick, predicted_respawn_tick=predicted_tick,
                              reference_first_visible=first_tick(truth_vis), predicted_first_visible=first_tick(own_vis),
                              reference_visible_ticks=int(truth_vis.sum()),
                              missed_visible_ticks=int((truth_vis & ~own_vis).sum()))
                if source_tick and predicted_tick:
                    ref_xy = truth[source_tick-1, 8:10] * 16
                    pred_xy = own[predicted_tick-1, 8:10] * 16
                    record.update(reference_spawn_xy=ref_xy.tolist(), predicted_spawn_xy=pred_xy.tolist(),
                                  spawn_error_tiles=float(torch.linalg.vector_norm(ref_xy-pred_xy)),
                                  same_cell=bool(torch.equal(ref_xy.floor(), pred_xy.floor())))
                measurements.append(record)
    comparable = [r for r in measurements if 'same_cell' in r]
    exact = [r for r in comparable if r['reference_respawn_tick'] == r['predicted_respawn_tick']]
    return dict(step=step, horizon=135, context=CONTEXT, display_sr_sha256=digest(args.display_sr),
                protocol='first_kill_context_end_fixed_VAL_v1', source_future_feedback=False,
                visibility='sprite alpha after wall clipping before weapon',
                windows=len(measurements), comparable=len(comparable), exact_tick=len(exact),
                same_spawn_cell=sum(r['same_cell'] for r in comparable),
                mean_spawn_error_tiles=float(np.mean([r['spawn_error_tiles'] for r in comparable])) if comparable else None,
                reference_visible_ticks=sum(r['reference_visible_ticks'] for r in measurements),
                missed_visible_ticks=sum(r['missed_visible_ticks'] for r in measurements), details=measurements)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world', required=True, type=Path)
    parser.add_argument('--display-sr', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=3, choices=range(1, 13))
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = evaluate(args)
    atomic_json(args.output, report)
    print({k:v for k,v in report.items() if k != 'details'})
