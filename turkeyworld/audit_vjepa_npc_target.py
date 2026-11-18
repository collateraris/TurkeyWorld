"""Diagnostic only: paired quiet H180 forecasts, with or without Source player XY."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .dreamer_policy_world import WorldState
from .engine import Game
from .jepa_training_utils import SequenceData
from .models import load_world_checkpoint
from .rssm_combat_protocol import environment_profile
from .small_enemy_data import digest
from .train_vjepa_sr_dynamics import ENEMY
from .training_pipeline import atomic_json, now
from .vjepa_policy_world import VJEPAImaginationAdapter
from .vjepa_sr_feedback import blend_memory, feedback_active
from .workspace_paths import ROOT


@torch.no_grad()
def paired_step(adapter, current, action, source_next_xy):
    """First half is production prior. Second uses only the matching post-tick player XY.

    Rebuild the newest state/map token and render feedback from the hybrid state,
    so that the next transition receives a consistent state and memory. The NPC
    at this tick is predicted BEFORE the intervention; no reference NPC is read.
    """
    n = len(source_next_xy)
    if source_next_xy.shape != (n, 2) or len(action) != 2 * n:
        raise ValueError('expected [ordinary, corrected] paired batch')
    action = adapter._action(current, action)
    with adapter._prior_mode():
        adapter._bind_map_context(current)
        world = adapter.world
        result = world.transition(current.state, action, current.map_features, current.memory)
    state = result['state'].clone()
    state[n:, :2] = source_next_xy
    backend = world.vjepa_backend
    core = backend.core
    token = core.pack_context(result['latent_prediction'][n:, None], torch.zeros_like(action[n:, None]),
                              state[n:, None])[:, 1]
    token = token + backend._map_context(current.map_features[n:], state[n:])
    memory = result['memory'].reshape(len(state), core.context_frames, core.block_tokens, core.width).clone()
    memory[n:, -1, 1] = token
    memory = memory.reshape_as(result['memory'])
    if adapter.feedback_renderer is not None:
        active = feedback_active(state, adapter.feedback_settings)
        if bool(active.any()):
            observation = adapter.feedback_renderer.observation(state, current.grid, navigation_only=world.config.navigation_only)
            memory = blend_memory(world, memory, observation, adapter.feedback_settings, active)
    following = WorldState(state.float(), memory.flatten(1).float(), current.map_features, current.grid)
    adapter._validate_state(following)
    return following


def source_masks(grid, initial, target):
    # These Source labels are computed after the rollout and never enter it.
    game = Game.__new__(Game)
    game.map = grid
    game.map_height, game.map_width = grid.shape
    path = np.vstack([initial, target])
    # Engine chooses the goal after moving player(t+1), before moving NPC(t+1).
    visible = np.array([game._line_clear(*(npc * 16), *(player * 16))
                        for npc, player in zip(path[:-1, 8:10], path[1:, :2])])
    velocity = np.diff(path[:, 8:10], axis=0) * 16
    speed = np.linalg.norm(velocity, axis=-1)
    dot = (velocity[1:] * velocity[:-1]).sum(axis=-1)
    turn = np.r_[False, (speed[1:] > 1e-4) & (speed[:-1] > 1e-4) &
                 (dot < np.cos(np.pi/6) * speed[1:] * speed[:-1])]
    post = np.convolve(turn.astype(int), np.ones(8, dtype=int), mode='full')[:len(turn)] > 0
    return {'visible': visible, 'occluded': ~visible, 'within_8_ticks_of_turn_ge30deg': post,
            'occluded_recent_turn': (~visible) & post, 'occluded_no_recent_turn': (~visible) & (~post)}


def summarize_rows(rows):
    def agg(entries):
        count = sum(e['ticks'] for e in entries)
        return {'ticks': count, **{k: sum(e[k]*e['ticks'] for e in entries if e['ticks'])/count if count else None
                                   for k in ('own_mean_tiles', 'true_player_mean_tiles',
                                             'own_velocity_error_tiles_s', 'true_player_velocity_error_tiles_s')}}
    return dict(windows=len(rows),
                H180={k: float(np.mean([r[k] for r in rows]))
                      for k in ('own_H180', 'true_player_H180', 'true_minus_own_H180')},
                improved_windows=sum(r['true_minus_own_H180'] < 0 for r in rows),
                offsets={str(t): {k: float(np.mean([r['offsets'][str(t)][k] for r in rows]))
                                  for k in ('own','true_player')} for t in (1,16,48,112,180)},
                source_groups={k: agg([r['source_groups'][k] for r in rows]) for k in rows[0]['source_groups']})


@torch.inference_mode()
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--world', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    p.add_argument('--batch-windows', type=int, default=2)
    p.add_argument('--cpu-threads', type=int, default=4)
    a = p.parse_args(argv)
    if a.batch_windows < 1: p.error('--batch-windows must be positive')
    torch.set_num_threads(a.cpu_threads)
    world, saved = load_world_checkpoint(a.world, 'cpu')
    step = saved['training_steps']
    del saved
    if not world.vjepa_sr_feedback: raise ValueError('Expected V-JEPA SR feedback')
    adapter = VJEPAImaginationAdapter(world.to(a.device))
    data = SequenceData(ENEMY.parent/'val', split='val', image_size=32, mode='stream', device='cpu')
    panel = json.loads((ROOT/'training_runs/rssm_moe_nav15_enemy25_cap20_20261113/rssm_validation_latest.json').read_text())['enemy_motion']['panel']['windows']
    index = {Path(e['path']).name: i for i,e in enumerate(data.episodes)}
    rows = []
    receipt = dict(protocol='paired_quiet_H180_player_xy_v1', checkpoint=str(a.world.resolve()), step=step,
                   horizon=180, total_windows=len(panel), generated_at=now(), diagnostic_only=True,
                   source_inputs_after_context={'own': [], 'true_player': ['player_x', 'player_y']},
                   source_RGB_used=False, source_NPC_future_input=False, feedback='frozen Base/SR on each branch state')
    with environment_profile(world, navigation_only=False):
        for start in range(0, len(panel), a.batch_windows):
            group = panel[start:start+a.batch_windows]
            n = len(group)
            batches, past, files = [], [], []
            for row in group:
                i, t = index[row['episode']], row['start']
                episode = data.episodes[i]
                batch = data.batch(1, context=1, horizon=180, indices=np.array([[i,t]]))
                past.append(torch.from_numpy(np.asarray(episode['states'][[max(0,j) for j in range(t-3,t+1)]])))
                batches.append(batch)
                files.append(dict(episode=row['episode'], start=t, sha256=digest(episode['path'])))
            state = torch.stack(past).to(a.device)
            grid = torch.cat([b['map'] for b in batches]).to(a.device)
            actions = torch.cat([b['actions'] for b in batches]).to(a.device)
            targets = torch.cat([b['target_states'] for b in batches]).to(a.device)
            if not torch.all(targets[...,13] > .5) or torch.any(actions[...,3] > .5):
                raise ValueError('Fixed panel no longer contains quiet, living NPCs only')
            duplicate = lambda x: torch.cat([x,x])
            current = adapter.initialize(duplicate(state), duplicate(grid), state.new_zeros(2*n,4,3,32,32))
            pts = []
            for tick in range(180):
                current = paired_step(adapter,current,duplicate(actions[:,tick]), targets[:,tick,:2])
                pts.append(current.state)
            predictions = torch.stack(pts, dim=1)
            if not torch.all(predictions[...,13] > .5): raise ValueError('NPC changed life during the paired forecast')
            errors = ((predictions[...,8:10]-duplicate(targets[...,8:10]))*16).norm(dim=-1).cpu().numpy()
            # Difference own consecutive NPC displacements against true consecutive
            # displacements, both from their own previous XY (no per-step resets).
            initial_npc = state[:, -1:,8:10]
            truth_vel = torch.diff(torch.cat([initial_npc,targets[...,8:10]],1),dim=1) * 480.
            pred_vel = torch.diff(torch.cat([duplicate(initial_npc), predictions[...,8:10]],1),dim=1) * 480.
            vel_err = (pred_vel - duplicate(truth_vel)).norm(dim=-1).cpu().numpy()
            for j,file in enumerate(files):
                own, corrected = errors[j], errors[n+j]
                groups = source_masks(grid[j].cpu().numpy(),state[j,-1].cpu().numpy(),targets[j].cpu().numpy())
                rows.append({**file, 'own_H180':float(own[-1]), 'true_player_H180':float(corrected[-1]),
                             'true_minus_own_H180':float(corrected[-1]-own[-1]),
                             'offsets':{str(t):dict(own=float(own[t-1]),true_player=float(corrected[t-1]))
                                        for t in (1,16,48,112,180)},
                             'source_groups':{k:dict(ticks=int(m.sum()),own_mean_tiles=float(own[m].mean()) if m.any() else None,
                                                     true_player_mean_tiles=float(corrected[m].mean()) if m.any() else None,
                                                     own_velocity_error_tiles_s=float(vel_err[j,m].mean()) if m.any() else None,
                                                     true_player_velocity_error_tiles_s=float(vel_err[n+j,m].mean()) if m.any() else None)
                                              for k,m in groups.items()}})
            partial = {**receipt, 'complete':len(rows)==len(panel),'per_window':rows,'summary':summarize_rows(rows)}
            atomic_json(a.output, partial)
            print(json.dumps(dict(step=step, windows=len(rows), H180=partial['summary']['H180'])),flush=True)


if __name__ == '__main__':
    main()
