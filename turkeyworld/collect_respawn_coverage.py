"""Collect independent one-tick TRAIN respawns with varied valid player positions and clocks."""
import argparse
from pathlib import Path
import numpy as np

from .engine import BEHAVIOR_VERSION, Game


def collect(path, seed):
    rng = np.random.default_rng(seed)
    game = Game(seed=seed, endless=True)
    grid = game.map.copy()
    positions = np.argwhere(grid == 0)
    states, targets = [], []
    action = np.zeros(4, dtype=np.float32)
    # Sample each floor cell and each clock interval; labels come only from
    # actually advancing the original game. The corpse does not move this tick.
    for y, x in positions:
        for phase in range(8):
            for offset in [(0., 0.), rng.uniform(-.24,.24,2), rng.uniform(-.24,.24,2)]:
                next_tick = 720 + phase * 90 + int(rng.integers(90))
                game.time = (next_tick - 1) / 30
                game.kills = 1
                game.player.update(x=float(x+.5+offset[0]), y=float(y+.5+offset[1]), hp=100.)
                enemy = game.enemies[0]
                enemy.update(alive=False, hp=0., vx=0., vy=0., death_timer=game.RESPAWN_DELAY-1/30)
                state = game.state_vector()
                game.step(action, dt=1/30)
                after = game.state_vector()
                if state[13] != 0. or after[13] != 1.:
                    raise RuntimeError('Expected a dead-to-alive transition')
                states.append(state)
                targets.append(after[8:10])
    path.parent.mkdir(exist_ok=True,parents=True)
    with path.open('xb') as out:
        np.savez_compressed(out, protocol='respawn_camera_coverage_v1', behavior_version=BEHAVIOR_VERSION,
                            split='train', seed=seed, states=np.stack(states), grid=grid, targets=np.stack(targets))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=731950)
    args = parser.parse_args()
    collect(args.out, args.seed)
