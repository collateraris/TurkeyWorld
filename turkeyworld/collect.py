"""Collect aligned, episode-separated ground-truth trajectories for world models."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import numpy as np

from .engine import ENTITY_SLOTS, BEHAVIOR_VERSION, Game
from .geometry_targets import build_geometry_targets


def visible(tilemap, start, end):
    delta = np.asarray(end) - np.asarray(start)
    distance = float(np.linalg.norm(delta))
    points = np.asarray(start)[None] + np.linspace(0, 1, max(2, int(distance / .06)))[:, None] * delta
    x, y = points[:, 0].astype(int), points[:, 1].astype(int)
    return bool(np.all(tilemap[y, x] == 0))


class MixedPolicy:
    """Correlated exploration plus aiming, misses, and close-range combat."""
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.held = np.zeros(4, dtype=np.float32)

    def __call__(self, game, index):
        state = game.state_vector()
        tilemap = np.asarray(game.map)
        height, width = tilemap.shape
        pos = state[:2] * (width, height)
        angle = math.atan2(float(state[2]), float(state[3]))
        entities = state[8:-2].reshape(ENTITY_SLOTS, 10)
        candidates = []
        for entity in entities:
            target = entity[:2] * (width, height)
            if entity[5] > .5 and visible(tilemap, pos, target):
                candidates.append((float(np.linalg.norm(target-pos)), target))
        # Long correlated segments expose simultaneous controls and idle/fire cases.
        if index % 24 == 0:
            self.held = np.array([self.rng.choice([-1., 0., 1.]),
                                  self.rng.choice([-1., 0., 0., 1.]),
                                  self.rng.uniform(-1, 1),
                                  float(self.rng.random() < .2)], dtype=np.float32)
        if candidates and self.rng.random() < .85:
            distance, target = min(candidates, key=lambda item: item[0])
            desired = math.atan2(target[1]-pos[1], target[0]-pos[0])
            error = (desired-angle+math.pi) % (2*math.pi)-math.pi
            movement = (1. if distance > 2.2 else -.4 if distance < 1.25 else 0.)
            return np.array([movement if abs(error) < .4 else 0.,
                             .25 * math.sin(index * .03), np.clip(error * 3., -1, 1),
                             float(abs(error) < .075 or self.rng.random() < .03)], dtype=np.float32)
        return self.held.copy()


class PatrolPolicy:
    """Reference-only navigation policy for long continuous motion examples.

    Its feedback is used while collecting training data, never to correct the
    deployed model. The outer corridor permits continuous travel without stops
    for camera rotation. NPC damage remains enabled and death ends the episode.
    """
    def __init__(self):
        self.waypoints = np.array([[14.25, 1.75], [14.25, 14.25],
                                   [1.75, 14.25], [1.75, 1.75]], dtype=np.float64)
        self.index = 0

    def __call__(self, game, index):
        p = game.player
        position = np.array([p["x"], p["y"]])
        delta = self.waypoints[self.index] - position
        if np.linalg.norm(delta) < .18:
            self.index = (self.index + 1) % len(self.waypoints)
            delta = self.waypoints[self.index] - position
        direction = delta / max(float(np.linalg.norm(delta)), 1e-8)
        c, s = math.cos(p["angle"]), math.sin(p["angle"])
        forward = direction[0] * c + direction[1] * s
        right = -direction[0] * s + direction[1] * c
        return np.array([forward, right, 0., 0.], dtype=np.float32)


def collect(destination, episodes=4, frames=300, size=256, hr_size=0, seed=0, hz=30,
            geometry=None, states_only=False, policy_name="mixed"):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if states_only and hr_size:
        raise ValueError("--states-only cannot be combined with --hr-size")
    if geometry is None:
        geometry = not states_only
    if hr_size and hr_size != size * 4:
        raise ValueError("--hr-size must be exactly 4 * --size; use 256/1024 for the target configuration")
    estimate = 0 if states_only else (frames+1) * (size*size + hr_size*hr_size) * 3
    if estimate > 512 * 1024**2:
        raise ValueError("One clip would exceed 512 MiB before compression. Use shorter clips, e.g. --frames 32 for 256/1024.")
    paths = []
    for episode in range(episodes):
        episode_seed = seed + episode
        game = Game(seed=episode_seed)
        if policy_name == "mixed":
            policy = MixedPolicy(episode_seed + 5000)
        elif policy_name == "patrol":
            policy = PatrolPolicy()
        else:
            raise ValueError("policy_name must be mixed or patrol")
        initial_state = json.dumps(game.get_state())
        # HUD text is composed by the application; it must not change the
        # physical scene or become an inconsistent LR/HR reconstruction target.
        observations = None if states_only else [game.render(size=size, overlays=False)]
        highres = [game.render(size=hr_size, overlays=False)] if hr_size else None
        states = [game.state_vector()]
        geometry_frames = [build_geometry_targets(game.map, states[0])] if geometry else None
        actions, events, hit_ids, attack_flags = [], [], [], []
        for index in range(frames):
            action = policy(game, index)
            event = game.step(action, dt=1/hz)
            # a_t belongs to precisely this s_t -> s_(t+1) transition.
            actions.append(action)
            events.append([float(bool(event.get(key, False))) for key in
                           ("shot", "hit", "kill", "player_hit")])
            hit_ids.append(int(event.get("hit_id", -1)))
            flags = np.zeros(ENTITY_SLOTS, dtype=np.float32)
            flags[event.get("attacker_ids", [])] = 1.
            attack_flags.append(flags)
            states.append(game.state_vector())
            if geometry_frames is not None:
                geometry_frames.append(build_geometry_targets(game.map, states[-1]))
            if observations is not None:
                observations.append(game.render(size=size, overlays=False))
            if highres is not None:
                highres.append(game.render(size=hr_size, overlays=False))
            if states[-1][4] <= 0:
                break  # Never concatenate a reset into a continuous training clip.
        payload = dict(map=np.asarray(game.map, dtype=np.int8),
                       states=np.asarray(states, dtype=np.float32),
                       actions=np.asarray(actions, dtype=np.float32),
                       events=np.asarray(events, dtype=np.float32),
                       hit_ids=np.asarray(hit_ids, dtype=np.int64),
                       attack_flags=np.asarray(attack_flags, dtype=np.float32),
                       initial_state=np.asarray(initial_state),
                       policy=np.asarray(policy_name), format_version=np.asarray(4),
                       behavior_version=np.asarray(BEHAVIOR_VERSION),
                       seed=np.asarray(episode_seed), dt=np.asarray(1/hz))
        if observations is not None:
            payload["frames"] = np.asarray(observations, dtype=np.uint8)
        if highres is not None:
            payload["hr_frames"] = np.asarray(highres, dtype=np.uint8)
        if geometry_frames is not None:
            payload["depth"] = np.asarray([g["depth"] for g in geometry_frames], dtype=np.float32)
            payload["visibility"] = np.asarray([g["visibility"] for g in geometry_frames], dtype=np.float32)
        path = destination / f"episode_{episode_seed:06d}.npz"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}; choose a new --seed or output directory")
        np.savez_compressed(path, **payload)
        paths.append(str(path))
        totals = np.asarray(events).sum(0).astype(int).tolist()
        print(json.dumps({"file": str(path), "transitions": len(actions),
                          "events_shot_hit_kill_player_hit": totals, "size_bytes": path.stat().st_size}))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--hr-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hz", type=int, default=30)
    geometry_flags = parser.add_mutually_exclusive_group()
    geometry_flags.add_argument("--no-geometry", dest="geometry", action="store_false",
                        help="Omit training-only wall-depth and object-center visibility labels")
    geometry_flags.add_argument("--geometry", dest="geometry", action="store_true",
                               help="Include geometry labels even for states-only collection")
    parser.set_defaults(geometry=None)
    parser.add_argument("--states-only", action="store_true", help="Store long state/action/event trajectories without RGB")
    parser.add_argument("--policy", choices=["mixed", "patrol"], default="mixed")
    args = parser.parse_args()
    if min(args.episodes, args.frames, args.size, args.hz) <= 0 or args.hr_size < 0:
        parser.error("Counts and sizes must be positive; --hr-size may be zero")
    collect(args.output, args.episodes, args.frames, args.size, args.hr_size, args.seed,
            args.hz, geometry=args.geometry, states_only=args.states_only,
            policy_name=args.policy)


if __name__ == "__main__":
    main()
