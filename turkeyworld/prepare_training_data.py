"""Prepare resumable, seeded RGB clips and continuous state-only episodes.

Every transition is produced by the unmodified Game engine. Short clips use
randomized valid initial configurations; long recordings start at a normal
reset and stop on player death. No resets, health repairs or state corrections
are inserted into a recording. Use rgb/train and rgb/val with EpisodeWindows,
and long/train and long/val with the long-episode loader.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from . import engine as _engine
from .engine import ENTITY_SLOTS, EVENT_NAMES, Game
from .geometry_targets import build_geometry_targets

FORMAT_VERSION = 4
POLICIES = ("combat", "misses", "attack", "sweep", "respawn")


def current_behavior_version(expected=None):
    """Identify actual engine semantics; an optional CLI value is an assertion.

    The fallback identifies older engines explicitly. Missing episode tags
    are never treated as evidence of matching the running engine.
    """
    actual = getattr(_engine, "BEHAVIOR_VERSION", "legacy_random_v1")
    if not isinstance(actual, str) or not actual:
        raise ValueError("Engine BEHAVIOR_VERSION must be a nonempty string")
    if expected is not None and expected != actual:
        raise ValueError(f"Requested behavior version {expected!r} differs from running engine {actual!r}")
    return actual


def _require_episode_behavior(data, expected, path):
    actual = str(data["behavior_version"]) if "behavior_version" in data.files else None
    if actual != expected:
        raise ValueError(f"Episode behavior version differs from the running engine: {path} ({actual!r} != {expected!r})")
    return actual


def _wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _visible_targets(game):
    p = game.player
    targets = []
    for enemy in game.enemies:
        if enemy["alive"] and game._line_clear(p["x"], p["y"], enemy["x"], enemy["y"]):
            dx, dy = enemy["x"] - p["x"], enemy["y"] - p["y"]
            targets.append((math.hypot(dx, dy), math.atan2(dy, dx), enemy["id"]))
    return sorted(targets)


class ClipPolicy:
    """Seeded aiming, deliberate misses, attack exposure and camera exploration."""

    def __init__(self, seed, name, hz, frames):
        self.rng = np.random.default_rng(seed)
        self.name, self.dt, self.frames = name, 1 / hz, frames
        self.held = np.zeros(4, dtype=np.float32)
        self.miss_offset = float(self.rng.choice([-1., 1.]) * self.rng.uniform(.12, .35))

    def __call__(self, game, index):
        if self.name == "sweep":
            if index % 12 == 0:
                self.held = np.array([self.rng.uniform(-1, 1), self.rng.uniform(-1, 1),
                                      self.rng.choice([-1., -.5, .5, 1.]),
                                      float(self.rng.random() < .3)], dtype=np.float32)
            return self.held.copy()
        targets = _visible_targets(game)
        if not targets:
            return np.array([.2, .3 * math.sin(index * .05), .6, 0.], dtype=np.float32)
        distance, desired, _ = targets[0]
        if self.name == "misses" and index < self.frames // 2:
            desired += self.miss_offset
        error = _wrap(desired - game.player["angle"])
        turn = float(np.clip(error / (2.2 * self.dt), -1, 1))
        remaining_error = error - turn * 2.2 * self.dt
        fire = abs(remaining_error) < .06
        forward = -.5 if distance < 1.4 else .35 if distance > 3 else 0.
        strafe = .2 * math.sin(index * .13)
        if self.name == "attack" and index < self.frames // 2:
            forward = strafe = 0.
            fire = False
        if self.name == "respawn":
            forward = strafe = 0.
        return np.array([forward, strafe, turn, float(fire)], dtype=np.float32)


class SurvivalPolicy:
    """Run the outer corridor while aiming/shooting; ordinary damage stays on."""

    def __init__(self, seed, hz):
        self.dt = 1 / hz
        self.waypoints = np.array([[1.75, 1.75], [14.25, 1.75],
                                   [14.25, 14.25], [1.75, 14.25]])
        self.direction = 1 if seed % 2 else -1
        self.index = 0

    def __call__(self, game, index):
        p = game.player
        position = np.array([p["x"], p["y"]])
        delta = self.waypoints[self.index] - position
        if np.linalg.norm(delta) < .24:
            self.index = (self.index + self.direction) % len(self.waypoints)
            delta = self.waypoints[self.index] - position
        direction = delta / max(float(np.linalg.norm(delta)), 1e-8)
        c, s = math.cos(p["angle"]), math.sin(p["angle"])
        # Movement and aim are independent: strafe maintains the running route.
        targets = _visible_targets(game)
        desired = targets[0][1] if targets else math.atan2(delta[1], delta[0])
        error = _wrap(desired - p["angle"])
        turn = float(np.clip(error / (2.2 * self.dt), -1, 1))
        next_angle = p["angle"] + turn * 2.2 * self.dt
        nc, ns = math.cos(next_angle), math.sin(next_angle)
        forward = float(direction[0] * nc + direction[1] * ns)
        strafe = float(-direction[0] * ns + direction[1] * nc)
        fire = bool(targets and abs(error - turn * 2.2 * self.dt) < .035)
        return np.array([forward, strafe, turn, float(fire)], dtype=np.float32)


class MovementPolicy:
    """Held movement controls, turns and strafing without fabricated combat."""
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.action = np.zeros(4, np.float32)

    def __call__(self, game, index):
        if index % 12 == 0:
            self.action = np.array([self.rng.choice([-1., -.5, .5, 1.]),
                                    self.rng.uniform(-1, 1), self.rng.uniform(-1, 1), 0.], np.float32)
        return self.action.copy()


class ReturnMemoryPolicy:
    """Pure-CPU return-combat controls shared in spirit with the long probes."""
    def __init__(self, seed, hz):
        self.hz, self.sign = hz, 1. if seed % 2 else -1.
        self.survival = SurvivalPolicy(seed, hz)

    def __call__(self, game, index):
        tick = int(index * 30 / self.hz)
        if tick == 0:
            return np.array([0., 0., 0., 1.], np.float32)
        if tick <= 60 or 112 < tick <= 144:
            return np.array([1., 0., 0., 0.], np.float32)
        if tick <= 103 or 144 < tick <= 187:
            return np.array([0., 0., self.sign, 0.], np.float32)
        if tick <= 112:
            e, p = game.enemies[0], game.player
            error = _wrap(math.atan2(e["y"]-p["y"], e["x"]-p["x"])-p["angle"])
            turn = float(np.clip(error/(2.2/self.hz), -1, 1))
            return np.array([0., 0., turn, float(abs(error-turn*2.2/self.hz) < .035)], np.float32)
        if tick <= 204:
            return np.array([0., 0., 0., 1.], np.float32)
        return self.survival(game, index)


def _place_for_collision(game, rng):
    candidates = []
    for iy, ix in np.argwhere(game.map == 0):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if game.map[iy+dy, ix+dx]:
                x, y = float(ix)+.5+dx*.25, float(iy)+.5+dy*.25
                if game._walkable(x, y, game.PLAYER_RADIUS):
                    candidates.append((x, y, math.atan2(dy, dx)))
    if not candidates:
        raise ValueError("No valid wall-contact initialization")
    x, y, angle = candidates[int(rng.integers(len(candidates)))]
    state = game.get_state()
    state["player"].update(x=x, y=y, angle=angle)
    game.set_state(state)


def _perturb_reference_initial_state(game, rng, source_path=None):
    """TRAIN-ONLY valid-state augmentation, explicitly not model-visited DAgger."""
    state = game.get_state()
    if source_path:
        with np.load(source_path, allow_pickle=False) as saved:
            _require_episode_behavior(saved, current_behavior_version(), source_path)
            if "split" in saved.files and str(saved["split"]) != "train":
                raise ValueError("Reference-state augmentation must use training sources")
            state = json.loads(str(saved["initial_state"]))
        state["seed"] = game.seed
        state["rng_state"] = game.rng.bit_generator.state
    for entity, radius in [(state["player"], game.PLAYER_RADIUS)] + [(e, game.ENEMY_RADIUS) for e in state["enemies"]]:
        origin = np.array([entity["x"], entity["y"]])
        for _ in range(64):
            point = origin + rng.uniform(-.65, .65, 2)
            if game._walkable(*point, radius):
                entity.update(x=float(point[0]), y=float(point[1]))
                break
    state["player"].update(angle=float(rng.uniform(-math.pi, math.pi)), hp=100., ammo=99)
    for entity in state["enemies"]:
        entity["hp"] = float(rng.choice([0., 25., 60.], p=[.15, .35, .50]))
        entity["alive"] = entity["hp"] > 0
        entity["death_timer"] = float(rng.uniform(0, 2.9)) if not entity["alive"] else 0.
        entity["vx"] = entity["vy"] = 0.
    state["kills"] = sum(not entity["alive"] for entity in state["enemies"])
    game.set_state(state)


def _stagger_attack_initialization(game, rng):
    """Several genuine later attacks remain observable after context frames."""
    state = game.get_state()
    p = state["player"]
    placed = []
    for enemy, cooldown in zip(state["enemies"], (.13, .35, .57, .79)):
        for angle in rng.uniform(-math.pi, math.pi, 256):
            x, y = p["x"]+.86*math.cos(angle), p["y"]+.86*math.sin(angle)
            if (game._walkable(x, y, game.ENEMY_RADIUS) and game._line_clear(p["x"], p["y"], x, y)
                    and all(math.hypot(x-px, y-py) > .2 for px, py in placed)):
                enemy.update(x=x, y=y, hp=60., alive=True, attack_cooldown=cooldown)
                placed.append((x, y))
                break
    game.set_state(state)


def _random_initial_state(game, rng, policy):
    """Randomized initialization only; subsequent states always come from step."""
    state = game.get_state()
    floor = np.argwhere(game.map == 0)[:, ::-1].astype(np.float64) + .5
    player_xy = floor[int(rng.integers(len(floor)))] + rng.uniform(-.22, .22, 2)
    state["player"].update(x=float(player_xy[0]), y=float(player_xy[1]),
                           angle=float(rng.uniform(-math.pi, math.pi)))
    target_xy = None
    if policy == "attack":
        for angle in rng.uniform(-math.pi, math.pi, 64):
            candidate = player_xy + .86 * np.array([math.cos(angle), math.sin(angle)])
            if (game._walkable(*candidate, game.ENEMY_RADIUS)
                    and game._line_clear(*player_xy, *candidate)):
                target_xy = candidate
                break
    elif policy != "sweep":
        candidates = [xy for xy in floor if 2.0 < np.linalg.norm(xy - player_xy) < 5.5
                      and game._line_clear(*player_xy, *xy)]
        if candidates:
            target_xy = candidates[int(rng.integers(len(candidates)))]
    placed = []
    for ident, enemy in enumerate(state["enemies"]):
        candidates = [xy for xy in floor if np.linalg.norm(xy - player_xy) >= 2.0
                      and all(np.linalg.norm(xy - other) >= .75 for other in placed)]
        xy = (target_xy if ident == 0 and target_xy is not None
              else candidates[int(rng.integers(len(candidates)))])
        placed.append(xy)
        enemy.update(x=float(xy[0]), y=float(xy[1]))
    if target_xy is not None:
        delta = target_xy - player_xy
        offset = float(rng.uniform(-.18, .18)) if policy == "combat" else 0.
        state["player"]["angle"] = math.atan2(delta[1], delta[0]) + offset
    game.set_state(state)


def _atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Windows readers can briefly hold the destination without delete sharing.
    # Keep the last complete status visible until its replacement succeeds.
    deadline = time.monotonic() + 2.
    delay = .02
    while True:
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(.1, delay * 1.5)


def _episode_stats(path, job):
    """Inspect compact labels only; RGB is not decompressed when resuming."""
    with np.load(path, allow_pickle=False) as data:
        behavior_version = _require_episode_behavior(data, job.get("behavior_version", current_behavior_version()), path)
        span = float(data["rgb_time_span"]) if "rgb_time_span" in data.files else 0.
        if span != (float(job.get("rgb_time_span", 0.)) if job["kind"] == "rgb" else 0.):
            raise ValueError(f"Existing episode RGB time span differs from its job: {path}")
        states, actions, events = (data[key] for key in ("states", "actions", "events"))
        if (int(data["seed"]) != job["seed"] or int(data["format_version"]) != FORMAT_VERSION
                or states.shape != (len(actions) + 1, 90) or actions.shape != (len(actions), 4)
                or events.shape != (len(actions), 4) or len(actions) < 1):
            raise ValueError(f"Existing episode does not match its job: {path}")
        required = {"respawn_flags", "initial_state", "hit_ids", "attack_flags"}
        if job["kind"] == "rgb":
            required.update(("frames", "depth", "visibility"))
        if not required.issubset(data.files):
            raise ValueError(f"Incomplete episode: {path}")
        hits = int(events[:, 1].sum())
        positions = states[:, :2] * 16
        tiles = np.clip(positions.astype(int), 0, 15)
        bins = ((np.arctan2(states[:, 2], states[:, 3]) + math.pi) / (2 * math.pi) * 16).astype(int) % 16
        return {"job_id": job["job_id"], "path": job["relative_path"], "seed": job["seed"],
                "behavior_version": behavior_version,
                "rgb_time_span": span,
                "initial_game_time": float(data["initial_game_time"]) if "initial_game_time" in data.files else float(states[0, -1])*300,
                "split": job["split"], "kind": job["kind"], "policy": job["policy"],
                "transitions": len(actions), "requested_transitions": job["frames"],
                "simulated_seconds": len(actions) / job["hz"],
                "terminal": bool(states[-1, 4] <= 0),
                "completed_requested_length": len(actions) == job["frames"],
                "events": {name: int(events[:, i].sum()) for i, name in enumerate(EVENT_NAMES)},
                "missed_shots": int(events[:, 0].sum()) - hits,
                "attack_slots": int(data["attack_flags"].sum()),
                "respawns": int(data["respawn_flags"].sum()),
                "player_distance": float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
                "player_tiles": np.unique(tiles[:, 1] * 16 + tiles[:, 0]).tolist(),
                "heading_bins": np.unique(bins).tolist(), "size_bytes": path.stat().st_size,
                "blocked_movement_steps": int(((np.linalg.norm(data["actions"][:, :2], axis=1) > .5)
                    & (np.linalg.norm(np.diff(positions, axis=0), axis=1) < .5 * 2.4 * float(data["dt"]))
                    & (states[:-1, 4] > 0)).sum()),
                "initialization_source": str(data["initialization_source"]) if "initialization_source" in data.files else "source_engine",
                "model_visited": bool(data["model_visited"]) if "model_visited" in data.files else False,
                "hit_events_after_first_action": int(events[1:, 1].sum()),
                "attack_events_after_first_action": int(events[1:, 3].sum()),
                "moving_transitions": int((np.linalg.norm(np.diff(positions, axis=0), axis=1) > .001).sum())}


def _generate_job(job):
    job = dict(job, behavior_version=current_behavior_version(job.get("behavior_version")))
    path = Path(job["path"])
    if path.exists():
        result = _episode_stats(path, job)
        result["resumed"] = True
        return result
    started = time.perf_counter()
    game = Game(seed=job["seed"], endless=job["endless"])
    rng = np.random.default_rng(job["seed"] + 77 + int(job.get("initialization_attempt", 0))*997)
    if job["kind"] == "rgb":
        if job["policy"] == "return_memory":
            policy = ReturnMemoryPolicy(job["seed"], job["hz"])
        else:
            _random_initial_state(game, rng, job["policy"])
            if job["policy"] == "attack" and job.get("delayed_attack"):
                _stagger_attack_initialization(game, rng)
            if job["policy"] == "perturbed":
                if job["split"] != "train":
                    raise ValueError("Perturbed reference-state oracle clips are training-only")
                _perturb_reference_initial_state(game, rng, job.get("reference_path"))
            if job["policy"] == "collision":
                _place_for_collision(game, rng)
                policy = lambda current, index: np.array([1., .2*math.sin(index*.2), 0., 0.], np.float32)
            elif job["policy"] == "movement":
                policy = MovementPolicy(job["seed"]+5000)
            else:
                if job["policy"] == "kill":
                    game.enemies[0]["hp"] = 35.
                    game.player["cooldown"] = .15
                policy = ClipPolicy(job["seed"] + 5000, job["policy"], job["hz"], job["frames"])
    else:
        policy = SurvivalPolicy(job["seed"], job["hz"])
    rgb_time_span = float(job.get("rgb_time_span", 0.)) if job["kind"] == "rgb" else 0.
    if not math.isfinite(rgb_time_span) or rgb_time_span < 0:
        raise ValueError("RGB time span must be finite and nonnegative")
    initial_time_offset = 0.
    if rgb_time_span:
        initial_time_offset = int(rng.integers(math.ceil(rgb_time_span*job["hz"]))) / job["hz"]
        game.time = initial_time_offset  # Valid initial clock; no elapsed-play claim or in-clip resets.
    warmup = 0
    if job["policy"] == "return_memory":
        for index in range(int(job.get("return_offset", 96))):
            game.step(policy(game, index), dt=1/job["hz"])
            warmup += 1
            if game.player["hp"] <= 0:
                break
    # Make actual kills before sampling the moment a dead slot can respawn.
    # The initial state records the exact warmed engine state, without edits.
    if job["policy"] == "respawn" and game.endless:
        for index in range(math.ceil(3.5 * job["hz"])):
            game.step(policy(game, index), dt=1 / job["hz"])
            warmup += 1
            if any(not enemy["alive"] and enemy["death_timer"] >= 2.45 for enemy in game.enemies):
                break
            if game.player["hp"] <= 0:
                break
    if game.player["hp"] <= 0:
        raise RuntimeError(f"Warmup ended in death for {job['job_id']}; no clip was saved")
    initial_game_time = game.time
    initial_state = json.dumps(game.get_state(), ensure_ascii=False)
    capacity = job["frames"]
    states = np.empty((capacity + 1, 90), dtype=np.float32)
    actions = np.empty((capacity, 4), dtype=np.float32)
    events = np.empty((capacity, 4), dtype=np.float32)
    hits = np.empty(capacity, dtype=np.int64)
    attacks = np.zeros((capacity, ENTITY_SLOTS), dtype=np.float32)
    respawns = np.zeros((capacity, ENTITY_SLOTS), dtype=np.float32)
    rgb = job["kind"] == "rgb"
    frames = np.empty((capacity + 1, job["size"], job["size"], 3), dtype=np.uint8) if rgb else None
    highres = (np.empty((capacity + 1, job["hr_size"], job["hr_size"], 3), dtype=np.uint8)
               if rgb and job["hr_size"] else None)
    geometry = rgb or job["long_geometry"]
    depth = np.empty((capacity + 1, 32), dtype=np.float32) if geometry else None
    visibility = np.empty((capacity + 1, ENTITY_SLOTS), dtype=np.float32) if geometry else None

    def observe(index):
        states[index] = game.state_vector()
        if frames is not None:
            frames[index] = game.render(size=job["size"], overlays=False)
        if highres is not None:
            highres[index] = game.render(size=job["hr_size"], overlays=False)
        if geometry:
            labels = build_geometry_targets(game.map, states[index])
            depth[index], visibility[index] = labels["depth"], labels["visibility"]

    observe(0)
    count = 0
    for index in range(capacity):
        action = policy(game, index + warmup if job["policy"] == "return_memory" else index)
        before_alive = np.array([enemy["alive"] for enemy in game.enemies])
        event = game.step(action, dt=1 / job["hz"])
        actions[index] = action
        events[index] = [float(bool(event.get(name, False))) for name in EVENT_NAMES]
        hits[index] = event.get("hit_id", -1)
        attacks[index, event.get("attacker_ids", [])] = 1.
        respawns[index] = ~before_alive & np.array([enemy["alive"] for enemy in game.enemies])
        count = index + 1
        observe(count)
        if game.player["hp"] <= 0:
            break
    payload = {"format_version": np.asarray(FORMAT_VERSION),
               "behavior_version": np.asarray(job["behavior_version"]),
               "rgb_time_span": np.asarray(rgb_time_span), "initial_time_offset": np.asarray(initial_time_offset),
               "initial_game_time": np.asarray(initial_game_time),
               "map": game.map.astype(np.int8), "states": states[:count + 1],
               "actions": actions[:count], "events": events[:count],
               "hit_ids": hits[:count], "attack_flags": attacks[:count],
               "respawn_flags": respawns[:count], "initial_state": np.asarray(initial_state),
               "seed": np.asarray(job["seed"]), "dt": np.asarray(1 / job["hz"]),
               "policy": np.asarray(job["policy"]), "split": np.asarray(job["split"]),
               "endless": np.asarray(game.endless), "warmup_steps": np.asarray(warmup),
               "randomized_initialization": np.asarray(rgb and job["policy"] != "return_memory"),
               "initialization_source": np.asarray("perturbed_reference_state" if job["policy"] == "perturbed" else
                    "reference_return_warmup" if job["policy"] == "return_memory" else "source_engine"),
               "model_visited": np.asarray(False),
               "initialization_attempt": np.asarray(int(job.get("initialization_attempt", 0))),
               "requested_transitions": np.asarray(capacity)}
    if frames is not None:
        payload["frames"] = frames[:count + 1]
        payload["frame_source"] = np.asarray("reference_engine")
    if highres is not None:
        payload["hr_frames"] = highres[:count + 1]
    if geometry:
        payload.update(depth=depth[:count + 1], visibility=visibility[:count + 1])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    os.replace(temporary, path)
    result = _episode_stats(path, job)
    result.update(resumed=False, wall_seconds=time.perf_counter() - started)
    return result


def _jobs(output, config, counts):
    jobs = []
    for kind in ("rgb", "long"):
        for split in ("train", "val"):
            for index in range(counts[f"{kind}_{split}"]):
                # Disjoint ranges keep every seed in exactly one split/kind.
                band = (1 if split == "val" else 0) + (2 if kind == "long" else 0)
                seed = config["seed"] + band * 1_000_000 + index
                relative = f"{kind}/{split}/episode_{seed:010d}.npz"
                job = dict(config, seed=seed, kind=kind, split=split,
                           rgb_time_span=config.get("rgb_time_span", 0.) if kind == "rgb" else 0.,
                           frames=config["clip_frames"] if kind == "rgb" else config["long_frames"],
                           policy=POLICIES[index % len(POLICIES)] if kind == "rgb" else "survival",
                           relative_path=relative, path=str(output / relative),
                           job_id=f"{kind}/{split}/{index}")
                jobs.append(job)
    return jobs


def _summary(records, total, started, new_completed, errors):
    groups = {}
    for record in records.values():
        group = groups.setdefault(f"{record['kind']}/{record['split']}",
                                  {"files": 0, "transitions": 0, "size_bytes": 0,
                                   "completed_requested_length": 0, "terminal_episodes": 0,
                                   "respawns": 0, "missed_shots": 0,
                                   "events": {name: 0 for name in EVENT_NAMES},
                                   "player_tiles": set(), "heading_bins": set()})
        for key in ("transitions", "size_bytes", "respawns", "missed_shots", "completed_requested_length"):
            group[key] += record[key]
        group["files"] += 1
        group["terminal_episodes"] += int(record["terminal"])
        for name in EVENT_NAMES:
            group["events"][name] += record["events"][name]
        group["player_tiles"].update(record["player_tiles"])
        group["heading_bins"].update(record["heading_bins"])
    for group in groups.values():
        group["player_tiles"] = sorted(group["player_tiles"])
        group["heading_bins"] = sorted(group["heading_bins"])
    elapsed = time.perf_counter() - started
    remaining = max(0, total - len(records))
    return {"status": "complete" if not remaining and not errors else "running",
            "behavior_version": current_behavior_version(),
            "planned_files": total, "completed_files": len(records), "remaining_files": remaining,
            "elapsed_seconds": round(elapsed, 3),
            "eta_seconds": round(elapsed * remaining / new_completed, 1) if new_completed else None,
            "errors": errors, "groups": groups,
            "quality_claim": "Source recordings only; no learned-model accuracy or survival guarantee."}


def prepare(output, *, rgb_train=512, rgb_val=64, clip_frames=32, size=128,
            hr_size=0, long_train=16, long_val=4, long_frames=9000,
            seed=0, hz=30, workers=1, endless=True, long_geometry=False, behavior_version=None, rgb_time_span=0.):
    """Generate jobs independently; rerunning the same configuration resumes."""
    counts = dict(rgb_train=rgb_train, rgb_val=rgb_val, long_train=long_train, long_val=long_val)
    if any(value < 0 or value >= 1_000_000 for value in counts.values()) or not sum(counts.values()):
        raise ValueError("Episode counts must be nonnegative, below one million, and not all zero")
    if min(clip_frames, long_frames, workers) < 1 or not 4 <= hz <= 240:
        raise ValueError("Frames/workers must be positive and hz must be between 4 and 240")
    if not math.isfinite(rgb_time_span) or rgb_time_span < 0:
        raise ValueError("RGB time span must be finite and nonnegative")
    if not 32 <= size <= 2048 or hr_size not in (0, size * 4):
        raise ValueError("size must be between 32 and 2048; hr-size must be zero or 4*size")
    if (clip_frames + 1) * (size * size + hr_size * hr_size) * 3 > 256 * 1024**2:
        raise ValueError("One RGB job exceeds 256 MiB; reduce clip-frames or image sizes")
    behavior_version = current_behavior_version(behavior_version)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = dict(clip_frames=clip_frames, size=size, hr_size=hr_size, long_frames=long_frames,
                  seed=seed, hz=hz, endless=bool(endless), long_geometry=bool(long_geometry),
                  behavior_version=behavior_version, rgb_time_span=float(rgb_time_span))
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.get("config", {}).setdefault("rgb_time_span", 0.)
        if manifest.get("config") != config or manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError("Dataset configuration changed; choose another output directory")
    manifest = {"format_version": FORMAT_VERSION, "behavior_version": behavior_version, "config": config, "counts": counts,
                "split_rule": "Seed bands: rgb/train=base, rgb/val=base+1M, long/train=base+2M, long/val=base+3M",
                "rgb_initialization": "Random valid poses and optional seeded clock offset before warmup; engine-only transitions thereafter; clock offset is not elapsed gameplay",
                "long_initialization": "Normal engine reset; no resets/health changes within a recording",
                "paths": {"world_training": "rgb/train", "world_validation": "rgb/val",
                          "long_training": "long/train", "long_validation": "long/val"}}
    _atomic_json(manifest_path, manifest)
    jobs = _jobs(output, config, counts)
    records, pending = {}, []
    for job in jobs:
        path = Path(job["path"])
        if path.exists():
            records[job["job_id"]] = _episode_stats(path, job)
        else:
            pending.append(job)
    started = time.perf_counter()
    errors, new_completed = [], 0
    status_path, progress_path = output / "status.json", output / "progress.jsonl"
    _atomic_json(status_path, _summary(records, len(jobs), started, new_completed, errors))

    def completed(job, result=None, error=None):
        nonlocal new_completed
        if error is not None:
            entry = {"job_id": job["job_id"], "path": job["relative_path"], "error": repr(error)}
            errors.append(entry)
        else:
            records[job["job_id"]] = result
            new_completed += 1
            entry = result
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        summary = _summary(records, len(jobs), started, new_completed, errors)
        _atomic_json(status_path, summary)
        print(json.dumps({"completed": len(records), "total": len(jobs),
                          "job_id": job["job_id"], "eta_seconds": summary["eta_seconds"],
                          "error": entry.get("error"), "transitions": entry.get("transitions"),
                          "respawns": entry.get("respawns"), "events": entry.get("events")}), flush=True)

    if workers == 1:
        for job in pending:
            try:
                result = _generate_job(job)
            except Exception as exc:
                completed(job, error=exc)
            else:
                completed(job, result)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_generate_job, job): job for job in pending}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    completed(job, error=exc)
                else:
                    completed(job, result)
    summary = _summary(records, len(jobs), started, new_completed, errors)
    summary["status"] = "failed" if errors else "complete"
    _atomic_json(status_path, summary)
    if errors:
        raise RuntimeError(f"{len(errors)} dataset jobs failed; see {progress_path}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rgb-train", type=int, default=512)
    parser.add_argument("--rgb-val", type=int, default=64)
    parser.add_argument("--rgb-time-span", type=float, default=0., help="Seeded RGB-only initial game-time range in seconds; not recorded survival")
    parser.add_argument("--clip-frames", type=int, default=32)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--hr-size", type=int, default=0)
    parser.add_argument("--long-train", type=int, default=16)
    parser.add_argument("--long-val", type=int, default=4)
    parser.add_argument("--long-frames", type=int, default=9000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hz", type=int, default=30)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--behavior-version", help="Assert the running engine behavior version; never relabel another engine")
    parser.add_argument("--classic", action="store_true")
    parser.add_argument("--long-geometry", action="store_true")
    args = vars(parser.parse_args(argv))
    args["endless"] = not args.pop("classic")
    summary = prepare(**args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
