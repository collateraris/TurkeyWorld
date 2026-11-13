"""Deterministic, headless-friendly ground-truth raycaster shooter.

This is the DATA SOURCE / classical gameplay reference, not neural inference.
Assets are loaded once. A generated 4x2 turkey atlas and 3x1 gun atlas are
optional: without them a deliberately simple programmatic fallback is used.
"""

from __future__ import annotations

import copy
import math
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .spawn import BEHAVIOR_VERSION, SPAWN_TICK_HZ, deterministic_spawn_tile
from .simulation_clock import advance_timer_seconds

GLOBAL_DIM = 8
ENTITY_DIM = 10
ENTITY_SLOTS = 1
STATE_DIM = GLOBAL_DIM + ENTITY_DIM * ENTITY_SLOTS + 2
ACTION_DIM = 4
EVENT_DIM = 4
EVENT_NAMES = ("shot", "hit", "kill", "player_hit")
STATE_FIELDS = {
    "global": ("player_x/map_width", "player_y/map_height", "sin_angle", "cos_angle",
               "hp/100", "ammo/99", "cooldown/0.22", "shot_flash/0.10"),
    "entity": ("x/map_width", "y/map_height", "vx/2", "vy/2", "hp/100", "alive",
               "attack_cooldown/1.2", "hurt_timer/0.15", "death_timer/0.6", "animation_phase"),
    "tail": ("kills", "time/300"),
}

_ROWS = (
    "1111111111111111",
    "1000000000000001",
    "1000000000000001",
    "1000222000110001",
    "1000200000100001",
    "1000200000100001",
    "1000000000000001",
    "1011100022201101",
    "1000000020000001",
    "1000000000000001",
    "1002200000110001",
    "1000200000100001",
    "1000000000000001",
    "1000000110000001",
    "1000000000000001",
    "1111111111111111",
)


class Game:
    """One persistent enemy, wall-blocked hitscan and optional endless combat.

    The default finite mode keeps permanent corpses for model training.
    Endless play reuses dead slots and supplies unlimited ammunition.

    Action = [forward, strafe_right, turn_right, fire], with movement/turn
    in [-1,1], fire in [0,1]. Positive angle rotates clockwise on screen.
    step() returns a dictionary containing four boolean labels named by
    EVENT_NAMES and additional diagnostic fields. get_state() is JSON-safe.
    """

    STATE_DIM = STATE_DIM
    BEHAVIOR_VERSION = BEHAVIOR_VERSION
    ENTITY_SLOTS = ENTITY_SLOTS
    FIRE_INTERVAL = 0.22
    FLASH_DURATION = 0.10
    HURT_DURATION = 0.15
    DEATH_DURATION = 0.60
    ATTACK_INTERVAL = 1.20
    PLAYER_RADIUS = 0.20
    ENEMY_RADIUS = 0.25
    RESPAWN_DELAY = 3.0
    SPAWN_DISTANCE = 4.0

    def __init__(self, seed: int = 0, assets_dir: str | Path | None = None,
                 *, endless: bool = False, enemies_enabled: bool = True):
        self.endless = bool(endless)
        self.enemies_enabled = bool(enemies_enabled)
        self.map = np.array([[int(c) for c in row] for row in _ROWS], dtype=np.uint8)
        self.map_height, self.map_width = self.map.shape
        self.assets_dir = Path(assets_dir) if assets_dir else Path(__file__).resolve().parents[1] / "assets"
        self._textures = self._make_textures()
        self._turkeys = self._load_atlas("turkey_atlas.png", 4, 2) or self._fallback_turkeys()
        self._guns = self._load_atlas("gun_atlas.png", 3, 1) or self._fallback_guns()
        self._flow_cell = None
        self._flow = None
        self.reset(seed)

    def reset(self, seed: int | None = None):
        self.seed = int(seed if seed is not None else getattr(self, "seed", 0))
        self.rng = np.random.default_rng(self.seed)
        self.player = {"x": 2.5, "y": 2.5, "angle": 0.0, "hp": 100.0,
                       "ammo": 99, "cooldown": 0.0, "shot_flash": 0.0}
        spawns = [(5.5, 2.5)]
        self.enemies = []
        for ident, (x, y) in enumerate(spawns):
            self.enemies.append(self._new_enemy(ident, x, y))
        if not self.enemies_enabled:
            for enemy in self.enemies:
                enemy.update(hp=0., alive=False, death_timer=self.DEATH_DURATION)
        self.kills = 0
        self.time = 0.0
        self.last_events = {name: False for name in EVENT_NAMES}
        self._flow_cell = None
        return self.get_state()

    @property
    def done(self):
        return self.player["hp"] <= 0 or (not self.endless and self.kills >= ENTITY_SLOTS)

    def _new_enemy(self, ident, x, y):
        return {"id": ident, "x": x, "y": y, "vx": 0.0, "vy": 0.0,
                "hp": 60.0, "alive": True, "attack_cooldown": 0.0,
                "hurt_timer": 0.0, "death_timer": 0.0,
                "animation_phase": 0.0}

    def get_state(self):
        result = copy.deepcopy({"version": 1, "seed": self.seed, "time": self.time,
                              "behavior_version": BEHAVIOR_VERSION,
                              "endless": self.endless,
                              "player": self.player, "enemies": self.enemies,
                              "kills": self.kills, "rng_state": self.rng.bit_generator.state})
        if not self.enemies_enabled:
            result['enemies_enabled'] = False
        return result

    def set_state(self, state):
        """Restore an exact valid state; invalid/inside-wall states raise ValueError."""
        candidate = copy.deepcopy(state)
        enemies_enabled = bool(candidate.get('enemies_enabled', True))
        if not enemies_enabled and any(e['alive'] or e['hp'] > 0 for e in candidate['enemies']):
            raise ValueError('Navigation profile requires inactive enemy slots')
        if len(candidate.get("enemies", [])) != ENTITY_SLOTS:
            raise ValueError(f"Expected exactly {ENTITY_SLOTS} persistent enemy slots")
        if [e.get("id") for e in candidate["enemies"]] != list(range(ENTITY_SLOTS)):
            raise ValueError("Enemy IDs must remain ordered 0..7")
        for entity, radius in [(candidate["player"], self.PLAYER_RADIUS)] + [
                (e, self.ENEMY_RADIUS) for e in candidate["enemies"]]:
            if not math.isfinite(float(entity["x"])) or not math.isfinite(float(entity["y"])):
                raise ValueError("Coordinates must be finite")
            if not self._walkable(float(entity["x"]), float(entity["y"]), radius):
                raise ValueError("Cannot restore an entity inside a wall")
        values = self._vector_from(candidate["player"], candidate["enemies"],
                                   candidate["kills"], candidate["time"])
        if not np.isfinite(values).all():
            raise ValueError("State contains non-finite numbers")
        self.seed = int(candidate.get("seed", self.seed))
        self.endless = bool(candidate.get("endless", False))
        self.enemies_enabled = enemies_enabled
        self.time = float(candidate["time"])
        self.player = candidate["player"]
        self.enemies = candidate["enemies"]
        self.kills = int(candidate["kills"])
        self.rng = np.random.default_rng(self.seed)
        if "rng_state" in candidate:
            self.rng.bit_generator.state = candidate["rng_state"]
        self._flow_cell = None
        self.last_events = {name: False for name in EVENT_NAMES}

    def _vector_from(self, p, enemies, kills, game_time):
        vector = [p["x"] / self.map_width, p["y"] / self.map_height,
                  math.sin(p["angle"]), math.cos(p["angle"]), p["hp"] / 100,
                  p["ammo"] / 99, p["cooldown"] / self.FIRE_INTERVAL,
                  p["shot_flash"] / self.FLASH_DURATION]
        for e in enemies:
            vector.extend([e["x"] / self.map_width, e["y"] / self.map_height,
                           e["vx"] / 2, e["vy"] / 2, e["hp"] / 100, float(e["alive"]),
                           e["attack_cooldown"] / self.ATTACK_INTERVAL,
                           e["hurt_timer"] / self.HURT_DURATION,
                           e["death_timer"] / self.DEATH_DURATION, e["animation_phase"]])
        vector.extend([kills / ENTITY_SLOTS, game_time / 300])
        return np.array(vector, dtype=np.float32)

    def state_vector(self):
        """20 float32 values, layout given by module STATE_FIELDS."""
        return self._vector_from(self.player, self.enemies, self.kills, self.time)

    def set_state_vector(self, vector):
        """Decode normalized state for simulator queries; invalid geometry is rejected.

        This is an explicit adapter for *training data collection*. It is not an
        automatic collision-correction layer inside the learned neural model.
        """
        z = np.asarray(vector, dtype=np.float64)
        if z.shape != (STATE_DIM,) or not np.isfinite(z).all():
            raise ValueError(f"Expected finite ({STATE_DIM},) state")
        s = self.get_state()
        p = s["player"]
        p.update(x=float(z[0] * self.map_width), y=float(z[1] * self.map_height),
                 angle=math.atan2(z[2], z[3]), hp=float(np.clip(z[4], 0, 1) * 100),
                 ammo=int(round(float(np.clip(z[5], 0, 1) * 99))),
                 cooldown=float(np.clip(z[6], 0, 1) * self.FIRE_INTERVAL),
                 shot_flash=float(np.clip(z[7], 0, 1) * self.FLASH_DURATION))
        for i, e in enumerate(s["enemies"]):
            q = z[GLOBAL_DIM + i * ENTITY_DIM:GLOBAL_DIM + (i + 1) * ENTITY_DIM]
            alive = bool(q[5] >= 0.5 and q[4] > 0)
            e.update(x=float(q[0] * self.map_width), y=float(q[1] * self.map_height),
                     vx=float(np.clip(q[2], -1, 1) * 2), vy=float(np.clip(q[3], -1, 1) * 2),
                     hp=float(np.clip(q[4], 0, 1) * 100) if alive else 0.0, alive=alive,
                     attack_cooldown=float(np.clip(q[6], 0, 1) * self.ATTACK_INTERVAL),
                     hurt_timer=float(np.clip(q[7], 0, 1) * self.HURT_DURATION),
                     death_timer=float(np.clip(q[8] * self.DEATH_DURATION, 0,
                                               self.RESPAWN_DELAY if self.endless else self.DEATH_DURATION)),
                     animation_phase=float(q[9] % 1))
        s["kills"] = (max(0, int(round(float(z[-2] * ENTITY_SLOTS)))) if self.endless
                      else sum(not e["alive"] for e in s["enemies"]))
        s["time"] = max(0.0, float(z[-1] * 300))
        self.set_state(s)

    def _walkable(self, x, y, radius):
        if x - radius < 0 or y - radius < 0 or x + radius >= self.map_width or y + radius >= self.map_height:
            return False
        for iy in range(int(y - radius), int(y + radius) + 1):
            for ix in range(int(x - radius), int(x + radius) + 1):
                if self.map[iy, ix]:
                    near_x, near_y = np.clip(x, ix, ix + 1), np.clip(y, iy, iy + 1)
                    if (x - near_x) ** 2 + (y - near_y) ** 2 < radius * radius:
                        return False
        return True

    def _move(self, entity, dx, dy, radius):
        # Substeps avoid tunnelling even for the largest supported step duration.
        count = max(1, int(math.ceil(max(abs(dx), abs(dy)) / 0.08)))
        for _ in range(count):
            if self._walkable(entity["x"] + dx / count, entity["y"], radius):
                entity["x"] += dx / count
            if self._walkable(entity["x"], entity["y"] + dy / count, radius):
                entity["y"] += dy / count

    def _ray_distance(self, dx, dy):
        """Distance along a unit ray from player to the first wall."""
        return self._wall_distance(self.player["x"], self.player["y"], dx, dy)

    def _wall_distance(self, x, y, dx, dy):
        mx, my = int(x), int(y)
        ddx = abs(1 / dx) if abs(dx) > 1e-12 else float("inf")
        ddy = abs(1 / dy) if abs(dy) > 1e-12 else float("inf")
        sx, sy = (1 if dx >= 0 else -1), (1 if dy >= 0 else -1)
        tx = ((mx + 1 - x) if dx >= 0 else (x - mx)) * ddx
        ty = ((my + 1 - y) if dy >= 0 else (y - my)) * ddy
        for _ in range(self.map_width + self.map_height + 8):
            if tx < ty:
                distance = tx
                tx += ddx
                mx += sx
            else:
                distance = ty
                ty += ddy
                my += sy
            if not (0 <= mx < self.map_width and 0 <= my < self.map_height) or self.map[my, mx]:
                return distance
        return 100.0

    def _line_clear(self, x0, y0, x1, y1):
        distance = math.hypot(x1 - x0, y1 - y0)
        if distance < 1e-8:
            return True
        return self._wall_distance(x0, y0, (x1 - x0) / distance, (y1 - y0) / distance) >= distance - 1e-6

    def _navigation(self):
        cell = (int(self.player["x"]), int(self.player["y"]))
        if cell == self._flow_cell:
            return self._flow
        field = np.full(self.map.shape, 10000, dtype=np.int32)
        field[cell[1], cell[0]] = 0
        queue = deque([cell])
        while queue:
            x, y = queue.popleft()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < self.map_width and 0 <= ny < self.map_height and not self.map[ny, nx] and field[ny, nx] > field[y, x] + 1:
                    field[ny, nx] = field[y, x] + 1
                    queue.append((nx, ny))
        self._flow_cell, self._flow = cell, field
        return field

    def _respawn_enemies(self):
        """Replace expired corpses by the shared three-second anchor cycle."""
        if not self.enemies_enabled:
            return set()
        ready = [e for e in self.enemies
                 if not e["alive"] and e["death_timer"] >= self.RESPAWN_DELAY - 1e-8]
        if not ready:
            return set()
        p = self.player
        spawned = set()
        tick = round(self.time * SPAWN_TICK_HZ)
        for enemy in ready:
            location = deterministic_spawn_tile(self.map == 0, (p['x'], p['y']),
                [(e['x'], e['y']) for e in self.enemies], [e['alive'] for e in self.enemies],
                enemy['id'], tick)
            if location is not None:
                x, y = location
                enemy.update(self._new_enemy(enemy["id"], x, y))
                spawned.add(enemy['id'])
        return spawned

    def step(self, action, dt=1 / 30):
        a = np.asarray(action, dtype=np.float64)
        if a.shape != (ACTION_DIM,) or not np.isfinite(a).all():
            raise ValueError("Action must contain four finite numbers")
        if not math.isfinite(dt) or not 0 < dt <= 0.25:
            raise ValueError("dt must be finite and in (0,0.25]")
        a = np.clip(a, [-1, -1, -1, 0], [1, 1, 1, 1])
        p = self.player
        events = {name: False for name in EVENT_NAMES}
        events.update(hit_id=-1, killed_id=-1, player_damage=0.0, attacker_ids=[])
        # The serialized interface is float32. Decode its near-lattice timer
        # phase only when advancing a tick; the supplied state is never
        # rounded or replaced by set_state_vector()/conditional Source labels.
        self.time += dt
        p["cooldown"] = max(0.0, p["cooldown"] - dt)
        p["shot_flash"] = max(0.0, p["shot_flash"] - dt)
        for e in self.enemies:
            e["hurt_timer"] = max(0.0, e["hurt_timer"] - dt)
            e["attack_cooldown"] = max(0.0, e["attack_cooldown"] - dt)
            if not e["alive"]:
                e["death_timer"] = advance_timer_seconds(float(e["death_timer"]), duration=self.DEATH_DURATION,
                    dt=float(dt), increasing=True,
                    maximum=self.RESPAWN_DELAY if self.endless else self.DEATH_DURATION)
                e["vx"] = e["vy"] = 0.0
        if p["hp"] <= 0:
            self.last_events = events
            return events
        respawned = self._respawn_enemies() if self.endless else set()
        p["angle"] = float((p["angle"] + a[2] * 2.2 * dt + math.pi) % (2 * math.pi) - math.pi)
        c, s = math.cos(p["angle"]), math.sin(p["angle"])
        forward, strafe = a[0], a[1]
        magnitude = max(1.0, math.hypot(forward, strafe))
        dx = (c * forward - s * strafe) * 2.4 * dt / magnitude
        dy = (s * forward + c * strafe) * 2.4 * dt / magnitude
        self._move(p, dx, dy, self.PLAYER_RADIUS)
        if a[3] >= 0.5 and (self.endless or p["ammo"] > 0) and p["cooldown"] <= 1e-8:
            if not self.endless:
                p["ammo"] -= 1
            p["cooldown"] = self.FIRE_INTERVAL
            p["shot_flash"] = self.FLASH_DURATION
            events["shot"] = True
            best = self._ray_distance(c, s)
            target = None
            for e in self.enemies:
                if not e["alive"]:
                    continue
                rx, ry = e["x"] - p["x"], e["y"] - p["y"]
                along, perpendicular = rx * c + ry * s, abs(rx * s - ry * c)
                if along > 0 and perpendicular < self.ENEMY_RADIUS:
                    entry = along - math.sqrt(self.ENEMY_RADIUS ** 2 - perpendicular ** 2)
                    if 0 <= entry < best:
                        best, target = entry, e
            if target is not None:
                events["hit"], events["hit_id"] = True, target["id"]
                target["hp"] = max(0.0, target["hp"] - 35.0)
                target["hurt_timer"] = self.HURT_DURATION
                if target["hp"] <= 0:
                    target["alive"] = False
                    target["vx"] = target["vy"] = 0.0
                    target["death_timer"] = 0.0
                    self.kills += 1
                    events["kill"], events["killed_id"] = True, target["id"]
        field = self._navigation()
        for e in self.enemies:
            if not e["alive"] or e['id'] in respawned:
                continue
            rx, ry = p["x"] - e["x"], p["y"] - e["y"]
            distance = math.hypot(rx, ry)
            clear = self._line_clear(e["x"], e["y"], p["x"], p["y"])
            e["animation_phase"] = (e["animation_phase"] + dt * 2.6) % 1.0
            oldx, oldy = e["x"], e["y"]
            if distance > 0.78:
                tx, ty = p["x"], p["y"]
                if not clear:
                    ex, ey = int(e["x"]), int(e["y"])
                    choices = [(field[ny, nx], nx, ny) for nx, ny in
                               ((ex + 1, ey), (ex - 1, ey), (ex, ey + 1), (ex, ey - 1))
                               if 0 <= nx < self.map_width and 0 <= ny < self.map_height and not self.map[ny, nx]]
                    if choices:
                        _, nx, ny = min(choices)
                        tx, ty = nx + 0.5, ny + 0.5
                vx, vy = tx - e["x"], ty - e["y"]
                length = max(1e-8, math.hypot(vx, vy))
                speed = 0.70 + e["id"] * 0.018
                self._move(e, vx / length * speed * dt, vy / length * speed * dt, self.ENEMY_RADIUS)
            e["vx"], e["vy"] = (e["x"] - oldx) / dt, (e["y"] - oldy) / dt
            if distance <= 1.0 and clear and e["attack_cooldown"] <= 1e-8:
                e["attack_cooldown"] = self.ATTACK_INTERVAL
                p["hp"] = max(0.0, p["hp"] - 8.0)
                events["player_hit"] = True
                events["player_damage"] += 8.0
                events["attacker_ids"].append(e["id"])
        self.last_events = events
        return events

    @staticmethod
    def event_vector(events):
        return np.array([float(bool(events.get(name, False))) for name in EVENT_NAMES], dtype=np.float32)

    def _make_textures(self):
        yy, xx = np.mgrid[:64, :64]
        rng = np.random.default_rng(782)
        noise = rng.integers(-9, 10, (64, 64, 1))
        brick = np.zeros((64, 64, 3), dtype=np.int16) + [154, 65, 46]
        mortar = (yy % 16 < 2) | ((xx + (yy // 16 % 2) * 16) % 32 < 2)
        brick[mortar] = [76, 61, 47]
        brick = np.clip(brick + noise, 0, 255).astype(np.uint8)
        earth = np.zeros((64, 64, 3), dtype=np.int16) + [114, 84, 43]
        earth[yy < 10 + ((xx // 5) % 3) * 3] = [76, 157, 36]
        earth = np.clip(earth + noise, 0, 255).astype(np.uint8)
        return np.stack([np.zeros_like(brick), brick, earth])

    def _load_atlas(self, filename, columns, rows):
        path = self.assets_dir / filename
        if not path.is_file():
            return None
        with Image.open(path) as source:
            image = source.convert("RGBA")
            return [np.asarray(image.crop((round(x * image.width / columns), round(y * image.height / rows),
                                           round((x + 1) * image.width / columns), round((y + 1) * image.height / rows)))).copy()
                    for y in range(rows) for x in range(columns)]

    @staticmethod
    def _fallback_turkeys():
        frames = []
        for frame in range(8):
            im = Image.new("RGBA", (96, 96))
            d = ImageDraw.Draw(im)
            if frame == 6:
                d.ellipse((10, 66, 85, 89), fill=(89, 58, 42, 255), outline=(35, 29, 24))
                d.polygon([(60, 67), (80, 58), (89, 70)], fill=(158, 70, 47, 255))
            else:
                bob = 2 if frame == 2 else 0
                for dx, dy in [(-24, 13), (-15, 2), (0, -2), (15, 2), (24, 13)]:
                    d.ellipse((23 + dx, 17 + dy + bob, 73 + dx, 70 + dy + bob),
                              fill=(104, 64, 44, 255), outline=(32, 25, 24))
                d.ellipse((22, 39 + bob, 75, 85), fill=(136, 102, 71, 255), outline=(31, 24, 23), width=2)
                d.ellipse((37, 20 + bob, 62, 54 + bob), fill=(75, 133, 155, 255), outline=(29, 29, 31), width=2)
                d.ellipse((40, 26 + bob, 47, 34 + bob), fill="white")
                d.ellipse((51, 26 + bob, 58, 34 + bob), fill="white")
                d.ellipse((44, 29 + bob, 47, 33 + bob), fill="black")
                d.ellipse((51, 29 + bob, 54, 33 + bob), fill="black")
                d.polygon([(45, 36 + bob), (61, 37 + bob), (51, 44 + bob)], fill=(239, 160, 47, 255))
                d.ellipse((43, 40 + bob, 50, 57 + bob), fill=(201, 49, 45, 255))
                shift = 3 if frame == 1 else -3 if frame == 2 else 0
                d.line((37, 81, 34 + shift, 93), fill=(226, 147, 46), width=4)
                d.line((59, 81, 63 - shift, 93), fill=(226, 147, 46), width=4)
                if frame == 4:
                    d.line((29, 21, 68, 72), fill=(255, 224, 155), width=3)
            frames.append(np.asarray(im).copy())
        return frames

    @staticmethod
    def _fallback_guns():
        frames = []
        for frame in range(3):
            im = Image.new("RGBA", (128, 128))
            d = ImageDraw.Draw(im)
            recoil = 7 if frame == 2 else 0
            d.polygon([(39, 128), (48, 71 + recoil), (76, 71 + recoil), (90, 128)], fill=(31, 37, 42), outline=(10, 15, 20))
            d.rectangle((52, 44 + recoil, 72, 95 + recoil), fill=(75, 84, 88), outline=(15, 22, 27), width=3)
            d.rectangle((58, 37 + recoil, 66, 59 + recoil), fill=(18, 24, 29))
            d.polygon([(0, 128), (25, 91), (45, 89), (49, 112), (42, 128)], fill=(190, 136, 89))
            d.polygon([(128, 128), (103, 91), (83, 89), (79, 112), (86, 128)], fill=(190, 136, 89))
            if frame == 1:
                d.polygon([(62, 6), (68, 26), (85, 20), (73, 37), (90, 45), (69, 47),
                           (62, 62), (55, 46), (35, 45), (51, 35), (44, 20), (57, 27)],
                          fill=(255, 220, 85, 255))
            frames.append(np.asarray(im).copy())
        return frames

    @staticmethod
    def _blit_rgba(target, sprite, x, y, width, height, depth_mask=None, tint=None):
        screen_h, screen_w = target.shape[:2]
        x0, x1 = max(0, x), min(screen_w, x + width)
        y0, y1 = max(0, y), min(screen_h, y + height)
        if x0 >= x1 or y0 >= y1:
            return
        sx = np.clip(((np.arange(x0, x1) - x) * sprite.shape[1] / width).astype(int), 0, sprite.shape[1] - 1)
        sy = np.clip(((np.arange(y0, y1) - y) * sprite.shape[0] / height).astype(int), 0, sprite.shape[0] - 1)
        sample = sprite[sy[:, None], sx[None, :]].astype(np.uint16)
        alpha = sample[..., 3:4].astype(np.uint16)
        if depth_mask is not None:
            alpha = alpha * depth_mask[x0:x1][None, :, None]
        rgb = sample[..., :3]
        if tint is not None:
            rgb = np.minimum(255, rgb * np.asarray(tint, dtype=np.float32)).astype(np.uint16)
        old = target[y0:y1, x0:x1].astype(np.uint16)
        target[y0:y1, x0:x1] = ((rgb * alpha + old * (255 - alpha)) // 255).astype(np.uint8)

    def render(self, size=256, overlays=True):
        """Return a square RGB uint8 image; no pygame or real-time clock required.

        overlays=False removes HUD/crosshair for clean model targets. The weapon
        remains in the image because its animation is part of the visual scene.
        """
        size = int(size)
        if not 32 <= size <= 2048:
            raise ValueError("size must be in [32,2048]")
        frame = np.empty((size, size, 3), dtype=np.uint8)
        cy = size // 2
        sky = np.linspace([42, 91, 188], [123, 185, 238], cy).astype(np.uint8)
        frame[:cy] = sky[:, None]
        ground = np.linspace([76, 137, 34], [94, 165, 43], size - cy).astype(np.uint8)
        frame[cy:] = ground[:, None]
        # Simple horizon clouds remain fixed in world-angle coordinates.
        p = self.player
        c, s = math.cos(p["angle"]), math.sin(p["angle"])
        fov = math.radians(68)
        tangent = math.tan(fov / 2)
        projection = size / (2 * tangent)
        camera_x = (np.arange(size) + 0.5) / size * 2 - 1
        rx, ry = c - s * tangent * camera_x, s + c * tangent * camera_x
        mx = np.full(size, int(p["x"]), dtype=np.int32)
        my = np.full(size, int(p["y"]), dtype=np.int32)
        sx, sy = np.where(rx >= 0, 1, -1), np.where(ry >= 0, 1, -1)
        ddx = 1 / np.maximum(np.abs(rx), 1e-12)
        ddy = 1 / np.maximum(np.abs(ry), 1e-12)
        tx = np.where(rx >= 0, mx + 1 - p["x"], p["x"] - mx) * ddx
        ty = np.where(ry >= 0, my + 1 - p["y"], p["y"] - my) * ddy
        hit = np.zeros(size, dtype=bool)
        side = np.zeros(size, dtype=bool)
        depth = np.full(size, 100.0)
        material = np.ones(size, dtype=np.uint8)
        for _ in range(self.map_width + self.map_height + 8):
            active = ~hit
            on_x = (tx < ty) & active
            on_y = ~on_x & active
            distances = np.where(on_x, tx, ty)
            mx[on_x] += sx[on_x]
            my[on_y] += sy[on_y]
            tx[on_x] += ddx[on_x]
            ty[on_y] += ddy[on_y]
            tile = self.map[np.clip(my, 0, self.map_height - 1), np.clip(mx, 0, self.map_width - 1)]
            new_hit = active & (tile != 0)
            depth[new_hit] = distances[new_hit]
            side[new_hit] = on_y[new_hit]
            material[new_hit] = tile[new_hit]
            hit |= new_hit
            if hit.all():
                break
        depth = np.maximum(depth, 0.02)
        wall_height = projection / depth
        top = cy - wall_height / 2
        ycoords = np.arange(size)[:, None]
        v = (ycoords - top[None, :]) / wall_height[None, :]
        visible = (v >= 0) & (v < 1)
        wall_x = np.where(side, p["x"] + depth * rx, p["y"] + depth * ry)
        u = ((wall_x % 1) * 64).astype(int)
        texture_y = np.clip((v * 64).astype(int), 0, 63)
        wall_pixels = self._textures[material[None, :], texture_y, u[None, :]].astype(np.float32)
        shade = np.where(side, 0.76, 1.0) * np.clip(1 - depth * 0.024, 0.50, 1)
        wall_pixels = (wall_pixels * shade[None, :, None]).astype(np.uint8)
        frame[visible] = wall_pixels[visible]
        ordered = sorted(self.enemies, key=lambda e: (e["x"] - p["x"]) ** 2 + (e["y"] - p["y"]) ** 2, reverse=True) if self.enemies_enabled else []
        for e in ordered:
            dx, dy = e["x"] - p["x"], e["y"] - p["y"]
            forward, lateral = dx * c + dy * s, -dx * s + dy * c
            if forward <= 0.12:
                continue
            height = int(min(size * 5, projection * 0.93 / forward))
            width = height
            screen_x = int(size / 2 + projection * lateral / forward)
            bottom = int(cy + projection * 0.5 / forward)
            if not e["alive"]:
                index = 5 if e["death_timer"] < 0.35 else 6
            elif e["hurt_timer"] > 0:
                index = 4
            elif e["attack_cooldown"] > 1.0:
                index = 3
            elif abs(e["vx"]) + abs(e["vy"]) > 0.05:
                index = 1 if e["animation_phase"] < 0.5 else 2
            else:
                index = 0
            self._blit_rgba(frame, self._turkeys[index], screen_x - width // 2,
                            bottom - height, width, height, depth > forward)
        gun_frame = 1 if p["shot_flash"] > 0.045 else 2 if p["shot_flash"] > 0 else 0
        gun_size = int(size * 0.67)
        self._blit_rgba(frame, self._guns[gun_frame], size // 2 - gun_size // 2,
                        size - gun_size, gun_size, gun_size)
        if p["hp"] <= 0:
            frame = ((frame.astype(np.uint16) + np.array([70, 8, 5])) // 2).astype(np.uint8)
        if overlays:
            image = Image.fromarray(frame)
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, size, 17), fill=(15, 30, 33))
            if self.endless:
                alive = sum(e["alive"] for e in self.enemies)
                hud = f"HP {int(p['hp'])}  AMMO INF  ALIVE {alive}  KILLS {self.kills}"
            else:
                hud = f"HP {int(p['hp']):3d}   AMMO {p['ammo']:2d}   TURKEYS {ENTITY_SLOTS - self.kills}"
            draw.text((4, 3), hud, fill=(255, 239, 190))
            arm = max(2, size // 64)
            draw.line((size // 2 - arm, cy, size // 2 + arm, cy), fill=(250, 250, 220))
            draw.line((size // 2, cy - arm, size // 2, cy + arm), fill=(250, 250, 220))
            if self.done:
                label = "GAME OVER  R: restart" if p["hp"] <= 0 else "YOU WIN! R: restart"
                box_w = min(size, len(label) * 6 + 8)
                draw.rectangle((size // 2 - box_w // 2, cy - 16, size // 2 + box_w // 2, cy + 7), fill=(25, 30, 35))
                draw.text((size // 2 - box_w // 2 + 4, cy - 11), label, fill=(255, 224, 130))
            frame = np.asarray(image).copy()
        return frame
