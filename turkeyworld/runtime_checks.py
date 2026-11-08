"""Read-only state-integrity measurements for a free neural rollout.

This module never repairs a prediction. Wall and shot checks are evaluation
oracles only, with no connection back into the model state or its memory.
"""
from __future__ import annotations

from collections import Counter
import math
import numpy as np

EVENT_NAMES = ("shot", "hit", "kill", "player_hit")
PLAYER_RADIUS, NPC_RADIUS = .20, .25
POSE_TOL, HP_TOL, AMMO_TOL = .002, .05, .02


def entities(state):
    return np.asarray(state)[8:88].reshape(8, 10)


def circle_is_clear(tilemap, point, radius):
    """Circle-vs-tile geometry, independent of the reference Game object."""
    x, y = np.asarray(point, dtype=float)
    height, width = tilemap.shape
    if not np.isfinite([x, y]).all() or x-radius < 0 or y-radius < 0 or x+radius >= width or y+radius >= height:
        return False
    for row in range(int(y-radius), int(y+radius)+1):
        for col in range(int(x-radius), int(x+radius)+1):
            if tilemap[row, col]:
                nearest_x, nearest_y = np.clip(x, col, col+1), np.clip(y, row, row+1)
                if (x-nearest_x)**2 + (y-nearest_y)**2 < radius**2 - 1e-8:
                    return False
    return True


def segment_is_clear(tilemap, start, end, radius):
    """Check the reference game's separated X-then-Y movement convention.

    A straight chord can cut a rounded obstacle corner even when both legs of
    the game's legal slide stay clear. At the supported 30 Hz speed caps there
    is at most one .08-tile movement substep. Long invalid jumps are still
    sampled through each leg rather than checking only their endpoints.
    """
    start, end = np.asarray(start), np.asarray(end)
    elbow = np.array([end[0], start[1]])
    for first, last in ((start, elbow), (elbow, end)):
        length = float(np.linalg.norm(last-first))
        count = max(2, min(4096, int(math.ceil(length/.04))+1))
        if not all(circle_is_clear(tilemap, point, radius)
                   for point in np.linspace(first, last, count)):
            return False
    return True


def wall_distance(tilemap, start, direction):
    """DDA used solely by the evaluator, parameterized by the supplied ray."""
    x, y = map(float, start)
    dx, dy = map(float, direction)
    height, width = tilemap.shape
    if not np.isfinite([x, y, dx, dy]).all() or not 0 <= x < width or not 0 <= y < height:
        return 0.
    mx, my = math.floor(x), math.floor(y)
    if tilemap[my, mx]:
        return 0.
    sx, sy = (1 if dx >= 0 else -1), (1 if dy >= 0 else -1)
    ddx = abs(1/dx) if abs(dx) > 1e-12 else math.inf
    ddy = abs(1/dy) if abs(dy) > 1e-12 else math.inf
    tx = ((mx+1-x) if dx >= 0 else (x-mx))*ddx if math.isfinite(ddx) else math.inf
    ty = ((my+1-y) if dy >= 0 else (y-my))*ddy if math.isfinite(ddy) else math.inf
    for _ in range(width+height+4):
        if tx < ty:
            distance, tx, mx = tx, tx+ddx, mx+sx
        else:
            distance, ty, my = ty, ty+ddy, my+sy
        if not 0 <= mx < width or not 0 <= my < height or tilemap[my, mx]:
            return float(distance)
    return math.inf


def line_is_clear(tilemap, start, end):
    delta = np.asarray(end)-np.asarray(start)
    distance = float(np.linalg.norm(delta))
    return distance < 1e-8 or wall_distance(tilemap, start, delta/distance) >= distance-1e-5


def event_flags(events):
    if isinstance(events, dict):
        return np.array([bool(events.get(name, False)) for name in EVENT_NAMES])
    values = np.asarray(events, dtype=float).reshape(-1)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("events must contain four finite event flags")
    return values >= .5


def check_transition(tilemap, previous, current, action, events, dt=1/30):
    """Return violation dictionaries; all input arrays remain untouched.

    The 90-value normalized state contract is the game's public interface.
    Boolean/discrete-state checks are intentional: fractional ammo/alive values
    cannot be hidden by rounding an invalid legacy prediction into a valid one.
    """
    previous, current, action = [np.asarray(x, dtype=float) for x in (previous, current, action)]
    failures = []
    def fail(code, message):
        failures.append({"code": code, "message": message})
    if previous.shape != (90,) or current.shape != (90,) or action.shape != (4,):
        fail("shape", "Expected previous/current state90 and action4")
        return failures
    if not np.isfinite(current).all():
        fail("non_finite_state", "Predicted state contains NaN or infinity")
        return failures
    try:
        shot, hit, kill, player_hit = event_flags(events)
    except ValueError as error:
        fail("non_finite_events", str(error))
        return failures
    height, width = tilemap.shape
    scale = np.array([width, height])
    old, new = entities(previous), entities(current)
    p0, p1 = previous[:2]*scale, current[:2]*scale
    npc0, npc1 = old[:, :2]*scale, new[:, :2]*scale
    bounded_indices = [0, 1, 4, 5, 6, 7, 88]
    if np.any(current[bounded_indices] < -1e-5) or np.any(current[bounded_indices] > 1+1e-5):
        fail("player_range", "Player coordinates/HP/ammo/timers or kill count outside normalized range")
    if abs(np.linalg.norm(current[2:4])-1) > .002:
        fail("heading_norm", "Heading sin/cos is not a unit vector")
    if np.any(new[:, [0, 1, 4, 5, 6, 7, 8, 9]] < -1e-5) or np.any(new[:, [0, 1, 4, 5, 6, 7, 8, 9]] > 1+1e-5):
        fail("npc_range", "NPC position/HP/life/timers outside normalized range")
    if np.any(np.minimum(np.abs(new[:, 5]), np.abs(new[:, 5]-1)) > .001):
        fail("nonbinary_alive", "NPC alive flags must be discrete zero/one")
    alive0, alive1 = old[:, 5] >= .5, new[:, 5] >= .5
    if np.any((new[:, 4] > .0005) != alive1):
        fail("hp_alive_disagreement", "NPC HP zero and alive flag disagree")
    if np.any(~alive0 & alive1):
        fail("revival", "A dead persistent NPC slot became alive")
    if np.any((new[:, 4]-old[:, 4])*100 > HP_TOL):
        fail("npc_healing", "NPC HP increased although the game has no healing")
    if (current[4]-previous[4])*100 > HP_TOL:
        fail("player_healing", "Player HP increased without a respawn/healing mechanic")
    kills = current[88]*8
    if abs(kills-round(kills)) > .01 or abs(kills-np.count_nonzero(~alive1)) > .01:
        fail("kill_count", "Kill count is fractional or disagrees with persistent dead slots")
    if abs((current[89]-previous[89])*300-dt) > .002:
        fail("clock", "Simulation clock does not advance by the configured fixed timestep")
    for label, start, end, radius in [("player", p0, p1, PLAYER_RADIUS)] + [
            (f"npc_{i}", npc0[i], npc1[i], NPC_RADIUS) for i in range(8)]:
        if not circle_is_clear(tilemap, end, radius):
            fail("wall_collision", f"{label} overlaps a wall or map boundary")
        elif not segment_is_clear(tilemap, start, end, radius):
            fail("wall_tunneling", f"{label} crossed a wall between endpoints")
    displacement = float(np.linalg.norm(p1-p0))
    if displacement > 2.4*dt + POSE_TOL:
        fail("player_speed", "Player displacement exceeds 2.4 world tiles/second")
    controls = np.clip(action[:2], -1, 1)
    controls = controls/max(1., float(np.linalg.norm(controls)))
    commanded_delta = np.array([current[3]*controls[0]-current[2]*controls[1],
                                current[2]*controls[0]+current[3]*controls[1]])*(2.4*dt)
    actual_delta = p1-p0
    if previous[4] <= .0005:
        commanded_delta[:] = 0
    if np.any(np.abs(actual_delta) > np.abs(commanded_delta)+POSE_TOL) or np.any(actual_delta*commanded_delta < -POSE_TOL*np.abs(commanded_delta)):
        fail("movement_direction", "Motion must follow commanded axes; walls may block motion but cannot invent an opposite/uncommanded displacement")
    npc_distances = np.linalg.norm(npc1-npc0, axis=-1)
    if np.any(npc_distances > 1.2*dt+POSE_TOL):
        fail("npc_speed", "NPC displacement exceeds the event_v1 1.2 tile/second cap")
    if np.any(npc_distances[~alive0] > POSE_TOL) or np.any(np.abs(new[~alive0, 2:4]) > 1e-4):
        fail("corpse_moved", "An already dead NPC moved or retained nonzero velocity")
    if previous[4] <= .0005 and (displacement > POSE_TOL or shot or hit or kill or player_hit):
        fail("dead_player_activity", "A terminal dead player continued moving/firing/combat")
    old_angle, new_angle = math.atan2(previous[2], previous[3]), math.atan2(current[2], current[3])
    angle_change = (new_angle-old_angle+math.pi) % (2*math.pi)-math.pi
    expected_turn = float(np.clip(action[2], -1, 1))*2.2*dt if previous[4] > .0005 else 0.
    if abs(angle_change-expected_turn) > .003:
        fail("turn_response", "Heading does not follow the commanded fixed-step turn")
    ammo0, ammo1 = previous[5]*99, current[5]*99
    if abs(ammo1-round(ammo1)) > AMMO_TOL:
        fail("fractional_ammo", "Ammo count is not an integer")
    if abs(ammo0-ammo1-float(shot)) > AMMO_TOL:
        fail("shot_ammo", "A shot must spend exactly one round; no shot must preserve ammo")
    eligible_shot = previous[4] > .0005 and action[3] >= .5 and ammo0 >= .5 and previous[6]*.22-dt <= 1e-5
    if bool(shot) != bool(eligible_shot):
        fail("shot_eligibility", "Shot event disagrees with fire input, ammo, cooldown or player life")
    expected_cooldown = 1. if shot else max(0., previous[6]-dt/.22)
    expected_flash = 1. if shot else max(0., previous[7]-dt/.10)
    if abs(current[6]-expected_cooldown) > .002 or abs(current[7]-expected_flash) > .002:
        fail("weapon_timers", "Fire cooldown or muzzle-flash timer is inconsistent")
    losses = (old[:, 4]-new[:, 4])*100
    damaged = np.flatnonzero(losses > HP_TOL)
    if bool(hit) != bool(len(damaged)) or len(damaged) > 1 or (hit and not shot):
        fail("hit_hp", "Hit event and NPC damage are inconsistent")
    if shot:
        direction = np.array([current[3], current[2]])
        wall_limit = wall_distance(tilemap, p1, direction)
        candidates = []
        for i in range(8):
            relative = npc0[i]-p1
            along = float(relative @ direction)
            perpendicular = abs(relative[0]*direction[1]-relative[1]*direction[0])
            if alive0[i] and along > 0 and perpendicular < NPC_RADIUS:
                entry = along-math.sqrt(max(0., NPC_RADIUS**2-perpendicular**2))
                if 0 <= entry < wall_limit:
                    candidates.append((entry, i, perpendicular))
        if candidates:
            entry, expected_id, perpendicular = min(candidates)
            # Require unequivocal impacts, leaving only a small numerical
            # tolerance at tangent/wall boundaries rather than forgiving a
            # permanently "none" target classifier on a clear centered shot.
            if perpendicular < NPC_RADIUS-.005 and entry < wall_limit-POSE_TOL:
                if not hit or len(damaged) != 1 or int(damaged[0]) != expected_id:
                    fail("missed_visible_hit", "A clear nearest living NPC on the fired center ray must take damage")
    if len(damaged) == 1:
        target = int(damaged[0])
        if abs(losses[target]-min(35., old[target, 4]*100)) > HP_TOL:
            fail("weapon_damage", "A hitscan impact must remove up to 35 HP from one NPC")
        direction = np.array([current[3], current[2]])
        relative = npc0[target]-p1
        along = float(relative @ direction)
        perpendicular = abs(relative[0]*direction[1]-relative[1]*direction[0])
        if along <= 0 or perpendicular >= NPC_RADIUS+.005:
            fail("hit_off_ray", "Damaged NPC was not on the fired center ray")
        else:
            entry = along-math.sqrt(max(0., NPC_RADIUS**2-min(perpendicular, NPC_RADIUS)**2))
            if wall_distance(tilemap, p1, direction) < entry-POSE_TOL:
                fail("hit_through_wall", "A wall blocked the damaged NPC on the firing ray")
            closer = []
            for i in range(8):
                delta = npc0[i]-p1
                d = float(delta @ direction)
                q = abs(delta[0]*direction[1]-delta[1]*direction[0])
                if alive0[i] and d > 0 and q < NPC_RADIUS:
                    closer.append((d-math.sqrt(max(0., NPC_RADIUS**2-q*q)), i))
            if closer and min(closer)[1] != target:
                fail("hit_not_nearest", "The shot damaged a target behind a closer living NPC")
    died = alive0 & ~alive1
    if bool(kill) != bool(died.any()) or np.count_nonzero(died) > 1 or (kill and not hit):
        fail("kill_event", "Kill event must agree with one HP-zero/alive transition")
    hp_loss = (previous[4]-current[4])*100
    if bool(player_hit) != bool(hp_loss > HP_TOL):
        fail("player_hit_hp", "Player-hit event and player HP change disagree")
    eligible_attackers = []
    strictly_ready_attackers = []
    for i in range(8):
        if alive1[i] and np.linalg.norm(npc0[i]-p1) <= 1.+POSE_TOL and old[i, 6]*1.2-dt <= 1e-5 and line_is_clear(tilemap, npc0[i], p1):
            eligible_attackers.append(i)
            if np.linalg.norm(npc0[i]-p1) <= 1.-POSE_TOL:
                strictly_ready_attackers.append(i)
    if previous[4] > .0005 and strictly_ready_attackers:
        required_damage = min(previous[4]*100, 8*len(strictly_ready_attackers))
        if not player_hit or hp_loss < required_damage-HP_TOL:
            fail("missed_ready_attack", "Ready living NPCs in clear melee range must attack; a permanently silent attack head is not correct combat")
    for i in range(8):
        decayed = max(0., old[i, 6]-dt/1.2)
        allowed = [1.] if previous[4] > .0005 and i in strictly_ready_attackers else [decayed]
        # At the narrow range-boundary tolerance either discrete branch may
        # be legal. A cooldown that stays frozen at1 while not ready is not.
        if previous[4] > .0005 and i in eligible_attackers and i not in strictly_ready_attackers:
            allowed.append(1.)
        if min(abs(new[i, 6]-value) for value in allowed) > .002:
            fail("npc_attack_timer", f"NPC{i} attack cooldown failed its decay/reset contract")
    if hp_loss > HP_TOL:
        if not eligible_attackers:
            fail("player_damage_without_attacker", "No living, nearby, unobstructed, ready attacker could damage player")
        if hp_loss > 8*len(eligible_attackers)+HP_TOL:
            fail("player_damage_amount", "Player damage exceeds eligible NPC attacks")
        if current[4] > .0005 and abs(hp_loss/8-round(hp_loss/8)) > .01:
            fail("fractional_attack_damage", "NPC attacks must deal 8 HP each, except a lethal clipped hit")
    return failures


class RolloutMonitor:
    """Aggregate integrity and nontrivial active-session qualification."""
    def __init__(self, tilemap, scenario, requested_steps, dt=1/30, max_examples=50):
        self.tilemap = np.asarray(tilemap)
        self.scenario, self.requested_steps, self.dt = scenario, requested_steps, dt
        self.max_examples = max_examples
        self.steps = 0
        self.counts, self.examples = Counter(), []
        self.first_failure = None
        self.path_tiles = self.reference_path_tiles = 0.
        self.movement_commands = self.responsive_movement_steps = self.reference_movement_steps = 0
        self.turn_commands = self.fire_commands = 0
        self.terminal_step = self.reference_terminal_step = None
        self.visited_cells = set()
        self.npc_path_tiles = np.zeros(8, dtype=float)
        self.reference_npc_path_tiles = np.zeros(8, dtype=float)
        self.latest_model_alive = np.ones(8, dtype=bool)

    def observe(self, previous, current, action, events, reference_previous=None, reference_current=None):
        self.steps += 1
        violations = check_transition(self.tilemap, previous, current, action, events, self.dt)
        for violation in violations:
            self.counts[violation["code"]] += 1
            entry = {"step": self.steps, "time_seconds": round(self.steps*self.dt, 6), **violation}
            if self.first_failure is None:
                self.first_failure = entry
            if len(self.examples) < self.max_examples:
                self.examples.append(entry)
        if np.isfinite(current).all():
            scale = np.array(self.tilemap.shape[::-1])
            distance = float(np.linalg.norm((np.asarray(current)[:2]-np.asarray(previous)[:2])*scale))
            self.path_tiles += distance
            old_npcs, new_npcs = entities(previous), entities(current)
            self.npc_path_tiles += np.linalg.norm((new_npcs[:, :2]-old_npcs[:, :2])*scale, axis=-1)
            self.latest_model_alive = new_npcs[:, 5] >= .5
            self.visited_cells.add(tuple(np.floor(np.asarray(current)[:2]*scale).astype(int)))
            commanded = bool(np.linalg.norm(np.asarray(action)[:2]) > .2)
            self.movement_commands += int(commanded)
            self.responsive_movement_steps += int(commanded and distance > .005)
            self.turn_commands += int(abs(action[2]) > .1)
            self.fire_commands += int(action[3] >= .5)
            if self.terminal_step is None and (current[4] <= .0005 or current[88]*8 >= 7.99):
                self.terminal_step = self.steps
            if reference_current is not None:
                ref_distance = float(np.linalg.norm((np.asarray(reference_current)[:2]-np.asarray(reference_previous)[:2])*scale))
                self.reference_path_tiles += ref_distance
                old_reference_npcs, new_reference_npcs = entities(reference_previous), entities(reference_current)
                self.reference_npc_path_tiles += np.linalg.norm(
                    (new_reference_npcs[:, :2]-old_reference_npcs[:, :2])*scale, axis=-1)
                self.reference_movement_steps += int(commanded and ref_distance > .005)
                if self.reference_terminal_step is None and (reference_current[4] <= .0005 or reference_current[88]*8 >= 7.99):
                    self.reference_terminal_step = self.steps
        return violations

    def report(self, reference_completed=True, reference_terminal_step=None):
        source_terminal = self.reference_terminal_step if reference_terminal_step is None else reference_terminal_step
        active_scenario = self.scenario in {"explore", "combat", "mixed"}
        completed = self.steps >= self.requested_steps
        meaningful_motion = (self.movement_commands >= min(30, self.requested_steps//2)
                             and self.path_tiles >= min(8., .5*self.requested_steps*self.dt)
                             and self.responsive_movement_steps >= max(1, .5*self.reference_movement_steps)
                             and len(self.visited_cells) >= min(6, max(2, self.requested_steps//30)))
        required_npcs = (self.reference_npc_path_tiles >= 1.) & self.latest_model_alive
        ratios = np.divide(self.npc_path_tiles, self.reference_npc_path_tiles,
                           out=np.zeros(8), where=self.reference_npc_path_tiles > 1e-8)
        frozen_npcs = np.flatnonzero(required_npcs & (ratios < .1)).tolist()
        npc_response_pass = not frozen_npcs
        model_ended_early = self.terminal_step is not None and self.terminal_step < self.requested_steps
        reference_ended_early = source_terminal is not None and source_terminal < self.requested_steps
        qualifying = bool(active_scenario and completed and meaningful_motion
                          and npc_response_pass and not model_ended_early
                          and reference_completed and not reference_ended_early)
        integrity = self.first_failure is None
        return {"completed_requested_steps": completed, "checked_steps": self.steps,
                "checked_simulated_seconds": round(self.steps*self.dt, 6),
                "integrity_pass": integrity, "qualifying_active_session": qualifying,
                "behavioral_checks_pass": bool(meaningful_motion and npc_response_pass),
                "state_integrity_active_300s_pass": bool(integrity and qualifying and self.steps*self.dt >= 300.-1e-6),
                "first_failure": self.first_failure, "violation_counts": dict(self.counts),
                "failure_examples": self.examples, "model_terminal_step": self.terminal_step,
                "reference_terminal_step": source_terminal,
                "coverage": {"path_tiles": self.path_tiles, "reference_path_tiles": self.reference_path_tiles,
                             "movement_command_steps": self.movement_commands,
                             "responsive_movement_steps": self.responsive_movement_steps,
                             "reference_responsive_movement_steps": self.reference_movement_steps,
                             "turn_command_steps": self.turn_commands, "fire_command_steps": self.fire_commands,
                             "distinct_player_cells": len(self.visited_cells)},
                "npc_responsiveness": {"model_path_tiles_by_id": self.npc_path_tiles.tolist(),
                                       "reference_path_tiles_by_id": self.reference_npc_path_tiles.tolist(),
                                       "path_ratio_by_id": ratios.tolist(),
                                       "evaluated_ids": np.flatnonzero(required_npcs).tolist(),
                                       "frozen_alive_ids": frozen_npcs,
                                       "pass": npc_response_pass,
                                       "note": "Broad anti-freeze gate: living NPCs whose reference travelled at least1tile must travel at least10% as far; this does not prove correct AI/combat."},
                "qualification_note": "Idle/turn/wall, early terminal episodes and frozen predictions cannot establish 300 seconds of active gameplay."}
