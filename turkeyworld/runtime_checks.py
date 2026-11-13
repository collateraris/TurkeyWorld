"""Read-only state-integrity measurements for a free neural rollout.

This module never repairs a prediction. Wall and shot checks are evaluation
oracles only, with no connection back into the model state or its memory.
"""
from __future__ import annotations

from .engine import GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS, STATE_DIM

from collections import Counter
import math
import numpy as np
from .simulation_clock import advance_timer_seconds

EVENT_NAMES = ("shot", "hit", "kill", "player_hit")
PLAYER_RADIUS, NPC_RADIUS = .20, .25
POSE_TOL, HP_TOL, AMMO_TOL = .002, .05, .02


def entities(state):
    return np.asarray(state)[GLOBAL_DIM:-2].reshape(ENTITY_SLOTS, ENTITY_DIM)


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


def check_transition(tilemap, previous, current, action, events, dt=1/30, *, endless=False):
    """Return violation dictionaries; all input arrays remain untouched.

    The STATE_DIM-value normalized state contract is the game's public interface.
    Boolean/discrete-state checks are intentional: fractional ammo/alive values
    cannot be hidden by rounding an invalid legacy prediction into a valid one.
    """
    previous, current, action = [np.asarray(x, dtype=float) for x in (previous, current, action)]
    failures = []
    def fail(code, message):
        failures.append({"code": code, "message": message})
    if previous.shape != (STATE_DIM,) or current.shape != (STATE_DIM,) or action.shape != (4,):
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
    original_old, new = entities(previous), entities(current)
    old = original_old.copy()
    p0, p1 = previous[:2]*scale, current[:2]*scale
    npc1 = new[:, :2]*scale
    original_alive = (original_old[:, 5] >= .5) & (original_old[:, 4] > .0005)
    alive1 = new[:, 5] >= .5
    revived = ~original_alive & alive1
    valid_respawns = np.zeros(ENTITY_SLOTS, dtype=bool)
    if endless:
        # Construct a virtual pre-combat spawn state only for measurement. It
        # never replaces a model prediction or feeds a correction into rollout.
        occupied = [old[i, :2]*scale for i in range(ENTITY_SLOTS) if original_alive[i]]
        floor_centers = np.argwhere(tilemap == 0)[:, ::-1].astype(float) + .5
        for i in range(ENTITY_SLOTS):
            ready = (not original_alive[i] and previous[4] > .0005
                     and advance_timer_seconds(float(original_old[i, 8])*.6,
                         duration=.6, dt=dt, increasing=True, maximum=3.) >= 3.-1e-8)
            if revived[i]:
                center = np.floor(npc1[i])+.5
                legal = True
                if not ready:
                    fail("respawn_eligibility", f"NPC{i} revived before three dead seconds or after player death")
                    legal = False
                if not circle_is_clear(tilemap, center, NPC_RADIUS):
                    fail("respawn_wall", f"NPC{i} spawned on a blocked tile")
                    legal = False
                if np.linalg.norm(npc1[i]-center) > 1.2*dt+POSE_TOL:
                    fail("respawn_center", f"NPC{i} spawn must begin at a floor tile center")
                    legal = False
                if np.linalg.norm(center-p0) < 4.-POSE_TOL:
                    fail("respawn_player_distance", f"NPC{i} spawned within four tiles of the player")
                    legal = False
                if any(np.linalg.norm(center-point) < 1.-POSE_TOL for point in occupied):
                    fail("respawn_spacing", f"NPC{i} spawned within one tile of another living NPC")
                    legal = False
                allowed_health = [60., 25.] if shot and hit else [60.]
                if min(abs(new[i, 4]*100-value) for value in allowed_health) > HP_TOL:
                    fail("respawn_health", f"NPC{i} must respawn with 60 HP, minus a possible same-tick shot")
                    legal = False
                if legal:
                    valid_respawns[i] = True
                    occupied.append(center)
                    old[i] = np.array([*(center/scale), 0., 0., .60, 1., 0., 0., 0., 0.])
            elif ready:
                # A permanently dead slot cannot masquerade as endless play.
                has_candidate = any(np.linalg.norm(center-p0) >= 4.-POSE_TOL
                                    and all(np.linalg.norm(center-point) >= 1.-POSE_TOL for point in occupied)
                                    for center in floor_centers)
                if has_candidate:
                    fail("missed_respawn", f"NPC{i} stayed dead despite an elapsed timer and a legal spawn tile")
    elif revived.any():
        fail("revival", "A dead persistent NPC slot became alive")
    npc0 = old[:, :2]*scale
    bounded_indices = [0, 1, 4, 5, 6, 7] + ([] if endless else [-2])
    if np.any(current[bounded_indices] < -1e-5) or np.any(current[bounded_indices] > 1+1e-5):
        fail("player_range", "Player coordinates/HP/ammo/timers or kill count outside normalized range")
    if abs(np.linalg.norm(current[2:4])-1) > .002:
        fail("heading_norm", "Heading sin/cos is not a unit vector")
    npc_bounded = [0, 1, 4, 5, 6, 7, 9] + ([] if endless else [8])
    if (np.any(new[:, npc_bounded] < -1e-5) or np.any(new[:, npc_bounded] > 1+1e-5)
            or (endless and (np.any(new[:, 8] < -1e-5) or np.any(new[:, 8] > 5.+1e-5)))):
        fail("npc_range", "NPC position/HP/life/timers outside normalized range")
    if np.any(np.minimum(np.abs(new[:, 5]), np.abs(new[:, 5]-1)) > .001):
        fail("nonbinary_alive", "NPC alive flags must be discrete zero/one")
    alive0 = old[:, 5] >= .5
    if np.any((new[:, 4] > .0005) != alive1):
        fail("hp_alive_disagreement", "NPC HP zero and alive flag disagree")
    if endless and np.any(revived & ~valid_respawns):
        fail("revival", "A dead persistent NPC slot became alive")
    if np.any((new[:, 4]-old[:, 4])*100 > HP_TOL):
        fail("npc_healing", "NPC HP increased although the game has no healing")
    if (current[4]-previous[4])*100 > HP_TOL:
        fail("player_healing", "Player HP increased without a respawn/healing mechanic")
    kills = current[-2]*ENTITY_SLOTS
    expected_kills = previous[-2]*ENTITY_SLOTS+np.count_nonzero(alive0 & ~alive1) if endless else np.count_nonzero(~alive1)
    if kills < -.01 or abs(kills-round(kills)) > .01 or abs(kills-expected_kills) > .01:
        fail("kill_count", "Kill count is fractional or disagrees with persistent dead slots")
    if abs((current[-1]-previous[-1])*300-dt) > .002:
        fail("clock", "Simulation clock does not advance by the configured fixed timestep")
    for label, start, end, radius in [("player", p0, p1, PLAYER_RADIUS)] + [
            (f"npc_{i}", npc0[i], npc1[i], NPC_RADIUS) for i in range(ENTITY_SLOTS)]:
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
    expected_ammo_spend = 0. if endless else float(shot)
    if abs(ammo0-ammo1-expected_ammo_spend) > AMMO_TOL:
        fail("shot_ammo", "Endless ammo must remain constant" if endless else
             "A shot must spend exactly one round; no shot must preserve ammo")
    eligible_shot = previous[4] > .0005 and action[3] >= .5 and (endless or ammo0 >= .5) and previous[6]*.22-dt <= 1e-5
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
        for i in range(ENTITY_SLOTS):
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
            for i in range(ENTITY_SLOTS):
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
    if endless:
        advanced_death = np.array([advance_timer_seconds(float(value)*.6,
            duration=.6, dt=dt, increasing=True, maximum=3.)/.6 for value in old[:, 8]])
        expected_death = np.where(~alive0, advanced_death, old[:, 8])
        expected_death[died] = 0.
        if np.any(np.abs(new[:, 8]-expected_death) > .002):
            fail("npc_death_timer", "Death timers must advance continuously and reset only on a kill or legal respawn")
    hp_loss = (previous[4]-current[4])*100
    if bool(player_hit) != bool(hp_loss > HP_TOL):
        fail("player_hit_hp", "Player-hit event and player HP change disagree")
    eligible_attackers = []
    strictly_ready_attackers = []
    for i in range(ENTITY_SLOTS):
        if alive1[i] and np.linalg.norm(npc0[i]-p1) <= 1.+POSE_TOL and old[i, 6]*1.2-dt <= 1e-5 and line_is_clear(tilemap, npc0[i], p1):
            eligible_attackers.append(i)
            if np.linalg.norm(npc0[i]-p1) <= 1.-POSE_TOL:
                strictly_ready_attackers.append(i)
    if previous[4] > .0005 and strictly_ready_attackers:
        required_damage = min(previous[4]*100, 8*len(strictly_ready_attackers))
        if not player_hit or hp_loss < required_damage-HP_TOL:
            fail("missed_ready_attack", "Ready living NPCs in clear melee range must attack; a permanently silent attack head is not correct combat")
    for i in range(ENTITY_SLOTS):
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
    def __init__(self, tilemap, scenario, requested_steps, dt=1/30, max_examples=50, *, endless=False):
        self.tilemap = np.asarray(tilemap)
        self.scenario, self.requested_steps, self.dt = scenario, requested_steps, dt
        self.max_examples = max_examples
        self.endless = bool(endless)
        self.steps = 0
        self.counts, self.examples = Counter(), []
        self.first_failure = None
        self.path_tiles = self.reference_path_tiles = 0.
        self.movement_commands = self.responsive_movement_steps = self.reference_movement_steps = 0
        self.turn_commands = self.fire_commands = 0
        self.terminal_step = self.reference_terminal_step = None
        self.visited_cells = set()
        self.npc_path_tiles = np.zeros(ENTITY_SLOTS, dtype=float)
        self.reference_npc_path_tiles = np.zeros(ENTITY_SLOTS, dtype=float)
        self.latest_model_alive = np.ones(ENTITY_SLOTS, dtype=bool)
        self.event_counts, self.reference_event_counts = Counter(), Counter()
        self.respawn_count = self.reference_respawn_count = 0
        self.model_player_alive = self.reference_player_alive = True

    def observe(self, previous, current, action, events, reference_previous=None, reference_current=None):
        self.steps += 1
        violations = check_transition(self.tilemap, previous, current, action, events, self.dt,
                                      endless=self.endless)
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
            npc_motion = np.linalg.norm((new_npcs[:, :2]-old_npcs[:, :2])*scale, axis=-1)
            revived = (old_npcs[:, 5] < .5) & (new_npcs[:, 5] >= .5)
            if self.endless:
                npc_motion[revived] = 0.  # Teleports do not count as neural AI response.
            self.npc_path_tiles += npc_motion
            self.respawn_count += int(revived.sum())
            self.latest_model_alive = new_npcs[:, 5] >= .5
            self.model_player_alive = bool(current[4] > .0005)
            for name, flag in zip(EVENT_NAMES, event_flags(events)):
                self.event_counts[name] += int(flag)
            self.visited_cells.add(tuple(np.floor(np.asarray(current)[:2]*scale).astype(int)))
            commanded = bool(np.linalg.norm(np.asarray(action)[:2]) > .2)
            self.movement_commands += int(commanded)
            self.responsive_movement_steps += int(commanded and distance > .005)
            self.turn_commands += int(abs(action[2]) > .1)
            self.fire_commands += int(action[3] >= .5)
            if self.terminal_step is None and (current[4] <= .0005 or
                                               (not self.endless and current[-2]*ENTITY_SLOTS >= ENTITY_SLOTS-.01)):
                self.terminal_step = self.steps
            if reference_current is not None:
                ref_distance = float(np.linalg.norm((np.asarray(reference_current)[:2]-np.asarray(reference_previous)[:2])*scale))
                self.reference_path_tiles += ref_distance
                old_reference_npcs, new_reference_npcs = entities(reference_previous), entities(reference_current)
                ref_motion = np.linalg.norm((new_reference_npcs[:, :2]-old_reference_npcs[:, :2])*scale, axis=-1)
                ref_revived = (old_reference_npcs[:, 5] < .5) & (new_reference_npcs[:, 5] >= .5)
                if self.endless:
                    ref_motion[ref_revived] = 0.
                self.reference_npc_path_tiles += ref_motion
                self.reference_respawn_count += int(ref_revived.sum())
                self.reference_player_alive = bool(reference_current[4] > .0005)
                self.reference_event_counts['kill'] += int(round(max(0.,
                    (reference_current[-2]-reference_previous[-2])*ENTITY_SLOTS)))
                self.reference_movement_steps += int(commanded and ref_distance > .005)
                if self.reference_terminal_step is None and (reference_current[4] <= .0005 or
                        (not self.endless and reference_current[-2]*ENTITY_SLOTS >= ENTITY_SLOTS-.01)):
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
                           out=np.zeros(ENTITY_SLOTS), where=self.reference_npc_path_tiles > 1e-8)
        frozen_npcs = np.flatnonzero(required_npcs & (ratios < .1)).tolist()
        npc_response_pass = not frozen_npcs
        model_ended_early = self.terminal_step is not None and self.terminal_step < self.requested_steps
        reference_ended_early = source_terminal is not None and source_terminal < self.requested_steps
        combat_pass = (not self.endless or
                       (self.event_counts['shot'] >= min(10, max(2, self.requested_steps//30))
                        and self.event_counts['hit'] >= 2 and self.event_counts['kill'] >= 1
                        and self.respawn_count >= 1 and self.reference_respawn_count >= 1))
        alive_pass = (not self.endless or (self.model_player_alive and self.reference_player_alive))
        qualifying = bool(active_scenario and completed and meaningful_motion
                          and npc_response_pass and not model_ended_early
                          and reference_completed and not reference_ended_early
                          and combat_pass and alive_pass)
        integrity = self.first_failure is None
        return {"completed_requested_steps": completed, "checked_steps": self.steps,
                "requested_steps": self.requested_steps,
                "requested_seconds": round(self.requested_steps*self.dt, 6),
                "checked_simulated_seconds": round(self.steps*self.dt, 6),
                "integrity_pass": integrity, "qualifying_active_session": qualifying,
                "behavioral_checks_pass": bool(meaningful_motion and npc_response_pass and combat_pass and alive_pass),
                "state_integrity_active_requested_pass": bool(integrity and qualifying
                    and self.steps*self.dt >= self.requested_steps*self.dt-1e-6),
                "state_integrity_active_300s_pass": bool(integrity and qualifying and self.steps*self.dt >= 300.-1e-6),
                "first_failure": self.first_failure, "violation_counts": dict(self.counts),
                "failure_examples": self.examples, "model_terminal_step": self.terminal_step,
                "reference_terminal_step": source_terminal,
                "endless": self.endless,
                "combat_coverage": {"events": dict(self.event_counts),
                                    "model_respawns": self.respawn_count,
                                    "reference_respawns": self.reference_respawn_count,
                                    "pass": combat_pass,
                                    "note": "Endless qualification requires actual shots, hits, a kill and a respawn in both paired trajectories."},
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
                "qualification_note": "Idle/turn/wall, early terminal episodes, frozen predictions and endless sessions without combat/respawns cannot establish active gameplay for the requested duration."}
