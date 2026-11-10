"""Integrity-evaluator fault injection and active-session qualification."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
import unittest
import math
import numpy as np

from turkeyworld.engine import Game
from turkeyworld.evaluate_rollout import build_action_tape
from turkeyworld.runtime_checks import RolloutMonitor, check_transition


class RolloutValidationTests(unittest.TestCase):
    def test_reference_combat_transitions_satisfy_integrity_contract(self):
        game = Game(seed=5)
        for index in range(90):
            previous = game.state_vector()
            action = np.array([0., 0., 0., float(index < 45)])
            event = game.step(action)
            failures = check_transition(game.map, previous, game.state_vector(), action, event)
            self.assertEqual(failures, [], f"Reference failure at step {index}: {failures}")

    def test_shot_without_ammo_spend_is_rejected_and_inputs_are_not_repaired(self):
        game = Game()
        previous = game.state_vector()
        action = np.array([0., 0., 0., 1.])
        event = game.step(action)
        corrupted = game.state_vector()
        corrupted[5] = previous[5]
        before = corrupted.copy()
        first = check_transition(game.map, previous, corrupted, action, event)
        second = check_transition(game.map, previous, corrupted, action, event)
        self.assertIn("shot_ammo", {failure["code"] for failure in first})
        self.assertEqual(first, second)
        np.testing.assert_array_equal(corrupted, before)

    def test_wall_tunneling_is_found_even_with_clear_endpoints(self):
        game = Game()
        tilemap = np.zeros((16, 16), dtype=np.uint8)
        tilemap[[0, -1], :] = 1
        tilemap[:, [0, -1]] = 1
        tilemap[:, 4] = 1
        previous = game.state_vector()
        current = previous.copy()
        current[:2] = [6.5/16, 2.5/16]
        current[-1] += (1/30)/300
        failures = check_transition(tilemap, previous, current, [1, 0, 0, 0], [0]*4)
        codes = {failure["code"] for failure in failures}
        self.assertIn("wall_tunneling", codes)
        self.assertIn("player_speed", codes)

    def test_legal_speed_in_the_wrong_direction_is_still_invalid(self):
        game = Game()
        previous = game.state_vector()
        current = previous.copy()
        current[0] -= .04/16
        current[-1] += (1/30)/300
        failures = check_transition(game.map, previous, current, [1, 0, 0, 0], [0]*4)
        self.assertIn("movement_direction", {failure["code"] for failure in failures})

    def test_reference_axis_separated_corner_slide_is_not_false_tunneling(self):
        game = Game()
        snapshot = game.get_state()
        snapshot["player"].update(x=4-.20001*math.cos(math.radians(50)),
                                  y=3-.20001*math.sin(math.radians(50)), angle=0.)
        game.set_state(snapshot)
        previous = game.state_vector()
        action = np.array([-1., 1., 0., 0.])
        event = game.step(action)
        failures = check_transition(game.map, previous, game.state_vector(), action, event)
        self.assertEqual(failures, [])

    def test_dead_slot_cannot_revive(self):
        game = Game()
        previous = game.state_vector()
        previous[12:14] = 0
        previous[-2] = 1/8
        current = previous.copy()
        current[12:14] = [.25, 1]
        current[-1] += (1/30)/300
        failures = check_transition(game.map, previous, current, [0]*4, [0]*4)
        self.assertIn("revival", {failure["code"] for failure in failures})

    def test_silent_attack_head_cannot_make_player_immortal(self):
        game = Game()
        previous = game.state_vector()
        previous[8:10] = [3./16, 2.5/16]
        current = previous.copy()
        current[-1] += (1/30)/300
        failures = check_transition(game.map, previous, current, [0]*4, [0]*4)
        self.assertIn("missed_ready_attack", {failure["code"] for failure in failures})

    def test_none_target_cannot_suppress_a_mandatory_centered_hit(self):
        game = Game()
        previous = game.state_vector()
        action = np.array([0., 0., 0., 1.])
        event = game.step(action)
        current = game.state_vector()
        current[12] = previous[12]
        event["hit"], event["hit_id"] = False, -1
        failures = check_transition(game.map, previous, current, action, event)
        self.assertIn("missed_visible_hit", {failure["code"] for failure in failures})

    def test_frozen_attack_cooldown_is_rejected(self):
        game = Game()
        previous = game.state_vector()
        previous[14] = 1.  # NPC0 normalized attack cooldown.
        current = previous.copy()
        current[-1] += (1/30)/300
        failures = check_transition(game.map, previous, current, [0]*4, [0]*4)
        self.assertIn("npc_attack_timer", {failure["code"] for failure in failures})

    def test_frozen_model_cannot_qualify_by_advancing_only_its_clock(self):
        tape = build_action_tape(5, "explore", 120, 1/30)
        monitor = RolloutMonitor(tape["map"], "explore", 120)
        previous = tape["initial_state"].copy()
        for index in range(120):
            current = previous.copy()
            current[-1] += (1/30)/300
            # This apparently sane state still ignores all movement commands.
            monitor.observe(previous, current, [1., 0., 0., 0.], [0]*4,
                            tape["reference_states"][index], tape["reference_states"][index+1])
            previous = current
        report = monitor.report(reference_completed=True)
        self.assertTrue(report["integrity_pass"])
        self.assertTrue(report["completed_requested_steps"])
        self.assertFalse(report["qualifying_active_session"])
        self.assertFalse(report["state_integrity_active_300s_pass"])

    def test_reference_terminal_episode_never_counts_as_long_active_session(self):
        game = Game()
        monitor = RolloutMonitor(game.map, "explore", 9000)
        # A sufficient activity summary still cannot erase an early reference
        # terminal state; these counters emulate the end of a recorded session.
        monitor.steps = 9000
        monitor.path_tiles = monitor.reference_path_tiles = 100
        monitor.movement_commands = monitor.responsive_movement_steps = monitor.reference_movement_steps = 1000
        monitor.visited_cells = {(index, 2) for index in range(10)}
        self.assertTrue(monitor.report(reference_completed=True)["qualifying_active_session"])
        terminal = monitor.report(reference_completed=True, reference_terminal_step=100)
        self.assertFalse(terminal["qualifying_active_session"])
        self.assertFalse(terminal["state_integrity_active_300s_pass"])
        final_step_terminal = monitor.report(reference_completed=True, reference_terminal_step=9000)
        self.assertTrue(final_step_terminal["qualifying_active_session"])

    def test_moving_player_with_frozen_living_npcs_does_not_qualify(self):
        tape = build_action_tape(5, "explore", 300, 1/30)
        monitor = RolloutMonitor(tape["map"], "explore", 300)
        previous = tape["initial_state"].copy()
        for index, action in enumerate(tape["actions"]):
            current = previous.copy()
            current[:4] = tape["reference_states"][index+1, :4]
            current[-1] += (1/30)/300
            monitor.observe(previous, current, action, [0]*4,
                            tape["reference_states"][index], tape["reference_states"][index+1])
            previous = current
        report = monitor.report(reference_completed=True)
        self.assertGreater(report["coverage"]["path_tiles"], 8)
        self.assertTrue(report["npc_responsiveness"]["frozen_alive_ids"])
        self.assertFalse(report["behavioral_checks_pass"])
        self.assertFalse(report["qualifying_active_session"])

    def test_diagnostic_idle_is_not_active_play(self):
        game = Game()
        monitor = RolloutMonitor(game.map, "idle", 9000)
        monitor.steps = 9000
        self.assertTrue(monitor.report()["integrity_pass"])
        self.assertFalse(monitor.report()["qualifying_active_session"])


if __name__ == "__main__":
    unittest.main()
