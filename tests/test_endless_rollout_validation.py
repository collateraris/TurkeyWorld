"""Endless evaluator contract and source-only 300-second qualification checks."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
import unittest

import numpy as np

from turkeyworld.engine import Game
from turkeyworld.evaluate_rollout import build_action_tape
from turkeyworld.runtime_checks import RolloutMonitor, check_transition


class EndlessRolloutValidationTests(unittest.TestCase):
    def respawn_pair(self):
        game = Game(seed=31, endless=True)
        game.enemies[0].update(alive=False, hp=0., death_timer=3.-1/30)
        game.kills = 17
        previous = game.state_vector()
        action = np.zeros(4, dtype=np.float32)
        event = game.step(action)
        return game, previous, game.state_vector(), action, event

    def codes(self, game, previous, current, action, events):
        return {failure['code'] for failure in
                check_transition(game.map, previous, current, action, events, endless=True)}

    def test_legal_respawn_passes_without_mutating_either_state(self):
        game, previous, current, action, events = self.respawn_pair()
        old_copy, new_copy = previous.copy(), current.copy()
        self.assertEqual(self.codes(game, previous, current, action, events), set())
        np.testing.assert_array_equal(previous, old_copy)
        np.testing.assert_array_equal(current, new_copy)

    def test_early_respawn_or_respawn_after_player_death_is_rejected(self):
        for timer, hp in [(1., 1.), (5., 0.)]:
            game, previous, current, action, events = self.respawn_pair()
            previous[16], previous[4] = timer, hp
            with self.subTest(timer=timer, hp=hp):
                self.assertIn('respawn_eligibility', self.codes(game, previous, current, action, events))
                self.assertIn('revival', self.codes(game, previous, current, action, events))

    def test_respawn_near_player_in_wall_or_with_wrong_health_is_rejected(self):
        for field, value, code in [(slice(GLOBAL_DIM, GLOBAL_DIM+2), [3.5/16, 2.5/16], 'respawn_player_distance'),
                                   (slice(GLOBAL_DIM, GLOBAL_DIM+2), [4.5/16, 3.5/16], 'respawn_wall'),
                                   (12, .42, 'respawn_health')]:
            game, previous, current, action, events = self.respawn_pair()
            current[field] = value
            with self.subTest(code=code):
                self.assertIn(code, self.codes(game, previous, current, action, events))

    def test_spawn_timer_cannot_jump_and_elapsed_slots_cannot_stay_dead(self):
        game, previous, current, action, events = self.respawn_pair()
        current[8:18] = previous[8:18]
        current[10:12] = 0.
        current[16] = 5.
        self.assertIn('missed_respawn', self.codes(game, previous, current, action, events))
        previous[16] = 1.
        self.assertIn('npc_death_timer', self.codes(game, previous, current, action, events))

    def test_cumulative_counter_and_infinite_ammo_are_checked(self):
        game, previous, current, action, events = self.respawn_pair()
        current[-2] = 0.
        self.assertIn('kill_count', self.codes(game, previous, current, action, events))
        current[-2] = previous[-2]
        current[5] -= 1/99
        self.assertIn('shot_ammo', self.codes(game, previous, current, action, events))

    def test_respawn_teleports_do_not_count_as_npc_responsiveness_or_a_win(self):
        game, previous, current, action, events = self.respawn_pair()
        monitor = RolloutMonitor(game.map, 'mixed', 9000, endless=True)
        monitor.observe(previous, current, action, events, previous, current)
        self.assertIsNone(monitor.terminal_step)
        self.assertEqual(monitor.npc_path_tiles[0], 0.)
        self.assertEqual(monitor.reference_npc_path_tiles[0], 0.)
        self.assertEqual(monitor.respawn_count, 1)

    def test_endless_motion_without_combat_cannot_qualify(self):
        game = Game(endless=True)
        monitor = RolloutMonitor(game.map, 'mixed', 9000, endless=True)
        monitor.steps = 9000
        monitor.path_tiles = monitor.reference_path_tiles = 100.
        monitor.movement_commands = monitor.responsive_movement_steps = monitor.reference_movement_steps = 1000
        monitor.visited_cells = {(index, 2) for index in range(10)}
        report = monitor.report()
        self.assertTrue(report['integrity_pass'])
        self.assertFalse(report['combat_coverage']['pass'])
        self.assertFalse(report['state_integrity_active_300s_pass'])

    def test_source_survival_controller_records_genuine_five_minute_combat(self):
        # This validates the evaluator's source/action tape, never a trained model.
        tape = build_action_tape(10000, 'mixed', 9000, 1/30, endless=True)
        self.assertIsNone(tape['reference_terminal_step'])
        self.assertEqual(tape['controller'], 'survival_patrol_with_combat')
        self.assertGreater(float(tape['reference_states'][-1, -2])*ENTITY_SLOTS, 8)
        monitor = RolloutMonitor(tape['map'], 'mixed', 9000, endless=True)
        states = tape['reference_states']
        for index, action in enumerate(tape['actions']):
            self.assertEqual(monitor.observe(states[index], states[index+1], action,
                                             tape['reference_events'][index],
                                             states[index], states[index+1]), [])
        report = monitor.report(True, tape['reference_terminal_step'])
        self.assertTrue(report['state_integrity_active_300s_pass'])
        self.assertGreater(report['combat_coverage']['model_respawns'], 8)


if __name__ == '__main__':
    unittest.main()
