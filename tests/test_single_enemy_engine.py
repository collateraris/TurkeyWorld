"""The source has exactly one persistent enemy and a predictable spawn cycle."""
import copy
import unittest

import numpy as np

from turkeyworld.engine import BEHAVIOR_VERSION, ENTITY_SLOTS, STATE_DIM, Game
from turkeyworld.spawn import SPAWN_ANCHORS, deterministic_spawn_tile


class SingleEnemyEngineTests(unittest.TestCase):
    def test_one_persistent_slot_and_twenty_values_finite_and_endless(self):
        self.assertEqual((ENTITY_SLOTS, STATE_DIM), (1, 20))
        self.assertEqual(BEHAVIOR_VERSION, 'deterministic_enemy_v4_single_spawn_cycle')
        for endless in (False, True):
            game = Game(endless=endless)
            self.assertEqual([enemy['id'] for enemy in game.enemies], [0])
            self.assertEqual(game.state_vector().shape, (20,))
            restored = Game(endless=endless)
            restored.set_state(game.get_state())
            np.testing.assert_array_equal(restored.state_vector(), game.state_vector())
            with self.assertRaises(ValueError): restored.set_state_vector(np.zeros(90))
            old = copy.deepcopy(game.get_state()); old['enemies'] *= 8
            with self.assertRaises(ValueError): restored.set_state(old)

    def test_real_kill_respawn_and_long_steps_never_add_another_enemy(self):
        game = Game(endless=True)
        kills = respawns = 0
        previous = True
        for _ in range(500):
            event = game.step([0, 0, 0, 1])
            kills += int(event['kill'])
            alive = game.enemies[0]['alive']
            respawns += int(alive and not previous)
            previous = alive
            self.assertEqual(len(game.enemies), 1)
            self.assertLessEqual(sum(enemy['alive'] for enemy in game.enemies), 1)
            self.assertTrue(np.isfinite(game.state_vector()).all())
        self.assertGreaterEqual(kills, 1)
        self.assertGreaterEqual(respawns, 1)
        self.assertEqual(game.state_vector()[-2], game.kills)

    def test_coarse_spawn_phase_repeats_and_ignores_source_rng(self):
        game = Game(endless=True)
        walkable = game.map == 0
        choose = lambda tick: deterministic_spawn_tile(walkable, (8.5, 8.5), [(5.5, 2.5)], [False], 0, tick)
        self.assertEqual(choose(0), choose(89))
        self.assertNotEqual(choose(89), choose(90))
        self.assertEqual(choose(0), choose(720))
        for x, y in SPAWN_ANCHORS:
            self.assertEqual(game.map[y, x], 0)
        first, second = Game(seed=1, endless=True), Game(seed=999, endless=True)
        first.rng.random(500)
        for _ in range(180):
            first.step([0, 0, 0, 1]); second.step([0, 0, 0, 1])
            np.testing.assert_array_equal(first.state_vector(), second.state_vector())

    def test_cycle_keeps_wall_and_player_distance_guards_with_static_fallback(self):
        floor = np.zeros((16, 16), bool); floor[14, 14] = True
        self.assertEqual(deterministic_spawn_tile(floor, (1.5, 1.5), [(5.5, 2.5)], [False], 0, 0), (14.5, 14.5))
        self.assertIsNone(deterministic_spawn_tile(floor, (14.5, 14.5), [(5.5, 2.5)], [False], 0, 0))
        self.assertEqual(Game().render(32, False).shape, (32, 32, 3))


if __name__ == '__main__':
    unittest.main()
