"""Behavioral checks for the ground-truth environment used as training oracle."""
import json
import math
import unittest
import numpy as np
from turkeyworld.engine import ENTITY_SLOTS, Game


class CombatAndReplayTests(unittest.TestCase):
    def test_same_input_after_restore_reproduces_trajectory(self):
        game = Game(seed=4)
        snapshot = json.loads(json.dumps(game.get_state()))
        rng = np.random.default_rng(44)
        actions = rng.uniform(-1, 1, (40, 4))
        actions[:, 3] = rng.integers(0, 2, 40)
        expected = []
        for action in actions:
            event = game.step(action)
            expected.append((game.state_vector(), event))
        game.set_state(snapshot)
        for action, (state, event) in zip(actions, expected):
            self.assertEqual(game.step(action), event)
            np.testing.assert_array_equal(game.state_vector(), state)

    def test_wall_blocks_bullets(self):
        game = Game()
        state = game.get_state()
        state['player'].update(x=2.5, y=4.5, angle=0)
        state['enemies'][0].update(x=6.5, y=4.5)
        game.set_state(state)
        hp = game.enemies[0]['hp']
        event = game.step([0, 0, 0, 1])
        self.assertTrue(event['shot'])
        self.assertEqual(game.enemies[0]['hp'], hp)

    def test_only_target_is_hit_and_death_persists(self):
        game = Game()
        first = game.step([0, 0, 0, 1])
        self.assertEqual(first['hit_id'], 0)
        self.assertEqual(len(game.enemies), 1)
        self.assertEqual(game.enemies[0]['hp'], 25)
        for _ in range(7):
            game.step([0, 0, 0, 0])
        second = game.step([0, 0, 0, 1])
        self.assertTrue(second['kill'])
        self.assertEqual(second['killed_id'], 0)
        position = (game.enemies[0]['x'], game.enemies[0]['y'])
        for _ in range(300):
            game.step([0, 0, .5, 0])
        self.assertFalse(game.enemies[0]['alive'])
        self.assertEqual(game.enemies[0]['hp'], 0)
        self.assertEqual(position, (game.enemies[0]['x'], game.enemies[0]['y']))

    def test_cooldown_and_empty_ammo(self):
        game = Game()
        self.assertTrue(game.step([0, 0, 0, 1])['shot'])
        self.assertFalse(game.step([0, 0, 0, 1])['shot'])
        game.player['ammo'] = 0
        game.player['cooldown'] = 0
        self.assertFalse(game.step([0, 0, 0, 1])['shot'])

    def test_wall_prevents_player_tunnelling(self):
        game = Game()
        state = game.get_state()
        state['player'].update(x=2.5, y=4.5, angle=0)
        game.set_state(state)
        for _ in range(20):
            game.step([1, 0, 0, 0], dt=.25)
        self.assertLess(game.player['x'], 4)
        self.assertGreater(game.player['x'], 3.7)

    def test_render_and_state_adapter(self):
        game = Game()
        before = game.state_vector()
        game.set_state_vector(before)
        np.testing.assert_array_equal(game.state_vector(), before)
        frame = game.render(size=128)
        self.assertEqual(frame.shape, (128, 128, 3))
        self.assertEqual(frame.dtype, np.uint8)


class EndlessCombatTests(unittest.TestCase):
    def test_repeated_kills_reuse_the_single_slot_and_keep_combat_running(self):
        game = Game(seed=19, endless=True)
        for kill_count in range(1, 13):
            # Stage one clear shot in the same persistent enemy slot.
            for enemy in game.enemies:
                enemy.update(x=12.5, y=12.5, attack_cooldown=100.0)
            game.enemies[0].update(x=5.5, y=2.5, hp=35.0)
            game.player['cooldown'] = 0.0
            event = game.step([0, 0, 0, 1])
            self.assertTrue(event['kill'])
            self.assertEqual(game.kills, kill_count)
            self.assertFalse(game.done)
            self.assertFalse(game.enemies[0]['alive'])
            for _ in range(12):
                game.step([0, 0, 0, 0], dt=.25)
            self.assertTrue(game.enemies[0]['alive'])
            self.assertEqual(game.enemies[0]['hp'], 60.0)
            self.assertEqual(len(game.enemies), ENTITY_SLOTS)
            self.assertEqual([enemy['id'] for enemy in game.enemies], [0])
        self.assertEqual(game.player['ammo'], 99)

    def test_corpses_wait_then_respawn_in_separate_reachable_locations(self):
        game = Game(seed=23, endless=True)
        for enemy in game.enemies:
            enemy.update(alive=False, hp=0.0, death_timer=0.0)
        game.kills = ENTITY_SLOTS
        for _ in range(11):
            game.step([0, 0, 0, 0], dt=.25)
        self.assertTrue(all(not enemy['alive'] for enemy in game.enemies))
        game.step([0, 0, 0, 0], dt=.25)
        field = game._navigation()
        for index, enemy in enumerate(game.enemies):
            self.assertTrue(enemy['alive'])
            self.assertTrue(game._walkable(enemy['x'], enemy['y'], game.ENEMY_RADIUS))
            self.assertLess(field[int(enemy['y']), int(enemy['x'])], 10000)
            # A new opponent may move toward the player in its first tick.
            self.assertGreaterEqual(math.hypot(enemy['x'] - game.player['x'],
                                               enemy['y'] - game.player['y']),
                                    game.SPAWN_DISTANCE - .25)
            for other in game.enemies[:index]:
                self.assertGreater(math.hypot(enemy['x'] - other['x'], enemy['y'] - other['y']), .5)

    def test_ammunition_is_unlimited_but_fire_cooldown_still_applies(self):
        game = Game(endless=True)
        game.player['ammo'] = 0
        self.assertTrue(game.step([0, 0, 0, 1])['shot'])
        self.assertFalse(game.step([0, 0, 0, 1])['shot'])
        for _ in range(7):
            game.step([0, 0, 0, 0])
        self.assertTrue(game.step([0, 0, 0, 1])['shot'])
        self.assertEqual(game.player['ammo'], 0)

    def test_no_respawn_or_combat_after_player_death(self):
        game = Game(endless=True)
        game.player['hp'] = 0.0
        for enemy in game.enemies:
            enemy.update(alive=False, hp=0.0, death_timer=game.RESPAWN_DELAY)
        self.assertTrue(game.done)
        for _ in range(20):
            events = game.step([1, 0, 1, 1], dt=.25)
            self.assertFalse(any(events[name] for name in ('shot', 'hit', 'kill', 'player_hit')))
        self.assertTrue(all(not enemy['alive'] for enemy in game.enemies))

    def test_restore_reproduces_respawns_and_following_combat(self):
        game = Game(seed=31, endless=True)
        game.enemies[0].update(alive=False, hp=0.0, death_timer=2.5)
        game.kills = 17
        snapshot = json.loads(json.dumps(game.get_state()))
        actions = [[0, 0, .2, index % 2] for index in range(24)]
        expected = []
        for action in actions:
            events = game.step(action, dt=.25)
            expected.append((game.get_state(), events))
        restored = Game(seed=99)
        restored.set_state(snapshot)
        self.assertTrue(restored.endless)
        for action, (state, events) in zip(actions, expected):
            self.assertEqual(restored.step(action, dt=.25), events)
            self.assertEqual(restored.get_state(), state)

    def test_vector_round_trip_keeps_cumulative_kills_and_respawn_progress(self):
        game = Game(endless=True)
        game.kills = 37
        game.enemies[0].update(alive=False, hp=0.0, death_timer=2.5)
        vector = game.state_vector()
        game.set_state_vector(vector)
        self.assertEqual(game.kills, 37)
        self.assertAlmostEqual(game.enemies[0]['death_timer'], 2.5, places=6)
        np.testing.assert_allclose(game.state_vector(), vector, atol=1e-7)

    def test_reset_keeps_mode_and_old_snapshots_restore_classic_rules(self):
        game = Game(seed=41, endless=True)
        initial = game.get_state()
        game.kills = 40
        game.enemies[0].update(alive=False, hp=0.0, death_timer=2.5)
        game.reset()
        self.assertEqual(game.get_state(), initial)
        classic = Game().get_state()
        classic.pop('endless')
        game.set_state(classic)
        self.assertFalse(game.endless)
        game.player['ammo'] = 0
        self.assertFalse(game.step([0, 0, 0, 1])['shot'])
        game.kills = ENTITY_SLOTS
        self.assertTrue(game.done)


if __name__ == '__main__':
    unittest.main()
