"""Behavioral checks for the ground-truth environment used as training oracle."""
import json
import unittest
import numpy as np
from turkeyworld.engine import Game


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

    def test_nearest_target_is_hit_and_death_persists(self):
        game = Game()
        first = game.step([0, 0, 0, 1])
        self.assertEqual(first['hit_id'], 0)
        self.assertEqual(game.enemies[1]['hp'], 60)
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


if __name__ == '__main__':
    unittest.main()
