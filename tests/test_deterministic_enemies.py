"""Source enemy behavior depends on state, game time and actions, never RNG."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
import copy
import unittest

import numpy as np

from turkeyworld.engine import BEHAVIOR_VERSION, Game
from turkeyworld.spawn import (SPAWN_ANCHORS, SPAWN_CYCLE_SECONDS, SPAWN_PHASE_TICKS,
                               deterministic_spawn_tile, spawn_tile_rank)


class DeterministicEnemySourceTests(unittest.TestCase):
    def test_initial_positions_and_animation_do_not_depend_on_seed(self):
        first, second = Game(seed=1, endless=True), Game(seed=999, endless=True)
        np.testing.assert_array_equal(first.state_vector(), second.state_vector())
        self.assertEqual(BEHAVIOR_VERSION, 'deterministic_enemy_v4_single_spawn_cycle')
        self.assertEqual(first.get_state()['behavior_version'], BEHAVIOR_VERSION)
        self.assertTrue(all(e['animation_phase'] == 0. for e in first.enemies))

    def test_respawns_and_all_following_decisions_ignore_rng_state(self):
        first, second = Game(seed=2, endless=True), Game(seed=9876, endless=True)
        for game in (first, second):
            game.time = 17.
            game.enemies[0].update(hp=0., alive=False, death_timer=2.99)
        first.rng.random(11)
        second.rng.random(10001)
        rng = np.random.default_rng(411)
        actions = rng.uniform(-1, 1, (180, 4))
        actions[:, 3] = rng.integers(0, 2, len(actions))
        for index, action in enumerate(actions):
            second.rng.random(index + 1)  # Arbitrary hidden-RNG drift cannot change gameplay.
            self.assertEqual(first.step(action), second.step(action))
            np.testing.assert_array_equal(first.state_vector(), second.state_vector())

    def test_coarse_anchor_cycle_is_stable_nearby_and_repeats_after_24_seconds(self):
        game = Game(endless=True)
        positions = [(0., 0.)] * ENTITY_SLOTS
        alive = [False] * ENTITY_SLOTS
        player = (8.5, 8.5)  # All eight anchors satisfy the four-tile exclusion.
        self.assertEqual(SPAWN_CYCLE_SECONDS, 24)
        for x, y in SPAWN_ANCHORS:
            self.assertEqual(game.map[y, x], 0)
        phases = []
        for phase in range(8):
            expected = tuple(value+.5 for value in SPAWN_ANCHORS[phase])
            choices = {deterministic_spawn_tile(game.map == 0, player, positions, alive, 0, tick)
                       for tick in (phase*90, phase*90+1, phase*90+45, phase*90+89)}
            self.assertEqual(choices, {expected})
            phases.append(expected)
            self.assertEqual(deterministic_spawn_tile(game.map == 0, player, positions, alive, 0,
                (phase+8)*SPAWN_PHASE_TICKS), expected)
        self.assertEqual(len(set(phases)), 8)
        # There is one actor; the retained compatibility argument cannot add
        # a hidden ID-dependent phase or hash to the point cycle.
        self.assertEqual(spawn_tile_rank(10, 0, 2*16+2), spawn_tile_rank(10, 7, 2*16+2))

    def test_anchor_safety_skips_walls_player_and_living_npc_then_uses_static_fallback(self):
        walkable = np.ones((16, 16), dtype=bool)
        walkable[[0, -1], :] = False
        walkable[:, [0, -1]] = False
        # Preferred0 is too close to player; anchor1 is wall; anchor2 occupied.
        walkable[2, 8] = False
        positions, alive = [(13.5, 2.5)], [True]
        self.assertEqual(deterministic_spawn_tile(walkable, (2.5, 2.5), positions, alive, 0, 0),
                         (13.5, 8.5))
        for x, y in SPAWN_ANCHORS:
            walkable[y, x] = False
        # The first legal non-anchor tile is selected in row-major order.
        self.assertEqual(deterministic_spawn_tile(walkable, (2.5, 2.5), [], [], 0, 0), (6.5, 1.5))
        self.assertEqual(deterministic_spawn_tile(walkable, (2.5, 2.5), [], [], 7, 700), (6.5, 1.5))
        self.assertIsNone(deterministic_spawn_tile(np.zeros((16, 16), dtype=bool),
                                                 (2.5, 2.5), [], [], 0, 0))

    def test_same_elapsed_spawn_time_uses_same_anchor_for_different_step_sizes(self):
        choices = []
        for dt in (1/60, 1/30, 1/20):
            game = Game(endless=True)
            game.time = 3.1-dt
            game.enemies[0].update(hp=0., alive=False, death_timer=3.-dt/2)
            game.step([0, 0, 0, 0], dt=dt)
            enemy = game.enemies[0]
            self.assertTrue(enemy['alive'])
            self.assertAlmostEqual(game.time, 3.1)
            choices.append((enemy['x'], enemy['y']))
        self.assertEqual(len(set(choices)), 1)

    def test_new_spawn_holds_center_and_zero_phase_until_following_tick(self):
        game = Game(endless=True)
        game.enemies[0].update(alive=False, hp=0., death_timer=2.99, animation_phase=.7)
        old = copy.deepcopy(game.enemies)
        expected = deterministic_spawn_tile(game.map == 0, (game.player['x'], game.player['y']),
            [(e['x'], e['y']) for e in old], [e['alive'] for e in old], 0, 1)
        game.step([0, 0, 0, 0])
        enemy = game.enemies[0]
        self.assertEqual((enemy['x'], enemy['y']), expected)
        self.assertEqual((enemy['vx'], enemy['vy'], enemy['animation_phase']), (0., 0., 0.))
        self.assertEqual((enemy['hp'], enemy['death_timer']), (60., 0.))
        game.step([0, 0, 0, 0])
        self.assertAlmostEqual(enemy['animation_phase'], 2.6 / 30)


class SharedSpawnPolicyTests(unittest.TestCase):
    def test_optional_tensor_policy_matches_source_and_is_independent_of_logits(self):
        import torch
        from turkeyworld.spawn import decode_spawn_positions
        from turkeyworld.structured_state import structured_transition

        for game_time in (0., 17., 123.4):
            with self.subTest(game_time=game_time):
                game = Game(seed=333, endless=True)
                game.time = game_time
                for e in game.enemies:
                    e.update(hp=0., alive=False, death_timer=2.99, animation_phase=.7)
                before = torch.from_numpy(game.state_vector())[None]
                logits = torch.randn(1, ENTITY_SLOTS, 256)
                positions, valid = decode_spawn_positions(logits, before, torch.tensor(game.map == 0),
                                                          1 / 30, deterministic=True)
                other, other_valid = decode_spawn_positions(-logits, before, torch.tensor(game.map == 0),
                                                            1 / 30, deterministic=True)
                self.assertTrue(torch.equal(positions, other))
                self.assertTrue(torch.equal(valid, other_valid))
                self.assertTrue(bool(valid.all()))
                targets = torch.full((1, ENTITY_SLOTS+1), -10.)
                targets[:, ENTITY_SLOTS] = 10.
                typed = structured_transition(before, before.clone(), torch.zeros(1, 4), targets,
                    torch.full((1, ENTITY_SLOTS), -10.), endless=True, spawn_positions=positions, spawn_valid=valid)
                game.step([0, 0, 0, 0])
                torch.testing.assert_close(typed['state'], torch.from_numpy(game.state_vector())[None],
                                           rtol=0, atol=1e-6)

    def test_learned_runtime_decoder_remains_deterministic_and_uses_head_preferences(self):
        import torch
        from turkeyworld.spawn import decode_spawn_positions

        game = Game(endless=True)
        game.enemies[0].update(hp=0., alive=False, death_timer=2.99)
        state = torch.from_numpy(game.state_vector())[None]
        logits = torch.zeros(1, ENTITY_SLOTS, 256)
        logits[0, 0, 1 * 16 + 14] = 100.
        first, available = decode_spawn_positions(logits, state, torch.tensor(game.map == 0), 1 / 30)
        second, second_available = decode_spawn_positions(logits, state, torch.tensor(game.map == 0), 1 / 30)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(available, second_available))
        torch.testing.assert_close(first[0, 0], torch.tensor([14.5 / 16, 1.5 / 16]))


if __name__ == '__main__':
    unittest.main()
