"""Source state serialization and conditional labels share the death tick clock."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.engine import Game
from turkeyworld.rssm_combat_branches import (_source_step_at_own_state,
    normalize_combat_branch_protocol)
from turkeyworld.structured_state import structured_transition


def corpse_game():
    game = Game(seed=17, endless=True)
    game.enemies[0].update(alive=False, hp=0., death_timer=0.)
    game.kills = 1
    return game


def typed_step(previous, source_target, *, dt=1/30):
    state = torch.from_numpy(previous.copy())[None]
    # Supply only a chosen spawn proposal; the neural event rules still decide
    # eligibility from exactly the supplied own previous state and clock.
    spawn = torch.from_numpy(source_target[8:10].copy())[None, None]
    return structured_transition(state, state.clone(), torch.zeros(1, 4),
        torch.tensor([[-10., 10.]]), torch.full((1, 1), -10.), endless=True,
        training=False, spawn_positions=spawn, spawn_valid=torch.ones(1, 1, dtype=torch.bool), dt=dt)


class SourceTimerInteropTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_literal_normalized_state_roundtrip_and_strict_guard_are_unchanged(self):
        game = corpse_game(); protocol = normalize_combat_branch_protocol({'version': 2})
        for q in (np.float32(4.944444179534912), np.float32(89/18),
                  np.nextafter(np.float32(89/18), np.float32(np.inf))):
            previous = game.state_vector(); previous[16] = q
            original = previous.copy()
            game.set_state_vector(previous)
            np.testing.assert_array_equal(game.state_vector(), original)
            # The scalar setter remains the raw q*.6 decode, not a repaired
            # 89/30 state that can differ by one normalized float32 ULP.
            self.assertEqual(game.enemies[0]['death_timer'], float(q)*.6)
            label = _source_step_at_own_state(previous, np.zeros(4), game.map, protocol, 1/30)
            np.testing.assert_array_equal(label['previous'], original)
            np.testing.assert_array_equal(previous, original)
            self.assertEqual(label['target'][13], 1.)
            predicted = typed_step(previous, label['target'])
            self.assertTrue(bool(predicted['respawn_flags'][0, 0]))
        invalid = corpse_game().state_vector(); invalid[16] = np.float32(5.001)
        with self.assertRaisesRegex(ValueError, 'noncanonical_direct_own_state_no_repair'):
            _source_step_at_own_state(invalid, np.zeros(4), game.map, protocol, 1/30)
        self.assertEqual(protocol['canonical_atol'], 2e-7)

    def test_30hz_death_zero_has_exactly_90_ticks_and_conditional_queries_keep_late_birth(self):
        source = corpse_game(); own = source.state_vector()
        protocol = normalize_combat_branch_protocol({'version': 2})
        births = []
        for tick in range(1, 91):
            previous = own.copy()
            label = _source_step_at_own_state(previous, np.zeros(4), source.map, protocol, 1/30)
            result = typed_step(previous, label['target'])
            own = result['state'][0].detach().numpy()
            source.step(np.zeros(4))
            self.assertEqual(bool(label['target'][13]), bool(result['respawn_flags'][0, 0]))
            self.assertEqual(bool(source.enemies[0]['alive']), tick == 90)
            if bool(result['respawn_flags'][0, 0]): births.append(tick)
            if tick < 90:
                self.assertEqual(own[13], 0.)
        self.assertEqual(births, [90])

    def test_four_step_canonical_label_window_retains_spawn_after_initial_clock(self):
        source = corpse_game(); protocol = normalize_combat_branch_protocol({'version': 2})
        for before_ticks in (86, 87, 88, 89):
            own = source.state_vector(); own[16] = np.float32(before_ticks/18)
            born = []
            for offset in range(1, 5):
                label = _source_step_at_own_state(own, np.zeros(4), source.map, protocol, 1/30)
                result = typed_step(own, label['target'])
                if bool(result['respawn_flags'][0, 0]): born.append(offset)
                own = result['state'][0].detach().numpy()
            self.assertEqual(born, [90-before_ticks])

    def test_native_Source_30hz_event_pose_and_rgb_match_legacy_without_quantizing_other_clocks(self):
        initial = corpse_game().get_state()
        actions = np.zeros((128, 4)); actions[:, 2] = .08; actions[::7, 3] = 1.
        expected = []
        legacy = corpse_game(); legacy.set_state(initial)
        def old_elapsed(value, *, duration, dt, increasing=False, maximum=None):
            self.assertTrue(increasing)
            result = max(0., value+dt)
            return min(result, maximum) if maximum is not None else result
        with patch('turkeyworld.engine.advance_timer_seconds', side_effect=old_elapsed):
            for action in actions:
                event = legacy.step(action)
                expected.append((copy.deepcopy(event), legacy.state_vector().copy(), legacy.render(32, overlays=False)))
        actual = corpse_game(); actual.set_state(initial)
        for action, (event, state, rgb) in zip(actions, expected):
            self.assertEqual(actual.step(action), event)
            # Native Source float32 tapes and pixels remain unchanged; only
            # internal double death timer representation can differ.
            np.testing.assert_array_equal(actual.state_vector(), state)
            np.testing.assert_array_equal(actual.render(32, overlays=False), rgb)

    def test_offgrid_and_generic_dt_keep_existing_deadline_semantics(self):
        for initial, dt, count in ((2.84, 1/30, 5), (2.5, .25, 2), (2.95, .017, 3),
                                   (np.float32(2.5), np.float32(.25), 2)):
            game = corpse_game(); game.enemies[0]['death_timer'] = initial
            for tick in range(1, count+1):
                game.step(np.zeros(4), dt=dt)
                self.assertEqual(game.enemies[0]['alive'], tick == count)
        dead_player = corpse_game(); dead_player.player['hp'] = 0.
        dead_player.enemies[0]['death_timer'] = float(np.float32(89/18))*.6
        dead_player.step(np.zeros(4))
        self.assertFalse(dead_player.enemies[0]['alive'])


if __name__ == '__main__': unittest.main()
