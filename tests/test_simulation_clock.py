"""Physical Source/normalized state timer agreement without a neural model."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.engine import Game
from turkeyworld.simulation_clock import (
    advance_timer_seconds, advance_normalized_timer, normalized_timer_ready)
from turkeyworld.structured_state import structured_transition


class SimulationClockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_serialized_threshold_phase_recovers_on_advance_not_on_decode(self):
        # The production failure is one float32 ULP below round(89/18).
        q = np.float32(4.944444179534912)
        game = Game(seed=3, endless=True)
        state = game.state_vector()
        state[12:14] = 0
        state[16] = q
        game.set_state_vector(state)
        np.testing.assert_array_equal(game.state_vector(), state)
        self.assertLess(game.enemies[0]['death_timer'] + 1/30, 3 - 1e-8)
        scalar = advance_timer_seconds(game.enemies[0]['death_timer'],
            duration=.6, dt=1/30, increasing=True, maximum=3.)
        self.assertEqual(scalar, 3.)
        value = torch.tensor([q])
        advanced = advance_normalized_timer(value, duration=.6, dt=1/30,
            increasing=True, maximum=3.)
        self.assertEqual(advanced.item(), 5.)
        self.assertTrue(normalized_timer_ready(value, duration=.6, dt=1/30,
            increasing=True, deadline=3.).item())

    def test_ninety_encoded_steps_have_no_accumulating_phase_error(self):
        scalar = 0.
        tensor = torch.zeros(1)
        for tick in range(1, 91):
            scalar = advance_timer_seconds(scalar, duration=.6, dt=1/30,
                increasing=True, maximum=3.)
            tensor = advance_normalized_timer(tensor, duration=.6, dt=1/30,
                increasing=True, maximum=3.)
            self.assertEqual(np.float32(scalar/.6), np.float32(tick/18))
            self.assertEqual(np.float32(tensor.item()), np.float32(tick/18))
        for _ in range(10):
            tensor = advance_normalized_timer(tensor, duration=.6, dt=1/30,
                increasing=True, maximum=3.)
            self.assertEqual(tensor.item(), 5.)

    def test_deadline_is_not_widened_by_an_extra_tick(self):
        for dtype in (torch.float32, torch.float64):
            values = torch.tensor([88/18, 89/18, 90/18], dtype=dtype)
            ready = normalized_timer_ready(values, duration=.6, dt=1/30,
                increasing=True, deadline=3.)
            self.assertEqual(ready.tolist(), [False, True, True])
            before = torch.tensor([88/18], dtype=dtype)
            first = advance_normalized_timer(before, duration=.6, dt=1/30,
                increasing=True, maximum=3.)
            self.assertLess(first.item(), 5.)
        # A real off-grid pose is not moved to a nearby simulation frame.
        original = 2.99 - 1/30
        actual = advance_timer_seconds(original, duration=.6, dt=1/30,
            increasing=True, maximum=3.)
        self.assertEqual(actual, original+1/30)
        self.assertEqual(actual, 2.99)

    def test_offgrid_and_non_tick_steps_keep_original_arithmetic(self):
        for dt, value in ((1/30, .234567), (.031, 4.944444179534912)):
            tensor = torch.tensor([value])
            result = advance_normalized_timer(tensor, duration=.6, dt=dt,
                increasing=True, maximum=3.)
            old = (tensor.clamp(0, 5)+dt/.6).clamp_max(5)
            torch.testing.assert_close(result, old, rtol=0, atol=0)
            old_ready = tensor*.6+dt >= 3.-1e-6
            ready = normalized_timer_ready(tensor, duration=.6, dt=dt,
                increasing=True, deadline=3.)
            torch.testing.assert_close(ready, old_ready, rtol=0, atol=0)
            seconds = value*.6
            self.assertEqual(advance_timer_seconds(seconds, duration=.6, dt=dt,
                increasing=True, maximum=3.), min(3., seconds+dt))

    def test_elapsed_timer_derivative_survives_phase_encode(self):
        value = torch.tensor([20/18, 20.1234/18], requires_grad=True)
        result = advance_normalized_timer(value, duration=.6, dt=1/30,
            increasing=True, maximum=3.)
        result.sum().backward()
        torch.testing.assert_close(value.grad, torch.ones_like(value), rtol=0, atol=0)

    def test_structured_source_birth_agrees_for_bad_serialized_phase(self):
        game = Game(seed=3, endless=True)
        before = game.state_vector()
        before[12:14] = 0
        before[16] = np.float32(4.944444179534912)
        game.set_state_vector(before)
        event = game.step(np.zeros(4))
        after = game.state_vector()
        self.assertTrue(game.enemies[0]['alive'])
        old = torch.from_numpy(before)[None]
        result = structured_transition(old, old.clone(), torch.zeros(1, 4),
            torch.tensor([[-12., 12.]]), torch.tensor([[-12.]]), endless=True,
            spawn_positions=torch.from_numpy(after[8:10].copy())[None, None],
            spawn_valid=torch.ones(1, 1, dtype=torch.bool))
        self.assertTrue(result['respawn_flags'].item())
        np.testing.assert_array_equal(result['events'].numpy()[0], Game.event_vector(event))
        torch.testing.assert_close(result['state'][0, 8:18], torch.from_numpy(after[8:18]),
                                   rtol=0, atol=1e-7)

    def test_only_death_timer_forward_representation_changes_before_birth(self):
        game = Game(seed=3, endless=True)
        state = torch.from_numpy(game.state_vector())[None]
        state[:, 6:8] = torch.tensor([[.47, .31]])
        state[:, 12:14] = 0
        state[:, 14:17] = torch.tensor([[.71, .63, 12/18]])
        args = (state, state.clone(), torch.tensor([[.2, -.1, .3, 1.]]),
                torch.tensor([[-12., 12.]]), torch.tensor([[-12.]]))
        current = structured_transition(*args, endless=True)
        def legacy(value, *, duration, dt, increasing=False, maximum=None):
            return (value.clamp(0, maximum/duration)+dt/duration).clamp_max(maximum/duration)
        with patch('turkeyworld.structured_state.advance_normalized_timer', side_effect=legacy):
            previous = structured_transition(*args, endless=True)
        keep = [index for index in range(20) if index != 16]
        torch.testing.assert_close(current['state'][:, keep], previous['state'][:, keep], rtol=0, atol=0)
        for field in ('events', 'target_logits', 'attack_logits', 'shot', 'respawn_flags'):
            torch.testing.assert_close(current[field], previous[field], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
