"""Event bookkeeping invariants, independent of renderer quality or learned weights."""
import unittest

import numpy as np
import torch

from turkeyworld.engine import Game
from turkeyworld.structured_state import structured_transition


class StructuredStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def initial(self, batch=1):
        state = torch.from_numpy(Game(seed=3).state_vector())[None]
        return state.repeat(batch, 1)

    def logits(self, batch=1, target=8):
        targets = torch.full((batch, 9), -12.0)
        targets[:, target] = 12.0
        return targets, torch.full((batch, 8), -12.0)

    def run_step(self, state, action=None, target=8, attacks=None, candidate=None, training=False):
        target_logits, attack_logits = self.logits(len(state), target)
        if attacks is not None:
            attack_logits = attacks
        if action is None:
            action = torch.zeros(len(state), 4)
        if candidate is None:
            candidate = state.clone()
        return structured_transition(state, candidate, action, target_logits, attack_logits,
                                     training=training)

    def test_no_false_decay_over_nine_thousand_idle_steps(self):
        state = self.initial()
        initial = state.clone()
        generator = torch.Generator().manual_seed(41)
        actions = torch.zeros(1, 4)
        targets, attacks = self.logits()
        with torch.inference_mode():
            for _ in range(9000):
                # Even wildly wrong proposals for non-motion fields must not
                # erode HP, flags, ammunition or create combat events.
                proposal = torch.rand((1, 90), generator=generator)
                result = structured_transition(state, proposal, actions, targets, attacks)
                state = result['state']
                self.assertEqual(float(result['events'].sum()), 0.0)
        torch.testing.assert_close(state[:, :2], initial[:, :2], rtol=0, atol=0)
        torch.testing.assert_close(state[:, 4:6], initial[:, 4:6], rtol=0, atol=0)
        old = initial[:, 8:-2].reshape(1, 8, 10)
        new = state[:, 8:-2].reshape(1, 8, 10)
        torch.testing.assert_close(new[:, :, 4:6], old[:, :, 4:6], rtol=0, atol=0)
        self.assertEqual(float(state[0, -2]), 0.0)
        self.assertAlmostEqual(float(state[0, -1]), 1.0, places=3)

    def test_two_accepted_hits_kill_and_persist(self):
        state = self.initial()
        fire = torch.tensor([[0., 0., 0., 1.]])
        first = self.run_step(state, fire, target=0)
        self.assertEqual(first['accepted_target'].tolist(), [0])
        self.assertEqual(first['events'].tolist(), [[1., 1., 0., 0.]])
        self.assertAlmostEqual(float(first['state'][0, 12]), .25, places=6)
        self.assertAlmostEqual(float(first['state'][0, 5]), 98 / 99, places=6)
        blocked = self.run_step(first['state'], fire, target=0)
        self.assertFalse(bool(blocked['shot'][0]))
        self.assertEqual(float(blocked['npc_damage'].sum()), 0)
        state = first['state']
        for _ in range(7):
            state = self.run_step(state)['state']
        second = self.run_step(state, fire, target=0)
        self.assertEqual(second['events'].tolist(), [[1., 1., 1., 0.]])
        dead = second['state'][:, 8:18]
        self.assertEqual(float(dead[0, 4]), 0)
        self.assertEqual(float(dead[0, 5]), 0)
        self.assertEqual(float(dead[0, 8]), 0)
        self.assertEqual(float(second['state'][0, -2]), 1 / 8)
        position = dead[:, :2].clone()
        state = second['state']
        for _ in range(30):
            proposal = torch.ones_like(state)
            state = self.run_step(state, candidate=proposal)['state']
        torch.testing.assert_close(state[:, 8:10], position, rtol=0, atol=0)
        torch.testing.assert_close(state[:, 10:14], torch.zeros(1, 4), rtol=0, atol=0)
        self.assertEqual(float(state[0, 16]), 1)

    def test_empty_ammo_and_dead_target_are_rejected(self):
        fire = torch.tensor([[0., 0., 0., 1.]])
        state = self.initial()
        state[:, 5] = 0
        empty = self.run_step(state, fire, target=0)
        self.assertEqual(empty['events'].tolist(), [[0., 0., 0., 0.]])
        state = self.initial()
        state[:, 12:14] = 0  # NPC0 HP and alive
        dead = self.run_step(state, fire, target=0)
        self.assertEqual(dead['accepted_target'].tolist(), [-1])
        self.assertEqual(dead['events'].tolist(), [[1., 0., 0., 0.]])
        self.assertEqual(float(dead['state'][0, -2]), 1 / 8)

    def test_melee_guards_cooldown_and_killed_attack_cancel(self):
        state = self.initial()
        state[:, 8:10] = torch.tensor([[2.9 / 16, 2.5 / 16]])
        attacks = torch.full((1, 8), -12.)
        attacks[:, 0] = 12
        first = self.run_step(state, attacks=attacks)
        self.assertTrue(bool(first['accepted_attacks'][0, 0]))
        self.assertAlmostEqual(float(first['state'][0, 4]), .92, places=6)
        repeat = self.run_step(first['state'], attacks=attacks)
        self.assertFalse(bool(repeat['player_hit'][0]))
        self.assertAlmostEqual(float(repeat['state'][0, 4]), .92, places=6)
        far = self.initial()
        far_hit = self.run_step(far, attacks=attacks)
        self.assertFalse(bool(far_hit['player_hit'][0]))
        doomed = state.clone()
        doomed[:, 12] = .30
        shot = self.run_step(doomed, torch.tensor([[0., 0., 0., 1.]]),
                             target=0, attacks=attacks)
        self.assertTrue(bool(shot['kill'][0]))
        self.assertFalse(bool(shot['accepted_attacks'][0, 0]))
        self.assertEqual(float(shot['state'][0, 4]), 1)

    def test_no_healing_and_terminal_motion_freeze(self):
        state = self.initial()
        state[:, 4] = .3
        state[:, 12] = .2
        proposal = torch.ones_like(state)
        result = self.run_step(state, candidate=proposal)
        self.assertAlmostEqual(float(result['state'][0, 4]), .3, places=6)
        self.assertAlmostEqual(float(result['state'][0, 12]), .2, places=6)
        state[:, 4] = 0
        state[:, 6:8] = 1
        state[:, 14] = 1  # NPC attack cooldown
        terminal = self.run_step(state, torch.ones(1, 4), target=0,
                                 attacks=torch.full((1, 8), 12.), candidate=proposal)
        torch.testing.assert_close(terminal['state'][:, :4], state[:, :4], rtol=0, atol=0)
        torch.testing.assert_close(terminal['state'][:, 8:10], state[:, 8:10], rtol=0, atol=0)
        self.assertEqual(float(terminal['events'].sum()), 0)
        self.assertLess(float(terminal['state'][0, 6]), float(state[0, 6]))
        self.assertGreater(float(terminal['state'][0, -1]), float(state[0, -1]))

    def test_movement_speed_limits_heading_and_velocity(self):
        state = self.initial()
        proposal = torch.ones_like(state)
        result = self.run_step(state, torch.tensor([[1., 0., 1., 0.]]), candidate=proposal)
        moved = result['state']
        player_distance = torch.linalg.vector_norm((moved[:, :2] - state[:, :2]) * 16)
        self.assertLessEqual(float(player_distance), 2.4 / 30 + 1e-6)
        self.assertAlmostEqual(float(moved[0, 2]), np.sin(2.2 / 30), places=6)
        self.assertAlmostEqual(float(moved[0, 3]), np.cos(2.2 / 30), places=6)
        slots = moved[:, 8:-2].reshape(1, 8, 10)
        old = state[:, 8:-2].reshape(1, 8, 10)
        distances = torch.linalg.vector_norm((slots[:, :, :2] - old[:, :, :2]) * 16, dim=-1)
        self.assertLessEqual(float(distances.max()), 1.2 / 30 + 1e-6)
        torch.testing.assert_close(slots[:, :, 2:4], (slots[:, :, :2] - old[:, :, :2]) * (16 * 30 / 2),
                                   rtol=1e-4, atol=1e-6)

    def test_hard_forward_decisions_have_finite_nonzero_training_gradients(self):
        previous = self.initial(batch=2).requires_grad_()
        candidate = (previous.detach() + .01).requires_grad_()
        targets = torch.zeros(2, 9, requires_grad=True)
        attacks = torch.ones(2, 8, requires_grad=True)
        # NPC0 in melee range, while all targets begin nonlethal at 0.6 HP.
        close = previous.detach().clone()
        close[:, 8:10] = torch.tensor([[2.9 / 16, 2.5 / 16]])
        previous = close.requires_grad_()
        action = torch.tensor([[1., 0., .2, 1.], [0., 0., 0., 0.]])
        result = structured_transition(previous, candidate, action, targets, attacks, training=True)
        inference = structured_transition(previous, candidate, action, targets, attacks, training=False)
        torch.testing.assert_close(result['state'], inference['state'], rtol=0, atol=0)
        loss = result['state'].square().mean() + .01 * result['event_logits'].square().mean()
        loss.backward()
        for tensor in (previous, candidate, targets, attacks):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(bool(torch.isfinite(tensor.grad).all()))
        self.assertGreater(float(targets.grad.abs().sum()), 0)
        self.assertGreater(float(attacks.grad.abs().sum()), 0)
        self.assertTrue(bool(torch.isfinite(result['event_logits']).all()))

    def test_oracle_bookkeeping_matches_when_motion_and_decisions_are_supplied(self):
        game = Game(seed=2)
        # Motion/visibility remain learned in production. Here only the test
        # supplies oracle XY and observed decisions to isolate bookkeeping.
        actions = [[0, 0, 0, 1]] + [[.1, 0, .05, 0]] * 7 + [[0, 0, 0, 1]] + [[0, 0, 0, 0]] * 12
        for action_values in actions:
            before = game.state_vector()
            event = game.step(action_values)
            after = game.state_vector()
            target, attack = self.logits(target=event['hit_id'] if event['hit_id'] >= 0 else 8)
            old_slots = before[8:-2].reshape(8, 10)
            new_slots = after[8:-2].reshape(8, 10)
            accepted = (new_slots[:, 6] > .999) & (old_slots[:, 6] < .1)
            attack[0, torch.from_numpy(accepted)] = 12
            result = structured_transition(torch.from_numpy(before)[None], torch.from_numpy(after)[None],
                                           torch.tensor([action_values], dtype=torch.float32), target, attack)
            np.testing.assert_allclose(result['state'][0].numpy(), after, rtol=2e-5, atol=1e-5)
            np.testing.assert_array_equal(result['events'][0].numpy(), game.event_vector(event))

    def test_all_ninety_nine_bullets_can_be_fired_without_a_stranded_round(self):
        state = self.initial()
        fire = torch.tensor([[0., 0., 0., 1.]])
        targets, attacks = self.logits()
        shots = 0
        with torch.inference_mode():
            for _ in range(800):
                result = structured_transition(state, state, fire, targets, attacks)
                state = result['state']
                shots += int(result['shot'][0])
        self.assertEqual(shots, 99)
        self.assertEqual(float(state[0, 5]), 0)

    def test_half_network_inputs_promote_to_persistent_float32_for_nine_thousand_steps(self):
        state = self.initial().half()
        action = torch.zeros(1, 4, dtype=torch.float16)
        targets, attacks = self.logits()
        targets, attacks = targets.half(), attacks.half()
        with torch.inference_mode():
            for _ in range(9000):
                result = structured_transition(state, state.half(), action, targets, attacks)
                state = result['state']
        self.assertEqual(state.dtype, torch.float32)
        self.assertAlmostEqual(float(state[0, -1]), 1., places=3)
        self.assertEqual(float(state[0, 4]), 1.)

    def test_time_is_not_clipped_at_three_hundred_seconds(self):
        state = self.initial()
        state[:, -1] = 1
        result = self.run_step(state)
        self.assertGreater(float(result['state'][0, -1]), 1)


if __name__ == '__main__':
    unittest.main()
