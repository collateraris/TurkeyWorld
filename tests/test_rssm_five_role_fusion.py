import unittest

import torch

from turkeyworld.rssm_five_role_fusion import (FIVE_ROLES, RESIDUAL_KEYS,
    RSSMFiveRoleFusion, five_role_metadata, own_player_mode)


class RSSMFiveRoleFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def inputs(self):
        state = torch.zeros(3, 20)
        state[:, 4] = 1
        state[:, 3] = 1
        state[1:, 8:10] = .4
        state[1:, 12:14] = 1
        action = torch.zeros(3, 4)
        action[2, 3] = 1
        return torch.randn(3, 3, 16), state, action

    def test_exact_zero_corrections_and_five_distinct_private_transformers(self):
        module = RSSMFiveRoleFusion(16, 4, 32)
        context, state, action = self.inputs()
        result = module(context, state, action)
        for name in RESIDUAL_KEYS:
            self.assertTrue(torch.equal(result[name], torch.zeros_like(result[name])), name)
        self.assertEqual(result['interaction_spawn_tile_residual'].shape, (3, 1, 256))
        self.assertEqual(result['interaction_spawn_anchor_residual'].shape, (3, 1, 9))
        self.assertEqual(tuple(result['role_names']), FIVE_ROLES)
        self.assertEqual(len({id(expert) for expert in module.experts}), 5)
        self.assertTrue(torch.all(result['router_probabilities'] > 0))
        self.assertTrue(torch.allclose(result['router_probabilities'].sum(-1), torch.ones(3, 5)))
        self.assertFalse(five_role_metadata(module)['new_recurrent_memory'])

    def test_own_modes_distinguish_no_enemy_quiet_combat_and_respawning_enemy(self):
        _, state, action = self.inputs()
        self.assertEqual(own_player_mode(state, action).tolist(), [0, 1, 2])
        self.assertEqual(own_player_mode(state, action, navigation_only=True).tolist(), [0, 0, 0])
        state[1, 12:14] = 0; state[1, 16] = .8
        self.assertEqual(own_player_mode(state, action).tolist(), [0, 2, 2])
        state[:, 18] = 5  # kills are not a player-hit flag or a routing reset.
        self.assertEqual(own_player_mode(state, action).tolist(), [0, 2, 2])

    def test_all_five_experts_and_task_heads_receive_finite_gradients(self):
        torch.manual_seed(19)
        module = RSSMFiveRoleFusion(16, 4, 32)
        optimizer = torch.optim.AdamW(module.parameters(), lr=.001)
        context, state, action = self.inputs()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            own_context = context.clone().requires_grad_()
            result = module(own_context, state, action)
            loss = sum((result[name] - 1).square().mean() for name in RESIDUAL_KEYS)
            loss = loss + .01 * result['router_role_loss'] + .001 * result['router_balance_loss']
            loss.backward()
            optimizer.step()
        for index, expert in enumerate(module.experts):
            gradients = [p.grad for p in expert.parameters() if p.grad is not None]
            self.assertTrue(all(bool(torch.isfinite(value).all()) for value in gradients))
            self.assertGreater(sum(float(value.abs().sum()) for value in gradients), 0, index)
        for head in (*module.player_heads, module.enemy_head, module.combat_head, module.router, module.player_mode_router):
            self.assertGreater(float(head.weight.grad.abs().sum()), 0)
        self.assertGreater(float(own_context.grad.abs().sum()), 0)

    def test_nonfinite_inputs_rejected_without_repair(self):
        module = RSSMFiveRoleFusion(16, 4, 32)
        context, state, action = self.inputs()
        state[0, 0] = float('nan')
        with self.assertRaises(FloatingPointError):
            module(context, state, action)


if __name__ == '__main__':
    unittest.main()
