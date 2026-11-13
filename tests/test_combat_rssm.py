"""Persistent combat-only memory, causal observations and old-stream RNG isolation."""
from contextlib import contextmanager
import inspect
import unittest
from unittest.mock import patch

import torch
from torch import nn

from turkeyworld.combat_rssm import (CombatRSSM, CombatRSSMConfig,
                                    COMBAT_RSSM_PARAMETERS, _preserve_sampling_rng)
from turkeyworld.engine import Game
from turkeyworld.rssm import balanced_kl


def small_model(dtype=torch.float32):
    return CombatRSSM(CombatRSSMConfig(deter_dim=24, hidden_dim=24, stoch=3, classes=4,
        embed_dim=12, action_dim=12, blocks=4), width=16, observation_dim=20, fusion_width=8).to(dtype=dtype)


def inputs(batch=2, *, width=16, observation_dim=20):
    state = torch.zeros(batch, 20)
    state[:, :2] = torch.tensor([.2, .2]); state[:, 3:6] = 1.
    state[:, 8:10] = torch.tensor([.4, .2]); state[:, 12:14] = torch.tensor([.6, 1.])
    return (state, torch.randn(batch, 4), state[:, :2]+.001,
            torch.randn(batch, width), torch.randn(batch, 2, width),
            torch.randn(batch, width), torch.randn(batch, width), torch.randn(batch, observation_dim))


def prior(model, memory, values, **kwargs):
    state, action, proposal, maps, spatial, player, enemy, _ = values
    return model.prior(memory, state, action, proposal, maps, spatial,
                       player_context=player, enemy_context=enemy, **kwargs)


class CombatRSSMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_actual20m_count_includes_adapter_and_outputs_are_zero_without_duplicate_HP_XY(self):
        with torch.device('meta'):
            production = CombatRSSM()
        self.assertEqual(sum(p.numel() for p in production.parameters()), 20_007_020)
        self.assertEqual(COMBAT_RSSM_PARAMETERS, 20_007_020)
        self.assertLess(abs(COMBAT_RSSM_PARAMETERS/20_000_000-1), .0005)
        self.assertEqual(production.memory_size, 1600)
        self.assertEqual(production.config.memory_size, 1600)
        self.assertEqual(sum(p.numel() for p in production.fusion_adapter.parameters()), 131328)
        model = small_model().eval(); values = inputs()
        memory = model.initial(2, dtype=torch.float16)
        self.assertEqual(memory.dtype, torch.float32)
        result = prior(model, memory, values)
        expected = {'target_residual': (2, 2), 'attack_residual': (2, 1),
                    'spawn_tile_residual': (2, 1, 256), 'spawn_anchor_residual': (2, 1, 9)}
        for key, shape in expected.items():
            self.assertEqual(tuple(result[key].shape), shape)
            self.assertFalse(bool(result[key].count_nonzero()))
        context = model.fusion_context(result['memory'])
        self.assertEqual(context.shape, (2, 8)); self.assertFalse(bool(context.count_nonzero()))
        self.assertEqual(result['rssm_prior_logits'].shape, (2, 3, 4))
        self.assertTrue(torch.equal(result['rssm_z'].sum(-1), torch.ones(2, 3)))
        self.assertFalse({'state', 'HP', 'player_xy', 'npc_xy', 'events'} & result.keys())
        self.assertNotIn('shared_rgb_embed', inspect.signature(model.prior).parameters)
        with self.assertRaises(TypeError): prior(model, memory, values, shared_rgb_embed=values[-1])

    def test_both_current_actor_contexts_have_gradients_but_numerical_and_observed_values_do_not(self):
        model = small_model().train(); values = list(inputs())
        source = torch.randn(2, 5, requires_grad=True)
        player, enemy = nn.Linear(5, 16), nn.Linear(5, 16)
        numeric_source = torch.randn(2, 5, requires_grad=True); map_layer = nn.Linear(5, 16)
        observation_source = torch.randn(2, 5, requires_grad=True); observation_layer = nn.Linear(5, 20)
        for index in (0, 1, 2): values[index].requires_grad_()
        values[3] = map_layer(numeric_source)
        values[4] = values[3][:, None].expand(-1, 2, -1)
        values[5], values[6] = player(source), enemy(source)
        values[7] = observation_layer(observation_source)
        p = prior(model, model.initial(2), values, sample=False)
        post = model.posterior(p, values[7], sample=False)
        kl, _ = balanced_kl(post['rssm_posterior_logits'], p['rssm_prior_logits'], free_nats=0)
        loss = p['pooled'].square().mean()+post['pooled'].square().mean()+kl \
            +torch.nn.functional.cross_entropy(p['target_residual'], torch.zeros(2, dtype=torch.long)) \
            +model.fusion_context(p['memory']).sum()
        loss.backward()
        for module in (player, enemy, model.core, model.conditioner, model.observation_projection,
                       model.readout, model.task_head, model.fusion_adapter):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
        self.assertGreater(float(source.grad.abs().sum()), 0)
        self.assertIsNone(numeric_source.grad); self.assertIsNone(observation_source.grad)
        self.assertTrue(all(p.grad is None for p in map_layer.parameters()))
        self.assertTrue(all(p.grad is None for p in observation_layer.parameters()))
        for index in (0, 1, 2): self.assertIsNone(values[index].grad)

    def test_stochastic_combat_prior_posterior_preserve_the_old_actor_CPU_random_draw_clock(self):
        model = small_model().train(); values = inputs(); memory = model.initial(2)
        state = torch.get_rng_state().clone()
        expected_first, expected_second = torch.rand(9), torch.rand(9)
        torch.set_rng_state(state)
        actual_first = torch.rand(9)
        between = torch.get_rng_state().clone()
        p = prior(model, memory, values, sample=True)
        self.assertTrue(torch.equal(torch.get_rng_state(), between))
        post = model.posterior(p, values[-1], sample=True)
        model.fusion_context(post['memory'])
        self.assertTrue(torch.equal(torch.get_rng_state(), between))
        actual_second = torch.rand(9)
        self.assertTrue(torch.equal(actual_first, expected_first))
        self.assertTrue(torch.equal(actual_second, expected_second))
        self.assertTrue(torch.equal(p['rssm_z'].sum(-1), torch.ones(2, 3)))
        self.assertTrue(torch.equal(post['rssm_z'].sum(-1), torch.ones(2, 3)))
        self.assertFalse(any('rng' in key.lower() or 'counter' in key.lower() for key, _ in model.named_buffers()))

    def test_posterior_reuses_actual_prior_and_future_observation_cannot_change_it(self):
        model = small_model().eval(); values = inputs()
        with patch.object(Game, 'step', side_effect=AssertionError('No Source engine')), \
             patch.object(Game, 'render', side_effect=AssertionError('No neural/classical renderer')):
            p = prior(model, model.initial(2), values)
            saved = {k: v.clone() for k, v in p.items() if torch.is_tensor(v)}
            with patch.object(model.core, 'imagine_step', side_effect=AssertionError('No second prior from RGB')):
                a = model.posterior(p, values[-1])
                b = model.posterior(p, values[-1]+10)
        self.assertFalse(torch.equal(a['rssm_posterior_logits'], b['rssm_posterior_logits']))
        self.assertTrue(torch.equal(a['rssm_h'], b['rssm_h']))
        self.assertTrue(torch.equal(a['rssm_prior_logits'], p['rssm_prior_logits']))
        for key, value in saved.items(): self.assertTrue(torch.equal(value, p[key]), key)

    def test_memory_continues_on_death_offscreen_and_tbptt_only_cuts_gradient(self):
        model = small_model().eval(); values = list(inputs())
        warm = model.observe_initial(values[-1], values[0], values[3], values[4],
            player_context=values[5], enemy_context=values[6])
        carried = warm['memory'].detach().requires_grad_(True)
        cold = model.initial(2)
        values[0][:, 4] = 0; values[0][:, 12:14] = 0; values[0][:, 8] = .1
        with patch.object(model.core, 'initial', side_effect=AssertionError('Never reset on death/no enemy')):
            a, b = prior(model, carried, values), prior(model, cold, values)
            self.assertFalse(torch.equal(a['memory'], b['memory']))
            boundary = a['memory']; own, detached = boundary, boundary.detach()
            for _ in range(8):
                own = prior(model, own, values)['memory']
                detached = prior(model, detached, values)['memory']
            self.assertTrue(torch.equal(own, detached))
            self.assertIsNone(torch.autograd.grad(detached.square().mean(), boundary,
                retain_graph=True, allow_unused=True)[0])
            own.square().mean().backward()
        self.assertGreater(float(carried.grad.abs().sum()), 0)
        self.assertTrue(bool(torch.isfinite(carried.grad).all()))

    def test_half_network_keeps_finite_FP32_memory_and_zero_residuals(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = small_model(dtype).train(); values = inputs()
                warm = model.observe_initial(values[-1], values[0], values[3], values[4],
                    player_context=values[5], enemy_context=values[6], sample=True)
                p = prior(model, warm['memory'], values, sample=True)
                post = model.posterior(p, values[-1], sample=True)
                self.assertEqual(post['memory'].dtype, torch.float32)
                self.assertTrue(bool(torch.isfinite(post['memory']).all()))
                self.assertFalse(bool(post['target_residual'].count_nonzero()))
                self.assertFalse(bool(model.fusion_context(post['memory']).count_nonzero()))
                post['pooled'].square().mean().backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))

    def test_cuda_sampling_scope_targets_only_current_device_without_allocating_a_GPU(self):
        captured = []
        @contextmanager
        def fake_fork(*, devices):
            captured.append(devices); yield
        with patch('torch.random.fork_rng', side_effect=fake_fork):
            with _preserve_sampling_rng(torch.device('cuda:3')): pass
            with _preserve_sampling_rng(torch.device('cpu')): pass
        self.assertEqual(captured, [[3], []])

    def test_invalid_config_inputs_memory_and_observation_raise_without_repair(self):
        for config in ({'deter_dim': 23}, {'stoch': True}, {'classes': 1}, {'unimix': float('nan')}):
            with self.assertRaises(ValueError): CombatRSSMConfig(**config)
        model = small_model(); values = inputs()
        with self.assertRaises(ValueError): model.initial(True)
        with self.assertRaises(ValueError): prior(model, None, values)
        with self.assertRaises(ValueError): prior(model, torch.zeros(2, 3), values)
        bad = list(values); bad[5] = torch.zeros(2, 15)
        with self.assertRaises(ValueError): prior(model, model.initial(2), bad)
        bad = list(values); bad[3] = bad[3].clone(); bad[3][0, 0] = float('nan')
        with self.assertRaises(FloatingPointError): prior(model, model.initial(2), bad)
        memory = model.initial(2); memory[0, 0] = float('inf')
        with self.assertRaises(FloatingPointError): prior(model, memory, values)
        p = prior(model, model.initial(2), values, sample=False)
        with self.assertRaises(ValueError): model.posterior(p, torch.zeros(2, 3, 32, 32))
        with self.assertRaises(ValueError): prior(model, model.initial(2), values, sample=1)


if __name__ == '__main__':
    unittest.main()
