"""Independent NPC memory, causal observations and isolated shared inputs."""
from copy import deepcopy
import inspect
import unittest
from unittest.mock import patch

import torch
from torch import nn

from turkeyworld.enemy_rssm import EnemyRSSM, EnemyRSSMConfig
from turkeyworld.rssm import balanced_kl
from turkeyworld.engine import Game


def small_model():
    return EnemyRSSM(EnemyRSSMConfig(deter_dim=24, hidden_dim=24, stoch=3, classes=4,
        embed_dim=12, action_dim=12, blocks=4), width=16, observation_dim=20)


def inputs(batch=2, width=16, observation_dim=20):
    state = torch.zeros(batch, 20)
    state[:, :2] = torch.tensor([.2, .2]); state[:, 3:6] = 1.
    state[:, 8:10] = torch.tensor([.4, .2]); state[:, 12:14] = torch.tensor([.6, 1.])
    return (state, torch.randn(batch, 4), state[:, :2]+.001,
            torch.randn(batch, width), torch.randn(batch, 2, width), torch.randn(batch, observation_dim))


class EnemyRSSMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_default_count_packing_eval_argmax_and_separate_actor_shapes(self):
        model = EnemyRSSM().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 2_438_019)
        self.assertEqual(sum(p.numel() for p in model.core.parameters()), 1_579_904)
        self.assertEqual(model.config.memory_size, 640)
        state, action, proposed, fmap, tokens, observation = inputs(width=512, observation_dim=512)
        memory = model.initial(2, dtype=torch.float16)
        self.assertEqual((memory.shape, memory.dtype), (torch.Size([2, 640]), torch.float32))
        rng = torch.get_rng_state().clone()
        result = model.prior(memory, state, action, proposed, fmap, tokens)
        repeated = model.prior(memory, state, action, proposed, fmap, tokens)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key in ('memory', 'pooled', 'motion', 'attack_logits', 'rssm_prior_logits'):
            self.assertTrue(torch.equal(result[key], repeated[key]), key)
        self.assertEqual(result['pooled'].shape, (2, 512))
        self.assertEqual(result['motion'].shape, (2, 2))
        self.assertEqual(result['attack_logits'].shape, (2, 1))
        self.assertEqual(result['rssm_prior_logits'].shape, (2, 8, 32))
        self.assertTrue(torch.equal(result['rssm_z'].sum(-1), torch.ones(2, 8)))
        self.assertNotIn('shared_rgb_embed', inspect.signature(model.prior).parameters)
        with self.assertRaises(TypeError):
            model.prior(memory, state, action, proposed, fmap, tokens, shared_rgb_embed=observation)

    def test_enemy_backward_trains_own_core_but_no_shared_player_map_spatial_rgb_or_numeric_inputs(self):
        model = small_model().train()
        shared = nn.Linear(5, 16)
        shared_observation = nn.Linear(5, 20)
        base = torch.randn(2, 5, requires_grad=True)
        context = shared(base); observation = shared_observation(base)
        state, action, proposed, _, _, _ = inputs()
        state.requires_grad_(True); action.requires_grad_(True); proposed.requires_grad_(True)
        tokens = context[:, None].expand(-1, 2, -1)
        prior = model.prior(model.initial(2), state, action, proposed, context, tokens, sample=False)
        posterior = model.posterior(prior, observation, sample=False)
        kl, _ = balanced_kl(posterior['rssm_posterior_logits'], prior['rssm_prior_logits'], free_nats=0)
        loss = prior['motion'].square().mean()+posterior['pooled'].square().mean()+kl \
            +torch.nn.functional.binary_cross_entropy_with_logits(prior['attack_logits'], torch.ones(2, 1))
        loss.backward()
        for value in (state, action, proposed, base): self.assertIsNone(value.grad)
        self.assertTrue(all(p.grad is None for p in shared.parameters()))
        self.assertTrue(all(p.grad is None for p in shared_observation.parameters()))
        for module in (model.core, model.conditioner, model.observation_projection, model.readout,
                       model.motion_head, model.attack_head):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)

    def test_posterior_reuses_exact_pre_rgb_prior_and_future_rgb_cannot_modify_its_predictions(self):
        model = small_model().eval()
        state, action, proposed, fmap, tokens, observation = inputs()
        with patch.object(Game, 'step', side_effect=AssertionError('No engine in enemy core')), \
                patch.object(Game, 'render', side_effect=AssertionError('No renderer in enemy core')):
            prior = model.prior(model.initial(2), state, action, proposed, fmap, tokens)
            saved = {k: v.detach().clone() for k, v in prior.items() if torch.is_tensor(v)}
            with patch.object(model.core, 'imagine_step', side_effect=AssertionError('No second prior from future RGB')):
                left = model.posterior(prior, observation)
                right = model.posterior(prior, observation+10.)
        self.assertFalse(torch.equal(left['rssm_posterior_logits'], right['rssm_posterior_logits']))
        self.assertTrue(torch.equal(left['rssm_h'], right['rssm_h']))
        self.assertTrue(torch.equal(left['rssm_prior_logits'], prior['rssm_prior_logits']))
        for key, value in saved.items(): self.assertTrue(torch.equal(prior[key], value), key)

    def test_dead_behind_camera_memory_is_not_reset_and_tbptt_detach_changes_only_graph(self):
        model = small_model().eval()
        state, action, proposed, fmap, tokens, observation = inputs()
        warm = model.observe_initial(observation, state, fmap, tokens)
        carried = warm['memory'].detach().requires_grad_(True)
        state[:, 8:10] = torch.tensor([.1, .2])  # Behind the player's positive-X view.
        state[:, 12:14] = 0.                  # Dead slot still has its own h/z.
        cold = model.initial(2)
        with patch.object(model.core, 'initial', side_effect=AssertionError('Never reset on death/offscreen')):
            result = model.prior(carried, state, action, proposed, fmap, tokens)
            cold_result = model.prior(cold, state, action, proposed, fmap, tokens)
            self.assertFalse(torch.equal(result['memory'], cold_result['memory']))
            own, detached = result['memory'], result['memory'].detach()
            boundary = own
            for _ in range(4):
                own = model.prior(own, state, action, proposed, fmap, tokens)['memory']
                detached = model.prior(detached, state, action, proposed, fmap, tokens)['memory']
            self.assertTrue(torch.equal(own, detached))
            self.assertIsNone(torch.autograd.grad(detached.square().mean(), boundary, allow_unused=True, retain_graph=True)[0])
            own.square().mean().backward()
        self.assertGreater(float(carried.grad.abs().sum()), 0.)
        self.assertTrue(torch.isfinite(carried.grad).all())

    def test_half_precision_network_keeps_finite_fp32_hz_and_stochastic_one_hot_gradients(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = small_model().to(dtype=dtype).train()
                state, action, proposed, fmap, tokens, observation = inputs()
                initial = model.observe_initial(observation, state, fmap, tokens, sample=True)
                prior = model.prior(initial['memory'], state, action, proposed, fmap, tokens, sample=True)
                posterior = model.posterior(prior, observation, sample=True)
                self.assertEqual(posterior['memory'].dtype, torch.float32)
                self.assertTrue(torch.isfinite(posterior['memory']).all())
                self.assertTrue(torch.equal(posterior['rssm_z'].sum(-1), torch.ones(2, 3)))
                posterior['pooled'].square().mean().backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
                self.assertGreater(sum(float(g.float().abs().sum()) for g in gradients), 0.)

    def test_bad_configuration_shapes_memory_and_nonfinite_values_fail_without_repair(self):
        for values in ({'deter_dim': 23}, {'classes': 1}, {'stoch': True}, {'unimix': float('nan')}):
            with self.assertRaises(ValueError): EnemyRSSMConfig(**values)
        model = small_model(); state, action, proposed, fmap, tokens, observation = inputs()
        with self.assertRaises(ValueError): model.prior(torch.zeros(2, 10), state, action, proposed, fmap, tokens)
        with self.assertRaises(ValueError): model.prior(None, state, action, proposed, fmap, tokens)
        with self.assertRaises(ValueError): model.condition(state, action, proposed, fmap, tokens[:, :1])
        with self.assertRaises(ValueError): model.initial(True)
        prior = model.prior(model.initial(2), state, action, proposed, fmap, tokens, sample=False)
        with self.assertRaises(ValueError): model.posterior(prior, observation[:, :3])
        with self.assertRaises(ValueError): model.prior(model.initial(2), state, action, proposed, fmap, tokens, sample=1)
        bad = state.clone(); bad[0, 0] = float('nan')
        with self.assertRaises(FloatingPointError): model.condition(bad, action, proposed, fmap, tokens)


if __name__ == '__main__': unittest.main()
