from copy import deepcopy
import itertools
import json
import math
import unittest

import torch
from torch import nn

from turkeyworld.dreamer_actor_critic import (PROTOCOL, ActorCriticConfig, FactorizedActor,
    DreamerActorCritic, ReturnPercentileNormalizer, action_indices)
from turkeyworld.dreamer_actor_critic_math import (make_value_bins, twohot_encode, twohot_mean,
    twohot_loss, twohot_target_clip_stats, lambda_returns, imagination_weights)


class DreamerActorCriticMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_raw_symexp_and_explicit_symlog_twohot_roundtrip_mass_and_clamping(self):
        for encoding in ('symexp_twohot', 'symlog_twohot'):
            with self.subTest(encoding=encoding):
                bins = make_value_bins(encoding=encoding)
                values = torch.tensor([-1e6, -123., -1., -.01, 0., .01, 1., 123., 1e6], requires_grad=True)
                labels = twohot_encode(values, bins, encoding)
                self.assertFalse(labels.requires_grad)
                torch.testing.assert_close(labels.sum(-1), torch.ones(len(values)), atol=1e-7, rtol=0)
                self.assertTrue(bool((labels >= 0).all()))
                self.assertTrue(bool(((labels > 0).sum(-1) <= 2).all()))
                recovered = twohot_mean(labels.clamp_min(1e-30).log(), bins, encoding)
                torch.testing.assert_close(recovered, values.detach(), atol=1e-3, rtol=2e-5)
                extremes = twohot_encode(torch.tensor([-1e20, 1e20]), bins, encoding)
                self.assertEqual(float(extremes[0, 0]), 1.)
                self.assertEqual(float(extremes[1, -1]), 1.)
                loss = twohot_loss(torch.zeros_like(labels, requires_grad=True), values, bins, encoding)
                loss.sum().backward()
                self.assertIsNone(values.grad)

    def test_uniform_huge_symexp_support_has_bitexact_zero_mean(self):
        bins = make_value_bins()
        self.assertGreater(float(bins[-1]), 4e8)
        self.assertTrue(torch.equal(bins, -bins.flip(0)))
        value = twohot_mean(torch.zeros(8, 15, 255), bins)
        self.assertTrue(torch.equal(value, torch.zeros_like(value)))
        model = DreamerActorCritic(ActorCriticConfig(4, hidden_dim=16, layers=1))
        self.assertTrue(torch.equal(model.critic.value(torch.randn(3, 4)), torch.zeros(3)))

    def test_game_supports_cover_reward_return_bounds_and_report_raw_clipping(self):
        reward_bins = make_value_bins(low=-math.log1p(2), high=math.log1p(2))
        value_bins = make_value_bins(low=-math.log1p(700), high=math.log1p(700))
        self.assertAlmostEqual(float(reward_bins[-1]), 2., places=6)
        self.assertAlmostEqual(float(value_bins[-1]), 700., delta=.001)
        reward = twohot_target_clip_stats(torch.tensor([-2., 0., 1.1]), reward_bins)
        self.assertEqual(reward['clipped_count'], 0)
        self.assertEqual(reward['target_count'], 3)
        value = twohot_target_clip_stats(torch.tensor([-2/(1-332/333), 1.1/(1-332/333)]), value_bins)
        self.assertEqual(value['clipped_count'], 0)
        outside = twohot_target_clip_stats(torch.tensor([-3., 0., 4.]), reward_bins)
        self.assertEqual(outside['clipped_count'], 2)
        self.assertAlmostEqual(outside['clipped_fraction'], 2/3, places=6)
        # Counters inspect raw units for either explicit encoding.
        legacy = make_value_bins(low=-math.log1p(2), high=math.log1p(2), encoding='symlog_twohot')
        same = twohot_target_clip_stats(torch.tensor([-3., 0., 4.]), legacy, 'symlog_twohot')
        self.assertEqual(outside, same)

    def test_lambda_return_matches_independent_n_step_mixture_and_terminal_outcomes(self):
        reward = torch.tensor([[1., 2., 3., 4.]])
        continuation = torch.tensor([[1., .5, 0., 1.]])
        values = torch.tensor([[10., 20., 30., 40., 50.]])
        gamma, lam = .9, .7
        actual = lambda_returns(reward, continuation, values, gamma=gamma, lambda_=lam)
        expected = []
        for start in range(4):
            total = 0.
            for length in range(1, 5-start):
                discounted = 0.; multiplier = 1.
                for offset in range(length):
                    at = start+offset
                    discounted += multiplier*float(reward[0, at])
                    multiplier *= gamma*float(continuation[0, at])
                n_step = discounted+multiplier*float(values[0, start+length])
                mixture = lam**(length-1)*(1-lam if length < 4-start else 1.)
                total += mixture*n_step
            expected.append(total)
        torch.testing.assert_close(actual[0], torch.tensor(expected), atol=1e-5, rtol=0)
        self.assertEqual(float(actual[0, 2]), 3.)  # terminal action keeps its reward
        mc = lambda_returns(torch.tensor([[1., 2., 3.]]), torch.tensor([[1., 1., 0.]]),
            torch.tensor([[99., 99., 99., 1000.]]), gamma=1., lambda_=1.)
        torch.testing.assert_close(mc, torch.tensor([[6., 5., 3.]]), atol=0, rtol=0)
        one_step = lambda_returns(reward, continuation, values, gamma=gamma, lambda_=0.)
        torch.testing.assert_close(one_step, reward+gamma*continuation*values[:, 1:])

    def test_action_before_terminal_is_weighted_and_later_transitions_are_not(self):
        continues = torch.tensor([[1., 0., 1., 1.], [1., 1., 1., 1.]], requires_grad=True)
        weights = imagination_weights(continues, gamma=.5, initial_alive=torch.tensor([1., 0.]))
        torch.testing.assert_close(weights, torch.tensor([[1., .5, 0., 0.], [0., 0., 0., 0.]]), atol=0, rtol=0)
        self.assertFalse(weights.requires_grad)

    def test_invalid_targets_continuations_and_configuration_are_rejected(self):
        with self.assertRaises(ValueError): make_value_bins(1)
        with self.assertRaises(ValueError): twohot_encode(torch.tensor([float('nan')]), make_value_bins())
        with self.assertRaises(ValueError): twohot_encode(torch.ones(1), torch.tensor([0., 0., 1.]))
        with self.assertRaises(ValueError): lambda_returns(torch.ones(2, 3), torch.full((2, 3), 2.), torch.ones(2, 4))
        with self.assertRaises(ValueError): imagination_weights(torch.tensor([[float('nan')]]))
        config = ActorCriticConfig(5252)
        self.assertEqual(config.gamma, 332/333)
        self.assertFalse(config.slow_target); self.assertEqual(config.slow_regularization, 1.)
        self.assertEqual(ActorCriticConfig.from_dict(json.loads(json.dumps(config.to_dict()))), config)
        bad = config.to_dict(); bad['hidden_dim'] = True
        with self.assertRaises(ValueError): ActorCriticConfig.from_dict(bad)
        bad = config.to_dict(); bad['invented_reward'] = 1
        with self.assertRaises(ValueError): ActorCriticConfig.from_dict(bad)


class DreamerActorCriticGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def config(self, **kw):
        return ActorCriticConfig(4, hidden_dim=32, layers=1, **kw)

    def test_all54_factorized_actions_form_a_normalized_distribution_with_correct_entropy(self):
        actor = FactorizedActor(self.config())
        with torch.no_grad(): actor.output.weight.zero_(); actor.output.bias.zero_()
        actions = torch.tensor(list(itertools.product((-1., 0., 1.), (-1., 0., 1.), (-1., 0., 1.), (0., 1.))))
        result = actor.evaluate_actions(torch.zeros(54, 4), actions)
        self.assertEqual(result['logits'].shape, (54, 11))
        self.assertAlmostEqual(float(result['log_prob'].exp().sum().detach()), 1., places=6)
        torch.testing.assert_close(result['log_prob'], torch.full((54,), -math.log(54)), atol=1e-6, rtol=0)
        torch.testing.assert_close(result['entropy'], torch.full((54,), math.log(54)), atol=1e-6, rtol=0)
        first = actor.act(torch.zeros(1024, 4), generator=torch.Generator().manual_seed(173))
        second = actor.act(torch.zeros(1024, 4), generator=torch.Generator().manual_seed(173))
        self.assertTrue(torch.equal(first, second)); self.assertFalse(first.requires_grad)
        for index in range(3): self.assertEqual(set(first[:, index].tolist()), {-1., 0., 1.})
        self.assertEqual(set(first[:, 3].tolist()), {0., 1.})
        with self.assertRaises(ValueError): action_indices(torch.tensor([[.1, 0., 0., 1.]]))

    def tape(self, model, *, require_world_grad=False):
        if require_world_grad:
            world = nn.Linear(4, 4)
            reward_head = nn.Linear(4, 1)
            continue_head = nn.Linear(4, 1)
            features = world(torch.randn(8, 4, 4)); features.retain_grad()
            rewards = reward_head(features[:, 1:]).squeeze(-1)
            continues = continue_head(features[:, 1:]).squeeze(-1).sigmoid()
        else:
            world = reward_head = continue_head = None
            features = torch.randn(8, 4, 4)
            rewards = torch.ones(8, 3)
            continues = torch.ones(8, 3)
        actions = model.actor.act(features[:, :-1])
        return dict(features=features, actions=actions, rewards=rewards, continues=continues), (world, reward_head, continue_head)

    def test_actor_and_critic_gradients_cannot_reach_world_reward_continue_or_each_other(self):
        model = DreamerActorCritic(self.config())
        tape, source_modules = self.tape(model, require_world_grad=True)
        losses, metrics = model.losses(tape)
        losses['actor'].backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.actor.parameters() if p.grad is not None), 0.)
        self.assertTrue(all(p.grad is None for p in model.critic.parameters()))
        self.assertTrue(all(p.grad is None for p in model.slow_critic.parameters()))
        self.assertTrue(all(p.grad is None for source in source_modules for p in source.parameters()))
        self.assertIsNone(tape['features'].grad)
        model.zero_grad(set_to_none=True)
        losses, _ = model.losses(tape, update_normalizer=False)
        losses['critic'].backward()
        self.assertTrue(all(p.grad is None for p in model.actor.parameters()))
        self.assertTrue(all(p.grad is None for p in model.slow_critic.parameters()))
        self.assertTrue(all(p.grad is None for source in source_modules for p in source.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.critic.parameters() if p.grad is not None), 0.)
        self.assertTrue(metrics['world_inputs_detached']); self.assertTrue(metrics['critic_targets_detached'])
        self.assertFalse(metrics['actor_uses_critic_gradient'])
        self.assertEqual(metrics['critic_target_clipped_count'], 0)
        self.assertEqual(metrics['critic_slow_target_clipped_count'], 0)
        json.dumps(metrics, allow_nan=False)

    def test_critic_future_bootstrap_logit_has_no_target_gradient(self):
        model = DreamerActorCritic(self.config())
        tape, _ = self.tape(model)
        captured = []
        def capture(module, args, output):
            output.retain_grad(); captured.append(output)
        handle = model.critic.register_forward_hook(capture)
        losses, _ = model.losses(tape, update_normalizer=False)
        losses['critic'].backward(); handle.remove()
        self.assertEqual(len(captured), 1)
        self.assertEqual(float(captured[0].grad[:, -1].abs().sum()), 0.)
        self.assertGreater(float(captured[0].grad[:, :-1].abs().sum()), 0.)

    def test_reinforce_actor_learns_a_positive_synthetic_fire_reward(self):
        model = DreamerActorCritic(self.config(entropy_scale=0., return_norm=False, gamma=0., slow_regularization=0.))
        optimizer = torch.optim.Adam(model.actor.parameters(), lr=.04)
        actions = torch.tensor(list(itertools.product((-1., 0., 1.), (-1., 0., 1.), (-1., 0., 1.), (0., 1.))))
        tape = dict(features=torch.zeros(54, 2, 4), actions=actions[:, None],
                    rewards=actions[:, 3:4].clone(), continues=torch.ones(54, 1))
        def fire_probability():
            logits = model.actor(torch.zeros(1, 4)).detach()
            return float(logits[:, -2:].softmax(-1)[0, 1])
        initial = fire_probability()
        for _ in range(30):
            optimizer.zero_grad(set_to_none=True)
            losses, _ = model.losses(tape, update_normalizer=False)
            losses['actor'].backward(); optimizer.step()
        self.assertGreater(fire_probability(), initial+.3)
        self.assertTrue(all(p.grad is None for p in model.critic.parameters()))

    def test_slow_value_is_exact_ema_and_never_requires_grad(self):
        model = DreamerActorCritic(self.config())
        before = {name: p.detach().clone() for name, p in model.slow_critic.named_parameters()}
        with torch.no_grad():
            for p in model.critic.parameters(): p.add_(1)
        model.update_slow_critic()
        for name, parameter in model.slow_critic.named_parameters():
            source = dict(model.critic.named_parameters())[name]
            torch.testing.assert_close(parameter, before[name]+.02*(source-before[name]), rtol=0, atol=1e-7)
            self.assertFalse(parameter.requires_grad); self.assertIsNone(parameter.grad)
        self.assertEqual(int(model.slow_updates), 1)

    def test_percentile_ema_has_reference_no_debias_floor_and_is_resumable(self):
        normalizer = ReturnPercentileNormalizer(self.config())
        normalizer.update(torch.arange(101).float())
        self.assertAlmostEqual(float(normalizer.low), .05, places=6)
        self.assertAlmostEqual(float(normalizer.high), .95, places=6)
        self.assertEqual(float(normalizer.scale()), 1.)
        restored = ReturnPercentileNormalizer(self.config()); restored.load_state_dict(normalizer.state_dict())
        normalizer.update(torch.arange(101).float()); restored.update(torch.arange(101).float())
        self.assertAlmostEqual(float(restored.scale()), 1.791, places=5)
        self.assertTrue(torch.equal(normalizer.low, restored.low)); self.assertEqual(int(restored.updates), 2)
        model = DreamerActorCritic(self.config())
        tape, _ = self.tape(model)
        model.losses(tape, update_normalizer=False)
        self.assertEqual(int(model.return_normalizer.updates), 0)

    def test_dead_initial_actors_have_zero_finite_losses(self):
        model = DreamerActorCritic(self.config())
        tape, _ = self.tape(model); tape['initial_alive'] = torch.zeros(8)
        losses, metrics = model.losses(tape)
        self.assertEqual(float(losses['actor'].detach()), 0.)
        self.assertEqual(float(losses['critic'].detach()), 0.)
        self.assertEqual(metrics['alive_weighted_transitions'], 0.)


if __name__ == '__main__': unittest.main()
