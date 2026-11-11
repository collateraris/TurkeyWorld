"""Categorical RSSM recurrence, separate KL gradients and persistent memory."""
from dataclasses import asdict, replace
import unittest

import torch

from turkeyworld.rssm import DreamerRSSM, RSSMConfig, balanced_kl


def small_config(**values):
    config = dict(deter_dim=32, hidden_dim=24, stoch=4, classes=4,
                  embed_dim=12, action_dim=6, blocks=4)
    config.update(values)
    return RSSMConfig(**config)


class RSSMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_prior_posterior_and_memory_have_distinct_observation_paths(self):
        model = DreamerRSSM(small_config()).eval()
        initial = model.initial(3)
        action = torch.randn(3, 6)
        random_state = torch.random.get_rng_state().clone()
        post, prior = model.observe_step(initial, action, torch.randn(3, 12))
        # In evaluation the categorical state uses argmax, with no RNG draw.
        random_state = torch.random.get_rng_state().clone()
        repeated = model.imagine_step(initial, action)
        self.assertTrue(torch.equal(random_state, torch.random.get_rng_state()))
        for key in prior:
            torch.testing.assert_close(prior[key], repeated[key], rtol=0, atol=0)
        self.assertIs(post['deter'], prior['deter'])
        changed = model.posterior(prior, torch.full((3, 12), 20.))
        self.assertFalse(torch.equal(post['logits'], changed['logits']))
        self.assertEqual(prior['stoch'].shape, (3, 4, 4))
        torch.testing.assert_close(prior['stoch'].sum(-1), torch.ones(3, 4), rtol=0, atol=0)
        packed = model.pack(prior)
        self.assertEqual(packed.shape, (3, 48))
        self.assertEqual(packed.dtype, torch.float32)
        unpacked = model.unpack(packed)
        for key in ('deter', 'stoch'):
            torch.testing.assert_close(unpacked[key], prior[key], rtol=0, atol=0)
        changed_action = model.imagine_step(initial, action+1)
        self.assertFalse(torch.equal(prior['deter'], changed_action['deter']))

    def test_stochastic_one_hot_has_straight_through_gradient_and_uniform_floor(self):
        model = DreamerRSSM(small_config()).train()
        logits = torch.zeros(64, 4, 4, requires_grad=True)
        sampled = model.categorical(logits)
        self.assertTrue(bool(((sampled == 0) | (sampled == 1)).all()))
        self.assertGreater(sampled.argmax(-1).unique().numel(), 1)
        (sampled*torch.arange(4.)).sum().backward()
        self.assertGreater(logits.grad.abs().sum().item(), 0)
        peaked = torch.full((1, 4, 4), -1000.)
        peaked[..., 0] = 1000.
        probabilities = model.probabilities(peaked)
        self.assertTrue(bool((probabilities >= .01/4).all()))
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(1, 4))

    def test_dynamics_and_representation_kl_stop_opposite_distributions(self):
        posterior = torch.randn(2, 3, 4, 4, requires_grad=True)
        prior = torch.randn(2, 3, 4, 4, requires_grad=True)
        loss, parts = balanced_kl(posterior, prior, free_nats=0)
        q_grad, p_grad = torch.autograd.grad(parts['dynamics_kl'], (posterior, prior),
                                           retain_graph=True, allow_unused=True)
        self.assertIsNone(q_grad)
        self.assertGreater(p_grad.abs().sum().item(), 0)
        q_grad, p_grad = torch.autograd.grad(parts['representation_kl'], (posterior, prior),
                                           retain_graph=True, allow_unused=True)
        self.assertIsNone(p_grad)
        self.assertGreater(q_grad.abs().sum().item(), 0)
        torch.testing.assert_close(loss, parts['dynamics_kl']+.1*parts['representation_kl'])
        self.assertFalse(parts['raw_kl'].requires_grad)

    def test_free_nats_floor_has_no_gradient_below_threshold(self):
        posterior = torch.zeros(2, 4, 4, requires_grad=True)
        prior = torch.zeros(2, 4, 4, requires_grad=True)
        loss, parts = balanced_kl(posterior, prior)
        self.assertAlmostEqual(loss.item(), 1.1, places=6)
        loss.backward()
        self.assertEqual(prior.grad.abs().sum().item(), 0)
        self.assertEqual(posterior.grad.abs().sum().item(), 0)
        self.assertEqual(parts['raw_kl'].item(), 0)

    def test_tbptt_detach_preserves_numerical_continuation_without_reset(self):
        model = DreamerRSSM(small_config()).eval()
        actions = torch.randn(2, 7, 6)
        state = model.initial(2)
        for offset in range(3):
            state = model.imagine_step(state, actions[:, offset])
        boundary = model.pack(state)
        resumed = model.unpack(boundary.detach())
        for offset in range(3, 7):
            state = model.imagine_step(state, actions[:, offset])
            resumed = model.imagine_step(resumed, actions[:, offset])
        torch.testing.assert_close(model.pack(resumed), model.pack(state), rtol=0, atol=0)
        previous_gradient = torch.autograd.grad(model.pack(resumed).square().mean(), boundary,
                                                allow_unused=True)[0]
        self.assertIsNone(previous_gradient)
        self.assertGreater(model.pack(resumed).abs().sum().item(), 0)

    def test_half_and_bfloat16_weights_keep_finite_fp32_persistent_memory(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = DreamerRSSM(small_config()).to(dtype=dtype)
                post, prior = model.observe_step(model.initial(2), torch.randn(2, 6),
                                                  torch.randn(2, 12), sample=False)
                memory = model.pack(post)
                self.assertEqual(memory.dtype, torch.float32)
                self.assertTrue(bool(torch.isfinite(memory).all()))
                kl, _ = balanced_kl(post['logits'], prior['logits'], free_nats=0)
                (memory.square().mean()+kl).backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(bool(torch.isfinite(value).all()) for value in gradients))
                self.assertGreater(sum(value.float().abs().sum().item() for value in gradients), 0)

    def test_invalid_configuration_shapes_and_sampling_are_rejected(self):
        for values in ({'deter_dim': 31}, {'stoch': True}, {'classes': 1},
                       {'unimix': float('nan')}, {'unimix': 1.}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                small_config(**values)
        model = DreamerRSSM(small_config())
        with self.assertRaises(ValueError):
            model.unpack(torch.zeros(1, 49))
        with self.assertRaises(ValueError):
            model.imagine_step(model.initial(1), torch.zeros(1, 5))
        with self.assertRaises(ValueError):
            model.categorical(torch.zeros(1, 4, 4), sample=1)
        with self.assertRaises(ValueError):
            model.posterior(model.initial(1), torch.zeros(1, 13))
        with self.assertRaises(ValueError):
            balanced_kl(torch.zeros(1, 4, 4), torch.zeros(1, 4, 3))

    def test_production_dynamics_count_includes_map_and_training_decoder(self):
        from turkeyworld.models import NeuralWorldModel, parameter_counts
        from turkeyworld.train_vq_world import teacher_vq_config
        config = teacher_vq_config({'image_size': 32, 'n_embed': 1488},
                                  '175m_vqgan', True, True, True)
        config = replace(config, architecture='dreamer_rssm', player_motion_mode='axis_fraction',
                         rssm_deter_dim=1008, rssm_hidden_dim=1024)
        with torch.device('meta'):
            model = NeuralWorldModel(config)
        counts = parameter_counts(model)
        self.assertEqual(counts['dynamics_including_map'], 25_007_048)
        self.assertEqual(counts['rssm_core'], 11_967_536)
        self.assertEqual(counts['world_total'], 103_491_431)
        self.assertTrue(hasattr(model.vjepa_backend, 'observation_decoder'))
        self.assertFalse(hasattr(model.core, 'encoder'))
        self.assertFalse(hasattr(model, 'target_encoder'))

    def test_legacy_config_normalization_adds_rssm_defaults_consistently(self):
        from turkeyworld.models import WorldModelConfig
        from turkeyworld.train import normalize_world_config
        config = asdict(WorldModelConfig())
        previous = {key: value for key, value in config.items() if not key.startswith('rssm_')}
        self.assertEqual(normalize_world_config(previous), normalize_world_config(config))


if __name__ == '__main__':
    unittest.main()
