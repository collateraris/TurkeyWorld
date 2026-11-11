"""Combat labels reuse the existing prior/observation pass exactly once."""
import copy
import unittest
from unittest.mock import patch

import torch

from test_rssm_combat_objectives import source_batch
from test_rssm_combat_training import raw_model
from turkeyworld.rssm_combat_protocol import environment_profile
from turkeyworld.engine import Game
from turkeyworld.rssm_migration import configure_rssm_training
from turkeyworld.rssm_objectives import rssm_dynamics_backward
import turkeyworld.rssm_objectives as objectives


class CombatIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def _run(self, weights, batch, combat_protocol):
        model = raw_model()
        model.load_state_dict(weights)
        configure_rssm_training(model)
        model.core.eval()
        main_queries, source_priors, encoded = [], [], []
        transition = model.transition
        observation = model.vjepa_backend.observe_step

        def main(state, action, fmap, memory):
            main_queries.append((state.detach().clone(), memory.detach().clone()))
            return transition(state, action, fmap, memory)

        def observed(*args, **kwargs):
            result = observation(*args, **kwargs)
            source_priors.append(result['prior_result']['target_logits'].detach().clone())
            return result

        hook = model.vjepa_backend.obs_encoder.register_forward_pre_hook(
            lambda module, args: encoded.append(args[0].detach().clone()))
        try:
            with torch.random.fork_rng(), environment_profile(model, navigation_only=False), \
                    patch.object(model, 'transition', side_effect=main), \
                    patch.object(model.vjepa_backend, 'observe_step', side_effect=observed) as observations, \
                    patch.object(objectives, 'balanced_kl', wraps=objectives.balanced_kl) as kl, \
                    patch.object(model, 'render', side_effect=AssertionError('No frame generator')):
                torch.random.default_generator.manual_seed(71)
                report = rssm_dynamics_backward(model, batch, context=1, observed_context=1,
                    auxiliary_batch=batch, auxiliary_context=1, auxiliary_observed_context=1,
                    expected_horizon=4, chunk_size=2, combat_protocol=combat_protocol)
                counts = (observations.call_count, kl.call_count)
        finally:
            hook.remove()
        self.assertTrue(all(parameter.grad is None for parameter in model.flow_renderer.parameters()))
        return report, main_queries, source_priors, encoded, counts

    def _batch(self):
        batch = source_batch(4)
        # Distinct RGB at each time makes accidental duplicate/future encoding observable.
        batch['frames'] = torch.arange(5).float()[None, :, None, None, None].expand(4, 5, 3, 32, 32)/5
        batch['map'] = torch.from_numpy(Game().map).long()[None].repeat(4, 1, 1)
        batch['map'][-1] = 0
        batch['map'][-1, [0, -1], :] = 1
        batch['map'][-1, :, [0, -1]] = 1
        batch['map'][-1, 2, 3] = 1
        return batch

    def test_optin_adds_combat_once_without_second_kl_or_source_observation(self):
        weights = copy.deepcopy(raw_model().state_dict())
        batch = self._batch()
        baseline, base_queries, _, base_encoded, base_counts = self._run(weights, batch, None)
        combat, queries, _, encoded, counts = self._run(weights, batch, {})
        self.assertEqual(counts, base_counts)
        self.assertEqual(counts, (4, 4))
        self.assertEqual(len(encoded), len(base_encoded))
        self.assertEqual(len(encoded), 6)  # main init + auxiliary init + four Source observations
        for old, new in zip(base_encoded, encoded):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
        for old, new in zip(base_queries, queries):
            for before, after in zip(old, new):
                torch.testing.assert_close(before, after, rtol=0, atol=0)
        for name in ('rssm_aux_posterior_kl', 'rssm_aux_posterior_reconstruction', 'rssm_aux_posterior_state'):
            self.assertEqual(combat[name], baseline[name])
        added = sum(combat['rssm_aux_'+name] for name in (
            'combat_events', 'combat_target', 'combat_attack', 'combat_damage_npc',
            'combat_damage_player', 'combat_death_npc', 'combat_spawn', 'combat_npc_motion'))
        self.assertAlmostEqual(combat['rssm_auxiliary_loss']-baseline['rssm_auxiliary_loss'], added, places=5)
        self.assertTrue(combat['rssm_aux_combat_objectives_active'])
        self.assertFalse(combat['rssm_posterior_feedback_to_prior'])

    def test_future_rgb_changes_neither_main_queries_nor_first_pre_rgb_combat_prior(self):
        weights = copy.deepcopy(raw_model().state_dict())
        original = self._batch()
        changed = {name: value.clone() for name, value in original.items()}
        changed['frames'][:, 1:] = 1-changed['frames'][:, 1:]
        _, queries, priors, _, _ = self._run(weights, original, {})
        report, changed_queries, changed_priors, _, _ = self._run(weights, changed, {})
        for old, new in zip(queries, changed_queries):
            for before, after in zip(old, new):
                torch.testing.assert_close(before, after, rtol=0, atol=0)
        torch.testing.assert_close(priors[0], changed_priors[0], rtol=0, atol=0)
        self.assertEqual(report['rssm_prior_future_rgb_inputs'], 0)
        self.assertTrue(report['rssm_aux_source_queries_before_target_rgb'])


if __name__ == '__main__':
    unittest.main()
