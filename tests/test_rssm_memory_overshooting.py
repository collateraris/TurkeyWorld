"""Autonomous categorical memory learns stopped Source posterior targets."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import torch

from test_dual_rssm_backend import model
from test_dual_rssm_objectives import source_batch
from turkeyworld.rssm_migration import configure_rssm_training
from turkeyworld.rssm_loss_balance import GroupLossAccumulator
from turkeyworld.rssm_memory_overshooting import (
    MEMORY_OVERSHOOTING_PROTOCOL as PROTOCOL,
    normalize_memory_overshooting_protocol, memory_overshooting_backward, _categorical_kl)
from turkeyworld.rssm_coordinate_gradients import COORDINATE_GRADIENT_PROTOCOL


class MemoryOvershootingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def run_world(self, world, batch, *, groups=None, context=1, observed=1):
        configure_rssm_training(world)
        world.core.train()
        # Preserve mixed child modes, not just the root core.training boolean.
        world.core.prior_head[0].eval()
        modes = [(module, module.training) for module in world.modules()]
        weights = {name: value.clone() for name, value in world.state_dict().items()}
        calls, targets, encoded = [], [], []
        transition, observe = world.transition, world.vjepa_backend.observe_step

        def capture_transition(state, action, fmap, memory):
            self.assertFalse(world.core.training)
            self.assertTrue(world.training)
            self.assertTrue(world.vjepa_backend.training)
            calls.append((state.detach().clone(), memory.detach().clone(), state.requires_grad))
            result = transition(state, action, fmap, memory)
            calls[-1] += (result['rssm_prior_logits'].detach().clone(),
                          result.get('enemy_rssm_prior_logits', torch.empty(0)).detach().clone())
            return result

        def capture_observation(*args, **kwargs):
            self.assertEqual(kwargs.get('sample'), False)
            self.assertFalse(torch.is_grad_enabled())
            result = observe(*args, **kwargs)
            targets.append(result)
            return result

        hook = world.vjepa_backend.obs_encoder.register_forward_pre_hook(
            lambda module, args: encoded.append((args[0].detach().clone(), torch.is_grad_enabled(), len(calls))))
        rng = torch.get_rng_state().clone()
        try:
            with patch.object(world, 'transition', side_effect=capture_transition), \
                 patch.object(world.vjepa_backend, 'observe_step', side_effect=capture_observation), \
                 patch.object(world, 'render', side_effect=AssertionError('No frozen renderer')):
                report = memory_overshooting_backward(world, batch, context=context, observed_context=observed,
                    protocol=PROTOCOL, coordinate_gradient_protocol=COORDINATE_GRADIENT_PROTOCOL,
                    loss_groups=groups)
        finally:
            hook.remove()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(module.training == previous for module, previous in modes))
        self.assertTrue(all(torch.equal(value, world.state_dict()[name]) for name, value in weights.items()))
        return report, calls, targets, encoded

    def test_dual_actual_rssm_gradients_targets_stopped_assets_frozen_and_single_backward(self):
        world = model(True)
        batch = source_batch()
        batch['frames'].requires_grad_()
        batch['states'].requires_grad_()
        report, calls, targets, encoded = self.run_world(world, batch)
        self.assertEqual(report['offsets'], [1, 4])
        self.assertEqual(report['backward_calls'], 1)
        self.assertEqual(report['optimizer_steps'], 0)
        self.assertEqual(report['kl_floor'], 0.)
        self.assertEqual(report['prior_future_rgb_inputs'], 0)
        self.assertTrue(report['no_future_feedback'])
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(targets), 4)
        self.assertTrue(all(not row[2] for row in calls))
        self.assertTrue(all(row[2] == 0 for row in encoded))
        # Target initialization + four future target observations have no graph;
        # only the separate final observed-context initialization has a graph.
        self.assertEqual([row[1] for row in encoded], [False]*5+[True])
        for target in targets:
            for key in ('rssm_posterior_logits', 'rssm_h', 'enemy_rssm_posterior_logits', 'enemy_rssm_h'):
                self.assertFalse(target[key].requires_grad, key)
        self.assertGreater(report['player_kl'], 0.)
        self.assertGreater(report['enemy_kl'], 0.)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in world.core.parameters() if p.grad is not None), 0.)
        enemy = world.vjepa_backend.enemy_rssm
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in enemy.core.parameters() if p.grad is not None), 0.)
        self.assertGreater(float(world.vjepa_backend.obs_encoder.net[0].weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in world.parameters() if not p.requires_grad))
        self.assertIsNone(batch['frames'].grad)
        self.assertIsNone(batch['states'].grad)

    def test_future_rgb_changes_targets_and_loss_but_not_own_state_or_either_memory(self):
        original = model(True)
        data = source_batch()
        changed = {name: value.clone() for name, value in data.items()}
        changed['frames'][:, 1:] = 1-changed['frames'][:, 1:]
        left, queries, targets, _ = self.run_world(deepcopy(original), data)
        right, other_queries, other_targets, _ = self.run_world(deepcopy(original), changed)
        for old, new in zip(queries, other_queries):
            for index in (0, 1, 3, 4): torch.testing.assert_close(old[index], new[index], rtol=0, atol=0)
        self.assertFalse(torch.equal(targets[0]['rssm_posterior_logits'], other_targets[0]['rssm_posterior_logits']))
        self.assertFalse(torch.equal(targets[0]['enemy_rssm_posterior_logits'], other_targets[0]['enemy_rssm_posterior_logits']))
        self.assertNotEqual(left['loss'], right['loss'])

    def test_full_sixteen_steps_all_offsets_and_unbroken_own_hz_gradient(self):
        import numpy as np
        from turkeyworld.engine import Game
        game = Game(seed=41, endless=True, enemies_enabled=True)
        states = [torch.tensor(game.state_vector())]
        frames = [torch.tensor(game.render(32, overlays=False)).permute(2, 0, 1).float()/255]
        action = np.asarray([.25, .1, .1, 0.], np.float32)
        for _ in range(17):
            game.step(action, 1/30)
            states.append(torch.tensor(game.state_vector()))
            frames.append(torch.tensor(game.render(32, overlays=False)).permute(2, 0, 1).float()/255)
        states = torch.stack(states)[None]
        batch = {'states': states, 'frames': torch.stack(frames)[None],
            'actions': torch.tensor(action)[None, None].repeat(1, 16, 1),
            'target_states': states[:, 2:].clone(), 'map': torch.tensor(game.map)[None]}
        world = model(True)
        transition = world.transition
        memories = []
        def remember(*args, **kwargs):
            result = transition(*args, **kwargs)
            result['memory'].retain_grad()
            memories.append(result['memory'])
            return result
        world.transition = remember
        report, calls, targets, encoded = self.run_world(world, batch, context=2, observed=1)
        self.assertEqual(report['offsets'], [1, 4, 8, 16])
        self.assertEqual(report['horizon'], 16)
        self.assertEqual(len(calls), 16)
        self.assertEqual(len(targets), 16)
        self.assertEqual(report['initial_observed_frames'], 1)
        torch.testing.assert_close(calls[0][0], states[:, 1], rtol=0, atol=0)
        # Outgoing memory at tick1 is not a direct loss input; this gradient
        # therefore proves that subsequent own ticks retain its temporal path.
        self.assertGreater(float(memories[0].grad[:, :48].abs().sum()), 0.)
        self.assertGreater(float(memories[0].grad[:, 48:].abs().sum()), 0.)
        self.assertTrue(all(row[2] == 0 for row in encoded))

    def test_actor_coefficients_apply_once_without_new_kl_floor(self):
        world = model(True)
        groups = GroupLossAccumulator(scales={'player': 2., 'enemy': 4., 'interaction': 3.})
        report, _, _, _ = self.run_world(world, source_batch(), groups=groups)
        player = .1*(report['player_kl']+.1*report['player_h'])
        enemy = .1*(report['enemy_kl']+.1*report['enemy_h'])
        self.assertAlmostEqual(report['loss'], player+enemy, places=6)
        self.assertAlmostEqual(report['weighted_loss'], player/2+enemy/4, places=6)
        self.assertEqual(groups.calls, 1)
        self.assertEqual(groups.report()['kl_floor'], 0.)
        self.assertAlmostEqual(groups.report()['total'], report['weighted_loss'], places=6)
        self.assertEqual(groups.report()['interaction']['raw'], 0.)

    def test_categorical_normalization_has_no_free_floor_or_latent_count_multiplier(self):
        target = torch.tensor([[[1., -1.]]], requires_grad=True)
        prior = torch.tensor([[[-.5, .5]]], requires_grad=True)
        single = _categorical_kl(target, prior, .01)
        repeated = _categorical_kl(target.expand(3, 7, 2), prior.expand(3, 7, 2), .01)
        torch.testing.assert_close(single, repeated)
        self.assertEqual(float(_categorical_kl(target, target.detach(), .01)), 0.)
        single.backward()
        self.assertIsNone(target.grad)
        self.assertGreater(float(prior.grad.abs().sum()), 0.)

    def test_strict_settings_legacy_noop_and_short_single_actor(self):
        self.assertIsNone(normalize_memory_overshooting_protocol(None))
        self.assertEqual(normalize_memory_overshooting_protocol(PROTOCOL), PROTOCOL)
        for invalid in ({**PROTOCOL, 'version': True}, {**PROTOCOL, 'weight': .2},
                        {**PROTOCOL, 'offsets': [1, 4, 16]}, {**PROTOCOL, 'future_rgb_feedback': True},
                        {**PROTOCOL, 'target_stop_gradient': False}):
            with self.assertRaises(ValueError): normalize_memory_overshooting_protocol(invalid)
        inactive = memory_overshooting_backward(None, None, context=1, observed_context=1, protocol=None)
        self.assertEqual(inactive['backward_calls'], 0)
        batch = source_batch()
        batch = {**batch, 'states': batch['states'][:, :2], 'frames': batch['frames'][:, :2],
                 'target_states': batch['target_states'][:, :1], 'actions': batch['actions'][:, :1]}
        report, _, _, _ = self.run_world(model(False), batch)
        self.assertEqual(report['offsets'], [1])
        self.assertEqual(report['enemy_kl'], 0.)
        self.assertEqual(report['enemy_h'], 0.)
        self.assertTrue(torch.isfinite(torch.tensor(report['loss'])))

    def test_prior_failure_restores_all_core_modes_without_backward_or_updates(self):
        world = model(True)
        configure_rssm_training(world)
        world.core.train(); world.core.prior_head[0].eval()
        modes = [(module, module.training) for module in world.modules()]
        before = {name: value.clone() for name, value in world.state_dict().items()}
        with patch.object(world, 'transition', side_effect=FloatingPointError('probe failure')):
            with self.assertRaisesRegex(FloatingPointError, 'probe failure'):
                memory_overshooting_backward(world, source_batch(), context=1, observed_context=1, protocol=PROTOCOL)
        self.assertTrue(all(module.training == old for module, old in modes))
        self.assertTrue(all(p.grad is None for p in world.parameters()))
        self.assertTrue(all(torch.equal(value, world.state_dict()[name]) for name, value in before.items()))


if __name__ == '__main__': unittest.main()
