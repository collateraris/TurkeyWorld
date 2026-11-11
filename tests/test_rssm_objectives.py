"""RSSM prior450, isolated Source posterior labels and honest teacher transfer."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from turkeyworld.rssm import DreamerRSSM, RSSMConfig
from turkeyworld.rssm_objectives import rssm_dynamics_backward
from turkeyworld.structured_state import player_command_delta, structured_transition
from test_long_dynamics_training import batch as long_batch
from test_motion_recovery_training import source_batch


class SpyRSSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = DreamerRSSM(RSSMConfig(deter_dim=8, hidden_dim=8, stoch=2, classes=2,
            embed_dim=4, action_dim=4, blocks=2))
        self.obs_encoder = nn.Linear(3, 4)
        self.readout = nn.Linear(12, 8)
        self.pass_head = nn.Linear(8, 2)
        self.feature_head = nn.Linear(8, 128).requires_grad_(False)
        self.observation_decoder = nn.Linear(8, 3)
        self.map_encoder = nn.Linear(1, 2)
        self.renderer = nn.Linear(1, 1).requires_grad_(False)
        self.readout_neck = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 8))
        self.calls, self.observations, self.initializations = [], [], []

    @property
    def vjepa_backend(self):
        return self

    def encode_map(self, grid):
        return self.map_encoder(grid.float().mean((1, 2))[:, None])

    def initialize_from_image(self, frames, states, features):
        self.initializations.append((frames.detach().clone(), states.detach().clone()))
        latent = self.core.initial(len(states))
        for index in range(frames.shape[1]):
            embed = self.obs_encoder(frames[:, index].mean((2, 3)))
            latent, _ = self.core.observe_step(latent, states.new_zeros(len(states), 4), embed, sample=False)
        return self.core.pack(latent)

    def _readout(self, state, action, latent):
        pooled = self.readout(self.core.features(latent))
        logits = self.pass_head(pooled)
        command = player_command_delta(state, action)
        raw = command*logits.sigmoid()
        candidate = torch.cat([state[:, :2]+raw, state[:, 2:]], -1)
        result = structured_transition(state, candidate, action,
            state.new_tensor([[-8., 8.]]).expand(len(state), -1), state.new_full((len(state), 1), -8.),
            training=self.training, endless=True, respawn_enabled=False)
        return {**result, 'memory': self.core.pack(latent), 'rssm_prior_logits': latent['logits'],
            'rssm_h': latent['deter'], 'rssm_z': latent['stoch'], 'rssm_readout': pooled,
            'features': self.feature_head(pooled), 'latent_prediction': pooled[:, None],
            'player_pass_logits': logits, 'raw_player_delta': raw, 'player_command_delta': command}

    def transition(self, state, action, features, memory):
        self.calls.append((state.detach().clone(), memory.detach().clone()))
        latent = self.core.imagine_step(self.core.unpack(memory), action, sample=False)
        return self._readout(state, action, latent)

    def observe_step(self, state, action, features, memory, rgb):
        count_before = len(self.calls)
        prior_result = self.transition(state, action, features, memory)
        prior = dict(deter=prior_result['rssm_h'], stoch=prior_result['rssm_z'], logits=prior_result['rssm_prior_logits'])
        # The observation is consumed after the prior result already exists.
        posterior = self.core.posterior(prior, self.obs_encoder(rgb.mean((2, 3))), sample=False)
        post = self._readout(state, action, posterior)
        reconstructed = self.observation_decoder(post['rssm_readout']).sigmoid()[:, :, None, None].expand_as(rgb)
        self.observations.append({'prior_calls_before': count_before, 'rgb': rgb.detach().clone(),
                                  'prior_logits': prior_result['player_pass_logits'].detach().clone()})
        return {**post, 'prior_result': prior_result, 'rssm_prior_logits': prior['logits'],
                'rssm_posterior_logits': posterior['logits'], 'rssm_reconstruction': reconstructed}


class RSSMObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_full450_prior_carries_memory_before_separate_posterior_and_has_no_ema_or_render(self):
        world = SpyRSSM()
        data, auxiliary = long_batch(), source_batch(context=1, horizon=4, batch_size=1)
        original = {key: value.clone() for key, value in data.items()}
        before = {key: value.clone() for key, value in world.state_dict().items()}
        progress = []
        with patch.object(world.renderer, 'forward', side_effect=AssertionError('No renderer')):
            metrics = rssm_dynamics_backward(world, data, phase='source', auxiliary_batch=auxiliary,
                auxiliary_context=1, observed_context=1, progress_callback=lambda n, total: progress.append((n, total)))
        self.assertEqual(metrics['rssm_prior_horizon'], 450)
        self.assertEqual(metrics['rssm_prior_tbptt_chunks'], 29)
        self.assertEqual(metrics['rssm_prior_numerical_resets'], 0)
        self.assertEqual(progress[-1], (450, 450))
        self.assertEqual(world.observations[0]['prior_calls_before'], 450)
        self.assertEqual(len(world.observations), 4)
        self.assertFalse(metrics['rssm_posterior_feedback_to_prior'])
        self.assertEqual(metrics['rssm_prior_future_rgb_inputs'], 0)
        self.assertFalse(metrics['rssm_ema_used'])
        self.assertEqual(metrics['rssm_optimizer_steps'], 0)
        self.assertEqual(metrics['rssm_prior_context_frames'], 1)
        torch.testing.assert_close(world.initializations[0][0], data['frames'][:, 1:2], rtol=0, atol=0)
        torch.testing.assert_close(world.calls[0][0], data['states'][:, 1], rtol=0, atol=0)
        self.assertFalse(torch.equal(world.calls[16][0], data['states'][:, 17]))
        self.assertGreater(float(world.obs_encoder.weight.grad.norm()), 0.)
        self.assertGreater(float(world.observation_decoder.weight.grad.norm()), 0.)
        self.assertGreater(float(world.pass_head.weight.grad.norm()), 0.)
        self.assertIsNone(world.map_encoder.weight.grad)
        self.assertIsNone(world.renderer.weight.grad)
        for key, value in data.items(): torch.testing.assert_close(value, original[key], rtol=0, atol=0)
        for key, value in world.state_dict().items(): torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_future_source_labels_never_change_main_prior_or_teacher_inputs_and_teacher_is_frozen(self):
        student, teacher = SpyRSSM(), SpyRSSM()
        data = source_batch(context=1, horizon=4, batch_size=1)
        baseline = copy.deepcopy(student.state_dict())
        teacher_before = copy.deepcopy(teacher.state_dict())
        metrics = rssm_dynamics_backward(student, data, phase='distill', teacher=teacher, context=1, teacher_weight=.1)
        self.assertTrue(metrics['rssm_teacher_used'])
        self.assertGreater(metrics['rssm_teacher_readout'], 0.)
        self.assertGreater(metrics['rssm_aux_posterior_reconstruction'], 0.)
        self.assertTrue(teacher.training)  # temporary eval is restored
        for key, value in teacher.state_dict().items(): torch.testing.assert_close(value, teacher_before[key], rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))
        changed = {key: value.clone() for key, value in data.items()}
        changed['states'][:, 1:, :2] += .001
        changed['target_states'][:, :, :2] += .001
        changed['frames'][:, 1:] = 1-changed['frames'][:, 1:]
        second = SpyRSSM(); second.load_state_dict(baseline)
        second_teacher = SpyRSSM(); second_teacher.load_state_dict(teacher_before)
        rssm_dynamics_backward(second, changed, phase='distill', teacher=second_teacher, context=1)
        for old, new in zip(student.calls[:4], second.calls[:4]):
            torch.testing.assert_close(old[0], new[0], rtol=0, atol=0)
            torch.testing.assert_close(old[1], new[1], rtol=0, atol=0)
        for old, new in zip(teacher.calls, second_teacher.calls):
            torch.testing.assert_close(old[0], new[0], rtol=0, atol=0)
        # The first Source prior query also precedes the changed next RGB.
        torch.testing.assert_close(student.observations[0]['prior_logits'], second.observations[0]['prior_logits'], rtol=0, atol=0)

    def test_real_rssm_backend_posterior_reconstruction_and_motion_gradients_frozen_assets(self):
        from test_single_enemy_core import model
        from turkeyworld.rssm_migration import build_rssm_student
        teacher = model()
        world, _ = build_rssm_student(teacher, config_overrides=dict(rssm_deter_dim=32, rssm_hidden_dim=32,
            rssm_stoch=4, rssm_classes=4, rssm_embed_dim=16, rssm_action_dim=20, rssm_blocks=4))
        data = source_batch(context=2, horizon=3, batch_size=1)
        aux = source_batch(context=1, horizon=3, batch_size=1)
        frozen = {name: value.clone() for name, value in world.named_parameters() if not value.requires_grad}
        with patch.object(world, 'render', side_effect=AssertionError('No frozen renderer')):
            metrics = rssm_dynamics_backward(world, data, phase='source', expected_horizon=3,
                auxiliary_batch=aux, auxiliary_context=1, observed_context=1)
        self.assertGreater(float(world.vjepa_backend.obs_encoder.net[0].weight.grad.norm()), 0.)
        self.assertGreater(float(world.vjepa_backend.observation_decoder[0].weight.grad.norm()), 0.)
        self.assertGreater(float(world.vjepa_backend.player_pass_head[-1].weight.grad.norm()), 0.)
        self.assertFalse(metrics['rssm_ema_used'])
        self.assertEqual(metrics['rssm_aux_source_observed_frames'], 1)
        for name, parameter in world.named_parameters():
            if name in frozen:
                self.assertIsNone(parameter.grad, name)
                torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)

    def test_phase_and_horizon_guards_fail_before_gradients(self):
        world, data = SpyRSSM(), long_batch(horizon=4)
        for options in (dict(phase='distill'), dict(phase='source'),
                        dict(phase='distill', teacher=SpyRSSM(), teacher_weight=.5)):
            with self.assertRaises(ValueError): rssm_dynamics_backward(world, data, **options)
        self.assertTrue(all(parameter.grad is None for parameter in world.parameters()))


if __name__ == '__main__': unittest.main()
