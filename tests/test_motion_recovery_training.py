"""Independent short-stream labels, balanced contacts and observed-only RGB."""
from dataclasses import replace
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from turkeyworld.engine import Game
from turkeyworld.motion_recovery_training import (pass_fraction_targets, balanced_pass_weights,
                                                  short_motion_backward)
from turkeyworld.structured_state import player_command_delta, structured_transition
from turkeyworld.component_training import configure_component_phase
from turkeyworld.long_dynamics_training import long_dynamics_backward
from test_long_dynamics_training import SpyWorld, Encoder, batch as long_batch
from turkeyworld.jepa_training_utils import EMATargetEncoder


def source_batch(context=2, horizon=6, batch_size=2):
    all_states, all_frames, all_actions = [], [], []
    for example in range(batch_size):
        game = Game(seed=example, endless=True, enemies_enabled=False)
        states = [torch.tensor(game.state_vector())]
        frames = [torch.tensor(game.render(32, overlays=False)).permute(2, 0, 1).float()/255]
        controls = []
        for index in range(context+horizon-1):
            action = [1. if example == 0 or index % 2 == 0 else -1., 0., 0., 0.]
            game.step(action, dt=1/30)
            controls.append(action)
            states.append(torch.tensor(game.state_vector()))
            frames.append(torch.tensor(game.render(32, overlays=False)).permute(2, 0, 1).float()/255)
        all_states.append(torch.stack(states)); all_frames.append(torch.stack(frames))
        all_actions.append(torch.tensor(controls[context-1:]))
    states = torch.stack(all_states)
    return dict(states=states, frames=torch.stack(all_frames), actions=torch.stack(all_actions),
                target_states=states[:, context:].clone(), map=torch.tensor(game.map)[None].expand(batch_size, -1, -1).clone())


class ShortWorld(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Module(); self.core.encoder = Encoder()
        self.core.encode_video = self.core.encoder.forward
        self.logits = nn.Parameter(torch.tensor([-.8, .2]))
        self.map_encoder = nn.Linear(1, 2)
        self.renderer = nn.Linear(1, 1).requires_grad_(False)
        self.calls, self.initializations = [], []

    def encode_map(self, grid):
        return self.map_encoder(grid.float().mean((1, 2))[:, None])

    def initialize_from_latents(self, latent, states, features):
        self.initializations.append(states.detach().clone())
        return latent[:, -1].expand(-1, 2, -1)

    def transition(self, state, action, map_features, memory):
        self.calls.append(state.detach().clone())
        logits = self.logits[None].expand(len(state), -1)+memory.mean((1, 2))[:, None]*.1
        command = player_command_delta(state, action)
        raw = command*logits.sigmoid()
        candidate = torch.cat([state[:, :2]+raw, state[:, 2:]], -1)
        result = structured_transition(state, candidate, action, state.new_tensor([[-5., 5.]]).expand(len(state), -1),
            state.new_full((len(state), 1), -5.), training=True, endless=True, respawn_enabled=False)
        return {**result, 'memory': memory+.01*state[:, :1, None], 'player_pass_logits': logits,
            'player_pass_fraction': logits.sigmoid(), 'player_command_delta': command, 'raw_player_delta': raw}


class MotionRecoveryTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_global_class_mass_is_not_diluted_by_time_or_many_blocked_negatives(self):
        gradients, losses = [], []
        for blocked_count in (1, 50):
            fractions = torch.zeros(1, 1+blocked_count, 2)
            fractions[:, 0, 0] = 1
            active = torch.zeros_like(fractions, dtype=torch.bool); active[:, :, 0] = True
            logits = torch.full_like(fractions, -3., requires_grad=True)
            weights = balanced_pass_weights(fractions, active)
            loss = (F.binary_cross_entropy_with_logits(logits, fractions, reduction='none')*weights).sum()
            loss.backward(); gradients.append(float(logits.grad[0, 0, 0])); losses.append(float(loss.detach()))
            self.assertEqual(float(weights[:, :, 1].sum()), 0.)
            self.assertAlmostEqual(float(weights.sum()), 1., places=6)
        self.assertAlmostEqual(gradients[0], gradients[1], places=6)
        self.assertAlmostEqual(losses[0], losses[1], places=6)
        self.assertLess(gradients[1], -.45)

    def test_partial_and_inactive_axis_targets_and_saturated_bad_pass_logits_are_learnable(self):
        state = torch.zeros(2, 20); state[:, :2] = .3; state[:, 3:6] = 1
        actions = torch.tensor([[1., 0., 0., 0.], [1., 1., 0., 0.]])
        commands = player_command_delta(state, actions)
        target = state.clone(); target[:, :2] += commands*torch.tensor([[1., 0.], [.25, 0.]])
        fraction, active, _ = pass_fraction_targets(state, target, actions)
        torch.testing.assert_close(fraction, torch.tensor([[1., 0.], [.25, 0.]]), atol=1e-5, rtol=1e-5)
        self.assertFalse(bool(active[0, 1])); self.assertTrue(bool(active[1].all()))
        bad = torch.tensor([[-30., 30.]], requires_grad=True)
        labels = torch.tensor([[1., 0.]])
        loss = (F.binary_cross_entropy_with_logits(bad, labels, reduction='none')
                *balanced_pass_weights(labels, torch.ones_like(labels, dtype=torch.bool))).sum()
        loss.backward()
        self.assertLess(float(bad.grad[0, 0]), -.49)
        self.assertGreater(float(bad.grad[0, 1]), .49)

    def test_short_stream_is_independent_causal_nonmutating_and_freezes_map_renderer(self):
        model = ShortWorld(); data = source_batch(context=2, horizon=6)
        snapshot = {key: value.clone() for key, value in data.items()}
        model_before = {key: value.clone() for key, value in model.state_dict().items()}
        with patch.object(Game, 'step', side_effect=AssertionError('No engine queries')), \
                patch.object(Game, 'render', side_effect=AssertionError('No renderer queries')):
            metrics = short_motion_backward(model, data, context=2, observed_context=1)
        self.assertEqual(len(model.calls), 12)
        self.assertEqual(metrics['motion_aux_backward_calls'], 3)  # own +4+2 labeled queries
        self.assertEqual(metrics['motion_aux_draws'], 2)
        self.assertFalse(metrics['motion_aux_gt_feedback_to_long'])
        self.assertFalse(metrics['motion_aux_future_rgb_online_encoded'])
        self.assertEqual(len(model.core.encoder.calls), 2)
        for images in model.core.encoder.calls:
            torch.testing.assert_close(images, data['frames'][:, 1:2], rtol=0, atol=0)
        for state in model.initializations:
            torch.testing.assert_close(state, data['states'][:, 1:2], rtol=0, atol=0)
        self.assertIsNone(model.map_encoder.weight.grad)
        self.assertIsNone(model.renderer.weight.grad)
        self.assertGreater(float(model.logits.grad.norm()), 0.)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, model_before[key]))  # backward only, no step
        for key, value in data.items():
            torch.testing.assert_close(value, snapshot[key], rtol=0, atol=0)
        altered = {key: value.clone() for key, value in data.items()}
        altered['states'][:, 2:, :2] += .001
        altered['target_states'][:, :, :2] += .001
        altered['frames'][:, 2:] = 1-altered['frames'][:, 2:]
        copy_model = ShortWorld(); copy_model.load_state_dict(model_before)
        short_motion_backward(copy_model, altered, context=2, observed_context=1)
        for old, new in zip(model.calls[:6], copy_model.calls[:6]):
            torch.testing.assert_close(old, new, rtol=0, atol=0)  # future labels never change own inputs

    def test_real_axis_fraction_vjepa_has_head_gradients_and_frozen_codec_map(self):
        from test_single_enemy_core import model as real_world
        from turkeyworld.models import NeuralWorldModel
        previous = real_world()
        world = NeuralWorldModel(replace(previous.config, player_motion_mode='axis_fraction'))
        world.load_state_dict(previous.state_dict(), strict=False)
        configure_component_phase(world, 'dynamics')
        data = source_batch(context=1, horizon=4, batch_size=1)
        optimizer = torch.optim.AdamW([p for p in world.parameters() if p.requires_grad], lr=1e-3)
        with patch.object(world, 'render', side_effect=AssertionError('No RGB path')):
            metrics = short_motion_backward(world, data, context=1, observed_context=1)
        self.assertTrue(torch.isfinite(world.vjepa_backend.player_pass_head[-1].weight.grad).all())
        self.assertGreater(float(world.vjepa_backend.player_pass_head[-1].weight.grad.norm()), 0.)
        before = world.vjepa_backend.player_pass_head[-1].weight.detach().clone()
        self.assertTrue(all(p.grad is None for p in world.flow_renderer.parameters()))
        self.assertTrue(all(p.grad is None for p in world.vjepa_backend.renderer.map_encoder.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, world.vjepa_backend.player_pass_head[-1].weight))
        self.assertEqual(metrics['motion_aux_train_loss'], metrics['motion_aux_loss'])

    def test_long_single_context_changes_only_observations_not_start_state_or_450_carry(self):
        world = SpyWorld()
        data = long_batch(horizon=450)
        target = EMATargetEncoder(world.core.encoder)
        metrics = long_dynamics_backward(world, target, data, observed_context=1)
        self.assertEqual(metrics['initial_observed_frames'], 1)
        self.assertEqual(metrics['initial_observed_frame_indices'], [1])
        self.assertEqual(len(world.transitions), 450)
        torch.testing.assert_close(world.core.encoder.calls[0], data['frames'][:, 1:2], rtol=0, atol=0)
        torch.testing.assert_close(world.transitions[0]['state'], data['states'][:, 1], rtol=0, atol=0)
        self.assertEqual(metrics['long_tbptt_chunks'], 29)
        self.assertEqual(metrics['long_memory_reset_count'], 0)


if __name__ == '__main__': unittest.main()
