"""450-step numerical carry, one-update gradients and real tiny V-JEPA TBPTT."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from test_single_enemy_core import model as real_model
from turkeyworld.component_training import configure_component_phase
from turkeyworld.engine import Game
from turkeyworld.jepa_training_utils import EMATargetEncoder
from turkeyworld.long_dynamics_training import long_dynamics_backward
from turkeyworld.structured_state import STATE_DIM


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([.7, .5, .3, .1]))
        self.calls = []

    def forward(self, images):
        self.calls.append(images.detach().clone())
        value = images.mean((2, 3, 4))
        return value[:, :, None, None]*self.scale[None, None, None]+.1


class SpyWorld(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Module()
        self.core.encoder = Encoder()
        self.core.encode_video = self.core.encoder.forward
        self.predictor = nn.Linear(4, 4)
        with torch.no_grad():
            self.predictor.weight.copy_(torch.eye(4)*.1)
            self.predictor.bias.zero_()
        self.map_encoder = nn.Linear(1, 2).requires_grad_(False)
        self.motion = nn.Parameter(torch.tensor(.0008))
        self.initializations, self.transitions = [], []

    def encode_map(self, grid):
        return self.map_encoder(grid.float().mean((1, 2))[:, None])

    def initialize_from_latents(self, latents, states, maps):
        self.initializations.append(states.detach().clone())
        return latents[:, -1].expand(-1, 2, -1)

    def transition(self, state, action, map_features, memory):
        self.transitions.append({'state': state.detach().clone(), 'memory': memory.detach().clone(),
            'state_graph': state.grad_fn is not None, 'memory_graph': memory.grad_fn is not None})
        move = self.motion*action[:, 0]+.00001*memory[:, 0, 0]
        following = torch.cat([state[:, :1]+move[:, None], state[:, 1:-1], state[:, -1:]+1/9000], -1)
        next_memory = .99*memory+.01*self.predictor(memory)
        return {'state': following, 'memory': next_memory, 'latent_prediction': next_memory.mean(1, keepdim=True)}


def batch(horizon=450, size=2):
    states = torch.zeros(1, horizon+2, STATE_DIM)
    states[:, :, 0] = .2+torch.arange(horizon+2)*.001
    states[:, :, 1], states[:, :, 3:6] = .2, 1.
    states[:, :, -1] = torch.arange(horizon+2)/9000
    frames = (torch.arange(horizon+2).float()/500)[None, :, None, None, None].expand(1, -1, 3, size, size).clone()
    actions = torch.zeros(1, horizon, 4); actions[:, :, 0] = 1
    return {'states': states, 'frames': frames, 'context_frames': frames[:, :2],
        'actions': actions, 'target_states': states[:, 2:],
        'map': torch.zeros(1, 16, 16, dtype=torch.long)}


class LongDynamicsTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_full450_carries_own_state_memory_and_backwards_all29_chunks_without_any_reset(self):
        world, data = SpyWorld(), batch()
        ema = EMATargetEncoder(world.core.encoder)
        originals = {name: value.clone() for name, value in data.items()}
        before = {name: value.detach().clone() for name, value in world.named_parameters()}
        progress = []
        with patch.object(Game, 'step', side_effect=AssertionError('No engine')), \
                patch.object(Game, 'render', side_effect=AssertionError('No rendering')), \
                patch.object(ema, 'update', side_effect=AssertionError('Caller owns EMA update')):
            metrics = long_dynamics_backward(world, ema, data,
                progress_callback=lambda done, total: progress.append((done, total)))
        self.assertEqual(metrics['long_horizon'], 450)
        self.assertEqual(metrics['long_tbptt_chunks'], 29)
        self.assertEqual(metrics['long_forward_transitions'], 450)
        self.assertEqual(metrics['long_coordinate_offsets'], list(range(1, 451)))
        self.assertEqual(progress, [(n, 450) for n in range(16, 450, 16)]+[(450, 450)])
        self.assertEqual(len(world.core.encoder.calls), 1)
        torch.testing.assert_close(world.core.encoder.calls[0], data['frames'][:, :2], rtol=0, atol=0)
        self.assertEqual(len(world.initializations), 1)
        torch.testing.assert_close(world.initializations[0], data['states'][:, :2], rtol=0, atol=0)
        self.assertEqual([len(call[0]) for call in ema.encoder.calls], [16]*28+[2])
        torch.testing.assert_close(torch.cat(ema.encoder.calls, 1), data['frames'][:, 2:], rtol=0, atol=0)
        for offset in range(16, 450, 16):
            self.assertFalse(world.transitions[offset]['state_graph'])
            self.assertFalse(world.transitions[offset]['memory_graph'])
            # Boundary values equal the previous OWN transition's output,
            # rather than the Source pose at this already-diverged position.
            previous = world.transitions[offset-1]
            expected = previous['state'][:, 0]+(world.motion.detach()+.00001*previous['memory'][:, 0, 0])
            torch.testing.assert_close(world.transitions[offset]['state'][:, 0], expected, rtol=0, atol=0)
            self.assertFalse(torch.equal(world.transitions[offset]['state'], data['states'][:, offset+1]))
        for name, parameter in world.named_parameters():
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)  # No helper optimizer step.
        for name, value in data.items():
            torch.testing.assert_close(value, originals[name], rtol=0, atol=0)
        for parameter in (world.motion, world.predictor.weight, world.core.encoder.scale):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        self.assertEqual(ema.steps, 0)
        optimizer = torch.optim.SGD(world.parameters(), lr=1e-4)
        optimizer.step()
        self.assertFalse(torch.equal(world.motion, before['motion']))

    def test_time_weighting_and_partial_final_chunk_match_the_untruncated_forward_objective(self):
        first = SpyWorld()
        second = copy.deepcopy(first)
        first_ema, second_ema = EMATargetEncoder(first.core.encoder), EMATargetEncoder(second.core.encoder)
        data = batch()
        truncated = long_dynamics_backward(first, first_ema, data, chunk_size=16)
        full = long_dynamics_backward(second, second_ema, data, chunk_size=450)
        for key in ('train_loss', 'state', 'jepa', 'ar_xy_player', 'ar_xy_npc',
                    'ar_wall_player', 'ar_wall_npc', 'regularizer'):
            self.assertAlmostEqual(truncated[key], full[key], places=5, msg=key)
        for a, b in zip(first.transitions, second.transitions):
            torch.testing.assert_close(a['state'], b['state'], rtol=0, atol=0)
            torch.testing.assert_close(a['memory'], b['memory'], rtol=0, atol=0)
        self.assertEqual(full['long_tbptt_chunks'], 1)

    def test_real_tiny_vjepa_has_context_predictor_motion_gradients_and_frozen_renderer(self):
        world, data = real_model(), batch(horizon=17, size=32)
        configure_component_phase(world, 'dynamics')
        frozen = {name: value.detach().clone() for name, value in world.named_parameters() if not value.requires_grad}
        ema = EMATargetEncoder(world.core.encoder)
        with patch.object(world, 'render_base', side_effect=AssertionError('No Base')), \
                patch.object(world.flow_renderer, 'loss', side_effect=AssertionError('No Flow')), \
                patch.object(world.flow_renderer, 'sample', side_effect=AssertionError('No sampler')):
            metrics = long_dynamics_backward(world, ema, data, expected_horizon=17)
        self.assertEqual(metrics['long_tbptt_chunks'], 2)
        for parameter in (world.core.encoder.patch_embed.weight, world.core.predictor.projection.weight,
                          world.vjepa_backend.delta_head[-1].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        for name, parameter in world.named_parameters():
            if name in frozen:
                self.assertIsNone(parameter.grad, name)
                torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)

    def test_wrong_duration_and_nonfinite_own_predictions_fail_before_an_optimizer_update(self):
        with self.assertRaisesRegex(ValueError, 'exactly450'):
            long_dynamics_backward(SpyWorld(), EMATargetEncoder(Encoder()), batch(horizon=17))
        world = SpyWorld()
        with torch.no_grad():
            world.motion.fill_(float('nan'))
        with self.assertRaisesRegex(FloatingPointError, 'own state/memory'):
            long_dynamics_backward(world, EMATargetEncoder(world.core.encoder), batch())


if __name__ == '__main__':
    unittest.main()
