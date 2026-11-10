"""Real optimizer boundaries for isolated Base/Flow, dynamics and joint phases."""
import copy
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from test_single_enemy_core import model as small_model
from turkeyworld.adversarial_training import AdversarialTrainer
from turkeyworld.aligned_training import aligned_short_rollout_loss
from turkeyworld.component_training import (configure_component_phase, render_only_loss,
                                            dynamics_only_loss)
from turkeyworld.engine import Game
from turkeyworld.jepa_training_utils import EMATargetEncoder
from turkeyworld.object_losses import balanced_image_l1


def source_batch():
    game = Game(seed=173, endless=True)
    game.enemies[0].update(alive=False, hp=0., death_timer=.6,
                           x=game.player['x'], y=game.player['y'])
    frames, states = [], []
    for step in range(4):
        frames.append(game.render(32, overlays=False))
        states.append(game.state_vector())
        if step < 3:
            game.step([1., 0., 0., 0.])
    frames = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None].float()/255
    states = torch.from_numpy(np.stack(states))[None]
    return {'states': states, 'frames': frames, 'context_frames': frames[:, :2],
        'previous_states': states[:, 1:-1], 'target_states': states[:, 2:],
        'target_frames': frames[:, 2:], 'actions': torch.tensor([[[1., 0., 0., 0.]]*2]),
        'events': torch.zeros(1, 2, 4), 'hit_ids': torch.full((1, 2), -1, dtype=torch.long),
        'attack_flags': torch.zeros(1, 2, 1), 'map': torch.tensor(game.map)[None].long(),
        'target_object_masks': torch.zeros(1, 2, 32, 32),
        'target_background_frames': frames[:, 2:].clone()}


class ComponentTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_configuration_reapplies_modes_and_clears_stale_gradients_without_replacing_parameters(self):
        world = small_model()
        identities = {name: id(p) for name, p in world.named_parameters()}
        for phase in ('renderer', 'dynamics', 'joint'):
            world.train()
            world.flow_renderer.output.weight.grad = torch.ones_like(world.flow_renderer.output.weight)
            status = configure_component_phase(world, phase)
            self.assertEqual(status['component_phase'], phase)
            self.assertEqual(status['map'], 'trainable' if phase == 'joint' else 'frozen')
            self.assertEqual(status['renderer_enabled'], phase != 'dynamics')
            self.assertEqual(status['dynamics_enabled'], phase != 'renderer')
            self.assertEqual(world.core.training, phase != 'renderer')
            self.assertEqual(world.vjepa_backend.renderer.training, phase != 'dynamics')
            self.assertEqual(world.flow_renderer.training, phase != 'dynamics')
            self.assertEqual(world.vjepa_backend.renderer.map_encoder.training, phase == 'joint')
            self.assertEqual(world.core.predictor.projection.weight.requires_grad, phase != 'renderer')
            self.assertEqual(world.vjepa_backend.map_projection[0].weight.requires_grad, phase != 'renderer')
            self.assertEqual(world.vjepa_backend.renderer.rgb32.weight.requires_grad, phase != 'dynamics')
            self.assertEqual(world.flow_renderer.output.weight.requires_grad, phase != 'dynamics')
            self.assertTrue(all(p.requires_grad == (phase == 'joint')
                                for p in world.vjepa_backend.renderer.map_encoder.parameters()))
            self.assertTrue(all(p.grad is None for p in world.parameters()))
            self.assertFalse(world.flow_renderer.codec.training)
            self.assertTrue(all(not p.requires_grad for p in world.flow_renderer.codec.parameters()))
        self.assertEqual(identities, {name: id(p) for name, p in world.named_parameters()})
        with self.assertRaisesRegex(ValueError, 'renderer, dynamics or joint'):
            configure_component_phase(world, 'unknown')

    def test_renderer_sequence_and_snapshot_queries_use_gt_pose_without_world_encoder_or_ema(self):
        world, batch = small_model(), source_batch()
        optimizer = torch.optim.AdamW(world.parameters(), lr=1e-3, weight_decay=.1)
        configure_component_phase(world, 'renderer')
        frozen = {name: p.detach().clone() for name, p in world.named_parameters() if not p.requires_grad}
        recorded = []
        original_render = world.render_base
        def capture(state, features, maps):
            recorded.append((state.detach().clone(), features.detach().clone(), maps.requires_grad))
            return original_render(state, features, maps)
        ema = Mock(side_effect=AssertionError('No EMA encoder or update'))
        with patch.object(world, 'transition', side_effect=AssertionError('No world transition')), \
                patch.object(world.core.encoder, 'forward', side_effect=AssertionError('No source encoder')), \
                patch.object(world.core.predictor, 'forward', side_effect=AssertionError('No dynamics predictor')), \
                patch.object(world, 'render_base', side_effect=capture):
            loss, metrics, endpoint = render_only_loss(world, ema, batch)
            loss.backward()
            generated = world.flow_renderer.sample(endpoint['base'].detach(), endpoint['state'],
                map_condition=endpoint['map_condition'], checkpoint_velocity=True)
            critic = AdversarialTrainer('cpu', base_channels=4, num_scales=1, warmup_steps=0,
                ramp_steps=1, r1_gamma=0., r1_interval=16)
            gan, _ = critic.generator_loss(generated, endpoint['target'], endpoint['base'], endpoint['state'])
            (gan+.5*balanced_image_l1(generated, endpoint['target'], endpoint['object_mask'])).backward()
            optimizer.step()
            snapshots = {'states': batch['target_states'][:, -1], 'frames': batch['target_frames'][:, -1],
                'object_masks': batch['target_object_masks'][:, -1],
                'background_frames': batch['target_background_frames'][:, -1], 'map': batch['map']}
            _, snapshot_metrics, _ = render_only_loss(world, ema, snapshots)
        self.assertTrue(metrics['renderer_only'] and snapshot_metrics['renderer_only'])
        self.assertEqual(critic.updates, 1)
        ema.assert_not_called()
        for state, features, map_requires_grad in recorded:
            torch.testing.assert_close(state, batch['target_states'][:, -1], rtol=0, atol=0)
            self.assertEqual(int(features.count_nonzero()), 0)
            self.assertFalse(map_requires_grad)
        for name, parameter in world.named_parameters():
            if name in frozen:
                torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
                self.assertIsNone(parameter.grad, name)
        self.assertGreater(float(world.flow_renderer.output.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(world.vjepa_backend.renderer.rgb32.weight.grad.abs().sum()), 0.)

    def test_dynamics_has_no_image_calls_and_keeps_renderer_weights_output_and_adam_moments_exact(self):
        world, batch = small_model(), source_batch()
        optimizer = torch.optim.AdamW(world.parameters(), lr=1e-3, weight_decay=.1)
        configure_component_phase(world, 'renderer')
        loss, _, _ = render_only_loss(world, None, batch)
        loss.backward(); optimizer.step()
        flow_weight = world.flow_renderer.output.weight
        old_moments = copy.deepcopy(optimizer.state[flow_weight])
        configure_component_phase(world, 'dynamics')
        frozen = {name: p.detach().clone() for name, p in world.named_parameters() if not p.requires_grad}
        core_before = world.core.predictor.projection.weight.detach().clone()
        def fixed_pose_rgb():
            with torch.inference_mode():
                maps = world.encode_map(batch['map'])
                base = world.render_base(batch['target_states'][:, -1], torch.zeros(1, 128), maps)
                return world.flow_renderer.sample(base, batch['target_states'][:, -1], map_condition=batch['map'])
        rgb_before = fixed_pose_rgb()
        ema = EMATargetEncoder(world.core.encoder)
        # Masks/backgrounds are image labels and must be unnecessary here.
        dynamics_batch = {k: v for k, v in batch.items() if k not in ('target_object_masks', 'target_background_frames')}
        with patch.object(world, 'render_base', side_effect=AssertionError('No Base rendering')), \
                patch.object(world.flow_renderer, 'loss', side_effect=AssertionError('No Flow objective')), \
                patch.object(world.flow_renderer, 'sample', side_effect=AssertionError('No image sampler')), \
                patch.object(ema, 'update', side_effect=AssertionError('Objective must not update EMA')):
            loss, metrics, endpoint = dynamics_only_loss(world, ema, dynamics_batch, object_region_boost=4.)
            loss.backward(); optimizer.step()
        self.assertIsNone(endpoint)
        self.assertNotIn('flow', metrics)
        self.assertNotIn('base_image', metrics)
        self.assertEqual(metrics['rgb_supervised_steps'], 0)
        self.assertGreater(float(world.vjepa_backend.map_projection[0].weight.grad.abs().sum()), 0.)
        self.assertFalse(torch.equal(core_before, world.core.predictor.projection.weight))
        for name, parameter in world.named_parameters():
            if name in frozen:
                torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
                self.assertIsNone(parameter.grad, name)
        for key, value in old_moments.items():
            torch.testing.assert_close(optimizer.state[flow_weight][key], value, rtol=0, atol=0)
        torch.testing.assert_close(rgb_before, fixed_pose_rgb(), rtol=0, atol=0)

    def test_joint_restores_world_base_flow_and_shared_map_gradients_while_codec_stays_frozen(self):
        world, batch = small_model(), source_batch()
        configure_component_phase(world, 'dynamics')
        configure_component_phase(world, 'joint')
        codec = {name: value.clone() for name, value in world.flow_renderer.codec.state_dict().items()}
        optimizer = torch.optim.AdamW(world.parameters(), lr=1e-3)
        loss, metrics, endpoint = aligned_short_rollout_loss(world, EMATargetEncoder(world.core.encoder),
            batch, object_region_boost=4., detach_flow_base=True)
        loss.backward()
        for parameter in (world.core.predictor.projection.weight,
                world.vjepa_backend.renderer.rgb32.weight, world.flow_renderer.output.weight,
                world.vjepa_backend.renderer.map_encoder.tile_embedding.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        optimizer.step()
        self.assertIn('flow', metrics)
        self.assertIsNotNone(endpoint)
        for name, value in world.flow_renderer.codec.state_dict().items():
            torch.testing.assert_close(value, codec[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in world.flow_renderer.codec.parameters()))


if __name__ == '__main__':
    unittest.main()
