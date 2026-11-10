"""Real single-NPC Source data reaches all training losses without hidden resets."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_single_enemy_core import model as small_model
from turkeyworld.adversarial_training import AdversarialTrainer
from turkeyworld.aligned_training import aligned_short_rollout_loss
from turkeyworld.combat_render_training import combat_image_objective
from turkeyworld.jepa_training_utils import EMATargetEncoder
from turkeyworld.object_losses import balanced_image_l1, foreground_contrastive_loss
from turkeyworld.spawn_render_training import spawn_image_objective
from turkeyworld.staged_curriculum import SnapshotData, StagedSequenceData, renderer_snapshot_loss
from turkeyworld.staged_data import collect_staged
from turkeyworld.train_staged_world import StagedWorldExperiment


def activate_renderer(model):
    with torch.no_grad():
        model.flow_renderer.output.weight.normal_(std=.01)
        model.flow_renderer.residual_adapter.output.weight.normal_(std=.01)


def sampled_refinement(model, endpoint):
    generated = model.flow_renderer.sample(endpoint['base'].detach(), endpoint['state'].detach(),
        checkpoint_velocity=True, map_condition=endpoint['map_condition'])
    d = AdversarialTrainer('cpu', base_channels=4, num_scales=1, warmup_steps=0, ramp_steps=1,
                           r1_gamma=0., r1_interval=16)
    gan, metrics = d.generator_loss(generated, endpoint['target'], endpoint['base'], endpoint['state'],
        counterfactual=endpoint['background'], counterfactual_mask=endpoint['object_mask'])
    loss = gan+.5*balanced_image_l1(generated, endpoint['target'], endpoint['object_mask'])
    loss = loss+.5*foreground_contrastive_loss(generated, endpoint['target'], endpoint['background'], endpoint['object_mask'])
    return loss, metrics, d


class SingleEnemyTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_real_source_joint_world_flow_gan_combat_spawn_four_backward_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            collect_staged(root, train_clips=10, val_clips=0, stages=('moving1',), frames=8, workers=1)
            directory = root/'moving1/rgb/train'
            choices = {}
            for path in sorted(directory.glob('*.npz')):
                with np.load(path, allow_pickle=False) as saved:
                    for name, array in (('hit', saved['hit_ids'] >= 0), ('spawn', saved['respawn_flags'].any(-1))):
                        future = np.flatnonzero(array[1:])+1
                        if len(future) and name not in choices:
                            choices[name] = (path, int(future[0]))
            self.assertEqual(set(choices), {'hit', 'spawn'})
            paths = list(dict.fromkeys(choices[name][0] for name in ('hit', 'spawn')))
            data = StagedSequenceData(directory, mode='stream', episode_paths=paths)
            pairs = [(paths.index(choices[name][0]), choices[name][1]-1) for name in ('hit', 'spawn')]
            batch = data.batch(2, context=2, horizon=1, indices=pairs)
            self.assertEqual(batch['states'].shape, (2, 3, 20))
            self.assertEqual(batch['attack_flags'].shape, (2, 1, 1))
            world = small_model().train(); activate_renderer(world)
            primary, metrics, endpoint = aligned_short_rollout_loss(world, EMATargetEncoder(world.core.encoder),
                batch, context=2, object_region_boost=4., detach_flow_base=True)
            primary.backward()
            self.assertEqual(metrics['global_spawn_event_count'], 1.)
            self.assertEqual(endpoint['combat_payload']['contexts']['state'].shape, (2, 1, 20))
            self.assertEqual(endpoint['combat_payload']['contexts']['spatial'].shape[2], 2)
            refinement, gan_metrics, discriminator = sampled_refinement(world, endpoint)
            refinement.backward()
            combat, combat_metrics = combat_image_objective(world, endpoint['combat_payload'])
            combat.backward()
            spawn, spawn_metrics = spawn_image_objective(world, endpoint['combat_payload'])
            spawn.backward()
            self.assertGreater(combat_metrics['combat_image_selected_samples'], 0)
            self.assertEqual(spawn_metrics['spawn_image_selected_samples'], 1)
            self.assertEqual(discriminator.updates, 1)
            self.assertTrue(gan_metrics['enabled'])
            self.assertTrue(all(torch.isfinite(value) for value in (primary, refinement, combat, spawn)))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in world.parameters() if p.grad is not None))
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in world.flow_renderer.codec.parameters()))
            self.assertGreater(float(world.vjepa_backend.spawn_head[-1].weight.grad.abs().sum()), 0.)
            self.assertGreater(float(world.flow_renderer.block_spatial_condition[0].output.weight.grad.abs().sum()), 0.)

    def test_real_static_snapshot_update_never_calls_or_changes_world_dynamics_or_map(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            collect_staged(root, train_clips=1, val_clips=0, stages=('static_enemy',), frames=4, workers=1)
            data = SnapshotData(root/'static_enemy/snapshots/train')
            batch = data.batch(2, indices=[(0, 0), (0, 1)])
            self.assertNotIn('actions', batch)
            world = small_model().train(); activate_renderer(world)
            before = {name: parameter.detach().clone() for name, parameter in world.named_parameters()
                      if name.startswith(('vjepa_backend.core.', 'vjepa_backend.delta_head.',
                          'vjepa_backend.spatial_encoder.', 'vjepa_backend.renderer.map_encoder.'))}
            optimizer = torch.optim.AdamW([p for p in world.parameters() if p.requires_grad], lr=1e-4)
            with patch.object(world, 'transition', side_effect=AssertionError('Static data cannot query dynamics')),\
                    patch.object(world.core, 'encode_video', side_effect=AssertionError('No image encoder')):
                loss, metrics, endpoint = renderer_snapshot_loss(world, None, batch)
                loss.backward()
                refinement, _, _ = sampled_refinement(world, endpoint)
                refinement.backward()
                optimizer.step()
            self.assertTrue(metrics['renderer_only'])
            self.assertFalse(metrics['dynamics_updated'])
            for name, parameter in world.named_parameters():
                if name in before:
                    self.assertIsNone(parameter.grad, name)
                    torch.testing.assert_close(parameter, before[name], rtol=0, atol=0, msg=name)
            self.assertTrue(all(parameter.grad is None for parameter in world.flow_renderer.codec.parameters()))
            self.assertGreater(float(world.vjepa_backend.renderer.render_condition[0].weight.grad.abs().sum()), 0.)

    def test_real_long_chunks_carry_own_state_memory_and_gradients_without_future_rgb(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            collect_staged(root, stages=('full_game',), full_train_clips=1, full_val_clips=0,
                           full_frames=65, workers=1)
            data = StagedSequenceData(root/'full_game/rgb/train', mode='stream')
            world = small_model().train()
            harness = SimpleNamespace(args=SimpleNamespace(context=2, chunk=16),
                curriculum=SimpleNamespace(state={'stage_updates': 0}), rng=np.random.default_rng(73),
                amp=lambda **unused: nullcontext())
            pools = SimpleNamespace(get=lambda name: data)
            inputs, outputs = [], []
            original = world.transition
            def tracked(state, action, maps, memory):
                inputs.append((state.detach().clone(), memory.detach().clone()))
                result = original(state, action, maps, memory)
                outputs.append((result['state'].detach().clone(), result['memory'].detach().clone()))
                return result
            with patch.object(world, 'transition', side_effect=tracked),\
                    patch.object(world, 'initialize_from_image', wraps=world.initialize_from_image) as initialize,\
                    patch.object(world.core, 'encode_video', wraps=world.core.encode_video) as encode:
                report = StagedWorldExperiment.long_backward(harness, world, pools, 1)
            self.assertEqual(report['long_horizon'], 64)
            self.assertEqual(report['long_memory_reset_count'], 0)
            self.assertEqual(len(inputs), 64)
            self.assertEqual(initialize.call_count, 1)
            self.assertEqual(encode.call_count, 1)
            self.assertEqual(encode.call_args.args[0].shape[1], 2)
            for index in (16, 32, 48):
                torch.testing.assert_close(inputs[index][0], outputs[index-1][0], rtol=0, atol=0)
                torch.testing.assert_close(inputs[index][1], outputs[index-1][1], rtol=0, atol=0)
            self.assertGreater(float(world.vjepa_backend.delta_head[-1].weight.grad.abs().sum()), 0.)
            self.assertGreater(float(world.core.predictor.projection.weight.grad.abs().sum()), 0.)
            self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in world.parameters() if parameter.grad is not None))
            self.assertTrue(all(parameter.grad is None for parameter in world.flow_renderer.parameters()))


if __name__ == '__main__':
    unittest.main()
