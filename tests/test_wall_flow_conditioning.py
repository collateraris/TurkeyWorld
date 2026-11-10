"""Immutable raw-map wall tokens condition both velocity nets without an oracle."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
import copy
from dataclasses import asdict
import unittest
from unittest.mock import patch

import torch

from turkeyworld.latent_flow_renderer import LatentFlowRenderer, LatentVelocityNetwork, WallFlowCondition
from turkeyworld.vqgan import VQGAN


def state(batch=1):
    value = torch.zeros(batch, STATE_DIM)
    value[:, :2] = torch.tensor([.2, .35])
    value[:, 3:6] = 1
    npc = value[:, GLOBAL_DIM:-2].reshape(batch, ENTITY_SLOTS, ENTITY_DIM)
    npc[:, :, 0] = torch.linspace(.4, .8, ENTITY_SLOTS)
    npc[:, :, 1] = .4
    npc[:, :, 4] = .6; npc[:, :, 5] = 1
    return value


def maps(batch=1):
    grid = torch.zeros(batch, 16, 16, dtype=torch.long)
    grid[:, 0], grid[:, -1], grid[:, :, 0], grid[:, :, -1] = 1, 2, 3, 4
    grid[:, 5:10, 7] = 6
    return grid


def codec_config():
    return dict(image_size=32, base_channels=8, latent_channels=16, embedding_dim=4,
                n_embed=32, attention=False, data_init=False, kmeans_iterations=0)


def renderer(*, enabled=True, adapter=True):
    model = LatentFlowRenderer(kind='teacher', image_size=32, width=32, depth=1, heads=4,
        codec_config=codec_config(), spatial_conditioning=enabled,
        adapter_width=32 if adapter else 0, adapter_depth=1 if adapter else 0, adapter_heads=4)
    codec = VQGAN(codec_config())
    model.load_codec({'config': codec.get_config(), 'model': codec.state_dict()})
    return model


def activate(model):
    with torch.no_grad():
        for network in (model, getattr(model, 'residual_adapter', None)):
            if network is None:
                continue
            network.output.weight.normal_(std=.02)
            network.blocks[0].modulation[-1].bias.fill_(.1)
            if hasattr(network, 'wall_condition'):
                network.wall_condition.output.weight.normal_(std=.02)


class WallFlowConditioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_default_keeps_old_keys_values_and_standalone_entity_branch_needs_no_map(self):
        torch.manual_seed(846)
        legacy = renderer(enabled=False)
        torch.manual_seed(846)
        explicit = renderer(enabled=False)
        self.assertFalse(legacy.uses_wall_conditioning)
        self.assertEqual(legacy.state_dict().keys(), explicit.state_dict().keys())
        self.assertFalse(any('wall_condition' in name for name in legacy.state_dict()))
        for name, value in legacy.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0, atol=0)
        legacy.velocity(torch.rand(1, 4, 8, 8), torch.tensor([.5]), torch.rand(1, 4, 8, 8), state())
        network = LatentVelocityNetwork(4, 8, 32, 1, 4)
        network.enable_spatial_conditioning(32)
        self.assertFalse(hasattr(network, 'wall_condition'))
        network(torch.rand(1, 4, 8, 8), torch.tensor([.5]), torch.rand(1, 4, 8, 8), state())

    def test_separate_zero_wall_branch_preserves_velocity_old_weights_and_rng(self):
        network = LatentVelocityNetwork(4, 8, 32, 1, 4)
        activate(network)
        network.enable_spatial_conditioning(32)
        inputs = (torch.rand(1, 4, 8, 8), torch.tensor([.5]), torch.rand(1, 4, 8, 8), state())
        before = network(*inputs)
        weights = copy.deepcopy(network.state_dict())
        rng = torch.get_rng_state().clone()
        network.enable_wall_conditioning()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        after = network(*inputs, map_condition=maps())
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        for name, value in weights.items():
            torch.testing.assert_close(value, network.state_dict()[name], rtol=0, atol=0)
        added = set(network.state_dict())-set(weights)
        self.assertTrue(added)
        self.assertTrue(all(name.startswith('wall_condition.') for name in added))

    def test_wall_material_and_camera_pose_change_actual_velocity_and_train_parameters(self):
        network = LatentVelocityNetwork(4, 8, 32, 1, 4)
        network.enable_wall_conditioning()
        activate(network)
        with torch.no_grad():
            for layer in network.state_input.modules():
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.zero_(); layer.bias.zero_()
        noisy, base = torch.rand(1, 4, 8, 8), torch.rand(1, 4, 8, 8)
        pose = state().requires_grad_()
        grid = maps()
        output = network(noisy, torch.tensor([.4]), base, pose, map_condition=grid)
        changed = grid.clone(); changed[:, 5:10, 7] = 0; changed[:, 8, 4:11] = 8
        moved = pose.detach().clone(); moved[:, 0] += .1
        turned = pose.detach().clone(); turned[:, 2:4] = torch.tensor([1., 0.])
        for other_grid, other_pose in ((changed, pose.detach()), (grid, moved), (grid, turned)):
            altered = network(noisy, torch.tensor([.4]), base, other_pose, map_condition=other_grid)
            self.assertFalse(torch.equal(output, altered))
        output.square().mean().backward()
        self.assertGreater(float(pose.grad[:, :2].abs().sum()), 0.)
        self.assertGreater(float(pose.grad[:, 2:4].abs().sum()), 0.)
        for parameter in (network.wall_condition.features[0].weight, network.wall_condition.material.weight,
                          network.wall_condition.query.weight, network.wall_condition.output.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)

    def test_all_floor_null_token_and_bfloat16_backward_are_finite_without_oracle(self):
        from turkeyworld.engine import Game
        module = WallFlowCondition(32)
        with torch.no_grad():
            module.output.weight.normal_(std=.02)
        pose = state(2).requires_grad_()
        with patch.object(Game, 'render', side_effect=AssertionError('No engine renderer')),\
                patch.object(Game, 'step', side_effect=AssertionError('No engine transition')):
            tokens, valid = module.wall_tokens(pose, torch.zeros(16, 16, dtype=torch.long))
            self.assertEqual(tokens.shape, (2, 257, 32))
            self.assertTrue(torch.equal(valid.sum(-1), torch.ones(2, dtype=torch.long)))
            with torch.autocast('cpu', dtype=torch.bfloat16):
                value = module(torch.rand(2, 4, 32), pose, torch.zeros(16, 16, dtype=torch.long))
                objective = value.float().square().mean()
            objective.backward()
            self.assertTrue(torch.isfinite(value).all())
            self.assertTrue(pose.grad is None or torch.isfinite(pose.grad).all())

    def test_map_required_cached_batch_exact_and_explicit_subset_overrides_cache(self):
        model = renderer(); activate(model)
        latent, base = torch.rand(1, 4, 8, 8), torch.rand(1, 4, 8, 8)
        with self.assertRaisesRegex(ValueError, 'encode_map|map_condition'):
            model.velocity(latent, torch.tensor([.3]), base, state())
        cached = maps(3)
        model.set_map_condition(cached)
        snapshot = cached.clone()
        cached.zero_()
        torch.testing.assert_close(model._map_condition, snapshot, rtol=0, atol=0)
        self.assertFalse(any(name.endswith('_map_condition') for name in model.state_dict()))
        with self.assertRaisesRegex(ValueError, 'batch'):
            model.velocity(latent, torch.tensor([.3]), base, state())
        explicit = maps()[0]
        first = model.velocity(latent, torch.tensor([.3]), base, state(), map_condition=explicit)
        second = model.velocity(latent, torch.tensor([.3]), base, state(), map_condition=explicit[None])
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertTrue(torch.equal(model._map_condition, snapshot))
        with self.assertRaisesRegex(ValueError, 'integer'):
            model.set_map_condition(explicit.float())
        with self.assertRaisesRegex(ValueError, 'material'):
            model.set_map_condition(torch.full((16, 16), 16))

    def test_both_velocity_nets_receive_exact_map_and_quality_loss_uses_explicit_map(self):
        model = renderer(); activate(model)
        grid = maps()
        received = []
        hooks = [network.wall_condition.register_forward_pre_hook(
            lambda _, args: received.append(args[2].detach().clone())) for network in (model, model.residual_adapter)]
        try:
            model.set_map_condition(torch.zeros(2, 16, 16, dtype=torch.long))
            value = model.velocity(torch.rand(1, 4, 8, 8), torch.tensor([.4]), torch.rand(1, 4, 8, 8),
                                   state(), map_condition=grid)
            value.square().mean().backward()
            self.assertEqual(len(received), 2)
            self.assertTrue(all(torch.equal(value, grid) for value in received))
            for network in (model, model.residual_adapter):
                self.assertGreater(float(network.wall_condition.output.weight.grad.abs().sum()), 0.)
            model.eval()
            loss, _ = model.loss(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32), state(),
                noise=torch.randn(1, 4, 8, 8), time=torch.tensor([.4]), map_condition=grid)
            self.assertTrue(torch.isfinite(loss))
        finally:
            for hook in hooks:
                hook.remove()

    def test_checkpoint_uses_immutable_argument_after_cache_and_original_map_change(self):
        from turkeyworld.latent_flow_renderer import checkpoint
        torch.manual_seed(974)
        plain = renderer(); activate(plain)
        checked = copy.deepcopy(plain)
        base = torch.rand(1, 3, 32, 32)
        pose = state()
        noise = torch.randn(1, 4, 8, 8)
        grid = maps()
        plain.set_map_condition(grid)
        checked.set_map_condition(grid)
        expected = plain.latent_sample(base, pose, noise=noise, steps=3)
        with patch('turkeyworld.latent_flow_renderer.checkpoint', wraps=checkpoint) as calls:
            actual = checked.latent_sample(base, pose, noise=noise, steps=3, checkpoint_velocity=True)
            self.assertEqual(calls.call_count, 6)
            self.assertEqual(len(calls.call_args_list[0].args), 7)  # fn plus six velocity inputs
            self.assertTrue(torch.equal(calls.call_args_list[0].args[-1], grid))
            checked.set_map_condition(torch.full((2, 16, 16), 7, dtype=torch.long))
            grid.zero_()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            expected.square().mean().backward()
            actual.square().mean().backward()
        expected_parameters = dict(plain.named_parameters())
        for name, parameter in checked.named_parameters():
            original = expected_parameters[name]
            if original.grad is None:
                self.assertIsNone(parameter.grad, name)
            else:
                torch.testing.assert_close(parameter.grad, original.grad, rtol=2e-5, atol=2e-7, msg=name)
        self.assertGreater(float(checked.wall_condition.features[0].weight.grad.abs().sum()), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in checked.codec.parameters()))

    def test_model_encode_map_sets_nonpersistent_raw_map_and_parameter_counts_are_exact(self):
        from turkeyworld.models import NeuralWorldModel, WorldModelConfig
        model = NeuralWorldModel(WorldModelConfig(kind='student', architecture='vjepa2_ac', image_size=32,
            endless=True, spatial_entities=True, spatial_combat=True, renderer_mode='rectified_flow',
            renderer_codec='vqgan', renderer_codec_config=codec_config(), flow_width=32, flow_depth=1,
            flow_heads=4, vjepa_width=32, vjepa_encoder_depth=1, vjepa_predictor_depth=1,
            vjepa_heads=4, vjepa_patch_size=4))
        raw = maps(2)
        features = model.encode_map(raw)
        self.assertEqual(features.shape, (2, 128, 16, 16))
        self.assertTrue(model.flow_renderer.uses_wall_conditioning)
        self.assertTrue(torch.equal(model.flow_renderer._map_condition, raw))
        self.assertNotEqual(model.flow_renderer._map_condition.data_ptr(), raw.data_ptr())
        self.assertFalse(any('_map_condition' in name for name in model.state_dict()))
        with torch.device('meta'):
            small, main, adapter = WallFlowCondition(32), WallFlowCondition(256), WallFlowCondition(512)
        self.assertEqual(sum(parameter.numel() for parameter in small.parameters()), 11_712)
        self.assertEqual(sum(parameter.numel() for parameter in main.parameters()), 88_320)
        self.assertEqual(sum(parameter.numel() for parameter in adapter.parameters()), 175_872)


if __name__ == '__main__':
    unittest.main()
