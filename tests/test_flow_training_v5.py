"""Opt-in fine latent patches, in-block conditions and region-balanced Flow."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
import math
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from turkeyworld.latent_flow_renderer import LatentFlowRenderer
from turkeyworld.object_losses import (balanced_image_l1, balanced_mse, balanced_region_losses,
    balanced_edge_l1, decoded_ssim_loss)
from turkeyworld.vqgan import VQGAN


def renderer(*, improved=True, blocks=False, patch_size=2, kind='student', **options):
    codec = dict(image_size=32, base_channels=8, latent_channels=16, embedding_dim=4,
                 n_embed=32, attention=False, data_init=False, kmeans_iterations=0)
    values = dict(kind=kind, image_size=32, mode='flow_matching' if kind == 'teacher' else 'rectified_flow',
        width=32, depth=2, heads=4, adapter_width=32, adapter_depth=2, adapter_heads=4,
        codec_config=codec, patch_size=patch_size, spatial_conditioning=blocks,
        block_conditioning=blocks, improved_loss=improved)
    values.update(options)
    model = LatentFlowRenderer(**values)
    source = VQGAN(codec)
    model.load_codec({'config': source.get_config(), 'model': source.state_dict()})
    return model


def inputs(batch=2):
    state = torch.zeros(batch, STATE_DIM)
    state[:, :2] = torch.tensor([.2, .35]); state[:, 3:6] = 1.
    npc = state[:, GLOBAL_DIM:-2].reshape(batch, ENTITY_SLOTS, ENTITY_DIM)
    npc[:, :, 0] = torch.linspace(.4, .8, ENTITY_SLOTS); npc[:, :, 1] = .4
    npc[:, :, 4:6] = torch.tensor([.6, 1.])
    maps = torch.zeros(batch, 16, 16, dtype=torch.long)
    maps[:, 0], maps[:, -1], maps[:, :, 0], maps[:, :, -1] = 1, 2, 3, 4
    maps[:, 5:10, 7] = 6
    return torch.rand(batch, 3, 32, 32), state, maps


class FlowTrainingV5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_legacy_default_keys_weights_and_no_map_function_are_exact(self):
        torch.manual_seed(173)
        default = renderer(improved=False, patch_size=4)
        torch.manual_seed(173)
        explicit = renderer(improved=False, patch_size=4, block_conditioning=False)
        self.assertEqual(set(default.state_dict()), set(explicit.state_dict()))
        self.assertFalse(any('block_spatial_condition' in name for name in default.state_dict()))
        for name, value in default.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0, atol=0)
        base, state, _ = inputs()
        with torch.no_grad():
            default.output.weight.normal_(std=.02)
            explicit.load_state_dict(default.state_dict(), strict=True)
            torch.testing.assert_close(default.sample(base, state, steps=1), explicit.sample(base, state, steps=1), rtol=0, atol=0)

    def test_patch_two_is_used_by_both_networks_and_reconstructs_exact_latent_grid(self):
        model = renderer()
        for network in (model, model.residual_adapter):
            self.assertEqual((network.patch, network.grid), (2, 4))
            self.assertEqual(network.position.shape, (1, 16, 32))
            self.assertEqual(network.image_input.kernel_size, (2, 2))
            self.assertEqual(network.output.out_features, 4*2*2)
        base, state, _ = inputs()
        velocity = model.velocity(torch.randn(2, 4, 8, 8), torch.tensor([.2, .7]), torch.zeros(2, 4, 8, 8), state)
        self.assertEqual(velocity.shape, (2, 4, 8, 8))
        self.assertEqual(model.sample(base, state, steps=1).shape, (2, 3, 32, 32))
        for invalid in (0, True, 3, 16):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError): renderer(patch_size=invalid)

    def test_zero_block_branches_preserve_old_velocity_and_all_blocks_receive_gradients(self):
        torch.manual_seed(173)
        legacy = renderer(improved=False, blocks=True, patch_size=4, block_conditioning=False)
        torch.manual_seed(173)
        expanded = renderer(improved=False, blocks=True, patch_size=4)
        old_keys = set(legacy.state_dict())
        for name in old_keys:
            torch.testing.assert_close(legacy.state_dict()[name], expanded.state_dict()[name], rtol=0, atol=0)
        new = set(expanded.state_dict())-old_keys
        self.assertTrue(new)
        self.assertTrue(all(name.startswith(('block_spatial_condition.', 'block_wall_condition.',
            'residual_adapter.block_spatial_condition.', 'residual_adapter.block_wall_condition.')) for name in new))
        with torch.no_grad():
            for network in (legacy, legacy.residual_adapter): network.output.weight.normal_(std=.02)
        expanded.load_state_dict(legacy.state_dict(), strict=False)
        _, state, maps = inputs()
        noisy, base, time = torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8), torch.tensor([.3, .8])
        first = legacy.velocity(noisy, time, base, state, map_condition=maps)
        second = expanded.velocity(noisy, time, base, state, map_condition=maps)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        second.square().mean().backward()
        for network in (expanded, expanded.residual_adapter):
            for group in (network.block_spatial_condition, network.block_wall_condition):
                self.assertEqual(len(group), len(network.blocks))
                for read in group:
                    self.assertIsNotNone(read.output.weight.grad)
                    self.assertGreater(float(read.output.weight.grad.abs().sum()), 0.)

    def test_activated_in_block_reads_see_npc_direction_and_wall_map_without_engine(self):
        from turkeyworld.engine import Game
        model = renderer(blocks=True)
        _, state, maps = inputs()
        with torch.no_grad():
            for network in (model, model.residual_adapter):
                network.output.weight.normal_(std=.03)
                for read in (*network.block_spatial_condition, *network.block_wall_condition):
                    read.output.weight.normal_(std=.03)
        noisy, base, time = torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8), torch.tensor([.3, .8])
        with patch.object(Game, 'step', side_effect=AssertionError('No oracle')),\
                patch.object(Game, 'render', side_effect=AssertionError('No source RGB')):
            original = model.velocity(noisy, time, base, state, map_condition=maps)
            moved = state.clone(); moved[:, 10:12] = torch.tensor([.2, -.1])
            direction = model.velocity(noisy, time, base, moved, map_condition=maps)
            changed = maps.clone(); changed[:, 5:10, 7] = 0
            walls = model.velocity(noisy, time, base, state, map_condition=changed)
        self.assertGreater(float((direction-original).detach().abs().sum()), 0.)
        self.assertGreater(float((walls-original).detach().abs().sum()), 0.)
        original.square().mean().backward()
        for network in (model, model.residual_adapter):
            for read in network.block_spatial_condition:
                self.assertGreater(float(read.features[0].weight.grad.abs().sum()), 0.)
            for read in network.block_wall_condition:
                self.assertGreater(float(read.features[0].weight.grad.abs().sum()), 0.)

    def test_separate_region_area_normalization_empty_full_soft_and_detached_targets(self):
        for size in (8, 16):
            for width in (1, 2, 4):
                mask = torch.zeros(1, 1, size, size); mask[:, :, :width] = 1.
                prediction = (2*mask+4*(1-mask)).expand(-1, 3, -1, -1).clone().requires_grad_()
                target = torch.zeros_like(prediction, requires_grad=True)
                mask.requires_grad_()
                self.assertEqual(float(balanced_image_l1(prediction, target, mask).detach()), 3.)
                self.assertEqual(float(balanced_mse(prediction, target, mask).detach()), 10.)
                balanced_image_l1(prediction, target, mask).backward()
                self.assertIsNone(target.grad); self.assertIsNone(mask.grad)
                self.assertTrue(torch.isfinite(prediction.grad).all())
        prediction = torch.full((2, 3, 8, 8), 2., requires_grad=True)
        target = torch.zeros_like(prediction)
        for mask in (torch.zeros(2, 1, 32, 32), torch.ones(2, 1, 32, 32), torch.full((2, 1, 32, 32), .25)):
            self.assertEqual(float(balanced_image_l1(prediction, target, mask).detach()), 2.)
            self.assertEqual(float(balanced_mse(prediction, target, mask).detach()), 4.)
        torch.testing.assert_close(balanced_image_l1(prediction, target), F.l1_loss(prediction, target))

    def test_balanced_velocity_endpoint_loss_ignores_boost_weights_and_logs_unweighted_values(self):
        model = renderer(decoded_ssim_weight=0., decoded_edge_weight=0.).eval()
        base, state, _ = inputs(batch=1)
        target = torch.zeros_like(base, requires_grad=True)
        mask = torch.zeros(1, 1, 32, 32); mask[:, :, :16, :16] = 1.; mask.requires_grad_()
        latent_mask = F.interpolate(mask.detach(), (8, 8), mode='area')
        field = (2*latent_mask+4*(1-latent_mask)).expand(-1, 4, -1, -1).clone().requires_grad_()
        zeros = torch.zeros(1, 4, 8, 8)
        with patch.object(model.codec, 'encode', return_value={'quantized': zeros}),\
                patch.object(model, 'velocity', return_value=field):
            loss, metrics = model.loss(target, base, state, noise=zeros, time=torch.tensor([.5]),
                object_mask=mask, pixel_weights=torch.full_like(mask, 999.))
        self.assertAlmostEqual(float(loss.detach()), 10.15, places=5)
        self.assertEqual(float(metrics['flow_velocity']), 10.)
        self.assertEqual(float(metrics['flow_velocity_unweighted']), 13.)
        self.assertEqual(float(metrics['flow_endpoint']), 1.5)
        self.assertEqual(float(metrics['flow_endpoint_unweighted']), 1.75)
        loss.backward()
        self.assertGreater(float(field.grad.abs().sum()), 0.)
        self.assertIsNone(mask.grad); self.assertIsNone(target.grad)

    def test_decoded_endpoint_ssim_edges_have_flow_gradients_with_codec_strictly_frozen(self):
        model = renderer().train()
        with torch.no_grad(): model.output.weight.normal_(std=.01)
        base, state, _ = inputs(batch=2); base.requires_grad_()
        target = torch.rand_like(base, requires_grad=True)
        mask = torch.zeros(2, 1, 32, 32); mask[:, :, 10:20, 12:21] = 1.
        captured = []
        handle = model.codec.decoder.register_forward_hook(lambda module, args, result: captured.append(result))
        codec_before = {name: value.clone() for name, value in model.codec.state_dict().items()}
        try:
            loss, metrics = model.loss(target, base, state, time=torch.tensor([.25, .75]),
                object_mask=mask, detach_base=True)
        finally: handle.remove()
        self.assertEqual(len(captured), 1)
        image_term = .05*decoded_ssim_loss(captured[0], target, mask)+.05*balanced_edge_l1(captured[0], target, mask)
        gradient, = torch.autograd.grad(image_term, model.output.weight, retain_graph=True)
        self.assertGreater(float(gradient.abs().sum()), 0.)
        loss.backward()
        self.assertIsNone(base.grad); self.assertIsNone(target.grad)
        self.assertFalse(model.codec.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.codec.parameters()))
        for name, value in model.codec.state_dict().items():
            torch.testing.assert_close(value, codec_before[name], rtol=0, atol=0)
        self.assertAlmostEqual(float(metrics['flow_decoded_ssim_weighted']), .05*float(metrics['flow_decoded_ssim_loss']), places=6)
        self.assertAlmostEqual(float(metrics['flow_decoded_edge_weighted']), .05*float(metrics['flow_decoded_edge_l1']), places=6)

    def test_time_bin_sums_counts_aggregate_ratios_without_mean_of_means_bias(self):
        from turkeyworld.gradient_accumulation import MetricMean
        model = renderer(decoded_ssim_weight=0., decoded_edge_weight=0.).eval()
        accumulated = MetricMean(); total_sums, total_counts = [0.]*5, [0.]*5
        for times in (torch.tensor([.05, .25, .45, .65, .85]), torch.tensor([.05, .1, .9])):
            batch = len(times); base, state, _ = inputs(batch)
            zero = torch.zeros(batch, 4, 8, 8)
            field = torch.arange(1, batch+1).float()[:, None, None, None].expand_as(zero)
            with patch.object(model.codec, 'encode', return_value={'quantized': zero}),\
                    patch.object(model, 'velocity', return_value=field):
                _, metrics = model.loss(base, base, state, noise=zero, time=times)
            accumulated.add({key: float(value) for key, value in metrics.items() if key.startswith('flow_t') and key[-1] in ('m', 't')})
            self.assertEqual(sum(int(metrics[f'flow_t{i}_count']) for i in range(5)), batch)
            self.assertAlmostEqual(sum(float(metrics[f'flow_t{i}_sum']) for i in range(5))/batch,
                                   float(metrics['flow_velocity']), places=5)
            for index in range(5):
                total_sums[index] += float(metrics[f'flow_t{index}_sum'])
                total_counts[index] += float(metrics[f'flow_t{index}_count'])
        means = accumulated.mean()
        for index in range(5):
            self.assertAlmostEqual(means[f'flow_t{index}_sum']/means[f'flow_t{index}_count'],
                                   total_sums[index]/total_counts[index])

    def test_stratification_preserves_initial_logit_normal_cdf_and_explicit_uniform_mixture(self):
        batch = 256
        for fraction in (0., .25, 1.):
            model = renderer(kind='teacher', time_uniform_fraction=fraction)
            torch.manual_seed(173)
            time = model.draw_training_time(batch, 'cpu')
            logit_cdf = .5*(1+torch.erf(torch.logit(time)/math.sqrt(2)))
            quantiles = ((1-fraction)*logit_cdf+fraction*time).sort().values
            index = torch.arange(batch)
            self.assertTrue(((quantiles >= index/batch-1e-6) & (quantiles <= (index+1)/batch+1e-6)).all())
        uniform = renderer(kind='student').draw_training_time(batch, 'cpu').sort().values
        self.assertTrue(((uniform >= index/batch) & (uniform < (index+1)/batch)).all())

    def test_explicit_time_noise_override_sampling_and_empty_npc_autocast_is_finite(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                model = renderer().train()
                with torch.no_grad(): model.output.weight.normal_(std=.01)
                base, state, _ = inputs(batch=2)
                noise = torch.randn(2, 4, 8, 8)
                with patch.object(model, 'draw_training_time', side_effect=AssertionError('Explicit times supplied')),\
                        torch.autocast('cpu', dtype=dtype):
                    loss, metrics = model.loss(torch.rand_like(base), base, state, noise=noise,
                        time=torch.tensor([.1, .9]), object_mask=torch.zeros(2, 1, 32, 32))
                self.assertEqual(loss.dtype, torch.float32)
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(float(metrics['flow_velocity_foreground']), 0.)
                self.assertTrue(metrics['flow_time_explicit'])
                self.assertFalse(metrics['flow_time_stratified'])
                loss.backward()
                self.assertTrue(torch.isfinite(model.output.weight.grad).all())
                self.assertTrue(all(p.grad is None for p in model.codec.parameters()))

    def test_identical_rgb_has_zero_ssim_edges_and_missing_mask_is_honest_background_loss(self):
        image = torch.rand(2, 3, 32, 32, requires_grad=True)
        self.assertEqual(float(decoded_ssim_loss(image, image.detach()).detach()), 0.)
        self.assertEqual(float(balanced_edge_l1(image, image.detach()).detach()), 0.)
        for size in (1, 3, 8):
            tiny = image[:, :, :size, :size]
            self.assertTrue(torch.isfinite(decoded_ssim_loss(tiny, tiny.detach())))
        model = renderer().eval(); base, state, _ = inputs()
        _, metrics = model.loss(base, base, state, time=torch.tensor([.2, .8]))
        self.assertFalse(metrics['flow_npc_mask_supplied'])
        self.assertEqual(float(metrics['flow_velocity_foreground']), 0.)

    def test_ssim_uses_receptive_occupancy_and_separate_foreground_background_normalization(self):
        import turkeyworld.object_losses as losses
        target = torch.full((1, 3, 32, 32), .5, requires_grad=True)
        prediction = target.detach().clone(); prediction[:, :, 12:16, 12:16] = .9
        prediction.requires_grad_()
        mask = torch.zeros(1, 1, 32, 32); mask[:, :, 12:16, 12:16] = 1.; mask.requires_grad_()
        original = losses.balanced_region_losses
        observed = []
        def observe(first, second, supplied_mask, **kwargs):
            observed.append(supplied_mask.detach().clone())
            return original(first, second, supplied_mask, **kwargs)
        with patch.object(losses, 'balanced_region_losses', side_effect=observe):
            balanced = decoded_ssim_loss(prediction, target, mask)
        self.assertEqual(len(observed), 1)
        torch.testing.assert_close(observed[0], F.avg_pool2d(mask.detach(), 11, stride=1), rtol=0, atol=0)
        global_loss = decoded_ssim_loss(prediction, target)
        self.assertGreater(float(balanced.detach()), float(global_loss.detach()))
        balanced.backward()
        self.assertGreater(float(prediction.grad.abs().sum()), 0.)
        self.assertIsNone(mask.grad); self.assertIsNone(target.grad)
        empty = decoded_ssim_loss(prediction.detach(), target.detach(), torch.zeros_like(mask))
        torch.testing.assert_close(empty, global_loss.detach(), rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
