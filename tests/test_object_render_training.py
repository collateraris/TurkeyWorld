"""Exact NPC pixel masks supervise losses without entering world-model inputs."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
import copy
import tempfile
import unittest
from unittest.mock import patch

import torch

import test_aligned_training as aligned_fixture
from test_joint_renderer_training import TinyVQWorld
from test_joint_pretrain_adversarial import GANHarness, GANData
from turkeyworld.aligned_training import aligned_short_rollout_loss
from turkeyworld.jepa_training_utils import EMATargetEncoder
from turkeyworld.latent_flow_renderer import LatentFlowRenderer
from turkeyworld.object_losses import object_region_weights, weighted_mse, weighted_image_l1, weighted_image_loss
from turkeyworld.vqgan import VQGAN


def tiny_flow(size=32):
    config = dict(image_size=size, base_channels=8, latent_channels=16,
                  embedding_dim=4, n_embed=32, data_init=False, kmeans_iterations=0)
    flow = LatentFlowRenderer(kind='student', image_size=size, mode='rectified_flow',
                             width=32, depth=1, heads=4, codec_config=config)
    codec = VQGAN(config)
    flow.load_codec({'codec_config': codec.get_config(), 'model': codec.state_dict()})
    return flow


class ObjectRenderTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_frozen_codec(self, flow, before):
        self.assertFalse(flow.codec.training)
        for name, value in flow.codec.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)
        for parameter in flow.codec.parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)

    def test_real_latent_flow_weights_velocity_and_endpoint_and_boosts_object_gradient(self):
        torch.manual_seed(351)
        flow = tiny_flow()
        target = torch.rand(1, 3, 32, 32, requires_grad=True)
        base, state = torch.rand(1, 3, 32, 32), torch.zeros(1, STATE_DIM)
        state[:, 3:6] = 1.
        mask = torch.zeros(1, 1, 32, 32)
        mask[:, :, 8:24, 8:24] = 1.
        mask.requires_grad_()
        weights = object_region_weights(mask)
        codec_before = {name: value.clone() for name, value in flow.codec.state_dict().items()}
        with torch.no_grad():
            latent = flow.normalize_latents(flow.codec.encode(target)['quantized'])
        # A constant desired velocity isolates the foreground/background
        # gradient ratio from incidental source texture and random noise.
        noise, moment = latent-1., torch.tensor([.5])
        captured = []
        velocity = flow.velocity

        def track_velocity(*args, **kwargs):
            value = velocity(*args, **kwargs)
            value.retain_grad()
            captured.append(value)
            return value

        with patch.object(flow, 'velocity', side_effect=track_velocity):
            loss, metrics = flow.loss(target, base, state, noise=noise, time=moment, pixel_weights=weights)
        prediction = captured[0]
        endpoint = .5*noise+.5*latent+.5*prediction
        expected_velocity = weighted_mse(prediction, latent-noise, weights)
        expected_endpoint = weighted_image_l1(endpoint, latent, weights)
        torch.testing.assert_close(loss, expected_velocity+.1*expected_endpoint, rtol=0, atol=0)
        torch.testing.assert_close(metrics['flow_velocity'], expected_velocity.detach(), rtol=0, atol=0)
        torch.testing.assert_close(metrics['flow_endpoint'], expected_endpoint.detach(), rtol=0, atol=0)
        loss.backward()
        ratio = prediction.grad[0, 0, 3, 3]/prediction.grad[0, 0, 0, 0]
        self.assertAlmostEqual(float(ratio), 4., places=5)
        self.assertGreater(float(flow.output.weight.grad.abs().sum()), 0.)
        self.assertTrue(bool(torch.isfinite(flow.output.weight.grad).all()))
        self.assertIsNone(target.grad)
        self.assertIsNone(mask.grad)
        self.assert_frozen_codec(flow, codec_before)

    def weighted_batch(self):
        batch = aligned_fixture.AlignedTrainingTests.batch()
        masks = torch.zeros(1, 4, 1, 16, 16)
        for index in range(4):
            masks[:, index, :, 2+index:6+index, 4:10] = 1
        batch['target_object_masks'] = masks.requires_grad_()
        return batch

    def test_aligned_real_base_and_flow_use_exact_future_masks_only_for_losses(self):
        torch.manual_seed(352)
        model = aligned_fixture.AlignedTrainingTests.model()
        model.flow_renderer = tiny_flow(16)
        changed_model = copy.deepcopy(model)
        batch = self.weighted_batch()
        altered = {key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                   for key, value in batch.items()}
        altered['target_object_masks'] = 1-altered['target_object_masks']
        codec_before = {name: value.clone() for name, value in model.flow_renderer.codec.state_dict().items()}
        bases = []
        original_render = model.render_base

        def render(*args, **kwargs):
            value = original_render(*args, **kwargs)
            bases.append(value)
            return value

        torch.manual_seed(353)
        with patch.object(model, 'render_base', side_effect=render), \
             patch.object(model.flow_renderer, 'loss', wraps=model.flow_renderer.loss) as flow_loss:
            loss, metrics, endpoint = aligned_short_rollout_loss(model, EMATargetEncoder(model.core.encoder),
                batch, object_region_boost=4., render_last_only=False)
        self.assertEqual(flow_loss.call_count, 4)
        expected_bases = []
        for index, call in enumerate(flow_loss.call_args_list):
            expected = object_region_weights(batch['target_object_masks'][:, index], 4.)
            self.assertTrue(torch.equal(call.kwargs['pixel_weights'], expected))
            self.assertFalse(call.kwargs['pixel_weights'].requires_grad)
            expected_bases.append(weighted_image_loss(bases[index], batch['target_frames'][:, index], expected))
        self.assertAlmostEqual(metrics['base_image'], float(torch.stack(expected_bases).mean().detach()), places=6)
        torch.testing.assert_close(endpoint['pixel_weights'],
            object_region_weights(batch['target_object_masks'][:, -1]), rtol=0, atol=0)
        torch.manual_seed(353)
        changed_loss, _, changed_endpoint = aligned_short_rollout_loss(changed_model,
            EMATargetEncoder(changed_model.core.encoder), altered, object_region_boost=4., render_last_only=False)
        # Changing ONLY future target masks cannot change any main trajectory,
        # encoder observation or neural condition. It changes the objective.
        for first, second in zip(model.state_inputs[:4], changed_model.state_inputs[:4]):
            self.assertTrue(torch.equal(first, second))
        for first, second in zip(model.core.encoder.calls, changed_model.core.encoder.calls):
            self.assertTrue(torch.equal(first, second))
        torch.testing.assert_close(endpoint['ar_state'], changed_endpoint['ar_state'], rtol=0, atol=0)
        torch.testing.assert_close(endpoint['ar_memory'], changed_endpoint['ar_memory'], rtol=0, atol=0)
        torch.testing.assert_close(endpoint['base'], changed_endpoint['base'], rtol=0, atol=0)
        self.assertNotEqual(float(loss.detach()), float(changed_loss.detach()))
        loss.backward()
        self.assertGreater(float(model.feature.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.core.encoder.linear.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.flow_renderer.output.weight.grad.abs().sum()), 0.)
        self.assertIsNone(batch['target_object_masks'].grad)
        self.assert_frozen_codec(model.flow_renderer, codec_before)

    def test_enabled_object_supervision_rejects_missing_masks_before_model_queries(self):
        model, batch = aligned_fixture.AlignedTrainingTests.model(), aligned_fixture.AlignedTrainingTests.batch()
        with self.assertRaisesRegex(ValueError, 'offline target NPC masks'):
            aligned_short_rollout_loss(model, EMATargetEncoder(model.core.encoder), batch, object_region_boost=4.)
        self.assertEqual(model.state_inputs, [])
        self.assertEqual(model.core.encoder.calls, [])
        self.assertEqual(model.flow_renderer.calls, [])

    def test_joint_gan_uses_weighted_sampled_rgb_without_resetting_accumulation_or_update_clock(self):
        class ObjectGANHarness(GANHarness):
            def loss_function(self, model, ema, batch, **options):
                self.last_boost = options.pop('object_region_boost')
                loss, metrics, endpoint = super().loss_function(model, ema, batch, **options)
                endpoint['pixel_weights'] = object_region_weights(batch['object_mask'], self.last_boost)
                endpoint['object_mask'] = batch['object_mask']
                return loss, metrics, endpoint

        torch.manual_seed(354)
        model, data = TinyVQWorld(), GANData()
        model.flow_renderer = tiny_flow(64)
        model.flow_renderer.kind, model.flow_renderer.mode, model.flow_renderer.steps = 'teacher', 'flow_matching', 12
        data.value['object_mask'] = torch.zeros(1, 1, 64, 64)
        data.value['object_mask'][:, :, 8:32, 12:40] = 1
        data.value['object_mask'].requires_grad_()
        expected = object_region_weights(data.value['object_mask'])
        real_l1, weighted_sample_calls = weighted_image_l1, []

        def record_l1(prediction, target, weights=None):
            if weights is not None and weights.shape == expected.shape and torch.equal(weights, expected):
                weighted_sample_calls.append(True)
            return real_l1(prediction, target, weights)

        with tempfile.TemporaryDirectory() as directory:
            experiment = ObjectGANHarness(directory)
            experiment.args.object_aware_rendering = True
            experiment.args.npc_region_boost = 4.
            experiment.args.optimizer_update_budget = 6000
            with patch('turkeyworld.object_losses.weighted_image_l1', side_effect=record_l1):
                experiment.fit(model, data, 'teacher500vq_world', 271, pretrain=True)
        self.assertEqual(experiment.last_boost, 4.)
        self.assertEqual(len(weighted_sample_calls), 4)
        self.assertEqual(data.calls, 4)
        self.assertEqual(experiment.global_step, 271)
        self.assertEqual(experiment.fragments_seen, 1084)
        self.assertEqual(experiment._adversarial_trainer.updates, 32)
        self.assertEqual(experiment.args.optimizer_update_budget, 6000)
        self.assertEqual(experiment.state['loss_protocol'], 'joint_dynamics_flow_matching_gan_objects_v1')
        self.assertIn('sampled_npc_rgb_mae', experiment.records[-1])
        self.assertIsNone(data.value['object_mask'].grad)


if __name__ == '__main__':
    unittest.main()
