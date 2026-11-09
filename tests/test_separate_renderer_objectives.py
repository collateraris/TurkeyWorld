"""Base reconstruction and conditional-flow regression have separate gradients."""
import tempfile
import unittest
from unittest.mock import patch

import torch

import test_aligned_training as aligned_fixture
from test_object_render_training import tiny_flow
from test_joint_pretrain_adversarial import GANHarness, GANData
from test_joint_renderer_training import TinyVQWorld
from turkeyworld.aligned_training import aligned_short_rollout_loss
from turkeyworld.jepa_training_utils import EMATargetEncoder
from turkeyworld.object_losses import object_region_weights, weighted_image_loss


class SeparateRendererObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_detach_preserves_flow_values_but_base_only_receives_rgb_gradient(self):
        torch.manual_seed(712)
        flow = tiny_flow().eval()
        # A fresh zero-initialized output would hide the conditional gradient.
        torch.nn.init.normal_(flow.output.weight, std=.02)
        base = torch.rand(1, 3, 32, 32, requires_grad=True)
        target = torch.rand_like(base, requires_grad=True)
        state = torch.zeros(1, 90)
        state[:, 3:6] = 1
        mask = torch.zeros(1, 1, 32, 32, requires_grad=True)
        with torch.no_grad():
            mask[:, :, 8:24, 8:24] = 1
        weights = object_region_weights(mask)
        noise = torch.randn(1, flow.latent_channels, flow.latent_size, flow.latent_size)
        options = dict(noise=noise, time=torch.tensor([.5]), pixel_weights=weights)
        original, _ = flow.loss(target, base, state, **options)
        original_gradient = torch.autograd.grad(original, base)[0]
        self.assertGreater(float(original_gradient.abs().sum()), 0.)
        detached, _ = flow.loss(target, base, state, detach_base=True, **options)
        torch.testing.assert_close(original, detached, rtol=0, atol=0)
        self.assertIsNone(torch.autograd.grad(detached, base, allow_unused=True, retain_graph=True)[0])
        # Detach changes the flow objective only; reconstruction still trains Base.
        rgb = weighted_image_loss(base, target.detach(), weights)
        expected = torch.autograd.grad(rgb, base, retain_graph=True)[0]
        actual = torch.autograd.grad(rgb+detached, base, retain_graph=True)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        detached.backward()
        self.assertGreater(float(flow.output.weight.grad.abs().sum()), 0.)
        self.assertIsNone(target.grad)
        self.assertIsNone(mask.grad)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in flow.codec.parameters()))

    def test_aligned_route_detaches_each_flow_condition_and_exposes_only_last_background_label(self):
        model = aligned_fixture.AlignedTrainingTests.model()
        model.flow_renderer = tiny_flow(16)
        batch = aligned_fixture.AlignedTrainingTests.batch()
        batch['target_background_frames'] = torch.rand_like(batch['target_frames'])
        with patch.object(model.flow_renderer, 'loss', wraps=model.flow_renderer.loss) as flow_loss:
            _, metrics, endpoint = aligned_short_rollout_loss(model, EMATargetEncoder(model.core.encoder),
                batch, detach_flow_base=True, render_last_only=False)
        self.assertEqual(flow_loss.call_count, 4)
        self.assertTrue(all(call.kwargs['detach_base'] for call in flow_loss.call_args_list))
        self.assertTrue(metrics['flow_base_detached'])
        torch.testing.assert_close(endpoint['background'], batch['target_background_frames'][:, -1], rtol=0, atol=0)

    def test_joint_loop_routes_negative_labels_and_preserves_four_microstep_clock(self):
        class SeparateHarness(GANHarness):
            def loss_function(self, model, ema, batch, **options):
                self.seen_options.append((options.pop('detach_flow_base'), options.pop('object_region_boost')))
                loss, metrics, endpoint = super().loss_function(model, ema, batch, **options)
                endpoint.update(object_mask=batch['mask'], background=batch['background'])
                return loss, metrics, endpoint

        torch.manual_seed(713)
        model, data = TinyVQWorld(), GANData()
        model.flow_renderer = tiny_flow(64)
        model.flow_renderer.kind, model.flow_renderer.mode, model.flow_renderer.steps = 'teacher', 'flow_matching', 12
        data.value['mask'] = torch.ones(1, 1, 64, 64, requires_grad=True)
        data.value['background'] = torch.zeros_like(data.value['target'], requires_grad=True)
        with tempfile.TemporaryDirectory() as directory:
            experiment = SeparateHarness(directory)
            experiment.seen_options = []
            for name, value in dict(detach_flow_base=True, object_aware_rendering=True,
                                    npc_region_boost=4., gan_npc_negatives=True,
                                    npc_negative_weight=1., npc_negative_min_fraction=.005).items():
                setattr(experiment.args, name, value)
            experiment.fit(model, data, 'teacher500vq_world', 271, pretrain=True)
        self.assertEqual(experiment.seen_options, [(True, 4.)]*4)
        self.assertEqual(experiment.global_step, 271)
        self.assertEqual(experiment.fragments_seen, 1084)
        self.assertEqual(experiment._adversarial_trainer.updates, 32)
        self.assertEqual(experiment._adversarial_trainer.counterfactual_updates, 1)
        self.assertEqual(experiment.records[-1]['adversarial_cf_completed_updates'], 1)
        self.assertEqual(experiment.records[-1]['adversarial_cf_valid_fraction'], 1.)
        self.assertTrue(experiment.state['renderer_training']['flow_base_detached'])
        self.assertTrue(experiment.state['renderer_training']['gan_npc_negatives'])
        self.assertEqual(experiment.state['loss_protocol'], 'joint_dynamics_flow_matching_gan_objects_separate_v2')
        self.assertIsNone(data.value['mask'].grad)
        self.assertIsNone(data.value['background'].grad)


if __name__ == '__main__':
    unittest.main()
