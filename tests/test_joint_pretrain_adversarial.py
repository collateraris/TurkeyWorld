"""Joint world/CFM and final-image GAN graphs obey one accumulation clock."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from test_joint_renderer_training import JointHarness, TinyVQWorld, OneBatch
    from turkeyworld.adversarial_training import AdversarialTrainer
    from turkeyworld.latent_flow_renderer import LatentFlowRenderer
    from turkeyworld.vqgan import VQGAN

    class GANHarness(JointHarness):
        def __init__(self, directory):
            super().__init__(directory, joint=True, accumulation=4, from_scratch=True)
            self.args.adversarial = True
            self.args.adversarial_during_pretrain = True
            self.global_step, self.fragments_seen = 270, 1080
            self._adversarial_trainer = self._adversarial_phase = None
            self.setup_calls, self.saved_discriminators = 0, []

        def restore(self, model, optimizer, phase):
            super().restore(model, optimizer, phase)
            return 270

        def _setup_adversarial(self, phase):
            self.setup_calls += 1
            self._adversarial_trainer = AdversarialTrainer('cpu', weight=.005,
                feature_weight=.01, warmup_steps=32, ramp_steps=128, r1_gamma=0., base_channels=4)
            # Exercise the warmup boundary without running31 unrelated D groups.
            self._adversarial_trainer.updates = 31
            self.discriminator_before = {name: value.clone() for name, value
                in self._adversarial_trainer.discriminator.state_dict().items()}
            self._adversarial_phase = phase

        def loss_function(self, model, ema, batch, *, render, **options):
            self.render_requests.append(render)
            core = (model.core.encoder(torch.tensor([[1., 2.]]))-.5).square().mean()
            base = model.vjepa_backend.renderer(batch['rgb']).sigmoid()
            camera = (base-batch['target']).square().mean()
            loss, metrics = core+camera, {'core': float(core.detach()), 'base_image': float(camera.detach())}
            if render:
                noise = torch.zeros(1, model.flow_renderer.latent_channels,
                                    model.flow_renderer.latent_size, model.flow_renderer.latent_size)
                flow, flow_metrics = model.flow_renderer.loss(batch['target'], base, batch['state'],
                                                             noise=noise, time=torch.tensor([.5]))
                loss = loss+flow
                metrics.update(flow=float(flow.detach()), **{k: float(v) for k, v in flow_metrics.items()})
            return loss, metrics, {'base': base, 'state': batch['state'], 'target': batch['target']}

        def save(self, model, optimizer, phase, step, metrics, extra=None):
            # This refuses partial D groups, just as the real checkpoint path does.
            self.saved_discriminators.append(copy.deepcopy(self._adversarial_trainer.state_dict()))
            return super().save(model, optimizer, phase, step, metrics, extra)

    class GANData:
        def __init__(self):
            generator = torch.Generator().manual_seed(290)
            self.value = {'rgb': torch.rand(1, 3, 64, 64, generator=generator),
                          'target': torch.rand(1, 3, 64, 64, generator=generator),
                          'state': torch.zeros(1, STATE_DIM)}
            self.value['state'][:, 3:6] = 1
            self.value['state'].requires_grad_()
            self.calls = 0

        def batch(self, batch_size, **options):
            assert batch_size == 1
            self.calls += 1
            return self.value


@unittest.skipIf(torch is None, 'Optional PyTorch dependency is not installed')
class JointPretrainAdversarialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_general_gan_setting_does_not_enable_gan_in_pretraining_without_explicit_flag(self):
        torch.manual_seed(269)
        model, data = TinyVQWorld(), OneBatch()
        with tempfile.TemporaryDirectory() as directory:
            experiment = JointHarness(directory, joint=True, accumulation=4, from_scratch=True)
            experiment.args.adversarial = True
            experiment.args.adversarial_during_pretrain = False
            with patch.object(experiment, '_setup_adversarial', side_effect=AssertionError('No pretrain D setup')), \
                 patch.object(experiment, '_adversarial_loss', side_effect=AssertionError('No pretrain GAN')), \
                 patch.object(model.flow_renderer, 'sample', side_effect=AssertionError('No sampled refinement')):
                experiment.fit(model, data, 'teacher500vq_world', 1, pretrain=True)
        self.assertEqual(data.calls, 4)
        self.assertEqual(experiment.global_step, 1)
        self.assertEqual(experiment.state['loss_protocol'], 'joint_dynamics_flow_matching_v1')
        self.assertFalse(experiment.state['renderer_training']['adversarial_active'])
        self.assertGreater(float(model.flow_renderer.output.weight.detach().abs().sum()), 0.)

    def test_real_heun_gan_updates_once_per_four_microbatches_and_respects_d_warmup(self):
        torch.manual_seed(270)
        model = TinyVQWorld()
        codec_config = dict(image_size=64, base_channels=8, latent_channels=16,
                            embedding_dim=4, n_embed=32, data_init=False, kmeans_iterations=0)
        model.flow_renderer = LatentFlowRenderer(kind='teacher', image_size=64,
            mode='flow_matching', width=32, depth=1, heads=4, steps=12, codec_config=codec_config)
        codec = VQGAN(codec_config)
        model.flow_renderer.load_codec({'codec_config': codec.get_config(), 'model': codec.state_dict()})
        frozen = {name: value.clone() for name, value in model.flow_renderer.codec.state_dict().items()}
        core_before = model.core.encoder.weight.detach().clone()
        data = GANData()
        samples, core_gradient_checks, flow_gradient_checks = [], [], []
        core_after_world_clip = None
        real_sample, real_clip = model.flow_renderer.sample, torch.nn.utils.clip_grad_norm_

        def sampled(base, state, **options):
            self.assertFalse(base.requires_grad)
            self.assertFalse(state.requires_grad)
            self.assertTrue(options.get('checkpoint_velocity'))
            self.assertEqual(options.get('steps', model.flow_renderer.steps), 12)
            samples.append((base.detach().clone(), state.detach().clone()))
            return real_sample(base, state, **options)

        def clipped(parameters, *args, **options):
            nonlocal core_after_world_clip
            parameters = list(parameters)
            if not any(parameter is model.core.encoder.weight for parameter in parameters):
                flow_ids = {id(parameter) for parameter in model.flow_renderer.parameters()}
                self.assertTrue(all(id(parameter) in flow_ids for parameter in parameters))
                torch.testing.assert_close(model.core.encoder.weight.grad, core_after_world_clip, rtol=0, atol=0)
                flow_gradient_checks.append(True)
                return real_clip(parameters, *args, **options)
            # Core has exactly this supervised scalar loss; neither sampled RGB
            # nor GAN may add a gradient through detached render conditions.
            expected = 2*(model.core.encoder.weight.detach() @ torch.tensor([[1.], [2.]])-.5)
            expected = expected*torch.tensor([[1., 2.]])
            torch.testing.assert_close(model.core.encoder.weight.grad, expected, rtol=1e-5, atol=1e-6)
            core_gradient_checks.append(model.core.encoder.weight.grad.detach().clone())
            result = real_clip(parameters, *args, **options)
            core_after_world_clip = model.core.encoder.weight.grad.detach().clone()
            return result

        with tempfile.TemporaryDirectory() as directory:
            experiment = GANHarness(directory)
            with patch.object(model.flow_renderer, 'sample', side_effect=sampled), \
                 patch('turkeyworld.vjepa_training.torch.nn.utils.clip_grad_norm_', side_effect=clipped):
                experiment.fit(model, data, 'teacher500vq_world', 272, pretrain=True)
        self.assertEqual(experiment.setup_calls, 1)
        self.assertEqual(data.calls, 8)
        self.assertEqual(len(samples), 8)
        self.assertEqual(len(core_gradient_checks), 2)
        self.assertEqual(len(flow_gradient_checks), 2)
        self.assertEqual(experiment.global_step, 272)
        self.assertEqual(experiment.fragments_seen, 1088)
        self.assertEqual(experiment.render_requests, [True]*8)
        trainer = experiment._adversarial_trainer
        self.assertEqual(trainer.updates, 33)
        first, second = experiment.records
        self.assertEqual(first['adversarial_completed_updates'], 32)
        self.assertEqual(second['adversarial_completed_updates'], 33)
        self.assertEqual(first['adversarial_weight'], 0.)
        self.assertAlmostEqual(second['adversarial_weight'], .005/128)
        self.assertFalse(bool(first['adversarial_enabled']))
        self.assertTrue(bool(second['adversarial_enabled']))
        self.assertEqual(experiment.state['loss_protocol'], 'joint_dynamics_flow_matching_gan_v1')
        self.assertFalse(torch.equal(model.core.encoder.weight, core_before))
        self.assertGreater(float(model.flow_renderer.output.weight.detach().abs().sum()), 0.)
        self.assertTrue(any(not torch.equal(value, experiment.discriminator_before[name])
                            for name, value in trainer.discriminator.state_dict().items()))
        self.assertTrue(experiment.saved_discriminators)
        self.assertTrue(all(saved['updates'] in (32, 33) for saved in experiment.saved_discriminators))
        self.assertIsNone(trainer._pending_accumulation_steps)
        self.assertEqual(trainer._pending_microsteps, 0)
        self.assertFalse(model.flow_renderer.codec.training)
        for name, value in model.flow_renderer.codec.state_dict().items():
            self.assertTrue(torch.equal(value, frozen[name]), name)
        for parameter in model.flow_renderer.codec.parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)


if __name__ == '__main__':
    unittest.main()
