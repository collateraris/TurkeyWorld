"""The real pretraining loop can train latent CFM without a sampled GAN path."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None


if torch is not None:
    from turkeyworld.latent_flow_renderer import LatentFlowRenderer
    from turkeyworld.vjepa_training import VJEPAExperiment
    from turkeyworld.vqgan import VQGAN

    class TinyVQWorld(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(kind='teacher', renderer_codec='vqgan',
                                          renderer_mode='flow_matching')
            self.core = torch.nn.Module()
            self.core.encoder = torch.nn.Linear(2, 1, bias=False)
            with torch.no_grad():
                self.core.encoder.weight.copy_(torch.tensor([[.4, -.3]]))
            codec_config = dict(image_size=32, base_channels=8, latent_channels=16,
                embedding_dim=4, n_embed=32, data_init=False, kmeans_iterations=0)
            self.flow_renderer = LatentFlowRenderer(kind='teacher', image_size=32,
                mode='flow_matching', width=32, depth=1, heads=4, codec_config=codec_config)
            codec = VQGAN(codec_config)
            self.flow_renderer.load_codec({'config': codec.get_config(), 'model': codec.state_dict()})
            self.vjepa_backend = torch.nn.Module()
            self.vjepa_backend.renderer = torch.nn.Conv2d(3, 3, 1)

        def requires_grad_(self, enabled=True):
            super().requires_grad_(enabled)
            self.flow_renderer.codec.requires_grad_(False)
            return self

    class OneBatch:
        def __init__(self):
            generator = torch.Generator().manual_seed(1513)
            self.value = {'rgb': torch.rand(1, 3, 32, 32, generator=generator),
                          'target': torch.rand(1, 3, 32, 32, generator=generator),
                          'state': torch.zeros(1, 90)}
            self.value['state'][:, 3:6] = 1
            self.calls = 0

        def batch(self, batch_size, *, context, horizon, rng):
            assert batch_size == context == horizon == 1
            self.calls += 1
            return self.value

    class JointHarness(VJEPAExperiment):
        def __init__(self, directory, *, joint=None, accumulation=1, from_scratch=False):
            self.root, self.device = Path(directory), torch.device('cpu')
            self.started = time.time()
            self.args = SimpleNamespace(ema=.996, lr=.001, batch=1, context=1, horizon=1,
                plateau_patience=6, plateau_delta=.005, quality_passes=3,
                gradient_accumulation_steps=accumulation, log_every=1, save_every=100,
                control_run_dir=None, from_scratch=from_scratch)
            if joint is not None:
                self.args.train_renderer_during_pretrain = joint
            self.rng = np.random.default_rng(17)
            self.global_step, self.fragments_seen = 0, 0
            self.state, self.extra = {'teacher_quality': {}}, {}
            self.render_requests, self.records, self.saves = [], [], []

        def loss_function(self, model, ema, batch, *, render, **options):
            self.render_requests.append(render)
            core = (model.core.encoder(torch.tensor([[1., 2.]])) - .5).square().mean()
            base = model.vjepa_backend.renderer(batch['rgb']).sigmoid()
            camera = (base-batch['target']).square().mean()
            loss, metrics = core+camera, {'core': float(core.detach()), 'base_image': float(camera.detach())}
            if render:
                flow, values = model.flow_renderer.loss(batch['target'], base, batch['state'],
                    noise=torch.zeros(1, 4, 8, 8), time=torch.full((1,), .5))
                loss = loss + flow
                metrics.update(flow=float(flow.detach()), **{k: float(v) for k, v in values.items()})
            return loss, metrics, {'base': base, 'state': batch['state'], 'target': batch['target']}

        def restore(self, model, optimizer, phase):
            self.optimizer = optimizer
            self.codec_trainable_at_restore = any(
                parameter.requires_grad for parameter in model.flow_renderer.codec.parameters())
            self.camera_trainable_at_restore = any(
                parameter.requires_grad for parameter in model.vjepa_backend.renderer.parameters())
            return 0

        def save(self, model, optimizer, phase, step, metrics, extra=None):
            self.saves.append((step, dict(metrics)))
            return self.root / 'test-checkpoint.pt'

        def check_stop(self, *args):
            pass

        def _publish_model_sizes(self, model):
            pass

        def update(self, stage, current, total, label, **values):
            self.state.update(stage=stage, current=current, total=total, **values)

        def metric(self, value):
            self.records.append(dict(value))

        def preview_if_due(self, model):
            return None


@unittest.skipIf(torch is None, 'Optional PyTorch dependency is not installed')
class JointRendererTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def run_pretraining(self, joint=None, *, accumulation=1, from_scratch=False):
        torch.manual_seed(1412)
        model, data = TinyVQWorld(), OneBatch()
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as mocks:
            experiment = JointHarness(directory, joint=joint, accumulation=accumulation,
                                      from_scratch=from_scratch)
            for name in ('sample', 'latent_sample'):
                mocks.enter_context(patch.object(model.flow_renderer, name,
                    side_effect=AssertionError('No full Heun/Euler endpoint in joint world pretraining')))
            for name in ('_setup_adversarial', '_adversarial_loss'):
                mocks.enter_context(patch.object(experiment, name,
                    side_effect=AssertionError('No discriminator or GAN in world pretraining')))
            experiment.fit(model, data, 'teacher500vq_world', 1, pretrain=True)
        self.assertFalse(experiment.codec_trainable_at_restore)
        self.assertEqual(experiment.camera_trainable_at_restore, bool(joint and from_scratch))
        optimized = {id(parameter) for group in experiment.optimizer.param_groups for parameter in group['params']}
        self.assertTrue(optimized.isdisjoint(id(parameter) for parameter in model.flow_renderer.codec.parameters()))
        return experiment, model, data, before

    def assert_codec_and_camera_unchanged(self, model, before, *, camera_frozen=True):
        for name, value in model.state_dict().items():
            if name.startswith('flow_renderer.codec.') or (camera_frozen and name.startswith('vjepa_backend.renderer.')):
                self.assertTrue(torch.equal(value, before[name]), name)
        self.assertFalse(model.flow_renderer.codec.training)
        for parameter in model.flow_renderer.codec.parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)
        if camera_frozen:
            for parameter in model.vjepa_backend.renderer.parameters():
                self.assertIsNone(parameter.grad)

    def test_joint_pretrain_updates_zero_initialized_latent_output_and_core(self):
        experiment, model, data, before = self.run_pretraining(True)
        self.assertEqual(experiment.render_requests, [True])
        self.assertEqual(data.calls, 1)
        self.assertEqual(experiment.global_step, 1)
        self.assertEqual(float(before['flow_renderer.output.weight'].abs().sum()), 0.)
        output = model.flow_renderer.output.weight
        self.assertGreater(float(output.grad.abs().sum()), 0.)
        self.assertTrue(bool(torch.isfinite(output.grad).all()))
        self.assertGreater(float(output.detach().abs().sum()), 0.)
        self.assertFalse(torch.equal(model.core.encoder.weight, before['core.encoder.weight']))
        self.assertGreater(float(model.core.encoder.weight.grad.abs().sum()), 0.)
        self.assertIn('flow', experiment.records[-1])
        self.assert_codec_and_camera_unchanged(model, before)

    def test_joint_pretrain_accumulates_cfm_and_preserves_frozen_codec(self):
        experiment, model, data, before = self.run_pretraining(True, accumulation=2)
        self.assertEqual(experiment.render_requests, [True, True])
        self.assertEqual(data.calls, 2)
        self.assertEqual(experiment.global_step, 1)
        self.assertEqual(experiment.fragments_seen, 2)
        self.assertGreater(float(model.flow_renderer.output.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.flow_renderer.output.weight.detach().abs().sum()), 0.)
        self.assert_codec_and_camera_unchanged(model, before)

    def test_fresh_joint_pretrain_also_trains_base_camera_without_unfreezing_codec(self):
        experiment, model, data, before = self.run_pretraining(True, from_scratch=True)
        self.assertEqual(experiment.render_requests, [True])
        camera = model.vjepa_backend.renderer.weight
        self.assertGreater(float(camera.grad.abs().sum()), 0.)
        self.assertFalse(torch.equal(camera, before['vjepa_backend.renderer.weight']))
        self.assertGreater(float(model.flow_renderer.output.weight.grad.abs().sum()), 0.)
        self.assertFalse(torch.equal(model.core.encoder.weight, before['core.encoder.weight']))
        self.assertTrue(experiment.state['renderer_training']['active'])
        self.assertFalse(experiment.state['renderer_training']['adversarial_active'])
        self.assert_codec_and_camera_unchanged(model, before, camera_frozen=False)

    def test_missing_or_disabled_flag_preserves_legacy_dynamics_only_pretraining(self):
        for flag in (None, False):
            with self.subTest(flag=flag):
                experiment, model, data, before = self.run_pretraining(flag)
                self.assertEqual(experiment.render_requests, [False])
                self.assertEqual(data.calls, 1)
                for name, value in model.flow_renderer.state_dict().items():
                    self.assertTrue(torch.equal(value, before['flow_renderer.' + name]), name)
                self.assertIsNone(model.flow_renderer.output.weight.grad)
                self.assertFalse(torch.equal(model.core.encoder.weight, before['core.encoder.weight']))
                self.assertNotIn('flow', experiment.records[-1])
                self.assert_codec_and_camera_unchanged(model, before)


if __name__ == '__main__':
    unittest.main()
