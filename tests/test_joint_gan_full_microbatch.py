"""All microbatch members train sampled RGB/GAN before one accumulation flush."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
import copy
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_joint_renderer_training import JointHarness, TinyVQWorld
from turkeyworld.adversarial_training import AdversarialTrainer
from turkeyworld.latent_flow_renderer import LatentFlowRenderer
from turkeyworld.object_losses import object_region_weights, weighted_image_loss
from turkeyworld.vqgan import VQGAN


class ThreeMemberData:
    def __init__(self):
        self.calls, self.fragments, self.batches = 0, [], []
        self.rng = torch.Generator().manual_seed(4816)

    def batch(self, batch_size, **options):
        assert batch_size == 3
        identifiers = list(range(self.calls*3+1, self.calls*3+4))
        rgb = torch.rand(3, 3, 64, 64, generator=self.rng)
        target = torch.rand(3, 3, 64, 64, generator=self.rng).requires_grad_()
        state = torch.zeros(3, STATE_DIM)
        state[:, 0] = torch.tensor(identifiers)/1000
        state[:, 3:6] = 1
        state.requires_grad_()
        mask = torch.zeros(3, 1, 64, 64)
        mask[1, :, 16:48, 16:48] = 1
        mask[2, :, 8:24, 12:40] = 1
        background = torch.where(mask.bool().expand_as(target), torch.full_like(target, .1), target.detach())
        background = background.detach().requires_grad_()
        mask.requires_grad_()
        value = {'rgb': rgb, 'target': target, 'state': state, 'mask': mask,
                 'background': background, 'identifiers': identifiers}
        self.calls += 1
        self.fragments.extend(identifiers)
        self.batches.append(value)
        return value


class FullBatchHarness(JointHarness):
    def __init__(self, directory):
        super().__init__(directory, joint=True, accumulation=16, from_scratch=True)
        self.args.batch = 3
        self.args.adversarial = self.args.adversarial_during_pretrain = True
        self.args.object_aware_rendering = self.args.gan_npc_negatives = True
        self.args.detach_flow_base = True
        self.args.npc_region_boost = 4.
        self.args.npc_negative_weight = 1.
        self.args.npc_negative_min_fraction = .005
        self.args.optimizer_update_budget = 6000
        self.global_step, self.fragments_seen = 54, 864
        self._adversarial_trainer = self._adversarial_phase = None
        self.loss_batches, self.saved_d_states = [], []

    def restore(self, model, optimizer, phase):
        super().restore(model, optimizer, phase)
        return 54

    def _setup_adversarial(self, phase):
        self._adversarial_trainer = AdversarialTrainer('cpu', base_channels=4,
            weight=.005, feature_weight=.01, warmup_steps=32, ramp_steps=128, r1_gamma=0.)
        self._adversarial_trainer.updates = 54
        self._adversarial_trainer.counterfactual_updates = 12
        self._adversarial_phase = phase

    def loss_function(self, model, ema, batch, *, render, object_region_boost,
                      detach_flow_base, **options):
        assert render and detach_flow_base
        self.loss_batches.append(batch)
        core = (model.core.encoder(torch.tensor([[1., 2.], [-1., .5], [2., 1.]]))-.5).square().mean()
        base = model.vjepa_backend.renderer(batch['rgb']).sigmoid()
        weights = object_region_weights(batch['mask'], object_region_boost)
        camera = weighted_image_loss(base, batch['target'], weights)
        noise = torch.zeros(3, 4, model.flow_renderer.latent_size, model.flow_renderer.latent_size)
        flow, values = model.flow_renderer.loss(batch['target'], base, batch['state'].detach(),
            noise=noise, time=torch.full((3,), .5), pixel_weights=weights, detach_base=True)
        loss = core+camera+flow
        metrics = {'core': float(core.detach()), 'base_image': float(camera.detach()),
                   'flow': float(flow.detach()), **{key: float(value) for key, value in values.items()}}
        return loss, metrics, {'base': base, 'state': batch['state'], 'target': batch['target'],
            'object_mask': batch['mask'], 'pixel_weights': weights, 'background': batch['background']}

    def save(self, model, optimizer, phase, step, metrics, extra=None):
        self.saved_d_states.append(copy.deepcopy(self._adversarial_trainer.state_dict()))
        return super().save(model, optimizer, phase, step, metrics, extra)


class JointGANFullMicrobatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_batch3_accumulation16_samples_and_labels_all48_fragments_with_one_g_d_update(self):
        torch.manual_seed(5416)
        model, data = TinyVQWorld(), ThreeMemberData()
        config = dict(image_size=64, base_channels=8, latent_channels=16,
            embedding_dim=4, n_embed=32, data_init=False, kmeans_iterations=0)
        model.flow_renderer = LatentFlowRenderer(kind='teacher', image_size=64,
            mode='flow_matching', steps=12, width=32, depth=1, heads=4, codec_config=config)
        codec = VQGAN(config)
        model.flow_renderer.load_codec({'codec_config': codec.get_config(), 'model': codec.state_dict()})
        frozen = {name: value.clone() for name, value in model.flow_renderer.codec.state_dict().items()}
        generated_images, adv_batches = [], []
        original_sample = model.flow_renderer.sample

        with tempfile.TemporaryDirectory() as directory:
            experiment = FullBatchHarness(directory)
            original_adv = experiment._adversarial_loss

            def sampled(base, state, **options):
                self.assertEqual(base.shape, (3, 3, 64, 64))
                self.assertEqual(state.shape, (3, STATE_DIM))
                self.assertFalse(base.requires_grad)
                self.assertFalse(state.requires_grad)
                self.assertEqual(options, {'checkpoint_velocity': True})
                current = data.batches[-1]
                self.assertEqual(state[:, 0].tolist(), current['state'][:, 0].tolist())
                self.assertFalse(torch.equal(base, current['target'].detach()))
                self.assertFalse(torch.equal(base, current['background'].detach()))
                value = original_sample(base, state, **options)
                value.retain_grad()
                generated_images.append(value)
                return value

            def adversarial(predicted, real, condition, state, **options):
                current = data.batches[-1]
                self.assertEqual(tuple(predicted.shape), (3, 3, 64, 64))
                self.assertTrue(torch.equal(real, current['target'].detach()))
                self.assertTrue(torch.equal(options['counterfactual'], current['background']))
                self.assertTrue(torch.equal(options['counterfactual_mask'], current['mask']))
                self.assertEqual(options['accumulation_steps'], 16)
                self.assertEqual(options['microstep'], len(adv_batches))
                self.assertEqual(options['counterfactual_weight'], 1.)
                adv_batches.append(3)
                return original_adv(predicted, real, condition, state, **options)

            with patch.object(model.flow_renderer, 'sample', side_effect=sampled), \
                 patch.object(experiment, '_adversarial_loss', side_effect=adversarial):
                experiment.fit(model, data, 'teacher175vq_world', 55, pretrain=True)
        self.assertEqual(data.calls, 16)
        self.assertEqual(data.fragments, list(range(1, 49)))
        self.assertEqual(sum(adv_batches), 48)
        self.assertEqual(len(generated_images), 16)
        for image in generated_images:
            self.assertIsNotNone(image.grad)
            self.assertTrue(bool(torch.isfinite(image.grad).all()))
            self.assertTrue(bool((image.grad.abs().flatten(1).sum(1) > 0).all()))
        self.assertEqual(experiment.global_step, 55)
        self.assertEqual(experiment.fragments_seen, 912)
        self.assertEqual(experiment.args.optimizer_update_budget, 6000)
        self.assertEqual(experiment._adversarial_trainer.updates, 55)
        self.assertEqual(experiment._adversarial_trainer.counterfactual_updates, 13)
        self.assertEqual(float(experiment.optimizer.state[model.flow_renderer.output.weight]['step']), 1.)
        self.assertEqual(experiment.state['renderer_training']['sampled_microbatch_size'], 3)
        self.assertEqual(experiment.state['renderer_training']['adversarial_microbatch_size'], 3)
        last = experiment.records[-1]
        self.assertEqual(last['effective_batch_size'], 48)
        self.assertEqual(last['sampled_rgb_batch_size'], 3)
        self.assertEqual(last['adversarial_batch_size'], 3)
        self.assertEqual(last['adversarial_completed_updates'], 55)
        self.assertEqual(last['adversarial_cf_completed_updates'], 13)
        self.assertAlmostEqual(last['adversarial_cf_valid_fraction'], 2/3, places=6)
        self.assertEqual(last['fragments_seen'], 912)
        self.assertTrue(experiment.saved_d_states)
        self.assertTrue(all(item['updates'] == 55 for item in experiment.saved_d_states))
        self.assertTrue(all(item['counterfactual_updates'] == 13 for item in experiment.saved_d_states))
        for batch in data.batches:
            for name in ('target', 'state', 'mask', 'background'):
                self.assertIsNone(batch[name].grad)
        self.assertFalse(model.flow_renderer.codec.training)
        for name, value in model.flow_renderer.codec.state_dict().items():
            self.assertTrue(torch.equal(value, frozen[name]), name)
        self.assertTrue(all(not parameter.requires_grad and parameter.grad is None
                            for parameter in model.flow_renderer.codec.parameters()))


if __name__ == '__main__':
    unittest.main()
