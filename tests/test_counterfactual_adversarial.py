"""Offline missing-NPC negatives preserve GAN isolation and update boundaries."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from turkeyworld.adversarial_training import AdversarialTrainer


class ToyCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(.5))
        self.calls = []

    def forward(self, rgb, condition, state):
        value = rgb.mean((1, 2, 3))+condition.mean((1, 2, 3))*.1
        if state is not None:
            value = value+state[:, 0]*.1
        score = self.scale*value[:, None, None, None]
        self.calls.append({'rgb': rgb.detach().clone(), 'condition': condition.detach().clone(),
            'state': None if state is None else state.detach().clone(), 'training': self.training,
            'logits': [score.detach().clone(), (score*.5).detach().clone()]})
        return {'logits': [score, score*.5], 'features': [[score], [score*.5]]}


class CounterfactualAdversarialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(420)
        self.source = torch.rand(1, 3, 64, 64)
        self.fake = torch.rand_like(self.source)
        self.base = torch.rand_like(self.source)
        self.state = torch.zeros(1, 90)
        self.state[:, 3:6] = 1
        self.removed = self.source.clone()
        self.removed[:, :, 16:48, 16:48] = .25
        self.mask = torch.zeros(1, 1, 64, 64)
        self.mask[:, :, 16:48, 16:48] = 1

    def trainer(self, **options):
        config = dict(device='cpu', base_channels=4, warmup_steps=0, ramp_steps=1,
                      r1_gamma=0., r1_interval=16)
        config.update(options)
        return AdversarialTrainer(**config)

    def test_counterfactual_is_detached_same_condition_and_only_actual_generated_gets_g_gradient(self):
        trainer = self.trainer()
        generator = torch.nn.Conv2d(3, 3, 1)
        predicted = generator(self.fake).sigmoid()
        predicted.retain_grad()
        source, base = self.source.clone().requires_grad_(), self.base.clone().requires_grad_()
        state = self.state.clone().requires_grad_()
        removed, mask = self.removed.clone().requires_grad_(), self.mask.clone().requires_grad_()
        with patch.object(trainer.discriminator, 'forward', wraps=trainer.discriminator.forward) as calls:
            loss, metrics = trainer.generator_loss(predicted, source, base, state,
                counterfactual=removed, counterfactual_mask=mask)
        # Real/generated/removed D passes, G pass, eval real, eval removed.
        self.assertEqual(calls.call_count, 6)
        for call in calls.call_args_list:
            self.assertTrue(torch.equal(call.args[1], base.detach()))
            self.assertTrue(torch.equal(call.args[2], state.detach()))
        self.assertFalse(calls.call_args_list[2].args[0].requires_grad)
        self.assertTrue(torch.equal(calls.call_args_list[2].args[0], removed.detach()))
        self.assertTrue(all(parameter.grad is None for parameter in generator.parameters()))
        loss.backward()
        self.assertGreater(float(predicted.grad.abs().sum()), 0.)
        self.assertGreater(float(generator.weight.grad.abs().sum()), 0.)
        for target in (source, base, state, removed, mask):
            self.assertIsNone(target.grad)
        self.assertTrue(all(parameter.grad is None for parameter in trainer.discriminator.parameters()))
        self.assertEqual(metrics['cf_valid_fraction'], 1.)
        self.assertEqual(metrics['cf_completed_updates'], 1)
        self.assertEqual(trainer.counterfactual_updates, 1)
        with torch.no_grad():
            real_output = trainer.discriminator(source, base, state)
            removed_output = trainer.discriminator(removed, base, state)
            expected_gap = torch.stack([a.mean()-b.mean() for a, b in
                zip(real_output['logits'], removed_output['logits'])]).mean()
        self.assertAlmostEqual(metrics['presence_score_gap'], float(expected_gap), places=7)

    def test_mixed_batch_normalizes_negatives_per_image_and_preserves_natural_wall(self):
        trainer = self.trainer()
        critic = ToyCritic()
        trainer.discriminator = critic
        trainer.optimizer = torch.optim.SGD(critic.parameters(), lr=0.)
        source = torch.stack([torch.full((3, 64, 64), .6), torch.full((3, 64, 64), .4)])
        fake = torch.stack([torch.full((3, 64, 64), .2), torch.full((3, 64, 64), .3)]).requires_grad_()
        removed = torch.stack([torch.full((3, 64, 64), .8), torch.full((3, 64, 64), .9)])
        base = torch.stack([torch.full((3, 64, 64), .1), torch.full((3, 64, 64), .7)])
        state = self.state.repeat(2, 1)
        state[:, 0] = torch.tensor([.1, .8])
        mask = torch.zeros(2, 1, 64, 64)
        mask[1, :, 16:48, 16:48] = 1
        loss, metrics = trainer.generator_loss(fake, source, base, state,
            counterfactual=removed, counterfactual_mask=mask, counterfactual_weight=3.)
        real_pass, fake_pass, removed_pass = critic.calls[:3]
        self.assertTrue(torch.equal(removed_pass['rgb'], removed[1:2]))
        self.assertTrue(torch.equal(removed_pass['condition'], base[1:2]))
        self.assertTrue(torch.equal(removed_pass['state'], state[1:2]))
        real_hinge = torch.stack([F.relu(1-value).flatten(1).mean(1) for value in real_pass['logits']]).mean(0)
        fake_hinge = torch.stack([F.relu(1+value).flatten(1).mean(1) for value in fake_pass['logits']]).mean(0)
        removed_hinge = torch.stack([F.relu(1+value).flatten(1).mean(1) for value in removed_pass['logits']]).mean(0)[0]
        expected = real_hinge.mean()+(fake_hinge[0]+(fake_hinge[1]+3*removed_hinge)/4)/2
        self.assertAlmostEqual(metrics['discriminator_loss'], float(expected), places=7)
        self.assertAlmostEqual(metrics['cf_discriminator_loss'], float(removed_hinge), places=7)
        self.assertEqual(metrics['cf_valid_fraction'], .5)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(fake.grad).all()))

    def test_empty_tiny_missing_and_disabled_counterfactuals_match_legacy_exactly(self):
        for kind in ('empty', 'tiny', 'missing_mask', 'missing_image', 'zero_weight'):
            with self.subTest(kind=kind):
                baseline, augmented = self.trainer(), self.trainer()
                augmented.load_state_dict(copy.deepcopy(baseline.state_dict()))
                first, second = self.fake.clone().requires_grad_(), self.fake.clone().requires_grad_()
                kwargs = {'counterfactual': self.removed, 'counterfactual_mask': self.mask}
                if kind == 'empty':
                    kwargs['counterfactual_mask'] = torch.zeros_like(self.mask)
                elif kind == 'tiny':
                    kwargs['counterfactual_mask'] = torch.zeros_like(self.mask)
                    kwargs['counterfactual_mask'][:, :, 0, 0] = 1
                elif kind == 'missing_mask':
                    kwargs.pop('counterfactual_mask')
                elif kind == 'missing_image':
                    kwargs.pop('counterfactual')
                else:
                    kwargs['counterfactual_weight'] = 0.
                old_loss, old_metrics = baseline.generator_loss(first, self.source, self.base, self.state)
                new_loss, metrics = augmented.generator_loss(second, self.source, self.base, self.state, **kwargs)
                self.assertTrue(torch.equal(old_loss, new_loss))
                self.assertEqual(old_metrics, metrics)
                old_loss.backward(); new_loss.backward()
                self.assertTrue(torch.equal(first.grad, second.grad))
                for name, value in baseline.discriminator.state_dict().items():
                    self.assertTrue(torch.equal(value, augmented.discriminator.state_dict()[name]), name)
                self.assertEqual(metrics['cf_completed_updates'], 0)
                self.assertEqual(metrics['cf_valid_fraction'], 0.)

    def test_cf_counter_advances_once_per_completed_group_even_if_last_micro_has_no_objects(self):
        trainer = self.trainer(warmup_steps=1)
        with patch.object(trainer.optimizer, 'step', wraps=trainer.optimizer.step) as step:
            for group in range(2):
                for micro in range(4):
                    predicted = self.fake.clone().requires_grad_()
                    mask = self.mask if group == 0 and micro == 0 else torch.zeros_like(self.mask)
                    loss, metrics = trainer.generator_loss(predicted, self.source, self.base, self.state,
                        accumulation_steps=4, microstep=micro, counterfactual=self.removed, counterfactual_mask=mask)
                    (loss/4).backward()
                    self.assertEqual(metrics['step'], group+1)
                    self.assertEqual(metrics['ramp'], float(group))
                    self.assertEqual(metrics['completed_updates'], group+int(micro == 3))
                    self.assertEqual(trainer.counterfactual_updates, int(group > 0 or micro == 3))
                    if micro < 3:
                        with self.assertRaisesRegex(RuntimeError, 'completed discriminator accumulation'):
                            trainer.state_dict()
                self.assertEqual(step.call_count, group+1)
        self.assertEqual(trainer.updates, 2)
        self.assertEqual(trainer.counterfactual_updates, 1)
        self.assertEqual(trainer.state_dict()['counterfactual_updates'], 1)

    def test_r1_remains_real_only_with_counterfactual_negative(self):
        from turkeyworld.adversarial_training import r1_penalty
        trainer = self.trainer(r1_gamma=1., r1_interval=1)
        predicted = self.fake.clone().requires_grad_()
        with patch('turkeyworld.adversarial_training.r1_penalty', wraps=r1_penalty) as r1:
            loss, metrics = trainer.generator_loss(predicted, self.source, self.base, self.state,
                counterfactual=self.removed, counterfactual_mask=self.mask)
        self.assertEqual(r1.call_count, 1)
        self.assertTrue(torch.equal(r1.call_args.args[1], self.source))
        self.assertGreater(metrics['r1'], 0.)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(predicted.grad).all()))

    def test_new_and_old_checkpoint_resume_keep_architecture_optimizer_config_and_next_update(self):
        first = self.trainer()
        loss, _ = first.generator_loss(self.fake.clone().requires_grad_(), self.source, self.base, self.state,
            counterfactual=self.removed, counterfactual_mask=self.mask)
        loss.backward()
        saved = copy.deepcopy(first.state_dict())
        self.assertEqual(saved['counterfactual_updates'], 1)
        self.assertEqual(set(saved['config']), {'weight', 'feature_weight', 'warmup_steps', 'ramp_steps',
            'r1_gamma', 'r1_interval', 'lr', 'base_channels'})
        resumed = self.trainer()
        resumed.load_state_dict(copy.deepcopy(saved))
        left, right = self.fake.clone().requires_grad_(), self.fake.clone().requires_grad_()
        first_loss, first_metrics = first.generator_loss(left, self.source, self.base, self.state,
            counterfactual=self.removed, counterfactual_mask=self.mask)
        next_loss, next_metrics = resumed.generator_loss(right, self.source, self.base, self.state,
            counterfactual=self.removed, counterfactual_mask=self.mask)
        self.assertTrue(torch.equal(first_loss, next_loss))
        self.assertEqual(first_metrics, next_metrics)
        first_loss.backward(); next_loss.backward()
        self.assertTrue(torch.equal(left.grad, right.grad))
        for name, value in first.discriminator.state_dict().items():
            self.assertTrue(torch.equal(value, resumed.discriminator.state_dict()[name]), name)
        # Legacy optimizer states have exactly the old config/parameter names;
        # only the optional top-level diagnostic counter is absent.
        legacy = copy.deepcopy(saved)
        legacy.pop('counterfactual_updates')
        old = self.trainer()
        old.load_state_dict(legacy)
        self.assertEqual(old.updates, 1)
        self.assertEqual(old.counterfactual_updates, 0)
        self.assertTrue(old.optimizer.state)
        self.assertEqual(set(old.discriminator.state_dict()), set(first.discriminator.state_dict()))

    def test_invalid_new_arguments_and_counter_state_fail_without_starting_a_group(self):
        for changes in ({'counterfactual_weight': -1.}, {'counterfactual_min_fraction': 1.1},
                        {'counterfactual_mask': torch.full_like(self.mask, float('nan'))},
                        {'counterfactual_mask': torch.zeros(2, 1, 64, 64)},
                        {'counterfactual': self.removed[:, :, :32]}):
            trainer = self.trainer()
            options = {'counterfactual': self.removed, 'counterfactual_mask': self.mask, **changes}
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                trainer.generator_loss(self.fake, self.source, self.base, self.state, **options)
            self.assertIsNone(trainer._pending_accumulation_steps)
            self.assertEqual(trainer.updates, 0)
        trainer = self.trainer()
        state = trainer.state_dict()
        state['counterfactual_updates'] = 1
        with self.assertRaisesRegex(ValueError, 'Counterfactual update count'):
            trainer.load_state_dict(state)


if __name__ == '__main__':
    unittest.main()
