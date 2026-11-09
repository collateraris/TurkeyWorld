"""Explicit dense SR, safe clipping gradients and exact legacy compatibility."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from torch.nn import functional as F

from turkeyworld.models import (SingleFrameUpscaler, EDSRUpscaler, SRResidualBlock,
                                build_upscaler, load_sr_checkpoint)


class SRQualityModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_explicit64x16_profile_count_dense_blocks_and_two_learned_shuffle_stages(self):
        config = {'architecture': 'edsr', 'channels': 64, 'blocks': 16, 'scale': 4, 'res_scale': .1}
        before = dict(config)
        with torch.device('meta'):
            model = build_upscaler(config)
        self.assertEqual(config, before)
        self.assertEqual(model.config, config)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1_517_571)
        self.assertEqual(len(model.body), 16)
        self.assertTrue(all(isinstance(block, SRResidualBlock) for block in model.body))
        for block in model.body:
            self.assertEqual(block.res_scale, .1)
            self.assertEqual([module.groups for module in block.body if isinstance(module, torch.nn.Conv2d)], [1, 1])
            self.assertTrue(all(module.kernel_size == (3, 3) for module in block.body if isinstance(module, torch.nn.Conv2d)))
        self.assertEqual([module.upscale_factor for module in model.modules() if isinstance(module, torch.nn.PixelShuffle)], [2, 2])
        self.assertFalse(any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in model.modules()))

    def test_zero_residual_is_exact_bilinear_and_training_reaches_trunk_on_second_update(self):
        torch.manual_seed(91)
        model = build_upscaler(architecture='edsr', channels=8, blocks=2).train()
        low = torch.rand(2, 3, 8, 8)
        baseline = F.interpolate(low, scale_factor=4, mode='bilinear', align_corners=False)
        self.assertTrue(torch.equal(model(low), baseline))
        initial = {key: value.clone() for key, value in model.state_dict().items()}
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        target = torch.zeros_like(baseline)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            predicted = model(low)
            self.assertEqual(predicted.shape, target.shape)
            self.assertTrue(bool(((predicted >= 0) & (predicted <= 1)).all()))
            loss = F.l1_loss(predicted, target)
            loss.backward()
            self.assertGreater(float(model.output.weight.grad.abs().sum()), 0.)
            if step == 0:
                self.assertEqual(float(model.input.weight.grad.abs().sum()), 0.)
            else:
                for parameter in (model.input.weight, model.body[0].body[0].weight,
                                  model.body_output.weight, model.upsample[0].weight, model.upsample[3].weight):
                    self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                    self.assertGreater(float(parameter.grad.abs().sum()), 0.)
            optimizer.step()
        self.assertFalse(torch.equal(model.input.weight, initial['input.weight']))
        self.assertFalse(torch.equal(model.body[0].body[0].weight, initial['body.0.body.0.weight']))

    def test_saturated_forward_stays_bounded_but_training_gradient_can_correct_it(self):
        model = EDSRUpscaler(channels=4, blocks=1)
        with torch.no_grad():
            model.output.bias.fill_(2.)
        low = torch.full((1, 3, 4, 5), .5)
        train_rgb = model.train()(low)
        eval_rgb = model.eval()(low)
        self.assertTrue(torch.equal(train_rgb, eval_rgb))
        self.assertTrue(torch.equal(train_rgb, torch.ones(1, 3, 16, 20)))
        model.train().zero_grad(set_to_none=True)
        F.l1_loss(model(low), torch.zeros_like(train_rgb)).backward()
        self.assertTrue(bool((model.output.bias.grad > 0).all()))

    def test_legacy_default_count_config_keys_tensors_and_load_path_are_exact(self):
        torch.manual_seed(3)
        original = SingleFrameUpscaler()
        torch.manual_seed(3)
        factory = build_upscaler()
        self.assertIsInstance(factory, SingleFrameUpscaler)
        self.assertEqual(original.config, {'channels': 32, 'blocks': 4, 'scale': 4})
        self.assertEqual(factory.config, original.config)
        self.assertEqual(sum(p.numel() for p in factory.parameters()), 25_776)
        self.assertEqual(set(factory.state_dict()), set(original.state_dict()))
        self.assertIn('body.0.net.0.weight', factory.state_dict())
        self.assertIn('output.0.weight', factory.state_dict())
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, factory.state_dict()[key]), key)

    def test_both_architectures_checkpoint_roundtrip_render_identically(self):
        for model in (SingleFrameUpscaler(channels=4, blocks=1), EDSRUpscaler(channels=8, blocks=2)):
            with self.subTest(architecture=model.config), TemporaryDirectory() as directory:
                model.eval()
                with torch.no_grad():
                    if isinstance(model, EDSRUpscaler):
                        model.output.weight.normal_(std=.002)
                    else:
                        model.output[0].weight.normal_(std=.002)
                path = Path(directory)/'sr_for_game.pt'
                torch.save({'stage': 'sr', 'config': model.config, 'model': model.state_dict(),
                            'training_steps': 500}, path)
                restored, saved = load_sr_checkpoint(path)
                self.assertIs(type(restored), type(model))
                self.assertEqual(restored.config, model.config)
                self.assertEqual(saved['training_steps'], 500)
                low = torch.rand(1, 3, 32, 32)
                with torch.inference_mode():
                    self.assertTrue(torch.equal(restored(low), model(low)))
                    self.assertEqual(restored(low).shape, (1, 3, 128, 128))

    def test_unknown_architecture_and_invalid_dense_config_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown SR architecture'):
            build_upscaler(architecture='not_a_model')
        for changes in ({'channels': 0}, {'blocks': 0}, {'scale': 2}, {'res_scale': 0.}, {'res_scale': float('nan')}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                EDSRUpscaler(**changes)


if __name__ == '__main__':
    unittest.main()
