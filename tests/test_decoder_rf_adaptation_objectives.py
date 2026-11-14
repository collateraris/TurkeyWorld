"""Real RF3/VQ graphs: detached code generation and isolated paired-D training."""
from copy import deepcopy
import json
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_renderer_rf5_objectives import model, source
from turkeyworld.compact_codec import compact_codec_config
from turkeyworld.vqgan import VQGAN
from turkeyworld.renderer_student import _hash_tensors
from turkeyworld.decoder_rf_adaptation_objectives import (
    PROTOCOL, TRAINABLE_PREFIXES, configure_decoder_adaptation,
    decoder_rf_adaptation_protocol, decoder_rf_adaptation_loss,
    generate_decoder_training_codes, _branch_loss)


def net():
    return model(3)


class DecoderRFAdaptationObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_code_generation_calls_actual_RF3_then_terminal_Q_never_D_and_preserves_all_tensors_and_RNG(self):
        network, data = net(), source()
        before = _hash_tensors(network.state_dict())
        cpu = torch.get_rng_state().clone()
        numpy, python = np.random.get_state(), random.getstate()
        calls = []
        def record_quantized(module, args, result):
            calls.append((args[0].detach().clone(), result['quantized'].detach().clone()))
        handle = network.flow_renderer.codec.quantize.register_forward_hook(record_quantized)
        try:
            with patch.object(network.flow_renderer.codec, 'decode', side_effect=AssertionError('D must not run during code generation')), \
                 patch.object(network.flow_renderer, 'velocity', wraps=network.flow_renderer.velocity) as velocity, \
                 patch('turkeyworld.engine.Game.render', side_effect=AssertionError('No Source rendering')):
                rf, truth, conditions = generate_decoder_training_codes(network, data)
        finally:
            handle.remove()
        self.assertEqual(velocity.call_count, 3)
        self.assertEqual(len(calls), 3)  # Q(E(Base)), Q(denormalized RF endpoint), Q(E(Source)).
        self.assertTrue(torch.equal(rf, calls[1][1]))
        self.assertTrue(torch.equal(truth, calls[2][1]))
        self.assertFalse(torch.equal(rf, calls[1][0]))  # Quantized, not continuous endpoint.
        self.assertTrue(torch.equal(conditions['target'], data['target']))
        self.assertFalse(rf.requires_grad); self.assertFalse(truth.requires_grad)
        self.assertFalse(torch.is_inference(rf)); self.assertFalse(torch.is_inference(truth))
        self.assertTrue(all(not p.requires_grad for p in network.parameters()))
        self.assertTrue(all(not m.training for m in network.modules()))
        self.assertEqual(before, _hash_tensors(network.state_dict()))
        self.assertTrue(torch.equal(cpu, torch.get_rng_state()))
        self.assertTrue(np.array_equal(numpy[1], np.random.get_state()[1]))
        self.assertEqual(numpy[2:], np.random.get_state()[2:])
        self.assertEqual(python, random.getstate())

    def test_current_Source_RGB_masks_and_background_never_change_RF_codes_or_causal_features(self):
        network, data = net(), source()
        data['dynamic'] = torch.tensor([True, False])
        previous = data['previous_rgb'][0:1].clone()
        with patch.object(network, 'initialize_from_image', wraps=network.initialize_from_image) as initialize, \
             patch('turkeyworld.engine.Game.step', side_effect=AssertionError('Recorded Source only')):
            first = generate_decoder_training_codes(network, data)
        self.assertEqual(initialize.call_count, 1)
        self.assertTrue(torch.equal(initialize.call_args.args[0], previous))
        altered = deepcopy(data)
        for key in ('target', 'background', 'object_mask'):
            altered[key] = 1 - altered[key]
        second = generate_decoder_training_codes(network, altered)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[2]['features'], second[2]['features']))
        for name, value in first[2]['geometry'].items():
            self.assertTrue(torch.equal(value, second[2]['geometry'][name]), name)
        # Source-code labels are allowed through the frozen E, never through RF.
        with torch.no_grad():
            expected = network.flow_renderer.codec.encode(altered['target'])['quantized']
        self.assertTrue(torch.equal(second[1], expected))

    def test_prepared_loss_never_calls_RF_encoder_quantizer_or_guard_both_D_branches_get_gradient(self):
        network, data = net(), source()
        for key in ('target', 'background', 'object_mask', 'state'):
            data[key].requires_grad_(True)
        prepared = generate_decoder_training_codes(network, data)
        frozen = {name: value.clone() for name, value in network.state_dict().items()
                  if not name.startswith(TRAINABLE_PREFIXES)}
        codec = network.flow_renderer.codec
        outputs = []
        def retain(module, args, output):
            output.retain_grad(); outputs.append(output)
        hook = codec.decoder.register_forward_hook(retain)
        try:
            with patch.object(network.flow_renderer, 'latent_sample', side_effect=AssertionError('No RF after D enable')), \
                 patch.object(network.flow_renderer, '_check_codec', side_effect=AssertionError('No guard after D enable')), \
                 patch.object(codec, 'encode', side_effect=AssertionError('No E after D enable')), \
                 patch.object(codec.quantize, 'forward', side_effect=AssertionError('No Q after D enable')), \
                 patch('turkeyworld.engine.Game.render', side_effect=AssertionError('No Source rendering')):
                loss, metrics = decoder_rf_adaptation_loss(network, prepared=prepared)
                loss.backward()
        finally:
            hook.remove()
        self.assertEqual(len(outputs), 2)
        for output in outputs:
            self.assertGreater(float(output.grad.abs().sum()), 0)
        wanted = {n for n, _ in network.named_parameters() if n.startswith(TRAINABLE_PREFIXES)}
        for name, parameter in network.named_parameters():
            if name in wanted:
                self.assertTrue(parameter.requires_grad)
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            else:
                self.assertFalse(parameter.requires_grad, name)
                self.assertIsNone(parameter.grad, name)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in codec.decoder.parameters()), 0)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in codec.post_quant_conv.parameters()), 0)
        for key in ('target', 'background', 'object_mask', 'state'):
            self.assertIsNone(data[key].grad, key)
        optimizer = torch.optim.AdamW([p for p in network.parameters() if p.requires_grad], lr=1e-4)
        optimizer.step()
        for name, value in frozen.items():
            self.assertTrue(torch.equal(value, network.state_dict()[name]), name)
        self.assertEqual(metrics['protocol'], PROTOCOL)
        self.assertEqual((metrics['Source_examples'], metrics['decoder_examples']), (2, 4))
        self.assertEqual(metrics['branch_weights'], dict(rf_quantized=.5, source_quantized=.5))
        self.assertAlmostEqual(float(loss.detach()),
            .5 * metrics['decoder_rf_loss'] + .5 * metrics['decoder_source_loss'], places=7)
        self.assertFalse(metrics['GAN_active']); self.assertFalse(metrics['source_future_feedback'])
        for branch in metrics['branches'].values():
            self.assertAlmostEqual(branch['loss'], sum(branch['weighted_terms'].values()), places=6)
            self.assertAlmostEqual(branch['wall_boundary_error'], branch['boundary_l1'])
            self.assertGreaterEqual(branch['image_ssim'], -1)
        json.dumps(metrics, allow_nan=False)
        # The next code-generation pass freezes D; the next loss re-enables it.
        optimizer.zero_grad(set_to_none=True)
        second = generate_decoder_training_codes(network, data)
        self.assertTrue(all(not p.requires_grad for p in network.parameters()))
        next_loss, _ = decoder_rf_adaptation_loss(network, prepared=second)
        next_loss.backward(); optimizer.step()
        self.assertGreater(float(codec.decoder.conv_out.weight.grad.abs().sum()), 0)
        for name, value in frozen.items():
            self.assertTrue(torch.equal(value, network.state_dict()[name]), name)

    def test_visible_NPC_normalization_excludes_empty_images_and_empty_wall_support_is_explicit(self):
        target = torch.zeros(2, 3, 32, 32)
        prediction = torch.zeros_like(target)
        prediction[0, :, 2, 3] = .75
        prediction.requires_grad_()
        mask = torch.zeros(2, 1, 32, 32); mask[0, 0, 2, 3] = 1
        wall = dict(wall=torch.zeros_like(mask), boundary=torch.zeros_like(mask),
                    horizontal=torch.zeros(2, 1, 32, 31), vertical=torch.zeros(2, 1, 31, 32))
        loss, metrics = _branch_loss(prediction, target, mask, wall)
        self.assertEqual(metrics['foreground_images'], 1)
        self.assertEqual(metrics['background_images'], 2)
        self.assertEqual(metrics['foreground_l1'], .75)
        self.assertEqual(metrics['background_l1'], 0)
        self.assertEqual(metrics['boundary_images'], 0)
        self.assertEqual(metrics['wall_boundary_error'], 0)
        self.assertAlmostEqual(metrics['weighted_terms']['region_l1'], .375)
        self.assertAlmostEqual(metrics['weighted_terms']['wall_boundary'], 0)
        loss.backward()
        self.assertGreater(float(prediction.grad[0, :, 2, 3].abs().sum()), 0)

    def test_real_compact_decoder_policy_is_exactly_528403_parameters_and62_tensors(self):
        network = net()
        network.flow_renderer.codec = VQGAN(compact_codec_config())
        network.flow_renderer.codec.quantize.initialized.fill_(True)
        before = _hash_tensors(network.state_dict())
        rng = torch.get_rng_state().clone()
        policy = configure_decoder_adaptation(network)
        self.assertEqual(policy['trainable_parameters'], 528403)
        self.assertEqual(policy['trainable_parameter_tensors'], 62)
        self.assertEqual(before, _hash_tensors(network.state_dict()))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(network.flow_renderer.codec.training)
        self.assertFalse(network.flow_renderer.codec.quantize.training)
        self.assertFalse(network.flow_renderer.codec.encoder.training)
        self.assertTrue(network.flow_renderer.codec.decoder.training)
        settings = decoder_rf_adaptation_protocol(); settings['weights']['ssim'] = 99
        self.assertEqual(decoder_rf_adaptation_protocol()['weights']['ssim'], .2)

    def test_invalid_solver_uninitialized_codes_wrong_shapes_or_ambiguous_inputs_rejected(self):
        for mutate in (lambda n: setattr(n.flow_renderer, 'steps', 1),
                       lambda n: setattr(n.flow_renderer, 'steps', 5),
                       lambda n: n.flow_renderer.codec.quantize.initialized.fill_(False)):
            network = net(); mutate(network)
            with self.assertRaisesRegex(ValueError, 'initialized compact codec.*RF3'):
                generate_decoder_training_codes(network, source())
        network, data = net(), source()
        prepared = generate_decoder_training_codes(network, data)
        for wrong in ((prepared[0].requires_grad_(), prepared[1], prepared[2]),
                      (prepared[0][:1].detach(), prepared[1], prepared[2])):
            with self.assertRaisesRegex(ValueError, 'detached quantized embeddings'):
                decoder_rf_adaptation_loss(network, prepared=wrong)
        with self.assertRaises(ValueError): decoder_rf_adaptation_loss(network)
        with self.assertRaises(ValueError): decoder_rf_adaptation_loss(network, data, prepared=prepared)


if __name__ == '__main__':
    unittest.main()
