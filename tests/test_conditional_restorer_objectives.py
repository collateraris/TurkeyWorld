"""Real RF3/codec + conditional attention: only the new restorer learns."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import torch

from test_renderer_rf5_objectives import model, source
from turkeyworld.conditional_decoder_restorer import ConditionalDecoderRestorer
from turkeyworld.conditional_restorer_objectives import (
    PROTOCOL, TRAINABLE_PREFIXES, conditional_restorer_protocol,
    configure_conditional_restorer, generate_conditional_restorer_inputs,
    conditional_restorer_predictions, conditional_restorer_loss)
from turkeyworld.latent_flow_renderer import LatentFlowRenderer
from turkeyworld.renderer_student import _hash_tensors


def network():
    result = model(3)
    private_geometry = deepcopy(result.flow_renderer.geometry_opacity)
    # Actual RF integrator and VQ use the production 16-channel interface;
    # the frozen fixture's world remains small and independent of codec size.
    result.flow_renderer = LatentFlowRenderer(kind='student', image_size=32,
        mode='rectified_flow', steps=3, width=16, depth=1, heads=2, mlp_hidden=32,
        codec_config=dict(image_size=32, base_channels=4, latent_channels=8,
                          embedding_dim=16, n_embed=8, data_init=False))
    result.flow_renderer.codec_loaded.fill_(True)
    result.flow_renderer.geometry_opacity = private_geometry
    result.flow_renderer.conditional_restorer = ConditionalDecoderRestorer()
    configure_conditional_restorer(result)
    return result


class ConditionalRestorerObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_actual_RF_codes_then_two_frozen_D_decodes_and_only65_restorer_gradients(self):
        net, batch = network(), source()
        for key in ('state', 'target', 'background', 'object_mask'):
            batch[key].requires_grad_(True)
        frozen = {name: value.clone() for name, value in net.state_dict().items()
                  if not name.startswith(TRAINABLE_PREFIXES)}
        with patch.object(net.flow_renderer, 'latent_sample', wraps=net.flow_renderer.latent_sample) as rf, \
             patch('turkeyworld.engine.Game.step', side_effect=AssertionError('No Source transition')), \
             patch('turkeyworld.engine.Game.render', side_effect=AssertionError('No Source rendering')):
            prepared = generate_conditional_restorer_inputs(net, batch)
        self.assertEqual(rf.call_count, 1)
        self.assertEqual(prepared[0].shape, (2, 16, 8, 8))
        self.assertTrue(all(not p.requires_grad for p in net.parameters()))
        codec = net.flow_renderer.codec
        actual_decode = codec.decode
        seen = []
        def decode(codes):
            seen.append((torch.is_grad_enabled(), codes.requires_grad))
            return actual_decode(codes)
        with patch.object(codec, 'decode', side_effect=decode), \
             patch.object(codec, 'encode', side_effect=AssertionError('Prepared loss cannot call E')), \
             patch.object(codec.quantize, 'forward', side_effect=AssertionError('Prepared loss cannot call Q')), \
             patch.object(net.flow_renderer, 'latent_sample', side_effect=AssertionError('No RF after restorer enable')):
            loss, metrics = conditional_restorer_loss(net, prepared=prepared)
            loss.backward()
        self.assertEqual(seen, [(False, False), (False, False)])
        wanted = {n for n, _ in net.named_parameters() if n.startswith(TRAINABLE_PREFIXES)}
        self.assertEqual(len(wanted), 65)
        for name, parameter in net.named_parameters():
            if name in wanted:
                self.assertTrue(parameter.requires_grad, name)
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            else:
                self.assertFalse(parameter.requires_grad, name)
                self.assertIsNone(parameter.grad, name)
        restorer = net.flow_renderer.conditional_restorer
        self.assertGreater(float(restorer.output.weight.grad.abs().sum()), 0)
        optimizer = torch.optim.AdamW(restorer.parameters(), lr=1e-4)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        # Once the zero output opens, both actual cross-attention blocks learn.
        loss, _ = conditional_restorer_loss(net, prepared=prepared)
        loss.backward(); optimizer.step()
        for block in restorer.blocks:
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in block.parameters()), 0)
        for name, value in frozen.items():
            self.assertTrue(torch.equal(value, net.state_dict()[name]), name)
        for key in ('state', 'target', 'background', 'object_mask'):
            self.assertIsNone(batch[key].grad, key)
        self.assertEqual(metrics['protocol'], PROTOCOL)
        self.assertEqual(metrics['branch_weights'], dict(rf_quantized=.5, source_quantized=.5))
        self.assertEqual((metrics['Source_examples'], metrics['restorer_examples']), (2, 4))
        self.assertTrue(metrics['full_codec_frozen']); self.assertTrue(metrics['decoder_frozen'])
        self.assertFalse(metrics['Source_RGB_direct_restorer_input'])
        self.assertFalse(metrics['Source_RGB_into_RF_conditioning'])
        json.dumps(metrics, allow_nan=False)

    def test_zero_head_exact_parent_and_Source_targets_masks_background_never_condition_restorer(self):
        net, batch = network(), source()
        prepared = generate_conditional_restorer_inputs(net, batch)
        first = conditional_restorer_predictions(net, prepared)
        self.assertTrue(torch.equal(first['rf_old'], first['rf_new']))
        self.assertTrue(torch.equal(first['source_old'], first['source_new']))
        # Open the actual trained path so this is not merely zero-head equality.
        with torch.no_grad():
            net.flow_renderer.conditional_restorer.output.weight.normal_(0, .02)
        original = conditional_restorer_predictions(net, prepared)
        conditions = dict(prepared[2])
        for key in ('target', 'object_mask', 'background'):
            conditions[key] = 1 - conditions[key]
        observed = []
        def capture(module, args):
            observed.append(tuple(value.detach().clone() for value in args))
        hook = net.flow_renderer.conditional_restorer.register_forward_pre_hook(capture)
        try:
            changed = conditional_restorer_predictions(net, (prepared[0], prepared[1], conditions))
        finally:
            hook.remove()
        for key in original:
            self.assertTrue(torch.equal(original[key], changed[key]), key)
        self.assertEqual(len(observed), 2)
        for args in observed:
            self.assertEqual(len(args), 4)
            self.assertEqual([tuple(value.shape) for value in args],
                             [(2, 16, 8, 8), (2, 3, 32, 32), (2, 20), (2, 16, 16)])
            self.assertTrue(all(not value.requires_grad for value in args))
        self.assertTrue(torch.equal(observed[0][0], prepared[0]))
        self.assertTrue(torch.equal(observed[1][0], prepared[1]))

    def test_paired_formula_matches_legacy_region_edge_crop_boundary_weights_exactly(self):
        net = network()
        loss, metrics = conditional_restorer_loss(net, source())
        self.assertAlmostEqual(float(loss), .5 * metrics['restorer_rf_loss'] +
                               .5 * metrics['restorer_source_loss'], places=7)
        for row in metrics['branches'].values():
            self.assertAlmostEqual(row['loss'], sum(row['weighted_terms'].values()), places=6)
            self.assertAlmostEqual(row['weighted_terms']['wall_boundary'], .2 * row['boundary_l1'], places=7)
        settings = conditional_restorer_protocol()
        self.assertEqual(settings['weights'], dict(region_l1=1., ssim=.2, region_edge=.1,
            native_crop_l1=.25, native_crop_edge=.1, wall_boundary_l1=.2))
        self.assertFalse(settings['GAN_active'])
        self.assertEqual(settings['trainable_prefixes'], list(TRAINABLE_PREFIXES))

    def test_configuration_exact_new_parameters_and_does_not_mutate_parent_or_RNG(self):
        net = network()
        before, rng = _hash_tensors(net.state_dict()), torch.get_rng_state().clone()
        info = configure_conditional_restorer(net)
        self.assertEqual((info['trainable_parameters'], info['trainable_parameter_tensors']), (247539, 65))
        self.assertEqual(before, _hash_tensors(net.state_dict()))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(net.flow_renderer.conditional_restorer.training)
        self.assertFalse(net.flow_renderer.codec.training)
        self.assertFalse(net.flow_renderer.codec.decoder.training)
        self.assertFalse(net.flow_renderer.codec.quantize.training)
        settings = conditional_restorer_protocol(); settings['weights']['ssim'] = 99
        self.assertEqual(conditional_restorer_protocol()['weights']['ssim'], .2)

    def test_missing_restorer_solver_malformed_codes_and_ambiguous_inputs_rejected(self):
        net = network(); data = source()
        prepared = generate_conditional_restorer_inputs(net, data)
        with self.assertRaises(ValueError): conditional_restorer_loss(net)
        with self.assertRaises(ValueError): conditional_restorer_loss(net, data, prepared=prepared)
        with self.assertRaises(ValueError): conditional_restorer_loss(net, prepared=prepared[:2])
        with self.assertRaises(ValueError):
            conditional_restorer_loss(net, prepared=(prepared[0].requires_grad_(), *prepared[1:]))
        del net.flow_renderer.conditional_restorer
        with self.assertRaisesRegex(ValueError, 'explicit learned'):
            configure_conditional_restorer(net)


if __name__ == '__main__':
    unittest.main()
