"""Pure CPU restorer checks: warm exactness, spatial capacity and strict pins."""
from copy import deepcopy
import json
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.conditional_decoder_restorer import (
    KIND, SCHEMA, ConditionalDecoderRestorer, ConditionalDecoderRestorerConfig,
    conditional_decoder_restorer_metadata, validate_conditional_decoder_restorer_state,
    load_conditional_decoder_restorer)


def inputs(size=2):
    table = torch.linspace(-.8, .8, 19 * 16).reshape(19, 16)
    indices = torch.arange(size * 64).reshape(size, 8, 8) % len(table)
    codes = table[indices].permute(0, 3, 1, 2).contiguous()
    rgb = .25 + .5 * torch.linspace(0, 1, size * 3 * 32 * 32).reshape(size, 3, 32, 32)
    state = torch.zeros(size, 20)
    state[:, :2] = 5.5 / 16
    state[:, 3:5] = 1
    state[:, 8:10] = 9.5 / 16
    state[:, 12:14] = 1
    yy, xx = torch.meshgrid(torch.arange(16), torch.arange(16), indexing='ij')
    grid = ((xx + 2 * yy) % 16).expand(size, -1, -1).clone()
    return codes, rgb, state, grid


def module_hash(module):
    return conditional_decoder_restorer_metadata(module)['state_sha256']


class ConditionalDecoderRestorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_zero_output_is_bitexact_in_train_eval_and_mixed_input_dtypes_with_private_RNG_initialization(self):
        torch_state = torch.get_rng_state().clone()
        numpy, python = np.random.get_state(), random.getstate()
        model = ConditionalDecoderRestorer(seed=173)
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertTrue(np.array_equal(numpy[1], np.random.get_state()[1]))
        self.assertEqual(python, random.getstate())
        other = ConditionalDecoderRestorer(seed=174)
        self.assertNotEqual(module_hash(model), module_hash(other))
        codes, rgb, state, grid = inputs()
        # Preserve real endpoint values0/1 too, and permit finite autonomous
        # state outside the map without clamping/correcting its coordinates.
        rgb[0, :, 0, 0] = 0; rgb[1, :, 0, 0] = 1
        state[1, :2] = torch.tensor([-.2, 1.2])
        for training in (True, False):
            model.train(training)
            before = torch.get_rng_state().clone()
            with patch('turkeyworld.engine.Game.render', side_effect=AssertionError('No Source renderer')), \
                 patch('turkeyworld.engine.Game.step', side_effect=AssertionError('No Source replay')):
                result = model.forward_with_aux(codes, rgb, state, grid)
            self.assertTrue(torch.equal(rgb, result['image']))
            self.assertEqual(float(result['delta'].detach().abs().sum()), 0)
            self.assertEqual(result['screen_tokens'].shape, (2, 64, 64))
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
        image = model(codes.half(), rgb.half(), state.float(), grid)
        self.assertEqual(image.dtype, torch.float16)
        self.assertTrue(torch.equal(image, rgb.half()))
        metadata = conditional_decoder_restorer_metadata(model)
        self.assertEqual((metadata['parameters'], metadata['parameter_tensors']), (247539, 65))
        self.assertEqual(metadata['cross_attention_memory_tokens'], 259)
        self.assertEqual(metadata['kind'], KIND); self.assertEqual(metadata['schema'], SCHEMA)
        self.assertFalse(metadata['Source_RGB_inputs']); self.assertFalse(metadata['Source_raycast_or_occlusion_oracle'])
        json.dumps(metadata, allow_nan=False)

    def test_last_layer_learns_first_then_both_deep_cross_attention_blocks_and_material_state_encoders(self):
        model = ConditionalDecoderRestorer()
        codes, rgb, state, grid = inputs()
        target = (rgb.flip(-1) * .5 + .15).detach()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=0)
        before = module_hash(model)
        first = (model(codes, rgb, state, grid) - target).square().mean()
        first.backward()
        self.assertGreater(float(model.output.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.output.bias.grad.abs().sum()), 0)
        for name, parameter in model.named_parameters():
            if not name.startswith('output.') and parameter.grad is not None:
                self.assertEqual(float(parameter.grad.abs().sum()), 0, name)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        second = (model(codes, rgb, state, grid) - target).square().mean()
        second.backward()
        for index, block in enumerate(model.blocks):
            for name, component in [('self_attention', block.self_attention),
                                    ('cross_attention', block.cross_attention), ('feedforward', block.feedforward)]:
                gradients = [p.grad for p in component.parameters() if p.grad is not None]
                self.assertTrue(all(bool(torch.isfinite(value).all()) for value in gradients))
                self.assertGreater(sum(float(value.abs().sum()) for value in gradients), 0, f'{index}/{name}')
        for component in (model.material_embedding, model.map_coordinates, model.actor_state, model.input_projection):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in component.parameters() if p.grad is not None), 0)
        optimizer.step()
        self.assertNotEqual(before, module_hash(model))

    def test_activated_restorer_reads_map_and_player_NPC_state_without_a_visibility_gate_or_Source_calls(self):
        model = ConditionalDecoderRestorer().eval()
        with torch.no_grad():
            model.output.weight.fill_(.002)
            model.output.bias.fill_(.01)
        codes, rgb, state, grid = inputs(1)
        before = module_hash(model)
        altered_state = state.clone()
        altered_state[:, :2] += .15
        altered_state[:, 2:4] = torch.tensor([1., 0.])
        altered_state[:, 8:10] -= .2
        altered_state[:, 12:16] = 0  # No enemy-alive hard gate is imposed.
        with torch.no_grad(), patch('turkeyworld.engine.Game.render', side_effect=AssertionError('No Source oracle')):
            original = model(codes, rgb, state, grid)
            changed_state = model(codes, rgb, altered_state, grid)
            changed_map = model(codes, rgb, state, (grid + 7) % 16)
        self.assertGreater(float((original - changed_state).abs().max()), 1e-6)
        self.assertGreater(float((original - changed_map).abs().max()), 1e-6)
        self.assertEqual(before, module_hash(model))
        self.assertTrue(bool(torch.isfinite(changed_state).all()))
        with self.assertRaises(TypeError):
            model(codes, rgb, state, grid, Source_rgb=rgb)

    def test_camera_relative_tokens_keep_map_y_row_x_column_and_rotation_semantics(self):
        model = ConditionalDecoderRestorer()
        codes, rgb, state, grid = inputs(1)
        captured = []
        handle = model.map_coordinates[0].register_forward_pre_hook(lambda _m, args: captured.append(args[0].detach().clone()))
        try:
            with torch.no_grad():
                model(codes, rgb, state, grid)
                turned = state.clone(); turned[:, 2:4] = torch.tensor([1., 0.])
                model(codes, rgb, turned, grid)
        finally:
            handle.remove()
        position = 3 * 16 + 7
        first, second = captured[0][0, position], captured[1][0, position]
        self.assertTrue(torch.equal(first[:2], torch.tensor([7.5 / 8 - 1, 3.5 / 8 - 1])))
        self.assertTrue(torch.equal(first[2:4], torch.tensor([2 / 16, -2 / 16])))
        self.assertAlmostEqual(float(first[4]), 2 / 16)
        self.assertAlmostEqual(float(first[5]), -2 / 16)
        self.assertAlmostEqual(float(second[4]), -2 / 16)
        self.assertAlmostEqual(float(second[5]), -2 / 16)

    def test_direct_native32_output_can_learn_spatial_detail_and_correction_larger_than_point1(self):
        model = ConditionalDecoderRestorer().eval()
        # The full-resolution output convolution operates after learned
        # subpixel upsampling; no bilinear32->32 smoothing or fixed .1cap.
        with torch.no_grad():
            model.output.bias.fill_(1.)
        codes, rgb, state, grid = inputs(1)
        rgb.fill_(.1)
        aux = model.forward_with_aux(codes, rgb, state, grid)
        self.assertGreater(float(aux['delta'].detach().min()), .7)
        self.assertGreater(float((aux['image'] - rgb).detach().min()), .7)
        self.assertEqual(aux['delta'].shape, (1, 3, 32, 32))
        features = torch.zeros(1, 16, 32, 32)
        features[:, 0, ::2, ::2] = 1
        with torch.no_grad():
            model.output.weight.zero_(); model.output.bias.zero_()
            model.output.weight[:, 0, 1, 1] = 1
        pattern = model.output(features).tanh()
        self.assertGreater(float(pattern[:, :, ::2, ::2].detach().mean()), .7)
        self.assertEqual(float(pattern[:, :, 1::2, 1::2].detach().abs().sum()), 0)

    def test_complete_strict_state_reload_and_metadata_tamper_rejection_are_transactional_and_RNG_preserving(self):
        model = ConditionalDecoderRestorer().eval()
        with torch.no_grad(): model.output.weight.normal_(0, .01)
        values, metadata = deepcopy(model.state_dict()), conditional_decoder_restorer_metadata(model)
        rng = torch.get_rng_state().clone()
        restored = load_conditional_decoder_restorer(model.get_config(), values, metadata=metadata).eval()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(module_hash(model), module_hash(restored))
        data = inputs(1)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(*data), restored(*data)))
        changes = [
            lambda s, m: s.pop(next(iter(s))),
            lambda s, m: s.update(unexpected=torch.zeros(1)),
            lambda s, m: s.update({'output.weight': s['output.weight'].double()}),
            lambda s, m: s.update({'output.weight': s['output.weight'][:1]}),
            lambda s, m: s['output.bias'].fill_(float('nan')),
            lambda s, m: m.update(version=2),
            lambda s, m: m.update(schema='other'),
            lambda s, m: m.update(Source_RGB_inputs=True),
            lambda s, m: m.update(state_sha256='0' * 64),
            lambda s, m: m['config'].update(rgb_amplitude=.1),
        ]
        for change in changes:
            bad_state, bad_meta = deepcopy(values), deepcopy(metadata)
            change(bad_state, bad_meta)
            before, rng = module_hash(model), torch.get_rng_state().clone()
            with self.assertRaises(ValueError):
                validate_conditional_decoder_restorer_state(model, bad_state, metadata=bad_meta)
            self.assertEqual(before, module_hash(model))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_invalid_shapes_nonfinite_RGB_or_materialIDs_rejected_without_input_repairs(self):
        model = ConditionalDecoderRestorer()
        for alter in [
            lambda a: a.__setitem__(0, a[0][:, :15]),
            lambda a: a.__setitem__(1, a[1][:, :, :31]),
            lambda a: a[0].fill_(float('nan')),
            lambda a: a[1].fill_(1.1),
            lambda a: a[2].fill_(float('inf')),
            lambda a: a.__setitem__(2, a[2].long()),
            lambda a: a[3].fill_(16),
            lambda a: a[3].fill_(-1),
            lambda a: a.__setitem__(3, a[3].float()),
            lambda a: a.__setitem__(3, a[3][:, :15]),
        ]:
            data = list(inputs(1)); alter(data)
            with self.assertRaises(ValueError): model(*data)
        for config in ({'depth': 1}, {'width': 63}, {'rgb_amplitude': .1},
                       {'init_seed': True}, {'materials': 17}, {'state_dim': 21}):
            with self.assertRaises(ValueError): ConditionalDecoderRestorerConfig(**config)


if __name__ == '__main__':
    unittest.main()
