"""Learned2M -> real7M continuation; no reset of either old Adam age group."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import torch

from test_conditional_decoder_restorer import inputs
from test_conditional_restorer_expansion import learned_parent, same
from turkeyworld.conditional_decoder_restorer import (
    ConditionalDecoderRestorer, build_conditional_decoder_restorer,
    conditional_decoder_restorer_metadata, validate_conditional_decoder_restorer_state,
    load_conditional_decoder_restorer)
from turkeyworld.conditional_restorer_expansion import ExpandedConditionalDecoderRestorer, expand_conditional_restorer
from turkeyworld.conditional_restorer_7m import (
    KIND, LargeConditionalDecoderRestorer, LargeConditionalDecoderRestorerConfig,
    expand_conditional_restorer_7m)


def learned_2m():
    result, _ = expand_conditional_restorer(learned_parent())
    with torch.no_grad():
        result.expansion.output.weight.normal_(0, .0005)
        result.expansion.output.bias.copy_(torch.tensor([-.015, .01, .02]))
    return result


class ConditionalRestorer7MTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_trained_2m_function_all130_objects_flags_and_RNG_are_exact_at7m_activation(self):
        data = inputs(1)
        for training in (True, False):
            previous = learned_2m().train(training).requires_grad_(False)
            with torch.no_grad(): expected = previous(*data)
            self.assertFalse(torch.equal(expected, previous.legacy(*data)))
            state = {n: v.clone() for n, v in previous.state_dict().items()}
            pointers = {n: p.data_ptr() for n, p in previous.named_parameters()}
            flags = {n: p.requires_grad for n, p in previous.named_parameters()}
            modes = {n: m.training for n, m in previous.named_modules()}
            rng = torch.get_rng_state().clone()
            model, report = expand_conditional_restorer_7m(previous)
            self.assertIs(model.previous, previous)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(pointers, {n: p.data_ptr() for n, p in previous.named_parameters()})
            self.assertEqual(flags, {n: p.requires_grad for n, p in previous.named_parameters()})
            self.assertEqual(modes, {n: m.training for n, m in previous.named_modules()})
            with torch.no_grad(), patch('turkeyworld.engine.Game.render', side_effect=AssertionError('No Source')):
                aux = model.forward_with_aux(*data)
            self.assertTrue(torch.equal(expected, aux['image']))
            self.assertTrue(torch.equal(expected, aux['previous_image']))
            self.assertEqual(float(aux['expansion_delta'].abs().sum()), 0)
            self.assertTrue(all(torch.equal(v, model.state_dict()['previous.' + n]) for n, v in state.items()))
            self.assertEqual((report['combined_parameters'], report['combined_parameter_tensors']), (7000313, 195))
            self.assertEqual((report['added_parameters'], report['added_parameter_tensors']), (5002515, 65))
            self.assertEqual(report['old_parameter_name_map'], {n: 'previous.' + n for n in pointers})
            self.assertEqual(len(report['old_model_tensor_name_map']), 130)
            self.assertEqual(len(report['added_parameter_names']), 65)
            json.dumps(report, allow_nan=False)

    def test_two_old_Adam_groups_keep_all130_moments_options_and_mixed_ages_new65_start_fresh(self):
        first = learned_parent()
        data = inputs(1); target = data[1] * .5 + .15
        optimizer = torch.optim.AdamW(first.parameters(), lr=.001, weight_decay=.01)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            (first(*data) - target).square().mean().backward(); optimizer.step()
        previous, _ = expand_conditional_restorer(first)
        optimizer.add_param_group(dict(params=previous.expansion.parameters(), lr=.0007))
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            (previous(*data) - target).square().mean().backward(); optimizer.step()
        old = deepcopy(optimizer.state_dict())
        self.assertEqual(len(old['state']), 130)
        ages = [{float(old['state'][i]['step']) for i in group['params']} for group in old['param_groups']]
        self.assertEqual(ages, [{4.}, {2.}])
        named = dict(previous.named_parameters())
        model, receipt = expand_conditional_restorer_7m(previous)
        current = dict(model.named_parameters())
        self.assertTrue(all(named[n] is current[mapped] for n, mapped in receipt['old_parameter_name_map'].items()))
        optimizer.add_param_group(dict(params=model.expansion.parameters(), lr=.0001))
        adopted = optimizer.state_dict()
        self.assertTrue(same(old['state'], adopted['state']))
        self.assertTrue(same(old['param_groups'], adopted['param_groups'][:2]))
        self.assertEqual(len(adopted['param_groups'][2]['params']), 65)
        self.assertFalse(set(adopted['param_groups'][2]['params']) & set(adopted['state']))

    def test_every_added_parameter_is_real_and_gets_finite_gradient_after_zero_head_opens(self):
        model, _ = expand_conditional_restorer_7m(learned_2m())
        model.train().requires_grad_(True)
        data = inputs(1); target = data[1].flip(-1) * .5 + .15
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0)
        (model(*data) - target).square().mean().backward()
        self.assertGreater(float(model.expansion.output.weight.grad.abs().sum()), 0)
        for name, p in model.expansion.named_parameters():
            self.assertIsNotNone(p.grad, name)
            if not name.startswith('output.'):
                self.assertEqual(float(p.grad.abs().sum()), 0, name)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        rng = torch.get_rng_state().clone()
        (model(*data) - target).square().mean().backward()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, p in model.expansion.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(bool(torch.isfinite(p.grad).all()), name)
            self.assertGreater(float(p.grad.abs().sum()), 0, name)
        self.assertGreater(float(model.previous.expansion.output.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.previous.legacy.output.weight.grad.abs().sum()), 0)

    def test_trained7m_strict_factory_reload_and247k_2m_compatibility_remain_exact(self):
        previous = learned_2m()
        small = previous.legacy
        small_state, small_meta = deepcopy(small.state_dict()), conditional_decoder_restorer_metadata(small)
        middle_state, middle_meta = deepcopy(previous.state_dict()), conditional_decoder_restorer_metadata(previous)
        model, _ = expand_conditional_restorer_7m(previous)
        with torch.no_grad(): model.expansion.output.weight.normal_(0, .0002)
        metadata, state = conditional_decoder_restorer_metadata(model), deepcopy(model.state_dict())
        self.assertEqual(metadata['kind'], KIND)
        self.assertEqual(metadata['capacity_variant'], 'expanded_7m_v1')
        self.assertEqual((metadata['parameters'], metadata['parameter_tensors']), (7000313, 195))
        self.assertFalse(metadata['dead_padding_parameters'])
        self.assertEqual(model.get_config()['expansion_config']['hidden'], 720)
        rng = torch.get_rng_state().clone()
        loaded = load_conditional_decoder_restorer(model.get_config(), state, metadata=metadata).eval().requires_grad_(False)
        self.assertIsInstance(loaded, LargeConditionalDecoderRestorer)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        model.eval().requires_grad_(False)
        with torch.no_grad(): self.assertTrue(torch.equal(model(*inputs(1)), loaded(*inputs(1))))
        old = load_conditional_decoder_restorer(small.get_config(), small_state, metadata=small_meta)
        middle = load_conditional_decoder_restorer(previous.get_config(), middle_state, metadata=middle_meta)
        self.assertIs(type(old), ConditionalDecoderRestorer)
        self.assertIs(type(middle), ExpandedConditionalDecoderRestorer)
        self.assertEqual(conditional_decoder_restorer_metadata(old), small_meta)
        self.assertEqual(conditional_decoder_restorer_metadata(middle), middle_meta)
        self.assertIsInstance(build_conditional_decoder_restorer(model.get_config()), LargeConditionalDecoderRestorer)

    def test_mixed_half_parent_modes_and_new_state_map_sensitivity_have_no_Source_or_RNG_calls(self):
        previous = learned_2m().eval().half()
        previous.legacy.actor_state.train()
        previous.expansion.output.weight.requires_grad_(False)
        flags = {n: p.requires_grad for n, p in previous.named_parameters()}
        modes = {n: m.training for n, m in previous.named_modules()}
        rng = torch.get_rng_state().clone()
        model, _ = expand_conditional_restorer_7m(previous)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(flags, {n: p.requires_grad for n, p in previous.named_parameters()})
        self.assertEqual(modes, {n: m.training for n, m in previous.named_modules()})
        codes, rgb, state, grid = inputs(1)
        with torch.no_grad():
            expected = previous(codes.half(), rgb.half(), state, grid)
            actual = model(codes.half(), rgb.half(), state, grid)
        self.assertTrue(torch.equal(expected, actual))
        model.float().eval()
        with torch.no_grad(): model.expansion.output.weight.fill_(.0003)
        alternate = state.clone(); alternate[:, 8:10] -= .2; alternate[:, :2] += .1
        with torch.no_grad(), patch('turkeyworld.engine.Game.step', side_effect=AssertionError('No Source feedback')):
            a = model.forward_with_aux(codes, rgb, state, grid)
            b = model.forward_with_aux(codes, rgb, alternate, grid)
            c = model.forward_with_aux(codes, rgb, state, (grid + 5) % 16)
        self.assertGreater(float((a['expansion_delta'] - b['expansion_delta']).abs().max()), 1e-7)
        self.assertGreater(float((a['expansion_delta'] - c['expansion_delta']).abs().max()), 1e-7)

    def test_branch_config_binding_and_all195_tensors_reject_tampering_without_mutating_live_state(self):
        model, _ = expand_conditional_restorer_7m(learned_2m())
        state, metadata = deepcopy(model.state_dict()), conditional_decoder_restorer_metadata(model)
        for mutate in (
            lambda s, m: s.pop('previous.legacy.output.bias'),
            lambda s, m: s.pop('previous.expansion.output.bias'),
            lambda s, m: s.pop('expansion.output.bias'),
            lambda s, m: s['expansion.output.bias'].fill_(float('nan')),
            lambda s, m: m.update(kind='expanded_conditional_decoder_restorer_v1'),
            lambda s, m: m.update(parameters=7000000),
            lambda s, m: m.update(Source_RGB_inputs=True),
            lambda s, m: m['config']['expansion_config'].update(hidden=800),
        ):
            wrong, pin = deepcopy(state), deepcopy(metadata); mutate(wrong, pin)
            before, rng = conditional_decoder_restorer_metadata(model), torch.get_rng_state().clone()
            with self.assertRaises(ValueError):
                validate_conditional_decoder_restorer_state(model, wrong, metadata=pin)
            self.assertEqual(before, conditional_decoder_restorer_metadata(model))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        with self.assertRaises(ValueError): expand_conditional_restorer_7m(model)
        with self.assertRaises(ValueError): expand_conditional_restorer_7m(model.previous.legacy)
        config = model.get_config(); config['expansion_config']['width'] = 376
        with self.assertRaises(ValueError): LargeConditionalDecoderRestorerConfig(**config)


if __name__ == '__main__': unittest.main()
