"""Learned247k parent -> real~2M restorer, without losing its function/state."""
from copy import deepcopy
from dataclasses import asdict
import json
import unittest
from unittest.mock import patch

import torch

from test_conditional_decoder_restorer import inputs
from turkeyworld.conditional_decoder_restorer import (
    ConditionalDecoderRestorer, ConditionalDecoderRestorerConfig,
    build_conditional_decoder_restorer, conditional_decoder_restorer_metadata,
    validate_conditional_decoder_restorer_state, load_conditional_decoder_restorer)
from turkeyworld.conditional_restorer_expansion import (
    KIND, ExpandedConditionalDecoderRestorer, ExpandedConditionalDecoderRestorerConfig,
    expand_conditional_restorer)


def learned_parent():
    result = ConditionalDecoderRestorer()
    with torch.no_grad():
        result.output.weight.normal_(0, .002)
        result.output.bias.copy_(torch.tensor([.01, -.02, .015]))
    return result


def same(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(same(v, right[k]) for k, v in left.items())
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


class ConditionalRestorerExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_learned_parent_function_and_parameter_objects_are_exact_in_train_eval_at_zero_activation(self):
        data = inputs()
        for training in (True, False):
            parent = learned_parent().train(training).requires_grad_(False)
            with torch.no_grad(): old_image = parent(*data)
            self.assertFalse(torch.equal(old_image, data[1]))  # Real learned correction, not a trivial zero parent.
            old = {name: value.clone() for name, value in parent.state_dict().items()}
            pointers = {name: p.data_ptr() for name, p in parent.named_parameters()}
            flags = {name: p.requires_grad for name, p in parent.named_parameters()}
            modes = {name: m.training for name, m in parent.named_modules()}
            rng = torch.get_rng_state().clone()
            model, report = expand_conditional_restorer(parent)
            self.assertIs(model.legacy, parent)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(flags, {n: p.requires_grad for n, p in parent.named_parameters()})
            self.assertEqual(modes, {n: m.training for n, m in parent.named_modules()})
            self.assertEqual(pointers, {n: p.data_ptr() for n, p in parent.named_parameters()})
            with torch.no_grad(), patch('turkeyworld.engine.Game.render', side_effect=AssertionError('No Source renderer')):
                aux = model.forward_with_aux(*data)
            self.assertTrue(torch.equal(old_image, aux['image']))
            self.assertTrue(torch.equal(old_image, aux['legacy_image']))
            self.assertEqual(float(aux['expansion_delta'].abs().sum()), 0)
            self.assertTrue(all(torch.equal(value, model.state_dict()['legacy.' + name]) for name, value in old.items()))
            self.assertEqual(report['combined_parameters'], 1997798)
            self.assertEqual(report['added_parameters'], 1750259)
            self.assertEqual(report['combined_parameter_tensors'], 130)
            self.assertEqual(report['added_parameter_tensors'], 65)
            self.assertEqual(report['old_parameter_name_map'], {name: 'legacy.' + name for name in pointers})
            self.assertTrue(all(name.startswith('expansion.') for name in report['added_parameter_names']))
            json.dumps(report, allow_nan=False)

    def test_mixed_legacy_modes_flags_and_half_dtype_are_preserved_without_an_extra_random_draw(self):
        parent = learned_parent().eval().half()
        parent.actor_state.train()
        parent.output.weight.requires_grad_(False)
        flags = {n: p.requires_grad for n, p in parent.named_parameters()}
        modes = {n: m.training for n, m in parent.named_modules()}
        rng = torch.get_rng_state().clone()
        model, _ = expand_conditional_restorer(parent)
        self.assertEqual(flags, {n: p.requires_grad for n, p in parent.named_parameters()})
        self.assertEqual(modes, {n: m.training for n, m in parent.named_modules()})
        self.assertTrue(all(p.dtype == torch.float16 for p in model.expansion.parameters()))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        codes, rgb, state, grid = inputs(1)
        with torch.no_grad():
            expected = parent(codes.half(), rgb.half(), state, grid)
            actual = model(codes.half(), rgb.half(), state, grid)
        self.assertTrue(torch.equal(expected, actual))

    def test_existing_65_Adam_states_options_and_ages_survive_while_new_65_are_fresh(self):
        parent = learned_parent()
        data = inputs(1)
        target = data[1] * .5 + .15
        optimizer = torch.optim.AdamW(parent.parameters(), lr=.001, weight_decay=.01)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            (parent(*data) - target).square().mean().backward()
            optimizer.step()
        old_optimizer = deepcopy(optimizer.state_dict())
        self.assertEqual(len(old_optimizer['state']), 65)
        named = dict(parent.named_parameters())
        model, receipt = expand_conditional_restorer(parent)
        actual = dict(model.named_parameters())
        for old, new in receipt['old_parameter_name_map'].items():
            self.assertIs(named[old], actual[new])
        # The pure module migration preserves the actual old Parameter objects;
        # a phase can retain the old Adam verbatim and append fresh capacity.
        optimizer.add_param_group(dict(params=model.expansion.parameters(), lr=.001))
        adopted = optimizer.state_dict()
        self.assertTrue(same(old_optimizer['state'], adopted['state']))
        self.assertTrue(same(old_optimizer['param_groups'], adopted['param_groups'][:1]))
        self.assertFalse(set(adopted['param_groups'][1]['params']) & set(adopted['state']))
        self.assertEqual(len(adopted['param_groups'][1]['params']), 65)
        self.assertEqual(receipt['old_state_name_map'], receipt['old_model_tensor_name_map'])

    def test_every_added_parameter_receives_gradient_after_the_zero_output_opens(self):
        model, _ = expand_conditional_restorer(learned_parent())
        model.train().requires_grad_(True)
        data = inputs(1)
        target = data[1].flip(-1) * .5 + .15
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0)
        first = (model(*data) - target).square().mean()
        first.backward()
        self.assertGreater(float(model.expansion.output.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.expansion.output.bias.grad.abs().sum()), 0)
        for name, parameter in model.expansion.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            if not name.startswith('output.'):
                self.assertEqual(float(parameter.grad.abs().sum()), 0, name)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        rng = torch.get_rng_state().clone()
        second = (model(*data) - target).square().mean()
        second.backward()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, parameter in model.expansion.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0, name)
        for branch in (model.legacy, model.expansion):
            for block in branch.blocks:
                self.assertGreater(float(block.cross_attention.in_proj_weight.grad.abs().sum()), 0)

    def test_expanded_factory_metadata_and_strict_reload_preserve_trained_output_and_old_v1_loading(self):
        parent = learned_parent()
        old_config, old_state = parent.get_config(), deepcopy(parent.state_dict())
        old_metadata = conditional_decoder_restorer_metadata(parent)
        model, _ = expand_conditional_restorer(parent)
        with torch.no_grad(): model.expansion.output.weight.normal_(0, .001)
        config, state = model.get_config(), deepcopy(model.state_dict())
        metadata = conditional_decoder_restorer_metadata(model)
        self.assertEqual(metadata['kind'], KIND)
        self.assertEqual(metadata['parameters'], 1997798)
        self.assertEqual(metadata['added_parameters'], 1750259)
        self.assertFalse(metadata['dead_padding_parameters'])
        self.assertEqual(config['expansion_config']['hidden'], 288)
        self.assertEqual(config['expansion_config']['width'], 224)
        rng = torch.get_rng_state().clone()
        loaded = load_conditional_decoder_restorer(config, state, metadata=metadata).eval().requires_grad_(False)
        self.assertIsInstance(loaded, ExpandedConditionalDecoderRestorer)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        model.eval().requires_grad_(False)
        with torch.no_grad(): self.assertTrue(torch.equal(model(*inputs(1)), loaded(*inputs(1))))
        old_loaded = load_conditional_decoder_restorer(old_config, old_state, metadata=old_metadata)
        self.assertIs(type(old_loaded), ConditionalDecoderRestorer)
        self.assertEqual(conditional_decoder_restorer_metadata(old_loaded), old_metadata)
        self.assertIs(type(build_conditional_decoder_restorer(old_config)), ConditionalDecoderRestorer)
        self.assertIsInstance(build_conditional_decoder_restorer(config), ExpandedConditionalDecoderRestorer)

    def test_complete_architecture_state_and_input_pins_reject_tampering_transactionally(self):
        model, _ = expand_conditional_restorer(learned_parent())
        state, metadata = deepcopy(model.state_dict()), conditional_decoder_restorer_metadata(model)
        for change in (
            lambda s, m: s.pop('legacy.output.bias'),
            lambda s, m: s.pop('expansion.output.bias'),
            lambda s, m: s.update(unexpected=torch.zeros(1)),
            lambda s, m: s['expansion.output.bias'].fill_(float('nan')),
            lambda s, m: s['legacy.output.bias'].add_(.1),
            lambda s, m: m.update(kind='conditional_decoder_restorer_v1'),
            lambda s, m: m.update(parameters=2000000),
            lambda s, m: m.update(Source_RGB_inputs=True),
            lambda s, m: m['config']['expansion_config'].update(hidden=320),
        ):
            bad_state, bad_metadata = deepcopy(state), deepcopy(metadata)
            change(bad_state, bad_metadata)
            before, rng = conditional_decoder_restorer_metadata(model), torch.get_rng_state().clone()
            with self.assertRaises(ValueError):
                validate_conditional_decoder_restorer_state(model, bad_state, metadata=bad_metadata)
            self.assertEqual(before, conditional_decoder_restorer_metadata(model))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        with self.assertRaises(ValueError):
            ExpandedConditionalDecoderRestorerConfig(expansion_config=asdict(ConditionalDecoderRestorerConfig(width=224, heads=8, hidden=320)))
        with self.assertRaises(ValueError): expand_conditional_restorer(model)
        with self.assertRaises(ValueError): build_conditional_decoder_restorer(dict(architecture='unknown'))


if __name__ == '__main__':
    unittest.main()
