"""Expanded Base experts retain trained RGB and use full144D screen queries."""
from copy import deepcopy
from dataclasses import asdict
import json
import unittest
from unittest.mock import patch

import torch

from test_renderer_moe_five import trained_three
from test_renderer_enemy_objectives import batch
from turkeyworld.base_student import BaseStudentConfig, CompactBaseHead
from turkeyworld.base_expanded_experts import (BaseExpandedExpert, KIND, SETTINGS,
    base_expanded_expert_metadata, upgrade_base_experts, validate_base_expanded_expert_metadata)
from turkeyworld.renderer_moe import (FIVE_SCHEMA, migrate_renderer_moe_five,
    output_moe_metadata, renderer_moe_metadata, validate_renderer_moe_bundle)
from turkeyworld.renderer_moe_objectives import moe_conditions
from turkeyworld.renderer_student import _hash_tensors


def head():
    result = CompactBaseHead(BaseStudentConfig(output_moe_enabled=True,
        output_moe_role_schema=FIVE_SCHEMA, output_moe_experts=5))
    with torch.no_grad():
        result.output_moe.output.weight.normal_(0, .05)
        result.output_moe.output.bias.normal_(0, .02)
        result.output_moe.router.weight.normal_(0, .03)
        result.output_moe.split_router.weight.normal_(0, .02)
    return result


class BaseExpandedExpertsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_active_parameter_counts_and_constructor_flag(self):
        with torch.device('meta'):
            source = head()
            self.assertEqual(sum(p.numel() for p in source.parameters()), 1001157)
            metadata = upgrade_base_experts(source)
        self.assertEqual(metadata['parameters_per_expert'], 300058)
        self.assertEqual(metadata['base_parameters'], 2498447)
        self.assertEqual(metadata['added_parameters'], 1497290)
        self.assertEqual(metadata['added_parameter_tensors'], 80)
        self.assertEqual(len(metadata['added_parameter_names']), 80)
        self.assertEqual(len(metadata['legacy_parameter_name_map']), 146)
        self.assertEqual(metadata['query_context_width'], 144)
        json.dumps(metadata, allow_nan=False)
        for change in (dict(private_hidden=382), dict(query_context_width=8),
                       dict(parameters_per_expert=300126), dict(version=True)):
            with self.assertRaises(ValueError):
                validate_base_expanded_expert_metadata(dict(metadata, **change))

    def test_actual_trained_Base_RGB_is_bitexact_and_old_objects_flags_modes_RNG_preserved(self):
        model = trained_three(); migrate_renderer_moe_five(model)
        base = model.base_student
        data = batch((False,)); data['navigation_only'] = torch.zeros(1, dtype=torch.bool)
        conditions = moe_conditions(model, data)
        args = [conditions[key] for key in ('state', 'features', 'geometry')]
        before = {name: value.clone() for name, value in base.state_dict().items()}
        pointers = {name: id(value) for name, value in base.named_parameters()}
        # Keep mixed legacy submodule flags; migration must not recurse train().
        base.output_moe.experts[1].self_attn.eval()
        modes = {id(module): module.training for module in base.modules()}
        flags = {id(value): value.requires_grad for value in base.parameters()}
        rf_hash = _hash_tensors(model.flow_renderer.state_dict())
        rf_metadata = output_moe_metadata(model.flow_renderer)
        with torch.no_grad():
            expected = base(*args)
        rng = torch.get_rng_state().clone()
        metadata = upgrade_base_experts(base)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        current, parameters = base.state_dict(), dict(base.named_parameters())
        for name, value in before.items():
            mapped = metadata['legacy_parameter_name_map'][name]
            self.assertTrue(torch.equal(value, current[mapped]), name)
            self.assertEqual(id(parameters[mapped]), pointers[name])
        for module in base.modules():
            if id(module) in modes:self.assertEqual(module.training, modes[id(module)])
        for value in base.parameters():
            if id(value) in flags:self.assertEqual(value.requires_grad, flags[id(value)])
        with patch('turkeyworld.engine.Game.render', side_effect=AssertionError('No Source forward')):
            with torch.no_grad():actual = base(*args)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(_hash_tensors(model.flow_renderer.state_dict()), rf_hash)
        self.assertEqual(output_moe_metadata(model.flow_renderer), rf_metadata)
        self.assertTrue(base.config.output_moe_expanded_enabled)
        self.assertEqual(upgrade_base_experts(base), base_expanded_expert_metadata(base))

    def test_train_and_eval_zero_activation_preserve_real_nonzero_MoE_function(self):
        original = head(); expanded = deepcopy(original)
        tokens, state, priors = torch.randn(1, 256, 144), torch.zeros(1, 20), torch.rand(1, 256, 3)
        upgrade_base_experts(expanded)
        for training in (False, True):
            original.train(training); expanded.train(training)
            expected = original.output_moe(tokens, state, role_priors=priors)
            actual = expanded.output_moe(tokens, state, role_priors=priors)
            self.assertTrue(torch.equal(expected, actual))
        self.assertFalse(torch.equal(expanded.output_moe.experts[0].input_projection.weight,
                                     expanded.output_moe.experts[1].input_projection.weight))

    def test_zero_head_opens_then_all_five_private_Transformers_and_full_query_inputs_learn(self):
        base = head(); metadata = upgrade_base_experts(base)
        base.train().requires_grad_(True)
        optimizer = torch.optim.AdamW(base.output_moe.parameters(), lr=1e-3)
        tokens, state = torch.randn(1, 256, 144), torch.zeros(1, 20)
        priors, target = torch.rand(1, 256, 3), torch.randn(1, 256, 12)
        new_names = [name.removeprefix('output_moe.') for name in metadata['added_parameter_names']]
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = base.output_moe(tokens, state, role_priors=priors)
            loss = (output-target).square().mean()
            loss.backward()
            for expert in base.output_moe.experts:
                self.assertGreater(float(expert.output_projection.weight.grad.abs().sum()), 0)
                grad = sum(float(value.grad.abs().sum()) for value in expert.private_transformer.parameters())
                self.assertEqual(grad, 0.) if step == 0 else self.assertGreater(grad, 0.)
                if step:
                    self.assertGreater(float(expert.input_projection.weight.grad.abs().sum()), 0)
            for name, value in base.output_moe.named_parameters():
                if name in new_names:
                    self.assertIsNotNone(value.grad, name)
                    self.assertTrue(torch.isfinite(value.grad).all(), name)
            optimizer.step()
        # With the shared 8D projection made constant, the new private branch
        # still responds to all144D queries. It is not downstream of the old bottleneck.
        with torch.no_grad():
            base.output_moe.input.weight.zero_()
            base.output_moe.input.bias.fill_(.1)
        tokens = tokens.detach().requires_grad_()
        changed = base.output_moe(tokens, state, role_priors=priors)
        changed.square().mean().backward()
        self.assertGreater(float(tokens.grad.abs().sum()), 0)
        other = base.output_moe(torch.zeros_like(tokens), state, role_priors=priors)
        self.assertFalse(torch.equal(changed, other))

    def test_config_strict_reload_and_Metadata_binding_preserve_trained_private_RGB(self):
        original = head(); upgrade_base_experts(original)
        with torch.no_grad():
            for expert in original.output_moe.experts:
                expert.output_projection.weight.normal_(0, .02)
        original.eval().requires_grad_(False)
        restored = CompactBaseHead(asdict(original.config)).eval().requires_grad_(False)
        restored.load_state_dict(deepcopy(original.state_dict()), strict=True)
        tokens, state, priors = torch.randn(1, 256, 144), torch.zeros(1, 20), torch.rand(1, 256, 3)
        with torch.no_grad():
            self.assertTrue(torch.equal(original.output_moe(tokens, state, role_priors=priors),
                                        restored.output_moe(tokens, state, role_priors=priors)))
        self.assertEqual(output_moe_metadata(original), output_moe_metadata(restored))
        missing = deepcopy(original.state_dict()); del missing['output_moe.experts.0.output_projection.weight']
        with self.assertRaises(RuntimeError):restored.load_state_dict(missing, strict=True)
        for config in (dict(output_moe_expanded_enabled=True),
                       dict(output_moe_enabled=True, output_moe_expanded_enabled=True),
                       dict(output_moe_expanded_enabled='true')):
            with self.assertRaises(ValueError):BaseStudentConfig(**config)

    def test_expanded_Base_and_original_RF_pair_only_with_explicit_flag_and_strict_architecture(self):
        model = trained_three(); migrate_renderer_moe_five(model)
        upgrade_base_experts(model.base_student)
        payload = dict(config=asdict(model.base_student.config), output_moe=output_moe_metadata(model.base_student))
        header = renderer_moe_metadata(model.flow_renderer, model.base_student)
        self.assertEqual(payload['output_moe']['kind'], KIND)
        self.assertNotEqual(header['base']['kind'], header['flow']['kind'])
        validate_renderer_moe_bundle(model.flow_renderer, payload, header)
        for change in ('missing_flag', 'wrong_width', 'wrong_kind', 'header'):
            altered, binding = deepcopy(payload), deepcopy(header)
            if change == 'missing_flag':altered['config']['output_moe_expanded_enabled'] = False
            if change == 'wrong_width':
                altered['output_moe']['base_expert_expansion']['query_context_width'] = 8
                binding['base'] = deepcopy(altered['output_moe'])
            if change == 'wrong_kind':
                altered['output_moe']['kind'] = header['flow']['kind']
                binding['base'] = deepcopy(altered['output_moe'])
            if change == 'header':binding['base']['state_sha256'] = '0'*64
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_renderer_moe_bundle(model.flow_renderer, altered, binding)

    def test_invalid_or_partial_upgrade_rejects_before_mutation_and_preserves_RNG(self):
        for partial in (False, True):
            base = head()
            if partial:
                base.output_moe.experts[0] = BaseExpandedExpert(base.output_moe.experts[0])
            else:
                base.output_moe.experts[4].dropout.p = .1
            ids = [id(value) for value in base.parameters()]
            rng = torch.get_rng_state().clone()
            with self.assertRaises(ValueError):upgrade_base_experts(base)
            self.assertEqual(ids, [id(value) for value in base.parameters()])
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))


if __name__ == '__main__':unittest.main()
