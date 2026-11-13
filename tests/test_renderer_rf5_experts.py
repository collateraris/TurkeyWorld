import copy
import json
import unittest

import torch

from turkeyworld.renderer_moe import FIVE_SCHEMA, RendererTokenMoE
from turkeyworld.renderer_rf5_experts import (
    RF5PrivateExpert, SETTINGS, rf5_expert_metadata,
    upgrade_rf5_experts, validate_rf5_expert_metadata,
)


class RendererRF5ExpertsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self):
        return RendererTokenMoE(256, 64, 32, 4, role_schema=FIVE_SCHEMA, experts=5)

    def test_exact_count_meta_loader_and_strict_metadata(self):
        with torch.device('meta'):
            model = self.model()
            metadata = upgrade_rf5_experts(model)
        self.assertEqual(metadata['parameters_per_expert'], 503072)
        self.assertEqual(metadata['expert_parameters'], 2515360)
        self.assertEqual(metadata['added_parameters'], 2472640)
        self.assertEqual(metadata['moe_parameters'], 2525861)
        self.assertEqual(metadata['added_parameter_tensors'], 80)
        self.assertEqual(len(metadata['legacy_parameter_name_map']), 60)
        json.dumps(metadata, allow_nan=False)
        for key, value in [('private_hidden', 410), ('experts', 3), ('Base_changed', True),
                           ('version', True)]:
            bad = dict(metadata, **{key: value})
            with self.assertRaises(ValueError):
                validate_rf5_expert_metadata(bad)

    def test_initial_function_and_old_parameters_and_rng_are_exact(self):
        model = self.model()
        with torch.no_grad():
            model.output.weight.normal_()  # Exercise nonzero parent MoE residuals.
            model.output.bias.normal_()
        tokens = torch.randn(2, 16, 256)
        state = torch.zeros(2, 20)
        prior = torch.rand(2, 16, 3)
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        identities = {k: id(v) for k, v in model.named_parameters()}
        model.eval()
        with torch.no_grad():
            expected = model(tokens, state, role_priors=prior)
        old_training = copy.deepcopy(model).train()
        expected_training = old_training(tokens, state, role_priors=prior)
        rng = torch.get_rng_state().clone()
        metadata = upgrade_rf5_experts(model)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(torch.equal(model.experts[0].private_input.weight,
                                     model.experts[1].private_input.weight))
        self.assertFalse(model.training)
        names = metadata['legacy_parameter_name_map']
        parameters = dict(model.named_parameters())
        current = model.state_dict()
        for name, value in before.items():
            mapped = names.get(name, name)
            self.assertTrue(torch.equal(value, current[mapped]), name)
            if name in identities:
                self.assertEqual(id(parameters[mapped]), identities[name])
        with torch.no_grad():
            actual = model(tokens, state, role_priors=prior)
        self.assertTrue(torch.equal(expected, actual))
        model.train()
        self.assertTrue(all(expert.private_transformer.training for expert in model.experts))
        actual = model(tokens, state, role_priors=prior)
        self.assertTrue(torch.equal(expected_training, actual))
        ids = [id(p) for p in model.parameters()]
        self.assertEqual(upgrade_rf5_experts(model), rf5_expert_metadata(model))
        self.assertEqual(ids, [id(p) for p in model.parameters()])

    def test_private_branch_opens_then_receives_finite_transformer_gradients(self):
        model = self.model()
        with torch.no_grad():
            model.output.weight.normal_(std=.1)
        upgrade_rf5_experts(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        tokens, state = torch.randn(2, 16, 256), torch.zeros(2, 20)
        prior, target = torch.rand(2, 16, 3), torch.randn(2, 16, 64)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(tokens, state, role_priors=prior) - target).square().mean()
            loss.backward()
            for expert in model.experts:
                self.assertGreater(expert.private_output.weight.grad.norm().item(), 0)
                private_grad = sum(p.grad.square().sum().item() for p in
                    expert.private_transformer.parameters() if p.grad is not None)
                self.assertEqual(private_grad, 0) if step == 0 else self.assertGreater(private_grad, 0)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                                if p.grad is not None))
            optimizer.step()

    def test_strict_state_roundtrip_preserves_nonzero_expanded_function(self):
        model = self.model().double()
        upgrade_rf5_experts(model)
        with torch.no_grad():
            for expert in model.experts:
                expert.private_output.weight.normal_(std=.02)
            model.output.weight.normal_(std=.02)
        tokens, state = torch.randn(1, 16, 256, dtype=torch.float64), torch.zeros(1, 20)
        prior = torch.rand(1, 16, 3)
        expected = model(tokens, state, role_priors=prior)
        restored = self.model().double()
        upgrade_rf5_experts(restored)
        restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        self.assertTrue(torch.equal(expected, restored(tokens, state, role_priors=prior)))
        self.assertTrue(all(p.dtype == torch.float64 for p in restored.parameters()))
        missing = copy.deepcopy(model.state_dict())
        del missing['experts.0.private_output.weight']
        with self.assertRaises(RuntimeError):
            restored.load_state_dict(missing, strict=True)

    def test_masked_layer_contract_and_rejection_without_partial_mutation(self):
        legacy = torch.nn.TransformerEncoderLayer(32, 4, 64, dropout=0.,
            activation='gelu', batch_first=True, norm_first=True)
        source = torch.randn(2, 4, 32)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(4)
        expected = legacy(source, src_mask=mask, is_causal=True)
        expanded = RF5PrivateExpert(legacy)
        self.assertTrue(torch.equal(expected, expanded(source, src_mask=mask, is_causal=True)))
        base = RendererTokenMoE(144, 12, 8, 16, role_schema=FIVE_SCHEMA, experts=5)
        original = list(base.experts)
        with self.assertRaises(ValueError):
            upgrade_rf5_experts(base)
        self.assertEqual(list(base.experts), original)
        model = self.model()
        model.experts[4].dropout.p = .1
        old = list(model.experts)
        with self.assertRaises(ValueError):
            upgrade_rf5_experts(model)
        self.assertEqual(list(model.experts), old)

    def test_preserves_mixed_legacy_modes_and_freeze_flags_and_rejects_partial_upgrade(self):
        model = self.model().train()
        model.experts[1].self_attn.eval()
        model.experts[2].norm1.weight.requires_grad_(False)
        modes = {id(module): module.training for module in model.modules()}
        flags = {id(p): p.requires_grad for p in model.parameters()}
        rng = torch.get_rng_state().clone()
        upgrade_rf5_experts(model)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for module in model.modules():
            if id(module) in modes:
                self.assertEqual(module.training, modes[id(module)])
        for parameter in model.parameters():
            if id(parameter) in flags:
                self.assertEqual(parameter.requires_grad, flags[id(parameter)])
        partial = self.model()
        partial.experts[0] = RF5PrivateExpert(partial.experts[0])
        identities = [id(p) for p in partial.parameters()]
        with self.assertRaises(ValueError):
            upgrade_rf5_experts(partial)
        self.assertEqual(identities, [id(p) for p in partial.parameters()])


if __name__ == '__main__':
    unittest.main()
