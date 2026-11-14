"""Real expanded Base and exact six-group Adam continuation without RF work."""
from copy import deepcopy
import json
import random
import unittest

import numpy as np
import torch

from test_renderer_rf5_objectives import model as tiny_renderer, source
from turkeyworld.base_expanded_experts import upgrade_base_experts
from turkeyworld.base_geometry_phase import (configure_base_geometry_training,
    adopt_base_geometry_optimizer, base_geometry_frozen_fingerprint,
    base_geometry_parameter_counts, base_geometry_trainable_names)
from turkeyworld.renderer_moe_objectives import moe_conditions
from turkeyworld.renderer_moe import RendererTokenMoE, FIVE_SCHEMA
from turkeyworld.renderer_native_phase import _same
from turkeyworld.renderer_rf5_experts import upgrade_rf5_experts, rf5_expert_metadata
from turkeyworld.renderer_rf_codec_phase import _expected_groups


def parent_fixture():
    network = tiny_renderer(3)
    # RF is never executed in this phase, but adoption validates the actual
    # production-sized RF expert owner separately from the tiny frozen trunk.
    network.flow_renderer.output_moe = RendererTokenMoE(256, 64, 32, 4, role_schema=FIVE_SCHEMA, experts=5)
    upgrade_rf5_experts(network.flow_renderer.output_moe)
    groups = _expected_groups(network, rf5_expert_metadata(network.flow_renderer.output_moe))
    parameters = dict(network.named_parameters())
    for names in groups:
        for name in names: parameters[name].requires_grad_(True)
    optimizer = torch.optim.AdamW([dict(params=[parameters[name] for name in names],
        lr=1e-4 if i != 2 else 5e-5, weight_decay=.01) for i, names in enumerate(groups)])
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        sum(p.square().mean()+.01*p.mean() for g in optimizer.param_groups for p in g['params']).backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    # Mixed ages are intentional: old Base/private were frozen during RF training.
    for name in groups[2]: optimizer.state[parameters[name]]['step'].fill_(1)
    saved = dict(base=deepcopy(network.base_student.state_dict()), flow=deepcopy(network.flow_renderer.state_dict()),
        optimizer=deepcopy(optimizer.state_dict()), optimizer_parameter_names=groups,
        phase=dict(protocol='renderer_rf_expanded_experts_v1'), step=1108,
        renderer_training_steps=30717, base_training_steps=29209,
        sampler=dict(draws=123), far_wall_aux_sampler=dict(draws=234), geometry_wall_aux_sampler=dict(draws=345))
    with torch.no_grad(): network.flow_renderer.codec.decoder.conv_out.bias.add_(.02)
    upgrade_base_experts(network.base_student)
    configure_base_geometry_training(network)
    return network, saved


class BaseGeometryPhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_old_six_groups_all_mixed_moments_and_rng_exact_fresh80_have_no_state(self):
        network, saved = parent_fixture()
        parent, weights = deepcopy(saved), deepcopy(network.state_dict())
        rng, numpy, python = torch.get_rng_state().clone(), deepcopy(np.random.get_state()), random.getstate()
        optimizer, proof = adopt_base_geometry_optimizer(network, saved)
        result = optimizer.state_dict()
        self.assertEqual(len(result['param_groups']), 7)
        self.assertTrue(_same(result['state'], saved['optimizer']['state']))
        self.assertTrue(_same(result['param_groups'][:6], saved['optimizer']['param_groups']))
        self.assertEqual(len(proof['added_parameter_names']), 80)
        self.assertFalse(set(result['param_groups'][6]['params']) & set(result['state']))
        self.assertEqual(proof['added_parameters'], 1497290)
        self.assertEqual(base_geometry_parameter_counts(network)['base'], 2498447)
        self.assertEqual(len(base_geometry_trainable_names(network)), 226)
        self.assertTrue(_same(parent, saved)); self.assertTrue(_same(weights, network.state_dict()))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(_same(numpy, np.random.get_state())); self.assertEqual(python, random.getstate())
        self.assertTrue(proof['decoder_overlay_changed_tensor_names'])
        json.dumps(proof, allow_nan=False)

    def test_two_actual_Base_updates_advance_old_Base_and_fresh_experts_only(self):
        network, saved = parent_fixture(); optimizer, proof = adopt_base_geometry_optimizer(network, saved)
        fingerprint = base_geometry_frozen_fingerprint(network)
        before = deepcopy(optimizer.state_dict()); parameters = dict(network.named_parameters())
        names = [n for group in proof['optimizer_parameter_names'] for n in group]
        conditions = moe_conditions(network, source())
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            rgb = network.base_student(conditions['state'], conditions['features'], conditions['geometry'])
            (rgb-.2).square().mean().backward(); optimizer.step()
            self.assertEqual(fingerprint, base_geometry_frozen_fingerprint(network))
        after = optimizer.state_dict()
        for index, name in enumerate(names):
            if not parameters[name].requires_grad:
                self.assertTrue(_same(before['state'][index], after['state'][index]), name)
                self.assertIsNone(parameters[name].grad, name)
            elif index in before['state']:
                self.assertEqual(float(after['state'][index]['step']), float(before['state'][index]['step'])+2)
            else: self.assertEqual(float(after['state'][index]['step']), 2)
        for expert in network.base_student.output_moe.experts:
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in expert.private_transformer.parameters()), 0)
        self.assertTrue(_same(saved['optimizer'], before | {'param_groups': before['param_groups'][:6]}))

    def test_parent_tensor_name_moment_and_zero_output_tamper_rejected_transactionally(self):
        template, parent = parent_fixture()
        mutations = dict(
            names=lambda m,s:s['optimizer_parameter_names'][3].reverse(),
            moment=lambda m,s:s['optimizer']['state'][0]['exp_avg'].fill_(float('nan')),
            age=lambda m,s:s['optimizer']['state'][0]['step'].zero_(),
            RF=lambda m,s:next(m.flow_renderer.image_input.parameters()).add_(.01),
            EQ=lambda m,s:next(m.flow_renderer.codec.encoder.parameters()).add_(.01),
            old_Base=lambda m,s:m.base_student.output_moe.router.bias.add_(.01),
            fresh_output=lambda m,s:m.base_student.output_moe.experts[0].output_projection.bias.add_(.01))
        for name, mutate in mutations.items():
            network, saved = deepcopy(template), deepcopy(parent)
            with torch.no_grad(): mutate(network, saved)
            weights = deepcopy(network.state_dict())
            with self.subTest(case=name), self.assertRaises(ValueError): adopt_base_geometry_optimizer(network, saved)
            self.assertTrue(_same(weights, network.state_dict()))


if __name__ == '__main__': unittest.main()
