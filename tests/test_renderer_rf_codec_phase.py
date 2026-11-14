"""Exact six-group RF continuation, frozen D overlay and immutable moments."""
from copy import deepcopy
import json
import random
import unittest

import numpy as np
import torch
from torch import nn

from test_renderer_rf5_phase import fixture
from turkeyworld.renderer_native_phase import _same
from turkeyworld.renderer_rf5_phase import build_rf5_optimizer, configure_rf5_training
from turkeyworld.renderer_rf_codec_phase import (
    PROTOCOL, configure_rf_codec_training, adopt_rf_codec_optimizer,
    rf_codec_trainable_names, rf_codec_frozen_fingerprint, rf_codec_parameter_counts,
    rf_codec_protocol)


class Codec(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 3)
        self.quant_conv = nn.Linear(3, 3)
        self.quantize = nn.Embedding(8, 3)
        self.decoder = nn.Linear(3, 3)
        self.post_quant_conv = nn.Linear(3, 3)
        self.register_buffer('usage_updates', torch.tensor(7))


def full_parent():
    model, rf1 = fixture()
    model.flow_renderer.codec = Codec()
    rf1['flow'].pop('codec.weight'); rf1['flow'].pop('codec.bias')
    rf1['flow'].update({'codec.'+name: value.clone() for name, value in model.flow_renderer.codec.state_dict().items()})
    configure_rf5_training(model)
    optimizer, adoption = build_rf5_optimizer(model, rf1)
    # Initialize the real old six groups, including historical frozen groups.
    for group in optimizer.param_groups:
        for parameter in group['params']: parameter.requires_grad_(True)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        sum(parameter.square().mean()+.01*parameter.mean()
            for group in optimizer.param_groups for parameter in group['params']).backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.flow_renderer.steps = 3
    model.flow_renderer.register_buffer('latent_mean', torch.zeros(1, 3, 1, 1))
    model.flow_renderer.register_buffer('latent_std', torch.ones(1, 3, 1, 1))
    saved = dict(base=deepcopy(model.base_student.state_dict()), flow=deepcopy(model.flow_renderer.state_dict()),
        optimizer=deepcopy(optimizer.state_dict()), optimizer_parameter_names=adoption['optimizer_parameter_names'],
        step=1108, renderer_training_steps=30717, base_training_steps=29209,
        phase=dict(protocol='renderer_rf_expanded_experts_v1'),
        sampler=dict(draws=123), far_wall_aux_sampler=dict(draws=234), geometry_wall_aux_sampler=dict(draws=345),
        rng_torch=torch.get_rng_state().clone(), rng_numpy=deepcopy(np.random.get_state()), rng_python=random.getstate())
    # Frozen current D may change, whereas E/Q/RF/Base and all buffers may not.
    with torch.no_grad():
        model.flow_renderer.codec.decoder.bias.add_(.02)
        model.flow_renderer.codec.post_quant_conv.weight.mul_(.99)
    configure_rf_codec_training(model)
    return model, saved


class RendererRFCodecPhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_six_old_groups_mixed_ages_options_all_exact_no_new_group_or_state(self):
        model, saved = full_parent()
        parent, weights = deepcopy(saved), deepcopy(model.state_dict())
        rng, numpy, python = torch.get_rng_state().clone(), deepcopy(np.random.get_state()), random.getstate()
        optimizer, proof = adopt_rf_codec_optimizer(model, saved)
        self.assertTrue(_same(optimizer.state_dict(), saved['optimizer']))
        self.assertEqual(len(optimizer.param_groups), 6)
        self.assertEqual(proof['old_Adam_states'], len(saved['optimizer']['state']))
        self.assertEqual(proof['fresh_Adam_states'], 0)
        self.assertEqual(proof['fresh_optimizer_groups'], 0)
        self.assertEqual(proof['added_parameter_names'], [])
        self.assertEqual(proof['decoder_overlay_changed_tensor_count'], 2)
        self.assertTrue(all(name.startswith(('flow_renderer.codec.decoder.',
            'flow_renderer.codec.post_quant_conv.')) for name in proof['decoder_overlay_changed_tensor_names']))
        self.assertTrue(_same(saved, parent)); self.assertTrue(_same(model.state_dict(), weights))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(_same(numpy, np.random.get_state())); self.assertEqual(python, random.getstate())
        self.assertEqual(proof['parent_renderer_training_steps'], 30717)
        self.assertEqual(proof['frozen_base_training_steps'], 29209)
        json.dumps(proof, allow_nan=False)

    def test_actual_RF_expert_SGD_keeps_Base_world_D_EQ_opacity_buffers_and_frozen_Adam_exact(self):
        model, saved = full_parent()
        optimizer, proof = adopt_rf_codec_optimizer(model, saved)
        before = deepcopy(model.state_dict()); moments = deepcopy(optimizer.state_dict())
        fingerprint = rf_codec_frozen_fingerprint(model)
        names = [name for group in proof['optimizer_parameter_names'] for name in group]
        parameters = dict(model.named_parameters())
        frozen_ids = [i for i, name in enumerate(names) if not parameters[name].requires_grad]
        tokens = torch.linspace(-1, 1, 2*16*256).reshape(2, 16, 256)
        state = torch.zeros(2, 20); state[:, 0:5] = torch.tensor([.25, .25, 0., 1., 1.])
        state[:, 8:10] = torch.tensor([.4, .25]); state[:, 12:14] = 1
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = model.flow_renderer.output_moe(tokens, state)
            ((output-.3).square().mean()+model.flow_renderer.body(output[..., :3]).square().mean()).backward()
            optimizer.step()
            self.assertEqual(rf_codec_frozen_fingerprint(model), fingerprint)
        after = optimizer.state_dict()
        for index in frozen_ids:
            self.assertTrue(_same(moments['state'][index], after['state'][index]))
        for index, name in enumerate(names):
            if parameters[name].requires_grad and parameters[name].grad is not None:
                self.assertEqual(float(after['state'][index]['step']), float(moments['state'][index]['step'])+2)
        for name, parameter in model.named_parameters():
            if name not in rf_codec_trainable_names(model): self.assertIsNone(parameter.grad, name)
        self.assertTrue(any(not torch.equal(before[name], value) for name, value in model.state_dict().items()
                            if name.startswith('flow_renderer.output_moe.')))
        self.assertTrue(_same(saved['optimizer'], moments), 'Parent optimizer RAM must not alias current moments')

    def test_only_D_post_overlay_allowed_all_other_parent_edits_or_state_tamper_rejected(self):
        template, source = full_parent()
        mutations = {
            'encoder': lambda m, s: m.flow_renderer.codec.encoder.weight.add_(.01),
            'codebook': lambda m, s: m.flow_renderer.codec.quantize.weight.add_(.01),
            'Base': lambda m, s: m.base_student.body.weight.add_(.01),
            'RF': lambda m, s: m.flow_renderer.body.weight.add_(.01),
            'opacity': lambda m, s: m.flow_renderer.geometry_opacity[0].weight.add_(.01),
            'noise': lambda m, s: m.flow_renderer.fixed_noise.add_(.01),
            'normalization': lambda m, s: m.flow_renderer.latent_std.add_(.01),
            'codec_buffer': lambda m, s: m.flow_renderer.codec.usage_updates.add_(1),
            'names': lambda m, s: s['optimizer_parameter_names'][3].reverse(),
            'ids': lambda m, s: s['optimizer']['param_groups'][5]['params'].reverse(),
            'missing_moment': lambda m, s: s['optimizer']['state'].pop(0),
            'age_reset': lambda m, s: s['optimizer']['state'][0]['step'].zero_(),
            'nan_moment': lambda m, s: s['optimizer']['state'][0]['exp_avg'].fill_(float('nan')),
            'bad_shape': lambda m, s: s['optimizer']['state'][0].__setitem__('exp_avg_sq', torch.zeros(1)),
        }
        for name, mutate in mutations.items():
            model, saved = deepcopy(template), deepcopy(source)
            with torch.no_grad(): mutate(model, saved)
            before = deepcopy(model.state_dict())
            with self.subTest(case=name), self.assertRaises(ValueError):
                adopt_rf_codec_optimizer(model, saved)
            self.assertTrue(_same(before, model.state_dict()))

    def test_configure_rejects_restorer_and_frozen_fingerprint_covers_whole_nonRF_state(self):
        model, _ = full_parent(); model.train()
        policy = configure_rf_codec_training(model)
        self.assertEqual(policy['trainable_parameter_names'], rf_codec_trainable_names(model))
        self.assertEqual(set(rf_codec_trainable_names(model)),
                         {name for name, p in model.named_parameters() if p.requires_grad})
        self.assertFalse(model.base_student.training); self.assertTrue(model.flow_renderer.training)
        self.assertFalse(model.flow_renderer.codec.training)
        self.assertFalse(model.flow_renderer.geometry_opacity.training)
        before = rf_codec_frozen_fingerprint(model)
        with torch.no_grad(): model.flow_renderer.body.bias.add_(.01)
        self.assertEqual(before, rf_codec_frozen_fingerprint(model))
        for name in ('codec', 'map', 'world'):
            other = deepcopy(model)
            with torch.no_grad():
                if name == 'codec': other.flow_renderer.codec.decoder.bias.add_(.01)
                if name == 'map': other.map_buffer.add_(1)
                if name == 'world': other.world.bias.add_(.01)
            self.assertNotEqual(before, rf_codec_frozen_fingerprint(other), name)
        model.flow_renderer.conditional_restorer = nn.Identity()
        with self.assertRaisesRegex(ValueError, 'conditional restorer'):
            configure_rf_codec_training(model)
        model, _ = full_parent(); model.flow_renderer.steps = 5
        with self.assertRaisesRegex(ValueError, 'RF3'):
            configure_rf_codec_training(model)
        model, _ = full_parent(); model.world.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'Configure'):
            rf_codec_frozen_fingerprint(model)
        self.assertEqual(rf_codec_protocol()['name'], PROTOCOL)
        self.assertEqual(rf_codec_protocol()['fresh_optimizer_groups'], 0)


if __name__ == '__main__':
    unittest.main()
