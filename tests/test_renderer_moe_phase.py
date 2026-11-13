from copy import deepcopy
import random
import unittest

import numpy as np
import torch
from torch import nn

from turkeyworld.renderer_moe_phase import (renderer_moe_protocol, identity_digest,
    build_renderer_moe_optimizer, restore_renderer_moe_parent)
from turkeyworld.renderer_native_phase import _same


class Stream:
    def __init__(self, value): self.value = deepcopy(value)
    def restore_sampler(self, value): self.value = deepcopy(value)
    def sampler_state(self): return deepcopy(self.value)


class Head(nn.Module):
    def __init__(self, flow=False):
        super().__init__(); self.body = nn.Linear(3, 3)
        if flow:
            self.codec = nn.Linear(3, 3).requires_grad_(False)
            self.geometry_opacity = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))


class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.base_student = Head(); self.flow_renderer = Head(True)
        self.world = nn.Linear(3, 3).requires_grad_(False)


def fixture():
    model = Model()
    old_groups = [list(model.base_student.parameters()), list(model.flow_renderer.body.parameters()),
                  list(model.flow_renderer.geometry_opacity.parameters())]
    optimizer = torch.optim.AdamW([{'params': group, 'lr': lr} for group, lr in
        zip(old_groups, (.0001, .0001, .00005))], weight_decay=.01)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        sum(p.square().mean() for group in old_groups for p in group).backward(); optimizer.step()
    saved = dict(base=deepcopy(model.base_student.state_dict()), flow=deepcopy(model.flow_renderer.state_dict()),
        optimizer=deepcopy(optimizer.state_dict()), step=10000,
        sampler={'draws': 160000, 'rng': np.random.default_rng(17).bit_generator.state},
        far_wall_aux_sampler={'draws': 32000}, geometry_wall_aux_sampler={'draws': 16000},
        configuration={'world': 'old5152.pt', 'batch': 16},
        identity={'world_sha256': 'a' * 64, 'codec_sha256': 'c' * 64, 'train': {'fingerprint': 'train'},
            'val': {'fingerprint': 'val'}, 'panel': [0, 1], 'panel_fingerprint': 'panel',
            'frozen_feature_snapshot': {'world_sha256': 'a' * 64}},
        rng_torch=torch.get_rng_state().clone(), rng_numpy=deepcopy(np.random.get_state()),
        rng_python=deepcopy(random.getstate()), rng_cuda=[])
    for index in saved['optimizer']['param_groups'][2]['params']:
        saved['optimizer']['state'][index]['step'].fill_(1)
    model.flow_renderer.geometry_opacity.requires_grad_(False)
    for head in (model.base_student, model.flow_renderer):
        head.output_moe = nn.Linear(3, 3)
        nn.init.zeros_(head.output_moe.weight); nn.init.zeros_(head.output_moe.bias)
    identity = {**deepcopy(saved['identity']), 'parent_identity_sha256': identity_digest(saved['identity']),
        'feature_world_sha256': 'b' * 64, 'renderer_moe_protocol': renderer_moe_protocol()}
    cfg = {'inherited_parent_configuration': deepcopy(saved['configuration'])}
    streams = [Stream({}) for _ in range(3)]
    return model, saved, identity, cfg, streams


class RendererMoEPhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_full_parent_weights_mixed_age_Adam_options_three_samplers_and_RNG_preserved(self):
        model, saved, identity, cfg, streams = fixture()
        optimizer = build_renderer_moe_optimizer(model, saved)
        new = {n: p.clone() for n, p in model.named_parameters() if '.output_moe.' in n}
        proof = restore_renderer_moe_parent(saved, model, optimizer, *streams, identity, cfg, 'frozen',
                                           parent_checkpoint_sha256='d' * 64)
        self.assertEqual(proof['start_step'], 0); self.assertEqual(proof['parent_step'], 10000)
        self.assertEqual(proof['inherited_Adam_states'], 8)
        self.assertEqual(proof['new_Adam_states_at_start'], 0)
        self.assertTrue(_same(saved['optimizer']['state'], optimizer.state_dict()['state']))
        self.assertEqual(saved['optimizer']['param_groups'], optimizer.state_dict()['param_groups'][:3])
        for n, p in model.named_parameters():
            if n in new: self.assertTrue(torch.equal(p, new[n]))
        for stream, key in zip(streams, ('sampler', 'far_wall_aux_sampler', 'geometry_wall_aux_sampler')):
            self.assertTrue(_same(stream.sampler_state(), saved[key]))
        self.assertTrue(torch.equal(torch.get_rng_state(), saved['rng_torch']))
        self.assertTrue(_same(np.random.get_state(), saved['rng_numpy']))
        self.assertEqual(random.getstate(), saved['rng_python'])

    def test_three_actual_SGD_steps_keep_frozen_private_weights_moments_age_but_train_old_and_new(self):
        model, saved, identity, cfg, streams = fixture()
        optimizer = build_renderer_moe_optimizer(model, saved)
        restore_renderer_moe_parent(saved, model, optimizer, *streams, identity, cfg, 'frozen',
                                    parent_checkpoint_sha256='d' * 64)
        weights = deepcopy(model.state_dict()); moments = deepcopy(optimizer.state_dict())
        private_ids = moments['param_groups'][2]['params']
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            x = torch.ones(2, 3)
            loss = sum((head.body(x) + head.output_moe(x)).square().mean()
                       for head in (model.base_student, model.flow_renderer))
            loss.backward(); optimizer.step()
        self.assertTrue(all(p.grad is None for p in model.flow_renderer.geometry_opacity.parameters()))
        for n, p in model.state_dict().items():
            if 'geometry_opacity' in n or n.startswith(('world.', 'flow_renderer.codec.')):
                self.assertTrue(torch.equal(p, weights[n]))
        for index in private_ids:
            self.assertTrue(_same(moments['state'][index], optimizer.state_dict()['state'][index]))
        for group in optimizer.state_dict()['param_groups'][:2]:
            for index in group['params']: self.assertEqual(float(optimizer.state_dict()['state'][index]['step']), 6)
        for index in optimizer.state_dict()['param_groups'][3]['params']:
            self.assertEqual(float(optimizer.state_dict()['state'][index]['step']), 3)
        self.assertTrue(any(not torch.equal(p, weights[n]) for n, p in model.state_dict().items() if '.output_moe.' in n))

    def test_changed_parent_data_options_or_source_identity_is_rejected(self):
        for kind in ('data', 'options', 'feature', 'missing_moment'):
            model, saved, identity, cfg, streams = fixture()
            optimizer = build_renderer_moe_optimizer(model, saved)
            if kind == 'data': identity['train']['fingerprint'] = 'other'
            elif kind == 'options': optimizer.param_groups[0]['lr'] = .2
            elif kind == 'feature': identity['feature_world_sha256'] = saved['identity']['world_sha256']
            else: saved['optimizer']['state'].pop(next(iter(saved['optimizer']['state'])))
            with self.assertRaises(ValueError):
                restore_renderer_moe_parent(saved, model, optimizer, *streams, identity, cfg, 'frozen',
                                            parent_checkpoint_sha256='d' * 64)


if __name__ == '__main__': unittest.main()
