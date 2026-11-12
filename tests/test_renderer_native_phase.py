"""Actual Adam and actual native-area sampler continuity across an explicit phase."""
import copy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_renderer_distance_audit import FakeParent
from test_renderer_enemy_objectives import tiny_model
from turkeyworld.renderer_far_wall_protocol import far_wall_protocol
from turkeyworld.renderer_native_phase import (restore_renderer_native_parent,
                                               restore_renderer_native_rng, _same)
from turkeyworld.renderer_refinement import frozen_fingerprint
from turkeyworld.renderer_small_enemy_data import SmallEnemyBalancedFrames
from turkeyworld.renderer_student import _hash_tensors


class RendererNativePhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.original_rng = (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def tearDown(self):
        torch.set_rng_state(self.original_rng[0])
        np.random.set_state(self.original_rng[1])
        random.setstate(self.original_rng[2])

    @staticmethod
    def optimizer(model):
        return torch.optim.AdamW([
            {'params': model.base_student.parameters(), 'lr': 1e-4},
            {'params': [p for n, p in model.flow_renderer.named_parameters()
                        if not n.startswith('codec.') and p.requires_grad], 'lr': 1e-4}], weight_decay=.01)

    def sampler(self):
        path = self.root/'train.npz'
        states = np.zeros((6, 20), np.float32)
        states[:, 12:14] = 1
        states[:, 8] = np.array([1, 2, 4, 5, 1, 1])/16
        states[4, 12:14] = 0
        np.savez(path, states=states)
        parent = FakeParent(path); parent.split = 'train'
        for row, area in zip(parent.rows, (1, 8, 31, 90, 12, 0)):
            row['mask_fraction'] = area/1024
        return SmallEnemyBalancedFrames(parent)

    def fixture(self):
        old = tiny_model()
        optimizer = self.optimizer(old)
        data = self.sampler()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            sum(p.square().sum() for group in optimizer.param_groups for p in group['params']).backward()
            optimizer.step()
            data.batch(16)
        config = {'steps': 3, 'run_dir': 'v5', 'small_enemy_rebalance': True,
                  'batch': 16, 'lr': 1e-4, 'base_lr': 1e-4, 'world': 'frozen5152.pt',
                  'renderer_checkpoint': 'selected6600.pt', 'data_root': 'originalSource',
                  'couple_base_flow': False, 'source_latent_weight': .1}
        identity = {'train': copy.deepcopy(data.parent.identity), 'val': {'fingerprint': 'heldout'},
                    'panel': [0, 1], 'panel_fingerprint': 'old_main256',
                    'native_area_guard_fingerprint': 'old_native256',
                    'world_sha256': 'current5152', 'renderer_sha256': 'selected6600',
                    'codec_sha256': _hash_tensors(old.flow_renderer.codec.state_dict())}
        saved = {'base': copy.deepcopy(old.base_student.state_dict()),
                 'flow': copy.deepcopy(old.flow_renderer.state_dict()),
                 'optimizer': copy.deepcopy(optimizer.state_dict()), 'step': 3,
                 'configuration': config, 'identity': identity, 'frozen_fingerprint': frozen_fingerprint(old),
                 'sampler': copy.deepcopy(data.sampler_state()), 'best': {'step': 2, 'score': .2},
                 'streak': 4, 'enemy_streak': 1,
                 'rng_torch': torch.Generator().manual_seed(51).get_state(),
                 'rng_numpy': np.random.RandomState(71).get_state(),
                 'rng_python': random.Random(91).getstate(), 'rng_cuda': []}
        target = copy.deepcopy(old)
        with torch.no_grad():
            next(target.base_student.parameters()).add_(.05)
            target.flow_renderer.image_input.weight.add_(.05)
        target_optimizer = self.optimizer(target)
        fresh = self.sampler()
        new_config = {**config, 'run_dir': 'v6', 'steps': 6,
                      'native_phase_parent': 'v5/immutable_parent.pt', 'far_wall_supervision': True}
        new_identity = {**copy.deepcopy(identity), 'far_wall_protocol': far_wall_protocol()}
        return old, optimizer, data, saved, target, target_optimizer, fresh, new_config, new_identity

    def adopt(self, saved, target, optimizer, data, config, identity):
        return restore_renderer_native_parent(saved, target, optimizer, data, identity, config,
            saved['frozen_fingerprint'], parent_checkpoint_sha256='a'*64, far_wall_protocol=far_wall_protocol())

    def test_all_weights_adam_rng_and_actual_sampler_resume_then_next_adam_matches_uninterrupted(self):
        old, optimizer, data, saved, target, new_optimizer, fresh, config, identity = self.fixture()
        initial_codec = _hash_tensors(old.flow_renderer.codec.state_dict())
        expected_ids = data.batch(16)['indices']
        restore_renderer_native_rng(saved, device='cpu')
        expected_random = (torch.rand(4), np.random.random(4), random.random())
        torch.rand(31); np.random.random(31); random.random()
        receipt = self.adopt(saved, target, new_optimizer, fresh, config, identity)
        self.assertEqual(receipt['start_step'], 3)
        self.assertTrue(receipt['global_rng_restored'])
        self.assertEqual(receipt['parent_best'], saved['best'])
        self.assertEqual(receipt['parent_streak'], 4)
        self.assertTrue(_same(saved['optimizer'], new_optimizer.state_dict()))
        self.assertTrue(_same(saved['sampler'], fresh.sampler_state()))
        self.assertEqual(fresh.phase_draws, 48)
        self.assertEqual(receipt['parent_mix_counts'], dict.fromkeys(fresh.parent.mix_counts, 0))
        np.testing.assert_array_equal(expected_ids, fresh.batch(16)['indices'])
        self.assertTrue(torch.equal(torch.rand(4), expected_random[0]))
        np.testing.assert_array_equal(np.random.random(4), expected_random[1])
        self.assertEqual(random.random(), expected_random[2])
        self.assertTrue(_same(old.state_dict(), target.state_dict()))
        for model, opt in ((old, optimizer), (target, new_optimizer)):
            opt.zero_grad(set_to_none=True)
            sum(p.square().sum() for group in opt.param_groups for p in group['params']).backward()
            opt.step()
        self.assertTrue(_same(old.state_dict(), target.state_dict()))
        self.assertTrue(_same(optimizer.state_dict(), new_optimizer.state_dict()))
        self.assertEqual({int(v['step']) for v in new_optimizer.state.values()}, {4})
        self.assertEqual(_hash_tensors(target.flow_renderer.codec.state_dict()), initial_codec)

    def test_settings_whitelist_rejects_lr_batch_source_or_feature_changes_before_mutation(self):
        _, _, _, saved, target, optimizer, data, config, identity = self.fixture()
        before = (_hash_tensors(target.state_dict()), copy.deepcopy(optimizer.state_dict()),
                  copy.deepcopy(data.sampler_state()), torch.get_rng_state().clone())
        for field, value in [('lr', 2e-4), ('batch', 32), ('world', 'other.pt'),
                             ('renderer_checkpoint', 'other.pt'), ('data_root', 'changed'),
                             ('source_latent_weight', .03), ('couple_base_flow', True)]:
            with self.assertRaisesRegex(ValueError, 'inherited settings'):
                self.adopt(saved, target, optimizer, data, {**config, field: value}, identity)
            self.assertEqual(_hash_tensors(target.state_dict()), before[0])
            self.assertTrue(_same(optimizer.state_dict(), before[1]))
            self.assertTrue(_same(data.sampler_state(), before[2]))
            self.assertTrue(torch.equal(torch.get_rng_state(), before[3]))

    def test_additive_train_aux_pool_allowed_but_old_panels_and_protocol_tampering_rejected(self):
        _, _, _, saved, target, optimizer, data, config, identity = self.fixture()
        for field in ('panel_fingerprint', 'native_area_guard_fingerprint', 'world_sha256'):
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                self.adopt(saved, target, optimizer, data, config, {**identity, field: 'different'})
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            self.adopt(saved, target, optimizer, data, config, {**identity, 'unapproved_guard': {}})
        changed = copy.deepcopy(identity); changed['far_wall_protocol']['version'] = 999
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            self.adopt(saved, target, optimizer, data, config, changed)
        pool = {'protocol': 'source_train_far_wall_aux_pool_v1', 'split': 'train',
                'fingerprint': 'f'*64, 'parent_source_fingerprint': saved['identity']['train']['fingerprint']}
        bad = {**identity, 'far_wall_aux_pool': {**pool, 'split': 'val'}}
        with self.assertRaisesRegex(ValueError, 'TRAIN'):
            self.adopt(saved, target, optimizer, data, config, bad)
        receipt = self.adopt(saved, target, optimizer, data, config, {**identity, 'far_wall_aux_pool': pool})
        self.assertTrue(receipt['new_far_wall_aux_sampler_is_independent'])
        self.assertEqual(data.phase_draws, 48)

    def test_codec_noise_moments_sampler_or_incomplete_boundary_cannot_be_laundered(self):
        _, _, _, saved, target, optimizer, data, config, identity = self.fixture()
        before = _hash_tensors(target.state_dict())
        alterations = []
        bad = copy.deepcopy(saved); bad['flow']['latent_mean'].add_(.1); alterations.append(bad)
        bad = copy.deepcopy(saved); key = next(k for k in bad['flow'] if k.startswith('codec.'))
        bad['flow'][key].flatten()[0] += 1; alterations.append(bad)
        bad = copy.deepcopy(saved); next(iter(bad['optimizer']['state'].values()))['exp_avg'].fill_(float('nan')); alterations.append(bad)
        bad = copy.deepcopy(saved); bad['sampler']['actual_area_counts']['empty'] += 1; alterations.append(bad)
        bad = copy.deepcopy(saved); bad['step'] = 2; alterations.append(bad)
        bad = copy.deepcopy(saved); del bad['rng_numpy']; alterations.append(bad)
        for bad in alterations:
            with self.assertRaises(ValueError):
                self.adopt(bad, target, optimizer, data, config, identity)
            self.assertEqual(_hash_tensors(target.state_dict()), before)
            self.assertEqual(optimizer.state_dict()['state'], {})
            self.assertEqual(data.phase_draws, 0)

    def test_ordered_groups_cannot_include_codec_or_reorder_base_parameters(self):
        _, _, _, saved, target, _, data, config, identity = self.fixture()
        wrong = torch.optim.AdamW([
            {'params': list(reversed(list(target.base_student.parameters()))), 'lr': 1e-4},
            {'params': [p for n, p in target.flow_renderer.named_parameters()
                        if not n.startswith('codec.') and p.requires_grad], 'lr': 1e-4}], weight_decay=.01)
        with self.assertRaisesRegex(ValueError, 'ordered'):
            self.adopt(saved, target, wrong, data, config, identity)
        self.assertEqual(wrong.state_dict()['state'], {})

    def test_cuda_rng_restore_passes_only_cpu_byte_tensors_without_any_gpu_execution(self):
        saved = {'rng_torch': torch.Generator().manual_seed(3).get_state(),
                 'rng_numpy': np.random.RandomState(5).get_state(),
                 'rng_python': random.Random(7).getstate(),
                 'rng_cuda': [torch.arange(16, dtype=torch.uint8)]}
        captured = []
        with patch('torch.cuda.set_rng_state_all', side_effect=lambda states: captured.extend(states)):
            receipt = restore_renderer_native_rng(saved, device='cuda')
        self.assertTrue(receipt['cuda_rng_restored'])
        self.assertFalse(receipt['cuda_rng_restore_required'])
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].device.type, 'cpu')
        self.assertEqual(captured[0].dtype, torch.uint8)
        self.assertTrue(torch.equal(captured[0], saved['rng_cuda'][0]))


if __name__ == '__main__':
    unittest.main()
