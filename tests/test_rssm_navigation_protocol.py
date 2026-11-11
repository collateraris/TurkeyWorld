import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from turkeyworld.rssm_navigation_protocol import (AXIS_OUTCOMES,
    NavigationContactSampler, bind_navigation_revision, verify_restored_boundary)
from turkeyworld.rssm_navigation_revision import commit_revision_update


def fixture(root):
    corpus = root/'data'
    folder = corpus/'short/rgb/train'
    folder.mkdir(parents=True)
    records, paths = [], []
    cohorts = ('free', 'blocked', 'slide', 'turn_idle')
    for index in range(8):
        path = folder/f'episode_{index:04d}.npz'
        np.savez_compressed(path, contact_first16_cohort=np.asarray(cohorts[index % 4]))
        records.append({'relative_path': path.relative_to(corpus).as_posix(),
            'split': 'train', 'stage': 'short', 'pool': 'corner' if index < 4 else 'singlewall',
            'axis_outcomes_first16': [AXIS_OUTCOMES[index]] if index < 4 else [],
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        paths.append(path)
    (corpus/'manifest.json').write_text(json.dumps({'status': 'complete', 'records': records}), encoding='utf-8')
    contact, continuation, checkpoint = [root/name for name in ('old_contact.json', 'old_continuation.json', 'full_adam.pt')]
    for path in (contact, continuation, checkpoint):
        path.write_bytes(b'original bytes remain unchanged')
    state = {'loss_protocol': 'old_protocol', 'training_protocol': {},
        'rssm_training': {'phase': 'source', 'source_completed': 3000,
            'continuation': {'consecutive_passes': 0, 'status': 'training'}},
        'rssm_validation': {key: {'fingerprint': key+'-immutable'} for key in ('long450', 'one_frame450')}}
    options = dict(dataset_root=corpus, base_plan_sha='original', contact_plan_path=contact,
        continuation_plan_path=continuation, checkpoint_path=checkpoint, step=3200,
        branch_protocol={'version': 1, 'max_branches': 4, 'horizon': 16})
    return state, root/'new_plan.json', options, paths


class NavigationProtocolTests(unittest.TestCase):
    def test_restored_boundary_verifies_real_tensors_moments_and_rng(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        model(torch.ones(1, 2)).square().sum().backward()
        optimizer.step()
        rng = np.random.default_rng(73)
        reference = {'model': copy.deepcopy(model.state_dict()),
            'optimizer': copy.deepcopy(optimizer.state_dict()),
            'rng_state': copy.deepcopy(rng.bit_generator.state),
            'torch_rng_state': torch.get_rng_state().clone(), 'training_steps': 1}
        report = verify_restored_boundary(model, optimizer, rng, reference)
        self.assertTrue(report['passed'])
        self.assertEqual(report['optimizer_steps'], [1])
        self.assertGreater(report['compared']['tensors'], 4)
        with torch.no_grad():
            model.weight.add_(1)
        with self.assertRaisesRegex(ValueError, 'did not preserve'):
            verify_restored_boundary(model, optimizer, rng, reference)

    def test_opt_in_keeps_old_bytes_rng_and_committed_resume_clocks(self):
        with TemporaryDirectory() as temporary:
            state, plan_path, options, paths = fixture(Path(temporary))
            old_bytes = {key: options[key].read_bytes() for key in
                ('contact_plan_path', 'continuation_plan_path', 'checkpoint_path')}
            cpu_rng = torch.get_rng_state().clone()
            numpy_rng = np.random.get_state()
            ledger, selected = bind_navigation_revision(state, plan_path, **options)
            self.assertEqual(len(selected), 8)
            plan_bytes = plan_path.read_bytes()
            self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
            self.assertTrue(np.array_equal(numpy_rng[1], np.random.get_state()[1]))
            for key, expected in old_bytes.items():
                self.assertEqual(options[key].read_bytes(), expected)
            commit_revision_update(ledger, step=3201, source_completed=3001)
            state['rssm_training']['source_completed'] = 3001
            resumed, _ = bind_navigation_revision(state, plan_path, **{**options, 'step': 3201})
            self.assertIs(resumed, ledger)
            self.assertEqual(resumed['revision_updates'], 1)
            self.assertEqual(resumed['auxiliary_draws'], {'corner': 8, 'single_wall': 4, 'legacy': 4})
            self.assertEqual(plan_path.read_bytes(), plan_bytes)
            self.assertEqual(state['loss_protocol'], 'rssm_navigation_own_memory_contacts_v3')
            self.assertEqual(state['training_protocol']['main_window_start'], 0)
            with self.assertRaisesRegex(ValueError, 'requires --navigation'):
                bind_navigation_revision(state, plan_path, **{**options, 'dataset_root': None, 'step': 3201})

    def test_corpus_objective_and_clock_changes_fail_closed(self):
        with TemporaryDirectory() as temporary:
            state, path, options, files = fixture(Path(temporary))
            bind_navigation_revision(state, path, **options)
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                bind_navigation_revision(state, path, **{**options, 'branch_protocol': {'weight': 9}})
            wrong = copy.deepcopy(state)
            wrong['rssm_training']['navigation_revision']['auxiliary_draws']['corner'] += 1
            with self.assertRaisesRegex(ValueError, 'auxiliary counters'):
                bind_navigation_revision(wrong, path, **options)
            files[0].write_bytes(b'changed training labels')
            with self.assertRaisesRegex(ValueError, 'member changed'):
                bind_navigation_revision(state, path, **options)

    def test_sampler_balances_actual_xy_outcomes_and_preserves_legacy_replay(self):
        with TemporaryDirectory() as temporary:
            state, path, options, files = fixture(Path(temporary))
            calls = []
            class Pool:
                def __init__(self, paths, tag):
                    self.episodes = [{'path': str(value)} for value in paths]
                    self.tag = tag
                def batch(self, count, **kwargs):
                    calls.append((self.tag, count, kwargs))
                    return {'values': torch.as_tensor(kwargs['indices'][:, 0])[:, None]}
            new = Pool(files, 'new')
            old = Pool(files*2, 'old')
            sampler = NavigationContactSampler(new, old, options['dataset_root'])
            result, cohorts = sampler.sample({'updates': 7}, np.random.default_rng(17))
            self.assertEqual(result['values'].shape, (16, 1))
            indices = calls[0][2]['indices'][:, 0]
            self.assertEqual(indices[:8].tolist(), [0, 0, 1, 1, 2, 2, 3, 3])
            self.assertEqual(set(indices[8:]), {4, 5, 6, 7})
            self.assertEqual(sum(cohorts.values()), 12)
            self.assertEqual(calls[1][1], 4)
            self.assertEqual(calls[1][2]['indices'][0, 0], 7)
            self.assertTrue(all(call[2]['context'] == 1 and call[2]['horizon'] == 16 for call in calls))
            self.assertTrue(all(not call[2]['indices'][:, 1].any() for call in calls))

    def test_disabled_revision_does_not_change_existing_state(self):
        with TemporaryDirectory() as temporary:
            state, path, options, _ = fixture(Path(temporary))
            expected = copy.deepcopy(state)
            self.assertIsNone(bind_navigation_revision(state, path, **{**options, 'dataset_root': None}))
            self.assertEqual(state, expected)
            self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
