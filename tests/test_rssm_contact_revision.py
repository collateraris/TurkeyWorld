import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from turkeyworld.rssm_contact_revision import (
    COHORTS, ContactAuxiliarySampler, bind_contact_revision, commit_contact_revision)


class Pool:
    def __init__(self, paths, offset):
        self.episodes = [{'path': str(path)} for path in paths]
        self.offset = offset

    def batch(self, count, *, context, horizon, indices):
        assert context == 1 and horizon == 16
        return {'selected': torch.tensor(indices[:, :1]+self.offset)}


class ContactRevisionTests(unittest.TestCase):
    def fixture(self, root):
        data = root/'data'
        data.mkdir()
        records, paths = [], []
        for index, key in enumerate(COHORTS):
            path = data/f'{key}.npz'
            active = np.zeros((32, 2), bool)
            labels = np.zeros((32, 2), np.float32)
            if key in ('free', 'blocked'): active[:, 0] = True
            if key == 'free': labels[:, 0] = 1
            if key == 'slide': active[:] = True; labels[:, 1] = 1
            np.savez_compressed(path, axis_pass_targets=labels, axis_command_active=active)
            paths.append(path)
            records.append({'relative_path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        manifest = data/'manifest.json'
        manifest.write_text(json.dumps({'status': 'complete', 'records': records}), encoding='utf-8')
        plan = {'version': 2, 'base_plan_sha256': 'original', 'microbatch_size': 16,
            'gradient_accumulation_steps': 1, 'horizon': 450, 'auxiliary_horizon': 16,
            'new_auxiliary_examples': 12, 'old_auxiliary_examples': 4, 'cohorts': list(COHORTS),
            'validation_every': 100, 'preview_every': 10, 'loss_settings': {'version': 2},
            'start_optimizer_step': 339, 'dataset_directory': str(data),
            'dataset_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}
        path = root/'revision.json'
        path.write_text(json.dumps(plan), encoding='utf-8')
        state = {'loss_protocol': 'v1', 'rssm_training': {'source_completed': 139,
            'phase': 'source', 'distillation_completed': 200, 'distillation_updates': 200}}
        return state, path, paths

    def test_sampler_balances_cohorts_replays_default_and_commits_only_after_success(self):
        with TemporaryDirectory() as tmp:
            state, path, paths = self.fixture(Path(tmp))
            revision = bind_contact_revision(state, path, base_plan_sha='original', step=339)
            sampler = ContactAuxiliarySampler(Pool(paths, 0), Pool(paths*4, 100))
            rng = np.random.default_rng(9)
            saved_rng = copy.deepcopy(rng.bit_generator.state)
            batch, selected = sampler.sample(revision, rng)
            self.assertEqual(revision['new_auxiliary_draws'], 0)
            self.assertEqual(selected, {key: 3 for key in COHORTS})
            self.assertEqual(batch['selected'][:12].flatten().tolist(), [0]*3+[1]*3+[2]*3+[3]*3)
            self.assertEqual(int(batch['selected'][12]), 100)
            rng.bit_generator.state = saved_rng
            replayed, _ = sampler.sample(revision, rng)
            self.assertTrue(torch.equal(batch['selected'], replayed['selected']))
            state['rssm_training']['source_completed'] += 1
            commit_contact_revision(revision, step=340, selected=selected)
            resumed = bind_contact_revision(state, path, base_plan_sha='original', step=340)
            self.assertEqual(resumed['new_auxiliary_draws'], 12)
            self.assertEqual(resumed['old_auxiliary_draws'], 4)
            with self.assertRaises(ValueError): commit_contact_revision(revision, step=340, selected=selected)

    def test_content_revision_or_clock_tampering_cannot_silently_resume(self):
        with TemporaryDirectory() as tmp:
            state, path, paths = self.fixture(Path(tmp))
            revision = bind_contact_revision(state, path, base_plan_sha='original', step=339)
            revision['new_auxiliary_draws'] = 12
            with self.assertRaisesRegex(ValueError, 'clocks'):
                bind_contact_revision(state, path, base_plan_sha='original', step=339)
            revision['new_auxiliary_draws'] = 0
            paths[0].write_bytes(b'changed content')
            with self.assertRaisesRegex(ValueError, 'modified'):
                bind_contact_revision(state, path, base_plan_sha='original', step=339)

    def test_absent_plan_does_not_disable_saved_revision(self):
        with TemporaryDirectory() as tmp:
            state, path, _ = self.fixture(Path(tmp))
            bind_contact_revision(state, path, base_plan_sha='original', step=339)
            path.unlink()
            with self.assertRaisesRegex(ValueError, 'requires its immutable plan'):
                bind_contact_revision(state, path, base_plan_sha='original', step=339)

    def test_move_and_turn_cohort_is_taken_from_source_metadata(self):
        with TemporaryDirectory() as tmp:
            _, _, paths = self.fixture(Path(tmp))
            with np.load(paths[0]) as saved:
                labels, active = saved['axis_pass_targets'], saved['axis_command_active']
            np.savez_compressed(paths[0], axis_pass_targets=labels, axis_command_active=active,
                contact_first16_cohort=np.asarray('turn_idle'))
            extra = Path(tmp)/'extra_free.npz'
            np.savez_compressed(extra, axis_pass_targets=labels, axis_command_active=active,
                contact_first16_cohort=np.asarray('free'))
            sampler = ContactAuxiliarySampler(Pool(paths+[extra], 0), Pool(paths*4, 100))
            self.assertIn(0, sampler.buckets['turn_idle'])
            self.assertNotIn(0, sampler.buckets['free'])

    def test_unlisted_episode_is_rejected_even_when_manifest_is_unchanged(self):
        with TemporaryDirectory() as tmp:
            state, path, paths = self.fixture(Path(tmp))
            (paths[0].parent/'unlisted.npz').write_bytes(paths[0].read_bytes())
            with self.assertRaisesRegex(ValueError, 'unlisted'):
                bind_contact_revision(state, path, base_plan_sha='original', step=339)


if __name__ == '__main__':
    unittest.main()
