"""Private TRAIN-only far-wall pool, exact resume and unchanged MAIN clocks."""
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.renderer_far_wall_data import FarWallFrames, STRATA


class SourceParent:
    split = 'train'

    def __init__(self, size=9):
        self.rows = [{'file': 0, 'frame': i, 'category': 'empty_navigation' if i % 3 == 0 else 'medium'}
                     for i in range(size)]
        self.files = [{'path': 'genuine_source_train.npz', 'sha256': 'fixed_source_file_sha'}]
        self.identity = {'fingerprint': 'immutable_train_parent'}
        self.rng = np.random.default_rng(173)
        self.draws = 16
        self.case_counts = {'original': 16}
        self.explicit_reads = []
        self.fail = False

    def sampler_state(self):
        return {'draws': self.draws, 'case_counts': dict(self.case_counts),
                'rng': copy.deepcopy(self.rng.bit_generator.state)}

    def batch(self, size, *, indices=None):
        if indices is None:
            raise AssertionError('Auxiliary pool must not draw the MAIN sampler')
        self.explicit_reads.append(list(indices))
        if self.fail:
            raise RuntimeError('Source read failed')
        state = torch.zeros(size, 20)
        state[:, :2] = .5; state[:, 3] = 1
        state[:, 19] = torch.tensor(indices)
        mask = torch.zeros(size, 1, 32, 32)
        for row, index in enumerate(indices):
            if index % 3:
                mask[row].view(-1)[:index + 1] = 1
                state[row, 12:14] = 1
        return {'state': state, 'map': torch.zeros(size, 16, 16),
                'object_mask': mask, 'target': torch.zeros(size, 3, 32, 32),
                'row_index': torch.tensor(indices)}


def labels_for_ids(grid, states, masks):
    del grid, masks
    count = len(states)
    d8 = torch.zeros(count, 1, 32, 32)
    d12 = torch.zeros_like(d8); thin = torch.zeros_like(d8); height = torch.zeros_like(d8)
    for row, state in enumerate(states):
        index = int(state[19])
        if index % 3 == 0:
            d8[row, 0, 16, 1:4] = 1; d12[row, 0, 16, 1] = 1
            height[row, 0, 16, 1] = 1
        elif index % 3 == 1:
            thin[row, 0, 16, 1] = 1
        else:
            d8[row, 0, 16, 1:3] = 1
    return {'aggregates': {'depth_ge_5': {'thin_face': thin},
            'depth_ge_8': {'wall': d8}, 'depth_ge_12': {'wall': d12}}, 'thin_height': height}


class RendererFarWallDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def build(self, parent=None, **kwargs):
        parent = SourceParent() if parent is None else parent
        with patch('turkeyworld.renderer_far_wall_data.source_wall_depth_labels', side_effect=labels_for_ids):
            return FarWallFrames(parent, candidate_count=9, label_batch_size=4, **kwargs)

    def test_source_only_disjoint_pools_json_receipt_and_unchanged_main_pcg(self):
        parent = SourceParent(); before = parent.sampler_state()
        numpy_before = copy.deepcopy(np.random.get_state())
        torch_before = torch.get_rng_state().clone()
        pool = self.build(parent)
        self.assertEqual(pool.pool_buckets, {'depth_ge_12': [0, 3, 6],
            'thin_face_ge_5': [1, 4, 7], 'other_depth_ge_8': [2, 5, 8]})
        self.assertEqual(parent.sampler_state(), before)
        self.assertEqual(sorted(i for batch in parent.explicit_reads for i in batch), list(range(9)))
        self.assertEqual(pool.draws, 0)
        self.assertEqual(pool.identity['candidate_support_frames']['depth_ge_12'], 3)
        self.assertEqual(pool.identity['candidate_support_pixels']['depth_ge_8'], 15)
        self.assertFalse(pool.identity['validation_used'])
        self.assertFalse(pool.identity['gpu_used_for_pool'])
        self.assertEqual(np.random.get_state()[0], numpy_before[0])
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_before[1])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))
        json.dumps(pool.identity, allow_nan=False); json.dumps(pool.sampler_state(), allow_nan=False)
        for _ in range(3):
            ids = pool.batch(4)['row_index'].tolist()
            self.assertTrue(all(i in pool.facts for i in ids))
        self.assertEqual(pool.stratum_counts, dict.fromkeys(STRATA, 4))
        self.assertEqual(sum(pool.actual_area_counts.values()), 12)
        self.assertEqual(parent.sampler_state(), before)

    def test_exact_private_resume_and_explicit_reads_do_not_advance_any_sampler(self):
        pool = self.build()
        pool.batch(4); saved = pool.sampler_state()
        same = self.build(); same.restore_sampler(copy.deepcopy(saved))
        before_main = pool.parent.sampler_state()
        a = pool.batch(17)['row_index']; b = same.batch(17)['row_index']
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(pool.sampler_state(), same.sampler_state())
        snapshot = pool.sampler_state()
        pool.batch(3, indices=[0, 4, 8])
        self.assertEqual(snapshot, pool.sampler_state())
        self.assertEqual(before_main, pool.parent.sampler_state())
        for key, value in [('fingerprint', 'other'), ('protocol', 'old'), ('draws', saved['draws'] + 1)]:
            bad = copy.deepcopy(saved); bad[key] = value
            with self.assertRaises(ValueError): same.restore_sampler(bad)
        bad = copy.deepcopy(saved); bad['stratum_counts'] = dict(zip(STRATA, [1, 2, 1]))
        with self.assertRaises(ValueError): same.restore_sampler(bad)
        bad = copy.deepcopy(saved); bad['actual_support_counts']['depth_ge_12'] = 99
        with self.assertRaises(ValueError): same.restore_sampler(bad)

    def test_source_read_failure_rolls_back_only_private_successful_draws(self):
        pool = self.build(); before = pool.sampler_state(); main = pool.parent.sampler_state()
        pool.parent.fail = True
        with self.assertRaisesRegex(RuntimeError, 'Source read failed'):
            pool.batch(4)
        self.assertEqual(pool.sampler_state(), before)
        self.assertEqual(pool.parent.sampler_state(), main)

    def test_no_validation_or_unbounded_mining_or_missing_stratum_fallback(self):
        parent = SourceParent(); parent.split = 'val'
        with self.assertRaisesRegex(ValueError, 'TRAIN'):
            self.build(parent)
        with self.assertRaisesRegex(ValueError, 'bounded'):
            FarWallFrames(SourceParent(), candidate_count=4097)
        def missing(*args):
            labels = labels_for_ids(*args)
            labels['aggregates']['depth_ge_12']['wall'].zero_()
            return labels
        with patch('turkeyworld.renderer_far_wall_data.source_wall_depth_labels', side_effect=missing):
            with self.assertRaisesRegex(ValueError, 'misses depth_ge_12'):
                FarWallFrames(SourceParent(), candidate_count=9)
        pool = self.build()
        with self.assertRaises(ValueError): pool.batch(1, indices=[False])
        with self.assertRaises(ValueError): pool.batch(1, indices=[100])


if __name__ == '__main__':
    unittest.main()
