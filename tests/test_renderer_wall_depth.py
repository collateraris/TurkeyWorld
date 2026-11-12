"""Source-native distant wall support and normalization regressions."""
import json
import math
import unittest

import numpy as np
import torch

from turkeyworld.geometry_targets import _wall_depths
from turkeyworld.renderer_wall_depth import (DEPTH_BANDS, _source_ray_faces,
    _thin_face_columns, source_wall_depth_labels, depth_banded_wall_terms,
    depth_banded_wall_report)
from turkeyworld.renderer_wall_supervision import source_wall_masks


def room():
    grid = torch.ones(1, 16, 16, dtype=torch.long)
    grid[:, 1:-1, 1:-1] = 0
    state = torch.zeros(1, 20)
    state[0, :2] = torch.tensor([1.5, 8.5]) / 16
    state[0, 3] = 1
    state[0, 4] = 1
    return grid, state, torch.zeros(1, 1, 32, 32)


class RendererWallDepthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_source_dda_depth_and_disjoint_full_support(self):
        grid, state, mask = room()
        tangent = math.tan(math.radians(68) / 2)
        screen = (np.arange(32) + .5) / 32 * 2 - 1
        for angle in (0., .3, -.8, 1.7):
            state[0, 2:4] = torch.tensor([math.sin(angle), math.cos(angle)])
            labels = source_wall_depth_labels(grid, state, mask)
            decoded_angle = math.atan2(float(state[0, 2]), float(state[0, 3]))
            c, s = math.cos(decoded_angle), math.sin(decoded_angle)
            reference = _wall_depths(grid[0].numpy(), 1.5, 8.5,
                c - s * tangent * screen, s + c * tangent * screen, 100.)
            np.testing.assert_allclose(labels['depth'][0, 0, 0].numpy(), reference, rtol=1e-7)
            base = source_wall_masks(grid, state, mask)
            for key in ('wall', 'boundary', 'horizontal', 'vertical'):
                total = sum(v[key] for v in labels['bands'].values())
                torch.testing.assert_close(total, base[key], rtol=0, atol=0)
            self.assertEqual(tuple(labels['bands']), tuple(v[0] for v in DEPTH_BANDS))

    def test_source_npc_weapon_occluders_never_enter_depth_region_denominators(self):
        grid, state, mask = room()
        original = source_wall_depth_labels(grid, state, mask)
        # Cover every previously exposed wall pixel: none may remain in any
        # wall/edge/corner stratum, even when its depth is long.
        mask[:] = 1
        hidden = source_wall_depth_labels(grid, state, mask)
        self.assertGreater(int(original['aggregates']['depth_ge_12']['wall'].sum()), 0)
        for groups in (hidden['bands'], hidden['aggregates']):
            for labels in groups.values():
                self.assertTrue(all(not bool(v.any()) for v in labels.values()))

    def test_thin_faces_use_native_width_and_do_not_split_coplanar_tiles(self):
        depths = np.full(10, 8.)
        faces = np.asarray([[0, 4, 1]] * 4 + [[1, 7, 1]] +
                           [[0, 8, 2]] * 2 + [[0, 8, 3]] * 3)
        thin, change = _thin_face_columns(depths, faces)
        np.testing.assert_array_equal(thin, [0, 0, 0, 0, 1, 1, 1, 0, 0, 0])
        np.testing.assert_array_equal(np.flatnonzero(change), [3, 4, 6])
        depths[4:7] = 4.999
        self.assertFalse(bool(_thin_face_columns(depths, faces)[0].any()))
        grid, _, _ = room()
        ray = np.asarray([1., 1.])
        depth, keys = _source_ray_faces(grid[0].numpy(), 1.5, 8.5, ray, np.asarray([.02, -.02]))
        # Two rays hit neighboring tiles of the same outer-wall plane.
        self.assertTrue(np.array_equal(keys[0], keys[1]))
        self.assertTrue(bool((depth >= 12).all()))

    def test_far_terms_ignore_near_area_and_absent_images_and_detach_labels(self):
        grid, state, mask = room()
        state.requires_grad_(); mask.requires_grad_()
        labels = source_wall_depth_labels(grid, state, mask)
        far = labels['bands']['thin_12_plus']['wall']
        target = torch.zeros(1, 3, 32, 32, requires_grad=True)
        prediction = (far.expand_as(target) * .4).requires_grad_()
        terms = depth_banded_wall_terms(prediction, target, labels)
        distant = terms['bands']['thin_12_plus']
        self.assertGreater(distant['wall_pixels'], 0)
        self.assertAlmostEqual(float(distant['wall_l1'].detach()), .4, places=6)
        extended = {**labels, **{kind: {name: {key: torch.cat([value, torch.zeros_like(value).expand(7, *value.shape[1:])])
                              for key, value in row.items()} for name, row in labels[kind].items()}
                        for kind in ('bands', 'aggregates')}}
        p = torch.cat([prediction, torch.zeros(7, 3, 32, 32)])
        t = torch.zeros_like(p)
        many = depth_banded_wall_terms(p, t, extended)['bands']['thin_12_plus']
        self.assertEqual(many['wall_images'], 1)
        torch.testing.assert_close(distant['wall_l1'], many['wall_l1'], rtol=0, atol=0)
        distant['wall_l1'].backward()
        self.assertGreater(float(prediction.grad.abs().sum()), 0)
        self.assertIsNone(target.grad); self.assertIsNone(state.grad); self.assertIsNone(mask.grad)

    def test_report_full_native_json_and_same_source_support(self):
        grid, state, mask = room()
        source = torch.zeros(1, 3, 32, 32)
        arrays = {'map': grid, 'states': state, 'mask': mask, 'source': source,
                  'base': source + .3, 'rgb': source + .2, 'codec_reference': source + .1}
        report = depth_banded_wall_report(arrays)
        json.dumps(report, allow_nan=False)
        self.assertFalse(report['codec_reference_is_a_proven_lower_bound'])
        for name in report['base']['bands']:
            for field in ('wall_images', 'wall_pixels', 'boundary_images', 'corner_images'):
                self.assertEqual(report['base']['bands'][name][field], report['rf1']['bands'][name][field])
        thin = report['rf1']['bands']['thin_12_plus']
        self.assertAlmostEqual(thin['wall_l1'], .2, places=6)
        self.assertGreater(thin['thin_height_pixels'], 0)


if __name__ == '__main__':
    unittest.main()
