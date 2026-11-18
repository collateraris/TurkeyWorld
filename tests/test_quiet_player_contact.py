import unittest

import numpy as np

from turkeyworld.train_quiet_player_contact import bracket, features
from turkeyworld.train_local_axis_collision import walkable


class QuietPlayerContactTest(unittest.TestCase):
    def test_nearest_occupied_tile_features_at_face_and_corner(self):
        maps = np.zeros((1, 16, 16), dtype=np.uint8)
        maps[0, 6, 8] = 1
        ids = np.array([0, 0])
        xy = np.array([[7.75, 6.5], [7.75, 5.875]])
        np.testing.assert_allclose(features(xy, maps, ids), [[1.0, 0.0], [1.0, 0.25]])

        start = np.array([[7.25, 6.5], [7.25, 5.25]])
        end = np.array([[8.5, 6.5], [8.5, 6.5]])
        contact = bracket(maps, ids, start, end)
        offset = (end - start) * 1.0e-5
        self.assertTrue(walkable(maps, ids, contact - offset).all())
        self.assertFalse(walkable(maps, ids, contact + offset).any())
