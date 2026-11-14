import unittest
import numpy as np

from turkeyworld.dreamer_policy_data import window_index, mixture_probabilities


class ReplayIndexTest(unittest.TestCase):
    def test_terminal_is_not_padding_or_time_limit(self):
        states = np.zeros((8, 20), np.float32)
        states[:5, 4] = 1
        events = np.zeros((7, 4), np.float32)
        starts, event, death = window_index(states, events, 2)
        self.assertEqual(starts.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(np.flatnonzero(death).tolist(), [3, 4])
        self.assertEqual(event.tolist(), death.tolist())

    def test_rare_overlap_importance_recovers_uniform_mean(self):
        event = np.array([0, 0, 1, 1], bool)
        death = np.array([0, 0, 0, 1], bool)
        _, p = mixture_probabilities(4, event, death)
        self.assertAlmostEqual(p.sum(), 1)
        target = np.array([0., .1, 1., -1.])
        self.assertAlmostEqual(np.sum(p*((1/4)/p)*target), target.mean())
        self.assertGreater(p[-1], p[0])

    def test_no_rare_pool_is_uniform(self):
        mass, p = mixture_probabilities(3, [False]*3, [False]*3)
        np.testing.assert_allclose(mass, [1, 0, 0])
        np.testing.assert_allclose(p, [1/3]*3)


if __name__ == '__main__': unittest.main()
