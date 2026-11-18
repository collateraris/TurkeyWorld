"""Quiet pursuit inference consumes static maps & own poses, not teacher states."""
import unittest
import torch
from unittest import mock
from turkeyworld.learned_quiet_pursuit import QuietPursuitHead


class QuietPursuitTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.head = QuietPursuitHead().eval()
        # Controlled learned weights, not a trained release.
        with torch.no_grad():
            self.head.visibility.bias.fill_(1.0)
            self.head.stop_model.bias.fill_(1.0)
            self.head.contact.score.bias.fill_(1.0)
            self.head.edge.weight.fill_(0.04)
        self.state = torch.zeros(2, 20)
        self.state[:, 8:10] = torch.tensor([4.5, 3.5]) / 16
        self.grid = torch.zeros(2, 16, 16)
        self.grid[:, 0, :] = self.grid[:, -1, :] = self.grid[:, :, 0] = self.grid[:, :, -1] = 1.0

    @torch.no_grad()
    def test_post_player_target_and_engine_independence(self):
        tgt = torch.tensor([[7.5, 3.5], [4.5, 6.5]]) / 16
        with mock.patch("turkeyworld.engine.Game._navigation", side_effect=AssertionError), mock.patch(
            "turkeyworld.engine.Game._line_clear", side_effect=AssertionError
        ), mock.patch("turkeyworld.engine.Game._move", side_effect=AssertionError):
            pred = self.head(self.state, tgt, self.grid)
            self.assertGreater(pred[0, 0], 0.0)
            self.assertEqual(pred[0, 1], 0.0)
            self.assertEqual(pred[1, 0], 0.0)
            self.assertGreater(pred[1, 1], 0.0)

    @torch.no_grad()
    def test_maps_separate_cache_invalidation(self):
        self.grid[1, 3, 5] = 1.0
        npc = self.state[:, 8:10] * 16
        player = torch.tensor([[6.5, 3.5], [6.5, 3.5]])
        mix = self.head.route_components(npc, player, self.grid)
        flat = self.head.route_components(npc[0:1], player[0:1], self.grid[0:1])
        self.assertTrue(torch.equal(mix[0], flat[0]))
        self.assertFalse(torch.equal(mix[1], flat[0]))
        self.assertEqual(int(flat.argmax(-1)), 0)
        self.assertNotEqual(int(mix[1].argmax()), 0)
        self.assertFalse(any("cache" in k for k in self.head.state_dict()))
        self.head.to("cpu")
        self.assertIsNone(self.head._cache)
