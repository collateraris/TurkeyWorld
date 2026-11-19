import unittest

import numpy as np
import torch

from test_base_student import states, tiny_teacher
from turkeyworld.validate_vjepa_combat import summarize
from turkeyworld.vjepa_policy_world import VJEPAImaginationAdapter


class CombatValidationTest(unittest.TestCase):
    def test_event_alignment_health_and_persistent_life(self):
        initial = np.zeros((1, 20), dtype=np.float32)
        initial[0, 13] = 1
        true = np.zeros((1, 4, 20), dtype=np.float32)
        true[0, :, 13] = (1, 0, 1, 1)
        pred = true.copy()
        # Miss the reference NPC death and respawn. The last two coordinates
        # belong to different lives even though both slots are marked alive.
        pred[0, :, 13] = 1
        pred[0, 1:, 8] = 0.5
        pred[0, :, 4] = 0.25
        truth_events = np.zeros((1, 4, 4))
        truth_events[0, [1, 3], 1] = 1
        predicted_events = np.zeros_like(truth_events)
        predicted_events[0, [1, 2], 1] = 1
        metrics = summarize(pred, true, predicted_events, truth_events, initial)
        self.assertEqual(metrics["normalized_hp"]["player"]["mean_abs"], 0.25)
        self.assertEqual(metrics["xy"]["npc_same_life"]["ticks_scored"], 1)
        self.assertEqual(metrics["xy"]["npc_same_life"]["mean_path_tiles"], 0)
        self.assertIsNone(metrics["xy"]["npc_same_life"]["H180_tiles"])
        self.assertEqual(
            metrics["exact_tick_events"]["hit"],
            dict(reference=2, predicted=2, tp=1, fp=1, fn=1, precision=0.5, recall=0.5, f1=0.5),
        )
        self.assertIsNone(metrics["exact_tick_events"]["kill"]["f1"])

    def test_respawn_counts_keep_timing_and_placement_separate(self):
        initial = np.zeros((2, 20), dtype=np.float32)
        true = np.zeros((2, 4, 20), dtype=np.float32)
        true[:, :, 13] = (1, 0, 1, 1)
        pred = true.copy()
        pred[1, 2, 13] = 0  # Its second respawn arrives one tick late.
        pred[0, 0, 8] = 0.25  # First tick matches the birth, but is four tiles off.
        events = np.zeros((2, 4, 4))
        respawn = summarize(pred, true, events, events, initial)["exact_tick_respawns"]
        self.assertEqual((respawn["reference"], respawn["predicted"]), (4, 4))
        self.assertEqual((respawn["tp"], respawn["fp"], respawn["fn"]), (3, 1, 1))
        self.assertEqual(respawn["same_tile"], 2)
        self.assertAlmostEqual(respawn["xy_mean_tiles_on_matched_ticks"], 4 / 3)

    def test_predicted_events_share_the_runtime_transition(self):
        torch.set_num_threads(2)
        adapter = VJEPAImaginationAdapter(tiny_teacher())
        state = states()
        initial = adapter.initialize(
            state, torch.zeros(2, 16, 16, dtype=torch.long), torch.zeros(2, 3, 32, 32)
        )
        control = torch.zeros(2, 4)
        control[:, 3] = 1
        reference = adapter.transition_prior(initial, control)
        step, info = adapter.transition_with_info(initial, control)
        self.assertTrue(torch.equal(step.state, reference.state))
        self.assertTrue(torch.equal(step.memory, reference.memory))
        self.assertEqual(info["events"].shape, (2, 4))
        self.assertTrue(torch.equal(info["state"], step.state))
