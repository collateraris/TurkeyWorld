"""Own-prediction TBPTT, checkpoint continuation and return-memory labels."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.vjepa_long_rollout import (CURRICULUM_SCHEDULE, curriculum_horizon,
    generate_return_episode, object_memory_metrics, train_autoregressive_episode,
    train_curriculum)


def episode(steps=70):
    states = np.zeros((steps + 1, STATE_DIM), np.float32)
    states[:, 0] = .2 + np.arange(steps + 1) * .001
    states[:, 1] = .2
    states[:, 3:6] = 1
    for slot in range(ENTITY_SLOTS):
        start = GLOBAL_DIM+slot*ENTITY_DIM
        states[:, start:start+2] = [.4 + slot * .04, .6]
    grid = np.zeros((16, 16), np.int8)
    grid[[0, -1]] = 1
    grid[:, [0, -1]] = 1
    return {"states": states, "actions": np.zeros((steps, 4), np.float32),
            "events": np.zeros((steps, 4), np.float32), "hit_ids": np.full(steps, -1, np.int64),
            "attack_flags": np.zeros((steps, ENTITY_SLOTS), np.float32), "map": grid, "dt": 1 / 30,
            "initial_rgb": np.full((8, 8, 3), 64, np.uint8)}


class CounterWorld(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.002))
        self.config = SimpleNamespace(architecture="vjepa2_ac", simulation_hz=30., image_size=8, endless=False)
        self.inputs = []
        self.images = []
        self.map_calls = 0

    def encode_map(self, grid):
        self.map_calls += 1
        return self.weight * grid[:, None].float()

    def initialize_from_image(self, image, state, map_features):
        self.images.append(image.detach().clone())
        return state.new_ones((1, 2, 3)) * image.mean() + self.weight * 0

    def transition(self, state, action, map_features, memory):
        self.inputs.append((state.detach().clone(), memory.detach().clone()))
        state = state.clone()
        state[:, 0] = state[:, 0] + self.weight
        return {"state": state, "memory": memory + 1, "features": memory[:, 0, :1].expand(-1, 128),
                "event_logits": self.weight.expand(1, 4)}

    def render(self, *args):
        raise AssertionError("State-only rollout must never render future source/model images")


class VJEPALongRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_every_transition_uses_previous_prediction_and_carries_32_step_boundaries(self):
        data, model = episode(), CounterWorld()
        data["frames"] = np.full((71, 8, 8, 3), 255, np.uint8)
        data["frames"][0] = 64
        data.pop("initial_rgb")
        with patch("turkeyworld.engine.Game.step", side_effect=AssertionError("Engine transitions forbidden")):
            report = train_autoregressive_episode(model, data, horizon=70, chunk_size=32,
                                                  collision_weight=0, capture_trace=True)
        self.assertEqual(len(model.inputs), 70)
        self.assertEqual(len(model.images), 1)
        torch.testing.assert_close(model.images[0], torch.full((1, 3, 8, 8), 64 / 255))
        self.assertEqual(model.map_calls, 3)
        self.assertEqual([item["length"] for item in report["declaration"]["chunk_boundaries"]], [32, 32, 6])
        self.assertAlmostEqual(float(report["final_state"][0, 0]), .2 + 70 * .002, places=5)
        self.assertNotAlmostEqual(float(model.inputs[2][0][0, 0]), float(data["states"][2, 0]))
        for before, after in zip(report["trace"], report["trace"][1:]):
            torch.testing.assert_close(before["output_state"], after["input_state"], rtol=0, atol=0)
            torch.testing.assert_close(before["output_memory"], after["input_memory"], rtol=0, atol=0)
        self.assertFalse(report["declaration"]["future_state_feedback"])
        self.assertEqual(report["declaration"]["teacher_forced_auxiliary_transitions"], 0)
        # Future labels may change losses, never the model's recurrent inputs.
        changed = deepcopy(data)
        changed["states"][1:, 0] = .95
        second = train_autoregressive_episode(CounterWorld(), changed, horizon=70, collision_weight=0)
        torch.testing.assert_close(report["final_state"], second["final_state"], rtol=0, atol=0)

    def test_optimizer_and_carried_state_resume_exactly_without_reinitialization(self):
        data = episode()
        full = CounterWorld()
        full_optimizer = torch.optim.SGD(full.parameters(), lr=.0001, momentum=.9)
        expected = train_autoregressive_episode(full, data, full_optimizer, horizon=70, collision_weight=0)
        first = CounterWorld()
        optimizer = torch.optim.SGD(first.parameters(), lr=.0001, momentum=.9)
        partial = train_autoregressive_episode(first, data, optimizer, horizon=70,
            collision_weight=0, on_chunk=lambda progress: progress["offset"] < 64)
        self.assertTrue(partial["declaration"]["interrupted"])
        resumed = CounterWorld()
        resumed.load_state_dict(first.state_dict())
        resumed_optimizer = torch.optim.SGD(resumed.parameters(), lr=.0001, momentum=.9)
        resumed_optimizer.load_state_dict(optimizer.state_dict())
        with patch("turkeyworld.vjepa_long_rollout.source_initial_image", side_effect=AssertionError("No reinitialization on resume")):
            actual = train_autoregressive_episode(resumed, data, resumed_optimizer, horizon=70,
                                                   collision_weight=0, resume=partial["resume"])
        torch.testing.assert_close(actual["final_state"], expected["final_state"], rtol=0, atol=0)
        torch.testing.assert_close(actual["final_memory"], expected["final_memory"], rtol=0, atol=0)
        torch.testing.assert_close(resumed.weight, full.weight, rtol=0, atol=0)
        self.assertEqual(resumed.images, [])
        self.assertEqual(actual["declaration"]["source_image_initializations_this_call"], 0)
        self.assertEqual(actual["declaration"]["rollout_steps"], 70)

    def test_nine_thousand_transitions_are_real_calls_and_short_sources_are_rejected(self):
        model = CounterWorld()
        data = episode(9000)
        # Keep the diagnostic fixture bounded while the network still drifts.
        data["states"][:, 0] = .2
        report = train_autoregressive_episode(model, data, horizon=9000, collision_weight=0)
        self.assertEqual(len(model.inputs), 9000)
        self.assertEqual(report["declaration"]["rollout_steps"], 9000)
        self.assertEqual(len(model.images), 1)
        self.assertAlmostEqual(float(report["final_memory"][0, 0, 0]), 9000 + 64 / 255, places=2)
        with self.assertRaisesRegex(ValueError, "does not fit"):
            train_autoregressive_episode(CounterWorld(), episode(100), horizon=9000)

    def test_memory_probe_catches_hp_position_loss_and_false_resurrection(self):
        states = episode(3)["states"]
        states[:, 12] = .25
        states[:, 13] = 1
        visible = np.zeros((4, ENTITY_SLOTS), np.float32)
        visible[[0, 3], 0] = 1
        predicted = states.copy()
        predicted[1:, 12] = .60
        predicted[-1, GLOBAL_DIM:GLOBAL_DIM+2] = [.8, .6]
        report = object_memory_metrics(predicted, states, visible)
        self.assertAlmostEqual(report["reappearance_hp_mae"], 35, places=4)
        self.assertEqual(report["reappearance_samples"], 1)
        # One persistent actor cannot swap identity with another slot. Its
        # wrong recalled position remains measurable independently of ID.
        self.assertEqual(report["reappearance_object_id_accuracy"], 1.)
        self.assertAlmostEqual(report["reappearance_position_error_tiles"], 6.4, places=4)
        states[:, 13] = 0
        predicted = states.copy()
        predicted[-1, 13] = 1
        report = object_memory_metrics(predicted, states, visible)
        self.assertEqual(report["false_resurrections"], 1)
        states[-1, 13] = 1  # A reference-authorized endless respawn is legal.
        report = object_memory_metrics(predicted, states, visible)
        self.assertEqual(report["false_resurrections"], 0)
        self.assertEqual(report["reference_respawns"], 1)

    def test_return_corpus_replays_exactly_and_records_real_reappearance_combat(self):
        from turkeyworld.engine import Game
        import json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "return.npz"
            report = generate_return_episode(path, seed=101, steps=224, image_size=32)
            with np.load(path, allow_pickle=False) as data:
                game = Game(endless=True)
                game.set_state(json.loads(str(data["initial_state"])))
                for index, action in enumerate(data["actions"]):
                    events = game.step(action, dt=float(data["dt"]))
                    np.testing.assert_array_equal(game.state_vector(), data["states"][index + 1])
                    self.assertEqual(events["hit_id"], int(data["hit_ids"][index]))
                self.assertTrue(data["initial_rgb"].shape == (32, 32, 3))
                self.assertGreater(report["reappearance_samples"], 0)
                self.assertGreater(report["events"]["hit"], 0)
                self.assertGreater(report["events"]["shot"], 0)

    def test_curriculum_total_and_checkpointed_task_selection_are_deterministic(self):
        self.assertEqual([curriculum_horizon(i, 5) for i in range(5)], [32, 128, 512, 2048, 9000])
        self.assertEqual(sum(CURRICULUM_SCHEDULE), 14888)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            long_dir, returns = root / "long/train", root / "return"
            for folder in (long_dir, returns / "train"):
                folder.mkdir(parents=True)
                data = episode(10)
                data.update(split=np.asarray("train"), endless=np.asarray(False))
                np.savez(folder / "fixture.npz", **data)
            with patch("turkeyworld.vjepa_long_rollout.prepare_return_corpus", return_value={"seeds_disjoint": True}):
                full = train_curriculum(CounterWorld(), long_dir, return_directory=returns,
                                        schedule=(2, 3), chunk_size=2)
                partial = train_curriculum(CounterWorld(), long_dir, return_directory=returns,
                    schedule=(2, 3), chunk_size=2, on_chunk=lambda progress: False)
                resumed = train_curriculum(CounterWorld(), long_dir, return_directory=returns,
                    schedule=(2, 3), chunk_size=2, resume=partial["resume_state"])
            self.assertEqual(full["completed_transitions"], 5)
            self.assertEqual(resumed["completed_transitions"], 5)
            self.assertEqual(full["resume_state"]["plan"], resumed["resume_state"]["plan"])
            self.assertEqual(full["declaration"]["validation_episodes_used_for_training"], 0)

    def test_real_vjepa_backend_backpropagates_across_chunks_without_memory_reset(self):
        from turkeyworld.engine import EVENT_NAMES, Game
        from turkeyworld.models import NeuralWorldModel, WorldModelConfig
        model = NeuralWorldModel(WorldModelConfig(kind="student", image_size=128,
            architecture="vjepa2_ac", endless=True, vjepa_width=24,
            vjepa_encoder_depth=1, vjepa_predictor_depth=1, vjepa_heads=4, vjepa_context_frames=2))
        game = Game(endless=True)
        states, actions, events, hits, attacks = [game.state_vector()], [], [], [], []
        initial = game.render(size=128, overlays=False)
        for index in range(8):
            action = np.array([.1, 0., .1, float(index % 7 == 0)], np.float32)
            event = game.step(action)
            actions.append(action)
            states.append(game.state_vector())
            events.append([float(event[name]) for name in EVENT_NAMES])
            hits.append(event["hit_id"])
            attacking = np.zeros(ENTITY_SLOTS, np.float32)
            attacking[event["attacker_ids"]] = 1
            attacks.append(attacking)
        data = dict(states=np.asarray(states), actions=np.asarray(actions), events=np.asarray(events, np.float32),
                    hit_ids=np.asarray(hits), attack_flags=np.asarray(attacks), map=game.map,
                    initial_rgb=initial, dt=1 / 30)
        before = model.core.predictor.projection.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        result = train_autoregressive_episode(model, data, optimizer, horizon=8, chunk_size=4)
        self.assertEqual(result["declaration"]["optimizer_steps"], 2)
        self.assertTrue(torch.isfinite(result["final_memory"]).all())
        self.assertFalse(torch.equal(before, model.core.predictor.projection.weight))
        self.assertEqual(result["declaration"]["chunk_boundaries"][1]["memory_carry_max_abs_difference"], 0.)


if __name__ == "__main__":
    unittest.main()
