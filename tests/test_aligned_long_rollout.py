"""Independent long-rollout auxiliary memory and checkpoint continuation."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def episode(steps=9):
    states = np.zeros((steps + 1, STATE_DIM), np.float32)
    states[:, 0] = .2 + np.arange(steps + 1) * .001
    states[:, 1], states[:, 3:6] = .2, 1
    enemies = states[:, GLOBAL_DIM:-2].reshape(steps + 1, ENTITY_SLOTS, ENTITY_DIM)
    enemies[:, :, :2], enemies[:, :, 4:6] = [.5, .6], [.6, 1]
    enemies[:4, 0, 4:6] = 0
    grid = np.zeros((16, 16), np.int8)
    grid[[0, -1], :] = grid[:, [0, -1]] = 1
    events = np.zeros((steps, 4), np.float32)
    events[1, :2], events[2, 3] = 1, 1
    hits = np.full(steps, -1, np.int64)
    hits[1] = 0
    attacks = np.zeros((steps, ENTITY_SLOTS), np.float32)
    attacks[2, 0] = 1
    return {"states": states, "actions": np.zeros((steps, 4), np.float32), "events": events,
            "hit_ids": hits, "attack_flags": attacks, "map": grid, "dt": 1 / 30,
            "initial_rgb": np.full((8, 8, 3), 64, np.uint8),
            "frames": np.full((steps + 1, 8, 8, 3), 64, np.uint8)}


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class AlignedLongRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def model():
        class World(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.motion = torch.nn.Parameter(torch.tensor(.003))
                self.hit = torch.nn.Linear(2, ENTITY_SLOTS+1)
                self.attack = torch.nn.Linear(2, ENTITY_SLOTS)
                self.spawn = torch.nn.Linear(2, ENTITY_SLOTS*256)
                self.event = torch.nn.Linear(2, 4)
                self.config = SimpleNamespace(architecture="vjepa2_ac", simulation_hz=30., image_size=8, endless=True)
                self.inputs, self.outputs, self.render_states, self.initial_images = [], [], [], []

            def encode_map(self, grid):
                return self.motion * grid[:, None].float()

            def initialize_from_image(self, image, state, map_features):
                self.initial_images.append(image.detach().clone())
                return state.new_ones(1, 1, 2) * image.mean()

            def transition(self, state, action, map_features, memory):
                self.inputs.append((state.detach().clone(), memory.detach().clone()))
                value = torch.stack([state[:, 0], memory.flatten(1).mean(1)], -1)
                result_state = state.clone()
                result_state[:, 0] += self.motion
                result = {"state": result_state, "memory": memory + 1 + state[:, :1, None] * .1,
                          "features": value[:, :1].expand(-1, 128),
                          "target_logits": self.hit(value), "attack_logits": self.attack(value),
                          "spawn_logits": self.spawn(value).reshape(-1, ENTITY_SLOTS, 256), "event_logits": self.event(value)}
                for name in ("target_logits", "attack_logits", "spawn_logits", "event_logits"):
                    if result[name].requires_grad:
                        result[name].retain_grad()
                self.outputs.append(result)
                return result

            def render(self, state, features, map_features):
                self.render_states.append(state.detach().clone())
                return (state[:, :1] + features[:, :1] * 0)[:, :, None, None].expand(-1, 3, 8, 8)

        return World()

    def setUp(self):
        torch.manual_seed(61)

    def test_main_state_is_unchanged_by_future_gt_while_auxiliary_inputs_and_rendering_are_aligned(self):
        from turkeyworld.vjepa_long_rollout import train_autoregressive_episode
        data, first_model = episode(), self.model()
        first = train_autoregressive_episode(first_model, data, horizon=9, chunk_size=3,
            collision_weight=0, aligned_supervision=True, render_every=1)
        changed = deepcopy(data)
        changed["states"][1:, 0] = .8
        second_model = self.model()
        second_model.load_state_dict(first_model.state_dict())
        second = train_autoregressive_episode(second_model, changed, horizon=9, chunk_size=3,
            collision_weight=0, aligned_supervision=True)
        torch.testing.assert_close(first["final_state"], second["final_state"], rtol=0, atol=0)
        torch.testing.assert_close(first["final_memory"], second["final_memory"], rtol=0, atol=0)
        for index in range(9):
            torch.testing.assert_close(first_model.inputs[index * 2][0], second_model.inputs[index * 2][0], rtol=0, atol=0)
            torch.testing.assert_close(first_model.inputs[index * 2 + 1][0], torch.tensor(data["states"][index:index + 1]))
            torch.testing.assert_close(first_model.render_states[index], torch.tensor(data["states"][index + 1:index + 2]))
        self.assertFalse(torch.equal(first_model.inputs[5][1], second_model.inputs[5][1]))
        self.assertEqual(first["declaration"]["teacher_forced_auxiliary_transitions"], 9)
        self.assertTrue(first["declaration"]["local_labels_at_reference_pose_only"])

    def test_recorded_classification_labels_backpropagate_only_through_auxiliary_queries(self):
        from turkeyworld.vjepa_long_rollout import train_autoregressive_episode
        model = self.model()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
        train_autoregressive_episode(model, episode(), optimizer, horizon=9, chunk_size=3,
                                     collision_weight=0, aligned_supervision=True)
        for output in model.outputs[::2]:
            for name in ("target_logits", "attack_logits", "spawn_logits", "event_logits"):
                self.assertIsNone(output[name].grad)
        for output in model.outputs[1::2]:
            self.assertGreater(float(output["target_logits"].grad.abs().sum()), 0.)
            self.assertGreater(float(output["attack_logits"].grad.abs().sum()), 0.)
        self.assertTrue(any(float(output["spawn_logits"].grad.abs().sum()) > 0 for output in model.outputs[1::2]))

    def test_aligned_chunk_resume_is_bit_exact_and_requires_its_protocol_and_auxiliary_memory(self):
        from turkeyworld.vjepa_long_rollout import train_autoregressive_episode
        data, full_model = episode(), self.model()
        split_model = deepcopy(full_model)
        full_optimizer = torch.optim.SGD(full_model.parameters(), lr=1e-5, momentum=.9)
        full = train_autoregressive_episode(full_model, data, full_optimizer, horizon=9, chunk_size=3,
                                            collision_weight=0, aligned_supervision=True)
        split_optimizer = torch.optim.SGD(split_model.parameters(), lr=1e-5, momentum=.9)
        partial = train_autoregressive_episode(split_model, data, split_optimizer, horizon=9, chunk_size=3,
            collision_weight=0, aligned_supervision=True, on_chunk=lambda value: value["offset"] < 6)
        self.assertTrue(partial["resume"]["aligned_supervision"])
        self.assertIn("auxiliary_memory", partial["resume"])
        resumed = self.model()
        resumed.load_state_dict(split_model.state_dict())
        resumed_optimizer = torch.optim.SGD(resumed.parameters(), lr=1e-5, momentum=.9)
        resumed_optimizer.load_state_dict(split_optimizer.state_dict())
        result = train_autoregressive_episode(resumed, data, resumed_optimizer, horizon=9, chunk_size=3,
            collision_weight=0, aligned_supervision=True, resume=partial["resume"])
        torch.testing.assert_close(result["final_state"], full["final_state"], rtol=0, atol=0)
        torch.testing.assert_close(result["final_memory"], full["final_memory"], rtol=0, atol=0)
        torch.testing.assert_close(result["resume"]["auxiliary_memory"], full["resume"]["auxiliary_memory"], rtol=0, atol=0)
        for name, value in resumed.state_dict().items():
            torch.testing.assert_close(value, full_model.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(resumed.initial_images, [])
        with self.assertRaisesRegex(ValueError, "protocol"):
            train_autoregressive_episode(resumed, data, horizon=9, resume=partial["resume"], collision_weight=0)
        missing = {key: value for key, value in partial["resume"].items() if key != "auxiliary_memory"}
        with self.assertRaisesRegex(ValueError, "auxiliary memory"):
            train_autoregressive_episode(resumed, data, horizon=9, resume=missing,
                                         aligned_supervision=True, collision_weight=0)


if __name__ == "__main__":
    unittest.main()
