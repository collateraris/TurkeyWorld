"""Behavioral checks for continuous TBPTT, state-only data and checkpoints."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
from contextlib import redirect_stdout
from dataclasses import asdict
import io
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def make_state_episode(path, steps=8):
    states = np.zeros((steps + 1, STATE_DIM), dtype=np.float32)
    states[:, 0] = .2 + np.arange(steps + 1) * .001
    states[:, 1] = .2
    states[:, 3:6] = 1
    states[:, -2] = 1  # The sole enemy is a corpse in this controlled fixture.
    states[:, -1] = np.arange(steps + 1) / 9000
    for slot in range(ENTITY_SLOTS):
        start = GLOBAL_DIM+slot*ENTITY_DIM
        states[:, start:start+2] = [.5, .5]
    actions = np.zeros((steps, 4), dtype=np.float32)
    actions[:, 0] = .5
    grid = np.zeros((16, 16), dtype=np.int8)
    grid[[0, -1], :] = 1
    grid[:, [0, -1]] = 1
    np.savez(path, states=states, actions=actions, events=np.zeros((steps, 4), np.float32),
             map=grid, hit_ids=np.full(steps, -1, np.int64),
             attack_flags=np.zeros((steps, ENTITY_SLOTS), np.float32), dt=np.asarray(1 / 30))


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class StatefulTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_chunks_carry_predicted_state_and_memory_including_partial_tail(self):
        from turkeyworld.long_train import load_episode, run_segment

        class CounterModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.01))
                self.config = SimpleNamespace(architecture="legacy_cnn", image_size=128)
                self.map_calls = self.render_calls = 0

            def encode_map(self, grid):
                self.map_calls += 1
                return self.weight * grid[:, None].float()

            def transition(self, state, action, map_features, memory=None):
                memory = state.new_zeros((len(state), 1)) if memory is None else memory
                result_state = state.clone()
                result_state[:, 0] = state[:, 0] + self.weight
                return {"state": result_state, "memory": memory + 1,
                        "features": memory.expand(-1, 128),
                        "event_logits": self.weight.expand(len(state), 4)}

            def render(self, *args):
                self.render_calls += 1
                raise AssertionError("State-only default must never invoke the renderer")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.npz"
            make_state_episode(path, steps=10)
            episode = load_episode(path)
            model = CounterModel()
            result = run_segment(model, episode, prefix_start=1, burn_in=2,
                                 rollout=5, chunk=2, collision_weight=0,
                                 geometry_every=0, capture_trace=True)
            declaration, trace = result["declaration"], result["trace"]
            self.assertEqual([entry["length"] for entry in declaration["chunk_boundaries"]], [2, 2, 1])
            self.assertEqual(declaration["ground_truth_initializations"], 1)
            self.assertEqual(declaration["detach_boundaries"], 2)
            self.assertEqual(model.map_calls, 4)  # burn-in + three fresh chunk graphs
            self.assertEqual(model.render_calls, 0)
            self.assertEqual(declaration["rendered_supervision_steps"], 0)
            for previous, current in zip(trace, trace[1:]):
                torch.testing.assert_close(previous["output_state"], current["input_state"])
                torch.testing.assert_close(previous["output_memory"], current["input_memory"])
            self.assertAlmostEqual(float(result["final_state"][0, 0]), .201 + 7 * .01, places=5)
            self.assertEqual(float(result["final_memory"][0, 0]), 7.)
            # The sequence does not silently return to the recorded x=.204 at
            # a chunk boundary; its own intentionally drifting state survives.
            self.assertGreater(float(trace[1]["input_state"][0, 0]), .23)

    def test_state_only_training_backpropagates_saves_and_loads_event_config(self):
        from turkeyworld.long_train import main
        from turkeyworld.models import load_world_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            make_state_episode(data / "episode.npz", steps=7)
            output = root / "world.pt"
            with redirect_stdout(io.StringIO()):
                main(["--data", str(data), "--architecture", "legacy_cnn",
                      "--transition-mode", "event_v1", "--chunk", "2", "--rollout", "5",
                      "--burn-in", "1", "--updates", "1", "--curriculum", "",
                      "--render-every", "0", "--geometry-every", "0", "--image-size", "128",
                      "--device", "cpu", "--val-fraction", "0", "--output", str(output)])
            model, checkpoint = load_world_checkpoint(output)
            self.assertEqual(checkpoint["training_steps"], 3)
            self.assertEqual(checkpoint["segment_updates"], 1)
            self.assertEqual(checkpoint["config"]["transition_mode"], "event_v1")
            self.assertEqual(checkpoint["config"]["simulation_hz"], 30.)
            self.assertEqual(checkpoint["config"], asdict(model.config))
            self.assertTrue(torch.isfinite(model.delta_head[-1].weight).all())
            self.assertGreater(float(model.delta_head[-1].weight.detach().abs().sum()), 0)
            declaration = checkpoint["last_tbptt_declaration"]
            self.assertFalse(declaration["rgb_available"])
            self.assertEqual(declaration["rendered_supervision_steps"], 0)
            self.assertEqual(declaration["map_encoder_calls"], 4)
            for boundary in declaration["chunk_boundaries"][1:]:
                self.assertFalse(boundary["state_has_grad_fn"])
                self.assertFalse(boundary["memory_has_grad_fn"])
                self.assertEqual(boundary["state_carry_max_abs_difference"], 0.)
                self.assertEqual(boundary["memory_carry_max_abs_difference"], 0.)

    def test_collision_barrier_supplies_a_pose_gradient_near_wall(self):
        from turkeyworld.long_train import collision_barrier, map_signed_distance
        grid = np.zeros((16, 16), np.int8)
        grid[:, 5] = 1
        sdf = torch.tensor(map_signed_distance(grid))[None, None]
        state = torch.zeros(1, STATE_DIM)
        state[0, :2] = torch.tensor([4.9 / 16, 4.5 / 16])
        state.requires_grad_()
        loss = collision_barrier(state, sdf)
        loss.backward()
        self.assertGreater(float(loss.detach()), 0)
        self.assertGreater(float(state.grad[0, 0]), 0)  # descent moves left, away from the wall
        self.assertTrue(torch.isfinite(state.grad).all())

    def test_event_sampler_keeps_early_positive_event_after_burn_in(self):
        from turkeyworld.long_train import choose_segment, load_episode
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "early_hit.npz"
            make_state_episode(path, steps=10)
            episode = load_episode(path)
            episode.hit_ids[0] = 0
            chosen, prefix, burn, sampling = choose_segment([episode], random.Random(3),
                horizon=4, burn_in=32, event_fraction=1.)
            self.assertIs(chosen, episode)
            self.assertEqual(sampling["mode"], "positive_event")
            self.assertEqual(burn, 0)
            self.assertLessEqual(prefix + burn, sampling["event_index"])
            self.assertLess(sampling["event_index"], prefix + burn + 4)


if __name__ == "__main__":
    unittest.main()
