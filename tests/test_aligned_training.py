"""GT-label alignment is separate from the persistent autoregressive stream."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class AlignedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def model(width=4):
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, width)
                self.calls = []

            def forward(self, video):
                self.calls.append(video.detach().clone())
                return self.linear(video.mean((-1, -2)))[:, :, None]

        class Core(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = Encoder()
                self.predictor = torch.nn.Linear(width, width)
                self.context_frames = 3
                self.initializations = []

            def encode_video(self, video):
                return self.encoder(video)

        class Flow(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.tensor(.01))
                self.calls, self.noises = [], []

            def loss(self, target, base, state, noise=None, target_endpoint=None):
                self.calls.append({"state": state.detach().clone(), "target": target.detach().clone(),
                    "noise": None if noise is None else noise.detach().clone()})
                value = torch.nn.functional.mse_loss(base + self.bias,
                    target if target_endpoint is None else target_endpoint)
                return value, {"flow_velocity": value.detach()}

            def sample(self, base, state, noise=None):
                self.noises.append(noise.detach().clone())
                return base + self.bias + .01 * noise

        class World(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(simulation_hz=30.)
                self.core, self.flow_renderer = Core(), Flow()
                self.pose = torch.nn.Linear(width, STATE_DIM)
                self.feature = torch.nn.Linear(width, 128)
                self.hit = torch.nn.Linear(width, ENTITY_SLOTS+1)
                self.attack = torch.nn.Linear(width, ENTITY_SLOTS)
                self.spawn = torch.nn.Linear(width, ENTITY_SLOTS*256)
                self.state_inputs, self.outputs, self.render_states = [], [], []

            @property
            def vjepa_backend(self):
                return self

            def encode_map(self, grid):
                return grid[:, None].float()

            def initialize_from_latents(self, latents, states, map_features):
                self.core.initializations.append(states.detach().clone())
                return latents[:, -1]

            def initialize_from_image(self, frames, states, map_features):
                return self.initialize_from_latents(self.core.encode_video(frames), states, map_features)

            def transition(self, state, action, map_features, memory):
                self.state_inputs.append(state.detach().clone())
                latent = self.core.predictor(memory) + action[:, None, :1] * .01
                pooled = latent.mean(1)
                result = {"state": state + .01 * self.pose(pooled), "memory": latent,
                          "latent_prediction": latent, "features": self.feature(pooled),
                          "target_logits": self.hit(pooled), "attack_logits": self.attack(pooled),
                          "spawn_logits": self.spawn(pooled).reshape(-1, ENTITY_SLOTS, 256)}
                for key in ("target_logits", "attack_logits", "spawn_logits"):
                    if result[key].requires_grad:
                        result[key].retain_grad()
                self.outputs.append(result)
                return result

            def render_base(self, state, features, map_features):
                self.render_states.append(state.detach().clone())
                return (features[:, :3] + state[:, :1]).sigmoid()[:, :, None, None].expand(-1, -1, 16, 16)

        return World()

    @staticmethod
    def batch():
        frames = torch.rand(1, 6, 3, 16, 16)
        states = torch.zeros(1, 6, STATE_DIM)
        states[:, :, 0] = torch.arange(6) * .05 + .1
        states[:, :, 1] = .2
        states[:, :, 3:6] = 1
        enemies = states[:, :, GLOBAL_DIM:-2].reshape(1, 6, ENTITY_SLOTS, ENTITY_DIM)
        enemies[:, :, :, :2] = .5
        enemies[:, :, :, 4:6] = torch.tensor([.6, 1.])
        enemies[:, :4, 0, 4:6] = 0
        enemies[:, 4:, 0, :2] = torch.tensor([11.5 / 16, 9.5 / 16])
        events = torch.zeros(1, 4, 4)
        events[:, 0, :2] = 1
        events[:, 1, 3] = 1
        attacks = torch.zeros(1, 4, ENTITY_SLOTS)
        attacks[:, 1, 0] = 1
        return {"frames": frames, "context_frames": frames[:, :2], "target_frames": frames[:, 2:],
                "states": states, "previous_states": states[:, 1:-1], "target_states": states[:, 2:],
                "actions": torch.ones(1, 4, 4), "events": events,
                "hit_ids": torch.tensor([[0, -1, -1, -1]]), "attack_flags": attacks,
                "map": torch.zeros(1, 16, 16, dtype=torch.long)}

    def setUp(self):
        torch.manual_seed(18)

    def test_future_gt_changes_cannot_change_main_ar_but_auxiliary_poses_match_labels(self):
        from turkeyworld.aligned_training import aligned_short_rollout_loss
        from turkeyworld.jepa_training_utils import EMATargetEncoder
        model, batch = self.model(), self.batch()
        ema = EMATargetEncoder(model.core.encoder)
        _, metrics, first = aligned_short_rollout_loss(model, ema, batch)
        first_ar = [value.clone() for value in model.state_inputs[:4]]
        for index in range(4):
            torch.testing.assert_close(model.state_inputs[index + 4], batch["previous_states"][:, index])
        self.assertEqual([video.shape[1] for video in model.core.encoder.calls], [2, 5])
        self.assertEqual(metrics["auxiliary_steps"], 4)
        self.assertEqual(metrics["rgb_supervised_steps"], 1)
        changed = {key: value.clone() for key, value in batch.items()}
        changed["frames"][:, 2:] = .9
        changed["target_frames"][:] = .9
        changed["states"][:, 2:, 0] += .3
        changed["previous_states"] = changed["states"][:, 1:-1]
        changed["target_states"] = changed["states"][:, 2:]
        changed["hit_ids"][:] = -1
        changed["attack_flags"][:] = 1
        model.state_inputs.clear()
        _, _, second = aligned_short_rollout_loss(model, ema, changed)
        for before, after in zip(first_ar, model.state_inputs[:4]):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        torch.testing.assert_close(first["ar_state"], second["ar_state"], rtol=0, atol=0)
        torch.testing.assert_close(first["ar_memory"], second["ar_memory"], rtol=0, atol=0)
        self.assertFalse(torch.equal(first["state"], second["state"]))

    def test_rgb_and_flow_use_matching_gt_next_pose_for_every_requested_frame(self):
        from turkeyworld.aligned_training import aligned_short_rollout_loss
        from turkeyworld.jepa_training_utils import EMATargetEncoder
        model, batch = self.model(), self.batch()
        _, metrics, endpoint = aligned_short_rollout_loss(model, EMATargetEncoder(model.core.encoder), batch,
                                                          render_last_only=False)
        self.assertEqual(metrics["rgb_supervised_steps"], 4)
        self.assertEqual(len(model.render_states), 4)
        self.assertEqual(len(model.flow_renderer.calls), 4)
        for index, state in enumerate(model.render_states):
            torch.testing.assert_close(state, batch["target_states"][:, index], rtol=0, atol=0)
            torch.testing.assert_close(model.flow_renderer.calls[index]["state"], state, rtol=0, atol=0)
            torch.testing.assert_close(model.flow_renderer.calls[index]["target"], batch["target_frames"][:, index])
        torch.testing.assert_close(endpoint["state"], batch["target_states"][:, -1], rtol=0, atol=0)
        self.assertFalse(torch.equal(endpoint["state"], endpoint["ar_state"]))

    def test_only_auxiliary_event_heads_receive_recorded_labels_and_spawn_head_learns(self):
        from turkeyworld.aligned_training import aligned_short_rollout_loss
        from turkeyworld.jepa_training_utils import EMATargetEncoder
        model = self.model()
        ema = EMATargetEncoder(model.core.encoder)
        loss, metrics, _ = aligned_short_rollout_loss(model, ema, self.batch())
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        for output in model.outputs[:4]:
            self.assertIsNone(output["target_logits"].grad)
            self.assertIsNone(output["attack_logits"].grad)
            self.assertIsNone(output["spawn_logits"].grad)
        for output in model.outputs[4:]:
            self.assertGreater(float(output["target_logits"].grad.abs().sum()), 0.)
            self.assertGreater(float(output["attack_logits"].grad.abs().sum()), 0.)
        for parameter in (model.hit.weight, model.attack.weight, model.spawn.weight,
                          model.core.encoder.linear.weight, model.core.predictor.weight):
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in ema.parameters()))
        self.assertGreater(metrics["spawn"], 0.)

    def test_reflow_teacher_and_student_share_gt_pose_and_the_same_noise(self):
        from turkeyworld.aligned_training import aligned_short_rollout_loss
        from turkeyworld.jepa_training_utils import EMATargetEncoder
        student, teacher, batch = self.model(), self.model(width=6).eval().requires_grad_(False), self.batch()
        loss, metrics, endpoint = aligned_short_rollout_loss(student, EMATargetEncoder(student.core.encoder),
                                                            batch, teacher=teacher, distill_endpoint=True)
        self.assertIn("reflow", metrics)
        torch.testing.assert_close(student.flow_renderer.calls[-1]["noise"], teacher.flow_renderer.noises[0])
        torch.testing.assert_close(teacher.render_states[-1], endpoint["state"], rtol=0, atol=0)
        for index in range(4):
            torch.testing.assert_close(teacher.state_inputs[index + 4], batch["previous_states"][:, index])
        loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))

    def test_auxiliary_helper_excludes_respawn_teleport_from_motion_and_labels_spawn(self):
        from turkeyworld.aligned_training import aligned_auxiliary_step_loss
        batch = self.batch()
        previous, target = batch["previous_states"][:, 2], batch["target_states"][:, 2]
        predicted = target.clone().requires_grad_()
        spawn = torch.zeros(1, ENTITY_SLOTS, 256, requires_grad=True)
        result = {"state": predicted, "target_logits": torch.zeros(1, ENTITY_SLOTS+1, requires_grad=True),
                  "attack_logits": torch.zeros(1, ENTITY_SLOTS, requires_grad=True), "spawn_logits": spawn}
        losses = aligned_auxiliary_step_loss(result, previous, target, batch["hit_ids"][:, 2],
                                             batch["attack_flags"][:, 2], batch["events"][:, 2])
        self.assertEqual(float(losses["motion"].detach()), 0.)
        sum(losses.values()).backward()
        self.assertGreater(float(spawn.grad[:, 0].abs().sum()), 0.)
        # The one-slot equivalent of an unrelated non-respawning ID is a
        # second query in which that same persistent ID remains dead.
        idle_spawn = torch.zeros_like(spawn, requires_grad=True)
        idle_result = {**result, 'state': previous.detach(), 'spawn_logits': idle_spawn}
        idle_losses = aligned_auxiliary_step_loss(idle_result, previous, previous,
            batch['hit_ids'][:, 2], batch['attack_flags'][:, 2], batch['events'][:, 2])
        idle_losses['spawn'].backward()
        self.assertEqual(float(idle_spawn.grad.abs().sum()), 0.)


if __name__ == "__main__":
    unittest.main()
