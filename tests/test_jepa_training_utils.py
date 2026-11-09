"""Stop-gradient EMA targets and episode-aligned image/action supervision."""
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class JEPATrainingUtilsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_ema_exact_initial_copy_update_and_frozen_eval_forward(self):
        from turkeyworld.jepa_training_utils import EMATargetEncoder
        online = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.5))
        online.register_buffer("counter", torch.zeros((), dtype=torch.int64))
        ema = EMATargetEncoder(online, momentum=.5)
        original = online[0].weight.detach().clone()
        with torch.no_grad():
            online[0].weight.add_(2)
            online.counter.add_(7)
        ema.update(online)
        torch.testing.assert_close(ema.encoder[0].weight, original + 1)
        self.assertEqual(ema.steps, 1)
        self.assertEqual(int(ema.encoder.counter), 7)
        ema.train()
        self.assertFalse(any(module.training for module in ema.modules()))
        inputs = torch.rand(2, 3, requires_grad=True)
        self.assertFalse(ema(inputs).requires_grad)
        self.assertFalse(any(parameter.requires_grad for parameter in ema.parameters()))
        self.assertIsNone(inputs.grad)
        self.assertEqual(ema.parameter_count, sum(parameter.numel() for parameter in online.parameters()))

    def test_ema_checkpoint_restores_configuration_weights_and_step_count(self):
        from turkeyworld.jepa_training_utils import EMATargetEncoder
        online = torch.nn.Linear(3, 4)
        first = EMATargetEncoder(online)
        first.update(online)
        stream = io.BytesIO()
        torch.save(first.state_dict(), stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=True)
        restored = EMATargetEncoder(online)
        restored.load_state_dict(saved)
        self.assertEqual(restored.steps, 1)
        torch.testing.assert_close(first.encoder.weight, restored.encoder.weight, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "momentum/configuration mismatch"):
            EMATargetEncoder(online, momentum=.9).load_state_dict(saved)

    def test_latent_loss_detaches_targets_and_normalizes_each_token(self):
        from turkeyworld.jepa_training_utils import latent_prediction_loss
        torch.manual_seed(17)
        prediction = torch.randn(2, 4, 3, 8, requires_grad=True)
        target = torch.randn_like(prediction, requires_grad=True)
        loss, metrics = latent_prediction_loss(prediction, target)
        self.assertGreater(float(loss.detach()), 0)
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertGreater(float(prediction.grad.abs().sum()), 0)
        self.assertTrue(all(bool(torch.isfinite(value)) for value in metrics.values()))
        identity, _ = latent_prediction_loss(target.detach() * 2 + 4, target.detach())
        self.assertLess(float(identity), 1e-5)

    def test_variance_covariance_is_finite_for_one_or_many_tokens(self):
        from turkeyworld.jepa_training_utils import variance_covariance_loss
        for shape in ((1, 1, 1, 8), (2, 3, 4, 8), (2, 1)):
            with self.subTest(shape=shape):
                features = torch.randn(shape, requires_grad=True)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    loss, metrics = variance_covariance_loss(features)
                self.assertEqual(loss.dtype, torch.float32)
                self.assertTrue(bool(torch.isfinite(loss)))
                self.assertGreaterEqual(float(loss.detach()), 0.)
                loss.backward()
                self.assertTrue(bool(torch.isfinite(features.grad).all()))
                self.assertTrue(bool(torch.isfinite(metrics["covariance"])))
        collapse, _ = variance_covariance_loss(torch.ones(1, 1, 1, 8))
        self.assertGreater(float(collapse), .9)

    @staticmethod
    def write_episode(path, episode_id=0, transitions=10, split="train"):
        values = np.arange(transitions + 1, dtype=np.float32)
        frames = np.broadcast_to((values + episode_id * 40).astype(np.uint8)[:, None, None, None],
                                 (transitions + 1, 16, 16, 3)).copy()
        states = np.zeros((transitions + 1, 90), dtype=np.float32)
        states[:, 0], states[:, 1] = values, episode_id
        actions = np.broadcast_to(values[:-1, None], (transitions, 4)).copy()
        np.savez_compressed(path, frames=frames, states=states, actions=actions,
                            events=np.zeros((transitions, 4), np.float32),
                            hit_ids=np.full(transitions, -1, np.int64),
                            attack_flags=np.zeros((transitions, 8), np.float32),
                            map=np.full((16, 16), episode_id, np.int8),
                            dt=np.asarray(1 / 30), split=np.asarray(split))

    def test_sequence_batch_never_crosses_an_episode_and_aligns_future_actions(self):
        from turkeyworld.jepa_training_utils import SequenceData
        with tempfile.TemporaryDirectory() as folder:
            train = Path(folder) / "train"
            train.mkdir()
            self.write_episode(train / "a.npz", 1)
            self.write_episode(train / "b.npz", 2)
            data = SequenceData(train, device="cpu")
            self.assertEqual(data.files, 2)
            self.assertEqual(data.count, 20)
            self.assertEqual(data.window_count(), 12)
            first = data.batch(8, context=2, horizon=4, rng=np.random.default_rng(31))
            second = data.batch(8, context=2, horizon=4, rng=np.random.default_rng(31))
            torch.testing.assert_close(first["frames"], second["frames"], rtol=0, atol=0)
            self.assertEqual(tuple(first["frames"].shape), (8, 6, 3, 16, 16))
            self.assertEqual(tuple(first["context_frames"].shape), (8, 2, 3, 16, 16))
            self.assertEqual(tuple(first["target_frames"].shape), (8, 4, 3, 16, 16))
            self.assertEqual(tuple(first["actions"].shape), (8, 4, 4))
            torch.testing.assert_close(first["actions"][:, :, 0], first["previous_states"][:, :, 0])
            torch.testing.assert_close(first["target_states"][:, :, 0], first["previous_states"][:, :, 0] + 1)
            self.assertTrue(bool((first["states"][:, :, 1] == first["states"][:, :1, 1]).all()))
            frame_values = first["frames"][:, :, 0, 0, 0] * 255
            torch.testing.assert_close(frame_values, first["states"][:, :, 0] + first["states"][:, :, 1] * 40)
            self.assertTrue(all(episode["frames"].dtype == np.uint8 for episode in data.episodes))

    def test_fixed_validation_panel_is_balanced_unique_and_replayed_exactly(self):
        from turkeyworld.jepa_training_utils import SequenceData
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for index in range(4):
                self.write_episode(root / f"{index}.npz", episode_id=index, split="val")
            data = SequenceData(root, split="val")
            panel = data.fixed_panel(count=8, context=2, horizon=4, seed=73)
            self.assertEqual(panel, data.fixed_panel(count=8, context=2, horizon=4, seed=73))
            self.assertEqual(len(set(panel)), 8)
            self.assertEqual([sum(episode == index for episode, _ in panel) for index in range(4)], [2] * 4)
            first = data.batch(len(panel), indices=panel, rng=np.random.default_rng(1))
            second = data.batch(len(panel), indices=panel, rng=np.random.default_rng(99))
            torch.testing.assert_close(first["frames"], second["frames"], rtol=0, atol=0)
            self.assertEqual(first["episode_indices"].tolist(), [item[0] for item in panel])
            self.assertEqual(first["start_indices"].tolist(), [item[1] for item in panel])
            with self.assertRaisesRegex(ValueError, "crosses an episode"):
                data.batch(1, indices=[(0, 9)])
            with self.assertRaisesRegex(ValueError, "integer"):
                data.batch(1, indices=[(.5, 0.)])

    def test_split_is_explicit_and_memory_or_short_episode_limits_are_enforced(self):
        from turkeyworld.jepa_training_utils import SequenceData
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_episode(root / "a.npz", transitions=3, split="train")
            data = SequenceData(root, split="train")
            with self.assertRaisesRegex(ValueError, "complete context"):
                data.batch(1)
            with self.assertRaisesRegex(ValueError, "max_bytes"):
                SequenceData(root, max_bytes=1)
            with self.assertRaisesRegex(ValueError, "mix dataset splits"):
                SequenceData(root, split="val")
            self.write_episode(root / "b.npz", split="val")
            with self.assertRaisesRegex(ValueError, "mix dataset splits"):
                SequenceData(root)


if __name__ == "__main__":
    unittest.main()
