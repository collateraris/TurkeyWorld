"""TBPTT accumulation means, safe checkpoint boundaries and unchanged own states."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Optional PyTorch dependency is not installed')
class LongGradientAccumulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        class World(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.2))
                self.config = SimpleNamespace(architecture='vjepa2_ac', image_size=8, simulation_hz=30., endless=False)
                self.images = []

            def encode_map(self, grid):
                return grid.float()[:, None]*self.weight

            def initialize_from_image(self, image, state, map_features):
                self.images.append(image.detach().clone())
                return state.new_ones(1, 2, 3)*image.mean()

            def transition(self, state, action, map_features, memory):
                result = state.clone()
                result[:, 0] += self.weight
                return {'state': result, 'memory': memory+1, 'coefficient': self.weight}

        return World()

    def episode(self, steps):
        states = np.zeros((steps+1, 90), np.float32)
        states[:, 0] = .3+np.arange(steps+1)*.01
        states[:, 1], states[:, 3:6] = .2, 1.
        grid = np.zeros((16, 16), np.int8)
        grid[[0, -1], :] = grid[:, [0, -1]] = 1
        return dict(states=states, actions=np.zeros((steps, 4), np.float32),
            events=np.zeros((steps, 4), np.float32), hit_ids=np.full(steps, -1, np.int64),
            attack_flags=np.zeros((steps, 8), np.float32), map=grid, dt=1/30,
            initial_rgb=np.full((8, 8, 3), 64, np.uint8))

    @staticmethod
    def loss(result, previous, target_previous, target_next, *args, **kwargs):
        zero = result['coefficient']*0
        values = {key: zero for key in ('motion', 'memory', 'events', 'target', 'attack', 'spawn', 'collision')}
        values['state'] = (result['coefficient']-target_next[:, 0]).square().mean()
        return values

    def train(self, model, data, optimizer, **options):
        from turkeyworld.vjepa_long_rollout import train_autoregressive_episode
        with patch('turkeyworld.vjepa_long_rollout.autoregressive_losses', side_effect=self.loss):
            return train_autoregressive_episode(model, data, optimizer, chunk_size=1,
                state_weight=1., memory_weight=0., motion_weight=0., event_weight=0.,
                collision_weight=0., grad_clip=100., **options)

    def test_24_microchunks_and_final_partial_group_use_exact_group_means(self):
        model, data, callbacks = self.model(), self.episode(26), []
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        clipping = torch.nn.utils.clip_grad_norm_
        with patch.object(optimizer, 'step', wraps=optimizer.step) as step, \
             patch.object(optimizer, 'zero_grad', wraps=optimizer.zero_grad) as zero, \
             patch('turkeyworld.vjepa_long_rollout.nn.utils.clip_grad_norm_', wraps=clipping) as clip:
            report = self.train(model, data, optimizer, gradient_accumulation_steps=24,
                                on_chunk=callbacks.append, capture_trace=True)
        expected = torch.tensor(.2)
        for targets in (data['states'][1:25, 0], data['states'][25:27, 0]):
            gradient = 2*(expected-torch.tensor(targets).mean())
            expected = expected-.01*gradient
        torch.testing.assert_close(model.weight, expected, rtol=1e-6, atol=1e-7)
        self.assertEqual((step.call_count, zero.call_count), (2, 2))
        self.assertEqual(clip.call_count, 2)
        self.assertEqual(report['declaration']['accumulation_group_sizes'], [24, 2])
        self.assertEqual(report['declaration']['optimizer_steps'], 2)
        self.assertEqual([entry['offset'] for entry in callbacks if entry['optimizer_updated']], [24, 26])
        self.assertTrue(all(entry['resume'] is None for entry in callbacks if not entry['optimizer_updated']))
        self.assertEqual([entry['accumulation_microstep'] for entry in callbacks], list(range(1, 25))+[1, 2])
        self.assertEqual(callbacks[-1]['accumulation_group_size'], 2)
        self.assertAlmostEqual(callbacks[23]['optimizer_group_loss'],
                               float(np.mean((.2-data['states'][1:25, 0])**2)), places=7)
        self.assertEqual(report['resume']['accumulation_microstep'], 0)
        self.assertEqual(len(model.images), 1)
        for before, after in zip(report['trace'], report['trace'][1:]):
            torch.testing.assert_close(before['output_state'], after['input_state'], rtol=0, atol=0)
            torch.testing.assert_close(before['output_memory'], after['input_memory'], rtol=0, atol=0)
        self.assertTrue(all(not boundary['state_has_grad_fn'] for boundary in report['declaration']['chunk_boundaries']))

    def test_resume_after_flush_is_bit_exact_for_weights_optimizer_and_carried_memory(self):
        data, full, split = self.episode(26), self.model(), self.model()
        full_optimizer = torch.optim.SGD(full.parameters(), lr=.01, momentum=.9)
        expected = self.train(full, data, full_optimizer, gradient_accumulation_steps=24)
        split_optimizer = torch.optim.SGD(split.parameters(), lr=.01, momentum=.9)
        partial = self.train(split, data, split_optimizer, gradient_accumulation_steps=24,
                             on_chunk=lambda progress: progress['offset'] < 24)
        self.assertEqual(partial['resume']['offset'], 24)
        resumed = self.model()
        resumed.load_state_dict(split.state_dict())
        optimizer = torch.optim.SGD(resumed.parameters(), lr=.01, momentum=.9)
        optimizer.load_state_dict(split_optimizer.state_dict())
        actual = self.train(resumed, data, optimizer, gradient_accumulation_steps=24, resume=partial['resume'])
        torch.testing.assert_close(resumed.weight, full.weight, rtol=0, atol=0)
        torch.testing.assert_close(actual['final_state'], expected['final_state'], rtol=0, atol=0)
        torch.testing.assert_close(actual['final_memory'], expected['final_memory'], rtol=0, atol=0)
        self.assertEqual(resumed.images, [])
        torch.testing.assert_close(optimizer.state_dict()['state'][0]['momentum_buffer'],
            full_optimizer.state_dict()['state'][0]['momentum_buffer'], rtol=0, atol=0)

    def test_stop_request_inside_group_waits_for_a_checkpointable_flush(self):
        model = self.model()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        report = self.train(model, self.episode(26), optimizer, gradient_accumulation_steps=24,
                            on_chunk=lambda progress: progress['offset'] != 1)
        self.assertTrue(report['declaration']['interrupted'])
        self.assertEqual(report['declaration']['rollout_steps'], 24)
        self.assertEqual(report['declaration']['optimizer_steps'], 1)
        self.assertEqual(report['resume']['accumulation_microstep'], 0)

    def test_resume_rejects_partial_or_changed_protocol_and_legacy_means_default_one(self):
        model = self.model()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        partial = self.train(model, self.episode(3), optimizer, on_chunk=lambda progress: False)
        for key, value in (('accumulation_microstep', 1), ('gradient_accumulation_steps', 24)):
            modified = {**partial['resume'], key: value}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'protocol|partial'):
                self.train(self.model(), self.episode(3), None, resume=modified)
        legacy = {key: value for key, value in partial['resume'].items()
                  if key not in {'gradient_accumulation_steps', 'accumulation_microstep', 'chunk_size'}}
        self.train(self.model(), self.episode(3), None, resume=legacy)
        with self.assertRaisesRegex(ValueError, 'protocol'):
            self.train(self.model(), self.episode(3), None, resume=legacy, gradient_accumulation_steps=24)
        accumulated = self.train(model, self.episode(26), optimizer, gradient_accumulation_steps=24,
                                 on_chunk=lambda progress: progress['offset'] < 24)
        modified = {**accumulated['resume'], 'chunk_size': 2}
        with self.assertRaisesRegex(ValueError, 'protocol'):
            self.train(self.model(), self.episode(26), None, resume=modified, gradient_accumulation_steps=24)

    def test_nonfinite_accumulated_gradients_cannot_reach_optimizer_step(self):
        class BadGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value.detach()*0+1

            @staticmethod
            def backward(ctx, gradient):
                return gradient*float('nan')

        def loss(result, *args, **kwargs):
            values = self.loss(result, *args, **kwargs)
            values['state'] = BadGradient.apply(result['coefficient'])
            return values

        from turkeyworld.vjepa_long_rollout import train_autoregressive_episode
        model = self.model()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        with patch('turkeyworld.vjepa_long_rollout.autoregressive_losses', side_effect=loss), \
             patch.object(optimizer, 'step', wraps=optimizer.step) as step:
            with self.assertRaisesRegex(RuntimeError, 'non-finite'):
                train_autoregressive_episode(model, self.episode(3), optimizer, chunk_size=1,
                    collision_weight=0., gradient_accumulation_steps=24)
        step.assert_not_called()
        self.assertAlmostEqual(float(model.weight.detach()), .2)

    def test_curriculum_flushes_short_episodes_and_forwards_safe_update_metadata(self):
        from turkeyworld.vjepa_long_rollout import train_curriculum
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            long_dir, returns = root/'long'/'train', root/'return'
            for folder in (long_dir, returns/'train'):
                folder.mkdir(parents=True)
                np.savez(folder/'fixture.npz', **self.episode(6), split=np.asarray('train'), endless=np.asarray(False))
            model, progress = self.model(), []
            optimizer = torch.optim.SGD(model.parameters(), lr=.001)
            with patch('turkeyworld.vjepa_long_rollout.prepare_return_corpus', return_value={'seeds_disjoint': True}), \
                 patch('turkeyworld.vjepa_long_rollout.source_initial_image', return_value=np.full((8, 8, 3), 64, np.uint8)), \
                 patch('turkeyworld.vjepa_long_rollout.autoregressive_losses', side_effect=self.loss):
                result = train_curriculum(model, long_dir, optimizer, return_directory=returns,
                    schedule=(2, 5), chunk_size=1, gradient_accumulation_steps=24, on_chunk=progress.append)
            self.assertEqual(result['completed_transitions'], 7)
            self.assertEqual(result['declaration']['optimizer_steps'], 2)
            updates = [item for item in progress if item['optimizer_updated']]
            self.assertEqual([item['accumulation_group_size'] for item in updates], [2, 5])
            self.assertEqual(updates[-1]['optimizer_steps'], 2)
            self.assertTrue(all(item['resume_state'] is None for item in progress if not item['checkpoint_safe']))
            self.assertEqual(result['resume_state']['gradient_accumulation_steps'], 24)


if __name__ == '__main__':
    unittest.main()
