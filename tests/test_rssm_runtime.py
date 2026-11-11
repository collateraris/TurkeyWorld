"""RSSM playback observes one image and carries packed FP32 memory autonomously."""
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from turkeyworld.neural_play import NeuralRuntime


class PackedRSSMStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(architecture='dreamer_rssm', transition_mode='event_v1',
            simulation_hz=30., image_size=32, endless=True, navigation_only=True,
            kind='teacher', renderer_mode='direct')
        self.images, self.memories = [], []
        self.corrupt = None

    def encode_map(self, grid):
        return grid[:, None].to(self.anchor.dtype)

    def initial_memory(self, *args):
        raise AssertionError('RSSM must observe the actual initial source image')

    def initialize_from_image(self, image, state, map_features):
        if self.training:
            raise AssertionError('RSSM initialization must run in deterministic eval mode')
        self.images.append(image.detach().clone())
        hidden = state.new_full((len(state), 2048), float(image.float().mean()))
        latent = state.new_zeros((len(state), 32, 32))
        latent[:, :, 3] = 1.
        memory = torch.cat([hidden, latent.flatten(1)], -1)
        return memory.to(self.anchor.dtype)

    def step(self, state, action, memory, map_features):
        if self.training or memory.dtype != torch.float32:
            raise AssertionError('RSSM eval must receive FP32 persistent memory')
        self.memories.append(memory.detach().clone())
        current = state.clone()
        current[:, -1] += 1/9000
        next_memory = memory.clone()
        next_memory[:, :2048] += .125
        if self.corrupt == 'nan':
            next_memory[:, 0] = float('nan')
        if self.corrupt == 'shape':
            next_memory = next_memory[:, :-1]
        return {'state': current, 'memory': next_memory.to(self.anchor.dtype),
                'rgb': state.new_full((len(state), 3, 32, 32), .25), 'events': state.new_zeros(len(state), 4)}


def arguments(**changes):
    values = dict(world=Path('rssm.pt'), device='cpu', threads=1, sr=None, fp16=False,
        seed=0, display_size=32, smooth=True, headless=True, fps=60, warmup=2)
    values.update(changes)
    return Namespace(**values)


class RSSMRuntimeTests(unittest.TestCase):
    def runtime(self, model, args=None):
        self.load = patch('turkeyworld.neural_play.load_world_checkpoint', return_value=(model, {'training_steps': 7}))
        self.counts = patch('turkeyworld.neural_play.parameter_counts', return_value={'world_total': 1})
        self.render = patch('turkeyworld.neural_play.Game.render', autospec=True,
                            return_value=np.full((32, 32, 3), 64, np.uint8))
        self.step = patch('turkeyworld.neural_play.Game.step', side_effect=AssertionError('No Source future'))
        self.correct = patch('turkeyworld.neural_play.Game.set_state_vector', side_effect=AssertionError('No Source correction'))
        for item in (self.load, self.counts, self.step, self.correct):
            item.start()
            self.addCleanup(item.stop)
        self.render_mock = self.render.start()
        self.addCleanup(self.render.stop)
        return NeuralRuntime(args or arguments())

    def test_exactly_one_image_and_cached_fp32_memory_survives_warmup_without_core_tokens(self):
        model = PackedRSSMStub().train()
        runtime = self.runtime(model)
        self.assertFalse(model.training)
        self.assertFalse(hasattr(model, 'core'))
        self.assertEqual(runtime.memory.shape, (1, 3072))
        self.assertEqual(runtime.memory.dtype, torch.float32)
        expected = runtime.initial_memory.clone()
        self.assertEqual(len(model.images), 1)
        torch.testing.assert_close(model.images[0], torch.full((1, 3, 32, 32), 64/255))
        runtime.memory.add_(1)
        self.assertTrue(torch.equal(runtime.initial_memory, expected))
        runtime.warmup()
        self.assertTrue(torch.equal(runtime.memory, expected))
        self.assertNotEqual(runtime.memory.data_ptr(), runtime.initial_memory.data_ptr())
        runtime.step([1, 0, 0, 0])
        report = runtime.report(.01)
        self.assertEqual(self.render_mock.call_count, 1)
        self.assertEqual(report['engine_renders_for_initialization'], 1)
        self.assertEqual(report['engine_renders_during_rollout'], 0)
        self.assertEqual(report['engine_steps_during_rollout'], 0)
        self.assertEqual(report['rssm']['packed_memory_shape'], [1, 3072])
        self.assertEqual(report['rssm']['memory_precision'], 'torch.float32')
        self.assertEqual(report['rssm']['latent_selection'], 'deterministic_argmax_in_eval')
        self.assertNotIn('ema', report['rssm'])

    def test_half_network_keeps_packed_memory_fp32_and_reset_repeats_rollout(self):
        # Direct class construction simulates half network weights on CPU;
        # the public CLI correctly reserves --fp16 for CUDA.
        model = PackedRSSMStub()
        runtime = self.runtime(model, arguments(fp16=True))
        self.assertEqual(model.anchor.dtype, torch.float16)
        self.assertEqual(runtime.memory.dtype, torch.float32)
        runtime.step([0, 0, 0, 0])
        first = runtime.memory.clone()
        runtime.reset_from_cached_initial_state()
        torch.random.default_generator.manual_seed(999)
        runtime.step([0, 0, 0, 0])
        self.assertTrue(torch.equal(runtime.memory, first))
        self.assertTrue(all(value.dtype == torch.float32 for value in model.memories))
        self.assertEqual(self.render_mock.call_count, 1)

    def test_corrupt_memory_shape_or_nan_fails_before_committing_state(self):
        model = PackedRSSMStub()
        runtime = self.runtime(model)
        original = runtime.state.clone()
        for corruption in ('shape', 'nan'):
            model.corrupt = corruption
            with self.subTest(corruption=corruption), self.assertRaises(FloatingPointError):
                runtime.step([0, 0, 0, 0])
            self.assertTrue(torch.equal(runtime.state, original))

    def test_nonpacked_initial_memory_is_rejected(self):
        model = PackedRSSMStub()
        with patch.object(model, 'initialize_from_image', return_value=torch.zeros(1, 32, 32)):
            with self.assertRaisesRegex(ValueError, 'packed tensor'):
                self.runtime(model)


if __name__ == '__main__':
    unittest.main()
