"""Actual generated-RGB metrics, immutable held-out panels and strict gates."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
import json
import math
from types import SimpleNamespace
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None


class PanelData:
    def __init__(self, lengths=(7, 9, 11), size=16):
        self.count, self.files = sum(lengths), len(lengths)
        self.data = {'new': np.zeros((self.count, STATE_DIM), np.float32),
                     'images': np.zeros((self.count, size, size, 3), np.uint8)}
        self.data['new'][:, 3] = 1.
        self.data['new'][:, 0] = np.arange(self.count)/255
        self.data['images'][:] = np.arange(self.count, dtype=np.uint8)[:, None, None, None]
        self.grid = torch.zeros(1, 16, 16, dtype=torch.long)
        self.episode_ranges = []
        start = 0
        for index, length in enumerate(lengths):
            self.episode_ranges.append({'path': f'heldout_{index}.npz', 'start': start, 'stop': start+length})
            start += length

    def batch(self, count, rng, *, indices):
        return {'new': torch.from_numpy(self.data['new'][indices]),
                'images': torch.from_numpy(self.data['images'][indices]).permute(0, 3, 1, 2).float()/255,
                'map': self.grid.expand(count, -1, -1)}


class RenderProbe(torch.nn.Module if torch is not None else object):
    def __init__(self, size=16, bad_final=False, fail=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.child = torch.nn.Dropout(.5)
        self.config = SimpleNamespace(kind='teacher', renderer_mode='direct', image_size=size)
        self.size, self.bad_final, self.fail = size, bad_final, fail
        self.calls = []

    def encode_map(self, grid):
        self.calls.append(('map', tuple(grid.shape)))
        return grid[:, None].float()

    def render(self, state, features, encoded_map):
        self.calls.append(('final_render', state.detach().clone(), features.detach().clone()))
        torch.rand(())  # Evaluation must preserve the caller's training RNG.
        if self.fail:
            raise RuntimeError('deliberate render failure')
        values = torch.zeros_like(state[:, 0]) if self.bad_final else state[:, 0]
        return values[:, None, None, None].expand(-1, 3, self.size, self.size)

    def render_base(self, *args):
        raise AssertionError('The intermediate condition must not be measured')


@unittest.skipIf(torch is None, 'Optional PyTorch dependency is not installed')
class VisualQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_identical_rgb_has_zero_mae_unit_ssim_and_finite_psnr(self):
        from turkeyworld.visual_quality import image_quality_metrics
        rgb = torch.rand(3, 3, 32, 32, generator=torch.Generator().manual_seed(19))
        metrics = image_quality_metrics(rgb, rgb)
        torch.testing.assert_close(metrics['mae'], torch.zeros(3, dtype=torch.float64))
        torch.testing.assert_close(metrics['ssim'], torch.ones(3, dtype=torch.float64), rtol=0, atol=1e-10)
        torch.testing.assert_close(metrics['psnr'], torch.full((3,), 120., dtype=torch.float64))

    def test_constant_images_match_analytic_psnr_mae_and_gaussian_ssim(self):
        from turkeyworld.visual_quality import image_quality_metrics
        target = torch.full((1, 3, 24, 24), .5, dtype=torch.float64)
        generated = torch.full_like(target, .6)
        metrics = image_quality_metrics(generated, target)
        self.assertAlmostEqual(float(metrics['mae'][0]), .1, places=12)
        self.assertAlmostEqual(float(metrics['psnr'][0]), 20., places=10)
        analytic = (2*.5*.6+.01**2)/(.5**2+.6**2+.01**2)
        self.assertAlmostEqual(float(metrics['ssim'][0]), analytic, places=10)

    def test_joint_95_percent_gate_and_overall_means_are_both_required(self):
        from turkeyworld.visual_quality import summarize_visual_quality
        perfect = {'psnr': 60., 'ssim': 1., 'mae': 0., 'finite': True}
        poor = {'psnr': 30., 'ssim': .97, 'mae': .01, 'finite': True}
        result = summarize_visual_quality([dict(perfect) for _ in range(19)]+[poor])
        self.assertTrue(result['passed'])
        self.assertEqual(result['summary']['required_frames'], 19)
        self.assertFalse(summarize_visual_quality([dict(perfect) for _ in range(18)]+[poor, poor])['passed'])
        marginal = {'psnr': 35., 'ssim': .98, 'mae': 2/255, 'finite': True}
        bad = {'psnr': 1., 'ssim': .5, 'mae': .05, 'finite': True}
        result = summarize_visual_quality([dict(marginal) for _ in range(19)]+[bad])
        self.assertEqual(result['summary']['passing_frames'], 19)
        self.assertFalse(result['passed'])

    def test_nonfinite_or_out_of_range_generated_rgb_cannot_pass_a_frame(self):
        from turkeyworld.visual_quality import image_quality_metrics, summarize_visual_quality
        source = torch.zeros(2, 3, 16, 16)
        generated = source.clone()
        generated[0, 0, 0, 0] = float('nan')
        generated[1, 0, 0, 0] = -1.
        metrics = image_quality_metrics(generated, source)
        rows = [{key: value[index].item() for key, value in metrics.items()} for index in range(2)]
        report = summarize_visual_quality(rows)
        self.assertFalse(report['passed'])
        self.assertEqual(report['frame_passed'], [False, False])
        json.dumps(report, allow_nan=False)

    def test_selection_is_fixed_unique_and_balanced_across_unequal_files(self):
        from turkeyworld.visual_quality import select_validation_indices
        data = PanelData()
        first = select_validation_indices(data, count=12, seed=73)
        second = select_validation_indices(data, count=12, seed=73)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(np.unique(first)), 12)
        for episode in data.episode_ranges:
            self.assertEqual(int(((first >= episode['start']) & (first < episode['stop'])).sum()), 4)
        self.assertEqual(len(select_validation_indices(data, count=100)), data.count)
        self.assertEqual(len(select_validation_indices(data, count=2)), 2)

    def test_final_render_is_measured_and_source_rgb_is_not_a_model_input(self):
        from turkeyworld.visual_quality import evaluate_visual_quality, select_validation_indices
        data = PanelData()
        indices = select_validation_indices(data, count=12)
        model = RenderProbe()
        model.train()
        model.child.eval()
        rng_before = torch.random.get_rng_state().clone()
        report = evaluate_visual_quality(model, data, indices, batch_size=5)
        self.assertTrue(report['passed'])
        self.assertEqual(report['render_precision'], 'fp32')
        self.assertTrue(model.training)
        self.assertFalse(model.child.training)
        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
        states = torch.cat([call[1] for call in model.calls if call[0] == 'final_render'])
        torch.testing.assert_close(states, torch.tensor(data.data['new'][indices]))
        for call in model.calls:
            if call[0] == 'final_render':
                self.assertEqual(float(call[2].abs().sum()), 0.)
        bad = evaluate_visual_quality(RenderProbe(bad_final=True), data, indices)
        self.assertFalse(bad['passed'])
        json.dumps(report, allow_nan=False)

    def test_panel_fingerprint_is_stable_and_detects_source_changes(self):
        from turkeyworld.visual_quality import evaluate_visual_quality, select_validation_indices
        data, model = PanelData(), RenderProbe()
        indices = select_validation_indices(data, count=12)
        first = evaluate_visual_quality(model, data, indices)
        second = evaluate_visual_quality(model, data, indices)
        self.assertEqual(first['panel']['fingerprint'], second['panel']['fingerprint'])
        self.assertEqual(first['per_frame'], second['per_frame'])
        data.data['images'][indices[0], 0, 0, 0] += 1
        changed = evaluate_visual_quality(model, data, indices)
        self.assertNotEqual(first['panel']['fingerprint'], changed['panel']['fingerprint'])

    def test_modes_and_rng_restore_even_when_render_raises(self):
        from turkeyworld.visual_quality import evaluate_visual_quality
        data, model = PanelData(), RenderProbe(fail=True)
        model.train()
        model.child.eval()
        before = torch.random.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, 'deliberate'):
            evaluate_visual_quality(model, data, np.array([0, 1]))
        self.assertTrue(model.training)
        self.assertFalse(model.child.training)
        torch.testing.assert_close(torch.random.get_rng_state(), before)

    def test_teacher_gate_rejects_shortened_flow_sampler(self):
        from turkeyworld.visual_quality import evaluate_visual_quality
        model, data = RenderProbe(), PanelData()
        model.config.renderer_mode = 'flow_matching'
        model.flow_renderer = SimpleNamespace(steps=4)
        with self.assertRaisesRegex(ValueError, '12-step Heun'):
            evaluate_visual_quality(model, data, np.array([0]))

    def test_strict_render_precision_restores_flags_including_on_exception(self):
        from turkeyworld.visual_quality import strict_render_precision
        policy = torch.get_float32_matmul_precision()
        matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        try:
            torch.set_float32_matmul_precision('medium')
            torch.backends.cudnn.allow_tf32 = True
            with self.assertRaisesRegex(RuntimeError, 'precision failure'):
                with strict_render_precision():
                    self.assertEqual(torch.get_float32_matmul_precision(), 'highest')
                    self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                    self.assertFalse(torch.backends.cudnn.allow_tf32)
                    raise RuntimeError('precision failure')
            self.assertEqual(torch.get_float32_matmul_precision(), 'medium')
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = matmul
            torch.backends.cudnn.allow_tf32 = cudnn
            torch.set_float32_matmul_precision(policy)

    def test_render_protocol_changes_panel_fingerprint_and_declares_tf32_off(self):
        from turkeyworld.visual_quality import evaluate_visual_quality
        model, data = RenderProbe(), PanelData()
        indices = np.array([0, 1, 2])
        strict = evaluate_visual_quality(model, data, indices, amp=False)
        mixed = evaluate_visual_quality(model, data, indices, amp=True)
        panel = strict['panel']
        self.assertEqual(panel['metric_version'], 'gaussian_ssim_v1')
        self.assertEqual(panel['float32_matmul_precision'], 'highest')
        self.assertFalse(panel['tf32_matmul'])
        self.assertFalse(panel['tf32_cudnn'])
        self.assertNotEqual(panel['fingerprint'], mixed['panel']['fingerprint'])


if __name__ == '__main__':
    unittest.main()
