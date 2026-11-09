"""Quality-driven SR uses clean held-out SSIM, selected weights and exact resume."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_train_sr import arguments, write_pair


def quality_args(root, data, **changes):
    values = dict(architecture='edsr', channels=4, blocks=1, res_scale=.1,
        target_ssim=.9, quality_patience=10, quality_min_delta=.0005,
        lr_min=1e-6, codec_checkpoint=None, codec_fraction=.25, ssim_weight=.2,
        steps=20, validate_every=1, preview_every=1)
    values.update(changes)
    return arguments(root, data, **values)


def fake_evaluation(scores, decoded_score=None):
    def evaluate(model, data, panel, *, step=0, input_kind='clean_source_lr', **kwargs):
        score = scores(step) if callable(scores) else scores[step]
        if input_kind == 'codec_reconstruction' and decoded_score is not None:
            score = decoded_score
        report = {'step': step, 'frames': len(panel), 'input_kind': input_kind,
            'primary_quality_metric': input_kind == 'clean_source_lr',
            'summary': {'ssim_mean': score, 'psnr_mean': 30., 'mae_mean': .03},
            'panel_fingerprint': data.panel_fingerprint(panel), 'source_content_sha256': 'same-source',
            'source_held_out': True, 'world_rollout': False}
        count = min(4, len(panel))
        preview = {'source_lr': np.zeros((count, 32, 32, 3), np.uint8),
            'sr': np.zeros((count, 128, 128, 3), np.uint8),
            'source_hr': np.zeros((count, 128, 128, 3), np.uint8),
            'difference': np.zeros((count, 128, 128, 3), np.uint8),
            'metadata': {'step': step, 'input_kind': input_kind, 'kind': 'single_frame_super_resolution'}}
        return report, preview
    return evaluate


class FrozenCodec(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.5), requires_grad=False)

    def forward(self, image):
        return {'reconstruction': image*self.weight}


class SRQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def data(self, root):
        write_pair(root/'data'/'rgb'/'train')
        write_pair(root/'data'/'rgb'/'val', split='val', identity=2)
        return root/'data'

    def test_training_ssim_matches_canonical_metric_and_has_only_prediction_gradients(self):
        from turkeyworld.train_sr import differentiable_ssim_loss
        from turkeyworld.visual_quality import image_quality_metrics
        target = torch.rand(2, 3, 32, 32, requires_grad=True)
        prediction = (.8*target.detach()+.1).requires_grad_()
        canonical = 1-image_quality_metrics(prediction.detach(), target.detach())['ssim'].mean()
        loss = differentiable_ssim_loss(prediction, target)
        self.assertAlmostEqual(float(loss.detach()), float(canonical), places=5)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))
        self.assertGreater(float(prediction.grad.abs().sum()), 0)
        self.assertIsNone(target.grad)

    def test_target_requires_two_distinct_clean_checks_and_exports_edsr_before_budget(self):
        from turkeyworld.models import EDSRUpscaler, load_sr_checkpoint
        from turkeyworld.train_sr import SRExperiment
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = self.data(root)
            experiment = SRExperiment(quality_args(root/'sr', data))
            evaluate = fake_evaluation({1: .89, 2: .91, 3: .92})
            with patch('turkeyworld.train_sr.evaluate_sr', side_effect=evaluate), contextlib.redirect_stdout(io.StringIO()):
                path = experiment.run()
            saved = torch.load(path, weights_only=False)
            self.assertEqual((experiment.step, saved['training_steps']), (3, 3))
            self.assertTrue(saved['target_met'])
            self.assertEqual(saved['completion_reason'], 'target')
            self.assertFalse(saved['completed_budget'])
            self.assertNotIn('Completed SR budget', saved['quality_status'])
            self.assertEqual(saved['quality_state']['target_streak'], 2)
            state_before = copy.deepcopy(experiment.quality)
            experiment.update_quality(saved['validation'])
            self.assertEqual(state_before, experiment.quality)  # Same-step restart cannot add a pass.
            model, _ = load_sr_checkpoint(path)
            self.assertIsInstance(model, EDSRUpscaler)
            self.assertEqual(model.config['architecture'], 'edsr')
            self.assertEqual(json.loads((root/'sr'/'sr_ready.json').read_text())['completion_reason'], 'target')

    def test_plateau_selects_numeric_best_and_reports_previews_match_selected_weights(self):
        from turkeyworld.train_sr import SRExperiment
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = self.data(root)
            args = quality_args(root/'sr', data, quality_patience=2)
            evaluate = fake_evaluation({1: .8, 2: .8002, 3: .799})
            with patch('turkeyworld.train_sr.evaluate_sr', side_effect=evaluate), contextlib.redirect_stdout(io.StringIO()):
                experiment = SRExperiment(args)
                path = experiment.run()
            selected = torch.load(path, weights_only=False)
            best = torch.load(root/'sr'/'checkpoints'/'sr_best.pt', weights_only=False)
            last = torch.load(root/'sr'/'checkpoints'/'sr_last.pt', weights_only=False)
            self.assertEqual((experiment.step, selected['training_steps'], best['training_steps']), (3, 2, 2))
            self.assertFalse(selected['target_met'])
            self.assertFalse(selected['completed_budget'])
            self.assertEqual(selected['completion_reason'], 'plateau')
            self.assertEqual(selected['validation']['summary']['ssim_mean'], .8002)
            self.assertTrue(any(not torch.equal(value, last['model'][name]) for name, value in best['model'].items()))
            for name, value in best['model'].items():
                torch.testing.assert_close(value, selected['model'][name], rtol=0, atol=0)
            self.assertEqual(json.loads((root/'sr'/'sr_validation.json').read_text())['step'], 2)
            self.assertEqual(json.loads((root/'sr'/'sr_validation_last.json').read_text())['step'], 3)
            self.assertEqual(json.loads((root/'sr'/'preview.json').read_text())['step'], 2)
            # A completed plateau resume re-exports the selected best without more training.
            from turkeyworld.sr_data import SRPairData
            original_batch = SRPairData.batch
            def held_out_only(data, *values, **kwargs):
                self.assertEqual(data.split, 'val')
                return original_batch(data, *values, **kwargs)
            with patch.object(SRPairData, 'batch', held_out_only), \
                    patch('turkeyworld.train_sr.evaluate_sr', side_effect=evaluate), contextlib.redirect_stdout(io.StringIO()):
                resumed = SRExperiment(args)
                resumed.run()
            self.assertEqual(resumed.step, 3)
            self.assertEqual(resumed.quality['bad_checks'], 2)

    def test_scheduler_optimizer_rng_and_quality_history_resume_bitexact(self):
        from turkeyworld.train_sr import SRExperiment
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = self.data(root)
            evaluate = fake_evaluation(lambda step: .8)
            with patch('turkeyworld.train_sr.evaluate_sr', side_effect=evaluate), contextlib.redirect_stdout(io.StringIO()):
                baseline = SRExperiment(quality_args(root/'baseline', data, steps=6))
                baseline.run()
                first = SRExperiment(quality_args(root/'resumed', data, steps=3))
                first.run()
                resumed = SRExperiment(quality_args(root/'resumed', data, steps=6))
                resumed.run()
            original = torch.load(root/'baseline'/'checkpoints'/'sr_last.pt', weights_only=False)
            restored = torch.load(root/'resumed'/'checkpoints'/'sr_last.pt', weights_only=False)
            self.assertLess(restored['optimizer']['param_groups'][0]['lr'], 1e-4)
            self.assertEqual(original['scheduler'], restored['scheduler'])
            self.assertEqual(original['quality_state'], restored['quality_state'])
            self.assertEqual(original['rng_numpy'], restored['rng_numpy'])
            torch.testing.assert_close(original['rng_torch'], restored['rng_torch'], rtol=0, atol=0)
            for name, value in original['model'].items():
                torch.testing.assert_close(value, restored['model'][name], rtol=0, atol=0)
            for index, state in original['optimizer']['state'].items():
                for name, value in state.items():
                    torch.testing.assert_close(value, restored['optimizer']['state'][index][name], rtol=0, atol=0)

    def test_codec_augmentation_is_seeded_frozen_and_keeps_clean_examples(self):
        from turkeyworld.train_sr import SRExperiment
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            experiment = SRExperiment(quality_args(root/'sr', root/'data'))
            codec = FrozenCodec().eval()
            experiment.codec = codec
            clean = torch.full((100, 3, 32, 32), .6)
            before = clean.clone()
            rng = copy.deepcopy(experiment.rng.bit_generator.state)
            augmented, fraction = experiment.training_inputs(clean)
            changed = (augmented != clean).flatten(1).any(1)
            self.assertGreater(int(changed.sum()), 0)
            self.assertLess(int(changed.sum()), 100)
            self.assertAlmostEqual(fraction, float(changed.float().mean()), places=6)
            torch.testing.assert_close(augmented[changed], clean[changed]*.5)
            torch.testing.assert_close(augmented[~changed], clean[~changed])
            torch.testing.assert_close(clean, before, rtol=0, atol=0)
            experiment.rng.bit_generator.state = rng
            repeated, _ = experiment.training_inputs(clean)
            torch.testing.assert_close(augmented, repeated, rtol=0, atol=0)
            self.assertIsNone(codec.weight.grad)

    def test_decoded_validation_is_separate_and_cannot_satisfy_clean_target(self):
        from turkeyworld.train_sr import SRExperiment, evaluate_sr
        from turkeyworld.sr_data import SRPairData
        from turkeyworld.models import SingleFrameUpscaler
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data_root = self.data(root)
            data = SRPairData(data_root/'rgb'/'val')
            model, codec = SingleFrameUpscaler(channels=4, blocks=1).eval(), FrozenCodec().eval()
            panel = data.fixed_panel(4)
            clean, _ = evaluate_sr(model, data, panel)
            decoded, _ = evaluate_sr(model, data, panel, codec=codec, input_kind='codec_reconstruction')
            self.assertEqual(clean['panel_fingerprint'], decoded['panel_fingerprint'])
            self.assertEqual(clean['source_content_sha256'], decoded['source_content_sha256'])
            self.assertTrue(clean['primary_quality_metric'])
            self.assertFalse(decoded['primary_quality_metric'])
            self.assertNotEqual(clean['summary']['ssim_mean'], decoded['summary']['ssim_mean'])
            info = {'codec_fingerprint': 'verified-test', 'accepted_by_user': True, 'quality_gate_passed': False}
            args = quality_args(root/'quality', data_root, steps=2, codec_checkpoint=root/'codec.pt')
            evaluate = fake_evaluation(lambda step: .85, decoded_score=.99)
            with patch('turkeyworld.train_sr.load_sr_codec', return_value=(codec, info)), \
                    patch('turkeyworld.train_sr.evaluate_sr', side_effect=evaluate), contextlib.redirect_stdout(io.StringIO()):
                experiment = SRExperiment(args)
                path = experiment.run()
            selected = torch.load(path, weights_only=False)
            self.assertFalse(selected['target_met'])
            self.assertEqual(selected['validation']['summary']['ssim_mean'], .85)
            self.assertEqual(selected['validation_codec']['summary']['ssim_mean'], .99)
            self.assertEqual(experiment.quality['target_streak'], 0)


if __name__ == '__main__':
    unittest.main()
