"""SR display rows receive only the already generated autoregressive future."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from turkeyworld.vjepa_validation import save_autoregressive_preview


class StubUpscaler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.inputs = []
        self.grad_modes = []
        self.output = None

    def forward(self, value):
        self.inputs.append(value.detach().clone())
        self.grad_modes.append(torch.is_grad_enabled())
        self.output = F.interpolate(value, scale_factor=4, mode="nearest")+.02
        return self.output


class PredictionUpscalePreviewTests(unittest.TestCase):
    @staticmethod
    def preview():
        generated = torch.linspace(.03, .55, 4*3*32*32).reshape(4, 3, 32, 32)
        return {"context": np.full((2, 32, 32, 3), 90, np.uint8),
                "source": np.full((4, 32, 32, 3), 240, np.uint8),
                "generated": generated.mul(255).round().byte().permute(0, 2, 3, 1).numpy(),
                "generated_float": generated,
                "metadata": {"step": 10, "context": 2, "horizon": 4, "kind": "free_rollout",
                    "generated_finite": True, "source_held_out": True,
                    "quality_gate_evaluated": False, "caption": "Source future is comparison only"}}

    def test_five_rows_use_prediction_only_and_final_row_is_actual_bilinear512(self):
        preview = self.preview()
        original_source = preview["source"].copy()
        upscaler = StubUpscaler().eval().requires_grad_(False)
        with TemporaryDirectory() as directory, \
             patch("turkeyworld.engine.Game.step", side_effect=AssertionError("engine forbidden")), \
             patch("turkeyworld.engine.Game.render", side_effect=AssertionError("source feedback forbidden")):
            output = Path(directory)/"preview.png"
            metadata = save_autoregressive_preview(preview, output, sr_model=upscaler,
                                                   sr_metadata={"training_steps": 11})
            torch.testing.assert_close(upscaler.inputs[0], preview["generated_float"], rtol=0, atol=0)
            self.assertEqual(upscaler.grad_modes, [False])
            self.assertFalse(torch.allclose(upscaler.inputs[0], torch.full_like(upscaler.inputs[0], 240/255)))
            expected = F.interpolate(upscaler.output, size=(512, 512), mode="bilinear", align_corners=False)
            pixels = expected[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
            with Image.open(output) as canvas:
                self.assertEqual(canvas.size, (2048, 2684))
                final_row = canvas.crop((0, 4*536+16, 512, 4*536+16+512))
                np.testing.assert_array_equal(np.asarray(final_row), pixels)
            self.assertEqual(metadata["row_count"], 5)
            self.assertEqual([row["key"] for row in metadata["rows"]],
                             ["context", "source", "generated", "sr", "bilinear"])
            self.assertEqual([row["native_resolution"] for row in metadata["rows"]],
                             [[32, 32]]*3+[[128, 128], [512, 512]])
            self.assertEqual(metadata["sr_source"], "generated_free_rollout_rgb")
            self.assertFalse(metadata["future_source_feedback"])
            self.assertFalse(metadata["sr_world_feedback"])
            self.assertFalse(metadata["quality_gate_evaluated"])
            self.assertEqual(metadata["upscale_stages"][1]["method"], "bilinear")
            self.assertEqual(metadata["upscale_stages"][1]["input_resolution"], [128, 128])
            self.assertEqual(metadata["upscale_stages"][1]["output_resolution"], [512, 512])
            self.assertEqual(json.loads(output.with_suffix('.json').read_text())["row_count"], 5)
        np.testing.assert_array_equal(preview["source"], original_source)

    def test_no_sr_preserves_legacy_three_row_canvas_and_untrained_sr_is_rejected(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)/"preview.png"
            metadata = save_autoregressive_preview(self.preview(), output)
            with Image.open(output) as image:
                self.assertEqual(image.size, (512, 430))
            self.assertEqual(metadata["row_count"], 3)
            self.assertNotIn('sr_resolution', metadata)
            with self.assertRaisesRegex(ValueError, 'positive training_steps'):
                save_autoregressive_preview(self.preview(), output, sr_model=StubUpscaler(),
                                            sr_metadata={"training_steps": 0})

    def test_experiment_caches_frozen_preview_sr_without_touching_world_or_quality(self):
        from turkeyworld.vjepa_training import VJEPAExperiment
        with TemporaryDirectory() as directory:
            experiment = object.__new__(VJEPAExperiment)
            experiment.root = Path(directory)
            sr_checkpoint = experiment.root/'sr.pt'
            sr_checkpoint.touch()
            experiment.args = SimpleNamespace(context=2, horizon=4, seed=73, preview_every=10,
                quality_psnr=35., quality_ssim=.98, quality_mae=2/255, quality_fraction=.95,
                quality_passes=3, sr_checkpoint=sr_checkpoint)
            experiment.global_step, experiment.fragments_seen, experiment.device = 10, 80, torch.device('cpu')
            experiment.panel, experiment.validation = [(i, 0) for i in range(8)], object()
            experiment.state = {'stage': 'teacher175vq_world', 'teacher_quality': {'last_checked_step': 0},
                'teacher_plateau': {'bad_checks': 2}, 'training_batch': {'optimizer_step': 10}}
            protected = deepcopy(experiment.state)
            experiment.metric = Mock(side_effect=AssertionError('Preview cannot publish quality'))
            world = torch.nn.Linear(1, 1)
            world.config = SimpleNamespace(kind='teacher')
            world_before = {name: value.clone() for name, value in world.state_dict().items()}
            sr = StubUpscaler().train()
            with patch('turkeyworld.models.load_sr_checkpoint', return_value=(sr, {'training_steps': 11})) as load, \
                 patch('turkeyworld.vjepa_validation.evaluate_autoregressive',
                       side_effect=lambda *args, **kwargs: ({'seconds': .1}, self.preview())):
                experiment.prediction_preview_report(world)
                experiment.prediction_preview_report(world)
            self.assertEqual(load.call_count, 1)
            self.assertFalse(sr.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in sr.parameters()))
            self.assertTrue({id(p) for p in sr.parameters()}.isdisjoint(id(p) for p in world.parameters()))
            self.assertEqual((experiment.global_step, experiment.fragments_seen), (10, 80))
            for key, value in protected.items():
                self.assertEqual(experiment.state[key], value)
            for name, value in world.state_dict().items():
                torch.testing.assert_close(value, world_before[name], rtol=0, atol=0)
            self.assertEqual(experiment.state['prediction_preview']['row_count'], 5)

    def test_full_validation_at500_saves_five_rows_without_an_additional_rollout(self):
        from turkeyworld.vjepa_training import VJEPAExperiment
        with TemporaryDirectory() as directory:
            experiment = object.__new__(VJEPAExperiment)
            experiment.root, experiment.device = Path(directory), torch.device('cpu')
            sr_checkpoint = experiment.root/'sr.pt'
            sr_checkpoint.touch()
            experiment.args = SimpleNamespace(context=2, horizon=4, seed=73, preview_every=10,
                quality_psnr=35., quality_ssim=.98, quality_mae=2/255, quality_fraction=.95,
                quality_passes=3, sr_checkpoint=sr_checkpoint)
            experiment.global_step, experiment.fragments_seen = 500, 12345
            experiment.panel, experiment.validation = [(i, 0) for i in range(64)], object()
            experiment.state = {'stage': 'teacher175vq_world', 'teacher_plateau': {'bad_checks': 2}}
            experiment.metric = Mock()
            experiment.prediction_preview_report = Mock(side_effect=AssertionError('No extra AR pass'))
            world = torch.nn.Linear(1, 1)
            world.config = SimpleNamespace(kind='teacher')
            report = {'seconds': .1, 'summary': {'psnr_mean': 36., 'ssim_mean': .99,
                'mae_mean': 1/255, 'pass_fraction': .96, 'frames': 256}}
            with patch('turkeyworld.models.load_sr_checkpoint', return_value=(StubUpscaler(),
                       {'training_steps': 11, 'input_size': 32, 'output_size': 128})) as load, \
                 patch('turkeyworld.vjepa_validation.evaluate_autoregressive',
                       return_value=(report, self.preview())) as evaluate:
                self.assertIs(experiment.preview_if_due(world), report)
            self.assertEqual(evaluate.call_count, 1)
            self.assertEqual(evaluate.call_args.args[2], experiment.panel)
            experiment.prediction_preview_report.assert_not_called()
            self.assertEqual(load.call_count, 1)
            metadata = experiment.state['prediction_preview']
            self.assertEqual((metadata['step'], metadata['row_count']), (500, 5))
            self.assertTrue(metadata['quality_gate_evaluated'])
            self.assertEqual(metadata['scope'], 'held_out_quality_preview')
            self.assertEqual(metadata['upscaled_resolution'], [512, 512])
            self.assertEqual(experiment.state['teacher_quality']['last_checked_step'], 500)
            self.assertEqual(experiment.state['teacher_quality']['frame_count'], 256)
            self.assertEqual(experiment.state['teacher_plateau'], {'bad_checks': 2})
            self.assertEqual((experiment.global_step, experiment.fragments_seen), (500, 12345))


if __name__ == '__main__':
    unittest.main()
