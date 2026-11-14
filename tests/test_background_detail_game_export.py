"""Pure CPU export/argument guards; these tests never open a game window."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

import torch

spec = importlib.util.spec_from_file_location('background_detail_game_export',
    Path(__file__).resolve().parents[1]/'training_runs/diagnostic_tools/launch_background_detail_game_20261114.py')
operator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operator)


class BackgroundDetailGameExportTests(unittest.TestCase):
    def source(self):
        return dict(stage='sr', config=dict(architecture='background_detail_sr', frozen_sr_moe=True),
            model={'frozen.weight': torch.ones(2), 'detail_residual.output.weight': torch.ones(1)},
            training_steps=135, source_training_steps=5308, old_RGBA_head_used=False,
            optimizer={'state': {1: 'old'}}, optimizer_parameter_names=['detail_residual.output.weight'],
            sampler={'draws': 1080}, rng_torch=torch.ones(2), migration={'kind': 'new_detail_branch'},
            gameplay_qualified=False)

    def test_inference_export_preserves_exact_tensors_but_drops_training_state(self):
        source = self.source()
        result = operator.sr_play_export(source, Path('readonly_source.pt'), 'source-sha')
        self.assertEqual(result['config'], source['config'])
        self.assertIs(result['model'], source['model'])
        self.assertIs(result['migration'], source['migration'])
        for key in ('optimizer', 'optimizer_parameter_names', 'sampler', 'rng_torch'):
            self.assertNotIn(key, result); self.assertIn(key, source)
        self.assertEqual(result['training_steps'], 135)
        self.assertEqual(result['source_training_steps'], 5308)
        self.assertFalse(result['latest_play_snapshot']['alpha_used'])
        self.assertFalse(result['latest_play_snapshot']['atlas_used'])

    def test_old_RGBA_atlas_bindings_or_wrong_local_clock_are_rejected(self):
        for mutation in ('architecture', 'oldhead', 'atlas', 'binding', 'step0', 'oldclock', 'provenance'):
            with self.subTest(mutation=mutation):
                saved = self.source()
                if mutation == 'architecture': saved['config']['architecture'] = 'atlas_enemy_edsr'
                elif mutation == 'oldhead': saved['model']['enemy_restoration.output.weight'] = torch.ones(1)
                elif mutation == 'atlas': saved['model']['atlas_frames_rgba'] = torch.ones(1)
                elif mutation == 'binding': saved['sr_moe_binding'] = {}
                elif mutation == 'step0': saved['training_steps'] = 0
                elif mutation == 'oldclock': saved['training_steps'] = 5443
                else: saved['source_training_steps'] = 2308
                with self.assertRaises(ValueError): operator.sr_play_export(saved, 'source.pt', 'sha')

    def test_game_arguments_use_RF3_independent_RSSM_and_existing_E_T_controls(self):
        arguments = operator.game_arguments(Path('training_runs/isolated_game'))
        for option, expected in (('--rf-euler-steps', '3'), ('--display-size', '1024'),
                ('--steps', '450'), ('--comparison-mode', 'autonomous'),
                ('--shared-camera-entities', 'neural'), ('--initial-environment', 'navigation')):
            self.assertEqual(arguments[arguments.index(option)+1], expected)
        self.assertIn('--smooth', arguments)
        self.assertNotIn('--fp16', arguments)
        self.assertNotIn('--two-stage-upscale', arguments)
        self.assertNotIn('--headless', arguments)


if __name__ == '__main__': unittest.main()
