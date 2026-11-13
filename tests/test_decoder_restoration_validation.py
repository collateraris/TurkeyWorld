"""Actual restoration-profile validation, cohort means and saved preview scale."""
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from test_enemy_decoder_validation import GuidedPairs
from test_semantic_sr_validation import TinySR
from turkeyworld.enemy_decoder_sr import EnemyDecoderEDSRUpscaler
from turkeyworld.decoder_restoration_objectives import restoration_raw_metrics, LOSS_WEIGHTS
from turkeyworld.semantic_sr_validation import (evaluate_semantic_sr, preview_semantic_sr,
                                               save_semantic_sr_preview)


class DecoderRestorationValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def model(self):
        return EnemyDecoderEDSRUpscaler(channels=8, blocks=1, semantic_channels=4,
            semantic_blocks=1, enemy_decoder_channels=4, enemy_decoder_stable=True,
            enemy_decoder_visual_first=True, enemy_decoder_restoration=True,
            restoration_blend_scale=.3)

    def test_three_raw_modes_use_actual_conditional_tail_counts_and_restore_all_state(self):
        torch.manual_seed(173)
        model, data, manual = self.model(), GuidedPairs(), GuidedPairs()
        # Positive denominators are3/1/2 across batches; make the one-positive
        # middle batch different so uniform batch averaging is observably wrong.
        for item in (data, manual):
            with torch.no_grad(): item.hr[3].add_(.15)
        expected = []
        model.eval()
        with torch.inference_mode():
            for first in range(0, 8, 3):
                ids = list(range(first, min(first + 3, 8)))
                batch = manual.batch(len(ids), indices=ids)
                output = model.forward_with_aux(batch['lr_npr'].detach(), batch['hypothesis'].detach())
                expected.append(restoration_raw_metrics(output, batch))
        model.train(); model.enemy_decoder_visual.eval()
        modes = [module.training for module in model.modules()]
        sampler = deepcopy(data.sampler_state())
        rng = torch.get_rng_state().clone(); py = random.getstate(); nprng = np.random.get_state()
        refs = {'sr38100': TinySR(.04), 'sr40100': TinySR(.02)}
        # Constructing references consumes no RNG, and evaluation must not either.
        report, preview = evaluate_semantic_sr(model, data, list(range(8)), step=100,
            batch_size=3, baselines=refs)
        self.assertEqual(report['kind'], 'decoder_restoration_super_resolution_v3')
        objective = report['validation_objective']
        self.assertEqual(objective['protocol'], 'decoder_restoration_sr_v3')
        self.assertEqual(set(objective['components']), set(LOSS_WEIGHTS))
        self.assertEqual((objective['frames'], objective['batch_count']), (8, 3))
        self.assertEqual(set(report['restoration_raw_modes']), {'with_hint', 'no_hint', 'wrong_hint'})
        for raw in report['restoration_raw_modes'].values():
            self.assertEqual((raw['frames'], raw['visible_frames'], raw['empty_frames']), (8, 6, 2))
            self.assertEqual(raw['metric_counts'], {'raw_foreground_mae': 6,
                'raw_foreground_edge_mae': 6, 'raw_empty_mae': 2, 'raw_background_mae': 8})
            self.assertTrue(raw['before_gain_and_route_or_visible_mask_gates'])
            self.assertFalse(raw['source_future_as_input'])
        actual = report['restoration_raw']
        for name in expected[0]['metric_counts']:
            count = sum(row['metric_counts'][name] for row in expected)
            value = sum(row[name] * row['metric_counts'][name] for row in expected) / count
            self.assertAlmostEqual(actual[name], value, places=8, msg=name)
        incorrect = sum(row['raw_foreground_mae'] for row in expected) / 3
        self.assertGreater(abs(actual['raw_foreground_mae'] - incorrect), 1e-4)
        self.assertEqual(report['enemy_decoder']['restoration']['blend_scale'], .3)
        self.assertFalse(report['enemy_decoder']['restoration']['raw_metrics_are_final_image_quality'])
        self.assertEqual(sampler, data.sampler_state())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(py, random.getstate())
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(nprng, np.random.get_state())))
        self.assertEqual(modes, [module.training for module in model.modules()])
        self.assertTrue(all(parameter.grad is None for net in (model, *refs.values()) for parameter in net.parameters()))
        for value in (data.hr, data.hr_mask, data.npr): self.assertIsNone(value.grad)
        self.assertEqual(preview['metadata']['row_count'], 6)
        self.assertEqual(preview['metadata']['restoration_blend_scale'], .3)
        json.dumps(report, allow_nan=False)

    def test_cached_baselines_and_six_row_light_preview_preserve_saved_blend(self):
        model, data = self.model(), GuidedPairs()
        refs = {'sr38100': TinySR(.04), 'sr40100': TinySR(.02)}
        baseline, _ = evaluate_semantic_sr(model, data, list(range(8)), step=0,
            batch_size=3, baselines=refs)
        calls = {name: len(net.calls) for name, net in refs.items()}
        current, _ = evaluate_semantic_sr(model, data, list(range(8)), step=100,
            batch_size=3, baseline_report=baseline)
        for name, net in refs.items():
            self.assertEqual(len(net.calls), calls[name])
            self.assertEqual(current['domains']['npr']['models'][name], baseline['domains']['npr']['models'][name])
        sampler = deepcopy(data.sampler_state()); rng = torch.get_rng_state().clone()
        model.train(); model.enemy_decoder_rgba.eval()
        modes = [module.training for module in model.modules()]
        light = preview_semantic_sr(model, data, list(range(8)), step=15, baseline_report=baseline)
        self.assertEqual(light['metadata']['restoration_blend_scale'], .3)
        self.assertEqual(light['metadata']['restoration_phase'], 'blending')
        self.assertEqual(light['metadata']['comparison_modes'], ['with_hint', 'no_hint'])
        self.assertEqual(light['metadata']['row_count'], 6)
        self.assertEqual(light['metadata']['reference_validation_step'], 0)
        self.assertEqual(sampler, data.sampler_state()); self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(modes, [module.training for module in model.modules()])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'preview.png'
            metadata = save_semantic_sr_preview(light, path, every=15)
            with Image.open(path) as image: self.assertEqual(image.size, (1024, 2552))
            self.assertEqual(metadata['restoration_blend_scale'], .3)
            self.assertEqual(metadata['row_count'], 6)
            json.dumps(json.loads(path.with_suffix('.json').read_text(encoding='utf-8')), allow_nan=False)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


if __name__ == '__main__': unittest.main()
