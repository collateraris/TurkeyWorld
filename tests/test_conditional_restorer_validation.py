"""Actual conditional restorer comparisons on shared held-out RF3 codes."""
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_conditional_restorer_objectives import network
from test_decoder_rf_adaptation_validation import HeldOut
from turkeyworld.conditional_restorer_objectives import conditional_restorer_loss
from turkeyworld.conditional_restorer_validation import (
    PROTOCOL, KIND, evaluate_conditional_restorer, save_conditional_restorer_preview)
from turkeyworld.renderer_student import _hash_tensors


class ConditionalRestorerValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = network()
        cls.reference = deepcopy(cls.model.flow_renderer.codec).eval().requires_grad_(False)
        cls.panel, cls.geometry_panel = [0, 1, 2, 3, 4], [0, 2, 3]

    def test_actual_zero_append_four_arms_geometry_and_truthful_frozen_decoder_scope(self):
        net, reference, data = deepcopy(self.model), deepcopy(self.reference), HeldOut()
        with patch('turkeyworld.engine.Game.render', side_effect=AssertionError('No Source rendering')), \
             patch('turkeyworld.engine.Game.step', side_effect=AssertionError('No Source transition')), \
             patch.object(net.flow_renderer, 'latent_sample', wraps=net.flow_renderer.latent_sample) as calls:
            report, arrays = evaluate_conditional_restorer(net, reference, data, self.panel,
                step=0, batch_size=2, geometry_panel=self.geometry_panel)
        self.assertEqual(calls.call_count, 5)
        self.assertEqual(report['kind'], KIND); self.assertEqual(report['protocol'], PROTOCOL)
        for prefix in ('rf', 'source'):
            self.assertTrue(torch.equal(arrays[prefix + '_old'], arrays[prefix + '_new']))
            self.assertEqual(report['models'][prefix + '_old'], report['models'][prefix + '_new'])
        self.assertTrue(report['full_codec_tensors_unchanged'])
        self.assertTrue(report['decoder_post_quant_conv_tensors_unchanged'])
        self.assertTrue(report['conditional_restorer_tensors_unchanged_by_validation'])
        self.assertTrue(report['source_preservation']['passed'])
        self.assertTrue(report['geometry']['source_preservation']['passed'])
        self.assertFalse(report['rf_improvement']['validated_improvement'])
        self.assertFalse(report['quality_passed'])
        for key in ('world_rollout', 'sr_quality_evaluated', 'autonomous_rollout',
                    'Source_RGB_direct_restorer_input', 'Source_RGB_into_RF_conditioning',
                    'gameplay_qualified', 'automatic_active_promotion'):
            self.assertFalse(report[key], key)
        self.assertTrue(report['source_held_out']); self.assertEqual(report['native_resolution'], 32)
        json.dumps(report, allow_nan=False)

    def test_actual_objective_unequal_tail_and_frozen_model_reference_RNG_sampler_flags(self):
        net, reference, data = deepcopy(self.model), deepcopy(self.reference), HeldOut()
        for key in ('state', 'target', 'background', 'object_mask'):
            data.values[key].requires_grad_(True)
        hashes = (_hash_tensors(net.state_dict()), _hash_tensors(reference.state_dict()))
        modes = {n: m.training for n, m in net.named_modules()}
        flags = {n: p.requires_grad for n, p in net.named_parameters()}
        sampler = data.sampler_state()
        cpu, numpy, python = torch.get_rng_state().clone(), deepcopy(np.random.get_state()), random.getstate()
        seen = []
        actual_batch = data.batch
        def noisy_batch(size, *, indices):
            torch.rand(1); np.random.random(); random.random(); data.rng.random()
            data.draws += size
            for index in indices: data.case_counts[index] += 1
            return actual_batch(size, indices=indices)
        def objective(network, *, prepared):
            result = conditional_restorer_loss(network, prepared=prepared)
            seen.append((len(prepared[0]), float(result[0]), deepcopy(result[1])))
            return result
        with patch.object(data, 'batch', side_effect=noisy_batch), \
             patch('turkeyworld.conditional_restorer_validation.conditional_restorer_loss', side_effect=objective):
            report, _ = evaluate_conditional_restorer(net, reference, data, self.panel, step=2, batch_size=2)
        self.assertEqual([size for size, _, _ in seen], [2, 2, 1])
        expected = sum(size * value for size, value, _ in seen) / 5
        self.assertAlmostEqual(report['val_loss'], expected, places=8)
        obj = report['validation_objective']
        self.assertEqual(obj['loss'], report['val_loss'])
        self.assertEqual(obj['protocol'], PROTOCOL)
        self.assertEqual(obj['branch_weights'], dict(rf_quantized=.5, source_quantized=.5))
        self.assertEqual(obj['aggregation'], 'batch_image_weighted')
        self.assertEqual(obj['restorer_examples'], 10)
        self.assertEqual(obj['input_kind'], 'frozen_RF3_or_Source_quantized_codes_RGB_state_map')
        for branch in ('rf_quantized', 'source_quantized'):
            self.assertAlmostEqual(obj['components'][branch]['loss'], sum(
                size * metrics['branches'][branch]['loss'] for size, _, metrics in seen) / 5, places=8)
        self.assertEqual(hashes, (_hash_tensors(net.state_dict()), _hash_tensors(reference.state_dict())))
        self.assertEqual(modes, {n: m.training for n, m in net.named_modules()})
        self.assertEqual(flags, {n: p.requires_grad for n, p in net.named_parameters()})
        self.assertEqual(sampler, data.sampler_state())
        self.assertTrue(torch.equal(cpu, torch.get_rng_state()))
        self.assertTrue(np.array_equal(numpy[1], np.random.get_state()[1]))
        self.assertEqual(numpy[2:], np.random.get_state()[2:]); self.assertEqual(python, random.getstate())
        self.assertTrue(all(p.grad is None for p in net.parameters()))
        self.assertTrue(all(p.grad is None for p in reference.parameters()))
        for key in ('state', 'target', 'background', 'object_mask'):
            self.assertIsNone(data.values[key].grad, key)

    def test_nonzero_restorer_measured_old_arms_remain_exact_and_full_D_tamper_rejects(self):
        net, reference, data = deepcopy(self.model), deepcopy(self.reference), HeldOut()
        baseline, original = evaluate_conditional_restorer(net, reference, data, self.panel, step=0, batch_size=2)
        with torch.no_grad():
            net.flow_renderer.conditional_restorer.output.bias.add_(.3)
        report, arrays = evaluate_conditional_restorer(net, reference, data, self.panel,
            step=1, batch_size=2, baseline=baseline)
        self.assertTrue(torch.equal(original['rf_old'], arrays['rf_old']))
        self.assertTrue(torch.equal(original['source_old'], arrays['source_old']))
        self.assertFalse(torch.equal(arrays['rf_old'], arrays['rf_new']))
        self.assertFalse(report['source_preservation']['passed'])
        self.assertEqual(report['baseline_step'], 0)
        for key, value in [('panel_fingerprint', 'foreign'), ('reference_codec_sha256', 'foreign'),
                           ('data_identity_sha256', 'foreign'), ('solver_steps', 5), ('protocol', 'foreign')]:
            bad = deepcopy(baseline); bad[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'identical'):
                evaluate_conditional_restorer(net, reference, data, self.panel,
                    step=1, batch_size=2, baseline=bad)
        with torch.no_grad():
            net.flow_renderer.codec.decoder.conv_out.bias.add_(.001)
        with self.assertRaisesRegex(ValueError, 'exact frozen parent decoder'):
            evaluate_conditional_restorer(net, reference, data, self.panel, step=1)

    def test_empty_support_not_quality_and_five_row_preview_truthfully_names_frozen_D(self):
        from PIL import Image
        net, reference, data = deepcopy(self.model), deepcopy(self.reference), HeldOut()
        empty, _ = evaluate_conditional_restorer(net, reference, data, [3, 4], step=0)
        self.assertIsNone(empty['models']['rf_new']['summary']['foreground_mae'])
        self.assertFalse(empty['passed'])
        self.assertFalse(empty['rf_improvement']['validated_improvement'])
        report, arrays = evaluate_conditional_restorer(net, reference, data, self.panel, step=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'preview.png'
            metadata = save_conditional_restorer_preview(arrays, path, 0)
            with Image.open(path) as image:
                self.assertEqual(image.size, (768, 1085))
                pixel = arrays['rf_new'][0, :, 16, 16].mul(255).round().byte().tolist()
                self.assertEqual(image.getpixel((99, 2*217+25+99)), tuple(pixel))
        self.assertEqual(metadata['kind'], 'conditional_restorer_preview_v1')
        self.assertEqual(metadata['protocol'], PROTOCOL)
        self.assertEqual(metadata['panel_fingerprint'], report['panel_fingerprint'])
        self.assertEqual(metadata['columns'], ['large', 'partial', 'distant', 'empty_navigation'])
        self.assertEqual(metadata['rows'], ['source', 'rf_old', 'rf_new', 'source_old', 'source_new'])
        self.assertEqual(metadata['row_labels'][2], 'RESTORER | RF3 CODES')
        self.assertTrue(all('ADAPTED D' not in value for value in metadata['row_labels']))
        data.split = 'train'
        with self.assertRaisesRegex(ValueError, 'held-out'):
            evaluate_conditional_restorer(net, reference, data, self.panel, step=0)


if __name__ == '__main__':
    unittest.main()
