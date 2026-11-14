"""Actual tiny RF3/VQ comparisons, paired loss, isolation and honest support."""
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_renderer_rf5_objectives import model
from test_renderer_rf5_validation import TinyHeldOutSource
from turkeyworld.decoder_rf_adaptation_objectives import (
    PROTOCOL, configure_decoder_adaptation, decoder_rf_adaptation_loss,
    generate_decoder_training_codes)
from turkeyworld.decoder_rf_adaptation_validation import (
    KIND, BRANCHES, evaluate_decoder_adaptation, save_decoder_adaptation_preview)
from turkeyworld.renderer_student import _hash_tensors
from turkeyworld.staged_validation import _isolated_validation


class HeldOut(TinyHeldOutSource):
    def __init__(self):
        super().__init__()
        self.identity = dict(kind='real_tiny_recorded_Source_fixture', split='val', rows=5)
        self.case_counts = dict.fromkeys(range(5), 0)

    def sampler_state(self):
        return deepcopy(dict(rng=self.rng.bit_generator.state, draws=self.draws,
                             case_counts=self.case_counts))

    def restore_sampler(self, state):
        self.rng.bit_generator.state = deepcopy(state['rng'])
        self.draws = state['draws']
        self.case_counts = deepcopy(state['case_counts'])


def network():
    result = model(3)
    configure_decoder_adaptation(result)
    reference = deepcopy(result.flow_renderer.codec).eval().requires_grad_(False)
    return result, reference


class DecoderRFAdaptationValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model, cls.reference = network()
        cls.panel = [0, 1, 2, 3, 4]
        cls.geometry_panel = [0, 2, 3]

    def test_same_weights_four_branches_share_codes_and_keep_native_Source_scope(self):
        net, reference = deepcopy(self.model), deepcopy(self.reference)
        data = HeldOut()
        with patch('turkeyworld.engine.Game.render', side_effect=AssertionError('No Source render')), \
             patch('turkeyworld.engine.Game.step', side_effect=AssertionError('No Source transition')), \
             patch.object(net.flow_renderer, 'latent_sample', wraps=net.flow_renderer.latent_sample) as rf:
            report, arrays = evaluate_decoder_adaptation(net, reference, data, self.panel,
                step=0, batch_size=2, geometry_panel=self.geometry_panel)
        # One endpoint per actual batch; never one endpoint per compared decoder.
        self.assertEqual(rf.call_count, 3 + 2)
        self.assertEqual(report['kind'], KIND)
        self.assertEqual(report['protocol'], PROTOCOL)
        self.assertEqual(set(report['models']), set(BRANCHES))
        self.assertTrue(torch.equal(arrays['rf_old'], arrays['rf_new']))
        self.assertTrue(torch.equal(arrays['source_old'], arrays['source_new']))
        self.assertEqual(report['models']['rf_old'], report['models']['rf_new'])
        self.assertEqual(report['models']['source_old'], report['models']['source_new'])
        self.assertTrue(report['source_preservation']['passed'])
        self.assertFalse(report['rf_improvement']['validated_improvement'])
        self.assertFalse(report['passed'])
        self.assertEqual(report['geometry']['frames'], 3)
        self.assertEqual(report['geometry']['panel_fingerprint'], data.panel_fingerprint(self.geometry_panel))
        self.assertTrue(report['geometry']['source_preservation']['passed'])
        self.assertTrue(report['reference_codec_tensors_unchanged'])
        self.assertTrue(report['encoder_quantizer_tensors_unchanged'])
        self.assertTrue(report['source_held_out']); self.assertTrue(report['renderer_only'])
        for key in ('world_rollout', 'autonomous_rollout', 'sr_quality_evaluated',
                    'future_source_rgb_in_prior', 'future_source_pose_in_prior',
                    'Source_RGB_into_RF_conditioning', 'gameplay_qualified', 'automatic_active_promotion'):
            self.assertFalse(report[key], key)
        self.assertEqual(report['native_resolution'], 32)
        self.assertEqual(arrays['rf_new'].shape, (5, 3, 32, 32))
        json.dumps(report, allow_nan=False)

    def test_actual_50_50_objective_and_components_use_batch_image_weighted_unequal_tail(self):
        net, reference = deepcopy(self.model), deepcopy(self.reference)
        data = HeldOut()
        observed = []
        def objective(network, *, prepared):
            result = decoder_rf_adaptation_loss(network, prepared=prepared)
            observed.append((len(prepared[0]), float(result[0]), deepcopy(result[1])))
            return result
        with patch('turkeyworld.decoder_rf_adaptation_validation.decoder_rf_adaptation_loss',
                   side_effect=objective):
            report, _ = evaluate_decoder_adaptation(net, reference, data, self.panel, step=7, batch_size=2)
        self.assertEqual([size for size, _, _ in observed], [2, 2, 1])
        expected = sum(size * loss for size, loss, _ in observed) / 5
        self.assertAlmostEqual(report['val_loss'], expected, places=8)
        obj = report['validation_objective']
        self.assertEqual(obj['loss'], report['val_loss'])
        self.assertEqual(obj['aggregation'], 'batch_image_weighted')
        self.assertEqual(obj['branch_weights'], dict(rf_quantized=.5, source_quantized=.5))
        self.assertEqual(obj['input_kind'], 'RF3_quantized_and_Source_quantized_latents')
        self.assertEqual((obj['frames'], obj['decoder_examples']), (5, 10))
        for branch in ('rf_quantized', 'source_quantized'):
            expected_branch = sum(size * metrics['branches'][branch]['loss']
                                  for size, _, metrics in observed) / 5
            row = obj['components'][branch]
            self.assertAlmostEqual(row['loss'], expected_branch, places=8)
            self.assertAlmostEqual(row['loss'], sum(row['weighted_terms'].values()), places=6)
            self.assertEqual(row['foreground_images'], sum(
                metrics['branches'][branch]['foreground_images'] for _, _, metrics in observed))
        self.assertNotEqual(report['val_loss'], report['models']['rf_new']['summary']['mae_mean'])

    def test_full_RNG_private_sampler_modes_grad_flags_weights_and_source_grads_preserved(self):
        net, reference = deepcopy(self.model), deepcopy(self.reference)
        data = HeldOut()
        # Mixed child modes and old gradient buffers must survive validation too.
        net.flow_renderer.codec.decoder.eval()
        next(p for p in net.parameters() if p.requires_grad).grad = torch.ones_like(
            next(p for p in net.parameters() if p.requires_grad))
        for key in ('state', 'target', 'background', 'object_mask'):
            data.values[key].requires_grad_(True)
        before = (_hash_tensors(net.state_dict()), _hash_tensors(reference.state_dict()))
        modes = {n: m.training for n, m in net.named_modules()}
        reference_modes = {n: m.training for n, m in reference.named_modules()}
        flags = {n: p.requires_grad for n, p in net.named_parameters()}
        grads = {n: p.grad.clone() if p.grad is not None else None for n, p in net.named_parameters()}
        sampler = data.sampler_state()
        cpu, numpy, python = torch.get_rng_state().clone(), deepcopy(np.random.get_state()), random.getstate()
        original_batch = data.batch
        def noisy_batch(size, *, indices):
            torch.rand(1); np.random.random(); random.random(); data.rng.random()
            data.draws += size
            for index in indices:
                data.case_counts[int(index)] += 1
            return original_batch(size, indices=indices)
        with patch.object(data, 'batch', side_effect=noisy_batch):
            evaluate_decoder_adaptation(net, reference, data, self.panel, step=1, batch_size=2)
        self.assertEqual(before, (_hash_tensors(net.state_dict()), _hash_tensors(reference.state_dict())))
        self.assertEqual(modes, {n: m.training for n, m in net.named_modules()})
        self.assertEqual(reference_modes, {n: m.training for n, m in reference.named_modules()})
        self.assertEqual(flags, {n: p.requires_grad for n, p in net.named_parameters()})
        self.assertEqual(sampler, data.sampler_state())
        self.assertTrue(torch.equal(cpu, torch.get_rng_state()))
        self.assertTrue(np.array_equal(numpy[1], np.random.get_state()[1]))
        self.assertEqual(numpy[2:], np.random.get_state()[2:]); self.assertEqual(python, random.getstate())
        for name, parameter in net.named_parameters():
            if grads[name] is None:
                self.assertIsNone(parameter.grad, name)
            else:
                self.assertTrue(torch.equal(grads[name], parameter.grad), name)
        for key in ('state', 'target', 'background', 'object_mask'):
            self.assertIsNone(data.values[key].grad, key)
        self.assertTrue(all(p.grad is None for p in reference.parameters()))

    def test_changed_D_is_measured_on_both_shared_code_sets_without_changing_reference(self):
        net, reference = deepcopy(self.model), deepcopy(self.reference)
        data = HeldOut()
        baseline, old_arrays = evaluate_decoder_adaptation(net, reference, data, self.panel, step=0, batch_size=2)
        with torch.no_grad():
            net.flow_renderer.codec.decoder.conv_out.bias.add_(1.)
        report, arrays = evaluate_decoder_adaptation(net, reference, data, self.panel,
            step=1, batch_size=2, baseline=baseline)
        self.assertTrue(torch.equal(old_arrays['rf_old'], arrays['rf_old']))
        self.assertTrue(torch.equal(old_arrays['source_old'], arrays['source_old']))
        self.assertFalse(torch.equal(arrays['rf_old'], arrays['rf_new']))
        self.assertFalse(torch.equal(arrays['source_old'], arrays['source_new']))
        self.assertFalse(report['source_preservation']['passed'])
        self.assertNotEqual(report['models']['rf_old']['summary'], report['models']['rf_new']['summary'])
        self.assertEqual(report['baseline_step'], 0)
        self.assertEqual(_hash_tensors(reference.state_dict()), baseline['reference_codec_sha256'])

    def test_empty_foreground_is_unmeasured_not_zero_error_or_pass_and_train_data_rejects(self):
        net, reference = deepcopy(self.model), deepcopy(self.reference)
        data = HeldOut()
        report, _ = evaluate_decoder_adaptation(net, reference, data, [3, 4], step=0)
        for branch in BRANCHES:
            summary = report['models'][branch]['summary']
            self.assertEqual(summary['foreground_frames'], 0)
            self.assertIsNone(summary['foreground_mae'])
        self.assertFalse(report['quality_passed'])
        self.assertFalse(report['rf_improvement']['validated_improvement'])
        self.assertIn('insufficient_visible_NPC_frames', report['rf_improvement']['failure_reasons'])
        data.split = 'train'
        with self.assertRaisesRegex(ValueError, 'held-out'):
            evaluate_decoder_adaptation(net, reference, data, self.panel, step=0)

    def test_baseline_data_panel_solver_reference_and_trainable_RF_mismatches_fail_closed(self):
        net, reference = deepcopy(self.model), deepcopy(self.reference)
        data = HeldOut()
        baseline, _ = evaluate_decoder_adaptation(net, reference, data, self.panel, step=0, batch_size=2)
        for key, value in [('panel_fingerprint', 'foreign'), ('geometry_panel_fingerprint', 'foreign'),
                           ('data_identity_sha256', 'foreign'), ('reference_codec_sha256', 'foreign'),
                           ('solver_steps', 5), ('protocol', 'foreign')]:
            bad = deepcopy(baseline); bad[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'identical'):
                evaluate_decoder_adaptation(net, reference, data, self.panel,
                    step=1, batch_size=2, baseline=bad)
        next(net.base_student.parameters()).requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'only D'):
            evaluate_decoder_adaptation(net, reference, data, self.panel, step=0)
        configure_decoder_adaptation(net)
        with torch.no_grad():
            next(reference.encoder.parameters()).add_(.001)
        with self.assertRaisesRegex(ValueError, 'exact frozen encoder'):
            evaluate_decoder_adaptation(net, reference, data, self.panel, step=0)

    def test_five_row_preview_has_actual_pixels_pinned_indices_and_no_SR_claim(self):
        from PIL import Image
        net, reference = deepcopy(self.model), deepcopy(self.reference)
        data = HeldOut()
        report, arrays = evaluate_decoder_adaptation(net, reference, data, self.panel, step=3, batch_size=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'preview.png'
            metadata = save_decoder_adaptation_preview(arrays, path, 3)
            with Image.open(path) as image:
                self.assertEqual(image.size, (768, 1085))
                # The center of a displayed cell is exactly a nearest-native pixel.
                pixel = arrays['source'][0, :, 16, 16].clamp(0, 1).mul(255).round().byte().tolist()
                self.assertEqual(image.getpixel((16*6+3, 25+16*6+3)), tuple(pixel))
        self.assertEqual(metadata['rows'], ['source', 'rf_old', 'rf_new', 'source_old', 'source_new'])
        self.assertEqual(metadata['columns'], ['large', 'partial', 'distant', 'empty_navigation'])
        self.assertEqual(metadata['row_indices'], [0, 1, 2, 3])
        self.assertEqual(metadata['panel_fingerprint'], report['panel_fingerprint'])
        self.assertFalse(metadata['sr_quality_evaluated'])
        self.assertFalse(metadata['source_input_allowed_for_rf'])


if __name__ == '__main__':
    unittest.main()
