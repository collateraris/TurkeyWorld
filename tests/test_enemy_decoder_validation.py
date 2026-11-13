"""Validation sends only RGB and causal hypotheses to the guided SR model."""
from copy import deepcopy
import unittest

import numpy as np
import torch

from test_semantic_sr_validation import PairedData, TinySR
from turkeyworld.enemy_decoder_data import hr_box_and_visibility
from turkeyworld.enemy_animation import enemy_animation_index
from turkeyworld.enemy_decoder_sr import EnemyDecoderEDSRUpscaler
from turkeyworld.semantic_sr_validation import (evaluate_semantic_sr, preview_semantic_sr,
                                               _current_forward)
from turkeyworld.train_semantic_sr import mix_enemy_hypotheses


class GuidedPairs(PairedData):
    identity = {'kind': 'NPR_with_pinned_causal_RSSM_fixture', 'world_sha256': 'frozen_world'}

    def batch(self, count, *, indices):
        result = super().batch(count, indices=indices)
        hypothesis = torch.zeros(count, 21)
        hypothesis[:, :2] = .3
        hypothesis[:, 3:5] = 1
        hypothesis[:, 8] = .45
        hypothesis[:, 9] = .3
        hypothesis[:, 12:14] = 1
        hypothesis[:, 20] = 1
        hypothesis[:, 17] = torch.as_tensor(indices).float()/10
        result['hypothesis'] = hypothesis
        result['bbox_hr'], result['visible_labels'] = hr_box_and_visibility(result['mask_hr'])
        result['animation_label'] = enemy_animation_index(result['state'])
        return result


class TrackedDecoder(EnemyDecoderEDSRUpscaler):
    def __init__(self, stable=False):
        super().__init__(channels=8, blocks=1, semantic_channels=4, semantic_blocks=1,
                         enemy_decoder_channels=4, enemy_decoder_stable=stable)
        self.inference_calls = []

    def forward_with_aux(self, rgb, hypothesis=None):
        self.inference_calls.append((rgb.detach().clone(), hypothesis.detach().clone()))
        return super().forward_with_aux(rgb, hypothesis)


class EnemyDecoderValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_full_and_light_validation_use_pinned_hypotheses_with_separate_protocol(self):
        model, data = TrackedDecoder(), GuidedPairs()
        before = deepcopy(data.sampler_state()); rng = torch.get_rng_state().clone()
        refs = {'sr38100': TinySR(.04), 'sr40100': TinySR(.02)}
        report, _ = evaluate_semantic_sr(model, data, list(range(8)), step=100,
                                         batch_size=4, baselines=refs)
        self.assertEqual(report['kind'], 'enemy_decoder_super_resolution_v1')
        self.assertEqual(report['validation_objective']['protocol'], 'rssm_guided_enemy_decoder_sr_v1')
        self.assertEqual(report['validation_objective']['frames'], 8)
        self.assertFalse(report['source_future_state_allowed_for_sr'])
        self.assertEqual(report['enemy_decoder']['hypothesis_frames'], 8)
        self.assertEqual(before, data.sampler_state())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        expected = data.batch(4, indices=list(range(4)))['hypothesis']
        torch.testing.assert_close(model.inference_calls[0][1], expected)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertIsNone(data.hr.grad); self.assertIsNone(data.npr.grad)
        data.restore_sampler(before)
        preview = preview_semantic_sr(model, data, list(range(8)), step=15, baseline_report=report)
        self.assertEqual(preview['metadata']['kind'], report['kind'])
        self.assertEqual(preview['metadata']['step'], 15)
        self.assertEqual(before, data.sampler_state())
        stale = deepcopy(report); stale['kind'] = 'semantic_npr_super_resolution_v1'
        with self.assertRaisesRegex(ValueError, 'different Source/NPR/panel'):
            evaluate_semantic_sr(model, data, list(range(8)), baseline_report=stale)

    def test_future_labels_cannot_change_inference_and_missing_hypotheses_fail_closed(self):
        model, data = TrackedDecoder().eval(), GuidedPairs()
        batch = data.batch(4, indices=list(range(4)))
        with torch.no_grad():
            first = _current_forward(model, batch['lr_npr'], batch)['image']
            poisoned = {**batch, 'state': torch.full_like(batch['state'], float('nan')),
                        'mask_hr': torch.zeros_like(batch['mask_hr']),
                        'hr_source': torch.ones_like(batch['hr_source'])}
            second = _current_forward(model, batch['lr_npr'], poisoned)['image']
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'pinned causal'):
            _current_forward(model, batch['lr_npr'], {})

    def test_wrong_proposal_augmentation_is_deterministic_and_never_changes_labels(self):
        data = GuidedPairs(); batch = data.batch(8, indices=list(range(8)))
        original = batch['hypothesis'].clone()
        changed, count = mix_enemy_hypotheses(original, np.random.default_rng(7))
        again, repeated = mix_enemy_hypotheses(original, np.random.default_rng(7))
        self.assertEqual(count, 2); self.assertEqual(count, repeated)
        self.assertEqual(int((changed != original).any(-1).sum()), 2)
        torch.testing.assert_close(changed, again, rtol=0, atol=0)
        torch.testing.assert_close(original, batch['hypothesis'], rtol=0, atol=0)
        self.assertTrue(bool((changed[:, -1] == 1).all()))

    def test_stable_validation_reports_animation_coverage_without_future_conditioning(self):
        model, data = TrackedDecoder(stable=True), GuidedPairs()
        report, _ = evaluate_semantic_sr(model, data, list(range(8)), batch_size=4,
            baselines={'sr38100': TinySR(.04), 'sr40100': TinySR(.02)})
        animation = report['enemy_decoder']['animations']
        self.assertEqual(sum(animation['source_visible_counts'].values()), 6)
        self.assertGreater(animation['source_visible_counts']['death_start'], 0)
        self.assertTrue(animation['labels_are_Source_only'])
        self.assertTrue(animation['condition_is_predicted_RSSM_only'])
        self.assertEqual(report['domains']['npr']['models']['current']['cohorts']['animation_death_start']['frames'],
                         animation['source_visible_counts']['death_start'])
        self.assertEqual(animation['visual_frames'], report['enemy_decoder']['visible_mask_frames'])


if __name__ == '__main__':
    unittest.main()
