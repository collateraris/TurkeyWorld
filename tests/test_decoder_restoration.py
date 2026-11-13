from copy import deepcopy
import json
import unittest

import torch
from torch.nn import functional as F

from turkeyworld.enemy_decoder_sr import EnemyDecoderEDSRUpscaler, warp_box
from turkeyworld.decoder_restoration_objectives import (
    decoder_restoration_loss, restoration_crop_targets, restoration_raw_metrics,
    _route_terms, LOSS_WEIGHTS, PROTOCOL, REPLACED_CANONICAL)
from test_visual_first_objectives import fixture
from test_enemy_decoder_visual_first import force_visual, hypothesis


def model(restoration=True, scale=0.):
    return EnemyDecoderEDSRUpscaler(channels=4, blocks=1, semantic_channels=4,
        semantic_blocks=1, enemy_decoder_channels=4, enemy_decoder_stable=True,
        enemy_decoder_visual_first=True, enemy_decoder_restoration=restoration,
        restoration_blend_scale=scale)


def outputs(*, empty=False):
    output, batch = fixture(empty=empty)
    output.update(restoration=True, restoration_blend_scale=0.,
        enemy_applicable=torch.ones(4, 1, dtype=torch.bool))
    for key in ('enemy_bbox', 'visual_bbox'):
        output[key] = torch.tensor([[.25, .25, .75, .75]] * 4, requires_grad=True)
    output['hypothesis_bbox'] = output['enemy_bbox'].detach().clone()
    return output, batch


class DecoderRestorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_opt_in_has_no_new_tensors_and_saved_scale_reconstructs_exact_inference(self):
        torch.manual_seed(173)
        old = model(False).eval()
        new = model(True, .3).eval()
        self.assertEqual(list(old.state_dict()), list(new.state_dict()))
        self.assertEqual(list(dict(old.named_parameters())), list(dict(new.named_parameters())))
        new.load_state_dict(old.state_dict(), strict=True)
        cfg = {key: value for key, value in new.config.items() if key != 'architecture'}
        copy = EnemyDecoderEDSRUpscaler(**cfg).eval()
        copy.load_state_dict(new.state_dict(), strict=True)
        rgb = torch.rand(2, 3, 32, 32)
        torch.testing.assert_close(new(rgb, None), copy(rgb, None), rtol=0, atol=0)
        self.assertEqual(copy.restoration_blend_scale, .3)
        new.set_restoration_blend_scale(0.)
        self.assertEqual(new.config['restoration_blend_scale'], 0.)
        self.assertNotIn('enemy_decoder_restoration', old.config)
        for invalid in (True, float('nan'), -.1, 1.1):
            with self.assertRaises(ValueError): new.set_restoration_blend_scale(invalid)
        with self.assertRaises(ValueError): old.set_restoration_blend_scale(.3)
        with self.assertRaises(ValueError):
            EnemyDecoderEDSRUpscaler(enemy_decoder_restoration=True)

    def test_explicit_gradual_blend_bypasses_both_tiny_gains_without_altering_old_profile(self):
        m = model().eval()
        force_visual(m)
        with torch.no_grad():
            m.enemy_decoder_gain.weight.fill_(-1.)
            m.enemy_decoder_visual_gain.weight.zero_()
        rgb = torch.full((2, 3, 32, 32), .25)
        zero = m.forward_with_aux(rgb, None)
        self.assertEqual(float(zero['visual_gain']), 0.)
        m.set_restoration_blend_scale(1.)
        full = m.forward_with_aux(rgb, None)
        self.assertTrue(full['visual_route'].all())
        self.assertGreater(float((full['image'] - zero['image']).detach().abs().sum()), 1.)
        m.set_restoration_blend_scale(.3)
        partial = m.forward_with_aux(rgb, None)
        torch.testing.assert_close(partial['image'], zero['image'] + .3 * (full['image'] - zero['image']),
                                   rtol=1e-6, atol=1e-7)
        for hint in (hypothesis(), torch.cat((torch.zeros(2, 20), torch.ones(2, 1)), 1)):
            torch.testing.assert_close(m(rgb, hint), partial['image'], rtol=0, atol=0)
        # An uncertain visual route can use guided RGB, again without the old gain.
        force_visual(m, presence=10., confirmation=0.)
        with torch.no_grad(): m.enemy_decoder_gain.weight.zero_()
        m.set_restoration_blend_scale(0.)
        base = m(rgb, hypothesis())
        m.set_restoration_blend_scale(1.)
        guided = m.forward_with_aux(rgb, hypothesis())
        self.assertTrue(guided['guided_route'].all())
        self.assertGreater(float((guided['image'] - base).detach().abs().sum()), 0.)
        self.assertEqual(float(guided['enemy_gain']), 1.)
        self.assertTrue(all(not p.requires_grad for n, p in m.named_parameters()
                            if not n.startswith('enemy_decoder_')))

    def test_raw_RGB_and_alpha_get_nonzero_gradients_at_zero_blend_no_hint_and_Source_gradients_none(self):
        m = model().train()
        _, batch = fixture()
        rgb = torch.full((4, 3, 32, 32), .2)
        output = m.forward_with_aux(rgb, None)
        loss, metrics = decoder_restoration_loss(output, batch)
        loss.backward()
        final = m.enemy_decoder_rgba[-1].weight.grad
        self.assertTrue(torch.isfinite(final).all())
        self.assertGreater(float(final[:3].abs().sum()), 0.)
        self.assertGreater(float(final[3:].abs().sum()), 0.)
        self.assertEqual(metrics['restoration_blend_scale'], 0.)
        self.assertTrue(metrics['restoration_raw_losses_before_gain'])
        self.assertEqual(metrics['restoration_routes']['visual']['frames'], 4)
        self.assertEqual(metrics['restoration_routes']['guided']['frames'], 0)
        for key in ('hr_source', 'hr_background', 'mask_hr', 'mask_lr', 'bbox_hr', 'hypothesis'):
            self.assertIsNone(batch[key].grad, key)
        self.assertTrue(all(p.grad is None for n, p in m.named_parameters() if not n.startswith('enemy_decoder_')))
        for name in REPLACED_CANONICAL: self.assertNotIn(name, metrics)
        self.assertAlmostEqual(metrics['train_loss'], sum(metrics[k + '_weighted'] for k in LOSS_WEIGHTS), places=5)
        self.assertEqual(metrics['loss_protocol'], PROTOCOL)
        json.dumps(metrics, allow_nan=False)

    def test_occluded_visible_crop_retains_its_position_in_actual_larger_warp_box(self):
        # Only the right half of an enemy remains visible behind a wall. The
        # tight visible bbox would stretch it across the whole proposal.
        mask = torch.zeros(1, 1, 128, 128)
        mask[:, :, 32:96, 64:96] = 1
        source = torch.full((1, 3, 128, 128), .1)
        source[:, :1] += .7 * mask
        batch = {'hr_source': source.requires_grad_(), 'mask_hr': mask.requires_grad_(),
                 'hr_background': torch.full_like(source, .1).requires_grad_()}
        box = torch.tensor([[.25, .25, .75, .75]], requires_grad=True)
        target_rgb, target_alpha = restoration_crop_targets(batch, box)
        self.assertEqual(float(target_alpha[:, :, :, :15].sum()), 0.)
        torch.testing.assert_close(target_alpha[:, :, 1:-1, 17:-1], torch.ones(1, 1, 30, 14))
        torch.testing.assert_close(target_rgb[:, :1][target_alpha > 0],
                                   torch.full_like(target_alpha[target_alpha > 0], .8))
        rgba = torch.cat((target_rgb, target_alpha), 1).detach().requires_grad_()
        terms, receipt = _route_terms(rgba, box.detach(), torch.tensor([True]), batch)
        self.assertEqual(terms['restoration_rgb'], 0.)
        stretched = rgba.detach().clone(); stretched[:, 3:] = 1.
        wrong, _ = _route_terms(stretched, box.detach(), torch.tensor([True]), batch)
        self.assertLess(float(terms['restoration_alpha_bce'].detach()),
                        .1 * float(wrong['restoration_alpha_bce']))
        # Its reconstruction occupies the original right-side image pixels.
        warped = warp_box(rgba, box.detach())[:, 3:4]
        occupied = (warped > .5).nonzero()
        self.assertEqual((int(occupied[:, 2].min()), int(occupied[:, 2].max())), (32, 95))
        self.assertEqual((int(occupied[:, 3].min()), int(occupied[:, 3].max())), (64, 95))
        sum(terms.values()).backward()
        self.assertIsNone(box.grad)
        self.assertIsNone(batch['hr_source'].grad)
        self.assertIsNone(batch['mask_hr'].grad)
        self.assertEqual(receipt['positive_frames'], 1)

    def test_RGB_color_area_normalization_and_empty_alpha_hard_negatives(self):
        values = []
        for size in (8, 64):
            mask = torch.zeros(1, 1, 128, 128)
            mask[:, :, 32:32+size, 32:32+size] = 1
            source = torch.full((1, 3, 128, 128), .1) + .7 * mask
            batch = {'hr_source': source, 'mask_hr': mask, 'hr_background': torch.full_like(source, .1)}
            rgba = torch.full((1, 4, 32, 32), .3, requires_grad=True)
            terms, _ = _route_terms(rgba, torch.tensor([[0., 0., 1., 1.]]), torch.tensor([True]), batch)
            values.append(float(terms['restoration_rgb'].detach()))
        self.assertAlmostEqual(values[0], .5, places=6)
        self.assertAlmostEqual(values[1], .5, places=6)
        output, batch = outputs(empty=True)
        loss, metrics = decoder_restoration_loss(output, batch)
        loss.backward()
        for key in ('visual_rgba', 'enemy_rgba'):
            self.assertTrue(torch.isfinite(output[key].grad).all())
            # Positive gradient lowers spurious alpha on genuine empty rows.
            self.assertGreater(float(output[key].grad[:, 3:4].sum()), 0.)
        self.assertEqual(metrics['restoration_rgb'], 0.)
        self.assertEqual(metrics['restoration_routes']['visual']['positive_frames'], 0)
        json.dumps(metrics, allow_nan=False)

    def test_single_HR_pixel_survives_canonical_area_sampling_without_background_color(self):
        mask = torch.zeros(1, 1, 128, 128)
        mask[:, :, 65, 66] = 1
        source = torch.full((1, 3, 128, 128), .9)
        source[:, :, 65, 66] = torch.tensor([.2, .3, .4])
        rgb, alpha = restoration_crop_targets({'hr_source': source, 'mask_hr': mask},
                                               torch.tensor([[0., 0., 1., 1.]]))
        self.assertAlmostEqual(float(alpha.sum()), 1 / 16, places=7)
        color = rgb[0, :, 16, 16]
        torch.testing.assert_close(color, torch.tensor([.2, .3, .4]), rtol=0, atol=1e-7)
        self.assertEqual(float(rgb[:, :, 0, 0].sum()), 0.)

    def test_raw_metrics_keep_missed_foreground_in_denominator_and_correct_mean_counts(self):
        output, batch = outputs()
        output['visual_bbox'] = torch.tensor([[0., 0., .1, .1]] * 4)
        output['guided_route'] = torch.zeros(4, 1, dtype=torch.bool)
        before = deepcopy({key: value.detach() for key, value in output.items() if torch.is_tensor(value)})
        raw = restoration_raw_metrics(output, batch)
        self.assertEqual(raw['raw_box_miss_visible_frames'], 3)
        self.assertEqual(raw['metric_counts'], {'raw_foreground_mae': 3,
            'raw_foreground_edge_mae': 3, 'raw_empty_mae': 1, 'raw_background_mae': 4})
        self.assertAlmostEqual(raw['raw_foreground_mae'], .7 / 3, places=5)
        self.assertEqual(raw['raw_visual_frames'], 4)
        self.assertEqual(raw['raw_guided_frames'], 0)
        for key, value in before.items(): torch.testing.assert_close(output[key], value, rtol=0, atol=0)
        json.dumps(raw, allow_nan=False)


if __name__ == '__main__': unittest.main()
