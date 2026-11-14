import json
import unittest

import torch

from turkeyworld.neural_enemy_restoration_objectives import (
    PROTOCOL, FOREGROUND_PROTOCOL, LOSS_WEIGHTS, FOREGROUND_LOSS_WEIGHTS,
    FOREGROUND_TOPK_PIXELS, DETAIL_PROTOCOL, DETAIL_LOSS_WEIGHTS, DETAIL_SETTINGS,
    DETAIL_POSITIVE_TOPK_PIXELS, DETAIL_RGB_TOPK_PIXELS,
    _topk_foreground, _rgb_detail_crops,
    loss_weights, neural_enemy_restoration_loss)


def fixture(*, count=2, alpha_logit=-2., rgb_value=.1, box=(48, 52, 60, 64)):
    mask = torch.zeros(count, 1, 128, 128)
    y0, y1, x0, x1 = box
    mask[0, :, y0:y1, x0:x1] = 1
    source_background = torch.full((count, 3, 128, 128), .2)
    source = source_background * (1 - mask) + .9 * mask
    rgb = torch.full_like(source, rgb_value, requires_grad=True)
    logits = torch.full_like(mask, alpha_logit, requires_grad=True)
    alpha = logits.sigmoid()
    background = torch.full_like(source, .2)
    image = background * (1 - alpha) + rgb * alpha
    output = dict(image=image, background=background, enemy_rgb=rgb, alpha_logits=logits, alpha=alpha)
    batch = dict(hr_source=source.requires_grad_(), hr_background=source_background.requires_grad_(),
                 mask_hr=mask.requires_grad_())
    return output, batch


class NeuralEnemyRestorationObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_zero_alpha_cannot_hide_wrong_rgb_or_its_gradient(self):
        hidden, batch = fixture(alpha_logit=-40.)
        revealed, same_labels = fixture(alpha_logit=40.)
        loss, metrics = neural_enemy_restoration_loss(hidden, batch)
        _, visible_metrics = neural_enemy_restoration_loss(revealed, same_labels)
        self.assertGreater(metrics['enemy_rgb_l1'], .79)
        for name in ('enemy_rgb_l1', 'enemy_rgb_edge', 'enemy_rgb_ssim'):
            self.assertEqual(metrics[name], visible_metrics[name])
        loss.backward()
        mask = batch['mask_hr'].detach().expand_as(hidden['enemy_rgb'])
        gradient = hidden['enemy_rgb'].grad
        self.assertGreater(float((gradient.abs() * mask).sum()), .5)
        self.assertLess(float((gradient.abs() * (1 - mask)).sum()), 1e-10)
        self.assertTrue(all(value.grad is None for value in batch.values()))
        self.assertFalse(hidden['background'].requires_grad)
        self.assertFalse(metrics['predicted_alpha_gates_appearance_loss'])
        self.assertTrue(metrics['source_labels_detached'])

    def test_correct_rgb_does_not_hide_wrong_visibility(self):
        output, batch = fixture(alpha_logit=0.)
        rgb = batch['hr_source'].detach().clone().requires_grad_()
        output['enemy_rgb'] = rgb
        output['image'] = output['background'] * (1 - output['alpha']) + rgb * output['alpha']
        loss, metrics = neural_enemy_restoration_loss(output, batch)
        self.assertEqual(metrics['enemy_rgb_l1'], 0.)
        self.assertLess(metrics['enemy_rgb_ssim'], 1e-6)
        self.assertGreater(metrics['alpha_bce'], .6)
        self.assertGreater(metrics['alpha_dice'], .9)
        loss.backward()
        self.assertGreater(float(output['alpha_logits'].grad.abs().sum()), 0)
        self.assertTrue(all(value.grad is None for value in batch.values()))

    def test_empty_cases_penalize_paint_and_opacity_without_inventing_enemy_rgb_targets(self):
        clean, batch = fixture(count=1, alpha_logit=-40., rgb_value=.9)
        batch['mask_hr'] = torch.zeros_like(batch['mask_hr'])
        batch['hr_source'] = batch['hr_background'].detach().clone()
        painted, _ = fixture(count=1, alpha_logit=4., rgb_value=.9)
        clean_loss, clean_metrics = neural_enemy_restoration_loss(clean, batch)
        loss, metrics = neural_enemy_restoration_loss(painted, batch)
        self.assertGreater(float(loss.detach()), float(clean_loss.detach()) + .5)
        self.assertEqual(metrics['visible_images'], 0); self.assertEqual(metrics['empty_images'], 1)
        for key in ('enemy_rgb_l1', 'enemy_rgb_edge', 'enemy_rgb_ssim', 'alpha_dice'):
            self.assertEqual(metrics[key], 0.)
        self.assertGreater(metrics['empty_alpha'], .98)
        self.assertGreater(metrics['empty_image_change'], .68)
        self.assertLess(clean_metrics['empty_image_change'], 1e-6)
        loss.backward()
        self.assertGreater(float(painted['alpha_logits'].grad.sum()), 0)

    def test_visible_and_empty_group_balance_is_unchanged_by_duplicate_easy_empty_frames(self):
        small, labels_small = fixture(count=2)
        large, labels_large = fixture(count=10)
        _, first = neural_enemy_restoration_loss(small, labels_small)
        _, second = neural_enemy_restoration_loss(large, labels_large)
        for name in LOSS_WEIGHTS:
            self.assertAlmostEqual(first[name], second[name], places=6, msg=name)
        self.assertAlmostEqual(first['train_loss'], second['train_loss'], places=6)
        self.assertEqual(second['empty_images'], 9)

    def test_foreground_rgb_mean_is_area_normalized_for_one_pixel_and_large_enemy(self):
        tiny, labels_tiny = fixture(count=1, box=(0, 1, 0, 1))
        large, labels_large = fixture(count=1, box=(32, 96, 32, 96))
        loss, first = neural_enemy_restoration_loss(tiny, labels_tiny)
        _, second = neural_enemy_restoration_loss(large, labels_large)
        self.assertAlmostEqual(first['enemy_rgb_l1'], second['enemy_rgb_l1'], places=6)
        self.assertEqual(first['foreground_pixels'], 1.)
        self.assertEqual(first['foreground_ssim_images'], 1)
        self.assertGreater(first['enemy_rgb_ssim'], 0)
        loss.backward(); self.assertTrue(bool(torch.isfinite(tiny['enemy_rgb'].grad).all()))

    def test_full_foreground_and_all_empty_have_finite_terms_and_explicit_supports(self):
        for full in (False, True):
            output, batch = fixture(count=1, box=(0, 128, 0, 128))
            if not full:
                batch['mask_hr'] = torch.zeros_like(batch['mask_hr'])
                batch['hr_source'] = batch['hr_background'].detach().clone()
            loss, metrics = neural_enemy_restoration_loss(output, batch)
            self.assertTrue(bool(torch.isfinite(loss)))
            if full:
                self.assertEqual(metrics['background_images'], 0)
                self.assertEqual(metrics['background_preservation'], 0.)
            self.assertAlmostEqual(sum(metrics[key + '_weighted'] for key in LOSS_WEIGHTS),
                                   metrics['train_loss'], places=6)
            self.assertEqual(metrics['objective_protocol'], PROTOCOL)
            self.assertFalse(metrics['GAN_active']); self.assertFalse(metrics['atlas_pixels_used'])
            json.dumps(metrics, allow_nan=False)

    def test_unbounded_frozen_precomposite_uses_display_clipped_preservation_target(self):
        output, batch = fixture(count=1, alpha_logit=-40.)
        background = output['background'].clone()
        background[..., :8] = -.1; background[..., -8:] = 1.1
        output['background'] = background
        output['image'] = (background * (1 - output['alpha']) + output['enemy_rgb'] * output['alpha']).clamp(0, 1)
        target = background.clamp(0, 1)
        batch = dict(hr_source=target.clone().requires_grad_(),
                     hr_background=target.clone().requires_grad_(), mask_hr=torch.zeros_like(output['alpha']))
        loss, metrics = neural_enemy_restoration_loss(output, batch)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(metrics['background_preservation'], 0.)
        self.assertEqual(metrics['empty_image_change'], 0.)
        self.assertFalse(background.requires_grad)
        loss.backward()
        self.assertTrue(all(value.grad is None for value in batch.values()))

    def test_invalid_shapes_nonfinite_labels_and_trainable_background_are_rejected(self):
        for mutation in ('shape', 'nan', 'mask_range', 'background_grad'):
            output, batch = fixture()
            if mutation == 'shape': output['alpha'] = output['alpha'][:, :, :32, :32]
            elif mutation == 'nan': batch['hr_source'].data[0, 0, 0, 0] = float('nan')
            elif mutation == 'mask_range': batch['mask_hr'].data[0, 0, 0, 0] = 2.
            else: output['background'].requires_grad_()
            with self.assertRaises(ValueError, msg=mutation): neural_enemy_restoration_loss(output, batch)


class ForegroundEnemyRestorationObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def foreground_loss(self, output, batch):
        return neural_enemy_restoration_loss(output, batch, protocol=FOREGROUND_PROTOCOL)

    def test_default_and_explicit_legacy_reproduce_identical_loss_metrics_and_gradients(self):
        first, first_labels = fixture()
        second, second_labels = fixture()
        legacy, old = neural_enemy_restoration_loss(first, first_labels)
        explicit, repeated = neural_enemy_restoration_loss(second, second_labels, protocol=PROTOCOL)
        self.assertEqual(float(legacy.detach()), float(explicit.detach())); self.assertEqual(old, repeated)
        legacy.backward(); explicit.backward()
        for key in ('enemy_rgb', 'alpha_logits'):
            torch.testing.assert_close(first[key].grad, second[key].grad, rtol=0, atol=0)
        self.assertEqual(loss_weights(PROTOCOL), LOSS_WEIGHTS)
        self.assertEqual(loss_weights(FOREGROUND_PROTOCOL), FOREGROUND_LOSS_WEIGHTS)
        independent = loss_weights(FOREGROUND_PROTOCOL); independent['alpha_bce'] = 99
        self.assertEqual(loss_weights(FOREGROUND_PROTOCOL)['alpha_bce'], .5)
        for protocol in ('unknown', None, True):
            with self.assertRaises(ValueError): neural_enemy_restoration_loss({}, {}, protocol=protocol)

    def test_source_wall_colors_and_clean_background_do_not_change_v2_loss_gradients(self):
        first, first_labels = fixture(alpha_logit=0., rgb_value=.4)
        second, second_labels = fixture(alpha_logit=0., rgb_value=.4)
        mask = second_labels['mask_hr'].detach()
        walls = torch.linspace(0, 1, 128)[None, None, None, :].expand_as(second_labels['hr_source'])
        second_labels['hr_source'] = (first_labels['hr_source'].detach() * mask + walls * (1 - mask)).requires_grad_()
        second_labels['hr_background'] = (1 - walls).clone().requires_grad_()
        loss, metrics = self.foreground_loss(first, first_labels)
        changed, changed_metrics = self.foreground_loss(second, second_labels)
        self.assertEqual(float(loss.detach()), float(changed.detach()))
        for key in FOREGROUND_LOSS_WEIGHTS: self.assertEqual(metrics[key], changed_metrics[key], key)
        self.assertNotEqual(metrics['source_image_l1_diagnostic'], changed_metrics['source_image_l1_diagnostic'])
        loss.backward(); changed.backward()
        for key in ('enemy_rgb', 'alpha_logits'):
            torch.testing.assert_close(first[key].grad, second[key].grad, rtol=0, atol=0)
        self.assertTrue(all(value.grad is None for value in first_labels.values()))
        self.assertTrue(all(value.grad is None for value in second_labels.values()))
        # This fixture is sensitive to the original full-Source background
        # objective, rather than accidentally choosing equal background errors.
        original, labels = fixture(alpha_logit=0., rgb_value=.4)
        altered, altered_labels = fixture(alpha_logit=0., rgb_value=.4)
        altered_labels['hr_source'] = second_labels['hr_source'].detach()
        altered_labels['hr_background'] = second_labels['hr_background'].detach()
        neural_enemy_restoration_loss(original, labels)[0].backward()
        neural_enemy_restoration_loss(altered, altered_labels)[0].backward()
        self.assertFalse(torch.equal(original['enemy_rgb'].grad, altered['enemy_rgb'].grad))

    def test_empty_frame_never_trains_rgb_toward_source_walls(self):
        first, labels = fixture(count=1, alpha_logit=0., rgb_value=.8, box=(0, 0, 0, 0))
        second, changed = fixture(count=1, alpha_logit=0., rgb_value=.8, box=(0, 0, 0, 0))
        labels['hr_source'] = torch.full_like(labels['hr_source'], .95, requires_grad=True)
        changed['hr_source'] = torch.full_like(changed['hr_source'], .05, requires_grad=True)
        changed['hr_background'] = torch.full_like(changed['hr_background'], .9, requires_grad=True)
        loss, metrics = self.foreground_loss(first, labels)
        other, other_metrics = self.foreground_loss(second, changed)
        loss.backward(); other.backward()
        for key in ('foreground_image_l1', 'foreground_image_ssim', 'foreground_image_edge',
                    'enemy_rgb_l1', 'enemy_rgb_edge', 'enemy_rgb_ssim'):
            self.assertEqual(metrics[key], 0.); self.assertEqual(other_metrics[key], 0.)
        torch.testing.assert_close(first['enemy_rgb'].grad, second['enemy_rgb'].grad, rtol=0, atol=0)
        torch.testing.assert_close(first['alpha_logits'].grad, second['alpha_logits'].grad, rtol=0, atol=0)
        # RGB may learn to undo its own false paint toward the frozen .2 base,
        # but cannot be rewarded for drawing either Source wall color.
        self.assertGreater(float(first['enemy_rgb'].grad.sum()), 0)
        self.assertTrue(all(value.grad is None for value in labels.values()))

    def test_compact_false_alpha_has_strong_topk_penalty_and_local_corrective_gradient(self):
        mask = torch.zeros(1, 1, 128, 128)
        compact = torch.full_like(mask, 1e-4); compact[:, :, 40:44, 50:58] = .9
        diffuse = torch.full_like(mask, float(compact.mean()))
        def output(probability):
            logits = torch.logit(probability).detach().requires_grad_(); alpha = logits.sigmoid()
            background = torch.full((1, 3, 128, 128), .2)
            rgb = background.clone().requires_grad_()
            return dict(image=background*(1-alpha)+rgb*alpha, background=background,
                        enemy_rgb=rgb, alpha_logits=logits, alpha=alpha)
        concentrated, spread = output(compact), output(diffuse)
        labels = dict(hr_source=torch.full((1, 3, 128, 128), .2),
                      hr_background=torch.full((1, 3, 128, 128), .2), mask_hr=mask)
        loss, metrics = self.foreground_loss(concentrated, labels)
        other, other_metrics = self.foreground_loss(spread, labels)
        self.assertAlmostEqual(metrics['empty_alpha'], other_metrics['empty_alpha'], places=7)
        self.assertGreater(metrics['hard_negative_alpha_bce'], 2.)
        self.assertLess(other_metrics['hard_negative_alpha_bce'], .01)
        self.assertGreater(float(loss.detach()), float(other.detach())*100)
        self.assertEqual(metrics['hard_negative_topk_per_image'], [32])
        self.assertEqual(metrics['hard_negative_available_pixels_per_image'], [16384])
        loss.backward(); gradient = concentrated['alpha_logits'].grad
        patch = torch.zeros_like(mask, dtype=torch.bool); patch[:, :, 40:44, 50:58] = True
        self.assertTrue(bool((gradient[patch] > 0).all()))
        self.assertGreater(float(gradient[patch].mean()), float(gradient[~patch].abs().mean())*100)
        updated = (concentrated['alpha_logits'].detach() - gradient).sigmoid()
        self.assertLess(float(updated[patch].mean()), float(concentrated['alpha'].detach()[patch].mean()))

    def test_hard_preservation_sees_a_local_paint_patch_even_at_equal_global_mean_change(self):
        mask = torch.zeros(1, 1, 128, 128)
        change = torch.zeros_like(mask); change[:, :, 40:44, 50:58] = .45
        diffuse = torch.full_like(change, float(change.mean()))
        def output(delta):
            background = torch.full((1, 3, 128, 128), .2)
            logits = torch.full_like(mask, torch.logit(torch.tensor(.9)), requires_grad=True)
            alpha = logits.sigmoid(); rgb = (background + delta.expand_as(background)/.9).detach().requires_grad_()
            return dict(image=background*(1-alpha)+rgb*alpha, background=background,
                        enemy_rgb=rgb, alpha_logits=logits, alpha=alpha)
        first, second = output(change), output(diffuse)
        labels = dict(hr_source=torch.full((1, 3, 128, 128), .2),
                      hr_background=torch.full((1, 3, 128, 128), .2), mask_hr=mask)
        loss, metrics = self.foreground_loss(first, labels)
        _, other = self.foreground_loss(second, labels)
        self.assertAlmostEqual(metrics['background_preservation'], other['background_preservation'], places=7)
        self.assertGreater(metrics['hard_background_preservation'], other['hard_background_preservation']*100)
        loss.backward(); self.assertGreater(float(first['enemy_rgb'].grad[..., 40:44, 50:58].sum()), 0)

    def test_hard_negatives_exclude_true_enemy_pixels_on_visible_frames(self):
        output, batch = fixture(count=2, rgb_value=.9)
        probability = torch.full_like(batch['mask_hr'], 1e-4)
        probability[0, :, 48:52, 60:64] = .999
        probability[0, :, 10:14, 10:18] = .8
        logits = torch.logit(probability).detach().requires_grad_(); alpha = logits.sigmoid()
        output['alpha_logits'], output['alpha'] = logits, alpha
        output['image'] = output['background']*(1-alpha)+output['enemy_rgb']*alpha
        loss, metrics = self.foreground_loss(output, batch)
        # Equal visible/empty case mass: .5*(BCE(.8,0)+BCE(.0001,0)).
        self.assertAlmostEqual(metrics['hard_negative_alpha_bce'], .80476896, places=5)
        self.assertEqual(metrics['hard_negative_visible_images'], 1)
        self.assertEqual(metrics['hard_negative_empty_images'], 1)
        self.assertEqual(metrics['hard_negative_available_pixels_per_image'], [16368, 16384])
        loss.backward()
        self.assertTrue(bool((logits.grad[0, :, 10:14, 10:18] > 0).all()))
        self.assertLess(float(logits.grad[0, :, 48:52, 60:64].mean()), 0)

    def test_raw_rgb_stays_independent_of_opacity_in_the_foreground_phase(self):
        hidden, labels = fixture(alpha_logit=-40.)
        shown, same_labels = fixture(alpha_logit=40.)
        loss, metrics = self.foreground_loss(hidden, labels)
        _, other = self.foreground_loss(shown, same_labels)
        for key in ('enemy_rgb_l1', 'enemy_rgb_edge', 'enemy_rgb_ssim'):
            self.assertEqual(metrics[key], other[key])
        loss.backward()
        self.assertGreater(float((hidden['enemy_rgb'].grad.abs()*labels['mask_hr'].detach()).sum()), .5)
        self.assertFalse(metrics['source_background_reconstruction_gradient'])
        self.assertEqual(metrics['edge_ssim_background_context'], 'same_frozen_display_background_both_sides')

    def test_topk_caps_to_actual_source_background_support_and_weighted_total_is_exact(self):
        output, batch = fixture(count=2, box=(0, 128, 0, 128))
        mask = torch.ones_like(batch['mask_hr']); mask[1, :, :2, :2] = 0
        batch['mask_hr'] = mask.requires_grad_()
        loss, metrics = self.foreground_loss(output, batch)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(metrics['hard_negative_topk_pixels'], FOREGROUND_TOPK_PIXELS)
        self.assertEqual(metrics['hard_negative_topk_per_image'], [0, 4])
        self.assertEqual(metrics['hard_background_topk_per_image'], [0, 4])
        self.assertEqual(metrics['hard_negative_available_pixels_per_image'], [0, 4])
        self.assertAlmostEqual(sum(metrics[key+'_weighted'] for key in FOREGROUND_LOSS_WEIGHTS),
                               metrics['train_loss'], places=6)
        self.assertEqual(metrics['objective_protocol'], FOREGROUND_PROTOCOL)
        json.dumps(metrics, allow_nan=False)


class DetailEnemyRestorationObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def detail_loss(self, output, batch):
        return neural_enemy_restoration_loss(output, batch, protocol=DETAIL_PROTOCOL)

    def test_detail_revision_has_exact_fixed_weights_and_finite_weighted_total(self):
        output, labels = fixture()
        loss, metrics = self.detail_loss(output, labels)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(metrics['objective_protocol'], DETAIL_PROTOCOL)
        self.assertEqual(DETAIL_SETTINGS['effective_weights'], DETAIL_LOSS_WEIGHTS)
        self.assertEqual(DETAIL_SETTINGS['fixed_group_scales'],
                         dict(alpha_mask=.5, direct_RGB=4., composite_foreground_RGB=1., preservation=1.))
        self.assertEqual(DETAIL_SETTINGS['positive_topk_pixels'], 32)
        self.assertEqual(DETAIL_SETTINGS['negative_topk_pixels'], 32)
        self.assertEqual(DETAIL_SETTINGS['rgb_topk_pixels'], 32)
        self.assertEqual(DETAIL_SETTINGS['crop_scales'], ['native', 32, 64])
        self.assertFalse(DETAIL_SETTINGS['architecture_change'])
        self.assertFalse(DETAIL_SETTINGS['optimizer_reset'])
        self.assertFalse(DETAIL_SETTINGS['source_background_reconstruction_gradient'])
        self.assertTrue(DETAIL_SETTINGS['fixed_weights_without_EMA'])
        self.assertAlmostEqual(sum(metrics[key+'_weighted'] for key in DETAIL_LOSS_WEIGHTS),
                               metrics['train_loss'], places=6)
        selected = loss_weights(DETAIL_PROTOCOL); selected['enemy_rgb_l1'] = 99
        self.assertEqual(loss_weights(DETAIL_PROTOCOL)['enemy_rgb_l1'], 4.)
        self.assertEqual(loss_weights(PROTOCOL), LOSS_WEIGHTS)
        self.assertEqual(loss_weights(FOREGROUND_PROTOCOL), FOREGROUND_LOSS_WEIGHTS)
        json.dumps(DETAIL_SETTINGS, allow_nan=False); json.dumps(metrics, allow_nan=False)

    def test_source_background_changes_cannot_change_detail_or_crop_gradients(self):
        first, labels = fixture(alpha_logit=-.5, rgb_value=.4, box=(40, 44, 30, 42))
        second, changed = fixture(alpha_logit=-.5, rgb_value=.4, box=(40, 44, 30, 42))
        mask = changed['mask_hr'].detach()
        wall = torch.linspace(0, 1, 128)[None, None, None, :].expand_as(changed['hr_source'])
        changed['hr_source'] = (labels['hr_source'].detach()*mask + wall*(1-mask)).requires_grad_()
        changed['hr_background'] = (1-wall).clone().requires_grad_()
        loss, metrics = self.detail_loss(first, labels)
        altered, other = self.detail_loss(second, changed)
        self.assertEqual(float(loss.detach()), float(altered.detach()))
        for key in DETAIL_LOSS_WEIGHTS: self.assertEqual(metrics[key], other[key], key)
        self.assertNotEqual(metrics['source_image_l1_diagnostic'], other['source_image_l1_diagnostic'])
        loss.backward(); altered.backward()
        for key in ('enemy_rgb', 'alpha_logits'):
            torch.testing.assert_close(first[key].grad, second[key].grad, rtol=0, atol=0)
        self.assertTrue(all(value.grad is None for value in labels.values()))
        self.assertTrue(all(value.grad is None for value in changed.values()))

    def test_missing_foreground_hard_positive_only_raises_worst_supported_opacity(self):
        mask = torch.zeros(1, 1, 128, 128, dtype=torch.bool)
        mask[:, :, 40:48, 40:48] = True
        probability = torch.full(mask.shape, .1)
        probability[mask] = .99; probability[:, :, 40:44, 40:48] = .01
        logits = torch.logit(probability).detach().requires_grad_()
        values, present, counts = _topk_foreground(
            torch.nn.functional.softplus(-logits), mask, DETAIL_POSITIVE_TOPK_PIXELS)
        self.assertEqual(counts, [32]); self.assertTrue(bool(present.all()))
        values.mean().backward()
        missed = torch.zeros_like(mask); missed[:, :, 40:44, 40:48] = True
        self.assertTrue(bool((logits.grad[missed] < 0).all()))
        self.assertEqual(float(logits.grad[~missed].abs().sum()), 0.)
        self.assertGreater(float(values.detach()), 4.)
        # The extra support must come only from Source visibility: a guessed
        # location on an empty frame creates no positive/RGB target.
        empty = torch.zeros_like(mask)
        zero, present, counts = _topk_foreground(torch.nn.functional.softplus(-logits), empty, 32)
        self.assertEqual(counts, [0]); self.assertFalse(bool(present.any()))
        self.assertEqual(float(zero.detach()), 0.)

    def test_hard_rgb_prefers_the_worst_true_detail_instead_of_unlabelled_walls(self):
        support = torch.zeros(1, 1, 128, 128, dtype=torch.bool)
        support[:, :, 30:38, 40:48] = True
        error = torch.full(support.shape, 100.)
        error[support] = .1; error[:, :, 30:34, 40:48] = .8
        error.requires_grad_()
        values, _, counts = _topk_foreground(error, support, DETAIL_RGB_TOPK_PIXELS)
        self.assertEqual(counts, [32]); self.assertAlmostEqual(float(values.detach()), .8, places=6)
        values.mean().backward()
        worst = torch.zeros_like(support); worst[:, :, 30:34, 40:48] = True
        self.assertTrue(bool((error.grad[worst] > 0).all()))
        self.assertEqual(float(error.grad[~worst].abs().sum()), 0.)

    def test_positive_and_rgb_topk_cap_to_actual_foreground_and_normalize_per_image(self):
        mask = torch.zeros(4, 1, 128, 128, dtype=torch.bool)
        mask[0, :, 0, 0] = True; mask[1, :, 10:12, 10:14] = True
        mask[2, :, 40:48, 40:48] = True
        error = torch.ones(mask.shape, requires_grad=True)
        values, present, counts = _topk_foreground(error, mask, 32)
        self.assertEqual(counts, [1, 8, 32, 0]); self.assertEqual(present.tolist(), [True, True, True, False])
        self.assertEqual(values.detach().tolist(), [1., 1., 1., 0.])
        values[present].mean().backward()
        for index in range(3):
            self.assertAlmostEqual(float(error.grad[index].sum()), 1/3, places=6)
        self.assertEqual(float(error.grad[3].abs().sum()), 0.)

    def test_detail_raw_rgb_losses_and_gradients_survive_zero_predicted_opacity(self):
        hidden, labels = fixture(alpha_logit=-40.)
        shown, same_labels = fixture(alpha_logit=40.)
        loss, metrics = self.detail_loss(hidden, labels)
        _, other = self.detail_loss(shown, same_labels)
        direct = ('enemy_rgb_l1', 'enemy_rgb_edge', 'enemy_rgb_ssim', 'hard_foreground_rgb_l1',
                  'rgb_multiscale_crop_l1', 'rgb_multiscale_crop_edge', 'rgb_multiscale_crop_ssim')
        for key in direct:
            self.assertEqual(metrics[key], other[key], key); self.assertGreater(metrics[key], 0., key)
        loss.backward()
        gradient = hidden['enemy_rgb'].grad; mask = labels['mask_hr'].detach().expand_as(gradient)
        self.assertGreater(float((gradient.abs()*mask).sum()), 2.)
        self.assertLess(float((gradient.abs()*(1-mask)).sum()), 1e-10)

    def test_loss_only_crops_preserve_aspect_and_align_source_foreground(self):
        output, labels = fixture(count=1, alpha_logit=-1., rgb_value=.4, box=(40, 44, 30, 42))
        mask = labels['mask_hr'].detach(); background = output['background']
        appearance = output['enemy_rgb']*mask + background*(1-mask)
        reference = labels['hr_source'].detach()*mask + background*(1-mask)
        terms, shapes = _rgb_detail_crops(appearance, reference, mask)
        self.assertEqual(shapes, [[[8, 16], [16, 32], [32, 64]]])
        self.assertTrue(all(float(value.detach()) > 0 for value in terms.values()))
        sum(terms.values()).backward()
        gradient = output['enemy_rgb'].grad
        self.assertGreater(float((gradient.abs()*mask).sum()), 0.)
        self.assertEqual(float((gradient.abs()*(1-mask)).sum()), 0.)
        # A perfect native foreground remains perfect after the identical
        # transforms, including the same frozen context at occlusion edges.
        perfect = reference.detach().clone().requires_grad_()
        equal, _ = _rgb_detail_crops(perfect, reference, mask)
        for value in equal.values(): self.assertLess(abs(float(value.detach())), 1e-6)

    def test_tiny_corner_enemy_has_finite_native_and_scaled_detail_gradients(self):
        output, labels = fixture(count=1, box=(0, 1, 0, 1))
        loss, metrics = self.detail_loss(output, labels)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(metrics['hard_positive_topk_per_image'], [1])
        self.assertEqual(metrics['hard_foreground_rgb_topk_per_image'], [1])
        self.assertEqual(metrics['detail_crop_shapes_per_image'], [[[3, 3], [32, 32], [64, 64]]])
        self.assertGreater(metrics['rgb_multiscale_crop_ssim'], 0.)
        loss.backward()
        for key in ('enemy_rgb', 'alpha_logits'):
            self.assertTrue(bool(torch.isfinite(output[key].grad).all()))

    def test_empty_images_have_no_positive_or_detail_rgb_source_targets(self):
        first, labels = fixture(count=1, alpha_logit=0., rgb_value=.8, box=(0, 0, 0, 0))
        second, changed = fixture(count=1, alpha_logit=0., rgb_value=.8, box=(0, 0, 0, 0))
        labels['hr_source'] = torch.full_like(labels['hr_source'], .99, requires_grad=True)
        changed['hr_source'] = torch.full_like(changed['hr_source'], .01, requires_grad=True)
        changed['hr_background'] = torch.full_like(changed['hr_background'], .85, requires_grad=True)
        loss, metrics = self.detail_loss(first, labels); altered, _ = self.detail_loss(second, changed)
        for key in ('positive_alpha_consistency', 'hard_positive_alpha_bce', 'hard_foreground_rgb_l1',
                    'rgb_multiscale_crop_l1', 'rgb_multiscale_crop_edge', 'rgb_multiscale_crop_ssim',
                    'enemy_rgb_l1', 'enemy_rgb_edge', 'enemy_rgb_ssim', 'foreground_image_l1',
                    'foreground_image_ssim', 'foreground_image_edge'):
            self.assertEqual(metrics[key], 0., key)
        self.assertEqual(metrics['detail_crop_shapes_per_image'], [[]])
        self.assertEqual(metrics['hard_positive_topk_per_image'], [0])
        self.assertEqual(metrics['hard_foreground_rgb_topk_per_image'], [0])
        loss.backward(); altered.backward()
        for key in ('enemy_rgb', 'alpha_logits'):
            torch.testing.assert_close(first[key].grad, second[key].grad, rtol=0, atol=0)
        self.assertGreater(float(first['enemy_rgb'].grad.sum()), 0.)


if __name__ == '__main__': unittest.main()
