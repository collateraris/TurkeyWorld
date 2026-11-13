"""Gain-independent ROI GAN gradients and honest conditional labels."""
from copy import deepcopy
import json
import unittest

import torch

from test_decoder_restoration import outputs, model
from turkeyworld.sr_enemy_patchgan import (EnemyPatchDiscriminator, build_enemy_patch_pairs,
    discriminator_loss, generator_loss, make_enemy_patch_optimizer, reduce_enemy_patch_values)


def fixture(*, empty=False):
    output, batch = outputs(empty=empty)
    batch['lr_npr'] = torch.full((4, 3, 32, 32), .2, requires_grad=True)
    state = torch.zeros(4, 20)
    state[:, 3:5] = 1
    state[:, 8] = torch.tensor([1., 3., 7., 9.]) / 16
    state[:, 12:14] = 1
    batch['state'] = state.requires_grad_()
    return output, batch


def small_discriminator():
    return EnemyPatchDiscriminator(channels=(4, 8, 16, 16))


class EnemyPatchGANTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_default_parameter_budget_and_patch_features(self):
        d = EnemyPatchDiscriminator()
        self.assertEqual(sum(p.numel() for p in d.parameters()), 206889)
        logits, features = d(torch.rand(2, 3, 64, 64), torch.rand(2, 4, 64, 64), return_features=True)
        self.assertEqual(logits.shape, (2, 1, 8, 8))
        self.assertEqual([tuple(x.shape[-2:]) for x in features], [(32,32),(16,16),(8,8),(8,8)])
        self.assertTrue(all(hasattr(block[0], 'weight_u') for block in d.blocks))
        optimizer = make_enemy_patch_optimizer(d)
        self.assertEqual(optimizer.param_groups[0]['lr'], 1e-4)
        self.assertEqual(optimizer.param_groups[0]['betas'], (0., .99))

    def test_same_input_canvas_and_actual_box_no_Source_box_or_RGB_condition_leak(self):
        output, batch = fixture()
        actual = torch.full_like(batch['lr_npr'], .6, requires_grad=True)
        pairs = build_enemy_patch_pairs(output, batch, low_rgb=actual)
        self.assertEqual((pairs['condition'].shape, pairs['real'].shape, pairs['fake'].shape),
                         ((8,3,64,64),(8,4,64,64),(8,4,64,64)))
        torch.testing.assert_close(pairs['condition'], torch.full_like(pairs['condition'], .6))
        empty_alpha = (pairs['real'][:, 3:] == 0).expand(-1,3,-1,-1)
        torch.testing.assert_close(pairs['real'][:, :3][empty_alpha], pairs['condition'][empty_alpha], rtol=0, atol=0)
        self.assertTrue(pairs['fake'].requires_grad)
        self.assertFalse(pairs['condition'].requires_grad)
        self.assertFalse(pairs['real'].requires_grad)
        poison = {**batch, 'hr_source': torch.ones_like(batch['hr_source']),
                  'mask_hr': torch.zeros_like(batch['mask_hr']), 'bbox_hr': torch.ones_like(batch['bbox_hr'])}
        other = build_enemy_patch_pairs(output, poison, low_rgb=actual)
        torch.testing.assert_close(pairs['condition'], other['condition'], rtol=0, atol=0)
        torch.testing.assert_close(pairs['fake'], other['fake'], rtol=0, atol=0)
        self.assertEqual(pairs['receipt']['visible_distance_frames'], {'near':1,'mid':1,'far':1})
        self.assertEqual(pairs['receipt']['source_empty_frames'], 1)
        json.dumps(pairs['receipt'], allow_nan=False)

    def test_D_backward_detaches_generator_and_all_Source_labels_and_has_own_optimizer(self):
        output, batch = fixture()
        pairs = build_enemy_patch_pairs(output, batch)
        d = small_discriminator(); opt = make_enemy_patch_optimizer(d)
        before = {k:v.clone() for k,v in d.named_parameters()}
        loss, metrics = discriminator_loss(d, pairs)
        loss.backward(); opt.step()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in d.parameters() if p.grad is not None), 0.)
        self.assertTrue(any(not torch.equal(value, dict(d.named_parameters())[key]) for key,value in before.items()))
        for key in ('visual_rgba','enemy_rgba','visual_bbox','enemy_bbox'):
            self.assertIsNone(output[key].grad, key)
        for key in ('hr_source','hr_background','mask_hr','mask_lr','bbox_hr','hypothesis','state','lr_npr'):
            self.assertIsNone(batch[key].grad, key)
        self.assertEqual(metrics['source_visible_frames'], 3)
        self.assertTrue(torch.isfinite(loss))
        json.dumps(metrics, allow_nan=False)

    def test_G_has_RGB_alpha_gradients_at_zero_gain_and_D_weights_buffers_flags_modes_stay_exact(self):
        torch.manual_seed(173)
        m = model().train()
        _, batch = fixture()
        inputs = batch['lr_npr'].detach()
        output = m.forward_with_aux(inputs, None)
        self.assertEqual(output['restoration_blend_scale'], 0.)
        output['visual_bbox'].retain_grad()
        pairs = build_enemy_patch_pairs(output, batch, low_rgb=inputs)
        d = small_discriminator()
        # Establish spectral norm vectors using a discriminator-only pass.
        discriminator_loss(d, pairs)
        d.blocks[1].eval(); d.logits.bias.requires_grad_(False)
        weights = deepcopy(d.state_dict())
        modes = [module.training for module in d.modules()]
        flags = [p.requires_grad for p in d.parameters()]
        loss, metrics = generator_loss(d, pairs, feature_weight=.1)
        loss.backward()
        final = m.enemy_decoder_rgba[-1].weight.grad
        self.assertGreater(float(final[:3].abs().sum()), 0.)
        self.assertGreater(float(final[3:].abs().sum()), 0.)
        self.assertTrue(torch.isfinite(final).all())
        self.assertTrue(all(p.grad is None for p in d.parameters()))
        self.assertEqual(modes, [module.training for module in d.modules()])
        self.assertEqual(flags, [p.requires_grad for p in d.parameters()])
        for key, value in weights.items(): torch.testing.assert_close(value, d.state_dict()[key], rtol=0, atol=0)
        self.assertIsNone(output['visual_bbox'].grad)
        self.assertTrue(all(p.grad is None for name,p in m.named_parameters() if not name.startswith('enemy_decoder_')))
        for key in ('hr_source','hr_background','mask_hr','mask_lr','bbox_hr','hypothesis','state','lr_npr'):
            self.assertIsNone(batch[key].grad, key)
        self.assertEqual(metrics['adv_loss'], metrics['enemy_patch_g_hinge'])
        self.assertAlmostEqual(float(loss.detach()), metrics['adv_loss'] + .1*metrics['feature_matching_loss'], places=6)
        self.assertTrue(metrics['enemy_patch_generator_before_gain'])
        json.dumps(metrics, allow_nan=False)

    def test_missed_proposals_and_true_empty_rows_are_background_targets_without_GT_rescue(self):
        output, batch = fixture()
        wrong = torch.tensor([[0., 0., .1, .1]] * 4, requires_grad=True)
        output.update(visual_bbox=wrong, enemy_bbox=wrong)
        pairs = build_enemy_patch_pairs(output, batch)
        self.assertEqual(pairs['receipt']['missed_visible_box_pairs'], 6)
        self.assertEqual(float(pairs['real'][:,3:].sum()), 0.)
        torch.testing.assert_close(pairs['real'][:,:3], pairs['condition'], rtol=0, atol=0)
        torch.testing.assert_close(pairs['boxes'], wrong.detach().repeat(2,1), rtol=0, atol=0)
        loss, _ = generator_loss(small_discriminator(), pairs)
        loss.backward()
        self.assertIsNone(wrong.grad)
        self.assertIsNone(batch['bbox_hr'].grad)
        empty_output, empty_batch = fixture(empty=True)
        empty = build_enemy_patch_pairs(empty_output, empty_batch)
        self.assertEqual(empty['receipt']['source_empty_frames'], 4)
        self.assertEqual(float(empty['real'][:,3:].sum()), 0.)
        dloss, _ = discriminator_loss(small_discriminator(), empty)
        self.assertTrue(torch.isfinite(dloss))

    def test_one_HR_pixel_far_enemy_survives_ROI_sampling_and_is_not_filtered_out(self):
        output, batch = fixture(empty=True)
        with torch.no_grad():
            batch['mask_hr'][0,0,65,66] = 1
            batch['hr_source'][0,:,65,66] = torch.tensor([.8,.2,.1])
            batch['state'][0,8] = 8/16
        output['visual_bbox'] = torch.tensor([[0.,0.,1.,1.]]*4)
        output['enemy_applicable'] = torch.zeros(4,1,dtype=torch.bool)
        pairs = build_enemy_patch_pairs(output, batch)
        self.assertEqual(pairs['receipt']['source_visible_frames'], 1)
        self.assertEqual(pairs['receipt']['visible_distance_frames'], {'near':0,'mid':0,'far':1})
        self.assertEqual(pairs['receipt']['subpixel_native_visible_frames'], 1)
        self.assertAlmostEqual(float(pairs['real'][0,3:].sum()), .25, places=7)
        self.assertEqual(pairs['receipt']['pairs'], 4)

    def test_class_distance_and_route_normalization_does_not_duplicate_a_hinted_frame(self):
        # near3, mid1, far1, empty3 -> mean(visible-distance-mean3, empty7)=5.
        values = torch.tensor([1.,1.,1.,3.,5.,7.,7.,7.,1.], requires_grad=True)
        pairs = dict(source_count=8, source_index=torch.tensor([0,1,2,3,4,5,6,7,0]),
            source_visible=torch.tensor([1,1,1,1,1,0,0,0],dtype=torch.bool),
            distance_band=torch.tensor([0,0,0,1,2,0,1,2]))
        loss = reduce_enemy_patch_values(values, pairs)
        self.assertEqual(float(loss.detach()), 5.)
        loss.backward()
        self.assertAlmostEqual(float(values.grad[0]), 1/36, places=7)
        self.assertAlmostEqual(float(values.grad[8]), 1/36, places=7)
        self.assertAlmostEqual(float(values.grad[3]), 1/6, places=7)
        self.assertAlmostEqual(float(values.grad[5]), 1/6, places=7)

    def test_all_invalid_boxes_skip_honestly_without_fabricating_a_crop(self):
        output, batch = fixture()
        output['visual_bbox'] = torch.zeros(4,4)
        output['enemy_bbox'] = torch.zeros(4,4)
        pairs = build_enemy_patch_pairs(output, batch)
        self.assertEqual(pairs['receipt']['valid_source_frames'], 0)
        self.assertEqual(pairs['receipt']['pairs'], 0)
        d = small_discriminator()
        for helper in (discriminator_loss, generator_loss):
            loss, metrics = helper(d, pairs)
            self.assertEqual(float(loss.detach()), 0.)
            self.assertTrue(metrics['enemy_patch_skipped_no_valid_boxes'])


if __name__ == '__main__': unittest.main()
