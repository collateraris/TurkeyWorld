from copy import deepcopy
import unittest

import torch

from turkeyworld.background_detail_objectives import (background_detail_loss,LOSS_WEIGHTS,
    PROTOCOL,low_frequency_image_change,high_frequency_image)


def case(image_value=.4,target_value=.2):
    image=torch.full((1,3,128,128),image_value,requires_grad=True)
    output=dict(image=image,background=torch.full_like(image,.4),residual=torch.zeros_like(image,requires_grad=True))
    return output,dict(hr_source=torch.full_like(image,target_value),mask_hr=torch.zeros(1,1,128,128))


class BackgroundDetailObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_empty_mask_cannot_hide_Source_background_errors_or_full_RGB_gradient(self):
        output,batch=case();loss,metrics=background_detail_loss(output,batch);loss.backward()
        self.assertGreater(float(loss.detach()),0);self.assertGreater(metrics['image_l1'],0);self.assertGreater(metrics['background_l1'],0)
        self.assertEqual(metrics['foreground_l1'],0);self.assertEqual(metrics['visible_images'],0)
        self.assertEqual(metrics['empty_images'],1);self.assertGreater(float(output['image'].grad.abs().sum()),0)
        self.assertFalse(metrics['predicted_mask_gates_loss']);self.assertFalse(metrics['alpha_used'])

    def test_foreground_error_is_area_normalized_for_small_and_large_enemy(self):
        values=[]
        for size in(2,32):
            image=torch.zeros(1,3,128,128);image[:,:,10:10+size,20:20+size]=.25;image.requires_grad_()
            mask=torch.zeros(1,1,128,128);mask[:,:,10:10+size,20:20+size]=1
            output=dict(image=image,background=torch.zeros_like(image),residual=image)
            _,metrics=background_detail_loss(output,dict(hr_source=torch.zeros_like(image),mask_hr=mask))
            values.append(metrics['foreground_l1'])
            self.assertEqual(metrics['visible_images'],1);self.assertGreater(metrics['foreground_edge'],0)
        self.assertEqual(values,[.25,.25])

    def test_lowfreq_penalty_preserves_base_color_but_allows_zero_mean_detail(self):
        base=torch.full((1,3,128,128),.5);yy,xx=torch.meshgrid(torch.arange(128),torch.arange(128),indexing='ij')
        checker=((xx+yy)%2*2-1)[None,None].expand(1,3,-1,-1).float()*.05
        self.assertGreater(float(high_frequency_image(base+checker).abs().mean()),.04)
        self.assertLess(float(low_frequency_image_change(base+checker,base).max()),1e-6)
        self.assertAlmostEqual(float(low_frequency_image_change(base+.1,base)),.1,places=5)
        self.assertLess(float(high_frequency_image(base+.1).abs().mean()),1e-6)

    def test_details_are_supervised_on_Source_without_source_or_mask_gradients(self):
        output,batch=case();batch['mask_hr'][:,:,12:20,12:20]=1
        batch['hr_source'].requires_grad_();batch['mask_hr'].requires_grad_()
        loss,metrics=background_detail_loss(output,batch);loss.backward()
        self.assertIsNone(batch['hr_source'].grad);self.assertIsNone(batch['mask_hr'].grad)
        self.assertIsNone(output['background'].grad)
        self.assertTrue(metrics['full_image_supervision']);self.assertTrue(metrics['source_background_reconstruction_gradient'])
        self.assertEqual(metrics['background_edge_scope'],'outside_NPC_not_walls_only')

    def test_loss_components_are_weighted_once_and_perfect_zero_image_has_zero_error(self):
        output,batch=case();loss,metrics=background_detail_loss(output,batch)
        self.assertAlmostEqual(float(loss.detach()),sum(metrics[k]*v for k,v in LOSS_WEIGHTS.items()),places=6)
        self.assertAlmostEqual(float(loss.detach()),sum(metrics[k+'_weighted']for k in LOSS_WEIGHTS),places=6)
        self.assertEqual(metrics['objective_protocol'],PROTOCOL)
        output,batch=case(0.,0.);output['background'].zero_()
        loss,metrics=background_detail_loss(output,batch)
        self.assertEqual(float(loss.detach()),0);self.assertEqual(metrics['residual_l1'],0)

    def test_invalid_or_trainable_background_and_nonfinite_labels_fail_closed(self):
        for mutate in(lambda o,b:o['background'].requires_grad_(),
                      lambda o,b:b['hr_source'].fill_(float('nan')),
                      lambda o,b:b['mask_hr'].fill_(-1),
                      lambda o,b:o.pop('residual')):
            output,batch=case();mutate(output,batch)
            with self.assertRaises(ValueError):background_detail_loss(output,batch)


if __name__=='__main__':unittest.main()
