import unittest
from unittest.mock import patch

import torch

from turkeyworld.object_losses import balanced_mse
from turkeyworld.renderer_enemy_objectives import enemy_refinement_loss
from test_renderer_enemy_objectives import tiny_model, batch


class RendererEnemyLatentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_optional_latent_target_is_frozen_loss_only_with_actual_rf_gradient_and_no_base_gradient(self):
        model = tiny_model()
        source = batch((False, True))
        for name, value in source.items():
            if value.is_floating_point():
                source[name] = value.requires_grad_(True)
        before = {name: value.clone() for name, value in model.flow_renderer.codec.state_dict().items()}
        base_before = {name: value.clone() for name, value in model.base_student.state_dict().items()}
        flow_before = model.flow_renderer.output.weight.clone()
        seen = {}
        def capture(prediction, target, mask):
            seen['prediction'], seen['target'] = prediction, target
            seen['loss'] = balanced_mse(prediction, target, mask)
            return seen['loss']
        with patch('turkeyworld.renderer_enemy_objectives.balanced_mse', side_effect=capture):
            total, metrics = enemy_refinement_loss(model, source, latent_weight=.1)
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(seen['prediction'].requires_grad)
        self.assertFalse(seen['target'].requires_grad)
        seen['loss'].backward()  # Isolate the new term from existing Base/RGB losses.
        self.assertGreater(float(model.flow_renderer.output.weight.grad.abs().sum()), 0)
        self.assertTrue(all(value.grad is None for value in model.base_student.parameters()))
        self.assertTrue(all(value.grad is None for name, value in model.named_parameters() if not value.requires_grad))
        self.assertTrue(all(value.grad is None for value in source.values() if value.is_floating_point()))
        optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad], lr=1e-4)
        optimizer.step()
        self.assertFalse(torch.equal(flow_before, model.flow_renderer.output.weight))
        self.assertTrue(all(torch.equal(value, model.flow_renderer.codec.state_dict()[name]) for name, value in before.items()))
        self.assertTrue(all(torch.equal(value, model.base_student.state_dict()[name]) for name, value in base_before.items()))
        self.assertTrue(metrics['source_latent_active'])
        self.assertEqual(metrics['source_latent_weight'], .1)
        self.assertAlmostEqual(metrics['weighted_source_latent_mse'], .1*metrics['source_latent_mse'])

    def test_default_zero_preserves_loss_exactly_and_base_only_never_encodes_a_source_target(self):
        model = tiny_model()
        source = batch((False,))
        default, _ = enemy_refinement_loss(model, source)
        explicit, metrics = enemy_refinement_loss(model, source, latent_weight=0.)
        self.assertTrue(torch.equal(default, explicit))
        self.assertFalse(metrics['source_latent_active'])
        self.assertNotIn('source_latent_mse', metrics)
        with patch.object(model.flow_renderer.codec, 'encode', side_effect=AssertionError('No codec in Base-only')):
            loss, metrics = enemy_refinement_loss(model, source, base_only=True, latent_weight=.1)
            self.assertTrue(torch.isfinite(loss))
        self.assertFalse(metrics['source_latent_active'])
        self.assertEqual(metrics['source_latent_weight'], 0)


if __name__ == '__main__':
    unittest.main()
