"""Group accounting covers actual dual prior/posterior graphs without resets."""
from copy import deepcopy
import unittest

import torch

from test_dual_rssm_backend import model
from test_dual_rssm_objectives import source_batch
from turkeyworld.rssm_loss_balance import GroupLossAccumulator
from turkeyworld.rssm_migration import configure_rssm_training
from turkeyworld.rssm_objectives import rssm_dynamics_backward


class GroupedRSSMObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def backward(self, world, batch, groups=None):
        configure_rssm_training(world);world.core.eval()
        return rssm_dynamics_backward(world,batch,context=1,observed_context=1,
            auxiliary_batch=batch,auxiliary_context=1,auxiliary_observed_context=1,
            expected_horizon=4,chunk_size=2,numeric_state_feedback='detached',
            combat_protocol={'target_supervision':'shot_distance_balanced'},loss_groups=groups)

    def test_unit_group_coefficients_conserve_every_prior_posterior_combat_gradient_and_floor(self):
        initial=model(True);batch=source_batch();legacy=deepcopy(initial);grouped=deepcopy(initial)
        a=self.backward(legacy,batch);b=self.backward(grouped,batch,GroupLossAccumulator())
        self.assertAlmostEqual(a['train_loss'],b['train_loss'],places=5)
        self.assertAlmostEqual(b['train_loss_raw'],a['train_loss'],places=7)
        self.assertAlmostEqual(sum(b['loss_groups'][name]['weighted']for name in ('player','enemy','interaction')),
            b['train_loss'],places=7)
        self.assertAlmostEqual(b['loss_groups']['kl_floor'],2.2)
        self.assertAlmostEqual(b['train_loss_without_kl_floor'],b['train_loss']-2.2)
        self.assertGreater(b['loss_groups']['player']['raw'],0)
        self.assertGreater(b['loss_groups']['enemy']['raw'],0)
        self.assertGreater(b['loss_groups']['interaction']['raw'],0)
        old=dict(legacy.named_parameters())
        for name,value in grouped.named_parameters():
            other=old[name]
            self.assertEqual(value.grad is None,other.grad is None,name)
            if value.grad is not None:
                torch.testing.assert_close(value.grad,other.grad,rtol=2e-5,atol=2e-6,msg=name)

    def test_normalization_changes_actual_gradient_not_only_report_and_keeps_causal_rollout(self):
        initial=model(True);batch=source_batch()
        a=self.backward(deepcopy(initial),batch,GroupLossAccumulator())
        scaled=deepcopy(initial)
        b=self.backward(scaled,batch,GroupLossAccumulator(scales={'player':2.,'enemy':4.,'interaction':8.}))
        for name in ('player','enemy','interaction'):
            self.assertAlmostEqual(a['loss_groups'][name]['raw'],b['loss_groups'][name]['raw'],places=6)
            row=b['loss_groups'][name]
            self.assertAlmostEqual(row['weighted'],row['raw']/row['scale']+row['kl_floor'])
        self.assertLess(b['train_loss'],a['train_loss'])
        self.assertAlmostEqual(b['train_loss_raw'],a['train_loss_raw'],places=6)
        self.assertFalse(b['rssm_posterior_feedback_to_prior'])
        self.assertEqual(b['rssm_prior_future_rgb_inputs'],0)
        self.assertGreater(sum(float(p.grad.abs().sum())for p in scaled.core.parameters()if p.grad is not None),0)
        self.assertGreater(sum(float(p.grad.abs().sum())for p in scaled.vjepa_backend.enemy_rssm.parameters()if p.grad is not None),0)


if __name__=='__main__':unittest.main()
