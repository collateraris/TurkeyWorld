from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import torch

from test_renderer_enemy_objectives import batch
from test_renderer_moe_five import trained_three
from turkeyworld.renderer_moe import migrate_renderer_moe_five, configure_renderer_moe
from turkeyworld.renderer_moe_five_objectives import (source_five_role_labels, five_token_role_loss,
                                                     moe_five_refinement_loss)
from turkeyworld.renderer_refinement import frozen_fingerprint


class RendererMoEFiveObjectivesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_five_masks_partition_depth5_tints_hp0_corpse_attack_events_and_do_not_read_target_rgb(self):
        source = batch((False,) * 8)
        # Keep one pixel of real visible NPC away from the mocked gun.
        source['object_mask'].zero_(); source['object_mask'][:, :, 4, 4] = 1
        source['state'][:, 4] = 1; source['state'][:, 12] = 1
        source['state'][:, 13] = 1; source['state'][:, 14] = 0
        source['state'][:, 15] = 0; source['state'][:, 18] = 0; source['state'][:, 7] = 0
        source['state'][1, 15] = 1       # NPC hurt only, never player tint.
        source['state'][2, 13] = 0      # corpse remains enemy-combat.
        source['state'][3, 12] = 0      # NPC HP0.
        source['state'][4, 4] = 0       # global player HP0 tint.
        source['state'][5, 18] = 1      # A kill counter must never become a full-frame tint.
        source['source_enemy_combat_event'] = torch.tensor([False] * 6 + [True, False])
        source['source_player_hurt_event'] = torch.tensor([False] * 7 + [True])
        source['dynamic'].zero_()
        depth = torch.tensor([4.99, 5.] * 16).reshape(1, 1, 1, 32).expand(8, -1, -1, -1)
        gun = torch.zeros(32, 32).numpy(); gun[31, 31] = 1
        with patch('turkeyworld.renderer_wall_depth.source_wall_depth_labels', return_value={'depth': depth}), \
                patch('turkeyworld.renderer_wall_supervision._gun_masks', return_value=(gun, gun, gun)):
            labels = source_five_role_labels(source, device='cpu')
            self.assertTrue(torch.equal(labels.sum(1), torch.ones(8, 32, 32)))
            self.assertEqual(float(labels[0, 0, 0, 0]), 1.)
            self.assertEqual(float(labels[0, 1, 0, 1]), 1.)
            self.assertEqual(float(labels[0, 2, 4, 4]), 1.)
            self.assertTrue(all(labels[i, 3, 4, 4] == 1 for i in (1, 2, 3, 6)))
            self.assertTrue(bool((labels[4, 4] == 1).all()))
            self.assertTrue(torch.equal(labels[5], labels[0]))
            self.assertEqual(float(labels[7, 4, 31, 31]), 1.)
            self.assertEqual(float(labels[7, 4, 0, 0]), 0.)
            self.assertEqual(float(labels[0, 0, 31, 31]), 1.)  # idle gun despite depth5.
            self.assertEqual(float(labels[0, 4, 31, 31]), 0.)
            self.assertEqual(float(labels[1, 4, 0, 0]), 0.)
            poisoned = deepcopy(source); poisoned['target'] = 1 - poisoned['target']
            self.assertTrue(torch.equal(labels, source_five_role_labels(poisoned, device='cpu')))
            shooting = deepcopy(source); shooting['state'][0, 7] = 1
            shot = source_five_role_labels(shooting, device='cpu')
            self.assertEqual(float(shot[0, 4, 31, 31]), 1.)
            self.assertEqual(float(shot[0, 0, 31, 31]), 0.)
            attack = deepcopy(source); attack['state'][0, 14] = .9
            self.assertEqual(float(source_five_role_labels(attack, device='cpu')[0, 3, 4, 4]), 1.)

    def test_absent_roles_excluded_and_single_pixel_is_not_diluted_with_source_grad_stopped(self):
        labels = torch.zeros(1, 5, 32, 32, requires_grad=True)
        with torch.no_grad(): labels[:, 0] = 1; labels[:, 0, 0, 0] = 0; labels[:, 3, 0, 0] = 1
        logits = torch.zeros(1, 4, 5, requires_grad=True)
        loss, balance, info = five_token_role_loss(logits, logits.softmax(-1), labels)
        self.assertEqual(info['present_roles'], 2); self.assertGreater(info['label_mass'][3], 0)
        self.assertAlmostEqual(float(loss.detach()), float(torch.tensor(5.).log()), places=6)
        (loss + balance).backward()
        self.assertIsNone(labels.grad); self.assertGreater(float(logits.grad[..., 3].abs().sum()), 0)

    def test_actual_main_far_and_geometry_backward_reaches_all_experts_routers_with_source_and_frozen_none(self):
        model = trained_three(); migrate_renderer_moe_five(model); configure_renderer_moe(model)
        source = batch((False, False)); source['navigation_only'] = torch.tensor([False, True])
        for value in source.values():
            if value.is_floating_point(): value.requires_grad_()
        frozen = frozen_fingerprint(model)
        for kind in ('main', 'far', 'geometry'):
            model.zero_grad(set_to_none=True)
            with patch('turkeyworld.engine.Game.render', side_effect=AssertionError('Loss labels only')):
                loss, metrics = moe_five_refinement_loss(model, source, kind=kind)
                loss.backward()
            self.assertTrue(torch.isfinite(loss))
            for head in (model.base_student, model.flow_renderer):
                for expert in head.output_moe.experts:
                    self.assertGreater(sum(float(p.grad.abs().sum()) for p in expert.parameters() if p.grad is not None), 0)
                self.assertGreater(float(head.output_moe.router.weight.grad.abs().sum()), 0)
                self.assertGreater(float(head.output_moe.split_router.weight.grad.abs().sum()), 0)
            self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
            self.assertTrue(all(v.grad is None for v in source.values() if v.is_floating_point()))
            self.assertEqual(frozen, frozen_fingerprint(model)); json.dumps(metrics, allow_nan=False)
            for info in metrics['renderer_moe_heads'].values():
                self.assertEqual(len(info['gates_mean']), 5)
                self.assertEqual(len(info['parent_gates_mean']), 3)
                self.assertEqual(len(info['split_gates_mean']), 2)


if __name__ == '__main__': unittest.main()
