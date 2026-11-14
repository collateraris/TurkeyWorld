"""The revised objective retains the three-stream quotas and group accounting."""
from copy import deepcopy
import json
import unittest

from turkeyworld.rssm_accumulation import accumulated_metrics
from turkeyworld.rssm_player_navigation_revision import PROTOCOL, player_navigation_settings
from turkeyworld.train_rssm_moe import merge_main_replay_metrics, merge_player_contact_metrics


class PlayerNavigationIntegrationTests(unittest.TestCase):
    def streams(self):
        return (
            dict(moe_player_loss=2., moe_enemy_loss=3., moe_combat_loss=4.,
                 moe_router_role_loss=.1, moe_router_balance_loss=.01,
                 moe_direct_support_gate_eligible=True, moe_main_tbptt=64,
                 moe_auxiliary=dict(unchanged=True), moe_late_own_prefix=dict(unchanged=True)),
            dict(nav_player_loss=.4, nav_router_role_loss=.02, nav_router_balance_loss=.002,
                 nav_player_xy_loss=.3, nav_player_wall_loss=.1, nav_tbptt=64),
            dict(enemy_motion_enemy_loss=.8, enemy_motion_player_loss=.6,
                 enemy_motion_player_xy_loss=.5, enemy_motion_player_wall_loss=.1,
                 enemy_motion_router_role_loss=.03, enemy_motion_router_balance_loss=.003,
                 enemy_motion_tbptt=64),
        )

    def merge(self, *streams, enabled=True):
        return merge_main_replay_metrics(*streams, plan_sha256='a' * 64,
            layout=dict(combat=19, navigation=5, enemy_motion=8),
            auxiliary_batch_size=32, late_actor_count=4,
            player_navigation_protocol=player_navigation_settings() if enabled else None)

    def test_quiet_player_added_once_without_diluting_enemy_combat_or_router(self):
        streams = self.streams(); before = deepcopy(streams)
        old = self.merge(*streams, enabled=False)
        current = self.merge(*streams)
        self.assertEqual(streams, before)
        self.assertAlmostEqual(old['moe_player_loss'], 2.4)
        self.assertAlmostEqual(current['moe_player_loss'], 3.)
        self.assertAlmostEqual(current['train_loss'] - old['train_loss'], .6)
        for name in ('moe_enemy_loss', 'moe_combat_loss', 'moe_router_loss',
                     'moe_auxiliary', 'moe_late_own_prefix', 'moe_direct_support_gate_eligible'):
            self.assertEqual(current[name], old[name])
        self.assertEqual((current['main_combat_actor_count'], current['main_navigation_actor_count'],
                          current['main_enemy_motion_actor_count']), (19, 5, 8))
        self.assertEqual(current['main_actor_count'], 32)
        self.assertEqual(current['combat_auxiliary_actor_count'], 32)
        self.assertAlmostEqual(current['main_navigation_fraction'], 5/32)
        self.assertAlmostEqual(current['main_enemy_motion_fraction'], 8/32)
        self.assertAlmostEqual(current['main_combat_fraction'], 19/32)

    def test_contact_objective_conserves_total_and_exposure_without_overwriting_main_protocol(self):
        base = self.merge(*self.streams()); before = deepcopy(base)
        base['protocol'] = 'main_moe_protocol'
        contact = dict(player_contact_loss=.125, player_contact_horizon=8, player_contact_tbptt=64,
                       protocol='contact_protocol', scout_forward_transitions=128,
                       forward_transitions=32, selected_prefix_steps=[16, 32, 48],
                       memory_resets_after_initialization=0)
        final = merge_player_contact_metrics(base, contact, plan_sha256='b' * 64)
        self.assertEqual(final['protocol'], 'main_moe_protocol')
        self.assertEqual(final['player_contact_replay'], contact)
        self.assertAlmostEqual(final['moe_player_loss'], before['moe_player_loss'] + .125)
        self.assertAlmostEqual(final['train_loss'], before['train_loss'] + .125)
        self.assertEqual(final['player_contact_generated_transitions'], 160)
        self.assertEqual(final['player_contact_branch_forward_transitions'], 32)
        self.assertEqual(final['player_contact_horizon'], 8)
        self.assertEqual(final['player_contact_tbptt'], 64)
        self.assertEqual(final['objective_formula_revision'], PROTOCOL)
        self.assertEqual(final['player_navigation_loss_revision'], PROTOCOL)
        for name in ('moe_enemy_loss', 'moe_combat_loss', 'moe_router_loss'):
            self.assertEqual(final[name], before[name])
        self.assertAlmostEqual(final['loss_groups']['total'], final['train_loss'])
        self.assertEqual(accumulated_metrics([final]), final)
        json.dumps(final, allow_nan=False)

    def test_invalid_or_unknown_new_objective_is_rejected(self):
        for value in (None, True, -.1, float('nan'), float('inf')):
            with self.subTest(contact=value):
                with self.assertRaises(ValueError):
                    merge_player_contact_metrics(self.merge(*self.streams()),
                        dict(player_contact_loss=value), plan_sha256='b' * 64)
        bad = player_navigation_settings(); bad['tbptt'] = 16
        with self.assertRaises(ValueError):
            merge_main_replay_metrics(*self.streams(), plan_sha256='a' * 64,
                                       player_navigation_protocol=bad)


if __name__ == '__main__': unittest.main()
