"""The optional full-graph plan cannot reinterpret the immutable64/B32 plan."""
from copy import deepcopy
import json
import unittest

import test_rssm_player_navigation_revision as legacy
from turkeyworld.rssm_moe_batch_revision import (
    micro16_single_batch_revision_plan,bind_moe_batch_revision,
)
from turkeyworld.rssm_player_navigation_revision import (
    PROTOCOL,FULL_HORIZON_PROTOCOL,file_sha,digest,player_navigation_settings,
    player_navigation_settings_for_plan,player_navigation_plan,
    bind_player_navigation_revision,mark_player_navigation_baseline,commit_player_navigation_revision,
)
from turkeyworld.train_rssm_moe import merge_main_replay_metrics,merge_player_contact_metrics


class FullHorizonPlayerRevisionTests(unittest.TestCase):
    def setUp(self):
        self.old=legacy.PlayerNavigationRevisionTests('runTest');self.old.setUp()
        self.addCleanup(self.old.doCleanups)
        self.state,self.source,self.root=self.old.state,self.old.source,self.old.root
        self.old_plan_bytes=self.old.path.read_bytes();self.raw_gate=deepcopy(self.state['rssm_training']['moe_training'])
        batch=micro16_single_batch_revision_plan(self.state,source_checkpoint=self.source)
        path=self.old.write('batch16_single.json',batch)
        bind_moe_batch_revision(self.state,batch,path=path,sha256=file_sha(path),
            loaded_checkpoint_sha256=file_sha(self.source))
        self.plan=player_navigation_plan(self.state,source_checkpoint=self.source,
            source_checkpoint_sha256=file_sha(self.source),contact_root=self.old.contact,
            contact_sampler_identity=self.old.identity,tbptt=450,microbatch_size=16)
        self.path=self.old.write('player_navigation_tbptt450_batch16_plan.json',self.plan)

    def bind(self,*,state=None,plan=None,path=None):
        path=self.path if path is None else path
        return bind_player_navigation_revision(self.state if state is None else state,
            self.plan if plan is None else plan,path=path,sha256=file_sha(path),
            loaded_checkpoint_sha256=file_sha(self.source))

    def test_explicit_v2_settings_and_raw_source_lineage(self):
        record=self.bind()
        self.assertEqual(record['protocol'],FULL_HORIZON_PROTOCOL)
        self.assertEqual(record['version'],2)
        self.assertEqual(record['settings']['tbptt'],450)
        self.assertEqual(record['settings']['player_weights'],dict(combat=4/16,navigation=4/16,enemy_motion=8/16))
        self.assertEqual(player_navigation_settings_for_plan(self.plan),record['settings'])
        self.assertEqual(self.plan['source_pre_batch_gate_sha256'],digest(self.raw_gate))
        self.assertEqual(self.state['rssm_training']['moe_training']['batch_revision']['source_gate'],self.raw_gate)
        self.assertEqual(self.old.path.read_bytes(),self.old_plan_bytes)
        self.assertEqual(self.old.plan['kind'],PROTOCOL)
        self.assertEqual(self.old.plan['settings'],player_navigation_settings())
        self.assertEqual(self.old.plan['settings']['tbptt'],64)
        self.assertEqual(self.state['rssm_training']['moe_training']['final_horizon_goal'],self.raw_gate['final_horizon_goal'])

    def test_v2_contact_metrics_commit_use_explicit_variant_tag(self):
        record=self.bind();mark_player_navigation_baseline(self.state,step=self.state['global_training_step'])
        combat=dict(moe_player_loss=1.,moe_enemy_loss=2.,moe_combat_loss=3.,moe_router_role_loss=.1,moe_router_balance_loss=.01)
        nav=dict(nav_player_loss=.4,nav_router_role_loss=.02,nav_router_balance_loss=.002)
        enemy=dict(enemy_motion_enemy_loss=.8,enemy_motion_player_loss=.6,
                   enemy_motion_router_role_loss=.03,enemy_motion_router_balance_loss=.003)
        metrics=merge_main_replay_metrics(combat,nav,enemy,plan_sha256='a'*64,
            layout=dict(combat=4,enemy_motion=8,navigation=4),player_navigation_protocol=record['settings'])
        metrics=merge_player_contact_metrics(metrics,dict(player_contact_loss=.125),plan_sha256=record['sha256'],
            player_navigation_protocol=record['settings'])
        self.assertEqual(metrics['player_navigation_loss_revision'],FULL_HORIZON_PROTOCOL)
        self.assertAlmostEqual(metrics['moe_player_loss'],2.125)
        from turkeyworld.rssm_moe_protocol import commit_moe_update
        self.state['global_training_step']+=1;step=self.state['global_training_step']
        commit_moe_update(self.state,step=step,horizon=450,train_loss=metrics['train_loss'],
            microbatches=1,microbatch_size=16,effective_batch_size=16)
        commit_player_navigation_revision(self.state,step=step,contact_sampler_state=dict(draws=4),metrics=metrics)
        restored=json.loads(json.dumps(self.state));self.bind(state=restored)
        self.assertEqual(restored,self.state)

    def test_full_graph_cannot_use_old_batch_or_change_goals(self):
        with self.assertRaises(ValueError):
            player_navigation_plan(self.old.source_state,source_checkpoint=self.source,
                source_checkpoint_sha256=file_sha(self.source),contact_root=self.old.contact,
                contact_sampler_identity=self.old.identity,tbptt=450,microbatch_size=16)
        for mutate in(lambda p:p['settings'].update(tbptt=64),
                       lambda p:p.update(batch_revision_sha256='e'*64),
                       lambda p:p['settings'].update(H450_threshold=.5)):
            bad=deepcopy(self.plan);mutate(bad);path=self.old.write('bad450.json',bad)
            with self.assertRaises(ValueError):self.bind(plan=bad,path=path)


if __name__=='__main__':unittest.main()
