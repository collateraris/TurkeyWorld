"""Local-motion repair is additive to the unchanged full450 optimizer history."""
from copy import deepcopy
import json
import unittest

import numpy as np
import torch
import test_rssm_player_navigation_tbptt450_revision as parent
from turkeyworld.rssm_player_navigation_revision import(file_sha,mark_player_navigation_baseline,
    commit_player_navigation_revision)
from turkeyworld.rssm_moe_protocol import commit_moe_update
from turkeyworld.rssm_motion_correction_revision import(PROTOCOL,motion_correction_plan,
    bind_motion_correction_revision,mark_motion_correction_baseline,commit_motion_correction_revision,
    local_motion_horizons)
from turkeyworld.train_rssm_moe import merge_motion_correction_metrics


class MotionCorrectionRevisionTests(unittest.TestCase):
    def setUp(self):
        self.old=parent.FullHorizonPlayerRevisionTests('runTest');self.old.setUp();self.addCleanup(self.old.doCleanups)
        self.state,self.source,self.root=self.old.state,self.old.source,self.old.root
        self.old.bind();mark_player_navigation_baseline(self.state,step=self.state['global_training_step'])
        self.state['rssm_training']['navigation_replay_revision']=dict(private_sampler=dict(draws=6726,rng=83))
        self.plan=motion_correction_plan(self.state,source_checkpoint=self.source,source_checkpoint_sha256=file_sha(self.source))
        self.path=self.root/'motion.json';self.path.write_text(json.dumps(self.plan),encoding='utf-8')

    def bind(self,state=None):
        return bind_motion_correction_revision(self.state if state is None else state,self.plan,
            path=self.path,sha256=file_sha(self.path),loaded_checkpoint_sha256=file_sha(self.source))

    def test_binding_preserves_all_old_clocks_history_loss_RNG_and_sampler(self):
        before=deepcopy(self.state);nprng=deepcopy(np.random.get_state());trng=torch.get_rng_state().clone()
        record=self.bind()
        for key,value in before['rssm_training'].items():self.assertEqual(value,self.state['rssm_training'][key])
        for key in('loss','latest_metrics','loss_groups','_model','_optimizer','preview_assets'):
            self.assertEqual(before[key],self.state[key])
        self.assertEqual(record['updates'],0);self.assertFalse(record['baseline_complete'])
        self.assertEqual(record['settings']['main_batch']['enemy_motion'],8)
        self.assertEqual(self.state['active_objective_formula_revision'],PROTOCOL)
        self.assertTrue(torch.equal(torch.get_rng_state(),trng));self.assertTrue(np.array_equal(np.random.get_state()[1],nprng[1]))
        self.assertEqual(self.state['objective_history'][-1]['source_latest_metrics'],before['latest_metrics'])

    def test_requires_baseline_and_preserves_parent_goal_through_true_commit_resume(self):
        record=self.bind();source=self.state['global_training_step']
        metrics=dict(enemy_local_loss=.3,objective_formula_revision=PROTOCOL,motion_correction_loss_revision=PROTOCOL,
            enemy_local_forward_transitions=64,enemy_local_valid_source_labels=60,
            player_navigation_loss_revision=self.state['rssm_training']['player_navigation_revision']['protocol'],
            player_contact_loss=.2,player_contact_generated_transitions=160)
        with self.assertRaises(ValueError):commit_motion_correction_revision(self.state,step=source+1,metrics=metrics)
        mark_motion_correction_baseline(self.state,step=source);mark_motion_correction_baseline(self.state,step=source)
        self.state['global_training_step']+=1;step=self.state['global_training_step']
        commit_moe_update(self.state,step=step,horizon=450,train_loss=.99,microbatches=1,microbatch_size=16,effective_batch_size=16)
        commit_player_navigation_revision(self.state,step=step,contact_sampler_state=dict(draws=4),metrics=metrics)
        commit_motion_correction_revision(self.state,step=step,metrics=metrics)
        self.assertEqual((record['updates'],record['enemy_branch_transitions'],record['enemy_valid_local_labels']),(1,64,60))
        goal=self.state['rssm_training']['moe_training']['final_horizon_goal']
        self.assertEqual((goal['streak'],goal['settings']['threshold'],goal['settings']['required_consecutive_passes']),(1,1.,10))
        restored=json.loads(json.dumps(self.state));self.bind(restored);self.assertEqual(restored,self.state)
        with self.assertRaises(ValueError):commit_motion_correction_revision(self.state,step=step,metrics=metrics)

    def test_plan_tampering_or_changed_source_cannot_silently_enable_new_objective(self):
        self.plan['settings']['H450_threshold']=.3;self.path.write_text(json.dumps(self.plan),encoding='utf-8')
        with self.assertRaises(ValueError):self.bind()
        self.plan['settings']['H450_threshold']=1.;self.path.write_text(json.dumps(self.plan),encoding='utf-8')
        self.source.write_bytes(b'changed source')
        with self.assertRaises(ValueError):self.bind()

    def test_horizons_are_independent_of_four_tick_latent_slot(self):
        pairs=[local_motion_horizons(i)for i in range(16)]
        for task in(0,1):
            for horizon in(4,8,16,32):
                slots=[i%4 for i,pair in enumerate(pairs)if pair[task]==horizon]
                self.assertEqual(sorted(slots),[0,1,2,3])
        self.assertTrue(all(a!=b for a,b in pairs))
        for invalid in(-1,True,1.5):
            with self.assertRaises(ValueError):local_motion_horizons(invalid)

    def test_local_loss_added_to_enemy_once_without_changing_parent_player_tag(self):
        row=dict(moe_player_loss=1.,moe_enemy_loss=2.,moe_combat_loss=3.,moe_router_loss=.1,
            moe_router_role_loss=.09,moe_router_balance_loss=.01,player_navigation_loss_revision='parent_v2')
        merged=merge_motion_correction_metrics(row,dict(enemy_local_loss=.4,enemy_local_valid_source_labels=16),plan_sha256='a'*64)
        self.assertEqual(merged['player_navigation_loss_revision'],'parent_v2')
        self.assertEqual(merged['objective_formula_revision'],PROTOCOL)
        self.assertEqual(merged['moe_enemy_loss'],2.4);self.assertAlmostEqual(merged['train_loss'],6.5)
        self.assertEqual(merged['loss_groups']['total'],merged['train_loss'])
        self.assertEqual(row['moe_enemy_loss'],2.)


if __name__=='__main__':unittest.main()
