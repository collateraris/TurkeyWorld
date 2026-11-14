"""Serial motion rounds preserve the full source while owning a new stage clock."""
from copy import deepcopy
import json
import pickle
import random
import unittest

import numpy as np
import torch

import test_rssm_main_ladder_revision as parent
from turkeyworld.rssm_motion_only_revision import (
    PROTOCOL,HORIZONS,STAGE_UPDATES,ROUND_UPDATES,motion_only_settings,batch_settings,
    motion_only_plan,bind_motion_only,mark_motion_only_baseline,current_horizon,
    commit_motion_only,begin_motion_round,attach_corpus,validate_motion_only,
    runtime_player_settings,legacy_binding_state,file_sha,digest,
)
from turkeyworld.rssm_player_navigation_revision import player_navigation_settings


class MotionOnlyRevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        self.old=parent.MainLadderRevisionTests('runTest');self.old.setUp();self.addCleanup(self.old.doCleanups)
        self.state,self.source,self.root=self.old.state,self.old.source,self.old.root
        self.old.ready()
        self.state['rssm_training']['moe_training'].setdefault('samplers',{})['16']=dict(
            anchor_counts={i:i+2 for i in range(8)},rng=np.random.default_rng(83).bit_generator.state)
        self.before=deepcopy(self.state)
        self.nav=self.root/'new_nav';self.enemy=self.root/'new_enemy';self.nav.mkdir();self.enemy.mkdir()
        self.plan=motion_only_plan(self.state,source_checkpoint=self.source,
            source_checkpoint_sha256=file_sha(self.source),navdir=self.nav,enemydir=self.enemy)
        self.path=self.root/'motion_only.json';self.path.write_text(json.dumps(self.plan),encoding='utf-8')

    def bind(self,state=None,*,loaded_sha=None):
        return bind_motion_only(self.state if state is None else state,self.plan,path=self.path,
            loaded_sha=file_sha(self.source)if loaded_sha is None else loaded_sha)

    def ready(self):
        r=self.bind();mark_motion_only_baseline(self.state,step=self.state['global_training_step']);return r

    def metrics(self,**overrides):
        h=current_horizon(self.state)
        row=dict(objective_formula_revision=PROTOCOL,motion_only_loss_revision=PROTOCOL,
            main_horizon=h,moe_main_horizon=h,nav_tbptt=h,enemy_motion_tbptt=h,moe_main_tbptt=h,
            microbatch_size=16,effective_batch_size=16,gradient_accumulation_steps=1,completed_microbatches=1,optimizer_steps=1,
            main_navigation_actor_count=8,main_enemy_motion_actor_count=8,main_independent_actor_count=16,
            main_combat_actor_count=0,combat_auxiliary_actor_count=0,combat_late_actor_count=0,
            extra_contact_fragment_count=0,contact_scout_transitions=0,combat_batch_draws=0,
            moe_combat_loss=0.,direct_combat_supervision_calls=0,combat_role_router_supervision=False,
            player_contact_replay=dict(branches=4,direct_source_attempts=48,selection=dict(source_probes=12)),
            enemy_local_recovery=dict(branches=3,total_Source_step_queries=96),
            motion_only_streams=[dict(stream=stream,actors=8,horizon=h,tbptt=h,
                forward_transitions=8*h,memory_detaches_after_initialization=0)for stream in('navigation','enemy_motion')])
        row.update(overrides);return row

    def update(self,value=2.,*,metrics=None,nav_increment=8,enemy_increment=8):
        r=validate_motion_only(self.state);a=deepcopy(r['nav_sampler_state']);b=deepcopy(r['enemy_sampler_state'])
        for state,count in((a,nav_increment),(b,enemy_increment)):
            rng=np.random.default_rng();rng.bit_generator.state=state['rng'];rng.integers(0,99,size=count)
            state.update(draws=state['draws']+count,rng=rng.bit_generator.state)
        row=self.metrics()if metrics is None else metrics
        step=self.state['global_training_step']+1;self.state['global_training_step']=step
        return commit_motion_only(self.state,step=step,loss=value,nav_sampler_state=a,enemy_sampler_state=b,metrics=row)

    def receipt(self,index=0):
        nav=self.root/f'cycle_{index}'/'navigation'/'train';enemy=self.root/f'cycle_{index}'/'enemy_motion'/'train'
        nav.mkdir(parents=True);enemy.mkdir(parents=True)
        episodes=[]
        for directory in(nav,enemy):
            path=directory/'episode.json';path.write_text(json.dumps(dict(actual_Source=True,index=index)),encoding='utf-8')
            episodes.append(dict(split='train',path=str(path),sha256=file_sha(path)))
        return dict(protocol='rssm_motion_only_Source_cycle_corpus_v1',split='train',validation_unchanged=True,
            nav_train_directory=str(nav),enemy_train_directory=str(enemy),episodes=episodes,
            policy_checkpoint_sha256='a'*64,world_checkpoint_sha256='b'*64)

    def test_activation_archives_raw_typed_ledgers_without_touching_model_adam_or_rng(self):
        trng=torch.get_rng_state().clone();nrng=deepcopy(np.random.get_state());prng=random.getstate()
        r=self.bind()
        self.assertEqual((HORIZONS,STAGE_UPDATES,ROUND_UPDATES),((16,48,80,112,144,180),30,180))
        self.assertEqual(r['source_rssm_training'],self.before['rssm_training'])
        for key,value in self.before['rssm_training'].items():self.assertEqual(self.state['rssm_training'][key],value)
        for key in('_model','_optimizer','preview_assets'):self.assertEqual(self.state[key],self.before[key])
        self.assertTrue(torch.equal(trng,torch.get_rng_state()));self.assertEqual(prng,random.getstate())
        self.assertTrue(np.array_equal(nrng[1],np.random.get_state()[1]))
        self.assertEqual(r['nav_sampler_state'],self.before['rssm_training']['navigation_replay_revision']['sampler_state'])
        self.assertEqual(r['enemy_sampler_state'],self.before['rssm_training']['navigation_replay_revision']['enemy_sampler_state'])
        self.assertEqual(set(r['source_rssm_training']['moe_training']['samplers']['16']['anchor_counts']),set(range(8)))
        self.assertIsNone(self.state['loss']);self.assertIsNone(self.state['latest_metrics'])
        self.assertEqual(self.state['objective_history'][-1]['source_loss'],self.before['loss'])
        self.assertEqual(self.state['active_objective_formula_revision'],PROTOCOL)

    def test_loss_never_shortcuts_or_delays_exact_thirty_updates(self):
        r=self.ready()
        for _ in range(29):self.update(.00001)
        self.assertEqual((current_horizon(self.state),r['gate']['stage_updates']),(16,29))
        self.update(100.)
        self.assertEqual((current_horizon(self.state),r['gate']['stage_updates']),(48,0))
        self.assertEqual(r['gate']['advance_history'][0]['reason'],'fixed_stage_updates')
        self.assertIsNone(r['settings']['loss_threshold']);self.assertFalse(r['settings']['loss_is_advance_condition'])

    def test_all_six_horizons_complete_180_updates_without_quality_claim(self):
        r=self.ready();oldq=deepcopy(self.before['rssm_training'])
        for horizon in HORIZONS:
            self.assertEqual(current_horizon(self.state),horizon)
            for _ in range(30):self.update(10.)
        self.assertEqual(r['updates'],180);self.assertEqual(r['main_examples'],16*180)
        self.assertEqual(r['main_transitions'],16*30*sum(HORIZONS))
        self.assertTrue(r['gate']['completed']);self.assertFalse(r['gate']['quality_passed'])
        self.assertEqual(r['gate']['completion_reason'],'world_round_complete')
        self.assertEqual(r['gate']['horizon_update_counts'],{str(h):30 for h in HORIZONS})
        for key,value in oldq.items():self.assertEqual(self.state['rssm_training'][key],value)
        self.assertEqual(r['nav_sampler_state']['draws'],r['initial_nav_draws']+1440)
        self.assertEqual(r['enemy_sampler_state']['draws'],r['initial_enemy_draws']+1440)
        self.assertEqual((r['combat_draws'],r['combat_examples'],r['independent_contact_draws']),(0,0,0))
        self.assertEqual(r['reused_player_local_branches'],720);self.assertEqual(r['reused_enemy_local_branches'],540)
        self.assertEqual(r['local_Source_queries'],180*(48+12+96))
        before=deepcopy(r)
        with self.assertRaises(ValueError):self.update()
        self.assertEqual(r,before)

    def test_first_corpus_and_completed_new_round_preserve_cumulative_clocks(self):
        r=self.ready();initial=self.receipt(0);begin_motion_round(self.state,initial)
        self.assertEqual((len(r['corpus_revisions']),r['gate']['round_index'],r['updates']),(1,0,0))
        with self.assertRaises(ValueError):attach_corpus(self.state,initial)
        with self.assertRaises(ValueError):
            self.update();begin_motion_round(self.state,self.receipt(1))
        # The first valid update above completed; the unfinished round cannot restart.
        self.assertEqual(r['updates'],1)
        for _ in range(179):self.update()
        source=self.state['global_training_step'];sampler=deepcopy(r['nav_sampler_state'])
        gate=begin_motion_round(self.state,self.receipt(2))
        self.assertEqual((gate['round_index'],gate['current_horizon'],gate['updates']),(1,16,0))
        self.assertEqual(gate['source_optimizer_step'],source);self.assertEqual(r['updates'],180)
        self.assertEqual(r['nav_sampler_state'],sampler)
        self.update();self.assertEqual((r['updates'],gate['updates']),(181,1))
        self.assertEqual(r['main_transitions'],16*(30*sum(HORIZONS)+16))

    def test_invalid_partial_combat_or_wrong_horizon_receipts_do_not_mutate_ledger(self):
        r=self.ready();before=deepcopy(r);source=self.state['global_training_step']
        cases=[dict(moe_combat_loss=.1),dict(main_combat_actor_count=1),dict(combat_auxiliary_actor_count=1),
            dict(combat_late_actor_count=1),dict(combat_batch_draws=1),dict(extra_contact_fragment_count=1),
            dict(optimizer_steps=0),dict(optimizer_steps=2),
            dict(contact_scout_transitions=32),dict(completed_microbatches=0),dict(main_navigation_actor_count=7),
            dict(main_independent_actor_count=20),dict(nav_tbptt=8),dict(moe_main_horizon=450),
            dict(combat_role_router_supervision=True),dict(player_contact_replay=dict(branches=5)),
            dict(motion_only_loss_revision='old')]
        for change in cases:
            row=self.metrics(**change)
            with self.subTest(change=change),self.assertRaises(ValueError):self.update(metrics=row)
            self.assertEqual(r,before);self.state['global_training_step']=source
        row=self.metrics();row['motion_only_streams'][0]['memory_detaches_after_initialization']=1
        with self.assertRaises(ValueError):self.update(metrics=row)
        self.assertEqual(r,before)

    def test_nonfinite_or_wrong_sampler_draws_never_commit(self):
        r=self.ready();before=deepcopy(r);step=self.state['global_training_step']
        for bad in(float('nan'),float('inf'),-.1):
            with self.assertRaises(ValueError):self.update(bad)
            self.assertEqual(r,before);self.state['global_training_step']=step
        for a,b in((7,8),(8,9)):
            with self.assertRaises(ValueError):self.update(nav_increment=a,enemy_increment=b)
            self.assertEqual(r,before);self.state['global_training_step']=step

    def test_baseline_and_exact_source_or_immutable_plan_are_required(self):
        r=self.bind();before=deepcopy(r);source=self.state['global_training_step']
        with self.assertRaises(ValueError):self.update()
        self.assertEqual(r,before);self.state['global_training_step']=source
        changed=deepcopy(self.plan);changed['settings']['updates_per_stage']=20
        self.path.write_text(json.dumps(changed),encoding='utf-8')
        with self.assertRaises(ValueError):self.bind()
        self.path.write_text(json.dumps(self.plan),encoding='utf-8')
        self.source.write_bytes(b'foreign source')
        with self.assertRaises(ValueError):self.bind()

    def test_pickle_resume_retains_integer_keys_rng_and_does_not_rebind_old_world_gate(self):
        r=self.ready();self.update();original=deepcopy(self.state)
        saved=pickle.loads(pickle.dumps(self.state));bound=self.bind(saved,loaded_sha='c'*64)
        self.assertEqual(saved,original);self.assertEqual(bound,r)
        disk=json.loads(self.path.read_text(encoding='utf-8'))
        self.assertEqual(self.plan,disk)
        self.assertEqual(set(disk['source_rssm_training']['moe_training']['samplers']['16']['anchor_counts']),set(map(str,range(8))))
        view=legacy_binding_state(saved,disk)
        self.assertEqual(view['rssm_training'],self.before['rssm_training'])
        self.assertEqual(view['global_training_step'],r['source_optimizer_step'])
        saved['rssm_training']['navigation_replay_revision']['sampler_state']['draws']+=1
        with self.assertRaises(ValueError):validate_motion_only(saved)

    def test_corpus_rejects_val_missing_pins_and_modified_source_episode(self):
        r=self.ready();receipt=self.receipt()
        for key,value in(('split','val'),('validation_unchanged',False),('world_checkpoint_sha256','bad')):
            bad=deepcopy(receipt);bad[key]=value
            with self.assertRaises(ValueError):attach_corpus(self.state,bad)
        bad=deepcopy(receipt);bad['episodes'][0]['split']='val'
        with self.assertRaises(ValueError):attach_corpus(self.state,bad)
        from pathlib import Path
        val=self.root/'actual_frozen_val.json';val.write_bytes(b'held out')
        bad=deepcopy(receipt);bad['episodes'][0].update(path=str(val),split='train',sha256=file_sha(val))
        with self.assertRaises(ValueError):attach_corpus(self.state,bad)
        Path(receipt['episodes'][0]['path']).write_bytes(b'modified')
        with self.assertRaises(ValueError):attach_corpus(self.state,receipt)
        self.assertEqual(r['corpus_revisions'],[])

    def test_full_h180_runtime_override_changes_no_parent_geometry_or_weights(self):
        protocol=player_navigation_settings(tbptt=450,microbatch_size=16);before=deepcopy(protocol)
        value=runtime_player_settings(protocol,horizon=180,plan_sha256='a'*64)
        self.assertEqual(value['tbptt'],180);self.assertEqual(value['geometry'],before['geometry'])
        self.assertEqual(value['main_batch_settings']['combat_auxiliary'],0)
        self.assertEqual(value['player_weights'],dict(navigation=.5,enemy_motion=.5,combat=0.))
        self.assertEqual(protocol,before)
        for invalid in(176,208,240,450):
            with self.assertRaises(ValueError):runtime_player_settings(protocol,horizon=invalid,plan_sha256='a'*64)
        self.assertEqual(motion_only_settings()['target_game_steps'],180)
        self.assertEqual(batch_settings()['enemy_normalization_denominator_per_update'],8)


if __name__=='__main__':unittest.main()
