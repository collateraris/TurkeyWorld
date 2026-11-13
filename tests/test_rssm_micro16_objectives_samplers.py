from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_rssm_attack_positive_sampler import source_files,sampler
from test_rssm_target_sampler import data_for
from test_rssm_moe_training import TinyModel,data,oracle
from test_rssm_moe_navigation_objective import TinyNavigationModel,batch
from test_rssm_moe_enemy_motion import EnemyModel,quiet_batch
import test_rssm_navigation_replay_protocol as replay_tests
from test_rssm_navigation_replay_protocol import Tape
from turkeyworld.rssm_moe_training import moe_dynamics_backward
from turkeyworld.rssm_moe_navigation_objective import navigation_replay_backward
from turkeyworld.rssm_moe_enemy_motion import enemy_motion_backward
from turkeyworld.rssm_navigation_replay_protocol import (NavigationReplaySampler,EnemyMotionReplaySampler,
    commit_navigation_replay)
from turkeyworld.rssm_moe_protocol import commit_moe_update
from turkeyworld.rssm_moe_batch_revision import batch_revision_plan,bind_moe_batch_revision,microbatch_layout


class Micro16Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_real16_aux_two_late_backward_and_legacy_defaults_bitexact(self):
        torch.manual_seed(7);model=TinyModel()
        with patch('turkeyworld.rssm_combat_branches._source_step_at_own_state',side_effect=oracle):
            report=moe_dynamics_backward(model,data(4,count=9),data(4,count=16),
                auxiliary_actor_limit=16,late_own_prefix_actors=2,main_player_weight=9/16,
                main_router_weight=9/16,main_enemy_weight=36/54)
        self.assertEqual(report['moe_auxiliary']['actors'],16)
        self.assertEqual(report['moe_late_own_prefix']['actors'],2)
        self.assertEqual(report['moe_auxiliary']['direct_queries_attempted'],64)
        self.assertEqual(report['moe_late_own_prefix']['direct_queries_attempted'],8)
        for head in(model.p_core,model.e_core,model.fusion.player_head,model.fusion.enemy_head,
                     model.fusion.combat_head,model.fusion.router):
            self.assertGreater(sum(float(p.grad.abs().sum())for p in head.parameters()if p.grad is not None),0)
        a=TinyModel();b=deepcopy(a)
        with patch('turkeyworld.rssm_combat_branches._source_step_at_own_state',side_effect=oracle):
            first=moe_dynamics_backward(a,data(4),data(4))
            second=moe_dynamics_backward(b,data(4),data(4),auxiliary_actor_limit=64,late_own_prefix_actors=8)
        self.assertEqual(first,second)
        for x,y in zip(a.parameters(),b.parameters()):
            if x.grad is None:self.assertIsNone(y.grad)
            else:self.assertTrue(torch.equal(x.grad,y.grad))

    def test_small_NAV_and_enemy_backward_keep_weighted_terms_and_own_memory(self):
        for actors in(3,2):
            labels=batch(16,count=actors,context=2);labels['actions'][...,3]=0
            model=TinyNavigationModel()
            with patch('turkeyworld.rssm_combat_branches._source_step_at_own_state',side_effect=AssertionError('No oracle')):
                report=navigation_replay_backward(model,labels,context=2,weight=actors/16)
            self.assertEqual(report['nav_actor_count'],actors)
            self.assertEqual(report['nav_forward_transitions'],actors*16)
            self.assertEqual(report['nav_weight'],actors/16)
            self.assertGreater(float(model.p_core.weight.grad.abs().sum()),0.)
        labels={k:v.repeat((2,)+(1,)*(v.ndim-1))for k,v in quiet_batch(16).items()}
        model=EnemyModel()
        with patch('turkeyworld.rssm_combat_branches._source_step_at_own_state',side_effect=AssertionError('No oracle')):
            report=enemy_motion_backward(model,labels,context=2,enemy_weight=16/54,router_weight=4/16)
        self.assertEqual(report['enemy_motion_actor_count'],4)
        self.assertEqual(report['enemy_motion_forward_transitions'],64)
        self.assertGreater(float(model.enemy_head.weight.grad.abs().sum()),0.)

    def test_balanced16_preserves_old_counters_rotates_real_misses_and_exact_resume(self):
        with TemporaryDirectory()as folder:
            paths=source_files(Path(folder)/'train');chosen=sampler(paths);rng=np.random.default_rng(173)
            chosen.sample(64,rng);old=deepcopy(chosen.state_dict());rows=[]
            for _ in range(4):
                pairs=chosen.sample(16,rng);rows.append(deepcopy(chosen.last_sample))
                self.assertEqual(len(pairs),16)
                self.assertEqual(chosen.last_sample['quota'],dict(spawn=4,attack_negative=4,attack_positive=4,shooting_replay=4))
                self.assertEqual(chosen.last_sample['attack_subquotas'],dict(positive_near_wall=2,positive_open=2,negative_occluded=3,negative_other=1))
            totals=Counter()
            for row in rows:totals.update(row['target_subquotas'])
            self.assertEqual(dict(totals),dict(hit_near=4,hit_mid=4,hit_far=4,fired_miss=2,legacy=2))
            self.assertEqual(sum(r['fired_miss_subquotas']['wall_miss']for r in rows),1)
            self.assertEqual(sum(r['fired_miss_subquotas']['other_miss']for r in rows),1)
            self.assertIsNone(rows[0]['ordinary_sampler']);self.assertIsNotNone(rows[1]['ordinary_sampler'])
            counts=chosen.state_dict()
            self.assertEqual(counts['v2_draws'],old['v2_draws']+64)
            self.assertEqual(counts['positive_draws'],old['positive_draws']+16)
            self.assertEqual(counts['retained_legacy_ledger'],old['retained_legacy_ledger'])
            self.assertEqual(counts['micro16_sampling'],dict(kind='source_attack_balanced_micro16_v1',version=1,source_v2_draws=64,draws=64,batches=4))
            self.assertTrue(all(counts['anchor_counts'][k]-old['anchor_counts'][k]==2 for k in range(8)))
            saved_rng=deepcopy(rng.bit_generator.state);expected=chosen.sample(16,rng)
            restored=sampler(paths);restored.restore_state(counts)
            other=np.random.default_rng();other.bit_generator.state=saved_rng
            np.testing.assert_array_equal(expected,restored.sample(16,other))
            self.assertEqual(chosen.last_sample,restored.last_sample)
            self.assertEqual(chosen.state_dict(),restored.state_dict())
            self.assertEqual(rng.bit_generator.state,other.bit_generator.state)
            for name in('draws','batches','source_v2_draws'):
                broken=deepcopy(counts);broken['micro16_sampling'][name]+=1
                with self.assertRaises(ValueError):sampler(paths).restore_state(broken)

    def test_old32_64_output_receipts_ledgers_and_PCG_match_frozen_git_reference(self):
        root=Path(__file__).resolve().parents[1]
        result=subprocess.run(['git','show','HEAD:turkeyworld/rssm_spawn_curriculum.py'],cwd=root,
            capture_output=True,text=True,encoding='utf-8',check=True)
        namespace={'__name__':'turkeyworld._legacy_spawn_reference','__package__':'turkeyworld'}
        exec(compile(result.stdout,'frozen-git-rssm_spawn_curriculum.py','exec'),namespace)
        with TemporaryDirectory()as folder:
            paths=source_files(Path(folder)/'train')
            old=namespace['RecoveryCombatSampler'](data_for(paths),target_distance_balanced=True,attack_positive_balanced=True)
            new=sampler(paths);a=np.random.default_rng(173);b=np.random.default_rng(173)
            for size in(32,64,32,64):
                np.testing.assert_array_equal(old.sample(size,a),new.sample(size,b))
                self.assertEqual(old.last_sample,new.last_sample)
                self.assertEqual(old.state_dict(),new.state_dict())
                self.assertEqual(a.bit_generator.state,b.bit_generator.state)
                self.assertEqual(old.summary(batch_size=size),new.summary(batch_size=size))

    def test_piecewise_NAV_ledger_old_baseline_new_counts_and_private_resume(self):
        helper=replay_tests.NavigationReplayProtocolTests();helper.setUp();self.addCleanup(helper.doCleanups)
        state=helper.state;record=helper.bind();n=NavigationReplaySampler(Tape(False),record);e=EnemyMotionReplaySampler(Tape(True),record)
        for _ in range(2):
            rows=[]
            for _ in range(4):
                n.batch(context=2,horizon=4);e.batch(context=2,horizon=4)
                rows.append(dict(main_navigation_actor_count=10,main_enemy_motion_actor_count=16,main_combat_actor_count=38,main_discarded_combat_rng_count=26))
            state['global_training_step']+=1;step=state['global_training_step']
            commit_moe_update(state,step=step,horizon=4,train_loss=.8)
            commit_navigation_replay(state,step=step,sampler_state=n.state_dict(),enemy_sampler_state=e.state_dict(),micro_rows=rows)
        baseline=deepcopy(record);p=batch_revision_plan(state,source_checkpoint=helper.source)
        path=helper.root/'batch16.json';path.write_text(json.dumps(p));sha=hashlib.sha256(path.read_bytes()).hexdigest()
        bind_moe_batch_revision(state,p,path=path,sha256=sha,loaded_checkpoint_sha256=p['source_checkpoint_sha256'])
        helper.bind();self.assertEqual(baseline,record)
        rows=[]
        for i in range(4):
            layout=microbatch_layout(state,i)
            a,_=n.batch(context=2,horizon=4,actors=layout['navigation']);b,_=e.batch(context=2,horizon=4,actors=4)
            self.assertEqual(len(a['states']),layout['navigation']);self.assertEqual(len(b['states']),4)
            rows.append(dict(main_navigation_actor_count=layout['navigation'],main_enemy_motion_actor_count=4,
                main_combat_actor_count=layout['combat'],main_discarded_combat_rng_count=layout['discarded_combat_rng_draws']))
        state['global_training_step']+=1;step=state['global_training_step']
        commit_moe_update(state,step=step,horizon=4,train_loss=.8,microbatch_size=16,effective_batch_size=64)
        commit_navigation_replay(state,step=step,sampler_state=n.state_dict(),enemy_sampler_state=e.state_dict(),micro_rows=rows)
        for key,delta in dict(navigation_main_actors=10,enemy_motion_main_actors=16,combat_main_actors=38,discarded_combat_rng_draws=26).items():
            self.assertEqual(record[key],baseline[key]+delta)
        helper.bind();clone=deepcopy(state);helper.bind(clone)
        n2=NavigationReplaySampler(Tape(False),clone['rssm_training']['navigation_replay_revision'])
        _,a=n.batch(context=2,horizon=450,actors=3);_,b=n2.batch(context=2,horizon=450,actors=3)
        self.assertEqual(a,b)
        before=n.state_dict();bad=Tape(True);n.data=bad
        with self.assertRaises(ValueError):n.batch(context=2,horizon=4,actors=2)
        self.assertEqual(n.state_dict(),before)


if __name__=='__main__':unittest.main()
