"""A complete micro32 update preserves both historical batch segments."""
from contextlib import nullcontext
from copy import deepcopy
import json
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_rssm_navigation_replay_protocol as legacy
from turkeyworld.rssm_combat_memory_protocol import file_sha
from turkeyworld.rssm_moe_batch_revision import (
    PROTOCOL, MICRO32_PROTOCOL, MICRO32_SETTINGS, batch_revision_plan,
    micro32_batch_revision_plan, bind_moe_batch_revision, moe_batch_settings,
    microbatch_layout, batch_settings_for_plan,
)
from turkeyworld.rssm_moe_protocol import bind_moe_training, commit_moe_update, moe_training_horizon
from turkeyworld.rssm_navigation_replay_protocol import (
    NavigationReplaySampler, EnemyMotionReplaySampler, commit_navigation_replay,
)
from turkeyworld.renderer_native_phase import _same


class Micro32BatchRevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.old = legacy.NavigationReplayProtocolTests(methodName='runTest')
        self.old.setUp()
        self.addCleanup(self.old.doCleanups)
        self.state, self.source, self.root = self.old.state, self.old.source, self.old.root
        self.old.bind()
        self.navigation, self.enemy = legacy.Tape(False), legacy.Tape(True)
        for _ in range(2):
            self.update()
        self.legacy_gate = deepcopy(self.gate)
        self.plan16 = batch_revision_plan(self.state, source_checkpoint=self.source)
        self.path16 = self.write('batch16.json', self.plan16)
        self.bind(self.plan16, self.path16)
        for _ in range(3):
            self.update()
        self.before = deepcopy(self.state)
        self.plan = micro32_batch_revision_plan(self.state, source_checkpoint=self.source)
        self.path = self.write('batch32.json', self.plan)

    @property
    def gate(self):
        return self.state['rssm_training']['moe_training']

    @property
    def navigation_record(self):
        return self.state['rssm_training']['navigation_replay_revision']

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value), encoding='utf-8')
        return path

    def bind(self, plan=None, path=None, *, state=None, loaded=None):
        path = path or self.path
        return bind_moe_batch_revision(self.state if state is None else state,
            self.plan if plan is None else plan, path=path, sha256=file_sha(path),
            loaded_checkpoint_sha256=loaded or file_sha(self.source))

    def samplers(self):
        settings = moe_batch_settings(self.state)
        return (NavigationReplaySampler(self.navigation, self.navigation_record, batch_settings=settings),
                EnemyMotionReplaySampler(self.enemy, self.navigation_record, batch_settings=settings))

    def update(self, loss=.8):
        settings = moe_batch_settings(self.state)
        n, e = self.samplers()
        horizon = moe_training_horizon(self.state)
        rows = []
        for i in range(settings['gradient_accumulation_steps']):
            layout = microbatch_layout(self.state, i)
            n.batch(context=1, horizon=horizon, actors=layout['navigation'])
            e.batch(context=1, horizon=horizon, actors=layout['enemy_motion'])
            rows.append(dict(main_navigation_actor_count=layout['navigation'],
                main_enemy_motion_actor_count=layout['enemy_motion'],
                main_combat_actor_count=layout['combat'],
                main_discarded_combat_rng_count=layout['discarded_combat_rng_draws']))
        self.state['global_training_step'] += 1
        step = self.state['global_training_step']
        commit_moe_update(self.state, step=step, horizon=horizon, train_loss=loss,
            microbatches=settings['gradient_accumulation_steps'], microbatch_size=settings['microbatch_size'],
            effective_batch_size=settings['effective_batch_size'])
        commit_navigation_replay(self.state, step=step, sampler_state=n.state_dict(),
            enemy_sampler_state=e.state_dict(), micro_rows=rows)

    def test_activation_is_a_direct_child_and_preserves_gate_sampler_history_rng_and_old_tensors(self):
        np_before, py_before, torch_before = deepcopy(np.random.get_state()), random.getstate(), torch.get_rng_state().clone()
        record = self.bind()
        old_gate = deepcopy(self.gate)
        old_gate['batch_revision'] = record['source_gate']['batch_revision']
        old_gate['batch_protocol'] = PROTOCOL
        self.assertTrue(_same(old_gate, self.before['rssm_training']['moe_training']))
        self.assertEqual(record['source_gate'], self.before['rssm_training']['moe_training'])
        self.assertEqual(record['source_navigation_replay_revision'], self.before['rssm_training']['navigation_replay_revision'])
        self.assertEqual(self.navigation_record, self.before['rssm_training']['navigation_replay_revision'])
        for key in ('_model', '_optimizer', '_rng'):
            self.assertTrue(_same(self.state[key], self.before[key]))
        self.assertTrue(_same(np_before, np.random.get_state()))
        self.assertEqual(py_before, random.getstate())
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        self.assertEqual(record['updates'], 0)
        self.assertEqual(self.gate['batch_protocol'], MICRO32_PROTOCOL)
        self.assertEqual(record['previous_batch_revision_sha256'], file_sha(self.path16))
        with self.assertRaises(ValueError):
            micro32_batch_revision_plan(self.state, source_checkpoint=self.source)
        no_parent = deepcopy(self.before)
        no_parent['rssm_training']['moe_training'] = deepcopy(self.legacy_gate)
        no_parent['global_training_step'] = self.legacy_gate['last_optimizer_step']
        with self.assertRaises(ValueError):
            micro32_batch_revision_plan(no_parent, source_checkpoint=self.source)

    def test_exact_single_micro_cohorts_weights_and_auxiliary_separation(self):
        self.bind()
        settings = moe_batch_settings(self.state)
        self.assertEqual(settings, MICRO32_SETTINGS)
        self.assertEqual((settings['microbatch_size'], settings['gradient_accumulation_steps'],
            settings['effective_batch_size'], settings['auxiliary_batch_size'], settings['late_own_prefix_actors']),
            (32, 1, 32, 32, 4))
        row = microbatch_layout(self.state, 0)
        self.assertEqual((row['combat'], row['enemy_motion'], row['navigation']), (19, 8, 5))
        self.assertEqual((row['combat_rng_draws'], row['discarded_combat_rng_draws']), (32, 13))
        self.assertAlmostEqual(row['combat_player_weight'], 19/32)
        self.assertAlmostEqual(row['navigation_player_router_weight'], 5/32)
        self.assertAlmostEqual(row['enemy_motion_router_weight'], 8/32)
        self.assertAlmostEqual(row['combat_enemy_weight'], 19/27)
        self.assertAlmostEqual(row['enemy_motion_enemy_weight'], 8/27)
        self.assertAlmostEqual(row['combat_enemy_weight'] + row['enemy_motion_enemy_weight'], 1)
        for index in (-1, 1, True, 0.):
            with self.assertRaises(ValueError): microbatch_layout(self.state, index)

    def test_piecewise256_64_32_counts_and_exact_resume_after_odd_nav_enemy_draws(self):
        self.bind(); self.update()
        self.assertEqual(self.gate['main_examples'], 2*256 + 3*64 + 32)
        self.assertEqual(self.gate['main_transitions'], 4*(2*256 + 3*64 + 32))
        self.assertEqual((self.navigation_record['navigation_main_actors'],
            self.navigation_record['enemy_motion_main_actors'], self.navigation_record['combat_main_actors'],
            self.navigation_record['discarded_combat_rng_draws']),
            (2*40+3*10+5, 2*64+3*16+8, 2*152+3*38+19, 2*104+3*26+13))
        # These complete32 updates are deliberately not complete old10/16 draws.
        with self.assertRaises(ValueError): NavigationReplaySampler(self.navigation, self.navigation_record)
        with self.assertRaises(ValueError): EnemyMotionReplaySampler(self.enemy, self.navigation_record)
        n, e = self.samplers()
        restored = json.loads(json.dumps({k:v for k,v in self.state.items() if not k.startswith('_')}))
        bind_moe_training(restored, json.loads((self.root/'moe.json').read_text()), path=self.root/'moe.json',
            sha256=file_sha(self.root/'moe.json'), loaded_checkpoint_sha256='f'*64)
        self.bind(state=restored, loaded='f'*64)
        self.old.bind(restored)
        before = deepcopy(restored)
        self.bind(state=restored, loaded='f'*64)
        self.assertEqual(restored, before)
        settings = moe_batch_settings(restored)
        record = restored['rssm_training']['navigation_replay_revision']
        n2 = NavigationReplaySampler(self.navigation, record, batch_settings=settings)
        e2 = EnemyMotionReplaySampler(self.enemy, record, batch_settings=settings)
        for a,b,count in ((n,n2,5),(e,e2,8)):
            _, receipt_a = a.batch(context=2, horizon=450, actors=count)
            _, receipt_b = b.batch(context=2, horizon=450, actors=count)
            self.assertEqual(receipt_a, receipt_b)
        self.update()
        self.assertEqual(self.gate['main_examples'], 2*256 + 3*64 + 2*32)
        self.assertEqual(self.gate['batch_revision']['source_gate'], self.before['rssm_training']['moe_training'])

    def test_malformed_parent_settings_old_ledger_and_source_pcg_rejected(self):
        self.bind(); self.update()
        for mutate in (
            lambda g: g['batch_revision'].update(previous_batch_revision_sha256='f'*64),
            lambda g: g['batch_revision']['settings'].update(gradient_accumulation_steps=4),
            lambda g: g['batch_revision']['source_gate']['batch_revision'].update(main_examples=0),
            lambda g: g['batch_revision']['source_navigation_replay_revision']['sampler_state'].update(draws=0),
            lambda g: g.update(main_examples=g['updates']*32),
        ):
            bad=deepcopy(self.state);mutate(bad['rssm_training']['moe_training'])
            with self.assertRaises(ValueError):moe_training_horizon(bad)
        changed=deepcopy(self.plan);changed['previous_batch_revision_sha256']='f'*64
        path=self.write('bad-parent.json',changed)
        with self.assertRaises(ValueError):self.bind(changed,path,state=deepcopy(self.before))
        with self.assertRaises(ValueError):self.bind(state=deepcopy(self.before),loaded='f'*64)

    def test_single_micro_commit_rejects_four_micro_dimensions_and_partial_actor_receipts(self):
        self.bind(); before=deepcopy(self.gate)
        self.state['global_training_step']+=1
        with self.assertRaises(ValueError):
            commit_moe_update(self.state,step=self.state['global_training_step'],horizon=4,
                train_loss=.8,microbatch_size=32,effective_batch_size=32)
        self.assertEqual(self.gate,before)
        self.state['global_training_step']-=1
        n,e=self.samplers();n.batch(context=1,horizon=4,actors=5);e.batch(context=1,horizon=4,actors=8)
        self.state['global_training_step']+=1
        commit_moe_update(self.state,step=self.state['global_training_step'],horizon=4,train_loss=.8,
            microbatches=1,microbatch_size=32,effective_batch_size=32)
        old=deepcopy(self.navigation_record)
        with self.assertRaises(ValueError):
            commit_navigation_replay(self.state,step=self.state['global_training_step'],
                sampler_state=n.state_dict(),enemy_sampler_state=e.state_dict(),micro_rows=[])
        self.assertEqual(old,self.navigation_record)

    def test_cli_dimensions_come_from_strict_new_or_legacy_plan_without_loading_a_model(self):
        import turkeyworld.train_rssm_moe as trainer
        seen=[]
        class Fake:
            def __init__(self,args):seen.append((args.batch,args.accumulation_steps))
            def run(self):pass
        required=('source-checkpoint','moe-plan','combat-data-root','navigation-data-root',
            'recovery-data-root','renderer-checkpoint','sr','preview-assets-plan','preview-assets-revision-plan')
        for path,expected in ((self.path,(32,1)),(self.path16,(16,4)),(None,(64,4))):
            argv=['--run-dir',str(self.root/('cli'+str(len(seen))))]
            for name in required:argv+=['--'+name,str(self.source)]
            if path is not None:argv+=['--batch-revision-plan',str(path)]
            with patch.object(trainer,'RSSMMoETraining',Fake),patch.object(trainer,'KeepAwake',nullcontext):
                trainer.main(argv)
            self.assertEqual(seen[-1],expected)
        changed=deepcopy(self.plan);changed['settings']['effective_batch_size']=64
        with self.assertRaises(ValueError):batch_settings_for_plan(changed)

    def test_actual_backward_accepts19main32aux4late_with_undiluted_enemy_weights(self):
        from test_rssm_moe_training import TinyModel,data,oracle
        from turkeyworld.rssm_moe_training import moe_dynamics_backward
        from turkeyworld.train_rssm_moe import merge_main_replay_metrics
        self.bind();layout=microbatch_layout(self.state,0)
        model=TinyModel()
        with patch('turkeyworld.rssm_combat_branches._source_step_at_own_state',side_effect=oracle):
            metrics=moe_dynamics_backward(model,data(4,count=19),data(4,count=32),
                expected_horizon=4,auxiliary_actor_limit=32,late_own_prefix_actors=4,
                main_player_weight=layout['main_player_weight'],main_router_weight=layout['main_router_weight'],
                main_enemy_weight=layout['main_enemy_weight'])
        self.assertEqual(metrics['moe_auxiliary']['actors'],32)
        self.assertEqual(metrics['moe_late_own_prefix']['actors'],4)
        self.assertGreater(float(model.p_core.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.e_core.weight.grad.abs().sum()),0)
        nav=dict(nav_player_loss=.1,nav_router_role_loss=.01,nav_router_balance_loss=.001)
        enemy=dict(enemy_motion_enemy_loss=.2,enemy_motion_router_role_loss=.02,enemy_motion_router_balance_loss=.002)
        merged=merge_main_replay_metrics(metrics,nav,enemy,plan_sha256='a'*64,
            layout=layout,auxiliary_batch_size=32,late_actor_count=4)
        self.assertEqual((merged['main_actor_count'],merged['main_combat_actor_count'],
            merged['main_enemy_motion_actor_count'],merged['main_navigation_actor_count']), (32,19,8,5))
        self.assertEqual((merged['combat_auxiliary_actor_count'],merged['combat_late_actor_count']), (32,4))
        self.assertAlmostEqual(merged['moe_combat_loss'],metrics['moe_combat_loss'])
        json.dumps(merged,allow_nan=False)


if __name__=='__main__':unittest.main()
