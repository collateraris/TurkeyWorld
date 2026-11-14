from copy import deepcopy
import json
import math
import pickle
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_rssm_motion_correction_revision as parent
from test_rssm_player_geometry_objectives import MemoryCreditModel,exact_credit_batch
from test_rssm_moe_training import TinyModel,data as combat_batch
from test_rssm_moe_enemy_motion import EnemyModel,quiet_batch
from turkeyworld.rssm_moe_protocol import moe_training_horizon,commit_moe_update
from turkeyworld.rssm_moe_batch_revision import moe_batch_settings,microbatch_layout
from turkeyworld.rssm_moe_budget import curriculum_completion,loss_gate_settings
from turkeyworld.rssm_player_navigation_revision import (player_navigation_protocol,
    player_navigation_settings,FULL_HORIZON_PROTOCOL,file_sha,commit_player_navigation_revision)
from turkeyworld.rssm_motion_correction_revision import (mark_motion_correction_baseline,
    commit_motion_correction_revision,PROTOCOL as MOTION)
from turkeyworld.rssm_moe_navigation_objective import navigation_replay_backward
from turkeyworld.rssm_moe_training import moe_dynamics_backward,COMBAT_COMPONENTS
from turkeyworld.rssm_moe_enemy_motion import enemy_motion_backward
from turkeyworld.rssm_navigation_replay_protocol import commit_navigation_replay,PROTOCOL as REPLAY
from turkeyworld.rssm_main_ladder_revision import (PROTOCOL,HORIZONS,main_ladder_plan,
    bind_main_ladder_revision,mark_main_ladder_baseline,runtime_player_settings,
    validate_continued_ledgers,legacy_binding_state)


class MainLadderRevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        self.old=parent.MotionCorrectionRevisionTests('runTest');self.old.setUp();self.addCleanup(self.old.doCleanups)
        self.state,self.source,self.root=self.old.state,self.old.source,self.old.root
        self.old.bind();mark_motion_correction_baseline(self.state,step=self.state['global_training_step'])
        # Minimal actual private replay ledgers, including nonzero inherited draws.
        nav=dict(updates=17,last_optimizer_step=self.state['global_training_step'],
            source_optimizer_step=self.state['global_training_step']-17,
            source_moe_updates=self.state['rssm_training']['moe_training']['updates']-17,
            navigation_main_actors=73,enemy_motion_main_actors=136,combat_main_actors=71,discarded_combat_rng_draws=204,
            sampler_state=dict(protocol=REPLAY,draws=73,rng=np.random.default_rng(17).bit_generator.state),
            enemy_sampler_state=dict(protocol=REPLAY,draws=136,rng=np.random.default_rng(18).bit_generator.state))
        self.state['rssm_training']['navigation_replay_revision']=nav
        self.before=deepcopy(self.state)
        self.plan=main_ladder_plan(self.state,source_checkpoint=self.source,source_checkpoint_sha256=file_sha(self.source))
        self.path=self.root/'main_ladder.json';self.path.write_text(json.dumps(self.plan),encoding='utf-8')

    @property
    def gate(self):return self.state['rssm_training']['moe_training']

    def bind(self,state=None):
        return bind_main_ladder_revision(self.state if state is None else state,self.plan,
            path=self.path,sha256=file_sha(self.path),loaded_checkpoint_sha256=file_sha(self.source))

    def ready(self):
        r=self.bind();mark_main_ladder_baseline(self.state,step=self.state['global_training_step']);return r

    def update(self,value=.9,eligible=True):
        horizon=moe_training_horizon(self.state);step=self.state['global_training_step']+1
        self.state['global_training_step']=step
        commit_moe_update(self.state,step=step,horizon=horizon,train_loss=value,
            microbatches=1,microbatch_size=18,effective_batch_size=18,loss_gate_eligible=eligible)
        q=self.state['rssm_training'];nav=q['navigation_replay_revision']
        a=deepcopy(nav['sampler_state']);b=deepcopy(nav['enemy_sampler_state']);a['draws']+=6;b['draws']+=6
        commit_navigation_replay(self.state,step=step,sampler_state=a,enemy_sampler_state=b,
            micro_rows=[dict(main_navigation_actor_count=6,main_combat_actor_count=6,
                main_enemy_motion_actor_count=6,main_discarded_combat_rng_count=12)])
        metrics=dict(player_navigation_loss_revision=FULL_HORIZON_PROTOCOL,player_contact_loss=.1,
            enemy_local_loss=.2,objective_formula_revision=MOTION,motion_correction_loss_revision=MOTION)
        commit_player_navigation_revision(self.state,step=step,
            contact_sampler_state=dict(draws=q['player_navigation_revision']['contact_draws']+4),metrics=metrics)
        commit_motion_correction_revision(self.state,step=step,metrics=metrics)
        validate_continued_ledgers(self.state)

    def test_activation_preserves_parent_tensors_rng_samplers_and_starts_independent_stage(self):
        trng=torch.get_rng_state().clone();rng=deepcopy(np.random.get_state());r=self.bind()
        self.assertEqual(HORIZONS,(16,48,80,112,144,176,208,240,272,304,336,368,400,432,450))
        for key in('player_navigation_revision','motion_correction_revision','navigation_replay_revision'):
            self.assertEqual(self.state['rssm_training'][key],self.before['rssm_training'][key])
        self.assertEqual(self.state['_model'],self.before['_model']);self.assertEqual(self.state['_optimizer'],self.before['_optimizer'])
        self.assertTrue(torch.equal(trng,torch.get_rng_state()));self.assertTrue(np.array_equal(rng[1],np.random.get_state()[1]))
        self.assertEqual(r['source_gate'],self.before['rssm_training']['moe_training'])
        self.assertEqual((r['updates'],r['stage_streak'],moe_training_horizon(self.state)),(0,0,16))
        self.assertEqual(self.state['h450_total_loss_goal'],self.gate['final_horizon_goal'])

    def test_layout_full_credit_and_source_exposure_are_piecewise(self):
        r=self.ready();layout=microbatch_layout(self.state,0);b=moe_batch_settings(self.state)
        self.assertEqual((b['microbatch_size'],b['effective_batch_size']),(18,18))
        self.assertEqual(b['player_router_micro_weight'],'actor_count_divided_by_18')
        self.assertEqual([layout[k]for k in('combat','enemy_motion','navigation')],[6,6,6])
        self.assertEqual(layout['combat_auxiliary'],16);self.assertEqual(layout['combat_late'],2)
        for k in('main_player_weight','navigation_player_router_weight','enemy_motion_router_weight'):self.assertEqual(layout[k],1/3)
        for k in('main_enemy_weight','enemy_motion_enemy_weight'):self.assertEqual(layout[k],.5)
        self.update(.9)
        self.assertEqual(r['main_transitions'],18*16)
        self.assertEqual(self.gate['main_transitions'],r['source_gate']['main_transitions']+18*16)
        self.assertEqual(r['source_gate']['batch_revision'],self.gate['batch_revision'])
        restored=json.loads(json.dumps(self.state));self.bind(restored)
        self.assertEqual(restored,self.state)
        view=legacy_binding_state(restored,self.plan)
        self.assertEqual(view['rssm_training'],self.plan['source_rssm_training'])

    def test_only_ten_strict_raw_totals_advance_no_update_cap_or_source_support_gate(self):
        r=self.ready()
        for _ in range(60):self.update(1.)
        self.assertEqual((r['stage_index'],r['stage_streak'],r['stage_updates']),(0,0,60))
        for _ in range(9):self.update(.9,eligible=False)
        self.assertEqual(moe_training_horizon(self.state),16)
        self.update(1.)
        for _ in range(9):self.update(.99)
        self.assertEqual(moe_training_horizon(self.state),16)
        self.update(.99)
        self.assertEqual((moe_training_horizon(self.state),r['stage_streak'],r['stage_updates']),(48,0,0))
        self.assertEqual(loss_gate_settings(self.gate),(1.,10));self.assertIsNone(curriculum_completion(self.gate))

    def test_all_stages_and_final_h450_require_their_own_ten_updates(self):
        r=self.ready()
        for expected in HORIZONS:
            self.assertEqual(moe_training_horizon(self.state),expected)
            for _ in range(9):self.update(.1)
            self.assertFalse(self.gate['goal_met']);self.update(.1)
        self.assertTrue(self.gate['goal_met']);self.assertEqual(curriculum_completion(self.gate),'completed_loss_gate')
        self.assertEqual(r['main_transitions'],18*10*sum(HORIZONS))
        self.assertEqual(self.gate['final_horizon_goal']['h450_updates'],10)
        self.assertEqual(len(r['stage_history']),len(HORIZONS)-1)

    def test_nonfinite_partial_or_tampered_commit_cannot_advance(self):
        r=self.ready();before=deepcopy(self.gate);step=self.state['global_training_step']+1
        for bad in(float('nan'),float('inf'),-.1):
            self.state['global_training_step']=step
            with self.assertRaises(ValueError):commit_moe_update(self.state,step=step,horizon=16,train_loss=bad,
                microbatches=1,microbatch_size=18,effective_batch_size=18)
            self.assertEqual(self.gate,before)
        self.state['global_training_step']=step-1
        for key,val in(('main_transitions',999),('stage_streak',2)):
            original=r[key];r[key]=val
            with self.assertRaises(ValueError):moe_training_horizon(self.state)
            r[key]=original

    def test_runtime_full_bptt_reaches_first_memory_at_each_current_horizon(self):
        parent=player_navigation_settings(tbptt=450,microbatch_size=16)
        for horizon in(16,48,80,450):
            protocol=runtime_player_settings(parent,horizon=horizon,plan_sha256='a'*64)
            self.assertEqual(player_navigation_protocol(protocol),FULL_HORIZON_PROTOCOL)
            self.assertEqual(protocol['tbptt'],horizon)
            model=MemoryCreditModel();report=navigation_replay_backward(model,exact_credit_batch(horizon),
                context=1,weight=1.,player_navigation_protocol=protocol)
            self.assertEqual(report['nav_tbptt'],horizon)
            self.assertGreater(float(model.returned[0].grad.abs().sum()),0.)
            self.assertTrue(model.memory_input_flags[horizon-1])
        self.assertEqual(parent['tbptt'],450)
        broken=runtime_player_settings(parent,horizon=48,plan_sha256='a'*64);broken['player_weights']['combat']=.5
        with self.assertRaises(ValueError):player_navigation_protocol(broken)

    def test_main_and_quiet_real_backwards_share_current_H_graph_without_hidden_16_detach(self):
        protocol=runtime_player_settings(player_navigation_settings(tbptt=450,microbatch_size=16),
            horizon=48,plan_sha256='a'*64)
        model=TinyModel();flags=[];original=model.transition
        def transition(*args):flags.append(args[-1].requires_grad);return original(*args)
        model.transition=transition
        def fake_branch(*args,**kwargs):
            return dict(player=0.,enemy=0.,combat=0.,horizon=16,actors=4,
                direct_source_supervision_available=True,direct_per_tick_counts=[],
                direct_spawn_anchor_counts={},**{key:0. for key in COMBAT_COMPONENTS})
        with patch('turkeyworld.rssm_moe_training._own_branch_backward',side_effect=fake_branch):
            result=moe_dynamics_backward(model,combat_batch(48),combat_batch(16),expected_horizon=48,
                chunk_size=48,player_navigation_protocol=protocol)
        self.assertEqual(result['moe_main_tbptt'],48);self.assertTrue(flags[16]);self.assertTrue(flags[32])
        quiet=EnemyModel();flags=[];original=quiet.transition
        def quiet_transition(*args):flags.append(args[-1].requires_grad);return original(*args)
        quiet.transition=quiet_transition
        result=enemy_motion_backward(quiet,quiet_batch(48),context=2,expected_horizon=48,
            player_weight=1/3,player_navigation_protocol=protocol)
        self.assertEqual(result['enemy_motion_tbptt'],48);self.assertTrue(flags[16]);self.assertTrue(flags[32])
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all())for p in quiet.parameters()))

    def test_integer_anchor_keys_roundtrip_plan_but_pickle_source_sampler_stays_bit_exact(self):
        q=self.state['rssm_training']
        q['moe_training'].setdefault('samplers',{})['16']=dict(anchor_counts={i:i+3 for i in range(8)},
            rng=np.random.default_rng(91).bit_generator.state)
        self.before=deepcopy(self.state)
        self.plan=main_ladder_plan(self.state,source_checkpoint=self.source,source_checkpoint_sha256=file_sha(self.source))
        self.path.write_text(json.dumps(self.plan),encoding='utf-8')
        disk=json.loads(self.path.read_text(encoding='utf-8'))
        self.assertEqual(self.plan,disk)
        self.assertEqual(set(disk['source_rssm_training']['moe_training']['samplers']['16']['anchor_counts']),set(map(str,range(8))))
        r=self.bind()
        self.assertEqual(set(r['source_gate']['samplers']['16']['anchor_counts']),set(range(8)))
        restored=pickle.loads(pickle.dumps(self.state));self.bind(restored)
        view=legacy_binding_state(restored,disk)
        self.assertEqual(view['rssm_training'],self.before['rssm_training'])
        self.assertEqual(set(view['rssm_training']['moe_training']['samplers']['16']['anchor_counts']),set(range(8)))
        self.assertEqual(view['rssm_training']['moe_training']['samplers']['16']['rng'],
            self.before['rssm_training']['moe_training']['samplers']['16']['rng'])


if __name__=='__main__':unittest.main()
