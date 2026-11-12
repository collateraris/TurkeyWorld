"""Strict completed-update loss gates preserve actual old horizon/Adam history."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from test_rssm_replay_curriculum import setup as replay_setup,advance
from test_rssm_drift_only_protocol import update_drift,raw_report
from turkeyworld.rssm_replay_curriculum import bind_replay_curriculum,replay_training_horizon,ReplayComponentLossBalancer
from turkeyworld.rssm_coordinate_curriculum import bind_coordinate_curriculum
from turkeyworld.rssm_drift_only_protocol import (drift_only_plan,bind_drift_only,frozen_balance_values,summarize_drift_groups)
from turkeyworld.rssm_loss_gated_curriculum import (HORIZONS,loss_gated_curriculum_plan,
    bind_loss_gated_curriculum,commit_loss_gated_update,loss_gated_coordinate_expectation)


def plan_file(path,plan):
    path.write_text(json.dumps(plan,sort_keys=True),encoding='utf-8')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup(folder):
    state,rp,rk,cp,ck,parameter,optimizer,_=replay_setup(folder)
    _,balance=bind_replay_curriculum(state,rp,**rk)
    for _ in range(19):advance(state,balance,parameter,optimizer)
    dp=drift_only_plan(state,source_checkpoint='source5079.pt',source_checkpoint_sha256='source5079',
        base_plan_sha256='base',optimizer_parameter_names=['weight'])
    dpath=Path(folder)/'drift.json'
    dk=dict(path=dpath,sha256=plan_file(dpath,dp),source_checkpoint_sha256='source5079',
        loaded_checkpoint_sha256='source5079',base_plan_sha256='base',optimizer_parameter_names=['weight'])
    bind_drift_only(state,dp,**dk)
    for _ in range(33):update_drift(state,balance,parameter,optimizer)
    plan=loss_gated_curriculum_plan(state,source_checkpoint='source5112.pt',source_checkpoint_sha256='source5112',base_plan_sha256='base')
    path=Path(folder)/'loss_gate.json'
    kwargs=dict(path=path,sha256=plan_file(path,plan),source_checkpoint_sha256='source5112',
        loaded_checkpoint_sha256='source5112',base_plan_sha256='base')
    return state,plan,kwargs,balance,parameter,optimizer,(cp,ck,rp,rk,dp,dk)


def step(state,balance,parameter,optimizer,loss):
    summary,choice=update_drift(state,balance,parameter,optimizer)
    gate=commit_loss_gated_update(state,step=state['global_training_step'],horizon=choice['horizon'],
        train_loss=summary['total']if loss is None else loss)
    return gate,choice,summary


class LossGatedCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_metadata_only_initial16_preserves_all_old_clocks_adam_rng_ema_and_plans(self):
        with TemporaryDirectory()as folder:
            state,plan,kwargs,balance,parameter,optimizer,_=setup(folder)
            before=deepcopy(state);rng=torch.get_rng_state().clone()
            files={p:p.read_bytes()for p in Path(folder).glob('*.json')}
            gate=bind_loss_gated_curriculum(state,plan,**kwargs)
            self.assertEqual(state['global_training_step'],5112)
            self.assertEqual(gate['stage_index'],0);self.assertEqual(gate['current_horizon'],16)
            self.assertEqual(state['rssm_training']['coordinate_curriculum']['updates'],98)
            self.assertEqual(state['rssm_training']['replay_curriculum']['updates'],52)
            self.assertEqual(state['rssm_training']['drift_only_revision']['updates'],33)
            self.assertFalse(replay_training_horizon(state)['forced_long_replay'])
            self.assertTrue(torch.equal(torch.get_rng_state(),rng));self.assertEqual(state['rng_state'],before['rng_state'])
            self.assertTrue(torch.equal(state['model']['weight'],before['model']['weight']))
            for key,value in before['optimizer']['state'][0].items():self.assertTrue(torch.equal(value,state['optimizer']['state'][0][key]))
            self.assertEqual(state['rssm_training']['loss_balance_state'],before['rssm_training']['loss_balance_state'])
            self.assertEqual(state['loss_protocol'],before['loss_protocol'])
            for path,raw in files.items():self.assertEqual(path.read_bytes(),raw)

    def test_real_four_micro_one_adam_gate_uses_completed_mean_and_never_forces_fifth450(self):
        with TemporaryDirectory()as folder:
            state,plan,kwargs,balance,parameter,optimizer,_=setup(folder);bind_loss_gated_curriculum(state,plan,**kwargs)
            old_clock=float(optimizer.state[parameter]['step']);values=frozen_balance_values(balance.state)
            forced=state['rssm_training']['replay_curriculum']['forced_long_replay_updates']
            trained=[]
            for _ in range(10):
                gate,choice,summary=step(state,balance,parameter,optimizer,.3)
                trained.append(choice['horizon']);self.assertEqual(summary['microbatches'],4)
            self.assertEqual(trained,[16]*10)
            self.assertEqual(float(optimizer.state[parameter]['step']),old_clock+10)
            self.assertEqual(gate['updates'],10);self.assertEqual(gate['stage_updates'],10)
            self.assertEqual(state['rssm_training']['replay_curriculum']['forced_long_replay_updates'],forced)
            self.assertEqual(frozen_balance_values(balance.state),values)
            self.assertEqual(state['rssm_training']['coordinate_curriculum']['updates'],108)

    def test_strict_boundary_tenth_consecutive_only_and_one_advance_even_for_zero_loss(self):
        with TemporaryDirectory()as folder:
            state,plan,kwargs,balance,parameter,optimizer,_=setup(folder);bind_loss_gated_curriculum(state,plan,**kwargs)
            for loss in (.25,.2500000001):
                gate,_,_=step(state,balance,parameter,optimizer,loss)
                self.assertEqual(gate['current_horizon'],16);self.assertFalse(gate['last_gate_passed'])
            for index in range(9):
                gate,choice,_=step(state,balance,parameter,optimizer,math.nextafter(.25,0.))
                self.assertEqual(choice['horizon'],16);self.assertEqual(gate['current_horizon'],16)
                self.assertEqual(gate['consecutive_below_threshold'],index+1)
            gate,choice,_=step(state,balance,parameter,optimizer,0.)
            self.assertEqual(choice['horizon'],16);self.assertEqual(gate['current_horizon'],32)
            self.assertTrue(gate['last_stage_advanced']);self.assertEqual(gate['gate_advances'],1)
            self.assertEqual(gate['consecutive_below_threshold'],0)
            gate,choice,_=step(state,balance,parameter,optimizer,0.)
            self.assertEqual(choice['horizon'],32);self.assertEqual(gate['current_horizon'],32)
            self.assertEqual(gate['gate_advances'],1);self.assertEqual(gate['consecutive_below_threshold'],1)
            self.assertEqual(gate['advance_history'][0]['completed_updates_at_previous_stage'],12)
            self.assertEqual(gate['advance_history'][0]['qualifying_streak'],10)

    def test_exact_all16_horizons_then450_forever_and_actual_transition_conservation(self):
        with TemporaryDirectory()as folder:
            state,plan,kwargs,balance,parameter,optimizer,_=setup(folder);bind_loss_gated_curriculum(state,plan,**kwargs)
            source=deepcopy(state['rssm_training']['coordinate_curriculum'])
            horizons=[]
            expected=[horizon for horizon in HORIZONS for _ in range(10)]+[450]*3
            for _ in expected:
                gate,choice,_=step(state,balance,parameter,optimizer,0.)
                horizons.append(choice['horizon'])
            self.assertEqual(horizons,expected)
            self.assertEqual(gate['stage_index'],15);self.assertEqual(gate['gate_advances'],15)
            self.assertEqual(gate['current_horizon'],450);self.assertFalse(gate['last_stage_advanced'])
            self.assertEqual(gate['consecutive_below_threshold'],10);self.assertTrue(gate['goal_met'])
            self.assertEqual(gate['main_transitions'],sum(horizons)*256)
            c=state['rssm_training']['coordinate_curriculum']
            self.assertEqual(c['updates'],source['updates']+len(horizons))
            self.assertEqual(c['main_transitions'],source['main_transitions']+sum(horizons)*256)
            histogram,transitions,next_h=loss_gated_coordinate_expectation(state,c['updates'])
            self.assertEqual(histogram,c['horizon_update_counts'])
            self.assertEqual(transitions,c['main_transitions_per_example']);self.assertEqual(next_h,450)
            self.assertEqual(gate['horizon_update_counts']['96'],10)
            bind_loss_gated_curriculum(state,plan,**{**kwargs,'loaded_checkpoint_sha256':'continued'})

    def test_resume_old_coordinate_replay_drift_and_new_gate_preserves_next_actual_update(self):
        with TemporaryDirectory()as folder:
            state,plan,kwargs,balance,parameter,optimizer,parents=setup(folder);bind_loss_gated_curriculum(state,plan,**kwargs)
            for _ in range(3):step(state,balance,parameter,optimizer,.1)
            self.assertEqual(state['rssm_training']['loss_gated_curriculum']['consecutive_below_threshold'],3)
            path=Path(folder)/'saved.pt';torch.save(state,path)
            restored=torch.load(path,map_location='cpu',weights_only=False)
            cp,ck,rp,rk,dp,dk=parents
            bind_coordinate_curriculum(restored,cp,**{**ck,'loaded_checkpoint_sha256':'next'})
            _,other=bind_replay_curriculum(restored,rp,**{**rk,'loaded_checkpoint_sha256':'next'})
            bind_drift_only(restored,dp,**{**dk,'loaded_checkpoint_sha256':'next'})
            bind_loss_gated_curriculum(restored,plan,**{**kwargs,'loaded_checkpoint_sha256':'next'})
            parameter2=torch.nn.Parameter(restored['model']['weight'].clone())
            optimizer2=torch.optim.AdamW([parameter2],lr=1e-4);optimizer2.load_state_dict(restored['optimizer'])
            for _ in range(7):
                step(state,balance,parameter,optimizer,.1);step(restored,other,parameter2,optimizer2,.1)
            self.assertTrue(torch.equal(parameter,parameter2));self.assertEqual(state['rssm_training'],restored['rssm_training'])
            self.assertEqual(state['rssm_training']['loss_gated_curriculum']['current_horizon'],32)
            self.assertEqual(state['rssm_training']['loss_gated_curriculum']['consecutive_below_threshold'],0)
            for key,value in optimizer.state[parameter].items():self.assertTrue(torch.equal(value,optimizer2.state[parameter2][key]))

    def test_no_per_micro_promotion_when_only_full_update_loss_is_above_threshold(self):
        with TemporaryDirectory()as folder:
            state,plan,kwargs,balance,parameter,optimizer,_=setup(folder);bind_loss_gated_curriculum(state,plan,**kwargs)
            reports=[raw_report(value/2,value/2)for value in(0.,0.,0.,1.2)]
            mean=summarize_drift_groups(reports,step=5113)
            self.assertAlmostEqual(mean['total'],.3)
            gate,_,_=step(state,balance,parameter,optimizer,mean['total'])
            self.assertEqual(gate['current_horizon'],16);self.assertFalse(gate['last_gate_passed'])

    def test_nonfinite_partial_duplicate_and_tampered_history_are_fail_closed(self):
        with TemporaryDirectory()as folder:
            state,plan,kwargs,balance,parameter,optimizer,_=setup(folder);bind_loss_gated_curriculum(state,plan,**kwargs)
            _,choice=update_drift(state,balance,parameter,optimizer)
            before=deepcopy(state['rssm_training']['loss_gated_curriculum'])
            for options in ({'train_loss':float('nan')},{'train_loss':float('inf')},{'train_loss':.1,'microbatches':3},
                            {'train_loss':.1,'microbatches':4.},{'train_loss':-.1}):
                with self.assertRaises(ValueError):commit_loss_gated_update(state,step=5113,horizon=choice['horizon'],**options)
                self.assertEqual(state['rssm_training']['loss_gated_curriculum'],before)
            commit_loss_gated_update(state,step=5113,horizon=16,train_loss=.1)
            with self.assertRaises(ValueError):commit_loss_gated_update(state,step=5113,horizon=32,train_loss=.1)
            for _ in range(9):step(state,balance,parameter,optimizer,.1)
            bad=deepcopy(state);bad['rssm_training']['loss_gated_curriculum']['advance_history'][0]['train_loss']=.25
            with self.assertRaises(ValueError):bind_loss_gated_curriculum(bad,plan,**{**kwargs,'loaded_checkpoint_sha256':'later'})

    def test_source_parent_threshold_path_and_plan_tampering_rejected_without_binding(self):
        with TemporaryDirectory()as folder:
            state,plan,kwargs,*_=setup(folder)
            for key,value in [('source_checkpoint_sha256','wrong'),('settings',{}),('drift_only_plan_sha256','wrong'),
                              ('coordinate_plan_sha256','wrong'),('replay_plan_sha256','wrong')]:
                bad=deepcopy(plan);bad[key]=value
                with self.assertRaises(ValueError):bind_loss_gated_curriculum(deepcopy(state),bad,**kwargs)
            with self.assertRaises(ValueError):bind_loss_gated_curriculum(deepcopy(state),plan,**{**kwargs,'loaded_checkpoint_sha256':'wrong'})
            bad=deepcopy(state);bad['rssm_training']['drift_only_revision']['last_optimizer_step']-=1
            with self.assertRaises(ValueError):bind_loss_gated_curriculum(bad,plan,**kwargs)
            self.assertNotIn('loss_gated_curriculum',state['rssm_training'])

    def test_intermediate_equal_or_high_resets_ten_consecutive_count_and_micros_count_once(self):
        with TemporaryDirectory()as folder:
            state,plan,kwargs,balance,parameter,optimizer,_=setup(folder);bind_loss_gated_curriculum(state,plan,**kwargs)
            for _ in range(9):step(state,balance,parameter,optimizer,.1)
            self.assertEqual(state['rssm_training']['loss_gated_curriculum']['consecutive_below_threshold'],9)
            gate,_,summary=step(state,balance,parameter,optimizer,.25)
            self.assertEqual(summary['microbatches'],4)
            self.assertEqual(gate['consecutive_below_threshold'],0);self.assertEqual(gate['current_horizon'],16)
            for _ in range(9):step(state,balance,parameter,optimizer,.1)
            gate,_,_=step(state,balance,parameter,optimizer,.4)
            self.assertEqual(gate['consecutive_below_threshold'],0);self.assertEqual(gate['current_horizon'],16)
            for index in range(9):
                gate,_,_=step(state,balance,parameter,optimizer,.1)
                self.assertEqual(gate['current_horizon'],16)
                self.assertEqual(gate['consecutive_below_threshold'],index+1)
            gate,_,_=step(state,balance,parameter,optimizer,.1)
            self.assertEqual(gate['current_horizon'],32)
            self.assertEqual(gate['consecutive_below_threshold'],0)


if __name__=='__main__':unittest.main()
