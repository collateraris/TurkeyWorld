"""The new horizon clock preserves the old64x4 optimizer and Source ledgers."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from test_rssm_batch256_revision import fixture, update
from turkeyworld.rssm_batch256 import bind_batch256_revision
from turkeyworld.rssm_coordinate_curriculum import (
    COORDINATE_GRADIENT_PROTOCOL, coordinate_horizon, coordinate_curriculum_plan,
    bind_coordinate_curriculum, commit_coordinate_update,
)
from turkeyworld.rssm_kinematic_losses import KINEMATIC_PROTOCOL_V1
from turkeyworld.rssm_memory_overshooting import MEMORY_OVERSHOOTING_PROTOCOL


def setup(folder):
    state, batch_plan, probe, batch_kwargs, balance, parameter, optimizer = fixture(folder)
    bind_batch256_revision(state, batch_plan, probe, **batch_kwargs)
    plan = coordinate_curriculum_plan(state, source_checkpoint=Path(folder)/'source.pt',
        source_checkpoint_sha256='source5014', base_plan_sha256='base',
        kinematic_protocol=KINEMATIC_PROTOCOL_V1, overshooting_protocol=MEMORY_OVERSHOOTING_PROTOCOL)
    path = Path(folder)/'coordinate_plan.json'
    path.write_text(json.dumps(plan, sort_keys=True), encoding='utf-8')
    kwargs = dict(path=path, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_checkpoint_sha256='source5014', loaded_checkpoint_sha256='source5014',
        base_plan_sha256='base', cli_microbatch=64, cli_accumulation=4)
    return state, plan, kwargs, balance, parameter, optimizer


class CoordinateCurriculumTests(unittest.TestCase):
    def test_500th_update_reaches450_and_resume_clock_uses_completed_not_target_step(self):
        for completed, horizon in ((0,16),(99,16),(100,32),(199,32),(200,64),(299,64),
            (300,128),(399,128),(400,256),(498,256),(499,450),(500,450),(10000,450)):
            self.assertEqual(coordinate_horizon(completed), horizon)
        for bad in (-1, True, .0, '100', None):
            with self.assertRaises(ValueError): coordinate_horizon(bad)

    def test_initial_bind_is_metadata_only_preserves_model_adam_rng_old_plans_and_counters(self):
        with TemporaryDirectory() as folder:
            state, plan, kwargs, balance, parameter, optimizer = setup(folder)
            old = deepcopy(state); rng = torch.get_rng_state().clone()
            numpy_rng = deepcopy(np.random.get_state())
            plans = {p: p.read_bytes() for p in Path(folder).glob('*.json')}
            revision = bind_coordinate_curriculum(state, plan, **kwargs)
            self.assertEqual(revision['updates'], 0)
            self.assertEqual(revision['next_horizon'], 16)
            for name, value in old['rssm_training'].items(): self.assertEqual(state['rssm_training'][name], value)
            self.assertTrue(torch.equal(state['model']['weight'], old['model']['weight']))
            self.assertEqual(state['optimizer']['param_groups'], old['optimizer']['param_groups'])
            for name, value in old['optimizer']['state'][0].items():
                self.assertTrue(torch.equal(state['optimizer']['state'][0][name], value))
            self.assertEqual(state['rng_state'], old['rng_state'])
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertTrue(np.array_equal(np.random.get_state()[1], numpy_rng[1]))
            self.assertEqual(state['global_training_step'], old['global_training_step'])
            self.assertEqual(state['loss_protocol'], old['loss_protocol'])
            self.assertEqual(state['training_protocol']['combat_branch_protocol'], old['training_protocol']['combat_branch_protocol'])
            self.assertEqual(state['training_protocol']['coordinate_gradient_protocol'], COORDINATE_GRADIENT_PROTOCOL)
            self.assertEqual(state['training_protocol']['overshooting_protocol'], MEMORY_OVERSHOOTING_PROTOCOL)
            count = len(state['objective_history'])
            self.assertIs(bind_coordinate_curriculum(state, plan, **kwargs), revision)
            self.assertEqual(len(state['objective_history']), count)
            for path, content in plans.items(): self.assertEqual(path.read_bytes(), content)

    def test_four64_microbatches_one_real_adam_commit_and_exact_save_resume_next_update(self):
        with TemporaryDirectory() as folder:
            state, plan, kwargs, balance, parameter, optimizer = setup(folder)
            revision = bind_coordinate_curriculum(state, plan, **kwargs)
            start = state['global_training_step']; adam_step = float(optimizer.state[parameter]['step'])
            for _ in range(2):
                horizon = coordinate_horizon(revision['updates'])
                update(state, balance, parameter, optimizer, effective=256)
                commit_coordinate_update(state, step=state['global_training_step'], horizon=horizon,
                    sampler_draws=state['rssm_training']['combat_recovery_revision']['sampler_state']['draws'])
            self.assertEqual(float(optimizer.state[parameter]['step']), adam_step+2)
            self.assertEqual(revision['updates'], 2)
            self.assertEqual(revision['main_transitions'], 2*16*256)
            self.assertEqual(revision['main_examples'], 512)
            path = Path(folder)/'continued.pt'; torch.save(state, path)
            restored = torch.load(path, map_location='cpu', weights_only=False)
            q = bind_coordinate_curriculum(restored, plan, **{**kwargs, 'loaded_checkpoint_sha256':'new_saved_sha'})
            self.assertEqual(q['updates'], 2)
            from turkeyworld.rssm_component_balance import ComponentLossBalancer
            balance2 = ComponentLossBalancer(restored['rssm_training']['loss_balance_state'])
            parameter2 = torch.nn.Parameter(restored['model']['weight'].clone())
            optimizer2 = torch.optim.AdamW([parameter2], lr=1e-4); optimizer2.load_state_dict(restored['optimizer'])
            for target, b, p, opt in ((state,balance,parameter,optimizer),(restored,balance2,parameter2,optimizer2)):
                update(target,b,p,opt,effective=256)
                commit_coordinate_update(target,step=target['global_training_step'],horizon=16,
                    sampler_draws=target['rssm_training']['combat_recovery_revision']['sampler_state']['draws'])
            self.assertTrue(torch.equal(parameter, parameter2))
            for name, value in optimizer.state[parameter].items(): self.assertTrue(torch.equal(value, optimizer2.state[parameter2][name]))
            self.assertEqual(state['rssm_training'], restored['rssm_training'])
            self.assertEqual(state['global_training_step'], start+3)

    def test_actual_sampler_draws_allow_commit_while_checkpoint_sampler_state_is_stale(self):
        with TemporaryDirectory() as folder:
            state, plan, kwargs, balance, parameter, optimizer = setup(folder)
            revision = bind_coordinate_curriculum(state, plan, **kwargs)
            saved_sampler = deepcopy(state['rssm_training']['combat_recovery_revision']['sampler_state'])
            update(state,balance,parameter,optimizer,effective=256)
            actual_draws = state['rssm_training']['combat_recovery_revision']['sampler_state']['draws']
            state['rssm_training']['combat_recovery_revision']['sampler_state'] = saved_sampler
            commit_coordinate_update(state,step=state['global_training_step'],horizon=16,sampler_draws=actual_draws)
            self.assertEqual(revision['updates'], 1)
            with self.assertRaises(ValueError): bind_coordinate_curriculum(state,plan,**kwargs)
            # self.save() later publishes the live sampler state into its checkpoint ledger.
            state['rssm_training']['combat_recovery_revision']['sampler_state']['draws'] = actual_draws
            bind_coordinate_curriculum(state,plan,**{**kwargs,'loaded_checkpoint_sha256':'continued'})
            before = deepcopy(revision)
            with self.assertRaises(ValueError):commit_coordinate_update(state,step=state['global_training_step'],horizon=16,sampler_draws=actual_draws)
            self.assertEqual(revision, before)

    def test_horizon_histogram_500_metadata_commits_and_failed_or_partial_updates_do_not_advance(self):
        with TemporaryDirectory() as folder:
            state, plan, kwargs, *_ = setup(folder)
            revision = bind_coordinate_curriculum(state, plan, **kwargs)
            q = state['rssm_training']
            before = deepcopy(revision)
            with self.assertRaises(ValueError):commit_coordinate_update(state,step=state['global_training_step']+1,horizon=16,sampler_draws=revision['sampler_draws_at_start']+64)
            self.assertEqual(revision,before)
            for index in range(500):
                step=state['global_training_step']+1; state['global_training_step']=step
                for name in ('combat_v6_revision','batch256_revision','combat_only_revision','loss_balance_revision','batch_revision','combat_recovery_revision','feedback_revision'):
                    q[name]['last_optimizer_step']=step
                q['combat_completed']+=1;q['source_completed']+=1
                q['auxiliary_draws']+=256;q['combat_auxiliary_draws']+=256
                q['combat_recovery_revision']['sampler_state']['draws']+=256
                commit_coordinate_update(state,step=step,horizon=coordinate_horizon(index),
                    sampler_draws=q['combat_recovery_revision']['sampler_state']['draws'])
            self.assertEqual(revision['horizon_update_counts'],{'16':100,'32':100,'64':100,'128':100,'256':99,'450':1})
            self.assertEqual(revision['main_transitions_per_example'],49794)
            self.assertEqual(revision['next_horizon'],450)
            bind_coordinate_curriculum(state,plan,**{**kwargs,'loaded_checkpoint_sha256':'after500'})

    def test_source_plan_protocol_parent_and_resume_tampering_fail_without_binding(self):
        with TemporaryDirectory() as folder:
            state, plan, kwargs, *_ = setup(folder)
            for key, value in [('version', True),('curriculum_updates',499),('microbatch_size',32),
                ('validation_horizon',16),('source_checkpoint_sha256','wrong'),('source_future_feedback',True),
                ('parent_batch256_sha256','wrong'),('main_window_start',1),('coordinate_gradient_protocol',{}),
                ('kinematic_protocol',{'version':999}),('overshooting_protocol',{'version':999})]:
                changed=deepcopy(plan);changed[key]=value
                with self.assertRaises(ValueError):bind_coordinate_curriculum(deepcopy(state),changed,**kwargs)
            for changes in ({'loaded_checkpoint_sha256':'wrong'},{'cli_microbatch':32},{'cli_accumulation':1}):
                with self.assertRaises(ValueError):bind_coordinate_curriculum(deepcopy(state),plan,**{**kwargs,**changes})
            revision=bind_coordinate_curriculum(state,plan,**kwargs)
            for key,value in [('updates',1),('next_horizon',32),('main_examples',64),('main_transitions',1)]:
                bad=deepcopy(state);bad['rssm_training']['coordinate_curriculum'][key]=value
                with self.assertRaises(ValueError):bind_coordinate_curriculum(bad,plan,**kwargs)
            self.assertEqual(revision['updates'],0)


if __name__ == '__main__': unittest.main()
